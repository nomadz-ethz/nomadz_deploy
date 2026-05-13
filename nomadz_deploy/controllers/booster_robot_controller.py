from __future__ import annotations
import logging
import signal
import time
import threading
import multiprocessing as mp
from multiprocessing import synchronize

import numpy as np
import torch

import rclpy
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from booster_interface.msg import LowState, LowCmd, MotorCmd

# Diagnostic publishers (Foxglove visualisation; see docs §8). Imported at
# module level because rclpy is already a hard dependency and these are
# standard ROS 2 packages — if either isn't available, the deployment was
# already broken.
from std_msgs.msg import Bool, Float32, Int32, String
from geometry_msgs.msg import Vector3

from booster_robotics_sdk_python import (  # type: ignore
    B1LocoClient,
    RobotMode,
)

# Optional — older SDK builds may not expose GetMode/GetModeResponse.
# Used here only for post-ChangeMode verification in the legacy helpers.
try:
    from booster_robotics_sdk_python import GetModeResponse  # type: ignore
    _HAS_GETMODE = True
except ImportError:  # pragma: no cover
    _HAS_GETMODE = False

from .controller_cfg import ControllerCfg
from .base_controller import BaseController, BoosterRobot
from ..utils.synced_array import SyncedArray
from ..utils.metrics import SyncedMetrics
from ..utils.isaaclab import math as lab_math
from ..utils.remote_control_service import RemoteControlService
from ..utils.joystick_handler import JoystickHandler
from ..utils.keyboard_handler import KeyboardHandler


logger = logging.getLogger("booster_deploy")
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


class CountTimer:
    def __init__(self, dt: float = 0.002, use_sim_time: bool = False):
        self.dt = dt
        # Use multiprocessing.Value for inter-process communication
        self.counter = mp.Value('L', 0)
        self.use_sim_time = use_sim_time

    def tick_timer_if_sim(self):
        if self.use_sim_time:
            with self.counter.get_lock():
                self.counter.value += 1

    def get_time(self):
        if self.use_sim_time:
            with self.counter.get_lock():
                return self.counter.value * self.dt
        else:
            return time.perf_counter()


class BoosterRobotPortal:
    synced_state: SyncedArray
    synced_command: SyncedArray
    synced_action: SyncedArray
    exit_event: synchronize.Event

    def __init__(
        self,
        cfg: ControllerCfg,
        use_sim_time: bool = False,
        joystick_enabled: bool = False,
        keyboard_enabled: bool = False,
    ) -> None:
        self.cfg = cfg

        self.robot = BoosterRobot(cfg.robot)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.remoteControlService = RemoteControlService()

        # Operator-input handler that drives the recovery state machine.
        # Two interchangeable backends, both implementing the
        # JoystickHandler-shape interface (axes / consume_press / start /
        # stop / calibrate):
        #
        #   - JoystickHandler — pygame, our own external controller plugged
        #     into the host's USB. Production path.
        #   - KeyboardHandler — stdin/cbreak, useful before a pad is
        #     available (early bring-up, sim runs over SSH). Same FSM,
        #     same buttons; just slower to drive.
        #
        # The two flags are mutually exclusive; if both are passed the
        # joystick wins (it's the more deliberate choice for hardware).
        # See docs/RECOVERY_INTEGRATION_PLAN.md §4 for the two-controllers
        # split (manufacturer's evdev pad vs. ours).
        self.joystick_enabled = bool(joystick_enabled)
        self.keyboard_enabled = bool(keyboard_enabled) and not self.joystick_enabled
        self.joystick_handler: "JoystickHandler | KeyboardHandler | None" = None
        if self.joystick_enabled:
            try:
                self.joystick_handler = JoystickHandler()
                self.joystick_handler.calibrate()
                self.joystick_handler.start()
                self.logger.info("JoystickHandler enabled and calibrated.")
            except Exception as exc:
                self.logger.warning(
                    "Failed to initialise JoystickHandler (%s); falling back "
                    "to RemoteControlService.", exc)
                self.joystick_enabled = False
                self.joystick_handler = None
        elif self.keyboard_enabled:
            try:
                self.joystick_handler = KeyboardHandler()
                self.joystick_handler.calibrate()
                self.joystick_handler.start()
                self.logger.info("KeyboardHandler enabled (cbreak stdin).")
            except Exception as exc:
                self.logger.warning(
                    "Failed to initialise KeyboardHandler (%s); falling back "
                    "to RemoteControlService.", exc)
                self.keyboard_enabled = False
                self.joystick_handler = None

        # Use multiprocessing.Event for inter-process communication
        self.exit_event = mp.Event()
        # Gates the inference subprocess's publish loop. set() = run, clear()
        # = pause without termination (used by the X-press emergency-damp path
        # in the recovery state machine). The subprocess inherits this Event
        # because we set() it before forking.
        self.policy_run_event = mp.Event()
        self.policy_run_event.set()
        self.is_running = True
        self.timer = CountTimer(
            self.cfg.booster.low_state_dt, use_sim_time=use_sim_time)

        # Fall-detection state. Updated on the low_state subscription thread;
        # consumed by the recovery state machine on the main thread. Only
        # logs/publishes by default — auto-recovery is intentionally OFF
        # (see docs §13.6); the operator commits via a B-press.
        self._fall_event = threading.Event()
        self._fall_streak = 0
        # Cached projected-gravity-z so _wait_until_upright() can poll it
        # without re-decoding the IMU each tick.
        self._proj_g_z: float = -1.0

        def signal_handler(sig, frame):
            if mp.current_process().name == "MainProcess":
                print("\nKeyboard interrupt received. Shutting down...")
            self.exit_event.set()

        # Register signal handler
        signal.signal(signal.SIGINT, signal_handler)

        self._init_synced_buffer()
        self._init_metrics()

        self._cleanup_done = False
        self.inference_process = None  # Inference process reference
        self.low_cmd_publisher: rclpy.publisher.Publisher = None
        self.low_state_thread = None
        self.low_cmd_process: mp.Process | None = None

        rclpy.init()
        # Initialize communication. Callbacks may start immediately and
        # reference `is_running` and `exit_event`, so ensure those are set.
        self._init_communication()

    def _init_synced_buffer(self):
        action_dtype = np.dtype(
            [
                ("dof_target", float, (self.robot.num_joints,)),
                ("stiffness", float, (self.robot.num_joints,)),
                ("damping", float, (self.robot.num_joints,)),
            ]
        )
        self.synced_action = SyncedArray(
            "action",
            shape=(1,),
            dtype=action_dtype,
        )
        self._action_buf = np.ndarray((1,), dtype=action_dtype)

        state_dtype = np.dtype(
            [
                ("root_rpy_w", float, (3,)),
                ("root_ang_vel_b", float, (3,)),
                ("root_pos_w", float, (3,)),
                ("root_lin_vel_w", float, (3,)),
                ("joint_pos", float, (self.robot.num_joints,)),
                ("joint_vel", float, (self.robot.num_joints,)),
                ("feedback_torque", float, (self.robot.num_joints,)),
            ]
        )
        self.synced_state = SyncedArray(
            "state",
            shape=(1,),
            dtype=state_dtype
        )
        self._state_buf = np.zeros((1,), dtype=state_dtype)

        command_dtype = np.dtype(
            [
                ("vx", float),
                ("vy", float),
                ("vyaw", float),
            ]
        )
        self.synced_command = SyncedArray(
            "command",
            shape=(1,),
            dtype=command_dtype,
        )

    def _init_metrics(self):
        # initialize cross-process synced metrics
        max_events = self.cfg.booster.metrics_max_events
        self.metrics = {
            "low_state_handler": SyncedMetrics(
                "low_state_handler", max_events=max_events
            ),
            "policy_step": SyncedMetrics(
                "policy_step", max_events=max_events
            ),
        }

    def _init_communication(self) -> None:
        try:
            self.client = B1LocoClient()
            self.create_low_cmd_publisher("booster_deploy_low_cmd_pub")
            self._start_low_state_subscription()
            self._init_diagnostics()
            self.client.Init()
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _init_diagnostics(self) -> None:
        """Stand up the ``/nomadz/*`` diagnostic publishers (see docs §8).

        Publishers don't need their node spun (only subscribers/timers do),
        so we don't run an executor for this node — we just keep it alive
        and call ``publish()`` from whichever thread produces the message
        (low_state callback, joystick poll thread, recovery FSM). rclpy's
        ``Publisher.publish`` is thread-safe.
        """
        self._diag_node = rclpy.create_node("nomadz_diagnostics")
        # Burst-only events get RELIABLE so we never miss a state transition;
        # high-rate streams get BEST_EFFORT/depth=1 so they never queue up.
        qos_event = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        qos_stream = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._pub_recovery_state = self._diag_node.create_publisher(
            String, "/nomadz/recovery_state", qos_event)
        self._pub_mode = self._diag_node.create_publisher(
            String, "/nomadz/mode", qos_event)
        self._pub_proj_g_z = self._diag_node.create_publisher(
            Float32, "/nomadz/proj_g_z", qos_stream)
        self._pub_fall_flag = self._diag_node.create_publisher(
            Bool, "/nomadz/fall_flag", qos_event)
        self._pub_joystick_axes = self._diag_node.create_publisher(
            Vector3, "/nomadz/joystick_axes", qos_stream)
        self._pub_button_events = self._diag_node.create_publisher(
            String, "/nomadz/button_events", qos_event)
        self._pub_imu_rpy = self._diag_node.create_publisher(
            Vector3, "/nomadz/imu_rpy", qos_stream)
        self._pub_getup_rc = self._diag_node.create_publisher(
            Int32, "/nomadz/getup_return_code", qos_event)

        # Joystick axes thread (50 Hz). Skipped when no input handler is
        # enabled so the topic just stays silent on legacy runs. Joystick
        # and keyboard backends share the same diag publisher path.
        self._diag_thread: threading.Thread | None = None
        if self.joystick_handler is not None:
            self._diag_thread = threading.Thread(
                target=self._diag_loop,
                name="nomadz_diag",
                daemon=True,
            )
            self._diag_thread.start()

        # Track previous fall_flag so the low_state handler only publishes on
        # rising/falling edges (saves bandwidth and gives Foxglove a clean
        # step trace).
        self._last_fall_flag_published = False

    # ------------------------------------------------------------------
    # Diagnostic publish helpers — safe to call from any thread.
    # ------------------------------------------------------------------

    def _publish_recovery_state(self, name: str) -> None:
        msg = String()
        msg.data = name
        self._pub_recovery_state.publish(msg)

    def _publish_mode(self, name: str) -> None:
        msg = String()
        msg.data = name
        self._pub_mode.publish(msg)

    def _publish_proj_g_z(self, value: float) -> None:
        msg = Float32()
        msg.data = float(value)
        self._pub_proj_g_z.publish(msg)

    def _publish_fall_flag(self, value: bool) -> None:
        msg = Bool()
        msg.data = bool(value)
        self._pub_fall_flag.publish(msg)

    def _publish_joystick_axes(self, vx: float, vy: float, vyaw: float) -> None:
        msg = Vector3()
        msg.x = float(vx)
        msg.y = float(vy)
        msg.z = float(vyaw)
        self._pub_joystick_axes.publish(msg)

    def _publish_button_event(self, name: str) -> None:
        msg = String()
        msg.data = name
        self._pub_button_events.publish(msg)

    def _publish_imu_rpy(self, roll: float, pitch: float, yaw: float) -> None:
        msg = Vector3()
        msg.x = float(roll)
        msg.y = float(pitch)
        msg.z = float(yaw)
        self._pub_imu_rpy.publish(msg)

    def _publish_getup_return_code(self, rc: int) -> None:
        msg = Int32()
        msg.data = int(rc)
        self._pub_getup_rc.publish(msg)

    def _diag_loop(self) -> None:
        """Pump joystick axes onto ``/nomadz/joystick_axes`` at 50 Hz.

        Mode and per-press button events are published from the state
        machine itself (it knows when it issues a ChangeMode and when it
        consumes a press), so this loop only handles the continuous
        joystick stream.
        """
        period = 0.02  # 50 Hz
        next_t = time.monotonic()
        while not self.exit_event.is_set():
            try:
                if (
                    self.joystick_handler is not None
                    and self.joystick_handler.calibrated
                ):
                    vx, vy, vyaw = self.joystick_handler.axes()
                    self._publish_joystick_axes(vx, vy, vyaw)
            except Exception as exc:
                self.logger.debug("diag_loop joystick publish failed: %s", exc)

            next_t += period
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.monotonic()

    def _start_low_state_subscription(self) -> None:
        """Start ROS 2 subscription loop on a dedicated thread.

        The subscription is run on a dedicated thread and spins a
        SingleThreadedExecutor for the `/low_state` topic.
        """

        def low_state_service_executor():
            self.logger.info("Low state subscription started")
            low_state_node = rclpy.create_node("booster_deploy_low_state_sub")
            low_state_node.create_subscription(
                LowState,
                "/low_state",
                self._low_state_handler,
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST,
                ),
            )

            executor = SingleThreadedExecutor()
            executor.add_node(low_state_node)

            try:
                # loop: check exit_event and rclpy.ok()
                while rclpy.ok() and not self.exit_event.is_set():
                    executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                pass
            except Exception as exc:
                # Suppress RCLError if we are shutting down
                is_rcl_error = "RCLError" in type(exc).__name__
                is_shutting_down = self.exit_event.is_set() or not rclpy.ok()

                if is_rcl_error and is_shutting_down:
                    pass
                else:
                    self.logger.error(
                        "Low state subscription executor stopped: %s",
                        exc,
                        exc_info=True
                    )
            finally:
                executor.shutdown()
                low_state_node.destroy_node()
            self.logger.info("Low state subscription stopped")

        self.low_state_thread = threading.Thread(
            target=low_state_service_executor,
            name="low_state_executor",
            daemon=True,
        )
        self.low_state_thread.start()

    def _low_state_handler(self, low_state_msg: LowState):
        self.metrics["low_state_handler"].mark()
        try:
            if not self.is_running or self.exit_event.is_set():
                return

            # simulator tick
            self.timer.tick_timer_if_sim()

            # collect state data
            rpy = np.array(low_state_msg.imu_state.rpy, dtype=np.float32)
            gyro = np.array(low_state_msg.imu_state.gyro, dtype=np.float32)
            dof_pos = np.zeros(self.robot.num_joints, dtype=np.float32)
            dof_vel = np.zeros(self.robot.num_joints, dtype=np.float32)
            fb_torque = np.zeros(self.robot.num_joints, dtype=np.float32)

            for i, motor in enumerate(low_state_msg.motor_state_serial):
                dof_pos[i] = motor.q
                dof_vel[i] = motor.dq
                fb_torque[i] = motor.tau_est

            self._state_buf[0]["root_rpy_w"][:] = rpy
            self._state_buf[0]["root_ang_vel_b"][:] = gyro
            self._state_buf[0]["root_pos_w"][:] = np.zeros(
                3, dtype=np.float32
            )
            self._state_buf[0]["root_lin_vel_w"][:] = np.zeros(
                3, dtype=np.float32
            )
            self._state_buf[0]["joint_pos"][:] = dof_pos
            self._state_buf[0]["joint_vel"][:] = dof_vel
            self._state_buf[0]["feedback_torque"][:] = fb_torque
            self.synced_state.write(self._state_buf)

            # IMU-based fall detector. proj_g_z is the world-down vector
            # rotated into the body frame; for an upright robot it sits near
            # -1.0, and on a fall it climbs toward 0 or above. Closed-form
            # equivalent to LocomotionPolicy.compute_observation's
            # quat_apply_inverse(quat, [0,0,-1]) but cheap to compute here
            # without converting to torch. See docs §6 ("Fall detection").
            roll, pitch = float(rpy[0]), float(rpy[1])
            yaw = float(rpy[2])
            proj_g_z = -float(np.cos(roll) * np.cos(pitch))
            self._proj_g_z = proj_g_z
            cfg_b = self.cfg.booster
            if proj_g_z > cfg_b.fall_proj_g_z_threshold:
                self._fall_streak += 1
                if self._fall_streak >= cfg_b.fall_streak_threshold:
                    if not self._fall_event.is_set():
                        self.logger.warning(
                            "Fall detected (proj_g_z=%.3f > %.3f for %d ticks). "
                            "Auto-recovery is OFF — operator must press B.",
                            proj_g_z, cfg_b.fall_proj_g_z_threshold,
                            self._fall_streak,
                        )
                        self._fall_event.set()
            else:
                self._fall_streak = 0

            # Diagnostic publishers. proj_g_z and imu_rpy stream every tick
            # (cheap small messages). fall_flag only emits on level changes
            # so Foxglove gets a clean step trace.
            self._publish_proj_g_z(proj_g_z)
            self._publish_imu_rpy(roll, pitch, yaw)
            cur_fall = self._fall_event.is_set()
            if cur_fall != self._last_fall_flag_published:
                self._publish_fall_flag(cur_fall)
                self._last_fall_flag_published = cur_fall

            # update velocity commands to synced_command. When our own input
            # handler (joystick OR keyboard) is enabled it overrides the
            # manufacturer's RemoteControlService; otherwise legacy
            # behaviour stands.
            cmd = np.zeros((1,), dtype=self.synced_command.dtype)
            if self.joystick_handler is not None:
                vx, vy, vyaw = self.joystick_handler.axes()
                cmd[0]["vx"] = vx
                cmd[0]["vy"] = vy
                cmd[0]["vyaw"] = vyaw
            else:
                cmd[0]["vx"] = self.remoteControlService.get_vx_cmd()
                cmd[0]["vy"] = self.remoteControlService.get_vy_cmd()
                cmd[0]["vyaw"] = self.remoteControlService.get_vyaw_cmd()
            self.synced_command.write(cmd)

        except Exception as e:
            self.logger.error(f"Error in _low_state_handler: {e}")
            self.running = False
            self.exit_event.set()

    def create_low_cmd_publisher(self, name):
        self.publish_node = rclpy.create_node(name)
        publisher = self.publish_node.create_publisher(
            LowCmd,
            "joint_ctrl",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST
            )
        )
        self.low_cmd_publisher = publisher

        # construct low_cmd struct
        self.low_cmd = LowCmd()  # type: ignore
        self.low_cmd.cmd_type = LowCmd.CMD_TYPE_SERIAL   # type: ignore
        motor_cmd_buf = [
            MotorCmd() for _ in range(self.robot.num_joints)
        ]  # type: ignore
        for i in range(self.robot.num_joints):
            motor_cmd_buf[i].q = 0.0
            motor_cmd_buf[i].dq = 0.0
            motor_cmd_buf[i].tau = 0.0
            motor_cmd_buf[i].kp = 0.0
            motor_cmd_buf[i].kd = 0.0
            motor_cmd_buf[i].weight = 0.0
        self.low_cmd.motor_cmd.extend(motor_cmd_buf)
        self.motor_cmd = self.low_cmd.motor_cmd

        return publisher

    # ------------------------------------------------------------------
    # Portal helpers for the recovery state machine.
    #
    # The legacy entry points `start_custom_mode_conditionally` and
    # `start_rl_gait_conditionally` are preserved as back-compat wrappers
    # that wait for a manufacturer-controller button then call into the
    # underlying helpers. The recovery state machine in
    # `recovery_state_machine.py` calls the helpers directly so it can
    # drive its own button source (our pygame JoystickHandler).
    # ------------------------------------------------------------------

    def _ramp_to_prepare_state(self) -> bool:
        """Ramp the robot from its current pose to the cfg's prepare-state pose
        and switch into ``kCustom`` mode. Idempotent-safe: callable from any
        firmware mode reachable from the current state (DAMP/PREP/CUSTOM).

        Returns True if the ramp completed, False if exit_event was set
        partway through.
        """
        while rclpy.ok() and self.low_cmd_publisher.get_subscription_count() == 0:
            self.logger.info("Waiting for '/joint_ctrl' subscriber, retry in 0.5s")
            time.sleep(0.5)
            if self.exit_event.is_set():
                return False

        self.logger.info("Subscriber found, starting control loop")

        prepare_state = self.robot.cfg.prepare_state
        init_joint_pos = self.synced_state.read()[0]['joint_pos']
        for i in range(self.robot.num_joints):
            self.motor_cmd[i].q = init_joint_pos[i]
            self.motor_cmd[i].kp = float(prepare_state.stiffness[i])
            self.motor_cmd[i].kd = float(prepare_state.damping[i])

        self.low_cmd_publisher.publish(self.low_cmd)
        time.sleep(0.1)

        # change to custom mode — rc-checked. SDK ChangeMode returns int
        # status (0=success). Pre-v2.2 the rc was discarded, which is
        # exactly how the FSM reached PREP_READY while the firmware was
        # still in PROTECT/DAMP/PREP and ignored our joint targets.
        try:
            rc = self.client.ChangeMode(RobotMode.kCustom)
        except Exception as exc:
            self.logger.error(
                "_ramp_to_prepare_state: ChangeMode(kCustom) raised: %s", exc)
            return False
        if rc != 0:
            self.logger.error(
                "_ramp_to_prepare_state: ChangeMode(kCustom) rc=%s; "
                "firmware did NOT enter custom mode. Aborting ramp so "
                "we don't blast joint targets at a non-listening firmware.",
                rc)
            return False
        # Optional GetMode verification — see recovery_state_machine._change_mode
        # for the same pattern.
        if _HAS_GETMODE:
            try:
                gm = GetModeResponse()
                gm_rc = self.client.GetMode(gm)
                if gm_rc == 0 and int(gm.mode) != int(RobotMode.kCustom):
                    self.logger.error(
                        "_ramp_to_prepare_state: ChangeMode(kCustom) rc=0 "
                        "but firmware reports mode=%s; aborting ramp.",
                        int(gm.mode))
                    return False
            except Exception:
                pass  # binding doesn't expose it cleanly — best-effort

        trans = np.linspace(init_joint_pos, prepare_state.joint_pos, num=500)
        start_time = self.timer.get_time()
        for i in range(500):
            if self.exit_event.is_set():
                return False
            for j in range(self.robot.num_joints):
                self.motor_cmd[j].q = trans[i][j]
            self.low_cmd_publisher.publish(self.low_cmd)
            while self.timer.get_time() < start_time + (i + 1) * 0.002:
                time.sleep(0.0002)
        self.logger.info("Custom mode started, initialized with prepare pose")
        return True

    def _spawn_inference(self) -> bool:
        """Fork the inference subprocess and start the policy loop.

        Idempotent: if the subprocess is already alive this just makes sure
        ``policy_run_event`` is set (resuming a pause from the X-press path).
        On a fresh spawn, sets the event before fork so the child starts
        publishing immediately. After a fall-recovery the previous subprocess
        has been terminated and a fresh one is forked here, giving the policy
        a clean ``obs_history`` and ``last_action``.
        """
        # Ensure the run gate is open. Two cases this matters in:
        #   * Fresh spawn — child inherits set() event and runs.
        #   * Idempotent call after a pause — flips event so the existing
        #     subprocess unblocks.
        self.policy_run_event.set()

        if (
            self.inference_process is not None
            and self.inference_process.is_alive()
        ):
            return True

        self.inference_process = mp.Process(
            target=BoosterRobotPortal.inference_process_func,
            args=(self.cfg, self),
            daemon=True,
        )
        self.inference_process.start()
        self.logger.info("Inference process started (pid=%s)",
                         self.inference_process.pid)
        return True

    def _terminate_inference(self, timeout: float = 1.0) -> None:
        """Terminate the inference subprocess hard. Used on the fall-to-
        recovery transition so the post-recovery respawn gets a fresh JIT
        load and a clean policy state. Safe to call when no subprocess is
        alive.
        """
        proc = self.inference_process
        if proc is None:
            return
        if proc.is_alive():
            self.logger.info("Terminating inference subprocess (pid=%s)…", proc.pid)
            proc.terminate()
            proc.join(timeout=timeout)
            if proc.is_alive():
                self.logger.warning(
                    "Inference subprocess did not exit; sending SIGKILL.")
                proc.kill()
                proc.join(timeout=timeout)
        self.inference_process = None

    def _pause_inference(self) -> None:
        """Pause the inference loop without terminating it.

        The X-press emergency-damp path uses this so the policy doesn't have
        to re-load the JIT on the way back. The subprocess sits waiting in
        ``BoosterRobotController.run`` until ``policy_run_event`` is set
        again (or ``exit_event`` is set, which makes it exit).
        """
        self.policy_run_event.clear()
        self.logger.info("Inference paused (policy_run_event cleared).")

    def _resume_inference(self) -> None:
        """Resume a previously-paused inference loop. No-op if the subprocess
        is not alive; in that case the caller should ``_spawn_inference``.
        """
        self.policy_run_event.set()
        self.logger.info("Inference resumed (policy_run_event set).")

    def _wait_until_upright(
        self,
        stable_frames: int | None = None,
        proj_g_z_target: float = -0.95,
        timeout_s: float = 15.0,
    ) -> bool:
        """Block until the IMU reports the robot is reliably upright.

        ``stable_frames`` consecutive low_state ticks must each have
        ``proj_g_z < proj_g_z_target`` before we conclude recovery has
        succeeded. Defaults to ``cfg.booster.recover_stable_frames``.
        Returns False on timeout or exit_event so callers can abort.
        """
        if stable_frames is None:
            stable_frames = self.cfg.booster.recover_stable_frames
        streak = 0
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.exit_event.is_set():
                return False
            if self._proj_g_z < proj_g_z_target:
                streak += 1
                if streak >= stable_frames:
                    return True
            else:
                streak = 0
            time.sleep(0.02)
        self.logger.warning(
            "Upright timeout: proj_g_z=%.3f after %.1fs", self._proj_g_z, timeout_s)
        return False

    def _safe_shutdown(self) -> None:
        """Walk the robot back to a benign mode through legal mode-transitions.

        Replaces the v1 single-hop ``ChangeMode(kWalking)`` which is rejected
        from CUSTOM and from PROTECT. DAMP is reachable from every mode, so
        we go DAMP → PREP → WALK regardless of the starting state.
        """
        self.logger.info("Safe shutdown: kDamping → kPrepare → kWalking")
        for target, label in (
            (RobotMode.kDamping, "kDamping"),
            (RobotMode.kPrepare, "kPrepare"),
            (RobotMode.kWalking, "kWalking"),
        ):
            try:
                rc = self.client.ChangeMode(target)
            except Exception as exc:
                self.logger.error(
                    "Safe shutdown ChangeMode(%s) raised: %s; "
                    "aborting remaining transitions.", label, exc)
                return
            if rc != 0:
                # Don't pretend on Foxglove that we reached this mode —
                # only publish on confirmed success. Bail so we don't
                # try to jump WALK over an un-DAMP'd robot.
                self.logger.error(
                    "Safe shutdown ChangeMode(%s) rc=%s; firmware "
                    "rejected. Stopping shutdown sequence; the robot "
                    "may still be holding its previous mode.", label, rc)
                return
            self._publish_mode(label)
            time.sleep(0.2)

    # --- Back-compat wrappers ------------------------------------------------

    def start_custom_mode_conditionally(self):
        """Legacy entry point: wait for the manufacturer-controller's
        custom-mode button, then ramp into prepare state. Preserved for
        callers that aren't driven by the recovery state machine.
        """
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while not self.exit_event.is_set():
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)
        if self.exit_event.is_set():
            return False
        return self._ramp_to_prepare_state()

    def start_rl_gait_conditionally(self):
        """Legacy entry point: wait for the manufacturer-controller's RL-gait
        button, then spawn the inference subprocess.
        """
        print(f"{self.remoteControlService.get_rl_gait_operation_hint()}")
        while not self.exit_event.is_set():
            if self.remoteControlService.start_rl_gait():
                break
            time.sleep(0.1)
        if self.exit_event.is_set():
            return False
        ok = self._spawn_inference()
        if ok:
            print(f"{self.remoteControlService.get_operation_hint()}")
        return ok

    def cleanup(self) -> None:
        """Clean up resources (idempotent)."""
        if self._cleanup_done:
            return
        self._cleanup_done = True

        self.logger.info("Doing cleanup...")

        # stop threads and processes
        self.is_running = False
        self.exit_event.set()

        # wait for inference process
        if (
            self.inference_process is not None
            and self.inference_process.is_alive()
        ):
            self.logger.info("Waiting for inference process...")
            self.inference_process.join(timeout=2.0)
            if self.inference_process.is_alive():
                self.logger.warning(
                    "Inference process did not stop, terminating...")
                self.inference_process.terminate()
                self.inference_process.join(timeout=1.0)

        # close communications
        try:
            self.remoteControlService.close()
        except Exception as e:
            self.logger.error(f"Error closing remote control: {e}")

        if self.low_cmd_process is not None and self.low_cmd_process.is_alive():
            self.logger.info("Waiting for low cmd publisher process...")
            self.low_cmd_process.join(timeout=2.0)
            if self.low_cmd_process.is_alive():
                self.logger.warning(
                    "Low cmd publisher process did not stop, terminating...")
                self.low_cmd_process.terminate()
                self.low_cmd_process.join(timeout=1.0)

        try:
            thread = self.low_state_thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)

        except Exception as e:
            self.logger.error(f"Error waiting for low state thread: {e}")

        # Diagnostic publisher thread + node teardown (Foxglove layer).
        try:
            diag_thread = getattr(self, "_diag_thread", None)
            if diag_thread is not None and diag_thread.is_alive():
                diag_thread.join(timeout=1.0)
            diag_node = getattr(self, "_diag_node", None)
            if diag_node is not None:
                diag_node.destroy_node()
        except Exception as e:
            self.logger.error(f"Error tearing down diagnostics: {e}")

        # Stop our pygame joystick if we owned it.
        try:
            if self.joystick_handler is not None:
                self.joystick_handler.stop()
        except Exception as e:
            self.logger.error(f"Error stopping joystick handler: {e}")

        if rclpy.ok():
            rclpy.shutdown()

        self.logger.info("Cleanup complete")

        # Print synced metrics summary to stdout
        for name, metric in self.metrics.items():
            stats = metric.compute()
            print(
                f"METRICS {name}: count={stats['count']}, "
                f"freq={stats['freq_hz']:.3f}Hz, "
                f"mean_period={stats['mean_period_s']}, "
                f"min={stats['min_period_s']}, max={stats['max_period_s']}"
            )

    def run(self):
        """Main loop. Two paths:

        * **Joystick-enabled** (``--joystick`` from ``scripts/deploy.py``):
          delegates to :class:`RecoveryStateMachine` for full Y/B/X driven
          locomotion + recovery. The state machine returns when
          ``exit_event`` fires; we then route through legal mode transitions
          via :meth:`_safe_shutdown`.

        * **Legacy** (no joystick): preserves the original linear chain
          (``start_custom_mode_conditionally`` → ``start_rl_gait_conditionally``
          → idle until exit) so existing scripts and the manufacturer's
          controller flow keep working.
        """
        print("Initialization complete.")

        if self.joystick_handler is not None:
            # Imported here to avoid a top-level circular import — the state
            # machine module imports BoosterRobotPortal for typing only.
            # The FSM accepts either backend (JoystickHandler or
            # KeyboardHandler) — both implement the same consume_press /
            # axes / start / stop surface.
            from .recovery_state_machine import RecoveryStateMachine
            try:
                RecoveryStateMachine(self, self.joystick_handler).run()
            except Exception as exc:
                self.logger.error(
                    "Recovery state machine crashed: %s", exc, exc_info=True)
                self.exit_event.set()
        else:
            self._run_legacy()

        # Exit through legal mode transitions regardless of which path ran.
        self._safe_shutdown()

    def _run_legacy(self) -> None:
        """Original startup chain. Used when ``--joystick`` is not set."""
        if not self.start_custom_mode_conditionally():
            print("Custom mode initialization cancelled.")
            return
        if not self.start_rl_gait_conditionally():
            print("RL gait initialization cancelled.")
            return
        # main loop: wait for exit signal
        while self.is_running and not self.exit_event.is_set():
            if self.inference_process is not None:
                if not self.inference_process.is_alive():
                    self.logger.error("Inference process died unexpectedly")
                    self.is_running = False
                    self.exit_event.set()
                    break
            time.sleep(0.1)

    def __enter__(self) -> BoosterRobotPortal:
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()

    @staticmethod
    def inference_process_func(
        cfg: ControllerCfg,
        portal: BoosterRobotPortal,
    ) -> None:
        BoosterRobotController(cfg, portal).run()
        portal.logger.info("Inference process stopped.")


class BoosterRobotController(BaseController):
    '''Controller for Booster robots. Note that this controller runs in a
    separate process forked by BoosterRobotPortal.
    '''
    def __init__(self, cfg: ControllerCfg, portal: BoosterRobotPortal) -> None:
        super().__init__(cfg)
        self.portal = portal

    def update_vel_command(self):
        cmd = self.portal.synced_command.read()[0]

        self.vel_command.lin_vel_x = cmd["vx"] * self.vel_command.vx_max
        self.vel_command.lin_vel_y = cmd["vy"] * self.vel_command.vy_max
        self.vel_command.ang_vel_yaw = cmd["vyaw"] * self.vel_command.vyaw_max

    def update_state(self) -> None:
        state = self.portal.synced_state.read()[0]

        self.robot.data.joint_pos = torch.from_numpy(
            state["joint_pos"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.joint_vel = torch.from_numpy(
            state["joint_vel"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.feedback_torque = torch.from_numpy(
            state["feedback_torque"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.root_pos_w = torch.from_numpy(
            state["root_pos_w"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        rpy_t = torch.from_numpy(state["root_rpy_w"]).to(
            dtype=torch.float32).to(self.robot.data.device)
        self.robot.data.root_quat_w = lab_math.quat_from_euler_xyz(
            *rpy_t
        ).squeeze()
        self.robot.data.root_lin_vel_b = lab_math.quat_apply_inverse(
            self.robot.data.root_quat_w,
            torch.from_numpy(
                state["root_lin_vel_w"]).to(dtype=torch.float32).to(
                    self.robot.data.device)
        )
        self.robot.data.root_ang_vel_b = torch.from_numpy(
            state["root_ang_vel_b"]).to(dtype=torch.float32).to(
                self.robot.data.device)

    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        for i in range(self.robot.num_joints):
            self.portal.motor_cmd[i].q = float(dof_targets[i].item())
            kp_val = float(self.robot.joint_stiffness[i].item())
            kd_val = float(self.robot.joint_damping[i].item())
            self.portal.motor_cmd[i].kp = kp_val
            self.portal.motor_cmd[i].kd = kd_val
        self.portal.low_cmd_publisher.publish(self.portal.low_cmd)

    def stop(self):
        super().stop()
        self.portal.exit_event.set()

    def run(self):
        try:
            self.update_state()
            if self.vel_command is not None:
                self.update_vel_command()
            self.start()
            next_inference_time = self.portal.timer.get_time()
            while self.is_running and not self.portal.exit_event.is_set():
                # Honour the parent's pause gate. When the recovery state
                # machine clears policy_run_event (e.g. on X-press emergency
                # damp), we stop publishing without terminating; the parent
                # later set()s it to resume. We also bail if exit_event fires
                # while we're paused.
                if not self.portal.policy_run_event.is_set():
                    self.portal.policy_run_event.wait(timeout=0.1)
                    next_inference_time = self.portal.timer.get_time()
                    continue

                if self.portal.timer.get_time() < next_inference_time:
                    time.sleep(0.0002)
                    continue
                next_inference_time += self.cfg.policy_dt

                self.update_state()
                if self.vel_command is not None:
                    self.update_vel_command()
                self.portal.metrics["policy_step"].mark()
                dof_targets = self.policy_step()
                self.ctrl_step(dof_targets)
        finally:
            self.portal.exit_event.set()
            if hasattr(self.policy, "flush_policy_log_if_enabled"):
                self.policy.flush_policy_log_if_enabled()
