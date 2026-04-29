import sys
import atexit
from time import sleep
import select
import numpy as np
import torch
import mujoco
import mujoco.viewer
import warnings
from booster_assets import BOOSTER_ASSETS_DIR
from .base_controller import BaseController, ControllerCfg, VelocityCommand

# Suppress pygame warnings before importing joystick handler
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)
from ..utils.joystick_handler import JoystickHandler


def render_velocity_bars(
    vx: float,
    vy: float,
    vyaw: float,
    vx_max: float,
    vy_max: float,
    vyaw_max: float,
    bar_width: int = 30,
) -> str:
    """Render velocity commands as a fixed multi-line text block."""

    def create_bar(value: float, max_val: float, width: int) -> str:
        normalized = value / max_val if max_val != 0 else 0.0
        normalized = max(-1.0, min(1.0, normalized))

        center_pos = width // 2
        fill_positions = int(round(abs(normalized) * center_pos))

        cells = ["░"] * width
        cells[center_pos] = "│"

        if normalized > 0:
            for idx in range(center_pos + 1, min(width, center_pos + 1 + fill_positions)):
                cells[idx] = "█"
        elif normalized < 0:
            start = max(0, center_pos - fill_positions)
            for idx in range(start, center_pos):
                cells[idx] = "█"

        return "".join(cells)

    def render_axis(name: str, value: float, max_val: float) -> list[str]:
        labels = f"{-max_val:>6.1f}{' ' * max(1, bar_width - 12)}{max_val:>6.1f}"
        return [
            f"{name:<5}{value:>7.2f}",
            create_bar(value, max_val, bar_width),
            labels,
        ]

    lines = []
    for axis_name, axis_value, axis_max in (
        ("Vx", vx, vx_max),
        ("Vy", vy, vy_max),
        ("Vyaw", vyaw, vyaw_max),
    ):
        if lines:
            lines.append("")
        lines.extend(render_axis(axis_name, axis_value, axis_max))
    return "\n".join(lines)


class MujocoController(BaseController):
    def __init__(self, cfg: ControllerCfg, joystick_enabled: bool = False):
        # Create MuJoCo model before super().__init__ so that policies
        # constructed during base init can access self.mj_model.
        mjcf_path = self._expand_assets_placeholder_static(cfg.robot.mjcf_path)
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)

        super().__init__(cfg)
        self.joystick_enabled = joystick_enabled
        self._joystick_display_initialized = False

        # Initialize joystick if enabled
        self.joystick_handler = None
        if self.joystick_enabled:
            try:
                self.joystick_handler = JoystickHandler()
                self.joystick_handler.calibrate()
                self.joystick_handler.start()
                print("Joystick control enabled and calibrated.")
            except Exception as e:
                print(f"Failed to initialize joystick: {e}")
                print("Falling back to keyboard input.")
                self.joystick_enabled = False

        self.mj_model.opt.timestep = self.cfg.mujoco.physics_dt
        self.decimation = self.cfg.mujoco.decimation

        self._use_native_pd = False

        # # Apply Python-config kp/kd to position actuator parameters so that
        # # KP_SCALE (and any KP_OVERRIDE) in task configs takes effect with
        # # native PD. The XML's kp/kv values are overridden here at load time.
        # # For a <position> actuator: force = kp*(ctrl-q) - kv*qd, encoded as:
        # #   gainprm[0] = kp, biasprm[1] = -kp, biasprm[2] = -kv.
        # if self._use_native_pd:
        #     kp_arr = self.robot.joint_stiffness.numpy()
        #     kd_arr = self.robot.joint_damping.numpy()
        #     n_act = min(len(self.robot.cfg.joint_names), int(self.mj_model.nu))
        #     for i in range(n_act):
        #         self.mj_model.actuator_gainprm[i, 0] = float(kp_arr[i])
        #         self.mj_model.actuator_biasprm[i, 1] = -float(kp_arr[i])
        #         self.mj_model.actuator_biasprm[i, 2] = -float(kd_arr[i])

        # Fix ground contact: ensure friction is enabled (condim>=3)
        # and set friction to reasonable value for walking.
        ground_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
        )
        if ground_id >= 0:
            self.mj_model.geom_condim[ground_id] = 6
            self.mj_model.geom_friction[ground_id] = self.cfg.mujoco.ground_friction

        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Initialize only the robot's qpos slice (root + 22 DoF). Composite
        # scenes that add extra free-jointed bodies (e.g. a ball) leave those
        # qpos entries at the mj_resetData defaults; their initial state is
        # the responsibility of whichever task owns them, typically set in
        # the policy's reset() hook.
        robot_qpos_init = np.concatenate(
            [
                np.array(self.cfg.mujoco.init_pos, dtype=np.float32),
                np.array(self.cfg.mujoco.init_quat, dtype=np.float32),
                self.robot.default_joint_pos.numpy(),
            ]
        )
        self.mj_data.qpos[: len(robot_qpos_init)] = robot_qpos_init
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # render a second "ghost" robot (kinematic only) without
        # modifying the MuJoCo XML. This uses a second MjData to compute FK from
        # generalized coordinates and draws a duplicated set of geoms via
        # viewer.user_scn.
        self._ghost_mj_data = mujoco.MjData(self.mj_model)
        # Keep ghost initialized to the current simulated pose so it is valid
        # even before any policy calls set_reference_qpos().
        self._ghost_mj_data.qpos[:] = self.mj_data.qpos
        self._ghost_mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self._ghost_mj_data)
        self._ghost_rgba = np.array(
            self.cfg.mujoco.ghost_rgba, dtype=np.float32)
        self._ghost_scene_option = mujoco.MjvOption()

        # Reference qpos can be set explicitly by the policy.
        self._reference_qpos: np.ndarray | None = None

        # Logging buffers are initialized lazily in log_states().
        self._states: dict[str, list[np.ndarray]] | None = None
        self._last_log_flush_step: int = 0
        # Ensure logs are flushed if the process exits unexpectedly.
        atexit.register(self._flush_logged_states)

    def start(self):
        # Clear reference; policy.reset() may set a fresh one.
        self._reference_qpos = None
        return super().start()

    def render_reference_robot(
        self,
        viewer,
        # mj_data: mujoco.MjData,
        *,
        rgba: np.ndarray | None = None,
    ) -> None:
        """Render a kinematic robot pose into viewer.user_scn using mj_data."""
        mujoco.mjv_updateScene(
            self.mj_model,
            self._ghost_mj_data,
            self._ghost_scene_option,
            None,
            viewer.cam,
            int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
            viewer.user_scn,
        )
        if rgba is None:
            rgba = self._ghost_rgba

        for i in range(viewer.user_scn.ngeom):
            viewer.user_scn.geoms[i].rgba[:] = rgba

    def _render_command_arrow(self, viewer) -> None:
        """Draw a 3-D arrow showing the current command velocity in the viewer."""
        policy = self.policy
        # Collect (dir_x, dir_y, speed) from whichever command interface is active.
        if hasattr(policy, 'tar_dir') and hasattr(policy, 'tar_speed'):
            dir_x = float(policy.tar_dir[0])
            dir_y = float(policy.tar_dir[1])
            speed = float(policy.tar_speed) if not hasattr(policy.tar_speed, '__len__') else float(policy.tar_speed[0])
        elif self.vel_command is not None:
            cmd = self.vel_command
            speed = float(np.hypot(cmd.lin_vel_x, cmd.lin_vel_y))
            if speed > 1e-6:
                dir_x, dir_y = cmd.lin_vel_x / speed, cmd.lin_vel_y / speed
            else:
                dir_x, dir_y = 1.0, 0.0
        else:
            return

        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return

        root_pos = self.mj_data.qpos[:3].copy()
        arrow_z = root_pos[2]          # draw at root height
        arrow_from = np.array([root_pos[0], root_pos[1], arrow_z])
        arrow_to   = np.array([root_pos[0] + dir_x * speed,
                                root_pos[1] + dir_y * speed,
                                arrow_z])

        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.array([0.1, 0.8, 0.1, 0.9], dtype=np.float32),  # green
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.025,
                             arrow_from, arrow_to)
        viewer.user_scn.ngeom += 1

    def set_reference_qpos(
        self,
        qpos: np.ndarray | torch.Tensor | None,
    ) -> None:
        """Set the reference generalized coordinates (qpos) for ghost rendering.

        Policies should call this each step (or whenever updated). Pass None to
        clear the reference.
        """
        if qpos is None:
            self._reference_qpos = None
            return

        if isinstance(qpos, torch.Tensor):
            qpos_np = qpos.detach().cpu().numpy()
        else:
            qpos_np = np.asarray(qpos)

        qpos_np = qpos_np.astype(np.float32, copy=False).reshape(-1)
        if qpos_np.shape[0] != int(self.mj_model.nq):
            raise ValueError(
                f"reference qpos must have shape (nq,), got {qpos_np.shape} (nq={int(self.mj_model.nq)})"
            )
        self._reference_qpos = qpos_np.copy()
        # FK + offset
        self._ghost_mj_data.qpos[:] = self._reference_qpos
        self._ghost_mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self._ghost_mj_data)

    @staticmethod
    def _expand_assets_placeholder_static(path: str) -> str:
        """Replace {BOOSTER_ASSETS_DIR} placeholder in a path string."""
        try:
            return path.replace("{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR))
        except Exception:
            return path

    def _expand_assets_placeholder(self, path: str) -> str:
        """Replace {BOOSTER_ASSETS_DIR} placeholder in a path string.
        """
        return self._expand_assets_placeholder_static(path)

    def update_vel_command(self):
        cmd: VelocityCommand = self.vel_command

        # Use joystick if available
        if self.joystick_enabled and self.joystick_handler:
            vx, vy, vyaw = self.joystick_handler.get_velocities(
                cmd.vx_max, cmd.vy_max, cmd.vyaw_max
            )
            cmd.lin_vel_x = vx
            cmd.lin_vel_y = vy
            cmd.ang_vel_yaw = vyaw
            return

        # Fallback to keyboard input
        if select.select([sys.stdin], [], [], 0)[0]:
            try:
                parts = sys.stdin.readline().strip().split()
                if len(parts) == 3:
                    (cmd.lin_vel_x, cmd.lin_vel_y, cmd.ang_vel_yaw) = map(float, parts)
                    print(
                        f"Updated command to: x={cmd.lin_vel_x},"
                        f"y={cmd.lin_vel_y}, yaw={cmd.ang_vel_yaw}\n"
                        "Set command (x, y, yaw): ",
                        end="",
                    )
                else:
                    raise ValueError
            except ValueError:
                print(
                    "Invalid input. Enter three numeric values. "
                    "Set command (x, y, yaw): ",
                    end="",
                )

    def update_steering_command(self):
        """Read steering commands (vx vy vyaw) from joystick or stdin."""
        import torch, math

        def _vel_to_policy(vx, vy, vyaw):
            speed = math.sqrt(vx**2 + vy**2)
            if speed > 1e-6:
                dir_x, dir_y = vx / speed, vy / speed
            else:
                dir_x, dir_y = 1.0, 0.0
            self.policy.tar_dir = torch.tensor([dir_x, dir_y], dtype=torch.float32)
            self.policy.tar_speed = torch.tensor([speed], dtype=torch.float32)
            if hasattr(self.policy, 'tar_omega'):
                self.policy.tar_omega = torch.tensor([vyaw], dtype=torch.float32)

        if self.joystick_enabled and self.joystick_handler:
            joystick_cfg = self.cfg.steering_joystick_command
            vx, vy, vyaw = self.joystick_handler.get_velocities(
                joystick_cfg.vx_max,
                joystick_cfg.vy_max,
                joystick_cfg.vyaw_max,
            )
            _vel_to_policy(vx, vy, vyaw)
            return

        if select.select([sys.stdin], [], [], 0)[0]:
            try:
                parts = sys.stdin.readline().strip().split()
                if len(parts) == 3:
                    vx, vy, vyaw = map(float, parts)
                    _vel_to_policy(vx, vy, vyaw)
                    speed = math.sqrt(vx**2 + vy**2)
                    print(
                        f"Updated: vx={vx}, vy={vy}, vyaw={vyaw} "
                        f"(speed={speed:.2f})\n"
                        "Set command (vx vy vyaw): ",
                        end="",
                    )
                else:
                    raise ValueError
            except (ValueError, AttributeError):
                print(
                    "Invalid input. Enter 3 values: vx vy vyaw\n"
                    "Set command (vx vy vyaw): ",
                    end="",
                )

    def update_state(self) -> None:
        # Slice only the robot's actuated dofs (composite scenes may have
        # extra free-jointed bodies past this slice; they're not part of the
        # robot state).
        n_dof = len(self.robot.cfg.joint_names)
        dof_pos = self.mj_data.qpos.astype(np.float32)[7 : 7 + n_dof]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6 : 6 + n_dof]
        dof_torque = self.mj_data.qfrc_actuator[6 : 6 + n_dof].astype(np.float32)

        base_pos_w = self.mj_data.qpos.astype(np.float32)[:3]
        base_quat_wxyz = self.mj_data.qpos.astype(np.float32)[3:7]
        # MuJoCo free joint: qvel[0:3] is linear vel in world frame,
        # qvel[3:6] is angular vel in body frame.
        base_lin_vel_w = self.mj_data.qvel.astype(np.float32)[:3]
        base_ang_vel_b = self.mj_data.qvel.astype(np.float32)[3:6]

        # Convert world-frame linear velocity to body frame using
        # inverse (conjugate) rotation of the base quaternion.
        w, x, y, z = base_quat_wxyz
        # Rotate by conjugate quat (w, -x, -y, -z) in wxyz convention:
        # v_body = q_conj * v_world * q
        qv = np.array([-x, -y, -z], dtype=np.float32)
        t = 2.0 * np.cross(qv, base_lin_vel_w)
        base_lin_vel_b = base_lin_vel_w + w * t + np.cross(qv, t)

        self.robot.data.joint_pos = torch.from_numpy(
            dof_pos).to(self.robot.data.device)
        self.robot.data.joint_vel = torch.from_numpy(
            dof_vel).to(self.robot.data.device)
        self.robot.data.feedback_torque = torch.from_numpy(
            dof_torque).to(self.robot.data.device)
        self.robot.data.root_pos_w = torch.from_numpy(
            base_pos_w).to(self.robot.data.device)
        self.robot.data.root_quat_w = torch.from_numpy(
            base_quat_wxyz).to(self.robot.data.device)
        self.robot.data.root_lin_vel_b = torch.from_numpy(
            base_lin_vel_b).to(self.robot.data.device)
        self.robot.data.root_ang_vel_b = torch.from_numpy(
            base_ang_vel_b).to(self.robot.data.device)

    def log_states(self, dof_targets: np.ndarray) -> None:
        if self.cfg.mujoco.log_states is not None:
            if self._states is None:
                self._states = {
                    'step': [],
                    'sim_time_s': [],
                    'root_pos_w': [],
                    'root_quat_w': [],
                    'root_lin_vel_w': [],
                    'root_lin_vel_b': [],
                    'root_ang_vel_b': [],
                    'joint_pos': [],
                    'joint_vel': [],
                    'joint_torque': [],
                    'ctrl_applied': [],
                    'dof_targets': [],
                }
            base_pos_w = self.mj_data.qpos.astype(np.float32)[:3]
            base_quat_wxyz = self.mj_data.qpos.astype(np.float32)[3:7]
            base_lin_vel_w = self.mj_data.qvel.astype(np.float32)[:3]
            base_ang_vel_b = self.mj_data.qvel.astype(np.float32)[3:6]

            # Convert world-frame linear velocity to body frame using
            # inverse (conjugate) rotation of the base quaternion.
            w, x, y, z = base_quat_wxyz
            qv = np.array([-x, -y, -z], dtype=np.float32)
            t = 2.0 * np.cross(qv, base_lin_vel_w)
            base_lin_vel_b = base_lin_vel_w + w * t + np.cross(qv, t)

            dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
            dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
            dof_torque = self.mj_data.qfrc_actuator[6:].astype(np.float32)
            ctrl_applied = self.mj_data.ctrl.astype(np.float32).copy()

            self._states['step'].append(np.array([self._step_count], dtype=np.float32))
            self._states['sim_time_s'].append(np.array([self._elapsed_s], dtype=np.float32))
            self._states['root_pos_w'].append(base_pos_w)
            self._states['root_quat_w'].append(base_quat_wxyz)
            self._states['root_lin_vel_w'].append(base_lin_vel_w)
            self._states['root_lin_vel_b'].append(base_lin_vel_b)
            self._states['root_ang_vel_b'].append(base_ang_vel_b)
            self._states['joint_pos'].append(dof_pos)
            self._states['joint_vel'].append(dof_vel)
            self._states['joint_torque'].append(dof_torque)
            self._states['ctrl_applied'].append(ctrl_applied)
            self._states['dof_targets'].append(dof_targets)
            if len(self._states['root_pos_w']) % 100 == 0:
                self._flush_logged_states()

    def _flush_logged_states(self) -> None:
        if self.cfg.mujoco.log_states is None or self._states is None:
            return
        num_entries = len(self._states['root_pos_w'])
        if num_entries == 0 or num_entries == self._last_log_flush_step:
            return
        stacked = {k: np.stack(v) for k, v in self._states.items()}
        np.savez(f'{self.cfg.mujoco.log_states}.npz', **stacked)
        self._last_log_flush_step = num_entries
        print(
            f'saved {self.cfg.mujoco.log_states}.npz '
            f'with {num_entries} control steps'
        )

    def ctrl_step(self, dof_targets: torch.Tensor):
        dof_targets = dof_targets.cpu().numpy()  # type: ignore
        self.log_states(dof_targets)
        if self.vel_command is not None:
            self.update_vel_command()
        elif hasattr(self.policy, 'tar_dir'):
            self.update_steering_command()

        if self._use_native_pd:
            # Native position actuator: ctrl = target position.
            # MuJoCo computes force = kp*(ctrl-pos) - kd*vel internally
            # at each physics sub-step (closer to PhysX implicit PD).
            self.mj_data.ctrl[:] = dof_targets
            for i in range(self.decimation):
                mujoco.mj_step(self.mj_model, self.mj_data)
        else:
            # Manual PD computation (legacy path).
            # Slice only the robot's actuated dofs: root free joint takes the
            # first 7 qpos / 6 qvel slots, then 22 named joints. Composite
            # scenes (e.g. ball after the robot) live past this slice and
            # must not be touched here.
            n_dof = len(self.robot.cfg.joint_names)
            dof_pos = self.mj_data.qpos.astype(np.float32)[7 : 7 + n_dof]
            dof_vel = self.mj_data.qvel.astype(np.float32)[6 : 6 + n_dof]
            kp = self.robot.joint_stiffness.numpy()
            kd = self.robot.joint_damping.numpy()
            ctrl_limit = self.robot.effort_limit.numpy()
            for i in range(self.decimation):
                self.mj_data.ctrl[:n_dof] = np.clip(
                    kp * (dof_targets - dof_pos) - kd * dof_vel,
                    -ctrl_limit,
                    ctrl_limit,
                )
                mujoco.mj_step(self.mj_model, self.mj_data)
                dof_pos = self.mj_data.qpos.astype(np.float32)[7 : 7 + n_dof]
                dof_vel = self.mj_data.qvel.astype(np.float32)[6 : 6 + n_dof]

    def _make_key_callback(self):
        """Return a GLFW key callback that sets a flag on Backspace.

        The viewer calls mj_resetData *after* the callback returns, so we
        cannot re-spawn objects inside the callback itself. Instead we set a
        flag and handle it in the main loop on the next iteration, after the
        viewer's reset has already been applied to mj_data.
        """
        GLFW_KEY_BACKSPACE = 259
        self._pending_reset = False

        def key_callback(keycode):
            if keycode == GLFW_KEY_BACKSPACE:
                self._pending_reset = True

        return key_callback

    def _apply_pending_reset(self) -> None:
        """Re-initialize robot qpos and call policy.reset() after a viewer reset."""
        robot_qpos_init = np.concatenate([
            np.array(self.cfg.mujoco.init_pos, dtype=np.float32),
            np.array(self.cfg.mujoco.init_quat, dtype=np.float32),
            self.robot.default_joint_pos.numpy(),
        ])
        self.mj_data.qpos[:len(robot_qpos_init)] = robot_qpos_init
        self.mj_data.qvel[:] = 0.0
        self.policy.reset()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data,
                    key_callback=self._make_key_callback()) as viewer:

                self.viewer = viewer
                viewer.cam.elevation = -20
                if not self.joystick_enabled:
                    if self.vel_command is not None:
                        print("\nSet command (x, y, yaw): ", end="")
                    elif hasattr(self.policy, 'tar_dir'):
                        print("\nSet command (vx vy vyaw): ", end="")
                else:
                    print("\nJoystick Control Active:")
                    print("  Left Stick: ↑↓ Forward/Back | ←→ Left/Right")
                    if self.vel_command is not None:
                        print("  Right Stick: ←→ Turn Left/Right")
                    elif hasattr(self.policy, 'tar_dir'):
                        print("  Right Stick: ←→ Yaw Left/Right")
                    print("  Use controller to teleoperate robot.")
                    print("")
                    print("")
                    print("")
                    print("")
                    print("")
                    print("")
                    print("")
                    print("")
                    print("")
                    print("")
                    print("")
                    self._joystick_display_initialized = True
                self.update_state()
                self.start()
                _prev_time = self.mj_data.time
                while viewer.is_running() and self.is_running:
                    sleep(self.cfg.mujoco.physics_dt * self.cfg.mujoco.decimation)
                    _cur_time = self.mj_data.time
                    if _cur_time < _prev_time or (self._pending_reset and _cur_time < 1e-9):
                        self._pending_reset = False
                        self._apply_pending_reset()
                    _prev_time = _cur_time
                    self.update_state()
                    dof_targets = self.policy_step()
                    self.ctrl_step(dof_targets)

                    # Display velocity bars if joystick is enabled
                    if self.joystick_enabled:
                        if self._joystick_display_initialized:
                            print("\033[11A", end="")
                        if self.vel_command is not None:
                            print(render_velocity_bars(
                                self.vel_command.lin_vel_x,
                                -self.vel_command.lin_vel_y,
                                -self.vel_command.ang_vel_yaw,
                                self.vel_command.vx_max,
                                self.vel_command.vy_max,
                                self.vel_command.vyaw_max
                            ))
                        elif hasattr(self.policy, 'tar_dir'):
                            speed = float(self.policy.tar_speed.item()) if hasattr(self.policy.tar_speed, 'item') else float(self.policy.tar_speed)
                            omega = float(self.policy.tar_omega.item()) if hasattr(self.policy.tar_omega, 'item') else float(self.policy.tar_omega)
                            dir_x = float(self.policy.tar_dir[0].item()) if hasattr(self.policy.tar_dir[0], 'item') else float(self.policy.tar_dir[0])
                            dir_y = float(self.policy.tar_dir[1].item()) if hasattr(self.policy.tar_dir[1], 'item') else float(self.policy.tar_dir[1])

                            vx = speed * dir_x
                            vy = speed * dir_y
                            vyaw = omega

                            joystick_cfg = self.cfg.steering_joystick_command
                            print(render_velocity_bars(
                                vx,
                                -vy,
                                -vyaw,
                                joystick_cfg.vx_max,
                                joystick_cfg.vy_max,
                                joystick_cfg.vyaw_max,
                            ))

                    viewer.user_scn.ngeom = 0
                    if self.cfg.mujoco.visualize_reference_ghost:
                        # Render kinematic "ghost" robot from generalized coordinates.
                        self.render_reference_robot(
                            viewer,
                            rgba=self._ghost_rgba,
                        )

                    self._render_command_arrow(viewer)

                    self.viewer.cam.lookat[:] = self.mj_data.qpos.astype(np.float32)[0:3]
                    self.viewer.sync()
        finally:
            self._flush_logged_states()
            if hasattr(self.policy, 'flush_policy_log_if_enabled'):
                self.policy.flush_policy_log_if_enabled()
            if self.joystick_handler:
                self.joystick_handler.stop()
