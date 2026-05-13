"""Recovery state machine for the t1_walk locomotion + GetUp loop.

Drives the firmware mode transitions and inference-subprocess lifecycle so
that pressing **Y** on the operator's pygame joystick enters locomotion,
**B** triggers a full fall-recovery cycle (DAMP → PREP → GetUp → PREP →
restart inference), and **X** is an emergency damp.

The FSM lives in the parent process of :class:`BoosterRobotPortal`. It
consumes button presses from :class:`JoystickHandler` (rising-edge via
``consume_press``) and the IMU-derived ``_fall_event`` flag from the
portal; it drives :class:`B1LocoClient` mode changes and the portal's
``_spawn_inference`` / ``_terminate_inference`` / ``_pause_inference``
helpers.

See ``docs/RECOVERY_INTEGRATION_PLAN.md`` §6 for the full state graph and
§13 for the v2.1 design decisions baked into this module (notably: no
auto-recovery — fall_event only logs, the operator commits via B-press;
``policy_run_event`` is the X-press pause primitive; B-press / fall does
a full terminate-and-respawn so the resumed policy gets a fresh JIT and
``obs_history``).
"""
from __future__ import annotations

import enum
import logging
import time
from typing import TYPE_CHECKING

from booster_robotics_sdk_python import RobotMode  # type: ignore

from ..utils.joystick_handler import (
    JoystickHandler,
    BUTTON_B,
    BUTTON_X,
    BUTTON_Y,
)

if TYPE_CHECKING:
    from .booster_robot_controller import BoosterRobotPortal
    from ..utils.keyboard_handler import KeyboardHandler

    InputHandler = JoystickHandler | KeyboardHandler
else:
    # At runtime the FSM doesn't care which backend it has — it only
    # touches `.consume_press(button_id)`. Both handlers expose the same
    # signature, so the type hint below is for IDE help only.
    InputHandler = JoystickHandler


class State(enum.Enum):
    """States in the recovery FSM. See docs §6 for the diagram."""

    IDLE = "idle"                 # cold start; nothing publishing
    PREP_READY = "prep_ready"     # ramped to prepare pose, in CUSTOM
    RUNNING = "running"           # RL policy publishing on joint_ctrl
    FALLEN = "fallen"             # fall trigger received; tearing down
    RECOVERING = "recovering"     # firmware GetUp in progress
    STAGED = "staged"             # back in PREP after recovery; waits Y
    EXITING = "exiting"           # terminal; portal will safe-shutdown


class RecoveryStateMachine:
    """Finite state machine for locomotion + recovery on the real T1.

    Parameters
    ----------
    portal:
        The :class:`BoosterRobotPortal` instance that owns ``B1LocoClient``,
        the inference subprocess, the IMU stream, and the helper methods
        the FSM dispatches into. Held by reference; not copied.
    joystick:
        Our pygame :class:`JoystickHandler`. Must already be calibrated
        and started — the FSM only calls ``consume_press`` on it.
    """

    POLL_INTERVAL_S = 0.05  # 20 Hz outer loop; transitions are cheap

    def __init__(
        self,
        portal: "BoosterRobotPortal",
        joystick: "InputHandler",
    ) -> None:
        self.portal = portal
        # Despite the attribute name, this can be either a JoystickHandler
        # or a KeyboardHandler — they share the consume_press/axes API the
        # FSM needs. The portal picks one at __init__ time.
        self.joystick = joystick
        self.client = portal.client
        self.logger = logging.getLogger(__name__)
        self.state = State.IDLE

        # Per-state "have we shown the hint yet?" flags. We re-arm them on
        # every state transition so re-entering a state re-prints its hint.
        self._idle_hinted = False
        self._prep_hinted = False
        self._staged_hinted = False

    # ------------------------------------------------------------------
    # Driver loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Drive the FSM until ``portal.exit_event`` fires or we hit EXITING.

        The caller (``BoosterRobotPortal.run``) is responsible for the safe
        shutdown afterwards — this method does not call ``_safe_shutdown``.
        """
        self.logger.info("Recovery state machine started.")
        # Publish the initial state so Foxglove panels light up immediately
        # rather than waiting for the first transition.
        self.portal._publish_recovery_state(self.state.value)
        while not self.portal.exit_event.is_set():
            handler = {
                State.IDLE: self._tick_idle,
                State.PREP_READY: self._tick_prep_ready,
                State.RUNNING: self._tick_running,
                State.FALLEN: self._tick_fallen,
                State.RECOVERING: self._tick_recovering,
                State.STAGED: self._tick_staged,
            }.get(self.state)
            if handler is None:
                if self.state is State.EXITING:
                    break
                self.logger.error("Unknown state %s; exiting", self.state)
                break

            handler()

            # Steady polling cadence between ticks; transitions skip the
            # sleep so we don't add latency to chained mode changes.
            time.sleep(self.POLL_INTERVAL_S)

        self.logger.info("Recovery state machine exiting (final state=%s).",
                         self.state.value)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_state(self, new_state: State) -> None:
        if new_state is self.state:
            return
        self.logger.info(
            "Recovery state %s → %s", self.state.value, new_state.value)
        self.state = new_state
        # Reset all hint flags so re-entering a state shows its hint again.
        self._idle_hinted = False
        self._prep_hinted = False
        self._staged_hinted = False
        # Foxglove diagnostic.
        self.portal._publish_recovery_state(new_state.value)

    def _change_mode(self, mode: "RobotMode", label: str) -> bool:
        """Wrap ``client.ChangeMode`` so we publish ``/nomadz/mode`` and log
        any failure consistently. Returns True on success.
        """
        try:
            self.client.ChangeMode(mode)
        except Exception as exc:
            self.logger.error("ChangeMode(%s) failed: %s", label, exc)
            return False
        self.portal._publish_mode(label)
        return True

    def _consume(self, button_id: int, label: str) -> bool:
        """``consume_press`` + emit a ``/nomadz/button_events`` ROS message
        whenever a press is actually consumed. Returns True on consumption.
        """
        if self.joystick.consume_press(button_id):
            self.portal._publish_button_event(label)
            return True
        return False

    # ------------------------------------------------------------------
    # Per-state handlers
    # ------------------------------------------------------------------

    def _tick_idle(self) -> None:
        """Cold start. Y press ramps to prepare and enters PREP_READY."""
        if not self._idle_hinted:
            print("\n[recovery] IDLE — press Y on the operator joystick to "
                  "ramp into prepare state and enter custom mode.")
            self._idle_hinted = True

        if self._consume(BUTTON_Y, "Y"):
            self.logger.info("Y pressed; ramping to prepare state")
            ok = self.portal._ramp_to_prepare_state()
            if not ok:
                self._set_state(State.EXITING)
                return
            self.portal._publish_mode("kCustom")  # _ramp does the ChangeMode
            self._set_state(State.PREP_READY)

    def _tick_prep_ready(self) -> None:
        """Custom mode + prepare pose held; Y press starts the RL policy."""
        if not self._prep_hinted:
            print("[recovery] PREP_READY — press Y to start the RL policy.")
            self._prep_hinted = True

        if self._consume(BUTTON_Y, "Y"):
            self.portal._spawn_inference()
            print("[recovery] RUNNING — RL policy active.")
            print("[recovery]   Joystick axes drive vx/vy/vyaw.")
            print("[recovery]   B = recovery cycle  |  X = emergency damp")
            # Clear any fall flag accumulated during the ramp so we don't
            # immediately re-trigger.
            self.portal._fall_streak = 0
            self.portal._fall_event.clear()
            self._set_state(State.RUNNING)

    def _tick_running(self) -> None:
        """RL policy is publishing; watch for B/X/fall.

        Per §13.6 we do NOT auto-act on ``_fall_event`` — the detector logs
        and we keep the flag visible (Foxglove can render it), but the
        operator must press B to commit to recovery.
        """
        # Operator-initiated recovery (covers both real falls and intentional
        # tests where the policy fell over).
        if self._consume(BUTTON_B, "B"):
            self.logger.info("B pressed in RUNNING; entering FALLEN")
            self.portal._terminate_inference()
            self._set_state(State.FALLEN)
            return

        # Emergency damp. Pause (don't terminate) so the next entry into
        # RUNNING via PREP_READY can reuse the same JIT load — though the
        # IDLE → PREP_READY transition will re-ramp the prepare pose first.
        if self._consume(BUTTON_X, "X"):
            self.logger.info("X pressed; emergency damp")
            self.portal._pause_inference()
            self._change_mode(RobotMode.kDamping, "kDamping")
            self.portal._fall_streak = 0
            self.portal._fall_event.clear()
            self._set_state(State.IDLE)
            return

        # Fall detected: log only. Keep the flag set; operator decides.

    def _tick_fallen(self) -> None:
        """PROTECT/CUSTOM → DAMP → PREP. Setup for ``GetUp()``."""
        if not self._change_mode(RobotMode.kDamping, "kDamping"):
            self._set_state(State.EXITING)
            return
        time.sleep(0.2)
        if not self._change_mode(RobotMode.kPrepare, "kPrepare"):
            self._set_state(State.EXITING)
            return
        time.sleep(0.2)
        self._set_state(State.RECOVERING)

    def _tick_recovering(self) -> None:
        """Run firmware ``GetUp``, wait for stable upright, walk to PREP."""
        print("[recovery] RECOVERING — firmware GetUp in progress…")
        try:
            rc = self.client.GetUp()
        except Exception as exc:
            self.logger.error("GetUp() raised: %s", exc)
            # Publish a sentinel rc so Foxglove sees the failure.
            self.portal._publish_getup_return_code(-1)
            self._set_state(State.EXITING)
            return
        self.logger.info("GetUp() return code = %s", rc)
        self.portal._publish_getup_return_code(int(rc))
        if rc != 0:
            self.logger.error(
                "GetUp() returned non-zero (%s); aborting recovery", rc)
            self._set_state(State.EXITING)
            return
        # Firmware will land us in WALK; advertise that for the dashboard.
        self.portal._publish_mode("kWalking")

        if not self.portal._wait_until_upright():
            self.logger.error(
                "Did not see stable upright IMU after GetUp; aborting")
            self._set_state(State.EXITING)
            return

        # Firmware lands us in WALK; CUSTOM is unreachable directly from
        # WALK so we must hop via PREP.
        if not self._change_mode(RobotMode.kPrepare, "kPrepare"):
            self._set_state(State.EXITING)
            return
        time.sleep(0.2)

        # Recovery complete — clear fall flags so the next RUNNING tick
        # doesn't see stale state.
        self.portal._fall_streak = 0
        self.portal._fall_event.clear()
        self._set_state(State.STAGED)

    def _tick_staged(self) -> None:
        """Post-recovery; in PREP. Y press → CUSTOM + fresh inference subprocess."""
        if not self._staged_hinted:
            print("[recovery] STAGED — recovery complete. Press Y to resume "
                  "locomotion (fresh inference subprocess will be spawned).")
            self._staged_hinted = True

        if self._consume(BUTTON_Y, "Y"):
            if not self._change_mode(RobotMode.kCustom, "kCustom"):
                self._set_state(State.EXITING)
                return
            # Fresh subprocess: previous one was terminated in FALLEN.
            self.portal._spawn_inference()
            self._set_state(State.RUNNING)
