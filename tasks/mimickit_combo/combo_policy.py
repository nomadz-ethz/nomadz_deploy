"""Combined steering + dribbling policy with a mode state machine.

Wraps the two existing JitPolicy task wrappers (``MimicKitPolicy`` for
steering, ``DribblingPolicy`` for dribbling) behind one ``Policy`` so a
single MuJoCo scene (K1 + ball) can be driven in either mode and switched
live:

    STEERING  ⇄  DRIBBLING

Switch inputs (any of):
    * joystick Y button (rising edge)
    * viewer key 'M' (forwarded by MujocoController via the on_key hook)

Extras:
    * joystick A button / viewer key 'B' re-spawns the ball next to the robot
    * the controller's joystick HUD shows the active mode via ``mode_name``

Transition handling — both sub-policies emit absolute joint-position
targets in the same joint set, so a switch is mostly benign; to avoid a
target step-discontinuity we:
    1. seed the incoming policy's EMA state with the last commanded targets
    2. linearly blend from the last commanded targets to the incoming
       policy's output over ``switch_blend_steps`` control steps (~0.5 s)
The dribbling policy's ball-position history is kept warm in steering mode
by calling its ``compute_observation()`` every step (one history push per
control step either way), so entering DRIBBLING never starts blind.

Command plumbing: the controller's steering joystick path sets ``tar_dir``,
``tar_speed`` and ``tar_omega`` on this object; setters forward to both
sub-policies (omega only exists on steering — the dribbling trainer has no
yaw command). Each sub-policy applies its own speed clamp.
"""
from __future__ import annotations

from dataclasses import MISSING

import torch

from nomadz_deploy.controllers.base_controller import BaseController, Policy
from nomadz_deploy.controllers.controller_cfg import PolicyCfg
from nomadz_deploy.utils.isaaclab.configclass import configclass

from tasks.mimickit_steering.mimickit_policy import MimicKitPolicy, MimicKitPolicyCfg
from tasks.mimickit_dribbling.dribbling_policy import DribblingPolicy, DribblingPolicyCfg


MODE_STEERING = "steering"
MODE_DRIBBLING = "dribbling"

# Xbox-360 pad button indices (pygame).
BUTTON_A = 0
BUTTON_Y = 3

# GLFW keycodes forwarded from the MuJoCo viewer.
KEY_M = 77
KEY_B = 66


class ComboPolicy(Policy):
    """State machine over a steering and a dribbling sub-policy."""

    def __init__(self, cfg: "ComboPolicyCfg", controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        self._steering = MimicKitPolicy(cfg.steering, controller)
        self._dribbling = DribblingPolicy(cfg.dribbling, controller)

        if cfg.initial_mode not in (MODE_STEERING, MODE_DRIBBLING):
            raise ValueError(
                f"initial_mode must be '{MODE_STEERING}' or '{MODE_DRIBBLING}', "
                f"got {cfg.initial_mode!r}"
            )
        self._mode = cfg.initial_mode

        self._blend_steps = int(cfg.switch_blend_steps)
        self._blend_step = self._blend_steps  # start with no blend pending
        self._blend_from: torch.Tensor | None = None
        self._last_action_real = self.robot.default_joint_pos.clone()

    # --- Mode state machine ---

    @property
    def mode_name(self) -> str:
        return self._mode.upper()

    @property
    def active_policy(self):
        return self._dribbling if self._mode == MODE_DRIBBLING else self._steering

    def toggle_mode(self) -> None:
        self._mode = (
            MODE_STEERING if self._mode == MODE_DRIBBLING else MODE_DRIBBLING
        )
        incoming = self.active_policy
        # Seed the incoming policy's EMA state with the last commanded
        # targets (converted to its policy joint order) and start the blend.
        incoming._previous_policy_targets = self._last_action_real[
            incoming._real_to_policy_idx
        ].clone()
        self._blend_from = self._last_action_real.clone()
        self._blend_step = 0

    def respawn_ball(self) -> None:
        self._dribbling._spawn_ball()

    def on_key(self, keycode: int) -> None:
        if keycode == KEY_M:
            self.toggle_mode()
        elif keycode == KEY_B:
            self.respawn_ball()

    def _poll_joystick_buttons(self) -> None:
        handler = getattr(self.controller, "joystick_handler", None)
        if handler is None:
            return
        if handler.consume_button_press(BUTTON_Y):
            self.toggle_mode()
        if handler.consume_button_press(BUTTON_A):
            self.respawn_ball()

    # --- Command properties (controller feeds these from joystick/stdin) ---

    @property
    def tar_dir(self) -> torch.Tensor:
        return self.active_policy.tar_dir

    @tar_dir.setter
    def tar_dir(self, value) -> None:
        self._steering.tar_dir = value
        self._dribbling.tar_dir = value

    @property
    def tar_speed(self) -> torch.Tensor:
        return self.active_policy.tar_speed

    @tar_speed.setter
    def tar_speed(self, value) -> None:
        self._steering.tar_speed = value
        self._dribbling.tar_speed = value

    @property
    def tar_omega(self) -> torch.Tensor:
        if self._mode == MODE_STEERING:
            return self._steering.tar_omega
        return torch.zeros(1, dtype=torch.float32)

    @tar_omega.setter
    def tar_omega(self, value) -> None:
        # Dribbling has no yaw command; only the steering policy consumes it.
        self._steering.tar_omega = value

    # --- Policy interface ---

    def reset(self) -> None:
        self._steering.reset()
        self._dribbling.reset()  # also re-spawns the ball
        self._blend_step = self._blend_steps
        self._blend_from = None
        self._last_action_real = self.robot.default_joint_pos.clone()

    def inference(self) -> torch.Tensor:
        self._poll_joystick_buttons()

        if self._mode == MODE_DRIBBLING:
            action_real = self._dribbling.inference()
        else:
            action_real = self._steering.inference()
            # Keep the ball history warm (exactly one push per control step,
            # same rate as in dribbling mode); the obs itself is discarded.
            self._dribbling.compute_observation()

        if self._blend_step < self._blend_steps and self._blend_from is not None:
            alpha = (self._blend_step + 1) / self._blend_steps
            action_real = (1.0 - alpha) * self._blend_from + alpha * action_real
            self._blend_step += 1

        self._last_action_real = action_real.clone()
        return action_real

    def flush_policy_log_if_enabled(self) -> None:
        self._steering.flush_policy_log_if_enabled()
        self._dribbling.flush_policy_log_if_enabled()


@configclass
class ComboPolicyCfg(PolicyCfg):
    constructor = ComboPolicy
    # Unused — sub-policies carry their own checkpoints.
    checkpoint_path: str = ""

    steering: MimicKitPolicyCfg = MISSING
    dribbling: DribblingPolicyCfg = MISSING

    initial_mode: str = MODE_STEERING
    # Control steps to blend joint targets after a mode switch (15 @ 30 Hz ≈ 0.5 s).
    switch_blend_steps: int = 15
