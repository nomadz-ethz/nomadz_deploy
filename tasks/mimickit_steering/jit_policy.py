"""Generic deploy wrapper for a TorchScript actor.

Use case: a training run produces a checkpoint where actor weights are
stored alongside running stats for input and action normalizers. The
exporter (``scripts/export_amp_policy.py``) bakes the MLP, both
normalizers, and the action clip into a single scripted module; this class
loads that module and runs forward inference for a deploy task.

The base class is task-agnostic. It owns:
    * checkpoint loading via ``torch.jit.load``
    * real <-> policy joint reorder maps
    * EMA action smoothing on the policy-order targets
    * optional per-step diagnostic logging to ``{log_path}.npz``

Subclasses define:
    * ``POLICY_JOINT_NAMES`` - joint order the actor was trained against
    * ``compute_observation()`` - sim state -> 1D obs tensor
    * (optional) ``_record_extra_log()`` - extra named slices for the trace

Forward shape contract: the scripted actor takes a 1D obs tensor and
returns a 1D action tensor of length ``len(POLICY_JOINT_NAMES)``,
expressed as absolute joint position targets in policy order.
"""
from __future__ import annotations
from dataclasses import MISSING
import os

import numpy as np
import torch

from nomadz_deploy.controllers.base_controller import BaseController, Policy
from nomadz_deploy.controllers.controller_cfg import PolicyCfg
from nomadz_deploy.utils.isaaclab.configclass import configclass


POLICY_LOG_SAVE_INTERVAL = 100


class JitPolicy(Policy):
    """TorchScript-actor deployment policy. Subclass to define obs + joints."""

    POLICY_JOINT_NAMES: tuple[str, ...] = ()

    def __init__(self, cfg: JitPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot
        if not 0.0 < cfg.action_smoothing <= 1.0:
            raise ValueError(
                f"action_smoothing must be in (0, 1], got {cfg.action_smoothing}"
            )

        ckpt_path = cfg.checkpoint_path
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(self.task_path, ckpt_path)
        self._model: torch.jit.ScriptModule = torch.jit.load(
            ckpt_path, map_location="cpu"
        )
        self._model.eval()

        policy_joint_names = list(self.POLICY_JOINT_NAMES)
        if not policy_joint_names:
            raise ValueError(
                f"{type(self).__name__} must define POLICY_JOINT_NAMES"
            )
        real_joint_names = list(self.robot.cfg.joint_names)
        if sorted(real_joint_names) != sorted(policy_joint_names):
            raise ValueError(
                f"Robot joint set does not match policy joint set.\n"
                f"  Configured: {real_joint_names}\n"
                f"  Expected:   {policy_joint_names}"
            )
        self._real_to_policy_idx = torch.tensor(
            [real_joint_names.index(n) for n in policy_joint_names],
            dtype=torch.long,
        )
        self._policy_to_real_idx = torch.tensor(
            [policy_joint_names.index(n) for n in real_joint_names],
            dtype=torch.long,
        )

        self._action_smoothing = cfg.action_smoothing
        self._previous_policy_targets = self.robot.default_joint_pos[
            self._real_to_policy_idx
        ].clone()

        self._log_path = cfg.log_path
        self._policy_log: dict[str, list[np.ndarray]] | None = (
            {} if self._log_path is not None else None
        )
        self._last_policy_log_flush_step: int = 0

    def reset(self) -> None:
        self._previous_policy_targets = self.robot.default_joint_pos[
            self._real_to_policy_idx
        ].clone()

    def compute_observation(self) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute_observation()"
        )

    def inference(self) -> torch.Tensor:
        obs = self.compute_observation()

        with torch.no_grad():
            action_policy = self._model(obs)

        action_pre_smooth = action_policy.clone()
        if self._action_smoothing < 1.0:
            alpha = self._action_smoothing
            action_policy = (
                alpha * action_policy
                + (1.0 - alpha) * self._previous_policy_targets
            )
        self._previous_policy_targets = action_policy.clone()

        action_real = action_policy[self._policy_to_real_idx]

        if self._policy_log is not None:
            self._record_step(
                obs=obs,
                action_pre_smooth=action_pre_smooth,
                action_post_smooth=action_policy,
                action_real=action_real,
            )

        return action_real

    def flush_policy_log_if_enabled(self) -> None:
        if self._policy_log is None:
            return
        steps = len(self._policy_log.get("obs", []))
        self._flush_policy_log(steps)

    def _record_step(
        self,
        *,
        obs: torch.Tensor,
        action_pre_smooth: torch.Tensor,
        action_post_smooth: torch.Tensor,
        action_real: torch.Tensor,
    ) -> None:
        assert self._policy_log is not None
        entry: dict[str, np.ndarray] = {
            "obs": _to_numpy(obs),
            "action_pre_smooth_policy_order": _to_numpy(action_pre_smooth),
            "action_post_smooth_policy_order": _to_numpy(action_post_smooth),
            "action_real_order": _to_numpy(action_real),
        }
        self._record_extra_log(entry, obs=obs)
        for key, value in entry.items():
            self._policy_log.setdefault(key, []).append(value)

        steps_logged = len(self._policy_log["obs"])
        if steps_logged % POLICY_LOG_SAVE_INTERVAL == 0:
            self._flush_policy_log(steps_logged)

    def _record_extra_log(
        self, entry: dict[str, np.ndarray], *, obs: torch.Tensor
    ) -> None:
        """Subclasses may add task-specific named slices to ``entry``."""

    def _flush_policy_log(self, steps_logged: int) -> None:
        assert self._policy_log is not None and self._log_path is not None
        if steps_logged == 0 or steps_logged == self._last_policy_log_flush_step:
            return
        stacked = {k: np.stack(v) for k, v in self._policy_log.items()}
        np.savez(f"{self._log_path}.npz", **stacked)
        self._last_policy_log_flush_step = steps_logged
        print(
            f"saved policy trace to {self._log_path}.npz "
            f"at {steps_logged} inference steps"
        )


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32, copy=False)


@configclass
class JitPolicyCfg(PolicyCfg):
    constructor = JitPolicy
    checkpoint_path: str = MISSING  # type: ignore
    # EMA smoothing factor for action targets (0 < alpha <= 1).
    # 1.0 = no smoothing. Lower values bridge sim-to-sim PD differences
    # (e.g. PhysX implicit PD vs MuJoCo explicit PD).
    action_smoothing: float = 0.5
    # If set, writes a per-step diagnostic trace to ``{log_path}.npz``,
    # overwritten every POLICY_LOG_SAVE_INTERVAL steps.
    log_path: str | None = None
