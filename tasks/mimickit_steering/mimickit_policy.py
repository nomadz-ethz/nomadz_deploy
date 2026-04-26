"""Steering-policy deploy wrapper for the K1 robot.

Builds a 53-D observation matching the training environment's
``compute_xyomega_obs`` and feeds it to a TorchScript actor exported with
``scripts/export_amp_policy.py``. The scripted module bakes
observation/action normalization and action clipping; this class adds the
deploy-side concerns: command state, sim-state -> obs assembly, joint
reordering, EMA smoothing, and optional trace logging.

Observation layout (53D):
    gravity_body  (3)  - world gravity rotated into the body frame
    ang_vel_body  (3)  - body-frame angular velocity
    dof_pos       (22) - joint positions in policy joint order
    dof_vel       (22) - joint velocities in policy joint order
    local_tar_vel (2)  - target direction rotated into the heading frame,
                         scaled by target speed (vx, vy)
    tar_omega     (1)  - signed target yaw rate (rad/s)

Command convention:
    ``tar_dir`` is a unit XY vector in the world frame; it is rotated into
    the robot's heading frame and multiplied by ``tar_speed`` to produce
    ``local_tar_vel``. ``tar_omega`` is a signed scalar yaw rate.
"""
from __future__ import annotations

import torch

from nomadz_deploy.utils.isaaclab.configclass import configclass

from .jit_policy import JitPolicy, JitPolicyCfg


MIN_TARGET_SPEED_MPS = 0.0
MAX_TARGET_SPEED_MPS = 2.0
DEFAULT_TARGET_SPEED_MPS = MIN_TARGET_SPEED_MPS
MIN_TARGET_OMEGA_RAD_S = -1.5
MAX_TARGET_OMEGA_RAD_S = 1.5
DEFAULT_TARGET_OMEGA_RAD_S = 0.0
DEFAULT_TARGET_DIRECTION_WORLD_XY = (1.0, 0.0)
DIRECTION_EPS = 1e-6
WORLD_GRAVITY_DIRECTION = (0.0, 0.0, -1.0)


# DFS traversal of the K1 articulation tree, identical to the URDF /
# MuJoCo MJCF joint order. Reordering is computed by name so that a
# future ``RobotCfg.joint_names`` reshuffle can't silently break this
# policy.
POLICY_JOINT_NAMES = (
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
)


def _quat_wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)


def _quat_rotate_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_v = q[..., :3]
    q_w = q[..., 3:]
    t = 2.0 * torch.cross(q_v, v, dim=-1)
    return v + q_w * t + torch.cross(q_v, t, dim=-1)


def _quat_conjugate_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def _calc_heading_quat_inv_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Inverse of the yaw-only component of ``q``, returned in xyzw."""
    ref_dir = torch.tensor([1.0, 0.0, 0.0], dtype=q.dtype, device=q.device)
    rot_dir = _quat_rotate_xyzw(q.unsqueeze(0), ref_dir.unsqueeze(0)).squeeze(0)
    heading = torch.atan2(rot_dir[1], rot_dir[0])

    half_angle = -0.5 * heading
    sin_a = torch.sin(half_angle)
    cos_a = torch.cos(half_angle)

    return torch.tensor([0.0, 0.0, sin_a, cos_a], dtype=q.dtype, device=q.device)


def _normalize_direction_xy(
    direction_xy: torch.Tensor,
    *,
    default_direction_xy: tuple[float, float],
) -> torch.Tensor:
    direction_norm = torch.linalg.vector_norm(direction_xy)
    if float(direction_norm) < DIRECTION_EPS:
        return direction_xy.new_tensor(default_direction_xy)
    return direction_xy / direction_norm


def _append_zero_z(vector_xy: torch.Tensor) -> torch.Tensor:
    return torch.cat([vector_xy, vector_xy.new_zeros(1)], dim=0)


class MimicKitPolicy(JitPolicy):
    """K1 steering policy: tar_dir(2) + tar_speed(1) + tar_omega(1)."""

    POLICY_JOINT_NAMES = POLICY_JOINT_NAMES

    def __init__(self, cfg: MimicKitPolicyCfg, controller):
        super().__init__(cfg, controller)
        self._target_direction_world_xy = torch.tensor(
            DEFAULT_TARGET_DIRECTION_WORLD_XY, dtype=torch.float32
        )
        self._target_speed_mps = torch.tensor(
            [DEFAULT_TARGET_SPEED_MPS], dtype=torch.float32
        )
        self._target_omega_rad_s = torch.tensor(
            [DEFAULT_TARGET_OMEGA_RAD_S], dtype=torch.float32
        )

    @property
    def tar_dir(self) -> torch.Tensor:
        return self._target_direction_world_xy

    @tar_dir.setter
    def tar_dir(self, value) -> None:
        direction_xy = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if direction_xy.numel() != 2:
            raise ValueError(f"tar_dir must have 2 elements, got {direction_xy.shape}")
        self._target_direction_world_xy = _normalize_direction_xy(
            direction_xy,
            default_direction_xy=DEFAULT_TARGET_DIRECTION_WORLD_XY,
        )

    @property
    def tar_speed(self) -> torch.Tensor:
        return self._target_speed_mps

    @tar_speed.setter
    def tar_speed(self, value) -> None:
        speed_mps = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if speed_mps.numel() != 1:
            raise ValueError(f"tar_speed must have 1 element, got {speed_mps.shape}")
        self._target_speed_mps = torch.clamp(
            speed_mps, min=MIN_TARGET_SPEED_MPS, max=MAX_TARGET_SPEED_MPS
        )

    @property
    def tar_omega(self) -> torch.Tensor:
        return self._target_omega_rad_s

    @tar_omega.setter
    def tar_omega(self, value) -> None:
        omega = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if omega.numel() != 1:
            raise ValueError(f"tar_omega must have 1 element, got {omega.shape}")
        self._target_omega_rad_s = torch.clamp(
            omega, min=MIN_TARGET_OMEGA_RAD_S, max=MAX_TARGET_OMEGA_RAD_S
        )

    def compute_observation(self) -> torch.Tensor:
        data = self.robot.data

        root_quat_xyzw = _quat_wxyz_to_xyzw(data.root_quat_w)
        root_quat_inv_xyzw = _quat_conjugate_xyzw(root_quat_xyzw)
        gravity_world = root_quat_xyzw.new_tensor(WORLD_GRAVITY_DIRECTION)
        gravity_body = _quat_rotate_xyzw(
            root_quat_inv_xyzw.unsqueeze(0), gravity_world.unsqueeze(0)
        ).squeeze(0)

        # MuJoCo qvel[3:6] is already body-frame.
        ang_vel_body = data.root_ang_vel_b

        dof_pos_policy = data.joint_pos[self._real_to_policy_idx]
        dof_vel_policy = data.joint_vel[self._real_to_policy_idx]

        heading_rot_inv_xyzw = _calc_heading_quat_inv_xyzw(root_quat_xyzw)
        tar_dir_3d = _append_zero_z(self._target_direction_world_xy)
        local_tar_dir = _quat_rotate_xyzw(
            heading_rot_inv_xyzw.unsqueeze(0), tar_dir_3d.unsqueeze(0)
        ).squeeze(0)[:2]
        local_tar_vel = local_tar_dir * self._target_speed_mps

        return torch.cat([
            gravity_body,             # 3
            ang_vel_body,             # 3
            dof_pos_policy,           # 22
            dof_vel_policy,           # 22
            local_tar_vel,            # 2
            self._target_omega_rad_s, # 1
        ], dim=0)

    def _record_extra_log(self, entry, *, obs):
        import numpy as np
        obs_np = obs.detach().cpu().numpy().astype(np.float32, copy=False)
        entry.update({
            "obs_gravity_body":         obs_np[0:3],
            "obs_ang_vel_body":         obs_np[3:6],
            "obs_dof_pos_policy_order": obs_np[6:28],
            "obs_dof_vel_policy_order": obs_np[28:50],
            "obs_local_tar_vel":        obs_np[50:52],
            "obs_tar_omega":            obs_np[52:53],
            "tar_dir_world_xy":         self._target_direction_world_xy.detach().cpu().numpy().astype(np.float32),
            "tar_speed_mps":            self._target_speed_mps.detach().cpu().numpy().astype(np.float32),
            "tar_omega_rad_s":          self._target_omega_rad_s.detach().cpu().numpy().astype(np.float32),
        })


@configclass
class MimicKitPolicyCfg(JitPolicyCfg):
    constructor = MimicKitPolicy
