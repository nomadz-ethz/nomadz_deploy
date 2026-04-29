"""Dribbling-policy deploy wrapper.

Builds the 84-D observation that pairs the robot proprioceptive state with
a ball-position history and a steering command, matching the training
environment's ``compute_base_obs`` + ``compute_dribbling_observations``.
The scripted actor (exported via ``scripts/export_amp_policy.py``) bakes
observation/action normalization and action clipping; this class adds the
deploy-side concerns: command state, sim-state -> obs assembly, ball-body
lookup, camera FOV check, ball-position history maintenance, joint
reordering, EMA smoothing, and optional trace logging.

Observation layout (84D):
    gravity_body   (3)  - world gravity rotated into the body frame
    ang_vel_body   (3)  - body-frame angular velocity
    dof_pos        (22) - joint positions in policy joint order
    dof_vel        (22) - joint velocities in policy joint order
    ball_history   (30) - 15 entries x XY of ball position in heading frame;
                          zero for entries when ball was out of camera FOV
    local_tar_dir  (2)  - target direction rotated into heading frame
    tar_speed      (1)  - scalar target ball speed (m/s)
    is_in_view     (1)  - 1.0 if ball centroid is in camera FOV, else 0.0

Command convention:
    ``tar_dir`` is a unit XY vector in the world frame; it is rotated into
    the robot's heading frame to produce ``local_tar_dir``. ``tar_speed``
    is a scalar in m/s.

Generic-ness:
    Camera body name, FOV, position/pitch offsets, ball body name, history
    length, and ball spawn pose are all configured via
    ``DribblingPolicyCfg`` so this class can be reused for any policy that
    shares this 84-D obs structure on a different robot or camera setup.
"""
from __future__ import annotations
from dataclasses import field
import math
import os
import sys

import mujoco
import numpy as np
import torch

from nomadz_deploy.utils.isaaclab.configclass import configclass

# Reuse the generic JIT-policy base from the steering task.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
from tasks.mimickit_steering.jit_policy import JitPolicy, JitPolicyCfg  # noqa: E402


MIN_TARGET_SPEED_MPS = 0.0
MAX_TARGET_SPEED_MPS = 3.0
DEFAULT_TARGET_SPEED_MPS = 1.0
DEFAULT_TARGET_DIRECTION_WORLD_XY = (1.0, 0.0)
DIRECTION_EPS = 1e-6
WORLD_GRAVITY_DIRECTION = (0.0, 0.0, -1.0)


# DFS traversal of the K1 articulation tree, identical to the URDF /
# MuJoCo MJCF joint order. Subclasses for other robots can override this.
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


# ---------------------------------------------------------------------------
# Quaternion helpers (xyzw convention used inside this file).
# MuJoCo's xquat is wxyz; we convert at the boundary.
# ---------------------------------------------------------------------------

def _quat_wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)


def _quat_rotate_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_v = q[..., :3]
    q_w = q[..., 3:]
    t = 2.0 * torch.cross(q_v, v, dim=-1)
    return v + q_w * t + torch.cross(q_v, t, dim=-1)


def _quat_conjugate_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def _quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def _calc_heading_quat_inv_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Inverse of the yaw-only component of ``q`` (xyzw)."""
    ref_dir = torch.tensor([1.0, 0.0, 0.0], dtype=q.dtype, device=q.device)
    rot_dir = _quat_rotate_xyzw(q.unsqueeze(0), ref_dir.unsqueeze(0)).squeeze(0)
    heading = torch.atan2(rot_dir[1], rot_dir[0])
    half_angle = -0.5 * heading
    sin_a = torch.sin(half_angle)
    cos_a = torch.cos(half_angle)
    return torch.tensor(
        [0.0, 0.0, sin_a, cos_a], dtype=q.dtype, device=q.device
    )


def _normalize_direction_xy(
    direction_xy: torch.Tensor,
    *,
    default_direction_xy: tuple[float, float],
) -> torch.Tensor:
    norm = torch.linalg.vector_norm(direction_xy)
    if float(norm) < DIRECTION_EPS:
        return direction_xy.new_tensor(default_direction_xy)
    return direction_xy / norm


def _append_zero_z(vector_xy: torch.Tensor) -> torch.Tensor:
    return torch.cat([vector_xy, vector_xy.new_zeros(1)], dim=0)


class DribblingPolicy(JitPolicy):
    """Dribbling policy: tar_dir(2) + tar_speed(1) + ball-aware obs."""

    POLICY_JOINT_NAMES = POLICY_JOINT_NAMES

    def __init__(self, cfg: "DribblingPolicyCfg", controller):
        super().__init__(cfg, controller)
        self.cfg: DribblingPolicyCfg = cfg

        mj_model = self.controller.mj_model
        self._ball_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, cfg.ball_body_name
        )
        if self._ball_body_id < 0:
            raise RuntimeError(
                f"Ball body '{cfg.ball_body_name}' not found in MJCF"
            )
        self._cam_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, cfg.camera_body_name
        )
        if self._cam_body_id < 0:
            raise RuntimeError(
                f"Camera body '{cfg.camera_body_name}' not found in MJCF"
            )

        # qpos/qvel slice for the ball's free joint.
        ball_jnt_id = mj_model.body_jntadr[self._ball_body_id]
        if ball_jnt_id < 0:
            raise RuntimeError(
                f"Body '{cfg.ball_body_name}' has no joint; expected freejoint"
            )
        self._ball_qpos_adr = int(mj_model.jnt_qposadr[ball_jnt_id])
        self._ball_qvel_adr = int(mj_model.jnt_dofadr[ball_jnt_id])

        # FOV bounds in tan-of-half-angle form.
        self._fov_max_u = math.tan(math.radians(cfg.camera_fov_horizontal_deg / 2.0))
        self._fov_max_v = math.tan(math.radians(cfg.camera_fov_vertical_deg / 2.0))

        # Camera pitch offset baked as xyzw quaternion (rotation about local Y).
        # Negative pitch -> camera tilts down. Mirrors the training env's
        # construction so deploy-side gaze checks line up bit-for-bit with
        # the trained policy's expectations.
        half_pitch = -math.radians(cfg.camera_pitch_offset_deg) / 2.0
        self._cam_pitch_quat_xyzw = torch.tensor(
            [0.0, math.sin(half_pitch), 0.0, math.cos(half_pitch)],
            dtype=torch.float32,
        )
        self._cam_local_offset = torch.tensor(
            [cfg.camera_x_offset_m, cfg.camera_y_offset_m, cfg.camera_z_offset_m],
            dtype=torch.float32,
        )

        # Ball-position history in heading frame (oldest -> newest at index 0).
        self._ball_history_len = cfg.ball_history_len
        self._ball_pos_history = torch.zeros(
            [self._ball_history_len, 2], dtype=torch.float32
        )

        # Steering command state.
        self._target_direction_world_xy = torch.tensor(
            DEFAULT_TARGET_DIRECTION_WORLD_XY, dtype=torch.float32
        )
        self._target_speed_mps = torch.tensor(
            [float(cfg.default_target_speed_mps)], dtype=torch.float32
        )

    # --- Command properties (mirror the steering task's interface) ---

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
        speed = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if speed.numel() != 1:
            raise ValueError(f"tar_speed must have 1 element, got {speed.shape}")
        self._target_speed_mps = torch.clamp(
            speed, min=MIN_TARGET_SPEED_MPS, max=MAX_TARGET_SPEED_MPS
        )

    # --- Lifecycle ---

    def reset(self) -> None:
        super().reset()
        self._ball_pos_history.zero_()
        self._spawn_ball()

    def _spawn_ball(self) -> None:
        """Place the ball at a random position in an annulus around the robot."""
        ctrl = self.controller
        mj_data = ctrl.mj_data
        root_xy = np.array(ctrl.cfg.mujoco.init_pos[:2], dtype=np.float32)
        r_min, r_max = self.cfg.ball_spawn_radius_min_m, self.cfg.ball_spawn_radius_max_m
        r = np.sqrt(np.random.uniform(r_min ** 2, r_max ** 2))
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        offset = np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)
        ball_xy = root_xy + offset
        mj_data.qpos[self._ball_qpos_adr + 0] = ball_xy[0]
        mj_data.qpos[self._ball_qpos_adr + 1] = ball_xy[1]
        mj_data.qpos[self._ball_qpos_adr + 2] = self.cfg.ball_init_z_m
        # quaternion (wxyz), identity
        mj_data.qpos[self._ball_qpos_adr + 3] = 1.0
        mj_data.qpos[self._ball_qpos_adr + 4] = 0.0
        mj_data.qpos[self._ball_qpos_adr + 5] = 0.0
        mj_data.qpos[self._ball_qpos_adr + 6] = 0.0
        # velocities: stationary
        mj_data.qvel[self._ball_qvel_adr : self._ball_qvel_adr + 6] = 0.0
        mujoco.mj_forward(ctrl.mj_model, mj_data)

    # --- Observation ---

    def compute_observation(self) -> torch.Tensor:
        data = self.robot.data
        mj_data = self.controller.mj_data

        # Robot proprioception (gravity + ang_vel in body frame, joints in
        # policy order). Same construction as the steering wrapper.
        root_quat_xyzw = _quat_wxyz_to_xyzw(data.root_quat_w)
        root_quat_inv_xyzw = _quat_conjugate_xyzw(root_quat_xyzw)
        gravity_world = root_quat_xyzw.new_tensor(WORLD_GRAVITY_DIRECTION)
        gravity_body = _quat_rotate_xyzw(
            root_quat_inv_xyzw.unsqueeze(0), gravity_world.unsqueeze(0)
        ).squeeze(0)
        ang_vel_body = data.root_ang_vel_b

        dof_pos_policy = data.joint_pos[self._real_to_policy_idx]
        dof_vel_policy = data.joint_vel[self._real_to_policy_idx]

        # Ball world position.
        ball_pos_w = torch.tensor(
            mj_data.xpos[self._ball_body_id].copy(), dtype=torch.float32
        )
        # Robot root world position (from MuJoCo qpos: 3 root pos + 4 quat + 22 dofs).
        root_pos_w = torch.tensor(
            mj_data.qpos[0:3].copy(), dtype=torch.float32
        )

        # Camera world pose: cam_body world pose, then offset (in body frame)
        # and pitch quaternion applied.
        cam_body_pos_w = torch.tensor(
            mj_data.xpos[self._cam_body_id].copy(), dtype=torch.float32
        )
        cam_body_quat_xyzw = _quat_wxyz_to_xyzw(
            torch.tensor(mj_data.xquat[self._cam_body_id].copy(), dtype=torch.float32)
        )
        cam_pos_w = cam_body_pos_w + _quat_rotate_xyzw(
            cam_body_quat_xyzw.unsqueeze(0),
            self._cam_local_offset.unsqueeze(0),
        ).squeeze(0)
        cam_quat_xyzw = _quat_mul_xyzw(cam_body_quat_xyzw, self._cam_pitch_quat_xyzw)

        is_in_view = self._ball_in_view(cam_pos_w, cam_quat_xyzw, ball_pos_w)

        # Update ball history: zero entry if ball is out of view, otherwise
        # ball XY in robot's heading frame.
        heading_rot_inv_xyzw = _calc_heading_quat_inv_xyzw(root_quat_xyzw)
        ball_rel_w = ball_pos_w - root_pos_w
        ball_local = _quat_rotate_xyzw(
            heading_rot_inv_xyzw.unsqueeze(0), ball_rel_w.unsqueeze(0)
        ).squeeze(0)
        ball_local_xy = ball_local[:2]
        new_entry = ball_local_xy if is_in_view > 0.5 else torch.zeros(2)
        # Shift old entries one slot to the right (newest at index 0).
        self._ball_pos_history[1:] = self._ball_pos_history[:-1].clone()
        self._ball_pos_history[0] = new_entry
        ball_history_flat = self._ball_pos_history.reshape(-1)

        # Target dir in heading frame.
        tar_dir_3d = _append_zero_z(self._target_direction_world_xy)
        local_tar_dir = _quat_rotate_xyzw(
            heading_rot_inv_xyzw.unsqueeze(0), tar_dir_3d.unsqueeze(0)
        ).squeeze(0)[:2]

        in_view_obs = torch.tensor([1.0 if is_in_view > 0.5 else 0.0])

        return torch.cat([
            gravity_body,                   # 3
            ang_vel_body,                   # 3
            dof_pos_policy,                 # 22
            dof_vel_policy,                 # 22
            ball_history_flat,              # 30
            local_tar_dir,                  # 2
            self._target_speed_mps,         # 1
            in_view_obs,                    # 1
        ], dim=0)

    def _ball_in_view(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_xyzw: torch.Tensor,
        ball_pos_w: torch.Tensor,
    ) -> float:
        """Return 1.0 if ball centroid is inside camera FOV, else 0.0."""
        vec_world = ball_pos_w - cam_pos_w
        cam_quat_conj = _quat_conjugate_xyzw(cam_quat_xyzw)
        vec_local = _quat_rotate_xyzw(
            cam_quat_conj.unsqueeze(0), vec_world.unsqueeze(0)
        ).squeeze(0)
        x_fwd = float(vec_local[0])
        y_left = float(vec_local[1])
        z_up = float(vec_local[2])
        if x_fwd <= 0.1:
            return 0.0
        u = y_left / max(x_fwd, 1e-5)
        v = z_up / max(x_fwd, 1e-5)
        if abs(u) > self._fov_max_u or abs(v) > self._fov_max_v:
            return 0.0
        return 1.0

    # --- Logging hook ---

    def _record_extra_log(self, entry, *, obs):
        obs_np = obs.detach().cpu().numpy().astype(np.float32, copy=False)
        entry.update({
            "obs_gravity_body":         obs_np[0:3],
            "obs_ang_vel_body":         obs_np[3:6],
            "obs_dof_pos_policy_order": obs_np[6:28],
            "obs_dof_vel_policy_order": obs_np[28:50],
            "obs_ball_history":         obs_np[50:80],
            "obs_local_tar_dir":        obs_np[80:82],
            "obs_tar_speed":            obs_np[82:83],
            "obs_in_view":              obs_np[83:84],
            "tar_dir_world_xy":         self._target_direction_world_xy.detach().cpu().numpy().astype(np.float32),
            "tar_speed_mps":            self._target_speed_mps.detach().cpu().numpy().astype(np.float32),
        })


@configclass
class DribblingPolicyCfg(JitPolicyCfg):
    constructor = DribblingPolicy

    # Camera configuration (matches the training env). Override per-robot.
    camera_body_name: str = "Head_2"
    camera_fov_horizontal_deg: float = 87.0
    camera_fov_vertical_deg: float = 58.0
    camera_x_offset_m: float = 0.059
    camera_y_offset_m: float = 0.0115
    camera_z_offset_m: float = 0.098
    camera_pitch_offset_deg: float = -12.2

    # Ball body in the composite scene.
    ball_body_name: str = "ball"

    # Ball-position history length used in the obs vector.
    ball_history_len: int = 15

    # Ball spawn annulus (relative to robot root XY).
    ball_spawn_radius_min_m: float = 0.5
    ball_spawn_radius_max_m: float = 1.0
    ball_init_z_m: float = 0.08

    # Initial command speed (m/s). Set via TARGET_SPEED_MPS in the task __init__.
    default_target_speed_mps: float = DEFAULT_TARGET_SPEED_MPS
