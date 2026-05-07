"""Steering task variant used by the headless video pipeline.

Mirrors ``tasks/mimickit_steering`` but registers under a new task name
(``k1_mimickit_steering_video``) and points at its own model directory so
exporting / swapping a new steering checkpoint never touches the live
steering deployment task.

Reuses ``MimicKitPolicy`` / ``MimicKitPolicyCfg`` from
``tasks.mimickit_steering``; the joint order is shared, but the obs
layout is selectable via ``obs_command_layout`` to match whichever
MimicKit steering env produced the active checkpoint
(``"xyomega"`` for A027/A029, ``"vxomega"`` for A030).

To swap models:
    1. ``python scripts/export_amp_policy.py --ckpt <mimickit_run>/model.pt
       --out tasks/mimickit_steering_video/models/<NAME>_model.pt``
    2. Edit ``CHECKPOINT_FILENAME`` below to ``"<NAME>_model.pt"``.
    3. Set ``OBS_COMMAND_LAYOUT`` below to match the trainer env variant.
"""
import os

from nomadz_deploy.utils.registry import register_task
from nomadz_deploy.utils.isaaclab.configclass import configclass
from nomadz_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
    SteeringJoystickCommandCfg,
)
from nomadz_deploy.robots.booster import K1_CFG
from tasks.mimickit_steering.mimickit_policy import (
    MimicKitPolicyCfg,
    OBS_LAYOUT_VXOMEGA,
    OBS_LAYOUT_XYOMEGA,
)


_LOG_PATH_ENV = "MIMICKIT_STEERING_VIDEO_LOG_PATH"
_log_path_stem = os.environ.get(_LOG_PATH_ENV)
_mujoco_log_path = _log_path_stem if _log_path_stem else None
_policy_log_path = f"{_log_path_stem}_policy" if _log_path_stem else None


CONTROL_FREQUENCY_HZ = 30.0
SIMULATION_FREQUENCY_HZ = 120.0
CONTROL_DECIMATION = int(SIMULATION_FREQUENCY_HZ / CONTROL_FREQUENCY_HZ)

# Active checkpoint. Edit this when you export a new model.
CHECKPOINT_FILENAME = "A030_model.pt"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "models", CHECKPOINT_FILENAME
)
# Trainer obs layout this checkpoint was exported against.
#   A027/A029 -> OBS_LAYOUT_XYOMEGA (53D)
#   A030      -> OBS_LAYOUT_VXOMEGA (52D)
OBS_COMMAND_LAYOUT = OBS_LAYOUT_VXOMEGA

# Trunk height that places K1's zero-pose feet just above the MJCF floor.
MUJOCO_ZERO_POSE_ROOT_HEIGHT_M = 0.557

# Per-joint stiffness override / scale. Mirrors the live steering task.
KP_OVERRIDE: list[float] | None = None
KP_SCALE: float = 1.0

_base_kp = KP_OVERRIDE if KP_OVERRIDE is not None else list(K1_CFG.joint_stiffness)
_kp = [v * KP_SCALE for v in _base_kp]


@configclass
class K1MimicKitSteeringVideoCfg(ControllerCfg):
    policy_dt: float = 1.0 / CONTROL_FREQUENCY_HZ

    robot = K1_CFG.replace(  # type: ignore
        joint_stiffness=_kp,
        joint_damping=list(K1_CFG.joint_damping),
        default_joint_pos=[
            0.0, 0.0,          # Head yaw, pitch
            0.0, -1.57,        # L Shoulder pitch, roll
            0.0, 0.0,          # L Elbow pitch, yaw
            0.0, 1.57,         # R Shoulder pitch, roll
            0.0, 0.0,          # R Elbow pitch, yaw
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # L leg
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # R leg
        ],
    )

    vel_command = None
    steering_joystick_command = SteeringJoystickCommandCfg(
        vx_max=2.0,
        vy_max=2.0,
        vyaw_max=1.5,
    )

    policy: MimicKitPolicyCfg = MimicKitPolicyCfg(
        checkpoint_path=CHECKPOINT_PATH,
        enable_safety_fallback=False,
        log_path=_policy_log_path,
        action_smoothing=1.0,
        obs_command_layout=OBS_COMMAND_LAYOUT,
    )

    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, MUJOCO_ZERO_POSE_ROOT_HEIGHT_M],
        decimation=CONTROL_DECIMATION,
        log_states=_mujoco_log_path,
        ground_friction=[1.0, 0.005, 0.0001],
    )


register_task("k1_mimickit_steering_video", K1MimicKitSteeringVideoCfg())
