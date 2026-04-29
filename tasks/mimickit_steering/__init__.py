import os

from nomadz_deploy.utils.registry import register_task
from nomadz_deploy.utils.isaaclab.configclass import configclass
from nomadz_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
    SteeringJoystickCommandCfg,
)
from nomadz_deploy.robots.booster import K1_CFG
from .mimickit_policy import MimicKitPolicyCfg


# Set MIMICKIT_STEERING_LOG_PATH to an absolute or relative path stem (no
# extension). When set, the MuJoCo controller writes robot state to
# `{stem}.npz` and the policy writes its input/output trace to
# `{stem}_policy.npz`, both refreshed every 100 control steps.
_LOG_PATH_ENV = "MIMICKIT_STEERING_LOG_PATH"
_log_path_stem = os.environ.get(_LOG_PATH_ENV)
_mujoco_log_path = _log_path_stem if _log_path_stem else None
_policy_log_path = f"{_log_path_stem}_policy" if _log_path_stem else None


CONTROL_FREQUENCY_HZ = 30.0
SIMULATION_FREQUENCY_HZ = 120.0
CONTROL_DECIMATION = int(SIMULATION_FREQUENCY_HZ / CONTROL_FREQUENCY_HZ)
CHECKPOINT_PATH = "models/A027_model.pt"

# Trunk height that places K1's zero-pose feet just above the MJCF floor.
MUJOCO_ZERO_POSE_ROOT_HEIGHT_M = 0.557


# Per-joint stiffness (kp) override. None = use K1_CFG defaults.
# Joint order matches K1_CFG.joint_names (22 joints):
#   [0-1]   head:          AAHead_yaw, Head_pitch
#   [2-9]   arms:          L/R Shoulder Pitch, Roll, Elbow Pitch, Yaw
#   [10-15] left leg:      Hip Pitch, Roll, Yaw, Knee, Ankle Pitch, Roll
#   [16-21] right leg:     Hip Pitch, Roll, Yaw, Knee, Ankle Pitch, Roll
KP_OVERRIDE: list[float] | None = None
# Example — halve all leg gains:
# KP_OVERRIDE = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 40, 40, 40, 40, 15, 15, 40, 40, 40, 40, 15, 15]

# Uniform scale applied on top of KP_OVERRIDE (or the defaults). 1.0 = no change.
KP_SCALE: float = 1.0

_base_kp = KP_OVERRIDE if KP_OVERRIDE is not None else list(K1_CFG.joint_stiffness)
_kp = [v * KP_SCALE for v in _base_kp]


@configclass
class K1MimicKitSteeringCfg(ControllerCfg):
    policy_dt: float = 1.0 / CONTROL_FREQUENCY_HZ

    robot = K1_CFG.replace(  # type: ignore
        joint_stiffness=_kp,
        joint_damping=list(K1_CFG.joint_damping),
        default_joint_pos=[0.0] * 22,
    )

    # Steering commands are provided directly as tar_dir(2) + speed(1)
    # + omega(1) via stdin, not via the standard velocity command path.
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
        action_smoothing=1.0,   # add this — 1.0 = no smoothing
    )

    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, MUJOCO_ZERO_POSE_ROOT_HEIGHT_M],
        decimation=CONTROL_DECIMATION,
        log_states=_mujoco_log_path,
        ground_friction=[1.0, 0.005, 0.0001],  # [sliding, torsional, rolling]
    )


register_task("k1_mimickit_steering", K1MimicKitSteeringCfg())
