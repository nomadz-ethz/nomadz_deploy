import os

from nomadz_deploy.utils.registry import register_task
from nomadz_deploy.utils.isaaclab.configclass import configclass
from nomadz_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
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
CHECKPOINT_PATH = "models/A025_model.pt"

# Trunk height that places K1's zero-pose feet just above the MJCF floor.
MUJOCO_ZERO_POSE_ROOT_HEIGHT_M = 0.557


@configclass
class K1MimicKitSteeringCfg(ControllerCfg):
    policy_dt: float = 1.0 / CONTROL_FREQUENCY_HZ

    robot = K1_CFG.replace(  # type: ignore
        joint_stiffness=list(K1_CFG.joint_stiffness),
        joint_damping=list(K1_CFG.joint_damping),
        default_joint_pos=[0.0] * 22,
    )

    # Steering commands are provided directly as tar_dir(2) + speed(1)
    # + omega(1) via stdin, not via the standard velocity command path.
    vel_command = None

    policy: MimicKitPolicyCfg = MimicKitPolicyCfg(
        checkpoint_path=CHECKPOINT_PATH,
        enable_safety_fallback=False,
        log_path=_policy_log_path,
    )

    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, MUJOCO_ZERO_POSE_ROOT_HEIGHT_M],
        decimation=CONTROL_DECIMATION,
        log_states=_mujoco_log_path,
    )


register_task("k1_mimickit_steering", K1MimicKitSteeringCfg())
