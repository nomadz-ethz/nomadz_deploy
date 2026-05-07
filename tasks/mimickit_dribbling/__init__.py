"""Dribbling task registration.

Composes a MuJoCo scene from the K1 robot MJCF and a ball body, then
registers a controller config that points the controller at the composite
scene. The generated scene file is written next to this module so the
configured ``mjcf_path`` is stable across processes.
"""
import os

from booster_assets import BOOSTER_ASSETS_DIR

from nomadz_deploy.utils.registry import register_task
from nomadz_deploy.utils.isaaclab.configclass import configclass
from nomadz_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
    SteeringJoystickCommandCfg,
)
from nomadz_deploy.robots.booster import K1_CFG
from .dribbling_policy import DribblingPolicyCfg


# Set MIMICKIT_DRIBBLING_LOG_PATH to an absolute or relative path stem (no
# extension). When set, the MuJoCo controller writes robot state to
# `{stem}.npz` and the policy writes its input/output trace to
# `{stem}_policy.npz`, both refreshed every 100 control steps.
_LOG_PATH_ENV = "MIMICKIT_DRIBBLING_LOG_PATH"
_log_path_stem = os.environ.get(_LOG_PATH_ENV)
_mujoco_log_path = _log_path_stem if _log_path_stem else None
_policy_log_path = f"{_log_path_stem}_policy" if _log_path_stem else None


CONTROL_FREQUENCY_HZ = 30.0
SIMULATION_FREQUENCY_HZ = 120.0  # 4x substeps: reduces bang-bang contact impulse at kp=4584 N·m/rad
CONTROL_DECIMATION = int(SIMULATION_FREQUENCY_HZ / CONTROL_FREQUENCY_HZ)
CHECKPOINT_PATH = "models/B024_model.pt"

# Trunk height that places K1's zero-pose feet just above the MJCF floor.
MUJOCO_ZERO_POSE_ROOT_HEIGHT_M = 0.557

KP_OVERRIDE: list[float] | None = None
KD_OVERRIDE: list[float] | None = None
# Example — halve all leg gains:
# KP_OVERRIDE = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 40, 40, 40, 40, 15, 15, 40, 40, 40, 40, 15, 15]

# Uniform scale applied on top of KP_OVERRIDE (or the defaults). 1.0 = no change.
KP_SCALE: float = 1.0
KD_SCALE: float = 1.0
_base_kp = KP_OVERRIDE if KP_OVERRIDE is not None else list(K1_CFG.joint_stiffness)
_kp = [v * KP_SCALE for v in _base_kp]

_base_kd = KD_OVERRIDE if KD_OVERRIDE is not None else list(K1_CFG.joint_damping)
_kd = [v * KD_SCALE for v in _base_kd]

# ---------------------------------------------------------------------------
# Compose K1 + ball into a single scene MJCF.
#
# The composite scene is written into the K1 asset directory (next to
# K1_22dof.xml) so MuJoCo's relative mesh-path resolution (`meshdir="meshes/"`
# in the K1 compiler tag) just works without absolute-path gymnastics.
# The file name is prefixed with `_` to mark it as a generated artifact.
# ---------------------------------------------------------------------------
_K1_DIR = os.path.join(BOOSTER_ASSETS_DIR, "robots", "K1")
_K1_MJCF = os.path.join(_K1_DIR, "K1_22dof_orig.xml")
_SCENE_MJCF = os.path.join(_K1_DIR, "_k1_dribbling_scene.xml")

# Ball physics roughly match the dribbling training env (mass 0.45 kg,
# radius 0.08 m, friction in the trained range, restitution 0.6).
BALL_RADIUS_M = 0.08
BALL_MASS_KG = 0.45
BALL_FRICTION = 0.4
BALL_RGBA = "1.0 0.6 0.0 1.0"

# Throw-cube projectile. The body is always present in the scene; the
# controller only launches it when enable_throw=True. Between throws it
# rests far off-screen at THROW_PARK_POS so it can't interact with the
# robot or ball.
#
# Tunable knobs ↓
THROW_CUBE_SIDE_M = 0.25           # full side length of the cube, in meters
THROW_CUBE_MASS_KG = 5.0           # mass of the cube, in kilograms
# (Visual / scene plumbing — usually leave alone)
THROW_CUBE_RGBA = "0.1 0.2 0.9 1.0"
THROW_PARK_POS = "10 10 0.02"      # m; well outside the dribbling area
_THROW_CUBE_HALFSIZE_M = THROW_CUBE_SIDE_M / 2  # MuJoCo box `size` is half-extent

# solref="timeconst dampratio": both values must be positive to avoid MuJoCo
# "mixed solref format" warning. dampratio < 1 = underdamped = bouncy contact.
# Lower dampratio → more bounce (0 = fully elastic, 1 = critically damped).
_SCENE_XML = f"""<mujoco model="k1_dribbling">
  <include file="K1_22dof_orig.xml"/>
`
  <worldbody>
    <body name="ball" pos="0 0 {BALL_RADIUS_M}">
      <freejoint name="ball_joint"/>
      <geom name="ball_geom" type="sphere"
            size="{BALL_RADIUS_M}" mass="{BALL_MASS_KG}"
            friction="{BALL_FRICTION} 0.005 0.02" condim="6"
            solimp="0.95 0.99 0.001" solref="0.008 0.2"
            rgba="{BALL_RGBA}"/>
    </body>

    <body name="throw_cube" pos="{THROW_PARK_POS}">
      <freejoint name="throw_cube_joint"/>
      <geom name="throw_cube_geom" type="box"
            size="{_THROW_CUBE_HALFSIZE_M} {_THROW_CUBE_HALFSIZE_M} {_THROW_CUBE_HALFSIZE_M}"
            mass="{THROW_CUBE_MASS_KG}"
            friction="0.5 0.005 0.0001" condim="3"
            rgba="{THROW_CUBE_RGBA}"/>
    </body>
  </worldbody>
</mujoco>
"""

# Write only if changed to avoid spurious file modifications on reimport.
_existing = ""
if os.path.exists(_SCENE_MJCF):
    with open(_SCENE_MJCF) as _f:
        _existing = _f.read()
if _existing != _SCENE_XML:
    with open(_SCENE_MJCF, "w") as _f:
        _f.write(_SCENE_XML)


@configclass
class K1MimicKitDribblingCfg(ControllerCfg):
    policy_dt: float = 1.0 / CONTROL_FREQUENCY_HZ

    # Override mjcf_path to point at the composite scene; gains and joint
    # table still come from K1_CFG.
    robot = K1_CFG.replace(  # type: ignore
        mjcf_path=_SCENE_MJCF,
        joint_stiffness=_kp,
        #joint_stiffness=list(K1_CFG.joint_stiffness),
        joint_damping=_kd,
        # default_joint_pos=[0.0] * 22,
        default_joint_pos=[
            0.0, 0.0,          # Head yaw, pitch
            0.0, -1.57,         # L Shoulder pitch, roll
            0.0, 0.0,          # L Elbow pitch, yaw
            0.0, 1.57,        # R Shoulder pitch, roll
            0.0, 0.0,          # R Elbow pitch, yaw
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # L leg
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # R leg
        ]
            )

    # Steering commands (tar_dir + tar_speed) come via stdin / property
    # setters, not via the standard velocity command path.
    vel_command = None
    steering_joystick_command = SteeringJoystickCommandCfg(
        vx_max=3.0,
        vy_max=3.0,
        vyaw_max=0.0,
    )

    policy: DribblingPolicyCfg = DribblingPolicyCfg(
        checkpoint_path=CHECKPOINT_PATH,
        enable_safety_fallback=False,
        log_path=_policy_log_path,
        action_smoothing=1.0,
    )

    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, MUJOCO_ZERO_POSE_ROOT_HEIGHT_M],
        decimation=CONTROL_DECIMATION,
        log_states=_mujoco_log_path,
        ground_friction=[1.0, 0.005, 0.0001],
        enable_push= False,
        push_force_min=50.0,
        push_force_max=75.0,
        push_interval_min=5.0,
        push_interval_max=5.0,
        push_duration=0.1,
        push_body="Trunk",
        enable_fall_reset=True,
        fall_height_threshold=0.3,
        fall_grace_period=0.5,
        enable_throw=False,
        throw_object_body="throw_cube",
        throw_object_joint="throw_cube_joint",
        throw_speed_min=5.0,
        throw_speed_max=8.0,
        throw_distance=2.0,
        throw_height_min=0.5,
        throw_height_max=1.2,
        throw_aim_offset=0.1,
        throw_spin_max=5.0,
        throw_interval_min=5.0,
        throw_interval_max=5.0,
    )


register_task("k1_mimickit_dribbling", K1MimicKitDribblingCfg())
