"""Combined steering + dribbling task registration (``k1_mimickit_combo``).

Composes one MuJoCo scene (K1 + RoboCup field + ball) and registers a
controller config whose policy is a state machine over the steering (A029,
xyomega) and dribbling sub-policies — switch live with the joystick Y button
or the viewer 'M' key. See ``combo_policy.py`` for the switching semantics.

The scene is a RoboCup HSL 2026 Middle-Division setup (the K1's division:
robot height < 1.25 m) instead of the default MuJoCo flat ground: M-Field
(14 x 9 m, white markings, two goals) and a FIFA size 3 ball — see
``nomadz_deploy/utils/robocup_field.py`` for the rule-book sources.

Run:
    python scripts/deploy.py --task k1_mimickit_combo --mujoco --joystick

Checkpoints (both verified present on disk; the task-default A030/B024
files referenced by the standalone tasks are missing):
    steering  — tasks/mimickit_steering/models/A029_model.pt  (53-D xyomega)
    dribbling — tasks/mimickit_dribbling.model.pt              (84-D)
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
from nomadz_deploy.utils.robocup_field import (
    FIFA_SIZE_3,
    M_FIELD,
    compose_field_scene,
)
from tasks.mimickit_steering.mimickit_policy import (
    MimicKitPolicyCfg,
    OBS_LAYOUT_XYOMEGA,
)
from tasks.mimickit_dribbling.dribbling_policy import DribblingPolicyCfg
from .combo_policy import ComboPolicyCfg, MODE_STEERING


CONTROL_FREQUENCY_HZ = 30.0
# 120 Hz is the regime both checkpoints were validated in (the steering
# video task and the dribbling contact tuning both use it).
SIMULATION_FREQUENCY_HZ = 120.0
CONTROL_DECIMATION = int(SIMULATION_FREQUENCY_HZ / CONTROL_FREQUENCY_HZ)

_TASKS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEERING_CHECKPOINT = os.path.join(
    _TASKS_DIR, "mimickit_steering", "models", "A029_model.pt"
)
DRIBBLING_CHECKPOINT = os.path.join(_TASKS_DIR, "mimickit_dribbling.model.pt")

# Trunk height that places K1's zero-pose feet just above the MJCF floor.
MUJOCO_ZERO_POSE_ROOT_HEIGHT_M = 0.557

# ---------------------------------------------------------------------------
# Compose K1 + RoboCup field + ball into a single scene MJCF, written next to
# the K1 robot file so its relative meshdir="meshes/" resolves. Built from
# K1_22dof_fixed.xml (the file the live steering task uses —
# K1_22dof_orig.xml referenced by the dribbling task does not exist in this
# checkout), with its default flat ground replaced by the field.
# ---------------------------------------------------------------------------
_K1_DIR = os.path.join(BOOSTER_ASSETS_DIR, "robots", "K1")
_ROBOT_MJCF = os.path.join(_K1_DIR, "K1_22dof_fixed.xml")
_SCENE_MJCF = os.path.join(_K1_DIR, "_k1_combo_scene.xml")

# HSL 2026 Middle Division allows a FIFA size 3 or 4 ball (rules Table 4);
# size 3 (r=0.094 m, 0.31 kg) stays closest to the ball the dribbling policy
# was trained on (r=0.08 m, 0.45 kg). Contact params keep the trained tuning.
FIELD_SPEC = M_FIELD
BALL_SPEC = FIFA_SIZE_3
BALL_CONTACT_ATTRS = {
    "friction": "0.4 0.005 0.02",
    "condim": "6",
    "solimp": "0.95 0.99 0.001",
    "solref": "0.008 0.2",
}

_SCENE_XML = compose_field_scene(
    _ROBOT_MJCF,
    model_name="k1_combo",
    field=FIELD_SPEC,
    ball=BALL_SPEC,
    ball_contact_attrs=BALL_CONTACT_ATTRS,
)

# Write only if changed to avoid spurious file modifications on reimport.
_existing = ""
if os.path.exists(_SCENE_MJCF):
    with open(_SCENE_MJCF) as _f:
        _existing = _f.read()
if _existing != _SCENE_XML:
    with open(_SCENE_MJCF, "w") as _f:
        _f.write(_SCENE_XML)


@configclass
class K1MimicKitComboCfg(ControllerCfg):
    policy_dt: float = 1.0 / CONTROL_FREQUENCY_HZ

    robot = K1_CFG.replace(  # type: ignore
        mjcf_path=_SCENE_MJCF,
        joint_stiffness=list(K1_CFG.joint_stiffness),
        joint_damping=list(K1_CFG.joint_damping),
        # Arms-down pose used by both MimicKit tasks.
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

    # Commands flow through the steering-style path (tar_dir/tar_speed/
    # tar_omega properties on the combo policy), not vel_command.
    vel_command = None
    steering_joystick_command = SteeringJoystickCommandCfg(
        vx_max=2.0,
        vy_max=2.0,
        vyaw_max=1.5,  # only consumed in STEERING mode
    )

    policy: ComboPolicyCfg = ComboPolicyCfg(
        initial_mode=MODE_STEERING,
        switch_blend_steps=15,
        steering=MimicKitPolicyCfg(
            checkpoint_path=STEERING_CHECKPOINT,
            obs_command_layout=OBS_LAYOUT_XYOMEGA,  # A029 is the 53-D variant
            enable_safety_fallback=False,
            action_smoothing=1.0,
        ),
        dribbling=DribblingPolicyCfg(
            checkpoint_path=DRIBBLING_CHECKPOINT,
            enable_safety_fallback=False,
            action_smoothing=1.0,
            # Spawn the ball resting on the turf (rules-size radius).
            ball_init_z_m=BALL_SPEC.radius,
        ),
    )

    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, MUJOCO_ZERO_POSE_ROOT_HEIGHT_M],
        decimation=CONTROL_DECIMATION,
        ground_friction=[1.0, 0.005, 0.0001],
        enable_fall_reset=True,
        fall_height_threshold=0.3,
        fall_grace_period=0.5,
    )


register_task("k1_mimickit_combo", K1MimicKitComboCfg())
