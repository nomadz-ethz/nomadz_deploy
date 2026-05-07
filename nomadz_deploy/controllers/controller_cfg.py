from typing import Callable, List, Optional
from dataclasses import MISSING
import torch

from ..utils.isaaclab.configclass import configclass


@configclass
class PrepareStateCfg:
    stiffness: List[float] = MISSING
    damping: List[float] = MISSING
    joint_pos: List[float] = MISSING


@configclass
class MujocoControllerCfg:
    init_pos: List[float] = [0.0, 0.0, 0.6]
    init_quat: List[float] = [1.0, 0.0, 0.0, 0.0]
    decimation: int = 10
    # When True, rewrite <motor> actuators into position-style <general>
    # actuators (kp/kd from robot cfg) and switch the integrator to
    # mjINT_IMPLICITFAST so the kd term is folded into the mass matrix.
    # When False, leave actuators as raw torque and apply manual PD in
    # Python each substep (the original repo's approach). High-impact /
    # bouncy-contact tasks (e.g. dribbling) need True; default-walking
    # tasks like t1_walk run fine with False.
    use_native_pd: bool = False
    # physics_dt will automatically be set by ControllerCfg
    physics_dt: float = None  # type: ignore
    # Ground geom friction [sliding, torsional, rolling]. Overrides the
    # MJCF value at load time so deployment is reproducible across assets.
    ground_friction: List[float] = [1.0, 0.005, 0.0001]
    log_states: Optional[str] = None
    visualize_reference_ghost: bool = False
    ghost_rgba: List[float] = [0.2, 0.8, 0.2, 0.25]

    # Random horizontal pushes on a body (default: trunk). Mirrors MimicKit's
    # push_randomization at training time so the deploy run can reproduce the
    # same disturbance regime.
    enable_push: bool = False
    push_force_min: float = 35.0       # Newtons
    push_force_max: float = 50.0       # Newtons
    push_interval_min: float = 5.0     # seconds between push events
    push_interval_max: float = 5.0     # seconds; equal min/max → fixed interval
    push_duration: float = 0.1         # seconds the force stays applied
    push_body: str = "Trunk"

    # Auto-reset when the robot falls. Fires when root z drops below
    # fall_height_threshold; after fall_grace_period seconds since the last
    # reset the check is enabled (so the very first physics tick — when the
    # robot is settling — doesn't immediately retrigger).
    enable_fall_reset: bool = False
    fall_height_threshold: float = 0.3   # meters; root z below this counts as fallen
    fall_grace_period: float = 0.5       # seconds; ignore fall checks this long after a reset

    # Throw a free-jointed object at the robot at random intervals. The
    # object body must already exist in the scene MJCF (typically composed
    # by the task config) — the controller just teleports it to a spawn
    # point at throw_distance from the trunk and gives it an initial
    # velocity aimed at the trunk's chest. Between throws the body sits
    # wherever it last landed.
    enable_throw: bool = False
    throw_object_body: str = "throw_cube"
    throw_object_joint: str = "throw_cube_joint"
    throw_speed_min: float = 5.0         # m/s, initial speed of the projectile
    throw_speed_max: float = 8.0
    throw_distance: float = 2.0          # m, horizontal spawn distance from trunk
    throw_height_min: float = 0.5        # m, spawn height above ground
    throw_height_max: float = 1.2
    throw_aim_offset: float = 0.1        # m, height above trunk to aim at (chest-ish)
    throw_spin_max: float = 5.0          # rad/s, random tumble (per axis); 0 to disable
    throw_interval_min: float = 5.0
    throw_interval_max: float = 5.0      # equal min/max → fixed interval

    ball_joint: str = "ball_joint"       # freejoint name of the ball in the scene
    ball_spawn_dist_min: float = 1.0     # m, min distance from robot at spawn/reset
    ball_spawn_dist_max: float = 2.0     # m, max distance from robot at spawn/reset


@configclass
class BoosterRobotControllerCfg:
    low_state_dt: float = 0.002
    metrics_max_events: int = 2000


@configclass
class RobotCfg:
    name: str = MISSING

    joint_names: list[str] = MISSING
    body_names: list[str] = MISSING

    sim_joint_names: list[str] = MISSING
    sim_body_names: list[str] = MISSING

    joint_stiffness: List[float] = MISSING
    joint_damping: List[float] = MISSING

    default_joint_pos: List[float] = MISSING
    effort_limit: List[float] = MISSING

    mjcf_path: str = MISSING

    prepare_state: PrepareStateCfg = MISSING

    def __post_init__(self):
        assert (
            len(self.joint_names)
            == len(self.joint_stiffness)
            == len(self.joint_damping)
            == len(self.default_joint_pos)
            == len(self.effort_limit)
        )


@configclass
class VelocityCommandCfg:
    vx_max: float = 1.0
    vy_max: float = 1.0
    vyaw_max: float = 1.0


@configclass
class SteeringJoystickCommandCfg:
    vx_max: float = 2.0
    vy_max: float = 2.0
    vyaw_max: float = 4.0


@configclass
class PolicyCfg:
    constructor: Callable = MISSING
    checkpoint_path: str = MISSING
    enable_safety_fallback: bool = True
    device: str | torch.device = "cpu"


@configclass
class EvaluatorCfg:
    constructor: Callable = MISSING
    # Rendering
    render: bool = True


@configclass
class ControllerCfg:
    """Controller configuration class.
    """

    policy_dt: float = 0.02
    robot: RobotCfg = MISSING
    vel_command: Optional[VelocityCommandCfg] = None
    steering_joystick_command: SteeringJoystickCommandCfg = SteeringJoystickCommandCfg()
    policy: PolicyCfg = MISSING

    mujoco: MujocoControllerCfg = MujocoControllerCfg()
    booster: BoosterRobotControllerCfg = BoosterRobotControllerCfg()
    evaluator: Optional[EvaluatorCfg] = None

    def __post_init__(self):
        self.mujoco.physics_dt = self.policy_dt / self.mujoco.decimation
