from nomadz_deploy.utils.isaaclab.configclass import configclass
from nomadz_deploy.utils.registry import register_task
from .locomotion import (
    K1WalkControllerCfg,
    T1WalkControllerCfg
)

# Register locomotion tasks


@configclass
class T1WalkControllerCfg1(T1WalkControllerCfg):
    '''Human-like walk for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/t1_walk.pt"
        # The portal-side recovery state machine owns fall handling for this
        # task; LocomotionPolicy's own safety_fallback would call
        # controller.stop() on a fall and tear down the inference subprocess
        # before the state machine sees the fall_event. Disable it here so
        # the FSM can drive the GetUp loop. See
        # docs/RECOVERY_INTEGRATION_PLAN.md §10.7.
        self.policy.enable_safety_fallback = False


register_task(
    "t1_walk", T1WalkControllerCfg1())
