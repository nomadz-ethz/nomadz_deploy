"""Headless MuJoCo obs/action logger for sim2sim parity debugging.

Same purpose as `scripts/deploy.py --mujoco --log-stem ...` but without the
mujoco viewer, so it works over SSH. Pins the steering command to a fixed
value and runs for a bounded number of control steps, then exits cleanly so
the policy-trace .npz and MuJoCo state .npz are flushed.

Example:
  cd /home/nomadz-control/booster_deploy
  python scripts/sim2sim_mj_logger.py \\
      --log-stem logs/sim2sim_mj \\
      --num-steps 20 \\
      --tar-dir-x 1.0 --tar-dir-y 0.0 --tar-speed 0.5 --tar-omega 0.0
"""

import argparse
import os
import sys

sys.path.append(".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="k1_mimickit_steering")
    parser.add_argument("--log-stem", required=True)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--tar-dir-x", type=float, default=1.0)
    parser.add_argument("--tar-dir-y", type=float, default=0.0)
    parser.add_argument("--tar-speed", type=float, default=0.5)
    parser.add_argument("--tar-omega", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # The policy wrapper picks up MIMICKIT_STEERING_LOG_PATH from env,
    # mirroring how `scripts/deploy.py --log-stem` wires it up.
    os.environ["MIMICKIT_STEERING_LOG_PATH"] = args.log_stem

    import pkgutil
    import tasks as tasks_pkg

    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        __import__(mod_info.name)
    from nomadz_deploy.utils.registry import get_task

    task_cfg = get_task(args.task)
    task_cfg.policy.device = args.device

    import torch

    from nomadz_deploy.controllers.mujoco_controller import MujocoController

    ctrl = MujocoController(task_cfg)

    # Force the steering command before the first inference call so step 0
    # is deterministic and matches the MimicKit logger's command.
    if hasattr(ctrl.policy, "tar_dir"):
        ctrl.policy.tar_dir = torch.tensor(
            [args.tar_dir_x, args.tar_dir_y], dtype=torch.float32
        )
        ctrl.policy.tar_speed = torch.tensor([args.tar_speed], dtype=torch.float32)
        ctrl.policy.tar_omega = torch.tensor([args.tar_omega], dtype=torch.float32)
    else:
        raise RuntimeError(
            f"Task '{args.task}' policy does not expose tar_dir/tar_speed/tar_omega; "
            "this script is specialized for steering policies."
        )

    ctrl.update_state()
    ctrl.start()

    try:
        for step in range(args.num_steps):
            ctrl.update_state()
            dof_targets = ctrl.policy_step()
            ctrl.ctrl_step(dof_targets)
    finally:
        ctrl._flush_logged_states()
        if hasattr(ctrl.policy, "flush_policy_log_if_enabled"):
            ctrl.policy.flush_policy_log_if_enabled()

    print(
        f"Completed {args.num_steps} control steps headless. "
        f"Traces: {args.log_stem}.npz (MuJoCo state), "
        f"{args.log_stem}_policy.npz (policy trace)."
    )


if __name__ == "__main__":
    main()
