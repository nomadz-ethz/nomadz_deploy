import argparse
import os
import sys
import warnings

# Suppress pygame warnings globally
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)

sys.path.append(".")

parser = argparse.ArgumentParser()
# require either --task or --list (mutually exclusive)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--task", type=str, help="Name of the configuration file.")
group.add_argument("-l", "--list", action="store_true", dest="list_tasks",
                   default=False, help="list available tasks")

parser.add_argument("--net", type=str, default="127.0.0.1",
                    help="Network interface for SDK communication.")
parser.add_argument("--mujoco", action="store_true", default=False,
                    help="deploy in mujoco simulation")
parser.add_argument("--webots", action="store_true", default=False,
                    help="deploy in webots simulation")
parser.add_argument(
    "--joystick", action="store_true", default=False,
    help="Enable joystick control for teleoperation")
parser.add_argument(
    "--vx-max", type=float, default=None,
    help="Override joystick max forward velocity.")
parser.add_argument(
    "--vy-max", type=float, default=None,
    help="Override joystick max lateral velocity.")
parser.add_argument(
    "--vyaw-max", type=float, default=None,
    help="Override joystick max yaw velocity.")
parser.add_argument(
    "--device", type=str, default="cpu",
    help="Device to run the evaluation on (e.g., 'cpu', 'cuda')")
parser.add_argument(
    "--log-stem",
    type=str,
    default=None,
    help=(
        "Path stem for debug logs. Example: --log-stem logs/k1_debug "
        "writes logs/k1_debug.npz (MuJoCo state) and "
        "logs/k1_debug_policy.npz (policy trace)."
    ),
)
args = parser.parse_args()


def main():
    # Optional: enable MimicKit steering debug logs without requiring manual
    # environment variable export.
    if args.log_stem:
        os.environ["MIMICKIT_STEERING_LOG_PATH"] = args.log_stem

    # load task registry and dispatch
    import pkgutil
    import tasks as tasks_pkg

    # auto-import all submodules under tasks (recursive) so they can register themselves
    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        full_name = mod_info.name
        try:
            __import__(full_name)
        except Exception as e:
            raise e
    from nomadz_deploy.utils.registry import get_task, list_tasks

    if args.list_tasks:
        print("Available tasks:")
        for task_name, cfg in list_tasks().items():
            cls = type(cfg)
            full_cls = f"{cls.__module__}.{cls.__qualname__}"
            print(f"  {task_name}\t:\t{full_cls}")
        sys.exit(0)

    try:
        task_cfg = get_task(args.task)
    except KeyError:
        print(f"Unknown task '{args.task}'. Available tasks: {list(list_tasks().keys())}")
        sys.exit(1)

    # Set device for policy
    task_cfg.policy.device = args.device

    if args.joystick:
        if task_cfg.vel_command is not None:
            if args.vx_max is not None:
                task_cfg.vel_command.vx_max = args.vx_max
            if args.vy_max is not None:
                task_cfg.vel_command.vy_max = args.vy_max
            if args.vyaw_max is not None:
                task_cfg.vel_command.vyaw_max = args.vyaw_max
        elif hasattr(task_cfg, "steering_joystick_command"):
            if args.vx_max is not None:
                task_cfg.steering_joystick_command.vx_max = args.vx_max
            if args.vy_max is not None:
                task_cfg.steering_joystick_command.vy_max = args.vy_max
            if args.vyaw_max is not None:
                task_cfg.steering_joystick_command.vyaw_max = args.vyaw_max

    # decide how to run based on flags
    if args.mujoco:
        # run mujoco controller
        from nomadz_deploy.controllers.mujoco_controller import MujocoController

        MujocoController(task_cfg, joystick_enabled=args.joystick).run()
    else:
        # initialize network and run robot portal
        try:
            from booster_robotics_sdk_python import ChannelFactory  # type: ignore
            ChannelFactory.Instance().Init(0, args.net)
        except ImportError as e:
            print(
                "Error: booster_robotics_sdk_python is not installed.\n"
                "Please install it to use real robot deployment.\n"
                "For MuJoCo simulation, use --mujoco flag instead."
            )
            sys.exit(1)

        # adjust ankle dampings for webots
        if args.webots:
            ankles = [-8, -7, -2, -1]  # indices of ankle joints
            for i in ankles:
                task_cfg.robot.joint_damping[i] = 0.5

        from nomadz_deploy.controllers.booster_robot_controller import BoosterRobotPortal
        with BoosterRobotPortal(task_cfg, use_sim_time=args.webots) as portal:
            portal.run()


if __name__ == "__main__":
    main()
