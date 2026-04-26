"""Headless MuJoCo video recorder for a steering policy.

Runs the task with a fixed steering command for a bounded duration and
writes an mp4 via MuJoCo's offscreen renderer — no display required, so it
works over SSH. Uses EGL by default; falls back to OSMesa if EGL is not
available.

Example:
  cd /home/nomadz-control/booster_deploy
  python scripts/sim2sim_mj_video.py \\
      --out logs/sim2sim_walk.mp4 \\
      --duration 10.0 \\
      --tar-dir-x 1.0 --tar-dir-y 0.0 --tar-speed 1.0 --tar-omega 0.0

The camera tracks the robot's trunk. Video FPS defaults to the policy
control rate.
"""

import argparse
import os
import sys

sys.path.append(".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="k1_mimickit_steering")
    parser.add_argument("--out", required=True, help="Output mp4 path")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Recording duration in seconds of sim time")
    parser.add_argument("--tar-dir-x", type=float, default=1.0)
    parser.add_argument("--tar-dir-y", type=float, default=0.0)
    parser.add_argument("--tar-speed", type=float, default=0.5)
    parser.add_argument("--tar-omega", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-elevation", type=float, default=-15.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video fps (one frame per control tick by default)")
    parser.add_argument("--gl", default="egl", choices=["egl", "osmesa", "glfw"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Must be set before importing mujoco so it picks the right GL backend.
    os.environ["MUJOCO_GL"] = args.gl
    os.environ.setdefault("PYOPENGL_PLATFORM", args.gl)

    import numpy as np
    import torch
    import mujoco
    import imageio.v2 as imageio

    import pkgutil
    import tasks as tasks_pkg

    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        __import__(mod_info.name)
    from nomadz_deploy.utils.registry import get_task
    from nomadz_deploy.controllers.mujoco_controller import MujocoController

    task_cfg = get_task(args.task)
    task_cfg.policy.device = args.device

    ctrl = MujocoController(task_cfg)

    if hasattr(ctrl.policy, "tar_dir"):
        ctrl.policy.tar_dir = torch.tensor(
            [args.tar_dir_x, args.tar_dir_y], dtype=torch.float32
        )
        ctrl.policy.tar_speed = torch.tensor([args.tar_speed], dtype=torch.float32)
        ctrl.policy.tar_omega = torch.tensor([args.tar_omega], dtype=torch.float32)
    else:
        raise RuntimeError(
            f"Task '{args.task}' does not expose tar_dir; this recorder is "
            "specialized for steering policies."
        )

    # One control tick per video frame keeps things simple and matches
    # the policy's control rate.
    control_hz = 1.0 / ctrl.cfg.policy_dt
    if abs(control_hz - args.fps) > 1e-3:
        print(
            f"[video] note: control rate is {control_hz:.1f} Hz but video fps is "
            f"{args.fps}; using 1 frame per control tick anyway."
        )
    num_steps = max(1, int(round(args.duration * control_hz)))
    print(
        f"[video] recording {num_steps} control steps "
        f"(~{num_steps / control_hz:.2f}s sim time)"
    )

    # MuJoCo's default offscreen framebuffer is 640x480; resize it to match
    # the requested render resolution before constructing the Renderer.
    ctrl.mj_model.vis.global_.offwidth = args.width
    ctrl.mj_model.vis.global_.offheight = args.height
    renderer = mujoco.Renderer(ctrl.mj_model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = args.camera_distance
    cam.elevation = args.camera_elevation
    cam.azimuth = args.camera_azimuth

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    writer = imageio.get_writer(
        args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=1
    )

    ctrl.update_state()
    ctrl.start()

    try:
        for step in range(num_steps):
            ctrl.update_state()
            dof_targets = ctrl.policy_step()
            ctrl.ctrl_step(dof_targets)

            # Track the trunk so the camera follows the walker.
            cam.lookat[:] = ctrl.mj_data.qpos[0:3]
            renderer.update_scene(ctrl.mj_data, camera=cam)
            frame = renderer.render()
            writer.append_data(frame)
    finally:
        writer.close()
        ctrl._flush_logged_states()
        if hasattr(ctrl.policy, "flush_policy_log_if_enabled"):
            ctrl.policy.flush_policy_log_if_enabled()

    print(f"[video] wrote {args.out}")


if __name__ == "__main__":
    main()
