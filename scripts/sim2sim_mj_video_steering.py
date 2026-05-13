"""Headless MuJoCo video recorder for the K1 steering policy.

Mirrors ``scripts/sim2sim_mj_video.py`` (the dribbling recorder), but uses
the steering policy's natural command interface: ``vx, vy, omega`` — a
world-frame velocity vector plus a yaw rate.

Schedule format: ``dur:vx,vy,omega[;dur:vx,vy,omega]...``
    dur      - segment duration in seconds
    vx, vy   - target velocity in world frame (m/s); magnitude = speed,
               direction = heading. Robot starts facing +X, so:
                   vx=1, vy=0  → forward
                   vx=0, vy=1  → left
                   vx=0, vy=-1 → right
    omega    - target yaw rate in rad/s (positive = counter-clockwise)

Examples:
    # pure yaw -> walk forward -> walk right (10s each, 30s total)
    python scripts/sim2sim_mj_video_steering.py \\
        --task k1_mimickit_steering_video \\
        --out  logs/steer_yaw_fwd_right.mp4 \\
        --schedule "10:0,0,1.5;10:1,0,0;10:0,-1,0"

    # yaw while walking (combined)
    python scripts/sim2sim_mj_video_steering.py \\
        --task k1_mimickit_steering_video \\
        --out  logs/steer_combined.mp4 \\
        --schedule "10:0,0,1.5;10:1,0,1.5;10:0,-1,1.5"

    # single-segment fallback (no --schedule)
    python scripts/sim2sim_mj_video_steering.py \\
        --task k1_mimickit_steering_video \\
        --out  logs/steer_walk.mp4 \\
        --duration 10.0 --vx 1.0 --vy 0.0 --omega 0.0
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(".")


def _parse_segment(seg: str):
    """Parse one ``dur:vx,vy,omega`` segment string."""
    dur_str, cmd_str = seg.split(":")
    parts = [float(x) for x in cmd_str.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"Schedule segment must have exactly 3 comma-separated values "
            f"(vx,vy,omega); got {len(parts)} in '{seg}'."
        )
    vx, vy, omega = parts
    return float(dur_str), vx, vy, omega


def _heading_yaw(qpos_wxyz):
    """Extract yaw angle from a MuJoCo root wxyz quaternion."""
    w, x, y, z = qpos_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="k1_mimickit_steering_video")
    parser.add_argument("--out", required=True, help="Output mp4 path")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Recording duration in seconds of sim time")
    parser.add_argument("--vx", type=float, default=1.0,
                        help="Target x velocity in world frame (m/s)")
    parser.add_argument("--vy", type=float, default=0.0,
                        help="Target y velocity in world frame (m/s)")
    parser.add_argument("--omega", type=float, default=0.0,
                        help="Target yaw rate (rad/s)")
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        help=(
            "Time-varying command schedule, overrides --vx/--vy/--omega/--duration. "
            "Format: 'dur:vx,vy,omega[;dur:vx,vy,omega]'. "
            "Example: '10:0,0,1.5;10:1,0,0;10:0,-1,0'."
        ),
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-elevation", type=float, default=-15.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-lookat-height", type=float, default=0.5,
                        help="Fixed world-frame Z height the camera looks at (m). "
                             "Camera tracks robot XY but not its vertical bobbing.")
    parser.add_argument("--fps", type=int, default=None,
                        help="Output video fps (defaults to the control rate so "
                             "playback is real-time; one frame per control tick)")
    parser.add_argument("--gl", default="egl", choices=["egl", "osmesa", "glfw"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.environ["MUJOCO_GL"] = args.gl
    os.environ.setdefault("PYOPENGL_PLATFORM", args.gl)

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

    if not hasattr(ctrl.policy, "tar_dir"):
        raise RuntimeError(
            f"Task '{args.task}' does not expose tar_dir; this recorder is "
            "specialized for steering policies."
        )
    if not hasattr(ctrl.policy, "tar_omega"):
        raise RuntimeError(
            f"Task '{args.task}' does not expose tar_omega; use the dribbling "
            "recorder (scripts/sim2sim_mj_video.py) for tasks without yaw input."
        )

    # Parse --schedule, falling back to a single-segment one built from flags.
    schedule: list[tuple[float, float, float, float]] = []
    if args.schedule:
        for seg in args.schedule.split(";"):
            seg = seg.strip()
            if not seg:
                continue
            schedule.append(_parse_segment(seg))
        total_duration = sum(d for d, *_ in schedule)
        print(
            f"[video] schedule: {len(schedule)} segments, "
            f"total {total_duration:.2f}s"
        )
    else:
        schedule.append((args.duration, args.vx, args.vy, args.omega))
        total_duration = args.duration

    # Body-frame command state — updated at segment transitions, applied
    # every step after rotating by the robot's current heading so that
    # vx/vy always mean "forward/left relative to the robot."
    cmd_vx, cmd_vy, cmd_omega = 0.0, 0.0, 0.0
    arrow_dir_world = np.array([1.0, 0.0])  # world-frame direction for arrow visualization

    def _update_command_world_frame() -> None:
        nonlocal arrow_dir_world
        yaw = _heading_yaw(ctrl.mj_data.qpos[3:7])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        wx = cos_y * cmd_vx - sin_y * cmd_vy
        wy = sin_y * cmd_vx + cos_y * cmd_vy
        speed = float(np.linalg.norm([wx, wy]))
        if speed > 1e-6:
            arrow_dir_world = np.array([wx / speed, wy / speed])
            ctrl.policy.tar_dir = torch.tensor([wx / speed, wy / speed], dtype=torch.float32)
        else:
            arrow_dir_world = np.array([1.0, 0.0])
            ctrl.policy.tar_dir = torch.tensor([1.0, 0.0], dtype=torch.float32)
        ctrl.policy.tar_speed = torch.tensor([speed], dtype=torch.float32)
        ctrl.policy.tar_omega = torch.tensor([cmd_omega], dtype=torch.float32)

    _, cmd_vx, cmd_vy, cmd_omega = schedule[0]

    control_hz = 1.0 / ctrl.cfg.policy_dt
    if args.fps is None:
        args.fps = int(round(control_hz))
        print(f"[video] fps not set; defaulting to control rate {args.fps} Hz")
    elif abs(control_hz - args.fps) > 1e-3:
        print(
            f"[video] note: control rate is {control_hz:.1f} Hz but video fps is "
            f"{args.fps}; using 1 frame per control tick anyway "
            f"(playback will be {args.fps / control_hz:.2f}× real-time)."
        )
    num_steps = max(1, int(round(total_duration * control_hz)))
    print(
        f"[video] recording {num_steps} control steps "
        f"(~{num_steps / control_hz:.2f}s sim time)"
    )

    segment_end_steps: list[int] = []
    cum_steps = 0
    for dur, *_ in schedule:
        cum_steps += int(round(dur * control_hz))
        segment_end_steps.append(cum_steps)

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
        current_seg = 0
        for step in range(num_steps):
            while (
                current_seg < len(schedule) - 1
                and step >= segment_end_steps[current_seg]
            ):
                current_seg += 1
                _, cmd_vx, cmd_vy, cmd_omega = schedule[current_seg]
                print(
                    f"[video] t={step / control_hz:5.2f}s  "
                    f"segment {current_seg + 1}/{len(schedule)}: "
                    f"vx={cmd_vx:+.2f} vy={cmd_vy:+.2f} omega={cmd_omega:+.2f}"
                )

            _update_command_world_frame()
            ctrl.update_state()
            dof_targets = ctrl.policy_step()
            ctrl.ctrl_step(dof_targets)

            cam.lookat[0] = ctrl.mj_data.qpos[0]
            cam.lookat[1] = ctrl.mj_data.qpos[1]
            cam.lookat[2] = args.camera_lookat_height
            renderer.update_scene(ctrl.mj_data, camera=cam)

            scene = renderer.scene
            if scene.ngeom < scene.maxgeom:
                # Position arrow at robot's head
                head_body_id = None
                for i in range(ctrl.mj_model.nbody):
                    if ctrl.mj_model.body(i).name == "Head_pitch":
                        head_body_id = i
                        break

                if head_body_id is not None:
                    head_pos = ctrl.mj_data.xpos[head_body_id].copy()
                else:
                    # Fallback: place arrow 0.5m above root
                    root_pos = ctrl.mj_data.qpos[0:3].copy()
                    head_pos = np.array([root_pos[0], root_pos[1], root_pos[2] + 0.5], dtype=np.float64)

                # Arrow points in target direction, fixed length for visibility
                arrow_length = 0.5
                arrow_from = head_pos.copy()
                arrow_to = head_pos + np.array([arrow_dir_world[0] * arrow_length, arrow_dir_world[1] * arrow_length, 0.0], dtype=np.float64)

                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3),
                    np.zeros(3),
                    np.zeros(9),
                    np.array([0.1, 0.8, 0.1, 0.9], dtype=np.float32),
                )
                mujoco.mjv_connector(
                    geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.025,
                    arrow_from, arrow_to,
                )
                scene.ngeom += 1

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
