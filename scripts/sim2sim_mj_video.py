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
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        help=(
            "Time-varying command schedule, overrides --tar-dir-* and "
            "--duration. Format: 'dur:dx,dy,sp[;dur:dx,dy,sp]...' — e.g. "
            "'15:1,0,2;15:0,1,2;15:0,-1,2;15:-1,0,2' for a 60s 4-segment run."
        ),
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-elevation", type=float, default=-20.0)
    parser.add_argument("--camera-azimuth", type=float, default=120.0)
    parser.add_argument("--camera-smooth", type=float, default=0.1,
                        help="EMA alpha for camera lookat smoothing (0=frozen, 1=no smoothing)")
    parser.add_argument("--fps", type=int, default=None,
                        help="Output video fps (defaults to the control rate so "
                             "playback is real-time; one frame per control tick)")
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
    from PIL import Image, ImageDraw, ImageFont

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
            "specialized for steering / dribbling policies."
        )

    # Parse --schedule if given, otherwise build a single-segment schedule
    # from --duration / --tar-*.
    schedule: list[tuple[float, float, float, float]] = []
    if args.schedule:
        for seg in args.schedule.split(";"):
            seg = seg.strip()
            if not seg:
                continue
            dur_str, cmd_str = seg.split(":")
            dx, dy, sp = (float(x) for x in cmd_str.split(","))
            schedule.append((float(dur_str), dx, dy, sp))
        total_duration = sum(d for d, *_ in schedule)
        print(
            f"[video] schedule: {len(schedule)} segments, "
            f"total {total_duration:.2f}s"
        )
    else:
        schedule.append(
            (args.duration, args.tar_dir_x, args.tar_dir_y, args.tar_speed)
        )
        total_duration = args.duration

    def _set_command(dx: float, dy: float, sp: float) -> None:
        ctrl.policy.tar_dir = torch.tensor([dx, dy], dtype=torch.float32)
        ctrl.policy.tar_speed = torch.tensor([sp], dtype=torch.float32)
        if hasattr(ctrl.policy, "tar_omega"):
            ctrl.policy.tar_omega = torch.tensor(
                [args.tar_omega], dtype=torch.float32
            )

    # Initialize with the first segment's command so the policy starts on it.
    _set_command(schedule[0][1], schedule[0][2], schedule[0][3])

    # One control tick per video frame keeps things simple and matches
    # the policy's control rate.
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

    # Pre-compute segment boundaries (in control steps) for fast lookup.
    segment_end_steps: list[int] = []
    cum_steps = 0
    for dur, *_ in schedule:
        cum_steps += int(round(dur * control_hz))
        segment_end_steps.append(cum_steps)

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

    # Try to find a TTF font for the HUD; fall back to PIL's bitmap default.
    _hud_font = None
    for _path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ):
        if os.path.exists(_path):
            _hud_font = ImageFont.truetype(_path, size=max(16, args.height // 28))
            break
    if _hud_font is None:
        _hud_font = ImageFont.load_default()

    def _overlay_hud(frame_rgb: np.ndarray, dir_xy: tuple, cmd_speed: float,
                     actual_speed: float) -> np.ndarray:
        """Draw a small HUD in the top-left with command + actual velocity."""
        img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img, "RGBA")
        lines = [
            f"dir:   ({dir_xy[0]:+.0f}, {dir_xy[1]:+.0f})",
            f"|v|cmd: {cmd_speed:.2f} m/s",
            f"|v|act: {actual_speed:.2f} m/s",
        ]
        # Measure for a translucent background box.
        pad = 8
        line_h = _hud_font.size + 4 if hasattr(_hud_font, "size") else 14
        widths = []
        for ln in lines:
            try:
                bbox = draw.textbbox((0, 0), ln, font=_hud_font)
                widths.append(bbox[2] - bbox[0])
            except AttributeError:
                widths.append(_hud_font.getsize(ln)[0])
        box_w = max(widths) + 2 * pad
        box_h = line_h * len(lines) + 2 * pad
        draw.rectangle([(0, 0), (box_w, box_h)], fill=(0, 0, 0, 140))
        for i, ln in enumerate(lines):
            draw.text((pad, pad + i * line_h), ln,
                      fill=(255, 255, 255, 255), font=_hud_font)
        return np.asarray(img)

    ctrl.update_state()
    ctrl.start()

    # Smoothed camera lookat — updated each frame via EMA so the camera
    # follows the robot without snapping to every jitter.
    cam_lookat = np.array(ctrl.mj_data.qpos[0:3], dtype=np.float64)
    cam_alpha = float(args.camera_smooth)

    try:
        current_seg = 0
        for step in range(num_steps):
            # Advance to the next schedule segment when we cross its boundary.
            while (
                current_seg < len(schedule) - 1
                and step >= segment_end_steps[current_seg]
            ):
                current_seg += 1
                _, dx, dy, sp = schedule[current_seg]
                _set_command(dx, dy, sp)
                print(
                    f"[video] t={step / control_hz:5.2f}s  "
                    f"segment {current_seg + 1}/{len(schedule)}: "
                    f"dir=({dx:+.2f},{dy:+.2f}) speed={sp:.2f}"
                )

            ctrl.update_state()
            dof_targets = ctrl.policy_step()
            ctrl.ctrl_step(dof_targets)

            # Track the trunk with EMA smoothing so the camera follows the
            # robot without jittering on every footstep impulse.
            cam_lookat += cam_alpha * (ctrl.mj_data.qpos[0:3] - cam_lookat)
            cam.lookat[:] = cam_lookat
            renderer.update_scene(ctrl.mj_data, camera=cam)

            # Draw a green arrow showing the current command direction
            # (length scales with target speed). Same idea as the live viewer's
            # _render_command_arrow, but appended directly into the offscreen
            # renderer's MjvScene before calling render().
            scene = renderer.scene
            if scene.ngeom < scene.maxgeom and hasattr(ctrl.policy, "tar_dir"):
                tar_dir = ctrl.policy.tar_dir
                tar_speed = ctrl.policy.tar_speed
                dir_x = float(tar_dir[0])
                dir_y = float(tar_dir[1])
                sp = float(tar_speed[0]) if hasattr(tar_speed, "__len__") else float(tar_speed)
                root_pos = ctrl.mj_data.qpos[0:3].copy()
                arrow_from = np.array(
                    [root_pos[0], root_pos[1], root_pos[2]], dtype=np.float64
                )
                arrow_to = np.array(
                    [root_pos[0] + dir_x * sp,
                     root_pos[1] + dir_y * sp,
                     root_pos[2]],
                    dtype=np.float64,
                )
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

            # HUD: commanded direction (e.g. (1,0)) + commanded/actual speed.
            tar_dir_now = ctrl.policy.tar_dir
            tar_speed_now = ctrl.policy.tar_speed
            dir_xy = (float(tar_dir_now[0]), float(tar_dir_now[1]))
            cmd_speed = (
                float(tar_speed_now[0])
                if hasattr(tar_speed_now, "__len__")
                else float(tar_speed_now)
            )
            vxy = ctrl.mj_data.qvel[0:2]
            actual_speed = float(np.linalg.norm(vxy))
            frame = _overlay_hud(frame, dir_xy, cmd_speed, actual_speed)

            writer.append_data(frame)
    finally:
        writer.close()
        ctrl._flush_logged_states()
        if hasattr(ctrl.policy, "flush_policy_log_if_enabled"):
            ctrl.policy.flush_policy_log_if_enabled()

    print(f"[video] wrote {args.out}")


if __name__ == "__main__":
    main()
