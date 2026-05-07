"""Step-response test to measure effective MuJoCo joint stiffness.

Spawns the K1 robot in mid-air with gravity disabled, freezes the trunk via
xfrc_applied, and commands a step input on a single joint (default:
Left_Knee_Pitch). Logs the joint position over time and computes rise time
+ steady-state tracking error.

Usage:
    python kp_step_response.py --kp 80
    python kp_step_response.py --kp 4584
    python kp_step_response.py --kp 80 --joint Left_Hip_Pitch --target_deg 20

Run the same test in MimicKit with the actual training-time stiffness, then
compare rise times. Whichever MuJoCo kp gives a matching rise time is the
correct sim-to-sim gain.
"""
import argparse
import os
import sys

import mujoco
import numpy as np

# Resolve the K1 MJCF path the same way the controllers do.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from booster_assets import BOOSTER_ASSETS_DIR  # noqa: E402

K1_MJCF = os.path.join(BOOSTER_ASSETS_DIR, "robots", "K1", "K1_22dof.xml")


def build_position_actuator(model, joint_idx, kp, kd):
    """Reconfigure a single actuator into a position actuator (kp, kd)."""
    model.actuator_gaintype[joint_idx] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[joint_idx] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_gainprm[joint_idx, 0] = kp
    model.actuator_gainprm[joint_idx, 1] = 0.0
    model.actuator_gainprm[joint_idx, 2] = 0.0
    model.actuator_biasprm[joint_idx, 0] = 0.0
    model.actuator_biasprm[joint_idx, 1] = -kp
    model.actuator_biasprm[joint_idx, 2] = -kd
    model.actuator_ctrllimited[joint_idx] = 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kp", type=float, default=80.0,
                   help="Proportional gain (N·m/rad). Try 80 vs 4584.")
    p.add_argument("--kd", type=float, default=2.0,
                   help="Derivative gain (N·m·s/rad).")
    p.add_argument("--joint", default="Left_Knee_Pitch",
                   help="Joint name to step.")
    p.add_argument("--target_deg", type=float, default=15.0,
                   help="Step target in degrees from zero.")
    p.add_argument("--duration", type=float, default=0.5,
                   help="Total simulation time in seconds.")
    p.add_argument("--dt", type=float, default=1.0 / 480.0,
                   help="Physics timestep.")
    p.add_argument("--integrator", default="implicit",
                   choices=["euler", "implicitfast", "implicit"])
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(K1_MJCF)
    model.opt.timestep = args.dt
    model.opt.gravity[:] = 0.0  # disable gravity for clean step response

    integrator_map = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    }
    model.opt.integrator = integrator_map[args.integrator]

    # Find the joint and its actuator
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, args.joint)
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, args.joint)
    if jnt_id < 0 or act_id < 0:
        raise RuntimeError(f"Joint or actuator '{args.joint}' not found")
    qpos_idx = model.jnt_qposadr[jnt_id]
    qvel_idx = model.jnt_dofadr[jnt_id]

    # Make every actuator a high-gain position actuator at zero target so the
    # rest of the body stays put. Then override our target joint's gains.
    n_act = int(model.nu)
    for i in range(n_act):
        build_position_actuator(model, i, kp=args.kp, kd=args.kd)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Pin the trunk in place using a free-joint weld via xfrc_applied is
    # awkward; easiest: set ctrl=0 for everything (so all joints are servoed
    # to zero), no gravity, and let the floating base float freely. With no
    # gravity and zero command, nothing happens.
    target_rad = np.deg2rad(args.target_deg)

    # Time series
    n_steps = int(args.duration / args.dt)
    times = np.zeros(n_steps)
    positions = np.zeros(n_steps)
    velocities = np.zeros(n_steps)
    torques = np.zeros(n_steps)

    # Apply step at t=0
    data.ctrl[:] = 0.0
    data.ctrl[act_id] = target_rad

    for k in range(n_steps):
        mujoco.mj_step(model, data)
        times[k] = data.time
        positions[k] = data.qpos[qpos_idx]
        velocities[k] = data.qvel[qvel_idx]
        torques[k] = data.qfrc_actuator[qvel_idx]

    # Metrics
    final_pos = positions[-1]
    steady_err_deg = np.rad2deg(target_rad - final_pos)
    # Rise time: first time we reach 90% of target
    threshold = 0.9 * target_rad
    rise_idx = np.argmax(positions >= threshold) if (positions >= threshold).any() else None
    rise_time_ms = times[rise_idx] * 1000 if rise_idx is not None and rise_idx > 0 else None
    # Overshoot
    peak = positions.max()
    overshoot_pct = ((peak - target_rad) / target_rad) * 100 if target_rad != 0 else 0.0
    # Max torque
    peak_torque = np.abs(torques).max()

    print(f"\n=== Step response: {args.joint} → {args.target_deg}° ===")
    print(f"  kp        = {args.kp} N·m/rad")
    print(f"  kd        = {args.kd} N·m·s/rad")
    print(f"  integrator = {args.integrator}")
    print(f"  dt        = {args.dt*1000:.2f} ms ({1/args.dt:.0f} Hz)")
    print(f"  duration  = {args.duration*1000:.0f} ms")
    print()
    print(f"  rise time (90%):    {rise_time_ms:.1f} ms" if rise_time_ms else "  rise time (90%):    NEVER REACHED")
    print(f"  overshoot:          {overshoot_pct:+.2f} %")
    print(f"  steady-state error: {steady_err_deg:+.3f}°")
    print(f"  peak torque:        {peak_torque:.2f} N·m")
    print()

    # Save a CSV for plotting
    csv_path = f"/tmp/step_response_kp{int(args.kp)}_{args.joint}.csv"
    with open(csv_path, "w") as f:
        f.write("time_s,pos_rad,vel_rad_s,torque_nm,target_rad\n")
        for i in range(n_steps):
            f.write(f"{times[i]:.6f},{positions[i]:.6f},{velocities[i]:.6f},"
                    f"{torques[i]:.4f},{target_rad:.6f}\n")
    print(f"  trajectory written to {csv_path}")


if __name__ == "__main__":
    main()
