"""Diff a training-side (IsaacLab) obs/action trace against a booster_deploy
MuJoCo policy trace. Intended for sim2sim observation-parity debugging of
a steering policy.

Usage:
  python scripts/compare_obs_traces.py \\
      --mimickit logs/sim2sim_mimickit.npz \\
      --mujoco   logs/sim2sim_mj_policy.npz \\
      --num_steps 3

Step 0 should match bit-for-bit (identical deterministic init + command +
zero velocities). Subsequent steps diverge as the two physics engines pull
apart — diffs after step 0 are expected and do not imply obs bugs.
"""

import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mimickit", required=True)
    p.add_argument("--mujoco", required=True)
    p.add_argument("--num_steps", type=int, default=3)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--top_k", type=int, default=5,
                   help="Print top-K worst indices per mismatched key.")
    args = p.parse_args()

    mk = np.load(args.mimickit)
    mj = np.load(args.mujoco)

    shared = sorted(set(mk.files) & set(mj.files))
    mk_only = sorted(set(mk.files) - set(mj.files))
    mj_only = sorted(set(mj.files) - set(mk.files))

    print(f"MimicKit: {args.mimickit}  ({len(mk.files)} keys)")
    print(f"MuJoCo:   {args.mujoco}  ({len(mj.files)} keys)")
    print(f"Shared keys ({len(shared)}): {shared}")
    if mk_only:
        print(f"MimicKit-only: {mk_only}")
    if mj_only:
        print(f"MuJoCo-only:   {mj_only}")
    print()

    for t in range(args.num_steps):
        print(f"=== step {t} ===")
        for k in shared:
            a = mk[k]
            b = mj[k]
            if t >= a.shape[0] or t >= b.shape[0]:
                print(f"  skip {k}: mimickit has {a.shape[0]} steps, mujoco has {b.shape[0]}")
                continue
            av = a[t].astype(np.float64)
            bv = b[t].astype(np.float64)
            if av.shape != bv.shape:
                print(f"  SHAPE_DIFF {k}: mimickit={av.shape} mujoco={bv.shape}")
                continue
            diff = np.abs(av - bv)
            d_max = float(diff.max()) if diff.size else 0.0
            ok = d_max < args.atol
            tag = "OK  " if ok else "DIFF"
            print(f"  {tag}  {k:40s}  max|diff|={d_max:.3e}  "
                  f"|mk|_max={np.abs(av).max():.3e}  |mj|_max={np.abs(bv).max():.3e}")
            if not ok:
                flat_a = av.reshape(-1)
                flat_b = bv.reshape(-1)
                flat_d = diff.reshape(-1)
                worst = np.argsort(-flat_d)[: args.top_k]
                for i in worst:
                    if flat_d[i] < args.atol:
                        break
                    print(
                        f"        [{i:3d}]  mk={flat_a[i]:+.6f}  "
                        f"mj={flat_b[i]:+.6f}  diff={flat_d[i]:.3e}"
                    )
        print()


if __name__ == "__main__":
    main()
