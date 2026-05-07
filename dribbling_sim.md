# MimicKit Dribbling — Sim2Sim Deployment Notes

Adds a deploy task `k1_mimickit_dribbling` that runs the MimicKit B012 K1
dribbling policy in MuJoCo with a ball spawned in the scene.

## Files added or changed

```
tasks/mimickit_dribbling/
├── __init__.py              # task registration; composes K1 + ball MJCF
├── dribbling_policy.py      # 84-D obs assembly, command props, FOV check, ball history
└── models/
    └── B012_model.pt        # TorchScript export of B012 (auto-inferred shape)

nomadz_deploy/controllers/
└── mujoco_controller.py     # qpos init + PD slice + state slice now use n_dof,
                             # so composite scenes (robot + extra bodies) work
                             # without breaking single-body scenes

scripts/
└── sim2sim_mj_video.py      # tar_omega assignment now optional (dribbling has no omega)
```

The composite scene is written to
`{BOOSTER_ASSETS_DIR}/robots/K1/_k1_dribbling_scene.xml` at task-import time
so the K1 mesh paths (`meshdir="meshes/"`) resolve natively.

## Policy / observation contract

The B012 checkpoint has obs_dim=84, act_dim=22, actor MLP [1024, 512].
`scripts/export_amp_policy.py` infers this from the state-dict and bakes
the obs/action normalizers + action clip into a TorchScript module.

Observation layout (84D, mirrors `task_k1_dribbling_env.compute_*_obs`):

| Slice | Dim | Component                                |
|-------|----:|------------------------------------------|
| 0:3   | 3   | gravity_body                             |
| 3:6   | 3   | ang_vel_body                             |
| 6:28  | 22  | dof_pos in policy joint order            |
| 28:50 | 22  | dof_vel in policy joint order            |
| 50:80 | 30  | ball XY history × 15, robot heading frame |
| 80:82 | 2   | local_tar_dir                            |
| 82:83 | 1   | tar_speed                                |
| 83:84 | 1   | is_in_view (1.0 / 0.0)                   |

Ball history slot 0 is zeroed when the ball is out of the camera's FOV;
otherwise it holds the ball XY in the robot's heading frame at that step.

## Camera

Uses Head_2 body world pose, applies the trained offset
`(x=0.059, y=0.0115, z=0.098)` and a -12.2° pitch quaternion. FOV defaults
to 87°×58°. All four are cfg-driven on `DribblingPolicyCfg`.

## Ball physics

| Property        | Value      | Source                          |
|-----------------|------------|---------------------------------|
| radius          | 0.08 m     | training env (sphere.usd geom)  |
| mass            | 0.45 kg    | training env                    |
| friction        | 0.7        | midpoint of training [0.55, 0.85] |
| restitution     | 0.6        | training env                    |
| condim          | 3          | needed for tangential friction  |

Spawn pose is configurable on the cfg; default is 1.5 m forward of root,
0.08 m above ground. Set in `DribblingPolicy.reset()` so it re-initializes
on every episode start.

## Running

```bash
cd /home/nomadz-control/nomadz_deploy
conda activate booster_deploy

# Interactive viewer
python scripts/deploy.py --task k1_mimickit_dribbling --mujoco

# Headless video (works over SSH)
python scripts/sim2sim_mj_video.py \
    --task k1_mimickit_dribbling \
    --out logs/dribbling_run.mp4 \
    --duration 10.0 \
    --tar-dir-x 1.0 --tar-speed 1.0
```

Steering input is the same (`tar_dir_x  tar_dir_y  speed`) as the steering
task — `tar_omega` does not apply to dribbling.

## Reusing for another model

The `DribblingPolicy(JitPolicy)` and `DribblingPolicyCfg` are written so a
checkpoint with the same 84-D obs / 22-D action can drop straight in:

1. Run `python scripts/export_amp_policy.py --ckpt <state_dict.pt> --out tasks/mimickit_dribbling/models/<name>.pt` (auto-infers any MLP shape).
2. Update `CHECKPOINT_PATH` in `tasks/mimickit_dribbling/__init__.py`.

A different obs layout means subclassing `DribblingPolicy.compute_observation()`;
a different robot means swapping `K1_CFG` and `POLICY_JOINT_NAMES`.

## Why the controller changes were needed

The MuJoCo controller's qpos initialization, manual PD step, and
`update_state` previously assumed `nq == 7 + 22` (root free joint + 22
named joints). Adding any free-jointed body past the robot (e.g. the ball,
which adds 7 qpos / 6 qvel) broke the slice math. The edits replace the
hardcoded slice with `7 : 7 + len(joint_names)`, which is identical to the
old behavior for single-body scenes and correctly skips the extra bodies
for composite scenes.

## Smoke-test checks performed

1. Composite scene compiles: `nq=36, nv=34, nbody=25` (K1 29/28/24 + ball 7/6/1).
2. JIT export of `MimicKit/output/K1_Dribbling_output/B012/model.pt`
   succeeds with `obs_dim=84 act_dim=22 hidden=[1024, 512]`, scripted vs eager `max_diff=0.0`.
3. Single inference step produces a 22-D action in the expected range.
4. Headless video recording of 10 s of sim time writes `logs/dribbling_test.mp4` (~2.9 MB).
