# nomadz_deploy — Booster K1/T1 deployment framework

Lightweight policy-deployment framework (fork of Booster Deploy): runs trained RL policies
on the real Booster robot (sim2real via ROS 2 / Booster SDK), in MuJoCo (sim2sim), or
Webots. Borrows IsaacLab-style `configclass` configs (vendored under
`nomadz_deploy/utils/isaaclab/` — no Isaac install needed).

**Python env: conda `booster_deploy`** → `/home/gan/miniconda3/envs/booster_deploy/bin/python`
(torch + mujoco + pygame; `booster_robotics_sdk_python` is only on the robot).

## Running

```bash
cd ~/NOMADZ/nomadz_deploy
python scripts/deploy.py --list                                   # show registered tasks
python scripts/deploy.py --task <NAME> --mujoco                   # sim2sim, stdin commands
python scripts/deploy.py --task <NAME> --mujoco --joystick        # gamepad teleop
python scripts/deploy.py --task <NAME>                            # real robot (needs SDK)
```

Useful flags: `--vx-max/--vy-max/--vyaw-max` (joystick limit overrides),
`--log-stem logs/foo` (writes `foo.npz` MuJoCo state + `foo_policy.npz` policy trace).
In MuJoCo, **Backspace** resets the robot (and re-spawns the ball in ball tasks);
auto fall-reset triggers when root z < `fall_height_threshold` if enabled.

## Architecture

```
scripts/deploy.py            # entry point; auto-imports every module under tasks/ so they self-register
nomadz_deploy/
├─ controllers/
│  ├─ base_controller.py     # BaseController loop contract: update_state → policy_step → ctrl_step
│  │                         # also Policy ABC, BoosterRobot/RobotData (state buffers), VelocityCommand
│  ├─ controller_cfg.py      # ControllerCfg / RobotCfg / PolicyCfg / MujocoControllerCfg (push, throw,
│  │                         # fall-reset, native-PD options) / SteeringJoystickCommandCfg
│  ├─ mujoco_controller.py   # sim2sim: viewer loop, PD (explicit Python or native implicit), joystick
│  │                         # HUD bars, ghost-reference rendering, random push / projectile-throw
│  └─ booster_robot_controller.py  # sim2real portal (ROS 2 /low_state + /joint_ctrl)
├─ robots/booster.py         # K1_CFG (22 DOF) and T1_23DOF_CFG: joint tables, kp/kd, effort limits,
│                            # mjcf_path, prepare_state (high-gain pose used when entering custom mode)
└─ utils/
   ├─ registry.py            # register_task(name, cfg) / get_task / list_tasks
   ├─ joystick_handler.py    # pygame Xbox-pad reader: axes → vx/vy/vyaw, button rising-edge events
   └─ isaaclab/              # vendored configclass & math helpers
tasks/                       # one folder per task: __init__.py registers a ControllerCfg
booster_assets/              # nested git repo: K1/T1 MJCF + meshes; BOOSTER_ASSETS_DIR points here
```

Control flow: each `policy_dt` (usually 1/30 s) the controller refreshes `robot.data`
from the sim/robot, calls `policy.inference()` → 22 absolute joint-position targets
(real joint order), then PD-tracks them for `decimation` physics substeps.

## Tasks

| Task name | What | Model | Commands |
|-----------|------|-------|----------|
| `t1_walk` | T1 velocity walking | `tasks/locomotion/models/t1_walk.pt` | `vel_command` (vx vy vyaw) |
| `k1_mimickit_steering` | K1 MimicKit steering walk | `tasks/mimickit_steering/models/A*.pt` | `tar_dir`+`tar_speed`+`tar_omega` |
| `k1_mimickit_dribbling` | K1 ball dribbling (84-D obs: ball history + camera FOV) | see ⚠ below | `tar_dir`+`tar_speed` (no omega) |
| `k1_mimickit_combo` | **Combined steering+dribbling with mode state machine** (see below) | A029 + dribbling model | joystick / keys switch modes |
| `k1_mimickit_steering_video` | headless mp4 recording variant | own `models/` dir | `--schedule` |
| `beyond_mimic` | motion-tracking replay | `tasks/beyond_mimic/models/` | — |

Two command plumbing paths exist:
- **`vel_command`** (locomotion): `ControllerCfg.vel_command` + `update_vel_command()`.
- **steering-style** (MimicKit tasks): `vel_command = None`; the policy exposes
  `tar_dir` (world-frame unit XY), `tar_speed`, optionally `tar_omega` properties, and the
  controller detects this via `hasattr(policy, 'tar_dir')` and feeds it from
  joystick (`steering_joystick_command` limits) or stdin.

### MimicKit policies

`tasks/mimickit_steering/jit_policy.py::JitPolicy` is the shared base: loads a TorchScript
actor exported by `scripts/export_amp_policy.py` (normalizers + action clip baked in),
handles real↔policy joint reordering (policy order = URDF DFS order, listed in
`POLICY_JOINT_NAMES`), EMA action smoothing, and `.npz` trace logging.
Subclasses implement `compute_observation()`:

- `MimicKitPolicy` (steering): 52-D `vxomega` or 53-D `xyomega` layout — **must match the
  checkpoint** (A027/A029 = xyomega, A030 = vxomega); mismatch = tensor-size error in the
  scripted normalizer.
- `DribblingPolicy`: 84-D = proprio(50) + ball-XY history(30, heading frame, zeroed when
  ball outside the head-camera FOV) + local_tar_dir(2) + tar_speed(1) + in_view(1).
  Needs `controller.mj_model` (ball/camera body lookup) → **MuJoCo-only as written**.

### ⚠ Known checkpoint gotchas (state as of 2026-06)

- `k1_mimickit_steering` cfg points at `models/A030_model.pt` which is **not on disk**
  (A018–A029 are; A018 is a broken export). Use A029 (xyomega) or restore A030.
- `k1_mimickit_dribbling` cfg points at `models/B024_model.pt` which is **not on disk**;
  the working dribbling TorchScript lives at `tasks/mimickit_dribbling.model.pt`
  (84→22, verified). The combo task references that file directly.
- The dribbling scene composer includes `K1_22dof_orig.xml`, which doesn't exist in
  `booster_assets/robots/K1/` (only `K1_22dof_fixed.xml` does; the original `K1_22dof.xml`
  is deleted in the working tree). The combo task composes from `K1_22dof_fixed.xml`.

## Combined steering+dribbling task (`k1_mimickit_combo`)

`tasks/mimickit_combo/` — one MuJoCo scene (K1 + RoboCup field + ball), two
sub-policies, a mode state machine (STEERING ⇄ DRIBBLING):

```bash
python scripts/deploy.py --task k1_mimickit_combo --mujoco --joystick
```

- **Y button** (or `M` key in the viewer) toggles mode; **A button** (or `B` key)
  re-spawns the ball next to the robot; Backspace = full reset.
- Left stick = vx/vy, right stick X = yaw (steering mode only; dribbling has no omega).
- Switching blends joint targets over ~0.5 s and seeds the incoming policy's EMA state
  from the last commanded targets; the dribbling ball-history buffer is kept warm every
  step (its `compute_observation()` runs even in steering mode) so the handoff is clean.
- Physics at 120 Hz / control 30 Hz (decimation 4) — the regime both checkpoints were
  validated in (`mimickit_steering_video` and the dribbling tuning notes).
- The scene is a RoboCup **HSL 2026 Middle Division** setup (rules release
  `rules-2026-v1.0`), composed by `nomadz_deploy/utils/robocup_field.py`: M-Field
  14×9 m green turf + white markings (visual-only geoms; turf plane keeps geom name
  `ground` for the controller's friction override), two collidable goals (2.6 m wide ×
  1.8 m high; nets drawn as visual-only strand grids, the ball is caught by
  invisible panels on the same faces), FIFA size-3 ball r=0.094 m / 0.31 kg.
  ⚠ The dribbling checkpoint was trained on r=0.08 m / 0.45 kg, so ball response
  differs slightly. S-/L-Field and FIFA size-4/5 presets live in the same module;
  the standalone dribbling task still uses its old flat-ground scene.

## Conventions / gotchas

- **Joint order**: `RobotCfg.joint_names` is the *real robot* order (head 2, arms 8,
  legs 12); MimicKit policies use URDF-DFS order; Isaac uses `sim_joint_names`. All
  reordering is name-based — never index by position across these lists.
- Quaternions: MuJoCo qpos is **wxyz**; the policy math helpers convert to xyzw at the
  boundary. MuJoCo free-joint qvel = world-frame linvel + **body-frame** angvel.
- `MujocoControllerCfg.use_native_pd`: False = explicit Python PD per substep (default,
  matches original repo); True = rewrite actuators to implicit position servos
  (closer to PhysX) — preferred for bouncy-contact tasks.
- Task `__init__.py` files run arbitrary code at import (scene-XML generation, registry);
  `deploy.py` imports *every* `tasks/` submodule, so a broken task breaks `--list`.
- Composite scenes (robot + ball/cube): robot qpos/qvel occupy slices `[0:7+22]` /
  `[0:6+22]`; extra free bodies live past that, owned by the task's policy
  (`reset()` spawns them).
- Generated scene XMLs are written into `booster_assets/robots/K1/` (prefixed `_`) so
  relative `meshdir="meshes/"` resolves; they are regenerated on import when the
  template changes.
- The attached gamepad is a 2.4G Xbox-360 clone at `/dev/input/js0` ("Generic X-Box pad"
  in pygame: 6 axes, 11 buttons; A=0 B=1 X=2 Y=3 LB=4 RB=5).
- `requirements.txt`: torch, mujoco, scipy, evdev, pygame.
