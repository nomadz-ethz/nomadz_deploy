# Booster Deploy

Booster Deploy is a lightweight deployment framework that supports running control policies on Booster robots (sim2real), MuJoCo (sim2sim), and Webots (internal sim2sim). The system adopts many well-established designs from IsaacLab to provide modular abstractions, allowing unified policy execution across both simulated and real robotic platforms.


## Prerequisites

| Environment | Notes |
|-------------|-------|
| Booster firmware >= v1.4 | Required for real robot deployments. |
| Python 3.10+ | Already installed on the robot |
| ROS 2 Humble | Required for `/low_state` + `/joint_ctrl` topics. Already installed on the robot. |
| MuJoCo / Webots | Optional; install if you plan to run the respective simulators. |


## Running Deployments

### Add and list tasks:
   1. Create a subfolder under `tasks/` for your task.
   2. Implement a `Policy`/`PolicyCfg` and provide a `ControllerCfg` referencing the policy.
   3. Place policy checkpoints under `models/` and reference the path in the config.
   4. Register your `ControllerCfg` config in the task registry (see existing tasks for the registration pattern).
   5. Check all available tasks:
      ```bash
      python3 scripts/deploy.py --list
      ```

### Run Sim2Sim (MuJoCo)

- Download and install BoosterAssets:
   - Clone the [booster_assets](https://github.com/BoosterRobotics/booster_assets) which contains Booster robot models and resources.
   - Install booster_assets python helper following the instructions in the repository.

- Install Python dependencies on local machine:
   ```
   pip install -r requirements.txt
   ```

- Launch the task in mujoco:
   ```bash
   python scripts/deploy.py --task <TASK_NAME> --mujoco
   ```

- Enable joystick teleoperation in MuJoCo with a connected gamepad:
   ```bash
   python scripts/deploy.py --task <TASK_NAME> --mujoco --joystick
   ```

- Override joystick command limits from the CLI:
   ```bash
   python scripts/deploy.py --task <TASK_NAME> --mujoco --joystick --vx-max 2.0 --vy-max 1.0 --vyaw-max 3.0
   ```

- Notes for joystick mode:
   - Left stick controls forward/backward and lateral motion.
   - Right stick left/right controls yaw.
   - Locomotion tasks use their `vel_command` limits by default.
   - `k1_mimickit_steering` also supports `--joystick` and uses its steering joystick limits by default.

- Example commands:
   ```bash
   python scripts/deploy.py --task t1_walk --mujoco --joystick
   python scripts/deploy.py --task k1_mimickit_steering --mujoco --joystick --vx-max 1.5 --vy-max 1.0 --vyaw-max 1.2
   ```

### Headless video recording (steering)

Record an mp4 of the K1 steering policy in MuJoCo over SSH (no display required). The pipeline is fully isolated from the live `k1_mimickit_steering` task — it uses its own task (`k1_mimickit_steering_video`), its own model directory, and a dedicated recorder script.

- Convert a MimicKit steering checkpoint to TorchScript:
   ```bash
   cd /home/nomadz-control/nomadz_deploy
   python scripts/export_amp_policy.py \
       --ckpt /home/nomadz-control/MimicKit/output/K1_Steering_output/<RUN>/model.pt \
       --out  tasks/mimickit_steering_video/models/<NAME>_model.pt
   ```
   Then update `CHECKPOINT_FILENAME` in [`tasks/mimickit_steering_video/__init__.py`](tasks/mimickit_steering_video/__init__.py) to `"<NAME>_model.pt"`.

- Record a video with a time-varying command schedule:
   ```bash
   conda activate gmr
   cd /home/nomadz-control/nomadz_deploy
   python scripts/sim2sim_mj_video_steering.py --task k1_mimickit_steering_video \
       --out logs/steer_yaw_fwd_right.mp4 \
       --schedule "10:0,0,1.5;10:1,0,0;10:0,-1,0"
   ```
   This yaws on the spot → walks forward → walks right (10 s each, 30 s total).

- Schedule format: `dur:vx,vy,omega[;dur:vx,vy,omega]…`
   - `dur` — segment duration in seconds
   - `vx,vy` — target velocity in the robot's **body frame** (m/s); updated every control step as the robot yaws, so `(1,0)` always means forward relative to wherever the robot is currently facing, `(0,-1)` = right, `(0,1)` = left
   - `omega` — target yaw rate in rad/s (positive = counter-clockwise)

- Camera follows the robot's XY position but holds a fixed world-frame height so vertical bobbing is clearly visible. Default lookat height is 0.5 m; override with `--camera-lookat-height <m>`.

- For the dribbling counterpart, see [`scripts/sim2sim_mj_video.py`](scripts/sim2sim_mj_video.py) and the `k1_mimickit_dribbling` task — same pipeline shape, but the schedule has 3 fields (`dx,dy,sp`, no omega) and the scene includes a ball.

### Run Sim2Real (Real Robots)

**IMPORTANT**: Make sure to install [Booster Firmware](https://booster.feishu.cn/wiki/E3q5wF5SnitXZgkY18Uc8odBnXb) >= v1.4 on the robot before proceeding.

**NOTE**: If you plan to deploy on the T1 Standard Edition robot, you need to choose to deploy on the **motion board** rather than the perception board.

- After you finish testing your task with Sim2Sim locally, copy the project to the robot.

- Install Booster Robotic SDK on robot:
   - Clone the latest [Booster Robotics SDK](https://github.com/BoosterRobotics/booster_robotics_sdk) repository into the robot.
   - Follow the build instructions in the SDK repository.
   - **Important**: Make sure to build and install the Python bindings:
     ```bash
     cd booster_robotics_sdk
     mkdir build && cd build
     cmake .. -DBUILD_PYTHON_BINDING=ON
     make -j$(nproc)
     sudo make install
     ```

- Install Python dependencies on the robot:
   ```
   pip install -r requirements.txt
   ```

- SSH into the robot and start the ROS 2 environment by sourcing the provided setup script:
   ```bash
   source /opt/booster/BoosterRos2Interface/install/setup.bash
   ```

- Launch the task on the robot and follow the prompts shown in the command line..
   ```bash
   python3 scripts/deploy.py --task <TASK_NAME>
   ```


## Repository Layout

```
booster_deploy/
├─ booster_deploy/           # Controllers, policies, utilities
├─ scripts/                  # Entry-point scripts (deploy.py)
├─ tasks/                    # Task registry and configs
├─ requirements.txt          # Python dependencies
└─ fastdds_profile.xml       # Default FastDDS settings for ROS 2
```

Key modules:
- `booster_deploy/`: Core module providing a unified abstraction for both simulators and physical robots, and handling communication via ROS 2 (implements a /low_state subscriber and a /low_cmd publisher to bridge policies to hardware).
- `booster_deploy/robots/`: Robot configuration modules. This folder contains booster robot configs by defining a `RobotCfg` describing:
    - joint names and body names
    - default joint positions
    - default joint stiffness (`joint_stiffness`) and damping (`joint_damping`)
    - effort limits
    - `mjcf_path` for MuJoCo model loading
    - `prepare_state` (prepare pose, stiffness and damping used when entering custom mode)

 - `tasks/`: User task definitions and implementations. Each task module contains:
    - `Policy`/`PolicyCfg` class implementing the inference logic;
    - a `ControllerCfg` class describing the task configuration including the policy;
    - registering a task with a `ControllerCfg` instance.

   Typical task layout (example):

   ```text
   tasks/my_task/
   ├─ __init__.py        # registers the task via utils.register.register_task
   ├─ task.py            # Policy and ControllerCfg implementation
   ├─ models/            # optional policy checkpoints
   └─ motions/           # optional motion primitives or recordings
   ```
