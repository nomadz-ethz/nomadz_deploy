# Recovery prototype — T1 deployment runbook

> **Status:** prototype, hoisted-bench validation only. **Do not drop the
> hoist** until every step in §10 (validation) has passed.
>
> This runbook turns the recovery state machine in
> `docs/RECOVERY_INTEGRATION_PLAN.md` (v2.1) into an end-to-end checklist
> for running it on the real T1: hardware setup → network → SSH → ROS 2
> sourcing → joystick → deploy → Foxglove → shutdown. Cross-references the
> `nomadz-neo.wiki` (T1-Robot-Configuration-Guide, T1-Development-Log) and
> the `RECOVERY_MOTION_REPORT.md` from the SDK.
>
> **Date:** 2026-05-08. **Author:** recovery prototype integration.

---

## Quick reference

```bash
# On the robot (motion board), one terminal:
ssh master@192.168.10.101                              # password 123456
cd ~/nomadz_deploy
source /opt/booster/BoosterRos2Interface/install/setup.bash

# Keyboard mode  — recommended for first run / no pad available:
python3 scripts/deploy.py --task t1_walk --keyboard

# Joystick mode  — production path with the pygame controller:
python3 scripts/deploy.py --task t1_walk --joystick

# Same robot, second terminal (same SSH login or another):
ssh master@192.168.10.101
source /opt/booster/BoosterRos2Interface/install/setup.bash
ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765

# On the workstation:
# Foxglove Studio → Open connection → Foxglove WebSocket
#   → ws://192.168.10.101:8765
# Layout → Import from file → nomadz_deploy/configs/foxglove_recovery.json
```

Operator inputs — the FSM accepts the same buttons from either backend:

| Press | In state | Effect |
|---|---|---|
| **Y** | IDLE | Ramp into prepare pose, enter CUSTOM mode → PREP_READY |
| **Y** | PREP_READY | Spawn inference subprocess → RUNNING (RL policy active) |
| **Y** | STAGED | Resume locomotion after a recovery (fresh subprocess) → RUNNING |
| **B** | RUNNING | Trigger full recovery cycle (DAMP→PREP→GetUp→PREP→STAGED) |
| **X** | RUNNING | Emergency damp (pause inference, ChangeMode kDamping) → IDLE |
| (sticks / wasd) | RUNNING | Drive vx/vy/vyaw |

In **joystick mode** (`--joystick`), the buttons are physical Y/B/X on the
pygame controller and the sticks are the analog inputs.

In **keyboard mode** (`--keyboard`), the same FSM reads from stdin in
cbreak mode: `y/b/x` are the buttons; `w/s` increment/decrement vx,
`a/d` for vy, `q/e` for vyaw, `Space` zeros all axes. Each axis press
adds 0.25 to the target so four presses saturate.

The manufacturer's joystick keeps doing whatever the firmware bound it to
(LT+UP for GetUp under DAMP/PREP, etc.). It is **independent** of our
state machine — operate one or the other, not both, to avoid confusion.

---

## 1. Pre-flight (do once per session, before powering anything)

* [ ] **Hoist rigged.** Robot suspended from a hoist with ~5 cm of slack
  at the chest harness. Enough rope to allow a fall *to the cushioned mat*
  but not enough to free-hang. Per `RECOVERY_MOTION_REPORT.md` §5.5: *"While
  developing new tricks on K1, it is recommended to use a Hoist at all
  times under CUSTOM mode."* Same applies to T1 here.
* [ ] **Floor area cleared in a 1.5 m radius.** During a `GetUp()` the
  robot rolls onto its back / pushes up; debris under it can wedge a limb
  and cause `GetUp` to abort with a non-zero return code (the FSM will
  then enter `EXITING` and you'll have to restart).
* [ ] **Cushion mat under the robot.** Even with the hoist, the recovery
  sequence is mechanically aggressive — the chest will impact the mat
  before the legs catch.
* [ ] **Battery ≥ 50 %.** Recovery draws large currents; a low battery is
  the most common cause of a stalled GetUp mid-cycle.
* [ ] **E-stop reachable.** The hardware E-stop on the back/side of the
  robot must be within arm's reach of the operator. **The X press is not
  an E-stop** — it stops the policy and damps joints, but firmware faults
  can still command motion.
* [ ] **Workstation laptop on the same wired subnet.** Wired interface on
  the workstation set to static `192.168.10.10/24` (per
  `nomadz-neo.wiki/T1-Robot-Configuration-Guide.md`). If your workstation
  is sharing internet to the robot, the iptables NAT rules from the wiki
  must be active or the robot can't reach the public network for any
  remote logging — but this is **not required** for the deploy itself,
  the deploy is fully local.
* [ ] **Booster firmware ≥ v1.6** preferred (so `GetUpWithMode` semantics
  match the wiki we used to design the FSM). v1.4 is the absolute minimum
  per `README.md`. Check via `client.GetStatus()` once you're up.

---

## 2. One-time setup (skip if already done on this robot)

This is per-robot, not per-session. If the robot has been rebuilt or the
SDK was reinstalled, redo it.

### 2.1 SDK on the motion board

Per `nomadz-neo.wiki/T1-Development-Log.md`, the SDK lives in
`~/Playground/booster_robotics_sdk` on each board. On the motion board:

```bash
ssh master@192.168.10.101
mkdir -p ~/Playground && cd ~/Playground
git clone https://github.com/BoosterRobotics/booster_robotics_sdk.git
cd booster_robotics_sdk

# Build prerequisites (one-off):
pip3 install pybind11-stubgen    # required by SDK CMakeLists for .pyi gen
                                 # — without this, `cmake -DBUILD_PYTHON_BINDING=ON`
                                 # fails at "pybind11-stubgen not found".

mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDING=ON   # critical — without this the
                                     # Python `booster_robotics_sdk_python`
                                     # module won't exist and our deploy
                                     # script can't import RobotMode/B1LocoClient.
make -j$(nproc)
sudo make install
```

> ⚠️ If you forget `-DBUILD_PYTHON_BINDING=ON`, `scripts/deploy.py` will
> bail with `ImportError: booster_robotics_sdk_python is not installed`
> at line 115 of `deploy.py`. The fix is to rebuild with the flag — the
> default-built C++-only SDK is **not** sufficient.

> ⚠️ If `cmake` fails with `pybind11-stubgen not found` (CMakeLists.txt
> line 52), `pip3 install pybind11-stubgen` is the fix. After installing,
> **clear the cmake cache before re-running** — CMake remembers the
> failed find:
>
> ```bash
> cd ~/Playground/booster_robotics_sdk/build
> rm -rf CMakeCache.txt CMakeFiles/
> cmake .. -DBUILD_PYTHON_BINDING=ON
> ```
>
> Verify the install with `which pybind11-stubgen`. If it's missing from
> `PATH` despite the pip install succeeding, either prefix `~/.local/bin`
> to `PATH`, or use `sudo pip3 install pybind11-stubgen` to drop it in
> `/usr/local/bin` instead.

### 2.2 nomadz_deploy on the motion board

If this is a fresh install:

```bash
# On the workstation:
rsync -av --exclude '.git' --exclude '__pycache__' --exclude 'logs' \
  /home/nomadz-control/nomadz_deploy/ master@192.168.10.101:~/nomadz_deploy/

# On the motion board:
ssh master@192.168.10.101
cd ~/nomadz_deploy
pip3 install -r requirements.txt   # torch, mujoco, scipy, evdev, pygame
                                   # (mujoco isn't strictly needed on robot,
                                   # but the import path expects it; ~50 MB)
```

### 2.3 Foxglove bridge on the motion board

```bash
sudo apt update
sudo apt install -y ros-humble-foxglove-bridge
```

(The `ros-humble-*` is what's already on the robot per `README.md`.
If you're on a different ROS distro for some reason, sub the right name.)

### 2.4 T1 Standard Edition warning

Per `README.md`: *"If you plan to deploy on the T1 Standard Edition robot,
you need to choose to deploy on the **motion board** rather than the
perception board."* This runbook assumes the motion board (`master@…101`).
If you're on a non-Standard T1 you may have a different topology — check
which board has the `joint_ctrl` publisher subscribed.

---

## 3. Power-on and boot

1. Lower the robot onto the mat, hoist tight enough that no joint takes
   weight at boot.
2. Press the power button on the back. Wait for the boot LEDs (≈ 30 s).
3. The firmware comes up in **PREP** mode by default. You'll see the
   robot stand on its own with the manufacturer's prepare pose. **Do not
   walk it** with the manufacturer's controller — leaving it in PREP is
   the legal hop point for our `_ramp_to_prepare_state` sequence.
4. If the robot booted into another mode (e.g. PROTECT after a power-cycle
   from a fall), use the manufacturer's joystick to drive
   PROTECT → DAMP → PREP. The wiki rules: PROTECT only goes to DAMP, then
   DAMP → PREP. Confirm by watching the LEDs / hearing the joints lock
   into the prepare pose.

---

## 4. Network sanity checks

From the workstation:

```bash
ping -c 3 192.168.10.101                  # motion board
ping -c 3 192.168.10.102                  # perception board (sanity, optional)
ssh master@192.168.10.101 'echo OK'       # password 123456
```

If `ping` works but `ssh` fails, the SSH key may have rotated — see
`T1-Development-Log.md` ("remove ssh identity key on boards") for how the
keys are managed. Re-run `ssh-copy-id` from the workstation if needed.

If neither works:

* On the workstation, confirm the wired interface is up and on
  `192.168.10.10/24`:
  ```bash
  ip -4 addr show eth1   # or the actual interface name
  ```
* Confirm the robot's wired interface has its static IP:
  ```bash
  ssh master@192.168.10.101 'ip -4 addr show enp1s0'
  # expect inet 192.168.10.101/24
  ```
* If the `robot-T1` NetworkManager profile is hijacking your default
  route on the workstation, undo it per `T1-Robot-Configuration-Guide.md`:
  ```bash
  nmcli connection modify "robot-T1" ipv4.never-default yes
  nmcli connection modify "robot-T1" ipv6.method disabled
  ```

---

## 5. ROS 2 environment on the motion board

Open **two SSH terminals** to the motion board (one for the deploy, one
for the Foxglove bridge). In **each** of them, source the ROS 2 env:

```bash
ssh master@192.168.10.101
source /opt/booster/BoosterRos2Interface/install/setup.bash
```

Verify that ROS 2 sees the robot's existing topics:

```bash
ros2 topic list | grep -E '/(low_state|joint_ctrl)'
```

You should see `/low_state` (the firmware is publishing it). If you do
**not** see `/low_state`, the firmware DDS publisher isn't reaching this
node — the most common causes:

* `ROS_DOMAIN_ID` mismatch: the firmware uses domain `0`. Make sure your
  shell hasn't exported a different domain. `echo $ROS_DOMAIN_ID` should
  be empty or `0`.
* `RMW_IMPLEMENTATION` mismatch: the SDK uses Fast DDS, so ROS 2 must
  also use `rmw_fastrtps_cpp`. `echo $RMW_IMPLEMENTATION` should be empty
  (default) or `rmw_fastrtps_cpp`. If it's set to Cyclone DDS, the topic
  names will not bridge.

Once `/low_state` is visible, you're cleared for §6.

---

## 6. Keyboard mode (recommended first run)

Before plugging in any operator pad, **drive the FSM from the keyboard**.
This validates the deploy path, the ROS 2 sourcing, the firmware mode
transitions, and the Foxglove dashboard end-to-end without depending on
pygame, the pad's button mapping, or USB pass-through over SSH. Once
keyboard mode works, switching to joystick is a one-flag change.

### 6.0.1 Why start with keyboard

* No driver issues. `KeyboardHandler` reads stdin in cbreak mode — no
  pygame, no SDL device probing, no `joystick.get_button()` indices to
  empirically verify (§6.1 in joystick mode is skipped entirely).
* Works over plain SSH. As long as your terminal is a real TTY (not a
  pipe — see §6.0.4), the runbook works without any USB pass-through.
* Same FSM. The recovery state machine reads the *same* `consume_press`
  surface on either backend, so a successful keyboard run is direct
  evidence the FSM is wired up correctly. When you later plug in the
  pad, the only failure modes can be pad-specific (button index, pygame
  init, calibration).

### 6.0.2 Bindings

`KeyboardHandler` (`nomadz_deploy/utils/keyboard_handler.py`) maps:

| Key | Effect |
|---|---|
| `y` / `b` / `x` | Press BUTTON_Y / BUTTON_B / BUTTON_X (single-shot) |
| `w` / `s` | `vx += 0.25` / `vx -= 0.25` (clamped to ±1) |
| `a` / `d` | `vy += 0.25` / `vy -= 0.25` |
| `q` / `e` | `vyaw += 0.25` / `vyaw -= 0.25` |
| `Space` | Zero all three axes |
| `Ctrl-C` | SIGINT → portal cleanup → safe shutdown |

> ⚠️ **Single-shot press semantics for buttons.** A `y` keystroke fires
> the FSM's BUTTON_Y once. The terminal's repeat-key behaviour does not
> produce repeat presses — the consumer drains rising edges only.

### 6.0.3 Launch

In **terminal 1** (the one you'll type into):

```bash
cd ~/nomadz_deploy
python3 scripts/deploy.py --task t1_walk --keyboard
```

You should see, in order:

1. `Keyboard handler ready. Use w/s/a/d/q/e for axes (Space = zero); y/b/x for buttons.`
2. `KeyboardHandler enabled (cbreak stdin).`
3. `Low state subscription started`
4. `Recovery state machine started.`
5. `[recovery] IDLE — press Y on the operator joystick to ramp into prepare state and enter custom mode.`
   *(The hint says "joystick" — it's emitted from the FSM regardless of
   backend. In keyboard mode, just type `y`.)*

The Foxglove bridge / layout from §8 work identically in keyboard mode
because the diag publishers don't care which backend produced the axes.
You can run §6.0.3 + §8 together as your end-to-end smoke test.

### 6.0.4 Common keyboard-mode gotchas

* **`KeyboardHandler needs a real TTY on stdin`** — you're running the
  deploy through a pipe, `nohup`, or `tmux`-detach without an attached
  client. Run it in a regular interactive shell. `tmux attach` works,
  detached sessions do not.
* **No characters register.** Either the cbreak switch failed (rare), or
  another process has the controlling terminal. Check `tty` in the
  terminal — should print `/dev/pts/<n>`.
* **Terminal "broken" after Ctrl-C / crash.** The cleanup path restores
  `tcsetattr` on shutdown, but a hard crash can skip it. Type `reset`
  and press Enter to fix; Bash will figure out the rest.
* **Letters appearing in your shell prompt later on.** Means the cleanup
  didn't run — same fix (`reset`).
* **Robot driven into a wall by a stuck axis.** WASD is incremental; if
  you press `w` four times the robot keeps walking forward at full vx
  even after you stop typing. Press `Space` to zero, or `s` four times
  to back out, or `x` to emergency-damp.

### 6.0.5 When to graduate to joystick mode

Switch to `--joystick` once **all** of these are true:

* The full validation sequence (§10) passes in keyboard mode.
* You have a pygame-compatible pad plugged in (see §6 below for the
  pad-side checks).
* You want analog stick control instead of incremental WASD — useful
  for actual walking demos but not for FSM bring-up.

Joystick and keyboard cannot be active at the same time. If you pass
both `--joystick` and `--keyboard`, the joystick wins (the more
deliberate hardware choice).

---

## 6. Connect the operator joystick

Plug **our pygame controller** (the one calibrated for the motion board's
USB) into a free USB port on the motion board.

Verify:

```bash
ls /dev/input/js*           # should show /dev/input/js0 (or jsN)
```

(Optional but recommended — confirm pygame can see it:)

```bash
python3 -c "
import pygame; pygame.init(); pygame.joystick.init()
n = pygame.joystick.get_count(); print('pads:', n)
if n: 
    j = pygame.joystick.Joystick(0); j.init()
    print('name:', j.get_name(), 'axes:', j.get_numaxes(), 'buttons:', j.get_numbuttons())"
```

Expected: at least one pad, ≥ 4 axes, ≥ 4 buttons.

> ⚠️ **Do not plug in the manufacturer's joystick at the same time** —
> pygame opens index 0, and there is no guarantee whose pad that is. If
> both are plugged in, our deploy may calibrate against the manufacturer
> pad and our buttons will go to the firmware. Either physically unplug
> the manufacturer's pad, or set `SDL_JOYSTICK_DEVICE=/dev/input/js<n>`
> to the right one before launch (rare; usually unplug is simpler).

### 6.1 Button-index sanity check (do this once per pad)

`nomadz_deploy/utils/joystick_handler.py` uses `BUTTON_A=0, BUTTON_B=1,
BUTTON_X=2, BUTTON_Y=3` — the standard SDL Linux mapping for Xbox-style
pads. **It varies by driver and pad.** Run `jstest` to confirm:

```bash
sudo apt install -y joystick   # if missing
jstest /dev/input/js0
```

Press A, B, X, Y in turn and note the index numbers reported. If they
don't match `0/1/2/3`, edit the constants at the top of
`nomadz_deploy/utils/joystick_handler.py` before launching the deploy.
This is the only empirical step in the whole bring-up.

---

## 7. Launch the deploy script (joystick mode)

> If you haven't already done the **keyboard-mode smoke test in §6.0**,
> do that first. Joystick mode adds pygame / pad / button-mapping
> failure modes on top of everything in keyboard mode.

In **terminal 1** (the one we'll watch the FSM logs in):

```bash
cd ~/nomadz_deploy
python3 scripts/deploy.py --task t1_walk --joystick
```

You should see, in roughly this order:

1. `Initialized joystick: <pad name>`
2. `Calibrating joystick... Please keep sticks centered.` — **do not
   touch the sticks for 2 seconds.**
3. `Calibration complete.`
4. `JoystickHandler enabled and calibrated.`
5. `Low state subscription started`
6. `Recovery state machine started.`
7. `[recovery] IDLE — press Y on the operator joystick to ramp into
   prepare state and enter custom mode.`

If you stop at step 1 (no joystick name printed), §6's joystick
verification failed; revisit that.

If you stop at step 5 (low_state thread doesn't start logging), the ROS 2
sourcing in §5 didn't take in this shell — re-source `setup.bash`.

If you reach step 6 but no IDLE prompt appears, the joystick is connected
but `JoystickHandler.calibrate()` raised silently — check the log for
warnings. Operator pad pulls a strong centre offset on some drivers;
re-calibrate with sticks more centered and rerun.

> ⚠️ **`--task t1_walk` without `--joystick` AND without `--keyboard` is
> now a foot-gun.** We set `enable_safety_fallback=False` on the t1_walk
> cfg (see `tasks/locomotion/__init__.py`) so the FSM owns fall handling.
> **Without the FSM, there is no fall handling at all** — the policy
> keeps walking the dead robot. Always pass either `--joystick` or
> `--keyboard` for t1_walk on real hardware. Sim runs without an input
> backend are fine because the MuJoCo controller has its own
> `enable_fall_reset`.

### 7.1 What happens internally on launch

The portal:

1. Constructs `BoosterRobot` from the t1_walk cfg (joints / stiffness / damping).
2. Instantiates `RemoteControlService` (manufacturer's evdev pad). Coexists
   but is unused by the FSM.
3. Instantiates `JoystickHandler`, calibrates, starts its 100 Hz read thread.
4. Inits `B1LocoClient`, opens the `/joint_ctrl` publisher and `/low_state`
   subscriber, opens `/nomadz/*` diagnostic publishers.
5. Starts a 50 Hz diag thread that publishes `/nomadz/joystick_axes`.
6. Calls `B1LocoClient.Init()`.
7. Hands off to `RecoveryStateMachine.run()`.

---

## 8. Foxglove visualisation

In **terminal 2** (separate SSH to motion board, ROS 2 env sourced as in §5):

```bash
ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765
```

You should see `Foxglove WebSocket listening on 8765`. Leave it running.

On the workstation:

1. Open Foxglove Studio.
2. **Open connection** → **Foxglove WebSocket**.
3. URL: `ws://192.168.10.101:8765`.
4. Once connected, **Layout** → **Import from file…** and pick
   `nomadz_deploy/configs/foxglove_recovery.json` (copy it from the robot
   first if you haven't).

You should immediately see:

* **Recovery State** indicator (top-left): grey "IDLE" until you press Y.
* **Mode** indicator: blank/grey until our FSM issues its first ChangeMode.
* **Fall flag** indicator: green "OK" (proj_g_z is well below threshold).
* **proj_g_z plot** streaming at ~500 Hz, sitting around -1.0 with the
  -0.5 threshold visible as the upper edge of the y-axis.
* **IMU rpy plot** streaming, near zero on roll/pitch.
* **Joystick axes plot** streaming, near zero (sticks centred).

If panels stay grey, run `ros2 topic list | grep /nomadz` in terminal 2 —
all 8 `/nomadz/*` topics should be listed. If none are, your two
terminals aren't on the same DDS domain (re-source `setup.bash` in both).

If `proj_g_z` shows but `joystick_axes` doesn't, the diag thread didn't
start — that means `joystick_enabled=False` despite passing `--joystick`,
which means the JoystickHandler ctor failed (look back in terminal 1 for
a warning).

See `nomadz_deploy/configs/README.md` for a description of every panel.

---

## 9. Operating the FSM (the actual demo)

With the IDLE prompt up:

1. **Press Y.** Robot ramps from current pose to t1_walk's prepare pose
   over ~1 s, enters CUSTOM mode. Foxglove: state → PREP_READY, mode →
   "CUSTOM" (purple). The robot is now publishing zero-target LowCmd at
   the prepare-state stiffness/damping.
2. **Press Y again.** Inference subprocess forks, `t1_walk.pt` JIT-loads
   (~200 ms), policy starts publishing. Foxglove: state → RUNNING (green).
   Move the left stick — the robot should respond.
3. **Test X (emergency damp).** Press X. Inference pauses
   (`policy_run_event.clear()`), firmware enters DAMP, joints go limp.
   Foxglove: state → IDLE, mode → "DAMP" (grey). Robot sags onto its
   harness/feet.
4. **Re-arm with Y → Y.** Same as steps 1+2.
5. **Test B (full recovery cycle), no real fall.** Press B in RUNNING.
   FSM: terminate inference subprocess → ChangeMode(kDamping) →
   ChangeMode(kPrepare) → GetUp() → wait for upright → ChangeMode(kPrepare)
   → STAGED. Foxglove: state cycles RUNNING → FALLEN → RECOVERING →
   STAGED. The mode indicator walks through DAMP → PREP → WALK → PREP.
   GetUp RC indicator turns green ("GetUp OK (0)"). When STAGED prompts
   "Press Y to resume", press Y; fresh inference subprocess spawns.
6. **Real fall test.** With the hoist *taut enough to catch* but
   slack enough that the robot can pitch ~20° before the rope arrests
   it, push the robot mid-walk. The IMU detector should trip and
   `/nomadz/fall_flag` should turn red. **Auto-recovery is OFF** — the
   FSM stays in RUNNING and waits. Press B to commit. Recovery cycle
   runs as in step 5.
7. **Ctrl-C** in terminal 1 to clean up. The signal handler sets
   `exit_event`; the FSM exits its loop; `_safe_shutdown` walks
   DAMP → PREP → WALK; cleanup terminates the inference subprocess and
   destroys the diag node. Foxglove reflects the mode walk in real time.

### 9.1 What "RUNNING" feels like

* Joystick axes (left X / left Y / right X) drive (vy / vx / vyaw),
  inverted to match the robot's body frame. Pushing left-stick-up walks
  forward; pushing right-stick-right yaws clockwise (looking down).
* Per-joint PD gains are the t1_walk cfg's `joint_stiffness` /
  `joint_damping` — same as training. If the robot looks "twitchy"
  compared to sim, the PD ratio is mismatched (rare; check
  `tasks/locomotion/locomotion.py:T1WalkControllerCfg`).

### 9.2 What "RECOVERING" looks like

* The robot rolls onto its back if face-down (or vice versa), pushes off
  with arms, swings legs underneath, stands. ~5–8 s end-to-end. The hoist
  rope should not take weight during this; if it does, the GetUp may
  abort.
* `RECOVERING` shows yellow in Foxglove for the duration. After
  `/nomadz/getup_return_code` shows 0 and `proj_g_z` re-anchors near
  -1.0, the FSM exits to STAGED.

---

## 10. Hoisted-bench validation order (do this in sequence the first time)

This is the §11 of `RECOVERY_INTEGRATION_PLAN.md`, repeated here as a
runbook checklist. **Do not skip ahead.** The first half (1k–6k) is in
**keyboard mode** so a pad failure doesn't block the FSM bring-up; the
second half (1j onward) re-runs the exhausting subset in joystick mode
to validate the pad-specific path.

### Phase A — keyboard mode (no pad needed)

* [ ] **(1k)** Manufacturer-controller A press (legacy run, no
  `--keyboard` and no `--joystick`) still arms custom mode. Confirms
  our changes didn't break the legacy path. Skip this if you don't have
  the manufacturer pad nearby.
* [ ] **(2k)** Launch `--keyboard`. KeyboardHandler initialises, FSM
  reaches IDLE. `/nomadz/recovery_state` shows "IDLE" in Foxglove.
* [ ] **(3k)** Type `y`, `y`. Robot ramps to prepare, enters CUSTOM,
  inference subprocess starts, robot walks. Foxglove cycles
  IDLE → PREP_READY → RUNNING.
* [ ] **(4k)** Type `x`. Inference pauses, firmware DAMPs, joints go
  limp. State → IDLE.
* [ ] **(5k)** Type `y`, `y`. Robot back in RUNNING. Type `b`. Full
  recovery cycle runs on the hoist. Foxglove walks
  RUNNING → FALLEN → RECOVERING → STAGED. GetUp RC = 0.
* [ ] **(6k)** Type `y`. Inference subprocess respawns; robot walks
  again. Type `Ctrl-C`; cleanup runs cleanly; terminal restored.

### Phase B — joystick mode

* [ ] **(1j)** Plug in the pygame pad. Verify `jstest /dev/input/js0`
  reports BUTTON_A/B/X/Y at indices 0/1/2/3 (or update the constants in
  `joystick_handler.py`).
* [ ] **(2j)** Launch `--joystick`. Same A/B drill as 2k–4k.
* [ ] **(3j)** Press B from RUNNING — full recovery cycle as 5k.
* [ ] **(4j)** Press Y from STAGED — same as 6k but via pad.
* [ ] **(5j)** Tilt the hoisted robot to trip the IMU detector.
  `/nomadz/fall_flag` turns red. **State machine stays in RUNNING** —
  auto-recovery is off (per `revision.txt` decision §6). Operator
  presses B → recovery cycle.
* [ ] **(6j)** Free-fall test (low slack on the hoist). Operator
  intervenes with B-press once the robot is on the mat. Recovery cycle
  succeeds.
* [ ] **(7j)** Foxglove dashboard shows everything live during steps 5j
  and 6j — the recording is the demo-grade evidence.
* [ ] **(8j)** Only after Phase A and 1j–7j pass: drop the hoist. Even
  then keep an operator on X (or on `x` in keyboard mode).

---

## 11. Shutdown

Normal shutdown:

1. **Ctrl-C** in terminal 1. Wait for the cleanup log:
   `Cleanup complete` followed by metric dumps. **Do not yank power
   while cleanup is running** — the safe-shutdown sequence is mid-flight
   and a hard reset can leave joints in an inconsistent state.
2. The robot is now in **kWalking** (the safe-shutdown target). If you
   intend to power off, drive it into PREP via the manufacturer's
   joystick first, then DAMP, then power off.
3. Stop the Foxglove bridge in terminal 2 (Ctrl-C).
4. Power-off button on the robot.

Emergency shutdown (anything looks wrong):

1. Hardware E-stop on the back of the robot.
2. Optionally also Ctrl-C the deploy script.
3. Diagnose before re-powering. Check `dmesg` on the motion board for
   any DDS / network errors; check `ros2 topic echo /low_state --once`
   for sensor sanity.

---

## 12. Troubleshooting catalogue

The FSM is small enough that most issues fall into one of these buckets.

### `ImportError: booster_robotics_sdk_python is not installed`

**Cause:** SDK was built without `-DBUILD_PYTHON_BINDING=ON`, or wasn't
`make install`'d. **Fix:** §2.1 with the flag.

### Calibration prints nothing / hangs

**Cause:** Pygame opened the wrong device (e.g., the manufacturer's pad).
**Fix:** unplug the manufacturer's pad. If only ours is plugged in,
`evtest /dev/input/event*` to confirm the kernel sees button events.

### Y press in IDLE does nothing

Possibilities:

* Wrong button index — see §6.1. Edit `BUTTON_Y` in
  `nomadz_deploy/utils/joystick_handler.py`.
* The pad's edge-trigger isn't being seen because the press is too short
  for our 100 Hz poll. **Hold the press for ~50 ms.**
* The `_button_lock` is in contention with the diag thread. Very
  unlikely; would show as missed presses across the board, not just Y.

### `Waiting for '/joint_ctrl' subscriber, retry in 0.5s` loops forever

**Cause:** The firmware isn't subscribing to our `joint_ctrl`. Either the
firmware is in a mode that doesn't accept LowCmd (e.g. WALK; CUSTOM is
required), or DDS isn't bridging. **Fix:** drive the robot to PREP via
the manufacturer's joystick before pressing Y, and double-check ROS 2
sourcing per §5.

### `ChangeMode(kCustom) failed`

The firmware rejected the transition. Per the wiki, only DAMP / PREP can
transition to CUSTOM. **Fix:** drive to PREP first (manufacturer's
joystick), then re-press Y in IDLE.

### `GetUp() returned non-zero`

Most common causes:

* `is_recovery_available` was false at the moment we called GetUp. We
  don't read that flag in the first cut (it's not Python-bound — see
  `RECOVERY_MOTION_REPORT.md` §5). Wait 2 seconds in PREP after a fall
  before pressing B.
* A joint is wedged — debris, tangle in the hoist line, robot caught on
  its own arm. **Inspect physically before retrying.**
* Battery sag during the recovery's high-current limbs swing. Re-charge.

The FSM enters EXITING on a non-zero rc to avoid a runaway. Restart the
deploy and try again.

### Foxglove panels stay grey

* Run `ros2 topic list | grep /nomadz` in terminal 2 — if empty, the two
  terminals aren't on the same DDS domain. Re-source `setup.bash`.
* Check `ros2 topic hz /nomadz/proj_g_z` — should be ~500 Hz. If 0 Hz,
  the deploy isn't running or the diag publishers weren't init'd; the
  deploy log will show why.
* If only some panels are dead (e.g. joystick_axes silent but proj_g_z
  alive), it's the diag thread specifically. See the deploy log for
  `JoystickHandler` warnings.

### Robot enters PROTECT mode unexpectedly

PROTECT auto-engages on a fall or a joint-limit error. From
`RECOVERY_MOTION_REPORT.md` §5.1: *"You can try to reenter DAMP mode
under PROTECT mode (Soft restart)."* So PROTECT → DAMP → PREP → CUSTOM.
The FSM's `_tick_fallen` does exactly this sequence, so pressing B
should recover. If it doesn't (PROTECT → DAMP rejected), the firmware
may need a hard restart — power-cycle.

### `RuntimeError: No joystick detected.`

`pygame.joystick.get_count() == 0`. Either the pad isn't plugged in, or
the kernel doesn't expose it as a `joystick` device (we need a
`/dev/input/js*`). On some 8BitDo / generic HID pads you have to enable
"Direct Input" mode (D-pad button mode setting on the pad itself).

### `Falling detected, stopping policy for safety`

This message comes from `LocomotionPolicy.compute_observation`'s safety
fallback. **You should never see this** for `t1_walk` because we
disabled it in `tasks/locomotion/__init__.py`. If you do, your
`tasks/locomotion/__init__.py` reverted; re-apply
`self.policy.enable_safety_fallback = False` in `T1WalkControllerCfg1.__post_init__`.

### Inference subprocess dies unexpectedly during RUNNING

In the legacy path the parent log shows `Inference process died
unexpectedly`; under the FSM, the next tick's RUNNING handler will
eventually see the subprocess gone — but we don't currently *probe* for
that in `_tick_running`. Symptom: button presses still register but no
joint motion, sticks have no effect. **Fix for now:** Ctrl-C and restart;
investigate the subprocess log. (Adding an "is_alive() check" to
`_tick_running` is a small follow-up improvement.)

### Operator presses B but recovery doesn't start

* Confirm Foxglove's `/nomadz/button_events` log shows "B" — if not,
  pygame missed the press (see "Y press in IDLE does nothing"). In
  keyboard mode the same log fires for `b` keystrokes.
* Confirm state was RUNNING when you pressed (button consumption only
  fires the recovery if the state matches).

### Keyboard mode: characters echo into the deploy log

cbreak failed to suppress echo, or another tool (tmux, screen) is
intercepting. Try in a plain `bash` outside tmux. If it still echoes,
your `tput`/`stty` may have been misconfigured by a shell rc; run
`reset` and retry.

### Keyboard mode: terminal is broken after the deploy exits

Cleanup didn't restore termios. Run `reset` and press Enter — the bash
prompt will return to normal. If it happens reliably, file a bug — the
expected path is `KeyboardHandler.stop()` calling `tcsetattr` in the
finally branch.

### Keyboard mode: WASD axes never reach -1 or +1

Each press adds `axis_step = 0.25`, so four presses saturate. If the
robot stops moving before saturating, the policy may be receiving
clamped commands from a per-task `vx_max` (see
`tasks/locomotion/locomotion.py:T1WalkControllerCfg.vel_command`).
Override with `--vx-max <m/s>` etc. on the deploy command line.

### Keyboard mode and joystick mode both selected — only one wins

By design: passing both `--joystick` and `--keyboard` resolves to
joystick (the more deliberate choice). To force keyboard, drop
`--joystick`. The portal log line at startup tells you which backend
took ("JoystickHandler enabled and calibrated." vs.
"KeyboardHandler enabled (cbreak stdin).").

---

## 13. Known limitations of this prototype

These are documented for the next iteration:

1. **No `is_recovery_available` flag** read from the firmware (Python
   binding gap). We rely on a 200 ms sleep after PREP before calling
   GetUp. Brittle if the firmware needs longer.
2. **No auto-recovery.** Fall detector logs only; operator must B-press.
   Per design choice, not a bug.
3. **`GetUpWithMode` not bound** in Python. We use bare `GetUp()` which
   lands in WALK by default. Fine for our PREP→CUSTOM hop afterwards.
4. **MuJoCo recovery mock not implemented.** B press in MuJoCo does
   nothing useful for recovery rehearsal.
5. **Subprocess liveness not probed in RUNNING.** A dead inference
   subprocess goes silent until Ctrl-C.
6. **Pygame button index assumption** — see §6.1; one-time check.
7. **Manufacturer's evdev controller still active** as a parallel input
   path. If an operator inadvertently presses something on it, the
   firmware will react (e.g. LT+UP triggers GetUp). Either physically
   stash it or accept the redundancy.
8. **`enable_safety_fallback=False` is now wired into the t1_walk cfg
   unconditionally**, so legacy runs (no `--joystick` AND no
   `--keyboard`) lose fall protection. Always pass one of the two for
   `t1_walk`. (Future fix: `enable_safety_fallback` becomes
   `report_to_parent_fall_event` — see `RECOVERY_INTEGRATION_PLAN.md`
   §10.7 option 2.)
9. **Keyboard mode axes are incremental, not analog.** Useful for
   bring-up; not great for fine-grained walking demos. The pad is the
   intended driver for §10 phase B onward.

Each of these has a tracking note in `RECOVERY_INTEGRATION_PLAN.md` §13.

---

## 14. Cross-references

| Source | Relevance |
|---|---|
| `docs/RECOVERY_INTEGRATION_PLAN.md` | Architecture, decisions log, file map |
| `/home/nomadz-control/booster_robotics_sdk/docs/RECOVERY_MOTION_REPORT.md` | SDK-side recovery API, mode-transition rules, fall-state telemetry |
| `~/nomadz-neo.wiki/T1-Robot-Configuration-Guide.md` | SSH, network, Wi-Fi, internet sharing setup |
| `~/nomadz-neo.wiki/T1-Development-Log.md` | SDK install on each board, factory-reset before-return checklist |
| `nomadz_deploy/configs/foxglove_recovery.json` | The Foxglove layout this runbook drives |
| `nomadz_deploy/configs/README.md` | Per-panel description |
| `nomadz_deploy/controllers/recovery_state_machine.py` | The FSM source, useful when something looks weird in the indicator |
| `nomadz_deploy/controllers/booster_robot_controller.py` | Portal helpers, fall detector, diag publishers |
| `nomadz_deploy/utils/joystick_handler.py` | Pygame backend for `--joystick`; BUTTON_* index constants |
| `nomadz_deploy/utils/keyboard_handler.py` | cbreak-stdin backend for `--keyboard`; key bindings |
| `tasks/locomotion/__init__.py` | Where `enable_safety_fallback=False` is set for t1_walk |
| `nomadz_deploy/README.md` | Original deploy framework docs |

---

*End of runbook v1. If anything in this document was contradicted by
real hardware behaviour during your validation run, file a finding in
`docs/` so the next operator doesn't hit the same wall.*
