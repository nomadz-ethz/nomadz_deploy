# Wiring `GetUp()` Recovery into the `t1_walk` Deployment

> Companion to `/home/nomadz-control/booster_robotics_sdk/docs/RECOVERY_MOTION_REPORT.md`.
> That report tells us *what* the SDK exposes; this one tells us *how* to plumb it into
> the existing `nomadz_deploy` real-robot stack so that running `t1_walk.pt` can
> survive a fall and resume on its own — driven by **our own joystick** (pygame /
> `JoystickHandler`), with **Foxglove** visualisation on top of the existing ROS 2 +
> Fast DDS messaging layer.
>
> **This revision (v2)** addresses comments in `docs/revision.txt`: clarifies the
> messaging stack, corrects the joystick story (it's our pygame controller we extend,
> not the manufacturer's evdev one), promotes SDK Python-binding work from "follow-up"
> to a documented fork-edit task, brings in lessons from the sister `nomadz-neo` C++
> stack, separates the state machine into its own file, and adds a concrete Foxglove
> visualisation section.

## TL;DR

* **Task entry point already exists.** `python scripts/deploy.py --task t1_walk` (real robot) or `--task t1_walk --mujoco` (sim) loads `tasks/locomotion/models/t1_walk.pt` via the registered `T1WalkControllerCfg1`.
* **Messaging stack: ROS 2 (`rclpy`) on top of Fast DDS, sharing a domain with the SDK's raw-DDS channels.** Our Python code subscribes to `/low_state` and publishes to `joint_ctrl`; the SDK's C++ side speaks to `rt/low_state` and `rt/joint_ctrl` on the same Fast DDS domain — ROS 2's DDS middleware handles the namespace translation. **Foxglove can plug directly into the ROS 2 side** with no extra bridge process.
* **Real-robot orchestrator is `BoosterRobotPortal`** in `nomadz_deploy/controllers/booster_robot_controller.py`. It owns `B1LocoClient`, the `/low_state` subscription thread, the `joint_ctrl` publisher, and the inference subprocess. Recovery is a state-machine extension that *consumes* the portal's primitives — but lives in **a new file** `nomadz_deploy/controllers/recovery_state_machine.py`.
* **Joystick correction.** `JoystickHandler` (pygame) is **our own external controller plugged into our laptop** — currently used in MuJoCo for axes. We extend it with edge-triggered button reads (BTN_Y / BTN_B / BTN_X) and wire it into the real-robot path too. `RemoteControlService` (evdev) is a parallel/legacy code path that talks to a different controller; the recovery flow does **not** depend on it.
* **SDK Python-binding gaps are now fork-fix items, not workarounds.** `booster_robotics_sdk` is our fork — we add `FallDownStateSubscriber` and `GetUpWithMode` bindings cleanly, with a CHANGELOG entry. For the *first cut* we still derive falls from the IMU stream (per user direction in `revision.txt`), but the binding work is on the same plan, not deferred forever.
* **Foxglove layout.** We publish a small set of diagnostic ROS 2 topics from the portal (`/nomadz/recovery_state`, `/nomadz/proj_g_z`, `/nomadz/mode`, `/nomadz/joystick_axes`, `/nomadz/button_events`). A `configs/foxglove_recovery.json` layout pins them next to `/low_state` for live debugging.
* **Button mapping (locked in v2.1):** **Y = enter / re-enter locomotion (CUSTOM + RL policy)**, **B = trigger recovery**, **X = emergency damp**. **A is *not* used by the state machine** — the manufacturer's controller's A binding stays out-of-band, and on our pygame controller A is left free. **No auto fall-recovery** for now: the IMU detector trips `_fall_event` and *logs*; the operator must press B to commit to the recovery sequence.

---

## 1. Messaging stack — what the wires are actually carrying

This was unclear in v1. Concrete picture:

| Layer | What it speaks | Topic naming | Where it lives |
|---|---|---|---|
| **Robot firmware** | Fast DDS | `rt/low_state`, `rt/joint_ctrl`, `rt/fall_down`, `rt/odometer_state` | On the robot. |
| **`booster_robotics_sdk` (C++)** | Fast DDS via `booster::common::DdsTopicChannel<MSG>` (`channel_factory.hpp:7`), Eprosima Fast DDS profiles in `nomadz-neo/configs/fastdds.xml` | `rt/...` | The SDK statically links Fast DDS. `B1LocoClient` is RPC-style on top of DDS. |
| **`booster_interface` ROS 2 messages** | `ament_cmake` ROS 2 message package (defined in `nomadz-neo/src/interface/booster_ros2_interface/`, `package.xml`, `*.msg`) | — | Provides `LowState`, `LowCmd`, `MotorCmd`, `MotorState`, `ImuState`, `RemoteControllerState`, etc. |
| **`nomadz_deploy` Python control** | `rclpy` (`booster_robot_controller.py:12-15`) using `booster_interface.msg.*` | `/low_state` (sub), `joint_ctrl` (pub) | Talks to Fast DDS through `rmw_fastrtps_cpp`. |

**Why `/low_state` ↔ `rt/low_state` "just works":** ROS 2 with `rmw_fastrtps_cpp` rewrites topic names on the wire — ROS 2's `/low_state` becomes the DDS topic `rt/low_state` (the `rt/` prefix is ROS 2's namespacing convention for "real topics"), so a node using `rclpy` and a node using raw Fast DDS see the same topic *as long as they share the same DDS domain*. Both `nomadz_deploy` and the SDK default to domain 0 (`scripts/deploy.py` passes `args.net` → `ChannelFactory.Init(0, args.net)`).

**QoS asymmetry already in code** (`booster_robot_controller.py:180-184, 272-276`):
* `/low_state` subscription: BEST_EFFORT, depth=1, KEEP_LAST. Sensor stream — losses are fine, latency matters.
* `joint_ctrl` publisher: RELIABLE, depth=1, KEEP_LAST. Command stream — every packet must land.

**Practical consequences for this plan:**
1. **Foxglove via the Foxglove WebSocket bridge** can subscribe to *both* the existing ROS 2 topics (`/low_state`, `joint_ctrl`) and any *new* diagnostic topics we publish from the portal — no DDS-side work required.
2. **Adding a `FallDownState` subscriber** is symmetric: either bind the SDK's C++ DDS reader to Python (clean, fits the existing pattern), or write an `rclpy` node that subscribes to `rt/fall_down` directly using a custom message type. The SDK route is cleaner because it shares the SDK's pre-rolled Fast DDS profile and message IDL.

---

## 2. Where the existing pieces fit

| Concern | File | Role today |
|---|---|---|
| CLI dispatch / task lookup | `scripts/deploy.py` | Walks `tasks/`, registers, then forks into `MujocoController` or `BoosterRobotPortal`. |
| Real-robot owner process | `nomadz_deploy/controllers/booster_robot_controller.py` (`BoosterRobotPortal`) | Owns `B1LocoClient`, `/low_state` sub, `joint_ctrl` pub, signal handler, inference subprocess. |
| Real-robot inference process | same file (`BoosterRobotController`, forked via `mp.Process`) | Reads `synced_state`, runs `policy.inference()`, writes `motor_cmd[i]` and publishes. |
| MuJoCo path (sim2sim) | `nomadz_deploy/controllers/mujoco_controller.py` | Reference for what `t1_walk.pt` expects; key callback / ghost / push / fall-reset hooks suggest the right shape for a sim recovery mock. |
| Locomotion policy | `tasks/locomotion/locomotion.py` (`LocomotionPolicy`) | Computes obs, runs jit, returns `dof_targets`. **Has its own fall-detect that calls `controller.stop()`** — see §6.4. |
| Task registry | `nomadz_deploy/utils/registry.py` + `tasks/locomotion/__init__.py` | `register_task("t1_walk", T1WalkControllerCfg1())` already done. |
| **Our** joystick (pygame, axes today, MuJoCo-only today) | `nomadz_deploy/utils/joystick_handler.py` (`JoystickHandler`) | Reads pygame axes 0/1/3 via `joystick.get_axis()`; **no button code today**. The class we extend. |
| Manufacturer's evdev controller | `nomadz_deploy/utils/remote_control_service.py` (`RemoteControlService`) | evdev. BTN_A → custom mode, BTN_B → RL gait. Currently the *real-robot* startup gate. Parallel path; not on the recovery critical path. |
| SDK `GetUp` / `LieDown` / `ChangeMode` | `booster_robotics_sdk/python/binding.cpp:586,594,420` + `b1_loco_client.hpp:209,200,57` | Python has the verbs. |
| SDK fall telemetry | `FallDownState.h` + `b1_api_const.hpp:12` (`rt/fall_down`) | C++ only today. **Plan: add Python binding in the fork (§9).** |
| SDK `RobotMode` enum (Python) | `binding.cpp:250-256` | Exposes `kUnknown / kDamping / kPrepare / kWalking / kCustom`. **Adding `kSoccer` is part of the binding work.** |

---

## 3. What's already wired vs. what we add

### Already there

* High-level loop: portal `__init__` → `start_custom_mode_conditionally` → `start_rl_gait_conditionally` → spawn inference → `run()` watches it.
* `/low_state` subscription writes RPY/gyro/joint state into `synced_state`. **The IMU stream is already in the parent process** — fall detection plugs in next to the existing `_low_state_handler`.
* `B1LocoClient` is `Init`'d in `_init_communication`; we can call `GetUp/LieDown/ChangeMode` from anywhere in the parent.
* `JoystickHandler` already calibrates and runs a 100 Hz axis-poll thread (`joystick_handler.py`), wired into `MujocoController` only.
* `BoosterRobotPortal.run()` already calls `client.ChangeMode(RobotMode.kWalking)` on shutdown — which is currently illegal from `CUSTOM` (see §7) and needs to route via `PREP`.

### Missing

1. **Pygame button reads on `JoystickHandler`.** Currently it calls `pygame.event.pump()` and `joystick.get_axis()`; it never calls `joystick.get_button(i)`. ~20-line extension (sketch in §6.1).
2. **Edge-triggered button events.** `get_button()` returns the current 0/1; we need a "rising-edge consume" wrapper so a long press doesn't fire `GetUp()` 50 times.
3. **`JoystickHandler` instantiation in `BoosterRobotPortal`.** Today it's MuJoCo-only — we need to import and start it in the real-robot portal too, alongside (or instead of) `RemoteControlService` for velocity commands.
4. **Fall detection from IMU.** Project gravity from `state["root_rpy_w"]`, trip when `proj_g[2] > -0.5` for N consecutive samples.
5. **`RecoveryStateMachine` class** in a new file `nomadz_deploy/controllers/recovery_state_machine.py`. Owns the state, calls `B1LocoClient`, asks the portal to spawn/stop the inference subprocess.
6. **`policy_run_event`** (an `mp.Event`) inside `BoosterRobotController` for **pause/resume of the inference loop** (X-press emergency damp, and any pause that doesn't need a fresh policy state). Used as the primary pause mechanism per the v2.1 decisions. Separately, the **fall-to-recovery transition does a full terminate-and-respawn** of the inference subprocess (cheap given `t1_walk.pt` JIT load is small) — that gives a fresh `obs_history` / `last_action` so the resumed policy isn't reasoning over pre-fall observations.
7. **Disabling `LocomotionPolicy.enable_safety_fallback`** for `t1_walk` (or rerouting it to signal the parent's fall_event instead of calling `controller.stop()`).
8. **Cleanup-path fix.** Route shutdown via DAMP→PREP→WALK instead of the single illegal hop.
9. **Diagnostic ROS 2 publishers + Foxglove layout** (§8).
10. **(Fork edit) SDK Python bindings** for `FallDownStateSubscriber`, `GetUpWithMode`, and the `kSoccer` enum value (§9). Optional for the first cut, scheduled after the IMU-based prototype validates.

---

## 4. Two controllers — clarification

Per `revision.txt`, this repo has *two* physical controllers in play; v1 of this doc conflated them.

* **Manufacturer-provided controller** (referenced by `RemoteControlService`, evdev, `BTN_A` / `BTN_B`). Ships with the robot. Talks to the operator's computer via Linux's input subsystem (`/dev/input/event*`). Today wired to "press A = enter custom mode, press B = start RL gait" in the portal startup sequence. Parallel to our state machine — the operator can still use it for whatever the manufacturer's firmware expects, but our state machine does **not** read its buttons.
* **Our external controller** (referenced by `JoystickHandler`, pygame, axes 0/1/3). Plugged into our laptop. Already programmed for axes (left stick → vx/vy, right stick X → vyaw). **This is the one we extend with Y/B/X buttons for the recovery state machine.**

In code, this means:

* `JoystickHandler` gets the button extensions (§6.1).
* `BoosterRobotPortal` instantiates `JoystickHandler` in `__init__` *in addition to* `RemoteControlService` (don't break the existing startup path) — but the new state machine reads only from `JoystickHandler`.
* The `--joystick` CLI flag in `scripts/deploy.py` (currently MuJoCo-only) gets plumbed into `BoosterRobotPortal` too. When set, the portal uses `JoystickHandler` for velocity commands; when unset, it falls back to `RemoteControlService` (current behaviour).

---

## 5. Prior art in `nomadz-neo`

`~/nomadz-neo` is our sister C++ ROS 2 stack for the RoboCup Booster soccer demo. It is *not* the same as `nomadz_deploy` (different language, different lifecycle, different policies), but it has already solved the recovery-state-machine problem on the same robot family, so its choices are useful evidence.

What it does (last commit 2026-03-31; not stale):

* **Subscribes to `fall_down_recovery_state`** (a ROS 2 topic) — `nomadz-neo/src/brain/src/brain.cpp:194`. The callback at `:1389-1415` unpacks a `RawBytesMsg` into `RobotRecoveryStateData { state, is_recovery_available, current_planner_index }`. The state enum has the same four values as `FallDownStateType` in the SDK header (`IS_READY / IS_FALLING / HAS_FALLEN / IS_GETTING_UP`).
* **`CheckAndStandUp` behavior-tree node** (`brain_tree.cpp:1441-1478`) orchestrates the recovery: detects `HAS_FALLEN`, calls `client->standUp()` (which routes to `LocoApiId::kGetUp`), retries up to a config limit, polls `currentRobotModeIndex` (1 = damping, 8/20 = "robot is back to normal walking", 10 = post-standup) to confirm success.
* **`/remote_controller_state` ROS 2 topic** (`brain.cpp:42`, `RemoteControllerState.msg`) carries the *manufacturer* controller's state as a ROS 2 message — `lx/ly/rx/ry`, `A/B/X/Y`, `LB/RB/LT/RT`, hat. So the manufacturer's controller is *also* readable from ROS 2, not just evdev. This is an option if we ever want to avoid evdev entirely; for now it's just useful evidence that the manufacturer's controller buttons are observable to ROS code.
* **Foxglove layout** at `nomadz-neo/booster_soccer.layout` with custom Python layers (`draw_field`, `draw_robot`, `draw_ball`) consuming `/booster_soccer/*` topics. Our recovery layout (§8) follows the same pattern, just with different topics.

What we lift from it:

1. **Retry-with-mode-poll pattern.** `CheckAndStandUp` retries `GetUp()` up to N times, gating on `current_planner_index` reaching a known-good value. We do the same but gate on IMU-derived "upright" since we don't have the planner index in Python yet.
2. **Topic naming.** `/booster_soccer/...` namespace prefix for diagnostics; we'll use `/nomadz/...` (already in the §8 layout).
3. **`RemoteControllerState` ROS 2 message** is the *clean* path to read the manufacturer's controller from Python. Useful follow-up if `RemoteControlService` (evdev) turns out flaky — we just subscribe to the message instead. **Not on the critical path** for the first cut because we're driving everything from `JoystickHandler`.

What we *don't* port from it:

* The behavior-tree machinery (`BehaviorTree.CPP`) is overkill for a 6-state machine; a Python `enum.Enum` + a dispatch loop is enough.
* The C++ `RawBytesMsg` unpacking is a workaround for the SDK not exposing the typed message — we'll fix that at the SDK level by adding the Python binding (§9).

---

## 6. The state machine

Mode-transition rules from the K1 V1.6 wiki, restated (this is the load-bearing fact for the whole plan):

```
DAMP    → PREP
PREP    → DAMP | WALK | CUSTOM | (SOCCER)
WALK    → DAMP | PREP | CUSTOM | (SOCCER)   # in practice route via PREP for CUSTOM
CUSTOM  → PREP | DAMP                         # NOT directly to WALK
PROTECT → DAMP                                # auto-engaged on fall; behaves like DAMP
```

So the recovery cycle is **always** five `ChangeMode` calls plus one `GetUp`:

```
RUNNING (CUSTOM)                # RL policy publishing on joint_ctrl
   │  fall detected (logs only, waits for B) — or B pressed directly
   ▼
FALLEN
   │  TERMINATE inference subprocess (cheap; gives fresh policy state on resume)
   │  ChangeMode(kDamping)      # PROTECT → DAMP soft-restart (or CUSTOM → DAMP)
   │  ChangeMode(kPrepare)      # DAMP → PREP
   ▼
RECOVERING
   │  GetUp()                   # firmware lands in WALK (Python lacks GetUpWithMode in v1)
   │  poll: proj_g[2] < -0.95 for 5 stable frames (~250 ms hold)
   │  ChangeMode(kPrepare)      # WALK → PREP (CUSTOM is unreachable directly from WALK)
   ▼
STAGED (PREP)
   │  wait for Y press
   │  ChangeMode(kCustom)       # PREP → CUSTOM
   │  RESPAWN inference subprocess (fresh JIT load, fresh obs_history)
   ▼
RUNNING (CUSTOM)

# X press in RUNNING is a separate, lighter path:
#   policy_run_event.clear()  (no terminate)  →  ChangeMode(kDamping)  →  IDLE.
# This is the "I just want to stop the policy without going through full recovery" exit.
```

A B-press in `RUNNING` (operator-triggered for testing, no real fall) follows exactly the same path — CUSTOM → DAMP is allowed, so the soft-restart works whether or not the firmware has auto-flipped to PROTECT.

### Edge-triggered buttons (in `JoystickHandler`)

`pygame` exposes both polling (`joystick.get_button(i)` → 0/1) and events (`pygame.event.get()` filtered to `JOYBUTTONDOWN/UP`). The current handler uses neither — only `pump()` to keep the queue clean and `get_axis()` for sticks. We extend with polling because it composes cleanly with the existing 100 Hz `_read_loop`:

```python
# Sketch — final lives in nomadz_deploy/utils/joystick_handler.py.
BUTTON_A, BUTTON_B, BUTTON_X, BUTTON_Y = 0, 1, 2, 3   # standard Xbox-style mapping

class JoystickHandler:
    def __init__(self, deadzone=0.1, num_buttons=10):
        ...
        self._button_prev = [False] * num_buttons
        self._button_pressed_edges: deque[int] = deque()
        self._button_lock = threading.Lock()

    def update_values(self):
        ...                                    # existing axis read
        with self._button_lock:
            for i in range(len(self._button_prev)):
                cur = bool(self.joystick.get_button(i))
                if cur and not self._button_prev[i]:
                    self._button_pressed_edges.append(i)
                self._button_prev[i] = cur

    def consume_press(self, button_id: int) -> bool:
        with self._button_lock:
            try:
                self._button_pressed_edges.remove(button_id)
                return True
            except ValueError:
                return False
```

The state machine calls `joystick.consume_press(BUTTON_Y)` etc. — each press fires once, no debounce hacks needed.

### Fall detection (in the parent's `_low_state_handler`)

The portal already extracts `rpy` (`booster_robot_controller.py:231`). Add directly after the existing `synced_state.write` call:

```python
# Compute projected gravity[2] in body frame. Equivalent to
# LocomotionPolicy.compute_observation but in numpy, in the parent.
gz = -np.cos(rpy[0]) * np.cos(rpy[1])     # exact: q^-1 * (0,0,-1) projected onto z_body
if gz > self.cfg.booster.fall_proj_g_z_threshold:    # default -0.5
    self._fall_streak += 1
    if self._fall_streak >= self.cfg.booster.fall_streak_threshold:    # default 5
        self._fall_event.set()
else:
    self._fall_streak = 0
```

`_fall_event` is a `threading.Event` consumed by the state machine's `_wait_for_any` helper.

> The exact closed-form is `gz = -cos(roll)*cos(pitch)` (apply quaternion-from-rpy to `(0,0,-1)`). The `LocomotionPolicy.compute_observation` does the same thing via `lab_math.quat_apply_inverse`; we keep them in sync by sharing the threshold constant.

---

## 7. Architecture (revised)

```
parent (BoosterRobotPortal)
├── B1LocoClient                              # mode changes, GetUp, LieDown
├── /low_state subscription thread            # streams IMU + joints into synced_state
│      └── _low_state_handler (extended)      # +fall detector → self._fall_event
├── RemoteControlService                      # evdev (manufacturer); kept for back-compat startup
├── *** new *** JoystickHandler              # pygame (our controller) — axes + edge buttons
├── *** new *** RecoveryStateMachine         # in recovery_state_machine.py; orchestrates
│      ↳ reads: joystick.consume_press(...), self._fall_event
│      ↳ writes: client.ChangeMode/GetUp, policy_run_event, diagnostic publishers
├── *** new *** ROS 2 diagnostic publishers   # /nomadz/recovery_state etc. (see §8)
├── joint_ctrl publisher (handle)             # used by inference subprocess
└── inference subprocess (BoosterRobotController)
       ├── policy_run_event (mp.Event)        # *** new *** parent gates publishing here
       └── jit-loaded t1_walk.pt
```

Why parent-side state machine:
* `B1LocoClient`, `RemoteControlService`, `JoystickHandler` already (or will) live in the parent. Cross-process synchronisation for mode changes is unnecessary.
* The IMU is already in `synced_state` on the parent.
* The inference subprocess can be paused (`policy_run_event.clear()`) without termination — the existing inner loop already polls `self.portal.exit_event` (`booster_robot_controller.py:543`); we add a second condition.

---

## 8. Visualisation — Foxglove + ROS 2 diagnostic topics

The user wants live visibility into the state machine, IMU, and mode transitions. We have ROS 2 already (§1), so this is "publish a few `std_msgs` topics from the portal and write a Foxglove layout".

### 8.1 Diagnostic topics to publish from the portal

All under the `/nomadz/...` namespace so they don't collide with `/low_state` and friends.

| Topic | Type | Rate | Where set |
|---|---|---|---|
| `/nomadz/recovery_state` | `std_msgs/String` | on transition | `RecoveryStateMachine` on every state change |
| `/nomadz/mode` | `std_msgs/String` | 5 Hz | `RecoveryStateMachine` polls `client.GetMode()` (already bound) |
| `/nomadz/proj_g_z` | `std_msgs/Float32` | 500 Hz (low_state rate) | `_low_state_handler` |
| `/nomadz/fall_flag` | `std_msgs/Bool` | on rising edge | `_low_state_handler` when `_fall_event` set |
| `/nomadz/joystick_axes` | `geometry_msgs/Vector3` (x=vx, y=vy, z=vyaw) | 100 Hz (joystick rate) | `JoystickHandler` poll loop |
| `/nomadz/button_events` | `std_msgs/String` (button name) | on press edge | `JoystickHandler` |
| `/nomadz/imu_rpy` | `geometry_msgs/Vector3` (x=roll, y=pitch, z=yaw) | 500 Hz | `_low_state_handler` |
| `/nomadz/getup_return_code` | `std_msgs/Int32` | on each `GetUp()` call | `RecoveryStateMachine` |

These are all small messages — the bandwidth cost is negligible compared to the existing `/low_state` stream.

### 8.2 Foxglove layout

Save at `nomadz_deploy/configs/foxglove_recovery.json`. Panels:

1. **State indicator** — `Indicator` panel on `/nomadz/recovery_state` (latest string), color-coded per state (`RUNNING` green, `FALLEN` red, `RECOVERING` yellow, `STAGED` blue).
2. **`proj_g_z` plot** — `Plot` panel on `/nomadz/proj_g_z`, with a horizontal line at the threshold (`-0.5`) so you can see the trip live. Add `/nomadz/fall_flag` as a step trace overlaid.
3. **IMU rpy plot** — `Plot` panel on `/nomadz/imu_rpy.x`, `.y`, `.z`. Useful to confirm the recovery actually orients the robot correctly.
4. **Mode timeline** — `Indicator` on `/nomadz/mode` + `Log` panel filtering on string changes ("CUSTOM → DAMP" etc.). Compare against `/nomadz/recovery_state` to spot illegal-transition rejects (mode reads back something unexpected).
5. **Joystick** — `Plot` panel on `/nomadz/joystick_axes.x/y/z` plus a `Log` panel on `/nomadz/button_events`. So the human reviewing the layout can see "operator pressed B at t=12.3s" right above the mode timeline.
6. **Joint trace** — `Plot` panel on selected fields of `/low_state.motor_state_serial[].q` (the existing topic). Foxglove can slice array fields. Useful to see the policy resume cleanly after recovery.
7. **GetUp return code** — small `Indicator` on `/nomadz/getup_return_code`. Should be `0`; a non-zero value is exactly what we want to *see* during a failed recovery.
8. **3D view (optional)** — Foxglove's `3D` panel on the existing `/low_state` IMU + joint state, if there's a robot URDF available. Out of scope for v1.

Equivalent to the `nomadz-neo/booster_soccer.layout` we found in §5, but for the recovery scenario.

### 8.3 How to launch Foxglove against this

`foxglove_bridge` is the standard ROS 2 → Foxglove WebSocket bridge. Two options:

* **Run the bridge alongside the portal:**
  ```bash
  ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765
  ```
  Then point Foxglove Studio at `ws://<robot-or-laptop>:8765`. The Foxglove side imports the JSON layout.
* **Use Foxglove's "ROS 2 native" connection** (newer versions). Same JSON layout; just different connection backend.

We'll add a one-paragraph note in the project README pointing at the layout file and the bridge command.

---

## 9. SDK fork edits (clean follow-up, on the same plan)

`booster_robotics_sdk` is our fork. Two binding gaps to close, both small:

### 9.1 `FallDownStateSubscriber` Python binding

Mirror the existing `B1LowStateSubscriber` pattern in `booster_robotics_sdk/python/binding.cpp` (around line 707). New code:

```cpp
// Add near the FallDownState include (top of binding.cpp).
#include "booster/idl/b1/FallDownState.h"

// Inside PYBIND11_MODULE(...):
py::class_<booster_interface::msg::FallDownState>(m, "FallDownState")
    .def(py::init<>())
    .def_property("fall_down_state",
        py::overload_cast<>(&booster_interface::msg::FallDownState::fall_down_state, py::const_),
        py::overload_cast<booster_interface::msg::FallDownStateType>(&booster_interface::msg::FallDownState::fall_down_state))
    .def_property("is_recovery_available",
        py::overload_cast<>(&booster_interface::msg::FallDownState::is_recovery_available, py::const_),
        py::overload_cast<bool>(&booster_interface::msg::FallDownState::is_recovery_available));

py::enum_<booster_interface::msg::FallDownStateType>(m, "FallDownStateType")
    .value("IS_READY",      booster_interface::msg::FallDownStateType::IS_READY)
    .value("IS_FALLING",    booster_interface::msg::FallDownStateType::IS_FALLING)
    .value("HAS_FALLEN",    booster_interface::msg::FallDownStateType::HAS_FALLEN)
    .value("IS_GETTING_UP", booster_interface::msg::FallDownStateType::IS_GETTING_UP)
    .export_values();

// And the subscriber wrapper, modelled on B1LowStateSubscriber (see binding.cpp:707-718).
py::class_<robot::b1::B1FallDownStateSubscriber, std::shared_ptr<robot::b1::B1FallDownStateSubscriber>>(
    m, "B1FallDownStateSubscriber")
    .def(py::init<const py::function &>(), py::arg("handler"))
    .def("InitChannel",   &robot::b1::B1FallDownStateSubscriber::InitChannel)
    .def("CloseChannel",  &robot::b1::B1FallDownStateSubscriber::CloseChannel)
    .def("GetChannelName",&robot::b1::B1FallDownStateSubscriber::GetChannelName);
```

The C++ side needs a matching `B1FallDownStateSubscriber` class — model on `B1LowStateSubscriber` in `include/booster/robot/b1/`. Topic name: `booster::robot::b1::kTopicFallDown` from `b1_api_const.hpp:12` (`rt/fall_down`).

### 9.2 `GetUpWithMode` Python binding

One line in `binding.cpp` (next to the existing `GetUp` def at line 586):

```cpp
.def("GetUpWithMode", &robot::b1::B1LocoClient::GetUpWithMode, py::arg("mode"),
     R"pbdoc(
     /**
      * @brief Get up and end in a specific mode.
      *
      * @param mode RobotMode { kWalking, kSoccer }
      * @return 0 if success, otherwise return error code
      */
     )pbdoc")
```

And add `kSoccer` to the `RobotMode` enum block at `binding.cpp:250-256`:

```cpp
.value("kSoccer", robot::RobotMode::kSoccer)
```

### 9.3 Rebuild + documentation hygiene

* Rebuild the wheel and reinstall in the dev environment.
* Bump the wheel version (e.g. `0.4.0+nomadz.1`).
* Add an entry to `booster_robotics_sdk/CHANGELOG.md` (create if missing):
  ```
  ## [0.4.0+nomadz.1] - 2026-05-08
  ### Added
  - Python bindings for `FallDownState`, `FallDownStateType`, `B1FallDownStateSubscriber`.
  - Python binding for `B1LocoClient.GetUpWithMode(mode)`.
  - `kSoccer` value in the Python `RobotMode` enum.
  ### Why
  Required by `nomadz_deploy` recovery state machine — see
  `nomadz_deploy/docs/RECOVERY_INTEGRATION_PLAN.md`.
  ```
* Mark the wheel as fork-built in the pip metadata so we never confuse it with upstream.
* Add a section to `booster_robotics_sdk/docs/RECOVERY_MOTION_REPORT.md` §5.4 (the Python pseudocode block) noting that the binding gap it describes has now been closed in this fork.

### 9.4 What to do *before* the binding work lands

Per `revision.txt`, IMU-derived fall detection is acceptable for the first cut. So the order of work is:

1. Ship the IMU detector + state machine + JoystickHandler buttons (§6, §7) — works today, no SDK rebuild.
2. Validate on the hoist (§10).
3. Then close the binding gaps (§9.1, §9.2) and switch the fall detector over to `B1FallDownStateSubscriber`. The state machine itself doesn't change — only the source of `_fall_event` does.

---

## 10. Step-by-step implementation plan

PR-sized chunks, runnable in order:

### 10.1 — Edge-triggered button bus on `JoystickHandler`

* Add `BUTTON_*` constants and the `_button_prev` / `_button_pressed_edges` deque + lock.
* Extend `update_values()` to call `joystick.get_button(i)` for `i in range(num_buttons)`.
* Add `consume_press(button_id) -> bool` method.
* Add a `axes()` convenience that returns `(vx, vy, vyaw)` normalised to [-1, 1] without the max-scaling (so the parent can scale per task).
* Unit-test with a fake pygame stub (mock `joystick.get_button` to return a scripted sequence).

### 10.2 — Plumb `--joystick` into the real-robot path

* `scripts/deploy.py`: pass `joystick_enabled=args.joystick` to `BoosterRobotPortal`.
* `BoosterRobotPortal.__init__`: instantiate and start `JoystickHandler` if enabled. Keep `RemoteControlService` for back-compat startup.
* `_low_state_handler`: when `joystick_enabled`, read velocity command from `JoystickHandler` instead of `RemoteControlService`.

### 10.3 — Fall detector in the portal

* In `_low_state_handler`, append the closed-form `gz = -cos(roll)*cos(pitch)` computation and the streak counter.
* New cfg fields on `BoosterRobotControllerCfg` (`controller_cfg.py:80-83`): `fall_proj_g_z_threshold: float = -0.5`, `fall_streak_threshold: int = 5`, `recover_stable_frames: int = 5`. (No `auto_recover` — auto-recovery is intentionally disabled per §13.6; the detector only logs and publishes `/nomadz/fall_flag`.)
* Initialise `self._fall_event = threading.Event()` and `self._fall_streak = 0` in `__init__`.

### 10.4 — `RecoveryStateMachine` in a new file

* New file: `nomadz_deploy/controllers/recovery_state_machine.py`. **Self-contained class, only depends on `B1LocoClient`, the joystick, and a small "portal API" (start/stop inference, get_state).**
* Public surface:
  ```python
  class RecoveryStateMachine:
      def __init__(self, portal: BoosterRobotPortal, joystick: JoystickHandler): ...
      def run(self) -> None:
          """Replaces the linear startup chain in BoosterRobotPortal.run().
          Returns when self.portal.exit_event is set."""
  ```
* Implements the §6 state graph using `time.sleep(0.05)` polling between events.
* Calls portal helpers `_spawn_inference()`, `_pause_inference()`, `_resume_inference()`, `_wait_until_upright()`, `_safe_shutdown()` — added in 10.5.

### 10.5 — Portal helpers + cleanup-path fix

* Move the existing `start_custom_mode_conditionally` body into `_ramp_to_prepare_state()` (no-arg, expects to be called when joints are at rest).
* Move the existing `start_rl_gait_conditionally` body into `_spawn_inference()` (drops the button wait — the state machine owns that). Make this idempotent so the state machine can call it on every `STAGED → RUNNING` transition.
* Add `_terminate_inference()` (`inference_process.terminate(); join(timeout=1.0)`) for the fall-to-recovery path — the policy gets a fresh JIT load + fresh `obs_history` on the next `_spawn_inference`.
* Add `_pause_inference()` / `_resume_inference()` that flip a new `self.policy_run_event = mp.Event()` for the X-press path (no terminate, no JIT reload, just stop publishing). The inference loop in `BoosterRobotController.run` (`booster_robot_controller.py:543`) gains a wait on `policy_run_event` between `update_state()` and `policy_step()`. The `mp.Event` is created in the portal `__init__` (before the subprocess is spawned) so the child inherits the same handle.
* Add `_wait_until_upright(stable_frames=5)` polling `self._proj_g_z` cached in `_low_state_handler`.
* Replace the final `client.ChangeMode(RobotMode.kWalking)` in `run()` with `_safe_shutdown()`:
  ```python
  def _safe_shutdown(self):
      self.client.ChangeMode(RobotMode.kDamping)   # always reachable
      time.sleep(0.2)
      self.client.ChangeMode(RobotMode.kPrepare)
      time.sleep(0.2)
      self.client.ChangeMode(RobotMode.kWalking)
  ```

### 10.6 — Replace `BoosterRobotPortal.run()` body

* Replace its linear chain with `RecoveryStateMachine(self, self.joystick).run()` followed by `_safe_shutdown()`.

### 10.7 — Disable `LocomotionPolicy.enable_safety_fallback`

* In `tasks/locomotion/__init__.py`'s `T1WalkControllerCfg1.__post_init__`, set `self.policy.enable_safety_fallback = False`.
* (Optional follow-up) extend `LocomotionPolicy` so its existing fall-detect signals the parent's `_fall_event` instead of calling `controller.stop()`. Defer until 10.6 is validated.

### 10.8 — Diagnostic publishers + Foxglove layout

* In `BoosterRobotPortal._init_communication`, create one extra ROS 2 node `nomadz_diagnostics` with publishers for the topics in §8.1.
* Wire them into the relevant code paths (mostly one-liners in `_low_state_handler`, the joystick handler, and the state machine).
* Commit `nomadz_deploy/configs/foxglove_recovery.json`.

### 10.9 — (Fork edit, follow-up) close SDK binding gaps

Per §9. Ship after 10.1–10.8 are validated on the hoist.

---

## 11. Validation order (hoisted robot)

Each step is independently testable; do not skip ahead:

1. **A press still arms custom mode (legacy path).** Confirm the existing `RemoteControlService` behaviour still works without `--joystick`.
2. **`--joystick` lights up `JoystickHandler` axes.** Without the state machine, just confirm `synced_command` reflects our pygame controller's stick. Manual mode change for now.
3. **Y press starts inference subprocess** under the new state machine. RL policy walks normally.
4. **X press damps cleanly.** Inference pauses (`policy_run_event.clear()`) → `ChangeMode(kDamping)` → joints go limp. No crash, state machine returns to `IDLE`.
5. **Manual B press from `RUNNING` (no real fall) does the full recovery cycle.** This is the §6 graph in miniature: CUSTOM → DAMP → PREP → GetUp → WALK → PREP → STAGED. Confirm `client.GetUp()` returns 0 and the firmware accepts every transition.
6. **Y from `STAGED` resumes locomotion.** Inference subprocess respawns / `policy_run_event` is set, `t1_walk.pt` walks again.
7. **Trip the IMU detector by tilting the hoisted robot.** Confirm `_fall_event` fires and the diagnostic publishers light up. State machine should *log only* and wait for an operator B press — auto-recovery is intentionally off.
8. **Free fall (low slack) → operator B press → recovery.** Wiki-recommended "drive into a falling pose" test from `b1_loco_example_client.cpp:362`'s `gu`/`ld` flow.
9. **Foxglove layout shows everything live** during steps 5 and 8 — `recovery_state` color cycles through FALLEN/RECOVERING/STAGED/RUNNING, `proj_g_z` plot shows the dip, `mode` indicator transitions cleanly. This is also the *demo*-grade evidence that the loop works.
10. **Only after 1–9 pass:** drop the hoist. Even then, keep an operator on X.

---

## 12. Files we expect to touch

| File | Change |
|---|---|
| `nomadz_deploy/utils/joystick_handler.py` | Add `BUTTON_*` constants; extend `update_values()` to read buttons; add `consume_press`, `axes`. ~30 lines. |
| `nomadz_deploy/controllers/booster_robot_controller.py` | Instantiate `JoystickHandler` (`--joystick`); add fall detector + `_proj_g_z` cache to `_low_state_handler`; refactor `start_custom_mode_conditionally` / `start_rl_gait_conditionally` into `_ramp_to_prepare_state` / `_spawn_inference`; add `_pause_inference` / `_resume_inference` / `_wait_until_upright` / `_safe_shutdown`; rewire `run()` to call `RecoveryStateMachine`. |
| `nomadz_deploy/controllers/recovery_state_machine.py` | **(NEW)** the state-machine class. ~150 lines. |
| `nomadz_deploy/controllers/controller_cfg.py` | Add `fall_proj_g_z_threshold`, `fall_streak_threshold`, `recover_stable_frames` to `BoosterRobotControllerCfg`. (No `auto_recover` field — the v2.1 decisions removed auto-recovery.) |
| `tasks/locomotion/__init__.py` | Set `enable_safety_fallback = False` on the `t1_walk` cfg. |
| `scripts/deploy.py` | Pass `joystick_enabled=args.joystick` to `BoosterRobotPortal`. (No `--auto-recover` flag added.) |
| `nomadz_deploy/configs/foxglove_recovery.json` | **(NEW)** the Foxglove layout from §8.2. |
| `docs/RECOVERY_INTEGRATION_PLAN.md` | This file. |
| `booster_robotics_sdk/python/binding.cpp` | (FORK EDIT, post-validation) `B1FallDownStateSubscriber`, `FallDownState`, `FallDownStateType`, `GetUpWithMode`, `kSoccer`. |
| `booster_robotics_sdk/include/booster/robot/b1/b1_fall_down_state_subscriber.hpp` | (FORK EDIT, NEW) C++ subscriber class mirroring `B1LowStateSubscriber`. |
| `booster_robotics_sdk/CHANGELOG.md` | (FORK EDIT) Document the new bindings per §9.3. |

---

## 13. Resolved decisions (v2.1)

The v2 doc had seven open questions; this is the locked-in answer to each.

1. **Coexist with the manufacturer's joystick path.** `JoystickHandler` (pygame, our controller) is added *alongside* `RemoteControlService` (evdev, manufacturer's controller). The two read different physical devices and don't fight; the recovery state machine reads only from `JoystickHandler`. **No removal of `RemoteControlService`.**
2. **`policy_run_event` is the primary pause mechanism.** Used for the X-press emergency damp (no JIT reload, just stop publishing).
3. **Terminate-and-respawn the inference subprocess on the fall-to-recovery path.** `t1_walk.pt`'s JIT load is cheap, and the fresh subprocess gives clean `obs_history` / `last_action`, which is exactly what we want after a fall. Concretely: X press uses `policy_run_event`, B press / fall uses `terminate(); join(); spawn()`. Both paths exist; the state machine picks per transition.
4. **`A` is unused by the state machine.** Removed from the button table. Operator muscle memory is fine — A is now free, the manufacturer's controller's A binding is unaffected.
5. **No MuJoCo recovery mock.** Skipped for v2.1. Real-robot validation only.
6. **No auto-recovery.** The IMU detector trips `_fall_event` and publishes `/nomadz/fall_flag` — but the state machine only reacts to a B press. **No `--auto-recover` flag, no `auto_recover` cfg field.** Revisit once the loop is proven on hardware.
7. **`proj_g_z` threshold stays at `-0.5`.** Same value `LocomotionPolicy.compute_observation` already uses; revisit empirically only if false positives show up in the IMU recordings from validation step 7.

These are settled — no more questions before implementation begins. The next ambiguity to resolve will surface during 10.1 (button extension): pygame button indices vary by OS / controller model, so the validation step 1.5 ("press each button on our pad and confirm `update_values()` sees the right index") becomes the empirical step that pins down the `BUTTON_Y / BUTTON_B / BUTTON_X` constants.

---

## 14. Pointers back into the recovery report

* `RECOVERY_MOTION_REPORT.md` §1 — `B1LocoClient::GetUp / GetUpWithMode / LieDown` and `LocoApiId::kGetUp = 2008`.
* §5.1 — verbatim wiki rules; the cheat sheet in §6 above is the same table compressed.
* §5.2 — full corrected recovery sequence diagram. **§6 of this plan is exactly that sequence in our code.**
* §5.3 — C++ pseudocode using `ChannelFactory::CreateRecvChannel<FallDownState>`. We close that gap in §9.1.
* §5.4 — Python pseudocode anticipating the binding gaps. Our plan is the production version of that snippet.
* §5.5 — caveats. The hoist warning still applies.

---

*End of plan v2. Nothing in either repo has been modified by writing this document.*
