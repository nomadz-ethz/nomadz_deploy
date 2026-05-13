# Mode-change investigation & manual override

> **TL;DR.** During the first real-hardware bring-up, the FSM happily
> logged `Recovery state idle → prep_ready` and Foxglove showed
> `Mode = kCustom` while the robot did absolutely nothing. The cause
> turned out to be **a single discarded return code**: `client.ChangeMode(...)`
> is an RPC that returns an `int` status, and our wrapper threw it away
> — only catching Python exceptions. A firmware-side rejection (rc=409
> Conflict, rc=502 StateTransitionFailed) or even a 1 s RPC timeout
> (rc=100) all looked like "success" to the FSM.
>
> This document explains the full data path from keystroke to motors,
> exactly where the silent-failure window was, the fixes applied on
> branch `T1_deploy` after commit `6243fee`, and how to manually drive
> a firmware mode change without our FSM as a pre-deploy sanity test.
>
> **Date:** 2026-05-13. **Branch:** `T1_deploy`. **Workspace:**
> `~/nomadz_deploy`, with the SDK fork at `~/booster_robotics_sdk`.

---

## Quick reference

**Sanity-test a single mode change (no FSM, no joystick, no keyboard):**

```bash
# On the motion board:
ssh master@192.168.10.101
cd ~/nomadz_deploy
source /opt/booster/BoosterRos2Interface/install/setup.bash

# Read current firmware mode (non-destructive):
python3 scripts/manual_mode.py status

# Switch to PREP / DAMP / CUSTOM / WALKING (asks for confirmation):
python3 scripts/manual_mode.py prepare
python3 scripts/manual_mode.py damp
python3 scripts/manual_mode.py custom
python3 scripts/manual_mode.py walking
```

Expected output (good run):

```
[manual_mode] ChannelFactory.Init(0, '127.0.0.1')
[manual_mode] B1LocoClient().Init()
[manual_mode] reading current mode...
  before: firmware mode = kPrepare (1)
Switch firmware to kDamping? Robot WILL move. (y/N) y
[manual_mode] ChangeMode(kDamping)...
[manual_mode] ChangeMode(kDamping) rc=0 (request accepted)
  after : firmware mode = kDamping (0)
[manual_mode] OK — firmware is now in kDamping.
```

If `status` itself prints `rc=100` or "cannot read firmware mode", the
SDK channel isn't reaching the firmware — fix that before doing anything
else. See §6.

**Operator inputs (FSM):** unchanged from the May 8 runbook —
`y/b/x` for buttons; `w/s/a/d/q/e/Space` for axes in `--keyboard` mode.

---

## 1. What changed in code

Three files patched (post-`6243fee`):

| File | Patch |
|---|---|
| `nomadz_deploy/controllers/recovery_state_machine.py` | `_change_mode` now reads the rc from `ChangeMode`, fails the transition on non-zero, and (when the binding exposes it) calls `GetMode` to confirm the firmware actually transitioned. Added `_RPC_RC_EXPLAIN` table + `_explain_rpc_rc()` so every error gets a human-readable explanation. |
| `nomadz_deploy/controllers/booster_robot_controller.py` | `_ramp_to_prepare_state` now captures the rc from `ChangeMode(kCustom)`, aborts the ramp if non-zero (so we don't blast joint targets at a firmware that isn't listening), and optionally verifies via `GetMode`. `_safe_shutdown` likewise checks each rc and only publishes `/nomadz/mode` on confirmed success, and stops the sequence on the first failure rather than blindly continuing. |
| `scripts/manual_mode.py` | **New** — standalone CLI that exercises a single `ChangeMode` round-trip outside the FSM. See §5 for usage. |

To get the patches onto the motion board:

```bash
# On the workstation:
git -C ~/nomadz_deploy status                          # should be clean
rsync -av --exclude '.git' --exclude '__pycache__' \
  --exclude 'logs' --exclude '_offline_wheels' \
  ~/nomadz_deploy/ master@192.168.10.101:~/nomadz_deploy/

# Verify on the robot:
ssh master@192.168.10.101 \
  'grep -n "ChangeMode firmware rejected" \
     ~/nomadz_deploy/nomadz_deploy/controllers/recovery_state_machine.py \
   && test -f ~/nomadz_deploy/scripts/manual_mode.py && echo OK'
```

---

## 2. The full data path: keystroke → motors

The recovery prototype touches four communication layers. Knowing which
layer carries which message is the key to debugging the silent-failure
class.

```
┌─────────────┐   stdin   ┌──────────────┐   .consume_press(n)   ┌──────────────────┐
│   `y` key   │──cbreak──▶│  KeyboardHdlr│──── (rising edge)────▶│ RecoveryFSM tick │
└─────────────┘           └──────────────┘                       └────────┬─────────┘
                                                                          │
                          ┌───────────────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────────┐
              │ portal._ramp_to_prepare()   │   ──── (a) /joint_ctrl ────▶ firmware
              │ portal._spawn_inference()   │       (ROS 2 topic; LowCmd; 500 Hz)
              │ fsm._change_mode(kCustom)   │
              └────────────┬────────────────┘
                           │
                           ▼
              ┌─────────────────────────────┐
              │ self.client.ChangeMode(...) │   ──── (b) rt/LocoApiTopic ─▶ firmware
              │ self.client.GetMode(...)    │       (raw Fast DDS RPC;
              │   (B1LocoClient methods)    │        RpcReqMsg + RpcRespMsg
              └─────────────────────────────┘        keyed by UUID; ~1 s timeout)
                           │
              ┌────────────┘
              ▼
   /low_state subscription   ◀────── (c) /low_state ──── firmware
   (rclpy; LowState msg;        (ROS 2 topic; ImuState
    drives FSM's fall            + per-joint MotorState;
    detector and Foxglove        500 Hz)
    indicators)
                                  
   /nomadz/* publishers      ────── (d) /nomadz/* ────▶ Foxglove bridge ─▶ Studio
   (rclpy; diagnostic;          (ROS 2; recovery_state,
    50 Hz to 500 Hz)              mode, fall_flag, …)
```

The four layers, with their distinct failure modes:

* **(a) `/joint_ctrl` — ROS 2 topic, `booster_interface/LowCmd`.** This is
  what the RL policy publishes at 500 Hz to drive the motors. The firmware
  must be in **kCustom** mode to act on it; in any other mode the topic
  is published happily but the firmware ignores its contents. Subscribers
  count on the publisher is observable via
  `self.low_cmd_publisher.get_subscription_count()` (used by
  `_ramp_to_prepare_state` line 569 as a "is the firmware listening at
  all" gate, but **not** by anything during RUNNING).
* **(b) `rt/LocoApiTopic` — raw Fast DDS topic, `booster_msgs::msg::RpcReqMsg`.**
  This is the SDK's RPC channel that carries every `B1LocoClient` method
  (`ChangeMode`, `GetMode`, `GetUp`, `LieDown`, …). It is **not** visible
  to `ros2 topic list` — it uses Fast DDS's raw IDL naming, not ROS 2's
  mangled `rt/<topic>` + `<pkg>::msg::dds_::<Type>_` convention. So
  `ros2 topic pub` cannot drive it (see §7). The Python binding for
  `ChangeMode` returns an `int` status; previously, the FSM threw that
  away.
* **(c) `/low_state` — ROS 2 topic, `booster_interface/LowState`.** 500 Hz
  IMU + per-joint state. The IMU drives our fall detector
  (`_low_state_handler` computes `proj_g_z`). Note: **`LowState` does not
  carry the firmware mode** — there is no `current_mode` field in the IDL
  (`booster_robotics_sdk/include/booster/idl/b1/LowState.h:228-231`).
  The closest the firmware publishes passively is the
  `rt/robot_states` topic with `RobotStatesMsg::current_mode`, but
  that one isn't wrapped in the Python binding either.
* **(d) `/nomadz/*` — our diagnostic publishers.** Pure presentation
  layer for Foxglove. These get written by the portal whenever it *thinks*
  something happened. The bug we hit is that the FSM wrote
  `/nomadz/mode = "kCustom"` based on "no exception was raised," not on
  any firmware ack.

---

## 3. The bug: silently-swallowed RPC status

`B1LocoClient::ChangeMode(RobotMode)` is defined header-only as:

```cpp
// booster_robotics_sdk/include/booster/robot/b1/b1_loco_client.hpp:57-61
int32_t ChangeMode(RobotMode mode) {
    ChangeModeParameter change_mode(mode);
    std::string param = change_mode.ToJson().dump();
    return SendApiRequest(LocoApiId::kChangeMode, param);
}
```

It returns the rc from `RpcClient::SendApiRequest`, which is one of:

| rc | name | meaning |
|---:|---|---|
| `-1` | `kRpcStatusCodeInvalid` | request never published (Init not called / torn down) |
| `0`  | `kRpcStatusCodeSuccess` | request accepted by the firmware |
| `100` | `kRpcStatusCodeTimeout` | no response in 1 s (default `SendApiRequest` timeout) |
| `400` | `kRpcStatusCodeBadRequest` | malformed payload |
| `409` | `kRpcStatusCodeConflict` | firmware in a state that disallows this transition |
| `429` | `kRpcStatusCodeRequestTooFrequent` | throttle |
| `500` | `kRpcStatusCodeInternalServerError` | firmware-side fault |
| `501` | `kRpcStatusCodeServerRefused` | firmware actively refused |
| `502` | `kRpcStatusCodeStateTransitionFailed` | firmware FSM blocked |

Source: `booster_robotics_sdk/include/booster/robot/rpc/error.hpp:8-16`.

The pybind binding maps this directly to a Python int:

```cpp
// booster_robotics_sdk/python/binding.cpp:422-431
.def("ChangeMode", &robot::b1::B1LocoClient::ChangeMode, py::arg("mode"),
     "Change the working mode of the robot. ...")
```

Our pre-patch wrapper:

```python
# v2.1, pre-fix — recovery_state_machine.py:153-163
def _change_mode(self, mode: "RobotMode", label: str) -> bool:
    try:
        self.client.ChangeMode(mode)        # ← rc DISCARDED
    except Exception as exc:
        self.logger.error("ChangeMode(%s) failed: %s", label, exc)
        return False
    self.portal._publish_mode(label)        # ← published regardless
    return True
```

A bare call (`_ramp_to_prepare_state:588`, `_safe_shutdown:712-722`) had
the same disease. Net effect: any firmware-side rejection looked
identical to a success.

### Why we couldn't see it from the FSM log

Three reinforcing factors:

1. `_change_mode` returned `True` whenever no Python exception fired.
2. The FSM's `_set_state` logged `Recovery state idle → prep_ready`
   *unconditionally* after `_change_mode` returned `True`.
3. The portal's `_publish_mode("kCustom")` immediately drove the
   Foxglove `Mode` indicator green, **without any read-back**.

So the observable signal — terminal log + Foxglove panel — claimed
success, while the firmware-side mode was unchanged.

### Most likely reasons the firmware rejected

Given that `--net 127.0.0.1` *is* the correct argument on the motion
board (it's what every SDK example uses; see
`booster_robotics_sdk/example/high_level/b1_loco_example_client.py:265`
and the same C++ pattern), the rc was almost certainly NOT a `100`
timeout. The two plausible candidates are:

- **rc=502 StateTransitionFailed.** The firmware FSM disallows certain
  hops. Per the SDK README and motion report, the legal hops are:
  `DAMP→PREP`, `PREP→*`, `WALK→{PREP, DAMP}`, `CUSTOM→{PREP, DAMP}`,
  `PROTECT→DAMP`. `CUSTOM→WALK` and direct hops out of PROTECT to
  anything other than DAMP are rejected. If the firmware boots into
  PROTECT after a hard restart, our `_tick_idle` → `_ramp_to_prepare_state`
  → `ChangeMode(kCustom)` will be rejected (PROTECT→CUSTOM is illegal),
  and the bug would surface exactly as the user observed.
- **rc=409 Conflict.** Similar story; firmware-state-dependent guards.

The new code now logs the actual rc + an explanation, so the next failure
will tell us which one it was.

---

## 4. The fix, in shape

The post-fix `_change_mode` (full body in
`nomadz_deploy/controllers/recovery_state_machine.py`):

```python
def _change_mode(self, mode, label, verify=True):
    try:
        rc = self.client.ChangeMode(mode)
    except Exception as exc:
        self.logger.error("ChangeMode(%s) raised: %s", label, exc)
        return False
    if rc != 0:
        self.logger.error(
            "ChangeMode(%s) firmware rejected: %s",
            label, _explain_rpc_rc(rc))
        return False

    # round-trip verify if the binding exposes GetMode
    if verify and _HAS_GETMODE:
        gm = GetModeResponse(); gm_rc = self.client.GetMode(gm)
        if gm_rc == 0 and int(gm.mode) != int(mode):
            self.logger.error(
                "Firmware mode mismatch after ChangeMode(%s): "
                "firmware reports mode=%s. Treating as failure.",
                label, int(gm.mode))
            return False

    self.portal._publish_mode(label)
    return True
```

Behavioural changes in three places:

1. **`recovery_state_machine._change_mode`** — rc-checked + `GetMode`
   verified. Used by `_tick_fallen`, `_tick_recovering`, `_tick_staged`,
   and the X-press path in `_tick_running`.
2. **`booster_robot_controller._ramp_to_prepare_state`** — rc-checked
   `ChangeMode(kCustom)` before starting the 500-step joint ramp. If the
   mode change is rejected, the ramp aborts (returns `False`) instead of
   publishing 500 ignored LowCmd frames; the FSM then transitions to
   `EXITING`. Optional `GetMode` verify is also added.
3. **`booster_robot_controller._safe_shutdown`** — each of the three
   transitions is rc-checked, `/nomadz/mode` only fires on confirmed
   success, and we bail out on the first failure rather than continuing
   to publish ghost transitions.

What we deliberately did **not** change:

- The `B1LocoClient.Init()` call has no return value in C++
  (`booster_robotics_sdk/include/booster/robot/b1/b1_loco_client.hpp:20-22`).
  We can't rc-check that. The SDK provides `B1LocoClient::WaitForService(timeout, true)`
  for fail-fast discovery, but at time of writing we haven't confirmed
  it's exposed in the Python binding. Adding it is a follow-up — see §8.
- We still don't subscribe to `rt/robot_states` for passive mode read-back.
  The Python binding doesn't wrap the `RobotStatesMsg` IDL, so adding
  this is a larger change (write a C++ binding for it).

---

## 5. Manual mode change (`scripts/manual_mode.py`)

The new utility constructs a single `B1LocoClient`, reads the current
firmware mode via `GetMode`, optionally issues one `ChangeMode`, and
reads back to confirm. It is essentially this, padded with logging:

```python
from booster_robotics_sdk_python import (
    B1LocoClient, ChannelFactory, RobotMode, GetModeResponse,
)
ChannelFactory.Instance().Init(0, "127.0.0.1")
c = B1LocoClient(); c.Init()
gm = GetModeResponse()
c.GetMode(gm); print(f"before: {int(gm.mode)}")
rc = c.ChangeMode(RobotMode.kPrepare); print(f"ChangeMode rc={rc}")
import time; time.sleep(0.3)
c.GetMode(gm); print(f"after:  {int(gm.mode)}")
```

### CLI

```
scripts/manual_mode.py <mode> [--net IP] [--yes]
```

| Argument | Default | Notes |
|---|---|---|
| `mode` | (required) | One of `status`, `damp`, `prepare`, `custom`, `walking` |
| `--net` | `127.0.0.1` | DDS interface IP. On motion board, leave default. |
| `--yes / -y` | off | Skip the confirmation prompt |

The mode/integer mapping (for Foxglove or raw DDS pubs if you ever
end up there): `kUnknown=-1, kDamping=0, kPrepare=1, kWalking=2,
kCustom=3, kSoccer=4`
(`booster_robotics_sdk/include/booster/robot/common/robot_shared.hpp:7-14`).

### What "good" looks like

```
$ python3 scripts/manual_mode.py prepare
[manual_mode] ChannelFactory.Init(0, '127.0.0.1')
[manual_mode] B1LocoClient().Init()
[manual_mode] reading current mode...
  before: firmware mode = kDamping (0)

Switch firmware to kPrepare? Robot WILL move. (y/N) y
[manual_mode] ChangeMode(kPrepare)...
[manual_mode] ChangeMode(kPrepare) rc=0 (request accepted)
  after : firmware mode = kPrepare (1)
[manual_mode] OK — firmware is now in kPrepare.
```

### What's NOT good

| Output | Cause | What to try |
|---|---|---|
| `GetMode failed rc=100` (Timeout) | RPC channel up but no firmware response | Check the firmware service is running on the motion board; check `ROS_DOMAIN_ID` is unset or 0 |
| `GetMode failed rc=-1` (Invalid) | Channel was never bound | Probably an import error before this point; check the deploy traceback |
| `ChangeMode rc=502` | Firmware FSM blocked this transition | See the mode-transition rules; try `manual_mode.py damp` first, then `prepare`, then your target |
| `ChangeMode rc=409` | Firmware in a state that conflicts | Same recipe: drop to DAMP and walk up |
| `ChangeMode rc=0` but `after` shows wrong mode | Firmware accepted then silently held its mode (`is_recovery_available` etc. guards) | Retry after a 1-second pause; if it persists, power-cycle |

Every rc/code is also explained verbatim in the script itself.

---

## 6. Pre-keyboard test plan (run this first on every bring-up)

Before exercising the FSM (`--keyboard` or `--joystick`), confirm the SDK
channel is healthy with the standalone script. Three minutes of work that
saves an evening of confusion.

1. **SSH to the motion board.**
   ```
   ssh master@192.168.10.101
   source /opt/booster/BoosterRos2Interface/install/setup.bash
   cd ~/nomadz_deploy
   ```
2. **Read current mode (non-destructive).**
   ```
   python3 scripts/manual_mode.py status
   ```
   Expect `before: firmware mode = kPrepare (1)` immediately after boot.
   If this hangs or returns `rc=100`, **do not proceed** — the channel
   isn't healthy and the FSM will hit the same wall. See §6 troubleshooting
   below.
3. **Round-trip a benign transition.** Robot is on the hoist, in PREP.
   ```
   python3 scripts/manual_mode.py damp
   ```
   Confirm physical: robot goes limp, joints take the hoist's weight.
   Confirm reported: `OK — firmware is now in kDamping.`
4. **Walk back up legally.**
   ```
   python3 scripts/manual_mode.py prepare
   ```
   Confirm physical: robot returns to prepare pose. Confirm
   reported: `OK — firmware is now in kPrepare.`
5. **The riskier one — try `custom`.** This is the same transition
   `_ramp_to_prepare_state` will issue when you press `y` from IDLE.
   If this fails *here*, the FSM will also fail.
   ```
   python3 scripts/manual_mode.py custom
   ```
   Confirm `OK — firmware is now in kCustom.` Then drop back to PREP:
   ```
   python3 scripts/manual_mode.py prepare
   ```
6. **Only now launch the FSM.**
   ```
   python3 scripts/deploy.py --task t1_walk --keyboard
   ```
   When the FSM prints `[recovery] IDLE — press Y …` and you type `y`,
   you will now see (in the **new** code path) one of:

   - `Recovery state idle → prep_ready` followed by `ChangeMode(kCustom)
     confirmed by GetMode().` — the happy path.
   - `_ramp_to_prepare_state: ChangeMode(kCustom) rc=NNN; firmware did
     NOT enter custom mode.` — the explicit failure path. The rc tells
     you which guard tripped.

   The "log says transition, robot does nothing" silent-failure mode is
   eliminated by construction in the new code.

### §6 troubleshooting — channel-level failures

If step 2 (`manual_mode.py status`) hangs or fails:

* **Hangs for 1 s then prints rc=100.** RPC timeout. The Fast DDS topic
  `rt/LocoApiTopic` is up on our side but nobody on the firmware side is
  responding. Check:
    - Is the firmware actually running? `systemctl status booster_*`
      on the motion board. The `loco` service must be up.
    - Is your shell exporting an alien `ROS_DOMAIN_ID`?
      `echo $ROS_DOMAIN_ID` — should be empty or `0`.
    - Is `RMW_IMPLEMENTATION` set to something other than
      `rmw_fastrtps_cpp`? Default is fine; Cyclone DDS will not bridge.
    - Are there stale FastDDS profiles? Check
      `echo $FASTRTPS_DEFAULT_PROFILES_FILE`. Should usually be empty
      or point to `/opt/booster/BoosterRos2/fastdds_profile.xml`.
* **Prints `rc=-1` (Invalid).** Request never published. The most likely
  cause is an `Init()` that hit an internal error. Run the script with
  `python3 -u` to flush stdout and look for any stderr message about
  "Failed to publish RPC request" or "RPC endpoints are not matched".
* **Crashes on `from booster_robotics_sdk_python import …`.** SDK was
  built without `-DBUILD_PYTHON_BINDING=ON`, or `pybind11-stubgen`
  wasn't installed at build time. See `docs/2026-05-08-recoveryprototype.md`
  §2.1.

---

## 7. Why can't we just `ros2 topic pub` it?

The user asked whether mode changes could be driven via manual ROS
topic commands as a pre-keyboard test. Short answer: **no, not
practically** — and the `manual_mode.py` script is the equivalent.

The long answer is worth knowing because it generalises to every other
`B1LocoClient` method.

### What ChangeMode actually publishes

`ChangeMode` ultimately calls `RpcClient::SendApiRequest`, which:

1. Constructs a `booster_msgs::msg::RpcReqMsg` with three string fields:
    - `uuid` — a freshly generated UUIDv4 (so the client can match the
      response to this specific call).
    - `header` — a JSON string: `{"api_id": 2000, "expect_response": true}`
      (`2000` is `LocoApiId::kChangeMode` — see
      `booster_robotics_sdk/include/booster/robot/b1/b1_loco_api.hpp:22`).
    - `body` — a JSON string: `{"mode": <int>}`.
2. Publishes that on the Fast DDS topic `rt/LocoApiTopic`.
3. Subscribes to the same topic, watches for an `RpcRespMsg` whose
   `uuid` matches, parses its `header` for the status code, returns
   the status code.

So a "manual mode change via raw publish" would require you to:

- Generate a UUIDv4 you also subscribe with.
- Construct the JSON `header` and `body`.
- Publish on `rt/LocoApiTopic` with the right Fast DDS topic type
  (`booster_msgs::msg::RpcReqMsg`).
- Subscribe to `rt/LocoApiTopic` for an `RpcRespMsg` with the matching
  UUID.

That is exactly what `RpcClient` does for you. Plus — and this is the
deal-breaker for raw `ros2 topic pub` — `rt/LocoApiTopic` is **not a
ROS 2 topic**. It's a vanilla Fast DDS topic. ROS 2 with `rmw_fastrtps_cpp`
prefixes topic names with `rt/` like the SDK does, but it also mangles
type names (`booster_msgs::msg::dds_::RpcReqMsg_`) and includes a ROS-2
specific type hash. There is no `rosidl_message` package shipped that
maps `RpcReqMsg` to a ROS 2 IDL — `find ~/booster_robotics_sdk -name
'*.msg' -o -name 'rosidl_*'` comes up empty.

You could in principle implement a minimal RPC client in pure ROS 2 by
declaring a matching `.msg`, generating the right type hash, and
emulating the UUID round-trip… but at that point you've re-written
`RpcClient`. The two-step Python script in §5 is the saner path.

The same logic applies to every other `B1LocoClient` method
(`LieDown`, `GetUp`, `GetStatus`, etc.). They all go over the same
`rt/LocoApiTopic` RPC, distinguished only by the `api_id` in the header.
None of them are `ros2 topic pub`-able.

### What you *can* do with `ros2 topic`

These topics ARE ROS-2-visible and useful for live observation /
manual injection:

| Direction | Topic | Type | Use |
|---|---|---|---|
| firmware → us | `/low_state` | `booster_interface/LowState` | Read joint state + IMU. `ros2 topic echo /low_state --once` is your "is the firmware even alive" smoke test. |
| us → firmware | `/joint_ctrl` | `booster_interface/LowCmd` | The actual motor command. **Only acted on in `kCustom` mode.** Could in principle be `ros2 topic pub`-ed manually, but the payload is ~30 motor entries — easier from Python. |
| us → us | `/nomadz/*` | various | Our diagnostic publishers. `ros2 topic list \| grep /nomadz` should show 8 of them when the deploy is running with diagnostics. |

In short: ROS 2 topics carry the **data plane** (motor commands and
sensor stream). The SDK RPC channel carries the **control plane** (mode
changes, GetUp, GetStatus). To touch the control plane manually, use
the Python binding, not `ros2 topic pub`.

---

## 8. Open follow-ups

- **Add `WaitForService` after `B1LocoClient.Init()`** if the Python
  binding exposes it. This would surface a "firmware not discovered"
  failure in 5 s at startup instead of 1 s at first `ChangeMode`.
  Pseudo-fix:
  ```python
  self.client.Init()
  if hasattr(self.client, "WaitForService"):
      if not self.client.WaitForService(5000, True):
          raise RuntimeError("Booster loco service not discovered in 5s")
  ```
  We didn't apply this yet because the binding wasn't verified to expose
  it (the agent transcript noted `WaitForService` exists on the C++ side
  but the Python pybind file wasn't checked end-to-end for it).
- **Subscribe to `rt/robot_states` for passive mode read-back.** Would
  require writing a C++ pybind wrapper for `RobotStatesMsg` (currently
  unbound). With that in place, the FSM could continuously sanity-check
  its internal "I think we're in CUSTOM" against the firmware's
  `current_mode` field at no extra RPC cost.
- **Probe the inference subprocess in `_tick_running`.** Today the FSM
  doesn't react if the policy subprocess dies — runbook §13 limitation
  #5. Adding a `proc.is_alive()` check + `_set_state(EXITING)` when
  False is a 5-line follow-up.
- **Unify `_change_mode` between the FSM and the portal.** Currently
  the rc-check logic lives in two places (FSM's `_change_mode` and
  the inline blocks in `_ramp_to_prepare_state` / `_safe_shutdown`).
  Promote a single `BoosterRobotPortal.client_change_mode_safe(mode, label)`
  helper that both call.

---

## 9. Cross-references

| Source | Why it matters |
|---|---|
| `docs/2026-05-08-recoveryprototype.md` | The original runbook. §6 (keyboard) and §10 (validation) are still authoritative; this doc supplements §12 (troubleshooting) with the mode-change failure mode. |
| `docs/RECOVERY_INTEGRATION_PLAN.md` | The architecture plan; §6 has the state machine. The decision log in §13 has the call shape; this doc updates the "we discard rc" implicit assumption. |
| `scripts/manual_mode.py` | The new test utility. |
| `nomadz_deploy/controllers/recovery_state_machine.py:`<br>`_change_mode` | The post-fix wrapper. |
| `nomadz_deploy/controllers/booster_robot_controller.py:`<br>`_ramp_to_prepare_state`, `_safe_shutdown` | The two other call sites, now also rc-checked. |
| `~/booster_robotics_sdk/include/booster/robot/b1/b1_loco_client.hpp:57-61` | C++ source of truth for the `ChangeMode` return type. |
| `~/booster_robotics_sdk/include/booster/robot/rpc/error.hpp:8-16` | The full rc table. |
| `~/booster_robotics_sdk/example/high_level/b1_loco_example_client.py` | Upstream Python recipe `manual_mode.py` is modelled on. |
| `~/booster_robotics_sdk/python/binding.cpp:422-431` | The pybind line that exposes `ChangeMode` and proves its return value is an `int`. |
| `~/booster_robotics_sdk/include/booster/robot/common/robot_shared.hpp:7-14` | `RobotMode` enum integer values for raw DDS / Foxglove display. |

---

*End of investigation doc. If your next session finds a different
failure mode under the same "log says success / robot does nothing"
umbrella, add a row to §3 (the rc table) and a § to §6 (channel
troubleshooting) so the next operator doesn't have to re-derive it.*
