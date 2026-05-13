#!/usr/bin/env python3
"""Standalone firmware mode-change utility for the T1 motion board.

This script bypasses the full deploy stack and the recovery FSM. It
constructs a single ``B1LocoClient`` and uses it to issue one
``ChangeMode`` RPC, then reads the firmware mode back via ``GetMode``
to confirm the transition actually took effect.

Use this as a pre-deploy sanity test:

    python3 scripts/manual_mode.py status         # just read current mode
    python3 scripts/manual_mode.py prepare        # → kPrepare
    python3 scripts/manual_mode.py damp           # → kDamping
    python3 scripts/manual_mode.py custom         # → kCustom (from PREP/DAMP)
    python3 scripts/manual_mode.py walking        # → kWalking

If ``status`` works, the SDK channel is healthy. If ``prepare``/``damp``
also work, mode changes from this host can reach the firmware — which
means the keyboard / joystick deploy *should* work, and any remaining
"FSM says mode changed but robot didn't move" symptom is a higher-level
bug (e.g. /joint_ctrl publisher subscriber count is zero) rather than
a channel problem.

Why is this needed even though we have a ChangeMode in the deploy?
Because the SDK ``ChangeMode`` is an RPC over a raw Fast DDS topic
(``rt/LocoApiTopic``, message ``booster_msgs::msg::RpcReqMsg``) that
is NOT directly accessible via ``ros2 topic pub`` — it's not a
ROS-2-mappable message, and the request/response routing depends on
per-call UUIDs handled inside the SDK's ``RpcClient``. The minimal
way to invoke it manually is the six lines below.

See also:
    docs/2026-05-13-mode-change-investigation.md — why we wrote this
        utility and how the silent-failure bug was diagnosed.
    booster_robotics_sdk/example/high_level/b1_loco_example_client.py
        — upstream example this is modelled after.
"""
from __future__ import annotations

import argparse
import sys
import time

from booster_robotics_sdk_python import (  # type: ignore
    B1LocoClient,
    ChannelFactory,
    RobotMode,
)

try:
    from booster_robotics_sdk_python import GetModeResponse  # type: ignore
    _HAS_GETMODE = True
except ImportError:  # pragma: no cover
    _HAS_GETMODE = False


# Friendly CLI names → RobotMode enum. The SDK enum integer values are
# kUnknown=-1, kDamping=0, kPrepare=1, kWalking=2, kCustom=3, kSoccer=4
# (booster_robotics_sdk/include/booster/robot/common/robot_shared.hpp).
MODE_BY_NAME = {
    "damp":    RobotMode.kDamping,
    "prepare": RobotMode.kPrepare,
    "walking": RobotMode.kWalking,
    "custom":  RobotMode.kCustom,
}

# Reverse mapping for read-back display. Integer-keyed so an unexpected
# value from the firmware still prints something useful.
MODE_FROM_INT = {
    -1: "kUnknown",
    0:  "kDamping",
    1:  "kPrepare",
    2:  "kWalking",
    3:  "kCustom",
    4:  "kSoccer",
}

# SDK RPC status codes — same table as recovery_state_machine._RPC_RC_EXPLAIN.
# Kept in sync by hand; if you edit one, edit the other. See
# booster_robotics_sdk/include/booster/robot/rpc/error.hpp.
RC_EXPLAIN = {
    -1: "Invalid: request never published (Init not called / RpcClient torn down).",
    0:  "Success.",
    100: "Timeout: channel is up but no firmware response in 1s. "
         "Check --net, that the firmware is up, and ROS_DOMAIN_ID.",
    400: "BadRequest: malformed payload (mode enum value rejected).",
    409: "Conflict: firmware in a state that disallows this transition "
         "(e.g. WALK → CUSTOM directly; must hop via PREP).",
    429: "RequestTooFrequent: throttle — sleep 0.2s between calls.",
    500: "InternalServerError: firmware-side fault.",
    501: "ServerRefused: firmware actively refused this op.",
    502: "StateTransitionFailed: firmware state machine couldn't perform "
         "this transition (e.g. PROTECT and guards not met).",
}


def _explain(rc: int) -> str:
    return RC_EXPLAIN.get(int(rc), f"unrecognised status code {rc}")


def _print_mode(client: B1LocoClient, prefix: str) -> int:
    """Return the integer firmware mode, or -1 on failure."""
    if not _HAS_GETMODE:
        print(f"  {prefix}: GetMode not exposed in this SDK build — "
              "skipping read-back.")
        return -1
    gm = GetModeResponse()
    rc = client.GetMode(gm)
    if rc != 0:
        print(f"  {prefix}: GetMode failed rc={rc} ({_explain(rc)})")
        return -1
    val = int(gm.mode)
    print(f"  {prefix}: firmware mode = {MODE_FROM_INT.get(val, f'?{val}')} "
          f"({val})")
    return val


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "mode",
        choices=list(MODE_BY_NAME) + ["status"],
        help="Target mode, or 'status' to just read the current mode "
             "without changing it.")
    p.add_argument(
        "--net", default="127.0.0.1",
        help="DDS network interface for ChannelFactory.Init(). "
             "When running on the motion board itself, leave as 127.0.0.1 "
             "(every SDK example does this). To reach the robot from "
             "another host, pass that host's NIC IP. Note: this is "
             "an *IPv4 address*, not an interface name like 'enp1s0'.")
    p.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the are-you-sure prompt for destructive modes.")
    args = p.parse_args()

    print(f"[manual_mode] ChannelFactory.Init(0, {args.net!r})")
    ChannelFactory.Instance().Init(0, args.net)

    print("[manual_mode] B1LocoClient().Init()")
    client = B1LocoClient()
    client.Init()

    # Pre-read — also doubles as a "is the channel even alive" check.
    # If this returns rc=100 (timeout), the rest of the script will fail
    # the same way, and we want to surface that clearly.
    print("[manual_mode] reading current mode...")
    pre = _print_mode(client, "before")
    if pre == -1 and _HAS_GETMODE:
        print("\n[manual_mode] FATAL: cannot read firmware mode. The SDK "
              "channel is not connected to the robot. Check:")
        print("  • Are you on the motion board (or correct --net)?")
        print("  • Is the firmware service actually running?")
        print("    (ssh master@<robot>: systemctl status booster_*)")
        print("  • Does the env have a stray ROS_DOMAIN_ID set?")
        return 1

    if args.mode == "status":
        return 0

    target = MODE_BY_NAME[args.mode]
    target_int = int(target)
    target_str = MODE_FROM_INT[target_int]

    if not args.yes:
        ans = input(
            f"\nSwitch firmware to {target_str}? Robot WILL move. (y/N) "
        ).strip().lower()
        if ans != "y":
            print("Aborted.")
            return 0

    print(f"[manual_mode] ChangeMode({target_str})...")
    try:
        rc = client.ChangeMode(target)
    except Exception as exc:
        print(f"[manual_mode] ChangeMode raised: {exc}")
        return 2
    if rc != 0:
        print(f"[manual_mode] ChangeMode({target_str}) FAILED rc={rc}")
        print(f"  → {_explain(rc)}")
        return 3
    print(f"[manual_mode] ChangeMode({target_str}) rc=0 (request accepted)")

    # Give the firmware a moment to actually transition before reading.
    time.sleep(0.3)

    post = _print_mode(client, "after ")
    if post == -1:
        # GetMode not bound — can't verify, but the rc=0 from ChangeMode
        # is the best signal we have.
        print("[manual_mode] (no read-back; trusting rc=0.)")
        return 0
    if post != target_int:
        print(f"[manual_mode] ⚠️  Firmware reports {MODE_FROM_INT.get(post, post)}, "
              f"NOT {target_str}. Either the transition was rejected "
              "after the fact or it has not completed yet.")
        return 4
    print(f"[manual_mode] OK — firmware is now in {target_str}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
