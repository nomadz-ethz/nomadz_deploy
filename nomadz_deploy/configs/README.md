# Foxglove layouts

## `foxglove_recovery.json`

Live dashboard for the t1_walk + recovery state machine. Pairs with
`docs/RECOVERY_INTEGRATION_PLAN.md` §8.

### What it shows

Top row (indicators):

| Panel | Topic | What it tells you |
|---|---|---|
| **Recovery State** | `/nomadz/recovery_state` | Current FSM state: IDLE / PREP_READY / RUNNING / FALLEN / RECOVERING / STAGED. Color-coded so a glance tells you whether the policy is publishing. |
| **Mode** | `/nomadz/mode` | Last firmware mode the FSM commanded via `ChangeMode`. |
| **Fall flag** | `/nomadz/fall_flag` | Red when the IMU-derived detector has tripped. Auto-recovery is OFF, so this just signals "operator should consider B-press". |
| **GetUp RC** | `/nomadz/getup_return_code` | Last firmware return code from `client.GetUp()`. Should be `0`; non-zero means the recovery cycle aborted into EXITING. |

Middle / bottom (plots and logs):

| Panel | Topic | What it tells you |
|---|---|---|
| **proj_g_z plot** | `/nomadz/proj_g_z` | Projected gravity z in body frame. Upright ≈ -1; sample crosses -0.5 → fall trip after `fall_streak_threshold` consecutive ticks. |
| **IMU rpy plot** | `/nomadz/imu_rpy` (Vector3) | Roll/pitch/yaw stream. Watch pitch & roll for sanity checks during recovery. |
| **Joystick axes plot** | `/nomadz/joystick_axes` (Vector3) | Normalised vx/vy/vyaw [-1, 1] from the operator pad. Confirms inputs are reaching the portal. |
| **Button events** | `/nomadz/button_events` (String, "Y"/"B"/"X") | One message per consumed press. Useful as a timing reference against the state-transition log. |
| **Recovery log** | `/nomadz/recovery_state` (String) | History of state transitions, alongside the live indicator. |

### How to launch

1. Run the recovery deployment (this brings the publishers online):

    ```bash
    python scripts/deploy.py --task t1_walk --joystick
    ```

2. In a second terminal, start the Foxglove WebSocket bridge:

    ```bash
    ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765
    ```

3. Open Foxglove Studio → **Open connection** → **Foxglove WebSocket** →
   `ws://<host>:8765`.

4. **Layout → Import from file…** → pick this JSON.

If panels stay grey, double-check (a) `ros2 topic list | grep /nomadz` shows
the topics, and (b) the bridge process can see the same DDS domain as the
deploy script.
