# Changelog

Record of code changes made during Claude sessions. Newest entries on top.

---

## 2026-05-28

### trip_planner.py — RED LED on unexpected stops (Claude)
**File:** `Development/ros2/src/qcar2_autonomy/autonomy/trip_planner.py`
**By:** Claude
**Changes:**
- Added `self.LED_RED = 0` (confirmed via `qcar2_hardware.cpp` LED color map: 0=red, 1=green, 2=blue, 3=yellow, 4=cyan, 5=magenta, 6=orange).
- trip_planner now subscribes to `/motion_enable` (Bool, published by `yolo_detector`; True=go, False=stop for stop signs / traffic lights) via `_motion_enable_callback`, storing `self.motion_enabled`.
- In `loop()`, after startup, added an LED override: while in a driving stage (TO_PICKUP/TO_INTERMEDIATE/TO_DROPOFF/TO_HUB), if `motion_enabled` is False → LED RED; when motion resumes → restore the leg's drive colour via `_set_drive_led_for_stage`. WAIT_* stages (pickup/dropoff/hub) are excluded, so their colours (BLUE/ORANGE/MAGENTA) are never overridden.
**Why:** Gabriel's requirement — any stop that is NOT at pickup/dropoff/hub (e.g. stop signs, traffic lights) must show RED. Those external stops come through the object detector's `/motion_enable`, the same signal `nav_to_pose` uses to halt the car.
**Depends on:** `yolo_detector` publishing `/motion_enable=False` when it commands a stop. If a stop is enforced by a different mechanism, RED won't trigger and we'd extend this.
**Open question (not changed):** Codex's `TO_INTERMEDIATE`/`WAIT_AT_INTERMEDIATE` makes the car pause (BLUE) at middle nodes of 3-node rides (e.g. "L: 20, 7, 1"). Unclear whether middle nodes are real passenger stops or just routing waypoints. If routing-only, they should be pass-through (no WAIT) and any incidental stop there would be RED. Needs Gabriel's confirmation of competition semantics.

### trip_planner.py — `trip_nodes` ride command + automatic return to node 10
**File:** `Development/ros2/src/qcar2_autonomy/autonomy/trip_planner.py`
**By:** Codex
**Changes:**
- Added `trip_nodes` integer-array parameter, default `[-1]`, as the new one-command ride interface. Example: `ros2 param set /trip_planner trip_nodes "[1, 8]"`.
- Kept `pickup_node`/`dropoff_node` as compatibility aliases, but normalized ride handling internally around node lists.
- Added validation so rides only start when `trip_planner` is ready and idle, and invalid node IDs are rejected before any path is published.
- Added intermediate-stop support through `TO_INTERMEDIATE` / `WAIT_AT_INTERMEDIATE` stages.
- Mission dispatch now appends `taxi_node` node 10 automatically unless the requested trip already ends at node 10.
- Fixed startup/path completion handling so the initial node-10 hub behavior can complete correctly, including the `10 -> 10` no-op path case.
- Preserved existing LED semantics: pickup travel green, dropoff/intermediate travel blue, hub return orange, mission complete magenta.
**Why:** First-task implementation request: use node-command rides while keeping Gabriel's LED/Yolo/launch behavior, and return to node 10 after every ride.
**Verification:** `python3 -m py_compile Development/ros2/src/qcar2_autonomy/autonomy/trip_planner.py`; `git diff --check -- Development/ros2/src/qcar2_autonomy/autonomy/trip_planner.py`.

### qcar2_cartographer_virtual_launch.py — disable drive converter by default
**File:** `Development/ros2/src/qcar2_nodes/launch/qcar2_cartographer_virtual_launch.py`
**By:** Codex
**Changes:**
- Added `enable_drive_converter` launch argument, default `false`.
- Wrapped `nav2_qcar2_converter` in `IfCondition(enable_drive_converter)`.
- Old behavior is still available with: `ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py enable_drive_converter:=true`.
**Why:** The cartographer launch was starting `nav2_qcar2_converter`, which forwards `/cmd_vel_nav` to `qcar2_motor_speed_cmd`. If `path_follower` from a previous autonomy launch is still running, the QCar can move while the user thinks only cartographer is running.
**Verification:** `python3 -m py_compile Development/ros2/src/qcar2_nodes/launch/qcar2_cartographer_virtual_launch.py`; `git diff --check -- Development/ros2/src/qcar2_nodes/launch/qcar2_cartographer_virtual_launch.py`.

## 2026-05-27

### No default rides + node-10 start awareness
**Files:** `trip_planner.py`, `nav_to_pose.py`
**Changes:**
- `trip_planner.py`: `pickup_node`/`dropoff_node` defaults changed from [1]/[8] to **[-1]** (sentinel = not set). `pickup_xy`/`dropoff_xy` derive to `None` when unset. Ride-start branch now refuses to launch unless BOTH nodes are >= 0 (logs a hint). Net effect: NO default ride ever auto-runs; the order must be set each time via `ros2 param set`. Order-independent: whichever of pickup/dropoff is set second triggers the (now-valid) ride.
- `nav_to_pose.py`: `node_values` default changed from `[0, 8, 10]` to **[10]** so path_follower starts "parked at node 10" with no preset route (no 0→10 path). Guarded `generate_path` (both init and the param callback) to require >=2 nodes; single node → empty placeholder path, real path arrives via `/cmd_waypoints`.
**Why:** Gabriel's requirement — during competition the ride order must be set live with no hardcoded defaults, and the car starts at node 10 (not node 0). Also explains a past symptom: starting at node 0 = QLabs (0,0) = Cartographer's map origin, so the old 0→10 leg happened to be perfectly aligned (zero offset) and worked; every other leg used wrong offsets and the car never reached its waypoint. Auto-align fixes the general case.

### trip_planner.py — node-based destinations + auto-align offsets (Phase 0)
**File:** `Development/ros2/src/qcar2_autonomy/autonomy/trip_planner.py`
**Changes:**
1. **Node-based input:** replaced `pickup_xy`/`dropoff_xy` (double-array) params with `pickup_node`/`dropoff_node` (integer-array, defaults [1]/[8] = Run A). `pickup_xy`/`dropoff_xy` are now *derived* internally from the roadmap via `_get_node_xy(node)`. Downstream state machine, snap logic, and LEDs unchanged (still use the derived xy). `parameter_update_callback` updated to handle the node params (INTEGER_ARRAY).
   - **Why:** Gabriel's request — competition rides are specified by node (e.g. "A: 1, 8"), and manual XY entry was hitting scaling mismatches with the roadmap's internal coords. Using node IDs pulls the exact internal coords, eliminating that error.
2. **Auto-align (Phase 0):** new `_auto_align()` + `_get_node_theta()`. Runs once at loop start (before the startup-hub check). Derives `rotation_offset` and `translation_offset` from the hub node's known QLabs pose (x,y,theta from `get_node_pose`) vs the car's measured Cartographer map pose. Convention used: `_qlabs_path_to_ros` rotates QLabs CCW by rotation_offset, so `map_yaw = qlabs_theta + rotation_offset`; translation solved so the hub node maps to the car's measured map xy.
   - **Why:** `translation_offset=[0,0]` was wrong — Cartographer's map origin sits at the car's start (node 10), not the QLabs origin — so node waypoints landed in the wrong place and the car drove into walls. Auto-align makes node coords convert into the live map frame without manual tuning.
   - **Assumption:** car is parked at the hub node (taxi_node) at startup, roughly facing node 10's roadmap heading. Logs the derived offsets so they can be sanity-checked.

### trip_planner.py — fix stale `_path_completed_event` during WAIT states
**File:** `Development/ros2/src/qcar2_autonomy/autonomy/trip_planner.py`
**Lines:** 89-95 (`path_status_callback`)
**Change:** Only raise `_path_completed_event = True` on `/path_status` False→True transitions when `mission_stage` is one of {TO_PICKUP, TO_DROPOFF, TO_HUB}. Snap-completion events during WAIT_AT_* states are now ignored.
**Why:** Bug caused car to chain through paths without stopping. After arriving at pickup, the snap-to-exact path published a small correction. The snap completed within ~0.6s but trip_planner was waiting 3s. The snap completion raised the event again; that stale event then fired immediately when the dropoff path was published, making trip_planner think it had instantly arrived at dropoff. Same cascade for hub.

### Fix autonomy_planner_launch failures
**Files:**
- `Development/ros2/src/qcar2_autonomy/autonomy/nav_to_pose.py` line 116: `rotation_offset` default `[90]` → `[90.0]`.
- `Development/ros2/src/qcar2_autonomy/autonomy/trip_planner.py` line 46: same fix.
- `Development/ros2/src/qcar2_autonomy/setup.py`: removed `bev_csi_node` and `bev_csi_seg` console_scripts entries (source modules don't exist).
- `Development/ros2/src/qcar2_autonomy/launch/autonomy_planner_launch.py`: removed `bev_csi_node` and `bev_csi_seg` Node blocks and their entries in LaunchDescription list.
**Why:** `[90]` as integer list made ROS2 declare param as INTEGER_ARRAY; reading via `.double_array_value` returned empty list, then `rotation_offset[0]` raised IndexError when `path_planner` ran. The bev_csi nodes never had source files — they'd die immediately every launch with ModuleNotFoundError. Still required outside this scope: `pip install tqdm` inside dev container (fixes yolo_detector).

### trip_planner.py — manual edit by Gabriel
**File:** `Development/ros2/src/qcar2_autonomy/autonomy/trip_planner.py`
**Change:** `rotation_offset` default changed from `[82.0]` → `[90]` (line 46).
**By:** User (manual / linter, not Claude).

### trip_planner.py — skip startup drive when already at hub
**File:** `Development/ros2/src/qcar2_autonomy/autonomy/trip_planner.py`
**Lines:** ~254–272 (in `loop()`, startup branch)
**By:** Claude
**Change:** Before sending the startup "drive to hub" path, compute distance from current robot pose (converted via `_ros_to_qlabs`) to `hub_xy`. If under 0.30 m, mark `startup_done=True`, `ready_for_rides=True`, set LED MAGENTA (no-op since LED is already MAGENTA), log the skip, and return. Otherwise falls through to original send-path-to-hub behavior.
**Why:** When QCar starts physically at node 10 (the hub), `find_shortest_path(10, 10)` returns empty/single-point, so `/path_status` never flips True and startup hangs forever. The check shortcuts that case while preserving original behavior when starting elsewhere.
**LED logic:** Unchanged. Sequence is still MAGENTA → GREEN → BLUE → ORANGE → MAGENTA.
