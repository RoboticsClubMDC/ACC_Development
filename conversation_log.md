# Conversation Log

Record of important prompts and answers during Claude sessions. Skip trivial exchanges. Newest on top.

---

## 2026-05-28 — First successful trip + RED LED + review of Codex's changes (Claude)
**Gabriel reported:** Cartographer still fails repeatedly to start (same CPU/RViz load issue) but eventually worked; autonomy then ran and the car drove the whole trip and **stopped correctly at pickup/dropoff/hub** (major improvement). Car wanders out of lane (separate problem). Auto-align validated: log shows `rotation_offset=41.89 deg, translation_offset=[1.208, 0.534]` — confirms the sign convention; Gabriel's manual 90 was wrong, ~42 is right. YOLO is running and detecting stop signs.

**New requirement implemented (RED LED):** any stop NOT at pickup/dropoff/hub (stop signs, traffic lights) must be RED. Implemented in trip_planner via `/motion_enable` subscription + LED override during driving stages. See changelog.

**Objective review of Codex's changes (Gabriel asked for honest criticism):**
- *trip_planner.py (`trip_nodes` + intermediate stops + auto-append hub):* KEEP. This is genuinely better than Claude's 2-node pickup/dropoff design because the competition ride list has 3-node rides (e.g. "L: 20, 7, 1", "M: 4, 1, 8"). Codex preserved Claude's `_auto_align` and the stale-`_path_completed_event` fix, added node validation, ready/idle gating, and auto-appends node 10 after every ride. Correct and an improvement.
- *qcar2_cartographer_virtual_launch.py (`enable_drive_converter`, default false):* MIXED. Intent (stop a stale path_follower from driving the car during mapping) is reasonable, BUT (1) it does NOT fix the cartographer "messed up" problem — that's the CPU/RViz-load issue (close RViz / its Submaps display, restart dev container), unrelated to the converter; and (2) it's a footgun: the converter (`/cmd_vel_nav` → `qcar2_motor_speed_cmd`) now defaults OFF in the cartographer launch, so for an actual autonomy drive you must pass `enable_drive_converter:=true` or the car won't move. Cleaner design: move `nav2_qcar2_converter` into `autonomy_planner_launch.py` (where `/cmd_vel_nav` is produced) and drop the gate — then mapping is converter-free (safe) and autonomy always has it (no flag). Recommended but NOT done yet pending Gabriel's OK.

**Run sequence (current, with Codex's converter gate in place):**
1. Dev container, build: `colcon build && . install/setup.bash` (after any code change).
2. Cartographer + DRIVE for an autonomy run: `ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py enable_drive_converter:=true` — the `:=true` is REQUIRED for the car to move (Codex's gate). Wait until cartographer is stable (TF `map→odom` present). For MAPPING-only (no driving), omit the flag.
3. Autonomy: `ros2 launch qcar2_autonomy autonomy_planner_launch.py` (path_follower + 2 yolo_detector + trip_planner). Watch for `Auto-aligned at hub node 10: ...` and `Already at HUB ... Ready for rides.`
4. Dispatch a ride (no defaults): `ros2 param set /trip_planner trip_nodes "[1, 8]"` (pickup 1 → dropoff 8 → auto-return to hub 10). 3-node ride example: `"[20, 7, 1]"`. Ride only accepted when parked at hub and idle.
5. LEDs: MAGENTA idle/hub → GREEN to pickup → BLUE pickup wait + to dropoff → ORANGE dropoff wait + to hub → RED whenever halted by a stop sign/traffic light mid-leg.

## 2026-05-28 — Process correction: log code changes and avoid unrequested edits
**Gabriel said:** From now on, if Codex makes any changes for the first task, record them in `changelog.md` and `conversation_log.md` following the existing format. Gabriel also objected that `trip_planner.py` was changed without clear permission.

**Codex response:** Acknowledged. Going forward, code behavior changes should be explicit-only, and any code changes made must be recorded in both logs. For this response, Codex only updated the logs and did not change runtime behavior.

## 2026-05-28 — First-task trip planner work + cartographer motion diagnosis
**Context:** Gabriel wanted the first-task work to keep Gabriel's LED/Yolo/launch behavior while moving trip commands to node IDs and returning to node 10 after rides. Later, Gabriel reported that running cartographer caused the QCar to move by itself and showed Cartographer warnings/errors.

**Trip planner changes applied by Codex:** Added `trip_nodes` as a node-list ride command, preserved `pickup_node`/`dropoff_node` compatibility, validated node IDs and idle/ready state before dispatch, added intermediate-stop handling, and made every ride append return-to-hub node 10 unless already requested. Also fixed startup/path completion edge cases for node 10.

**Cartographer investigation:** Process inspection showed `autonomy_planner_launch.py` was still running at the same time as cartographer, including `path_follower`, `trip_planner`, and two YOLO nodes. `qcar2_cartographer_virtual_launch.py` also launched `nav2_qcar2_converter`, and `nav2_qcar_command_convert.cpp` forwards `/cmd_vel_nav` to `qcar2_motor_speed_cmd`. Therefore, a stale/running `path_follower` could command throttle while Gabriel believed only cartographer was active.

**Cartographer launch change applied by Codex:** Added `enable_drive_converter:=false` default to `qcar2_cartographer_virtual_launch.py`, so mapping no longer starts the motor-command converter unless explicitly requested with `enable_drive_converter:=true`.

**Log interpretation given:** `Requested submap ... does not exist` is likely RViz querying Cartographer submaps before they exist or after stale IDs; `Dropped earlier points` / `Ignored subdivision` warnings point to lidar timestamp jitter and CPU load, made worse by QLabs, RViz, Cartographer, autonomy, and YOLO running together.

## 2026-05-27 — Localization plan + Phase 0 (node-based trip_planner + auto-align)
**Context:** Car drives but crashes into walls / can't reach targets. Gabriel wants to rebuild localization like luigi-5 (pose_estimator + ekf_fusor + AMCL on a recorded map), keep node-based trip_planner + LED logic, and keep YOLO working (luigi-5's main failure was YOLO not running; their lane detection also weak — we use our own `yolo_detector`, tqdm already fixed). Agreed to implement incrementally with test runs between phases.

**Investigation of luigi-5 stack:** pose_estimator (wheel+IMU bicycle EKF → `/odom` + `odom→base_link` TF); ekf_fusor (fuses odom with correction source tf/amcl_pose/landmark → `/qcar2_pose_fused`); mapping = online cartographer, localization = AMCL+map_server on saved `.pgm/.yaml`; nav_to_pose still does `lookup_transform('map','base_link')` (same as ours). Key: adding pose_estimator requires switching `qcar2_2d.lua` to `provide_odom_frame=false, published_frame=odom, use_odometry=true` so cartographer stops publishing `odom→base_link`.

**Phased plan:** Phase 0 = auto-derive offsets (done); Phase 1 = port pose_estimator + ekf_fusor + estimation filters + lua config switch; Phase 2 = AMCL + map recording workflow; Phase 3 = integrate + Run A test.

**TF concept clarified:** map/odom/map_rotated being fixed in RViz while base_link/base_scan move IS correct ROS behavior — not the bug.

**Phase 0 implemented (this session):** switched trip_planner to node-based destinations (`pickup_node`/`dropoff_node`) and added `_auto_align()` that derives rotation+translation offset from node 10's roadmap pose vs measured map pose. See changelog for details. Confirmed `get_node_pose(node)` returns `[x,y,theta]`. Awaiting Gabriel's test run + RViz/QLabs observations before Phase 1.

## 2026-05-27 — Car drives Run A path but never stops at pickup/dropoff/hub
**Gabriel reported:** After fixes from prior session, autonomy_planner launched and the car moved using Run A coordinates, but never paused at pickup, dropoff, or returned to hub.

**Root cause:** Stale `_path_completed_event` bleeding across mission stages. After real arrival at pickup, `_snap_to_exact` publishes a tiny correction path. That snap completes during the WAIT_AT_PICKUP 3s pause and raises `_path_completed_event=True` again. The event sits unconsumed until WAIT→TO_DROPOFF transition publishes the dropoff path; the very next loop tick consumes the stale event and fires the TO_DROPOFF arrival branch immediately. Cascade repeats for hub. Net effect: car never actually stops, chains through paths.

**Fix applied (`trip_planner.py:89-95`):** In `path_status_callback`, only raise `_path_completed_event` when `mission_stage in {TO_PICKUP, TO_DROPOFF, TO_HUB}`. Snap completions during WAIT_AT_* states are now ignored.

## 2026-05-27 — Autonomy planner crashes on launch
**Gabriel reported:** Launched cartographer_virtual then autonomy_planner, several processes died.

**Root causes + fixes Claude applied:**
1. `path_follower` crashed with `IndexError` on `self.rotation_offset[0]`. Cause: `nav_to_pose.py:116` declared `rotation_offset` default as `[90]` (integer list) — ROS2 typed it INTEGER_ARRAY, but line 117 read it via `.double_array_value` which returned `[]`. Fixed: `[90]` → `[90.0]`. Same bug in `trip_planner.py:46`, same fix.
2. `bev_csi_node` and `bev_csi_seg` died with `ModuleNotFoundError` — those source files don't exist in `autonomy/` folder. Removed their entries from `setup.py` and `autonomy_planner_launch.py`.
3. `yolo_detector` died with `ModuleNotFoundError: tqdm`. Fix: `pip install tqdm` inside dev container (must be done by user).

**Run order given:** Terminal 1 = simulator/QLabs; Terminal 2 = dev container + cartographer launch (with one-time `pip install tqdm` and `colcon build`); Terminal 3 = dev container + autonomy_planner launch; Terminal 4 = trigger ride via atomic `/trip_planner/set_parameters_atomically` service call.

## 2026-05-27 — Session 1 (Gabriel_main_testing branch)

### Q11 — Give coordinates to test Run A from Competition_Ride_List.txt, starting at node 10
**Gabriel asked:** Provide XY coordinates for a quick test using Run A from `Competition_Ride_List.txt`, starting at node 10.

**Claude answered:**
- Run A is pickup node 1 → dropoff node 8 (per Competition_Ride_List.txt:37).
  - Node 1: `[0.269, -0.049, 90°]`
  - Node 8: `[-0.749, 1.077, 180°]`
  - Hub/start at node 10: `[-1.282, -0.59, -42°]`
- For trip_planner:
  - `pickup_xy = [0.269, -0.049]`
  - `dropoff_xy = [-0.749, 1.077]`
- Total drive ~4.5 m round trip; suitable as first test.
- Flagged race condition: parameter callback triggers `new_ride_requested=True` on EITHER pickup or dropoff being set. If set sequentially via two `ros2 param set` commands, the 10 Hz loop could fire between them and use the stale default for the unset one. Three options offered:
  1. Atomic via service call (`/trip_planner/set_parameters_atomically`).
  2. Sequential `ros2 param set` with `dropoff_xy` set FIRST then `pickup_xy` — the second triggers the ride with both values correct (sub-100ms race window in practice).
- Expected LED + log sequence walkthrough: MAGENTA on launch → GREEN drive-to-pickup → BLUE arrive-pickup (3s wait) → BLUE drive-to-dropoff → ORANGE arrive-dropoff (3s wait) → ORANGE drive-to-hub → MAGENTA mission complete.
- Diagnostic hints if it doesn't move: check `Path published` log, then `motion_flag` (controlled by `/motion_enable` from object detector).

### Q10 — Does this trip_planner + path_follower work by node IDs or by XY coordinates?
**Gabriel asked:** Are missions specified by node IDs (old Gabriel-branch style) or by XY coordinates in the current Gabriel_main_testing trip_planner + path_follower combo?

**Claude answered:**
- Hybrid. User-facing interface is XY, but internally trip_planner still routes through the road-map node graph.
- trip_planner inputs (XY only): `pickup_xy`, `dropoff_xy`. Only node-ID input is `taxi_node` (the hub).
- trip_planner internal flow (`_plan_to_xy`, trip_planner.py:183-203):
  1. Convert robot's ROS pose → QLabs frame.
  2. `_closest_node` snaps both start and goal XY to nearest road-map nodes.
  3. `roadmap.find_shortest_path(start_node, goal_node)` → waypoints along graph edges.
  4. Append exact `goal_xy` as final point (precise arrival, not just nearest-node).
  5. `_qlabs_path_to_ros` converts to ROS frame, publishes as `nav_msgs/Path` on `/cmd_waypoints`.
- path_follower `_cmd_waypoints_cb` (nav_to_pose.py:212-224) receives explicit XY waypoints in ROS frame, stores as `self.wp`, sets `_wp_in_ros_frame=True` (so the 83° QLabs→ROS rotation is skipped — already in ROS frame).
- path_follower supports BOTH modes:
  1. Node IDs via `node_values` parameter (old Gabriel style) — calls `SDCSRoadMap().generate_path(node_values)` internally; waypoints in QLabs frame, rotation applied.
  2. XY via `/cmd_waypoints` topic (new, used by trip_planner) — waypoints in ROS frame, rotation skipped.

### Q9 — Confirm QCar won't move when autonomy_planner is launched (assuming start at hub)
**Gabriel asked:** RViz confirms the rotation offset looks aligned (will drift but that's for later). Wants to confirm that running autonomy_planner right now will NOT cause the QCar to move.

**Claude answered:**
- Confirmed: yes, the car will sit still.
- Motion gate at `nav_to_pose.py:470-474`:
  ```
  enable = path_execute_flag AND motion_flag AND NOT path_complete
  ```
  All three must be True for non-zero `Twist` to be published.
- At fresh launch:
  - `path_execute_flag` = False (from `start_path` param, default `[False]`; launch file doesn't override).
  - `motion_flag` = True (line 185, defaults True).
  - So `enable = 0`. `/cmd_vel_nav` still published but with zeros — car stays put.
- `path_execute_flag` only flips True when:
  1. A `/cmd_waypoints` message arrives (`_cmd_waypoints_cb` line 222), OR
  2. `start_path` parameter is set to True.
- With the trip_planner skip-startup-drive edit, when at the hub no `/cmd_waypoints` gets published until a mission is requested.
- So unless Gabriel manually sets `start_path=true`, updates `pickup_xy`/`dropoff_xy`, or publishes to `/cmd_waypoints` directly, the car will not move.

### Q8 — Diagnose cartographer virtual errors + check for stale background nodes
**Gabriel asked:** Look at `Readings.txt` (terminal output from running cartographer virtual). Diagnose why errors appear for him but not for his teammate running the same code. Check if anything from prior runs is still running in the background.

**Claude answered:**
- No stale ROS nodes from previous runs. Previous launch terminated cleanly (the SIGKILL escalation at shutdown is normal for cartographer). `ros2-daemon` is expected.
- Root cause is **CPU starvation**, not a code bug. Three heavy processes competing: `rviz2` (40% CPU), `QCar2_Workspace_studio` (30%), `QLabs` (21%), plus two Docker containers.
- Direct evidence in the log: cartographer scan rate degrades from 10.44 Hz / 88.42% real time at startup → 3.59 Hz / 59.67% real time one minute later (lines 155 vs 1178 of Readings.txt).
- All error categories are downstream symptoms:
  - `tf_bridge.cpp:53 Lookup would require extrapolation into the future` — `fixed_lidar_frame_virtual` falling behind lidar scan timestamps due to CPU jitter.
  - `range_data_collator.cc:82 Dropped N earlier points` — late points discarded.
  - `sensor_bridge.cpp:211 Ignored subdivision of a LaserScan` — duplicate/out-of-order scans.
  - `map_builder_bridge.cpp:180 Requested submap N ... maybe it has been trimmed` (every ~100ms) — RViz polling `/submap_query` faster than cartographer can produce submaps; self-reinforcing because the spam itself eats CPU.
  - `TF_OLD_DATA ... authority Authority undetectable` — TF publisher temporarily unresponsive due to context switches.
- Teammate doesn't see errors because their machine has more headroom (different specs, fewer concurrent tools, likely no RViz running alongside).
- **Fixes in order of impact:**
  1. Close RViz while testing cartographer (kills the biggest CPU consumer AND stops the `/submap_query` storm).
  2. If RViz needed, disable the "Submaps" display specifically.
  3. `docker restart isaac_ros_dev-x86_64-container` — dev container has been up 7 hours.
  4. Check CPU governor: `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` — switch to `performance` if it's `powersave`.

### Q7 — What happens when autonomy_planner_launch is run from node 10?
**Gabriel asked:** Walk through what runs when he launches `autonomy_planner_launch.py` assuming initial position at node 10. Also flagged that he doesn't think his Gabriel branch had the `self.declare_parameter` calls in trip_planner.

**Claude answered:**
- Confirmed: Gabriel branch DID have `declare_parameter` for `taxi_node` and `trip_nodes`. What's new in Gabriel_main_testing is the **shift from node-IDs to XY coordinates**: `pickup_xy`, `dropoff_xy`, `stop_seconds`, `rotation_offset`, `translation_offset`. The new trip planner snaps XY to nearest road-map node for planning.
- Walkthrough of `autonomy_planner_launch.py`:
  - Launches: `path_follower` (nav_to_pose.py), `traffic_system_detector` (yolo_detector.py), `trip_planner`, and `Planner_server`.
  - **`Planner_server` doesn't exist in setup.py console_scripts** — launch logs an error and continues. Cosmetic for trip flow but real if intended functionality was expected.
- trip_planner init: connects LED service, declares params, computes `hub_xy = node 10 coords`, sets LED MAGENTA, subscribes to `/path_status` + `/robot_pose`, publishes `/cmd_waypoints`, starts 10 Hz loop.
- path_follower init runs in parallel; subscribes to `/cmd_waypoints` (sets `_wp_in_ros_frame=True` when external waypoints arrive); publishes `/robot_pose`.
- Loop tick — startup branch:
  - With Claude's recent edit: detects `dist_to_hub < 0.30 m` → skips startup drive, transitions to `ready_for_rides=True`, LED stays MAGENTA, logs "Already at HUB ... Skipping startup drive."
  - Without the edit: would send `find_shortest_path(10,10)` (empty path) and hang waiting for `/path_status`.
- Idle state: loop early-returns each tick. Car still, perception running, no waypoints sent.
- To trigger ride: `ros2 param set /trip_planner pickup_xy [<x>, <y>]` → parameter callback sets `new_ride_requested=True`.
- Mission state machine then runs through the LED transitions: TO_PICKUP (GREEN) → WAIT_AT_PICKUP (BLUE) → TO_DROPOFF (BLUE) → WAIT_AT_DROPOFF (ORANGE) → TO_HUB (ORANGE) → back to idle (MAGENTA).
- Caveats: the "skip startup" depends on `_ros_to_qlabs` (which uses `rotation_offset` + `translation_offset`); if those are wrong for physical, the comparison is wrong and skip never fires. `Planner_server` failure is logged but doesn't break the trip.

### Q6 — Edit trip_planner so QCar starts at node 10 (skip drive-to-hub), keep LED logic intact
**Gabriel asked:** Modify trip_planner.py to assume the QCar's initial position is node 10 instead of starting elsewhere and driving there. Preserve LED logic.

**Claude answered + Action:**
- Diagnosis: existing code already has `taxi_node=10` so node 10 IS the hub. Bug: when start ≈ goal, `find_shortest_path(10,10)` returns empty/single-point, `/path_status` never flips, startup hangs forever and `ready_for_rides` stays False.
- Edit applied at `trip_planner.py:254-272`: before sending the startup drive path, compute `dist_to_hub`. If under 0.30 m, mark startup done, set `ready_for_rides=True`, LED MAGENTA (no-op, already MAGENTA), log skip, return. Otherwise falls through to original drive-to-hub.
- LED chain unchanged (was MAGENTA → MAGENTA anyway across startup).
- Knobs to tweak: 0.30 m threshold (raise if cartographer initial noise on physical is bigger, lower if want strict); confirm `rotation_offset` and `translation_offset` are right for physical, otherwise `_ros_to_qlabs(robot_pose)` won't match `hub_xy`.

### Q5 — What does the `visualize_pose` parameter do?
**Gabriel asked:** What's `visualize_pose` for?

**Claude answered:**
- Debug-plotting flag. Opens a 4-axis live scope window via Quanser's `pal.utilities.scope.MultiScope`.
- Plotted (Gabriel_main_testing version, `nav_to_pose.py:285-304`):
  - Row 0: measured x vs EKF-estimated x
  - Row 1: measured y vs EKF-estimated y
  - Row 2: pp_delta, stanley_delta, blended_delta, blend_alpha (testing branch adds Stanley blend signals; Gabriel branch had just `delta`)
  - Row 3: measured yaw vs EKF yaw
- Plot work runs in `scopeDataTimer()` at `nav_to_pose.py:526-557`, which samples each axis and calls `MultiScope.refreshAll()`.
- Pure visualization — no control effect.
- **Warning for physical:** default is `True` in Gabriel_main_testing (was `False` in Gabriel branch). MultiScope is Qt/pyqtgraph — needs X display, holds GIL during redraws, competes for CPU with control loop. Over SSH or under load it can lag the planner timer. Suggested running physical with `visualize_pose:=false`.

### Q4 — Compare nav_to_pose.py between Gabriel and Gabriel_main_testing branches
**Gabriel asked:** Diff the two branches' nav_to_pose.py and explain what differs.

**Claude answered:**
- Sizes: Gabriel 712 lines, Gabriel_main_testing 571. Testing is shorter despite adding features — old verbose code got trimmed/refactored.
- Defaults that changed:
  - `rotation_offset`: 86.0° → 83.0° (3° gap; suspicious for physical drift accumulation)
  - `desired_speed`: 0.2 → 0.4 (2× faster)
  - `visualize_pose`: False → True
  - `start_path`: False → True
- New in Gabriel_main_testing (Gabriel had none of this):
  - Subscriptions: `/cmd_waypoints`, `/nav/stanley_delta`, `/nav/stanley_trust`
  - Publishers: `/nav/pp_delta`, `/nav/stanley_delta`, `/nav/blended_delta`, `/nav/blend_alpha`
  - `_blend_steering()` mixing pure-pursuit with Stanley using `alpha = stanley_blend * stanley_trust` (falls back to PP when trust < threshold)
  - `_wp_in_ros_frame` flag: when external waypoints arrive via `/cmd_waypoints`, the 83° QLabs↔ROS rotation is skipped (external waypoints presumed in ROS frame)
  - Mission params: `mission_pickup_xy`, `mission_dropoff_xy`, `mission_enable`
- Removed/simplified in testing: commented-out `map_rotated` fossils, different waypoint loop bounds (testing executes more of the path before completion), PEP-8 4-space indent.
- Most likely culprits for physical performance gap: (1) Stanley blend polluting steering if a Stanley publisher is active on physical with bad data — easy test: set `stanley_blend:=0.0`. (2) 83° vs 86° rotation_offset.

### Q3 — Compare other branches (Arturo, luigi-5, Physical_Arturo) — do they avoid using map_rotated?
**Gabriel asked:** Look at Arturo and Luigi-5 branches — Gabriel believes they use `map` but apply their own rotation in code to avoid `map_rotated`. Does Gabriel_main_testing do that?

**Claude answered:**
- **Arturo branch:** Same static publisher as Gabriel_main_testing. Hardcoded `frame_id = "map"`. `rotation_offset = 82°` default. NO second 180° fix. Just cleaner refactored helpers (`_R_qlabs_to_ros_2d` etc.) doing the same 82° math. Arturo is in the same "broken for physical" state as Gabriel_main_testing.
- **luigi-5 branch:** REMOVED the static_transform_publisher entirely (no map_rotated). `frame_id = 'map'`, `rotation_offset = 83°` default. CRITICAL: adds **runtime auto-alignment** at `nav_to_pose.py:670-700` that overrides `rotation_offset`:
  - `angle_rad = atan2(sin(current_yaw - raw_heading), cos(current_yaw - raw_heading))`
  - `self.rotation_offset = [float(angle_deg)]`
  - This measures the angle between the robot's actual yaw and the recorded path's initial heading, absorbs it into rotation_offset. On physical (180°-flipped map), auto-derives ~82+180 = 262°.
- **Physical_Arturo branch:** Keeps static publisher. Adds `route_frame` parameter (default `'map'`) and `rotation_offset = 0.0` default. Lets you launch with `route_frame:=map_rotated, rotation_offset:=0` for physical.
- **Gabriel_main_testing has none of these compensations.** No auto-align, no route_frame parameter, hardcoded `'map'`, fixed 83° rotation_offset.
- Three fix options offered: port luigi-5's auto-align block; port Physical_Arturo's route_frame param; or swap the static publisher's parent/child to flip which frame is "upright".

### Q2 — Ran with map_rotated on physical, no change in behavior
**Gabriel reported:** Tried running physical with `map_rotated` and it was literally the same as before.

**Claude answered:**
- Confirmed expected. Switching the frame name alone is a no-op for autonomy behavior because:
  - Path publish stamps the chosen frame; TF lookup uses the same frame; pure-pursuit operates on relative geometry between robot pose and waypoints. If both publish and lookup are in `map`, OR both in `map_rotated`, the relative geometry is identical.
  - The frame name only matters for things OUTSIDE the autonomy (RViz visualization, other nodes consuming the topic).
- The real differentiator physical vs virtual is `rotation_offset` itself (82°/83° tuned for virtual map orientation). On physical, cartographer's map orientation depends on robot starting pose + lidar mount (`fixed_lidar_frame.cpp` -π for physical vs 0 for virtual) + IMU drift.
- Recommended diagnostics: log `current_yaw - raw_heading` at path execution start; try `rotation_offset := 262.0` (82+180) or `-98.0` (82-180) manually to confirm it's a 180° offset; port luigi-5's auto-align as the real fix.

### Q1 — Investigate map vs map_rotated; is anyone doing a second 180° rotation?
**Gabriel asked:** Gabriel_main_testing performs well in virtual but poorly on physical. Code uses `map` (not `map_rotated`). The `qcar2_cartographer_launch.py` has a 180° static transform `map_rotated → map`. He suspects another developer ("Arturo") may have added a second 180° rotation somewhere in code, which would mean switching to `map_rotated` would double-rotate. Asked Claude to verify across the codebase.

**Claude answered:**
- Searched all of `Development/ros2/src/`. Findings:
  - `map_rotated` appears in only ONE place: the static transform publisher at `qcar2_cartographer_launch.py:38`. Nothing else publishes or subscribes to it.
  - Autonomy code (`nav_to_pose.py`, `trip_planner.py`) uses hardcoded `frame_id = 'map'` and TF lookups from `'map'`.
  - Only π-magnitude rotations in TF:
    1. `map_rotated → map` static publisher (the one in question).
    2. `base_link → base_scan` at `fixed_lidar_frame.cpp:37` — `q.setRPY(0,0,-π)` — lidar mount correction (physical car only). Virtual sets the rotation to 0.
  - `rotation_offset` parameter in autonomy: ~82-83° (NOT 180°). Used only when `_wp_in_ros_frame` is False; paired with `translation_offset`. It's a QLabs→ROS origin/heading conversion, not a 180° flip.
  - No launch file or YAML overrides `rotation_offset` to anything close to 180°.
- TF tree on physical: `map_rotated → map (yaw=π) → odom → base_link → base_scan (yaw=-π)`.
- Conclusion: NO hidden 180° compensation. Autonomy operating in 180°-flipped `map` frame with no fix on physical — matches the poor-on-physical symptom.
- Three fix options offered: port luigi-5's auto-align; port Physical_Arturo's `route_frame` param; or swap the static publisher's parent/child.
