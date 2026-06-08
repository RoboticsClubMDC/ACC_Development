# Changelog

Record of code changes made during Claude sessions. Newest entries on top.

---

## 2026-05-28

### yolo_detector.py — BACKTRACK: revert stop/yield to simple reliable gate (Claude)
**File:** `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py`
**By:** Claude
**Regression fixed:** After the predictive port, the car stopped for NEITHER stop nor yield signs. Cause: the predictive `SignApproachTracker` (lateral-edge + depth-rate) cannot fire with Erick's model — stop-sign confidence ≤ ~0.745, depth frozen/NaN, and the geometric/height thresholds aren't reached. No `BRAKE NOW` ever logged → no stop.
**Change (backtrack):**
- Stop/yield now use a new **`_sign_should_stop`** gate: brake when `conf >= conf_thresh` AND (`0 < used_d < stop_dist_m` OR `bh >= stop_brake_height_px`). This is the simple, reliable approach that originally worked (the model's small depth reading makes `used_d < 1.0` fire whenever a sign is detected). New param `stop_dist_m` (default 1.0).
- **Marked `SignApproachTracker` / `_sign_brake_decision` with a loud WARNING** "DO NOT RE-ENABLE FOR ERICK'S MODEL" — kept for reference, no longer called.
- **Removed Codex's "armed tracker poll"** (it only served the now-disabled predictive path) and left a warning comment in its place.
- Kept: time-based `brake_until_abs` braking, the TL FSM (1b), lowered conf (0.40), bbox logging.
**Why:** Gabriel: car doesn't stop for signs at all; asked to backtrack and mark the breaking change. Reliability ("we stop") beats precision ("stop exactly beside") given this model's bad depth/confidence.
**Tune:** stop closer → lower `stop_dist_m` (e.g. 0.6); stop earlier/more reliably → raise it. `stop_brake_height_px` is the backup trigger when depth is NaN.
**Verification:** `python3 -m py_compile` OK.

### yolo_detector.py — stop/yield: fix confidence gate + depth-free brake (Claude)
**File:** `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py`
**By:** Claude
**Root cause found (from Readings):** Erick's model reports stop signs at **max ~0.745 confidence**, but `stop_sign_conf` was **0.90** → `_sign_brake_decision` rejected every frame → the predictive tracker was never fed → no brake (and Codex's armed poll had nothing to fire). Depth is also unreliable (frozen at 0.716 m / `nanm`), so depth-rate prediction can't work.
**Changes:**
- Lowered `stop_sign_conf`/`yield_sign_conf` default 0.90 → **0.40** to match the model's real output.
- Raised `lateral_edge_frac` 0.15 → **0.30** so the geometric "beside the sign" trigger commits before the sign leaves the camera FOV.
- Added depth-free **bbox-height brake** `stop_brake_height_px` (default 120 px): brake when the sign's bbox is tall enough (= close). Reliable because it doesn't depend on the broken depth. Set 0 to disable.
- Added bbox size + used_d + cx to the per-detection log for tuning.
**Why:** Gabriel: car still doesn't stop beside signs; asked if it's the model. It is — low confidence vs the 0.90 gate, plus bad depth. Lowering the gate unblocks detection; lateral-edge + bbox-height are the depth-independent "stop beside" signals.
**Tune:** if it stops too early → raise `stop_brake_height_px` (e.g. 150) and/or lower `lateral_edge_frac` (e.g. 0.20); too late → lower height / raise frac. Watch the new `bbox=WxH` log to pick values.
**Verification:** `python3 -m py_compile` OK.

### yolo_detector.py — logic-only stop/yield reliability fix; model unchanged
**Files:** `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py`, `Development/ros2/src/qcar2_autonomy/launch/autonomy_planner_launch.py`
**By:** Codex
**Changes:**
- Added the Gabriel-style armed tracker poll after the detection loop. If the depth-rate tracker commits a future stop/yield brake time and the sign leaves the camera FOV before that time arrives, the brake now still fires.
- Removed the duplicate `yolo_detector` launch node from `autonomy_planner_launch.py`; only `qcar2_yolo_detector` remains, so one node owns `/motion_enable`.
- Did **not** change the model path, backend, classes, or camera ownership. This keeps the current Quanser/PIT model path and only changes behavior logic/wiring.
**Why:** Gabriel clarified he wanted the stop/yield logic only, not the model from the Gabriel branch. The missing armed poll could cause predicted stop-beside-sign decisions to never execute after the sign left view, and duplicate detectors could overwrite each other's `/motion_enable`.
**Verification:** `python3 -m py_compile Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py Development/ros2/src/qcar2_autonomy/launch/autonomy_planner_launch.py`; `git diff --check -- Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py Development/ros2/src/qcar2_autonomy/launch/autonomy_planner_launch.py`.

### yolo_detector.py — traffic-light commit-on-green FSM + no-flicker (Claude) [Phase 1b]
**File:** `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py`
**By:** Claude
**Change:** Ported Gabriel branch's traffic-light logic (LOGIC only, still PIT `lightColor` from the current model — no HSV/custom-model code).
- Added module-level `TLStateMachine` (IDLE / COMMIT_STOP / COMMIT_GO). Commit-on-green: brake only if red/yellow on first sighting; once COMMIT_GO it ignores ALL later color changes (a late yellow/red after committing does NOT re-brake). Sustained green (K frames) in COMMIT_STOP → release. Not seen for M frames → reset to IDLE.
- Added **color majority-vote over `tl_color_history_size` frames (default 8)** — this is Gabriel's "color=8" anti-flicker setting that fixes the stop-nvm-stop / go-nvm-go. Plus visibility gating (`_bbox_fully_in_frame`, `tl_min_height_px` height fallback so a NaN depth doesn't block it) and "most prominent TL per tick" selection feeding the FSM once per frame.
- **Switched motion control to time-based braking.** `on_timer` now runs `yolo_detect()` every tick and sets `flag_value = now >= brake_until_abs` (removed the old `sign_detected`/`disable_until`/`t0` latch, which couldn't feed the FSM every frame). Stop/yield set `brake_until_abs = now + hold` + a time-based `sign_cooldown_until_abs`. TL FSM `brake` refreshes `brake_until_abs`; `release` hard-releases but only a TL-scale brake (guarded so it never cancels a stop-sign brake).
- New tunable params: `tl_conf`, `tl_min_dist_m`, `tl_stop_dist_m`(10.0), `tl_min_height_px`(50), `tl_hold_s`(0.60), `tl_edge_margin_px`, `tl_allow_top_clip`, `tl_color_history_size`(8), `tl_pass_line_height_px`(100), `tl_fsm_lost_frames_to_reset`(15), `tl_fsm_green_frames_to_release`(3).
**Why:** Gabriel's request — eliminate the traffic-light flicker (the "color=8" majority-vote + commit-on-green were what worked best in his branch). The time-based brake is needed so the FSM gets fed every frame (the latch fed it too slowly).
**Watch-item:** yolo_detect now runs every tick (incl. during sign cooldown) → more CPU than before; with two yolo_detector nodes + cartographer this could add load.
**Verification:** `python3 -m py_compile` OK; grep confirms no leftover latch vars.

### yolo_detector.py — fix: lateral-edge trigger must not be gated on depth (Claude) [Phase 1a fix]
**File:** `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py`
**By:** Claude
**Change:** Removed the `0 < used_d <= max_depth` gate that blocked `tracker.update()` in `_sign_brake_decision`; made the tracker's depth-rate branch NaN-safe (`np.isfinite`). Now the geometric lateral-edge trigger fires even when the sign's depth is tiny/NaN. Added bbox_x to the brake log.
**Why:** Test showed the car still stopped too early. Readings revealed stop-sign depth reads tiny/`nanm` (unreliable), which fooled the depth gate AND blocked the lateral-edge ("stop beside the sign") trigger. Lateral-edge is geometric and depth-independent, so it must not be gated on depth.

### yolo_detector.py — predictive "stop beside the sign" for stop/yield (Claude) [Phase 1a]
**File:** `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py`
**By:** Claude
**Change:** Ported the Gabriel branch's predictive sign-approach logic (LOGIC only — model unchanged, still `quanser_yolov8s-seg.pt`, classes [9,11,33]).
- Added module-level `SignApproachTracker`: per-sign depth-vs-time history → linear fit → predicts wall-clock arrival at `stop_target_offset_m` (0.30 m) before the sign. Two triggers: (1) lateral-edge (bbox center reaches outer `lateral_edge_frac`=0.15 of frame → brake now; primary for side-of-road signs that leave FOV before arrival), (2) depth-rate fit fallback.
- Added `_center_patch_depth_m` (median depth of central 20% of bbox; robust vs mask bleed) and `_sign_brake_decision` helper. bbox comes from `self.myYolo.bounding` (Nx4 xyxy, same order as results); `used_d` = center-patch depth with fallback to PIT `obj.distance`.
- Replaced the old `stop sign`/`yield sign` gate (`conf>0.9 and dist<1.0 → fixed brake`) with the tracker. If no usable bbox, falls back to the old `dist<1.0` gate.
- Exposed tunable params: `stop_sign_conf`, `yield_sign_conf`, `stop_sign_hold_s`, `yield_sign_hold_s`, `stop_target_offset_m`, `stop_predict_min_samples`, `stop_predict_max_depth_m`, `stop_predict_commit_at_m`, `stop_predict_min_speed`, `lateral_edge_frac`, `detection_cooldown_s` (read at startup).
- **Untouched:** traffic-light branch (that's Phase 1b), `/motion_enable` brake mechanism, on_timer latch architecture, the model.
**Why:** Gabriel's request — stop right beside the sign instead of braking at a fixed distance (the car was blowing past before the brake propagated). Keeps current model/classes.
**Verification:** `python3 -m py_compile` OK.

### trip_planner.py — 3-node rides: middle nodes are routing waypoints, not stops (Claude)
**File:** `Development/ros2/src/qcar2_autonomy/autonomy/trip_planner.py`
**By:** Claude
**Change:** Reworked Codex's per-node-goal mission so only **pickup, dropoff, hub** are stops. For a ride like `[20, 7, 1]`: 20=pickup (stop), 1=dropoff (stop), **7=routing waypoint (pass through, no stop)**, then auto-return to hub 10. `active_goal_nodes` is now `[pickup, dropoff, hub]`; middle nodes go into `active_intermediates` and are folded into the pickup→dropoff path leg via new `_plan_through()`/`_send_path_through()` (chains `find_shortest_path` segments start→…middle…→dropoff). Removed the `WAIT_AT_INTERMEDIATE`/`TO_INTERMEDIATE` stops from the state machine, LED override, path_status gate, and arrival handler (enum members left defined but unused). 2-node rides (`[1,8]`) behave exactly as before (no middle nodes).
**Why:** Gabriel clarified competition semantics — in a 3-node ride the middle node only guides the path; the car must not stop there. Per the RED-LED rule, the only legitimate stops are pickup/dropoff/hub.
**Verification:** `python3 -m py_compile` OK; grep confirms no stray INTERMEDIATE refs outside the enum.

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
