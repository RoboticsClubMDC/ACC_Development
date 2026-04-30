# VO Change Log

This file tracks cleanup, calibration decisions, and test observations for
the QCar2 visual odometry work.

## 2026-03-27

Scope of this pass:
- Cleanup only.
- No executable VO logic changed.
- No tuning or calibration constants were changed in code.

Changes made:
- Created this changelog file.
- Deleted `autonomy/qlabs_vo_baseline.yaml`.
- Removed misleading or stale comment/docstring text from:
  - `autonomy/visual_odometry.py`
  - `autonomy/vo_node.py`

Why `qlabs_vo_baseline.yaml` was removed:
- It contained stale VO-specific overrides from an older tuning pass.
- It mixed guessed values and deprecated assumptions such as the old `fx=161`
  path and older depth-scaling notes.
- Keeping it in the active tree risks reintroducing bad launch parameters.

Comment cleanup goals:
- Keep comments tied to explicit calibration sources.
- Remove comments that presented guessed values as if they were authoritative.
- Remove stale references to non-active file names and outdated run commands.

Files intentionally kept in place during this pass:
- `autonomy/visual_odometry_old.py`
- `autonomy/vo_node_old.py`
- `autonomy/vo_terminal_dashboard.py`
- Other VO helper utilities pending later review

Current source-of-truth documents for discussion:
- `Virtual_Stage_ROS_FAQ.md`
- `Camera Intrinsics Post.txt`
- `user_manual_system_hardware.pdf`

Open follow-up items:
- Verify the virtual `MONO16` depth scale against live runtime data.
- Decide whether the current virtual depth-to-color alignment homography
  remains acceptable or should be replaced.
- Replace placeholder physical depth/alignment assumptions with measured values.

## 2026-03-27

Scope of this pass:
- Add a standalone depth probe for `video3d_frame_get_meters(...)`.
- Keep the active `rgbd` node and active VO path unchanged.

Changes made:
- Added `qcar2_nodes/src/rgbd_get_meters_probe.cpp`.
- Added build/install entries for the new `rgbd_get_meters_probe` executable in
  `qcar2_nodes/CMakeLists.txt`.

Why this probe was added:
- The main unresolved question in the current VO path is whether the ROS2
  `MONO16` depth stream should keep using the empirical `15707.0` divisor or
  whether Quanser's depth API can provide authoritative metric depth directly.
- The Quanser C API explicitly exposes `video3d_frame_get_meters(...)` for
  depth streams, and older Quanser ROS examples already used it.
- A separate probe lets us test raw depth and metric depth on the same frame
  without changing the active `rgbd.cpp`, `visual_odometry.py`, or `vo_node.py`
  behavior.

What the new probe does:
- Opens only the depth stream, not the full RGBD publish path.
- Calls `video3d_frame_get_data(...)` and `video3d_frame_get_meters(...)` on
  the same depth frame.
- Publishes:
  - `camera/depth_image_probe_raw` as `MONO16`
  - `camera/depth_image_probe_meters` as `32FC1`
  - `camera/depth_probe_status` as a diagnostic string
- Logs a periodic comparison between:
  - raw uint16 depth
  - API meters from `get_meters`
  - the current virtual estimate derived from `15707.0 * 0.1 m/unit`

Important testing note:
- The probe opens the depth camera directly. For the cleanest test, it should
  not be run at the same time as the normal `rgbd` node unless the backend is
  confirmed to support concurrent opens.

Environment/package note:
- A package inventory spot-check on this machine found `librealsense2`
  installed.
- No Isaac ROS Visual SLAM apt package was found at this point.

## 2026-03-27

Scope of this pass:
- Runtime geometry fix for physical vs. virtual VO mode selection.
- No changes yet to `rgbd.cpp` depth publication format.

Changes made:
- Updated `autonomy/visual_odometry.py` so `camera_mode` now selects the full
  calibration stack instead of only switching RGB intrinsics.
- Updated `autonomy/vo_node.py` so physical runs no longer inherit the virtual
  `depth_scale=15707` by default.
- Added mode-aware depth diagnostics so physical mode no longer reports
  virtual-only "QLabs units" conversions.

Why these changes were needed:
- The active code had the correct virtual and physical calibration tables
  declared, but the constructor fallback path still defaulted depth scale,
  alignment, and camera-to-body extrinsics to the virtual settings.
- As a result, `camera_mode='physical'` was only partially physical.
- `vo_node.py` also forced `depth_scale=15707.0` into every run, which made
  the physical mode selection ineffective even before VO processing started.

Details of the fix:
- `visual_odometry.py`
  now builds a single mode-specific defaults table for:
  RGB intrinsics, depth intrinsics, depth scale, alignment matrix,
  camera-to-body extrinsics, depth unit label, and depth validity gates.
- `visual_odometry.py`
  now keeps raw depth intrinsics available and uses them when alignment is
  disabled, instead of always backprojecting with RGB intrinsics.
- `vo_node.py`
  now treats `depth_scale <= 0` as "use the mode default from
  visual_odometry.py".
- `vo_node.py`
  now uses `alignment_mode`:
  `auto` => virtual uses alignment, physical disables it by default until a
  validated physical RGB-depth registration is available.

Important remaining limitation:
- Physical mode is now protected from inheriting virtual-only parameters, but
  physical RGB-depth alignment is still not solved in the active ROS2 path.
- The current physical default keeps alignment off intentionally because the
  placeholder identity matrix is not a validated registration.
- Publishing metric depth directly from `rgbd.cpp` via `get_meters` remains the
  preferred follow-up change.

## 2026-03-27

Scope of this pass:
- Minimal cleanup only for the next manual VO run workflow.

Changes made:
- Kept the `rgbd.cpp` cleanup that removed malformed pasted diff text from the
  stream-open error branch.
- Reverted the temporary launch-file additions from the previous pass because
  the preferred workflow is still manual terminals for Cartographer, `vo_node`,
  and autonomy nodes.

Why this was kept:
- `rgbd.cpp` had stray pasted text in active source, which could break a rebuild
  of `qcar2_nodes`.

Why the launch changes were reverted:
- The user does not want a separate VO test launch file at this stage.
- The user is not using `qcar2_manual_drive_launch.py` in the active workflow.
- The current preferred workflow remains:
  - scenario / QLabs setup
  - Cartographer launch
  - `ros2 run qcar2_autonomy vo_node`
  - either `path_follower` or `autonomy_planner_launch.py`
  - a separate terminal tool for viewing `/vo/fault_status`

## 2026-04-30

Scope of this pass:
- Physical camera pipeline validation only.
- No motion/pose-quality testing in this pass.
- No VO source code or launch file logic changes.

Test context:
- Car was stationary and elevated (wheels not touching mat), which is acceptable
  for calibration/probe validation.
- Branch state in `/home/nvidia/Documents/ACC_Development` was confirmed as
  `Gabriel` and synchronized with `origin/Gabriel`.

Sequence executed:
1. Confirmed no active ROS/camera owners (`ros2 node list`, `ps -ef` checks).
2. Ran pre-flight calibration snapshot:
   - `rs-enumerate-devices -c`
   - `rs-enumerate-devices -o`
3. Ran isolated probe (Step 4):
   - `ros2 run qcar2_nodes rgbd_get_meters_probe --ros-args -p device_type:=physical -p frame_width_depth:=640 -p frame_height_depth:=480 -p frame_rate:=30.0`
4. Verified probe topics and sample telemetry:
   - `/camera/depth_image_probe_raw`
   - `/camera/depth_image_probe_meters`
   - `/camera/depth_probe_status`

Artifacts saved:
- `/home/nvidia/vo_calib_logs/realsense_calib_2026-04-30_120521.txt`
- `/home/nvidia/vo_calib_logs/realsense_options_2026-04-30_120521.txt`

Key observations:
- Physical D435 detected and streaming successfully in probe mode.
- `Depth Units` default reported as `0.001` in the options snapshot.
- 640x480 intrinsics were available in snapshot:
  - Depth: `fx=384.7583`, `fy=384.7583`, `ppx=324.0233`, `ppy=237.5673`
  - Color: `fx=607.3273`, `fy=607.3451`, `ppx=324.9502`, `ppy=249.8685`
- Color-depth extrinsics were reported in snapshot (non-identity transform).
- Probe status sample confirmed API-meters path:
  - `raw=8797`, `meters_api=8.7970`, `raw_est_m=0.0560`, `est/api=0.0064`
- Probe publish rate on `/camera/depth_image_probe_meters` was near target:
  - observed ~`28-29 Hz` for a `30 Hz` request.

Interpretation:
- `video3d_frame_get_meters(...)` is functioning on this physical setup.
- The old virtual estimate (`raw/(15707*0.1)`) remains strongly inconsistent
  with physical API meters in this run, so it should not be treated as a
  physical depth conversion truth.

Follow-up intent:
- Proceed to controlled physical stack bring-up (Section 2) only after this
  isolated probe validation, keeping VO in `camera_mode:=physical`.

## 2026-04-30 (manual rerun by user)

Scope of this pass:
- Manual user-run validation of Step 1 and Step 4.
- Confirm behavior after accidental terminal suspend (`Ctrl+Z`).
- No VO runtime code changes.

What happened:
- User ran Step 1 manually and saved logs in the repo-local folder:
  - `vo_calib_logs/realsense_calib_2026-04-30_121711.txt`
  - `vo_calib_logs/realsense_options_2026-04-30_121711.txt`
- Probe was started successfully, then accidentally suspended with `Ctrl+Z`.
- A second probe start failed with:
  - `video3d_start_streaming failed: Error was returned from the underlying OS-specific layer.`
- Root cause matched stream ownership conflict (first probe process still held
  the depth stream).
- After stopping the old probe process, rerun succeeded.

Manual probe evidence reported by user:
- `/camera/depth_probe_status` was visible.
- Status sample:
  - `frame=959 ... raw=9056 meters_api=9.0560 raw_est_m=0.0577 est/api=0.0064 ...`
- Probe console sample included:
  - `Depth probe opened on 0 at 640x480 @ 30.0 Hz`
  - periodic `[GET_METERS]` frames with `meters_api` values around 8.3-9.6 m
    in reported snippets.
- `timeout 12s ros2 topic hz /camera/depth_image_probe_meters` showed roughly
  `23-27 Hz` during this manual run.

Important note on `ros2 topic hz` timeout:
- The `ExternalShutdownException` seen after the timed `ros2 topic hz` command
  is expected when `timeout` terminates the ROS CLI process.

Conclusion:
- Physical depth probe path is confirmed working.
- Isolation requirement is confirmed operationally: only one stream owner at a
  time for reliable probe startup.

## 2026-04-30 (read-only pre-drive config audit)

Scope of this pass:
- Read-only verification of VO/camera launch readiness before physical manual
  drive tests.
- No ROS runtime execution and no VO algorithm code changes.

Audit outcomes:
- `qcar2_manual_cartographer_launch.py` includes
  `qcar2_manual_drive_launch.py`, so manual stack + cartographer are launched
  together in that path.
- `qcar2_manual_drive_launch.py` includes the `rgbd` node, so camera topics are
  expected in this launch path.
- `qcar2_launch.py` still has `realsense_camera_node` commented out in its
  `LaunchDescription`; this is important because `qcar2_cartographer_launch.py`
  includes `qcar2_launch.py`.
- `rgbd.cpp` defaults remain aligned with current physical VO calibration:
  `device_type="physical"`, `640x480` for color/depth, `frame_rate=30.0`.
- `vo_node.py` still defaults `camera_mode='virtual'`, but the active test
  command path sets `camera_mode:=physical`, `alignment_mode:=auto`,
  `depth_scale:=0.0` explicitly.
- `visual_odometry.py` physical 640x480 intrinsics match current RealSense
  snapshot values used in today’s tests.

Operational implication:
- Preferred physical VO path remains:
  - `qcar2_manual_cartographer_launch.py`
  - `vo_node` launched with explicit physical parameters
- Avoid using `qcar2_cartographer_launch.py` for VO camera tests unless
  `qcar2_launch.py` is adjusted to include `rgbd`.

## 2026-04-30 (manual-drive test prep guidance)

Scope of this pass:
- Prepared operator-facing launch sequence for first physical manual-drive VO
  check.
- Confirmed control mapping from source without running runtime nodes.

Key clarification:
- In current manual launch path, drive control is joystick/gamepad based, not
  WASD keyboard control.
- `qcar2_nodes/src/command.cpp` mapping:
  - `LB` arm enable
  - `RT` throttle
  - Left stick X for steering
  - `A` toggles reverse direction

Execution policy retained:
- Runtime commands are user-executed manually.
- Documentation/log files may be updated directly in-repo.

## 2026-04-30 (WASD keyboard manual-drive integration)

Scope of this pass:
- Add keyboard-based manual drive option so physical VO tests can run without a
  game controller.
- Keep changes focused on launch/control plumbing; no VO algorithm math changes.

Reference reviewed:
- Physical_Arturo branch keyboard node pattern:
  `qcar2_autonomy/autonomy/manual_drive.py`
  (publishes `Twist` to `/cmd_vel_nav`).

Changes made:
- Added `qcar2_autonomy/autonomy/manual_drive.py` in current branch with
  conservative defaults for first physical tests:
  - `forward_speed=0.10`
  - `reverse_speed=0.08`
  - `turn_rate=0.25`
  - publishes to `/cmd_vel_nav`
- Added `manual_drive` console entry point to
  `qcar2_autonomy/setup.py`.
- Added keyboard-specific base launch:
  - `qcar2_nodes/launch/qcar2_keyboard_drive_launch.py`
  - Includes `lidar`, `rgbd`, `csi`, `qcar2_hardware`, and
    `nav2_qcar2_converter`
  - Intentionally excludes joystick `command` node
- Added keyboard-specific cartographer launch:
  - `qcar2_nodes/launch/qcar2_keyboard_cartographer_launch.py`
  - Includes `qcar2_keyboard_drive_launch.py` + cartographer nodes
- Updated `Easy_Start.txt` with a new keyboard section (2.1) and explicit
  per-terminal commands.

Why this wiring is needed:
- `manual_drive.py` sends `Twist` on `/cmd_vel_nav`.
- `qcar2_hardware` consumes `qcar2_motor_speed_cmd` (`MotorCommands`).
- `nav2_qcar2_converter` bridges `/cmd_vel_nav` -> `qcar2_motor_speed_cmd`.
- Excluding joystick command prevents controller-zero-command interference when
  no controller is desired.

Verification performed:
- Syntax checks only (no hardware runtime execution):
  - `python3 -m py_compile` passed for:
    - `autonomy/manual_drive.py`
    - `qcar2_keyboard_drive_launch.py`
    - `qcar2_keyboard_cartographer_launch.py`

## 2026-04-30 (TEST 1 Physical output analysis, read-only)

Scope of this pass:
- Analyze `TEST 1 Physical` terminal output in `VO_readings.txt`.
- Confirm what `/vo/fault_status` indicates about current VO behavior.
- No runtime command execution, no VO code changes.

Data source:
- `Development/ros2/src/qcar2_autonomy/VO_readings.txt`
  section marker: `TEST 1 Physical`.

Observed monitor-state distribution (1061 status lines):
- `agree`: 513
- `vo_suspect`: 451
- `warming`: 78
- `init`: 19

Quality snapshots:
- Overall averages:
  - inliers: `178.90`
  - weight (`w`): `0.512`
  - disagreement (`rho`): `0.083`
- `agree` frames:
  - average inliers: `264.22`
  - average weight: `0.708`
- `vo_suspect` frames:
  - average inliers: `83.72`
  - average weight: `0.400`
  - average rho: `0.140`
  - zero-inlier frames: `28`

Interpretation:
- VO camera tracking is functioning and often strong (high inliers/high weight
  periods with low `rho`).
- During harder segments, quality gates correctly down-weight VO and move state
  to `vo_suspect`.
- Current monitor behavior is conservative rather than failing open.

Important architectural note:
- This run is not yet IMU-fused VIO.
- VO frame deltas come from camera processing in `visual_odometry.py`
  (ORB + matching + RANSAC + SVD).
- Cartographer is used for monitor comparison and periodic anchoring/re-anchor
  in `vo_node.py`, not as a per-frame overwrite of VO.

Likely next tuning items (no changes made in this pass):
- Revisit `turn_gate_deg` (default 5 deg) for physical turning behavior.
- Re-tune `spread_bad`, `spread_good`, and `min_vo_weight` for physical scenes.
- Improve fault-status observability by logging suspect reason codes.
- Continue physical depth/alignment hardening toward fully validated metric
  depth path.

## 2026-04-30 (anchoring frequency audit, read-only)

Scope of this pass:
- Answer whether VO is currently re-anchoring and how often.
- Use code inspection + `TEST 1 Physical` log evidence.
- No code/runtime changes.

Code behavior confirmed:
- Initial anchor occurs once on first valid frame via:
  `self.vo.reset(self.cart_x, self.cart_y, self.cart_psi)`.
- Soft re-anchor attempts happen only after consecutive `agree` decisions,
  then `_try_reanchor(...)` enforces both gates:
  - `reanchor_cooldown_s = 8.0`
  - `reanchor_min_dist = 0.25 m`
- If both gates pass, VO performs:
  `self.vo.soft_reset(self.cart_x, self.cart_y, self.cart_psi)`
  then enters warmup (`warmup_after_reanchor_s = 1.0`).

Practical implication:
- Re-anchoring is not per-frame and cannot happen every millisecond.
- Absolute upper bound is one re-anchor per 8 seconds, and only after moving
  at least 0.25 m since the last anchor.
- Between anchors, motion integration (`dx, dy, dpsi`) remains camera-driven.

`TEST 1 Physical` evidence summary:
- `warming` lines: 78
- `warming` segments: 9
- `agree -> warming` transitions: 7

Interpretation:
- The run shows occasional soft re-anchors, not continuous frame-by-frame
  Cartographer overwriting.
- However, each anchor does re-pin global VO pose/yaw to Cartographer, so the
  anchored VO stream is not fully independent over long time horizons.

Potential follow-up (not implemented in this pass):
- Add a `reanchor_enabled` parameter and expose two streams:
  - `VO_raw` (no re-anchor, strict redundancy)
  - `VO_stabilized` (current anchored behavior)

## 2026-04-30 (axis-consistency inspection, read-only)

Scope of this pass:
- Read-only interpretation of user-highlighted `TEST 1 Physical` slice.
- No code edits and no runtime execution.

User-highlighted window:
- `VO_readings.txt` lines 2221-2255.

Computed observations:
- Net motion in this slice:
  - VO:  `Δx=+0.050`, `Δy=+0.270`
  - Cart:`Δx=-0.030`, `Δy=+0.250`
- Consecutive-step sign agreement:
  - X axis: `1/17`
  - Y axis: `14/17`

Interpretation:
- Y-direction change is consistently tracked in this interval.
- X-direction consistency is weak in the same interval.
- This supports axis-wise evaluation as a primary metric, not just absolute
  pose closeness.

Policy note (discussion only, not implemented):
- Short-term competition strategy may force `vo_psi = cart_psi` for stability,
  while preserving raw camera yaw diagnostics (`vo_psi_raw`, `dpsi_raw`) for
  continued VO-only improvement work.

## 2026-04-30 (alignment source audit + policy confirmation, read-only)

Scope of this pass:
- Read-only search for physical alignment/camera-info sources in active code.
- Confirm user policy decisions for upcoming iterations.
- No runtime node execution and no VO code modifications.

Policy confirmations from discussion:
- Future analyses should auto-detect high-value windows (startup stability,
  one-axis-dominant motion, and post-anchor drift windows), not only
  user-pointed line ranges.
- Short-term competition policy set: force `vo_psi = cart_psi`.

Codebase findings:
- `qcar2_nodes/src/rgbd.cpp` currently publishes image topics only:
  `camera/color_image`, `camera/depth_image`.
- No `sensor_msgs/msg/CameraInfo` publisher path found in active source.
- In `visual_odometry.py` physical alignment remains placeholder:
  `PHYSICAL_ALIGNMENT_M = np.eye(3)` and physical default `use_alignment=False`.
- In `vo_node.py`, `alignment_mode=auto` resolves to alignment OFF for
  physical mode.

Calibration artifacts findings:
- `vo_calib_logs/realsense_calib_2026-04-30_121711.txt` contains:
  - 640x480 depth intrinsics,
  - 640x480 color intrinsics,
  - stream extrinsics (`Depth -> Color`, `Color -> Depth`) with rotation and
    translation.
- This confirms physical geometry is available in saved artifacts, but is not
  yet wired as a validated runtime alignment path inside current VO code.

Angle-sign clarification:
- `-180` and `180` yaw at startup are equivalent wrapped angles (same heading);
  this boundary flip is expected representation behavior.

Dashboard compatibility note:
- Future `/vo/fault_status` format updates should be mirrored in
  `vo_terminal_dashboard.py` in the same pass to keep diagnostics coherent.

## 2026-04-30 (hardware alignment clarification, read-only)

Scope of this pass:
- Clarify whether hardware-provided geometry should be used immediately.
- Distinguish available calibration data from active VO alignment behavior.
- No code/runtime changes.

Findings:
- Hardware geometry is available in saved calibration artifacts:
  - depth/color intrinsics (640x480),
  - depth<->color extrinsics.
- Quanser video3d API header exposes runtime accessors:
  - `video3d_stream_get_camera_intrinsics(...)`
  - `video3d_stream_get_extrinsics(...)`
  - `video3d_stream_get_depth_scale(...)`
- Active VO physical alignment path remains placeholder:
  `PHYSICAL_ALIGNMENT_M = np.eye(3)` with physical default `use_alignment=False`.

Clarification:
- The caution is not "do not use hardware geometry".
- The caution is "do not enable placeholder 2D warp as if it were true
  physical alignment".
- True depth->color alignment is depth-dependent 3D reprojection, not a single
  global homography for all distances.

Recommended direction (discussion only):
- Keep placeholder warp disabled until the VO path applies validated
  depth-dependent reprojection (or equivalent aligned-depth output) using real
  intrinsics/extrinsics/depth-scale.

## 2026-04-30 (physical alignment + yaw policy implementation)

Scope of this pass:
- Implement physical depth->color alignment improvements in VO path.
- Apply official short-term yaw policy (`vo_psi = cart_psi`).
- Update terminal dashboard parsing for current compact fault-status format.
- No ROS runtime node launches or hardware-motion commands executed.

Code changes made:

1) `autonomy/visual_odometry.py`
- Added physical `Depth -> Color` extrinsic transform constant from the saved
  RealSense calibration snapshot.
- Added alignment model selection:
  - virtual: homography alignment (`alignment_M`)
  - physical: projective depth->color reprojection
- Implemented `_align_depth_projective(...)`:
  - backprojects depth pixels with depth intrinsics,
  - transforms depth-camera points to color-camera frame,
  - projects to color pixels,
  - z-buffer resolves collisions with nearest depth.
- Physical mode defaults now set:
  - `use_alignment=True`
  - `alignment_model='projective'`

2) `autonomy/vo_node.py`
- Added parameter `force_cart_yaw` (default `True`).
- In `_vo_tick`, when enabled:
  - injects cart yaw before VO update,
  - pins output VO yaw to cart yaw after update,
  - keeps internal VO heading pinned to cart yaw.
- Result: competition-phase heading stabilization while retaining camera-based
  translation estimation.

3) `autonomy/vo_terminal_dashboard.py`
- Added regex parser for current compact `/vo/fault_status` format:
  `state rho w | vo(...) ct(...) | dx dy dpsi inl sp`.
- Kept legacy key-value fallback parsing.
- Updated dashboard summary to display state, rho, weight, drift,
  inliers, and spread from the compact message.

Validation:
- `python3 -m py_compile` passed for:
  - `autonomy/visual_odometry.py`
  - `autonomy/vo_node.py`
  - `autonomy/vo_terminal_dashboard.py`
- Offline sanity check run for `DepthProjector` alignment methods only.

Notes:
- This pass did not edit `VO_readings.txt`.
- Physical VO remained depth-based throughout; this pass specifically replaced
  placeholder-style physical alignment behavior with a projective path.

## 2026-04-30 (end-of-day clarification: virtual impact)

Scope:
- Clarified whether latest pipeline changes are physical-only.
- No additional source edits in this pass.

Clarification:
- Physical alignment changes are mode-specific (projective path for physical).
- Virtual alignment path remains homography-based.
- `force_cart_yaw` default is currently global in `vo_node.py`; therefore,
  virtual runs can also have yaw pinned unless the parameter is disabled.
- Dashboard parser change is global but diagnostic-only (no VO math impact).
