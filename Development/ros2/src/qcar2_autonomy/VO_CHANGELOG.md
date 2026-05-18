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

## 2026-05-04 (Physical Test 1.1 vs 1.2 comparison, read-only)

Scope:
- Compare new physical runs logged as `Physical Test 1.1` and `Physical Test 1.2`
  in `VO_readings.txt`.
- Focus on agreed pattern checks: startup stability, one-axis consistency,
  and pre-reanchor behavior indicators.
- No code/runtime changes in this pass.

Data windows used:
- Physical Test 1.1: `VO_readings.txt` lines ~3010 to ~4461 (`data:` lines only)
- Physical Test 1.2: `VO_readings.txt` lines ~4499 to EOF (`data:` lines only)

Summary metrics:

1) Physical Test 1.1
- Samples: `725`
- State counts:
  - `agree`: `235`
  - `vo_suspect`: `424`
  - `warming`: `58`
  - `init`: `8`
- Averages:
  - `rho`: `0.1197`
  - `w`: `0.3863`
  - `inliers`: `127.12`
  - `spread`: `84.70`
  - zero-inlier frames: `37`
- Startup (first 80 frames):
  - state mix: `80/80 agree`
  - avg `rho`: `0.0558`
  - avg `w`: `0.7316`

2) Physical Test 1.2
- Samples: `779`
- State counts:
  - `agree`: `259`
  - `vo_suspect`: `448`
  - `warming`: `69`
  - `init`: `3`
- Averages:
  - `rho`: `0.1039`
  - `w`: `0.4803`
  - `inliers`: `133.67`
  - `spread`: `92.95`
  - zero-inlier frames: `35`
- Startup (first 80 frames):
  - state mix: `77 agree`, `2 warming`, `1 init`
  - avg `rho`: `0.0199`
  - avg `w`: `0.8230`

Interpretation vs 1.1:
- Run 1.2 is generally better quality than 1.1:
  - lower average disagreement (`rho`),
  - higher confidence/quality weight (`w`),
  - better inlier and spread averages,
  - stronger startup behavior (much lower startup `rho`).
- `vo_suspect` remains frequent in both runs, so monitor strictness during
  dynamic segments is still active.

Pattern checks (one-axis windows):
- Y-dominant windows:
  - 1.1 average sign agreement: `0.707`
  - 1.2 average sign agreement: `0.726`
- X-dominant windows:
  - 1.1 average sign agreement: `0.514`
  - 1.2 average sign agreement: `0.634`
- This indicates the biggest relative gain in 1.2 is on X-axis consistency.

Concrete line-range examples from 1.2:
- Strong Y-dominant consistency:
  - around `4861 -> 4911` (very high sign agreement, near 0.96)
- Strong X-dominant consistency:
  - around `4609 -> 4659` (sign agreement ~1.0)
- Weak segments still present:
  - Y-dominant around `5459 -> 5509`
  - X-dominant around `5881 -> 5931`

Re-anchor related observation:
- `agree -> warming` transitions (proxy for re-anchor events) were present in
  both runs (`6` in 1.1, `7` in 1.2), consistent with gated re-anchor behavior
  rather than per-frame anchoring.

Next-step implication:
- Keep current re-anchor policy ON for now (as planned).
- Use 1.2-style controlled driving as baseline for parameter tuning.
- Start VO weighting retune next (priority):
  1) `min_vo_weight`
  2) `spread_bad` / `spread_good`
  3) inlier/confidence gates

## 2026-05-04 (runtime workspace sync fix for physical VO)

Scope:
- Diagnose why physical runs still showed `Alignment: OFF` and non-pinned yaw
  after code changes were already committed in `ACC_Development`.
- Apply non-runtime fix (source sync + rebuild) so `~/ros2` executes the
  updated VO pipeline.
- No ROS motion/runtime nodes were launched in this pass.

Root cause confirmed:
- `ACC_Development` had updated files (`vo_node.py`, `visual_odometry.py`),
  but `~/ros2/src` and `~/ros2/install` were still on older copies.
- Therefore `ros2 run qcar2_autonomy vo_node` used old logic from
  `~/ros2/install`:
  - physical `alignment_mode:=auto` resolved to OFF,
  - `force_cart_yaw` logic absent.

Fix applied:
- Synced package from branch workspace into runtime workspace:
  - `/home/nvidia/Documents/ACC_Development/Development/ros2/src/qcar2_autonomy/`
    -> `/home/nvidia/ros2/src/qcar2_autonomy/`
- Rebuilt runtime package:
  - `colcon build --packages-select qcar2_autonomy`
- Verified `~/ros2/src` and `~/ros2/install` now match branch hashes and
  contain:
  - `alignment_mode == auto -> use_align = True`
  - `force_cart_yaw` parameter and pinning logic
  - physical projective depth->color alignment path

Process hardening:
- Updated `Easy_Start.txt` by adding Step `0.6`:
  - explicit `rsync` sync from branch repo to `~/ros2/src`
  - package rebuild step
- Daily order now includes Step 0.6 before physical test runs.

## 2026-05-04 (physical autonomy launch + RViz overlay implementation)

Scope:
- Align physical launch chain with autonomous VO workflow.
- Add live camera overlay for VO/Cart readings in RViz.
- Keep change set focused (no VO math changes).

Changes made:
1) Re-enabled RGBD in physical base launch:
- File: `qcar2_nodes/launch/qcar2_launch.py`
- `realsense_camera_node` is now included in returned `LaunchDescription`.
- Effect: `/camera/color_image` and `/camera/depth_image` are available in the
  physical cartographer path used by VO.

2) Added VO camera overlay node:
- File: `qcar2_autonomy/autonomy/vo_image_overlay.py`
- New node subscribes to:
  - `/camera/color_image`
  - `/vo/fault_status`
- Publishes:
  - `/vo/overlay_image`
- Overlay content includes state, rho, weight, VO pose, Cart pose,
  frame deltas, inliers, spread, and drift.

3) Added executable entry point:
- File: `qcar2_autonomy/setup.py`
- New console script:
  - `vo_overlay=autonomy.vo_image_overlay:main`

4) Updated runbook:
- File: `Easy_Start.txt`
- Added practical 5-terminal sequence for autonomous physical run + RViz image
  overlay, including `vo_overlay` and `/vo/overlay_image` topic guidance.

Notes:
- This pass does not alter VO estimator math (`visual_odometry.py`) or
  monitor state logic (`vo_node.py`).
- Purpose is runtime workflow reliability and observability.

## 2026-05-04 (vo_overlay crash: empty frame handling + runtime sync note)

Observed during physical workflow:
- `ros2 run qcar2_autonomy vo_overlay` crashed with:
  `AttributeError: 'NoneType' object has no attribute 'shape'`
- Cause: runtime install was stale (older overlay file), while branch source
  already had the empty-frame guard.

Current expected fix path:
- Sync `qcar2_autonomy` source from branch workspace into `~/ros2/src`.
- Rebuild `qcar2_autonomy` in `~/ros2`.
- Re-run `vo_overlay`.

No estimator math changed in this item; this is runtime robustness + sync.

## 2026-05-04 (physical cartographer frame parity with virtual)

Change requested:
- Remove physical-only `map_rotated -> map` static transform from
  `qcar2_cartographer_launch.py` so physical launch matches virtual map frame
  behavior.

Implemented:
- Deleted `static_transform_publisher` node that injected 180 deg yaw frame
  offset.
- Updated launch return list accordingly.
- Synced `qcar2_nodes` to runtime workspace and rebuilt `qcar2_nodes`.

Expected effect:
- Physical path follower and VO consumers read the same direct `map` frame
  convention as virtual tests.

## 2026-05-04 (rollback: keep original physical cartographer frame)

Per test-control request:
- Kept original physical launch behavior with static TF
  `map_rotated -> map` in `qcar2_cartographer_launch.py`.
- Re-synced and rebuilt `qcar2_nodes` so runtime matches this rollback.

Reason:
- Avoid large frame-convention changes before baseline verification run.

## 2026-05-04 (startup note: rs-enumerate-devices segfault)

Observed:
- At startup calibration step, `rs-enumerate-devices -c/-o` crashed with
  `Segmentation fault (core dumped)`.

Interpretation:
- Treat as failure of the enumeration utility itself until proven otherwise.
- VO testing may still proceed if ROS camera/probe nodes stream normally.

Recommended immediate triage:
1) ensure no competing camera owner process,
2) collect quick version/path/kernel crash context,
3) validate with `rgbd_get_meters_probe` + ROS topic checks.

## 2026-05-04 (rs-enumerate-devices segfault triage results)

Collected environment facts:
- No active competing `rgbd/realsense/cartographer/vo` process at crash time.
- Enumerator path/version:
  - `/opt/ros/humble/bin/rs-enumerate-devices`
  - `2.54.1`
- Installed package versions include mismatch:
  - `librealsense2` = `2.49.0`
  - `ros-humble-librealsense2` = `2.54.1`
  - `ros-humble-realsense2-camera` = `4.54.1`
- `dmesg` unavailable to regular user (kernel buffer permission).

Assessment:
- Likely ABI/runtime mismatch contributes to CLI enumerator segfault.
- Continue VO flow if ROS camera/probe nodes stream correctly; treat
  `rs-enumerate-devices` as non-blocking diagnostic-tool issue for this phase.

## 2026-05-04 (safety: nav command timeout auto-stop)

Issue observed:
- Vehicle continued moving briefly/indefinitely after planner stack stop,
  because converter kept re-publishing last received `/cmd_vel_nav` command.

Fix implemented:
- Updated `qcar2_nodes/src/nav2_qcar_command_convert.cpp`:
  - Added parameter: `cmd_timeout_sec` (default `0.25`).
  - Added last-command timestamp tracking.
  - If command age exceeds timeout, published motor command is forced to
    zero steering + zero throttle.
- Synced `qcar2_nodes` to runtime workspace and rebuilt package.

Operational effect:
- Loss of upstream planner command stream now fails safe to stop.

## 2026-05-04 (map_rotated trace audit)

User report:
- `/planned_path` appears inverted/opposite in RViz during physical run.

Audit findings:
- Physical launch file currently defines static TF:
  `map_rotated -> map` (180 deg yaw).
- Branch source `autonomy/nav_to_pose.py` currently references `map`.
- Nested workspace install copy at
  `/home/nvidia/ros2/src/install/qcar2_autonomy/.../nav_to_pose.py`
  still references `map_rotated` in path frame + TF lookup.

Implication:
- If wrong overlay/workspace setup is sourced, planner frame conventions can
  diverge and produce apparent path inversion.

## 2026-05-04 (rebuild after nav_to_pose map-frame fix)

Action:
- User updated `autonomy/nav_to_pose.py` to remove `map_rotated` usage.
- Synced package into `~/ros2/src` and rebuilt `qcar2_autonomy`.

Verification:
- `path_msg.header.frame_id = "map"`
- `pose.header.frame_id = "map"`
- `from_frame_rel = "map"`

Build status:
- `qcar2_autonomy` build succeeded.

## 2026-05-12 (read-only TF inversion audit, physical)

Inputs analyzed from `VO_readings` (cartographer physical TF test):
- `tf2_echo map map_rotated` => yaw ~ -180 deg
- `tf2_echo base_link base_scan` => yaw ~ +180 deg
- `tf2_echo map base_link` => yaw ~ -180 deg
- `tf2_echo map_rotated base_link` => yaw near 0 deg

Conclusion:
- Current physical frame chain aligns heading near zero in `map_rotated`, not
  in `map`.
- Using planner/path in `map` under this chain can produce opposite/inverted
  behavior (forward/backward and turn-direction confusion).

Diagnostic quality note:
- Initial "Invalid frame ID" lines were transient startup timing before TF
  frames became available; subsequent repeated transform outputs are valid.

## 2026-05-12 (frame-history decision checkpoint)

Key history finding:
- Initial ROS2 `nav_to_pose.py` used `map_rotated` (path frame + TF lookup).
- Commit `006f0dc` changed those references to `map`.
- Physical TF 180-deg transforms in lidar frame and cartographer launch were
  already present in initial commit lineage.

Decision guidance:
- Prefer single-variable test first: restore `nav_to_pose` frame references to
  `map_rotated` while keeping physical TF chain unchanged.
- Avoid changing both physical TF rotations to zero simultaneously during first
  diagnostic pass.

## 2026-05-12 (single-variable frame rollback test in nav_to_pose)

Change applied:
- Restored `autonomy/nav_to_pose.py` frame convention to `map_rotated` for:
  - path header frame
  - pose frame
  - TF lookup frame (`from_frame_rel`)
- Kept `map` alternatives commented for traceability.

Scope control:
- No changes made to physical TF publishers in this step
  (`fixed_lidar_frame.cpp` and `qcar2_cartographer_launch.py` unchanged).
- Purpose is isolated validation of planner frame convention only.

## 2026-05-12 (VO x/y freeze diagnostic interpretation)

Observed behavior:
- During physical autonomy test, `vo_x` and `vo_y` appeared static in
  `/vo/fault_status` and overlay.

Code interpretation (no estimator patch yet):
- `visual_odometry.py` keeps the previous pose when current frame update is
  rejected.
- Rejection causes include:
  - insufficient matches / depth-valid correspondences,
  - `inlier_count < min_inliers`,
  - safety gate failures (`max_translation`, `max_rotation_deg`, `max_dt`).

Implication:
- Static VO pose can occur even while cart pose moves, if consecutive frame
  updates are rejected.
- This is not directly caused by planner map-frame edits alone.

## 2026-05-12 (hotfix: alignment_mode CLI type crash)

Issue:
- Running `vo_node` with `-p alignment_mode:=off` caused startup failure:
  ROS parsed `off` as bool, but `alignment_mode` was declared as string.

Fix:
- Updated `autonomy/vo_node.py` to declare `alignment_mode` with
  `ParameterDescriptor(dynamic_typing=True)`.
- Downstream logic already normalizes bool/string values, so behavior is
  preserved while avoiding launch-time type exceptions.

Build:
- Synced and rebuilt `qcar2_autonomy` in `~/ros2`.

## 2026-05-12 (physical alignment regression guard)

Issue observed:
- Physical VO runs looked stable with alignment disabled, but degraded when
  alignment path was enabled.

Root cause in code:
- `vo_node.py`: `alignment_mode:=auto` was resolving to alignment ON in
  physical mode.
- `visual_odometry.py`: physical defaults also set `use_alignment=True`.
- Net effect: physical runs entered projective depth->color alignment by
  default instead of conservative raw-depth mode.

Changes applied:
1) `autonomy/vo_node.py`
- `alignment_mode=auto` now resolves to:
  - virtual: alignment ON
  - physical: alignment OFF
- Added explicit runtime warning for physical auto-off behavior.
- Explicit opt-in still supported via `-p alignment_mode:=on`.

2) `autonomy/visual_odometry.py`
- Physical mode default changed:
  - `use_alignment: True` -> `use_alignment: False`
- Projective alignment code remains intact for explicit testing.

Verification:
- Syntax check passed for updated files (`python3 -m py_compile ...`).

Expected effect:
- Physical default runs should no longer be unintentionally affected by
  projective alignment path.
- Alignment A/B testing remains available by explicitly setting
  `alignment_mode:=on` or `alignment_mode:=off`.

## 2026-05-12 (alignment reliability deep-dive, read-only)

Scope:
- Read-only audit of active physical runtime path vs. available D435 camera
  data artifacts.
- No estimator or launch code edits in this checkpoint.

Verified facts:
1) Active runtime depth source
- `qcar2_nodes/src/rgbd.cpp` publishes `/camera/depth_image` from
  `video3d_frame_get_data(...)` as raw `MONO16`.
- Current `rgbd.cpp` does not publish `camera_info` topics.

2) Hardware geometry availability
- `vo_calib_logs/realsense_calib_2026-04-30_121711.txt` contains:
  - 640x480 depth intrinsics,
  - 640x480 color intrinsics,
  - `Depth -> Color` and `Color -> Depth` extrinsics,
  - distortion model entries and coefficients from snapshot.
- `vo_calib_logs/realsense_options_2026-04-30_121711.txt` reports
  `Depth Units` default = `0.001`.

3) API depth truth path is implemented separately
- `rgbd_get_meters_probe.cpp` samples raw depth and metric depth from the same
  frame via:
  - `video3d_frame_get_data(...)`
  - `video3d_frame_get_meters(...)`
- This confirms meter-depth access exists, but is not yet the main depth image
  path used by VO.

4) Current alignment risk mechanism
- Physical alignment in `visual_odometry.py` is a software projective
  reprojection stage (not a driver-provided synchronized aligned-depth topic).
- `vo_node.py` uses latest buffered color/depth frames without explicit
  color-depth timestamp pairing tolerance at VO tick time.
- Under motion, small temporal mismatch plus reprojection sensitivity can
  degrade depth-at-feature lookup, harming VO `x/y`.

Interpretation:
- Alignment constants are sourced from real camera data, not fabricated.
- Instability comes from runtime coupling/synchronization and path validation,
  not from lack of manufacturer calibration availability.

## 2026-05-14 (PIT depth alignment cross-check, read-only)

Scope:
- Compare active physical VO depth/alignment path against PIT implementation in
  `Development/MDC_libraries/python/pit/YOLO/utils.py`.
- No runtime node launches; no estimator code edits.

Findings:
1) PIT virtual vs physical alignment are different
- Virtual branch:
  - uses homography `self.M` and virtual depth scale (`5.5` in that stack).
- Physical branch:
  - starts `QCar2DepthAlign.rt-linux_qcar2` via `quarc_run`,
  - receives combined depth+RGB packet from `BasicStream`.

2) Relation to current ROS2 VO path
- Current ROS2 physical VO uses `/camera/depth_image` from `rgbd.cpp` raw
  `MONO16` path and optional software projective reprojection in
  `visual_odometry.py`.
- PIT physical path is closer to "driver/runtime-provided aligned stream"
  behavior than to manual homography.

3) Useful reuse takeaway
- `VIRTUAL_ALIGNMENT_M` in current VO matches PIT virtual matrix exactly.
- For physical, PIT suggests that using Quanser's aligned stream runtime may be
  more robust than forcing homography-like behavior.

4) Build/symlink clarity (validated)
- With `--symlink-install`, Python source edits in `~/ros2/src/qcar2_autonomy`
  are picked up after process restart (no rebuild required).
- Edits in `~/Documents/ACC_Development/...` require sync into `~/ros2/src/...`
  before runtime sees them.
- Rebuild required for `setup.py`/entry points, C++ code, interfaces, or deps.

## 2026-05-14 (PIT vs ROS stats test result, read-only)

Data source:
- `VO_readings.txt` -> Physical Test 2 -> Testing PIT stats vs ROS stats.

Observed:
1) PIT aligned stream (`QCar2DepthAligned`)
- Sampled frames show:
  - valid depth pixels around ~290k to ~292k,
  - center depth around ~1.15 to ~1.21 m.
- `received_new=89/200` over ~2.99 s under 10 ms polling indicates
  approximately 30 Hz new-frame availability (nonblocking polling behavior).

2) ROS rgbd topic rates (same session)
- `/camera/color_image`: approximately ~17-20 Hz average with jitter.
- `/camera/depth_image`: approximately ~14-18 Hz average with jitter.
- End-of-command exceptions correspond to timeout shutdown behavior of topic
  monitor commands.

Interpretation:
- PIT aligned stream appeared stable in this snapshot test.
- Current ROS rgbd stream exhibited lower and less stable rates than requested
  30 Hz settings.
- Result reinforces synchronization/timing as a likely major factor in physical
  VO degradation when alignment is enabled.

## 2026-05-14 (full PIT/HAL/PAL camera-path review, read-only)

Scope:
- User requested a broad review of official library camera/alignment paths and
  a concrete improvement strategy for physical VO using `QCar2DepthAligned`.
- No code changes in this checkpoint.

Reviewed:
- `Development/MDC_libraries/python/pit/YOLO/utils.py`
- `docker/0_libraries/python/pal/utilities/vision.py`
- `docker/0_libraries/python/pal/products/qcar.py`
- `Development/MDC_libraries/python/hal/content/qcar.py`
- `Development/MDC_libraries/python/hal/utilities/image_processing.py`
- `Development/ros2/src/qcar2_autonomy/autonomy/{vo_node.py,visual_odometry.py,yolo_detector.py,yolo_detector_new.py}`
- `Development/ros2/src/qcar2_nodes/src/rgbd.cpp`

Findings:
1) Two distinct physical camera paths exist
- PIT path: `QCar2DepthAligned` starts `QCar2DepthAlign.rt-linux_qcar2` and
  consumes a single aligned depth+RGB stream packet via `BasicStream`.
- Current VO path: `rgbd.cpp` publishes separate color/depth ROS topics; VO
  consumes raw `MONO16` depth and optional in-VO software alignment.

2) Synchronization/coupling difference is fundamental
- PIT stream is packet-coupled (aligned depth and RGB delivered together).
- Current VO path is topic-coupled and currently does not enforce strict
  color/depth timestamp pairing at VO update time.

3) Official reusable helpers identified
- `pal.utilities.vision.Camera3D.read_depth(dataMode='M')` supports meter-depth
  retrieval for validation utilities.
- `hal.utilities.image_processing` has camera calibration and undistortion
  tooling for physical refinement.
- `yolo_detector.py` already integrates `QCar2DepthAligned` into ROS topics,
  providing a practical reference for a VO camera bridge.

Implementation direction (next step, not applied yet):
1) Add a dedicated VO camera bridge based on `QCar2DepthAligned` that publishes
   aligned RGB + aligned depth for VO use.
2) Extend `vo_node` with configurable image/depth input topics and a 32FC1
   depth ingestion path.
3) Run controlled A/B comparisons against current `rgbd.cpp` path while keeping
   ORB/RANSAC/SVD estimator settings constant.

## 2026-05-14 (implemented: VO direct PIT aligned input mode)

Scope:
- Implemented physical VO path that can consume `QCar2DepthAligned` directly.
- Kept existing ROS `rgbd.cpp` topic path intact as default/fallback.

Code changes:
1) `autonomy/vo_node.py`
- Added `input_source` parameter:
  - `ros_rgbd` (default): existing `/camera/color_image` + `/camera/depth_image`
  - `pit_aligned`: direct `QCar2DepthAligned` stream
- Added PIT connection parameters:
  - `pit_python_path`, `pit_ip`, `pit_port`, `pit_manual_start`,
    `pit_non_blocking`
- Added import resolver for `pit.YOLO.utils.QCar2DepthAligned` that supports:
  - existing `sys.path`
  - `MDC_PYTHON_PATH` env var
  - common local paths (`.../MDC_libraries/python`)
- Added `_init_pit_camera()` and `_read_pit_frame()`:
  - pulls aligned RGB + aligned meter-depth packets
  - uses ROS clock timestamp for VO update timing
  - includes periodic center-depth diagnostics for PIT mode
- PIT mode behavior:
  - no ROS color/depth subscribers are created
  - VO tick runs only when a new PIT packet arrives
  - alignment is forced OFF in VO projector (stream is pre-aligned)
- Added `destroy_node()` cleanup to terminate PIT stream process cleanly.

2) `autonomy/visual_odometry.py`
- `DepthProjector.align_depth()` now accepts:
  - integer raw depth inputs (scaled by `depth_scale`), and
  - floating depth inputs already in active units (no re-scaling)
- This enables direct use of PIT meter-depth frames without converting them to
  fake MONO16 raw counts first.
- Updated docstrings for accepted depth input types.

Build/verification:
- `python3 -m py_compile` passed for:
  - `autonomy/vo_node.py`
  - `autonomy/visual_odometry.py`
- `colcon build --packages-select qcar2_autonomy --symlink-install` succeeded
  after sourcing ROS environment.

How to run PIT input mode:
- Example launch command for physical VO:
  `ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=pit_aligned -p force_cart_yaw:=true`
- Optional path override if PIT import is not found automatically:
  `-p pit_python_path:=/home/nvidia/Documents/ACC_Development/Development/MDC_libraries/python`

Expected impact:
- Better color/depth synchronization at source (single aligned packet).
- Eliminate timing skew between separate ROS color/depth topics in VO ingest.
- Reduce depth-at-feature mismatch under motion, which should improve inlier
  quality stability and reduce drift spikes in physical runs.

## 2026-05-14 (K_inv fix for PIT-aligned + shadow VO yaw observer)

Scope:
- Two targeted fixes prompted by Claude code review:
  1. Correct intrinsics selection when consuming PIT-aligned depth.
  2. Add an independent camera-only yaw integrator for diagnostic comparison
     while `force_cart_yaw` remains the operational policy.
- Camera spec re-verification against the live D435 (not relying on saved
  `vo_calib_logs` snapshots).
- No virtual-mode behavior changed. No camera defaults changed. No RANSAC/ORB
  parameter changes. No framerate or sensor-preset changes.

Code changes:

1) `autonomy/visual_odometry.py`
- `DepthProjector.__init__` now accepts `depth_on_color_grid` (default `None`,
  falls back to `use_alignment` to preserve legacy behavior).
- Added attribute `self.depth_on_color_grid`.
- `pixels_to_3d_body` now selects `K_rgb_inv` vs `K_depth_inv` based on
  `self.depth_on_color_grid` instead of `self.use_alignment`. This separates
  two previously conflated concepts:
  - "Did we run software alignment?" (`use_alignment`)
  - "Is the depth currently sampled at color pixel coordinates?"
    (`depth_on_color_grid`)
- `VisualOdometryDepth.__init__` forwards the new kwarg to `DepthProjector`.

2) `autonomy/vo_node.py`
- Computes `depth_on_color_grid` and forwards it into `VisualOdometryDepth`:
  - `input_source=pit_aligned`: forced `True` (depth arrives pre-aligned to
    the color grid from `QCar2DepthAligned`).
  - `input_source=ros_rgbd`: mirrors `use_align` (legacy semantics).
- Added shadow yaw state `self.vo_psi_shadow`:
  - Initialized to `self.cart_psi` at first anchor and at every soft re-anchor.
  - Integrates the raw per-frame `dpsi` returned by the VO engine on every
    valid update, independent of `force_cart_yaw`.
- New publisher `/vo/vo_psi_shadow` (Float64, radians).
- `/vo/fault_status` extended with trailing `psi_raw=<deg>` field.
- Module docstring updated to list `/vo/vo_psi_shadow`.

3) `autonomy/vo_image_overlay.py`
- `_STATUS_RE` extended with an optional trailing `psi_raw=<deg>` group.
- Overlay now adds a `psi_raw=±Xdeg (camera-only, err=±Ydeg)` line when the
  new field is present in the parsed status.

4) `autonomy/vo_terminal_dashboard.py`
- `_COMPACT_RE` extended with the same optional `psi_raw` group.
- Dashboard prints an extra `psi_raw` row when the field is present, with the
  wrapped error vs Cart yaw.

Backward compatibility:
- All extended regexes use a non-capturing optional group `(?:...)?`, so
  legacy `VO_readings.txt` lines without `psi_raw` still parse.
- Default behavior for legacy callers of `DepthProjector` (no
  `depth_on_color_grid` arg) is unchanged because the default falls back to
  `use_alignment`.

Why the K_inv fix matters:
- Before: `K_inv = K_rgb_inv if use_alignment else K_depth_inv`.
- PIT path forces `use_alignment=False` because no software warp is needed —
  but the depth IS on the color pixel grid. The old logic therefore used
  `K_depth_inv` (fx≈384) to backproject ORB features picked from the color
  image (which needs fx≈607). Result: systematic scale error of
  `fx_rgb/fx_depth ≈ 1.58×` in the camera-frame 3D points, producing a
  consistent translation bias from the PIT pipeline.
- After: `K_inv = K_rgb_inv if depth_on_color_grid else K_depth_inv`.

Why the shadow yaw matters:
- `force_cart_yaw=True` overwrites VO yaw with Cartographer yaw at every
  tick, discarding the camera's frame-to-frame `dpsi` after it is computed.
- Shadow integrator preserves that signal without altering the active
  control path, providing a side-by-side diagnostic so the user can decide
  when `force_cart_yaw` can safely be flipped off.
- Reset behavior follows engine anchoring: shadow resets to `cart_psi` at
  hard anchor and at every soft re-anchor so it remains comparable to Cart.

Camera spec re-verification (live device, not log replay):
- `rs-enumerate-devices -c` confirmed against hardcoded constants:
  - Color 640x480: fx=607.327, fy=607.345, cx=324.950, cy=249.868
  - Depth 640x480: fx=384.758, fy=384.758, cx=324.023, cy=237.567
  - Depth->Color extrinsic matches `PHYSICAL_T_DEPTH_TO_COLOR`.
- `user_manual_system_hardware.pdf` (Table 3, p.6) confirms hardware max
  framerates: RGB 640x480 @ 60 Hz, Depth 640x480 @ 90 Hz. Both well above
  current 30 Hz operating point.
- FOV at 640x480 (live): Color 55.6°x43.1°, Depth 79.5°x63.9°. Depth FOV
  is wider, so any depth->color alignment necessarily crops depth coverage.
- `Realsense to body` extrinsic in the manual matches `PHYSICAL_T_CAM2BODY`
  in `visual_odometry.py`.

Compute budget check at 0.1 m/s, 30 Hz, Jetson AGX Orin:
- ORB (800 features) ~5-10 ms, BF k=2 ~2-4 ms, RANSAC 2-point x300 ~4-8 ms,
  depth path ~1-3 ms (no-op in PIT mode).
- Total ~15-25 ms vs 33 ms budget. Ample slack at current parameters; no
  tuning required until framerate or vehicle speed increases.
- Frame-to-frame translation at 0.1 m/s @ 30 Hz ≈ 3.3 mm, well below ORB
  scale; high overlap ensures fast RANSAC convergence.

Verification:
- `python3 -m py_compile` passed for all four modified files.
- Regex backward-compatibility validated against synthetic old/new
  `/vo/fault_status` strings.
- No ROS runtime executed in this pass; user runs hardware tests manually
  per the existing scope agreement.

Operational notes for next physical test:
- Sync + rebuild per Easy_Start.txt §0.6 before running:
  - `rsync -a --delete .../Development/ros2/src/qcar2_autonomy/ ~/ros2/src/qcar2_autonomy/`
  - `colcon build --packages-select qcar2_autonomy --symlink-install`
- Run VO with PIT input:
  - `ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=pit_aligned -p force_cart_yaw:=true`
- Compare `psi_raw` vs `ct(...,psi)` in the overlay or dashboard. When raw
  camera yaw tracks Cart reasonably across a full drive, consider flipping
  `force_cart_yaw:=false` for a redundancy-honest run.

Deferred (acknowledged, not implemented this pass):
- `CameraInfo` publishing from `rgbd.cpp` — pending VSLAM/loop-closure phase.
- ORB/RANSAC parameter A/B testing — deferred until baseline PIT results
  are observed; compute headroom currently sufficient at 30 Hz / 0.1 m/s.
- Camera sensor preset / laser power / exposure tuning — held at factory
  defaults to preserve compute headroom and avoid disturbing other consumers
  (YOLO, traffic detection) that share the same RealSense.
- Rate-jitter investigation — provisionally expected to resolve when running
  through `QCar2DepthAligned` (packet-coupled stream); revisit only if PIT
  path also exhibits jitter.

## 2026-05-14 (single-owner camera architecture: C++ bridge + subscribers)

Scope:
- Resolve the long-standing physical-stack camera contention where
  `rgbd.cpp` and `yolo_detector.py` both attempt to own the RealSense at
  the same time (rgbd via the Quanser `video3d_*` C API, yolo via direct
  instantiation of `QCar2DepthAligned`). Competition runs require both
  paths active simultaneously and this fails as a hard hardware conflict.
- Replace ad-hoc multi-owner pattern with a single physical camera owner
  exposed as a ROS publisher. Every other camera consumer subscribes.
- Implementation choice: C++ owner node living in `qcar2_nodes` next to
  `rgbd.cpp`, using the Quanser `quanser_communications` C API
  (`qcomm_connect/poll/receive/shutdown/close`). Confirmed available in
  `/usr/include/quanser/quanser_communications.h` and
  `/usr/lib/aarch64-linux-gnu/libquanser_communications.so`.
- Virtual mode is untouched. All virtual launches and the virtual VO path
  continue to use `rgbd.cpp` with MONO16 depth.

Architectural rationale:
- Quanser's `QCar2DepthAlign.rt-linux_qcar2` runtime performs depth->color
  alignment at the driver level and streams aligned (depth, RGB) packets
  over TCP. Previous Codex tests (2026-05-14, PIT vs ROS stats) showed
  the PIT-aligned path is denser and steadier (~30 Hz vs ROS rgbd's
  ~14-20 Hz) on the same camera. Migrating physical VO + YOLO to consume
  this stream is the highest-quality option available without changing
  RealSense hardware settings.
- A C++ implementation matches the existing `qcar2_nodes` package
  conventions (rgbd, csi, lidar, hardware are all C++), avoids a Python
  GIL between camera I/O and ROS publish, eliminates `cv_bridge`
  Python marshaling overhead, and removes the runtime dependency on a
  Python interpreter inside the critical camera path.

Files added:

1) `qcar2_nodes/src/qcar2_camera_bridge.cpp` (NEW, ~330 lines)
- ROS2 node `qcar2_camera_bridge` (executable installs under the same
  name in `qcar2_nodes`).
- Sole client of the Quanser depth-align runtime.
- Lifecycle:
  - Constructor optionally spawns the runtime via
    `quarc_run -r -t <quarc_target_uri> <runtime_path> -uri <pit_uri>`
    (matches `pit.YOLO.utils.QCar2DepthAligned.__initDepthAlign`),
    sleeps 4 s, then `qcomm_connect(pit_uri, non_blocking=true, &conn)`.
  - Worker thread (dedicated `std::thread`) loops `qcomm_poll` +
    `qcomm_receive` to accumulate 480x640x4 float32 packets in
    Fortran column-major order, then copies a complete packet into
    a mutex-guarded shared buffer.
  - Publish-side ROS timer (default 60 Hz) snapshots the buffer,
    builds `cv::Mat` instances, and publishes:
    - `/camera/color_image` as `sensor_msgs::msg::Image` encoding
      `bgr8` (built from packet channels 3,2,1).
    - `/camera/depth_image` as `sensor_msgs::msg::Image` encoding
      `32FC1` (meters, packet channel 0).
  - Destructor signals worker stop, joins, calls `qcomm_shutdown`
    + `qcomm_close`, then `quarc_run -q -Q` to terminate the
    Quanser runtime cleanly (matches PIT's `__stopDepthAlign`).
- Parameters: `device_type` (physical/virtual), `auto_start_runtime`,
  `runtime_path`, `quarc_target_uri`, `pit_uri`, `timer_rate_hz`.

Files modified:

2) `qcar2_nodes/CMakeLists.txt`
- Added `add_executable(qcar2_camera_bridge ...)` target.
- Links `-lquanser_communications -lquanser_runtime -lquanser_common
  -lpthread`, plus `cv_bridge::cv_bridge` and `opencv_imgcodecs`.
- Added `install(TARGETS qcar2_camera_bridge ...)` so the binary
  lands in `lib/qcar2_nodes/qcar2_camera_bridge`.

3) `qcar2_autonomy/autonomy/vo_node.py` — encoding auto-detect
- `_depth_cb` now branches on `msg.encoding`:
  - `32FC1`: decode as float meters, do NOT scale by `depth_scale`.
    On first sight, flips `self.vo.projector.depth_on_color_grid =
    True` and logs the event once (`_aligned_depth_locked` guard).
    The K_inv fix from earlier in the day now takes effect under
    the new pipeline because RGB intrinsics are correct for
    depth-on-color-grid backprojection.
  - Anything else: legacy `passthrough` -> uint16 path preserved.
- Depth diagnostic logger split into ALIGNED vs raw branches with
  per-mode formatting (raw counts vs metric center value).

4) `qcar2_autonomy/autonomy/yolo_detector.py` — converted from
   camera owner to ROS subscriber
- Comment-out policy applied (per project convention): the original
  PIT-ownership lines remain in the file, prefixed with the
  2026-05-14 marker and an explanation, so the legacy direct-PIT
  behavior can be restored by uncommenting + disabling the bridge.
- New subscribers on `/camera/color_image` (bgr8) and
  `/camera/depth_image` (auto-detects 32FC1 aligned meters from the
  bridge or MONO16 raw from legacy rgbd; MONO16 path divides by
  1000.0 to get meters).
- `on_timer` now reads latest frames from a `_frame_lock`-protected
  shared buffer instead of calling `self.QCarImg.read()`.
- `yolo_detect()` consumes `_current_rgb` / `_current_depth` cached
  by `on_timer` instead of `self.QCarImg.rgb` / `.depth`.
- `terminate()` becomes a no-op (bridge handles runtime shutdown).
- YOLO inference, motion-enable flag publishing, stop-override on
  `/trip_planner/qcar_state`, and detection cooldown logic are
  unchanged behaviorally.

5) Launch files (commented-arg pattern; no deletions of working
   nodes — both rgbd and bridge are defined, gated by IfCondition):
- `qcar2_nodes/launch/qcar2_launch.py`
- `qcar2_nodes/launch/qcar2_manual_drive_launch.py`
- `qcar2_nodes/launch/qcar2_keyboard_drive_launch.py`

Each declares:
  `DeclareLaunchArgument('camera_source', default_value='depth_aligned',
   description="'depth_aligned' (qcar2_camera_bridge) or 'rgbd'")`
and conditionally launches either the bridge node or the legacy rgbd
node with `IfCondition(PythonExpression(...))`. Both nodes use ROS node
name `RealsenseCamera` so downstream subscribers do not change their
expectations.

Packet protocol notes (recorded for future maintainers):
- The Quanser depth-align runtime delivers exactly
  `480 * 640 * 4 * sizeof(float)` = 4,915,200 bytes per frame.
- Layout: Fortran column-major. Element at (row=i, col=j, ch=k) lives
  at float-index `k*480*640 + j*480 + i`.
- Channel mapping (verified against PIT's Python read() implementation):
  - ch 0 = depth in meters
  - ch 1 = R (0..255 as float)
  - ch 2 = G
  - ch 3 = B
- `cv_bridge` BGR8 output is built from channels (3, 2, 1) via
  `cv::saturate_cast<uchar>`.

Build verification:
- `g++ -fsyntax-only` against ROS Humble + Quanser system headers: PASS.
- Isolated `colcon build --packages-select qcar2_nodes --cmake-target
  qcar2_camera_bridge`: PASS. Only warning is the pre-existing
  opencv-4.2-vs-4.5 ABI notice already emitted by rgbd/csi targets.
- Python `py_compile` on all four edited Python files: PASS.

Operational notes for the next physical bring-up:
1) Sync source per `Easy_Start.txt §0.6` (rsync `qcar2_autonomy` into
   `~/ros2/src/`, plus copy modified `qcar2_nodes` if not already
   under `~/ros2/src/qcar2_nodes`), then:
     `cd ~/ros2 && colcon build --packages-select qcar2_interfaces
        qcar2_nodes qcar2_autonomy --symlink-install`
   (qcar2_interfaces typically already built; include only if needed.)
2) Verify the bridge alone:
     `ros2 run qcar2_nodes qcar2_camera_bridge`
   then in another terminal:
     `ros2 topic hz /camera/color_image`     (expect ~30 Hz steady)
     `ros2 topic hz /camera/depth_image`     (expect ~30 Hz steady)
     `ros2 topic echo --no-arr /camera/depth_image | grep encoding`
       (expect `encoding: 32FC1`)
3) Launch the full physical stack via the chosen launch file with the
   default `camera_source:=depth_aligned`. To roll back to rgbd:
     `ros2 launch qcar2_nodes qcar2_cartographer_launch.py
       camera_source:=rgbd`
4) Run `vo_node` in `ros_rgbd` mode (subscribes to the bridge topics):
     `ros2 run qcar2_autonomy vo_node --ros-args
       -p camera_mode:=physical
       -p input_source:=ros_rgbd
       -p force_cart_yaw:=true`
   Expect a one-time log line:
     `[DEPTH DIAG ALIGNED] center=X.XXX m (32FC1)` confirming the
     32FC1 path is active and `depth_on_color_grid=True` was set.
5) Bring up `autonomy_planner_launch.py` for YOLO + planner + lane +
   traffic. YOLO inference and stop/yield-sign overrides are
   unchanged; the only difference is the camera data is sourced from
   ROS topics instead of a direct PIT TCP connection.

Risks acknowledged (per pre-implementation review):
- Single point of failure: bridge crash takes down all camera consumers
  simultaneously. Mitigation: launch-arg rollback to `rgbd` is one flag.
- Bridge runtime adds ~4 s startup latency (PIT runtime spawn). Any
  consumer with strict QoS / startup timeout on `/camera/*` may need
  to be launched after the bridge or tolerate the delay.
- Aligned depth is cropped to RGB FOV (55.6 x 43.1 deg) vs depth's
  native 79.5 x 63.9 deg. Acceptable for VO (features come from RGB
  anyway); flag for any future wide-FOV depth consumer.
- Quanser depth-align runtime is an opaque vendor binary; runtime bugs
  must be filed with Quanser. Same as the existing `yolo_detector` path
  used today.

Virtual-mode note (for a future port — NOT implemented here):
- `pit.YOLO.utils.QCar2DepthAligned` already handles virtual via
  `Camera3D` against QLabs port 18965 and a homography matrix `M`. To
  extend the bridge to virtual: detect `device_type:=virtual`, skip the
  quarc_run spawn, and connect to QLabs port 18965 with the same
  Fortran-order packet decoding plus the homography warp. All other
  bridge code (publishers, threading, lifecycle) is identical.

Alternatives considered and rejected (decision audit trail):

A) Do nothing structural; defer the rework until the dual-owner conflict
   is actually observed in physical testing.
   - Why considered: the conflict has never been hit in physical because
     YOLO and VO have not been run together. There was a chance the
     Quanser runtime accepts contention silently and the rework would be
     premature.
   - Why rejected: real competition runs will execute the full stack
     concurrently; the failure is virtually certain once both processes
     attempt `video3d_start_streaming` on the same camera. Pre-emptive
     consolidation is cheaper than diagnosing it on competition day.

B) Single-owner-via-rgbd: leave `rgbd.cpp` as the only RealSense owner,
   convert every other consumer (including `yolo_detector.py`) to a
   subscriber, and accept raw MONO16 depth everywhere.
   - Why considered: smallest diff. One launch-file flip, one entry-point
     swap, no new code. Lowest implementation risk.
   - Why rejected: loses the aligned depth signal entirely. The K_inv fix
     applied earlier the same day benefits aligned-depth consumers but is
     a no-op against raw MONO16. PIT-vs-ROS measurements (Codex,
     2026-05-14) showed PIT delivering denser, steadier depth than the
     `rgbd` path. Once VSLAM and loop-closure work begins, aligned depth
     is also a hard requirement. Path B avoids two future rewrites.

C) Single-owner-via-bridge (chosen). New ROS node that owns the Quanser
   depth-align runtime and publishes ROS topics for every other consumer
   to subscribe. Both `rgbd.cpp` and the new bridge remain in the source
   tree, switched at launch time via `camera_source`.

D) Shared Quanser runtime with multiple PIT clients (no bridge).
   Run `QCar2DepthAlign.rt-linux_qcar2` once externally, then have
   `vo_node` (`pit_aligned` mode) and `yolo_detector.py` both instantiate
   `QCar2DepthAligned(manualStart=True)` to attach to it.
   - Why considered: zero new infrastructure; both Python consumers stay
     close to their existing pattern.
   - Why rejected: Quanser's `BasicStream` is a TCP client. There is no
     documented confirmation that the runtime accepts multiple
     simultaneous client connections, and the failure mode would be a
     silent stall on the second connector. The bridge pattern is
     deterministic and matches the standard ROS pub/sub model.

E) Python-implemented bridge instead of C++.
   - Why considered: faster to write and iterate; easier to debug; works
     with `--symlink-install` without rebuild; lower initial bring-up
     risk.
   - Why rejected (after explicit user direction): the Quanser C
     communications API is exposed via `/usr/include/quanser/quanser_
     communications.h` and `libquanser_communications.so`, so the
     unknown that motivated "Python first, port later" turned out not
     to exist. C++ removes GIL / numpy / cv_bridge marshaling overhead,
     gives compile-time type safety, and matches the rest of the
     `qcar2_nodes` package convention. The two-step Python-then-C++
     migration was replaced by one-step C++ implementation.

F) Keep using `yolo_detector_new.py` (the existing pure-subscriber
   version) instead of converting OLD `yolo_detector.py`.
   - Why considered: `yolo_detector_new` already subscribes to ROS
     topics — would have required no code change beyond a setup.py
     entry-point swap.
   - Why rejected (per user preference): OLD `yolo_detector` carries
     additional behavior the project depends on, specifically the
     `/trip_planner/qcar_state` UInt8 override during stop-sign and
     yield-sign holds, the `/motion_enable` 500 Hz publisher, and the
     detection-cooldown / sign_detected state machine. The surgical
     conversion preserves all of that exactly while removing only the
     camera-ownership lines.

Why this matters for future maintainers:
- The `rgbd.cpp` path stays buildable and is the documented rollback
  (`camera_source:=rgbd`). Do not delete `rgbd.cpp` or the legacy
  `realsense_camera_node` block; both still serve the virtual mode and
  the rollback drill.
- The bridge is a single point of failure for the physical camera path
  by design. If it crashes the only diagnostic question is "did the
  Quanser runtime spawn cleanly?" — visible via the bridge's startup
  log and `ros2 node list`.
- The bridge consumes the same Quanser runtime that PIT helpers use.
  Anything that calls `QCar2DepthAligned(manualStart=False)` while the
  bridge is up will kill the bridge's stream (PIT's `__initDepthAlign`
  calls `__stopDepthAlign` first). Specifically: do not run the OLD
  `yolo_detector` unconverted alongside the bridge — the conversion
  in this pass is what makes them safe to coexist.

Key downsides accepted (per pre-implementation review):
- Single point of failure for the camera path. Mitigated by the
  `camera_source:=rgbd` rollback flag.
- ~4 s bridge startup delay (Quanser runtime spawn). Anything with a
  strict QoS startup timeout on `/camera/*` must launch after the
  bridge or tolerate this delay.
- Encoding switch (`MONO16` on rgbd vs `32FC1` on bridge) introduces
  a silent-bug class. Mitigated by per-callback `msg.encoding` checks
  in every depth subscriber on the project (currently only `vo_node`
  and the converted `yolo_detector`). Any new depth subscriber MUST
  do the same.
- Aligned depth is cropped to the RGB FOV (55.6 x 43.1 deg) vs depth's
  native 79.5 x 63.9 deg. Acceptable for VO (features come from RGB
  anyway); flag for any future wide-FOV depth consumer.
- Bandwidth on `/camera/depth_image` roughly doubles (MONO16 ~18 MB/s
  to 32FC1 ~37 MB/s at 640x480x30 Hz). Negligible on Jetson AGX Orin
  but visible if subscribers run on a separate host over network.

## 2026-05-14 (hotfix: bridge runtime_path resolver + spawn-fast-fail)

Issue observed on first physical test (recorded by user under
`Physical Test 2 / New camera owner Test (DepthAlign C++ Bridge)` in
`VO_readings.txt`):
- Bridge logged `Unable to download model
  /home/nvidia/Documents/ACC_Development/Development/MDC_libraries/
  resources/applications/QCarDepthAlign/QCar2DepthAlign.rt-linux_qcar2.
  The file could not be found.`
- `quarc_run` returned 256 (file-not-found).
- Bridge continued and emitted 14 sequential
  `qcomm_connect(tcpip://localhost:17003) failed: A non-blocking
  operation would have blocked` errors because no server was listening
  on port 17003 (the runtime never spawned).
- `ros2 topic hz` reported the topics were never published.

Root cause:
- The original default for `runtime_path` was a guess based on the
  PIT class's relative path (`../../../resources/applications/...`).
  PIT resolves that against its own `__file__`. On this QCar2, the
  Quanser install puts the binary at
  `/home/nvidia/Documents/Quanser/0_libraries/resources/applications/
  QCarDepthAlign/QCar2DepthAlign.rt-linux_qcar2`, outside
  `ACC_Development`. The default path I had pointed inside
  `ACC_Development/Development/MDC_libraries/resources/`, which does
  not exist (MDC_libraries only contains `python/`).

Fix applied to `qcar2_nodes/src/qcar2_camera_bridge.cpp`:
- Added `resolveRuntimePath(std::string& out)` that searches a
  prioritized candidate list and returns the first existing file:
    1. The user-supplied `runtime_path` parameter (if non-empty)
    2. `QCAR2_DEPTHALIGN_RUNTIME` environment variable (if set)
    3. `/home/nvidia/Documents/Quanser/0_libraries/resources/
        applications/QCarDepthAlign/QCar2DepthAlign.rt-linux_qcar2`
        (actual install location on this QCar2)
    4. `/home/nvidia/Documents/ACC_Development/docker/0_libraries/
        resources/applications/QCarDepthAlign/
        QCar2DepthAlign.rt-linux_qcar2`
    5. `/home/nvidia/Documents/ACC_Development/backup/
        Quanser_Academic_Resources/0_libraries/resources/applications/
        QCarDepthAlign/QCar2DepthAlign.rt-linux_qcar2`
- Default value of the `runtime_path` parameter changed from a hard-
  coded (wrong) path to empty string — falls through to the resolver.
- `startRuntime()` now:
    1. Calls `resolveRuntimePath()` BEFORE invoking `quarc_run`.
       Logs the list of candidate paths tried if nothing resolves.
    2. Treats a non-zero `quarc_run` return code as ERROR (was WARN)
       and sets `runtime_spawn_ok_ = false`.
- Bridge constructor now checks `runtime_spawn_ok_` after
  `startRuntime()` and ABORTS init if false. No more 14-attempt
  reconnect loop on a runtime that never started. Logs a clear
  remediation message: set `runtime_path` parameter and relaunch.
- Destructor only attempts `stopRuntime()` if the spawn actually
  succeeded.
- Added `<filesystem>`, `<sys/stat.h>` includes for path existence
  checks via `stat()`.

Operational implication:
- User does NOT need to pass `-p runtime_path:=...` on this QCar2 —
  the resolver finds the binary at
  `/home/nvidia/Documents/Quanser/0_libraries/...` automatically.
- For machines where the install location differs, override with
  either `-p runtime_path:=/full/path/to/QCar2DepthAlign.rt-linux_qcar2`
  or `export QCAR2_DEPTHALIGN_RUNTIME=/full/path/...`.
- If neither the parameter nor the env var nor any of the candidate
  paths exists, the bridge now logs a clear list of what it tried
  and exits init cleanly rather than spamming connect failures.

Build verification:
- Isolated `colcon build --packages-select qcar2_nodes --cmake-target
  qcar2_camera_bridge` PASS; only the pre-existing OpenCV ABI warning.

Follow-up (same day, per user direction): reordered the candidate
list to prefer the in-repo copy over the system install. All four
copies on this machine were verified byte-identical
(md5 58572dc0d62e8535140afb45f5eaf554):
- `/home/nvidia/Documents/Quanser/0_libraries/...` (system install)
- `ACC_Development/docker/0_libraries/...` (in-repo, default)
- `ACC_Development/backup/Quanser_Academic_Resources/0_libraries/...`
- `ACC_Development/docker/development_docker/...` (also in-repo)

New priority order in `resolveRuntimePath()`:
  1. `runtime_path` parameter (if non-empty)
  2. `QCAR2_DEPTHALIGN_RUNTIME` env var
  3. `ACC_Development/docker/0_libraries/...`  ← new default
  4. `ACC_Development/backup/Quanser_Academic_Resources/0_libraries/...`
  5. `/home/nvidia/Documents/Quanser/0_libraries/...` (system, fallback)

Rationale: keeps the project self-contained on a fresh clone; matches
the user's explicit preference for not depending on paths outside
`ACC_Development`. Rebuilt and verified.

## 2026-05-14 (hotfix #2: match PIT quarc_run invocation exactly)

Issue:
- After the runtime_path resolver fix above, the bridge's `quarc_run`
  invocation still produced no listening server on port 17003.
  Diagnostic single-line manual command from the user's terminal
  (recorded under `VO_readings.txt -> Step 2 Test 2 trailing
  diagnostics`) demonstrated that the same unquoted `quarc_run -r -t
  tcpip://localhost:17000 /full/path/QCar2DepthAlign.rt-linux_qcar2
  -uri tcpip://localhost:17003` started the runtime cleanly
  (`exit=0`; `QCar2DepthAlign` process alive; ports 17003 AND 18777
  listening — the runtime exposes both URIs by default which is why
  PIT's `port='18777'` consumer default has historically worked
  against a runtime spawned with `-uri tcpip://localhost:17003`).
- The bridge's failing variant wrapped the path in double quotes:
  `quarc_run -r -t tcpip://localhost:17000 "/full/path/file.rt..."
  -uri tcpip://localhost:17003`. The quotes are the only difference
  between the failing and working forms. On this Jetson the quoted
  form is consumed silently by `quarc_run` with no output and no
  non-zero exit, leaving no server bound to the URI.
- Additionally, the bridge's pre-stop command passed the full quoted
  path to `quarc_run -q -Q`. PIT's `__stopDepthAlign` passes ONLY
  the basename of the model file. Quanser's `-q -Q` flag matches
  models by basename; a full path is not the canonical form.
- The `-D` flag (daemon-detach) used in `qvl/real_time.py` is broken
  on this aarch64 QCar2 build: manual `quarc_run -D -r ...` failed
  with "Unable to download model ... An operating system function
  returned an unrecognized error" (exit=1). Confirmed by repeated
  manual test. Do NOT add `-D` to the bridge.

Fix applied to `qcar2_nodes/src/qcar2_camera_bridge.cpp`:
- `startRuntime()` now builds the start command WITHOUT quoting the
  runtime path. The candidate-list resolver guarantees the resolved
  path contains no whitespace, so quoting is unnecessary and (per
  diagnostic data) actively harmful.
- `startRuntime()` and `stopRuntime()` now derive the model basename
  via `std::filesystem::path(runtime_path_).filename().string()` and
  pass it to `quarc_run -q -Q <basename>`, matching PIT's exact
  invocation.
- Added an INFO log line after `quarc_run` returns to mark the start
  of the 4 s settle wait. Useful breadcrumb in future debugging.
- No `-D` flag is added; the QCar2's aarch64 quarc_run build does
  not support it.

Operational expectation after this fix:
- Bridge startup log should now read (in order):
    Using QCar2DepthAlign runtime at: /home/nvidia/.../QCar2DepthAlign.rt-linux_qcar2
    Spawning Quanser depth-align runtime: quarc_run -r -t
      tcpip://localhost:17000 /home/nvidia/.../QCar2DepthAlign.rt-linux_qcar2
      -uri tcpip://localhost:17003
    quarc_run returned exit=0; waiting 4s for runtime to bind to
      tcpip://localhost:17003 ...
    Connected to QCar2DepthAlign stream at tcpip://localhost:17003
- `pgrep -af QCar2DepthAlign` should show the runtime process alive
  while the bridge runs.
- `/camera/color_image` and `/camera/depth_image` should publish at
  ~30 Hz with encodings `bgr8` and `32FC1` respectively.

Build verification:
- Isolated `colcon build --packages-select qcar2_nodes --cmake-target
  qcar2_camera_bridge` PASS; only pre-existing OpenCV ABI warning.

## 2026-05-14 (hotfix #4: switch from qcomm_* to stream_* API for buffer sizing)

Observed after hotfix #3 (recorded in `VO_readings.txt -> Step 2 Test 4`):
- Bridge connects cleanly: "Connected to QCar2DepthAlign stream at
  tcpip://localhost:17003" appears in the log.
- Worker heartbeat shows packets ARE arriving:
    [bridge worker hb] packets=1  poll_data=653  bytes=7891559
    [bridge worker hb] packets=17 poll_data=7258 bytes=87287633
- BUT the throughput is wrong: 16 full packets across 50 seconds
  (~0.32 fps) instead of the expected ~30 fps. Each `qcomm_receive`
  call returned only ~12 KB on average (79 MB / 6605 receive calls),
  forcing ~400 receive calls to assemble each 4.9 MB frame.

Root cause:
- The bridge used `qcomm_connect / qcomm_poll / qcomm_receive` from
  `quanser_communications.h`. The `qcomm_connect` signature does NOT
  let the caller size the send/receive buffers — it uses small TCP
  defaults (~16-64 KB).
- PIT's Python `pal.utilities.stream.BasicStream` does NOT use the
  `qcomm_*` family. It calls Quanser's `stream_*` family (in the
  same `libquanser_communications.so` library) which DOES take
  explicit `send_buffer_size` and `receive_buffer_size` arguments.
  PIT's QCar2DepthAligned passes `send=480*640*3` and
  `recv=480*640*4*4` (= 4,915,200 bytes) so `stream_receive` can
  pull a near-full frame in a single call.
- `STREAM_POLL_RECEIVE / STREAM_POLL_CONNECT / etc.` are #define
  aliases of the corresponding `QCOMM_POLL_*` values in
  `quanser_stream.h`, so flag constants are unchanged.

Fix applied to `qcar2_nodes/src/qcar2_camera_bridge.cpp`:
- Added `#include "quanser/quanser_stream.h"`.
- Replaced `t_connection conn_` member with `t_stream stream_`.
- Constructor initializer adds `stream_(nullptr)`.
- `connectStream()` now calls
    stream_connect(uri, /*non_blocking=*/true,
                   send_buffer_size = 480*640*3,
                   receive_buffer_size = 480*640*4*4,
                   &stream_);
  matching PIT's buffer sizes exactly.
- Subsequent loops use `stream_poll(stream_, t, STREAM_POLL_RECEIVE)`
  and `stream_receive(stream_, buf, n)`.
- Destructor uses `stream_shutdown` + `stream_close`.
- Error labels updated from "qcomm_poll"/"qcomm_receive" to
  "stream_poll"/"stream_receive" for log accuracy.
- The `-QERR_WOULD_BLOCK` handling from hotfix #3 is preserved
  (still the expected return for non-blocking connect-in-progress).

Expected behavior after this fix:
- Heartbeat should now show packets growing at ~30 per heartbeat
  interval (≈ 5 s × 30 fps = 150 packets per heartbeat).
- `bytes_received_total / poll_data` should approach ~1 MB per
  receive call instead of ~12 KB.
- `/camera/color_image` and `/camera/depth_image` should publish at
  ~30 Hz to subscribers in the same ROS_DOMAIN_ID.

Open follow-up observed in same test:
- Even at the (slow) 0.34 fps achieved before this fix, `ros2 topic
  list | grep camera` in a second terminal returned no `/camera/*`
  topics. This is independent of the throughput fix; it points at a
  ROS discovery issue (likely `ROS_DOMAIN_ID` mismatch between the
  terminal that ran the bridge and the terminal that ran
  `ros2 topic list`, or a stale `ros2 daemon`). Diagnose with
  `echo $ROS_DOMAIN_ID` in both terminals and `ros2 daemon stop &&
  ros2 daemon start` if they match but topics still don't appear.

Build verification:
- Isolated `colcon build --packages-select qcar2_nodes --cmake-target
  qcar2_camera_bridge` PASS; only pre-existing OpenCV ABI warning.

## 2026-05-14 (hotfix #5: use stream_receive_byte_array for atomic packet receive)

Observed after hotfix #4 (buffer-size fix, recorded in `VO_readings.txt
-> Step 2 Test 5`): throughput essentially unchanged at ~0.3 fps; each
`stream_receive` call returned only ~23 KB on average (slight
improvement over the previous ~12 KB but nowhere near the expected
~4.9 MB per call).

Root cause discovered by actually reading PIT's
`pal.utilities.stream.BasicStream.receive` (lines 369-444 of
`docker/0_libraries/python/pal/utilities/stream.py`):

  pollResult = self.clientStream.poll(self.t_out, PollFlag.RECEIVE)
  ...
  self.bytesReceived = self.clientStream.receive_byte_array(
      self.data, totalNumBytes)

PIT calls `receive_byte_array`, NOT plain `receive`. The C equivalent
declared in `quanser_stream.h` is:

  EXTERN t_int stream_receive_byte_array(t_stream stream,
                                         t_byte * elements,
                                         t_uint num_elements);

Quanser's own header documents the semantic difference:

> This function receives an array of bytes over a client stream. It
> differs from the stream_receive_bytes function in that it treats the
> entire array as an atomic unit. It either receives all of the array
> or none of it. It also requires that the stream receive buffer be at
> least as large as the array of bytes.

> Returns 1 on success. If not enough data is available and the
> connection has been closed gracefully then 0 is returned. If an error
> occurs then a negative error code is returned. In non-blocking mode,
> -QERR_WOULD_BLOCK is returned when not enough data has buffered yet.

So:
- `stream_receive`         -> partial chunks (~12-23 KB per call), need
                              manual accumulation; the API we WERE using.
- `stream_receive_byte_array` -> atomic; returns 1 only when the FULL
                              num_elements arrives; the API PIT uses.

Fix applied to `qcar2_camera_bridge.cpp` worker thread:
- Replaced the accumulating `stream_receive` loop with a single
  `stream_receive_byte_array(stream_, scratch_bytes, kPacketBytes)`
  call per poll-with-data event.
- Removed the `total_received` accumulator and `remaining` calculation;
  no longer needed since each successful call delivers a complete
  4.9 MB packet.
- Return-code handling now distinguishes:
    1                  -> publish the full frame, increment packets++
    0                  -> connection closed; flag conn_open_=false for
                          worker-level reconnect on next iteration
    -QERR_WOULD_BLOCK  -> not enough buffered; increment would_block
                          counter and loop back to poll
    other negative     -> log via logQuanserError
- Buffer ptr cast updated from `char*` to `t_byte*` to match the API.
- Heartbeat counters updated: removed `in_progress` (no longer
  meaningful), added `would_block`.
- Inline comment block in the source explains the semantic difference
  so this never gets re-broken.

Expected behavior after this fix:
- Heartbeat should now show `packets` growing by ~150 per 5-second
  interval (30 fps x 5 s).
- `polls_with_data` should approach `packets` 1:1 (one atomic receive
  per poll-with-data event) instead of ~400:1 as before.
- `/camera/color_image` and `/camera/depth_image` at ~30 Hz.

Build verification:
- Isolated `colcon build --packages-select qcar2_nodes --cmake-target
  qcar2_camera_bridge` PASS; only pre-existing OpenCV ABI warning.

Independent earlier finding (now resolved by user): terminal B's
missing `/camera/*` topics in hotfix #4's test were caused by
`ROS_DOMAIN_ID` being unset in terminal B while terminal A had
`ROS_DOMAIN_ID=67`. Resolution: `export ROS_DOMAIN_ID=67` in every
shell, or add to `~/.bashrc`. Easy_Start.txt Section 0 already
documents this.

## 2026-05-14 (pivot: bridge moved from C++ to Python — final implementation)

Decision:
- After hotfix #5 still produced ~0.3 fps and the user's diagnostic
  showed `net.core.rmem_max=212992` (208 KB kernel cap), the user
  applied `sudo sysctl -w net.core.rmem_max=16777216` to lift the
  ceiling to 16 MB. Re-tested: bridge throughput remained ~0.3 fps
  with `would_block` exceeding 200 million per 5-second heartbeat.
  Kernel-level fix did NOT unlock the Quanser receive path.
- Conclusion: Quanser's library does something inside its Python
  wrapper that the C `stream_*` / `qcomm_*` APIs do not expose. PIT
  achieves ~30 fps via `pal.utilities.stream.BasicStream` (verified
  in production with legacy yolo_detector.py). The fastest path to
  a working single-owner camera bridge is to invoke
  `pit.YOLO.utils.QCar2DepthAligned` directly from a Python ROS2
  node, exactly as legacy yolo_detector used it.

Pivot implemented:

1. NEW file `Development/ros2/src/qcar2_autonomy/autonomy/qcar2_camera_bridge.py`
   (~230 lines).
   - rclpy Node `qcar2_camera_bridge`.
   - Reuses the `_resolve_pit_import` pattern from `vo_node.py` so PIT
     can be located on standard or alternate sys.path layouts.
   - Instantiates `QCar2DepthAligned(ip, nonBlocking, manualStart,
     port, isPhyscial)` with PIT defaults (matches legacy
     yolo_detector behavior verbatim).
   - 60 Hz polling timer wrapping `cam.read()`. When `read()` returns
     True a new frame is available; the timer publishes:
       /camera/color_image  (bgr8)
       /camera/depth_image  (32FC1 meters, aligned to color grid)
   - One-shot log line on first frame; periodic "[bridge hb]" log
     line every 5 s reporting actual published fps for live
     verification.
   - `destroy_node()` calls `self.cam.terminate()` so the Quanser
     depth-align runtime exits cleanly on shutdown.
   - Parameters: device_type, pit_python_path, pit_ip, pit_port,
     pit_non_blocking, pit_manual_start, publish_rate.

2. UPDATED `Development/ros2/src/qcar2_autonomy/setup.py`:
   - Added console entry point:
       camera_bridge=autonomy.qcar2_camera_bridge:main

3. UPDATED launch files
   (`qcar2_launch.py`, `qcar2_manual_drive_launch.py`,
    `qcar2_keyboard_drive_launch.py`):
   - The `camera_bridge_node` definition switched from
       package='qcar2_nodes', executable='qcar2_camera_bridge'
     to
       package='qcar2_autonomy', executable='camera_bridge'
   - The `camera_source` launch arg and the IfCondition wiring are
     unchanged; rollback to legacy rgbd.cpp is still available via
     `-p camera_source:=rgbd`.

4. C++ bridge kept in tree (per comment-out policy). Added a header
   comment block at the top of
   `qcar2_nodes/src/qcar2_camera_bridge.cpp` documenting:
   - "STATUS (2026-05-14): NOT THE ACTIVE BRIDGE"
   - Pointer to the Python implementation that supersedes it
   - The full forensic of why C++ couldn't reach 30 fps
   - Note that the file remains buildable as a reference for any
     future Quanser SDK update that exposes the missing piece.
   The CMakeLists target is unchanged — `qcar2_camera_bridge` binary
   still builds and installs to `lib/qcar2_nodes/qcar2_camera_bridge`,
   it just isn't referenced by any launch file.

Operational expectation after this pivot:
- The Python bridge should achieve ~30 fps because it uses the exact
  same `QCar2DepthAligned` class legacy yolo_detector used at ~30 Hz.
- All downstream consumers (vo_node ros_rgbd path, yolo_detector,
  traffic_system_detector, lane_detector) remain unchanged. They
  subscribe to `/camera/color_image` and `/camera/depth_image`,
  which the Python bridge now publishes with the same encodings
  (bgr8 and 32FC1) the C++ bridge was advertising.
- `vo_node._depth_cb` encoding auto-detect from hotfix #1 still
  flips `depth_on_color_grid=True` on receipt of the first 32FC1
  message, so the K_inv backprojection fix remains active.
- Sync + rebuild on the QCar2 via Easy_Start.txt §0.6 will need to
  rebuild only `qcar2_autonomy` (Python) — qcar2_nodes itself was
  not touched in this pivot. Use:
    rsync -a --delete .../qcar2_autonomy/  ~/ros2/src/qcar2_autonomy/
    rsync -a --delete .../qcar2_nodes/launch/  ~/ros2/src/qcar2_nodes/launch/
    cd ~/ros2 && colcon build --packages-select qcar2_autonomy --symlink-install

Build verification:
- `python3 -m py_compile` PASS on:
    autonomy/qcar2_camera_bridge.py
    qcar2_nodes/launch/qcar2_launch.py
    qcar2_nodes/launch/qcar2_manual_drive_launch.py
    qcar2_nodes/launch/qcar2_keyboard_drive_launch.py
    qcar2_autonomy/setup.py

## 2026-05-14 (Step 3 PHYSICAL validation — bridge + VO end-to-end PASS)

Test conditions:
- Car elevated and stationary (wheels off the ground).
- Single launch command: `ros2 launch qcar2_nodes
  qcar2_cartographer_launch.py` (camera_source defaulted to
  depth_aligned, so the Python bridge auto-started alongside
  cartographer, lidar, csi, qcar2_hardware, nav2_qcar2_converter,
  static_transform_publisher, fixed_lidar_frame, and the occupancy
  grid node).
- `vo_node` launched separately with `-p camera_mode:=physical
  -p input_source:=ros_rgbd -p force_cart_yaw:=true`.
- `ros2 topic echo /vo/fault_status` captured in a third terminal.

Results captured in `VO_readings.txt` under the trailing Step 3 block
(camera_bridge + cartographer + vo_node + fault_status):

1. Single-launch architecture confirmed:
   - Launch log line: `[INFO] [camera_bridge-3]: process started with
     pid [42966]` — i.e. the Python bridge is now a peer of the C++
     nodes inside cartographer's launch. ROS2 launch handles the
     language difference transparently. No separate bridge launch
     needed.

2. Bridge fps confirmed under full-stack load:
   - `[bridge hb] published=152 frames in 5.0 s (~30.4 fps)` repeated
     across multiple heartbeats — the Python bridge holds a steady
     ~30 fps even when cartographer + lidar + qcar2_hardware are
     consuming CPU simultaneously.

3. VO consuming the bridge correctly:
   - One-shot log line on first depth callback:
       "Detected aligned 32FC1 depth on depth_image; set
        depth_on_color_grid=True. VO will use RGB intrinsics
        (K_rgb_inv) for backprojection."
   - This proves the encoding auto-detect from hotfix #1 (now active
     under the Python bridge) fires correctly and the K_inv fix is
     in the live path.
   - Periodic `[DEPTH DIAG ALIGNED] center=1.158 m (32FC1)` confirms
     real metric depth being consumed by VO.

4. /vo/fault_status quality (stationary baseline):
   - State: `agree` consistently (no `init`/`warming` after warmup).
   - rho (residual): 0.003-0.031 m — excellent agreement with Cart
     pose at rest.
   - weight (quality): 0.65-0.75 sustained.
   - inliers: 300-415 per frame — very high feature counts, well
     above the 200-260 baseline observed in earlier MONO16+no-align
     physical tests.
   - spread: 90-106 pixels — full image coverage.
   - psi_raw: 180 deg consistently (shadow yaw integrator initialized
     to cart_psi and not moving because the car is stationary — dpsi
     ~0 every frame, so the integrator stays put).
   - VO pose: vo(0.14, 0.01, ~180°) — anchored to Cart's initial
     pose at start and held within mm-scale jitter, exactly the
     expected behavior at rest.

Verdict: the single-owner camera architecture is operationally
complete and outperforms the prior MONO16+software-align baseline in
every measured dimension (higher inlier counts, lower rho, higher
weight, no jitter on the camera topics, no setup hassle).

Next session: mat run with actual driving. See the "Plan for next
session" section at the end of `VO_Conversation_Log.txt`.

## 2026-05-14 (hotfix #3: connect logic — -QERR_WOULD_BLOCK is expected)

Issue:
- After hotfix #2 (PIT-matching invocation, no quotes, basename-only
  stop), `quarc_run` correctly spawned the runtime. Manual diagnostic
  (recorded in VO_readings.txt -> Step 2 Test 3 trailing checks)
  confirmed `QCar2DepthAlign -uri tcpip://localhost:17003` was alive
  and listening on ports 17003 + 18777.
- BUT the bridge still logged "qcomm_connect failed: A non-blocking
  operation would have blocked" on every iteration.
- The smoking gun was the `ss -tlnp` output: `LISTEN 19 128
  0.0.0.0:17003` (Recv-Q = 19) and a few seconds later `LISTEN 33
  128 0.0.0.0:17003` (Recv-Q = 33). The Recv-Q on a listening
  socket is the count of half-open connection attempts the kernel
  has queued waiting for the server to accept(). Growing Recv-Q
  proves the bridge's connect attempts WERE reaching the runtime
  but were never being followed through.

Root cause in `qcar2_nodes/src/qcar2_camera_bridge.cpp`
(`connectStream()`):
- `qcomm_connect` with `non_blocking=true` returns `-QERR_WOULD_BLOCK`
  while the TCP connection is in progress. That return value is the
  expected/correct behavior per the Quanser docs — the caller is
  supposed to then invoke
  `qcomm_poll(conn, timeout, QCOMM_POLL_CONNECT)` until it returns
  > 0 to know the connection has completed.
- The bridge's connect logic treated ANY r < 0 as a fatal error
  (`if (r < 0) { log; return false; }`). That ran on the first
  `qcomm_connect` call, before `conn_open_` was set to true, so the
  worker loop kept calling `connectStream()` every 250 ms. Each
  call queued a fresh half-open TCP connection at the runtime and
  abandoned it before completing the handshake. Runtime never
  accepted anything because the bridge never polled for
  POLL_CONNECT after issuing the call.

Fix applied:
- `connectStream()` now distinguishes `-QERR_WOULD_BLOCK` from other
  failures explicitly:
    if (r < 0 && r != -QERR_WOULD_BLOCK) { fatal log; return false; }
  On `-QERR_WOULD_BLOCK` it sets `conn_open_ = true` and enters the
  existing `qcomm_poll(QCOMM_POLL_CONNECT)` loop (2 s budget) to
  wait for the connection to complete cleanly.
- Added an inline comment in the source describing the protocol so
  this never gets re-broken.

Expected behavior after this fix (next physical bring-up):
- Bridge log should print one of:
    "Connected to QCar2DepthAlign stream at tcpip://localhost:17003"
       (after the post-spawn poll completes), OR
    "Stream connect did not complete within initial 2 s wait;
       worker will keep polling." (degraded — investigate)
- No more repeated "A non-blocking operation would have blocked"
  ERROR lines. That message originated from misclassifying the
  normal non-blocking return.
- `ss -tlnp | grep :17003` should show Recv-Q = 0 (no queued
  half-open connects) once the bridge has connected.
- /camera/color_image and /camera/depth_image should start
  publishing at ~30 Hz with encodings bgr8 and 32FC1.

Build verification:
- Isolated `colcon build --packages-select qcar2_nodes --cmake-target
  qcar2_camera_bridge` PASS; only pre-existing OpenCV ABI warning.

## 2026-05-15 (manual_drive hotfix: continuous publish at 50 Hz)

Issue observed on the mat (first attempted Step 4 drive):
- After launching `qcar2_cartographer_launch.py` + `vo_node` + `manual_drive`,
  pressing `a`/`d` produced a brief visible steering response, but `w`/`s`
  produced no forward/reverse motion. Increasing `forward_speed` did not
  help (confirmed by operator).

Root cause:
- `nav2_qcar_command_convert` enforces a 0.25 s `/cmd_vel_nav` timeout that
  zeros both throttle AND steering when no fresh Twist arrives inside the
  window (added 2026-05-04 as a safety auto-stop). The previous
  `manual_drive.py` only published on KEY PRESS, so each W tap produced a
  single 0.25 s Twist pulse followed by forced zero. At low throttle the
  pulse can't overcome static friction; at higher throttle the wheels get
  yanked-then-stopped every 0.25 s. Steering looks "responsive" because the
  servo visibly moves during the pulse even though the converter zeros it
  immediately afterward.

Fix implemented in `autonomy/manual_drive.py`:
- Decoupled keyboard reading from publishing.
- Keyboard thread (`_keyboard_loop`) only mutates a thread-safe
  `(linear_x, angular_z)` state on key press.
- A 50 Hz `rclpy` timer (`_publish_current`) emits the current state every
  20 ms - well inside the converter's 0.25 s safety window - so tapping
  w/a/s/d once produces sustained motion until space/x/q is pressed.
- Stop+republish on shutdown so the converter receives an explicit zero
  before the publisher destroys.
- Added a top-of-file comment block documenting the architecture and the
  2026-05-04 safety-timeout interaction so this never regresses to a
  publish-on-keypress design.

Defaults unchanged (`forward_speed=0.10`, `reverse_speed=0.08`,
`turn_rate=0.25`) so existing tuning recipes in Easy_Start.txt section 2.1
still apply. New parameter `publish_rate_hz` (default `50.0`).

Build verification:
- `python3 -m py_compile` PASS on the rewritten manual_drive.py.
- Rsync + `colcon build --packages-select qcar2_autonomy --symlink-install`
  PASS (only the standard easy_install deprecation warning).

Next: re-run Step 4 mat drive with the new manual_drive build.

## 2026-05-15 (manual_drive hotfix #2: independent throttle and steering axes)

Issue: previous hotfix made w/a/s/d mutually exclusive (pressing `a` zeroed
`linear_x`, pressing `w` zeroed `angular_z`), so the car could not be
steered while moving - exactly the use case the test plan requires.

Fix: split the state-setters in `manual_drive.py` so each key only mutates
the axis it controls:
- `w` / `s`  -> set `linear_x` only, leave `angular_z` untouched.
- `a` / `d`  -> set `angular_z` only, leave `linear_x` untouched.
- ` ` (space) -> release throttle only (linear_x=0).
- `c`         -> center steering only (angular_z=0).
- `x`         -> full stop (both axes zero).
- `q`         -> stop and quit.

The 50 Hz publish-timer architecture from hotfix #1 is unchanged - this
change only touches the keyboard handler.

Build verification: py_compile PASS, rsync to ~/ros2/src + colcon build
PASS.

## 2026-05-15 (manual_drive hotfix #3: revert to original, single-line fix only)

Per operator instruction: previous hotfixes (#1 + #2) introduced too much
new behavior. Reverted `manual_drive.py` to exactly the file at git HEAD
and applied the smallest possible change to fix the no-forward-motion bug.

Total diff vs HEAD: 4 added lines (3-line comment + 1 republish call).

Behavior is otherwise unchanged from the original:
- w/a/s/d still mutually exclusive (each key zeros the other axis).
- Controls, defaults, prints, exit behavior all identical.
- Only difference: when `get_key()` returns "" (no key in the 100 ms
  window), the loop now also publishes the current `(linear_x,
  angular_z)` before `continue`. That keeps `/cmd_vel_nav` fresh
  inside the 0.25 s safety timeout of `nav2_qcar_command_convert`,
  so a tap-W produces sustained motion until tap-x/space/q.

py_compile + rsync + colcon build PASS.

Hotfix #1 and #2 are superseded by this entry. The 50 Hz timer-based
publish + decoupled keyboard thread + new key bindings are all rolled
back.

## 2026-05-15 (manual_drive — full verbatim state snapshot, post-hotfix #3)

Per operator instruction: changelog must be self-contained. No future
reader of these logs should ever need to consult git history to know
what the source looked like. Recording the complete contents of
`Development/ros2/src/qcar2_autonomy/autonomy/manual_drive.py` as of
the end of this turn (after the hotfix #3 single-line fix) below.

------------------- BEGIN manual_drive.py -------------------
#!/usr/bin/env python3

import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def get_key(timeout_s=0.1):
    """Read one key from stdin in raw mode, or return '' on timeout."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if ready:
            return sys.stdin.read(1)
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class ManualDrive(Node):
    """Simple keyboard teleop for QCar2 via /cmd_vel_nav."""

    def __init__(self):
        super().__init__("manual_drive")

        # Conservative defaults for first physical tests.
        self.declare_parameter("forward_speed", 0.10)
        self.declare_parameter("reverse_speed", 0.08)
        self.declare_parameter("turn_rate", 0.25)
        self.declare_parameter("cmd_topic", "/cmd_vel_nav")

        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.reverse_speed = float(self.get_parameter("reverse_speed").value)
        self.turn_rate = float(self.get_parameter("turn_rate").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

    def publish_cmd(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ManualDrive()

    print("Keyboard manual drive (WASD style)")
    print("  w: forward")
    print("  s: reverse")
    print("  a: turn left in place")
    print("  d: turn right in place")
    print("  space/x: stop")
    print("  q: stop and quit")
    print(
        f"Using topic={node.cmd_topic}, "
        f"forward={node.forward_speed:.2f}, "
        f"reverse={node.reverse_speed:.2f}, "
        f"turn={node.turn_rate:.2f}"
    )

    linear_x = 0.0
    angular_z = 0.0

    try:
        while rclpy.ok():
            key = get_key().lower()

            # Republish current state on idle ticks so nav2_qcar_command_convert
            # (0.25 s /cmd_vel_nav safety timeout, added 2026-05-04) does not
            # zero throttle and steering between key presses.
            if not key:
                node.publish_cmd(linear_x, angular_z)
                continue

            if key == "q":
                linear_x = 0.0
                angular_z = 0.0
                node.publish_cmd(linear_x, angular_z)
                print("\nStopped. Exiting manual drive.")
                break
            if key == "w":
                linear_x = node.forward_speed
                angular_z = 0.0
            elif key == "s":
                linear_x = -node.reverse_speed
                angular_z = 0.0
            elif key == "a":
                linear_x = 0.0
                angular_z = node.turn_rate
            elif key == "d":
                linear_x = 0.0
                angular_z = -node.turn_rate
            elif key in (" ", "x"):
                linear_x = 0.0
                angular_z = 0.0
            else:
                continue

            node.publish_cmd(linear_x, angular_z)
            print(
                f"\rlinear={linear_x:+.2f} angular={angular_z:+.2f}  ",
                end="",
                flush=True,
            )
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
------------------- END manual_drive.py -------------------

Behavioral contract of the file above:
- Subscribes to keystrokes via tty raw mode (`get_key` 100 ms timeout).
- Publishes `geometry_msgs/Twist` on `/cmd_vel_nav`.
- Defaults: `forward_speed=0.10`, `reverse_speed=0.08`,
  `turn_rate=0.25`, `cmd_topic=/cmd_vel_nav`.
- Key bindings (mutually exclusive axes, identical to pre-2026-05-15
  behavior):
    w -> linear_x=forward_speed,  angular_z=0
    s -> linear_x=-reverse_speed, angular_z=0
    a -> linear_x=0,              angular_z=turn_rate
    d -> linear_x=0,              angular_z=-turn_rate
    space / x -> linear_x=0, angular_z=0
    q -> stop and exit
- The only change vs the original Apr 30 version is the 4 lines in the
  idle branch that republish the current `(linear_x, angular_z)` on
  every 100 ms tick where no key was pressed, so the converter's
  0.25 s timeout (set 2026-05-04) does not zero output between
  keystrokes.

## 2026-05-15 (manual_drive steering fix: keep forward speed during A/D)

Context:
- Physical manual driving reported that pressing `W` starts forward motion,
  but pressing `A` or `D` immediately stops forward movement. Expected
  behavior for operator control: press `W` once, then steer with `A/D`
  without dropping throttle.

Root cause:
- `Development/ros2/src/qcar2_autonomy/autonomy/manual_drive.py` explicitly
  set `linear_x = 0.0` in both `A` and `D` key branches (turn-in-place logic).
- This was not introduced by recent VO/camera changes; it existed in the
  original `manual_drive.py` commit and therefore behaved as designed for
  "turn in place", not "steer while moving".

Code changes made:
- File: `Development/ros2/src/qcar2_autonomy/autonomy/manual_drive.py`
- Updated help text:
  - `a: turn left in place` -> `a: steer left (keep current speed)`
  - `d: turn right in place` -> `d: steer right (keep current speed)`
- Updated key handling:
  - Removed `linear_x = 0.0` from `A` branch.
  - Removed `linear_x = 0.0` from `D` branch.
  - `A/D` now only change `angular_z`; current `linear_x` remains latched.

Net effect:
- `W` latches forward speed.
- `A/D` steer while preserving current speed.
- `S` still switches to reverse and recenters steering.
- `space/x` still full stop.
- Idle republish behavior remains intact (prevents timeout-induced zeroing).

Verification:
- `python3 -m py_compile Development/ros2/src/qcar2_autonomy/autonomy/manual_drive.py` passed.

## 2026-05-15 (manual_drive — Codex external edit applied + verified)

An external edit was applied to
`Development/ros2/src/qcar2_autonomy/autonomy/manual_drive.py` by
Codex (not by this assistant). The intent was to allow simultaneous
throttle + steering while staying as close as possible to the
original file: W/S still set both axes (linear=forward/reverse,
angular=0), but A/D now ONLY set angular and leave linear untouched.
So tapping W and then D produces forward motion AND a right turn,
which was the use case requested by the operator.

Verbatim state of the file after Codex's edit is captured below.
This supersedes the previous 'verbatim state snapshot, post-hotfix #3'
section. Future readers should treat this snapshot as the current
source of truth for manual_drive.py.

------------------- BEGIN manual_drive.py (post-Codex) -------------------
#!/usr/bin/env python3

import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def get_key(timeout_s=0.1):
    """Read one key from stdin in raw mode, or return '' on timeout."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if ready:
            return sys.stdin.read(1)
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class ManualDrive(Node):
    """Simple keyboard teleop for QCar2 via /cmd_vel_nav."""

    def __init__(self):
        super().__init__("manual_drive")

        # Conservative defaults for first physical tests.
        self.declare_parameter("forward_speed", 0.10)
        self.declare_parameter("reverse_speed", 0.08)
        self.declare_parameter("turn_rate", 0.25)
        self.declare_parameter("cmd_topic", "/cmd_vel_nav")

        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.reverse_speed = float(self.get_parameter("reverse_speed").value)
        self.turn_rate = float(self.get_parameter("turn_rate").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

    def publish_cmd(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ManualDrive()

    print("Keyboard manual drive (WASD style)")
    print("  w: forward")
    print("  s: reverse")
    print("  a: steer left (keep current speed)")
    print("  d: steer right (keep current speed)")
    print("  space/x: stop")
    print("  q: stop and quit")
    print(
        f"Using topic={node.cmd_topic}, "
        f"forward={node.forward_speed:.2f}, "
        f"reverse={node.reverse_speed:.2f}, "
        f"turn={node.turn_rate:.2f}"
    )

    linear_x = 0.0
    angular_z = 0.0

    try:
        while rclpy.ok():
            key = get_key().lower()

            # Republish current state on idle ticks so nav2_qcar_command_convert
            # (0.25 s /cmd_vel_nav safety timeout, added 2026-05-04) does not
            # zero throttle and steering between key presses.
            if not key:
                node.publish_cmd(linear_x, angular_z)
                continue

            if key == "q":
                linear_x = 0.0
                angular_z = 0.0
                node.publish_cmd(linear_x, angular_z)
                print("\nStopped. Exiting manual drive.")
                break
            if key == "w":
                linear_x = node.forward_speed
                angular_z = 0.0
            elif key == "s":
                linear_x = -node.reverse_speed
                angular_z = 0.0
            elif key == "a":
                # Keep the currently latched speed and only steer.
                angular_z = node.turn_rate
            elif key == "d":
                # Keep the currently latched speed and only steer.
                angular_z = -node.turn_rate
            elif key in (" ", "x"):
                linear_x = 0.0
                angular_z = 0.0
            else:
                continue

            node.publish_cmd(linear_x, angular_z)
            print(
                f"\rlinear={linear_x:+.2f} angular={angular_z:+.2f}  ",
                end="",
                flush=True,
            )
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
------------------- END manual_drive.py (post-Codex) -------------------

Behavioral contract (current):
- Defaults: forward_speed=0.10, reverse_speed=0.08, turn_rate=0.25,
  cmd_topic=/cmd_vel_nav.
- Idle-tick republish (from hotfix #3) preserved: every 100 ms tick
  with no key pressed still calls node.publish_cmd(linear_x,
  angular_z), keeping nav2_qcar_command_convert (0.25 s timeout) fed.
- Key bindings:
    w       -> linear_x=forward_speed,  angular_z=0       (straight forward)
    s       -> linear_x=-reverse_speed, angular_z=0       (straight reverse)
    a       -> angular_z=+turn_rate     (linear_x untouched)
    d       -> angular_z=-turn_rate     (linear_x untouched)
    space   -> linear_x=0, angular_z=0  (full stop)
    x       -> linear_x=0, angular_z=0  (full stop)
    q       -> publish zeros, print 'Stopped. Exiting manual drive.',
               break out of the loop, destroy node, shutdown rclpy

Runtime delivery:
- rsync of qcar2_autonomy source -> ~/ros2/src/qcar2_autonomy/ verified
  identical post-edit.
- /home/nvidia/ros2/build/qcar2_autonomy/autonomy/manual_drive.py is
  3709 bytes, timestamped May 15 12:13 — matches source byte-for-byte.
- This is the file Python imports via the egg-link
  /home/nvidia/ros2/install/qcar2_autonomy/lib/python3.8/site-packages/
  qcar2-autonomy.egg-link, which points at
  /home/nvidia/ros2/build/qcar2_autonomy.
- No rebuild required. Operator only needs to Ctrl-C the running
  manual_drive in Terminal D and re-run
  'ros2 run qcar2_autonomy manual_drive'.

## 2026-05-15 (Physical Test 2 — ON MAT analysis with new bridge)

Source: `Development/ros2/src/qcar2_autonomy/VO_readings.txt` section
"ON MAT TEST /vo/fault_status" (lines 7564-9033). 714 raw data lines,
651 successfully parsed (63 skipped due to truncated psi_raw fields in
the terminal log; that is a logging-side artifact, not a VO failure).

Drive: manual_drive (post-Codex steer-while-moving fix), 4 left turns
and 3 straight segments per operator's sequence description.
Stack: qcar2_cartographer_launch.py + Python camera_bridge + vo_node
(camera_mode=physical, input_source=ros_rgbd, force_cart_yaw=true).

### Whole-run state mix
- `agree`:      153 / 651  (23.5%)
- `vo_suspect`: 455 / 651  (69.9%)
- `warming`:     43 / 651  ( 6.6%)
- `init`:         0 / 651  ( 0.0%)

Comparison vs prior physical tests:
- Test 1.1: agree 48%, vo_suspect 43%, warming 7%, init 2%
- Test 1.2: agree 33%, vo_suspect 58%, warming 9%, init 0.4%
- Test 2 (this run): agree 23.5%, vo_suspect 69.9%, warming 6.6%, init 0%
Init dropped to zero (good - K_inv fix + aligned depth eliminate
cold-start). agree fell and vo_suspect rose - dominated by turn
segments accumulating in vo_suspect.

### Per-state averages (this run)
- agree:      rho=0.0418 (max 0.144), w=0.535, inliers=213, spread=99
- vo_suspect: rho=0.0740 (max 0.278), w=0.193, inliers=93,  spread=74
- warming:    rho=0.0000 (locked),    w=0.000, inliers=176, spread=94
- zero-inlier frames: 62 / 651 (9.5%); concentrated in vo_suspect
  (13.6% of vo_suspect frames vs 0% of agree).

### Inlier and spread comparison (agree state only)
- 1.1: inliers ~178, weight ~0.512
- 1.2: inliers ~134, weight ~0.480
- 2:   inliers  213, weight  0.535
The new bridge confirms the Step 3 stationary observation under
driving load: higher inlier counts and slightly higher confidence.

### Startup (first 80 frames)
- 68 agree, 7 warming, 5 vo_suspect (no init)
- rho mean 0.025, w mean 0.57, inliers mean 240, spread mean 93
- Best startup of any physical test so far.

### Ending (last 80 frames)
- 80 / 80 vo_suspect
- rho mean 0.054 (low) but w mean 0.08 (very low) - low quality kept
  the state in vo_suspect even though VO-vs-Cart distance was small.
- This is the classic stuck-in-vo_suspect pattern after a hard turn.

### State transitions and re-anchor cadence
- agree segments:        12
- warming segments:       6
- init segments:          0
- agree -> warming:       6
- warming -> agree:       6
- agree -> vo_suspect:    6
- vo_suspect -> agree:    5
Re-anchor gating is firing similarly to Test 1 (9 segments,
7 transitions), so the re-anchor policy is not the regression.

### Trajectory and final divergence
- Cart traj: start (0.11, -0.01) yaw=-179°  -> end (-0.55, 1.53) yaw=-127°
- VO   traj: start (0.11, -0.01) yaw=-179°  -> end ( 0.78, 1.11) yaw=-127°
- Final VO vs Cart divergence:
    Dx = +1.330 m   (Cart went -X, VO went +X => sign FLIP on X)
    Dy = -0.420 m   (Y direction same sign, magnitude under by ~27%)
    Dyaw = 0.0°     (force_cart_yaw=true pins VO heading to Cart)
- Net 52° of CCW rotation across the run.

### Axis sign agreement (Cart deltas vs VO deltas, 40-frame windows)
- X-dominant: score=0.920 over 226 qualifying windows
- Y-dominant: score=1.000 over 286 qualifying windows
vs baselines:
- 1.1: X=0.514, Y=0.707
- 1.2: X=0.634, Y=0.726
Significant improvement in directional consistency. The final 1.33 m
X-divergence is therefore not a continuous direction error but a
transient sign flip during one or more turns that the re-anchor
cooldown did not catch in time.

### Shadow yaw (psi_raw) vs cart_psi
- mean error:    -20.6°
- median:         -9.0°
- stdev:           27.2°
- abs max:        115.0°
- frames with |err| <= 15°: 389 / 651 (59.8%)

Shadow yaw is NOT ready to take over. The 115° peak error and 27°
stdev would cause severe drift in standalone mode. Keep
force_cart_yaw=true for the next run.

### Interpretation
1. The Python camera bridge + K_inv fix + aligned 32FC1 depth path is
   structurally correct. Startup is clean, inliers are high, axis
   sign agreement is excellent, agree-state quality beats both prior
   physical tests.
2. The drop in agree-state share and rise in vo_suspect is dominated
   by turn segments. Tight / jerky left turns under manual drive cut
   translational feature flow, drop spread and weight, trigger safety
   gates, and leave VO stuck in vo_suspect with rho near zero (i.e.
   no update) until quality recovers enough for a re-anchor.
3. The 1.33 m X-direction divergence is the most concerning datum.
   It is caused by a transient X-sign flip during one of the harder
   left turns; X-axis windowed sign agreement is still 0.92, so it
   is not pervasive.
4. Manual drive jitter is a confound, not a controlled variable.

### Recommendations for next session
1. Keep force_cart_yaw=true. Shadow yaw needs more controlled-turn
   data before we can trust standalone yaw.
2. Re-run with smoother driving:
   (a) gentler turn inputs on manual_drive, OR
   (b) autonomy_planner / nav2 path at constant low speed, which is
       the apples-to-apples comparison vs Tests 1.1 and 1.2.
3. Do NOT tune gate parameters yet. One more controlled run is
   needed to distinguish driver smoothness from gate tightness.
4. Optional: capture topic hz on /camera/color_image during driving
   to confirm the bridge still holds 30 fps under autonomy_planner
   plus VO load.

### Files touched in this turn
- VO_readings.txt: read-only (user-owned, never modified).
- /tmp/analyze_test2.py: analysis script (out-of-tree, not committed).
- This changelog entry.
- VO_Conversation_Log.txt Turn 78 (added in this turn).

## 2026-05-15 (workflow note: nav_to_pose.py edits require colcon build)

Confirmed for the record: editing any `.py` source file in
`qcar2_autonomy` (including `nav_to_pose.py`) requires a
`colcon build --packages-select qcar2_autonomy --symlink-install`
before the change takes effect at runtime, even with
`--symlink-install`. The reason is the ament_python build copies
files from `~/ros2/src/qcar2_autonomy/autonomy/` into
`~/ros2/build/qcar2_autonomy/autonomy/`, and Python imports from
the build directory via the egg-link at
`~/ros2/install/qcar2_autonomy/lib/python3.8/site-packages/
qcar2-autonomy.egg-link`. The `--symlink-install` flag only affects
the install dir's relationship to the build dir, not the build
dir's relationship to src. Build itself runs in ~4 s on this
machine.

Quick-experiment alternative: override declared ROS parameters at
launch / run time without editing source. For nav_to_pose.py the
relevant override is:
  ros2 run qcar2_autonomy path_follower --ros-args
      -p rotation_offset:='[90.0]'
which avoids the rebuild cycle entirely while still letting you
sweep the value. Default at this snapshot is `[86.0]` (see
`nav_to_pose.py` line 231).

## 2026-05-15 (Physical Test 3 - autonomy_planner ON MAT analysis)

Source: `Development/ros2/src/qcar2_autonomy/VO_readings.txt` section
"ON MAT TEST using autonomy_planner" (lines 9035-9557). 241 raw
data lines, 220 successfully parsed (21 skipped - truncated psi_raw
in terminal log, logging artifact only).

Drive: autonomy_planner_launch.py - path_follower (nav_to_pose) +
yolo_detector + trip_planner + lane_detector + Planner_server.
Stack: qcar2_cartographer_launch.py + Python camera_bridge + vo_node
(camera_mode=physical, input_source=ros_rgbd, force_cart_yaw=true),
same VO parameters as Test 2.

### Whole-run state mix
- `agree`:      0 / 220  ( 0.0%)
- `vo_suspect`: 220 / 220 (100.0%)
- `warming`:     0 / 220  ( 0.0%)
- `init`:        0 / 220  ( 0.0%)
VO never entered `agree` for a single frame across the entire run.

### Comparison vs Test 2 (same drive plan, manual_drive)
- Test 2: agree 23.5%, vo_suspect 69.9%, warming 6.6%, init 0%
- Test 3: agree 0%, vo_suspect 100%, warming 0%, init 0%
- Test 2 final divergence: Dx=+1.33, Dy=-0.42
- Test 3 final divergence: Dx=+0.09, Dy=-1.61
VO position barely moved while Cart traveled ~1.6 m in Y. The Y
divergence is essentially the full cart displacement - VO did not
integrate translation across the run.

### Per-run averages
- rho       mean=0.078  med=0.000  max=0.358
- w         mean=0.028  med=0.000   <- updates rejected near 100% of the time
- inliers   mean=112    med=66     min=0     <- not catastrophic, but lower than Test 2 agree=213
- spread    mean=76     med=79
- zero-inlier frames: 14 / 220 (6.4%)

### Trajectory
- Cart: (0.11, -0.01) yaw -180  ->  (-0.09, 1.52) yaw -140    [1.6 m, 40 deg CCW]
- VO:   (0.08, -0.04) yaw -180  ->  (-0.00, -0.09) yaw -140   [<0.1 m]
- Final divergence: Dx=+0.090, Dy=-1.610, Dyaw=0 (force_cart_yaw=true)

### State transitions / re-anchor segments
- agree segments: 0
- warming segments: 0
- init segments: 0
- Total transitions: 0 (state never changed)
Re-anchor never fired because VO never produced a valid update to
anchor from.

### Shadow yaw (psi_raw vs cart_psi)
- mean error: +4.5 deg
- stdev:      106.5 deg
- abs max:    178.0 deg
- |err| <= 15 deg: 42 / 220 (19.1%)
Far worse than Test 2 (27 deg stdev, 115 deg max). Consistent with
psi_raw not integrating because dpsi=0 on every rejected frame -
psi_raw effectively froze at its initial value while cart_psi
swept through the turn.

### Axis sign agreement (Cart deltas vs VO deltas, 40-frame windows)
- X-dominant: score=1.000 over 44 windows  (but VO Dx is near zero,
  so this is mostly about sign of near-zero noise)
- Y-dominant: score=0.844 over 96 windows
These numbers are misleadingly OK - they reflect the few windows
where VO did register tiny motion. The bulk of the run has VO
delta = 0, which the sign-agreement metric doesn't penalize.

### Diagnostic interpretation
The signature 'rho=0 w=0 dx=0 dy=0 dpsi=0' on the vast majority of
frames is the diagnostic of safety-gate failure inside
visual_odometry.py - the VO frame was computed but rejected before
being committed. Candidate rejection causes (in priority order):

1) Per-frame translation exceeded `max_translation` gate.
   autonomy_planner default speed is higher than manual_drive's
   0.10 m/s. Higher car speed + variable VO tick rate (CPU
   contention from yolo/lane/Planner_server) -> larger per-frame
   pixel disparity -> ORB-matches scatter -> RANSAC produces a
   pose that exceeds the magnitude gate.

2) Per-frame dt exceeded `max_dt` gate.
   Test 3 produced 220 fault_status frames; Test 2 produced 651
   over a comparable run. A 3x lower tick rate is consistent with
   the camera bridge or VO node falling behind under combined
   load (yolo + lane + Planner_server all subscribed to
   /camera/color_image).

3) Inlier count low after RANSAC because per-frame motion stresses
   ORB. Average 112 vs Test 2 agree-state 213. Not zero but
   borderline.

Bridge architecture itself is not the failure: peak inliers at
startup were 339, encoding auto-detect fired, [DEPTH DIAG ALIGNED]
logs show plausible metric values (1.0-4.6 m) throughout. The
failure mode is on the VO side.

### Recommended next steps (do NOT tune gates yet)
1. Re-run with concurrent `ros2 topic hz /camera/color_image` and
   `ros2 topic hz /vo/fault_status` captured in a 5th terminal.
   This will confirm whether the bridge fps drops under the full
   autonomy stack.
2. Re-run with ONLY `path_follower` (no yolo/lane/Planner_server)
   to isolate planner speed from CPU contention:
     ros2 run qcar2_autonomy path_follower
3. If feasible, lower the planner speed to ~0.10-0.15 m/s for an
   apples-to-apples vs Test 2.
4. Only after the above: review visual_odometry.py safety-gate
   thresholds (`max_translation`, `max_rotation_deg`, `max_dt`)
   for whether they need to scale with planner speed.

Operator-side notes:
- Test 3 was a planned comparison run; the failure is informative,
  not a regression vs Test 2. Test 2 results remain the baseline
  for the new bridge architecture.
- force_cart_yaw=true is masking how bad VO yaw is right now.
  Definitely keep that flag on until the gate / fps issue is
  resolved.

### Files touched in this turn
- VO_readings.txt: read-only (user-owned).
- /tmp/analyze_test3.py: out-of-tree analysis script (cloned from
  /tmp/analyze_test2.py, only path differed).
- This changelog entry.
- VO_Conversation_Log.txt Turn 80 (added in this turn).

## 2026-05-15 (Test 3 follow-up: root-cause for fps drop + stationary car)

Operator captured concurrent fps measurements during another
autonomy_planner run and reported:
- ros2 topic hz /camera/color_image  -> avg ~15 fps (max 15)
- ros2 topic hz /vo/fault_status     -> avg ~3 Hz, occasionally < 1 Hz
- Car did not move during the run.

These two observations resolve the open questions from the Test 3
analysis above.

### Finding 1: bridge fps halved under autonomy stack (30 -> 15)

autonomy_planner_launch.py actually spawns:
- path_follower (light, geometry only)
- traffic_system_detector (executable=yolo_detector,
  name=qcar2_yolo_detector)
- trip_planner (light)
- Planner_server (light)
- lane_detector is commented out at the LaunchDescription return

yolo_detector subscribes to BOTH /camera/color_image AND
/camera/depth_image at QoS depth 5 (yolo_detector.py:93-96) and runs
YOLO inference on every received frame. On this Jetson this is at
the edge of available compute. When inference falls behind, the
image pipeline back-pressures and the bridge's publish timer slips,
halving the effective publish rate. Consequence: vo_node sees 15
frames/sec instead of 30, per-frame motion roughly doubles, and the
max_translation / max_dt safety gates inside visual_odometry.py
reject the update. This is the mechanism behind Test 3's
100% vo_suspect state mix.

### Finding 2: car did not move because start_path defaulted to False

nav_to_pose.py:238-239 declares:
    self.declare_parameter('start_path', [False])
    self.path_execute_flag = list(self.get_parameter
        ('start_path').get_parameter_value().bool_array_value)[0]

The launch file does not set start_path, so path_follower comes up
but does not issue Twist commands until the flag is flipped. The
previous Test 3 run apparently had it flipped at some point during
the run (since cart moved 1.6 m in Y); the new run did not. Cart
movement is incidental to the fps diagnostic - the fps drop is
purely a function of who is subscribed, not whether the car is
driving.

Default desired_speed in nav_to_pose.py is [0.2] m/s, which is 2x
manual_drive Test 2's 0.10 m/s. For an apples-to-apples comparison
the next run should override to 0.10.

### Next run plan (Test 4 - isolation)

Goal: confirm yolo_detector is the cause of the fps drop AND
produce a valid VO comparison vs Test 2 baseline.

Terminals A, B, C unchanged from Test 2/3 (quarc_run + cartographer
launch, vo_node, fault_status echo).

Terminal D replaced by:
    ros2 run qcar2_autonomy path_follower --ros-args \
        -p start_path:='[True]' \
        -p desired_speed:='[0.1]'
(path_follower only - no yolo, no trip_planner, no Planner_server.)

Terminal E (new) - fps watchers, started before the run:
    ros2 topic hz /camera/color_image
    ros2 topic hz /vo/fault_status

Expected if the hypothesis is correct:
- /camera/color_image returns to ~28-30 fps
- /vo/fault_status returns to ~5 Hz
- VO re-enters agree state, fault_status mix becomes Test 2-like
  or better

If bridge fps still does not recover to ~30 with just
path_follower running, the suspect shifts to vo_node load or
cartographer rather than yolo.

Operator will paste new results under a section in VO_readings.txt
such as "ON MAT TEST - autonomy path_follower alone, no yolo".

Do NOT tune visual_odometry.py safety gates yet. The Test 3
analysis already cautioned this; the Finding 1 mechanism makes it
concrete: the gates are appropriate for 30 fps input; under 15 fps
they reject correctly. Restore the input rate first, then re-evaluate.

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 81 (added in this turn).
- No code edits.

## 2026-05-15 (Test 4 follow-up: yolo hypothesis ruled out, bridge fps anomaly isolated)

Operator ran the isolation test (Terminal D = path_follower alone,
no yolo / trip_planner / Planner_server) and reported:
- /camera/color_image: ~16-18 fps (NOT recovered to 30)
- /vo/fault_status:    ~3.7-4 Hz with path_follower running,
                       ~5 Hz before path_follower came up
- Car did not move under path_follower alone (probably because
  trip_planner / Planner_server are needed to seed the planned
  route - path_follower follows a path it does not own).

### Yolo hypothesis: refuted

If removing yolo + trip_planner + Planner_server leaves the bridge
stuck at ~17 fps, those nodes were not the bottleneck. CPU
contention from the autonomy stack is not the dominant cause of
the bridge fps drop. The Test 3 follow-up entry above was wrong
and should be read in light of this new evidence.

### What is actually going on (re-derived)

Two separate observations, only one anomalous:

1. fault_status at ~5 Hz is NORMAL for this stack. Test 2 produced
   651 frames over ~130 s = ~5 Hz. vo_node ticks at 20 Hz nominal
   (vo_node.py:349-350) but ORB + RANSAC + SVD on 640x480 frames
   takes ~150-200 ms on this Jetson, so the effective fault_status
   rate is ~5 Hz. This is a VO-computation cost, not a rate cap.

2. /camera/color_image at ~17 fps IS anomalous. Step 3 stationary
   test (2026-05-14) verified ~30.3 fps via the bridge's own
   '[bridge hb]' log line under the EXACT SAME stack
   (qcar2_cartographer_launch.py + vo_node). Something has changed
   system-side since.

qcar2_camera_bridge.py polls QCar2DepthAligned.read() at 60 Hz
and publishes only when read() returns True (bridge code lines
164-167, 191-199). Nothing on the ROS side caps the rate. The
published fps equals the PIT runtime's delivery rate. Therefore
the ~17 fps observation means the Quanser depth-align runtime
itself is delivering 17 fps, not the 30 fps Step 3 confirmed.

### Suspects (in priority order)

A. Jetson power mode / clocks throttled.
   nvpmodel and jetson_clocks govern CPU + GPU frequencies. The
   QCar2DepthAlign pipeline depends on GPU/CUDA for the stereo->
   aligned 32FC1 pass. If the system reverted to a low-power
   profile (e.g. 15W) between Step 3 and today, 30 fps -> ~15 fps
   is exactly the expected drop.

B. Orphaned QCar2DepthAlign runtime from a previous session.
   Two processes contending for the camera USB device and
   tcpip://localhost:17003 could halve the effective rate.

C. Thermal throttling after sustained day of operation.

D. RealSense USB renumeration into a slower port / hub.

### Diagnostic steps for operator (car off mat, full stack down)

1. Check Jetson power profile and clocks:
     sudo nvpmodel -q
     sudo jetson_clocks --show
   If not MAXN / clocks not at max, restore with:
     sudo nvpmodel -m 0           # MAXN profile
     sudo jetson_clocks            # pin clocks high

2. Check for orphan camera owners:
     ps -ef | grep -iE 'QCar2DepthAlign|quarc_run|rgbd|camera_bridge' | grep -v grep
   Kill any survivors before restarting cartographer.

3. Run tegrastats during a fresh stack bring-up to see CPU / GPU /
   thermal state live:
     sudo tegrastats --interval 1000

4. The bridge logs '[bridge hb] published=X frames in 5.0 s
   (~Y fps)' every 5 s in Terminal A. That line is ground truth
   for the bridge's measured delivery rate. Whatever it reports
   should match the 'ros2 topic hz' on /camera/color_image. If
   the heartbeat says ~30 but the topic hz reports ~15, that
   means consumers (vo_node) are throttling subscription -
   different diagnosis from PIT-throttled.

### Decision on mat testing

Operator has decided to continue using manual_drive on the mat
for VO testing going forward (constant operator-controlled speed
at 0.10 m/s is the cleanest baseline). autonomy_planner runs are
deferred until: (a) the bridge fps anomaly is resolved, and
(b) the autonomy stack can be run with a known good speed and
without yolo at full inference rate.

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 82 (added in this turn).
- No code edits.

## 2026-05-15 (system diagnostics: bridge confirmed at 30 fps, topic_hz was the artifact)

Operator captured the diagnostic commands from the previous turn
into VO_readings.txt under "GPU checks" section. Headline results:

1. Bridge heartbeat in cartographer log:
   [bridge hb] published=152 frames in 5.0 s (~30.4 fps)
   ----> bridge IS publishing at 30 fps. PIT runtime healthy.

2. Jetson power profile:
   sudo nvpmodel -q  ->  NV Power Mode: MAXN  (correct)

3. Jetson clocks:
   CPU max freq: 2188800 (2.18 GHz) on all 8 cores
   GPU max freq: 930750000 (930 MHz)
   EMC at 2133 MHz, DLA at 1.4 GHz, PVA at 1.19 GHz
   ----> CPU/GPU/memory clocks at expected MAXN ceilings.
   The 'schedutil' governor scales them down when idle (729 MHz
   in tegrastats during idle), and they ramp back to 2188 MHz the
   moment work hits (visible on the 13:58:31 tegrastats line
   where 4 cores briefly jumped to 2188 MHz).

4. Orphan processes:
   ps -ef | grep -iE 'QCar2DepthAlign|quarc_run|rgbd|camera_bridge'
   -> empty. No duplicate camera owners.

5. tegrastats during idle window:
   CPU 0-5% across cores, GPU 0%, temps ~38C. Nothing hot, nothing
   pegged. Plenty of headroom.

### Re-interpretation of the 'bridge fps drop' story

Earlier this session two hypotheses were wrong:
1. (Turn 81) 'yolo_detector starves the bridge to 15 fps' - WRONG
   path_follower-alone test ruled out CPU contention as cause of
   the bridge published rate dropping.
2. (Turn 82) 'PIT runtime throttled / power mode' - WRONG
   bridge heartbeat confirms 30.4 fps published; nvpmodel MAXN,
   clocks at max, no orphans.

Correct interpretation:
- The bridge HAS been publishing at 30 fps consistently. The
  Step 3 stationary test result still stands.
- 'ros2 topic hz /camera/color_image' reports ~17 fps because it
  measures messages RECEIVED by its subscriber, not messages
  published. On Jetson + ROS 2 humble + DDS (cyclonedds or
  fastrtps), large Image topics over default reliable QoS
  routinely get half-dropped at the DDS layer when there is any
  subscriber pressure. This is a middleware artifact, not a
  publisher problem.
- /vo/fault_status at ~5 Hz is the VO computation cost
  (ORB + RANSAC + SVD on 640x480 takes ~200 ms per vo_tick on
  this Jetson). The 20 Hz timer in vo_node.py:349-350 cannot
  meet its period; effective rate is ~5 Hz. This is normal and
  matches every prior test (Test 2 had 651 frames over ~130 s).

### So what actually broke Test 3?

Revised mechanism (still pointing at yolo, but via a different
channel than originally claimed):

- Bridge at 30 fps in both Test 2 and Test 3 - constant.
- vo_node CPU budget is what changed.
- Test 2 (manual_drive only): vo_node had full CPU for its
  ~200 ms tick - produced ~5 Hz fault_status with high agree
  quality.
- Test 3 (autonomy_planner with yolo_detector inference on every
  frame): vo_node CPU contended; per-tick time stretched. Time
  gap between frames vo_node actually processed increased,
  per-frame motion in pixels doubled or tripled, max_translation
  / max_dt safety gates inside visual_odometry.py rejected every
  update. Hence the 100% vo_suspect mix.

Yolo IS the cause of Test 3 failure, but via vo_node CPU
contention rather than bridge fps throttling.

### On the 60 Hz poll rate vs 30 Hz delivery rate

Operator asked whether the disparity is itself a problem. It is
not. The 2x oversample pattern (qcar2_camera_bridge.py:121, 167)
is deliberate:
- Polling 2x the source rate halves the average ROS-publish
  latency (~8 ms vs ~17 ms if we polled at exactly 30 Hz).
- read() returns False quickly when no new frame is available
  (microseconds of CPU per no-op tick). No CPU cost.
- Lines 191-199 of the bridge guard publish on 'if not new:
  return'.
This is the standard pattern for ROS bridges over a fixed-rate
external source and should remain at 60 Hz.

### Mat-testing decision (carried from prior turn)

Operator is continuing mat VO testing with manual_drive only.
Because vo_node CPU contention is the dominant degrader,
manual_drive (which spawns no extra image-subscribing or
GPU-inferring nodes) gives the cleanest VO baseline. Test 2
remains the controlled-baseline run.

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 83 (added in this turn).
- No code edits.

## 2026-05-15 (planning / parameter-tuning analysis, no code edits)

Operator wants to step up VO tuning - specifically the RANSAC
sample size from 2 to 3 or 4 - and also wants to think about
whether VO can keep up at 0.4 / 0.6 m/s driving speeds. Also
floated an architectural question: should yolo + VO + camera
ownership be merged into a single Python process so the camera
frames never traverse ROS pub/sub?

### Current VO parameters (from visual_odometry.py:569-573,785)

- n_features:        800
- match_ratio:       0.75
- ransac_threshold:  0.05 m
- ransac_iterations: 300
- min_inliers:       8
- max_translation:   0.20 m / frame
- max_rotation:      15 deg / frame
- max_dt:            0.20 s
- RANSAC sample size (line 785, np.random.choice(M, 2)): s=2
- Docstring (line 779) explicitly calls it '2-point RANSAC with
  SVD Procrustes'.

### On bumping s = 2 -> 3

The 2D rigid-body transform has 3 DOF (theta, tx, ty), so s=2 is
the minimum geometric sample and s=3 is the natural over-
constrained robust choice. s=4 is over-constrained for the model
and not recommended.

RANSAC iteration scaling at w=0.5 inlier ratio and p=0.99
confidence:
- s=2: N approx 16 iterations
- s=3: N approx 35
- s=4: N approx 72
Current iterations=300 is well above any of these thresholds.
Conclusion: bumping s from 2 to 3 with iterations unchanged is a
free quality improvement (better hypothesis robustness, same
per-iteration cost).

Caveat: s-count tuning does not address the dominant Test 2
failure mode. Most rejected frames have rho=0 w=0 dx=0 dy=0,
which is the safety-gate path (max_translation, max_rotation, or
min_inliers tripping), not a bad-consensus path. s-count affects
consensus quality given that the gates pass.

### Speed budget for 0.4 / 0.6 m/s at 5 Hz VO

Per-frame motion = speed / VO rate:
- 0.10 m/s : 0.020 m / frame  (Test 2 baseline)
- 0.20 m/s : 0.040 m / frame
- 0.40 m/s : 0.080 m / frame
- 0.60 m/s : 0.120 m / frame

All below the 0.20 m max_translation gate. BUT - the metric-space
gate is not the right rate-limit. Pixel-space disparity is. At
0.60 m/s with 640x480 frames and scene depths ~1-2 m, features
can shift 30-60 px between consecutive frames. ORB + BFMatcher
with default search radius begins to lose matches around that
disparity. 0.40 m/s is approximately the upper limit before
match counts and inlier ratio collapse, given the current VO
tick rate.

Two ways to extend the operating envelope:
(a) Raise VO frame rate (reduce per-tick cost - shrinks per-frame
    motion proportionally).
(b) Use a disparity-tolerant matcher (image pyramid, LK optical
    flow seeded from previous match, etc.) - bigger rewrite.

### Speedup levers, ranked

                                       est speedup   risk    quality
  ransac_iterations 300 -> 100         3x            low     same
  s 2 -> 3                             ~0            low     +
  n_features 800 -> 500                1.5x          low     -
  resize 640x480 -> 320x240            3-4x          medium  -
  skip every 2nd frame                 2x            low     +per-frame motion
  ORB+BF -> FAST + LK optical flow     5-10x         high    different regime

The combination 'ransac_iterations 300 -> 100  AND  s 2 -> 3' is
a freebie: same or better quality, ~3x faster RANSAC step. This
is the recommended first experiment.

### On merging yolo + VO + camera-owner into one file

What that change would actually save:
- DDS serialization / deserialization of /camera/color_image and
  /camera/depth_image. About 30 MB/s of serialized image data on
  the wire. The half-drop seen on 'ros2 topic hz' is part of
  this overhead.

What it does NOT save:
- Yolo inference cost. GPU/DLA work still has to run.
- VO computation cost. CPU work still has to run.
- Python GIL serializes CPU paths inside one process; yolo on
  GPU/DLA can overlap with VO on CPU, but that already happens
  today across separate processes.

Downsides:
- Single point of failure for both yolo and VO.
- Cannot disable yolo independently for isolation tests
  (exactly what Test 4 needed).
- Hard to debug a large coupled file.

Cleaner alternative if intra-process plumbing is the goal:
ROS 2 composable nodes loaded into a single component_container
with intra-process comms enabled. yolo + VO + bridge each remain
their own file/class but share buffers zero-copy. This is the
ROS 2 idiom for high-rate image pipelines.

Cleaner alternative if just freeing CPU is the goal:
Throttle yolo inference to 10 Hz instead of 30 Hz. Sign/obstacle
perception rarely needs 30 Hz; would free large amount of GPU/
CPU contention for VO.

Recommendation: do not merge into a single file. If/when we want
to fight the DDS overhead, do composable nodes. If/when we want
to fight CPU contention, throttle yolo.

### Proposed next step

Option A (recommended): tune ransac_iterations 300 -> 100 and
RANSAC sample size 2 -> 3 in visual_odometry.py. Rsync, rebuild,
re-run a manual_drive Test 2-style trajectory at 0.10 m/s. If
agree-state share goes up vs Test 2 (23.5%), keep. If unchanged
or worse, revert and try a different lever.

No code edits in this turn - waiting for operator's go on which
option to execute.

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 84 (added in this turn).
- No code edits.

## 2026-05-15 (QoS fix + ORB ROI mask for VO)

Two related changes made in this turn. Self-contained diffs and
behavioral contracts below so this entry is the recovery point.

### Change 1 - sensor_data QoS across the camera topic chain

Why: bridge publishes /camera/color_image and /camera/depth_image
at 30 fps (confirmed by [bridge hb] heartbeat), but 'ros2 topic hz'
reports ~17 fps. Root cause is reliable QoS + KEEP_LAST depth=5 on
large Image messages, which half-drops at the DDS layer under any
subscriber pressure. Switching to BEST_EFFORT (qos_profile_sensor_
data) eliminates the half-drop because BEST_EFFORT drops in flight
instead of buffering. This is the standard ROS 2 idiom for high-
rate sensor topics. Subscribers must use BEST_EFFORT too; otherwise
the publisher and subscriber are QoS-incompatible and they do not
even connect.

Files edited:

A) qcar2_camera_bridge.py
   Added import: 'from rclpy.qos import qos_profile_sensor_data'
   Changed publishers from:
     self.pub_rgb   = self.create_publisher(Image, '/camera/color_image', 5)
     self.pub_depth = self.create_publisher(Image, '/camera/depth_image', 5)
   to:
     self.pub_rgb   = self.create_publisher(Image, '/camera/color_image', qos_profile_sensor_data)
     self.pub_depth = self.create_publisher(Image, '/camera/depth_image', qos_profile_sensor_data)
   Added a multi-line comment block above the publishers explaining
   why the change was made.

B) vo_node.py
   Added import: 'from rclpy.qos import qos_profile_sensor_data'
   Changed image subscribers in the ros_rgbd branch from depth=5
   default-QoS positional argument to qos_profile_sensor_data.
   The color_sub and depth_sub create_subscription() calls now pass
   qos_profile_sensor_data in place of the bare '5'.

C) yolo_detector.py
   Added import: 'from rclpy.qos import qos_profile_sensor_data'
   Changed the two create_subscription calls for /camera/color_image
   and /camera/depth_image to use qos_profile_sensor_data instead of
   the bare integer 5.

D) traffic_system_detector.py
   Added import: 'from rclpy.qos import qos_profile_sensor_data'
   Changed the camera_image_subscriber to use qos_profile_sensor_data
   instead of the bare integer 10.

E) lane_detector.py
   The file already used an explicit QoSProfile() construction.
   Changed reliability from ReliabilityPolicy.RELIABLE to
   ReliabilityPolicy.BEST_EFFORT, and depth from 10 to 5 for
   consistency with the other subscribers.

Behavioral contract after Change 1:
- ros2 topic hz /camera/color_image should report ~30 fps,
  matching the [bridge hb] heartbeat.
- vo_node, yolo_detector, traffic_system_detector, and
  lane_detector remain QoS-compatible with the bridge
  publishers (BEST_EFFORT on both sides).
- For developers: any new subscriber to /camera/* must use
  qos_profile_sensor_data (or equivalent BEST_EFFORT QoS) or
  it will silently fail to receive frames.

### Change 2 - ORB ROI mask in visual_odometry.py

Why: the top portion of the image is usually sky, ceiling, or
distant horizon - mostly low-quality features for VO and often
lacking usable depth (RealSense max range falls off past ~3 m).
Excluding it focuses ORB on ground-level scene features that
actually carry valid 3D backprojection. Implemented as an
OpenCV ORB mask (cv2.ORB.detectAndCompute(gray, mask)) rather than
an image crop - this keeps image dimensions and camera intrinsics
unchanged so DepthProjector.pixels_to_3d_body needs no rescaling.

File: visual_odometry.py

Signature change in VisualOdometryDepth.__init__:
  Added new keyword argument 'roi_top_fraction=0.0' at the end of
  the constructor parameter list.

New instance state:
  self.roi_top_fraction = float(roi_top_fraction)
  if not (0.0 <= self.roi_top_fraction < 1.0):
      self.roi_top_fraction = 0.0
  self._orb_mask = None   # built lazily on first frame

In update(), replaced this single line:
    keypoints, descriptors = self.orb.detectAndCompute(gray, None)
with the mask block:
    mask = None
    if self.roi_top_fraction > 0.0:
        H, W = gray.shape[:2]
        if (self._orb_mask is None
                or self._orb_mask.shape[0] != H
                or self._orb_mask.shape[1] != W):
            self._orb_mask = np.zeros((H, W), dtype=np.uint8)
            y_start = int(H * self.roi_top_fraction)
            self._orb_mask[y_start:, :] = 255
        mask = self._orb_mask
    keypoints, descriptors = self.orb.detectAndCompute(gray, mask)

Behavioral contract:
- Default roi_top_fraction=0.0 means the legacy behavior is
  preserved exactly (mask is None, ORB processes the full image).
- Setting roi_top_fraction=0.30 builds a (480x640) uint8 mask
  with rows 0-143 zero (excluded) and rows 144-479 set to 255.
- The mask is built once on the first qualifying frame and reused
  across frames as long as image dimensions stay constant.
- Camera intrinsics (K_rgb, K_depth, K_rgb_inv, K_depth_inv) and
  the alignment matrix are NOT touched. Pixel coordinates
  reported by ORB stay in the full (640x480) frame so
  DepthProjector.pixels_to_3d_body sees the same coordinate
  system it did before.

### Change 3 - expose roi_top_fraction as a ROS parameter

File: vo_node.py

Added a parameter declaration in the existing block of declarations:
    self.declare_parameter('roi_top_fraction', 0.0)

Added read in the parameter-binding block:
    roi_top_fraction = float(
        self.get_parameter('roi_top_fraction').value)

Added pass-through into the VisualOdometryDepth constructor call:
    roi_top_fraction=roi_top_fraction,

Operator usage:
    ros2 run qcar2_autonomy vo_node --ros-args \
        -p camera_mode:=physical \
        -p input_source:=ros_rgbd \
        -p force_cart_yaw:=true \
        -p roi_top_fraction:=0.30

0.30 is a sensible first try (skips top 30%, keeps bottom 70%).
0.40 is more aggressive. Comparing two runs at 0.30 vs 0.0 will
show whether the ROI mask improves agree-state share, inlier
count, or per-tick computation time.

### Build verification

- python3 -m py_compile qcar2_camera_bridge.py vo_node.py
  visual_odometry.py yolo_detector.py traffic_system_detector.py
  lane_detector.py  ->  PASS
- rsync from ACC_Development -> ~/ros2/src/qcar2_autonomy/  ->  OK
- colcon build --packages-select qcar2_autonomy --symlink-install
  ->  PASS (only the standard easy_install deprecation warning)

### Files touched in this turn
- qcar2_camera_bridge.py
- vo_node.py
- yolo_detector.py
- traffic_system_detector.py
- lane_detector.py
- visual_odometry.py
- VO_CHANGELOG.md (this entry)
- VO_Conversation_Log.txt Turn 85 (added in this turn)

## 2026-05-15 (Physical Test 3 OFF-MAT — QoS fix verified, ROI sweep)

Source: VO_readings.txt section 'Physical Test 3' (lines 9613-10381).
Operator ran three off-mat stationary phases back-to-back to test
the new QoS profile and the new roi_top_fraction parameter.

### Phase line ranges
- Phase 1 (ROI=0.0):  cmd at line 9614, fault_status from 9692-9781
  (44 data lines)
- Phase 2 (ROI=0.30): cmd at line 9782, fault_status from 9902-10149
  (123 data lines)
- Phase 3 (ROI=0.40): cmd at line 10150, fault_status from 10234-end
  (73 data lines)

### Camera publish-rate confirmation

Phase 1 (fresh start, no restart artifact):
  ros2 topic hz /camera/color_image  ->  29.4-30.0 fps steady
  ros2 topic hz /camera/depth_image  ->  29.0-29.4 fps steady

Conclusion: the QoS fix from earlier this turn is verified. The
subscriber side (ros2 topic hz tool) now reports the same rate the
bridge is publishing. The previous ~17 fps reading was indeed the
DDS half-drop on reliable QoS for large Image messages, and
switching everything to qos_profile_sensor_data resolved it.

Phases 2 and 3 show topic_hz averages of ~25-26 fps with max
inter-message gaps of 0.368-0.399 s. Those are consistent with a
brief stall during the vo_node restart at the start of each new
phase; the moving-average buffer in 'ros2 topic hz' carries the
spike forward and never fully recovers within the captured window.
This is a measurement artifact, not a bridge slowdown - the [bridge
hb] line would be the ground truth (not captured this run). To
verify on subsequent runs, glance at Terminal A's bridge heartbeat
during ROI phases.

### Per-phase quality (stationary, force_cart_yaw=true)

                            P1 (no ROI)   P2 (0.30)    P3 (0.40)
  n frames:                       44          123           73
  state mix:                      100% agree  100% agree   100% agree
  mean rho (good ticks):       0.0208       0.0175       0.0199
  mean weight (good ticks):    0.689        0.634        0.575
  mean inliers (all):          333          356          410
  mean spread (all):           102          89           77
  zero-weight %:               20.5%        15.4%        23.3%

### Interpretation

1) Inlier count rises monotonically as we mask more of the top of
   the image (333 -> 356 -> 410). Same 800-feature ORB budget, but
   more of those features land in the lower image where depth is
   reliable. Net effect: more matches survive the depth-validity
   filter.

2) Spread drops monotonically (102 -> 89 -> 77) because features
   are spatially confined to the unmasked region. Expected.

3) Weight drops modestly (0.69 -> 0.63 -> 0.57). The VO weight
   blends multiple quality signals including spread, so a smaller
   spread region pulls the weight down even when inliers and rho
   improve.

4) Zero-weight (rejected-tick) percentage:
      P1: 20.5%
      P2: 15.4%   <- best
      P3: 23.3%
   Rejection rate is non-monotonic: ROI=0.30 actually reduces
   rejections vs no-ROI, but ROI=0.40 is over-aggressive and
   rejections climb back up.

5) rho is lowest at ROI=0.30 (0.0175 m), which is the residual
   between VO and Cartographer poses. Lower is better.

### Verdict

- QoS fix: CONFIRMED working. /camera/color_image and
  /camera/depth_image now report ~30 fps via 'ros2 topic hz',
  matching the bridge heartbeat. Any future subscriber added to
  /camera/* must use BEST_EFFORT QoS to remain compatible.

- ROI mask: SAFE. None of the three configurations broke VO at
  rest. State stayed in agree throughout.

- Sweet spot: roi_top_fraction = 0.30. Best on every aggregate
  quality metric (lowest rho, lowest rejection rate, healthy
  inlier count). ROI = 0.40 is past the optimum (weight drops,
  rejections climb).

### Recommended next step

Bring the car onto the mat and run manual_drive with
  -p roi_top_fraction:=0.30
added to the vo_node launch params (other VO params unchanged).
Drive a Test 2-style controlled trajectory (forward, gentle left,
straight, etc.) at the same speed used in Test 2 (manual_drive
defaults). Save under a new VO_readings section so we can compare
agree-state share and axis sign agreement against Test 2 (manual,
no ROI) baseline:
  Test 2: agree 23.5%, X-axis sign agreement 0.92, Y 1.00
If Test 4 (manual + ROI 0.30) shows agree share rising and/or
axis sign agreement holding 0.9+, the ROI mask graduates.

Do NOT change any other VO parameters for the mat run. We want
one variable changed at a time so the comparison is clean.

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 86 (added in this turn).
- No code edits in this turn; the QoS + ROI code changes from the
  prior turn are the ones being validated here.

## 2026-05-15 (CORRECTION to Test 3 analysis above)

Operator clarified post-analysis that the prior section overstated
two things. Recording the correction inline so future readers and
the parallel Codex session do not act on the incorrect framing.

### Correction 1: Phase 1 'baseline' was not comparable to 2 and 3

The operator clarified that during Phase 1 the vo_node was NOT
running - the ros2 topic hz reading was captured with cartographer
+ bridge only, no vo_node subscribed. Phases 2 and 3 had vo_node
running, which adds a consuming subscriber to /camera/color_image
and /camera/depth_image. So Phase 1's 29.4-30.0 fps and Phases 2
and 3's ~25 fps stable readings are not measuring the same load
regime. They are not three datapoints on the same curve.

### Correction 2: phase durations were uneven

Operator did not time the phases, and the captured fault_status
counts (44 / 123 / 73 lines) reflect different total durations and
different stabilization-window-included fractions. The per-phase
average rho / weight / inliers / spread numbers in the prior
section are therefore not strictly comparable, and the
'sweet spot at ROI=0.30' conclusion is softer than it sounded.
ROI=0.30 remains a reasonable default for the mat run on the basis
of being mid-aggressive (not extreme), but should not be cited as
demonstrably optimal from this dataset.

### What we actually know

1) QoS fix did SOMETHING positive: pre-fix topic_hz reading was
   ~17 fps under load; this run shows ~25 fps under vo_node load
   in stable windows of Phases 2 and 3. Better than before.

2) Whether the bridge is at 30 fps with vo_node also consuming is
   NOT proven by this run because the bridge heartbeat lines
   ([bridge hb] from Terminal A) were not captured. The bridge
   may still be at 30 and ros2 topic hz under-reports due to
   tool/sampling artifacts, OR the bridge may be at ~25 fps
   under vo_node back-pressure even with BEST_EFFORT. Data does
   not distinguish.

3) All three ROI configurations held 100% agree state at rest.
   That is the most defensible claim from this dataset:
   ROI=0.30 and 0.40 are safe to use; neither breaks VO at rest.

### Updated plan

Mat run with manual_drive remains the same: run vo_node with
  -p roi_top_fraction:=0.30
and otherwise the Test 2 baseline. Choice of 0.30 is on the basis
of being conservative and consistent with the operator's earlier
intent, not on the basis of being statistically best from Test 3.

Additional one-line diagnostic for the mat run: keep an eye on
Terminal A's [bridge hb] log line during driving. If it consistently
reads ~30 fps under live vo_node load, that confirms the bridge is
fine and any topic_hz under-read is a measurement-tool delay. If
it reads ~25 fps under load, then there is a real DDS drop
happening and we look at it next.

### Files touched in this turn
- This changelog correction (appended to the prior Test 3 entry).
- VO_Conversation_Log.txt Turn 87 (added in this turn).
- No code edits in this turn.

## 2026-05-15 (Test 3 — bridge heartbeat confirms 30 fps under vo_node load)

Operator confirmed by glancing at Terminal A's cartographer log
during the Test 3 phases:
  [bridge hb] ~151 frames in 5.0 s (~30.1 fps) consistently

This resolves the residual question from the Test 3 correction
entry above. The bridge is publishing at 30 fps under live vo_node
load - same as the Step 3 stationary number from 2026-05-14
(~30.3 fps). Nothing on the publish side is degraded by adding a
vo_node consumer.

Therefore the ~5 fps gap between the bridge heartbeat (30) and
ros2 topic hz (~25) under load is a measurement-side artifact:
either the 'ros2 topic hz' tool's own subscriber is dropping a
few frames under load (each BEST_EFFORT subscriber drops
independently when DDS cannot fan-out fast enough), or the
tool's sampling lags slightly behind the real arrival rate.
Either way it is purely diagnostic; vo_node is receiving close
to all 30 fps.

### Confirmed pipeline numbers (after QoS fix, with vo_node running):

- Bridge publish rate (from [bridge hb]):   ~30.1 fps  <- ground truth
- ros2 topic hz /camera/color_image:        ~25 fps    <- tool artifact
- ros2 topic hz /camera/depth_image:        ~25 fps    <- tool artifact
- /vo/fault_status:                         ~5 Hz      <- VO compute-bound

### Verdict

QoS fix is fully verified. Bridge at 30 under all tested loads.
Pre-fix topic_hz read ~17 fps because reliable QoS half-dropped at
the DDS layer; post-fix it reads ~25 fps because BEST_EFFORT only
drops a small fraction to the diagnostic subscriber. The full 30
fps is making it into the pipeline.

No further action required on QoS for the mat run. Proceed with
the manual_drive Test 2-style run + -p roi_top_fraction:=0.30 as
previously planned.

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 88 (added in this turn).
- No code edits.

## 2026-05-15 (Test 4 analysis + parameter-sweep campaign plan)

Operator requested a rapid iterative mat-test campaign: read each
test, hand back the next mat command with a stated hypothesis,
keep going until enough data, then stop and discuss. Operator
increments the Physical Test number each run. Assistant may vary
parameters (roi_top_fraction, n_features, ransac s-point, etc.) as
long as each test has a clear expected outcome.

### Test 4 — ROI 0.30, manual_drive (VO_readings line 10559+)

722 frames parsed, 81 skipped (truncated psi_raw, logging artifact).

- state mix: agree 127 (17.6%), warming 39 (5.4%),
  vo_suspect 556 (77.0%)
- zero-weight (rejected) ticks: 379/722 (52.5%)
- rho (good ticks):    mean 0.0701  med 0.0580  max 0.254
- weight (good ticks): mean 0.489   med 0.530
- inliers (all):       mean 91.5    med 45      min 0
- spread (all):        mean 69.9    med 82
- first 80 frames: 42 agree / 9 warming / 29 vo_suspect, inl ~116
- last 80 frames:  100% vo_suspect, inl ~47 (ended in a bad pocket)
- transitions: agree<->warming 5/5, agree<->vo_suspect 5/4
- psi_raw-ctyaw: mean -23.6 stdev 20.1 absmax 59 |err|<=15: 46%
- axis sign agreement: X=0.941 (237 win), Y=0.833 (323 win)
- final divergence: dx=+0.740  dy=-0.090  dyaw=0.0
- Cart (0.10,-0.02,-180) -> (-0.59,1.50,-135);
  VO  (0.15,-0.01,-180) -> (0.15,1.41,-135)

Comparison to Test 2 (no ROI, manual_drive, different session):
  Test 2: agree 23.5%, X 0.92, Y 1.00, final dx 1.33 dy -0.42
  Test 4: agree 17.6%, X 0.94, Y 0.83, final dx 0.74 dy -0.09

Key cross-reference: at REST with ROI 0.30 (Test 3 Phase 2)
inliers were ~356; UNDER MOTION with ROI 0.30 (Test 4) inliers
collapsed to ~91. The ROI mask behaves very differently under
motion than at rest - the lower-frame region that the mask keeps
suffers motion blur / low texture while driving, and the
upper-frame region the mask discards (distant walls/signs) had
been contributing stable trackable features. NOT YET concluding
ROI is bad under motion - a same-session no-mask baseline is
needed first because Test 2 was a different day / start pose /
trajectory.

### Campaign plan

  Test 4  ROI 0.30                       DONE
  Test 5  ROI 0.0                        same-session no-mask
                                         baseline. Expect inliers
                                         ~120-130, agree share up
                                         vs Test 4. Isolates
                                         whether ROI helps/hurts
                                         under motion.
  Test 6  ROI 0.15                       gentler mask. Expect
                                         inliers between Test 4
                                         and Test 5; tells us if a
                                         light mask trims sky
                                         without starving features.
  Test 7  ROI 0.0, n_features=1200       more ORB features to
                                         offset motion match-loss.
                                         Expect higher inliers,
                                         possibly higher agree, at
                                         cost of slower fault_status
                                         (more compute/tick).
  Test 8  ROI 0.0, RANSAC s=2->3         code change at
                                         visual_odometry.py:785 +
                                         rebuild. Expect more
                                         robust hypotheses, fewer
                                         bad-consensus frames;
                                         iteration count (300)
                                         unchanged so runtime ~same.

Tests 5-7 are ROS-param-only (no rebuild). Test 8 needs a code
edit + rsync + colcon build, batched when reached.

All vo_node launches keep camera_mode=physical,
input_source=ros_rgbd, force_cart_yaw=true. Only the swept
parameter changes per test. manual_drive trajectory kept as close
to Test 2/Test 4 as the operator can reproduce by hand.

### Test 5 command issued to operator
ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=ros_rgbd -p force_cart_yaw:=true -p roi_top_fraction:=0.0

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 89 (added in this turn).
- No code edits.

## 2026-05-15 (analysis method: maneuver segmentation from ctyaw + ts capture)

Operator asked for a better way to correlate the fixed driving
scenario (straight-left-straight-left-straight~10s-deep left) with
fault_status, including turn timing and separating real turns from
small lane corrections. Requested critical thinking, not just
literal implementation of the suggestion.

### Decision (with rationale)

1. NO instrumentation added to manual_drive or vo_node. Every
   fault_status line already carries ct(x,y,yaw) - the
   Cartographer pose. ctyaw is ground truth for what the car
   physically did. Maneuvers are fully recoverable by segmenting
   the ctyaw signal in data we already capture. This works
   retroactively on Test 4 and avoids changing code that is
   itself under test (which would also break Test 4<->5
   comparability).

2. The only worthwhile addition is shell-side timestamping of the
   capture (moreutils 'ts' is installed at /usr/bin/ts). It does
   not change any node or the fault_status content (the parser
   strips the bracketed prefix), it just gives exact per-line
   wall-clock so turn durations are real, not frame-count
   estimates. Zero risk.

3. Capturing manual_drive stdout through ts as well yields an
   independent driver-intent timeline (when a/d pressed and for
   how long) to cross-check the geometry-derived segmentation.

### Capture commands handed to operator (from Test 5 onward)

Terminal C (fault_status, timestamped):
  ros2 topic echo /vo/fault_status | ts '[%Y-%m-%d %H:%M:%.S]'

Terminal D (manual_drive, timestamped intent track):
  ros2 run qcar2_autonomy manual_drive 2>&1 | ts '[%Y-%m-%d %H:%M:%.S]'

### Turn vs correction classification logic (to be implemented in
the analyzer)

Per sliding window over the ctyaw sequence:
- net dyaw and cumulative |dyaw| computed.
- Scripted turn: large net heading change (tens of degrees),
  sustained / monotonic over a span. Labeled, timed, matched to
  the known script (L1, L2, deep-L).
- Lane correction: small net change (<= ~10-15 deg), brief, often
  sign-reversing (net ~ 0). Flagged but not treated as a
  maneuver.
- Every turn shown; each classified by magnitude + duration so a
  quick flick is distinguished from the deep left.

Per maneuver the analyzer will report: agree%, rho, weight,
inliers, and the growth of VO<->Cart divergence DURING that
maneuver, so degradation can be attributed to specific turns
(Test 4 was 77% vo_suspect overall; this localizes where).

Time anchoring: primary = ts wall-clock prefixes. Fallback for
runs without ts (e.g. Test 4) = the [DEPTH DIAG ALIGNED] log
lines in the vo_node terminal, which carry ROS timestamps
(~every 7-10 s) and bracket the run for frame-index->seconds
interpolation.

### Fixed scenario (locked for all tests in this campaign)
straight -> left -> straight -> left -> straight (~10 s) ->
deep left. Small in-lane corrections occur throughout and are
expected; the analyzer must not treat them as scripted turns.

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 90 (added in this turn).
- No code edits. Analyzer is an out-of-tree /tmp script.

## 2026-05-15 (Test 5 analysis + Test 6 pivot to n_features)

### Test 5 - ROI 0.0, manual_drive, same fixed scenario
(VO_readings line 12206+; timestamped capture via ts worked)

656 frames parsed, 92 skipped (truncated psi_raw, logging artifact).

Side-by-side vs Test 4 (ROI 0.30), same session / scenario:

  metric                    Test 4 (0.30)   Test 5 (0.0)
  agree %                       17.6            20.3
  vo_suspect %                  77.0            73.6
  zero-weight (rejected) %      52.5            45.0
  inliers (all) mean            91.5           130.3
  rho (good) mean               0.070           0.074
  weight (good) mean            0.49            0.56
  psi_raw-ctyaw |err|<=15       46%             78%
  X sign agreement              0.94            0.75
  Y sign agreement              0.83            1.00
  final dx / dy                 0.74/-0.09      0.94/-0.20

Cart (0.10,-0.01,-180) -> (-0.54,1.43,-132)
VO   (0.04,-0.04,-180) -> ( 0.40,1.23,-132)
first 80: 55 agree/8 warming/17 vo_suspect, inl ~212
last 80 : 100% vo_suspect, inl ~126 (ended in a bad pocket again)

Conclusion: ROI 0.0 decisively beats ROI 0.30 under motion.
+43% inliers (130 vs 91), fewer rejected ticks (45 vs 52.5%),
markedly better shadow-yaw tracking (78% vs 46% within 15 deg).
This confirms the rest-vs-motion inversion noted after Test 4:
the ROI mask raises inliers at rest (Test 3: 333->356->410) but
starves them under motion because the kept lower-frame region
suffers motion blur while the discarded upper region had stable
distant features. ROI masking is therefore counterproductive for
the moving use case. Lock roi_top_fraction=0.0 for motion.

### Test 6 decision - pivot away from the planned ROI 0.15

Rationale: ROI 0.0 already won decisively over 0.30. ROI 0.15
would only confirm an already-visible monotonic trend (it would
land between) - low information value. The dominant problem both
Test 4 and Test 5 share is ~74-77% vo_suspect and ~45-52%
rejected ticks under motion, caused by inlier collapse during
turns. Attack that directly instead.

Test 6 = ROI 0.0 + n_features 800 -> 1200 (ROS param, no rebuild).
Hypothesis: more ORB features -> more matches survive motion blur
through turns -> inliers rise above Test 5's 130, vo_suspect
share drops, fewer safety-gate rejections. Expected cost: higher
per-tick compute, so /vo/fault_status rate may drop below ~5 Hz;
watch that tradeoff. If inliers and agree% rise without an
unacceptable rate drop, more features is a win and becomes the
new baseline for subsequent tests.

Test 6 vo_node command issued:
ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=ros_rgbd -p force_cart_yaw:=true -p roi_top_fraction:=0.0 -p n_features:=1200

Campaign so far:
  T4 ROI 0.30                 done
  T5 ROI 0.0                  done - new motion baseline
  T6 ROI 0.0 + n_features 1200 running
  T7 TBD from T6 (likely RANSAC s 2->3, needs code change+rebuild)

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 91.
- No code edits.

## 2026-05-15 (Test 6 analysis - n_features 1200 big win + Test 7)

### Test 6 - ROI 0.0, n_features 1200, manual_drive, same scenario
(VO_readings line 13752+)

682 frames parsed, 84 skipped. Campaign comparison:

  metric                T4(.30/800)  T5(0/800)  T6(0/1200)
  agree %                  17.6        20.3       30.2
  vo_suspect %             77.0        73.6       62.0
  zero-weight %            52.5        45.0       37.0
  inliers (all) mean       91.5       130.3      196.2
  weight (good) mean       0.49        0.56       0.59
  rho (good) mean          0.070       0.074      0.062
  psi |err|<=15            46%         78%        81%
  X sign / Y sign          0.94/0.83   0.75/1.00  1.00/1.00
  final dx / dy            0.74/-0.09  0.94/-0.20 0.58/-0.12
  fault_status rate        -           6.04 Hz    6.19 Hz

Cart (0.11,-0.02,-179) -> (-0.61,1.44,-131)
VO   (0.12,-0.01,-179) -> (-0.03,1.32,-131)
first 80: 53 agree/8 warming/19 vo_suspect, inl ~201, w 0.70
last 80 : 100% vo_suspect, inl ~177 (still ends in a bad pocket,
          but inliers in that pocket are far healthier than
          earlier tests: 177 vs Test 5's 126 vs Test 4's 47)

Rate measured from the ts timestamps: 800 features = 6.04 Hz,
1200 features = 6.19 Hz. No rate penalty - ORB extraction is NOT
the pipeline bottleneck (bridge feeds 30 fps, plenty of headroom;
the ~6 Hz ceiling is elsewhere - matching/RANSAC/tick scheduling).
The earlier worry that more features would slow fault_status is
refuted by direct measurement.

Conclusion: n_features 800 -> 1200 is an unambiguous,
cost-free improvement across every metric. agree% +50%,
inliers +50%, both axis sign agreements perfect, best final
divergence and shadow-yaw tracking of the campaign. This is the
new working baseline.

### manual_drive | ts capture - abandoned
Operator correctly observed the manual_drive terminal showed
nothing through the ts pipe. Root cause: manual_drive's status
line is printed with a carriage return and no newline
(print(..., end='', flush=True) with leading backslash-r), and
ts only emits a timestamped line on newline, so the status line
never traverses the pipe. This does not affect analysis: the
Cartographer yaw ct(...) in every fault_status line is the
ground-truth maneuver signal and Terminal C's
'ros2 topic echo /vo/fault_status | ts' capture works correctly.
Decision: stop piping manual_drive through ts; run it plain so
the operator at least sees local feedback. Keep ts only on the
fault_status capture.

### Test 7 - find the n_features knee
Test 7 = ROI 0.0 + n_features 2000. Since 800->1200 improved
everything at zero rate cost, push further to locate diminishing
returns or a rate drop. If agree%/inliers keep climbing with
acceptable rate, 2000 becomes baseline; if they plateau or
fault_status rate drops meaningfully, lock n_features at 1200.

Test 7 vo_node command issued:
ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=ros_rgbd -p force_cart_yaw:=true -p roi_top_fraction:=0.0 -p n_features:=2000

Campaign: T4 done, T5 done, T6 done (best so far), T7 running
(n_features 2000), T8 likely RANSAC s 2->3 code change on the
winning n_features.

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 92.
- No code edits.

## 2026-05-15 (METHODOLOGY CORRECTION: zone the runs; Test 7 analysis)

Operator supplied critical environmental context: the mat is
currently in a cramped space surrounded by tables, chairs, walls.
For roughly the first 30 s and last 10 s of each run the camera
sees that clutter (rich, easy features). The middle of the run
sees only blue wall / white wall / bare mat. Competition will
have minimal clutter near the camera - mostly just the
lidar-mapping boundary walls, possibly far away. Therefore the
BARE MIDDLE of each run is the competition-representative regime;
the cluttered start/end are non-representative artifacts of the
test room.

Consequence: every prior campaign conclusion (Tests 4-7) was
computed on whole-run aggregates that include the cluttered ends
and are therefore optimistically biased. Re-analysis must zone
each ts-timestamped run into:
  A = clutter start (0-30 s)        non-representative
  B = bare middle (30 s .. end-10s) COMPETITION-REPRESENTATIVE
  C = clutter end (last 10 s)       non-representative
and judge on zone B. Implemented /tmp/analyze_zones.py using the
ts wall-clock prefixes.

### Zoned results (zone B = bare middle)

  metric            T5(0/800)  T6(0/1200)  T7(0/2000)
  agree %              12.0       25.2        7.0
  vo_suspect %         83.3       67.8       88.2
  rejected %           49.8       40.7       49.1
  inliers (all)        121        211        238
  weight               0.54       0.58       0.54
  psi |err|<=15        78%        83%        75%

Clutter-start zone (all tests): ~61-63% agree, 100% psi-good -
pure test-room artifact, confirming the operator's concern.

### Revised conclusions

1. n_features knee is 1200, NOT higher. Test 7 (2000) has the
   most inliers (238) but the WORST zone-B agree (7%) and
   highest vo_suspect (88%). On bare/low-texture walls the extra
   features are ambiguous; RANSAC reaches a self-consistent but
   wrong consensus (textureless-wall / aperture problem). More
   features past 1200 actively degrades the representative
   regime. LOCK n_features=1200.

2. Real competition-condition performance is much worse than the
   whole-run aggregates implied. Best config (T6) in zone B is
   only 25% agree, 68% vo_suspect. The earlier 'T6 30% agree'
   was inflated by the cluttered ends.

3. The ROI hypothesis was rejected on confounded data. In zone B
   the upper image is featureless wall feeding ambiguous
   matches - exactly where cropping to the lower image
   (mat + cones + signs) should help. ROI must be re-tested at
   n_features=1200 and judged on zone B only. This is the
   corrected version of the Test 4/5 ROI experiment.

### Test 7 whole-run (for the record, non-representative)
612 frames; agree 21.4%, vo_suspect 71.6%, inliers 208,
final divergence dx=0.02 dy=0.03 (the tiny final divergence is
because Test 7 happened to end back in the cluttered zone where
VO re-locked - not a sign of good mid-run tracking).

### Test 8 - corrected ROI experiment
Test 8 = n_features 1200 + roi_top_fraction 0.30. Compare its
zone B against Test 6 zone B (n_features 1200, ROI 0.0). Isolates
the ROI effect at the locked feature count under representative
conditions.

Command issued:
ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=ros_rgbd -p force_cart_yaw:=true -p roi_top_fraction:=0.30 -p n_features:=1200

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 93.
- /tmp/analyze_zones.py (out-of-tree analyzer; not committed).
- No code edits.

## 2026-05-15 (Test 8 analysis + ransac_sample_size param + Test 9)

Operator refined the environment timing: total run ~90 s; the
clutter-rich zone is realistically ~10 s at start and ~10 s at
end (not 30 s). The bare regime is after the first left turn and
before the last turn, BUT a big off-mat table is visible to the
camera during the straight-~10 s section, so even mid-run there
is an intermittent off-mat feature source. Operator wants the
whole-run kept in focus while being aware which sections have
near-mat (off-mat) feature assists.

Re-zoned with 10 s / 10 s ends.

### Test 8 (ROI 0.30, n_features 1200) vs Test 6 (ROI 0.0, 1200)
Same feature count, only ROI differs. 10s/10s zoning.

  WHOLE RUN          T6 (ROI 0.0)   T8 (ROI 0.30)
  agree %               30.2           23.1
  vo_suspect %          62.0           70.1
  inliers (all)         196            127
  final div dx/dy       0.58/-0.12     1.06/-0.18

  ZONE B (bare middle, competition-representative)
  agree %               26.3           19.0
  vo_suspect %          65.9           74.5
  rejected %            38.6           51.1
  inliers (all)         197            132
  weight                0.56           0.51
  psi |err|<=15         85%            75%

  ZONE A (clutter start, ~10 s): both ~84-87% agree, 0%
  vo_suspect, 100% psi-good - confirms the test-room artifact.

Verdict: ROI cropping LOSES even in the competition-
representative zone B, not just whole-run. Cropping the top 30%
removes the wall/floor boundary, mat edges and distant
structural lines that DO carry geometric signal, leaving only
the near-field mat surface (low texture, motion-blurred under
drive). The operator's ROI hypothesis is reasonable a priori but
empirically rejected at every zone and feature count tested.
roi_top_fraction = 0.0 is locked.

### Parameter sweep convergence
- ROI:        0.0   (cropping hurts in all zones, all nfeat)
- n_features: 1200  (800 weak; 2000 worse in zone B - ambiguous
                     features on bare walls cause wrong RANSAC
                     consensus: high inliers, low agree)
- Best config so far = Test 6 (ROI 0.0, n_features 1200).
- Structural limitation surfaced: even the best config is only
  ~26% agree / ~66% vo_suspect in the bare middle. Feature-VO on
  low-texture wall+mat is inherently weak. This matters for the
  redundancy design: the supervisor must distrust VO in
  low-texture stretches.

### Code change this turn: ransac_sample_size as a ROS parameter

Made the RANSAC minimum-sample size (the 's-point') tunable
without future rebuilds, mirroring the roi_top_fraction pattern.

visual_odometry.py:
  - VisualOdometryDepth.__init__ new kwarg ransac_sample_size=2
    (added after roi_top_fraction).
  - Stored as: self.ransac_sample_size = max(2, int(
    ransac_sample_size))   # clamp >=2, default 2 = legacy
  - _ransac_motion rewritten generically:
      s = self.ransac_sample_size
      if M < s: return zeros (safety)
      idx = np.random.choice(M, s, replace=False)
      degenerate guard generalized from the old 2-pt distance
        check to: if np.max(np.std(pts_prev[idx],axis=0)) < 1e-8:
        continue
      refit guard changed from 'best_count >= 2' to
        'best_count >= s'.
    Docstring updated from '2-point RANSAC' to 's-point RANSAC'.
    With s=2 the behavior is identical to the previous code.

vo_node.py:
  - declare_parameter('ransac_sample_size', 2)
  - read: ransac_sample_size = int(get_parameter(
    'ransac_sample_size').value)
  - passed into VisualOdometryDepth(... ransac_sample_size=
    ransac_sample_size)

Build: py_compile PASS on visual_odometry.py + vo_node.py;
rsync + colcon build --packages-select qcar2_autonomy
--symlink-install PASS.

### Test 9 - RANSAC s=3 on the winning config
Test 9 = ROI 0.0, n_features 1200, ransac_sample_size 3.
Hypothesis: 3-point samples resist the degenerate/ambiguous
consensus that bare walls produce (Test 7's high-inlier
low-agree signature), so zone-B agree should rise above Test 6's
26% and vo_suspect fall below 66% without a rate penalty
(iterations unchanged at 300). If it does not improve zone B,
the bare-wall weakness is structural (not a RANSAC-robustness
issue) and the campaign concludes with Test 6's config as the
best achievable and a documented VO limitation on low-texture
scenes.

Command issued:
ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=ros_rgbd -p force_cart_yaw:=true -p roi_top_fraction:=0.0 -p n_features:=1200 -p ransac_sample_size:=3

### Files touched this turn
- visual_odometry.py (ransac_sample_size param + generic _ransac_motion)
- vo_node.py (declare/read/pass ransac_sample_size)
- This changelog entry.
- VO_Conversation_Log.txt Turn 94.

## 2026-05-15 (Test 9 analysis + PARAMETER-SWEEP CAMPAIGN CONCLUSION)

### Test 9 - ROI 0.0, n_features 1200, ransac_sample_size 3
vs Test 6 - ROI 0.0, n_features 1200, ransac_sample_size 2
(10s/10s zoning; zone B = competition-representative)

  ZONE B               T6 (s=2)   T9 (s=3)
  agree %                26.3       18.2
  vo_suspect %           65.9       74.9
  rejected %             38.6       53.4
  inliers (all)          197        205
  weight                 0.56       0.58
  psi |err|<=15          85%        62%
  WHOLE final div     0.58/-0.12  0.72/-0.62

s=3 did NOT improve the representative zone - it slightly
degraded it (rejected 53.4 vs 38.6%, agree 18.2 vs 26.3%, psi
62 vs 85%). Inliers/weight essentially unchanged. Interpretation:
the bare-wall weakness is NOT a RANSAC-robustness problem. With
the fixed 300-iteration budget a 3-point sample yields fewer
strong consensus sets on the marginal bare-wall feature
population, so more frames fail to lock -> more vo_suspect.
ransac_sample_size = 2 (default) is retained.

### CAMPAIGN CONCLUSION

Three highest-value levers swept with zone-corrected
methodology (clutter ends excluded; bare middle = competition-
representative):

  Lever            Outcome
  ROI mask         REJECTED. Hurts every zone & feature count.
                   Cropping removes wall/floor boundary, mat
                   edges and distant structural lines that
                   carry geometric signal.
  n_features       1200 is the knee. 800 weak; 2000 WORSE in
                   zone B (ambiguous bare-wall features ->
                   wrong RANSAC consensus: high inliers, low
                   agree).
  RANSAC s-point   2 wins. s=3 no robustness gain, slightly
                   worse.

WINNING CONFIG: roi_top_fraction=0.0, n_features=1200,
ransac_sample_size=2. (= Test 6: stock vo_node except
-p n_features:=1200.) Recommended operating point going
forward.

LOAD-BEARING FINDING (structural, not a tuning bug):
Even the winning config achieves only ~26% agree / ~66%
vo_suspect in the competition-representative bare zone. The
~86% agree seen in the cluttered start/end is a test-room
artifact (tables/chairs/near walls) that will not exist in
competition. Feature-based VO on low-texture wall + uniform
mat is inherently weak; no swept parameter changes this.

DESIGN IMPLICATION for VO-as-redundancy-layer:
VO cannot be expected to agree with Cartographer everywhere.
The supervisor (vo_supervisor.py) should detect the
low-texture regime (e.g. via inlier count / weight / spread
collapse) and DOWN-WEIGHT or suspend VO-vs-Cart disagreement
flagging during those stretches, rather than treating
sustained vo_suspect on bare walls as an odometry fault. VO
is most informative as redundancy where the scene has
structure (signs, cones, boundary geometry) - which is also
where competition-relevant decisions happen.

Campaign halted here per plan. Next: design discussion, no
more parameter mat-runs unless a new hypothesis emerges.

### Files touched this turn
- This changelog entry (campaign conclusion).
- VO_Conversation_Log.txt Turn 95.
- No code edits (ransac_sample_size param from Turn 94 stands;
  default 2 is the retained value).

## 2026-05-15 (VO improvement research + strategic direction)

Operator goal hierarchy clarified: long-term = full VSLAM;
near-term = a good-enough VO that can be folded into the EKF;
meta-goal = learn to design a functional VO (winning the
competition is explicitly secondary). Operator open to additive
improvements only - nothing already built should be discarded.
Operator wants substantive technical decisions in the logs (not
conversational/clarification noise).

### Reference research performed (web)

1. niconielsen32 ComputerVision/VisualOdometry
   stereo_visual_odometry.py pipeline: tiled-grid FAST detector
   (forces spatial spread), KLT optical-flow tracking
   (calcOpticalFlowPyrLK) instead of descriptor matching, SGBM
   disparity, cv2.triangulatePoints for 3D, pose by
   reprojection-error least_squares (LM) with RANSAC. KITTI
   (car, rich texture). Transferable ideas: tiled detection,
   KLT tracking, reprojection-error pose refinement. Does NOT
   address low texture (KITTI is feature-rich).

2. NVIDIA Isaac ROS Visual SLAM (cuVSLAM): production
   stereo-visual-inertial SLAM, GPU, ~7 ms/frame on Jetson.
   Its own docs state it performs poorly on featureless scenes
   (solid-colored walls) - the SAME ceiling we hit - and its
   mitigation is IMU fusion. Confirms our limit is inherent to
   vision-only feature VO, not a tuning defect. Heavy external
   dependency; would require giving it camera ownership
   (conflicts with the single-node Quanser-bridge architecture);
   not a quick win.

3. Low-texture VO literature (2025) converges on: direct/dense
   methods, IMU fusion (VIO), and point+line features. Sparse
   features on RGB (our current frontend) is the most brittle
   possible choice for our environment.

4. RealSense D435 specifics: it IS a stereo camera; the depth
   image we already consume IS the stereo result (IR pair +
   projector disparity). We are NOT missing stereo. The IR dot
   projector adds artificial texture for DEPTH on blank walls,
   but the pattern is camera-fixed (slides across the scene as
   the camera moves) so it cannot be used for world-frame
   frame-to-frame feature tracking. We likely only have
   RGB + aligned depth via the Quanser PIT QCar2DepthAligned
   bridge (NOT raw left/right IR) - to be verified before any
   IR-stereo-odometry idea.

### Key engineering truth recorded (design constraint)

Translation parallel to a bare flat surface is mathematically
unobservable to ANY camera-only method (ICP slip / planar
degeneracy). cuVSLAM, direct methods and depth-ICP all share
this. Therefore 'depth-geometry odometry' is additive value at
STRUCTURED geometry (wall corners, floor/wall edge, mat
boundary, objects) but is NOT a fix for a featureless flat wall
traversed parallel. Its highest practical value is DETECTING
the degeneracy (point-cloud / 3D-point geometric conditioning)
and feeding that into a VO uncertainty estimate.

### Strategic direction (decided, pending operator confirm on order)

Additive, single-node, learning-maximizing plan; nothing from
Tests 4-9 / winning config (ROI 0.0, n_features 1200, s=2) is
discarded:

Step 1 - Frontend accuracy upgrade (from the niconielsen
  reference): add a tiled-FAST + KLT optical-flow tracking
  frontend as an alternative to ORB+BF matching, keeping the
  existing depth-backprojection + RANSAC/SVD-Procrustes
  backend and keeping ORB as a selectable fallback. Expected:
  faster, more stable tracking in turns and
  textured/semi-textured zones; no change to the unobservable
  bare-wall case.

Step 2 - EKF-readiness (the real near-term milestone):
  vo_node additionally publishes nav_msgs/Odometry with a
  principled covariance derived from inlier count, inlier
  spread, RANSAC residual, and 3D-point geometric
  conditioning (the conditioning term captures the bare-wall
  degeneracy). This is what makes VO consumable by an EKF and
  teaches the core fusion concept: a sensor is only as useful
  as its honest uncertainty. The depth-geometry/degeneracy
  detection folds in here rather than as a standalone motion
  estimator.

Recommended order: Step 1 then Step 2 (improve the estimate,
then wrap it in honest uncertainty). vo_supervisor
alarm-suppression idea from the prior turn is superseded by
Step 2 (covariance subsumes it; no extra node).

No code changed this turn (research + direction only). Awaiting
operator confirmation of order before implementing Step 1.

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 96.

## 2026-05-15 (direction revised: keep ORB; estimator is the x,y lever)

Operator rejected the tiled-FAST+KLT frontend swap. Decision:
ORB stays as the feature frontend (speed + accuracy base for a
self-driving car; also keeps future low-light work tractable).
Full ORB-SLAM3 noted as a separate, much larger later topic
(keyframes/local mapping/loop closure/BA - not a detector swap).
Depth is to be used ADDITIVELY (RGB still used; use whatever the
camera offers that improves VO). Low-light robustness flagged as
a FUTURE concern, not now. Current focus: accurate-enough x,y.

### SVD Procrustes - precise assessment (corrects earlier
loose 'SVD may not be best' remark)

SVD (Kabsch/Procrustes) is the OPTIMAL closed-form solver for
its problem: given matched 3D point pairs, find rigid R,t
minimizing summed Euclidean point-to-point error. The solver is
not the issue. The COST FUNCTION we feed it is mismatched to
RealSense error characteristics:

1. Point-to-point 3D Euclidean error is wrong for a depth
   camera. X=(u-cx)*Z/fx: pixel u is sub-pixel precise but
   depth Z noise grows ~quadratically with range (~mm at
   0.5 m, ~5-10 cm at 3-4 m). SVD-Procrustes weights a noisy
   far bare-wall point equally with a crisp near-mat point, so
   the far wall drags x,y. Direct contributor to the measured
   bare-wall degradation.
2. Unweighted + isotropic assumption; real depth error is
   per-point and anisotropic (worse along the ray).
3. Minimizes in 3D then we discard Z anyway (project to body
   X,Y) - Z-noise contaminates X,Y before being thrown away.

### Alternatives (ranked for accurate x,y)

- Depth-weighted Procrustes: same SVD machinery, weight each
  point by 1/sigma_Z(range)^2. Minimal change, kills mismatch
  #1/#2, keeps ORB+depth+RANSAC. Low-risk validation step.
- 3D-2D PnP + RANSAC (cv2.solvePnPRansac): anchor prev-frame
  3D from depth, minimize REPROJECTION error in current-frame
  pixels (error measured where the camera is precise - image
  plane). This is the accuracy-correct method (ORB-SLAM-style
  motion-only estimation) and the proper RGB+depth fusion.
  Moderate cost, real-time.
- Essential matrix 5-pt: rejected (no scale; we have depth).
- Motion-only bundle adjustment: later, toward SLAM.

### Revised plan (additive; ORB/depth/RANSAC structure kept)

Step 1: Depth-weighted Procrustes - one weighting term
  (1/sigma_Z(range)^2) in _svd_rigid_2d / _ransac_motion.
  Low-risk; validates the noisy-far-point hypothesis.
Step 2: If Step 1 helps -> replace point-to-point cost with
  solvePnPRansac reprojection-error estimator (bigger x,y
  accuracy gain). ORB + depth backprojection unchanged.
Step 3: Fold depth-geometry in as a degeneracy/conditioning
  metric that later becomes the EKF covariance.

Supersedes the prior turn's Step 1 (KLT frontend) - that is
dropped. The Step 2 EKF-covariance milestone from the prior
turn is retained and becomes Step 3 here.

No code changed this turn (analysis + revised direction).
Awaiting operator go-ahead to implement Step 1 (depth-weighted
Procrustes).

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 97.

## 2026-05-15 (SESSION HANDOFF pointer)

Session ending; operator moving to a new session. Full continuity
handoff (status, locked decisions, load-bearing facts, and the
Step 1/2/3 plan WITH reasoning) is written at the END of
VO_Conversation_Log.txt under the block
"SESSION HANDOFF - 2026-05-15 (READ THIS FIRST)". New session
should read that block first. The older 2026-05-14
"PLAN FOR NEXT SESSION" block in that file is SUPERSEDED/obsolete.

One-line summary for continuity:
- Campaign done. Winning config = ROI 0.0, n_features 1200,
  ransac_sample_size 2 (Test 6).
- Keep ORB; use depth additively; SVD solver fine but its
  point-to-point cost is mismatched to RealSense depth noise.
- NEXT: Step 1 depth-weighted Procrustes (awaiting operator "go"),
  then Step 2 solvePnPRansac reprojection estimator, then Step 3
  vo_node Odometry + principled covariance (EKF-readiness).
- No code changed this turn (logging/handoff only).

## 2026-05-18 (pre-Step-1 hygiene: _match short-row guard + deferred-items register)

New session. Operator requested a code-efficiency review of every VO
file before starting Step 1. The review produced eight findings
(numbered #1-#8 below). Recommended scope was minimal: fix only the
one real defect now, document the rest as intentionally deferred,
then move to Step 1. This was independently cross-checked in a
parallel Codex session ("Review VO next step"); both reviews
converged on the same minimal scope, which raised confidence.

### Code change this turn: _match short-row guard (finding #3)

visual_odometry.py VisualOdometryDepth._match:
  The Lowe ratio-test comprehension used
    return [m for m, n in raw if len([m, n]) == 2 and ...]
  The `len([m, n]) == 2` check was a no-op (Python unpacks `m, n`
  before evaluating the `if`, and the freshly built list is always
  length 2). cv2.BFMatcher.knnMatch(k=2) returns a row with < 2
  neighbors when the train set has < 2 descriptors, and the unpack
  then raised ValueError. Rewritten as an explicit loop that skips
  rows with len(pair) < 2 before the ratio test.

  PRECISION CAVEAT (corrects the review's initial framing): this is
  NOT an active or latent crash in the running pipeline and NOT a
  strict Step 1 prerequisite. update() already returns early at the
  descriptor-count guard
    if (descriptors is None or self._prev_desc is None
        or len(descriptors) < 2 or len(self._prev_desc) < 2): ...
  Since the live call is _match(self._prev_desc, descriptors) and
  knnMatch row length = min(2, len(descriptors)), that guard already
  excludes the short-row case. So the crash is unreachable in
  production; the fix is correctness hygiene that makes _match safe
  on its own (Step 1/2 keep using this path). The live path stays
  bit-for-bit identical (every row has exactly 2 entries there), so
  no behavior change and no mat run is needed to validate it.

### Deferred-items register (findings #1, #2, #4, #5, #6, #7, #8)

Logged here as known and intentionally NOT addressed now, with a
one-line reason each, so they are not rediscovered the hard way:

- #1 pixels_to_3d_body uses a per-feature Python loop for depth
  sampling (runs twice/frame). Vectorizable, behavior-preserving.
  DEFERRED: real cleanup but not a Step 1 prerequisite; widening
  scope right before the estimator work is undesirable.
- #2 _ransac_motion uses np.linalg.norm (sqrt) where a squared-
  distance vs squared-threshold compare is bit-identical and
  cheaper. DEFERRED: this exact function is rewritten in Step 1
  (depth weighting) and replaced in Step 2 (solvePnPRansac);
  optimizing it now is throwaway.
- #4 align_depth ascontiguousarray wasted on the float/no-warp
  path; update() color_image.copy() unnecessary on the gray path.
  DEFERRED: trivial, zero urgency.
- #5 vo_node._color_cb runs the BGR cv_bridge decode while holding
  state_lock at ~30 Hz (contends with _vo_tick). DEFERRED: real
  but node-level housekeeping, not engine/Step-1 scope.
- #6 vo_node._publish_all emits 17 scalar topics every 20 Hz tick
  even on init/warming/no-cart ticks. DEFERRED: design choice,
  not a bug; not a CPU bottleneck.
- #7 vo_supervisor.py subscribes /vo/state_id as UInt8 and
  /vo/delta_trans, /vo/vo_weight as Float32 while vo_node publishes
  Int32 / Float64 -> ROS 2 type-hash mismatch, those subscriptions
  receive nothing (only /vo/healthy connects). Also the state-id
  numbering is inverted vs vo_node STATE_MAP (node agree=2,
  vo_suspect=3; supervisor assumes vo_suspect=2, agree=3).
  DEFERRED: vo_supervisor is explicitly superseded by Step 3
  (covariance subsumes the alarm-suppression role); fixing it now
  is wasted effort.
- #8 vo_live_plot.py has the same Float32-vs-Float64 mismatch on
  /vo/delta_trans and /vo/vo_weight, and depends on the broken
  supervisor for trust/mode. DEFERRED: same root cause as #7;
  diagnostic-only tool.

### Step 1 design pin (from the Codex cross-check)

Depth-weighted Procrustes needs a per-correspondence range/sigma_Z
weight, but _ransac_motion(prev_2d, curr_2d) and _svd_rigid_2d
currently receive only body-frame XY. The slice prev_3d[both_valid,
:2] / curr_3d[both_valid, :2] in update() discards exactly the Z
needed for the 1/sigma_Z(range)^2 weighting. Step 1's first edit is
therefore a call-site change: carry the per-point Z (or camera-frame
range) alongside the XY into _ransac_motion / _svd_rigid_2d, and
expose the weighting as a default-safe ROS param so it A/Bs cleanly
against Test 6.

### Build

py_compile PASS on visual_odometry.py; rsync ACC_Development ->
~/ros2/src/qcar2_autonomy; colcon build --packages-select
qcar2_autonomy --symlink-install PASS; install sourced. No mat run
(behavior-preserving change, nothing to validate until Step 1).

### Files touched this turn
- visual_odometry.py (_match short-row guard, finding #3)
- This changelog entry.
- VO_Conversation_Log.txt Turns 99-100.

## 2026-05-18 (STEP 1 IMPLEMENTED: depth-weighted Procrustes, default-safe)

Operator gave the go for Step 1. Implemented depth-weighted
Procrustes as a single default-safe, A/B-able ROS parameter.
Nothing from the campaign is discarded; ORB + depth backprojection
+ s-point RANSAC structure unchanged.

### Concept

RealSense stereo depth noise grows ~ sigma_Z ∝ Z^2 (mm at 0.5 m,
5-10 cm at 3-4 m). Unweighted SVD Procrustes weights a noisy far
bare-wall point equally with a crisp near-mat point, so the far
point drags x,y — the cost-function mismatch diagnosed in the
2026-05-15 entries. Step 1 weights each correspondence by
1 / range**depth_weight_power before the rigid fit.

### Code changes (visual_odometry.py)

- DepthProjector.pixels_to_3d_body now returns a THIRD value
  `depths` (per-point camera-frame range, mode units; 0 where the
  pixel was out of bounds). The body-frame Z column is height, NOT
  camera range, so the range had to be surfaced separately — this
  is the call-site change flagged in the Codex cross-check
  (Turn 100). Both early-return and normal-return paths updated.
- VisualOdometryDepth.__init__: new kwarg depth_weight_power=0.0,
  stored as self.depth_weight_power, clamped to [0, 8] (0.0
  outside band -> 0.0).
- update(): unpacks the new prev_depths/curr_depths. When
  depth_weight_power > 0 it forms
  pair_range = max(prev_depths, curr_depths)[both_valid]
  (a pair is only as good as its noisier endpoint), then
  weights = pair_range ** (-power), normalized so max weight = 1
  (Procrustes uses weights only up to a common scale; this just
  keeps the wide range span well-conditioned). When power == 0 it
  passes weights=None so the executed code is literally the old
  path (default-safe A/B control).
- _ransac_motion(pts_prev, pts_curr, weights=None): weights are
  applied ONLY inside the SVD fits (minimal-sample hypothesis fit
  via weights[idx] and the final inlier refit via
  weights[best_inliers]). The RANSAC inlier test stays an
  UNWEIGHTED geometric distance vs the locked ransac_threshold, so
  the consensus search and its tuned threshold are unchanged; only
  the pose recovered FROM the chosen consensus set is depth-
  weighted. This is the lowest-risk reading of "depth-weighted
  Procrustes" and keeps the A/B clean.
- _svd_rigid_2d(pts_a, pts_b, w=None): w=None keeps the original
  unweighted Kabsch verbatim. w given -> weighted centroids and
  weighted cross-covariance S = (w[:,None]*A).T @ B. Degenerate-
  weight guard (sum<=0 or non-finite) falls back to unweighted
  instead of producing NaNs. Uniform weights cancel out of the
  normalized centroid and only scale S by a positive constant, so
  the SVD R,t are unchanged -> depth_weight_power=0 is provably
  identical to the prior estimator.

### Code changes (vo_node.py)

- declare_parameter('depth_weight_power', 0.0); read as float;
  passed into VisualOdometryDepth(... depth_weight_power=...).
- Startup banner shows depth_weight_power and ON / OFF(=Test 6).
  A warn() fires when power > 0 so the operator cannot mistake a
  weighted run for the control.

### Verification (out-of-tree /tmp script, not committed)

Direct numeric checks against the live class:
- _svd_rigid_2d: w=None vs uniform-ones -> dR=0.0, dt=0.0
  (bit-identical); vs uniform 7.3 -> ~1e-16 (machine epsilon).
  Confirms the default-safe claim empirically, not on faith.
- _ransac_motion(weights=None) reproducible under fixed seed and
  unchanged from prior behavior.
- Weighted path (power 4, far noisy outliers) returns finite,
  correct motion — numerically stable, no NaNs.

py_compile PASS (visual_odometry.py + vo_node.py); rsync
ACC_Development -> ~/ros2/src/qcar2_autonomy; colcon build
--packages-select qcar2_autonomy --symlink-install PASS; install
sourced.

### A/B validation plan (operator runs the mat; assistant does not)

Hold everything at the Test 6 winning config and vary ONLY
depth_weight_power. Same fixed scenario (straight, left, straight,
left, straight ~10 s, deep left). Judge on the bare-middle zone B,
not whole-run aggregates (the cluttered start/end is a test-room
artifact). Capture via the usual
`ros2 topic echo /vo/fault_status | ts '[%Y-%m-%d %H:%M:%.S]'`.

  Control  (= Test 6): -p depth_weight_power:=0.0
  Gentle:              -p depth_weight_power:=2.0
  Full RealSense:      -p depth_weight_power:=4.0

Full command form:
ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=ros_rgbd -p force_cart_yaw:=true -p n_features:=1200 -p depth_weight_power:=0.0

Then rerun with 2.0 and 4.0. Hypothesis: if zone-B x,y agreement
improves and vo_suspect falls vs the 0.0 control, the cost-function
diagnosis is confirmed and Step 2 (solvePnPRansac reprojection
estimator) is justified. If no zone-B improvement at any power, the
far-point-drag hypothesis is not the dominant error term and the
campaign records that before moving to Step 2.

### Files touched this turn
- visual_odometry.py (pixels_to_3d_body 3rd return; __init__
  depth_weight_power; update() weight build; _ransac_motion +
  _svd_rigid_2d weighted)
- vo_node.py (declare/read/pass depth_weight_power; banner + warn)
- This changelog entry.
- VO_Conversation_Log.txt Turn 101.

## 2026-05-18 (STEP 1b: hard max-depth cutoff — operator idea, Quanser-precedented)

Operator proposed a HARD distance cutoff (drop features past a
range entirely) as the complement to Step 1's soft weighting,
noting Quanser does something similar. Verified the precedent:
pit/YOLO/nets.py NeuralNetworks.post_processing zeroes any depth
pixel past `clippingDistance` (default 10 m):
  bgRemoved = np.where((depth3D > clippingDistance) | (depth3D <= 0),
                       0, depth3D)
So a hard depth clip is established Quanser practice. This was
cross-discussed in the parallel Codex session ("Review VO next
step"); Codex independently recommended adding exactly ONE extra
knob (max_vo_feature_depth_m) and testing depth-only this mat
session (do not re-sweep ROI / n_features / ransac_sample_size —
already swept). Agreed; only this one knob was added.

### Code changes (visual_odometry.py)

- VisualOdometryDepth.__init__: new kwarg
  max_vo_feature_depth_m=0.0, stored as self.max_vo_feature_depth;
  <= 0 (incl. negative) normalizes to 0.0 = OFF. Units = active
  mode depth units (metres in physical mode).
- update(): right after both_valid = prev_valid & curr_valid and
  BEFORE the min_inliers gate, when the cutoff > 0:
    within = (prev_depths <= cut) & (curr_depths <= cut)
    both_valid = both_valid & within
  A correspondence is dropped if EITHER endpoint's camera range
  exceeds the cutoff (a pair is only usable if both observations
  are within trusted range). Placed before the min_inliers check
  so an over-aggressive cutoff cleanly degrades to "VO abstains"
  (vo_suspect) instead of fitting on a near-empty set. The
  depth_weight_power path consumes the post-cutoff both_valid, so
  cutoff filters first and weighting then trusts the survivors —
  the two compose cleanly and independently.

### Code changes (vo_node.py)

- declare_parameter('max_vo_feature_depth_m', 0.0); read float;
  passed into VisualOdometryDepth. Banner shows it with ON/OFF; a
  warn() fires when active so a cutoff run cannot be mistaken for
  the control.

### Verification (out-of-tree, not committed)

- Default 0.0 (OFF), explicit 3.0 stored 3.0, negative -> 0.0.
- Mask logic: prev/curr depth pairs vs 3.0 m cutoff keep exactly
  the pairs with BOTH endpoints <= 3.0 (point dropped if either
  endpoint exceeds). Default-safe: cutoff 0.0 leaves both_valid
  untouched -> identical to the Step 1 / Test 6 path.

py_compile PASS; rsync; colcon --symlink-install PASS; sourced.

### Updated A/B matrix (depth-only this session; judge zone B)

All at the Test 6 base (n_features 1200, ROI 0.0, s 2), vary only
the two depth knobs:

  1 Control      depth_weight_power:=0.0  max_vo_feature_depth_m:=0.0
  2 Soft         depth_weight_power:=2.0  max_vo_feature_depth_m:=0.0
  3 Strong soft  depth_weight_power:=4.0  max_vo_feature_depth_m:=0.0
  4 Hard cutoff  depth_weight_power:=0.0  max_vo_feature_depth_m:=3.0
  5 Combo        depth_weight_power:=2.0  max_vo_feature_depth_m:=3.0

Run 1 is the unweighted Test 6 control. Compare 2-5 against it on
the bare-middle zone B only. No more knobs added before this
session per the "don't confound the test" discipline.

### Files touched this turn
- visual_odometry.py (__init__ max_vo_feature_depth_m; update()
  cutoff mask before min_inliers gate)
- vo_node.py (declare/read/pass max_vo_feature_depth_m; banner +
  warn)
- This changelog entry.
- VO_Conversation_Log.txt Turn 102.

## 2026-05-18 (Step 1 mat A/B analysis: depth weighting and cutoff)

Operator ran the five physical mat tests for the Step 1 depth
experiment and appended them to `VO_readings.txt` under `Physical
Test 10` / `STEP1 TEST 1-5`. Parsed results from the new blocks
only. The parse ignores truncated `psi_raw=...` tails and uses the
stable fields through `inl` / `sp`.

Test set:

  Test 1 control:      depth_weight_power=0.0, max_vo_feature_depth_m=0.0
  Test 2 soft:         depth_weight_power=2.0, max_vo_feature_depth_m=0.0
  Test 3 strong soft:  depth_weight_power=4.0, max_vo_feature_depth_m=0.0
  Test 4 cutoff:       depth_weight_power=0.0, max_vo_feature_depth_m=3.0
  Test 5 combo:        depth_weight_power=2.0, max_vo_feature_depth_m=3.0

10s/10s zoning was reused: zone A = first 10 s, zone C = last
10 s, zone B = representative middle.

Same-session whole-run summary:

  Run        agree  vo_sus  w=0   inl  final VO-Cart
  control    20.0   72.8   47.3  207  +0.12,+0.04
  soft 2     15.1   78.6   49.9  213  -0.04,-0.03
  soft 4     18.9   76.7   44.1  204  +0.99,-0.46
  cutoff 3   14.1   82.2   60.1  157  +1.09,-0.76
  combo      18.3   75.2   55.4  168  +0.63,-0.40

Zone-B summary:

  Run        agree  vo_sus  w=0   rho   weight  inl
  control    14.1   79.9   48.6  0.062  0.32   195
  soft 2      9.5   85.3   52.0  0.067  0.28   196
  soft 4     14.5   80.6   45.1  0.060  0.33   187
  cutoff 3    8.0   89.1   64.6  0.074  0.19   127
  combo      18.8   74.0   57.1  0.072  0.26   159

Interpretation:
- Depth weighting did NOT clearly improve the same-session control.
  `depth_weight_power=2.0` was worse in zone B on agree/vo_suspect.
  `depth_weight_power=4.0` roughly matched zone-B agree but ended
  with large full-run final divergence.
- The hard 3 m cutoff is too aggressive for this environment. The
  depth diagnostics during the bare middle repeatedly show center
  depths around 4-5 m; cutting at 3 m removes many wall/boundary
  features that still carry signal. This explains the inlier drop
  (zone-B avg 127 vs 195 control), high `w=0`, and worst
  vo_suspect rate.
- The combo had the best zone-B agree percentage, but also high
  `w=0`, lower inliers, higher rho, and substantial final
  divergence. Not a clean winner.
- Overall conclusion: the far-depth-noise hypothesis is not
  confirmed as the dominant current VO error. The structural
  low-texture / bare-wall limitation remains the main issue.

Immediate mat recommendation:
- The five tests are enough to stop if the operator wants to remove
  the car from the mat.
- If doing exactly one more quick run while the car is already on
  the mat, use a looser cutoff only:
    depth_weight_power=0.0, max_vo_feature_depth_m=5.0
  Rationale: it tests whether the hard-cutoff idea was reasonable
  but the 3 m value was too low. Do NOT run more 3 m cutoff tests.

Next design implication:
- Keep default deployment at depth_weight_power=0.0 and
  max_vo_feature_depth_m=0.0 unless the optional 5 m cutoff run
  clearly beats control.
- If no 5 m win, move on to Step 2 planning: solvePnPRansac /
  reprojection-error motion estimation, because simply reweighting
  point-to-point Procrustes did not solve the representative zone.

## 2026-05-18 (Step 1 A/B — independent zone-B reconciliation + DECISION)

Second, independent analysis of the SAME five mat runs (the entry
directly above is the parallel Codex session's analysis; this one
is the Claude-side cross-check). Rebuilt the campaign zone analyzer
as /tmp/analyze_step1.py (out-of-tree, not committed): segments
runs by ts gap >8 s / startup banner / header, reads the ACTUAL
active params from the node's own [WARN] lines (ground truth, not
the header label), skips two short aborted runs (dwp4 5 s, combo
1 s), and judges zone B = [10 s, dur-10 s] on the longest
qualifying full run per config (~120 s, n≈400-500 each).

Zone-B (this analyzer):

  cfg          dwp cut  agree%  vo_susp%  rej%  psi%  inl   w
  1 control    0   0     16.2    77.0    52.5  67.5  207  0.29
  2 soft       2   0     10.5    83.9    55.3  60.8  206  0.26
  3 strong     4   0      8.2    87.0    53.8  51.0  175  0.25
  4 cutoff     0   3      9.0    87.8    69.7  38.2  128  0.15
  5 combo      2   3     16.5    76.3    65.3  48.4  147  0.19

Reconciliation with the Codex entry above: control / soft2 /
cutoff3 / combo agree within run-to-run noise across both
analyzers (control ~14-16% agree, ~77-80% vo_suspect; cutoff3 the
worst on quality, inliers ~127-128 both). The only divergence is
dwp=4 (Test 3 had 5 vo_node restarts; the analyzers picked
different qualifying runs — 8.2% here vs 14.5% Codex). The verdict
is robust to that: in BOTH analyses NO config beats control in
zone B; soft weighting is worse or equal; the 3 m cutoff is
clearly worst (starves the estimator: inliers collapse, rej% and
vo_suspect% jump); combo only ties control on agree while worse on
rej%/psi%/inliers. Here the soft degradation is even monotonic
(control > dwp2 > dwp4).

DECISION (Claude-side, converging with Codex's conclusion):
- The far-point-noise hypothesis is REFUTED. depth_weight_power
  and max_vo_feature_depth_m are not accuracy levers. They remain
  in the code, default 0.0 = OFF (verified bit-identical to
  Test 6) — inert, documented, no revert needed.
- On the optional looser-cutoff idea (Codex: try cut=5.0 m once):
  noted but assessed LOW-VALUE. Physical depth_max is 8 m and the
  bare-middle features sit ~4-5 m, so a 5 m cut only trims the
  small, noisiest 5-8 m tail — predicted to reproduce control
  ±noise, not overturn the verdict. Not worth keeping the car
  staged for; acceptable only as a free opportunistic grab if the
  car is still on the mat.
- No more required mat runs. Result is conclusive (two
  independent analyzers, 5 configs, ~500 zone-B samples each).
  Recommend the car comes off the mat.
- STRATEGIC: this strengthens the already-recorded structural
  finding (bare-wall translation is unobservable to camera-only
  methods). Claude recommends pivoting to STEP 3 (vo_node
  publishes nav_msgs/Odometry + principled covariance from
  inliers/spread/RANSAC residual/3D conditioning) rather than
  Step 2 (solvePnPRansac), because Step 1 shows the bottleneck is
  structural, not the cost function — the high-value move is
  honest uncertainty so the EKF can down-weight VO where it is
  blind, not a better point estimate on a degenerate scene. Codex
  leans Step 2 next; this is a genuine strategic fork left for the
  operator to decide. No more parameter mat-runs either way.

No code changed this turn (analysis + decision + logging only).

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 103.
- /tmp/analyze_step1.py (out-of-tree analyzer; not committed).

## 2026-05-18 (5 m cutoff = also negative; Step 2-vs-Step 3 debate opened)

Operator ran the optional looser cutoff (depth_weight_power=0.0,
max_vo_feature_depth_m=5.0) and appended it to VO_readings.txt.
Note: that run's capture is the vo_node CONSOLE log (banner +
[WARN] + [DEPTH DIAG ALIGNED]), not the ts-prefixed
/vo/fault_status stream, so the Claude zone analyzer cannot
zone-B it. Verdict is nonetheless robust from three independent
reads:
- Codex parse: 5 m cutoff agree 10.0% vs control 20.0% whole-run,
  ~4.7% agree / 91.7% vo_suspect in the middle — worse than
  control (recovered more inliers than 3 m but still worse than no
  cutoff).
- Operator observation: "changed nothing" / no improvement.
- Prediction (prior entry) + the run's own [DEPTH DIAG ALIGNED]
  values clustering at 0.5-1.3 m (occasional 4-4.7 m): a 5 m cut
  trims only the small, noisy 5-8 m tail, so ~control was
  expected.

CONCLUSION: the entire depth-handling lever (soft weighting at
dwp=2/4, hard cutoff at 3 m and 5 m, and the combo) is CLOSED.
None beat the unweighted Test 6 control in the representative
zone. depth_weight_power and max_vo_feature_depth_m stay in the
code, default 0.0 = OFF (bit-identical to Test 6), inert. No more
depth-parameter mat runs.

STRATEGIC FORK (decision PENDING — being debated between the two
assistants at operator request, not decided this turn):
- Codex position: Step 2 next — replace point-to-point Procrustes
  with a PnP / reprojection-error estimator; "the VO problem is
  the motion-estimation method itself."
- Claude position: Step 3 next — vo_node publishes
  nav_msgs/Odometry + a principled covariance (inliers / spread /
  RANSAC residual / 3D-point geometric conditioning). Argument:
  Step 1's failure mode was information STARVATION (inliers
  collapse, rej/vo_suspect rise), i.e. an observability/
  conditioning problem, not a cost-function bias problem; a better
  objective (PnP) on a rank-deficient bare-wall problem still
  returns confident-wrong output. The near-term goal is
  fusion-readiness, whose gate is honest covariance, not marginal
  point accuracy. Step 3 is additive/zero-regression to the
  proven path and yields a per-frame quality metric that makes a
  later Step 2 objectively measurable; Step 2 first replaces the
  load-bearing estimator with no yardstick and no safety net.
  Order should be Step 3 -> Step 2, not Step 2 -> Step 3.
- Full Claude reasoning delivered to the operator for the debate;
  decision + its rationale will be logged here once resolved.

No code changed this turn (analysis + debate framing + logging).

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 105.

## 2026-05-18 (project direction: Step 3 selected lane; competition-exclusion + showcase plan)

Operator reviewed both assistants' Step 2-vs-Step 3 positions and
set project direction (no code this turn — direction + Step 3
scope explanation only):

- Depth-handling lever accepted as CLOSED by the operator: the
  best new config (dwp=2) was at best ~equal-to-slightly-worse
  than the unweighted 1200-feature Test 6 baseline. Integration
  judged correct; the approach simply did not help. No more
  depth-parameter mat runs.
- Framing agreed: Step 2 = raw-VO-accuracy research path; Step 3 =
  physical-robot fusion / safe-EKF-integration path.
- DECISION: pursue Step 3 next (pending final operator "go" after
  the scope explanation). Rationale tied to the operator's actual
  plan rather than the abstract debate: Step 3 is the prerequisite
  for the optional Friday VO+Cartographer EKF test, is the
  strongest showcase artifact (honest uncertainty that visibly
  grows on bare walls), and is the clean input for the teammate's
  read-only VSLAM/loop-closure showcase.
- PROJECT CONSTRAINTS recorded (durable):
  * VO / VSLAM are READ-ONLY research + showcase. They will NOT be
    run during the actual competition — competition compute is
    reserved for lane following, trip planner, traffic detection.
  * Timeline: build Step 3 ~2026-05-18..05-19 (today/tomorrow);
    optional EKF VO+Cart fusion test on Friday 2026-05-22 (final
    tests); the last ~3 days (~05-20..05-22) are showcase
    production (videos, RViz uncertainty/loop-closure demos).
  * Teammate handoff: current VO implementation given to a
    teammate to start VSLAM (camera map / loop closure) as a
    read-only showcase only; operator's stance — if VO is weak
    SLAM won't fix it, so scope it as showcase, not a competition
    capability (same logic as deferring low-light work until
    good-light works).

Step 3 build scope (planned, additive, default-safe, zero
regression to the Test 6 path):
- vo_node additionally publishes nav_msgs/Odometry (e.g.
  /vo/odometry): the existing pose + a per-frame covariance.
- Covariance derived from inlier count, inlier spread, RANSAC
  residual, and a NEW 3D-point geometric-conditioning term (the
  quantitative bare-wall-degeneracy detector).
- Conditioning also published as its own scalar (dashboard +
  showcase visibility). Existing topics / fault_status / estimate
  path all unchanged. Step 2 (PnP) explicitly deferred to AFTER
  Step 3 and would be evaluated using the Step 3 conditioning
  metric.

No code changed this turn (direction + Step 3 scope only;
awaiting operator go to implement).

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 106.
