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
