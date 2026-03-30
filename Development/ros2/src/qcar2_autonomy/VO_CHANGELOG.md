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
