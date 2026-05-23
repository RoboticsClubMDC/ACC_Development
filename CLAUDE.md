# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is the Quanser ACC 2026 competition workspace for the **QCar2**. It is a ROS 2 Humble project that runs inside an Isaac ROS dev container, against either the **virtual QCar2** (QLabs) or **physical QCar2** hardware. `Easy_Start.md` is the authoritative copy-paste runbook — keep it in sync when adding launch files, executable names, or topic contracts.

## Where work happens

- All ROS 2 source lives in [Development/ros2/src/](Development/ros2/src/). Inside the Isaac dev container this path is mounted at `/workspaces/isaac_ros-dev/Development/ros2`. The container is the only supported build/run environment — do not assume host installs work.
- `colcon build` is run from `Development/ros2` (i.e. `/workspaces/isaac_ros-dev/Development/ros2` inside the container), NOT from the repo root.
- The directories `build/`, `install/`, `log/`, `Development/ros2/build`, `Development/ros2/install`, and `Development/ros2/log` are generated. Never hand-edit them.
- Permanent Python/apt deps for the dev container go in [docker/development_docker/quanser_dev_docker_files/Dockerfile.quanser](docker/development_docker/quanser_dev_docker_files/Dockerfile.quanser), not in ad-hoc `pip install` inside a running container.

## Starting a session

Order is non-trivial; follow `Easy_Start.md` § "Normal Startup Order":

1. Launch QLabs / virtual QCar2 from `docker/virtual_qcar2` (host).
2. Launch the Isaac ROS dev container via `./isaac_ros_common/scripts/run_dev.sh /home/$USER/Documents/GitHub/ACC_Development/Development`.
3. In **every** new ROS terminal inside the container:
   ```bash
   cd /workspaces/isaac_ros-dev/Development/ros2
   source /opt/ros/humble/setup.bash
   source /workspace/cartographer_ws/install/setup.bash
   source install/setup.bash
   export ROS_DOMAIN_ID=69
   ```
   The Cartographer overlay is required because `pcl_conversions` / `pcl_ros` come from there, not from apt.

## Build commands

Full workspace:
```bash
colcon build --symlink-install
```

Single package (prefer this — full builds are slow and pull in the RTAB-Map source tree):
```bash
colcon build --symlink-install --packages-select qcar2_autonomy
colcon build --symlink-install --packages-select qcar2_perception
```

To temporarily skip the heavy local RTAB-Map source build:
```bash
touch src/rtabmap/COLCON_IGNORE src/rtabmap_ros/COLCON_IGNORE
```

To enable and build RTAB-Map headless (this branch only — `rtabmap_ros` 0.22.1, needs Humble header patches already applied in-tree): see `Easy_Start.md` § "RTAB-Map Source Build" for the exact `--cmake-args` flags. Use `--parallel-workers 2` to avoid OOM. RTAB **must** be built with `BUILD_APP=OFF BUILD_TOOLS=OFF BUILD_EXAMPLES=OFF WITH_QT=OFF`.

Always re-source `install/setup.bash` and re-export `ROS_DOMAIN_ID=69` after a build.

## Running components

Launch files live in `qcar2_nodes/launch/` (base/hardware/SLAM/Nav2) and `qcar2_perception/launch/` (perception + RTAB mapping). Common entry points:

- Base sensors + hardware (virtual): `ros2 launch qcar2_nodes qcar2_virtual_launch.py`
- Cartographer (virtual): `ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py`
- Foxglove bridge: `ros2 launch qcar2_nodes foxglove_bridge_launch.py` → `ws://localhost:8765`
- Perception core: `ros2 launch qcar2_perception perception_core_virtual.launch.py` (or `_physical`)
- RTAB-Map mapping stack: `ros2 launch qcar2_perception qcar2_rtabmap_mapping_virtual.launch.py` (or `_physical`)
- Autonomy planner: `ros2 launch qcar2_autonomy autonomy_planner_launch.py`
- Manual drive: `ros2 run qcar2_autonomy manual_drive` — auto-spawns `qcar2_nodes nav2_qcar2_converter` unless one is already running; disable with `-p auto_start_converter:=false`.

**Mutual exclusions** (will silently corrupt data if violated):
- Do not run `qcar2_nodes rgbd` together with `qcar2_perception d435_aligned_source` — the D435 aligned source owns the camera path in the perception stack.
- Do not run the old `qcar2_autonomy yolo_detector` together with the new `qcar2_perception semantic_yolo_detector`.
- The RTAB-Map mapping launches intentionally do **not** start Cartographer, `qcar2_nodes rgbd`, YOLO, semantic mapper, or Nav2. Don't add them — RTAB owns the map frame during a mapping run.

## Tests

ROS 2 ament Python tests:
```bash
colcon test --packages-select qcar2_autonomy
colcon test-result --verbose
```
There is no JS/TS toolchain in this repo.

## High-level architecture

The runtime decomposes into four layers. They communicate only through ROS topics/TF, and each layer has a **single authority** that must not be bypassed:

1. **Mapping** — builds the world model. Today: Cartographer for live 2D SLAM; the new path is RTAB-Map RGB-D + 2D LiDAR for offline map construction.
2. **Localization** — estimates the robot inside the frozen map. Final target is AMCL against a `golden_map.yaml`/`.pgm` exported from RTAB and aligned to competition coordinates via a BEV PNG.
3. **Perception / semantics** — `qcar2_perception` runs the D435 aligned source, YOLO detector, 3D object estimator, semantic landmark mapper, and consistency monitor. Semantics **audit** world consistency; they do not move pose.
4. **Control** — path follower + lane detector + reward grid + semantic watchdog all feed a **motion arbiter** which is the only node that publishes the final motor command. Lane detector does not directly steer; reward grid does not directly drive. Preserve this invariant when adding new nodes.

End-to-end direction (per `Easy_Start.md` § "Architecture Direction"):
```
RTAB-Map (offline map) → BEV alignment → golden_map → AMCL (live)
  → path follower → reward grid + lane safety + semantic watchdog → motion arbiter → motors
```

### Frame contract

Two regimes — do not mix them:

- **Cartographer regime**: `map → odom → base_link`, plus `base_link → base_scan` and `base_link → aligned_camera_optical_frame`.
- **RTAB mapping regime**: `rtab_map → rtab_odom → base_link`, plus the same sensor TFs.
  - `rtab_map` is a **root** frame during RTAB mapping; do not give it a static parent in `semantic_tf_launch.py`.
  - RTAB SLAM publishes `rtab_map → rtab_odom`; RTAB odometry publishes `rtab_odom → base_link`. Both must run with `frame_id:=base_link` — using `aligned_camera_optical_frame` will collide with the static camera TF.
  - D435 messages stamped in `aligned_camera_optical_frame` are fine; RTAB resolves them through the static TF.

### Semantic marker topics (must stay separated)

The perception stack publishes four distinct marker topics on purpose — collapsing them into one Foxglove layer destroys the debugging signal:

- `/perception/semantic_landmark_markers` — **stable** map landmarks only (the persistent semantic overlay).
- `/perception/semantic_hypothesis_markers` — candidate + confirmed landmarks (working memory).
- `/perception/semantic_current_markers` — landmarks currently visible to the D435.
- `/perception/semantic_residual_markers` — consistency-monitor residual lines, NOT landmark memory.

`semantic_map.json` is a session overlay; `reset_map_on_start: true` clears it each launch. If `semantic_map.json` is mysteriously refilling, a stale mapper is still running.

### YOLO model resolution

`semantic_yolo_detector` searches, in order: the installed `qcar2_autonomy/share/qcar2_autonomy/models`, then `Development/ros2/src/qcar2_autonomy/models`, then a legacy path. When adding/replacing model files, drop them into `qcar2_autonomy/models/` and rebuild — `setup.py` glob-installs `models/*` into the share dir.

## Packages at a glance

- [qcar2_nodes](Development/ros2/src/qcar2_nodes/) — C++ hardware/sensor nodes (`lidar`, `qcar2_hardware`, `rgbd`, `qcar2_odometry`, `nav2_qcar_command_convert`, `csi`) and the SLAM/Nav2 bringup launches.
- [qcar2_autonomy](Development/ros2/src/qcar2_autonomy/) — Python autonomy: path follower (`nav_to_pose`), Stanley lane controllers, manual/teleop drive, EKF pose estimator, visual odometry, trip planner, roadmap alignment. Executable-to-entry-point mapping is in [setup.py](Development/ros2/src/qcar2_autonomy/setup.py).
- [qcar2_perception](Development/ros2/src/qcar2_perception/) — Python perception: D435 aligned source, semantic YOLO detector, 3D object estimator, semantic landmark mapper, semantic consistency monitor, plus the RTAB-Map mapping launches.
- [qcar2_interfaces](Development/ros2/src/qcar2_interfaces/) — custom msg definitions.
- [rtabmap/](Development/ros2/src/rtabmap/) and [rtabmap_ros/](Development/ros2/src/rtabmap_ros/) — vendored RTAB-Map source (0.22.1) with Humble header-name patches. Treat as third-party; only patch when the upstream version is bumped.

## Topic contracts worth knowing

- `manual_drive` publishes `/cmd_vel_nav`. The QCar hardware consumes `qcar2_motor_speed_cmd`, so `nav2_qcar2_converter` must be running between them (manual_drive auto-spawns it).
- Perception aligned-camera topics: `/perception/d435/{rgb/image_raw, depth/image_rect, camera_info}`.
- `rgbd_sync` only publishes its heavy RGB-D message when something subscribes — `ros2 topic hz /rtabmap/rgbd_image` doubles as a check and a keep-alive subscriber.
- RTAB internal parameters declared as **strings** by this wrapper (e.g. `RGBD/CreateOccupancyGrid:="true"`, `Rtabmap/DetectionRate:="1.0"`) must keep their nested quotes, otherwise ROS 2 parses them as bool/float and throws `InvalidParameterTypeException`.

## Debugging entry points

- Build logs: `Development/ros2/log/latest_build/*/stdout_stderr.log`.
- Runtime logs: `~/.ros/log/latest/*.log`.
- TF sanity: `ros2 run tf2_ros tf2_echo <parent> <child>` for the frame-contract edges above.
- See `Easy_Start.md` § "Logs, Bags, And Debug Checks" for the full set of `ros2 topic hz` / `bag record` recipes, including the standard perception and RTAB smoke-test bag manifests.
