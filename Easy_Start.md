# QCar2 ACC Easy Start

Updated: 2026-05-22 19:00:55 EDT

This is the copy-paste runbook for the ACC QCar2 workspace. It is split into
startup flow, builds, perception, RTAB-Map, logs/debugging, and architecture
notes so the commands stay easy to scan while testing.

## Table Of Contents

1. [Normal Startup Order](#normal-startup-order)
2. [Every ROS Container Terminal](#every-ros-container-terminal)
3. [Start QLabs / Virtual QCar2](#start-qlabs--virtual-qcar2)
4. [Start Isaac ROS Dev Container](#start-isaac-ros-dev-container)
5. [Build Commands](#build-commands)
6. [Base QCar2 Nodes](#base-qcar2-nodes)
7. [Autonomy Commands](#autonomy-commands)
8. [Perception Layer](#perception-layer)
9. [RTAB-Map Source Build](#rtab-map-source-build)
10. [RTAB-Map RGB-D + LiDAR Mapping Launch](#rtab-map-rgb-d--lidar-mapping-launch)
11. [Logs, Bags, And Debug Checks](#logs-bags-and-debug-checks)
12. [Killing Stale ROS Nodes Between Runs](#killing-stale-ros-nodes-between-runs)
13. [Odometry Architecture (EKF owns odom)](#odometry-architecture-ekf-owns-odom)
14. [Architecture Direction](#architecture-direction)
15. [Change Log](#change-log)

## Normal Startup Order

Use this order for a normal virtual QCar2 ACC run:

1. Start QLabs / virtual QCar2 on the host.
2. Start the Isaac ROS dev container.
3. In each ROS terminal, source the workspace and set `ROS_DOMAIN_ID=69`.
4. Build only the package you changed, or run the normal workspace build.
5. Start base QCar2 nodes or Cartographer.
6. Start Foxglove bridge if you want browser visualization.
7. Start perception or autonomy, depending on the test.
8. Watch the log/debug topics before trusting behavior.

## Every ROS Container Terminal

Run this inside every new ROS container terminal:

```bash
cd /workspaces/isaac_ros-dev/Development/ros2
source /opt/ros/humble/setup.bash
source /workspace/cartographer_ws/install/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=69
```

Quick check:

```bash
ros2 pkg list | grep qcar2
```

Optional terminal tab/window title:

```bash
echo -ne "\033]0;QCar2 Cartographer\007"
```

Reusable helper for `~/.bashrc`:

```bash
termname() {
  echo -ne "\033]0;$1\007"
}
```

After reloading `~/.bashrc`, name terminals like this:

```bash
source ~/.bashrc
termname "QCar2 AMCL"
termname "QCar2 Perception"
termname "Foxglove Bridge"
```

## Start QLabs / Virtual QCar2

Host terminal:

```bash
cd /home/$USER/Documents/GitHub/ACC_Development/docker/virtual_qcar2
sudo docker run --rm -it --network host --name virtual-qcar2 quanser/virtual-qcar2 bash
```

Inside the QLabs container:

```bash
cd /home/qcar2_scripts/python
python3 Base_Scenarios_Python/Setup_Competition_Map.py
clear
```

## Start Isaac ROS Dev Container

Host terminal:

```bash
cd /home/$USER/Documents/GitHub/ACC_Development
./isaac_ros_common/scripts/run_dev.sh /home/$USER/Documents/GitHub/ACC_Development/Development
```

Expected Docker config file:

```bash
/home/$USER/Documents/GitHub/ACC_Development/isaac_ros_common/scripts/.isaac_ros_common-config
```

Expected config:

```bash
CONFIG_IMAGE_KEY="ros2_humble.ros_cartographer.user.quanser"
CONFIG_DOCKER_SEARCH_DIRS=(/home/$USER/Documents/GitHub/ACC_Development/docker/development_docker/quanser_dev_docker_files)
```

Permanent Python/Debian packages go here:

```bash
/home/$USER/Documents/GitHub/ACC_Development/docker/development_docker/quanser_dev_docker_files/Dockerfile.quanser
```

Verify Python packages inside the container:

```bash
python3 -c "import ultralytics, tqdm; print('ok')"
pip3 show ultralytics tqdm
```

## Build Commands

Use this after ordinary QCar2 source changes:

```bash
cd /workspaces/isaac_ros-dev/Development/ros2
source /opt/ros/humble/setup.bash
source /workspace/cartographer_ws/install/setup.bash
colcon build --symlink-install
source install/setup.bash
export ROS_DOMAIN_ID=69
```

Fast package rebuild examples:

```bash
colcon build --symlink-install --packages-select qcar2_autonomy
source install/setup.bash
export ROS_DOMAIN_ID=69
```

```bash
colcon build --symlink-install --packages-select qcar2_perception
source install/setup.bash
export ROS_DOMAIN_ID=69
```

Only do a clean build when the workspace/install state is actually broken:

```bash
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
export ROS_DOMAIN_ID=69
```

## Base QCar2 Nodes

Virtual sensor/hardware launch:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 launch qcar2_nodes qcar2_virtual_launch.py
```

Cartographer virtual launch:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py
```

Foxglove bridge:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 launch qcar2_nodes foxglove_bridge_launch.py
```

Foxglove Studio on the host:

```text
ws://localhost:8765
```

## Autonomy Commands

Autonomy planner:

```bash
colcon build --symlink-install --packages-select qcar2_autonomy
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 launch qcar2_autonomy autonomy_planner_launch.py
```

Path follower:

```bash
colcon build --symlink-install --packages-select qcar2_autonomy
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 run qcar2_autonomy path_follower
```

Old autonomy YOLO prototype:

```bash
colcon build --symlink-install --packages-select qcar2_autonomy
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 run qcar2_autonomy yolo_detector
```

Manual drive:

```bash
colcon build --symlink-install --packages-select qcar2_autonomy
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 run qcar2_autonomy manual_drive
```

Note:

- Added 2026-05-22 22:18:54 EDT: `manual_drive` publishes `/cmd_vel_nav`.
- If no QCar command converter is running, `manual_drive` starts
  `qcar2_nodes nav2_qcar2_converter`.
- If a launch file already started the converter, `manual_drive` reuses it.
- Disable converter auto-start if needed:

```bash
ros2 run qcar2_autonomy manual_drive --ros-args -p auto_start_converter:=false
```

## Perception Layer

Added: 2026-05-20 09:36:26 EDT

Purpose:

- Starts the `qcar2_perception` layer.
- Does not touch the old working `qcar2_autonomy` YOLO prototype.
- Publishes aligned D435 RGB/depth/camera_info.
- Publishes YOLO detections, 3D object estimates, semantic landmarks, and monitor output.

Important:

- Do not run `qcar2_nodes rgbd` at the same time as `d435_aligned_source`.
- Do not run the old `qcar2_autonomy yolo_detector` at the same time as the new perception stack.
- The D435 aligned source owns the camera path for this perception stack.

Build:

```bash
cd /workspaces/isaac_ros-dev/Development/ros2
colcon build --symlink-install --packages-select qcar2_perception
source install/setup.bash
export ROS_DOMAIN_ID=69
```

Virtual QLabs perception:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 launch qcar2_perception perception_core_virtual.launch.py
```

Physical QCar2 perception:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 launch qcar2_perception perception_core_physical.launch.py
```

Manual start, terminal 1:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 launch qcar2_perception semantic_tf_launch.py
```

Manual start, terminal 2 for virtual QLabs:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 run qcar2_perception d435_aligned_source --ros-args -p is_physical:=false -p distance_scale:=0.1
```

Manual start, terminal 2 for physical QCar2:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 run qcar2_perception d435_aligned_source --ros-args -p is_physical:=true -p distance_scale:=1.0
```

Manual start, terminal 3:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 run qcar2_perception semantic_yolo_detector
```

Cartographer plus perception:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py
```

```bash
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 launch qcar2_perception perception_core_virtual.launch.py
```

Perception meaning notes:

- Cartographer remains the geometry/localization authority.
- `semantic_map.json` is a session semantic overlay by default.
- `reset_map_on_start: true` means the semantic map starts empty each core launch.
- Stable landmarks are saved; candidates and confirmed landmarks are live memory/debug hypotheses.
- YOLO dead-zone mask draws a red polygon with `NA` on `/perception/yolo/image_annotated`.
- Stable map, live hypotheses, current observations, and residual checks are separate Foxglove layers.

Core Foxglove topics:

```text
/perception/d435/rgb/image_raw
/perception/d435/depth/image_rect
/perception/d435/camera_info
/perception/yolo/image_annotated
/perception/yolo/detections_2d
/perception/objects_3d
/perception/object_markers
/perception/semantic_landmarks
/perception/semantic_landmark_markers
/perception/semantic_hypothesis_markers
/perception/semantic_current_markers
/perception/semantic_residual_markers
/perception/semantic_localization_residual
/perception/health
```

Semantic marker topic meanings:

| Topic | Meaning |
| --- | --- |
| `/perception/semantic_landmark_markers` | Stable landmarks only. This is the semantic map overlay for the current run. |
| `/perception/semantic_hypothesis_markers` | Candidate and confirmed landmarks. This is working memory/debug hypotheses. |
| `/perception/semantic_current_markers` | Landmarks currently visible to the D435 semantic pipeline. |
| `/perception/semantic_residual_markers` | Consistency monitor residual lines only. These are not landmark memory. |

Perception quick checks:

```bash
ros2 topic hz /perception/d435/rgb/image_raw
ros2 topic hz /perception/d435/depth/image_rect
ros2 topic echo /perception/d435/camera_info --once
ros2 topic hz /perception/yolo/image_annotated
ros2 topic echo /perception/yolo/detections_2d
ros2 run tf2_ros tf2_echo base_link aligned_camera_optical_frame
```

## RTAB-Map Source Build

Updated: 2026-05-22 18:52:41 EDT

Why source build:

- This Isaac/Quanser apt setup can fail to resolve `ros-humble-rtabmap-ros`.
- Some ROS dependency packages can also be missing from apt here.
- `pcl_conversions` and `pcl_ros` may come from `/workspace/cartographer_ws/install`.
- `octomap_msgs` can be missing, but RTAB can still build enough for RGB-D smoke tests.

Dockerfile note:

- `Dockerfile.quanser` installs RTAB-Map source-build dependencies.
- TensorRT is commented out during this phase because huge NVIDIA wheels can stop the image before RTAB dependencies install.
- `ros-humble-message-filters` must be installed for `rtabmap_sync`.

RTAB source compatibility notes:

- Use matching RTAB core and ROS wrapper versions. For this workspace, `rtabmap_ros` is `0.22.1`, so RTAB core should match `0.22.1-humble`.
- This container needed Humble header-name compatibility patches:
  - `message_filters/*.hpp` includes become `*.h`
  - `tf2/utils.hpp` becomes `tf2/utils.h`
- RTAB must be built headless here:
  - `BUILD_APP=OFF`
  - `BUILD_TOOLS=OFF`
  - `BUILD_EXAMPLES=OFF`
  - `WITH_QT=OFF`

Check dependency visibility:

```bash
ros2 pkg prefix octomap_msgs
ros2 pkg prefix pcl_conversions
ros2 pkg prefix pcl_ros
ros2 pkg prefix message_filters
```

Ignore RTAB source while building only normal QCar2 packages:

```bash
cd /workspaces/isaac_ros-dev/Development/ros2
touch src/rtabmap/COLCON_IGNORE
touch src/rtabmap_ros/COLCON_IGNORE
```

Enable and build RTAB-Map smoke-test packages:

```bash
cd /workspaces/isaac_ros-dev/Development/ros2
rm -f src/rtabmap/COLCON_IGNORE
rm -f src/rtabmap_ros/COLCON_IGNORE
source /opt/ros/humble/setup.bash
source /workspace/cartographer_ws/install/setup.bash
colcon build --symlink-install \
  --parallel-workers 2 \
  --packages-up-to rtabmap_slam rtabmap_odom rtabmap_sync \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_APP=OFF \
    -DBUILD_TOOLS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DWITH_QT=OFF \
    -DWITH_G2O=OFF \
    -DWITH_GTSAM=OFF \
    -DWITH_POINTMATCHER=OFF
source install/setup.bash
export ROS_DOMAIN_ID=69
```

Verify RTAB executables:

```bash
ros2 pkg executables rtabmap_sync
ros2 pkg executables rtabmap_odom
ros2 pkg executables rtabmap_slam
```

Important executable names in this source branch:

```text
rtabmap_sync: rgbd_sync, rgbdx_sync, stereo_sync, rgb_sync
rtabmap_odom: rgbd_odometry, stereo_odometry, icp_odometry
rtabmap_slam: rtabmap
```

Do not use these names with `ros2 run` in this branch:

```text
rtabmap_rgbd_sync
rtabmap_rgbd_odometry
rtabmap_node
```

## RTAB-Map RGB-D + LiDAR Mapping Launch

Goal:

1. Prove RTAB can receive aligned D435 RGB-D.
2. Prove RTAB RGB-D odometry can publish `/rtabmap/odom`.
3. Keep RTAB connected to the normal QCar robot frames.
4. Prove RTAB SLAM can consume RGB-D, odom, and 2D LiDAR `/scan`.
5. Record a small bag for offline mapping.

Frame contract:

```text
rtab_map -> rtab_odom -> base_link
base_link -> base_scan
base_link -> aligned_camera_optical_frame
```

Important:

- `rtab_map` is a root frame during RTAB mapping. It does not get a static
  parent in `semantic_tf_launch.py`.
- RTAB SLAM publishes `rtab_map -> rtab_odom`.
- RTAB odometry publishes `rtab_odom -> base_link`.
- The static TF launch files publish only physical robot sensor transforms.
- Do not use `aligned_camera_optical_frame` as RTAB odometry `frame_id` for the
  mapping run. That makes RTAB publish `rtab_odom -> aligned_camera_optical_frame`
  and conflicts with our static `base_link -> aligned_camera_optical_frame`.
- RTAB odometry and SLAM should use `frame_id:=base_link`.
- The D435 image messages can still be stamped in `aligned_camera_optical_frame`;
  RTAB can transform them through the static camera TF.
- The 2D LiDAR scan is in `base_scan`, so `base_link -> base_scan` must be alive.

Build the launch package after edits:

```bash
colcon build --symlink-install --packages-select qcar2_perception qcar2_autonomy
source install/setup.bash
export ROS_DOMAIN_ID=69
```

Start full virtual RTAB mapping stack:

```bash
ros2 launch qcar2_perception qcar2_rtabmap_mapping_virtual.launch.py
```

Start full physical RTAB mapping stack:

```bash
ros2 launch qcar2_perception qcar2_rtabmap_mapping_physical.launch.py
```

What this launch starts:

```text
qcar2_nodes lidar
qcar2_nodes qcar2_hardware
base_link -> base_scan TF
base_link -> aligned_camera_optical_frame TF
qcar2_perception d435_aligned_source
rtabmap_sync rgbd_sync
rtabmap_odom rgbd_odometry
rtabmap_slam rtabmap
```

What this launch intentionally does not start:

```text
Cartographer
qcar2_nodes rgbd
YOLO / semantic mapper
Nav2
```

Verify the robot frame tree and data flow:

```bash
ros2 run tf2_ros tf2_echo base_link aligned_camera_optical_frame
ros2 run tf2_ros tf2_echo base_link base_scan
ros2 run tf2_ros tf2_echo rtab_odom base_link
ros2 run tf2_ros tf2_echo rtab_map rtab_odom
ros2 topic list | grep d435
ros2 topic hz /perception/d435/rgb/image_raw
ros2 topic hz /perception/d435/depth/image_rect
ros2 topic echo /perception/d435/camera_info --once
ros2 topic hz /scan
ros2 topic hz /rtabmap/rgbd_image
ros2 topic hz /rtabmap/odom
ros2 topic echo /rtabmap/odom --once
ros2 topic hz /rtabmap/info
```

Note:

- RTAB internal parameters like `RGBD/CreateOccupancyGrid` and
  `Rtabmap/DetectionRate` are declared as strings by this ROS wrapper. Keep the
  nested quotes or ROS 2 will parse `true` as a bool and abort with
  `InvalidParameterTypeException`.
- `rgbd_sync` only publishes the heavy RGB-D message when something subscribes.
- Running `ros2 topic hz /rtabmap/rgbd_image` is both a check and a subscriber.

Check RTAB SLAM:

```bash
ros2 topic list | grep rtabmap
ros2 topic echo /rtabmap/info --once
ros2 run tf2_ros tf2_echo rtab_map rtab_odom
```

Record first smoke-test bag:

```bash
mkdir -p ~/qcar2_rtab_bags
ros2 bag record -o ~/qcar2_rtab_bags/rtab_lidar_rgbd_01 \
  /scan \
  /tf \
  /tf_static \
  /perception/d435/rgb/image_raw \
  /perception/d435/depth/image_rect \
  /perception/d435/camera_info \
  /rtabmap/rgbd_image \
  /rtabmap/odom \
  /rtabmap/odom_info \
  /rtabmap/info \
  /rtabmap/mapData \
  /rtabmap/mapGraph
```

## Logs, Bags, And Debug Checks

This section is only for logs and visibility. Use it after launches are running.

Build logs:

```bash
cd /workspaces/isaac_ros-dev/Development/ros2
ls -lt log | head
find log/latest_build -maxdepth 3 -type f | sort
tail -n 80 log/latest_build/*/stdout_stderr.log
```

ROS run logs:

```bash
ls -lt ~/.ros/log | head
find ~/.ros/log/latest -maxdepth 2 -type f | sort
tail -n 80 ~/.ros/log/latest/*.log
```

Save a launch terminal log:

```bash
mkdir -p ~/qcar2_logs
ros2 launch qcar2_perception perception_core_virtual.launch.py 2>&1 | tee ~/qcar2_logs/perception_core_virtual.log
```

Common graph checks:

```bash
ros2 node list
ros2 topic list
ros2 topic list | grep perception
ros2 topic list | grep rtabmap
ros2 run tf2_ros tf2_echo map base_link
ros2 run tf2_ros tf2_echo base_link aligned_camera_optical_frame
```

Perception topic checks:

```bash
ros2 topic hz /perception/d435/rgb/image_raw
ros2 topic hz /perception/d435/depth/image_rect
ros2 topic echo /perception/d435/camera_info --once
ros2 topic hz /perception/yolo/image_annotated
ros2 topic echo /perception/yolo/detections_2d
ros2 topic echo /perception/health --once
ros2 topic echo /perception/semantic_localization_residual --once
```

Semantic marker layers to enable separately in Foxglove/RViz:

```text
/perception/semantic_landmark_markers
/perception/semantic_hypothesis_markers
/perception/semantic_current_markers
/perception/semantic_residual_markers
```

Bag commands:

```bash
mkdir -p ~/qcar2_bags
ros2 bag record -o ~/qcar2_bags/perception_debug_01 \
  /tf \
  /tf_static \
  /scan \
  /perception/d435/rgb/image_raw \
  /perception/d435/depth/image_rect \
  /perception/d435/camera_info \
  /perception/yolo/detections_2d \
  /perception/objects_3d \
  /perception/semantic_landmarks \
  /perception/health
```

```bash
ros2 bag info ~/qcar2_bags/perception_debug_01
ros2 bag play ~/qcar2_bags/perception_debug_01
```

Debug meanings:

- If `semantic_map.json` clears and then fills again, an old mapper is probably still running and rewriting it.
- If stable landmarks look mixed with candidates, check the separate semantic marker topics instead of one combined layer.
- If YOLO detects the QCar body/dead image region, check that `/perception/yolo/image_annotated` shows the red `NA` polygon.
- If map-frame semantic landmarks warn about transforms, start Cartographer before the perception core launch.

## Killing Stale ROS Nodes Between Runs

Ctrl-C on a `ros2 launch` does not always reap child processes. Leftover nodes
become "ghosts" that duplicate publishers (e.g. two `/foxglove_bridge`,
two `odom -> base_link` TFs) and silently corrupt the next run. Use this
between sessions.

One-shot in the dev container (or any terminal that can see the processes):

```bash
pkill -INT  -f "ros2 launch" 2>/dev/null
pkill -INT  -f "ros2 run"    2>/dev/null
sleep 2
pkill -TERM -f "qcar2|nav2_qcar2|foxglove_bridge|fixed_lidar|pose_estimator|cartographer|amcl|lifecycle_manager|map_server|nav2_map_server" 2>/dev/null
sleep 1
pkill -KILL -f "qcar2|nav2_qcar2|foxglove_bridge|fixed_lidar|pose_estimator|cartographer|amcl|lifecycle_manager|map_server|nav2_map_server" 2>/dev/null
ros2 daemon stop 2>/dev/null
ros2 daemon start
```

Why three signal stages: `SIGINT` (same as Ctrl-C) gives launchers a chance to
unwind cleanly. `SIGTERM` is graceful for plain executables. `SIGKILL` is the
last resort. The `ros2 daemon stop/start` resets the discovery cache so
`ros2 node list` does not report stale entries.

Convenience alias for `~/.bashrc`:

```bash
ros2_killall() {
  pkill -INT  -f "ros2 launch" 2>/dev/null
  pkill -INT  -f "ros2 run"    2>/dev/null
  sleep 2
  pkill -TERM -f "qcar2|nav2_qcar2|foxglove_bridge|fixed_lidar|pose_estimator|cartographer|amcl|lifecycle_manager|map_server|nav2_map_server" 2>/dev/null
  sleep 1
  pkill -KILL -f "qcar2|nav2_qcar2|foxglove_bridge|fixed_lidar|pose_estimator|cartographer|amcl|lifecycle_manager|map_server|nav2_map_server" 2>/dev/null
  ros2 daemon stop 2>/dev/null
  ros2 daemon start
  echo "ROS 2 processes killed. Remaining:"
  ps -ef | grep -E "qcar2|ros2 (launch|run)|foxglove_bridge|cartographer|amcl|lifecycle_manager|map_server" | grep -v grep || echo "  (none)"
}
```

After sourcing `~/.bashrc`, run `ros2_killall` between launches.

The QLabs container is separate. Because it runs `--network host` with
`ROS_DOMAIN_ID=69`, nodes inside it appear as peers but are not killed by host
`pkill`. Clean it with:

```bash
sudo docker exec virtual-qcar2 pkill -f "csi_camera|foxglove_bridge|ros2"
```

Or restart the QLabs container fully:

```bash
sudo docker rm -f virtual-qcar2
# then re-run the QLabs startup from "Start QLabs / Virtual QCar2"
```

Verify everything is clean:

```bash
ros2 node list                  # should be empty (or only your active launches)
ros2 topic list | wc -l         # should be ~10 framework topics if nothing is running
ps -ef | grep -E "qcar2|ros2|foxglove_bridge|cartographer|amcl" | grep -v grep
```

## Odometry Architecture (EKF owns odom)

Updated: 2026-05-23 EDT

Single source of truth for `odom -> base_link` and `/odom`:

```text
/qcar2_joint  ──┐
/qcar2_imu    ──┼─►  qcar2_autonomy pose_estimator (EKF)  ──►  /odom + odom->base_link TF
steering      ──┘                                                │
                                                                 ▼
                                          Cartographer (use_odometry = true) ──► map->odom
                                                                                 │
                                                                                 ▼
                                                              AMCL ──► map->odom (replaces Cartographer's after lap)
```

Rules:

- `pose_estimator` (`qcar2_autonomy/autonomy/pose_estimator.py`) is the **only** node that publishes `odom -> base_link` and `/odom`.
- The C++ node `qcar2_odometry` is **retired**. Do not launch it. It is no longer referenced in `qcar2_virtual_launch.py`, `qcar2_amcl_localization_launch.py`, or `qcar2_amcl_localization_virtual_launch.py`. The source file in `qcar2_nodes/src/qcar2_odometry.cpp` remains for reference only.
- Cartographer reads `/odom` (`use_odometry = true` in `qcar2_2d.lua`) as its motion prior.
- AMCL reads `odom -> base_link` from TF — same EKF source, no changes to AMCL.
- The EKF fuses wheel speed (encoders via `/qcar2_joint`) with IMU yaw rate (`/qcar2_imu`) using `gyro_weight` (default 0.65). Steering from `/cmd_vel_nav` / `/qcar2_motor_speed_cmd` feeds the bicycle-model term.
- The EKF has **no** `map -> base_link` correction. Earlier versions had one, which made the EKF a downstream observer of SLAM and prevented it from helping Cartographer. That has been removed; pure dead-reckoning fusion now.

Cartographer config for external odometry (`qcar2_nodes/config/qcar2_2d.lua`):

```lua
tracking_frame     = "base_scan"   -- sensor frame
published_frame    = "odom"        -- Cartographer publishes map -> odom
odom_frame         = "odom"
provide_odom_frame = false         -- do NOT have Cartographer manage odom
use_odometry       = true          -- subscribe to EKF /odom
```

Common pitfall: setting `published_frame = "base_link"` with `provide_odom_frame = false` makes Cartographer publish `map -> base_link` directly, which collides with the EKF's `odom -> base_link` (two parents for the same frame). Symptom is `/map` never appearing. Keep `published_frame = "odom"`.

If you ever need to disable TF broadcast (e.g. running two odom sources for comparison):

```bash
ros2 run qcar2_autonomy pose_estimator --ros-args -p publish_tf:=false
```

## Architecture Direction

Final target:

```text
RTAB-Map offline builds/optimizes the map
  -> BEV PNG aligns that map to ideal competition coordinates
  -> AMCL localizes live against the frozen wall map
  -> roadmap/path follower drives in that fixed frame
  -> reward grid + lane safety + semantic watchdog audit the run
  -> motion arbiter sends the only final motor command
```

Rules:

- Mapping creates the world model.
- Localization estimates robot pose inside that world.
- Semantics audits world consistency but does not directly move pose.
- Lane detector does not directly command steering.
- Reward grid does not directly drive the motor.
- Motion arbiter is the only final command authority.

Immediate execution order:

1. RTAB-Map RGB-D smoke test.
2. Record small bag with `/scan`, D435 RGB-D, TF, camera info, odom if available.
3. Replay bag into RTAB offline.
4. Export/inspect occupancy map and graph.
5. Attach YOLO semantic observations to RTAB keyframes.
6. Align exported map with BEV PNG.
7. Generate `golden_map.yaml` and `golden_map.pgm`.
8. Launch AMCL on frozen map.
9. Verify scan overlay and covariance.
10. Add roadmap/path follower.
11. Add reward grid.
12. Add semantic watchdog events.
13. Add motion arbiter.

## Change Log

### 2026-05-23 EDT — discovery: Cartographer precision over AMCL

Observed during testing after the EKF + max_range fixes: Cartographer's live SLAM is **noticeably more accurate** than AMCL on the saved map from the same run. AMCL holds the geometry only because we tuned the scan-match weights extremely tight (`laser_likelihood_max_dist=0.30`, `max_beams=180`, `sigma_hit=0.20`); it does not converge to the same submillimeter alignment Cartographer achieves live.

**Why this is structural, not a tuning gap:**

- Cartographer is **graph-based SLAM** (Hess et al. 2016, "Real-Time Loop Closure in 2D LIDAR SLAM"). Local SLAM does correlative scan matching against probabilistic occupancy submaps; global SLAM optimizes a pose graph via Sparse Pose Adjustment (Konolige et al. 2010, Ceres-Solver under the hood). Every loop closure adds a new edge constraint and the optimizer **redistributes error retroactively across the entire trajectory**. The map literally gets better every overlapping lap. Drift is not "corrected" — it's eliminated.
- AMCL is **Adaptive Monte Carlo Localization** (Dellaert/Fox/Burgard/Thrun 1999), a particle filter over a frozen map. It can only update the current pose belief; there is no retroactive graph optimization. It is fundamentally a filter, not a smoother.
- AMCL is therefore capped by the quality of the saved PGM. Live Cartographer outperforms it because Cartographer is continuously refining both pose AND map. Once the map is frozen, AMCL inherits whatever quality existed at snapshot time and cannot improve on it.

**What the EKF contributes to Cartographer's accuracy gain:**

- The EKF (encoders + IMU yaw rate via bicycle model) provides `nav_msgs/Odometry` as a **motion prior** to Cartographer's local SLAM.
- With the prior, the correlative scan matcher's search is tightly centered → finds the correct local minimum reliably → submap insertions are crisp.
- Crisp submaps mean revisits look almost identical to previous traversals → branch-and-bound loop closure fires reliably → more edge constraints in the pose graph → tighter SPA optimization → less global drift.
- Net effect: the EKF doesn't *do* loop closure; it makes loop closure *reliable*. Stacked with the `max_range = 10` change (richer scan signatures) this is what unlocked the "soft-learning" behavior observed during multi-lap runs.

**The "anchor hard" tuning on AMCL works for the same reason and reveals the same ceiling:**

- Tight scan-match params (`z_hit=0.90`, `sigma_hit=0.20`, etc.) force AMCL to bind the current scan very strongly to map geometry — which is great when the map is correct, fragile when it isn't.
- The residual scan-vs-map offset observed in Foxglove is **not AMCL's fault** — it is the quality ceiling imposed by the frozen map. No further AMCL tuning will close it.

**Implications for the system design:**

- The current AMCL is being driven against an **abstract map** (Cartographer's self-referential output: origin at trajectory start, orientation at boot heading). For competition coordinates you want an **ideal map** — same geometry, but the YAML's `origin: [x, y, yaw]` field rigid-aligned to the world frame (e.g. corner of the track).
- Alignment is a **Procrustes / Kabsch** problem: given ≥3 landmark correspondences (abstract coords ↔ ideal coords), solve for the rigid transform via SVD. Closed form, no optimization. The semantic landmark mapper is the right input source for this.
- For competition runs this becomes: Cartographer (or RTAB-Map offline) → semantic landmark alignment → ideal `golden_map.yaml` → AMCL — which is exactly the architecture in the "Architecture Direction" section.

**Practical takeaways:**

1. Drive ≥3 overlapping laps before saving the map. Each pass adds loop-closure constraints; the third lap is typically near-optimal.
2. Use `cartographer_pbstream_to_ros_map` rather than `map_saver_cli` when accuracy matters — `write_state` forces a final SPA pass before export, producing a cleaner PGM.
3. Treat AMCL as **filter localization for race-time tracking**, not as a map-quality improver. It is fast and deterministic; that is its job.
4. Future work: implement landmark-based map alignment so AMCL operates in competition coordinates instead of Cartographer's abstract frame.

### 2026-05-24 EDT — BO test path must END FAR FROM TEST_ORIGIN (instant-complete fix)

After two failed BO runs, the recurring failure mode was finally pinned down:

**Symptom:** Most trials report `dur=0.0` with `arr` near 0.12 m and `std=0.000`. BO converges to a fake "best" point whose J was computed from zero seconds of driving.

**Cause:** The test path's FINAL waypoint was too close to TEST_ORIGIN (0.3 m). After APPROACH leaves the car within 0.30 m of TEST_ORIGIN, the car is often within 0.25 m of the final waypoint. `path_follower`'s `dist_to_final < 0.25` trigger fires on the first measurement tick → `/path_status = True` → measurement loop exits with no telemetry.

**The geometric constraint that must hold:**
```
distance(final_waypoint, TEST_ORIGIN) > APPROACH_ARRIVAL_RAD + 0.25 + margin
                                       0.30                + 0.25 + 0.30 = 0.85 m
```
With the L-shape final at (-0.5, 1.0) → 1.12 m from origin, we're 0.27 m above the floor. Safe.

**Updated test path geometry (in `scripts/bo_pd_tune.py`):**
```python
TEST_PATH_OFFSETS = [
    (1.0,  0.0),    # forward 1.0 m
    (1.0,  1.0),    # left    1.0 m
    (-0.5, 1.0),    # back    1.5 m   → FINAL at 1.12 m from TEST_ORIGIN
]
```

This is an OPEN L-shape, not a closed loop. Each trial ends at TEST_ORIGIN + (-0.5, 1.0). The next trial's APPROACH then drives the car ~1.1 m back to TEST_ORIGIN (~4 s) before starting measurement. So:
- Trial 1: APPROACH (instant, car already there) → MEASUREMENT (3 legs, ~25 s)
- Trial 2: APPROACH (~4 s drive back) → MEASUREMENT (~25 s)
- ... and so on, ~30 s per trial × 15 trials ≈ 8 min total.

**Other compounding fixes from the same debugging session:**
- `_drive_to` switched from `/path_status` wait (which can lag, queue, or never fire) to **position-based** completion: car within radius for N seconds.
- Measurement loop drains pending callbacks (`spin_once × 10`) before resetting `path_complete` to clear any stale /path_status from approach.
- Measurement loop ALSO has a position-based fallback (dist to final < 0.30 m for >0.5 s).
- APPROACH_ARRIVAL_RAD = 0.30 m, MUST be > path_follower's 0.25 m completion threshold — otherwise car stops driving before script considers it arrived → 25 s timeout every approach.
- APPROACH gains bumped to Kp=1.2, Kd=0.20 (was 1.0/0.30, too sluggish to reach 0.20 m in 20 s).

**Diagnostic log lines to watch for** (in BO terminal):
```
APPROACH arrived in 4.2s, final dist=0.18m                   ← good (took time, ended close)
Trial (kp=..., kd=...) → arr=0.15 sat=0.05 std=0.18 dur=22.4 dmean=0.61 | J=1.85
```
vs. broken:
```
APPROACH arrived in 0.3s, final dist=0.12m                   ← bad if happens trial 2+
Trial (...) → arr=0.18 std=0.000 dur=0.0 | J=0.22            ← instant-complete bug
```

If you see ALL trials with `dur=0.0`, the test path geometry is broken again (final waypoint too close to TEST_ORIGIN or to approach end-point). Re-check the table above.

### 2026-05-24 EDT — two-phase BO trials: APPROACH + MEASUREMENT

After the first BO run crashed the car and produced mostly junk J values (instant-complete or no-telemetry trials), the BO script (`scripts/bo_pd_tune.py`) was rewritten to run each trial in two distinct phases:

**Phase 1 — APPROACH (safe gains, not measured):**
- Drive the car back to TEST_ORIGIN with `APPROACH_KP=1.0, APPROACH_KD=0.30` (known-safe gains)
- Single-waypoint path = `[TEST_ORIGIN]`
- Timeout: 20 s
- Telemetry NOT counted toward J

**Phase 2 — MEASUREMENT (trial gains, measured):**
- Switch path_follower to the BO-suggested `(Kp, Kd)`
- Drive the 4-waypoint test path absolutely-anchored at TEST_ORIGIN
- All `/nav/*` telemetry during this phase contributes to cost function J
- Timeout: 45 s

**TEST_ORIGIN is captured when the user presses ENTER**, not hardcoded. The flow:

1. User starts BO script — it pauses and prints instructions
2. User puts path_follower in `manual` mode, drives car to a safe open spot
3. User exits manual mode (`q`) and presses ENTER in BO terminal
4. BO captures the current `/qcar2_pose_fused` as TEST_ORIGIN
5. All 15 trials use that origin as their anchor

This eliminates three issues from the first run:
- **Crashes**: user picks a safe open spot before BO starts
- **Drift across trials**: every trial returns to the SAME origin before measuring
- **Instant-complete trials**: the test path's final waypoint is reliably 0.3 m from origin (above the 0.25 m threshold), and the approach phase guarantees the car is AT the origin at trial start

**Test path geometry (offsets from TEST_ORIGIN):**

```
TEST_PATH_OFFSETS = [
    (1.0, 0.0),     # forward 1 m
    (1.0, 1.0),     # left  1 m
    (0.0, 1.0),     # back  1 m
    (0.0, 0.3),     # FINAL — 0.3 m north of TEST_ORIGIN
]
```

Total path length ~3.7 m. Segment lengths all ≥ 0.7 m (above lookahead at 0.4 m/s default speed). Drive time per trial: ~18 s. Combined with the approach phase (~10 s avg), each trial is ~30 s × 15 trials = ~8 min total BO time.

**Space requirements:** at least 1.5 m of clear space ahead and to the LEFT of TEST_ORIGIN. The test path forms a 1 m × 1 m square in that direction.

**Crash recovery:** if a trial crashes the car (e.g., BO picked a very aggressive Kp/Kd that ran into a wall), Ctrl-C the BO script, manually drive the car free, restart BO. Past trials are NOT preserved (skopt doesn't checkpoint by default).

**Per-trial log lines now print:**

```
[APPROACH] done in 8.2s, arrived 0.12m from target
Trial (kp=1.097, kd=0.267) → arr=0.18 sat=0.00 std=0.11 dur=16.4 dmean=0.65 | J=2.41
```

If you see `dur=0.0` consistently, the path_follower is still hitting instant-complete. If you see `[APPROACH] TIMEOUT`, the approach gains can't reach the origin — check path_follower's controller_mode is actually switching to autonomous when BO publishes /cmd_waypoints.

### 2026-05-24 EDT — unified control modes in path_follower (manual_drive folded in)

`path_follower` is now the **single owner** of `/cmd_vel_nav`. No more double-publisher tug-of-war when both `path_follower` and `manual_drive` are running. Manual-drive's WASD keystroke logic has been folded into `path_follower` as a thread, so there is exactly one node that ever publishes the steering bus.

**New parameter: `control_mode`** (string, default `"idle"`):

| Mode | What `path_follower` does | Use case |
|---|---|---|
| `idle` | Nothing — does NOT publish `/cmd_vel_nav` at all. Diagnostics still publish. | Default safe state. Lets you launch path_follower without it driving immediately. |
| `manual` | Spawns a daemon thread that reads WASD keystrokes from path_follower's terminal and publishes commands via `self.publisher` (the same one autonomous mode uses). | Drive the car by hand with the keyboard, in the same terminal that's running path_follower. |
| `autonomous` | Existing pure-pursuit behavior; computes PD steering from `/qcar2_pose_fused` + waypoints. | Normal driving along a path. |

**How to switch modes:**

```bash
# Idle (default) — does nothing
ros2 run qcar2_autonomy path_follower

# Manual — keystrokes from this terminal control the car
ros2 param set /path_follower control_mode "manual"
# WASD = steer, space/x = stop, +/- = speed, [/] = turn rate, q = exit to idle

# Autonomous — drive the path (default node_values = [0, 8, 10])
ros2 param set /path_follower control_mode "autonomous"
```

**Auto-switching:** the mode switches automatically into `autonomous` when:
- `/cmd_waypoints` arrives with a Path message (BO script, trip_planner, manual `ros2 topic pub`)
- `node_values` parameter changes (`ros2 param set /path_follower node_values "[...]"`)

These are explicit "I want to drive" intents.

**Manual-mode controls (WASD, same as the old `manual_drive` node):**

```
  w : forward    s : reverse
  a : steer left d : steer right
  space/x : stop
  + / - : change forward/reverse speed by 0.05
  [ / ] : change turn rate by 0.05
  q : exit manual mode → switch back to idle
```

Manual-mode tunables (all parameters):
- `manual_forward_speed` (default 0.25 m/s)
- `manual_reverse_speed` (default 0.20 m/s)
- `manual_turn_rate`     (default 0.50 rad)
- `manual_speed_step`    (default 0.05)

**Standalone `manual_drive` node still exists** — kept for backward compatibility and physical-hardware use cases. But if you run BOTH `manual_drive` AND `path_follower` in `manual` mode, they will fight again. **Use ONE OR THE OTHER**: either folded-in manual mode (preferred — single ownership) or the standalone `manual_drive` node with `path_follower` in `idle` mode.

**Implementation notes:**

- The keystroke thread requires path_follower's terminal to be a TTY. If stdin isn't a TTY (e.g., launched via launch file in background), `manual` mode logs a warning and degrades to behaving like `idle`. Best practice: run `ros2 run qcar2_autonomy path_follower` in a foreground terminal you can type into.
- `nav_command()` (line 655) early-returns if `control_mode != 'autonomous'`. This is the kill-switch that prevents the tug-of-war.
- Switching INTO `manual` runs `_setup_terminal()` (cbreak mode on stdin). Switching OUT runs `_restore_terminal()`. If path_follower crashes in manual mode, the terminal may need `reset` typed to recover.

**Implications for BO:**

The BO script (`scripts/bo_pd_tune.py`) publishes `/cmd_waypoints`, which now also auto-switches path_follower into `autonomous` mode. So the BO workflow becomes even simpler:

```bash
ros2 run qcar2_autonomy path_follower            # starts IDLE — does nothing
ros2 param set /path_follower control_mode "manual"
# Drive the car around for 30s to warm Cartographer (in this terminal)
# Press 'q' → returns to idle

python3 /workspaces/isaac_ros-dev/ros2/scripts/bo_pd_tune.py
# Press ENTER when ready
# /cmd_waypoints arrives → path_follower auto-switches to autonomous
# 15 trials run, then it goes back to idle when complete (final waypoint reached)
```

No second `manual_drive` terminal needed.

### 2026-05-24 EDT — path_follower idle by default + BO Enter-to-start gate

Two related changes that fix the manual-vs-autonomous bus fight and add a Cartographer warm-up window for BO.

**1. `path_follower` now starts in IDLE mode.** The `start_path` parameter default flipped from `True` to `False`. While idle, `path_follower`'s `enable` (line 565) is 0 → it publishes `/cmd_vel_nav` with zero linear/angular commands → does not fight `manual_drive` for the bus. Three ways to enter autonomous mode:

```bash
# Method 1 — send a path (BO uses this, also any /cmd_waypoints client)
ros2 topic pub --once /cmd_waypoints nav_msgs/msg/Path "..."

# Method 2 — toggle the start_path parameter
ros2 param set /path_follower start_path "[true]"

# Method 3 — set node_values (treated as "I want to drive these nodes")
ros2 param set /path_follower node_values "[0, 8, 10]"
```

All three set `self.path_execute_flag = True` internally. To exit autonomous mode:

```bash
ros2 param set /path_follower start_path "[false]"
```

**Implication for workflow:** you can now start `path_follower` early and `manual_drive` later (or vice versa) without them stepping on each other. Only when you explicitly issue a path command does the controller take over.

**2. `bo_pd_tune.py` now waits for ENTER before starting trials.** This is the Cartographer warm-up gate — manually drive the car around for ~30 s with `manual_drive` to give Cartographer a stable map, then return to the BO terminal and press ENTER. The script prints clear instructions:

```
======================================================================
  BO is READY but PAUSED.

  Before pressing ENTER:
    1. Run `ros2 run qcar2_autonomy manual_drive` in another terminal.
    2. Drive the car around the test area for ~30 s so
       Cartographer builds a stable local map.
    3. (Optional) Drive the car back near (0, 0) in map frame.
    4. Stop manual_drive (Ctrl-C) so the steering bus is free.

  Once you press ENTER:
    - 15 BO trials will run back-to-back (~12 min total).
    - Each trial drives the test triangle in map frame.
    - Do NOT interfere unless something goes wrong (Ctrl-C this script).
======================================================================

  Press ENTER when ready (or Ctrl-C to abort)...
```

**Full Cartographer + BO workflow (final form for tonight):**

```bash
# Build
colcon build --symlink-install --packages-select qcar2_autonomy qcar2_nodes
source install/setup.bash && export ROS_DOMAIN_ID=69
ros2_killall

# Stack
ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py     # SLAM + base
ros2 run qcar2_autonomy ekf_fusor                                  # filtered pose
ros2 launch qcar2_nodes foxglove_bridge_launch.py                  # bridge + watchdog
ros2 run qcar2_autonomy path_follower                              # IDLE (doesn't drive)

# (Optional) Tkinter slider for ad-hoc tuning
python3 /workspaces/isaac_ros-dev/ros2/scripts/pd_tuner.py

# BO — paused waiting for warm-up
python3 /workspaces/isaac_ros-dev/ros2/scripts/bo_pd_tune.py

# In another terminal: warm up Cartographer
ros2 run qcar2_autonomy manual_drive
# Drive ~30s, Ctrl-C manual_drive
# Return to BO terminal, press ENTER
# Wait ~12 minutes
cat /tmp/bo_pd_tune_result.json
```

### 2026-05-24 EDT — Controller watchdog + PD tuner GUI + Bayesian Optimization

Three new pieces of infrastructure, all wired to the existing path_follower diagnostic topics. No path_follower / EKF / Cartographer changes required to use any of them.

**1. `controller_watchdog` node (`qcar2_autonomy/autonomy/controller_watchdog.py`)**

Mirrors the pattern of `ekf_fusor` health: passive observer that subscribes to `/nav/blended_delta`, `/nav/psi`, `/nav/steering_saturation_rate` and publishes `/nav/controller_health` (`std_msgs/String`) at 2 Hz. States:

| State | Trigger |
|---|---|
| `healthy` | No issues |
| `warming_up` | Not enough samples yet (first ~250 ms) |
| `saturated` | `steering_saturation_rate > 0.5` (controller fighting hardware limit) |
| `wiggling` | `std(blended_delta) over 1s > 0.20 rad` (high-freq oscillation) |
| `late_reaction` | `|psi| > 0.5 rad` AND `|blended_delta| < 0.10 rad` for >0.5 s (target ahead, controller not steering enough → Kp too low or Kd too high) |

Run:
```bash
ros2 run qcar2_autonomy controller_watchdog
```

In Foxglove → add **Indicator** panel → topic `/nav/controller_health` → field `.data` → set color rules: `healthy`=green, `warming_up`=blue, `saturated`=red, `wiggling`=orange, `late_reaction`=yellow.

Thresholds are ROS parameters (`sat_threshold`, `wiggle_std_threshold`, `late_psi_threshold`, `late_delta_threshold`, `late_min_duration`) so you can adjust without editing code.

Intentionally NOT implemented: `off_path` (requires cross-track computation) and `stuck` (longer time window). Add these if you observe symptoms that match.

**2. `pd_tuner.py` Tkinter slider GUI (`scripts/pd_tuner.py`)**

Foxglove's Publish panel only fires on button click, not on slider drag. This script gives you continuous-drag slider control:

```bash
python3 /workspaces/isaac_ros-dev/ros2/scripts/pd_tuner.py
```

Opens a small window with two sliders (Kp 0–2, Kd 0–1). Drag → publishes to `/nav/kp_steering_set` / `/nav/kd_steering_set` → path_follower picks up the change on the next 80 Hz tick. Watch path_follower log for `kp_steering (topic) -> X.XXX` confirmation.

Requires X forwarding / desktop environment.

**3. `bo_pd_tune.py` — Bayesian Optimization automatic PD tuner (`scripts/bo_pd_tune.py`)**

Automatically finds the (Kp, Kd) pair that minimizes a controller-quality cost function. Wires into the existing system as a peer node — no controller/EKF/SLAM modifications.

How it wires:

```
┌──────────────────┐     publish (Float32)         ┌──────────────────┐
│  bo_pd_tune.py   │ ─── /nav/kp_steering_set ──▶ │  path_follower   │
│                  │ ─── /nav/kd_steering_set ──▶ │                  │
│                  │ ─── /cmd_waypoints (Path) ──▶│                  │
│   skopt          │                              └──────────────────┘
│   gp_minimize    │
│   surrogate(GP)  │   subscribe
│   acquisition(EI)│ ◀── /nav/blended_delta
│                  │ ◀── /nav/steering_saturation_rate
│                  │ ◀── /nav/distance_to_final
│                  │ ◀── /path_status (Bool)
│                  │ ◀── /qcar2_pose_fused
└──────────────────┘
```

Each trial runs the same fixed test path (4-waypoint triangle, ~3 m) and computes:
```
J = W_ARRIVAL * arrival_error
  + W_SATURATION * mean(saturation_rate)
  + W_DELTA_STD * std(blended_delta)
  + W_DURATION * trial_duration
  + W_DIST_MEAN * mean(distance_to_final)
```

Default weights (top of file): `1.0 / 2.0 / 0.5 / 0.1 / 0.3`. Lower J = better tune.

BO procedure:
- 5 random sample points (initial exploration)
- 10 acquisition-driven samples using Gaussian Process surrogate + Expected Improvement
- 15 trials total, ~12 minutes on QLabs
- Output: `/tmp/bo_pd_tune_result.json` with best (Kp, Kd) + full trial log
- Best gains are auto-applied at the end so the car drives with them immediately

Dependencies:
```bash
pip install scikit-optimize numpy
```

Run:
```bash
# First start the normal stack: cartographer + ekf_fusor + foxglove + path_follower
# Then in another sourced terminal:
python3 /workspaces/isaac_ros-dev/ros2/scripts/bo_pd_tune.py
```

**Why Bayesian Optimization, full name + reading list:**

Also called **Sequential Model-Based Optimization (SMBO)** or **Gaussian Process Optimization**. Core idea: when each trial is expensive (driving a car, training a neural network, etc.), don't grid-search blindly — model the cost function `J(Kp, Kd)` as a stochastic process (typically a Gaussian Process) and use that model to decide where to sample next. Balances exploration (try where uncertainty is high) and exploitation (try near known good points) via an acquisition function (default here: Expected Improvement).

Foundational reading, in order of usefulness for engineers:
1. **Frazier 2018** — "A Tutorial on Bayesian Optimization" (arXiv:1807.02811) — best intro
2. **Snoek, Larochelle, Adams 2012** — "Practical Bayesian Optimization of Machine Learning Algorithms" — application to hyperparameter tuning
3. **Jones, Schonlau, Welch 1998** — "Efficient Global Optimization of Expensive Black-Box Functions" — the EGO algorithm, modern BO's foundation
4. **Mockus 1975** — original Bayesian extremum-seeking formulation (historical)

Library used: [`scikit-optimize`](https://scikit-optimize.github.io/) — open source, ~200 LOC integration, GP surrogate + EI acquisition by default.

### 2026-05-24 EDT — Live-tunable PD gains + controller diagnostics for Foxglove

`path_follower` (`nav_to_pose.py`) now exposes its pure-pursuit damping PD gains as ROS 2 parameters and publishes a controller-health diagnostic set. Three things you can do that you couldn't before:

**1. Tune Kp / Kd live during a drive** — no rebuild, no restart. From any sourced terminal:

```bash
ros2 param set /path_follower kp_steering 1.2
ros2 param set /path_follower kd_steering 0.4
```

Or in Foxglove → add a **Parameters** panel → expand `/path_follower` → drag the `kp_steering` and `kd_steering` sliders. Changes take effect on the next path_planner tick.

**Defaults** (lowered after the deg→rad unit bug fix; see prior change log entry):
- `kp_steering = 1.0`
- `kd_steering = 0.3`

**Tuning rules of thumb:**

| Behavior | Action |
|---|---|
| Car wiggles at nodes | Raise `kd_steering` by 0.1 |
| Car turns sluggishly, scrapes walls on inside of curve | Lower `kd_steering` by 0.1 |
| Car undershoots curves | Raise `kp_steering` by 0.1 |
| Car overshoots / oscillates | Lower `kp_steering` by 0.1 OR raise `kd_steering` |

Hard limits: `0.0 ≤ Kd ≤ 1.5`, `0.5 ≤ Kp ≤ 1.8`. Outside these ranges the controller becomes unstable.

**2. New controller-health topics** (all `std_msgs/Float32`, accessible via `.data` in Foxglove Plot panels):

| Topic | Meaning | Healthy values |
|---|---|---|
| `/nav/distance_to_waypoint` | m to current target waypoint | decreases steadily toward 0.3, then jumps to next waypoint |
| `/nav/distance_to_final` | m to last waypoint of path | monotonically decreasing |
| `/nav/psi` | rad, angle from car nose to target | bounded ±0.5 in healthy operation; near ±π means "target behind car" |
| `/nav/steering_saturation_rate` | 0..1, fraction of last ~1s at the ±0.52 limit | should be near 0; >0.3 means controller is fighting the limit |
| `/nav/speed_cmd` | m/s commanded forward speed (after cos² scaling) | matches `desired_speed` on straights, drops in turns |
| `/nav/yaw_rate_imu` | rad/s, raw IMU gyro | should track `/nav/blended_delta * speed / wheelbase` |
| `/nav/progress_rate` | m/s, d(dist_to_final)/dt | should be ~ -speed when on path; near 0 means stuck |
| `/nav/wpi` | float-encoded current waypoint index | monotonically increasing through the path |
| `/nav/controller_mode` | 0=PP, 1=blended, 2=stopping (within 0.5m), 3=complete | 0 most of the time, 2 briefly, 3 at end |

**3. Recommended Foxglove dashboard for tuning**

Six Plot panels in a 2×3 grid, all in sliding-window mode (X axis: current time, 30s window):

```
┌──────────────────────────┬──────────────────────────┐
│ Plot: Distance            │ Plot: PD Output           │
│   /nav/distance_to_wp    │   /nav/pp_delta           │
│   /nav/distance_to_final │   /nav/blended_delta      │
│   /nav/progress_rate     │   /nav/yaw_rate_imu       │
├──────────────────────────┼──────────────────────────┤
│ Plot: Geometry            │ Plot: Saturation          │
│   /nav/psi               │   /nav/steering_saturation_rate │
│                          │   (reference line at 0.3)  │
├──────────────────────────┼──────────────────────────┤
│ Parameters panel          │ Plot: Speed               │
│   /path_follower/kp_steering   /nav/speed_cmd       │
│   /path_follower/kd_steering   /nav/wpi              │
└──────────────────────────┴──────────────────────────┘
```

The Parameters panel is the key one — that's how you tune live without touching the terminal.

**Repeatability test (separate procedure, run when controller is tuned):**

1. Drive (manually or with planner) to some point. Record pose:
   ```bash
   ros2 run tf2_ros tf2_echo map base_link --once
   # Note: target_x, target_y
   ```
2. Send the car elsewhere via `/cmd_waypoints` (a point ~1m away).
3. When it arrives, send it back to (target_x, target_y) via `/cmd_waypoints`:
   ```bash
   ros2 topic pub --once /cmd_waypoints nav_msgs/msg/Path \
     "{header: {frame_id: 'map'}, poses: [{pose: {position: {x: TARGET_X, y: TARGET_Y, z: 0.0}}}]}"
   ```
4. When it arrives, record actual pose. Compute `e_i = sqrt((x_actual - target_x)² + (y_actual - target_y)²)`.
5. Repeat 5–10 times.

Stats to compute:
- Mean error `μ_e` = accuracy (systematic offset)
- Std deviation `σ_e` = repeatability (random scatter)

Interpretation buckets:
- `σ_e < 5cm`: excellent
- `5–15cm`: healthy
- `15–30cm`: marginal — may fail precision tasks like pickup
- `>30cm`: localization drift or PP oscillation; needs investigation

### 2026-05-24 EDT — QLabs spawn pose at SDCSRoadMap node 0 (empirical calibration)

To spawn the virtual QCar 2 at SDCSRoadMap **node 0** (the default start of the path_follower's `node_values = [0, 8, 10]`), edit either `Setup_Competition_Map.py` or `Setup_Real_Scenario.py` inside the QLabs container (`/home/qcar2_scripts/python/Base_Scenarios_Python/`) — both ship with the same defaults.

Change the `initialPosition` and `initialOrientation` defaults (function signature **and** the call site) to:

```python
initialPosition    = [0.000, 0.130, 0.005]
initialOrientation = [0, 0, -33]   # empirical; NOT -90 like SDCSRoadMap says
```

**Why -33° and not -90°:** SDCSRoadMap reports node 0 yaw as `-π/2 = -90°`, but that's in the roadmap's own pixel-derived frame. Three rotation conventions stack between SDCSRoadMap and what QLabs renders:

1. **SDCSRoadMap node** yaw = `-90°` (roadmap frame)
2. **QLabs floor** is spawned with `rotation = [0, 0, -90]` (the roadmap image is laid down rotated -90° from QLabs world)
3. **`nav_to_pose.py:rotation_offset`** parameter = `83°` default (the QLabs→ROS frame correction inside `R_QLabs_ROS` at lines 408-411)

Combined, the QLabs world-frame yaw at which the car aligns nose-along-road at node 0 is `-33°` (validated by spawning at that orientation and observing the car sit straight on the lane in the QLabs viewer). Do not try to derive it analytically — the rotation_offset is itself a calibration value that depends on how Cartographer's `map` frame happens to settle, and the practical answer is the one that works.

**For other nodes**, the same `-33° vs -90°` offset (`+57°` correction) should apply if you want to spawn at a different node's nominal pose. So for any SDCSRoadMap node with yaw `θ_node`, the QLabs spawn yaw is approximately `θ_node + 57°`.

**Root cause to fix later (not now):** `nav_to_pose.py:auto_align_roadmap_to_current_pose()` exists to handle exactly this — automatically compute the rigid alignment between the SDCSRoadMap waypoints and the car's actual measured pose. If that worked cleanly, no spawn-pose hand-calibration would be needed. Revisit after Day 4 validation if there's time before competition.

### 2026-05-24 EDT — nav_to_pose now consumes /qcar2_pose_fused (Day 4 of EKF refactor)

`nav_to_pose.py` (the path follower) used to instantiate its own `QcarEKF` + `GyroKF` privately, run both prediction (`ekf_filter_timer`) and correction (`tf_timer` calling `qcar2_ekf.correction`) internally, and read state from `self.qcar2_ekf.xHat` for pure-pursuit. That meant the EKF was hidden inside one node, with no diagnostics, no Foxglove visibility, and no other consumer could share the same filtered pose.

Day 4 retires the embedded EKF in favor of consuming `/qcar2_pose_fused` from the `ekf_fusor` node built on Day 3.

**Concrete changes inside `nav_to_pose.py`:**

| Before | After |
|---|---|
| `self.qcar2_ekf = QcarEKF(...)` in `__init__` | Deleted. Replaced with subscription to `/qcar2_pose_fused` and three cache fields (`fused_pose_x/y/yaw`). |
| `self.gyro_kf = GyroKF(...)` in `__init__` | Deleted. ekf_fusor owns the gyro KF internally now. |
| `self.gyro_kf.correction(...)` + `self.qcar2_ekf.correction(...)` calls inside `tf_timer` | Deleted. ekf_fusor handles correction. `tf_timer` now only caches raw TF as a startup fallback and republishes `/robot_pose`. |
| `def ekf_filter_timer(self): ... qcar2_ekf.prediction(...)` | **Deleted entirely.** Prediction is now done at 80 Hz by ekf_fusor. |
| `self.ekf_filter_timer()` call at top of `path_planner` | Removed (was a no-op now). |
| `th = self.qcar2_ekf.xHat[2, 0]; p = [...xHat[0, 0], ...xHat[1, 0]]` inside pure-pursuit loop | Replaced with `p, th = self._read_pose_for_planner()` — new helper with priority chain. |
| Two `self.qcar2_ekf.xHat` reads inside `scopeDataTimer` (visualization) | Replaced with reads from the fused-pose cache. |
| `from autonomy.estimation import QcarEKF, GyroKF` | Removed (unused). |

**New pose-source priority chain (`_read_pose_for_planner`):**

```python
if self.fused_pose_x is not None:
    return (fused_pose_x, fused_pose_y), fused_pose_yaw   # PRIMARY
try:
    return (translation.x, translation.y), self.yaw       # FALLBACK 1: raw TF
except AttributeError:
    return (0.0, 0.0), 0.0                                # FALLBACK 2: origin
```

This means the path follower automatically degrades gracefully:
- Once ekf_fusor bootstraps and publishes `/qcar2_pose_fused`, it's the source of truth
- If ekf_fusor isn't running, the planner uses raw `map -> base_link` TF directly (Cartographer/AMCL feeding it)
- If TF is also missing, it stays at origin until something arrives

**What this unlocks:**

1. **Single point of truth.** Whatever pose ekf_fusor settles on is what the controller drives against. Diagnostics on `/qcar2_ekf/*` directly reflect what the controller sees.
2. **Outlier rejection benefits the controller.** When ekf_fusor's Mahalanobis gate rejects a bad measurement, the path planner doesn't see a spike — it sees a smooth pose. The gain we observed in the stress test (maha=150 rejections) now protects pure pursuit.
3. **Other consumers can use the same pose.** Semantic mapper, active-SLAM monitor, or future arbiter nodes can subscribe to `/qcar2_pose_fused` and have all of them see the same world state.
4. **Smaller `nav_to_pose.py`.** ~40 lines deleted. Easier to maintain and reason about.

**Compatibility:**

- `/robot_pose` still publishes (raw TF, unchanged behavior for any external subscribers)
- `tf_buffer` / `tf_listener` still active for `/robot_pose` and as the fallback path
- `auto_align_roadmap_to_current_pose()` and other helpers unchanged
- Pure-pursuit math unchanged
- Stanley blend, scope visualization, parameter callbacks unchanged

**Smoke test:**

```bash
cd /workspaces/isaac_ros-dev/ros2
colcon build --symlink-install --packages-select qcar2_autonomy qcar2_nodes qcar2_interfaces qcar2_perception
source install/setup.bash && export ROS_DOMAIN_ID=69
ros2_killall

# Terminal A: Cartographer + base hardware + pose_estimator
ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py
# Terminal B: ekf_fusor (TF correction mode)
ros2 run qcar2_autonomy ekf_fusor
# Terminal C: foxglove
ros2 launch qcar2_nodes foxglove_bridge_launch.py
# Terminal D: autonomy
ros2 launch qcar2_autonomy autonomy_planner_launch.py
```

Expected: path follower drives waypoints exactly as before. The only invisible-to-user change is *which topic the planner reads its pose from*. If the path is jittery or skews unexpectedly, it's because the fused pose is feeding it slightly different values than raw TF — which is the whole point (better filtered pose) but might require minor pure-pursuit tuning if it overcorrects.

**Next (Day 5):** integrate manual_drive as a *mode* of nav_to_pose rather than a separate node, so the path follower becomes the single command authority with autonomous/manual switching via parameter. This was the "matrix B" framing from earlier — manual_drive becomes the input-mapping function called when `control_mode:=manual`.

### 2026-05-24 EDT — ekf_fusor node (Day 3 of EKF refactor)

New ROS 2 node `qcar2_autonomy/autonomy/ekf_fusor.py` implements the Quanser-spec `ekf_fusor` from "ROS 2 For QCar 2" doc. Wraps the extracted `QcarEKF` + `GyroKF` filters into a standalone node that fuses encoder + IMU + steering (prediction) with Cartographer or AMCL pose (correction).

**Architecture position:** ekf_fusor is a **downstream consumer** of Cartographer/AMCL. It does NOT publish anything that feeds back into them — `pose_estimator.py` still owns `/odom` and `odom -> base_link` for the SLAM motion prior. Avoids the circular-reasoning bug we removed earlier.

**Run modes** (via `correction_source` param):

- `tf` — polls TF `map -> base_link` at 10 Hz. Use during Cartographer mapping.
- `amcl_pose` — subscribes to `/amcl_pose`, uses AMCL's published covariance. Use during AMCL runtime.
- `none` — prediction only, no correction. Debug only.

**Outputs:**

- `/qcar2_pose_fused` — `PoseWithCovarianceStamped`, main consumer-facing output
- `/qcar2_ekf/odometry_fused` — `nav_msgs/Odometry` with twist (speed + yaw rate)
- `/qcar2_ekf/p_diag` — diagonal of state covariance (Foxglove plot: watch this shrink after corrections)
- `/qcar2_ekf/k_diag` — diagonal of last Kalman gain (Foxglove plot: K converging means filter is settling)
- `/qcar2_ekf/innovation` — last innovation vector `[Δx, Δy, Δθ]`
- `/qcar2_ekf/innovation_mahalanobis` — scalar test statistic for outlier gating
- `/qcar2_ekf/health` — string status (`healthy` / `degraded` / `prediction_only` / `outlier_streak` / `bootstrapping`)
- `/qcar2_ekf/mode` — current correction source

**Operational safety details** (the difference between demo-grade and competition-reliable):

1. **Outlier gate** — Mahalanobis distance `y^T S^-1 y` compared against χ²_3 threshold (default 11.345 at 99% confidence). Bad corrections (e.g. AMCL momentary glitch) are rejected, logged, and counted.
2. **Stale-correction detection** — tracks time since last accepted correction. Transitions to `degraded` after 2 s, `prediction_only` after 5 s. In `prediction_only` mode, covariance is actively inflated so downstream consumers can see uncertainty growing.
3. **Bootstrap** — does not predict before either the first correction arrives OR a non-zero `initial_pose` param is set. Avoids integrating noise from (0,0,0) before a real pose is known.
4. **dt clamp** — skips predict step if `dt > max_dt` (default 0.25 s) to avoid bad linearization.
5. **TF broadcast optional** — default `publish_tf: false` so we don't fight Cartographer/AMCL/pose_estimator for the same TF edge. If enabled, broadcasts `map -> base_link_fused` (separate child frame to avoid conflict).
6. **AMCL covariance pass-through** — uses `pose.covariance` from `/amcl_pose` if non-zero, falls back to a configurable default otherwise. So R adapts to AMCL's own confidence.

**Parameters** (sensible defaults; tune in launch file):

```yaml
ekf_fusor:
  ros__parameters:
    wheelbase: 0.256                          # m, from manual Table 11
    predict_rate: 80.0                        # Hz
    map_frame: "map"
    base_frame: "base_link"
    correction_source: "tf"                   # tf | amcl_pose | none
    amcl_pose_topic: "/amcl_pose"
    use_gyro_kf: true
    q_diag: [0.0005, 0.0005, 0.002]
    r_carto_diag: [0.02, 0.02, 0.01]
    r_amcl_default_diag: [0.05, 0.05, 0.03]
    max_dt: 0.25
    tf_correction_rate: 10.0
    stale_correction_warning_sec: 2.0
    stale_correction_predict_only_sec: 5.0
    mahalanobis_gate_chi2: 11.345
    publish_tf: false
    fused_child_frame: "base_link_fused"
    initial_pose: [0.0, 0.0, 0.0]
    steering_limit: 0.52
```

**Registered as console script** in `setup.py`:

```bash
ros2 run qcar2_autonomy ekf_fusor
ros2 run qcar2_autonomy ekf_fusor --ros-args -p correction_source:=amcl_pose
```

**Smoke test:**

```bash
cd /workspaces/isaac_ros-dev/ros2
colcon build --symlink-install --packages-select qcar2_autonomy qcar2_nodes qcar2_interfaces qcar2_perception
source install/setup.bash && export ROS_DOMAIN_ID=69
ros2_killall

# Terminal A: Cartographer
ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py

# Terminal B: ekf_fusor in TF mode
ros2 run qcar2_autonomy ekf_fusor

# Terminal C: watch the filter converge
ros2 topic echo /qcar2_ekf/p_diag           # diagonal should drop after first correction
ros2 topic echo /qcar2_ekf/innovation_mahalanobis  # should be small after bootstrap
ros2 topic echo /qcar2_ekf/health           # should be "bootstrapping" then "healthy"

# Terminal D (Foxglove): add Plot panels for /qcar2_ekf/{p_diag, k_diag, innovation_mahalanobis}
```

**Next step (Day 4):** refactor `nav_to_pose.py` to consume `/qcar2_pose_fused` instead of computing its own EKF internally. After that the embedded EKF inside the path follower becomes redundant and can be deleted (the QcarEKF instance in nav_to_pose's `PathFollower.__init__` stays as a dead path until then).

### 2026-05-24 EDT — extracted Kalman filter primitives (Day 2 of EKF refactor)

`QcarEKF` and `GyroKF` (the 2D bicycle-model EKF and the yaw + yaw-rate-bias filter) used to live as private classes inside `qcar2_autonomy/autonomy/nav_to_pose.py`. They've been moved to a new subpackage so the upcoming `ekf_fusor` node and any other consumer (semantic mapper, active-SLAM monitor, diagnostics) can reuse the same math.

New location:

```
qcar2_autonomy/autonomy/estimation/
├── __init__.py        # re-exports QcarEKF, GyroKF
└── filters.py         # the actual filter implementations
```

`nav_to_pose.py` now imports via:

```python
from autonomy.estimation import QcarEKF, GyroKF
```

**Behavior is unchanged.** This is a pure refactor — same math, same constants (wheelbase 0.256 m from Table 11), same predict + correct logic, same use sites inside `PathFollower`. Validated by:

```bash
cd /workspaces/isaac_ros-dev/ros2
colcon build --symlink-install --packages-select qcar2_autonomy
source install/setup.bash && export ROS_DOMAIN_ID=69
# Run cartographer launch as before; nav_to_pose should behave identically.
ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py
```

Next step (Day 3-4): build `ekf_fusor.py` as a standalone node that imports the same filters and exposes the fused pose + EKF diagnostics for Foxglove.

### 2026-05-23 EDT — hardware-spec corrections (gear ratio, wheelbase, steering)

Three constants in the autonomy stack disagreed with the QCar 2 User Manual - System Hardware v1.0 (2024-10-01). Corrected against Table 11 (dimensions), page 12 (drive train), and page 17 (steering range).

| Constant | Before | After | Reason |
|---|---|---|---|
| `GEAR_RATIO` (`pose_estimator.py`) | `(13*19) / (70*30)` = 0.1176 | `(13*19) / (70*37)` = 0.0954 | Manual: motor pinion (13/70) × differential (19/37). The 30 was wrong → linear velocity overestimated by ~23%, dead-reckoning ballooned. |
| `WHEELBASE` (`pose_estimator.py`, `nav_to_pose.py:QcarEKF`) | 0.257 | 0.256 | Manual Table 11. Note: `nav_to_pose.py` already used 0.256 in its pure-pursuit calc at line 487 — file was internally inconsistent before this fix. |
| Steering limit (`pose_estimator.py:steering_limit`, `nav_to_pose.py:max_steering_angle`) | 0.6 rad | 0.52 rad | Manual Table 11: max steering = ±30° = ±0.52 rad. Commanding 0.6 produced bicycle-model yaw rates the servo physically cannot deliver, so predicted ω diverged from actual ω. |

**Expected impact:**

- Cartographer's motion prior (via EKF `/odom`) becomes ~23% more accurate in linear velocity. Submap insertion alignment should tighten.
- AMCL's dead-reckoning between corrections should drift much less at speed — this was likely the root cause of "AMCL keeps slipping at high speed" symptom.
- Yaw predictions from the bicycle model match servo limits. May allow AMCL's motion-model alphas (`alpha1..5`) to be re-tightened from the conservative 0.2 defaults.

**Verification after rebuild:**

```bash
# Drive straight 1 m by hand-pushing or QLabs known motion. Compare:
ros2 topic echo /odom --once   # x should increase by ~1.0 m, not ~1.23 m
```

**LiDAR static TF (`fixed_lidar_frame_virtual.cpp`) intentionally NOT changed.** Manual page 10 confirms physical LiDAR is mounted with 180° yaw offset (`rplidar_to_body` rotation matrix `[[-1,0,0],[0,-1,0],[0,0,1]]`), matching the physical file's `setRPY(0,0,-π)`. Virtual file with `setRPY(0,0,0)` is correct IF QLabs publishes the scan pre-rotated into the body frame's forward convention. Defer this change until physical-hardware testing confirms which convention QLabs uses.

### 2026-05-23 EDT

- Wired the EKF as the single owner of `/odom` and `odom -> base_link` TF.
- Rewrote `qcar2_autonomy/autonomy/pose_estimator.py` to publish `nav_msgs/Odometry` on `/odom` and broadcast `odom -> base_link`. Removed the `map -> base_link` correction loop that made the EKF a downstream observer of SLAM.
- Retired the `qcar2_odometry` C++ node from active launches: removed from `qcar2_amcl_localization_launch.py` and `qcar2_amcl_localization_virtual_launch.py`. Already commented out of `qcar2_virtual_launch.py`.
- Enabled `use_odometry = true` in `qcar2_nodes/config/qcar2_2d.lua` so Cartographer fuses the EKF `/odom` as a motion prior.
- Reason: Cartographer was running on LiDAR-only scan matching with no motion prior, causing drift in open stretches between distinctive geometry. The homemade EKF existed but was a sidecar publishing `/robot_pose` that nobody consumed.

### 2026-05-22 19:00:55 EDT

- Promoted the runbook to `Easy_Start.md`.
- Added a table of contents and split the file into run, build, perception, RTAB, logs/debugging, and architecture sections.
- Added a dedicated logs/debugging section for build logs, ROS run logs, launch log capture, graph checks, topic checks, semantic marker layers, and bag commands.
- Kept the current QCar2 perception, Cartographer, RTAB, Docker, and architecture notes.

### 2026-05-22 18:52:41 EDT

- Reorganized this file into Markdown-style sections for copy-paste use.
- Added corrected RTAB executable names for this source branch:
  - `ros2 run rtabmap_sync rgbd_sync`
  - `ros2 run rtabmap_odom rgbd_odometry`
  - `ros2 run rtabmap_slam rtabmap`
- Added RTAB RGB-D smoke-test commands wired to the current D435 topics:
  - `/perception/d435/rgb/image_raw`
  - `/perception/d435/depth/image_rect`
  - `/perception/d435/camera_info`
- Kept the QCar2 perception, Cartographer, Docker, and architecture notes in grouped sections.

### 2026-05-22 19:08:34 EDT

- Fixed RTAB SLAM copy-paste command so RTAB internal parameters are passed as
  strings:
  - `RGBD/CreateOccupancyGrid:="true"`
  - `Rtabmap/DetectionRate:="1.0"`
- Reason: ROS 2 otherwise parses `true` as a bool, but this RTAB wrapper
  declares those slash-style RTAB parameters as strings.

### 2026-05-22 21:46:48 EDT

- Corrected RTAB smoke-test frame contract:
  - `rtab_map -> rtab_odom -> base_link`
  - `base_link -> base_scan`
  - `base_link -> aligned_camera_optical_frame`
- Updated RTAB odometry/SLAM commands to use `frame_id:=base_link` and
  `publish_tf:=true` for the temporary RTAB mapping run.
- Added `subscribe_scan:=true` so RTAB consumes the 2D LiDAR `/scan`, matching
  the final plan where LiDAR provides wall/track geometry and D435 provides
  visual/semantic evidence.

### 2026-05-22 21:52:25 EDT

- Added RTAB mapping TF launch files, later replaced by the full RTAB mapping
  launch files below.
- Clarified that `rtab_map` should not be added as a static transform; RTAB
  SLAM publishes `rtab_map -> rtab_odom`, and RTAB odometry publishes
  `rtab_odom -> base_link`.

### 2026-05-22 22:26:31 EDT

- Replaced the TF-only RTAB launch files with full mapping launches:
  - `qcar2_perception qcar2_rtabmap_mapping_virtual.launch.py`
  - `qcar2_perception qcar2_rtabmap_mapping_physical.launch.py`
- The full launches start LiDAR, hardware, static sensor TFs, D435 aligned
  source, RGB-D sync, RTAB odometry, and RTAB SLAM.
- They intentionally do not start Cartographer, `qcar2_nodes rgbd`, YOLO,
  semantic mapper, or Nav2.

### 2026-05-22 23:03:54 EDT

- Fixed `semantic_yolo_detector` model path resolution.
- The detector now searches:
  - installed `qcar2_autonomy/share/qcar2_autonomy/models`
  - `/workspaces/isaac_ros-dev/Development/ros2/src/qcar2_autonomy/models`
  - legacy `/workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/models`
- `qcar2_autonomy/setup.py` now installs files from `models/` so launched
  perception nodes do not depend on stale absolute source paths.

### 2026-05-22 22:18:54 EDT

- Updated `qcar2_autonomy manual_drive` so it auto-starts
  `qcar2_nodes nav2_qcar2_converter` only when no converter node is already
  running.
- Reason: RTAB-only sessions do not launch Nav2/Cartographer bringup, but
  `manual_drive` still publishes `/cmd_vel_nav` and the QCar hardware consumes
  `qcar2_motor_speed_cmd`.

### 2026-05-22 18:20:44 EDT

- RTAB-Map local source build reached `rtabmap_sync`, `rtabmap_odom`, and `rtabmap_slam`.
- Patched Humble header-name compatibility issues in local `rtabmap_ros`.
- Disabled RTAB GUI/tools/examples for headless Docker build.

### 2026-05-20

- Added `qcar2_perception` startup flow.
- Added D435 aligned source, semantic YOLO detector, 3D object estimator, semantic mapper, and semantic consistency monitor notes.
- Split semantic visualization topics by meaning.
- Added YOLO dead-zone mask note with visible `NA` overlay.
