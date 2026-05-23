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
