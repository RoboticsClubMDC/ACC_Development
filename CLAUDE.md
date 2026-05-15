# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

MDC North AI/Robotics ACC2026 — autonomous taxi mission on a Quanser QCar2 physical robot. Active branch is `Physical_Arturo`.

ROS distro: **Kilted**. The workspace root for all build/run commands is `Development/ros2/`.

## Build & Run

All commands from `Development/ros2/` unless noted.

```bash
# Source ROS (already in .bashrc)
source /opt/ros/kilted/setup.bash

# Build only the packages we touch
colcon build --packages-select qcar2_interfaces qcar2_autonomy
source install/setup.bash
export ROS_DOMAIN_ID=7

# Full rebuild (slower)
colcon build && source install/setup.bash && export ROS_DOMAIN_ID=7
```

**Record a new route map (teach mode):**
```bash
# Terminal 1 – cartographer SLAM must already be running
ros2 launch qcar2_autonomy teach_path_launch.py
# Drive with WASD; map auto-saves to recording_maps/ on every node
# Ctrl+C when done
```

**Run the full autonomy stack:**
```bash
ros2 launch qcar2_autonomy autonomy_planner_launch.py
```

**Dispatch a ride at runtime (while autonomy stack is running):**
```bash
ros2 param set /trip_planner pickup_xy "[x, y]"
ros2 param set /trip_planner dropoff_xy "[x, y]"
```

**Virtual simulation (Docker + QLabs):**
```bash
# QLabs scenario
cd ~/Documents/ACC_Development/docker/virtual_qcar2
sudo docker run --rm -it --network host quanser/virtual-qcar2 bash
# inside: python3 Base_Scenarios_Python/Setup_Competition_Map.py

# ROS Isaac container
cd ~/Documents/ACC_Development/isaac_ros_common
./scripts/run_dev.sh ~/Documents/ACC_Development/Development
# inside: colcon build && source install/setup.bash && export ROS_DOMAIN_ID=7
#         ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py
```

## Architecture

All autonomy code lives in `Development/ros2/src/qcar2_autonomy/autonomy/`. Executables are declared in `setup.py`; launch files in `launch/`.

### Node graph (physical robot autonomy)

```
cartographer → /tf (map→base_link)
                        │
                  nav_to_pose (path_follower)
                  - EKF pose estimation (bicycle model + gyro KF)
                  - Pure pursuit path following at 80 Hz
                  - Subscribes: /cmd_waypoints, /tf, /qcar2_joint, /imu
                  - Publishes: /cmd_vel_raw, /robot_pose, /path_status
                        │
                  lane_keeping
                  - Sidewalk repulsion safety layer
                  - Subscribes: /cmd_vel_raw, /sidewalk_detection/no_go_margin
                  - Publishes: /cmd_vel_nav  ──→  hardware
                        ▲
                  trip_planner
                  - Taxi state machine (IDLE→TO_PICKUP→WAIT→TO_DROPOFF→WAIT→TO_HUB)
                  - Plans legs from the recorded map on disk
                  - Subscribes: /robot_pose, /path_status
                  - Publishes: /cmd_waypoints

Perception (parallel):
  lane_detection      → /lane_detection/lane_selected  (mono8 mask)
  sidewalk_detection  → /sidewalk_detection/no_go_margin  (dilated mono8)
  yolo_detector       → /motion_enable  (stop sign / traffic light gating)
  lane_stanley_node   ← /lane_detection/lane_selected  → steering corrections
```

### Recorded map system

Maps are JSON files in `Development/ros2/src/qcar2_autonomy/recording_maps/`, named `taught_map_YYYYMMDD_HHMMSS.json`. Each file contains a `frame_id` and a list of `{x, y, yaw}` nodes.

`trip_planner` calls `find_latest_recording_map(frame_id='map')` on startup — it picks the **newest file by mtime** whose `frame_id` matches `route_frame`. To switch maps, drop a newer file in that directory.

`recorded_map_utils.py` provides all map I/O and path math:
- `filter_recorded_nodes` — down-samples by min spacing and min yaw change
- `densify_polyline` — interpolates control nodes to 3 cm waypoint spacing
- `build_directed_dense_recorded_segment` — extracts a forward-only slice (with optional wrap-around for loop maps)
- `closest_recorded_node_index` — heading-aware nearest-node search

### Coordinate frames

Goal coordinates (`hub_xy`, `pickup_xy`, `dropoff_xy`) are expressed in **SDCS PNG/reference coordinates** by default (`goals_are_sdcs_frame=True`). `trip_planner._goal_to_route_frame()` converts them to the ROS `map` frame using:
- `sdcs_to_map_rotation_deg = -79.63`
- `sdcs_to_map_translation = [-0.087, -0.054]`
- `sdcs_to_map_scale = 1.022`

These constants were fitted from manually sampled SDCS node positions. If the physical map layout changes they need to be re-fitted.

### Map quality requirements

For the route to work correctly, the recorded map must be a **single clean lap**:
- Start at the hub (hub ≈ index 0)
- Drive the full course once in the direction of travel
- End within **0.75 m** of the start (triggers `recorded_loop=True`)

A multi-lap or out-of-order recording will cause `trip_planner` to generate near-full-route paths for short legs, or return empty paths when a backward segment is requested with `one_way_route=True`.

### key parameters (trip_planner)

| Parameter | Default | Effect |
|---|---|---|
| `one_way_route` | `True` | Enforce forward-only index traversal |
| `allow_route_wrap` | `True` | Allow wrapping end→start on loop maps |
| `route_wrap_gap_tolerance_m` | `1.0` | Max gap to still treat map as wrappable |
| `heading_aware_start` | `True` | Score start node by heading match |
| `goal_on_route_tolerance_m` | `0.12` | Warn if goal is off the recorded path |
