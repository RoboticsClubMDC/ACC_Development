# ACC_Development
Repository for Quanser ACC competition 2026

[Virtual_Running_VO_Redundancy_Guide.md](https://github.com/user-attachments/files/25638398/Virtual_Running_VO_Redundancy_Guide.md)
# Running Visual Odometry Redundancy on the Virtual QCar 2 <!-- omit in toc -->

This guide covers the commands and procedure for running the Visual Odometry (VO) redundancy system on the virtual QCar 2. The VO system uses the camera to independently verify the position estimates from Cartographer (IMU + encoder odometry) and flags disagreements in real time.

- [Overview](#overview)
- [File Locations](#file-locations)
- [Setup](#setup)
- [Starting the VO Redundancy System](#starting-the-vo-redundancy-system)
- [Monitoring the System](#monitoring-the-system)
- [A/B Comparison: Calibrated vs Default Intrinsics](#ab-comparison-calibrated-vs-default-intrinsics)
- [Demo Videos](#demo-videos)

## Overview

The system has three layers:

- **Visual Odometry Engine** (`visual_odometry.py`) — Extracts ORB features from the camera, matches them frame-to-frame, back-projects into 3D using depth, and estimates motion via RANSAC + SVD Procrustes.
- **Redundancy Monitor** (`vo_node.py`) — Compares VO displacement against Cartographer displacement over a sliding window. Classifies each cycle as `agree`, `vo_suspect`, or `odom_suspect` using a weighted trust score and a four-gate decision tree.
- **Supervisor** (`vo_supervisor.py`) — Converts the monitor output into navigation modes (`NORMAL`, `VO_UNTRUSTED`, `ODOM_WARNING`, `ODOM_FAULT`) with a stop advisory and continuous trust level.

## File Locations

All VO files are located in:

```
Development/ros2/src/qcar2_autonomy/autonomy/
```

- `visual_odometry.py` — VO engine (ORB + RANSAC + SVD)
- `vo_node.py` — Redundancy monitor node (v5)
- `vo_supervisor.py` — Supervisor node (Part 3)
- `vo_capture.py` — Fault status capture utility
- `vo_terminal_dashboard.py` — Optional terminal dashboard
- `vo_live_plot.py` — Optional matplotlib live plotter
- `qlabs_vo_baseline.yaml` — Validated QLabs parameter file

## Setup

Make sure that you have gone through the entire [ACC Software Setup Instructions](./Virtual_ROS_Software_Setup.md) and the [Running Nav 2 Guide](./Virtual_Running_Nav2_Guide.md) before continuing with this guide.

1. Open QLabs to the Plane World.

2. Open a new terminal (`CTRL + ALT + T`) and run the Quanser Virtual Environment Container.

    ```bash
    cd /home/$USER/Documents/ACC_Development/docker/virtual_qcar2
    sudo docker run --rm -it --network host --name virtual-qcar2 quanser/virtual-qcar2 bash
    ```

3. Spawn the QCar in the competition environment by running the following commands in the Quanser Virtual Environment Container.

    ```bash
    cd /home/qcar2_scripts/python
    python3 Base_Scenarios_Python/Setup_Competition_Map.py
    ```

4. Open a new terminal and start the Development Container.

    ```bash
    cd /home/$USER/Documents/ACC_Development/isaac_ros_common
    ./scripts/run_dev.sh /home/$USER/Documents/ACC_Development/Development
    ```

5. Build and source the nodes in ROS2.

    ```bash
    colcon build && . install/setup.bash && export ROS_DOMAIN_ID=67
    ```

6. Launch the Cartographer SLAM and navigation stack. The VO system depends on this for the camera topics and the odometry TF that it compares against.

    ```bash
    ros2 launch qcar2_nodes qcar2_slam_and_nav_bringup_virtual_launch.py
    ```

7. Start the autonomous navigation. Open a new terminal, attach to the Development Container, and run the autonomy planner.

    ```bash
    cd /home/$USER/Documents/ACC_Development/isaac_ros_common
    ./scripts/run_dev.sh /home/$USER/Documents/ACC_Development/Development
    ```

    ```bash
    . install/setup.bash && export ROS_DOMAIN_ID=67
    ros2 launch qcar2_autonomy autonomy_planner_launch.py
    ```

    The QCar should now be driving autonomously around the track. You can alternatively set a `2D Goal Pose` in `rviz2` as described in the [Nav 2 Guide](./Virtual_Running_Nav2_Guide.md).

## Starting the VO Redundancy System

8. Open a new terminal and attach to the Development Container.

    ```bash
    cd /home/$USER/Documents/ACC_Development/isaac_ros_common
    ./scripts/run_dev.sh /home/$USER/Documents/ACC_Development/Development
    ```

9. Start the VO node with the calibrated QLabs parameters.

    ```bash
    . install/setup.bash && export ROS_DOMAIN_ID=67
    ros2 run qcar2_autonomy vo_node --ros-args \
        --params-file /workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/autonomy/qlabs_vo_baseline.yaml
    ```

    You should see a startup banner. Verify that it shows `fx=161.0` and `turn_gate=5.0deg`:

    ```
    VO NODE v5 (Part 2: vector residual + weighted trust)
      depth: shift=0 unit=0.0010 scale=15.7  -> SCALE mode
      ...
      intrinsics: fx=161.0  fy=161.0  cx=321.2  cy=238.5
    ```

    If it shows `fx=483.7` instead, the YAML path is incorrect.

10. Open another terminal, attach to the Development Container, and start the supervisor.

    ```bash
    cd /home/$USER/Documents/ACC_Development/isaac_ros_common
    ./scripts/run_dev.sh /home/$USER/Documents/ACC_Development/Development
    ```

    ```bash
    . install/setup.bash && export ROS_DOMAIN_ID=67
    ros2 run qcar2_autonomy vo_supervisor
    ```

    You should see `VO Supervisor started (degraded_timeout=30s)`.

## Monitoring the System

11. Open another terminal, attach to the Development Container, and echo the supervisor mode to see the system's decisions in real time.

    ```bash
    cd /home/$USER/Documents/ACC_Development/isaac_ros_common
    ./scripts/run_dev.sh /home/$USER/Documents/ACC_Development/Development
    ```

    ```bash
    . install/setup.bash && export ROS_DOMAIN_ID=67
    ros2 topic echo /vo/supervisor/mode
    ```

    You will see output like this as the car drives:

    ```
    data: NORMAL          ← straight, VO and odom agree
    data: VO_UNTRUSTED    ← turn, VO abstaining (model assumption violated)
    data: INITIALIZING    ← warmup after re-anchor
    data: ODOM_WARNING    ← pending odom disagreement (not yet confirmed)
    data: ODOM_FAULT      ← confirmed odom fault (3+ consecutive, stop advised)
    ```

12. You can also verify all topics are publishing by running:

    ```bash
    ros2 topic list | grep "^/vo/"
    ```

    You should see:

    ```
    /vo/delta_trans
    /vo/fault_status
    /vo/healthy
    /vo/state_id
    /vo/supervisor/mode
    /vo/supervisor/stop_advised
    /vo/supervisor/trust_level
    /vo/vo_weight
    ```

## A/B Comparison: Calibrated vs Default Intrinsics

To demonstrate the impact of proper camera calibration, you can run the VO node **without** the parameter file. This uses the code defaults (`fx=483.7` from the Intel D435 spec sheet, `turn_gate=8°`, `features=800`).

```bash
. install/setup.bash && export ROS_DOMAIN_ID=67
ros2 run qcar2_autonomy vo_node
```

The banner will show `fx=483.7` and `turn_gate=8.0deg`. With these settings, VO compresses translations by approximately 3x because the QLabs virtual camera has a much wider FOV than the physical D435 spec suggests. This causes more frequent false `ODOM_WARNING` and `ODOM_FAULT` events on straight segments compared to the calibrated run.

## Demo Videos

Two recordings are included in this repository showing the system running on the QLabs virtual track.

**Calibrated Run** (`Screencast_from_2026-02-27_21-04-19.webm`): VO node running with `qlabs_vo_baseline.yaml` (`fx=161`, `turn_gate=5°`). Shows clean state separation — `NORMAL` on straights, `VO_UNTRUSTED` on turns, brief `INITIALIZING` after re-anchors, and rare transition-zone `ODOM_WARNING` events.

**Default Intrinsics Run** (`Screencast_from_2026-02-27_21-57-41.webm`): VO node running without the parameter file (`fx=483.7`, `turn_gate=8°`). Shows more frequent false fault detections due to 3x translation compression from incorrect intrinsics, demonstrating why proper camera calibration matters.
