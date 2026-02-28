ACC_Development
Repository for Quanser ACC competition 2026

This guide covers the commands and procedure for running the Visual Odometry (VO) redundancy system on the virtual QCar 2. The VO system independently verifies Cartographer (IMU + encoder odometry) using camera-based visual odometry and flags disagreements in real time.

[VO_Technical_Reference.pdf](https://github.com/user-attachments/files/25639831/VO_Technical_Reference.pdf)

- [Overview](#overview)
- [File Locations](#file-locations)
- [Setup](#setup)
- [Starting the VO System](#starting-the-vo-system)
- [Monitoring the Output](#monitoring-the-output)
- [A/B Comparison (Calibrated vs Default Intrinsics)](#ab-comparison-calibrated-vs-default-intrinsics)
- [Expected Topics](#expected-topics)
- [Notes](#notes)
- [Demo Videos](#demo-videos)

## Overview

The system has three layers.

The **Visual Odometry Engine** (`visual_odometry.py`) handles ORB feature extraction, frame-to-frame matching, 3D back-projection using depth, and RANSAC + SVD Procrustes motion estimation. It produces an independent pose estimate from camera images alone.

The **Redundancy Monitor** (`vo_node.py`) runs a sliding-window comparison of the VO pose against the Cartographer pose. It computes the vector residual between the two, applies a weighted trust score based on confidence, inlier count, and feature spread, gates out turns where VO is unreliable, and classifies each evaluation cycle as `agree`, `vo_suspect`, or `odom_suspect`.

The **Supervisor** (`vo_supervisor.py`) converts the monitor output into operational modes that downstream controllers can act on: `NORMAL`, `VO_UNTRUSTED`, `ODOM_WARNING`, `ODOM_FAULT`, `INITIALIZING`, and `DEGRADED`.

## File Locations[Uploading VO_Technical_Reference.pdf…]()


All VO files are located in `Development/ros2/src/qcar2_autonomy/autonomy/`:

| File | Purpose |
|------|---------|
| `visual_odometry.py` | VO engine (ORB + RANSAC + SVD) |
| `vo_node.py` | Redundancy monitor node |
| `vo_supervisor.py` | Supervisor node |
| `vo_capture.py` | Fault status capture utility |
| `vo_live_plot.py` | Matplotlib live plotter |
| `qlabs_vo_baseline.yaml` | Validated QLabs parameter file |

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

6. Launch the Cartographer SLAM and navigation stack. The VO system depends on this for the camera topics and the odometry TF.

    ```bash
    ros2 launch qcar2_nodes qcar2_slam_and_nav_bringup_virtual_launch.py
    ```

7. Open a new terminal and attach it to the Development Container.

    ```bash
    cd /home/$USER/Documents/ACC_Development/isaac_ros_common
    ./scripts/run_dev.sh /home/$USER/Documents/ACC_Development/Development
    ```

8. Start the autonomy planner so the car is driving around the track.

    ```bash
    . install/setup.bash && export ROS_DOMAIN_ID=67
    ros2 launch qcar2_autonomy autonomy_planner_launch.py
    ```

    The QCar should now be driving autonomously. You can also set a `2D Goal Pose` in `rviz2` as described in the [Nav 2 Guide](./Virtual_Running_Nav2_Guide.md).

## Starting the VO System

Each of the following terminals needs to be attached to the Development Container and sourced before running the command.

```bash
cd /home/$USER/Documents/ACC_Development/isaac_ros_common
./scripts/run_dev.sh /home/$USER/Documents/ACC_Development/Development
. install/setup.bash && export ROS_DOMAIN_ID=67
```

9. Open a new terminal and start the VO node with the calibrated QLabs parameters.

    ```bash
    ros2 run qcar2_autonomy vo_node --ros-args \
      --params-file /workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/autonomy/qlabs_vo_baseline.yaml
    ```

    Verify the startup banner shows `fx=161.0` and `turn_gate=5.0deg`. If it shows `fx=483.7` instead, the YAML file path is wrong.

10. Open a new terminal and start the supervisor.

    ```bash
    ros2 run qcar2_autonomy vo_supervisor
    ```

11. Open a new terminal and echo the supervisor mode to see the system working.

    ```bash
    ros2 topic echo /vo/supervisor/mode
    ```

    As the car drives you should see `NORMAL` on straight segments, `VO_UNTRUSTED` during turns, brief `INITIALIZING` after re-anchors, and occasional `ODOM_WARNING` at turn-to-straight transitions.

## Monitoring the Output

There are two ways to watch the numeric signals in real time.

12. Terminal echo (always works, no extra dependencies):

    ```bash
    ros2 topic echo /vo/supervisor/trust_level
    ```

    This shows the trust score as a float: 1.0 on straights, dips during turns, craters to 0.1 on a confirmed fault.

13. Live matplotlib plot (requires `python3-tk` in the container):

    ```bash
    sudo apt install -y python3-tk
    python3 /workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/autonomy/vo_live_plot.py
    ```

    This opens three stacked panels showing the residual, the VO weight, and the trust level in real time, with the current supervisor mode displayed at the bottom.

    Note that `rqt_plot` is not available in the Isaac ROS container, which is why we use the matplotlib plotter instead.

## A/B Comparison (Calibrated vs Default Intrinsics)

To demonstrate the impact of proper camera calibration, you can run the VO node two different ways.

The **calibrated run** uses the `qlabs_vo_baseline.yaml` file (step 9 above), which sets `fx=161` and `turn_gate=5°`. This produces clean state separation: `NORMAL` on straights, `VO_UNTRUSTED` on turns, with rare and brief `ODOM_WARNING` events at transitions only.

The **default intrinsics run** launches the VO node without the YAML file:

```bash
ros2 run qcar2_autonomy vo_node
```

This uses the code defaults of `fx=483.7` and `turn_gate=8°`. Because the QLabs virtual camera actually has `fx=161`, the default value compresses all VO translations by roughly 3x, causing the monitor to see a persistent disagreement on straight segments and generate frequent false `ODOM_WARNING` and `ODOM_FAULT` events.

## Expected Topics

After starting the VO node and supervisor, you can verify all topics are present:

```bash
ros2 topic list | grep "^/vo/"
```

This should return:

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

## Notes

All terminals must use the same `ROS_DOMAIN_ID` or they will not see each other's topics.

`ODOM_FAULT` is only triggered after 3 consecutive confirmed detections, so a single transient blip will not halt the car.

`VO_UNTRUSTED` during turns is expected and correct behavior. The SVD rigid-body model does not produce reliable translations when the camera is rotating through a scene with depth variation, so the system correctly abstains.

`ExternalShutdownException` during `Ctrl+C` is harmless and can be ignored.

The live plot (`vo_live_plot.py`) is a pure subscriber and does not change any VO computation. Heavy GUI rendering may add system overhead during screen recording.

## Demo Videos

Two recordings are included in the repository:

`Screencast_from_2026-02-27_21-04-19.webm` shows the calibrated run with `fx=161` and `turn_gate=5°`.

`Screencast_from_2026-02-27_21-57-41.webm` shows the default intrinsics run with `fx=483.7` and `turn_gate=8°`.

Comparing the two demonstrates the impact of proper camera intrinsics on fault detection accuracy.
