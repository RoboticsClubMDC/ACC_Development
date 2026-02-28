# ACC_Development
Repository for the Quanser ACC Competition 2026.

------------------------------------------------------------------------

# Running Visual Odometry Redundancy on the Virtual QCar 2

This guide covers the commands and procedure for running the Visual
Odometry (VO) redundancy system on the virtual QCar 2.

The VO system independently verifies Cartographer (IMU + encoder
odometry) using camera-based visual odometry and flags disagreements in
real time.

------------------------------------------------------------------------

## Overview

The system has three layers:

### 1️⃣ Visual Odometry Engine (`visual_odometry.py`)

-   ORB feature extraction
-   Frame-to-frame matching
-   3D back-projection using depth
-   RANSAC + SVD Procrustes motion estimation

### 2️⃣ Redundancy Monitor (`vo_node.py`)

-   Sliding window comparison of VO vs Cartographer
-   Vector residual ρ comparison
-   Weighted trust score (confidence + inliers + spread)
-   Turn gating
-   Classification:
    -   `agree`
    -   `vo_suspect`
    -   `odom_suspect`

### 3️⃣ Supervisor (`vo_supervisor.py`)

Converts monitor output into operational modes:
-   `NORMAL`
-   `VO_UNTRUSTED`
-   `ODOM_WARNING`
-   `ODOM_FAULT`
-   `INITIALIZING`
-   `DEGRADED`

------------------------------------------------------------------------

## File Locations

All VO files are located in:

```
Development/ros2/src/qcar2_autonomy/autonomy/
```

-   `visual_odometry.py` — VO engine (ORB + RANSAC + SVD)
-   `vo_node.py` — Redundancy monitor node (v5)
-   `vo_supervisor.py` — Supervisor node (Part 3)
-   `vo_capture.py` — Fault status capture utility
-   `vo_terminal_dashboard.py` — Optional terminal dashboard
-   `vo_live_plot.py` — Matplotlib live plotter
-   `qlabs_vo_baseline.yaml` — Validated QLabs parameter file

------------------------------------------------------------------------

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

------------------------------------------------------------------------

## Running the System (Demo Setup)

For each new terminal below, attach to the Development Container first:

```bash
cd /home/$USER/Documents/ACC_Development/isaac_ros_common
./scripts/run_dev.sh /home/$USER/Documents/ACC_Development/Development
```

Then source and set the domain ID:

```bash
. install/setup.bash && export ROS_DOMAIN_ID=67
```

------------------------------------------------------------------------

### Terminal A — VO Node (Calibrated)

```bash
ros2 run qcar2_autonomy vo_node --ros-args \
  --params-file /workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/autonomy/qlabs_vo_baseline.yaml
```

Verify the startup banner shows:

```
intrinsics: fx=161.0  fy=161.0  cx=321.2  cy=238.5
turn_gate=5.0deg
```

If it shows `fx=483.7` instead, the YAML path is incorrect.

------------------------------------------------------------------------

### Terminal B — Supervisor

```bash
ros2 run qcar2_autonomy vo_supervisor
```

You should see `VO Supervisor started (degraded_timeout=30s)`.

------------------------------------------------------------------------

### Terminal C — Mode Output

```bash
ros2 topic echo /vo/supervisor/mode
```

Expected behavior as the car drives:

-   `NORMAL` → Straight segments, VO and odom agree
-   `VO_UNTRUSTED` → Turns, VO abstaining (model assumption violated)
-   `ODOM_WARNING` → Pending odom disagreement (not yet confirmed)
-   `ODOM_FAULT` → Confirmed odom fault (3+ consecutive, stop advised)
-   `INITIALIZING` → Warmup after re-anchor

------------------------------------------------------------------------

### Terminal D — Numeric Monitoring (Optional)

**Option 1: Terminal echo** (zero dependencies, always works):

```bash
ros2 topic echo /vo/supervisor/trust_level
```

Shows the trust score (1.0 on straights, dips during turns, craters on fault).

**Option 2: Live matplotlib plot** (requires `python3-tk` installed in container):

```bash
sudo apt install -y python3-tk
python3 /workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/autonomy/vo_live_plot.py
```

Displays three real-time panels: residual ρ (red), VO weight (teal), and trust level (yellow) with the current supervisor mode at the bottom.

> **Note:** `rqt_plot` is not available in the Isaac ROS container. Use one of the options above instead.

------------------------------------------------------------------------

## A/B Comparison

### Calibrated Run (Recommended)

Uses the `qlabs_vo_baseline.yaml` parameter file:

-   `fx=161` — Corrected focal length for QLabs virtual camera
-   `turn_gate=5°` — Tighter turn detection

More stable straight detection, fewer false odom faults.

------------------------------------------------------------------------

### Default Intrinsics Run

```bash
ros2 run qcar2_autonomy vo_node
```

Uses code defaults:

-   `fx=483.7` — Intel D435 spec sheet value (wrong for QLabs)
-   `turn_gate=8°` — Wider turn gate

Demonstrates 3x translation compression and increased false `ODOM_WARNING` and `ODOM_FAULT` events because VO consistently undermeasures displacement.

------------------------------------------------------------------------

## Expected Topics

```bash
ros2 topic list | grep "^/vo/"
```

Should show:

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

------------------------------------------------------------------------

## Notes

-   All terminals must use the same `ROS_DOMAIN_ID`.
-   `ODOM_FAULT` is only triggered after 3 consecutive confirmed detections.
-   `VO_UNTRUSTED` during turns is expected behavior, not failure.
-   `ExternalShutdownException` during Ctrl+C is harmless.
-   The live plot (`vo_live_plot.py`) is a pure subscriber and does not affect VO computation. Heavy GUI rendering may add system overhead during screen recording.

------------------------------------------------------------------------

## Demo Videos

Included in repository:

-   **Calibrated Run** (`Screencast_from_2026-02-27_21-04-19.webm`) — fx=161, turn_gate=5°
-   **Default Intrinsics Run** (`Screencast_from_2026-02-27_21-57-41.webm`) — fx=483.7, turn_gate=8°

These demonstrate the impact of proper camera calibration on fault detection accuracy.
