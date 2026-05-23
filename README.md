# ACC_Development — `Gabriel` branch

Quanser **ACC (Autonomous Connected Cars) competition 2026** codebase, on the `Gabriel` branch. The work on this branch is building a **Visual Odometry (VO) redundancy layer** for the QCar2 — an independent camera-based pose estimate that verifies the QCar's primary Cartographer localization and flags disagreements.

If you've just landed on this branch and you're trying to figure out what's going on, this README is your map. It won't dive into algorithm internals — for that, follow the links to the logs.

---

## What this work is, in one paragraph

The QCar2 is normally localized by **Cartographer**, which fuses lidar + IMU + wheel odometry. That's a single sensor stack — if any of those degrade together (reflective surfaces, IMU drift, wheel slip), Cartographer can be wrong without saying so. We add **VO** as an independent second opinion from the camera, whose failure modes (low texture, motion blur, depth noise) don't overlap with the lidar/IMU/wheel ones. Our VO reports **honest per-frame uncertainty**, so an EKF can blend the two estimates and automatically down-weight VO when it's unreliable. That's the whole pitch: redundancy + honest uncertainty + automatic blending.

---

## Where to go for what

| If you want to… | Open this |
|---|---|
| Run the system step-by-step (canonical procedure, on-car and off-car) | [`Easy_Start.txt`](Easy_Start.txt) |
| Read every dated decision, result, and reversal | [`Development/ros2/src/qcar2_autonomy/VO_CHANGELOG.md`](Development/ros2/src/qcar2_autonomy/VO_CHANGELOG.md) (newest at the top) |
| Read the running narrative of each work session | [`VO_Conversation_Log.txt`](VO_Conversation_Log.txt) |
| See the project rules & guardrails (for Claude or new collaborators) | [`CLAUDE.md`](CLAUDE.md) |
| Browse the VO source code | [`Development/ros2/src/qcar2_autonomy/autonomy/`](Development/ros2/src/qcar2_autonomy/autonomy/) |
| Browse launch files | [`Development/ros2/src/qcar2_nodes/launch/`](Development/ros2/src/qcar2_nodes/launch/) |
| See result plots (trajectories, EKF, yaw comparison, 2D map) | [`VO_Images/`](VO_Images/) |
| Physical camera intrinsics (real D435) | [`Camera Intrinsics Post.txt`](Camera%20Intrinsics%20Post.txt) |
| Virtual camera intrinsics (QLabs D435) | [`Virtual_Stage_ROS_FAQ.md`](Virtual_Stage_ROS_FAQ.md) |
| Hardware extrinsics (camera-to-body) | [`user_manual_system_hardware.pdf`](user_manual_system_hardware.pdf) |

---

## The VO source files

All in [`Development/ros2/src/qcar2_autonomy/autonomy/`](Development/ros2/src/qcar2_autonomy/autonomy/):

| File | What it does |
|---|---|
| `visual_odometry.py` | The VO engine — features, depth backprojection, RANSAC, motion estimation. Selectable frontend (ORB/KLT) and estimator (SVD/PnP). |
| `vo_node.py` | The ROS node that runs the engine, publishes `/vo/odometry` with honest covariance, and compares against Cartographer in real time. |
| `vo_supervisor.py` | Turns the monitor output into operational modes a controller can act on. |
| `vo_odom_tf_relay.py` | Re-frames `/vo/odometry` for RTAB-Map consumption (clean odom frame + TF). |
| `vo_terminal_dashboard.py` | Terminal dashboard (entry point `vo_dashboard`). |
| `vo_image_overlay.py` | Image/feature overlay viewer (entry point `vo_overlay`). |
| `qcar2_camera_bridge.py` | Camera bridge — publishes RGB, depth, `camera_info`, and the static `base_link → camera` TF. |
| `nav_to_pose.py` | Nav2 goal sender; also contains the `QcarEKF` reference (bicycle-model EKF). |
| `manual_drive.py` | WASD keyboard manual drive (used for physical VO testing). |
| `*_old.py` | Frozen reference copies — don't edit, don't delete; used for diffing against current code. |

---

## High-level milestones (where we are)

- **VO engine running** on both physical QCar and QLabs virtual, publishing pose + honest per-frame covariance.
- **Selectable knobs added** (all default-safe): `vo_frontend` ∈ {orb, klt}, `vo_estimator` ∈ {svd, pnp}, `feature_grid` (spread features across the image).
- **2×2 frontend × estimator campaign** completed: **KLT+PnP** wins, with roughly 3× lower invalid-frame rate than the default ORB+SVD.
- **RTAB-Map (SLAM) integration**: A/B/C maps built on the QCar (pure-visual / our VO / Cartographer-fused); per-config maps built off-car for KLT+SVD, ORB+PnP, KLT+PnP.
- **Off-car workflow**: any Ubuntu box with a `humble-rtabmap` Docker image can replay the recorded source bag (`vslam_test12`) — no QCar required.
- **Offline EKF demo**: VO + Cartographer fused using `nav_to_pose.py`'s `QcarEKF`, showing honest-covariance blending in action (the `ekf_demo_vo_plus_cartographer.png` plot in `VO_Images/`).
- **Every decision (including reversals) logged** in `VO_CHANGELOG.md` so the reasoning trail is reproducible.

---

For step-by-step run instructions, the canonical reference is [`Easy_Start.txt`](Easy_Start.txt). For *why* anything is the way it is, read [`VO_CHANGELOG.md`](Development/ros2/src/qcar2_autonomy/VO_CHANGELOG.md) — newest entries are at the top.
