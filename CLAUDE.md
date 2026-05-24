# CLAUDE.md

Handoff briefing for the next Claude Code session on the **Quanser ACC 2026 QCar 2** workspace. Read this top-to-bottom before making any changes — there are several non-obvious decisions and a few "do not recreate" items that aren't visible from the code alone.

`Easy_Start.md` is the operational runbook (copy-paste commands). `CLAUDE.md` (this file) is the architectural + historical briefing. Keep both in sync when you change anything.

---

## 0. What this project is

- Quanser ACC 2026 competition entry. Hardware target: **physical QCar 2** (Jetson AGX Orin, RPLidar A2, Intel D435 RGB-D, four CSI cameras, IMU, Ackermann steering). Development target: **virtual QCar 2** via QLabs Docker container.
- ROS 2 Humble, all source under `Development/ros2/src/`. The container is the **only** supported build/run environment.
- The competition has **three concrete tasks**:
  1. **Stop for traffic lights / stop signs** (object detection + brake)
  2. **Maintain allowed lane / driving rules** (geometric path follower + lane detection + safe-distance)
  3. **Complete the taxi trip**: HUB → pickup → dropoff → HUB → repeat
- Repeatability across **many trips in one session** is the dominant requirement. By trip 6, accumulated localization error must still be small enough to stop at a passenger pickup. That's why we built so much infrastructure around EKF/AMCL accuracy.

---

## 1. Where work happens

- All ROS 2 source: [`Development/ros2/src/`](Development/ros2/src/). Inside the dev container this is mounted at `/workspaces/isaac_ros-dev/Development/ros2`.
- `colcon build` runs from `Development/ros2` (NOT the repo root).
- The active packages (only **4**, not 18 — the old RTAB-Map vendored 14 sub-packages got deleted):
  - `qcar2_nodes` (C++ hardware) — lidar, qcar2_hardware, rgbd, csi, nav2_qcar_command_convert, fixed_lidar_frame[_virtual], + launches
  - `qcar2_autonomy` (Python) — path_follower, pose_estimator, ekf_fusor, controller_watchdog, trip_planner, manual_drive, lane/sidewalk detection, LCroadmap_alignment_node, + helpers
  - `qcar2_perception` (Python) — d435_aligned_source, semantic_yolo_detector, object_3d_estimator, semantic_landmark_mapper, semantic_consistency_monitor
  - `qcar2_interfaces` (msg defs only)
- Generated directories (NEVER edit): `build/`, `install/`, `log/`, `Development/ros2/build/`, `Development/ros2/install/`, `Development/ros2/log/`.
- Permanent Python/apt deps for the dev container go in [`docker/development_docker/quanser_dev_docker_files/Dockerfile.quanser`](docker/development_docker/quanser_dev_docker_files/Dockerfile.quanser), NOT ad-hoc `pip install` in a running container.

---

## 2. Starting a session

Follow `Easy_Start.md` § "Normal Startup Order":

1. Launch QLabs on the host: `docker/virtual_qcar2`.
2. Enter the Isaac dev container: `./isaac_ros_common/scripts/run_dev.sh ~/Documents/GitHub/ACC_Development/Development`.
3. In **every** new ROS terminal:
   ```bash
   cd /workspaces/isaac_ros-dev/Development/ros2
   source /opt/ros/humble/setup.bash
   source /workspace/cartographer_ws/install/setup.bash
   source install/setup.bash
   export ROS_DOMAIN_ID=69
   ```
   The Cartographer overlay is required because `pcl_conversions` / `pcl_ros` come from there.

---

## 3. Build commands

```bash
# Active packages only (RTAB-Map source has been DELETED — no need to skip it):
colcon build --symlink-install --packages-select qcar2_autonomy qcar2_nodes qcar2_interfaces qcar2_perception
source install/setup.bash && export ROS_DOMAIN_ID=69
```

Full workspace `colcon build` is fine too — there are no more heavy subpackages.

Re-source `install/setup.bash` and re-export `ROS_DOMAIN_ID=69` after every build.

---

## 4. Current architecture (the runtime stack)

```
QLabs / physical sensors
   │
   ▼
qcar2_hardware + lidar + csi + rgbd    ← /qcar2_joint /qcar2_imu /scan /qcar2_csi
   │
   ▼
pose_estimator (predict-only)          ← publishes /odom and odom→base_link TF
   │
   ▼
Cartographer (build phase)             ← consumes /odom + /scan + /qcar2_imu
   │                                       publishes map→odom TF + /map
   ▼
ekf_fusor (full EKF with correction)   ← subscribes to TF map→base_link OR /amcl_pose
   │                                       publishes /qcar2_pose_fused (PoseWithCovarianceStamped)
   ▼                                       + /qcar2_ekf/{p_diag, k_diag, innovation, innovation_mahalanobis, health, mode}
path_follower                          ← reads /qcar2_pose_fused; modes: idle | manual | autonomous
   │                                     publishes /cmd_vel_nav + /nav/* diagnostics
   ▼
nav2_qcar2_converter                    ← /cmd_vel_nav → /qcar2_motor_speed_cmd
   │
   ▼
qcar2_hardware                          ← /qcar2_motor_speed_cmd → motors
```

Plus passive observers (one Foxglove-bound topic each):
- `controller_watchdog` → `/nav/controller_health` (HEALTHY / SATURATED / WIGGLING / LATE_REACTION / WARMING_UP)
- `cartographer_occupancy_grid_node` → `/map`

Trip mission state machine `trip_planner` is the top layer above path_follower; it sends `/cmd_waypoints` between HUB / pickup / dropoff. Today it pre-empts path_follower's default `node_values`; the auto-switch into `autonomous` mode is automatic when `/cmd_waypoints` arrives.

### What's bundled by which launch

| Launch | Spawns |
|---|---|
| `qcar2_cartographer_virtual_launch.py` | qcar2_virtual (base hardware) + pose_estimator + **ekf_fusor** + cartographer + occupancy_grid + nav2_qcar2_converter + fixed_lidar_frame_virtual TF |
| `qcar2_cartographer_launch.py` (physical) | same as above but with physical base + `fixed_lidar_frame` (with `setRPY(0,0,-π)` for the 180° LiDAR mount) |
| `qcar2_amcl_localization_virtual_launch.py` | qcar2_virtual + pose_estimator + AMCL + map_server + lifecycle_manager + fixed_lidar_frame_virtual |
| `qcar2_amcl_localization_launch.py` | same as above but physical |
| `foxglove_bridge_launch.py` | foxglove_bridge + **controller_watchdog** |
| `autonomy_planner_launch.py` | path_follower + trip_planner + lane_detection + lane_stanley_node + bev_csi_node + sidewalk_detection + bev_csi_seg |

Minimal stack for autonomous driving in QLabs (3 terminals + QLabs):
```bash
ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py    # SLAM + base + pose stack
ros2 launch qcar2_nodes foxglove_bridge_launch.py               # viz + watchdog
ros2 run qcar2_autonomy path_follower                            # idle by default
```

---

## 5. Critical design decisions (do not undo without reading)

### 5.1 The EKF refactor

The old architecture had `pose_estimator.py` as a single predict-only "complementary filter" with no measurement update — and the original `nav_to_pose.py` had a private EKF buried inside. We extracted, restructured, and added a real measurement update:

- `pose_estimator.py` — **predict-only** EKF, owns `odom → base_link` TF and `/odom` topic. Inputs: encoder (`/qcar2_joint`), IMU (`/qcar2_imu`), steering (`/cmd_vel_nav`, `/qcar2_motor_speed_cmd`). Bicycle model with `gyro_weight=0.65` blending IMU gyro vs steering-derived ω.
- `qcar2_autonomy/autonomy/estimation/filters.py` — extracted `QcarEKF` + `GyroKF` classes (originally inside `nav_to_pose.py`). Importable, testable in isolation.
- `ekf_fusor.py` — **standalone full EKF with measurement update**. Uses the extracted filters. Subscribes to raw sensors for prediction, then either TF `map → base_link` (Cartographer mode) or `/amcl_pose` (runtime mode) for correction. Mahalanobis outlier gate at χ²_3 = 11.345 (99% confidence). Publishes `/qcar2_pose_fused` (PoseWithCovarianceStamped) + 8 diagnostic topics.
- `nav_to_pose.py` — embedded EKF removed; now consumes `/qcar2_pose_fused`. Falls back to raw `map → base_link` TF if fused pose isn't available yet.

**Hardware constants** (per QCar 2 User Manual - System Hardware v1.0):
- `ENCODER_TICKS_PER_REV = 720 × 4` (quadrature)
- `GEAR_RATIO = (13 × 19) / (70 × 37)` ← **the denominator was 30 in original code (23% wrong)**
- `WHEEL_RADIUS = 0.033 m`
- `WHEELBASE = 0.256 m` ← **was 0.257 in original code**
- Steering limit `0.52 rad = ±30°` ← **was 0.60 (impossible for hardware)**

The IMU `/qcar2_imu.angular_velocity.z` is in **rad/s per ROS standard**, NOT deg/s. The original `nav_to_pose.py` had a `* π/180` deg→rad conversion in the gyro damping term, which made the D gain 57× weaker than intended. **Removed.**

### 5.2 Unified `path_follower` control modes

`path_follower` is now the **single owner** of `/cmd_vel_nav`. No double-publish fights. The `control_mode` parameter has three values:

| Mode | Behavior | Trigger |
|---|---|---|
| `idle` (default) | Publishes nothing. Bus is free. | Initial state |
| `manual` | WASD keystrokes from path_follower's terminal publish `/cmd_vel_nav` directly | `ros2 param set /path_follower control_mode "manual"` |
| `autonomous` | Pure pursuit drives waypoints | `/cmd_waypoints` arrives, `node_values` param changes, or manual param set |

The `manual_drive` standalone node still exists for backward compat, but the preferred way to drive by hand is `control_mode=manual`. Running both at once = double-publish fight; don't.

### 5.3 PD gain tuning (BO + Option-B middle of robust cluster)

Pure pursuit + gyro damping. Live-tunable via `kp_steering` / `kd_steering` parameters or topic-based (`/nav/kp_steering_set` / `/nav/kd_steering_set`).

**Current defaults**: `Kp = 1.10, Kd = 0.20`. Source: Bayesian Optimization with skopt (`scripts/bo_pd_tune.py`). BO's literal best was `Kp=1.08, Kd=0.08` but that's effectively undamped; **Option B** picked the safer middle of the low-J cluster (Kp ≈ 1.05–1.19, Kd robust 0.0–0.27) for real competition driving where tight corners matter.

### 5.4 Frame contract (single regime — Cartographer)

```
map → odom → base_link → base_scan
                       → aligned_camera_optical_frame
```

- Cartographer publishes `map → odom` (with `provide_odom_frame=false` and `published_frame="odom"` in `qcar2_2d.lua`).
- pose_estimator publishes `odom → base_link`.
- Static TFs from `fixed_lidar_frame[_virtual].cpp` publish `base_link → base_scan`.

The physical LiDAR is mounted with 180° yaw (per manual page 10's `rplidar_to_body` extrinsic). `fixed_lidar_frame.cpp` correctly applies `setRPY(0, 0, -π)`. The virtual variant uses `setRPY(0, 0, 0)` — believed correct if QLabs publishes scans pre-rotated, but verify on first physical run.

The **old RTAB regime** (`rtab_map → rtab_odom → base_link`) is gone — RTAB-Map has been retired.

### 5.5 Spawn pose at SDCSRoadMap node 0

Empirical calibration (see Easy_Start.md change log): to spawn the virtual QCar at node 0 of SDCSRoadMap (the default first waypoint of path_follower's `node_values=[0,8,10]`), edit `Setup_Real_Scenario.py` or `Setup_Competition_Map.py` in the QLabs container:
```python
initialPosition    = [0.000, 0.130, 0.005]
initialOrientation = [0, 0, -33]     # NOT -90 (SDCSRoadMap reports -90, but 3 rotation conventions stack)
```

The −33° value is empirical, accounting for: SDCSRoadMap node yaw + QLabs floor rotation (-90°) + nav_to_pose's `rotation_offset=83°`. Don't try to derive analytically.

### 5.6 Mutual exclusions (will silently corrupt data)

- Do NOT run `qcar2_nodes rgbd` together with `qcar2_perception d435_aligned_source` — d435_aligned_source owns the camera path.
- Do NOT run the standalone `manual_drive` node AND `path_follower` with `control_mode=manual` simultaneously — both publish `/cmd_vel_nav`.
- AMCL launches start a fresh `qcar2_virtual_launch.py` internally; do NOT also start it manually beforehand or you'll have duplicate `qcar2_hardware` nodes fighting QLabs.

---

## 6. What's deleted / retired (do NOT recreate)

This list keeps the next AI from "discovering" old patterns and reintroducing them:

| Deleted | Why |
|---|---|
| `Development/ros2/src/rtabmap/` and `rtabmap_ros/` (vendored RTAB-Map, ~170 MB, ~1700 files) | Architecture moved to Cartographer + AMCL. RTAB-Map never made it past smoke tests. Don't try to revive — clone from upstream if ever needed. |
| `qcar2_perception/launch/qcar2_rtabmap_mapping_*.launch.py` | Same. |
| `qcar2_autonomy/autonomy/yolo_detector_MARKERS_CPU_ABC.py` (old YOLO prototype) | Replaced by `qcar2_perception/semantic_yolo_detector`. Setup.py entry pruned. References removed from `autonomy_planner_launch.py`. |
| `qcar2_autonomy/autonomy/teleop_csi.py` | Used QLabs Python API directly, bypassed ROS, wouldn't work on physical. Manual_drive (ROS) + path_follower's manual mode supersede it. |
| `qcar2_nodes/src/qcar2_odometry.cpp` | Encoder-only C++ odometry. Retired in favor of `pose_estimator` + `ekf_fusor` (Python). Removed from CMakeLists.txt. |
| Easy_Start.txt, "nav_to_pose references.txt" | Scratch files, superseded by Easy_Start.md. |

Historical change-log entries in Easy_Start.md that mention these have `[DELETED]` markers so you know the things they describe no longer exist.

---

## 7. What's still on the roadmap (the unfinished story)

The runtime stack is complete enough to drive a node-to-node lap in QLabs and (probably) on physical. The competition deliverables that are still TBD:

1. **`golden_map.yaml` alignment** — Cartographer produces an "abstract" map (origin = boot pose). For competition, waypoints are in fixed world coordinates. `LCroadmap_alignment_node.py` is the start of the Procrustes/Kabsch alignment layer; it's registered in setup.py but not wired into any launch yet. **Build this next** when you save your first physical Cartographer map.
2. **AMCL on golden_map** — once `golden_map.yaml` exists, switch from Cartographer-live to AMCL on the frozen map for runtime. AMCL launches exist (`qcar2_amcl_localization_*`). `ekf_fusor`'s `correction_source` parameter switches from `tf` to `amcl_pose` for this.
3. **Reward grid** — not built. Should overlay lane / obstacle / stop / target layers as channels for the motion arbiter to consume. Spec discussed in Easy_Start.md change log.
4. **Motion arbiter** — not built. Should be the ONLY node publishing motor commands during competition. Today `path_follower` fills that role; an arbiter would sit between it and hardware, prioritizing emergency stop > obstacle avoidance > lane keeping > path following.
5. **Traffic light state classifier** — `semantic_yolo_detector` can detect a light, but classifying RED/YELLOW/GREEN from the cropped image isn't wired into a stop/go state machine yet.
6. **HUB re-localization on trip end** — single highest-ROI runtime safety net for multi-trip drift. When `trip_planner` detects "arrived at HUB", publish `/initialpose` with HUB's known coordinates. Resets accumulated drift between trips. ~50 lines of code.
7. **Stop-line precision for pickup/dropoff** — pure pursuit doesn't stop precisely. For the taxi pickup/dropoff (judges need the car within ~10 cm), add a "creep phase" inside `trip_planner` when within 0.5 m of pickup/dropoff.

---

## 8. Scripts in `Development/ros2/scripts/`

Full copy-paste recipes are in `Easy_Start.md` § "Scripts Reference". Summary:

| Script | What it does |
|---|---|
| `termname.sh` | Sets terminal title. Source helper. |
| `ros2_killall.sh` | Sweeps stale ROS procs between launches. Source helper. |
| `carto_to_amcl.sh` | End-to-end Cartographer → save → AMCL pipeline. Press ENTER to freeze map. |
| `pd_tuner.py` | Tkinter sliders → publishes to `/nav/kp_steering_set`, `/nav/kd_steering_set`. Live PD tuning. |
| `bo_pd_tune.py` | Bayesian Optimization of (Kp, Kd) with 15 trials. Two-phase per trial (APPROACH safe gains → MEASUREMENT trial gains). Outputs `/tmp/bo_pd_tune_result.json`. Requires `pip install scikit-optimize`. |
| `stress_test_for_EKF_and_mahalanobis.py` | Publishes alternating GOOD/BAD `/amcl_pose` to validate ekf_fusor's Mahalanobis outlier gate. Requires `ekf_fusor` running with `correction_source:=amcl_pose`. |

---

## 9. Topic contracts

- `/qcar2_pose_fused` (PoseWithCovarianceStamped) — **THE pose**. Any consumer needing the car's position in `map` frame should subscribe to this, not raw TF.
- `/cmd_vel_nav` (Twist) — single owner is `path_follower`. `nav2_qcar2_converter` converts to `/qcar2_motor_speed_cmd`.
- `/nav/controller_health` (String) — controller_watchdog status. Foxglove Indicator panel.
- `/qcar2_ekf/innovation_mahalanobis` (Float32) — outlier-gate metric. Healthy < 3; outlier threshold 11.345 (χ²_3 @ 99%).
- `/nav/{distance_to_waypoint, distance_to_final, psi, steering_saturation_rate, speed_cmd, yaw_rate_imu, progress_rate, wpi, controller_mode}` — controller diagnostics. All Float32 with `.data`. Foxglove Plot bindings.
- `/nav/kp_steering_set`, `/nav/kd_steering_set` (Float32) — live PD tuning. Receivers in path_follower.
- Perception topics on `/perception/d435/{rgb/image_raw, depth/image_rect, camera_info}` and `/perception/semantic_{landmark_markers, hypothesis_markers, current_markers, residual_markers}` — keep these four marker streams **separate** in Foxglove; collapsing destroys debugging signal.

---

## 10. Frame contract details

- Static `base_link → base_scan` from `fixed_lidar_frame[_virtual]`. Required.
- Static `base_link → aligned_camera_optical_frame` from `semantic_tf_launch.py` (in qcar2_perception).
- Cartographer publishes `map → odom`.
- pose_estimator publishes `odom → base_link`.
- AMCL (when running) publishes `map → odom` in place of Cartographer.

`tf2_echo` sanity edges before any autonomous drive:
```bash
ros2 run tf2_ros tf2_echo map odom
ros2 run tf2_ros tf2_echo odom base_link
ros2 run tf2_ros tf2_echo base_link base_scan
```

---

## 11. Memory of past sessions (in `~/.claude/projects/.../memory/`)

The previous Claude Code session built up project memories about the user's role, project history, decisions made. Read `MEMORY.md` in that dir before answering questions about "how did we end up at X" — the explanations are there. Update memory when the user explicitly confirms a decision or corrects an approach.

---

## 12. Things to be careful about

- **Don't add `* π/180` to anything that's already in rad/s.** ROS standard is rad/s; the QCar IMU honors it.
- **Don't recreate RTAB-Map.** If you need pose-graph optimization beyond Cartographer, use Cartographer's own loop closure or upgrade to GTSAM/iSAM2 directly.
- **Don't publish `/cmd_vel_nav` from a new node.** Anything that needs to influence motor commands should publish to a topic that `path_follower` (or the future motion arbiter) reads.
- **Don't change `path_follower`'s test-path dist_to_final completion radius from 0.25 m without checking BO geometry.** The BO test-path final waypoint is at 1.12 m from TEST_ORIGIN specifically because of this — too close and you get instant-complete bugs.
- **Don't set `WaypointDist` floor in path_planner below 0.05 m** — pure pursuit's `δ = atan(2L sin(ψ)/Ld)` denominator blows up.
- **Don't run two `pose_estimator` instances.** We've hit this several times; the second one comes from forgetting to kill the previous Cartographer launch. Use `ros2_killall` between sessions.
- **Don't enable Cartographer's `provide_odom_frame=true`** with an external odometry source. The combo creates dual ownership of `odom → base_link`. Current config is `provide_odom_frame=false` + `published_frame="odom"`.

---

## 13. Debugging entry points

- Build logs: `Development/ros2/log/latest_build/*/stdout_stderr.log`
- Runtime logs: `~/.ros/log/latest/*.log`
- Cartographer + BO temp logs: `/tmp/carto.log`, `/tmp/amcl.log`, `/tmp/bo_pd_tune_result.json`, `/tmp/final_pose.txt`
- TF sanity: `ros2 run tf2_ros tf2_echo <parent> <child>`
- Topic existence + rate: `ros2 topic list | grep <pattern>` ; `ros2 topic hz <topic>`
- Easy_Start.md § "Logs, Bags, And Debug Checks" has the full debug recipe set.

---

## 14. Hardware reference (physical QCar 2 only)

See `Easy_Start.md` § "Physical QCar 2 Reference" for the full sensor extrinsic matrices from the User Manual - System Hardware v1.0. Critical highlights:

- **Body frame {B}**: x = forward, y = left, z = up. Origin between front and rear axles on the ground.
- **Wheelbase**: 0.256 m (front_axle x=0.130 to rear_axle x=−0.130 → 0.260, but spec value is 0.256 m; use 0.256).
- **Steering limit**: ±30° = ±0.52 rad. Servo time constant 0.16 s.
- **IMU at (0.011, 0, 0.089) with identity rotation** — body-frame aligned, rad/s.
- **LiDAR at (-0.012, 0, 0.193) with 180° yaw** — `fixed_lidar_frame.cpp` compensates.
- **D435 at (0.095, 0.032, 0.172) looking forward** — for camera painting use `T_map_from_body × T_body_from_realsense × p_cam`.

For virtual QCar 2 in QLabs, all distances are 10× the physical (per manual). The ROS-side TF tree is always in real meters regardless.

---

## 15. When you're stuck

Search order:
1. `Easy_Start.md` change log — every architectural decision and bug fix is logged with date and reason.
2. The relevant module's docstring — most files have a header explaining their role.
3. Git history — `git log --oneline --all -- <file>`.
4. The "Things to be careful about" section above (§ 12).
5. Ask the user — describe what you tried and which file you'd edit.

Don't guess at architecture. The ekf_fusor + path_follower + control_mode stack was built carefully over several sessions; surface-level rewrites tend to reintroduce bugs we already fixed.

---

## 16. Final state at this handoff

- ✅ EKF refactor complete (pose_estimator predict + ekf_fusor full EKF with measurement update)
- ✅ PD tuned by BO, defaults Kp=1.10, Kd=0.20
- ✅ Unified control_mode in path_follower
- ✅ Cartographer launches bundle ekf_fusor automatically
- ✅ Foxglove launch bundles controller_watchdog
- ✅ Workspace cleaned (RTAB-Map deleted, old YOLO deleted, qcar2_odometry retired, stray scratch files removed)
- ✅ Easy_Start.md and CLAUDE.md current as of last edit

What user is doing next: **moving from QLabs to physical QCar 2 over SSH**. The validation gate is in Easy_Start.md's most recent change-log entry — six topic-hz / TF / manual-drive checks before any autonomous lap on physical.

Good luck. Be careful with the bus.
