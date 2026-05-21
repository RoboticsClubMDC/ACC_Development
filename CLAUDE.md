# CLAUDE.md — ACC_Development_gabriel

This file gives Claude Code the persistent context it needs when working in this repo. Keep it accurate.

## Project

Quanser ACC (Autonomous Connected Cars) competition 2026 codebase. The active development focus is **Visual Odometry (VO) as a redundancy layer** for the QCar2's Nav2 / Cartographer stack — the camera-based pose estimate independently verifies the IMU + encoder odometry and flags disagreements.

Primary user: Gabriel Licona. Work branch: `Gabriel`. Main branch: `main`.

**Folder note (2026-05-20)**: Gabriel's working tree lives at `/home/nvidia/Documents/ACC_Development_gabriel/` (renamed from `ACC_Development` after a teammate's delete/restore). Arturo's separate clone at `/home/nvidia/Documents/ACC_Development/` is on branch `Physical_Arturo` and is **off-limits** — never read from it for rsync sources, never edit it, do not compare branches with it unless explicitly asked.

## Hardware (QCar2 — read-only context)

- **Compute**: NVIDIA Jetson, JetPack R35.6 (L4T), kernel 5.10.216-tegra, aarch64
- **Camera**: Intel RealSense D435 (serial 243522071970, firmware 05.13.00.55), USB 3.0 (`8086:0b07`)
- **ROS distros installed**: `noetic`, `galactic`, `humble` — VO code targets **humble**
- **Connection**: User reaches the QCar2 via remote desktop / SSH over VS Code Remote-SSH.

## Scope of work — what Claude may and may NOT touch

**Allowed to read anywhere on the filesystem** for context (RealSense specs, ROS topic shapes, Quanser HAL/PAL APIs, etc.). Wandering outside `ACC_Development_gabriel` for *information* is fine.

**Off-limits for edits AND for rsync sources**:
- `/home/nvidia/Documents/ACC_Development original/` — pristine baseline, never modified
- `/home/nvidia/Documents/ACC_Development/` — Arturo's clone (`Physical_Arturo` branch). Different `qcar2_autonomy/autonomy/` contents (lane_assist_blend, lane_keeping, path_teacher, …). Never rsync from here into `~/ros2/`.
- `/home/nvidia/Documents/ACC_Development_backup_gabriel/` — local backup snapshot of Gabriel's tree; read-only safety net.
- Anything under `/opt/ros/`, system drivers, Quanser core libraries (`Quanser_cam_intructor/hal/`, `Quanser_cam_intructor/pal/` are reference, not edit targets)
- RealSense / camera drivers, udev rules, kernel modules — we do not retune the stack
- Any system-level configuration that affects the QCar2 base behavior

**In-scope for edits** (the VS Code workspace — `/home/nvidia/Documents/ACC_Development_gabriel/`):
- `Development/ros2/src/qcar2_autonomy/` — the autonomy package we own
- `Development/ros2/src/qcar2_nodes/launch/` and command conversion (when explicitly asked)
- Top-level docs in `ACC_Development_gabriel/` (Easy_Start.txt, VO_CHANGELOG, VO_readings, etc.)

If a change would require touching a driver, Quanser core, or system config, **stop and surface it** instead of editing.

## Repo layout (the parts that matter)

```
ACC_Development_gabriel/
├── Development/ros2/src/
│   ├── qcar2_autonomy/autonomy/        ← primary edit target
│   │   ├── visual_odometry.py           # VO engine: ORB + depth backprojection + RANSAC + SVD Procrustes
│   │   ├── vo_node.py                   # Redundancy monitor (VO vs Cartographer)
│   │   ├── vo_supervisor.py             # Redundancy state machine + act-on-faults (stop_advised from /vo/healthy)
│   │   ├── vo_terminal_dashboard.py     # Terminal dashboard (entry point: vo_dashboard)
│   │   ├── vo_image_overlay.py          # Image/feature overlay (entry point: vo_overlay; future RANSAC viz panel)
│   │   ├── vo_odom_tf_relay.py          # Adapts /vo/odometry for RTAB-Map: re-frame to 'odom' + TF + covariance sanitize
│   │   ├── nav_to_pose.py               # Nav2 goal sender
│   │   ├── manual_drive.py              # WASD keyboard manual drive (recently added)
│   │   ├── lane_detector.py / traffic_system_detector.py / yolo_detector*.py
│   │   └── *_old.py                     # frozen reference copies — do not edit
│   └── qcar2_nodes/                     # Quanser-provided nodes + launch files (edit launch only when asked)
├── Quanser_cam_intructor/               ← reference (HAL/PAL/products) — do not edit
├── isaac_ros_common/                    ← reference — do not edit
├── backup/                              ← reference (student-competition-resources, Quanser_Academic_Resources)
├── docker/                              ← reference unless a docker task is requested
├── vo_calib_logs/                       ← calibration log dumps; new ones are added here, do not prune
├── Camera Intrinsics Post.txt           ← physical D435 intrinsics (source of truth for physical mode)
├── Virtual_Stage_ROS_FAQ.md             ← virtual D435 intrinsics (source of truth for virtual mode)
├── user_manual_system_hardware.pdf      ← camera-to-body extrinsics
├── Easy_Start.txt                       ← canonical run/test procedure
├── VO_CHANGELOG.md / VO_readings.txt    ← evolving notes; user updates these manually
└── VO_Conversation_Log.txt              ← human-curated log of past sessions
```

The actual runtime workspace is `~/ros2/`, NOT `ACC_Development_gabriel/Development/ros2/`. The source in this repo is rsync'd into `~/ros2/src/` before `colcon build` (see Easy_Start.txt §0.6). When VO behavior on the physical car doesn't match recent code edits, the rsync was probably skipped. **Sanity check**: `readlink -f ~/ros2/src/qcar2_autonomy/autonomy/visual_odometry.py` is a real file (not a symlink) and its contents should match `~/Documents/ACC_Development_gabriel/Development/ros2/src/qcar2_autonomy/autonomy/visual_odometry.py` — if a `diff -q` between them comes back differing or empty-on-Gabriel-side, the wrong source folder was rsync'd.

## Build & run cheat sheet

```bash
# Per-session setup (from Easy_Start.txt §0)
source /opt/ros/humble/setup.bash
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.8/dist-packages
export ROS_DOMAIN_ID=67
cd ~/ros2 && source install/setup.bash

# After editing VO code (sync + build) — note: source is ACC_Development_gabriel, NOT ACC_Development
rsync -a --delete \
  ~/Documents/ACC_Development_gabriel/Development/ros2/src/qcar2_autonomy/ \
  ~/ros2/src/qcar2_autonomy/
cd ~/ros2 && colcon build --packages-select qcar2_autonomy && source install/setup.bash
```

ROS entry points (from `qcar2_autonomy/setup.py`): `path_follower`, `manual_drive`, `traffic_system_detector`, `lane_detector`, `yolo_detector`, `trip_planner`, `dataset_collector`, `Planner_server`, `vo_node`, `vo_supervisor`, `vo_dashboard`, `vo_overlay`.

## VO toolbox knobs (status as of 2026-05-19)

Three independent runtime knobs on `vo_node`, all default-safe (defaults reproduce pre-toolbox behavior):
- `vo_frontend` ∈ `{orb, klt}` — ORB descriptor matching vs KLT pyramidal LK tracking (shared ORB detector seed).
- `vo_estimator` ∈ `{svd, pnp}` — closed-form Procrustes (3D↔3D) vs `cv2.solvePnPRansac` (3D↔2D), both converted back to the same body-frame planar motion.
- `feature_grid` ∈ `{0, 8, …}` — grid-distributed feature selection; `8` is the recommended operating point on the mat (beat Test-6 baseline on every metric); `12` over-thins (do not use).

All 4 `{orb,klt}×{svd,pnp}` combos verified to run; default `orb+svd, feature_grid=0` is byte-equivalent to pre-toolbox VO.

## RTAB-Map / SLAM showcase plan

**Install status (verified 2026-05-20)**: `rtabmap_ros` 0.21.1 is **already fully installed** on the QCar's humble/L4T via apt (all 13 component packages; binaries `/opt/ros/humble/bin/rtabmap` and `rtabmap-databaseViewer` on PATH). No install action required.

**Showcase architecture (decided 2026-05-20, "record-once / playback-three-ways")**: record one canonical physical-run bag with vo_node + cartographer + camera_bridge live, then replay the bag through RTAB-Map three times with different pose sources. Comparison gives a directly comparable A/B/C across:

| `odom_source` | What it tests | Visual? |
|---|---|---|
| `rtabmap_odom` (default) | RTAB-Map's own internal VO on the bagged RGB-D | Pure visual (no IMU/lidar) |
| `vo_node` | Our `/vo/odometry` (from bag) feeds RTAB-Map | "Visual + IMU yaw correction" if recorded with `force_cart_yaw=true`; pure-visual if recorded with `force_cart_yaw=false` (record both for the comparison) |
| `cartographer` | Cartographer's `/odom` (from bag) feeds RTAB-Map | NOT pure-visual — lidar + IMU + wheel pose, camera-augmented |

Implementation:
- Launch file: `Development/ros2/src/qcar2_nodes/launch/qcar2_rtabmap_launch.py` wraps `rtabmap_launch/rtabmap.launch.py` and exposes `odom_source ∈ {rtabmap_odom, vo_node, cartographer}` as the single switch. Pinned remaps: `rgb_topic:=/camera/color_image`, `depth_topic:=/camera/depth_image`, `camera_info_topic:=/camera/camera_info`, `frame_id:=base_link`, `approx_sync:=true`, `rgbd_sync:=true`, `database_path:=~/vo_rtab_bags/rtabmap.db`. Defaults to bag playback (`use_sim_time:=true`). Pure glue — does not redefine any RTAB-Map internals.
- Recording recipe + per-mode playback (with `--exclude` lists to avoid topic conflicts) + comparison protocol all live in `Easy_Start.txt §7`.
- Persistent bag dir: `~/vo_rtab_bags/` (exists). Never `/tmp` — we lost one bag that way.
- One canonical best-bag policy: keep one on QCar, transfer dbs off to personal machine, prune old bags.

Frontend choice for RTAB-Map: **ORB only**. KLT has no descriptor → can't match against the appearance database for loop closure / relocalization. When bagging for SLAM, keep `vo_frontend:=orb` (the default). KLT and PnP knobs are for VO-redundancy A/B testing, not SLAM.

Open observation (to investigate via bag, no live time needed): user has seen unphysical x-jumps in both `/vo/odometry` (~0.6 m/s instantaneous at a commanded 0.1 m/s) and `/odom` (Cartographer) through curve segments. Bag captures both topics → can analyze synchronized jumps offline. See Easy_Start §7.7.

## Camera resolution — verdict (revised 2026-05-20)

Earlier "skip, needs recalibration" was wrong. RealSense exposes per-resolution intrinsics in the SDK (`rs-enumerate-devices -c` lists them per profile) and `camera_bridge` already publishes `/camera/camera_info` for whatever resolution is configured. Cleanest evolution: add `intrinsics_source ∈ {camera_info, physical_480, virtual_480}` parameter to `visual_odometry.py`, default `camera_info` → VO becomes resolution-agnostic. Hardcoded tables stay as fallbacks.

Revised verdicts:
- **480p** (current baseline) — keep.
- **720p** — worth a test campaign once VSLAM baseline is up. Note: at higher resolution, *lower* `n_features` (not raise) to keep per-frame ORB cost ~constant, since each detected feature is more expensive to score at higher resolution.
- **1080p** — probably skip. ORB cost scales ~linearly with pixels; with cartographer + vo_node + bridge co-running on the Jetson, real-time budget is tight.

Test matrix (for the resolution session, not today):
- 480p × n_features ∈ {400, 800, 1200} — characterizes the low-end.
- 720p × n_features ∈ {600, 800, 1200} — does more resolution beat more features at iso-CPU?
- Same `feature_grid=8`, same bag-driven evaluation across all cells.

## EKF fusion — note for when we wire `robot_localization`

The EKF is **not** a flat average of inputs (correcting an oversimplification a Quanser engineer offered). It is a recursive Bayesian filter: predict step projects state with motion model + grows covariance by process noise; update step applies Kalman gain that weights each measurement **inversely by its covariance** relative to the predicted-state covariance. Practical consequence for our pipeline: `/vo/odometry` already publishes honest covariance driven by `confidence` and `/vo/conditioning`, so the EKF automatically downweights VO when conditioning is poor. The redundancy showcase pitch is therefore "VO publishes honest uncertainty and the EKF does the right blend," not "we average them."

Cartographer's covariance, by contrast, is roughly constant by default — it does not flag itself as uncertain when it jumps. That's a Cartographer-side EKF-tuning concern to address when `robot_localization` is wired. RTAB-Map builds the map/trajectory off the bag, off-car, for RViz playback.

## Camera intrinsics — which to use

`visual_odometry.py` carries **two** calibration tables:
- **Virtual mode** values come from `Virtual_Stage_ROS_FAQ.md` (D435 sim intrinsics)
- **Physical mode** values come from `Camera Intrinsics Post.txt` (real D435 post-calibration)

Don't cross-pollinate them. Mode-specific tables at 640×480 are the source of truth; other resolutions exist in the txt file if needed.

## Git workflow

- Always work on branch `Gabriel`. Verify with `git branch --show-current` before edits.
- One consolidated push to `origin/Gabriel` at end of day — not repeatedly during the day.
- Never force-push, never amend pushed commits, never push to `main`.
- The `install/`, `build/`, `log/` directories and `vo_calib_logs/` are noise — don't stage them. The repo already has a `.gitignore` (check it before adding new untracked files).

## Conventions

- Python: existing style is descriptive class-level docstrings, inline math comments where geometry is non-obvious. Match that — don't strip explanatory math comments thinking they're chatty.
- Files ending in `_old.py` are intentional frozen references for diffing. Don't delete, don't refactor.
- New VO calibration runs append a timestamped file to `vo_calib_logs/` — keep that pattern.
- When the user says "physical" vs "virtual" they mean real QCar2 hardware vs QLabs simulation; the VO pipeline branches on this, so be precise.
- **Always-log rule**: every code change or design decision in this session must be appended to BOTH `Development/ros2/src/qcar2_autonomy/VO_CHANGELOG.md` (dated technical entry) AND `VO_Conversation_Log.txt` (Turn-style summary — user prompt rephrased formally + assistant response, retaining file paths and parameter names). Do this in the same turn as the change, not later. `VO_readings.txt` is user-owned — never touch it.

## Terminals — what Claude can and can't do

Claude runs commands in its own non-interactive subshells via the Bash tool. It **cannot** see, attach to, or send keystrokes into the user's open VS Code integrated terminals or `gnome-terminal` windows. If the user wants a command run in *their* terminal, they have to paste/run it themselves; Claude can only run things in its own ephemeral shells (which inherit env from the user's profile on launch but don't share state across calls).

For long-running tasks (colcon builds, ROS nodes), Claude can launch them in the background and stream output — but the process is owned by Claude's shell, not by a user terminal.
