# CLAUDE.md — ACC_Development

This file gives Claude Code the persistent context it needs when working in this repo. Keep it accurate.

## Project

Quanser ACC (Autonomous Connected Cars) competition 2026 codebase. The active development focus is **Visual Odometry (VO) as a redundancy layer** for the QCar2's Nav2 / Cartographer stack — the camera-based pose estimate independently verifies the IMU + encoder odometry and flags disagreements.

Primary user: Gabriel Licona. Work branch: `Gabriel`. Main branch: `main`.

## Hardware (QCar2 — read-only context)

- **Compute**: NVIDIA Jetson, JetPack R35.6 (L4T), kernel 5.10.216-tegra, aarch64
- **Camera**: Intel RealSense D435 (serial 243522071970, firmware 05.13.00.55), USB 3.0 (`8086:0b07`)
- **ROS distros installed**: `noetic`, `galactic`, `humble` — VO code targets **humble**
- **Connection**: User reaches the QCar2 via remote desktop / SSH over VS Code Remote-SSH.

## Scope of work — what Claude may and may NOT touch

**Allowed to read anywhere on the filesystem** for context (RealSense specs, ROS topic shapes, Quanser HAL/PAL APIs, etc.). Wandering outside `ACC_Development` for *information* is fine.

**Off-limits for edits**:
- `/home/nvidia/Documents/ACC_Development original/` — original baseline, never modified
- Anything under `/opt/ros/`, system drivers, Quanser core libraries (`Quanser_cam_intructor/hal/`, `Quanser_cam_intructor/pal/` are reference, not edit targets)
- RealSense / camera drivers, udev rules, kernel modules — we do not retune the stack
- Any system-level configuration that affects the QCar2 base behavior

**In-scope for edits** (the VS Code workspace):
- `Development/ros2/src/qcar2_autonomy/` — the autonomy package we own
- `Development/ros2/src/qcar2_nodes/launch/` and command conversion (when explicitly asked)
- Top-level docs in `ACC_Development/` (Easy_Start.txt, VO_CHANGELOG, VO_readings, etc.)

If a change would require touching a driver, Quanser core, or system config, **stop and surface it** instead of editing.

## Repo layout (the parts that matter)

```
ACC_Development/
├── Development/ros2/src/
│   ├── qcar2_autonomy/autonomy/        ← primary edit target
│   │   ├── visual_odometry.py           # VO engine: ORB + depth backprojection + RANSAC + SVD Procrustes
│   │   ├── vo_node.py                   # Redundancy monitor (VO vs Cartographer)
│   │   ├── vo_supervisor.py             # NORMAL / VO_UNTRUSTED / ODOM_WARNING / ODOM_FAULT / INITIALIZING / DEGRADED
│   │   ├── vo_capture.py                # Fault status capture utility
│   │   ├── vo_live_plot.py              # Matplotlib live plotter
│   │   ├── vo_terminal_dashboard.py     # Terminal dashboard
│   │   ├── vo_image_overlay.py          # Image overlay (untracked, in progress)
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

The actual runtime workspace is `~/ros2/`, NOT `ACC_Development/Development/ros2/`. The source in this repo is rsync'd into `~/ros2/src/` before `colcon build` (see Easy_Start.txt §0.6). When VO behavior on the physical car doesn't match recent code edits, the rsync was probably skipped.

## Build & run cheat sheet

```bash
# Per-session setup (from Easy_Start.txt §0)
source /opt/ros/humble/setup.bash
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.8/dist-packages
export ROS_DOMAIN_ID=67
cd ~/ros2 && source install/setup.bash

# After editing VO code (sync + build)
rsync -a --delete \
  ~/Documents/ACC_Development/Development/ros2/src/qcar2_autonomy/ \
  ~/ros2/src/qcar2_autonomy/
cd ~/ros2 && colcon build --packages-select qcar2_autonomy && source install/setup.bash
```

ROS entry points (from `qcar2_autonomy/setup.py`): `path_follower`, `manual_drive`, `traffic_system_detector`, `lane_detector`, `yolo_detector`, `trip_planner`, `dataset_collector`, `Planner_server`, `vo_node`, `vo_supervisor`, `vo_dashboard`, `vo_overlay`.

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
