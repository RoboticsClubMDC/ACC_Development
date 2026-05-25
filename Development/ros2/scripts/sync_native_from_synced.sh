#!/usr/bin/env bash
# sync_native_from_synced.sh — runs ON THE QCAR 2 (Jetson, native side).
#
# The laptop sync script mirrors the whole repo to:
#     ~/Documents/ACC_Development_luigi/
#
# Native ROS should mirror from:
#     ~/Documents/ACC_Development_luigi/Development/ros2/        (the "synced" tree)
#
# to its OWN workspace:
#     ~/ros2/                                                      (the "native" tree)
#
# This script mirrors synced -> native while protecting build/, install/, and
# log/. That lets `colcon build` in ~/ros2 pick up source/script changes without
# the laptop wiping native build artifacts. Run it in --watch mode and the chain is:
#
#     LAPTOP edit
#         └── sync_qcar2.sh --watch  ──ssh/rsync──▶  ~/Documents/ACC_Development_luigi/
#                                                            └── Development/ros2/
#                                                            └── sync_native_from_synced.sh --watch ──▶  ~/ros2/
#                                                                       └── (you) colcon build && ros2 launch
#
# Usage on QCar 2:
#   ~/Documents/ACC_Development_luigi/Development/ros2/scripts/sync_native_from_synced.sh
#   ~/Documents/ACC_Development_luigi/Development/ros2/scripts/sync_native_from_synced.sh --watch
#
# Prereqs on QCar 2:
#   mkdir -p ~/ros2
#   sudo apt install inotify-tools     # only needed for --watch

set -euo pipefail

SYNCED_ROS2="$HOME/Documents/ACC_Development_luigi/Development/ros2/"
NATIVE_ROS2="$HOME/ros2/"

mkdir -p "$NATIVE_ROS2"

RSYNC_FLAGS=(
  -av
  --delete
  --exclude 'build/'
  --exclude 'install/'
  --exclude 'log/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.venv/'
)

do_sync() {
  echo "[sync_native] $(date +%H:%M:%S) mirroring ${SYNCED_ROS2} -> ${NATIVE_ROS2}"
  rsync "${RSYNC_FLAGS[@]}" "$SYNCED_ROS2" "$NATIVE_ROS2"
  echo "[sync_native] done."
}

if [[ "${1:-}" == "--watch" ]]; then
  command -v inotifywait >/dev/null || {
    echo "Install inotify-tools first: sudo apt install inotify-tools"; exit 1; }
  do_sync
  while inotifywait -r -e modify,create,delete,move \
      --exclude '(build|install|log|__pycache__|\.git)' \
      "$SYNCED_ROS2" >/dev/null 2>&1; do
    sleep 0.3   # coalesce burst writes from a multi-file rsync push
    do_sync
  done
else
  do_sync
fi
