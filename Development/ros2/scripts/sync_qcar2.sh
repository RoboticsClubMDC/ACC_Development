#!/usr/bin/env bash
# sync_qcar2.sh - runs on the laptop.
#
# One command does the physical-QCar edit loop:
#   1. Copy the laptop clock to the QCar 2 so ROS/Foxglove timestamps agree.
#   2. Mirror the full ACC_Development repo to the QCar 2 branch folder:
#        laptop: ~/Documents/GitHub/ACC_Development/
#        qcar2 : ~/Documents/ACC_Development_luigi/
#
# The QCar-side sync_native_from_synced.sh then mirrors:
#        ~/Documents/ACC_Development_luigi/Development/ros2/
#     -> ~/ros2/
# while protecting native build/, install/, and log/.
#
# Usage:
#   sync_qcar2.sh
#   sync_qcar2.sh --watch

set -euo pipefail

REMOTE="${QCAR2_REMOTE:-qcar2}"
REMOTE_USER="${QCAR2_REMOTE_USER:-nvidia}"
REMOTE_ROOT="${QCAR2_REMOTE_ROOT:-/home/nvidia/Documents/ACC_Development_luigi/}"
LOCAL_ROOT="${QCAR2_LOCAL_ROOT:-/home/${USER}/Documents/GitHub/ACC_Development/}"
SUDO_PW="${QCAR2_SUDO_PW:-nvidia}"

RSYNC_FLAGS=(
  -avz
  --delete
  --exclude '.git/'
  --exclude 'build/'
  --exclude 'install/'
  --exclude 'log/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.venv/'
  --exclude 'Development/ros2/src/rtabmap/'
  --exclude 'Development/ros2/src/rtabmap_ros/'
  --exclude 'Development/ros2/build/'
  --exclude 'Development/ros2/install/'
  --exclude 'Development/ros2/log/'
)

require_local_root() {
  if [[ ! -d "$LOCAL_ROOT/Development/ros2/src" ]]; then
    echo "[sync_qcar2] ERROR: LOCAL_ROOT does not look like ACC_Development: $LOCAL_ROOT" >&2
    exit 1
  fi
}

sync_clock() {
  local stamp_utc tz_local laptop_epoch remote_epoch drift drift_abs

  stamp_utc=$(date -u +"%Y-%m-%d %H:%M:%S")
  tz_local=$(timedatectl show --property=Timezone --value 2>/dev/null || echo "UTC")

  echo "[sync_qcar2] syncing QCar clock: ${stamp_utc} UTC, timezone ${tz_local}"

  ssh -T "$REMOTE" bash -s <<EOF
set -e
echo '${SUDO_PW}' | sudo -S -p '' timedatectl set-ntp false 2>/dev/null || true
echo '${SUDO_PW}' | sudo -S -p '' timedatectl set-timezone '${tz_local}' 2>/dev/null || true
echo '${SUDO_PW}' | sudo -S -p '' date -u -s '${stamp_utc}' >/dev/null
echo '${SUDO_PW}' | sudo -S -p '' hwclock --systohc 2>/dev/null || true
echo "[sync_qcar2] qcar2 clock: \$(date)"
EOF

  laptop_epoch=$(date -u +%s)
  remote_epoch=$(ssh -T "$REMOTE" 'date -u +%s')
  drift=$((laptop_epoch - remote_epoch))
  drift_abs=${drift#-}
  echo "[sync_qcar2] clock drift after sync: ${drift}s"
  if [[ "$drift_abs" -gt 2 ]]; then
    echo "[sync_qcar2] WARNING: drift is above 2 s. Check sudo password/network delay." >&2
  fi
}

do_sync() {
  local sync_time="${1:-1}"
  require_local_root
  if [[ "$sync_time" == "1" ]]; then
    sync_clock
  else
    echo "[sync_qcar2] skipping clock sync in watch update; do not step ROS time while nodes are running"
  fi
  echo "[sync_qcar2] $(date +%H:%M:%S) mirroring ${LOCAL_ROOT} -> ${REMOTE}:${REMOTE_ROOT}"
  ssh -T "$REMOTE" "mkdir -p '${REMOTE_ROOT}'"
  rsync "${RSYNC_FLAGS[@]}" "$LOCAL_ROOT" "${REMOTE}:${REMOTE_ROOT}"
  echo "[sync_qcar2] done."
}

if [[ "${1:-}" == "--watch" ]]; then
  command -v inotifywait >/dev/null || {
    echo "Install inotify-tools first: sudo apt install inotify-tools" >&2
    exit 1
  }
  do_sync 1
  while inotifywait -r -e modify,create,delete,move \
      --exclude '(build|install|log|__pycache__|\.git)' \
      "$LOCAL_ROOT" >/dev/null 2>&1; do
    sleep 0.3
    do_sync 0
  done
else
  do_sync 1
fi
