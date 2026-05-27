#!/usr/bin/env bash
# sync_native_remote.sh — laptop-side wrapper that triggers
# sync_native_from_synced.sh on the QCar 2 over SSH.
#
# What it does:
#   - Connects to qcar2 (via ~/.ssh/config alias 'qcar2').
#   - Runs ~/Documents/ACC_Development_luigi/Development/ros2/scripts/
#       sync_native_from_synced.sh on the remote.
#   - Streams its stdout/stderr back to your laptop terminal.
#
# Use one of two modes:
#   ./sync_native_remote.sh           # one-shot mirror, exits when done
#   ./sync_native_remote.sh --watch   # remote watch loop, blocks until Ctrl-C
#
# Prereqs:
#   - sync_qcar2.sh has already mirrored the repo at least once
#     (so the script exists on the QCar at the path above).
#   - ssh qcar2 works without password (one-time `ssh-copy-id qcar2`).
#   - On QCar2: `sudo apt install inotify-tools` if you plan to use --watch.
#
# Convenience: pair with sync_qcar2.sh --watch in another laptop terminal
# so the chain (laptop edit → synced → native) is fully automatic.

set -e

REMOTE="${QCAR2_REMOTE:-qcar2}"
REMOTE_SCRIPT="\$HOME/Documents/ACC_Development_luigi/Development/ros2/scripts/sync_native_from_synced.sh"

WATCH_FLAG=""
for arg in "$@"; do
  case "$arg" in
    --watch) WATCH_FLAG="--watch" ;;
    *) echo "Unknown arg: $arg"; echo "Usage: $0 [--watch]"; exit 1 ;;
  esac
done

if ! command -v ssh >/dev/null 2>&1; then
  echo "ERROR: ssh not in PATH"; exit 1
fi

echo "[sync_native_remote] target: $REMOTE"
echo "[sync_native_remote] remote command: $REMOTE_SCRIPT $WATCH_FLAG"
echo ""

# -t so the remote script's terminal output (rsync progress lines) flushes
# in real time. Pass through Ctrl-C cleanly via -o ServerAliveInterval.
exec ssh -t -o ServerAliveInterval=30 "$REMOTE" "bash -lc '$REMOTE_SCRIPT $WATCH_FLAG'"
