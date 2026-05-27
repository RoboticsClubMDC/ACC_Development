#!/usr/bin/env bash
# carto_to_amcl.sh — one-terminal pipeline: launch Cartographer, drive the
# car with WASD keys in THIS terminal, then Ctrl-C to freeze the map and
# transition to AMCL seeded with the final pose.
#
# Flow:
#   1. Launches Cartographer in the background (logs to /tmp/carto.log).
#   2. Waits for /map to appear.
#   3. Hands the terminal over to `manual_drive` — drive with WASD in this
#      same terminal. Cartographer keeps mapping in the background while
#      you drive.
#   4. When the map looks good, Ctrl-C to exit manual_drive (does NOT kill
#      the script — SIGINT is trapped during this phase).
#   5. Captures the final TF, saves the map to ~/qcar2_maps/, kills carto.
#   6. Launches AMCL on the saved map, seeds /initialpose with the captured
#      pose. AMCL stays running in the foreground.
#
# Usage (inside the Isaac ROS dev container):
#   ./carto_to_amcl.sh           # default: virtual / QLabs
#   ./carto_to_amcl.sh physical  # use physical-robot launch files
#
# Side effects:
#   - Writes ~/qcar2_maps/competition_map.{pgm,yaml}
#   - Leaves AMCL (+ ekf_fusor + map_server + lifecycle_manager) running in
#     the foreground; Ctrl-C the script ONLY when you're done with AMCL.
#
# Logs go to /tmp/carto.log and /tmp/amcl.log so the current terminal stays
# clean for status output.

set -e

MODE="${1:-virtual}"
MAP_DIR="${HOME}/qcar2_maps"
MAP_NAME="competition_map"

case "$MODE" in
  virtual)
    CARTO_LAUNCH="qcar2_cartographer_virtual_launch.py"
    AMCL_LAUNCH="qcar2_amcl_localization_virtual_launch.py"
    ;;
  physical)
    CARTO_LAUNCH="qcar2_cartographer_launch.py"
    AMCL_LAUNCH="qcar2_amcl_localization_launch.py"
    ;;
  *)
    echo "Usage: $0 [virtual|physical]"
    exit 1
    ;;
esac

# Sanity check: rely on the parent shell having already sourced ROS. This avoids
# stale install/setup.bash messages and keeps the script free of workspace-path
# assumptions.
if ! command -v ros2 >/dev/null 2>&1; then
  echo "ERROR: 'ros2' not in PATH. Source /opt/ros/humble/setup.bash and your"
  echo "workspace install/setup.bash before running this script."
  exit 1
fi
if [[ -z "${ROS_DOMAIN_ID:-}" ]]; then
  echo "WARNING: ROS_DOMAIN_ID is not set. The QCar2 workspace uses 69."
  echo "         export ROS_DOMAIN_ID=69 and rerun."
  exit 1
fi

mkdir -p "$MAP_DIR"

# Kill the entire Cartographer launch process group. Cartographer's launch
# spawns pose_estimator, qcar2_hardware, fixed_lidar_frame, etc. via Include —
# pattern-based pkill leaves stragglers and produces duplicate nodes once AMCL
# spawns its own copies. Process-group kill catches every descendant.
kill_carto_group() {
  if [[ -n "${CARTO_PGID:-}" ]]; then
    kill -INT  -"$CARTO_PGID" 2>/dev/null || true
    sleep 2
    kill -TERM -"$CARTO_PGID" 2>/dev/null || true
    sleep 1
    kill -KILL -"$CARTO_PGID" 2>/dev/null || true
  fi
  # Defensive: kill any cartographer_* still floating (e.g. if user launched
  # Cartographer separately before running this script).
  pkill -KILL -f "cartographer_node|cartographer_occupancy_grid_node" 2>/dev/null || true
}

# Refuse to start if Cartographer is already up — better than racing.
if ros2 node list 2>/dev/null | grep -q "cartographer"; then
  echo "ERROR: A cartographer_* node is already running. Stop it first, then rerun."
  exit 1
fi

echo "[1/6] Launching Cartographer ($CARTO_LAUNCH)..."
# setsid puts the launch into its own process group so we can kill the whole
# tree (Cartographer + included qcar2_virtual + pose_estimator + lidar_tf) in
# one shot at transition time.
setsid ros2 launch qcar2_nodes "$CARTO_LAUNCH" >/tmp/carto.log 2>&1 &
CARTO_PID=$!
CARTO_PGID=$CARTO_PID  # setsid makes the child its own group leader

# Wait until /map exists (Cartographer's occupancy grid is publishing).
echo "      Waiting for /map topic..."
TIMEOUT=60
while ! ros2 topic info /map >/dev/null 2>&1; do
  sleep 1
  TIMEOUT=$((TIMEOUT - 1))
  if [[ $TIMEOUT -le 0 ]]; then
    echo "ERROR: /map never appeared. Check /tmp/carto.log."
    kill_carto_group
    exit 1
  fi
done

echo ""
echo "[1.5/6] Capturing INITIAL pose (where carto started)..."
# Wait a couple of seconds for map → base_link TF to settle, then snapshot.
# This pose is what AMCL needs as a seed on a future load-without-record
# session (where the user re-uses the saved map and wants AMCL to know
# where the car is at startup).
sleep 2
INIT_OUT=$(timeout 4 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null || true)
INIT_PARSED=$(echo "$INIT_OUT" \
  | grep -E "Translation|Quaternion" \
  | awk -F'[][]' '{print $2}' \
  | tr ',' ' ')
INIT_X=$(echo "$INIT_PARSED" | sed -n '1p' | awk '{print $1}')
INIT_Y=$(echo "$INIT_PARSED" | sed -n '1p' | awk '{print $2}')
INIT_QZ=$(echo "$INIT_PARSED" | sed -n '2p' | awk '{print $3}')
INIT_QW=$(echo "$INIT_PARSED" | sed -n '2p' | awk '{print $4}')
if [[ -z "$INIT_X" || -z "$INIT_Y" ]]; then
  echo "      WARN: could not parse initial pose. Skipping."
  INIT_X="0.0"; INIT_Y="0.0"; INIT_QZ="0.0"; INIT_QW="1.0"
fi
echo "      initial:  x=$INIT_X  y=$INIT_Y  qz=$INIT_QZ  qw=$INIT_QW"

echo ""
echo "      Cartographer is up. Now entering MANUAL DRIVE mode in this terminal."
echo ""
echo "      Controls (WASD-ish, see manual_drive node for exact bindings):"
echo "        w / s     forward / back"
echo "        a / d     steer left / right"
echo "        space     stop"
echo "        q         finish driving and FREEZE the map → AMCL"
echo "        Ctrl-C    also works (script catches the signal and continues)"
echo ""
echo "      Drive one slow clean lap. When the map looks good, press q (or Ctrl-C)."
echo "      Then wait a few seconds — the script will save the map, kill carto,"
echo "      and launch AMCL automatically."
echo ""
sleep 1

# Run manual_drive in foreground. set +e so the SIGINT exit code from
# manual_drive doesn't abort the rest of the script. manual_drive owns
# stdin (raw termios) during this phase — the user's keys drive the car,
# Cartographer keeps recording in the background.
#
# Trap SIGINT during this phase so Ctrl-C cleanly exits manual_drive
# WITHOUT killing the script (otherwise bash defers the SIGINT, kills
# manual_drive's child, then exits itself before saving the map).
#
# IMPORTANT: manual_drive defaults to publishing /cmd_vel_nav, which the
# nav2_qcar2_converter consumes. That's what we want during mapping.
# (The PP/Stanley/blender stack is not running here — only carto + drive.)
trap 'echo ""; echo "      [carto_to_amcl] SIGINT — finishing manual drive..."' INT
set +e
ros2 run qcar2_autonomy manual_drive
MANUAL_RC=$?
set -e
trap - INT
echo ""
echo "      Manual drive exited (rc=$MANUAL_RC). Proceeding to freeze map..."

echo "[2/6] Capturing final pose from TF (map -> base_link)..."
# Some ROS 2 humble builds don't have tf2_echo --once, so use `timeout` and
# parse the first snapshot. timeout returns non-zero when it kills the child,
# hence the `|| true` (we do not want set -e to abort here).
POSE_OUT=$(timeout 4 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null || true)
echo "$POSE_OUT" > /tmp/final_pose.txt

# Parse the FIRST "Translation: [x, y, z]" line and the FIRST
# "Quaternion [qx, qy, qz, qw]" line from the captured snapshots.
PARSED=$(echo "$POSE_OUT" \
  | grep -E "Translation|Quaternion" \
  | awk -F'[][]' '{print $2}' \
  | tr ',' ' ')

X=$(echo "$PARSED" | sed -n '1p' | awk '{print $1}')
Y=$(echo "$PARSED" | sed -n '1p' | awk '{print $2}')
QZ=$(echo "$PARSED" | sed -n '2p' | awk '{print $3}')
QW=$(echo "$PARSED" | sed -n '2p' | awk '{print $4}')

if [[ -z "$X" || -z "$Y" || -z "$QZ" || -z "$QW" ]]; then
  echo "ERROR: Could not parse pose from tf2_echo. Raw output saved to /tmp/final_pose.txt:"
  echo "$POSE_OUT"
  echo ""
  echo "Cartographer is still running. Capture the pose manually with:"
  echo "  ros2 run tf2_ros tf2_echo map base_link"
  echo "Then save the map and start AMCL manually."
  exit 1
fi

echo "      pose:  x=$X  y=$Y  qz=$QZ  qw=$QW"
echo "      saved to /tmp/final_pose.txt"

# Persist BOTH the initial (carto-start) and final (carto-end) poses
# next to the saved map. The companion script `amcl_load.sh` reads this
# YAML when bringing AMCL back up against the saved map without recording.
POSE_YAML="$MAP_DIR/${MAP_NAME}_pose.yaml"
cat > "$POSE_YAML" <<EOF
# Companion file written by carto_to_amcl.sh.
# Used by amcl_load.sh to seed /initialpose when AMCL is launched against
# the saved map without re-recording.
initial_pose:
  position:    {x: $INIT_X, y: $INIT_Y, z: 0.0}
  orientation: {x: 0.0, y: 0.0, z: $INIT_QZ, w: $INIT_QW}
final_pose:
  position:    {x: $X, y: $Y, z: 0.0}
  orientation: {x: 0.0, y: 0.0, z: $QZ, w: $QW}
EOF
echo "      pose YAML: $POSE_YAML"

echo "[3/6] Saving map to $MAP_DIR/$MAP_NAME..."
ros2 run nav2_map_server map_saver_cli -f "$MAP_DIR/$MAP_NAME"
if [[ ! -f "$MAP_DIR/$MAP_NAME.yaml" ]]; then
  echo "ERROR: map_saver_cli did not produce $MAP_NAME.yaml"
  kill_carto_group
  exit 1
fi

echo "[4/6] Killing Cartographer (keeping foxglove_bridge / manual_drive alive)..."
kill -INT $CARTO_PID 2>/dev/null || true
sleep 2
kill_carto_group
sleep 1

echo "[5/6] Launching AMCL ($AMCL_LAUNCH) with saved map..."
# setsid AMCL too — otherwise on any timeout/exit, the AMCL launch tree leaks
# and accumulates duplicate lifecycle managers / map_servers across retries.
setsid ros2 launch qcar2_nodes "$AMCL_LAUNCH" map:="$MAP_DIR/$MAP_NAME.yaml" >/tmp/amcl.log 2>&1 &
AMCL_PID=$!
AMCL_PGID=$AMCL_PID

# From here on, if we exit for any reason (error, signal, normal exit), make
# sure the AMCL tree dies with us. Otherwise leftover processes break the next
# run.
trap 'echo ""; echo "Cleaning up..."; \
      kill -INT -"$AMCL_PGID" 2>/dev/null || true; sleep 2; \
      kill -KILL -"$AMCL_PGID" 2>/dev/null || true; \
      exit' INT TERM EXIT

# Wait for AMCL lifecycle to activate (i.e. /amcl_pose subscribable).
echo "      Waiting for AMCL to come up..."
TIMEOUT=60
while ! ros2 topic info /amcl_pose >/dev/null 2>&1; do
  sleep 1
  TIMEOUT=$((TIMEOUT - 1))
  if [[ $TIMEOUT -le 0 ]]; then
    echo "ERROR: AMCL never registered /amcl_pose. Check /tmp/amcl.log."
    kill_carto_group
    exit 1
  fi
done
# Wait until AMCL is genuinely ready by checking its lifecycle state, not just
# topic existence. Topic info appears as soon as the node is created; the node
# isn't actually accepting /initialpose until Activated.
echo "      Waiting for AMCL lifecycle to reach 'active'..."
ACTIVE_TIMEOUT=30
while true; do
  state=$(ros2 lifecycle get /amcl 2>/dev/null | head -1 || true)
  if [[ "$state" == *"active"* ]]; then
    break
  fi
  sleep 1
  ACTIVE_TIMEOUT=$((ACTIVE_TIMEOUT - 1))
  if [[ $ACTIVE_TIMEOUT -le 0 ]]; then
    echo "WARN: AMCL not 'active' after 30s (state: $state). Seeding anyway."
    break
  fi
done
sleep 1  # AMCL needs a beat after Activate to wire its /initialpose subscriber

echo "[6/6] Seeding /initialpose with captured pose..."
# Publish repeatedly for ~3 seconds so AMCL definitely catches one message
# regardless of late subscriber registration.
ros2 topic pub -r 2 --times 6 /initialpose geometry_msgs/PoseWithCovarianceStamped "{
  header: {frame_id: 'map'},
  pose: {
    pose: {
      position: {x: $X, y: $Y, z: 0.0},
      orientation: {x: 0.0, y: 0.0, z: $QZ, w: $QW}
    },
    covariance: [0.25, 0, 0, 0, 0, 0,
                 0, 0.25, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0.07]
  }
}" >/dev/null 2>&1

echo ""
echo "=================================================================="
echo "  AMCL is live at ($X, $Y)."
echo "  Map:   $MAP_DIR/$MAP_NAME.yaml"
echo "  Pose:  /tmp/final_pose.txt"
echo "  Logs:  /tmp/carto.log  /tmp/amcl.log"
echo ""
echo "  Ctrl-C to stop AMCL."
echo "=================================================================="

# Block on AMCL. The EXIT trap above handles cleanup of the AMCL process group
# whether we exit normally, on signal, or on error — so foxglove_bridge /
# manual_drive in other terminals stay alive, but every leftover AMCL process
# dies with this script.
wait $AMCL_PID
