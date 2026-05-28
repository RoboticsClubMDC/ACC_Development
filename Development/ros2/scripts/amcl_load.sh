#!/usr/bin/env bash
# amcl_load.sh — launch AMCL against a previously saved Cartographer map
# (no re-recording). Seeds /initialpose automatically from the last pose
# saved by carto_to_amcl.sh.
#
# Flow:
#   1. Reads ~/qcar2_maps/${MAP_NAME}_pose.yaml (final_pose from last
#      carto_to_amcl.sh run).
#   2. Launches qcar2_amcl_localization_*_launch.py with the saved map.
#   3. Waits for AMCL lifecycle to be 'active'.
#   4. Publishes /initialpose with the captured last pose.
#   5. Blocks; Ctrl-C to stop AMCL.
#
# Usage (inside the Isaac ROS dev container):
#   ./amcl_load.sh                  # default: virtual, last pose, HUB=node 10
#   ./amcl_load.sh physical         # physical robot, last pose
#   ./amcl_load.sh virtual initial  # seed with initial (carto-start) pose
#   ./amcl_load.sh virtual final 10 # override the HUB node (default 10)
#
# Seeds /initialpose with the saved pose as AMCL's initial guess — this is the
# ACTUAL pose the car is parked at, refined by AMCL's first scan. The temp-start
# (-1) node then reads the live AMCL-corrected /qcar2_pose_fused, so navigation
# starts from the true first-scan pose, not the raw YAML number.
#
# After seeding it runs the SAME automatic first-trip-to-HUB + LED sequence as
# carto_to_amcl.sh: CYAN (localizing) → GREEN (drive [-1, HUB] with the STANLEY
# lane stack; Stanley gated off on 8→10/10→1) → arrival-align → MAGENTA (parked
# at HUB, ready for rides). Set STANLEY=0 for a bare PP-only first trip.
#
# Pre-req: ~/qcar2_maps/competition_map.{yaml,pgm} and the companion
# competition_map_pose.yaml must already exist (i.e., carto_to_amcl.sh
# ran successfully at least once).

set -e

MODE="${1:-virtual}"
# Default pose is chosen AFTER we can read the YAML (prefer the saved HUB
# parked pose = where the car actually ends each session). 2nd arg overrides:
# initial | final | hub.
WHICH_POSE="${2:-}"
# Optional 3rd arg overrides the HUB node (default 10) for the auto first-trip.
export HUB_NODE="${3:-10}"
MAP_DIR="${HOME}/qcar2_maps"
MAP_NAME="competition_map"
MAP_YAML="$MAP_DIR/$MAP_NAME.yaml"
POSE_YAML="$MAP_DIR/${MAP_NAME}_pose.yaml"

# Shared LED + first-trip-to-HUB helper (set_led, nav_to_hub).
source "$(dirname "$0")/hub_handoff.sh"

case "$MODE" in
  virtual)  AMCL_LAUNCH="qcar2_amcl_localization_virtual_launch.py" ;;
  physical) AMCL_LAUNCH="qcar2_amcl_localization_launch.py" ;;
  *) echo "Usage: $0 [virtual|physical] [initial|final]"; exit 1 ;;
esac

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ERROR: 'ros2' not in PATH. Source ROS and your workspace first."
  exit 1
fi
if [[ -z "${ROS_DOMAIN_ID:-}" ]]; then
  echo "WARNING: ROS_DOMAIN_ID is not set (workspace uses 69). Exiting."
  exit 1
fi
if [[ ! -f "$MAP_YAML" ]]; then
  echo "ERROR: $MAP_YAML not found. Run carto_to_amcl.sh first."
  exit 1
fi
if [[ ! -f "$POSE_YAML" ]]; then
  echo "ERROR: $POSE_YAML not found. Run carto_to_amcl.sh (recent version) first."
  exit 1
fi

# Default pose: prefer the saved HUB parked pose (the car's ACTUAL position at
# the end of each session = where it sits on reload). If carto didn't save one
# (older run / NAV_TO_HUB=0), fall back to the final (near-node-0) seed.
if [[ -z "$WHICH_POSE" ]]; then
  if grep -q '^hub_pose:' "$POSE_YAML"; then
    WHICH_POSE="hub"
  else
    WHICH_POSE="final"
  fi
fi
case "$WHICH_POSE" in
  initial|final|hub) ;;
  *) echo "Usage: $0 [virtual|physical] [initial|final|hub]"; exit 1 ;;
esac
if [[ "$WHICH_POSE" == "hub" ]] && ! grep -q '^hub_pose:' "$POSE_YAML"; then
  echo "WARN: no hub_pose in $POSE_YAML — falling back to 'final'."
  WHICH_POSE="final"
fi
echo "Seeding from '$WHICH_POSE' pose (the car should be physically there)."

# When seeding the HUB pose, the car is ALREADY parked at the HUB → tell
# nav_to_hub to skip the drive and just confirm + MAGENTA.
if [[ "$WHICH_POSE" == "hub" ]]; then
  export AT_HUB=1
fi

# Parse the chosen pose block from the YAML.
# YAML is short and rigid; awk is enough. We grab the first {x: A, y: B}
# from the requested block.
SECTION="${WHICH_POSE}_pose"
POSE_BLOCK=$(awk -v sec="$SECTION:" '
  $0 ~ sec {found=1; next}
  found && /^[a-z]/ {found=0}
  found {print}
' "$POSE_YAML")

X=$(echo "$POSE_BLOCK"  | grep position    | sed -E 's/.*x:\s*([-0-9.eE+]+).*/\1/')
Y=$(echo "$POSE_BLOCK"  | grep position    | sed -E 's/.*y:\s*([-0-9.eE+]+).*/\1/')
QZ=$(echo "$POSE_BLOCK" | grep orientation | sed -E 's/.*z:\s*([-0-9.eE+]+).*/\1/')
QW=$(echo "$POSE_BLOCK" | grep orientation | sed -E 's/.*w:\s*([-0-9.eE+]+).*/\1/')

if [[ -z "$X" || -z "$Y" || -z "$QZ" || -z "$QW" ]]; then
  echo "ERROR: could not parse $SECTION from $POSE_YAML"
  cat "$POSE_YAML"
  exit 1
fi

echo "Seed pose ($WHICH_POSE):  x=$X  y=$Y  qz=$QZ  qw=$QW"
echo "Map:                     $MAP_YAML"

# Defensive: refuse to start if AMCL is already up.
if ros2 node list 2>/dev/null | grep -q "^/amcl$"; then
  echo "ERROR: an /amcl node is already running. Stop it first."
  exit 1
fi

echo "[1/3] Launching AMCL ($AMCL_LAUNCH)..."
setsid ros2 launch qcar2_nodes "$AMCL_LAUNCH" map:="$MAP_YAML" >/tmp/amcl.log 2>&1 &
AMCL_PID=$!
AMCL_PGID=$AMCL_PID

trap 'echo ""; echo "Cleaning up..."; \
      [[ -n "$PF_PGID" ]] && { kill -INT -"$PF_PGID" 2>/dev/null || true; }; \
      kill -INT -"$AMCL_PGID" 2>/dev/null || true; sleep 2; \
      [[ -n "$PF_PGID" ]] && { kill -KILL -"$PF_PGID" 2>/dev/null || true; }; \
      kill -KILL -"$AMCL_PGID" 2>/dev/null || true; \
      exit' INT TERM EXIT

echo "      Waiting for /amcl_pose..."
TIMEOUT=60
while ! ros2 topic info /amcl_pose >/dev/null 2>&1; do
  sleep 1
  TIMEOUT=$((TIMEOUT - 1))
  if [[ $TIMEOUT -le 0 ]]; then
    echo "ERROR: AMCL never registered /amcl_pose. Check /tmp/amcl.log."
    exit 1
  fi
done

echo "[2/3] Waiting for AMCL lifecycle 'active'..."
ACTIVE_TIMEOUT=30
while true; do
  state=$(ros2 lifecycle get /amcl 2>/dev/null | head -1 || true)
  if [[ "$state" == *"active"* ]]; then break; fi
  sleep 1
  ACTIVE_TIMEOUT=$((ACTIVE_TIMEOUT - 1))
  if [[ $ACTIVE_TIMEOUT -le 0 ]]; then
    echo "WARN: AMCL not 'active' after 30s (state: $state). Seeding anyway."
    break
  fi
done
sleep 1

# Hardware is up → LED CYAN while we seed + AMCL settles on its first scan.
set_led "$LED_CYAN"

echo "[3/3] Seeding /initialpose with $WHICH_POSE pose (AMCL's initial guess;"
echo "      the actual first-scan pose; the -1 node reads the refined fused pose)..."
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

# Automatic first trip to HUB (LED GREEN → drive [-1, HUB] → align → MAGENTA).
echo ""
if [[ "${NAV_TO_HUB:-1}" == "1" ]]; then
  nav_to_hub || echo "[hub] WARN: first-trip-to-HUB did not complete; AMCL stays up."
else
  set_led "$LED_GREEN"
  echo "[hub] NAV_TO_HUB=0 — skipping auto path_follower. Bring up your own stack"
  echo "      (e.g. lane+Stanley) and set node_values \"[-1, $HUB_NODE]\" yourself."
fi

echo ""
echo "=================================================================="
echo "  AMCL is live (seeded $WHICH_POSE pose at $X, $Y)."
echo "  Map:    $MAP_YAML"
echo "  Pose:   $POSE_YAML  (selected: $WHICH_POSE)"
echo "  Logs:   /tmp/amcl.log  /tmp/path_follower.log"
echo "  Parked at HUB node $HUB_NODE (MAGENTA). Ready for RIDES behavior."
echo "  Ctrl-C to stop AMCL + path_follower."
echo "=================================================================="

wait $AMCL_PID
