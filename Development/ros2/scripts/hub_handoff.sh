#!/usr/bin/env bash
# hub_handoff.sh — SOURCED helper (not run directly) shared by carto_to_amcl.sh
# and amcl_load.sh. Provides:
#   - set_led <id>        : set the QCar LED color via qcar2_hardware param
#   - nav_to_hub          : the automatic "first trip to HUB" before rides:
#                           launch path_follower PP-only, drive the temp-start
#                           (-1) route to the HUB node, wait for the arrival
#                           align to seat the car, then go MAGENTA.
#
# LED sequence the scripts drive (per ACC ride convention):
#   CYAN (4)    — while mapping / localizing / seeding AMCL
#   GREEN (1)   — driving the first trip to the HUB (node -1 → HUB)
#   MAGENTA (5) — parked + seated at the HUB, ready for RIDES behavior
#
# Overridable env: HW_NODE (default /qcar2_hardware), HUB_NODE (default 10).

HW_NODE="${HW_NODE:-/qcar2_hardware}"
HUB_NODE="${HUB_NODE:-10}"
LED_CYAN=4
LED_GREEN=1
LED_MAGENTA=5

# PF_PGID is exported so the caller's cleanup trap can kill path_follower too.
PF_PGID=""

set_led() {
  if ros2 param set "$HW_NODE" led_color_id "$1" >/dev/null 2>&1; then
    echo "      LED → $1"
  else
    echo "      (LED set $1 failed — is $HW_NODE up with a led_color_id param?)"
  fi
}

# Launch path_follower PP-only and run the first trip to the HUB node, seating
# the car on it with the arrival-align maneuver. Blocks until seated (or the
# align times out and path_follower reports complete anyway).
nav_to_hub() {
  set_led "$LED_GREEN"
  echo "[hub] First trip to HUB node $HUB_NODE (temp-start -1 from seeded pose)."
  # DEFAULT = Stanley (lane-following) stack: the -1→HUB first trip already runs
  # the full arbiter lane stack — Stanley on open road, gated OFF on the no-lane
  # legs 8→10 and 10→1, and the arbiter goes PATH-ONLY at node 10 so the
  # arrival-align seats the car. Set STANLEY=0 for a bare PP-only first trip.
  if [[ "${STANLEY:-1}" == "1" ]]; then
    echo "      Stanley stack (lane-primary; Stanley off on 8→10/10→1; PP-align at HUB)."
    echo "      log: /tmp/stanley_stack.log"
    setsid ros2 launch qcar2_autonomy stanley_arbiter_stack_launch.py \
      >/tmp/stanley_stack.log 2>&1 &
    PF_PGID=$!
  else
    # STANLEY=0 → PP-only (cmd_topic=/cmd_vel_nav). Pre-arm node_values=[-1,HUB]
    # so it starts quiet (temp-start-pending) instead of spamming auto-align on
    # the default route; the param set below triggers build + autonomous.
    echo "      STANLEY=0 → PP-only (no lane following)."
    setsid ros2 run qcar2_autonomy path_follower --ros-args \
      -p cmd_topic:=/cmd_vel_nav \
      -p node_values:="[-1, $HUB_NODE]" >/tmp/path_follower.log 2>&1 &
    PF_PGID=$!
  fi

  echo "      Waiting for /path_follower to come up..."
  local t=30
  until ros2 node list 2>/dev/null | grep -q "/path_follower"; do
    sleep 1; t=$((t - 1))
    if [[ $t -le 0 ]]; then
      echo "      WARN: /path_follower never appeared (see /tmp/path_follower.log)."
      return 1
    fi
  done
  # Let AMCL-corrected /qcar2_pose_fused start flowing so the temp-start node
  # lands on the true pose, not a warm-up sample.
  sleep 2

  echo "      Arming temp-start route [-1, $HUB_NODE]..."
  ros2 param set /path_follower node_values "[-1, $HUB_NODE]" >/dev/null 2>&1 \
    || echo "      WARN: failed to set node_values on /path_follower."

  echo "      Driving + aligning onto node $HUB_NODE (tail /tmp/path_follower.log)..."
  # /path_status (Bool) becomes True only AFTER the arrival-align seats the car
  # (or the 20 s align timeout fires). Poll until then.
  until ros2 topic echo /path_status --once 2>/dev/null | grep -q 'data: true'; do
    sleep 1
  done

  set_led "$LED_MAGENTA"
  echo "[hub] Parked + seated at HUB node $HUB_NODE — MAGENTA. Ready for RIDES."
}
