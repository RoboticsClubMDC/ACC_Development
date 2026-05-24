#!/usr/bin/env bash
# ros2_killall.sh — graceful → forceful kill of all QCar2-related ROS 2
# processes, plus a daemon reset to clear stale node-graph entries.
#
# Usage:
#   1. Source this file (manually or from ~/.bashrc):
#        source /workspaces/isaac_ros-dev/ros2/scripts/ros2_killall.sh
#   2. Call between launches when you want a clean slate:
#        ros2_killall
#
# Why three signal stages: SIGINT first (same as Ctrl-C) lets ros2 launch
# unwind its child tree cleanly. SIGTERM next is graceful for plain
# executables. SIGKILL is last-resort. The ros2 daemon stop/start at the
# end resets the discovery cache so `ros2 node list` doesn't report
# already-dead nodes.
#
# Note: this does NOT touch processes inside the virtual-qcar2 (QLabs)
# container, since those run in their own PID namespace. To clean those:
#   sudo docker exec virtual-qcar2 pkill -f "csi_camera|foxglove_bridge|ros2"

ros2_killall() {
    pkill -INT  -f "ros2 launch" 2>/dev/null
    pkill -INT  -f "ros2 run"    2>/dev/null
    sleep 2
    pkill -TERM -f "qcar2|nav2_qcar2|foxglove_bridge|fixed_lidar|pose_estimator|cartographer|amcl|lifecycle_manager|map_server|nav2_map_server|ekf_fusor" 2>/dev/null
    sleep 1
    pkill -KILL -f "qcar2|nav2_qcar2|foxglove_bridge|fixed_lidar|pose_estimator|cartographer|amcl|lifecycle_manager|map_server|nav2_map_server|ekf_fusor" 2>/dev/null
    ros2 daemon stop 2>/dev/null
    ros2 daemon start
    echo "ROS 2 processes killed. Remaining:"
    ps -ef | grep -E "qcar2|ros2 (launch|run)|foxglove_bridge|cartographer|amcl|lifecycle_manager|map_server|ekf_fusor" | grep -v grep || echo "  (none)"
}
