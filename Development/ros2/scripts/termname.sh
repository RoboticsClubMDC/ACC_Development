#!/usr/bin/env bash
# termname.sh — set the terminal window/tab title.
#
# Usage:
#   1. Source this file (manually or from ~/.bashrc):
#        source /workspaces/isaac_ros-dev/ros2/scripts/termname.sh
#   2. Call termname with the title you want:
#        termname "QCar2 Cartographer"
#        termname "QCar2 EKF Fusor"
#        termname "QCar2 Manual Drive"
#
# This works in GNOME Terminal, xterm, KDE Konsole, VSCode integrated
# terminal, tmux, and most other ANSI-aware terminals. The escape sequence
# \033]0;TITLE\007 sets both the window title and the tab title.
#
# If you want this available in every new shell, add to ~/.bashrc:
#   source /workspaces/isaac_ros-dev/ros2/scripts/termname.sh

termname() {
    if [[ -z "$1" ]]; then
        echo "Usage: termname \"<title>\""
        return 1
    fi
    printf '\033]0;%s\007' "$1"
}
