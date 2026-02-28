#!/usr/bin/env python3
"""vo_live_plot.py — Lightweight live plotter for VO redundancy signals.

Replaces rqt_plot when it's not available in the container.
Uses matplotlib animation to show three signals in real time:
  - delta_trans (residual rho) — how much VO and odom disagree
  - vo_weight — how much we trust VO right now
  - trust_level — supervisor's overall navigation trust

Usage:
  export ROS_DOMAIN_ID=67
  python3 vo_live_plot.py

Press Ctrl+C or close the window to stop.
"""

import threading
from collections import deque

import matplotlib
matplotlib.use('TkAgg')  # works in most X11 environments
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String


# How many seconds of history to show
WINDOW_SEC = 30
# Plot update rate (ms)
INTERVAL_MS = 200


class PlotData:
    """Thread-safe ring buffers for plot data."""
    def __init__(self, maxlen=300):
        self.t_rho = deque(maxlen=maxlen)
        self.v_rho = deque(maxlen=maxlen)
        self.t_weight = deque(maxlen=maxlen)
        self.v_weight = deque(maxlen=maxlen)
        self.t_trust = deque(maxlen=maxlen)
        self.v_trust = deque(maxlen=maxlen)
        self.mode = "WAITING"
        self.lock = threading.Lock()
        self._t0 = None

    def _elapsed(self, now_ns):
        if self._t0 is None:
            self._t0 = now_ns
        return (now_ns - self._t0) / 1e9


class PlotNode(Node):
    def __init__(self, data):
        super().__init__('vo_live_plot')
        self.data = data
        self.create_subscription(
            Float32, '/vo/delta_trans', self._rho_cb, 10)
        self.create_subscription(
            Float32, '/vo/vo_weight', self._weight_cb, 10)
        self.create_subscription(
            Float32, '/vo/supervisor/trust_level', self._trust_cb, 10)
        self.create_subscription(
            String, '/vo/supervisor/mode', self._mode_cb, 10)

    def _rho_cb(self, msg):
        t = self.data._elapsed(self.get_clock().now().nanoseconds)
        with self.data.lock:
            self.data.t_rho.append(t)
            self.data.v_rho.append(msg.data)

    def _weight_cb(self, msg):
        t = self.data._elapsed(self.get_clock().now().nanoseconds)
        with self.data.lock:
            self.data.t_weight.append(t)
            self.data.v_weight.append(msg.data)

    def _trust_cb(self, msg):
        t = self.data._elapsed(self.get_clock().now().nanoseconds)
        with self.data.lock:
            self.data.t_trust.append(t)
            self.data.v_trust.append(msg.data)

    def _mode_cb(self, msg):
        with self.data.lock:
            self.data.mode = msg.data


def main():
    rclpy.init()
    data = PlotData(maxlen=int(WINDOW_SEC / (INTERVAL_MS / 1000) * 2))
    node = PlotNode(data)

    # Spin ROS in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # ── Matplotlib setup ──
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.suptitle('VO Redundancy Monitor — Live', fontsize=13, fontweight='bold')
    fig.patch.set_facecolor('#1e1e1e')

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor('#2d2d2d')
        ax.tick_params(colors='#cccccc', labelsize=8)
        ax.spines['bottom'].set_color('#555555')
        ax.spines['left'].set_color('#555555')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, color='#666666')

    # Residual plot
    line_rho, = ax1.plot([], [], color='#ff6b6b', linewidth=1.5, label='residual ρ')
    thresh_line = ax1.axhline(y=0.35, color='#ff6b6b', linestyle='--', alpha=0.5, label='threshold')
    ax1.set_ylabel('delta_trans (m)', color='#ff6b6b', fontsize=9)
    ax1.set_ylim(-0.05, 1.0)
    ax1.legend(loc='upper right', fontsize=7, facecolor='#2d2d2d', edgecolor='#555555', labelcolor='#cccccc')

    # Weight plot
    line_w, = ax2.plot([], [], color='#4ecdc4', linewidth=1.5, label='vo_weight')
    weight_thresh = ax2.axhline(y=0.30, color='#4ecdc4', linestyle='--', alpha=0.5, label='min_weight')
    ax2.set_ylabel('vo_weight', color='#4ecdc4', fontsize=9)
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(loc='upper right', fontsize=7, facecolor='#2d2d2d', edgecolor='#555555', labelcolor='#cccccc')

    # Trust level plot
    line_t, = ax3.plot([], [], color='#ffe66d', linewidth=1.5, label='trust_level')
    ax3.set_ylabel('trust_level', color='#ffe66d', fontsize=9)
    ax3.set_ylim(-0.05, 1.1)
    ax3.set_xlabel('time (s)', color='#cccccc', fontsize=9)
    ax3.legend(loc='upper right', fontsize=7, facecolor='#2d2d2d', edgecolor='#555555', labelcolor='#cccccc')

    # Mode text annotation
    mode_text = fig.text(0.5, 0.01, 'Mode: WAITING', ha='center',
                         fontsize=11, fontweight='bold', color='#ffe66d',
                         fontfamily='monospace')

    MODE_COLORS = {
        'NORMAL': '#4ecdc4',
        'VO_UNTRUSTED': '#ffa726',
        'ODOM_WARNING': '#ff7043',
        'ODOM_FAULT': '#ff1744',
        'INITIALIZING': '#78909c',
        'DEGRADED': '#e040fb',
        'WAITING': '#78909c',
    }

    def update(frame):
        with data.lock:
            tr = list(data.t_rho)
            vr = list(data.v_rho)
            tw = list(data.t_weight)
            vw = list(data.v_weight)
            tt = list(data.t_trust)
            vt = list(data.v_trust)
            mode = data.mode

        if tr:
            line_rho.set_data(tr, vr)
        if tw:
            line_w.set_data(tw, vw)
        if tt:
            line_t.set_data(tt, vt)

        # Auto-scroll x axis
        all_t = tr + tw + tt
        if all_t:
            t_max = max(all_t)
            t_min = max(0, t_max - WINDOW_SEC)
            for ax in (ax1, ax2, ax3):
                ax.set_xlim(t_min, t_max + 1)

        # Update mode display
        color = MODE_COLORS.get(mode, '#cccccc')
        mode_text.set_text(f'Mode: {mode}')
        mode_text.set_color(color)

        return line_rho, line_w, line_t, mode_text

    ani = animation.FuncAnimation(
        fig, update, interval=INTERVAL_MS, blit=False, cache_frame_data=False)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()