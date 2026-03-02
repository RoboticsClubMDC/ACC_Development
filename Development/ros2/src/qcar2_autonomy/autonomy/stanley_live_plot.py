#!/usr/bin/env python3

import threading
from collections import deque

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


WINDOW_SEC  = 30
INTERVAL_MS = 200
CTE_THRESH  = 0.3
CTE_YLIM    = 2.0
HE_YLIM     = 1.0


class PlotData:
    def __init__(self, maxlen=300):
        self.t_cte = deque(maxlen=maxlen); self.v_cte = deque(maxlen=maxlen)
        self.t_he  = deque(maxlen=maxlen); self.v_he  = deque(maxlen=maxlen)
        self.lock  = threading.Lock()
        self._t0   = None

    def _elapsed(self, now_ns):
        if self._t0 is None:
            self._t0 = now_ns
        return (now_ns - self._t0) / 1e9


class PlotNode(Node):
    def __init__(self, data):
        super().__init__('stanley_live_plot')
        self.data = data
        self.create_subscription(Float32, '/lane_stanley/cte',           self._cte_cb, 10)
        self.create_subscription(Float32, '/lane_stanley/heading_error', self._he_cb,  10)

    def _t(self):
        return self.data._elapsed(self.get_clock().now().nanoseconds)

    def _cte_cb(self, msg):
        t = self._t()
        with self.data.lock:
            self.data.t_cte.append(t); self.data.v_cte.append(msg.data)

    def _he_cb(self, msg):
        t = self._t()
        with self.data.lock:
            self.data.t_he.append(t); self.data.v_he.append(msg.data)


def main():
    rclpy.init()
    data = PlotData(maxlen=int(WINDOW_SEC / (INTERVAL_MS / 1000) * 2))
    node = PlotNode(data)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle('Lane Stanley — CTE & Heading Error', fontsize=13, fontweight='bold', color='#cccccc')
    fig.patch.set_facecolor('#1e1e1e')

    for ax in (ax1, ax2):
        ax.set_facecolor('#2d2d2d')
        ax.tick_params(colors='#cccccc', labelsize=8)
        ax.spines['bottom'].set_color('#555555')
        ax.spines['left'].set_color('#555555')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, color='#666666')

    line_cte, = ax1.plot([], [], color='#ff6b6b', linewidth=1.5, label='cross-track error')
    ax1.axhline(y=0,            color='#555555', linestyle='-',  alpha=0.5)
    ax1.axhline(y= CTE_THRESH,  color='#ff6b6b', linestyle='--', alpha=0.6, label=f'±{CTE_THRESH}m')
    ax1.axhline(y=-CTE_THRESH,  color='#ff6b6b', linestyle='--', alpha=0.6)
    ax1.set_ylabel('CTE (m)', color='#ff6b6b', fontsize=9)
    ax1.set_ylim(-CTE_YLIM, CTE_YLIM)
    ax1.legend(loc='upper right', fontsize=7, facecolor='#2d2d2d', edgecolor='#555555', labelcolor='#cccccc')

    line_he, = ax2.plot([], [], color='#ffd54f', linewidth=1.5, label='heading error')
    ax2.axhline(y=0, color='#555555', linestyle='-', alpha=0.5)
    ax2.set_ylabel('heading error (rad)', color='#ffd54f', fontsize=9)
    ax2.set_ylim(-HE_YLIM, HE_YLIM)
    ax2.set_xlabel('time (s)', color='#cccccc', fontsize=9)
    ax2.legend(loc='upper right', fontsize=7, facecolor='#2d2d2d', edgecolor='#555555', labelcolor='#cccccc')

    def update(frame):
        with data.lock:
            t_cte = list(data.t_cte); v_cte = list(data.v_cte)
            t_he  = list(data.t_he);  v_he  = list(data.v_he)

        if t_cte: line_cte.set_data(t_cte, v_cte)
        if t_he:  line_he.set_data(t_he, v_he)

        all_t = t_cte + t_he
        if all_t:
            t_max = max(all_t)
            t_min = max(0, t_max - WINDOW_SEC)
            for ax in (ax1, ax2):
                ax.set_xlim(t_min, t_max + 1)

        return line_cte, line_he

    ani = animation.FuncAnimation(
        fig, update, interval=INTERVAL_MS, blit=False, cache_frame_data=False)

    plt.tight_layout()

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()