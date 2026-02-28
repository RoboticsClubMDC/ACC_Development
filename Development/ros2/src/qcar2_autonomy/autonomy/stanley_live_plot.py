#!/usr/bin/env python3
"""stanley_live_plot.py — Live steering comparison plot.
Matches Gabriel's vo_live_plot visual style exactly.
"""

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


class PlotData:
    def __init__(self, maxlen=300):
        self.t_pp  = deque(maxlen=maxlen); self.v_pp  = deque(maxlen=maxlen)
        self.t_st  = deque(maxlen=maxlen); self.v_st  = deque(maxlen=maxlen)
        self.t_bl  = deque(maxlen=maxlen); self.v_bl  = deque(maxlen=maxlen)
        self.t_cte = deque(maxlen=maxlen); self.v_cte = deque(maxlen=maxlen)
        self.t_al  = deque(maxlen=maxlen); self.v_al  = deque(maxlen=maxlen)
        self.mode  = 'WAITING'
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
        self.create_subscription(Float32, '/nav/pp_delta',      self._pp_cb,  10)
        self.create_subscription(Float32, '/nav/stanley_delta', self._st_cb,  10)
        self.create_subscription(Float32, '/nav/blended_delta', self._bl_cb,  10)
        self.create_subscription(Float32, '/nav/blend_alpha',   self._al_cb,  10)
        self.create_subscription(Float32, '/lane_stanley/cte',  self._cte_cb, 10)

    def _t(self):
        return self.data._elapsed(self.get_clock().now().nanoseconds)

    def _pp_cb(self, msg):
        t = self._t()
        with self.data.lock:
            self.data.t_pp.append(t); self.data.v_pp.append(msg.data)

    def _st_cb(self, msg):
        t = self._t()
        with self.data.lock:
            self.data.t_st.append(t); self.data.v_st.append(msg.data)

    def _bl_cb(self, msg):
        t = self._t()
        with self.data.lock:
            self.data.t_bl.append(t); self.data.v_bl.append(msg.data)

    def _al_cb(self, msg):
        t = self._t()
        with self.data.lock:
            self.data.t_al.append(t); self.data.v_al.append(msg.data)
            a = msg.data
            if a < 0.05:
                self.data.mode = 'PURE PURSUIT ONLY'
            elif a < 0.4:
                self.data.mode = 'PP + STANLEY (low blend)'
            else:
                self.data.mode = 'PP + STANLEY (active)'

    def _cte_cb(self, msg):
        t = self._t()
        with self.data.lock:
            self.data.t_cte.append(t); self.data.v_cte.append(msg.data)


def main():
    rclpy.init()
    data = PlotData(maxlen=int(WINDOW_SEC / (INTERVAL_MS / 1000) * 2))
    node = PlotNode(data)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle('Pure Pursuit vs Stanley — Live', fontsize=13, fontweight='bold')
    fig.patch.set_facecolor('#1e1e1e')

    for ax in (ax1, ax2):
        ax.set_facecolor('#2d2d2d')
        ax.tick_params(colors='#cccccc', labelsize=8)
        ax.spines['bottom'].set_color('#555555')
        ax.spines['left'].set_color('#555555')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, color='#666666')

    # ── Row 1: steering ──
    line_pp, = ax1.plot([], [], color='#4fc3f7', linewidth=1.5, label='pure pursuit')
    line_st, = ax1.plot([], [], color='#ffb74d', linewidth=1.5, label='stanley')
    line_bl, = ax1.plot([], [], color='#81c784', linewidth=2.0, label='blended')
    ax1.axhline(y=0, color='#555555', linestyle='--', alpha=0.5)
    ax1.set_ylabel('steering (rad)', color='#cccccc', fontsize=9)
    ax1.set_ylim(-0.7, 0.7)
    ax1.legend(loc='upper right', fontsize=7, facecolor='#2d2d2d',
               edgecolor='#555555', labelcolor='#cccccc')

    # ── Row 2: CTE ──
    CTE_THRESH = 0.3   # ← change this to adjust the margin lines (meters)
    line_cte, = ax2.plot([], [], color='#ff6b6b', linewidth=1.5, label='cross-track error')
    ax2.axhline(y=0,           color='#555555', linestyle='-',  alpha=0.5)
    ax2.axhline(y= CTE_THRESH, color='#ff6b6b', linestyle='--', alpha=0.7, label=f'±{CTE_THRESH}m margin')
    ax2.axhline(y=-CTE_THRESH, color='#ff6b6b', linestyle='--', alpha=0.7)
    ax2.set_ylabel('CTE (m)', color='#ff6b6b', fontsize=9)
    ax2.set_ylim(-2.0, 2.0)
    ax2.set_xlabel('time (s)', color='#cccccc', fontsize=9)
    ax2.legend(loc='upper right', fontsize=7, facecolor='#2d2d2d',
               edgecolor='#555555', labelcolor='#cccccc')

    mode_text = fig.text(0.5, 0.01, 'Mode: WAITING', ha='center',
                         fontsize=11, fontweight='bold', color='#ffe66d',
                         fontfamily='monospace')

    MODE_COLORS = {
        'PURE PURSUIT ONLY':        '#4fc3f7',
        'PP + STANLEY (low blend)': '#ffb74d',
        'PP + STANLEY (active)':    '#81c784',
        'WAITING':                  '#78909c',
    }

    def update(frame):
        with data.lock:
            t_pp  = list(data.t_pp);  v_pp  = list(data.v_pp)
            t_st  = list(data.t_st);  v_st  = list(data.v_st)
            t_bl  = list(data.t_bl);  v_bl  = list(data.v_bl)
            t_cte = list(data.t_cte); v_cte = list(data.v_cte)
            mode  = data.mode

        if t_pp:  line_pp.set_data(t_pp, v_pp)
        if t_st:  line_st.set_data(t_st, v_st)
        if t_bl:  line_bl.set_data(t_bl, v_bl)
        if t_cte: line_cte.set_data(t_cte, v_cte)

        all_t = t_pp + t_st + t_bl + t_cte
        if all_t:
            t_max = max(all_t)
            t_min = max(0, t_max - WINDOW_SEC)
            for ax in (ax1, ax2):
                ax.set_xlim(t_min, t_max + 1)

        color = MODE_COLORS.get(mode, '#cccccc')
        mode_text.set_text(f'Mode: {mode}')
        mode_text.set_color(color)

        return line_pp, line_st, line_bl, line_cte, mode_text

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