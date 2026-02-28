#!/usr/bin/env python3
"""
stanley_live_plot.py
====================
Live comparison plot: Pure Pursuit vs Stanley vs Blended steering.

Shows three rows:
  Row 1: Steering — pp_delta (blue) vs stanley_delta (orange) vs blended (green)
  Row 2: Cross-track error (meters) from lane Stanley
  Row 3: Heading error (radians) + Stanley trust level

Usage:
  python3 stanley_live_plot.py

Press Ctrl+C or close the window to exit.
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


WINDOW_SEC  = 40
INTERVAL_MS = 100
MAXLEN      = int(WINDOW_SEC / (INTERVAL_MS / 1000) * 1.5)


class PlotData:
    def __init__(self):
        self.lock = threading.Lock()
        self._t0  = None

        self.t_pp      = deque(maxlen=MAXLEN)
        self.v_pp      = deque(maxlen=MAXLEN)

        self.t_st      = deque(maxlen=MAXLEN)
        self.v_st      = deque(maxlen=MAXLEN)

        self.t_bl      = deque(maxlen=MAXLEN)
        self.v_bl      = deque(maxlen=MAXLEN)

        self.t_alpha   = deque(maxlen=MAXLEN)
        self.v_alpha   = deque(maxlen=MAXLEN)

        self.t_cte     = deque(maxlen=MAXLEN)
        self.v_cte     = deque(maxlen=MAXLEN)

        self.t_he      = deque(maxlen=MAXLEN)
        self.v_he      = deque(maxlen=MAXLEN)

        self.t_trust   = deque(maxlen=MAXLEN)
        self.v_trust   = deque(maxlen=MAXLEN)

    def elapsed(self, now_ns):
        if self._t0 is None:
            self._t0 = now_ns
        return (now_ns - self._t0) / 1e9


class PlotNode(Node):
    def __init__(self, data: PlotData):
        super().__init__('stanley_live_plot')
        self.data = data

        def sub(topic, cb):
            self.create_subscription(Float32, topic, cb, 10)

        sub('/nav/pp_delta',              self._pp_cb)
        sub('/nav/stanley_delta',         self._st_cb)
        sub('/nav/blended_delta',         self._bl_cb)
        sub('/nav/blend_alpha',           self._alpha_cb)
        sub('/lane_stanley/cte',          self._cte_cb)
        sub('/lane_stanley/heading_error', self._he_cb)
        sub('/lane_stanley/trust',        self._trust_cb)

    def _t(self):
        return self.data.elapsed(self.get_clock().now().nanoseconds)

    def _pp_cb(self, m):
        t = self._t()
        with self.data.lock:
            self.data.t_pp.append(t); self.data.v_pp.append(m.data)

    def _st_cb(self, m):
        t = self._t()
        with self.data.lock:
            self.data.t_st.append(t); self.data.v_st.append(m.data)

    def _bl_cb(self, m):
        t = self._t()
        with self.data.lock:
            self.data.t_bl.append(t); self.data.v_bl.append(m.data)

    def _alpha_cb(self, m):
        t = self._t()
        with self.data.lock:
            self.data.t_alpha.append(t); self.data.v_alpha.append(m.data)

    def _cte_cb(self, m):
        t = self._t()
        with self.data.lock:
            self.data.t_cte.append(t); self.data.v_cte.append(m.data)

    def _he_cb(self, m):
        t = self._t()
        with self.data.lock:
            self.data.t_he.append(t); self.data.v_he.append(m.data)

    def _trust_cb(self, m):
        t = self._t()
        with self.data.lock:
            self.data.t_trust.append(t); self.data.v_trust.append(m.data)


# ---------------------------------------------------------------------------
# Plot setup
# ---------------------------------------------------------------------------

BG       = '#1e1e1e'
PANEL    = '#2d2d2d'
GRID_C   = '#444444'
C_PP     = '#4fc3f7'   # light blue  — pure pursuit
C_ST     = '#ffb74d'   # orange      — stanley
C_BLEND  = '#81c784'   # green       — blended output
C_ALPHA  = '#ce93d8'   # purple      — blend alpha
C_CTE    = '#ff6b6b'   # red         — CTE
C_HE     = '#ffe66d'   # yellow      — heading error
C_TRUST  = '#4ecdc4'   # teal        — trust


def style_ax(ax, ylabel, ycolor, ylim):
    ax.set_facecolor(PANEL)
    ax.set_ylabel(ylabel, color=ycolor, fontsize=9)
    ax.set_ylim(*ylim)
    ax.tick_params(colors='#cccccc', labelsize=8)
    ax.yaxis.label.set_color(ycolor)
    for sp in ax.spines.values():
        sp.set_color('#555555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color=GRID_C)


def main():
    rclpy.init()
    data = PlotData()
    node = PlotNode(data)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor(BG)
    fig.suptitle('Pure Pursuit vs Stanley — Live Steering Monitor',
                 fontsize=13, fontweight='bold', color='#eeeeee')

    # ---- Row 1: Steering commands ----
    style_ax(ax1, 'Steering (rad)', C_BLEND, (-0.7, 0.7))
    l_pp,    = ax1.plot([], [], color=C_PP,    linewidth=1.5, label='pure pursuit', alpha=0.85)
    l_st,    = ax1.plot([], [], color=C_ST,    linewidth=1.5, label='stanley',      alpha=0.85)
    l_blend, = ax1.plot([], [], color=C_BLEND, linewidth=2.2, label='blended',      zorder=5)
    ax1.axhline(0, color='#666666', linewidth=0.8, linestyle='--')
    ax1_r = ax1.twinx()
    ax1_r.set_facecolor(PANEL)
    ax1_r.set_ylim(-0.05, 1.1)
    ax1_r.set_ylabel('blend α', color=C_ALPHA, fontsize=8)
    ax1_r.tick_params(colors='#cccccc', labelsize=7)
    l_alpha, = ax1_r.plot([], [], color=C_ALPHA, linewidth=1.0,
                          linestyle=':', label='blend α', alpha=0.7)
    # Combined legend
    lines  = [l_pp, l_st, l_blend, l_alpha]
    labels = ['pure pursuit', 'stanley', 'blended', 'blend α']
    ax1.legend(lines, labels, loc='upper right', fontsize=7,
               facecolor=PANEL, edgecolor='#555555', labelcolor='#cccccc')

    # ---- Row 2: CTE ----
    style_ax(ax2, 'CTE (m)', C_CTE, (-2.0, 2.0))
    l_cte, = ax2.plot([], [], color=C_CTE, linewidth=1.5, label='cross-track error')
    ax2.axhline(0, color='#666666', linewidth=0.8, linestyle='--')
    ax2.legend(loc='upper right', fontsize=7, facecolor=PANEL,
               edgecolor='#555555', labelcolor='#cccccc')

    # ---- Row 3: Heading error + trust ----
    style_ax(ax3, 'Heading err (rad)', C_HE, (-0.8, 0.8))
    l_he,    = ax3.plot([], [], color=C_HE,    linewidth=1.5, label='heading error')
    ax3.axhline(0, color='#666666', linewidth=0.8, linestyle='--')
    ax3.set_xlabel('time (s)', color='#cccccc', fontsize=9)
    ax3_r = ax3.twinx()
    ax3_r.set_facecolor(PANEL)
    ax3_r.set_ylim(-0.05, 1.1)
    ax3_r.set_ylabel('trust', color=C_TRUST, fontsize=8)
    ax3_r.tick_params(colors='#cccccc', labelsize=7)
    l_trust, = ax3_r.plot([], [], color=C_TRUST, linewidth=1.2,
                           linestyle='--', label='trust', alpha=0.8)
    lines3  = [l_he, l_trust]
    labels3 = ['heading error', 'trust']
    ax3.legend(lines3, labels3, loc='upper right', fontsize=7,
               facecolor=PANEL, edgecolor='#555555', labelcolor='#cccccc')

    all_axes = [ax1, ax2, ax3, ax1_r, ax3_r]

    def update(_frame):
        with data.lock:
            t_pp    = list(data.t_pp);    v_pp    = list(data.v_pp)
            t_st    = list(data.t_st);    v_st    = list(data.v_st)
            t_bl    = list(data.t_bl);    v_bl    = list(data.v_bl)
            t_al    = list(data.t_alpha); v_al    = list(data.v_alpha)
            t_cte   = list(data.t_cte);   v_cte   = list(data.v_cte)
            t_he    = list(data.t_he);    v_he    = list(data.v_he)
            t_tr    = list(data.t_trust); v_tr    = list(data.v_trust)

        if t_pp:    l_pp.set_data(t_pp, v_pp)
        if t_st:    l_st.set_data(t_st, v_st)
        if t_bl:    l_blend.set_data(t_bl, v_bl)
        if t_al:    l_alpha.set_data(t_al, v_al)
        if t_cte:   l_cte.set_data(t_cte, v_cte)
        if t_he:    l_he.set_data(t_he, v_he)
        if t_tr:    l_trust.set_data(t_tr, v_tr)

        all_t = t_pp + t_st + t_bl + t_cte + t_he + t_tr
        if all_t:
            t_max = max(all_t)
            t_min = max(0.0, t_max - WINDOW_SEC)
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(t_min, t_max + 0.5)

        return l_pp, l_st, l_blend, l_alpha, l_cte, l_he, l_trust

    ani = animation.FuncAnimation(
        fig, update, interval=INTERVAL_MS, blit=False, cache_frame_data=False)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()