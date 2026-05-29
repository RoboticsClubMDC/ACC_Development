#!/usr/bin/env python3
"""
controller_drive — Logitech gamepad control node for QCar2.

Publishes geometry_msgs/Twist to /cmd_vel_nav (configurable).

Default axis / button map (Logitech F310/F710, XInput mode, Linux):
  Axis 0  : Left stick X      → steering     (-1 = full left, +1 = full right)
  Axis 1  : Left stick Y      → throttle     (use_trigger=False, -1 = up / forward)
  Axis 5  : Right trigger RT  → throttle     (use_trigger=True,  -1 = rest → +1 = full)
  Button 0: A  → reverse
  Button 1: B  → stop / brake
  Button 4: LB → deadman enable (hold to allow driving)

Backend priority: pygame → evdev → raw /dev/input/js0 (stdlib only).
"""


import fcntl
import math
import os
import struct
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty

# ── suppress SDL display / audio init when no display is attached ────────────
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')


# ─────────────────────────────────────────────────────────────────────────────
# Joystick backends
# ─────────────────────────────────────────────────────────────────────────────

class _PygameBackend:
    """Thin wrapper around pygame.joystick."""

    def __init__(self):
        import pygame as _pg
        self._pg = _pg
        _pg.init()
        _pg.joystick.init()
        if _pg.joystick.get_count() == 0:
            raise RuntimeError('pygame: no joystick found')
        self._joy = _pg.joystick.Joystick(0)
        self._joy.init()

    def name(self):        return self._joy.get_name()
    def num_axes(self):    return self._joy.get_numaxes()
    def num_buttons(self): return self._joy.get_numbuttons()

    def pump(self):
        self._pg.event.pump()

    def axis(self, idx):
        try:    return float(self._joy.get_axis(idx))
        except: return 0.0

    def button(self, idx):
        try:    return bool(self._joy.get_button(idx))
        except: return False

    def close(self):
        self._pg.quit()


class _EvdevBackend:
    """evdev-based backend (python3-evdev).  Uses ABS axes and key events."""

    # Standard Linux EV_ABS axis codes for gamepads
    _ABS_MAP = {
        0: 0,   # ABS_X  → axis 0  (left stick X)
        1: 1,   # ABS_Y  → axis 1  (left stick Y)
        2: 2,   # ABS_Z  → axis 2  (LT)
        3: 3,   # ABS_RX → axis 3  (right stick X)
        4: 4,   # ABS_RY → axis 4  (right stick Y)
        5: 5,   # ABS_RZ → axis 5  (RT)
    }
    # BTN_GAMEPAD (304) → button 0, BTN_SOUTH is 304 = A, etc.
    _BTN_BASE = 304

    def __init__(self, path=None):
        import evdev as _ev
        self._ev = _ev
        if path:
            self._dev = _ev.InputDevice(path)
        else:
            devices = [_ev.InputDevice(p) for p in _ev.list_devices()]
            joy = next((d for d in devices if _ev.ecodes.EV_ABS in d.capabilities()), None)
            if joy is None:
                raise RuntimeError('evdev: no gamepad found')
            self._dev = joy
        self._dev.grab()

        caps = self._dev.capabilities()
        abs_info = caps.get(_ev.ecodes.EV_ABS, [])
        self._axes = {}
        for code, info in abs_info:
            idx = self._ABS_MAP.get(code)
            if idx is not None:
                mid = (info.max + info.min) / 2.0
                rng = (info.max - info.min) / 2.0
                self._axes[idx] = {'code': code, 'mid': mid, 'range': rng, 'val': 0.0}

        self._buttons = {}
        for code, _ in caps.get(_ev.ecodes.EV_KEY, []):
            btn_idx = code - self._BTN_BASE
            if 0 <= btn_idx < 32:
                self._buttons[btn_idx] = False

        self._n_axes   = max(self._axes.keys(),   default=-1) + 1
        self._n_buttons = max(self._buttons.keys(), default=-1) + 1

    def name(self):        return self._dev.name
    def num_axes(self):    return self._n_axes
    def num_buttons(self): return self._n_buttons

    def pump(self):
        try:
            for ev in self._dev.read():
                if ev.type == self._ev.ecodes.EV_ABS:
                    for idx, info in self._axes.items():
                        if info['code'] == ev.code:
                            rng = info['range'] or 1.0
                            info['val'] = (ev.value - info['mid']) / rng
                elif ev.type == self._ev.ecodes.EV_KEY:
                    btn = ev.code - self._BTN_BASE
                    if btn in self._buttons:
                        self._buttons[btn] = bool(ev.value)
        except BlockingIOError:
            pass

    def axis(self, idx):
        return self._axes.get(idx, {}).get('val', 0.0)

    def button(self, idx):
        return self._buttons.get(idx, False)

    def close(self):
        try:
            self._dev.ungrab()
        except Exception:
            pass


class _Js0Backend:
    """
    Reads Linux joystick events directly from /dev/input/jsN (stdlib only).

    Event struct: uint32 time | int16 value | uint8 type | uint8 number
    type 0x01 = button (value 0/1)
    type 0x02 = axis   (value -32767..32767)
    0x80 flag = initial-state event (we process these the same way)
    """

    _FMT  = 'IhBB'
    _SIZE = struct.calcsize(_FMT)   # 8 bytes
    _BUTTON = 0x01
    _AXIS   = 0x02
    _INIT   = 0x80

    def __init__(self, path='/dev/input/js0'):
        self._fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, os.O_NONBLOCK)

        # Try to read device name via JSIOCGNAME ioctl (optional)
        try:
            buf = bytearray(128)
            # JSIOCGNAME(128): _IOC(READ=2, 'j'=106, 0x13, 128)
            _JSIOCGNAME = (2 << 30) | (128 << 16) | (106 << 8) | 0x13
            fcntl.ioctl(self._fd, _JSIOCGNAME, buf)
            self._name = buf.rstrip(b'\x00').decode('utf-8', errors='replace')
        except Exception:
            self._name = 'js0 device'

        # Try to read axis / button count via ioctl (optional)
        try:
            _JSIOCGAXES    = (2 << 30) | (1 << 16) | (106 << 8) | 0x11
            _JSIOCGBUTTONS = (2 << 30) | (1 << 16) | (106 << 8) | 0x12
            ax_buf  = bytearray(1)
            btn_buf = bytearray(1)
            fcntl.ioctl(self._fd, _JSIOCGAXES,    ax_buf)
            fcntl.ioctl(self._fd, _JSIOCGBUTTONS, btn_buf)
            self._n_axes   = ax_buf[0]
            self._n_buttons = btn_buf[0]
        except Exception:
            self._n_axes    = 20
            self._n_buttons = 20

        self._axis_state   = [0.0] * max(self._n_axes, 20)
        self._button_state = [False] * max(self._n_buttons, 20)

        # Drain initial-state events so we start with correct values
        self.pump()

    def name(self):        return self._name
    def num_axes(self):    return self._n_axes
    def num_buttons(self): return self._n_buttons

    def pump(self):
        try:
            while True:
                raw = os.read(self._fd, self._SIZE)
                if len(raw) < self._SIZE:
                    break
                _, value, etype, number = struct.unpack(self._FMT, raw)
                core_type = etype & ~self._INIT
                if core_type == self._BUTTON:
                    if number < len(self._button_state):
                        self._button_state[number] = bool(value)
                elif core_type == self._AXIS:
                    if number < len(self._axis_state):
                        self._axis_state[number] = value / 32767.0
        except BlockingIOError:
            pass
        except OSError:
            pass

    def axis(self, idx):
        try:    return self._axis_state[idx]
        except: return 0.0

    def button(self, idx):
        try:    return self._button_state[idx]
        except: return False

    def close(self):
        try:
            os.close(self._fd)
        except Exception:
            pass


def _open_best_backend(logger):
    """Try pygame → evdev → js0, return (backend, backend_name) or (None, '')."""

    # ── pygame ───────────────────────────────────────────────────────────────
    try:
        import pygame  # noqa: F401  (just to test availability)
        b = _PygameBackend()
        return b, 'pygame'
    except ImportError:
        logger.debug('pygame not installed, trying evdev')
    except RuntimeError as e:
        logger.debug(f'pygame: {e}')

    # ── evdev ────────────────────────────────────────────────────────────────
    try:
        import evdev  # noqa: F401
        b = _EvdevBackend()
        return b, 'evdev'
    except ImportError:
        logger.debug('evdev not installed, trying raw js0')
    except (RuntimeError, StopIteration) as e:
        logger.debug(f'evdev: {e}')

    # ── raw js0 ──────────────────────────────────────────────────────────────
    js_path = '/dev/input/js0'
    if os.path.exists(js_path):
        try:
            b = _Js0Backend(js_path)
            return b, f'raw {js_path}'
        except Exception as e:
            logger.error(f'Failed to open {js_path}: {e}')
    else:
        logger.error(f'{js_path} not found — plug in the controller and restart.')

    return None, ''


# ─────────────────────────────────────────────────────────────────────────────
# Signal math
# ─────────────────────────────────────────────────────────────────────────────

def _deadzone(value, dz):
    """Rescale value ∈ [-1, 1] to [-1, 1] after removing a symmetric dead zone."""
    if abs(value) < dz:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    return sign * (abs(value) - dz) / (1.0 - dz)


def _trigger_to_throttle(raw, dz):
    """Map trigger raw value (-1 = rest, +1 = full) → throttle ∈ [0, 1]."""
    val = (raw + 1.0) / 2.0          # → 0..1
    if val < dz:
        return 0.0
    return (val - dz) / (1.0 - dz)


def _slew(current, target, rate, dt):
    """Rate-limit how fast current approaches target (rate = units per second)."""
    delta = target - current
    limit = rate * dt
    if abs(delta) <= limit:
        return target
    return current + math.copysign(limit, delta)


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 Node
# ─────────────────────────────────────────────────────────────────────────────

class ControllerDrive(Node):

    def __init__(self):
        super().__init__('controller_drive')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('cmd_topic',             '/cmd_vel_raw')
        self.declare_parameter('max_speed',             0.04)    # m/s
        self.declare_parameter('max_turn',              0.4)    # rad/s
        self.declare_parameter('deadzone',              0.08)
        self.declare_parameter('publish_hz',            30.0)
        self.declare_parameter('speed_slew',            1.0)    # m/s per second
        self.declare_parameter('steer_slew',            12.0)   # rad/s per second
        self.declare_parameter('require_enable_button', True)
        self.declare_parameter('enable_button',         4)      # LB
        self.declare_parameter('reverse_button',        0)      # A
        self.declare_parameter('stop_button',           1)      # B
        self.declare_parameter('steer_axis',            0)      # left stick X
        self.declare_parameter('throttle_axis',         1)      # left stick Y (use_trigger=False)
        self.declare_parameter('trigger_axis',          5)      # right trigger (use_trigger=True)
        self.declare_parameter('use_trigger',           True)
        self.declare_parameter('undo_button',           3)      # Y
        self.declare_parameter('node_delete_wait',      30.0)

        p = self.get_parameter
        self.cmd_topic      = p('cmd_topic').get_parameter_value().string_value
        self.max_speed      = float(p('max_speed').get_parameter_value().double_value)
        self.max_turn       = float(p('max_turn').get_parameter_value().double_value)
        self.dz             = float(p('deadzone').get_parameter_value().double_value)
        hz                  = float(p('publish_hz').get_parameter_value().double_value)
        self.speed_slew     = float(p('speed_slew').get_parameter_value().double_value)
        self.steer_slew     = float(p('steer_slew').get_parameter_value().double_value)
        self.require_enable = bool(p('require_enable_button').get_parameter_value().bool_value)
        self.enable_btn     = int(p('enable_button').get_parameter_value().integer_value)
        self.reverse_btn    = int(p('reverse_button').get_parameter_value().integer_value)
        self.stop_btn       = int(p('stop_button').get_parameter_value().integer_value)
        self.steer_ax       = int(p('steer_axis').get_parameter_value().integer_value)
        self.throttle_ax    = int(p('throttle_axis').get_parameter_value().integer_value)
        self.trigger_ax     = int(p('trigger_axis').get_parameter_value().integer_value)
        self.use_trigger    = bool(p('use_trigger').get_parameter_value().bool_value)
        self.undo_btn          = int(p('undo_button').get_parameter_value().integer_value)
        self.node_delete_wait  = float(p('node_delete_wait').get_parameter_value().double_value)

        self.cmd_pub        = self.create_publisher(Twist, self.cmd_topic, 10)
        self.undo_pub       = self.create_publisher(Empty, '/undo_last_node', 2)

        self._speed_out          = 0.0
        self._turn_out           = 0.0
        self._dt                 = 1.0 / max(hz, 1.0)
        self._backend            = None
        self._prev_undo          = False
        self._undo_countdown     = 0.0
        self._last_countdown_sec = 0.0

        self._backend, backend_name = _open_best_backend(self.get_logger())

        if self._backend:
            name    = self._backend.name()
            n_axes  = self._backend.num_axes()
            n_btns  = self._backend.num_buttons()
            self.get_logger().info(
                f'[{backend_name}] Controller: "{name}"  '
                f'axes={n_axes}  buttons={n_btns}')
            self.get_logger().info(
                f'Publishing to {self.cmd_topic} @ {hz:.0f} Hz | '
                f'max_speed={self.max_speed:.2f} m/s  '
                f'max_turn={self.max_turn:.2f} rad/s  '
                f'deadzone={self.dz:.2f}')
            throttle_src = (
                f'axis{self.trigger_ax} (RT)'
                if self.use_trigger
                else f'axis{self.throttle_ax} (LS-Y)'
            )
            enable_str = (
                f'btn{self.enable_btn}=enable(LB)  '
                if self.require_enable else ''
            )
            self.get_logger().info(
                f'Controls: {enable_str}'
                f'throttle={throttle_src}  '
                f'steer=axis{self.steer_ax}(LS-X)  '
                f'reverse=btn{self.reverse_btn}(A)  '
                f'stop=btn{self.stop_btn}(B)  '
                f'undo_node=btn{self.undo_btn}(Y)'
            )
        else:
            self.get_logger().error(
                'No controller found. Node will publish zeros until one is connected.\n'
                'To install a better backend:\n'
                '  sudo apt install python3-pygame\n'
                '  sudo apt install python3-evdev'
            )

        self.create_timer(self._dt, self._loop)

    # ── Timer loop ───────────────────────────────────────────────────────────

    def _loop(self):
        if self._backend is None:
            self._publish(0.0, 0.0)
            return

        try:
            self._backend.pump()
        except Exception as e:
            self.get_logger().warn(f'Controller read error: {e}', throttle_duration_sec=2.0)
            self._publish(0.0, 0.0)
            return

        undo_now = self._backend.button(self.undo_btn)
        if undo_now and not self._prev_undo:
            self.undo_pub.publish(Empty())
            self._undo_countdown = self.node_delete_wait
            self._last_countdown_sec = math.floor(self._undo_countdown) + 1
        self._prev_undo = undo_now

        if self._undo_countdown > 0.0:
            self._undo_countdown -= self._dt
            remaining_sec = math.ceil(self._undo_countdown)
            if remaining_sec < self._last_countdown_sec:
                self._last_countdown_sec = remaining_sec
                if remaining_sec > 0:
                    print(f'\r  [REPOSITION] Recording resumes in {remaining_sec}s...   ',
                          end='', flush=True)
                else:
                    self._undo_countdown = 0.0
                    print('\r  [REPOSITION] Done — recording resumed.              ',
                          flush=True)

        enabled = (not self.require_enable) or self._backend.button(self.enable_btn)
        stopped = self._backend.button(self.stop_btn)
        reverse = self._backend.button(self.reverse_btn)

        if stopped or not enabled:
            target_speed = 0.0
            target_turn  = 0.0
        else:
            # ── Throttle ─────────────────────────────────────────────────────
            if self.use_trigger:
                raw_trigger  = self._backend.axis(self.trigger_ax)
                throttle_fwd = _trigger_to_throttle(raw_trigger, self.dz)
                sign         = -1.0 if reverse else 1.0
                target_speed = sign * throttle_fwd * self.max_speed
            else:
                # Left stick Y: -1 = pushed up = forward in standard mapping
                raw_y        = -self._backend.axis(self.throttle_ax)
                target_speed = _deadzone(raw_y, self.dz) * self.max_speed

            # ── Steering ─────────────────────────────────────────────────────
            # Left stick X: -1 = left, +1 = right
            # ROS angular.z: positive = left (CCW), so negate
            raw_x       = self._backend.axis(self.steer_ax)
            steer_norm  = _deadzone(raw_x, self.dz)
            target_turn = -steer_norm * self.max_turn

        # ── Slew limiting ─────────────────────────────────────────────────────
        self._speed_out = _slew(self._speed_out, target_speed,
                                self.speed_slew, self._dt)
        self._turn_out  = _slew(self._turn_out,  target_turn,
                                self.steer_slew, self._dt)

        self._publish(self._speed_out, self._turn_out)

    def _publish(self, speed, turn):
        msg = Twist()
        msg.linear.x  = float(speed)
        msg.angular.z = float(turn)
        self.cmd_pub.publish(msg)

    def destroy_node(self):
        try:
            self._publish(0.0, 0.0)
        except Exception:
            pass
        if self._backend:
            try:
                self._backend.close()
            except Exception:
                pass
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ControllerDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
