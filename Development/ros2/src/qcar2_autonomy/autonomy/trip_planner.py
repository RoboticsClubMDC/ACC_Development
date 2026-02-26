#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, SetParametersResult
from rclpy.parameter import ParameterType
from std_msgs.msg import Bool, UInt8
import time


class tripPlanner(Node):
    def __init__(self):
        super().__init__('trip_planner')

        self.path_follower_node = "path_follower"
        self.qcar_hardware_node = "qcar2_hardware"

        self.path_follower_client = self.create_client(SetParameters, f'/{self.path_follower_node}/set_parameters')
        self.qcar_hardware_client = self.create_client(SetParameters, f'/{self.qcar_hardware_node}/set_parameters')

        while not self.path_follower_client.wait_for_service(timeout_sec=4.0):
            self.get_logger().info(f'waiting for {self.path_follower_node} parameter service...')
        self.get_logger().info(f'connected to {self.path_follower_node} parameter service.')

        while not self.qcar_hardware_client.wait_for_service(timeout_sec=4.0):
            self.get_logger().info(f'waiting for {self.qcar_hardware_node} parameter service...')
        self.get_logger().info(f'connected to {self.qcar_hardware_node} parameter service.')

        # ---------------- user-facing params ----------------
        self.declare_parameter('taxi_node', [10])
        self.taxi_node = int(list(self.get_parameter("taxi_node").get_parameter_value().integer_array_value)[0])

        self.declare_parameter('pickup_xy', [0.0, 0.0])
        self.declare_parameter('dropoff_xy', [0.0, 0.0])

        self.declare_parameter('stop_seconds', [3.0])
        self.stop_seconds = float(list(self.get_parameter("stop_seconds").get_parameter_value().double_array_value)[0])

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        # ---------------- LED IDs ----------------
        # hardware mapping: 0 red, 1 green, 2 blue, 5 magenta, 6 orange
        self.LED_GREEN = 1
        self.LED_BLUE = 2
        self.LED_MAGENTA = 5
        self.LED_ORANGE = 6

        # ---------------- MissionStage values from PathFollower ----------------
        self.STAGE_DISABLED = 0
        self.STAGE_TO_PICKUP = 1
        self.STAGE_WAIT_AT_PICKUP = 2
        self.STAGE_TO_DROPOFF = 3
        self.STAGE_WAIT_AT_DROPOFF = 4
        self.STAGE_TO_HUB = 5
        self.STAGE_DONE = 6

        # ---------------- internal state ----------------
        self.current_path_status = False
        self.mission_stage = 0

        self.startup_sent = False
        self.ready_for_rides = False
        self.mission_running = False
        self.new_ride_requested = False

        # hub flash control (THIS fixes “it never becomes ready”)
        self.hub_flash_active = False
        self.hold_until = 0.0

        self._last_led = None

        # subscribe
        self.create_subscription(Bool, '/path_status', self.path_status_callback, 10)
        self.create_subscription(UInt8, '/path_follower/mission_state', self.mission_state_callback, 10)

        # loop
        self.timer = self.create_timer(0.1, self.loop)

    # ---------------- callbacks ----------------
    def path_status_callback(self, msg):
        self.current_path_status = bool(msg.data)

    def mission_state_callback(self, msg):
        self.mission_stage = int(msg.data)

    def parameter_update_callback(self, params):
        # Update stored pickup/dropoff
        for p in params:
            if p.name == 'pickup_xy' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.pickup_xy = list(p.value)
                self.get_logger().info(f"pickup_xy updated: {self.pickup_xy}")

            elif p.name == 'dropoff_xy' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.dropoff_xy = list(p.value)
                self.get_logger().info(f"dropoff_xy updated: {self.dropoff_xy}")

            elif p.name == 'stop_seconds' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.stop_seconds = float(list(p.value)[0])
                self.get_logger().info(f"stop_seconds updated: {self.stop_seconds}")

        # If we are ready, treat any update as a ride request
        if self.ready_for_rides and not self.mission_running:
            self.new_ride_requested = True

        return SetParametersResult(successful=True)

    # ---------------- time helpers ----------------
    def _start_hold(self, seconds):
        self.hold_until = time.time() + float(seconds)

    def _in_hold(self):
        return time.time() < self.hold_until

    # ---------------- ROS param helper ----------------
    def _set_param(self, client, name, value, ptype):
        param = Parameter()
        param.name = name
        param.value.type = ptype

        if ptype == ParameterType.PARAMETER_INTEGER_ARRAY:
            param.value.integer_array_value = list(value)
        elif ptype == ParameterType.PARAMETER_DOUBLE_ARRAY:
            param.value.double_array_value = list(value)
        elif ptype == ParameterType.PARAMETER_BOOL_ARRAY:
            param.value.bool_array_value = list(value)
        elif ptype == ParameterType.PARAMETER_INTEGER:
            param.value.integer_value = int(value)
        else:
            raise RuntimeError(f"Unsupported param type: {ptype}")

        req = SetParameters.Request()
        req.parameters = [param]
        client.call_async(req)

    # ---------------- LED helper ----------------
    def _set_led(self, led_color_id):
        if self._last_led == led_color_id:
            return
        self._last_led = led_color_id
        self._set_param(self.qcar_hardware_client, "led_color_id", int(led_color_id), ParameterType.PARAMETER_INTEGER)

    # ---------------- main loop ----------------
    def loop(self):
        # ---------------- STARTUP: go to hub ----------------
        if not self.ready_for_rides and not self.mission_running:
            # Driving to hub => GREEN
            self._set_led(self.LED_GREEN)

            if not self.startup_sent and not self.current_path_status:
                # node-based startup to taxi hub
                self._set_param(self.path_follower_client, "node_values",
                                [0, self.taxi_node], ParameterType.PARAMETER_INTEGER_ARRAY)
                self._set_param(self.path_follower_client, "start_path",
                                [True], ParameterType.PARAMETER_BOOL_ARRAY)
                self.startup_sent = True

            # Arrived hub => MAGENTA 3s, then READY
            if self.current_path_status and not self.hub_flash_active:
                self.hub_flash_active = True
                self._start_hold(3.0)
                self._set_led(self.LED_MAGENTA)

            if self.hub_flash_active and not self._in_hold():
                self.hub_flash_active = False
                self.ready_for_rides = True
                self.startup_sent = False
                self._set_led(self.LED_MAGENTA)

            return

        # ---------------- READY (waiting for ride) ----------------
        if self.ready_for_rides and not self.mission_running and not self.new_ride_requested:
            self._set_led(self.LED_MAGENTA)
            return

        # ---------------- START MISSION (XY) ----------------
        if self.ready_for_rides and not self.mission_running and self.new_ride_requested:
            # push mission params
            self._set_param(self.path_follower_client, "mission_use_xy",
                            [True], ParameterType.PARAMETER_BOOL_ARRAY)
            self._set_param(self.path_follower_client, "mission_pickup_xy",
                            self.pickup_xy, ParameterType.PARAMETER_DOUBLE_ARRAY)
            self._set_param(self.path_follower_client, "mission_dropoff_xy",
                            self.dropoff_xy, ParameterType.PARAMETER_DOUBLE_ARRAY)
            self._set_param(self.path_follower_client, "mission_stop_seconds",
                            [float(self.stop_seconds)], ParameterType.PARAMETER_DOUBLE_ARRAY)

            # enable mission (required)
            self._set_param(self.path_follower_client, "mission_enable",
                            [True], ParameterType.PARAMETER_BOOL_ARRAY)

            self.mission_running = True
            self.ready_for_rides = False
            self.new_ride_requested = False
            self._set_led(self.LED_GREEN)
            return

        # ---------------- MISSION RUNNING: LED from mission_stage ----------------
        if self.mission_running:
            if self.mission_stage in (self.STAGE_TO_PICKUP, self.STAGE_TO_DROPOFF, self.STAGE_TO_HUB):
                self._set_led(self.LED_GREEN)

            elif self.mission_stage == self.STAGE_WAIT_AT_PICKUP:
                self._set_led(self.LED_BLUE)

            elif self.mission_stage == self.STAGE_WAIT_AT_DROPOFF:
                self._set_led(self.LED_ORANGE)

            elif self.mission_stage == self.STAGE_DONE:
                # DONE => magenta 3s then ready
                if not self.hub_flash_active:
                    self.hub_flash_active = True
                    self._start_hold(3.0)
                    self._set_led(self.LED_MAGENTA)

                if self.hub_flash_active and not self._in_hold():
                    self.hub_flash_active = False
                    # disable mission so next ride can re-trigger cleanly
                    self._set_param(self.path_follower_client, "mission_enable",
                                    [False], ParameterType.PARAMETER_BOOL_ARRAY)

                    self.mission_running = False
                    self.ready_for_rides = True
                    self._set_led(self.LED_MAGENTA)

            else:
                # fallback
                self._set_led(self.LED_MAGENTA)


def main():
    rclpy.init()
    node = tripPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()