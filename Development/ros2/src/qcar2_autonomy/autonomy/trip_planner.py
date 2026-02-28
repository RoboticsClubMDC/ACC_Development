# =========================
# OVERALL: trip_planner.py  (NO PlannerServer)
# ✅ Subscribes to /path_follower/qcar_state (2/4/5/6)
# ✅ Directly sets /qcar2_hardware led_color_id
# ✅ YOUR requested LED policy:
#    - HUB idle/accepting rides: MAGENTA (5)
#    - Going to pickup: GREEN (1)
#    - After pickup: BLUE (2) stays BLUE until dropoff reached
#    - At dropoff: GREEN (1) (NO orange)
#    - Return to hub: GREEN then MAGENTA at hub
# =========================
#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, UInt8
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, SetParametersResult
from rclpy.parameter import ParameterType


class tripPlanner(Node):
    def __init__(self):
        super().__init__('trip_planner')

        # Nodes
        self.path_follower_node = "path_follower"
        self.qcar_hardware_node = "qcar2_hardware"

        # Clients
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

        # Scenario defaults (override anytime)
        self.declare_parameter('pickup_xy', [0.125, 4.395])
        self.declare_parameter('dropoff_xy', [-0.905, 0.800])

        self.declare_parameter('stop_seconds', [3.0])
        self.stop_seconds = float(list(self.get_parameter("stop_seconds").get_parameter_value().double_array_value)[0])

        self.pickup_xy = list(self.get_parameter("pickup_xy").get_parameter_value().double_array_value)
        self.dropoff_xy = list(self.get_parameter("dropoff_xy").get_parameter_value().double_array_value)

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        # ---------------- LED IDs (ACTUAL led_color_id) ----------------
        self.LED_GREEN = 1
        self.LED_BLUE = 2
        self.LED_MAGENTA = 5
        self.LED_ORANGE = 6  # unused per your policy

        # ---------------- internal state ----------------
        self.current_path_status = False
        self.startup_sent = False
        self.ready_for_rides = False
        self.mission_running = False
        self.new_ride_requested = False

        # follower publishes 4 driving, 2 pickup, 6 dropoff, 5 hub
        self.follower_event = 0

        # YOUR policy latches
        self.in_ride = False
        self.picked_up = False

        # LED state
        self._last_led = None

        # subscribe
        self.create_subscription(Bool, '/path_status', self.path_status_callback, 10)
        self.create_subscription(UInt8, '/path_follower/qcar_state', self.follower_event_callback, 10)

        # start at HUB idle magenta (accepting rides)
        self._set_led(self.LED_MAGENTA)

        # loop (mission control only; LEDs driven by follower callback)
        self.timer = self.create_timer(0.1, self.loop)

    # ---------------- callbacks ----------------
    def path_status_callback(self, msg):
        self.current_path_status = bool(msg.data)

    def follower_event_callback(self, msg):
            """
        PathFollower publishes:
        4 = driving
        2 = pickup stop
        6 = dropoff stop
        5 = hub/ready

        Exact scenario mapping:
        hub      -> MAGENTA
        driving  -> GREEN
        pickup   -> BLUE
        dropoff  -> ORANGE
        """
            s = int(msg.data)
            self.follower_event = s

            if s == 5:
                # HUB idle / accepting rides
                self._set_led(self.LED_MAGENTA)

            elif s == 4:
                # ANY driving segment
                self._set_led(self.LED_GREEN)

            elif s == 2:
                # At pickup
                self._set_led(self.LED_BLUE)

            elif s == 6:
                # At dropoff
                self._set_led(self.LED_ORANGE)

    def parameter_update_callback(self, params):
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

        # Any update while ready triggers new ride
        if self.ready_for_rides and not self.mission_running:
            self.new_ride_requested = True

        return SetParametersResult(successful=True)

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

    # ---------------- LED setter (DIRECT TO HARDWARE) ----------------
    def _set_led(self, led_color_id: int):
        led_color_id = int(led_color_id)
        if self._last_led == led_color_id:
            return
        self._last_led = led_color_id

        self._set_param(self.qcar_hardware_client, "led_color_id",
                        led_color_id, ParameterType.PARAMETER_INTEGER)

    # ---------------- main loop (NO LED forcing here) ----------------
    def loop(self):
        # STARTUP: go to hub once (if you spawn elsewhere)
        if not self.ready_for_rides and not self.mission_running:
            if not self.startup_sent and not self.current_path_status:
                self._set_param(self.path_follower_client, "node_values",
                                [0, self.taxi_node], ParameterType.PARAMETER_INTEGER_ARRAY)
                self._set_param(self.path_follower_client, "start_path",
                                [True], ParameterType.PARAMETER_BOOL_ARRAY)
                self.startup_sent = True

            if self.current_path_status:
                self.ready_for_rides = True
                self.startup_sent = False
            return

        # READY (waiting for ride) — LED will be MAGENTA due to follower_event=5
        if self.ready_for_rides and not self.mission_running and not self.new_ride_requested:
            return

        # START MISSION
        if self.ready_for_rides and not self.mission_running and self.new_ride_requested:
            self._set_param(self.path_follower_client, "mission_use_xy",
                            [True], ParameterType.PARAMETER_BOOL_ARRAY)
            self._set_param(self.path_follower_client, "mission_pickup_xy",
                            self.pickup_xy, ParameterType.PARAMETER_DOUBLE_ARRAY)
            self._set_param(self.path_follower_client, "mission_dropoff_xy",
                            self.dropoff_xy, ParameterType.PARAMETER_DOUBLE_ARRAY)
            self._set_param(self.path_follower_client, "mission_stop_seconds",
                            [float(self.stop_seconds)], ParameterType.PARAMETER_DOUBLE_ARRAY)

            self._set_param(self.path_follower_client, "mission_enable",
                            [True], ParameterType.PARAMETER_BOOL_ARRAY)

            self.mission_running = True
            self.ready_for_rides = False
            self.new_ride_requested = False
            return

        # MISSION RUNNING: end when follower reports HUB (5)
        if self.mission_running:
            if self.follower_event == 5:
                self._set_param(self.path_follower_client, "mission_enable",
                                [False], ParameterType.PARAMETER_BOOL_ARRAY)
                self.mission_running = False
                self.ready_for_rides = True


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