#! /usr/bin/env python3

import rclpy
from rclpy.node import Node

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, SetParametersResult
from rclpy.parameter import ParameterType

from std_msgs.msg import Bool, UInt8, Int32MultiArray, Float32, Float64MultiArray
import time


class tripPlanner(Node):
    def __init__(self):
        super().__init__('trip_planner')

        # ===================== LED hardware service (kept) =====================
        self.qcar_hardware_node = "qcar2_hardware"
        self.qcar_hardware_client = self.create_client(
            SetParameters, f'/{self.qcar_hardware_node}/set_parameters'
        )

        while not self.qcar_hardware_client.wait_for_service(timeout_sec=4.0):
            self.get_logger().info(f'waiting for {self.qcar_hardware_node} parameter service!.....')
        self.get_logger().info(f'connected to {self.qcar_hardware_node} parameter service!.....')

        # ===================== PARAMETERS =====================
        self.declare_parameter('taxi_node', [10])
        self.taxi_node = int(list(self.get_parameter("taxi_node").get_parameter_value().integer_array_value)[0])

        # Hub XY default (node 10 in competition docs / your note)
        self.declare_parameter('hub_xy', [-1.242, -0.495])
        self.hub_xy = list(self.get_parameter("hub_xy").get_parameter_value().double_array_value)

        # If false => node mode; if true => coordinate mode
        self.declare_parameter('use_xy', True)
        self.use_xy = bool(self.get_parameter("use_xy").value)

        # Node-mode trip (still supported)
        self.declare_parameter('trip_nodes', [2, 8])
        self.trip_nodes = list(self.get_parameter("trip_nodes").get_parameter_value().integer_array_value)

        # XY ride request inputs
        self.declare_parameter('pickup_xy', [0.0, 0.0])
        self.pickup_xy = list(self.get_parameter("pickup_xy").get_parameter_value().double_array_value)

        self.declare_parameter('dropoff_xy', [0.0, 0.0])
        self.dropoff_xy = list(self.get_parameter("dropoff_xy").get_parameter_value().double_array_value)

        # Stop seconds at pickup (PathFollower will enforce this)
        self.declare_parameter('stop_seconds', 3.0)
        self.stop_seconds = float(self.get_parameter('stop_seconds').value)

        # Speeds
        self.declare_parameter('speed_to_hub', 0.6)
        self.declare_parameter('speed_on_trip', 0.6)
        self.speed_to_hub = float(self.get_parameter('speed_to_hub').value)
        self.speed_on_trip = float(self.get_parameter('speed_on_trip').value)

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        # ===================== STATE =====================
        self.trip_super_state = 1.0  # 1: go hub, 2: ready
        self.current_trip_status = False
        self.new_ride_requested = False

        self.sent_init_to_hub = False
        self.sent_mission = False

        # timing
        self.trip_time = time.time()

        # path status from follower
        self.current_path_status = False
        self.path_status_subscription = self.create_subscription(
            Bool, '/path_status', self.path_status_callback, 1
        )

        # ===================== PUBS =====================
        # Node-mode still supported
        self.node_values_pub = self.create_publisher(Int32MultiArray, '/trip_planner/node_values', 10)

        # XY mission (ONE message = pickup+dropoff)
        self.mission_xy_pub = self.create_publisher(Float64MultiArray, '/trip_planner/mission_xy', 10)

        # Still allow “go to hub” as a single XY goal
        self.goal_xy_pub = self.create_publisher(Float64MultiArray, '/trip_planner/goal_xy', 10)

        self.start_path_pub = self.create_publisher(Bool, '/trip_planner/start_path', 10)
        self.desired_speed_pub = self.create_publisher(Float32, '/trip_planner/desired_speed', 10)

        # LED state publisher kept (not priority per your request)
        self.qcar_state = 4
        self.qcar_state_pub = self.create_publisher(UInt8, '/trip_planner/qcar_state', 10)
        self.publish_qcar_state()

        # main loop
        self.dt = 1 / 10
        self.timer1 = self.create_timer(self.dt, self.trip_planner_controller)

    # ===================== Publishers helpers =====================
    def publish_qcar_state(self):
        msg = UInt8()
        msg.data = int(self.qcar_state)
        self.qcar_state_pub.publish(msg)

    def publish_node_values(self, nodes_list):
        msg = Int32MultiArray()
        msg.data = [int(x) for x in nodes_list]
        self.node_values_pub.publish(msg)

    def publish_goal_xy(self, xy):
        msg = Float64MultiArray()
        msg.data = [float(xy[0]), float(xy[1])]
        self.goal_xy_pub.publish(msg)

    def publish_mission_xy(self, pickup_xy, dropoff_xy, stop_seconds):
        msg = Float64MultiArray()
        # [px, py, dx, dy, stop_seconds]
        msg.data = [float(pickup_xy[0]), float(pickup_xy[1]),
                    float(dropoff_xy[0]), float(dropoff_xy[1]),
                    float(stop_seconds)]
        self.mission_xy_pub.publish(msg)

    def publish_start_path(self, flag: bool):
        msg = Bool()
        msg.data = bool(flag)
        self.start_path_pub.publish(msg)

    def publish_desired_speed(self, v: float):
        msg = Float32()
        msg.data = float(v)
        self.desired_speed_pub.publish(msg)

    # ===================== Parameter callback =====================
    def parameter_update_callback(self, params):
        for param in params:
            if param.name == 'use_xy':
                self.use_xy = bool(param.value)
                self.get_logger().info(f"use_xy => {self.use_xy}")

            elif param.name == 'hub_xy' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.hub_xy = list(param.value)
                self.get_logger().info(f"hub_xy => {self.hub_xy}")

            elif param.name == 'pickup_xy' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.pickup_xy = list(param.value)
                self.get_logger().info(f"pickup_xy updated => {self.pickup_xy}")
                if self.trip_super_state == 2 and self.current_trip_status:
                    self.new_ride_requested = True
                    self.sent_mission = False
                    self.trip_time = time.time()

            elif param.name == 'dropoff_xy' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.dropoff_xy = list(param.value)
                self.get_logger().info(f"dropoff_xy updated => {self.dropoff_xy}")
                if self.trip_super_state == 2 and self.current_trip_status:
                    self.new_ride_requested = True
                    self.sent_mission = False
                    self.trip_time = time.time()

            elif param.name == 'stop_seconds':
                self.stop_seconds = float(param.value)
                self.get_logger().info(f"stop_seconds => {self.stop_seconds}")

            elif param.name == 'trip_nodes' and param.type_ == param.Type.INTEGER_ARRAY:
                self.trip_nodes = list(param.value)
                self.get_logger().info(f"trip_nodes updated => {self.trip_nodes}")
                if self.trip_super_state == 2 and self.current_trip_status and (not self.use_xy):
                    self.new_ride_requested = True
                    self.sent_mission = False
                    self.trip_time = time.time()

        return SetParametersResult(successful=True)

    # ===================== Main trip logic =====================
    def trip_planner_controller(self):
        t_current = time.time() - self.trip_time

        # ---------------- Super state 1: go to hub ----------------
        if self.trip_super_state == 1:
            if (not self.sent_init_to_hub) and (not self.current_path_status):
                self.publish_desired_speed(self.speed_to_hub)
                self.publish_start_path(True)

                if self.use_xy:
                    self.publish_goal_xy(self.hub_xy)
                else:
                    init_nodes = [0] + [int(n) for n in self.trip_nodes] + [int(self.taxi_node)]
                    self.publish_node_values(init_nodes)

                self.sent_init_to_hub = True
                self.trip_time = time.time()

            # Once hub is reached (PathFollower sets /path_status True)
            if self.current_path_status is True and t_current > 2.0:
                self.trip_super_state = 2
                self.current_trip_status = True
                self.trip_time = time.time()
                self.get_logger().info("Arrived at HUB. Ready for rides.")

        # ---------------- Super state 2: ready for rides ----------------
        if self.trip_super_state == 2:
            if self.new_ride_requested:

                if self.use_xy:
                    if len(self.pickup_xy) != 2 or len(self.dropoff_xy) != 2:
                        self.get_logger().info("Invalid pickup/dropoff XY (need 2 values each).")
                        return

                    # Send ONE mission message once
                    if not self.sent_mission:
                        self.publish_desired_speed(self.speed_on_trip)
                        self.publish_start_path(True)
                        self.publish_mission_xy(self.pickup_xy, self.dropoff_xy, self.stop_seconds)
                        self.sent_mission = True
                        self.get_logger().info(f"Mission sent (one-shot): pickup={self.pickup_xy}, dropoff={self.dropoff_xy}")

                    # Mission is complete ONLY when PathFollower returns to hub and publishes /path_status True
                    if self.current_path_status is True and t_current > 2.0:
                        self.get_logger().info("Mission complete (including hub return).")
                        self.new_ride_requested = False
                        self.sent_mission = False
                        self.trip_time = time.time()

                else:
                    # node-mode: just send full list and let follower run it (simple)
                    if not self.sent_mission:
                        nodes = [int(self.taxi_node)] + [int(n) for n in self.trip_nodes] + [int(self.taxi_node)]
                        self.publish_desired_speed(self.speed_on_trip)
                        self.publish_start_path(True)
                        self.publish_node_values(nodes)
                        self.sent_mission = True
                        self.trip_time = time.time()

                    if self.current_path_status is True and t_current > 2.0:
                        self.new_ride_requested = False
                        self.sent_mission = False
                        self.trip_time = time.time()

    def path_status_callback(self, msg):
        self.current_path_status = bool(msg.data)

    def send_request(self, param_name, param_value, param_type, client):
        # (kept for LED work later)
        param = Parameter()
        param.name = param_name
        param.value.type = param_type
        if param_type == ParameterType.PARAMETER_INTEGER:
            param.value.integer_value = int(param_value)

        request = SetParameters.Request()
        request.parameters = [param]
        client.call_async(request)
        return True


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