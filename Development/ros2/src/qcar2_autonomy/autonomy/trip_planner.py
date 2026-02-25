#! /usr/bin/env python3

import rclpy  # Python client library for ROS 2
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, SetParametersResult
from rclpy.parameter import ParameterType

from std_msgs.msg import Bool, UInt8

import time


class tripPlanner(Node):
    def __init__(self):
        super().__init__('trip_planner')

        # node names to set
        self.path_follower_node = "path_follower"
        self.qcar_hardware_node = "qcar2_hardware"

        # start clients for node parameters
        self.path_follower_client = self.create_client(
            SetParameters, f'/{self.path_follower_node}/set_parameters'
        )
        self.qcar_hardware_client = self.create_client(
            SetParameters, f'/{self.qcar_hardware_node}/set_parameters'
        )

        # waiting for services to become available...
        while not self.path_follower_client.wait_for_service(timeout_sec=4.0):
            self.get_logger().info(f'waiting for {self.path_follower_node} parameter service!.....')
        self.get_logger().info(f'connected to {self.path_follower_node} parameter service!.....')

        while not self.qcar_hardware_client.wait_for_service(timeout_sec=4.0):
            self.get_logger().info(f'waiting for {self.qcar_hardware_node} parameter service!.....')
        self.get_logger().info(f'connected to {self.qcar_hardware_node} parameter service!.....')

        self.parameter_change_retries = 5
        self.parameter_sleep_time = 2

        # define new parameters for taxi node to use
        self.declare_parameter('taxi_node', [10])
        self.taxi_node = list(self.get_parameter("taxi_node").get_parameter_value().integer_array_value)[0]

        # define new parameters for node to use
        self.declare_parameter('trip_nodes', [2, 8])
        self.trip_nodes = list(self.get_parameter("trip_nodes").get_parameter_value().integer_array_value)

        # parameter callback for new rides/update to taxi node
        self.add_on_set_parameters_callback(self.parameter_update_callback)

        """
        State definition for trip planner logic
        trip super states
        1 - going to taxi hub
        2 - ready for rides

        trip states (qcar_state published)
        1 - waiting red
        2 - pickup blue
        3 - dropoff yellow
        4 - driving green
        """

        self.trip_super_state = 1.0
        self.current_trip_state = 1.0
        self.path_nodes = []
        self.super_state_1_flags = [False, False]
        self.taxi_node = 10
        self.new_ride_requested = False
        self.trip_length = 0
        self.current_trip_status = False
        self.current_stop = 0
        self.goal_stop = 0
        self.stop_index = 0
        self.nodes_sent = False

        # LED timing
        self.led_time = 3
        self.led_time_t0 = time.time()
        self.led_timer_reset = False

        self.path_status_subscrition = self.create_subscription(
            Bool, '/path_status', self.path_status_callback, 1
        )

        # --- STATE + PUBLISHER (THIS WAS THE MISSING PART) ---
        self.qcar_state = 4  # use int for publishing
        self.previous_led_state = 0
        self.current_path_status = False

        self.qcar_state_pub = self.create_publisher(UInt8, '/trip_planner/qcar_state', 10)
        self.publish_qcar_state()  # publish initial state
        # -----------------------------------------------------

        self.led_set_logic()

        self.dt = 1 / 10
        self.trip_time = time.time()
        self.timer1 = self.create_timer(self.dt, self.trip_planner_controller)

    def publish_qcar_state(self):
        msg = UInt8()
        msg.data = int(self.qcar_state)
        self.qcar_state_pub.publish(msg)

    def parameter_update_callback(self, params):
        for param in params:
            if param.name == 'taxi_node' and param.type_ == param.Type.INTEGER_ARRAY:
                self.taxi_node = list(param.value)
                if len(self.taxi_node > 1):
                    self.get_logger().info('Incorrect number of nodes given... setting default')
                    self.taxi_node = 10

            elif param.name == 'trip_nodes' and param.type_ == param.Type.INTEGER_ARRAY:
                if self.trip_super_state == 1:
                    self.get_logger().info('Cant assign trip, not at the taxi hub!')

                # The QCar2 has to reach the taxi hub before a new trip can be requested
                if self.current_trip_status is True and self.trip_super_state == 2:
                    self.trip_nodes = list(param.value)
                    self.path_nodes = []
                    self.trip_length = len(self.trip_nodes)

                    if self.trip_length < 2:
                        self.get_logger().info('Invalid trip scenario... minimum 2 point required for trip')
                    else:
                        for i in range(2 + self.trip_length):
                            if i == 0 or i == len(self.trip_nodes) + 1:
                                self.path_nodes.append(10)
                            else:
                                self.path_nodes.append(self.trip_nodes[i - 1])

                        print("New trip requested!")
                        print(self.path_nodes)

                        self.new_ride_requested = True
                        self.stop_index = 0
                        self.trip_time = time.time()

                        # We travel slowly to pickup station and speed up during actual rides
                        # self.send_request(
                        #     param_name="desired_speed",
                        #     param_value=[1.0],
                        #     param_type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                        #     client=self.path_follower_client
                        # )

                if self.current_trip_status is False and self.trip_super_state == 2:
                    self.get_logger().info('Cant assign trip, current trip in progress!')

        return SetParametersResult(successful=True)

    def trip_planner_controller(self):
        # publish state at 10 Hz so anything subscribing always gets updates
        self.publish_qcar_state()

        t_current = time.time() - self.trip_time

        # Super state 1: drive to taxi hub
        if self.trip_super_state == 1:
            if t_current < 10 and len(self.path_nodes) == 0 and self.current_path_status is False:
                self.path_nodes.append(0)
                for node in self.trip_nodes:
                    self.path_nodes.append(node)
                self.path_nodes.append(self.taxi_node)

                if self.super_state_1_flags[0] is False:
                    self.super_state_1_flags[0] = self.send_request(
                        param_name="node_values",
                        param_value=self.path_nodes,
                        param_type=ParameterType.PARAMETER_INTEGER_ARRAY,
                        client=self.path_follower_client
                    )

                if self.super_state_1_flags[1] is False:
                    self.super_state_1_flags[1] = self.send_request(
                        param_name="start_path",
                        param_value=[True],
                        param_type=ParameterType.PARAMETER_BOOL_ARRAY,
                        client=self.path_follower_client
                    )

            if self.current_path_status is True and t_current > 10:
                if not self.led_timer_reset:
                    self.led_time_t0 = time.time()
                    self.led_timer_reset = True

                if time.time() - self.led_time_t0 < self.led_time:
                    self.qcar_state = 1  # red
                elif time.time() - self.led_time_t0 > self.led_time:
                    self.qcar_state = 4  # green driving/ready

                    self.path_nodes = []
                    self.trip_super_state = 2
                    self.current_trip_status = True
                    self.led_timer_reset = False

        # Super state 2: at taxi hub, run ride segments
        if self.trip_super_state == 2:
            self.qcar_state = 4
            if self.new_ride_requested:
                if self.stop_index + 1 <= len(self.path_nodes) - 1:
                    if self.nodes_sent is False:
                        self.current_stop = self.path_nodes[self.stop_index]
                        self.goal_stop = self.path_nodes[self.stop_index + 1]
                        trip = [self.current_stop, self.goal_stop]

                        if not self.nodes_sent:
                            self.nodes_sent = self.send_request(
                                param_name="node_values",
                                param_value=trip,
                                param_type=ParameterType.PARAMETER_INTEGER_ARRAY,
                                client=self.path_follower_client
                            )

                    if self.current_path_status is True and t_current > 2:
                        if not self.led_timer_reset:
                            self.led_time_t0 = time.time()
                            self.led_timer_reset = True

                        if time.time() - self.led_time_t0 < self.led_time:
                            if len(self.path_nodes) > 4:
                                if self.stop_index + 1 == 1:
                                    self.qcar_state = 2  # pickup blue

                                if (self.stop_index + 1 > 1 or self.stop_index + 1 == len(self.path_nodes) - 1 and
                                        self.stop_index + 1 != len(self.path_nodes) - 2):
                                    self.qcar_state = 1  # red stop

                                if self.stop_index + 1 == len(self.path_nodes) - 2:
                                    self.qcar_state = 3  # dropoff yellow

                            elif len(self.path_nodes) == 4:
                                if self.stop_index + 1 == 1:
                                    self.qcar_state = 2
                                elif self.stop_index + 1 == 2:
                                    self.qcar_state = 3
                                elif self.stop_index + 1 == 3:
                                    self.qcar_state = 1

                        elif time.time() - self.led_time_t0 > self.led_time:
                            self.led_timer_reset = False
                            self.nodes_sent = False
                            self.qcar_state = 4
                            self.stop_index += 1
                            self.trip_time = time.time()

        if self.previous_led_state != self.qcar_state:
            self.previous_led_state = self.qcar_state
            self.publish_qcar_state()  # publish immediately on change
            self.led_set_logic()

    def led_set_logic(self):
        # LED Red
        if self.qcar_state == 1:
            self.send_request(
                param_name="led_color_id",
                param_value=0,
                param_type=ParameterType.PARAMETER_INTEGER,
                client=self.qcar_hardware_client
            )

        # Arrived at pickup (blue)
        elif self.qcar_state == 2:
            self.send_request(
                param_name="led_color_id",
                param_value=2,
                param_type=ParameterType.PARAMETER_INTEGER,
                client=self.qcar_hardware_client
            )

        # Drop off (yellow)
        elif self.qcar_state == 3:
            self.send_request(
                param_name="led_color_id",
                param_value=3,
                param_type=ParameterType.PARAMETER_INTEGER,
                client=self.qcar_hardware_client
            )

        # Driving (green)
        elif self.qcar_state == 4:
            self.send_request(
                param_name="led_color_id",
                param_value=1,
                param_type=ParameterType.PARAMETER_INTEGER,
                client=self.qcar_hardware_client
            )

    def path_status_callback(self, msg):
        self.current_path_status = msg.data

    def send_request(self, param_name, param_value, param_type, client):
        param = Parameter()
        param.name = param_name
        param.value.type = param_type

        if param_type == ParameterType.PARAMETER_INTEGER_ARRAY:
            param.value.integer_array_value = param_value
        elif param_type == ParameterType.PARAMETER_INTEGER:
            param.value.integer_value = param_value
        elif param_type == ParameterType.PARAMETER_BOOL_ARRAY:
            param.value.bool_array_value = param_value
        elif param_type == ParameterType.PARAMETER_DOUBLE_ARRAY:
            param.value.double_array_value = param_value

        request = SetParameters.Request()
        request.parameters = [param]
        client.call_async(request)
        return True  # keep your "flags" behavior


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
