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

        trip states (qcar_state published, mapped to LEDs by Planner_server)
        1 - intermediate stop  -> red
        2 - pickup             -> blue
        3 - drop-off           -> orange
        4 - driving            -> green
        5 - at taxi hub        -> magenta
        """

        # Start in super_state 2 (idle at hub, ready for rides) so the user
        # can request a ride immediately by setting trip_nodes. Trip format:
        #   [current_position, pickup, dropoff]
        #   [current_position, pickup, stop, ..., dropoff]
        # The planner auto-appends taxi_node so the car returns home.
        self.trip_super_state = 2.0
        self.current_trip_state = 1.0
        self.path_nodes = []
        self.super_state_1_flags = [False, False]
        self.taxi_node = 10
        self.new_ride_requested = False
        self.trip_length = 0
        self.current_trip_status = True
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
        # Start magenta — idle at taxi hub awaiting a ride request.
        self.qcar_state = 5
        self.previous_led_state = 0
        self.current_path_status = False
        self.ride_complete = False

        self.qcar_state_pub = self.create_publisher(UInt8, '/trip_planner/qcar_state', 10)
        self.publish_qcar_state()  # publish initial state
        # -----------------------------------------------------

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
                taxi_list = list(param.value)
                if len(taxi_list) != 1:
                    self.get_logger().info('Incorrect number of nodes given... setting default')
                    self.taxi_node = 10
                else:
                    self.taxi_node = taxi_list[0]

            elif param.name == 'trip_nodes' and param.type_ == param.Type.INTEGER_ARRAY:
                # The QCar2 accepts new trips only while idling at the hub.
                if self.current_trip_status is False:
                    self.get_logger().info('Cant assign trip, current trip in progress!')
                    continue

                self.trip_nodes = list(param.value)
                self.trip_length = len(self.trip_nodes)
                if self.trip_length < 2:
                    self.get_logger().info(
                        'Invalid trip: need at least [current_position, pickup, dropoff].'
                    )
                    continue

                # trip_nodes already starts at the car's current node and ends
                # at the drop-off. Append the taxi hub so the car drives home
                # after the drop-off. Strip any user-supplied trailing hub to
                # avoid a duplicate node.
                trip = list(self.trip_nodes)
                if trip[-1] == self.taxi_node:
                    trip.pop()
                self.path_nodes = trip + [self.taxi_node]

                self.get_logger().info(f'New trip requested: {self.path_nodes}')

                self.new_ride_requested = True
                self.ride_complete = False
                self.current_trip_status = False  # ride in progress; block new requests
                self.stop_index = 0
                self.nodes_sent = False
                self.led_timer_reset = False
                self.trip_time = time.time()

                # We travel slowly to pickup station and speed up during actual rides
                self.send_request(
                    param_name="desired_speed",
                    param_value=[1.0],
                    param_type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    client=self.path_follower_client
                )

        return SetParametersResult(successful=True)

    def trip_planner_controller(self):
        # publish state at 10 Hz so anything subscribing always gets updates
        self.publish_qcar_state()

        t_current = time.time() - self.trip_time

        # Super state 1: drive to taxi hub
        if self.trip_super_state == 1:
            if 5 < t_current < 15 and len(self.path_nodes) == 0 and self.current_path_status is False:
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
                    self.qcar_state = 5  # magenta — parked at taxi hub
                else:
                    # Dwell complete; idle at hub waiting for a ride request.
                    self.qcar_state = 5
                    self.path_nodes = []
                    self.trip_super_state = 2
                    self.current_trip_status = True
                    self.led_timer_reset = False

        # Super state 2: at taxi hub, run ride segments
        if self.trip_super_state == 2:
            if not self.new_ride_requested:
                # Idle at hub between rides — keep LEDs magenta until a new
                # trip is requested via the trip_nodes parameter.
                self.qcar_state = 5
            elif self.ride_complete:
                # Ride finished — back at hub, hold magenta and re-arm.
                self.qcar_state = 5
            else:
                last = len(self.path_nodes) - 1  # index of taxi-hub return
                # Default to green while driving any segment.
                self.qcar_state = 4

                if self.stop_index + 1 <= last:
                    if self.nodes_sent is False:
                        self.current_stop = self.path_nodes[self.stop_index]
                        self.goal_stop = self.path_nodes[self.stop_index + 1]
                        trip = [self.current_stop, self.goal_stop]
                        self.nodes_sent = self.send_request(
                            param_name="node_values",
                            param_value=trip,
                            param_type=ParameterType.PARAMETER_INTEGER_ARRAY,
                            client=self.path_follower_client
                        )

                    # Once the segment is complete and a small grace
                    # window has elapsed, dwell 3 s at the arrived node
                    # with the LED color appropriate for that node.
                    if self.current_path_status is True and t_current > 2:
                        if not self.led_timer_reset:
                            self.led_time_t0 = time.time()
                            self.led_timer_reset = True

                        arrived_at = self.stop_index + 1
                        if arrived_at == 1:
                            arrived_state = 2          # pickup → blue
                        elif arrived_at == last:
                            arrived_state = 5          # back at taxi hub → magenta
                        elif arrived_at == last - 1:
                            arrived_state = 3          # drop-off → orange
                        else:
                            arrived_state = 1          # intermediate stop → red

                        elapsed = time.time() - self.led_time_t0
                        if elapsed < self.led_time:
                            self.qcar_state = arrived_state
                        else:
                            # 3 s dwell finished → advance to the next segment.
                            self.led_timer_reset = False
                            self.nodes_sent = False
                            self.stop_index += 1
                            self.trip_time = time.time()
                            if arrived_at >= last:
                                # We just dwelled at the taxi hub — ride is over.
                                self.ride_complete = True
                                self.new_ride_requested = False
                                self.current_trip_status = True  # re-arm for next ride
                                self.qcar_state = 5
                                self.stop_index = 0
                                self.path_nodes = []
                                self.get_logger().info(
                                    'Ride complete. Magenta @ taxi hub. Awaiting next trip_nodes.'
                                )

        if self.previous_led_state != self.qcar_state:
            self.previous_led_state = self.qcar_state
            self.publish_qcar_state()  # publish immediately on change

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
