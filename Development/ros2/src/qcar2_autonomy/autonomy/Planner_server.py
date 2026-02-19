#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import UInt8
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter
from rclpy.parameter import ParameterType


class PlannerServer(Node):
    """
    Subscribes to /trip_planner/qcar_state (UInt8: 1..4)
    and sets /qcar2_hardware led_color_id accordingly.

    Mapping:
      qcar_state 1 -> led_color_id 0 (red)
      qcar_state 2 -> led_color_id 2 (blue)
      qcar_state 3 -> led_color_id 3 (yellow)
      qcar_state 4 -> led_color_id 1 (green)
    """

    def __init__(self):
        # IMPORTANT: your runtime node name is /Planner_server
        super().__init__('Planner_server')

        self.declare_parameter('planner_state_topic', '/trip_planner/qcar_state')
        self.declare_parameter('qcar_hardware_node', 'qcar2_hardware')

        planner_state_topic = self.get_parameter('planner_state_topic').value
        qcar_hardware_node = self.get_parameter('qcar_hardware_node').value

        # Ground-truth service from your ros2 service list:
        self.qcar_service_name = f'/{qcar_hardware_node}/set_parameters'

        self.client = self.create_client(SetParameters, self.qcar_service_name)
        self.sub = self.create_subscription(UInt8, planner_state_topic, self.state_cb, 10)

        self.last_led_color_id = None

        self.get_logger().info(
            f'Planner_server listening on {planner_state_topic}, calling {self.qcar_service_name} to set led_color_id'
        )

    def state_cb(self, msg: UInt8):
        qcar_state = int(msg.data)

        if qcar_state == 1:
            led_color_id = 0  # red
        elif qcar_state == 2:
            led_color_id = 2  # blue
        elif qcar_state == 3:
            led_color_id = 3  # yellow
        elif qcar_state == 4:
            led_color_id = 1  # green
        else:
            self.get_logger().warn(f'Unknown qcar_state={qcar_state}, defaulting to green')
            led_color_id = 1

        if self.last_led_color_id == led_color_id:
            return

        self.last_led_color_id = led_color_id
        self.set_led_param(led_color_id)

    def set_led_param(self, led_color_id: int):
        if not self.client.service_is_ready():
            if not self.client.wait_for_service(timeout_sec=0.5):
                self.get_logger().warn(f'Service not available: {self.qcar_service_name}')
                return

        param = Parameter()
        param.name = 'led_color_id'
        param.value.type = ParameterType.PARAMETER_INTEGER
        param.value.integer_value = int(led_color_id)

        request = SetParameters.Request()
        request.parameters = [param]

        future = self.client.call_async(request)
        future.add_done_callback(lambda f: self._handle_response(f, led_color_id))

    def _handle_response(self, future, led_color_id: int):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f'Failed setting led_color_id={led_color_id}: {e}')
            return

        if not resp.results:
            self.get_logger().warn(f'No result returned when setting led_color_id={led_color_id}')
            return

        if not resp.results[0].successful:
            self.get_logger().warn(f'Set led_color_id={led_color_id} rejected: {resp.results[0].reason}')
            return

        self.get_logger().info(f'Set led_color_id={led_color_id}')


def main():
    rclpy.init()
    node = PlannerServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
