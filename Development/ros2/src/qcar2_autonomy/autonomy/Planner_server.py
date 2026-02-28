# # =========================
# # OVERALL: PlannerServer.py  (FINAL)
# # ✅ ONLY node that sets /qcar2_hardware led_color_id
# # ✅ Expects /trip_planner/qcar_state to already be the ACTUAL led_color_id (1/2/5/6)
# # =========================
# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node

# from std_msgs.msg import UInt8
# from rcl_interfaces.srv import SetParameters
# from rcl_interfaces.msg import Parameter
# from rclpy.parameter import ParameterType


# class PlannerServer(Node):
#     def __init__(self):
#         super().__init__('planner_server')

#         self.declare_parameter('planner_state_topic', '/trip_planner/qcar_state')
#         self.declare_parameter('qcar_hardware_node', 'qcar2_hardware')

#         planner_state_topic = self.get_parameter('planner_state_topic').value
#         qcar_hardware_node = self.get_parameter('qcar_hardware_node').value

#         self.qcar_service_name = f'/{qcar_hardware_node}/set_parameters'
#         self.client = self.create_client(SetParameters, self.qcar_service_name)
#         self.sub = self.create_subscription(UInt8, planner_state_topic, self.state_cb, 10)

#         self.last_led_color_id = None
#         self.get_logger().info(f'PlannerServer: sub={planner_state_topic}  svc={self.qcar_service_name}')

#     def state_cb(self, msg: UInt8):
#         led_color_id = int(msg.data)

#         # Allowed: 0..6 (you added orange=6)
#         if led_color_id < 0 or led_color_id > 6:
#             self.get_logger().warn(f'Invalid led_color_id={led_color_id}; forcing MAGENTA(5)')
#             led_color_id = 5

#         if self.last_led_color_id == led_color_id:
#             return

#         self.last_led_color_id = led_color_id
#         self.set_led_param(led_color_id)

#     def set_led_param(self, led_color_id: int):
#         if not self.client.service_is_ready():
#             if not self.client.wait_for_service(timeout_sec=0.5):
#                 self.get_logger().warn(f'Service not available: {self.qcar_service_name}')
#                 return

#         param = Parameter()
#         param.name = 'led_color_id'
#         param.value.type = ParameterType.PARAMETER_INTEGER
#         param.value.integer_value = int(led_color_id)

#         req = SetParameters.Request()
#         req.parameters = [param]

#         future = self.client.call_async(req)
#         future.add_done_callback(lambda f: self._handle_response(f, led_color_id))

#     def _handle_response(self, future, led_color_id: int):
#         try:
#             resp = future.result()
#         except Exception as e:
#             self.get_logger().error(f'Failed setting led_color_id={led_color_id}: {e}')
#             return

#         if not resp.results or not resp.results[0].successful:
#             reason = resp.results[0].reason if resp.results else "no results"
#             self.get_logger().warn(f'Set led_color_id={led_color_id} rejected: {reason}')
#             return

#         self.get_logger().info(f'Set led_color_id={led_color_id} OK')


# def main():
#     rclpy.init()
#     node = PlannerServer()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()