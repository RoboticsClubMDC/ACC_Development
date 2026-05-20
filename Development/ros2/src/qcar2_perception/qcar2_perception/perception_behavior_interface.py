import rclpy
from rclpy.node import Node


def main(args=None):
    rclpy.init(args=args)
    node = Node("perception_behavior_interface")
    node.get_logger().warn("perception_behavior_interface is a placeholder.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
