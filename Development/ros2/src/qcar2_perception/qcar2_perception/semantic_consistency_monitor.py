import rclpy
from rclpy.node import Node


def main(args=None):
    rclpy.init(args=args)
    node = Node("semantic_consistency_monitor")
    node.get_logger().warn("semantic_consistency_monitor is a placeholder.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

