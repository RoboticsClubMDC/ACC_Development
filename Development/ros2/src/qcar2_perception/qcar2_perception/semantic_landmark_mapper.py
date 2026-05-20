import rclpy
from rclpy.node import Node


def main(args=None):
    rclpy.init(args=args)
    node = Node("semantic_landmark_mapper")
    node.get_logger().warn("semantic_landmark_mapper is a placeholder.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

