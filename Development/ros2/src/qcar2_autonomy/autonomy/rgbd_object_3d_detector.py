import math
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from cv_bridge import CvBridge

class RGBDObject3DDetector(Node):
    def __init__(self):
        super().__init__('rgbd_object_3d_detector')

        self.bridge = CvBridge()

        self.color_img = None
        self.depth_img = None
        self.camera_info = None

        self.create_subscription(Image, '/camera/color_image', self.color_cb, 10)
        self.create_subscription(Image, '/camera/depth_image', self.depth_cb, 10)
        self.fx = 455.2
        self.fy = 459.43
        self.cx = 308.53
        self.cy = 213.56
        self.camera_frame = 'base_link'   # debug frame for RViz

        self.marker_pub = self.create_publisher(MarkerArray, '/object_markers', 10)

        self.timer = self.create_timer(0.2, self.process)

        self.get_logger().info('RGBD Object 3D Detector started.')

    def color_cb(self, msg):
        self.color_img = msg

    def depth_cb(self, msg):
        self.depth_img = msg

    def info_cb(self, msg):
        self.camera_info = msg

    def process(self):
        if self.color_img is None or self.depth_img is None:
            self.get_logger().info(
                f'Waiting: color={self.color_img is not None}, '
                f'depth={self.depth_img is not None}, '
                f'info={self.camera_info is not None}',
                throttle_duration_sec=2.0
            )
            return
        
        depth = self.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='passthrough')
        self.get_logger().info(
            f"depth dtype={depth.dtype}, shape={depth.shape}, "
            f"min={np.nanmin(depth)}, max={np.nanmax(depth)}",
            throttle_duration_sec=1.0
        )
        h, w = depth.shape[:2]
        u =  w // 2
        v = h // 2


        #likely to change after
        patch = depth[max(0, v-10):v+10, max(0, u-10):u+10].astype(np.float32)
        valid = patch[np.isfinite(patch) & (patch > 0)]

        if valid.size == 0:
            self.get_logger().warn('No valid depth near center.', throttle_duration_sec=1.0)
            return

        z = float(np.median(valid))


        # if depth is in milimeters, convert to meters, TODO: DEACTIVE THIS JUST
        if z > 20.0:
            z = z / 1000.0

        if not math.isfinite(z) or z <= 0.0:
            self.get_logger().warn('Invalid center depth.')
            return

        k = self.camera_info.k
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        self.get_logger().info(
            f'Center pixel ({u},{v}) depth={z:.3f}m -> camera point '
            f'X={x:.3f}, Y={y:.3f}, Z={z:.3f}',
            throttle_duration_sec=1.0
        )

        markers = MarkerArray()
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.camera_frame
        marker.ns = 'rgbd_debug'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.scale.z = 0.08
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 1.0

        markers.markers.append(marker)
        self.marker_pub.publish(markers)


def main():
    rclpy.init()
    node = RGBDObject3DDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()