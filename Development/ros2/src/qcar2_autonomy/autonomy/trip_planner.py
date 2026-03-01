#!/usr/bin/env python3

import time
import numpy as np
from enum import Enum

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, SetParametersResult
from rclpy.parameter import ParameterType

from hal.products.mats import SDCSRoadMap


class MissionStage(Enum):
    IDLE           = 0
    TO_PICKUP      = 1
    WAIT_AT_PICKUP = 2
    TO_DROPOFF     = 3
    WAIT_AT_DROPOFF = 4
    TO_HUB         = 5


class TripPlanner(Node):

    def __init__(self):
        super().__init__('trip_planner')

        self.roadmap = SDCSRoadMap()

        self.qcar_hardware_client = self.create_client(SetParameters, '/qcar2_hardware/set_parameters')
        if not self.qcar_hardware_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('qcar2_hardware LED service not available — LEDs disabled')
            self.qcar_hardware_client = None
        else:
            self.get_logger().info('connected to qcar2_hardware.')

        self.declare_parameter('taxi_node',          [10])
        self.declare_parameter('pickup_xy',          [0.125, 4.395])
        self.declare_parameter('dropoff_xy',         [-0.905, 0.800])
        self.declare_parameter('stop_seconds',       [3.0])
        self.declare_parameter('rotation_offset',    [82.0])
        self.declare_parameter('translation_offset', [0.0, 0.0])

        self.taxi_node          = int(list(self.get_parameter('taxi_node').get_parameter_value().integer_array_value)[0])
        self.pickup_xy          = list(self.get_parameter('pickup_xy').get_parameter_value().double_array_value)
        self.dropoff_xy         = list(self.get_parameter('dropoff_xy').get_parameter_value().double_array_value)
        self.stop_seconds       = float(list(self.get_parameter('stop_seconds').get_parameter_value().double_array_value)[0])
        self.rotation_offset    = list(self.get_parameter('rotation_offset').get_parameter_value().double_array_value)
        self.translation_offset = list(self.get_parameter('translation_offset').get_parameter_value().double_array_value)

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        self.hub_xy = list(self._get_node_xy(self.taxi_node))
        self.get_logger().info(f'Hub at node {self.taxi_node}: {self.hub_xy}')

        self.LED_GREEN   = 1
        self.LED_BLUE    = 2
        self.LED_MAGENTA = 5
        self.LED_ORANGE  = 6

        self.robot_pose             = None
        self.path_status            = False
        self._path_completed_event  = False

        self.startup_done          = False
        self._startup_path_sent    = False
        self.ready_for_rides       = False
        self.new_ride_requested    = False

        self.mission_stage = MissionStage.IDLE
        self.pause_until   = 0.0
        self.picked_up     = False
        self.dropped_off   = False
        self._last_led     = None

        self.waypoints_pub = self.create_publisher(Path, '/cmd_waypoints', 1)

        self.create_subscription(Bool,        '/path_status',  self.path_status_callback,  10)
        self.create_subscription(PoseStamped, '/robot_pose',   self.robot_pose_callback,   10)

        self._set_led(self.LED_MAGENTA)
        self.create_timer(0.1, self.loop)

    def path_status_callback(self, msg):
        prev = self.path_status
        self.path_status = bool(msg.data)
        if not prev and self.path_status:
            self._path_completed_event = True

    def robot_pose_callback(self, msg: PoseStamped):
        self.robot_pose = msg

    def parameter_update_callback(self, params):
        for p in params:
            if p.name == 'pickup_xy' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.pickup_xy = list(p.value)
                self.get_logger().info(f'pickup_xy updated: {self.pickup_xy}')
            elif p.name == 'dropoff_xy' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.dropoff_xy = list(p.value)
                self.get_logger().info(f'dropoff_xy updated: {self.dropoff_xy}')
            elif p.name == 'stop_seconds' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.stop_seconds = float(list(p.value)[0])
            elif p.name == 'rotation_offset' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.rotation_offset = list(p.value)
            elif p.name == 'translation_offset' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.translation_offset = list(p.value)

        if self.ready_for_rides and self.mission_stage == MissionStage.IDLE:
            self.new_ride_requested = True

        return SetParametersResult(successful=True)

    def _rot_matrix(self):
        angle = float(self.rotation_offset[0]) * np.pi / 180.0
        return np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle),  np.cos(-angle)]
        ])

    def _ros_to_qlabs(self, x, y):
        R_mat = self._rot_matrix()
        t     = np.array(self.translation_offset)
        return tuple((np.array([float(x), float(y)]) @ R_mat.T) - t)

    def _qlabs_path_to_ros(self, wp_2xn):
        R_mat   = self._rot_matrix()
        t_off   = np.array(self.translation_offset)
        path_msg = Path()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        for i in range(wp_2xn.shape[1]):
            pt   = (wp_2xn[:, i] + t_off) @ R_mat
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(pt[0])
            pose.pose.position.y = float(pt[1])
            path_msg.poses.append(pose)
        return path_msg

    def _get_node_xy(self, node_id):
        if hasattr(self.roadmap, 'get_node_pose'):
            pose = np.array(self.roadmap.get_node_pose(node_id)).reshape(-1)
        else:
            pose = np.array(self.roadmap.nodes[node_id].pose).reshape(-1)
        return float(pose[0]), float(pose[1])

    def _node_count(self):
        if hasattr(self.roadmap, 'nodes'):
            try:
                return len(self.roadmap.nodes)
            except Exception:
                pass
        if hasattr(self.roadmap, 'get_node_pose'):
            i = 0
            while True:
                try:
                    self.roadmap.get_node_pose(i)
                    i += 1
                    if i > 1000:
                        return i
                except Exception:
                    return i
        return 0

    def _closest_node(self, x, y):
        best_i, best_d = 0, float('inf')
        for i in range(self._node_count()):
            try:
                nx, ny = self._get_node_xy(i)
                d = (nx - x) ** 2 + (ny - y) ** 2
                if d < best_d:
                    best_d = d
                    best_i = i
            except Exception:
                continue
        return best_i

    def _plan_to_xy(self, goal_xy):
        if self.robot_pose is None:
            return None

        rx, ry   = float(self.robot_pose.pose.position.x), float(self.robot_pose.pose.position.y)
        px_q, py_q = self._ros_to_qlabs(rx, ry)

        start_node = self._closest_node(px_q, py_q)
        goal_node  = self._closest_node(float(goal_xy[0]), float(goal_xy[1]))

        if hasattr(self.roadmap, 'find_shortest_path'):
            base = self.roadmap.find_shortest_path(start_node, goal_node)
        else:
            base = self.roadmap.generate_path([start_node, goal_node])

        if base is None or np.size(base) == 0:
            self.get_logger().error(f'No path from node {start_node} to {goal_node}')
            return None

        goal_col = np.array(goal_xy).reshape(2, 1)
        return np.hstack([np.array(base), goal_col])

    def _send_path_to(self, goal_xy, label=''):
        wp = self._plan_to_xy(goal_xy)
        if wp is None:
            return False
        self.waypoints_pub.publish(self._qlabs_path_to_ros(wp))
        self.get_logger().info(f'Path published -> {label} ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})')
        return True

    def _snap_to_exact(self, goal_xy, label=''):
        if self.robot_pose is None:
            return self._send_path_to(goal_xy, label)

        rx, ry = float(self.robot_pose.pose.position.x), float(self.robot_pose.pose.position.y)
        R_mat  = self._rot_matrix()
        t      = np.array(self.translation_offset)
        cur_q  = np.array([rx, ry]) @ R_mat.T - t
        goal_q = np.array([float(goal_xy[0]), float(goal_xy[1])])

        dist = float(np.linalg.norm(cur_q - goal_q))
        if dist < 0.10:
            self.get_logger().info(f'Already within 0.10m of {label}, no snap needed')
            return True

        wp = np.stack([cur_q, goal_q], axis=1)
        self.waypoints_pub.publish(self._qlabs_path_to_ros(wp))
        self.get_logger().info(f'Snap path -> {label} ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) dist={dist:.3f}m')
        return True

    def _set_led(self, led_id: int):
        if self.qcar_hardware_client is None:
            return
        led_id = int(led_id)
        if self._last_led == led_id:
            return
        self._last_led = led_id
        param = Parameter()
        param.name                  = 'led_color_id'
        param.value.type            = ParameterType.PARAMETER_INTEGER
        param.value.integer_value   = led_id
        req                         = SetParameters.Request()
        req.parameters              = [param]
        self.qcar_hardware_client.call_async(req)

    def loop(self):
        now = time.time()

        if self.robot_pose is None:
            return

        if not self.startup_done:
            if not self._startup_path_sent:
                ok = self._send_path_to(self.hub_xy, label='HUB (startup)')
                if ok:
                    self._startup_path_sent = True
                return
            if self._path_completed_event:
                self._path_completed_event = False
                self.startup_done    = True
                self.ready_for_rides = True
                self._set_led(self.LED_MAGENTA)
                self.get_logger().info('Startup complete. Ready for rides.')
            return

        if self.ready_for_rides and not self.new_ride_requested:
            return

        if self.ready_for_rides and self.new_ride_requested:
            self.new_ride_requested = False
            self.ready_for_rides    = False
            self.picked_up          = False
            self.dropped_off        = False
            ok = self._send_path_to(self.pickup_xy, label='PICKUP')
            if ok:
                self.mission_stage = MissionStage.TO_PICKUP
                self._set_led(self.LED_GREEN)
            return

        if self.mission_stage == MissionStage.WAIT_AT_PICKUP and now >= self.pause_until:
            ok = self._send_path_to(self.dropoff_xy, label='DROPOFF')
            if ok:
                self.mission_stage = MissionStage.TO_DROPOFF
                self._set_led(self.LED_BLUE)
            return

        if self.mission_stage == MissionStage.WAIT_AT_DROPOFF and now >= self.pause_until:
            ok = self._send_path_to(self.hub_xy, label='HUB')
            if ok:
                self.mission_stage = MissionStage.TO_HUB
                self._set_led(self.LED_ORANGE)
            return

        if now < self.pause_until:
            return

        if not self._path_completed_event:
            return
        self._path_completed_event = False

        if self.mission_stage == MissionStage.TO_PICKUP:
            self._snap_to_exact(self.pickup_xy, label='PICKUP-snap')
            self.mission_stage = MissionStage.WAIT_AT_PICKUP
            self.picked_up     = True
            self._set_led(self.LED_BLUE)
            self.pause_until   = now + self.stop_seconds
            self.get_logger().info('Arrived at PICKUP.')

        elif self.mission_stage == MissionStage.TO_DROPOFF:
            self._snap_to_exact(self.dropoff_xy, label='DROPOFF-snap')
            self.mission_stage = MissionStage.WAIT_AT_DROPOFF
            self.dropped_off   = True
            self._set_led(self.LED_ORANGE)
            self.pause_until   = now + self.stop_seconds
            self.get_logger().info('Arrived at DROPOFF.')

        elif self.mission_stage == MissionStage.TO_HUB:
            self.mission_stage = MissionStage.IDLE
            self.picked_up     = False
            self.dropped_off   = False
            self._set_led(self.LED_MAGENTA)
            self.ready_for_rides = True
            self.get_logger().info('Mission complete. Ready for next ride.')


def main():
    rclpy.init()
    node = TripPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()