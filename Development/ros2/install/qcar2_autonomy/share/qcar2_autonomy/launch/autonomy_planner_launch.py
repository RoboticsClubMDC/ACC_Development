import subprocess

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # NOTE: the old `yolo_detector` (autonomy.yolo_detector_MARKERS_CPU_ABC)
    # was retired 2026-05-24. Semantic detection now lives in qcar2_perception
    # via `semantic_yolo_detector`. The traffic_system_detector that used to
    # alias the old yolo node has been removed from this launch too.

    path_follower = Node(
        package='qcar2_autonomy',
        executable='path_follower',
        name='path_follower',
        parameters=[{'visualize_pose': [True]}],
    )

    trip_planner = Node(
        package='qcar2_autonomy',
        executable='trip_planner',
        name='trip_planner',
        # parameters=[{
        #     'taxi_node': [10],
        #     'trip_nodes': [2, 4, 14, 20, 22, 10],
        # }]
    )

    lane_detection = Node(
        package='qcar2_autonomy',
        executable='lane_detection',
        name='lane_detection',
    )

    lane_stanley_node = Node(
        package='qcar2_autonomy',
        executable='lane_stanley_node',
        name='lane_stanley_node',
    )

    bev_csi_node = Node(
        package='qcar2_autonomy',
        executable='bev_csi_node',
        name='bev_csi_node',
    )

    sidewalk_detection = Node(
        package='qcar2_autonomy',
        executable='sidewalk_detection',
        name='sidewalk_detection',
    )

    bev_csi_seg = Node(
        package='qcar2_autonomy',
        executable='bev_csi_seg',
        name='bev_csi_seg',
    )

    ''' TODO: Once finished this launch file must also include
    - Lane detector to help smooth out tracking of lanes while driving
    - Planner server to coordinate which LEDs on the QCar should be on based on trip logic
    - Hook into qcar2_perception's semantic_yolo_detector for traffic-light/stop-sign awareness
    '''

    return LaunchDescription([
        path_follower,
        trip_planner,
        bev_csi_node,
        lane_detection,
        lane_stanley_node,
        sidewalk_detection,
        bev_csi_seg,
    ])