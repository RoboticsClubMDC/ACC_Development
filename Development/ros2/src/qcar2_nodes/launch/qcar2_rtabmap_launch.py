"""qcar2_rtabmap_launch.py - VSLAM showcase launch (intended for bag playback)

Wraps the stock rtabmap_launch/rtabmap.launch.py with the QCar2's camera
topic names pinned, and a single `odom_source` switch that selects one of
the three VSLAM showcase variants we want to compare on the same recorded
bag (record-once / playback-three-ways):

  odom_source:=rtabmap_odom   Pure-visual SLAM. RTAB-Map runs its own
                              internal visual odometry (rtabmap_odom node)
                              on the bagged RGB-D stream. Default.
  odom_source:=vo_node        Uses our /vo/odometry (from the bag) as the
                              pose source. rtabmap_odom is disabled.
                              Notes:
                                - If the bag was recorded with vo_node's
                                  force_cart_yaw=True (default), this run
                                  is "visual + IMU yaw correction", not
                                  pure-visual.
                                - For a true pure-visual ours-end-to-end
                                  comparison, record a second bag with
                                  force_cart_yaw:=false on vo_node.
  odom_source:=cartographer   Uses /odom (from the bag) as the pose
                              source. Map is built from camera, trajectory
                              from lidar+IMU+wheel. Provided as a
                              completeness comparison; not the headline.

Defaults assume bag playback (use_sim_time:=true). For a hypothetical
live run, pass use_sim_time:=false.

Topic-conflict reminder (handle on the `ros2 bag play` side, NOT here):
  - rtabmap_odom mode: exclude /odom and /tf from playback, otherwise
    the bag's Cartographer /odom will collide with rtabmap_odom's /odom
    publication, and the bag's /tf will fight rtabmap_odom's odom->base
    TF.
  - vo_node mode: exclude /odom from playback. /vo/odometry carries the
    pose; whether /tf needs to be included depends on whether vo_node
    publishes the odom->base_link TF (verify before recording).
  - cartographer mode: keep /odom and /tf; that's the point.

This file is pure glue. RTAB-Map's own VO, feature detection, descriptor
extraction, pose-graph optimization, loop closure, and point-cloud
generation all live in the installed ros-humble-rtabmap-* packages and
are not redefined here.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.substitutions import FindPackageShare


_VALID_ODOM_SOURCES = ('rtabmap_odom', 'vo_node', 'cartographer')


def _resolve_db_path(raw: str) -> str:
    return os.path.expanduser(os.path.expandvars(raw))


def _build_rtabmap_include(context, *args, **kwargs):
    odom_source = LaunchConfiguration('odom_source').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    database_path = _resolve_db_path(LaunchConfiguration('database_path').perform(context))
    localization = LaunchConfiguration('localization').perform(context)
    rtabmapviz = LaunchConfiguration('rtabmapviz').perform(context)
    rviz = LaunchConfiguration('rviz').perform(context)
    qos = LaunchConfiguration('qos').perform(context)

    if odom_source not in _VALID_ODOM_SOURCES:
        raise ValueError(
            f"odom_source must be one of {_VALID_ODOM_SOURCES!r}, got {odom_source!r}"
        )

    common_args = {
        'rgb_topic': '/camera/color_image',
        'depth_topic': '/camera/depth_image',
        'camera_info_topic': '/camera/camera_info',
        'frame_id': 'base_link',
        'approx_sync': 'true',
        'rgbd_sync': 'true',
        # The camera_bridge publishes /camera/* as Best Effort (sensor QoS),
        # and a bag of it replays Best Effort. RTAB-Map defaults to Reliable
        # (qos=1), which is INCOMPATIBLE with a Best Effort publisher -> no
        # frames are delivered. Default this wrapper to qos=2 (Best Effort)
        # so it matches both the live bridge and a recorded bag with no QoS
        # override needed. (If you must keep a Reliable RTAB-Map, instead
        # force the bag player Reliable via --qos-profile-overrides-path.)
        'qos': qos,
        'use_sim_time': use_sim_time,
        'database_path': database_path,
        'localization': localization,
        'rtabmapviz': rtabmapviz,
        'rviz': rviz,
        'queue_size': '30',
    }

    if odom_source == 'rtabmap_odom':
        common_args['visual_odometry'] = 'true'
    elif odom_source == 'vo_node':
        # /vo/odometry is stamped frame_id='map' with no odom->base_link TF,
        # which RTAB-Map cannot ingest as external odometry: the 'map' frame
        # collides with RTAB-Map's own map frame and the TF tree breaks
        # ("two or more unconnected trees"), leaving the map stuck at 1 node
        # (Physical Test 12 variant B, 2026-05-21). REQUIRES the relay node
        # vo_odom_tf_relay.py running, which republishes /vo/odometry as
        # /vo/odom_relay in a clean 'odom' frame + broadcasts odom->base TF:
        #   python3 .../qcar2_autonomy/autonomy/vo_odom_tf_relay.py \
        #     --ros-args -p use_sim_time:=true
        common_args['visual_odometry'] = 'false'
        common_args['odom_topic'] = '/vo/odom_relay'
    else:  # cartographer
        # The QCar's Cartographer does NOT publish a nav_msgs/Odometry on
        # /odom — it exposes the robot pose only through TF (map ->
        # base_link). So RTAB-Map must read odometry from the TF tree
        # (odom_frame_id) rather than subscribing to an odometry topic.
        # Caveat: 'map' carries loop-closure jumps (it is not a strictly
        # continuous odom frame), so this variant can show discontinuities;
        # acceptable for the showcase A/B/C comparison. publish_tf_map is
        # disabled so RTAB-Map does not fight Cartographer for the map frame.
        common_args['visual_odometry'] = 'false'
        common_args['odom_frame_id'] = 'map'
        common_args['publish_tf_map'] = 'false'

    rtabmap_launch_path = [
        FindPackageShare('rtabmap_launch'),
        '/launch/rtabmap.launch.py',
    ]

    return [
        LogInfo(msg=f"[qcar2_rtabmap_launch] odom_source={odom_source}  database_path={database_path}  localization={localization}"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(rtabmap_launch_path),
            launch_arguments=list(common_args.items()),
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'odom_source',
            default_value='rtabmap_odom',
            description=(
                'Pose source for RTAB-Map. One of: '
                'rtabmap_odom (pure visual, default) | '
                'vo_node (our /vo/odometry from bag) | '
                'cartographer (/odom from bag)'
            ),
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='true for bag playback (default), false for a live run',
        ),
        DeclareLaunchArgument(
            'qos',
            default_value='2',
            description=(
                'Sensor-input QoS reliability: 0=system default, 1=Reliable, '
                '2=Best Effort (default). camera_bridge and a bag of it are '
                'Best Effort, so 2 matches with no override. Use 1 only if '
                'the input publisher is Reliable.'
            ),
        ),
        DeclareLaunchArgument(
            'database_path',
            default_value='~/vo_rtab_bags/rtabmap.db',
            description='Persistent .db output path. Overwrite per run, e.g. rtabmap_odom.db, vo_node.db, cartographer.db',
        ),
        DeclareLaunchArgument(
            'localization',
            default_value='false',
            description='false=mapping (default), true=replay a saved db read-only',
        ),
        DeclareLaunchArgument(
            'rtabmapviz',
            default_value='true',
            description='Launch the rtabmap_viz GUI (recommended for the showcase)',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='false',
            description='Launch RViz alongside rtabmap_viz',
        ),
        OpaqueFunction(function=_build_rtabmap_include),
    ])
