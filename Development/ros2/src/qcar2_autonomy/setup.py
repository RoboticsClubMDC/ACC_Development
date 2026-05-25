from setuptools import find_packages, setup
import os 
from glob import glob

package_name = 'qcar2_autonomy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name,'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name,'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu2404',
    maintainer_email='ubuntu2404@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_follower = autonomy.nav_to_pose:main',
            'lane_detection=autonomy.lane_detection:main',
            'trip_planner=autonomy.trip_planner:main',
            'bev_csi_node=autonomy.bev_csi_node:main',
            'stanley_live_plot=autonomy.stanley_live_plot:main',
            'lane_stanley_node=autonomy.lane_stanley_node:main',
            'sidewalk_detection=autonomy.sidewalk_detection:main',
            'bev_csi_seg=autonomy.bev_csi_seg:main',
            'roadmap_alignment_node = autonomy.LCroadmap_alignment_node:main',
            'manual_drive = autonomy.manual_drive:main',
            'visual_odometry = autonomy.vo_node:main',
            'pose_estimator = autonomy.pose_estimator:main',
            'qcar2_ekf_odometry = autonomy.pose_estimator:main',
            'ekf_fusor = autonomy.ekf_fusor:main',
            'controller_watchdog = autonomy.controller_watchdog:main',
        ],
    },
)
