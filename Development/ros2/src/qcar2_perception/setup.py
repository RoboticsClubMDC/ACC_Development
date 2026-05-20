from setuptools import find_packages, setup
from glob import glob
import os

package_name = "qcar2_perception"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "maps"), glob("maps/*.json")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ubuntu2404",
    maintainer_email="ubuntu2404@todo.todo",
    description="Semantic perception layer for QCar2.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "d435_aligned_source = qcar2_perception.d435_aligned_source:main",
            "semantic_yolo_detector = qcar2_perception.semantic_yolo_detector:main",
            "object_3d_estimator = qcar2_perception.object_3d_estimator:main",
            "semantic_landmark_mapper = qcar2_perception.semantic_landmark_mapper:main",
            "semantic_consistency_monitor = qcar2_perception.semantic_consistency_monitor:main",
            "perception_behavior_interface = qcar2_perception.perception_behavior_interface:main",
        ],
    },
)