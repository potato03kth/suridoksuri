"""
Phase 1 launch: MissionNode — 경로 생성 후 MAVROS 미션 업로드.

실행:
  ros2 launch fc_ros phase1.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg    = get_package_share_directory("fc_ros")
    params = os.path.join(pkg, "params", "fc_ros_params.yaml")

    return LaunchDescription([
        Node(
            package="fc_ros",
            executable="mission_node",
            name="mission_node",
            parameters=[params],
            output="screen",
        ),
    ])
