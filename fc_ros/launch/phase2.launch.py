"""
Phase 2 launch: TelemetryNode + OffboardNode — Offboard 경로 추종.

실행:
  ros2 launch fc_ros phase2.launch.py

TelemetryNode: 진단·모니터링 용도 (VehicleState 로깅).
OffboardNode:  실제 제어 루프 (MAVROS 토픽 직접 구독, 자체 VehicleState 유지).
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
            executable="telemetry_node",
            name="telemetry_node",
            output="screen",
        ),
        Node(
            package="fc_ros",
            executable="offboard_node",
            name="offboard_node",
            parameters=[params],
            output="screen",
        ),
    ])
