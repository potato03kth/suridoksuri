"""
Phase 2 launch: TelemetryNode + OffboardNode — Offboard 경로 추종.

실행:
  ros2 launch fc_ros phase2.launch.py

TelemetryNode: 진단·모니터링 용도 (VehicleState 로깅).
OffboardNode:  실제 제어 루프 (MAVROS 토픽 직접 구독, 자체 VehicleState 유지).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg    = get_package_share_directory("fc_ros")
    params = os.path.join(pkg, "params", "fc_ros_params.yaml")

    # 기체 타입 런타임 오버라이드. 기본 "vtol"(YAML 값과 동일) — 개발 워크플로 불변.
    #   순수 멀티콥터 테스트:  ros2 launch fc_ros phase2.launch.py vehicle_type:=mc
    vehicle_type = LaunchConfiguration("vehicle_type")

    return LaunchDescription([
        DeclareLaunchArgument(
            "vehicle_type", default_value="vtol",
            description='기체 타입: "vtol"(기본) | "mc"(순수 멀티콥터, FW 천이 생략)'),
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
            # YAML 뒤에 dict를 두어 vehicle_type만 런타임 인자로 덮어쓴다.
            parameters=[params, {"vehicle_type": vehicle_type}],
            output="screen",
        ),
    ])
