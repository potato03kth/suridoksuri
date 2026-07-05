"""
Phase 2 launch: TelemetryNode + OffboardNode — Offboard 경로 추종.

실행:
  ros2 launch fc_ros phase2.launch.py

테스트용 파라미터 오버라이드 (YAML은 정식값 유지 — 테스트 임시값은 여기로만 준다):
  ros2 launch fc_ros phase2.launch.py vehicle_type:=mc
  ros2 launch fc_ros phase2.launch.py v_cruise:=18.0
  ros2 launch fc_ros phase2.launch.py waypoints:="[0.0,0.0,50.0, 300.0,0.0,50.0]"

TelemetryNode: 진단·모니터링 용도 (VehicleState 로깅).
OffboardNode:  실제 제어 루프 (MAVROS 토픽 직접 구독, 자체 VehicleState 유지).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
import yaml
from ament_index_python.packages import get_package_share_directory


def _make_nodes(context):
    pkg    = get_package_share_directory("fc_ros")
    params = os.path.join(pkg, "params", "fc_ros_params.yaml")

    # YAML 뒤에 dict를 두어 launch 인자로 받은 것만 덮어쓴다.
    # 빈 값(기본)이면 dict에 넣지 않아 YAML 값이 그대로 쓰인다 — 기본값 이중 관리 없음.
    overrides = {"vehicle_type": LaunchConfiguration("vehicle_type").perform(context)}

    v_cruise = LaunchConfiguration("v_cruise").perform(context)
    if v_cruise:
        overrides["v_cruise"] = float(v_cruise)

    waypoints = LaunchConfiguration("waypoints").perform(context)
    if waypoints:
        wps = [float(x) for x in yaml.safe_load(waypoints)]
        if len(wps) % 3 != 0:
            raise ValueError(
                f"waypoints must be a flat [x,y,z, ...] list (len % 3 == 0), got len={len(wps)}")
        overrides["waypoints"] = wps

    return [
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
            parameters=[params, overrides],
            output="screen",
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "vehicle_type", default_value="vtol",
            description='기체 타입: "vtol"(기본) | "mc"(순수 멀티콥터, FW 천이 생략)'),
        DeclareLaunchArgument(
            "v_cruise", default_value="",
            description="테스트용 순항속도 오버라이드 (m/s). 빈 값(기본)이면 YAML 값 사용"),
        DeclareLaunchArgument(
            "waypoints", default_value="",
            description='테스트용 WP 오버라이드, flat 1D: "[x,y,z, x,y,z, ...]". 빈 값(기본)이면 YAML 값 사용'),
        OpaqueFunction(function=_make_nodes),
    ])
