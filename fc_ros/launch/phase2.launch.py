"""
Phase 2 launch: TelemetryNode + OffboardNode — Offboard 경로 추종.

실행:
  ros2 launch fc_ros phase2.launch.py

테스트용 파라미터 오버라이드 (YAML은 정식값 유지 — 테스트 임시값은 여기로만 준다):
  ros2 launch fc_ros phase2.launch.py vehicle_type:=mc
  ros2 launch fc_ros phase2.launch.py v_cruise:=18.0
  ros2 launch fc_ros phase2.launch.py waypoints:="[0.0,0.0,50.0, 300.0,0.0,50.0]"
  ros2 launch fc_ros phase2.launch.py vehicle_type:=mc transition_alt:=4.0 waypoints:="[0.0,0.0,4.0, 8.0,0.0,4.0]"
  ros2 launch fc_ros phase2.launch.py vehicle_type:=mc mc_wp_settle_time:=3.0   # WP마다 3초씩 정착
  ros2 launch fc_ros phase2.launch.py d_end_thresh:=30.0        # 역천이 진입 거리 스윕 (SITL-7 C5)
  ros2 launch fc_ros phase2.launch.py entry_mode:=mid_flight    # ENTRY 상태 경로 (SITL-7 C10)
  ros2 launch fc_ros phase2.launch.py planner:=straight l1_dist:=30.0

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

# 위생 검사 로직은 노드와 공유한다 — 세 층이 같은 문자 집합을 보게 하고,
# launch/rclpy 없이 테스트 가능하게 하려는 것이다 (fc_ros/test/test_param_hygiene.py).
from fc_ros.param_hygiene import check_launch_argv, check_value


# launch 인자 위생 검사 — 배경·3겹 구조·사고 경위는 `fc_ros/param_hygiene.py` docstring.
# 요약: 2026-07-28 실비행 2건이 인자에 섞인 U+00A0 하나로 날아갔다(flight01 즉사,
# flight02 는 transition_alt 가 통째로 유실된 채 **조용히** 기동 → 천이 실패).


def _arg(name, context):
    """launch 인자를 읽고 위생 검사한다. 통과하면 값을 그대로 돌려준다."""
    return check_value(name, LaunchConfiguration(name).perform(context))


def _make_nodes(context):
    pkg    = get_package_share_directory("fc_ros")
    params = os.path.join(pkg, "params", "fc_ros_params.yaml")

    # YAML 뒤에 dict를 두어 launch 인자로 받은 것만 덮어쓴다.
    # 빈 값(기본)이면 dict에 넣지 않아 YAML 값이 그대로 쓰인다 — 기본값 이중 관리 없음.
    overrides = {"vehicle_type": _arg("vehicle_type", context)}

    for name in ("v_cruise", "transition_alt",
                 "mc_end_thresh", "mc_wp_settle_time",
                 # SITL-7 회귀 캠페인용 확장 (docs/sitl_vtol_campaign.md 3장)
                 "d_end_thresh", "v_approach", "l1_dist", "wp0_heading_tol",
                 "hold_timeout", "landing_timeout", "control_hz",
                 # SITL-7 R1 — 상태 타임아웃 4종 + 거리 상한 (현장 조정은 여기로만)
                 "climbing_timeout", "transition_fw_timeout",
                 "transition_mc_timeout", "entry_timeout", "range_limit_m",
                 # SITL-7 R5 — 천이 고도 계단 램프 (F-9)
                 "alt_slew_rate"):
        val = _arg(name, context)
        if val:
            overrides[name] = float(val)

    for name in ("entry_mode", "planner"):
        val = _arg(name, context)
        if val:
            overrides[name] = val

    waypoints = _arg("waypoints", context)
    if waypoints:
        wps = [float(x) for x in yaml.safe_load(waypoints)]
        if len(wps) % 3 != 0:
            raise ValueError(
                f"waypoints must be a flat [x,y,z, ...] list (len % 3 == 0), got len={len(wps)}")
        overrides["waypoints"] = wps

    waypoint_frame = _arg("waypoint_frame", context)
    if waypoint_frame:
        overrides["waypoint_frame"] = waypoint_frame

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
    args = [
        DeclareLaunchArgument(
            "vehicle_type", default_value="vtol",
            description='기체 타입: "vtol"(기본) | "mc"(순수 멀티콥터, FW 천이 생략)'),
        DeclareLaunchArgument(
            "v_cruise", default_value="",
            description="테스트용 순항속도 오버라이드 (m/s). 빈 값(기본)이면 YAML 값 사용"),
        DeclareLaunchArgument(
            "waypoints", default_value="",
            description='테스트용 WP 오버라이드, flat 1D: "[x,y,z, x,y,z, ...]". 빈 값(기본)이면 YAML 값 사용'),
        DeclareLaunchArgument(
            "transition_alt", default_value="",
            description="테스트용 천이/이륙 고도 오버라이드 (m). 빈 값(기본)이면 YAML 값 사용 — MC 저고도 벤치테스트 필수"),
        DeclareLaunchArgument(
            "mc_end_thresh", default_value="",
            description="MC WP 도달 판정 반경 오버라이드 (m). 빈 값(기본)이면 YAML 값 사용"),
        DeclareLaunchArgument(
            "mc_wp_settle_time", default_value="",
            description="MC WP 정착 유지 시간 오버라이드 (s). 0.0이면 정착 없이 통과(종전 fly-by). "
                        "빈 값(기본)이면 YAML 값 사용"),
        DeclareLaunchArgument(
            "waypoint_frame", default_value="",
            description='waypoints 기준계: "takeoff"(기본, 이륙지점 상대) | "local"(EKF 로컬 절대, 종전 동작). '
                        '빈 값이면 YAML 값 사용'),
        # ── SITL-7 VTOL 회귀 캠페인 확장 인자 (docs/sitl_vtol_campaign.md 3장) ──
        # 전부 빈 문자열 기본값 = YAML 값 사용. 테스트 임시값은 YAML을 고치지 않고
        # 여기로만 준다(프로젝트 규율).
        DeclareLaunchArgument(
            "d_end_thresh", default_value="",
            description="FW/VTOL 역천이 진입 거리 기준 오버라이드 (m). 시나리오 C5 스윕용. "
                        "빈 값(기본)이면 YAML 값 사용"),
        DeclareLaunchArgument(
            "entry_mode", default_value="",
            description='진입 모드: "pre_takeoff"(기본) | "mid_flight"(ENTRY 상태 경유). '
                        '시나리오 C10용. 빈 값이면 YAML 값 사용'),
        DeclareLaunchArgument(
            "planner", default_value="",
            description='경로 플래너: "auto"(기본) | "eta3" | "diterpin" | "straight". '
                        '빈 값이면 YAML 값 사용'),
        DeclareLaunchArgument(
            "v_approach", default_value="",
            description="ENTRY 접근 속도 / MC 위치 setpoint 슬루레이트 (m/s). "
                        "빈 값(기본)이면 YAML 값 사용"),
        DeclareLaunchArgument(
            "l1_dist", default_value="",
            description="L1 유도 lookahead 거리 (m). 빈 값(기본)이면 YAML 값 사용"),
        DeclareLaunchArgument(
            "wp0_heading_tol", default_value="",
            description="헤딩 정렬 허용 오차 (rad). TRANSITION_FW 정렬·ENTRY 공용. "
                        "빈 값(기본)이면 YAML 값 사용"),
        DeclareLaunchArgument(
            "hold_timeout", default_value="",
            description="WP1 홀드 타임아웃 (s). 빈 값(기본)이면 YAML 값 사용"),
        DeclareLaunchArgument(
            "landing_timeout", default_value="",
            description="AUTO.LAND 타임아웃 (s). 빈 값(기본)이면 YAML 값 사용"),
        DeclareLaunchArgument(
            "control_hz", default_value="",
            description="제어 루프 주파수 (Hz, ≥2). 빈 값(기본)이면 YAML 값 사용"),
        # ── SITL-7 R1 — 상태 타임아웃 4종 + 거리 상한 ────────────────────
        # 전부 0 이하를 주면 비활성(종전 무한대기 동작). 초과 시 안전 폴백
        # (`_request_override()` → manual 시도 → AUTO.LOITER).
        DeclareLaunchArgument(
            "climbing_timeout", default_value="",
            description="CLIMBING 체류 상한 (s). 0 이하면 비활성. 빈 값(기본)이면 YAML(120.0)"),
        DeclareLaunchArgument(
            "transition_fw_timeout", default_value="",
            description="TRANSITION_FW 체류 상한 (s). 0 이하면 비활성. 빈 값(기본)이면 YAML(90.0)"),
        DeclareLaunchArgument(
            "transition_mc_timeout", default_value="",
            description="TRANSITION_MC 체류 상한 (s). 0 이하면 비활성. 빈 값(기본)이면 YAML(30.0)"),
        DeclareLaunchArgument(
            "entry_timeout", default_value="",
            description="ENTRY 체류 상한 (s). 0 이하면 비활성. 빈 값(기본)이면 YAML(60.0)"),
        DeclareLaunchArgument(
            "range_limit_m", default_value="",
            description="이륙지점 기준 수평거리 상한 (m). 0 이하면 비활성. "
                        "빈 값(기본)이면 YAML(300.0). 300m 넘는 편도 경로를 시험할 땐 함께 키울 것"),
        DeclareLaunchArgument(
            "alt_slew_rate", default_value="",
            description="FW 위치 setpoint 고도 램프 (m/s). 0 이하면 비활성(= 종전 계단). "
                        "빈 값(기본)이면 YAML(3.0). F-9 — transition_alt != wp[-1].z 일 때의 천이 고도 계단"),
    ]

    # 선언 목록이 곧 검사 기준이다 — 인자를 추가해도 여기 손댈 필요가 없다.
    # `_make_nodes`(OpaqueFunction)보다 **먼저** 돌아야 오염된 인자로 노드가 뜨는 걸 막는다.
    check_launch_argv({a.name for a in args})

    return LaunchDescription(args + [OpaqueFunction(function=_make_nodes)])
