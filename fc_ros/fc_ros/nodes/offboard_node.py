"""
OffboardNode: create_timer 기반 상태머신으로 Offboard 경로 추종.

기존 OffboardFollower의 while + time.sleep 루프를 ROS2 타이머 콜백으로 변환.

상태 머신:
  ARM_TAKEOFF   : ARM + AUTO.TAKEOFF 명령
  CLIMBING      : 천이 고도 도달 대기
  TRANSITION_FW : MC→FW 천이 명령 + vtol_state==FW 대기
  STREAMING     : 위치 세트포인트 스트리밍 후 OFFBOARD 전환 요청
                  (FW: lookahead 위치 setpoint / MC: 현재위치 홀드)
  ENTRY         : entry_mode="mid_flight" 시에만 통과; WP0 진입 + 헤딩 정렬 대기
  FOLLOWING     : L1Guidance 기반 경로 추종
  TRANSITION_MC : FW→MC 역천이 명령 + vtol_state==MC 대기 (직선 감속)
  HOLD          : MC로 WP1 복귀·홀드 → 도달+안정 시 LANDING (WP1 지점 착륙)
  LANDING       : AUTO.LAND 명령 + disarmed 대기
  OVERRIDE      : 긴급 수동 전환 — manual 모드 시도 → 미진입 시 AUTO.LOITER 폴백
  DONE          : 착륙 완료 (타이머 계속 동작, 속도=0 유지)

MAVROS 토픽을 직접 구독해 TelemetryNode에 의존하지 않는다.
판정 순수 함수: fc_bridge.execution.state_logic (rclpy 없이 테스트 가능).

안전 감시 (2026-07-27 SITL-7 R1 — 제어법칙은 그대로, "실패 시 어디로 떨어지는지"만 정의):
  · 상태 체류 타임아웃 4종 — CLIMBING / TRANSITION_FW / TRANSITION_MC / ENTRY
  · 이륙지점 기준 수평거리 상한 (range_limit_m)
  둘 다 초과 시 **강제 진행이 아니라** 이미 실증된 `_request_override()`
  (manual 시도 → AUTO.LOITER 폴백)로 떨어진다. 전부 0 이하로 주면 비활성.
"""
from __future__ import annotations
from fc_ros.param_hygiene import check_choice, VEHICLE_TYPES, ENTRY_MODES
from fc_ros.adapters.setpoint_publisher import SetpointPublisher
from fc_ros.adapters.vehicle_state_bridge import (
    update_from_pose,
    update_from_twist,
    update_from_mavros_state,
    update_from_extended_state,
)
from fc_bridge.guidance.l1_guidance import L1Guidance
from fc_bridge.execution.state_logic import (
    climbing_reached, vtol_is_fw,
    trans_mc_trigger, mc_wp_advance, vtol_is_mc, landing_done,
    override_mode, override_reached, override_fallback_due, wp1_land_ready,
    after_climb_state, after_following_state, after_streaming_state,
    takeoff_request_fields,
    home_amsl_confirmed, home_amsl_sample_fresh,
    is_pilot_takeover, path_origin_ned, translate_path,
    offboard_reacquire_allowed, state_timeout_due,
    horiz_dist_from_origin, range_limit_exceeded, path_max_range,
    slew_setpoint,
)
from fc_bridge.execution.search_pattern import (
    SpiralState, spiral_advance, spiral_point_ne, spiral_complete,
    camera_footprint_m, ring_spacing_from_footprint, max_turn_speed,
    detection_altitude_ok,
)
from fc_bridge.execution.precision_land import (
    latch_candidate, descend_allowed, handoff_due, search_pass_next,
)
from fc_ros.adapters.vision_target_bridge import (
    VisionTargetBridge, subscribe as subscribe_vision,
)
from fc_bridge.comm.vehicle_state import VehicleState
from fc_bridge.utils.rotation import yaw_ned_to_quat_enu
import enum
import math
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, ExtendedState, Altitude
from mavros_msgs.srv import CommandBool, CommandLong, CommandTOL, SetMode
from std_msgs.msg import Bool

_MAVROS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


VTOL_STATE_MC = 3
VTOL_STATE_FW = 4

# TRANSITION_FW 헤딩 정렬: 연속 안정 요구 틱 수
_FW_STABLE_REQ = 20

# FW 위치 setpoint lookahead 거리 (m).
# PX4 FW 오프보드는 속도 setpoint를 무시하고 위치만 추종한다(미달 시 flower-pattern 선회).
# 목표점을 선회반경(SITL 로그상 ~37m)보다 충분히 멀리 둬 FW가 목표점을 orbit하지 않게 한다.
_FW_LOOKAHEAD = 70.0

# WP1 착륙 홀드: 도달+안정 연속 요구 틱 수
_HOLD_STABLE_REQ = 10

# 세그먼트 인덱스 급변 경고 임계 (경로점 개수, 2026-07-27 SITL-7 R2 진단).
# 정상 전진은 순항 20 m/s · 틱 0.1s · 경로점 간격 ~1m 라 2~3점/틱이다.
#
# ⚠️ 이 경고는 "버그다"가 아니라 **"진행 이외의 이유로 인덱스가 움직였다"**
# 는 관측 신호다. 두 가지가 여기 걸린다:
#   ①정상 — 코너 안쪽을 자르면 투영점이 원호를 따라 실제로 수십 점 전진한다
#     (R2 실측 `R2_b5__prefix` 192→221 Δ+29 @ pos=[185.7,15.5] cte=14.4m).
#   ②비정상 — 다른 레그를 잡아 인덱스가 **되감기거나** 창 밖으로 튄다.
# 창 탐색 도입 후에도 ①은 그대로 나올 수 있다(창 폭 안이므로). 막히는 것은
# ② 쪽이다 — 역행은 `_SEG_BACK_M`(5m), 전방은 창 폭(150m)으로 상한이 걸린다.
# 그래서 판정은 "경고가 0건인가"가 아니라 **되감김 건수와 Δ의 부호·크기**로 한다.
_SEG_JUMP_WARN = 20

# OVERRIDE: manual 모드 진입 대기 후 AUTO.LOITER 안전 폴백까지의 틱 수 (10Hz → 1s).
# headless SITL·RC 없음 시 PX4가 MANUAL/POSCTL을 거부하므로 폴백 필수.
_OVERRIDE_FALLBACK_TICKS = 10


class _State(enum.Enum):
    IDLE = "idle"
    ARM_TAKEOFF = "arm_takeoff"
    CLIMBING = "climbing"
    TRANSITION_FW = "transition_fw"
    STREAMING = "streaming"
    ENTRY = "entry"
    FOLLOWING = "following"
    TRANSITION_MC = "transition_mc"
    HOLD = "hold"
    # 비전 정밀착륙 2종 (2026-07-28 F2). `vision_landing:=false`(기본)면 어느
    # 쪽에도 진입하지 않으며 HOLD→LANDING 기존 경로가 그대로 유지된다.
    VISION_SEARCH = "vision_search"      # WP1 중심 나선 탐색
    PRECISION_LAND = "precision_land"    # 비전 유도 정렬·하강
    LANDING = "landing"
    OVERRIDE = "override"
    PILOT_TAKEOVER = "pilot_takeover"
    DONE = "done"


# 거리 상한 감시를 거는 상태 집합 (2026-07-27 SITL-7 R1).
# 기체가 실제로 임무 지령을 받으며 날고 있는 구간만 본다.
#  - IDLE/ARM_TAKEOFF: 아직 이륙지점을 캡처하기 전이라 기준점 자체가 없다
#  - LANDING: 이미 내려오는 중 — 여기서 OVERRIDE를 걸면 착륙을 방해한다
#  - OVERRIDE/PILOT_TAKEOVER/DONE: 이미 손을 뗀 상태 (재발동 방지)
#
# 2026-07-28 F2 추가 결정 (인수인계 문서 D-b):
#  - VISION_SEARCH: **포함한다.** 탐색은 의도적으로 WP1 바깥으로 나가지만 상한
#    300m 대비 탐색 반경(기본 30m)은 자릿수가 다르다. 나선 상태가 폭주하거나
#    중심점이 잘못 잡히면 "점점 커지는 원"은 스스로 멈출 조건이 없으므로
#    (`_step_transition_mc`의 "도망가는 캐럿"과 같은 실패 모드) 거리 상한이
#    유일한 기하학적 제동장치다.
#  - PRECISION_LAND: **제외한다.** LANDING과 같은 논리 — 이미 내려오는 중이라
#    여기서 OVERRIDE를 걸면 착륙을 방해한다.
_RANGE_GUARDED_STATES = frozenset({
    _State.CLIMBING, _State.TRANSITION_FW, _State.STREAMING, _State.ENTRY,
    _State.FOLLOWING, _State.TRANSITION_MC, _State.HOLD,
    _State.VISION_SEARCH,
})


class OffboardNode(Node):
    """
    ROS2 파라미터:
      control_hz         (float, 10.0)  — 제어 루프 주파수 (≥2Hz)
      l1_dist            (float, 20.0)  — L1 lookahead 거리 (m)
      entry_mode         (str,  "pre_takeoff") — "pre_takeoff" | "mid_flight"
      wp0_entry_radius   (float, 5.0)   — WP0 도달 판정 반경 (m)
      wp0_heading_tol    (float, 0.2)   — 헤딩 허용 오차 (rad)
      v_approach         (float, 5.0)   — ENTRY 접근 속도 (m/s)
      cmd_vel_frame_id   (str,  "base_link") — TwistStamped frame_id (MAVROS 버전에 따라 다름)
      waypoints          (float[], [0,0,50, 100,0,50]) — flat 1D, 코드에서 reshape(-1,3)
      planner            (str,  "auto") — "auto"(mc→straight/vtol→eta3) | "eta3" | "diterpin" | "straight"
      v_cruise           (float, 15.0)  — 순항 속도 (m/s)
      a_max_g            (float, 0.3)   — 횡방향 가속도 상한 (g)
      gravity            (float, 9.81)  — 중력 가속도 (m/s²)
      home_amsl_tol         (float, 0.5) — home_position AMSL 수렴 판정 허용오차 (m)
      home_amsl_min_samples (int,   3)   — 이 개수만큼 연속 tol 이내로 수렴해야 신뢰(2026-07-23 대응)
      waypoint_frame     (str,  "takeoff") — waypoints 해석 기준계.
                         "takeoff": 이륙지점 기준 상대좌표(기본, 2026-07-25 사고 대응)
                         "local":   EKF 로컬 프레임 절대좌표(종전 동작)
      mc_end_thresh      (float, 1.0)  — MC WP 도달 판정 반경 (m)
      mc_wp_settle_speed (float, 0.3)  — MC WP 정착 판정 수평속도 임계 (m/s)
      mc_wp_settle_time  (float, 1.0)  — MC WP 정착 유지 시간 (s). 0.0이면 종전 fly-by
      mc_wp_timeout      (float, 20.0) — MC WP당 정착 타임아웃 (s), 초과 시 강제 진행
      climbing_timeout       (float, 120.0) — CLIMBING 체류 상한 (s), 초과 시 안전 폴백
      transition_fw_timeout  (float,  90.0) — TRANSITION_FW 체류 상한 (s), 〃
      transition_mc_timeout  (float,  30.0) — TRANSITION_MC 체류 상한 (s), 〃
      entry_timeout          (float,  60.0) — ENTRY 체류 상한 (s), 〃
      range_limit_m          (float, 300.0) — 이륙지점 기준 수평거리 상한 (m), 〃
                             위 5개는 전부 0 이하이면 비활성. 초과 시 강제
                             진행이 아니라 `_request_override()` 안전 폴백이다.
    """

    def __init__(self):
        super().__init__("offboard_node")

        # ── ROS2 파라미터 ────────────────────────────────────
        self.declare_parameter("control_hz",        10.0)
        self.declare_parameter("l1_dist",           20.0)
        # 기체 타입: "vtol"(기본, MC↔FW 천이 포함) | "mc"(순수 멀티콥터, 천이 생략)
        self.declare_parameter("vehicle_type",      "vtol")
        self.declare_parameter("entry_mode",        "pre_takeoff")
        self.declare_parameter("wp0_entry_radius",  5.0)
        self.declare_parameter("wp0_heading_tol",   0.2)
        self.declare_parameter("v_approach",        5.0)
        self.declare_parameter("cmd_vel_frame_id",  "base_link")
        self.declare_parameter("transition_alt",    50.0)
        self.declare_parameter("d_end_thresh",      10.0)
        # d_end_thresh(10.0m)는 FW/VTOL 대형 항로(300m급)의 "역천이 진입 거리"
        # 기준이라 MC 테스트기체의 짧은 왕복경로(4~6m 스케일)엔 과대 — 경로
        # 전체가 그 반경 안에 들어가 FOLLOWING이 실제로 움직이기도 전에
        # "완료"로 오판한다(2026-07-24 flight03/flight05 실비행 재현). MC는
        # 이 값을 대신 쓴다.
        # MC 웨이포인트(경로점) 도달 판정 반경 (m) — 중간점 통과·경로완료 둘
        # 다에 씀. MC는 L1Guidance(FW 선회반경 회피용 lookahead 알고리즘)가
        # 필요 없어(직선 구간을 순서대로 지나가기만 하면 됨, 2026-07-24 사용자
        # 지적) 이산 웨이포인트 순차추종으로 전환 — d_end_thresh(FW/VTOL
        # 300m급 항로 기준 10.0m)는 MC 4~6m급 경로엔 애초에 무의미해 별도 값.
        self.declare_parameter("mc_end_thresh",     1.0)
        # MC 웨이포인트 "정착" 판정 (2026-07-27) — 반경 안에 들어오는 순간
        # 다음 WP로 넘어가던 fly-by를 폐기하고, 그 자리에 멈춰 선 것을 확인한
        # 뒤 출발한다. MC는 호버가 되므로 이쪽이 자연스럽고 WP 통과 정확도도
        # 올라간다(종전: 반경 경계 1.8~1.9m에서 통과 판정, 실제로 WP 위에
        # 가보지 않음). settle_time=0.0 이면 종전 fly-by 동작.
        self.declare_parameter("mc_wp_settle_speed", 0.3)   # m/s, 수평속도 임계
        self.declare_parameter("mc_wp_settle_time",  1.0)   # s, 정착 유지 시간
        self.declare_parameter("mc_wp_timeout",     20.0)   # s, WP당 정착 타임아웃
        self.declare_parameter("landing_timeout",   60.0)
        self.declare_parameter("v_terminal",        15.2)
        self.declare_parameter("decel_dist",        80.0)
        self.declare_parameter("wp1_land_radius",    3.0)
        self.declare_parameter("wp1_land_speed",     1.5)
        self.declare_parameter("hold_timeout",       30.0)
        self.declare_parameter(
            "waypoints",  [0.0, 0.0, 50.0, 100.0, 0.0, 50.0])
        self.declare_parameter("planner",    "auto")
        self.declare_parameter("v_cruise",   15.0)
        self.declare_parameter("a_max_g",    0.3)
        self.declare_parameter("gravity",    9.81)
        self.declare_parameter("home_amsl_tol",         0.5)
        self.declare_parameter("home_amsl_min_samples", 3)
        self.declare_parameter("home_amsl_max_age",     1.0)
        self.declare_parameter("climb_vz_tol",          0.3)
        self.declare_parameter("waypoint_frame",   "takeoff")
        # ── 상태 타임아웃 4종 + 거리 상한 (2026-07-27 SITL-7 R1) ──────
        # 근거·유도과정은 fc_ros_params.yaml 의 해당 항목 주석에 전부 적혀 있다.
        # 규약은 hold_timeout·mc_wp_timeout 과 동일(초 단위 float, 0 이하 = 비활성,
        # 경계 strict >)이지만 **초과 시 강제 진행이 아니라 안전 폴백**이다 —
        # 이미 실증된 `_request_override()`(manual 시도 → AUTO.LOITER)로 떨어뜨린다.
        self.declare_parameter("climbing_timeout",      120.0)
        self.declare_parameter("transition_fw_timeout",  90.0)
        self.declare_parameter("transition_mc_timeout",  30.0)
        self.declare_parameter("entry_timeout",          60.0)
        self.declare_parameter("range_limit_m",         300.0)

        # ── 비전 정밀착륙 (2026-07-28 F2) ────────────────────────────
        # 🔴 **기본 비활성.** 실기체 FW+OFFBOARD 실적이 0건이라
        # `sitl_vtol_remediation_plan.md` §4-1 4번이 "첫 실비행에 미검증 변수를
        # 둘로 만들지 말라"를 명문화하고 있다. 켜는 것은 별도 결정이다.
        self.declare_parameter("vision_landing",        False)
        # 탐색 고도(AGL, m). 25m 근거: 초록 매트(3.0m)의 coarse 검출 하한
        # (`distress_coarse.yaml` min_area=8000px²)이 주는 상한이 **33.57m**
        # (넓은 화각 가정 = 검출 안전측)이라 8.6m 여유가 있고, 그 고도의
        # 카메라 풋프린트가 33.4×18.8m(좁은 화각 가정 = 커버리지 안전측)라
        # WP 오차 ±9m 이내면 한 발짝도 움직이지 않고 잡힌다.
        # ⚠️ 25m 명목 픽셀면적은 min_area의 1.80배뿐이다(20m는 2.82배) —
        # 현장 검출률이 떨어지면 **이 값을 20m 쪽으로 내리는 것이 첫 카드**다.
        self.declare_parameter("vision_search_alt",      25.0)
        self.declare_parameter("vision_search_radius",   30.0)   # 최대 탐색반경(m)
        # 링 간격(m). 0 이하면 탐색고도 풋프린트에서 자동 산출(겹침 35%).
        self.declare_parameter("vision_search_spacing",   0.0)
        # 나선 선회속도(m/s). 0 이하면 `max_turn_speed`로 자동 산출.
        # 🔴 상한을 두는 이유는 승차감이 아니라 **커버리지**다 — 선회 구심가속도가
        # 기체를 기울이면 나디르 카메라 시선이 `AGL·tan(tilt)`만큼 밀리는데
        # (25m·6° = 2.63m) 이게 링 간격과 같은 자릿수라 탐색 구멍이 생긴다.
        self.declare_parameter("vision_search_speed",     0.0)
        self.declare_parameter("vision_search_dwell",     3.0)   # 제자리 확인(s)
        # 1회 탐색 상한(s). 🔴 **실측 역산값이다.** 기본 설정(25m/30m/겹침35%)에서
        # 나선 완주에 89s가 걸리고 고도정렬(~5s)+dwell(3s)을 더하면 97s다 —
        # 90s로 잡으면 **완주 직전에 잘려** 재탐색으로 넘어간다. 23s 여유를 둔다.
        # 파라미터로 반경·간격·속도를 바꾸면 이 값도 같이 재야 한다.
        self.declare_parameter("vision_search_timeout", 120.0)
        # 재탐색(2회차) — 고도를 낮춰 GSD를 벌고 반경을 줄여 시간을 아낀다.
        self.declare_parameter("vision_retry_alt",       15.0)
        self.declare_parameter("vision_retry_radius",    18.0)
        # 래치: 유효 setpoint가 이만큼 연속 + 좌표 산포가 임계 이내여야 탐색을
        # 멈춘다. 단발 오탐 하나로 엉뚱한 곳으로 날아가는 것을 막는다.
        self.declare_parameter("vision_latch_ticks",        3)
        self.declare_parameter("vision_latch_spread_m",   3.0)
        # 🔴 stale 판정 1.0s — 실측 발행 주파수가 **4.4Hz**(median 0.221s /
        # p95 0.310s)다. 10Hz를 가정해 0.5s로 잡으면 정상 지터와 여유가
        # 1.6배뿐이라 헛경보가 난다(인수인계 문서 §7).
        self.declare_parameter("vision_stale_timeout",    1.0)
        self.declare_parameter("vision_link_timeout",     3.0)
        # 🔴 D-a: vision이 `HOLD`(veto)로 빠졌을 때의 종결 시한. vision은 이 값을
        # **일부러 만들지 않았다** — `FailsafeContract`가 "FC가 자기 제어틱에서
        # 재는 값"으로 못박았다. 없으면 무한 호버링이다(실측 확인).
        self.declare_parameter("vision_veto_timeout",    10.0)
        # 정렬 허용오차(m). 이 안에 들어와야 하강한다 — 수평/수직 분리의 핵심.
        self.declare_parameter("vision_align_tol",        1.0)
        self.declare_parameter("vision_descend_speed",    0.8)   # m/s
        # AUTO.LAND 인계 고도(AGL, m). vision의 `terminal_agl_m`과 같은 숫자여야
        # 한다 — 계약상 "3m부터 AUTO.LAND 인계"다.
        self.declare_parameter("vision_land_handoff_agl", 3.0)
        self.declare_parameter("precision_land_timeout", 60.0)

        control_hz = self.get_parameter("control_hz").value
        self._dt = 1.0 / max(control_hz, 2.0)
        self._vehicle_type = check_choice(
            "vehicle_type", self.get_parameter("vehicle_type").value,
            VEHICLE_TYPES)
        self._is_mc = self._vehicle_type == "mc"
        self._entry_mode = check_choice(
            "entry_mode", self.get_parameter("entry_mode").value, ENTRY_MODES)
        self._wp0_r = self.get_parameter("wp0_entry_radius").value
        self._wp0_htol = self.get_parameter("wp0_heading_tol").value
        self._v_approach = self.get_parameter("v_approach").value
        frame_id = self.get_parameter("cmd_vel_frame_id").value
        self._transition_alt = float(
            self.get_parameter("transition_alt").value)
        self._d_end_thresh = float(self.get_parameter("d_end_thresh").value)
        self._mc_end_thresh = float(
            self.get_parameter("mc_end_thresh").value)
        self._mc_settle_speed = float(
            self.get_parameter("mc_wp_settle_speed").value)
        # 정착 유지 시간(s) → 제어 틱 수. 0.0이면 0틱 = 종전 fly-by 동작.
        self._mc_settle_req = int(round(
            float(self.get_parameter("mc_wp_settle_time").value) / self._dt))
        self._mc_wp_timeout = float(
            self.get_parameter("mc_wp_timeout").value)
        self._mc_wp_idx = 0
        self._mc_settled_ticks = 0
        self._mc_wp_elapsed = 0.0
        self._landing_timeout = float(
            self.get_parameter("landing_timeout").value)
        self._wp1_land_radius = float(
            self.get_parameter("wp1_land_radius").value)
        self._wp1_land_speed = float(
            self.get_parameter("wp1_land_speed").value)
        self._hold_timeout = float(self.get_parameter("hold_timeout").value)
        self._home_amsl_tol = float(
            self.get_parameter("home_amsl_tol").value)
        self._home_amsl_min_samples = int(
            self.get_parameter("home_amsl_min_samples").value)
        self._home_amsl_max_age = float(
            self.get_parameter("home_amsl_max_age").value)
        self._climb_vz_tol = float(
            self.get_parameter("climb_vz_tol").value)
        self._waypoint_frame = str(
            self.get_parameter("waypoint_frame").value).lower()
        # 파라미터 오타를 비행 중이 아니라 노드 기동 시점에 잡는다.
        path_origin_ned(np.zeros(3), self._waypoint_frame)
        self._climbing_timeout = float(
            self.get_parameter("climbing_timeout").value)
        self._fw_trans_timeout = float(
            self.get_parameter("transition_fw_timeout").value)
        self._mc_trans_timeout = float(
            self.get_parameter("transition_mc_timeout").value)
        self._entry_timeout = float(self.get_parameter("entry_timeout").value)
        self._range_limit_m = float(self.get_parameter("range_limit_m").value)

        # ── 비전 정밀착륙 파라미터 ──────────────────────────────
        self._vision_landing = bool(
            self.get_parameter("vision_landing").value)
        self._vs_alt = float(self.get_parameter("vision_search_alt").value)
        self._vs_radius = float(
            self.get_parameter("vision_search_radius").value)
        self._vs_spacing_param = float(
            self.get_parameter("vision_search_spacing").value)
        self._vs_speed_param = float(
            self.get_parameter("vision_search_speed").value)
        self._vs_dwell = float(self.get_parameter("vision_search_dwell").value)
        self._vs_timeout = float(
            self.get_parameter("vision_search_timeout").value)
        self._vs_retry_alt = float(
            self.get_parameter("vision_retry_alt").value)
        self._vs_retry_radius = float(
            self.get_parameter("vision_retry_radius").value)
        self._vs_latch_ticks = int(
            self.get_parameter("vision_latch_ticks").value)
        self._vs_latch_spread = float(
            self.get_parameter("vision_latch_spread_m").value)
        self._vs_stale_timeout = float(
            self.get_parameter("vision_stale_timeout").value)
        self._vs_link_timeout = float(
            self.get_parameter("vision_link_timeout").value)
        self._vs_veto_timeout = float(
            self.get_parameter("vision_veto_timeout").value)
        self._vs_align_tol = float(
            self.get_parameter("vision_align_tol").value)
        self._vs_descend_speed = float(
            self.get_parameter("vision_descend_speed").value)
        self._vs_handoff_agl = float(
            self.get_parameter("vision_land_handoff_agl").value)
        self._pl_timeout = float(
            self.get_parameter("precision_land_timeout").value)

        # ── 경로 계획 ─────────────────────────────────────────
        from fc_bridge.planning.planner_runner import (
            run_planner, resolve_planner_name, planner_eta_s)
        from fc_bridge.planning.terminal_decel import apply_terminal_decel

        raw_wps = np.array(self.get_parameter(
            "waypoints").value, dtype=float).reshape(-1, 3)
        planner_name = resolve_planner_name(
            self.get_parameter("planner").value, self._vehicle_type)
        vehicle_params = {
            "v_cruise": self.get_parameter("v_cruise").value,
            "a_max_g":  self.get_parameter("a_max_g").value,
            "gravity":  self.get_parameter("gravity").value,
        }
        v_terminal = float(self.get_parameter("v_terminal").value)
        decel_dist = float(self.get_parameter("decel_dist").value)

        # ⚠️ run_planner() 는 여기서 **동기로** 돈다 — 반환할 때까지 이 노드는
        # 콜백을 하나도 처리하지 않는다(F-12). 폐곡선 5WP 실측 263.5초.
        # 현장에서 "launch 가 죽었나" 싶어 재기동하면 gz·MAVROS·OFFBOARD
        # 스트림이 두 벌 도는 중복 인스턴스가 된다. 그래서 들어가기 전에
        # 예상 소요시간을, 나온 뒤에 실제 소요시간을 반드시 찍는다.
        # (비동기화는 R6 담당 — 여기서는 로그만 넣는다.)
        _eta_s = planner_eta_s(planner_name, len(raw_wps))
        _closed = (len(raw_wps) >= 3
                   and float(np.linalg.norm(raw_wps[0, :2] - raw_wps[-1, :2])) < 1.0)
        self.get_logger().info(
            f"경로 계획 시작 — planner={planner_name} WP {len(raw_wps)}개"
            f"{' (폐회로)' if _closed else ''} v_cruise={vehicle_params['v_cruise']} "
            f"→ 예상 약 {_eta_s:.0f}s 블로킹. "
            f"이 동안 노드는 응답하지 않는다 — 죽은 것이 아니니 재기동 금지 "
            f"(재기동하면 중복 인스턴스가 된다).")
        _plan_t0 = time.monotonic()
        path = run_planner(planner_name, raw_wps, vehicle_params)
        _plan_dt = time.monotonic() - _plan_t0
        self.get_logger().info(
            f"경로 계획 완료 — 소요 {_plan_dt:.1f}s (예상 {_eta_s:.0f}s), "
            f"경로점 {len(path.points)}개, 전장 {path.total_length:.1f}m")
        path_pts = np.array([pt.pos[:2] for pt in path.points])
        v_profile = np.array([pt.v_ref for pt in path.points])
        s_arc = np.array([pt.s for pt in path.points])
        v_profile = apply_terminal_decel(
            v_profile, s_arc, v_terminal, decel_dist)

        # ── 경로 데이터 ──────────────────────────────────────
        self._pts = np.asarray(path_pts, dtype=float)
        self._v = np.asarray(v_profile, dtype=float)
        # MC 이산 웨이포인트 순차추종용 — run_planner()가 만든 ~1m 간격 촘촘한
        # 보간점(self._pts) 전부가 아니라, 실제 waypoints:= 원본점(wp_index가
        # 있는 점)만 뽑아 순서대로 찾아간다. 보간점 전부를 대상으로 하면
        # 간격(~1m)이 mc_end_thresh(기본 2.0m)보다 좁아, 실제로 그 위치까지
        # 가지도 않았는데 "도달"로 판정되며 인덱스가 실제 이동보다 훨씬 빨리
        # 앞서가는 문제가 있었다(2026-07-24 시뮬레이션으로 확인 — 13점 중 하나도
        # 제대로 안 가보고 12틱 만에 "완료" 처리됨).
        self._mc_wps = np.array(
            [pt.pos[:2] for pt in path.points if pt.wp_index is not None],
            dtype=float)
        # FW 위치 setpoint 순항 고도 (h_up, 양수=위). WP 고도 사용.
        self._cruise_alt = float(raw_wps[-1, 2])
        # 역천이 직진용 최종 진행방향 (마지막 WP 레그, NED 단위벡터).
        # 역천이 시 끝점(근접)을 목표로 하면 FW가 급선회하므로, 이 방향으로
        # 먼 점을 목표로 발행해 직선 감속한다.
        if len(raw_wps) >= 2:
            _ed = raw_wps[-1, :2] - raw_wps[-2, :2]
            _edn = float(np.linalg.norm(_ed))
            self._end_dir = (_ed / _edn if _edn > 1e-9
                             else np.array([1.0, 0.0]))
        else:
            self._end_dir = np.array([1.0, 0.0])

        # ── 내부 상태 ────────────────────────────────────────
        self._vehicle_state = VehicleState()
        self._vs_lock = threading.Lock()
        self._l1_dist = float(self.get_parameter("l1_dist").value)
        self._guidance = L1Guidance(self._l1_dist, self._pts, self._v)
        # 경로 원점 (NED [N, E, h_up]). 이륙 순간 `_apply_path_origin()`이 채운다.
        # 그 전까지는 0 — 즉 종전 동작(로컬 절대좌표)과 동일하다.
        self._path_origin = np.zeros(3)
        self._path_origin_applied = False
        self._sm = _State.ARM_TAKEOFF
        self._offboard_requested = False
        self._stream_ticks = 0
        self._follow_ticks = 0
        self._current_mode = ""
        self._offboard_engaged_once = False
        # 위치 토픽을 한 번이라도 받았는지 (_cb_pose 참조)
        self._pose_received = False
        # OFFBOARD 재요청 최소 간격 (s). /mavros/state 는 PX4 HEARTBEAT 주기라
        # 1Hz인데 제어 루프는 10Hz — 같은 1초 묵은 mode 값으로 10번 재요청해도
        # 의미가 없고, 2026-07-25 사고 때 0.9초에 10회가 나간 원인이 이것이다.
        self._offboard_req_min_interval = 1.0
        self._last_offboard_req_t = None
        # MC STREAMING/FOLLOWING + HOLD 위치 setpoint 슬루레이트 제한용
        # (2026-07-20 사고 대응, HOLD 는 2026-07-27 F-6 으로 추가).
        self._mc_pos_ramp = None
        # 직전 틱의 L1 세그먼트 인덱스 (FW FOLLOWING 진단, 2026-07-27 R2).
        self._last_seg = None
        # ARM_TAKEOFF 시퀀스 플래그
        self._arm_sent = False
        self._takeoff_sent = False
        # 이륙 지점 지면 AMSL (/mavros/altitude.amsl, 2026-07-24부터 — 이전엔
        # /mavros/home_position/home.geo.altitude였으나 ellipsoid 혼동 버그로 교체됨,
        # 아래 _cb_home 참조). CommandTOL 목표고도(절대) 기준.
        # 수신 즉시 단발 신뢰하지 않고 최근 N개가 tol 이내로 수렴해야 확정한다
        # (2026-07-23 실비행 사고: 막 재시작된 MAVROS가 첫 수신값을 그대로 썼다가
        # 26.7m 스테일 오차로 3m 상승명령이 29.7m 상승으로 실행됨 — state_logic.py
        # home_amsl_confirmed() 참조).
        self._home_amsl_samples: list[float] = []
        self._home_amsl = None
        # 이륙 순간 로컬 지면 높이 (h_up). CLIMBING AGL 판정의 지면 기준(2026-07-07).
        self._takeoff_ground_h = 0.0
        # 이륙 순간 기체 수평위치 (NED [N, E, h_up]). 거리 상한 감시의 기준점.
        # `_path_origin` 과 값이 같아 보이지만 **같은 것이 아니다** —
        # `waypoint_frame:="local"` 이면 `_path_origin` 은 0(EKF 로컬 원점)이 되고,
        # EKF 로컬 원점은 이륙지점과 무관하다(2026-07-25 flight01 실측: 수평
        # 10.94m·수직 10.55m 어긋남). 거리 상한은 프레임 설정과 무관하게 항상
        # **이륙지점** 기준이어야 하므로 따로 캡처한다. None이면 아직 미이륙 →
        # 감시 비활성.
        self._takeoff_pos_ned = None
        # TRANSITION_FW 시퀀스 플래그
        self._fw_transition_sent = False
        self._fw_prime_ticks = 0     # OFFBOARD 프라이밍 틱 수
        self._fw_offboard_requested = False  # 천이 전 MC OFFBOARD 요청 여부
        self._fw_heading_aligned = False  # WP 방향 헤딩 정렬 완료 여부
        self._fw_stable_ticks = 0        # 헤딩 오차가 허용 범위 내 연속 틱 수
        # TRANSITION_MC / LANDING 시퀀스 플래그
        self._mc_transition_sent = False
        self._landing_sent = False
        self._landing_elapsed = 0.0
        self._landing_timeout_warned = False
        # HOLD (WP1 복귀·착륙) 플래그
        self._hold_ticks = 0
        self._hold_stable_ticks = 0
        self._hold_elapsed = 0.0
        # ── 비전 정밀착륙 상태 (2026-07-28 F2) ────────────────────
        # 탐색은 "고도정렬 → 제자리확인 → 나선"의 3단이고, 실패 시 저고도로
        # 한 번 더 돈다. `_vs_pass`가 0이면 1회차(25m/30m), 1이면 재탐색(15m/18m).
        self._vs_pass = 0
        self._vs_phase = "align"       # "align" | "dwell" | "spiral"
        self._vs_elapsed = 0.0
        self._vs_dwell_elapsed = 0.0
        self._vs_spiral = SpiralState(0.0, 0.0, 0.0)
        self._vs_r_start = 0.0         # 나선 시작 반경 (= 링 간격의 절반)
        self._vs_ramp = None           # 탐색 setpoint 슬루 램프 (NED [N,E,h_up])
        self._vs_center = None         # 나선 중심 (NED [N, E])
        self._vs_spacing = 0.0         # 실효 링 간격 (자동산출 결과 보관)
        self._vs_speed = 0.0           # 실효 선회속도
        # 래치 — 유효 setpoint를 연속으로 모아 산포를 본다.
        self._vs_latch_buf: list = []
        self._vs_latched = None        # 래치된 착륙점 (NED [N, E, h_up])
        # PRECISION_LAND 플래그
        self._pl_ramp = None
        self._pl_elapsed = 0.0
        self._pl_veto_elapsed = 0.0
        self._pl_lost_elapsed = 0.0
        self._pl_target_ne = None      # 최신 수평 목표 (NED [N, E])
        self._pl_alt = None            # FC가 스케줄하는 목표 고도 (h_up)
        # OVERRIDE (긴급 수동 전환) 플래그
        self._override_target = "MANUAL"
        self._override_ticks = 0
        self._override_fallback_sent = False
        # 상태별 체류시간 누적 (s). 타임아웃 4종용 — 각 상태는 한 번만 진입하므로
        # 진입 시 리셋 없이 step 안에서 누적한다(`_hold_elapsed` 와 같은 방식).
        self._climbing_elapsed = 0.0
        self._fw_trans_elapsed = 0.0
        self._mc_trans_elapsed = 0.0
        self._entry_elapsed = 0.0

        # ── 비전 토픽 구독 (vision_landing 일 때만) ───────────────
        # 🔴 구독조차 만들지 않는다 — 파라미터가 꺼져 있으면 이 노드는 vision의
        # 존재를 아예 모른다. shim이 안 떠 있어도 부작용이 0이어야 한다.
        self._vision = None
        if self._vision_landing:
            self._vision = VisionTargetBridge(
                stale_timeout_s=self._vs_stale_timeout,
                link_timeout_s=self._vs_link_timeout)
            subscribe_vision(self, self._vision)
            self.get_logger().info(
                f"비전 정밀착륙 활성 — 탐색고도 {self._vs_alt:.0f}m "
                f"반경 {self._vs_radius:.0f}m / 재탐색 {self._vs_retry_alt:.0f}m "
                f"반경 {self._vs_retry_radius:.0f}m")
            # 탐색고도가 검출 상한을 넘으면 나선을 아무리 예쁘게 그려도
            # 검출이 0건이다. 비행 중이 아니라 기동 시점에 알린다.
            if not detection_altitude_ok(self._vs_alt):
                self.get_logger().error(
                    f"⚠️ 탐색고도 {self._vs_alt:.1f}m 가 초록매트 coarse 검출 "
                    f"상한을 넘는다 — 그 고도에서는 타겟이 min_area 미달이라 "
                    f"탐색이 구조적으로 실패한다. vision_search_alt 를 낮춰라")

        # ── MAVROS 토픽 구독 ─────────────────────────────────
        self.create_subscription(
            PoseStamped,
            "/mavros/local_position/pose",
            self._cb_pose, _MAVROS_QOS)
        self.create_subscription(
            TwistStamped,
            "/mavros/local_position/velocity_local",
            self._cb_twist, _MAVROS_QOS)
        self.create_subscription(
            State,
            "/mavros/state",
            self._cb_state, _MAVROS_QOS)
        self.create_subscription(
            ExtendedState,
            "/mavros/extended_state",
            self._cb_extended, _MAVROS_QOS)
        self.create_subscription(
            Altitude,
            "/mavros/altitude",
            self._cb_home, _MAVROS_QOS)
        self.create_subscription(
            Bool,
            "/fc_ros/override",
            self._cb_override, 10)

        # ── 발행 / 서비스 ────────────────────────────────────
        pub = self.create_publisher(
            TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10)
        self._setpoint = SetpointPublisher(pub, frame_id=frame_id)
        # FW 구간 위치 setpoint (PoseStamped, ENU). MC 속도 setpoint와 별개 채널.
        self._pos_pub = self.create_publisher(
            PoseStamped, "/mavros/setpoint_position/local", _MAVROS_QOS)
        self._set_mode_cli = self.create_client(
            SetMode,      "/mavros/set_mode")
        self._arm_cli = self.create_client(CommandBool,  "/mavros/cmd/arming")
        self._cmd_cli = self.create_client(CommandLong,  "/mavros/cmd/command")
        self._takeoff_cli = self.create_client(CommandTOL, "/mavros/cmd/takeoff")

        # ── 제어 타이머 ──────────────────────────────────────
        self.create_timer(self._dt, self._control_callback)

    # ── MAVROS 구독 콜백 ─────────────────────────────────────

    def _cb_pose(self, msg: PoseStamped) -> None:
        with self._vs_lock:
            update_from_pose(self._vehicle_state, msg)
            # 위치를 한 번이라도 실제로 받았는지 (2026-07-25 감사).
            # /mavros/state 는 transient_local(래치)이라 노드 기동 즉시 armed=True 를
            # 알 수 있는 반면 /mavros/local_position/pose 는 volatile(래치 없음)이라
            # 다음 실시간 샘플까지 아무것도 안 온다. 그 창에서 이륙 시퀀스가 진행되면
            # pos_ned 가 기본값 zeros(3) 인 채로 캡처돼 _takeoff_ground_h=0 이 되고
            # _apply_path_origin 이 원점 0 을 적용(=무보정)하며, np.any(origin)==False
            # 라 로그조차 남지 않는다 → 2026-07-25 flight01 사고 동작으로 조용히 회귀.
            self._pose_received = True

    def _cb_twist(self, msg: TwistStamped) -> None:
        with self._vs_lock:
            update_from_twist(self._vehicle_state, msg)

    def _cb_state(self, msg: State) -> None:
        with self._vs_lock:
            update_from_mavros_state(self._vehicle_state, msg)
            self._current_mode = msg.mode
            if msg.mode == "OFFBOARD":
                # 한 번이라도 OFFBOARD를 잡은 뒤부터는 수동모드 = 조종사 인계로 본다.
                # (그 전에는 POSCTL이 그냥 시동 전 대기 모드라 인계와 구분되지 않는다)
                self._offboard_engaged_once = True

    def _cb_extended(self, msg: ExtendedState) -> None:
        with self._vs_lock:
            update_from_extended_state(self._vehicle_state, msg)

    def _cb_home(self, msg: Altitude) -> None:
        # amsl = 이륙 지점 지면 AMSL. CommandTOL 목표고도(절대)의 기준값.
        # 단발 스냅샷 대신 최근 샘플이 tol 이내로 수렴할 때만 확정한다(2026-07-23 대응).
        #
        # (2026-07-24 SITL H-2 체크리스트 재현 대응, geoid/ellipsoid 혼동 확정):
        # 원래 /mavros/home_position/home 의 geo.altitude를 썼으나, 이는 MAVROS가
        # ROS REP-103 관례(NavSatFix.altitude=WGS84 타원체고)를 맞추려 EGM96 보정을
        # 적용한 값 — 반면 CommandTOL(MAV_CMD_NAV_TAKEOFF param7)이 기대하는 값은
        # AMSL이라, 그 보정치(한국 기준 약 24.8m, geoid separation)만큼 목표고도가
        # 부풀려져 3~4m 상승명령이 실제로 ~28m까지 과상승했다(PX4_HOME_LAT/LON/ALT를
        # 실측값으로 지정한 통제 SITL 재현으로 확정, docs/sitl_verification_log.md
        # "작업 H-2" 참조). /mavros/altitude.amsl은 이 EGM96 보정을 거치지 않아
        # CommandTOL과 프레임이 일치한다.
        #
        # ⚠️ 근거 정정(2026-07-25 감사): 종전 주석은 "GLOBAL_POSITION_INT.alt를
        # relay"라고 적었으나 사실이 아니다. MAVROS altitude 플러그인이 relay하는
        # 것은 MAVLink **ALTITUDE(#141)** 의 altitude_amsl이다. 이 구분이 중요한
        # 이유는 PX4 ALTITUDE.hpp가 다음 폴백을 갖기 때문이다:
        #   z_global 유효 → -local_pos.z + ref_alt (진짜 AMSL)
        #   z_global 무효 → altitude_monotonic (= **기압고도**) 로 조용히 대체
        #   air_data도 무효 → NaN (아래 isfinite 가드로 배제)
        # 기압고도는 표준대기 기준이라 당일 기압에 따라 실제 AMSL과 수십 m 차이가
        # 날 수 있고, **몇 초간 매우 안정적이라 아래 수렴판정을 그대로 통과한다** —
        # 선례(ellipsoid고)와 동일한 실패 패턴이다. ARM/AUTO.TAKEOFF가 GPS 락을
        # 요구해 실제 발생 확률은 낮지만, 이륙 전 점검으로
        # `ros2 topic echo /mavros/altitude` 의 amsl과 monotonic이 **같으면**
        # 폴백 상태이므로 이륙을 보류할 것.
        #
        # (2026-07-24 이전 SITL 재현 대응, 여전히 유효): 오래된(래치/캐시된) 표본은
        # 수렴 표본에서 아예 제외한다 — 같은 세션 안에서 이전 PX4 인스턴스의 값이
        # 그대로 섞여 들어와 우연히 min_samples개가 서로 tol 이내로 일치해버리면
        # home_amsl_confirmed()가 그 stale 값을 그대로 확정하는 사고를 막기 위함
        # (실측: 이번 비행 지면 0.25m AMSL인데 이전 비행 값 47.5m대가 새어들어가
        # PX4에 이륙목표 50.47m AMSL 요청 — logs/2026-07-24_sitl_streaming_overshoot/).
        age_s = (self.get_clock().now()
                 - Time.from_msg(msg.header.stamp)).nanoseconds / 1e9
        if not home_amsl_sample_fresh(age_s, self._home_amsl_max_age):
            self.get_logger().warn(
                f"altitude 메시지 지연 age={age_s:.1f}s > "
                f"{self._home_amsl_max_age:.1f}s — 래치/캐시 의심, 수렴 표본에서 제외",
                throttle_duration_sec=2.0)
            return
        # NaN/inf 표본은 아예 수집하지 않는다 (2026-07-25 감사).
        # PX4 ALTITUDE(#141)의 altitude_amsl은 EKF 글로벌 기준(z_global)과 air_data가
        # 둘 다 무효이면 NaN이다. NaN은 모든 비교가 False라 home_amsl_confirmed()의
        # 수렴검사를 그냥 통과해 CommandTOL.altitude=NaN이 나간다.
        if not math.isfinite(msg.amsl):
            self.get_logger().warn(
                "altitude.amsl 이 유한하지 않음(NaN/inf) — 수렴 표본에서 제외",
                throttle_duration_sec=2.0)
            return
        self._home_amsl_samples.append(float(msg.amsl))
        del self._home_amsl_samples[:-20]  # 무한 성장 방지, 최근 20개만 유지
        self._home_amsl = home_amsl_confirmed(
            self._home_amsl_samples,
            tol=self._home_amsl_tol,
            min_samples=self._home_amsl_min_samples)

    def _get_state(self) -> VehicleState:
        with self._vs_lock:
            return self._vehicle_state.copy()

    # ── 제어 루프 (타이머 콜백) ──────────────────────────────

    def _control_callback(self) -> None:
        state = self._get_state()

        # 조종사 인계 감시 — 어떤 상태에 있든 최우선.
        # OFFBOARD를 한 번이라도 잡은 뒤 수동모드(POSCTL/MANUAL/…)로 빠졌다면
        # 사람이 기체를 가져간 것이다. 절대 되찾아오지 않는다.
        # (2026-07-25 flight01: FOLLOWING이 조종사의 POSCTL 인계를 0.9초 만에
        #  10회 재요청으로 뒤집어 조종사가 KILL 스위치를 써야 했다.)
        if (self._offboard_engaged_once
                and self._sm not in (_State.PILOT_TAKEOVER, _State.DONE)
                and is_pilot_takeover(self._current_mode)):
            self._enter_pilot_takeover()
            return

        # 거리 상한 감시 — 조종사 인계 다음 우선순위. 상태머신이 어디서 막혔든,
        # 어떤 이유로 캐럿이 도망갔든 "이륙지점에서 이만큼 멀어지면 손을 뗀다"는
        # 마지막 기하학적 제동장치다. (2026-07-27 SITL-7 R1)
        if self._range_guard_breached(state):
            return

        if self._sm == _State.ARM_TAKEOFF:
            self._step_arm_takeoff(state)

        elif self._sm == _State.CLIMBING:
            self._step_climbing(state)

        elif self._sm == _State.TRANSITION_FW:
            self._step_transition_fw(state)

        elif self._sm == _State.STREAMING:
            if self._is_mc:
                # MC도 최종 VTOL 기체와 동일하게 위치기반 세트포인트를 유지한다
                # (이 테스트기체의 목적이 최종기체 제어로직 검증이므로 속도
                # setpoint로 전환할 이유가 없음). OFFBOARD 확정 전까지는 매 틱
                # 현재위치를 그대로 스트리밍해, PX4가 확정 순간 이어받는
                # setpoint가 항상 실제위치와 일치하게 한다 — FW의 lookahead
                # 목표점(경로 끝점 WP1로 클램프됨, 아래)을 그대로 쓰면 실제
                # 위치와 무관한 먼 절대좌표가 되어(2026-07-20 실비행: 클라이밍
                # 오버슈트 중 이 값이 그대로 발행돼 OFFBOARD 확정 순간 PX4가
                # 급격한 자세보정을 시도 → 제어상실, 조종사 수동 회수) 위험함.
                self._mc_pos_ramp = np.array(state.pos_ned, dtype=float)
                self._publish_pos_setpoint(self._mc_pos_ramp, state.yaw)
            else:
                # FW 위치 setpoint로 lookahead 추종 (속도 setpoint는 FW가 무시 → flower-pattern).
                tgt = self._guidance.target_point_ned(state.pos_ned, _FW_LOOKAHEAD)
                self._publish_pos_setpoint(
                    np.array([tgt[0], tgt[1], self._cruise_alt]), state.yaw)

            if self._current_mode == "OFFBOARD":
                # 실제로 가는 상태를 그대로 찍는다 (F-14). 종전엔 entry_mode 와
                # 무관하게 항상 "→ FOLLOWING" 이라 ENTRY 로 갔을 때도 FOLLOWING
                # 이라고 찍혀, 하니스 상태창 추정과 결함 해석이 함께 어긋났다.
                nxt = _State(after_streaming_state(self._entry_mode))
                self.get_logger().info(
                    f"OFFBOARD 확인 → {nxt.name}"
                    + (f" (entry_mode={self._entry_mode})"
                       if nxt is _State.ENTRY else ""))
                self._sm = nxt
                self._follow_ticks = 0
                self._mc_wp_idx = 0
                return

            # 폴백: OFFBOARD 미활성 상태에서 20 tick 후 재요청
            self._stream_ticks += 1
            if self._stream_ticks == 20 and not self._offboard_requested:
                self._request_offboard()
                self._offboard_requested = True
                self.get_logger().info("OFFBOARD 전환 요청 (폴백)")

        elif self._sm == _State.ENTRY:
            self._entry_elapsed += self._dt
            if self._step_entry(state):
                self.get_logger().info("ENTRY 완료 -> FOLLOWING")
                self._sm = _State.FOLLOWING
            else:
                # C10 실측: 여기서 432.67초 동안 탈출하지 못했고 그동안 기체가
                # 이륙지점에서 5.85km 이탈했다. 장애주입 없이 파라미터 하나로.
                self._timeout_fallback(
                    "ENTRY", self._entry_elapsed, self._entry_timeout)

        elif self._sm == _State.FOLLOWING:
            if self._step_following(state):
                nxt = _State(after_following_state(self._is_mc))
                self.get_logger().info(
                    f"경로 추종 완료 -> {nxt.value}"
                    + (" (MC, 역천이 생략)" if self._is_mc else ""))
                self._sm = nxt

        elif self._sm == _State.TRANSITION_MC:
            self._step_transition_mc(state)

        elif self._sm == _State.HOLD:
            self._step_hold(state)

        elif self._sm == _State.VISION_SEARCH:
            self._step_vision_search(state)

        elif self._sm == _State.PRECISION_LAND:
            self._step_precision_land(state)

        elif self._sm == _State.LANDING:
            self._step_landing(state)

        elif self._sm == _State.OVERRIDE:
            self._step_override(state)

        elif self._sm == _State.PILOT_TAKEOVER:
            pass   # 세트포인트 발행 중단 — 아래 _enter_pilot_takeover() 참조

        elif self._sm == _State.DONE:
            # setpoint를 계속 발행하면 PX4가 OFFBOARD를 언제든 다시 받아줄 수 있는
            # 상태로 남는다. 임무가 끝난 뒤엔 스트림을 끊는 게 맞다
            # (`_request_override()` 주석의 "velocity-0 발행이 OFFBOARD를 살려둬
            #  FW가 직진 폭주" 사례와 같은 위험).
            pass

    # ── ARM_TAKEOFF ──────────────────────────────────────────

    def _step_arm_takeoff(self, state: VehicleState) -> None:
        """ARM 요청 → armed 확인 → CommandTOL 이륙 요청 → CLIMBING 전환.

        이륙 목표고도는 지면 AMSL(home_amsl)+transition_alt 절대고도로 보낸다
        (작업 H 수정, 2026-07-07): CommandTOL.altitude 는 AMSL 절대고도라
        transition_alt 를 그대로 실으면 지면보다 낮아 PX4가 이륙을 취소한다.

        home_amsl은 첫 수신값을 바로 쓰지 않고 최근 home_amsl_min_samples개가
        home_amsl_tol 이내로 수렴해야 확정된다(state_logic.home_amsl_confirmed).
        2026-07-23 실비행에서 막 재시작된 MAVROS가 PX4 부팅 초기(GPS 수직정확도
        미수렴 시점)에 래치된 오래된 값을 단발로 받아 26.7m 오차로 이륙목표가
        3m 대신 29.7m로 계산·실행된 사고 대응.
        """
        if not self._arm_sent:
            if not self._arm_cli.service_is_ready():
                self.get_logger().warn("/mavros/cmd/arming 서비스 없음")
                return
            req = CommandBool.Request()
            req.value = True
            self._arm_cli.call_async(req)
            self._arm_sent = True
            self.get_logger().info("ARM 요청")
            return

        if not state.armed:
            return  # ARM 완료 대기

        if self._home_amsl is None:
            if not self._home_amsl_samples:
                self.get_logger().warn(
                    "home_position 미수신 — 이륙 목표 AMSL 계산 불가, 대기",
                    throttle_duration_sec=2.0)
            else:
                self.get_logger().warn(
                    "home_position AMSL 미수렴"
                    f"(최근 {len(self._home_amsl_samples[-self._home_amsl_min_samples:])}개: "
                    f"{['%.1f' % v for v in self._home_amsl_samples[-self._home_amsl_min_samples:]]}, "
                    f"tol={self._home_amsl_tol:.1f}) — 이륙 보류, 수렴 대기",
                    throttle_duration_sec=2.0)
            return

        if not self._takeoff_sent:
            if not self._takeoff_cli.service_is_ready():
                self.get_logger().warn("/mavros/cmd/takeoff 서비스 없음")
                return
            # 위치 미수신 상태로 이륙하면 _takeoff_ground_h 와 경로 원점이 둘 다
            # zeros 로 캡처돼 보정이 조용히 무효화된다 (_cb_pose 주석 참조).
            if not self._pose_received:
                self.get_logger().warn(
                    "local_position/pose 미수신 — 이륙 지점/지면 기준을 잡을 수 없어 "
                    "이륙 보류, 수신 대기",
                    throttle_duration_sec=2.0)
                return
            # 이륙 순간 로컬 지면 높이 캡처 (로컬 원점≠지면 보정, CLIMBING AGL 판정용).
            self._takeoff_ground_h = float(state.pos_ned[2])
            # 거리 상한 감시 기준점 — waypoint_frame 설정과 무관하게 이륙지점.
            self._takeoff_pos_ned = np.array(state.pos_ned, dtype=float)
            # 같은 보정을 경로(수평 waypoints + 순항고도)에도 적용한다.
            # 2026-07-25 이전엔 CLIMBING 판정에만 적용돼 있어, 이륙은 정확했지만
            # OFFBOARD 진입 순간 변환 없이 EKF 로컬 프레임으로 갈아탔다.
            self._apply_path_origin(state.pos_ned)
            self._check_path_within_range()
            req = CommandTOL.Request()
            fields = takeoff_request_fields(self._transition_alt, self._home_amsl)
            for field, value in fields.items():
                setattr(req, field, value)
            self._takeoff_cli.call_async(req)
            self._takeoff_sent = True
            self.get_logger().info(
                f"CommandTOL 이륙 요청 alt={fields['altitude']:.1f}m AMSL "
                f"(지면 {self._home_amsl:.1f}+{self._transition_alt:.1f}) -> CLIMBING")
            self._sm = _State.CLIMBING

    def _apply_path_origin(self, takeoff_pos_ned) -> None:
        """`waypoints:=` 를 이륙지점 기준으로 평행이동한다 (2026-07-25 사고 대응).

        경로 데이터(`_pts`/`_mc_wps`/`_cruise_alt`)는 `__init__`에서 waypoints 값
        그대로 만들어져 있다. 이륙 순간 기체 위치를 알게 된 시점에 딱 한 번
        전체를 평행이동해, 이후 모든 하위 코드(L1Guidance·거리판정·setpoint
        발행)는 종전과 똑같이 EKF 로컬 절대좌표로 동작하게 한다 —
        비교/발행 지점마다 오프셋을 더하는 방식보다 누락 위험이 없다.

        `waypoint_frame:="local"`이면 원점이 0이라 아무것도 바뀌지 않는다.

        배경: EKF 로컬 원점은 이륙지점이 아니다. 2026-07-25 flight01에서 원점이
        이륙지점으로부터 수평 10.94m·수직 10.55m 어긋나 `[0,0,3]`이 실제로는
        "10.9m 옆 + 13.55m AGL"을 지령했다. `state_logic.path_origin_ned()` 참조.
        """
        if self._path_origin_applied:
            return
        origin = path_origin_ned(takeoff_pos_ned, self._waypoint_frame)
        self._path_origin = origin
        self._path_origin_applied = True
        if not np.any(origin):
            return   # "local" 프레임이거나 원점이 정확히 0 — 이동 불필요

        self._pts, self._mc_wps, self._cruise_alt = translate_path(
            self._pts, self._mc_wps, self._cruise_alt, origin)
        # 경로점이 바뀌었으므로 L1Guidance를 새 경로로 다시 만든다
        # (내부에 경로 배열을 들고 있어 재생성하지 않으면 옛 좌표를 계속 쓴다).
        self._guidance = L1Guidance(self._l1_dist, self._pts, self._v)
        wp0 = (f"[{self._mc_wps[0][0]:.2f},{self._mc_wps[0][1]:.2f}]"
               if len(self._mc_wps) else "(없음)")
        self.get_logger().info(
            f"경로 원점 = 이륙지점 [N={origin[0]:.2f}, E={origin[1]:.2f}, "
            f"h_up={origin[2]:.2f}] 적용 "
            f"(frame={self._waypoint_frame}) → "
            f"WP0={wp0} 순항고도 h_up={self._cruise_alt:.2f}")

    def _check_path_within_range(self) -> None:
        """계획된 경로가 애초에 거리 상한 밖인지 **이륙 순간에** 알린다.

        경로가 상한을 넘게 설계돼 있으면 그 경로는 정상 수행 자체가 불가능하다
        (종점 부근에서 반드시 안전 폴백이 걸린다). 그걸 비행 중에 발견하는
        것과 이륙 직전 로그에서 보는 것은 완전히 다르다 —
        `path_origin_ned()` 오타 검사를 노드 기동 시점으로 당겨놓은 것과 같은 이유다.

        ⚠️ 판정하지 않고 경고만 한다. 이륙을 막지 않는 이유: 상한은 현장에서
        launch 인자로 조정하는 값이고, 여기서 이륙을 거부하면 "왜 안 뜨지"가
        되어 현장에서 더 위험한 우회(감시 자체를 끄기)를 유도한다.
        """
        if self._range_limit_m <= 0:
            self.get_logger().warn(
                "거리 상한 감시 비활성 (range_limit_m <= 0) — "
                "이륙지점 기준 이탈 제동장치가 없다")
            return
        far = path_max_range(self._pts, self._takeoff_pos_ned)
        if far > self._range_limit_m:
            self.get_logger().warn(
                f"⚠️ 계획 경로가 거리 상한 밖이다 — 경로 최원점 {far:.0f}m > "
                f"상한 {self._range_limit_m:.0f}m. 그대로 날면 종점 부근에서 "
                f"안전 폴백(OVERRIDE)이 걸린다. "
                f"range_limit_m 을 키우거나 경로를 줄일 것")
        else:
            self.get_logger().info(
                f"거리 상한 감시 활성 {self._range_limit_m:.0f}m "
                f"(이륙지점 기준 수평, 경로 최원점 {far:.0f}m)")

    # ── CLIMBING ─────────────────────────────────────────────

    def _step_climbing(self, state: VehicleState) -> None:
        """운용 고도 도달 확인 → 다음 상태 전환.

        VTOL: TRANSITION_FW(MC→FW 천이). MC: STREAMING(천이 생략, OFFBOARD 진입).
        """
        if climbing_reached(state.pos_ned[2], self._transition_alt,
                            self._takeoff_ground_h,
                            vz_down=state.vel_ned[2],
                            vz_tol=self._climb_vz_tol):
            nxt = _State(after_climb_state(self._is_mc))
            self.get_logger().info(
                f"운용 고도 {self._transition_alt:.1f}m 도달 → {nxt.value}"
                + (" (MC, 천이 생략)" if self._is_mc else ""))
            self._sm = nxt
            return

        # 타임아웃 — PX4가 우리가 지령한 것과 다른 고도로 이륙했거나(래치된
        # home_amsl / MIS_TAKEOFF_ALT 폴백), 판정 허용대(±alt_tol, |vz|<=vz_tol)에
        # 영영 들어오지 못하면 여기서 영구 대기한다.
        self._climbing_elapsed += self._dt
        self._timeout_fallback(
            "CLIMBING", self._climbing_elapsed, self._climbing_timeout)

    # ── TRANSITION_FW ─────────────────────────────────────────

    def _step_transition_fw(self, state: VehicleState) -> None:
        """
        헤딩 정렬 후 MC→FW 직선 천이.

        Phase 1: MC+HOLD에서 hover 세트포인트 20 tick → OFFBOARD 요청
        Phase 2: MC OFFBOARD hover + yaw rate P제어로 WP 방향 헤딩 정렬
                 (twist.angular.z 없으면 PX4 MC는 yaw를 바꾸지 않음)
        Phase 3: 헤딩 정렬 완료 → WP 방향 전진 + MC→FW 천이 명령
        Phase 4: vtol_state==FW 대기 → STREAMING
        """
        # Phase 4: FW 전환 완료 확인
        if vtol_is_fw(state.vtol_state):
            self.get_logger().info("FW 전환 완료 -> STREAMING")
            self._sm = _State.STREAMING
            return

        # 타임아웃 — 이 상태의 탈출은 `vtol_state == FW` 단일 신호 하나뿐인데
        # 그 토픽은 BEST_EFFORT·초기값 0·신선도 검사 없음이다. 천이 명령이
        # 거부되면(1회 발행·결과 미확인) MC 상태로 종점까지 가서 영구 호버한다
        # (F-3). 헤딩 정렬(Phase 2)이 바람에 20틱을 못 채워도 여기 갇힌다(F-7).
        self._fw_trans_elapsed += self._dt
        if self._timeout_fallback("TRANSITION_FW", self._fw_trans_elapsed,
                                  self._fw_trans_timeout):
            return

        # 경로 시작 방향 (WP0→WP1 단위벡터, NED)
        if len(self._pts) > 1:
            seg = self._pts[1] - self._pts[0]
            seg_norm = float(np.linalg.norm(seg))
            seg = seg / (seg_norm + 1e-9)
        else:
            seg = np.array([1.0, 0.0])
        # chi_wp: NED 기준 (arctan2(E,N)), 0=North, 양수=East(CW)
        chi_wp = float(np.arctan2(seg[1], seg[0]))

        # Phase "ACTIVE TRANSITION": 천이 명령 발행 완료 → vtol_state==FW 대기 구간.
        # FW는 속도 setpoint를 무시하므로(꽃잎 선회) WP1을 위치 setpoint로 발행한다.
        # OFFBOARD 이탈 여부와 무관하게 위치 명령을 끊지 않는다.
        if self._fw_heading_aligned and self._fw_transition_sent:
            self._publish_pos_setpoint(
                np.array([self._pts[-1][0], self._pts[-1][1], self._cruise_alt]), chi_wp)
            if self._current_mode != "OFFBOARD":
                self._request_offboard()
                # throttle: 실제 명령은 `_request_offboard()`가 1Hz로 조이는데
                # 로그만 10Hz로 나가면 2026-07-25 조종사 인계 사고(0.9초에 10회
                # 재요청)와 육안 구별이 안 된다 — C3 실측 10건/0.90s vs
                # `vehicle_command 176 p2=6` 1건. (F-16)
                self.get_logger().warn(
                    f"천이 중 OFFBOARD 이탈 → 재요청 (mode={self._current_mode})",
                    throttle_duration_sec=self._offboard_req_min_interval)
            return

        # Phase 1: hover 세트포인트로 OFFBOARD 프라이밍 (HOLD 중에는 무시됨)
        if not self._fw_offboard_requested:
            self._setpoint.publish(np.zeros(3))  # hover — 방향 무관
            self._fw_prime_ticks += 1
            if self._fw_prime_ticks >= 20:
                # 아직 OFFBOARD를 한 번도 안 잡은 구간이라 상단 감시가 안 걸린다.
                # 여기서도 조종사가 이미 기체를 쥐고 있으면 뺏지 않는다.
                if is_pilot_takeover(self._current_mode):
                    self._enter_pilot_takeover()
                    return
                if not self._set_mode_cli.service_is_ready():
                    self.get_logger().warn("/mavros/set_mode 서비스 없음")
                    return
                req = SetMode.Request()
                req.custom_mode = "OFFBOARD"
                self._set_mode_cli.call_async(req)
                self._fw_offboard_requested = True
                self.get_logger().info("천이 전 MC OFFBOARD 요청 (헤딩 정렬 대기)")
            return

        # OFFBOARD 미확인: keepalive + **재요청** 후 대기.
        #
        # (2026-07-27 SITL-7 F-15 수정) 종전엔 hover 세트포인트만 계속 발행하고
        # 그대로 return 했다. Phase 1에서 `_fw_offboard_requested`가 이미 True로
        # 잠겨 재요청 경로가 아예 없었기 때문에, 헤딩 정렬(Phase 2, 실측 13~15초)
        # 구간에서 OFFBOARD를 잃거나 애초에 승인을 못 받으면 **복구 수단 없이
        # 무한 대기**했다. 이제 위 TRANSITION_FW 타임아웃이 최후 방어선이지만,
        # 그 전에 스스로 복구할 기회를 준다.
        #
        # ⚠️ 이 경로를 열면서 반드시 통과시켜야 하는 가드가 있다 — 조종사가
        # POSCTL 등으로 기체를 가져간 상황이면 재요청은 **절대 금지**다.
        # 2026-07-25 flight01에서 노드가 조종사의 POSCTL 인계를 0.9초에 10회
        # 재요청으로 뒤집어 조종사가 KILL 스위치를 써야 했다. 그래서 여기서는
        # 재요청이 아니라 `PILOT_TAKEOVER`(세트포인트 영구 정지)로 빠진다.
        # 이 구간은 `_offboard_engaged_once`가 아직 False일 수 있어(한 번도
        # OFFBOARD를 못 잡은 경우) 상단 최우선 감시가 안 걸리는 구간이기도 하다
        # — Phase 1의 동일한 가드(`is_pilot_takeover` → `_enter_pilot_takeover`)와
        # 같은 이유·같은 처리다.
        if self._current_mode != "OFFBOARD":
            if not offboard_reacquire_allowed(self._current_mode):
                self._enter_pilot_takeover()
                return
            self._setpoint.publish(np.zeros(3))
            self._request_offboard()
            self.get_logger().warn(
                f"정렬 구간 OFFBOARD 이탈 → 재요청 (mode={self._current_mode})",
                throttle_duration_sec=self._offboard_req_min_interval)
            return

        # Phase 2: MC OFFBOARD — hover + yaw rate P제어로 헤딩 정렬
        # gain 0.3 rad/s per rad, 포화 ±0.5 rad/s.
        # _FW_STABLE_REQ 틱 연속으로 |heading_err| < wp0_htol 이어야 Phase 3 진입.
        # (P gain 1.0은 overshoot 후 진동을 유발함 — 0.3으로 낮추고 정착 확인)
        if not self._fw_heading_aligned:
            heading_err = _wrap(chi_wp - state.yaw)  # 부호 있는 오차 (NED)
            if abs(heading_err) < self._wp0_htol:
                self._fw_stable_ticks += 1
                # yaw_rate=0 으로 멈추면 잔류 오차(~11°)에서 고착됨.
                # 안정 구간에서도 소량 P제어를 유지해 0° 쪽으로 계속 수렴.
                yaw_rate_fine = float(np.clip(-heading_err * 0.1, -0.1, 0.1))
                self._setpoint.publish(np.zeros(3), yaw_rate=yaw_rate_fine)
                if self._fw_stable_ticks >= _FW_STABLE_REQ:
                    self._fw_heading_aligned = True
                    self.get_logger().info(
                        f"헤딩 정렬 완료 "
                        f"target={np.degrees(chi_wp):.1f}° "
                        f"current={np.degrees(state.yaw):.1f}° "
                        f"err={np.degrees(heading_err):.1f}° "
                        f"({_FW_STABLE_REQ}틱 안정) → 전진 + 천이 명령")
                    # fall-through to Phase 3
                else:
                    self.get_logger().debug(
                        f"헤딩 안정 대기 {self._fw_stable_ticks}/{_FW_STABLE_REQ} "
                        f"err={heading_err:.3f} rad")
                    return
            else:
                self._fw_stable_ticks = 0  # 이탈 시 카운터 초기화
                # heading_err 양수(NED CW 필요) → ENU angular.z 음수
                # 부호 확정(2026-07-25 감사, 두 경로 교차검증 — 바꾸지 말 것):
                #  ① MAVROS setpoint_velocity 는 각속도에 transform_frame_enu_ned
                #     를 적용 → NED yaw_rate = -angular.z
                #  ② yaw_enu = π/2 - yaw_ned → d(yaw_enu)/dt = -d(yaw_ned)/dt
                # 시뮬(북 0°→동 90°, 우선회): err=+90° → angular.z=-0.47 →
                # NED yaw_rate=+0.47 → yaw 증가 → 오차 단조 감소(수렴).
                # +0.3 으로 바꾸면 발산한다.
                yaw_rate = float(np.clip(-heading_err * 0.3, -0.5, 0.5))
                self._setpoint.publish(np.zeros(3), yaw_rate=yaw_rate)
                self.get_logger().debug(
                    f"헤딩 정렬 중 err={heading_err:.3f} rad yaw_rate={yaw_rate:.2f}")
                return

        # Phase 3: 헤딩 정렬 완료 — WP1 위치 setpoint(직선) + 천이 명령.
        #
        # 🔴 이 setpoint 는 천이 *구간 중에는* 기체를 움직이지 못한다 (2026-07-28
        #    실기체 PX4 `c890d9db0a` 소스로 확정, flight02 실측과 일치).
        #    종전 주석은 "MC가 WP1 방향으로 가속하며 전이 → 사전가속 불필요"라고
        #    적혀 있었으나 **틀렸다.** `standard.cpp` TRANSITION_TO_FW 분기(:201-245)가
        #    천이 중 roll 을 FW 제어기 값으로, pitch 를 `FW_PSP_OFF`(이 기체 0.0)로
        #    덮어쓴다 — MC 위치제어는 자세 지령에서 배제되고 pitch 0° 고정이라
        #    멀티콥터가 수평 이동할 수단 자체가 없다. 전진 추력은 오직 pusher 에서
        #    나온다. SITL 에서 통한 건 무풍이라 표류가 안 보였기 때문이다.
        #    flight02 실측: 3.4초간 남서 65m 를 계속 지령했으나 기체는 정동 90~96°로
        #    표류, 고도 지령 21.2m 에 대해 50.5m 유지(하강 시도 없음).
        #
        #    그래도 이 발행을 유지하는 이유: ① 천이가 완료되는 순간 FW 가 곧바로
        #    같은 setpoint 로 직선 추종에 들어간다 ② OFFBOARD 유지에 스트림이 필요하다.
        #    천이 *완료 조건*은 우리 setpoint 와 무관하다 — `vtol_type.cpp:174-193`,
        #    대기속도계가 유효하면 `VT_TRANS_MIN_TM`(8s) AND airspeed >=
        #    `VT_ARSP_TRANS`(12m/s). 이 기체는 `SENS_EN_MS4525DO=1` 이라 위 가지를 타고,
        #    12m/s 를 못 내면 시간이 지나도 FW 로 안 넘어간 채 `VT_TRANS_TIMEOUT`(20s)에
        #    걸린다. (전문: `logs/2026-07-28_flight02/notes.md` ③·④)
        self._publish_pos_setpoint(
            np.array([self._pts[-1][0], self._pts[-1][1], self._cruise_alt]), chi_wp)

        if not self._fw_transition_sent:
            if not self._cmd_cli.service_is_ready():
                self.get_logger().warn("/mavros/cmd/command 서비스 없음")
                return
            req = CommandLong.Request()
            req.command = 3000   # MAV_CMD_DO_VTOL_TRANSITION
            req.param1 = 4.0   # 목표 상태 FW(4)
            self._cmd_cli.call_async(req)
            self._fw_transition_sent = True
            self.get_logger().info("MC→FW 천이 명령 요청 (위치 setpoint 직선 천이)")

    # ── STREAMING ────────────────────────────────────────────

    def _publish_pos_setpoint(self, pos_ned: np.ndarray, yaw_ned: float) -> None:
        """NED [N, E, h_up] + 헤딩 → ENU PoseStamped 발행 (/mavros/setpoint_position/local).

        PX4 FW 오프보드는 위치 setpoint만 추종한다(속도/가속도 무시). PoseStamped는
        MAVROS가 위치 type_mask를 설정하므로 flower-pattern 선회를 피한다.
        local_position/pose와 동일 프레임이므로 GPS(EKF) 기준 경로를 따른다.

        yaw_ned는 필수 인자다 — 과거 orientation 미설정 시 ROS2 기본값(단위쿼터니언,
        ENU yaw=0=NED yaw=90°)이 실제 헤딩과 무관하게 그대로 발행돼, OFFBOARD
        진입 첫 틱에 순간 yaw 점프(2026-07-21 flight04 yaw 스핀 사고)를 유발했다.
        호출부는 현재 실제 헤딩(`state.yaw`)을 넘겨 그 틱의 setpoint가 항상 실제
        기체 자세와 일치하게 한다(위치를 현재값으로 스트리밍하는 것과 동일한 원리).
        """
        msg = PoseStamped()
        msg.header.frame_id = "map"               # LOCAL_NED ↔ ENU world 프레임
        msg.pose.position.x = float(pos_ned[1])   # E → x_enu
        msg.pose.position.y = float(pos_ned[0])   # N → y_enu
        msg.pose.position.z = float(pos_ned[2])   # h_up = z_enu
        w, x, y, z = yaw_ned_to_quat_enu(yaw_ned)
        msg.pose.orientation.w = w
        msg.pose.orientation.x = x
        msg.pose.orientation.y = y
        msg.pose.orientation.z = z
        self._pos_pub.publish(msg)

    def _enter_pilot_takeover(self) -> None:
        """조종사 RC 인계 확정 — 세트포인트 스트림을 끊고 영구 정지한다.

        OFFBOARD 재요청은 물론이고 세트포인트 발행 자체를 멈춘다. PX4는
        `COM_OF_LOSS_T`(현 기체 1.0s) 동안 setpoint가 없으면 OFFBOARD를
        더 이상 허용하지 않으므로, 이 노드가 기체를 다시 가져갈 방법이 없어진다.
        복귀하려면 launch를 다시 띄워야 한다 — 의도된 동작이다.
        """
        if self._sm == _State.PILOT_TAKEOVER:
            return
        self._sm = _State.PILOT_TAKEOVER
        self.get_logger().warn(
            f"조종사 인계 감지 (mode={self._current_mode}) — "
            f"세트포인트 발행 중단, OFFBOARD 재요청 안 함. 기체는 조종사 것.")

    def _request_offboard(self) -> None:
        # 최후 방어선: 호출부가 무엇이든 수동모드에선 OFFBOARD를 뺏지 않는다.
        if is_pilot_takeover(self._current_mode):
            self._enter_pilot_takeover()
            return
        # 판정 근거인 _current_mode 가 1Hz 소스라 10Hz로 재요청해봐야 같은 값을
        # 열 번 보고 열 번 쏘는 것뿐이다. PX4 페일세이프(AUTO.LOITER/AUTO.RTL)를
        # 되받아치는 강도도 이 간격으로 제한된다.
        now = time.monotonic()
        if (self._last_offboard_req_t is not None
                and now - self._last_offboard_req_t < self._offboard_req_min_interval):
            return
        self._last_offboard_req_t = now
        if not self._set_mode_cli.service_is_ready():
            self.get_logger().warn("/mavros/set_mode 서비스 없음")
            return
        req = SetMode.Request()
        req.custom_mode = "OFFBOARD"
        self._set_mode_cli.call_async(req)

    def _request_arm(self) -> None:
        if not self._arm_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("/mavros/cmd/arming 서비스 없음")
            return
        req = CommandBool.Request()
        req.value = True
        self._arm_cli.call_async(req)

    # ── OVERRIDE ─────────────────────────────────────────────

    def _cb_override(self, msg: Bool) -> None:
        if msg.data:
            self._request_override()

    def _request_override(self) -> None:
        """긴급 수동 전환 진입: manual 목표 모드 요청 후 OVERRIDE 상태로.

        실제 모드 전환 확인·폴백은 _step_override 가 담당한다. 여기서 곧장
        DONE으로 가면(이전 구현) 모드 전환 거부 시 cmd_vel velocity-0 발행이
        OFFBOARD를 살려둬 FW가 직진 폭주한다 → OVERRIDE 상태로 명시 처리.
        """
        state = self._get_state()
        self._override_target = override_mode(state.vtol_state)  # MC→POSCTL, FW→MANUAL
        self._override_ticks = 0
        self._override_fallback_sent = False
        if self._set_mode_cli.service_is_ready():
            req = SetMode.Request()
            req.custom_mode = self._override_target
            self._set_mode_cli.call_async(req)
        else:
            self.get_logger().warn("/mavros/set_mode 서비스 없음 (override)")
        self._sm = _State.OVERRIDE
        self.get_logger().warn(f"긴급 수동 전환 실행 → {self._override_target} 요청")

    # ── 안전 폴백 (타임아웃 4종 + 거리 상한, 2026-07-27 SITL-7 R1) ────

    def _safety_fallback(self, reason: str) -> None:
        """감시에 걸렸을 때의 유일한 출구 — 검증된 `_request_override()` 재사용.

        **새 폴백 경로를 만들지 않는다.** S7 장애주입에서 실증된 안전경로는
        OVERRIDE(FW)/OVERRIDE(MC)/PILOT_TAKEOVER 3종뿐이고, 새 경로를 만들면
        그 경로부터 다시 실증해야 한다. `_request_override()`는
        `override_mode()`로 MC→POSCTL / FW→MANUAL을 고르고, 모드가 거부되면
        10틱 뒤 AUTO.LOITER로 폴백하며, 그 순간부터 setpoint 발행이 멈춘다
        (스트림이 살아 있으면 PX4 페일세이프가 안 걸린다는 것이 C10의 교훈).
        """
        self.get_logger().error(f"{reason} → 안전 폴백(OVERRIDE) 실행")
        self._request_override()

    def _timeout_fallback(self, label: str, elapsed_s: float,
                          timeout_s: float) -> bool:
        """상태 체류 타임아웃 판정 + 초과 시 안전 폴백. 초과했으면 True.

        `timeout_s <= 0`이면 비활성 — 종전(무한대기) 동작으로 되돌리는 탈출구다.
        """
        if not state_timeout_due(elapsed_s, timeout_s):
            return False
        self._safety_fallback(
            f"{label} 타임아웃 {timeout_s:.0f}s 초과 (체류 {elapsed_s:.1f}s)")
        return True

    def _range_guard_breached(self, state: VehicleState) -> bool:
        """이륙지점 기준 수평거리 상한 감시. 초과해 폴백했으면 True.

        하니스 `tools/sitl/run_scenario.py`의 `RangeGuard`와 같은 감시를 노드
        안에 넣은 것이다(C7에서 1564m에 실제 발동해 7km 비행을 막은 전례).
        차이는 셋: ①기준점이 EKF 로컬 원점이 아니라 **실제 이륙지점**
        ②`ros2 topic echo` 폴링(4s 주기)이 아니라 제어 틱(10Hz)마다
        ③런을 죽이는 게 아니라 **기체를 안전 상태로 떨어뜨린다.**
        """
        if self._takeoff_pos_ned is None:
            return False          # 아직 이륙 전 — 기준점 없음
        if self._sm not in _RANGE_GUARDED_STATES:
            return False
        d = horiz_dist_from_origin(state.pos_ned, self._takeoff_pos_ned)
        if not range_limit_exceeded(d, self._range_limit_m):
            return False
        self._safety_fallback(
            f"거리 상한 초과 — 이륙지점에서 {d:.0f}m "
            f"(상한 {self._range_limit_m:.0f}m, 상태={self._sm.value})")
        return True

    def _step_override(self, state: VehicleState) -> None:
        """manual 모드 진입 확인 → 미진입 시 AUTO.LOITER 안전 폴백.

        OFFBOARD setpoint를 더 이상 발행하지 않는다(스트림 중단). manual 모드
        (MANUAL/POSCTL)는 RC·조이스틱 같은 수동제어 소스가 필요해 headless SITL
        에선 거부된다. 1초 내 미진입이면 AUTO.LOITER를 발행해 자율 안전 홀드로
        전환, 기체가 OFFBOARD로 폭주하지 않게 한다. 실기체에선 조종사 RC 인계로
        목표 모드가 즉시 잡혀 폴백 전에 종료된다.
        """
        if override_reached(self._current_mode, self._override_target):
            self.get_logger().warn(
                f"수동/안전 모드 진입 확인 (mode={self._current_mode}) -> DONE")
            self._sm = _State.DONE
            return

        self._override_ticks += 1
        if override_fallback_due(
                self._current_mode, self._override_target,
                self._override_ticks, _OVERRIDE_FALLBACK_TICKS,
                self._override_fallback_sent):
            if not self._set_mode_cli.service_is_ready():
                self.get_logger().warn("/mavros/set_mode 서비스 없음 (override 폴백)")
                return
            req = SetMode.Request()
            req.custom_mode = "AUTO.LOITER"
            self._set_mode_cli.call_async(req)
            self._override_fallback_sent = True
            self.get_logger().warn(
                f"수동 모드({self._override_target}) 미진입 "
                f"(mode={self._current_mode}) -> AUTO.LOITER 안전 폴백 요청")

    # ── TRANSITION_MC ─────────────────────────────────────────

    def _step_transition_mc(self, state: VehicleState) -> None:
        """FW→MC 역천이 명령 발행 → vtol_state==MC 확인 → LANDING 전환.

        역천이 대기 중에도 OFFBOARD 세트포인트 스트림을 끊지 않는다.
        끊기면 PX4 offboard 상실 failsafe(COM_OF_LOSS_T) → RTL.
        최종 WP를 위치 setpoint로 유지 발행해 스트림 keepalive + 목적지 홀드.
        """
        # OFFBOARD keepalive — 역천이 구간 RTL 방지 + 직선 감속.
        # 끝점(근접)을 목표로 하면 FW가 근접 목표로 급선회한다(동향 꺾임).
        # 최종 진행방향으로 항상 lookahead만큼 앞선 점을 목표로 직진 유지.
        far_tgt = state.pos_ned[:2] + self._end_dir * _FW_LOOKAHEAD
        chi_end = float(np.arctan2(self._end_dir[1], self._end_dir[0]))
        self._publish_pos_setpoint(
            np.array([far_tgt[0], far_tgt[1], self._cruise_alt]), chi_end)

        if vtol_is_mc(state.vtol_state):
            self.get_logger().info("MC 전환 완료 -> HOLD (WP1 복귀)")
            self._sm = _State.HOLD
            return

        # 타임아웃 — 역천이가 실패하면 FW가 `_end_dir` 방향으로 **무한 직진**한다.
        # 목표점이 매 틱 현재위치 앞 70m로 다시 놓이는 "도망가는 캐럿"이라
        # 스스로 멈출 조건이 없고, 스트림이 살아 있어 PX4 offboard-loss
        # 페일세이프도 걸리지 않는다 (F-2). 거리 상한과 이중 방어.
        self._mc_trans_elapsed += self._dt
        if self._timeout_fallback("TRANSITION_MC", self._mc_trans_elapsed,
                                  self._mc_trans_timeout):
            return

        if self._current_mode != "OFFBOARD":
            self._request_offboard()
            self.get_logger().warn(
                f"역천이 중 OFFBOARD 이탈 → 재요청 (mode={self._current_mode})",
                throttle_duration_sec=self._offboard_req_min_interval)

        if not self._mc_transition_sent:
            if not self._cmd_cli.service_is_ready():
                self.get_logger().warn("/mavros/cmd/command 서비스 없음")
                return
            req = CommandLong.Request()
            req.command = 3000   # MAV_CMD_DO_VTOL_TRANSITION
            req.param1 = 3.0   # 목표 상태 MC(3)
            self._cmd_cli.call_async(req)
            self._mc_transition_sent = True
            self.get_logger().info("FW->MC 역천이 명령 요청")

    # ── HOLD (WP1 복귀·착륙) ──────────────────────────────────

    def _step_hold(self, state: VehicleState) -> None:
        """MC로 WP1 복귀·홀드 → 도달+안정 시 LANDING (WP1 지점 착륙).

        역천이는 FW 관성으로 WP1을 지나치므로, MC 전환 후 WP1으로 복귀해
        홀드한 뒤 그 자리에서 AUTO.LAND 하면 WP1 상공에서 수직 하강해 착륙한다.

        **위치 setpoint 는 슬루레이트(≤`v_approach` m/s)로 램프한다**
        (2026-07-27 SITL-7 F-6, 2026-07-21 "수정후보 ②"로 특정된 뒤 미수정으로
        남아 있던 항목). 종전엔 첫 틱부터 끝점을 그대로 발행했는데, 직전
        상태(`_step_transition_mc`)가 발행하던 것은 **기체 앞 70m**
        (`_FW_LOOKAHEAD`)의 직진 유지점이고 기체는 끝점을 이미 약 47m
        지나쳐 있어(역천이 오버슈트 ≈ 57 − `d_end_thresh`), 두 setpoint 사이가
        한 틱에 **110~117m** 벌어졌다 (각 런 metrics.json 의
        `setpoint_jump`, state=HOLD: 캠페인 실측 60.2~143.6m).
        ⚠️ 캠페인 최대 계단(214~300m)은 이것이 아니라
        `TRANSITION_FW→STREAMING` 스냅백이며 **그건 R5 소관**이다.

        램프 시작점은 **기체 현재 위치**다(직전 setpoint 가 아니다). 근거:
          ① `_step_following` 의 MC 분기·STREAMING 의 MC 분기가 쓰는 규약과
             같다 — "OFFBOARD 가 이어받는 setpoint 는 항상 실제 위치와
             일치시킨다"(2026-07-20 제어상실 사고 대응).
          ② 직전 setpoint(기체 앞 70m)에서 시작하면 램프가 117m 를
             v_approach 5 m/s 로 걷느라 23초를 쓰는데 `hold_timeout` 이 30초다.
             멈춰야 할 시점에 "앞으로 더 가라"를 이어받는 것이기도 하다.
        따라서 경계 계단은 |직전 setpoint − 현재위치| = `_FW_LOOKAHEAD`(70m)
        까지 줄고 그 뒤로는 틱당 ≤0.5m 다. **남는 70m 는 슬루레이트가 아니라
        `_step_transition_mc` 의 목표점 정의(도망가는 캐럿) 때문이며 그건
        R5(스냅백) 소관이다** — 여기서 건드리지 않는다.
        """
        wp1 = self._pts[-1]
        raw_target = np.array([wp1[0], wp1[1], self._cruise_alt])
        if self._mc_pos_ramp is None:
            self._mc_pos_ramp = np.array(state.pos_ned, dtype=float)
        self._mc_pos_ramp = slew_setpoint(
            self._mc_pos_ramp, raw_target, self._v_approach * self._dt)
        self._publish_pos_setpoint(self._mc_pos_ramp, state.yaw)

        if self._current_mode != "OFFBOARD":
            self._request_offboard()

        dist = float(np.linalg.norm(state.pos_ned[:2] - wp1))
        speed = float(np.linalg.norm(state.vel_ned[:2]))

        self._hold_ticks += 1
        if self._hold_ticks % 20 == 0:
            self.get_logger().info(
                f"WP1 홀드 dist={dist:.1f}m speed={speed:.1f}m/s "
                f"stable={self._hold_stable_ticks}/{_HOLD_STABLE_REQ}")

        if wp1_land_ready(dist, speed,
                          self._wp1_land_radius, self._wp1_land_speed):
            self._hold_stable_ticks += 1
            if self._hold_stable_ticks >= _HOLD_STABLE_REQ:
                self._exit_hold(state, f"WP1 도달·안정 (dist={dist:.1f}m "
                                       f"speed={speed:.1f}m/s)")
                return
        else:
            self._hold_stable_ticks = 0

        self._hold_elapsed += self._dt
        if self._hold_elapsed > self._hold_timeout:
            self.get_logger().warn(
                f"WP1 홀드 타임아웃 {self._hold_timeout:.0f}s 초과 "
                f"(dist={dist:.1f}m)")
            self._exit_hold(state, "홀드 타임아웃")

    def _exit_hold(self, state: VehicleState, why: str) -> None:
        """HOLD 종료 — 비전 착륙이 켜져 있으면 탐색으로, 아니면 종전대로 착륙.

        🔴 이 분기가 F2의 유일한 진입점이다. `vision_landing:=false`(기본)면
        종전 동작(`HOLD → LANDING`)과 **완전히 동일**하다.
        """
        if not self._vision_landing:
            self.get_logger().info(f"{why} → LANDING")
            self._sm = _State.LANDING
            return

        # 나선 중심은 **WP1**이다(기체 현재위치가 아니다). 역천이 오버슈트로
        # 기체가 WP1을 지나쳐 있을 수 있고, 타겟이 있을 것으로 기대되는 곳은
        # 어디까지나 계획된 마지막 WP다.
        self._vs_center = np.array(self._pts[-1], dtype=float)
        self._enter_vision_search(state, pass_idx=0)
        self.get_logger().info(f"{why} → VISION_SEARCH (비전 착륙 활성)")

    def _enter_vision_search(self, state: VehicleState, pass_idx: int) -> None:
        """탐색 1회차/재탐색 공통 진입 — 회차별 고도·반경을 잡고 나선을 초기화한다."""
        self._vs_pass = pass_idx
        self._vs_phase = "align"
        self._vs_elapsed = 0.0
        self._vs_dwell_elapsed = 0.0
        self._vs_latch_buf = []
        # 램프 시작점은 **기체 현재 위치**다 — `_step_hold`와 같은 규약
        # ("OFFBOARD가 이어받는 setpoint는 항상 실제 위치와 일치시킨다",
        #  2026-07-20 제어상실 사고 대응).
        self._vs_ramp = np.array(state.pos_ned, dtype=float)

        alt_agl = self._vs_search_alt()
        # 링 간격·선회속도 자동 산출 — 파라미터가 0 이하일 때만.
        _, short_m = camera_footprint_m(alt_agl)
        self._vs_spacing = (self._vs_spacing_param
                            if self._vs_spacing_param > 0.0
                            else ring_spacing_from_footprint(short_m))
        # 🔴 `_vs_speed`는 **상한**이지 실제 속도가 아니다. 실제 선회속도는 매
        # 틱 그 순간의 반경으로 다시 구한다(`_step_vision_search`) — 최대반경
        # 기준 속도를 나선 전체에 쓰면 작은 반경에서 tilt가 폭주한다.
        # 실측: v=5m/s를 r=5m에 쓰면 tilt 27° → 25m 고도에서 시선이 12.75m
        # 밀리는데 이건 링 간격(12.2m)과 **같은 자릿수**라 커버리지가 통째로
        # 틀어진다. r=2m면 51.9°로 비행 자체가 위험하다.
        self._vs_speed = (self._vs_speed_param
                          if self._vs_speed_param > 0.0
                          else self._v_approach)
        # 나선은 링 간격의 절반에서 시작한다 — 중심부는 직전 `dwell` 단계에서
        # 이미 풋프린트 안에 들어와 있었으므로 다시 훑을 이유가 없고, r→0
        # 근처는 등속을 유지하려면 각속도가 커져 tilt가 감당이 안 된다.
        self._vs_r_start = max(self._vs_spacing / 2.0, 1.0)
        self._vs_spiral = SpiralState(0.0, self._vs_r_start, 0.0)
        self.get_logger().info(
            f"탐색 {pass_idx + 1}회차 — 고도 {alt_agl:.0f}m AGL "
            f"반경 {self._vs_r_start:.1f}→{self._vs_search_radius():.0f}m "
            f"링간격 {self._vs_spacing:.1f}m 속도상한 {self._vs_speed:.1f}m/s "
            f"(풋프린트 단폭 {short_m:.1f}m)")

    def _vs_search_alt(self) -> float:
        """이번 회차의 탐색 고도 (AGL, m)."""
        return self._vs_alt if self._vs_pass == 0 else self._vs_retry_alt

    def _vs_search_radius(self) -> float:
        """이번 회차의 최대 탐색 반경 (m)."""
        return self._vs_radius if self._vs_pass == 0 else self._vs_retry_radius

    def _agl(self, state: VehicleState) -> float:
        """이륙지점 지면 기준 AGL (m). CLIMBING의 AGL 판정과 같은 기준(2026-07-07).

        ⚠️ 라이다 AGL이 아니다 — `fc_ros`에 거리계 배선이 없다(인수인계 D-d).
        평탄지 가정이며, 착륙지 지면이 이륙지점과 높이가 다르면 그만큼 틀린다.
        """
        return float(state.pos_ned[2]) - self._takeoff_ground_h

    # ── VISION_SEARCH (WP1 중심 나선 탐색) ─────────────────────

    def _vision_latch_update(self, vt) -> bool:
        """유효 setpoint를 연속으로 모아 산포를 본다. 래치가 서면 True.

        🔴 **단발 검출로 탐색을 멈추지 않는다.** 탐색 중에는 기체가 계속 움직여
        타겟이 화각을 스쳐 지나가므로, 한 프레임짜리 오탐 하나로 나선을 버리고
        엉뚱한 곳으로 날아가면 회복할 방법이 없다. `vision_latch_ticks` 틱
        연속 + 그 좌표들의 수평 산포가 `vision_latch_spread_m` 이내일 때만
        "같은 것을 계속 보고 있다"로 인정한다.

        연속성이 깨지면 버퍼를 **비운다** — 띄엄띄엄 본 것을 모아 평균 내면
        서로 다른 물체의 중점이라는 실재하지 않는 좌표가 나온다.
        """
        if not (vt.valid and vt.pos_ned is not None and not vt.veto):
            self._vs_latch_buf = []
            return False

        self._vs_latch_buf.append(np.array(vt.pos_ned, dtype=float))
        # 창을 최신 N개로 유지 — 아직 흔들리면 계속 지켜본다.
        self._vs_latch_buf = self._vs_latch_buf[-self._vs_latch_ticks:]
        latched = latch_candidate(
            self._vs_latch_buf, self._vs_latch_ticks, self._vs_latch_spread)
        if latched is None:
            return False
        self._vs_latched = latched
        return True

    def _step_vision_search(self, state: VehicleState) -> None:
        """탐색고도 정렬 → 제자리 확인 → 나선 확대. 래치가 서면 PRECISION_LAND.

        3단으로 나눈 이유는 **싼 것부터 시도**하기 위해서다. 탐색고도의 카메라
        풋프린트가 25m에서 33.4×18.8m라, WP 오차가 풋프린트 단폭의 절반
        (±9.4m) 안이면 나선을 한 바퀴도 돌지 않고 제자리에서 잡힌다. 나선은
        그게 실패했을 때의 수단이다.

        🔴 setpoint의 z는 쓰지 않는다. 이 상태의 목표고도는 **FC가 정한다** —
        `/vision/landing_setpoint`의 z는 착륙점의 절대고도(≈지면)라 그대로
        따라가면 탐색 중에 지면으로 내려가 버린다.
        """
        self._vs_elapsed += self._dt
        vt = self._vision.snapshot(time.monotonic())

        # ① 래치 — 어느 단계에 있든 본다(고도 정렬 중에 보여도 잡는다).
        if self._vision_latch_update(vt):
            self.get_logger().info(
                f"타겟 래치 (연속 {self._vs_latch_ticks}틱, "
                f"N={self._vs_latched[0]:.1f} E={self._vs_latched[1]:.1f}) "
                f"→ PRECISION_LAND")
            self._enter_precision_land(state)
            return

        # ② 생산자 사망 — 탐색을 계속할 이유가 없다. 나선을 마저 도는 것은
        #    배터리만 쓰고 "매끄럽지 못하게" 보인다.
        if vt.link_dead:
            self.get_logger().error(
                "vision 생산자 사망(link ERROR) → 탐색 중단, GPS 착륙")
            self._sm = _State.LANDING
            return

        center = self._vs_center
        alt_target = self._takeoff_ground_h + self._vs_search_alt()

        if self._vs_phase == "align":
            raw = np.array([center[0], center[1], alt_target])
            if (abs(self._agl(state) - self._vs_search_alt()) < 1.0
                    and float(np.linalg.norm(
                        state.pos_ned[:2] - center)) < self._wp1_land_radius):
                self._vs_phase = "dwell"
                self.get_logger().info(
                    f"탐색고도 도달 ({self._agl(state):.1f}m AGL) → 제자리 확인 "
                    f"{self._vs_dwell:.0f}s")
        elif self._vs_phase == "dwell":
            raw = np.array([center[0], center[1], alt_target])
            self._vs_dwell_elapsed += self._dt
            if self._vs_dwell_elapsed >= self._vs_dwell:
                self._vs_phase = "spiral"
                self.get_logger().info("제자리 미검출 → 나선 확대 시작")
        else:
            # 🔴 선회속도를 **그 순간의 반경으로** 다시 구한다. 나디르 카메라는
            # 기체가 기울면 시선이 `AGL·tan(tilt)`만큼 밀리고, 그 오프셋이 링
            # 간격과 같은 자릿수가 되면 나선이 훑었다고 믿는 곳과 실제로 본
            # 곳이 어긋나 탐색 구멍이 생긴다. 하한 1.0m/s는 나선이 멈춰버리는
            # 것을 막는다(반경이 작을수록 허용속도가 0으로 간다).
            v = min(max(max_turn_speed(self._vs_spiral.radius_m), 1.0),
                    self._vs_speed)
            self._vs_spiral = spiral_advance(
                self._vs_spiral, self._dt, v,
                self._vs_spacing, self._vs_r_start, self._vs_search_radius())
            ne = spiral_point_ne(self._vs_spiral, center)
            raw = np.array([ne[0], ne[1], alt_target])

        if self._vs_ramp is None:
            self._vs_ramp = np.array(state.pos_ned, dtype=float)
        self._vs_ramp = slew_setpoint(
            self._vs_ramp, raw, self._v_approach * self._dt)
        self._publish_pos_setpoint(self._vs_ramp, state.yaw)

        if self._current_mode != "OFFBOARD":
            self._request_offboard()

        if self._vs_elapsed % 5.0 < self._dt:
            self.get_logger().info(
                f"탐색 {self._vs_pass + 1}회차 [{self._vs_phase}] "
                f"r={self._vs_spiral.radius_m:.1f}m "
                f"rev={self._vs_spiral.revolutions:.2f} "
                f"t={self._vs_elapsed:.0f}/{self._vs_timeout:.0f}s "
                f"seen={vt.target_seen}")

        # ③ 종료 — 나선 완주 또는 타임아웃.
        done = (self._vs_phase == "spiral" and spiral_complete(
            self._vs_spiral, self._vs_search_radius(), self._vs_r_start))
        timed_out = self._vs_elapsed > self._vs_timeout
        if not (done or timed_out):
            return

        why = "나선 완주" if done else f"타임아웃 {self._vs_timeout:.0f}s"
        nxt = search_pass_next(self._vs_pass)
        if nxt is not None:
            self.get_logger().warn(
                f"탐색 {self._vs_pass + 1}회차 실패 ({why}) → "
                f"{self._vs_retry_alt:.0f}m 재탐색")
            self._enter_vision_search(state, pass_idx=nxt)
        else:
            self.get_logger().warn(
                f"탐색 {self._vs_pass + 1}회차 실패 ({why}) → GPS 착륙 폴백")
            self._sm = _State.LANDING

    # ── PRECISION_LAND (비전 유도 정렬·하강) ────────────────────

    def _enter_precision_land(self, state: VehicleState) -> None:
        self._sm = _State.PRECISION_LAND
        self._pl_elapsed = 0.0
        self._pl_veto_elapsed = 0.0
        self._pl_lost_elapsed = 0.0
        self._pl_target_ne = np.array(self._vs_latched[:2], dtype=float)
        # 하강은 현재 고도에서 시작한다 — 래치 시점의 고도를 유지한 채 먼저
        # 수평 정렬을 끝낸다.
        self._pl_alt = float(state.pos_ned[2])
        self._pl_ramp = np.array(state.pos_ned, dtype=float)

    def _step_precision_land(self, state: VehicleState) -> None:
        """수평은 vision, **수직은 FC**. 정렬이 서야 내려간다.

        🔴 이 분리가 이 상태의 핵심이다. `/vision/landing_setpoint`를 통째로
        `_publish_pos_setpoint`에 먹이면 `slew_setpoint`가 3D 벡터 하나로
        램프하므로 수평 정렬과 수직 하강이 한 예산을 나눠 쓴다 — 25m 상공에서
        15m 옆 타겟이면 3D 거리 29m를 `v_approach`로 걷느라 **수직 성분이
        4m/s 하강**이 된다. 정렬되기 전에 지면에 닿는다.

        그래서 수평 오차가 `vision_align_tol` 안에 들어온 틱에만 목표고도를
        `vision_descend_speed`만큼 내린다. vision 상태머신의
        CENTER_DESCEND("중심 맞추며 하강") 의도를 FC 쪽에서 구현한 것이다.
        """
        self._pl_elapsed += self._dt
        vt = self._vision.snapshot(time.monotonic())
        agl = self._agl(state)

        # ① AUTO.LAND 인계 — 고도 도달 또는 vision의 land 힌트.
        #    `closed_loop_floor_agl_m` = vision의 `terminal_agl_m` = 3.0m라
        #    계약상 이미 "3m부터 AUTO.LAND 인계"다.
        if handoff_due(agl, self._vs_handoff_agl, vt.land_handoff_hint):
            why = (f"인계고도 도달 ({agl:.1f}m AGL)"
                   if agl <= self._vs_handoff_agl
                   else f"vision land 인계 힌트 (state={vt.state})")
            self.get_logger().info(f"{why} → LANDING (AUTO.LAND)")
            self._sm = _State.LANDING
            return

        # ② 거부권 — D-a. vision이 HOLD/ABORT_ASCEND로 빠진 채 시한을 넘기면
        #    무한 호버링을 끊는다.
        if vt.veto:
            self._pl_veto_elapsed += self._dt
            if self._pl_veto_elapsed > self._vs_veto_timeout:
                self.get_logger().warn(
                    f"vision 거부권 {self._vs_veto_timeout:.0f}s 지속 "
                    f"(state={vt.state}) → GPS 착륙 폴백")
                self._sm = _State.LANDING
                return
        else:
            self._pl_veto_elapsed = 0.0

        # ③ 수평 목표 갱신. 신선하지 않으면 **마지막 목표 위에 그대로 머문다** —
        #    탐색으로 되돌아가지 않는다(정성 판정이 "재시도 없이"를 요구하고,
        #    vision 쪽 기조도 2026-07-28 "밀어붙이기"로 전환됐다).
        guided = bool(vt.valid and vt.pos_ned is not None and not vt.veto)
        if guided:
            self._pl_target_ne = np.array(vt.pos_ned[:2], dtype=float)
            self._pl_lost_elapsed = 0.0
        else:
            self._pl_lost_elapsed += self._dt

        err = float(np.linalg.norm(state.pos_ned[:2] - self._pl_target_ne))
        # ④ 수직 스케줄 — 정렬됐고 **그 틱에** 신선한 유도가 있을 때만 내려간다.
        #    유도가 끊기면 고도를 그대로 붙들고 수평만 유지한다(추측 하강 금지).
        aligned = descend_allowed(guided, err, self._vs_align_tol, vt.veto)
        if aligned:
            # 인계고도 아래로는 목표를 내리지 않는다 — 그 밑은 AUTO.LAND 몫이라
            # 여기서 더 내려봐야 ①번 인계 조건이 먼저 걸린다. 방어적 하한.
            floor_h = self._takeoff_ground_h + self._vs_handoff_agl
            self._pl_alt = max(
                self._pl_alt - self._vs_descend_speed * self._dt, floor_h)

        raw = np.array([self._pl_target_ne[0], self._pl_target_ne[1],
                        self._pl_alt])
        if self._pl_ramp is None:
            self._pl_ramp = np.array(state.pos_ned, dtype=float)
        self._pl_ramp = slew_setpoint(
            self._pl_ramp, raw, self._v_approach * self._dt)
        self._publish_pos_setpoint(self._pl_ramp, state.yaw)

        if self._current_mode != "OFFBOARD":
            self._request_offboard()

        if self._pl_elapsed % 2.0 < self._dt:
            self.get_logger().info(
                f"정밀착륙 AGL={agl:.1f}m 수평오차={err:.2f}m "
                f"{'하강' if aligned else '정렬대기'} "
                f"state={vt.state} age={vt.age_s:.2f}s "
                f"t={self._pl_elapsed:.0f}/{self._pl_timeout:.0f}s")

        # ⑤ 전체 타임아웃 — 정렬이 영영 안 서는 경우.
        if self._pl_elapsed > self._pl_timeout:
            self.get_logger().warn(
                f"정밀착륙 타임아웃 {self._pl_timeout:.0f}s 초과 "
                f"(수평오차 {err:.2f}m) → GPS 착륙 폴백")
            self._sm = _State.LANDING

    # ── LANDING ───────────────────────────────────────────────

    def _step_landing(self, state: VehicleState) -> None:
        """AUTO.LAND 명령 발행 → disarmed 확인 → DONE 전환."""
        if landing_done(state.armed):
            self.get_logger().info("착륙 완료 (disarmed) -> DONE")
            self._sm = _State.DONE
            return

        if not self._landing_sent:
            if not self._set_mode_cli.service_is_ready():
                self.get_logger().warn("/mavros/set_mode 서비스 없음")
                return
            req = SetMode.Request()
            req.custom_mode = "AUTO.LAND"
            self._set_mode_cli.call_async(req)
            self._landing_sent = True
            self.get_logger().info("AUTO.LAND 요청")

        self._landing_elapsed += self._dt
        if (not self._landing_timeout_warned
                and self._landing_elapsed > self._landing_timeout):
            self.get_logger().warn(
                f"AUTO.LAND 타임아웃 {self._landing_timeout:.0f}s 초과")
            self._landing_timeout_warned = True

    # ── ENTRY ────────────────────────────────────────────────

    def _step_entry(self, state: VehicleState) -> bool:
        """WP0 방향 접근. 도달 + 헤딩 정렬 완료 시 True."""
        wp0 = self._pts[0]
        pos2 = state.pos_ned[:2]
        dist = float(np.linalg.norm(wp0 - pos2))

        to_wp0 = wp0 - pos2
        if np.linalg.norm(to_wp0) < 1e-3:
            to_wp0 = np.array([1.0, 0.0])
        to_wp0 /= np.linalg.norm(to_wp0)

        chi_to_wp0 = float(np.arctan2(to_wp0[1], to_wp0[0]))
        heading_err = abs(_wrap(chi_to_wp0 - state.yaw))

        if dist < self._wp0_r and heading_err < self._wp0_htol:
            return True

        v_cmd = min(self._v_approach, dist * 0.5)
        vel_cmd = np.array([v_cmd * to_wp0[0], v_cmd * to_wp0[1], 0.0])
        self._setpoint.publish(vel_cmd)
        return False

    # ── FOLLOWING ────────────────────────────────────────────

    def _check_segment_progress(self, pos) -> str:
        """L1 세그먼트 인덱스 진행을 감시하고 로그 꼬리표를 반환한다 (FW 전용).

        `_find_segment()` 가 잡은 세그먼트는 "기체가 지금 경로의 어디를 타고
        있는가"다. 이것이 한 틱에 `_SEG_JUMP_WARN` 점 넘게 튀면 경로를 따라
        전진한 것이 아니라 **다른 레그를 잡은 것**이고, 그 순간
        `_lookahead_point()` 의 목표점이 통째로 다른 곳으로 옮겨간다.
        다만 **코너 안쪽을 자를 때의 전진 급변은 정상**이다 — 임계 상수
        `_SEG_JUMP_WARN` 주석에 두 경우의 구분을 적어 두었다.

        (2026-07-27 SITL-7 R2 진단. 종전 `_find_segment` 는 매 틱 전역 최근접
        탐색이라 이 급변이 구조적으로 가능했고, 그런데도 **관측 수단이 전혀
        없어** 캠페인 24런 어디에도 세그먼트 인덱스가 기록되지 않았다.
        수정 전후를 비교하려면 먼저 볼 수 있어야 한다.)

        경고는 1초 throttle 로 묶는다 — 실제 이벤트가 10Hz 로 연속 발생하면
        로그만으로는 2026-07-25 조종사 인계 사고와 구별이 어려워진다는 F-16 의
        교훈을 같은 방식으로 적용한 것이다.
        """
        seg = int(self._guidance.current_segment)
        n_seg = max(1, len(self._pts) - 1)
        prev = self._last_seg
        self._last_seg = seg
        if prev is not None and abs(seg - prev) > _SEG_JUMP_WARN:
            self.get_logger().warn(
                f"세그먼트 인덱스 급변 {prev}→{seg} (Δ{seg - prev:+d}, "
                f"전체 {n_seg}) pos=[{pos[0]:.1f},{pos[1]:.1f}] "
                f"— 경로상 전진이 아니라 다른 레그 선택일 수 있다",
                throttle_duration_sec=1.0)
        return f" seg={seg}/{n_seg}"

    def _step_following(self, state: VehicleState) -> bool:
        """경로 추종. 경로 끝 도달 시 True.

        FW·MC 둘 다 위치 setpoint를 발행한다(MC도 최종 VTOL 기체의 제어로직을
        그대로 검증해야 하므로 속도 setpoint로 전환하지 않음).

        FW는 L1Guidance의 연속경로 투영+lookahead(`_FW_LOOKAHEAD`, 선회반경보다
        충분히 큼)를 그대로 쓴다 — FW는 실제로 선회하며 경로를 따라가야 하므로
        이 알고리즘이 필요하다.

        MC는 L1Guidance를 쓰지 않는다 — 선회반경 개념이 없는 저속 이산 이동뿐이라
        애초에 이 알고리즘이 불필요하고(2026-07-24, 사용자 지적), 실제로 두 가지
        문제가 있었다: ①FW lookahead(70m)가 MC 짧은 경로 전체보다 커
        `L1Guidance._lookahead_point()`가 중간 waypoint와 무관하게 항상 경로의
        마지막 점으로 클램프됨(2026-07-20 실비행 제어상실 원인이자, 2026-07-24
        flight03/flight05: WP2 부호를 반대로 바꿔도 결과가 똑같았던 근본원인)
        ②왕복(팰린드롬) 직선경로는 왕복 두 구간이 기하학적으로 완전히 겹쳐
        `_find_segment()`의 최근접 탐색 자체가 통째로 무의미(SITL 시뮬레이션
        검증: lookahead를 줄여도 두 구간 전환 지점에서 고정점에 수렴해 경로를
        완주 못 하고 영원히 멈춤). 대신 실제 waypoints:= 원본점만 뽑은
        `self._mc_wps`(`run_planner()`가 만든 ~1m 간격 촘촘한 보간점 `self._pts`
        전부가 아니라 `wp_index`가 있는 점만 — 보간점 전부를 쓰면 간격이
        `mc_end_thresh`보다 좁아 실제로 가지도 않고 인덱스가 앞서가는 문제가
        있었음)을 순서대로 하나씩 찾아간다(`mc_wp_advance`,
        `fc_bridge/execution/state_logic.py`) — 경로 모양(왕복/직선/꺾임)과
        무관하게 항상 배열 순서대로 끝까지 간다. `self._mc_pos_ramp`로
        슬루레이트(≤`v_approach` m/s) 제한 하에 각 목표점에 점진 접근시켜
        순간점프를 막는다.

        MC는 각 WP에 **정착한 뒤** 다음 WP로 출발한다(2026-07-27, 사용자
        지적: "MC는 호버가 되는데 WP에 정착하지 않을 이유가 없다"). 종전엔
        거리 조건만 봐서 반경(`mc_end_thresh`) 경계를 스치는 순간 목표가
        바뀌어, WP 위에 실제로 가보지 않고 코너를 자르며 지나갔다(2026-07-25
        flight04: 1.8~1.9m 지점에서 통과 판정). 이제 반경 안 + 수평속도
        < `mc_wp_settle_speed`가 `mc_wp_settle_time` 동안 연속 유지돼야
        다음 WP로 넘어간다. 위치 setpoint는 그 사이 WP에 고정돼 있으므로
        (램프가 목표에 도달하면 `raw_target`으로 고정) 기체는 그 점 위에
        호버하며 정착한다. 정착에 실패해 갇히는 경우는 `mc_wp_timeout`이
        강제로 다음 WP로 넘긴다.
        """
        pos = state.pos_ned

        if self._is_mc:
            tgt = self._mc_wps[self._mc_wp_idx]
            raw_target = np.array([tgt[0], tgt[1], self._cruise_alt])
            if self._mc_pos_ramp is None:
                self._mc_pos_ramp = np.array(pos, dtype=float)
            self._mc_pos_ramp = slew_setpoint(
                self._mc_pos_ramp, raw_target, self._v_approach * self._dt)
            self._publish_pos_setpoint(self._mc_pos_ramp, state.yaw)
            cte = float(np.linalg.norm(pos[:2] - tgt))  # 진단용: 현재 목표점까지 남은 거리
            seg_info = ""
        else:
            tgt = self._guidance.target_point_ned(pos, _FW_LOOKAHEAD)
            chi_cmd, _, cte = self._guidance.compute(pos, state.vel_ned)
            self._publish_pos_setpoint(
                np.array([tgt[0], tgt[1], self._cruise_alt]), chi_cmd)
            seg_info = self._check_segment_progress(pos)

        # 진입 첫 틱 및 20틱마다 진단 로그 (경로 추종 / OFFBOARD 유지 확인)
        if self._follow_ticks == 0:
            self.get_logger().info(
                f"FOLLOWING 시작 pos=[{pos[0]:.1f},{pos[1]:.1f}] "
                f"tgt=[{tgt[0]:.1f},{tgt[1]:.1f}] cte={cte:.1f}m "
                f"mode={self._current_mode}" + seg_info)
        self._follow_ticks += 1
        if self._follow_ticks % 20 == 0:
            settle_info = (
                f" settle={self._mc_settled_ticks}/{self._mc_settle_req}"
                if self._is_mc else "")
            self.get_logger().info(
                f"FOLLOWING tick={self._follow_ticks} "
                f"mode={self._current_mode} "
                f"cte={cte:.1f}m "
                f"pos=[{pos[0]:.1f},{pos[1]:.1f}] "
                f"tgt=[{tgt[0]:.1f},{tgt[1]:.1f}]"
                + settle_info + seg_info)
        if self._current_mode != "OFFBOARD":
            # throttle: 실제 명령은 1Hz인데 로그만 10Hz로 나가던 것을 맞췄다 (F-16).
            self.get_logger().warn(
                f"FOLLOWING 중 OFFBOARD 이탈 → 재요청 (mode={self._current_mode})",
                throttle_duration_sec=self._offboard_req_min_interval)
            self._request_offboard()

        if not self._is_mc:
            last_pt = self._pts[-1]
            dist_to_end = float(np.linalg.norm(pos[:2] - last_pt))
            return trans_mc_trigger(dist_to_end, self._d_end_thresh)

        dist_to_wp = float(np.linalg.norm(pos[:2] - tgt))
        speed = float(np.linalg.norm(state.vel_ned[:2]))
        next_idx, self._mc_settled_ticks, done = mc_wp_advance(
            self._mc_wp_idx, dist_to_wp, self._mc_end_thresh,
            len(self._mc_wps),
            speed=speed, settle_speed=self._mc_settle_speed,
            settled_ticks=self._mc_settled_ticks,
            settle_req=self._mc_settle_req)
        self._mc_wp_elapsed += self._dt

        if done or next_idx != self._mc_wp_idx:
            self.get_logger().info(
                f"WP{self._mc_wp_idx} 정착 "
                f"(dist={dist_to_wp:.2f}m speed={speed:.2f}m/s "
                f"{self._mc_settle_req}틱 유지) → "
                + ("경로 완료" if done else f"WP{next_idx} 이동"))
            self._mc_wp_elapsed = 0.0
        elif self._mc_wp_elapsed > self._mc_wp_timeout:
            # 바람·GPS 오차로 반경 안에 정착하지 못해 한 WP에 갇히는 것을 막는
            # 폴백. 강제로 다음 WP(마지막이면 경로완료)로 넘겨 미션이 끝까지
            # 진행되게 한다 — hold_timeout·landing_timeout과 같은 규약.
            self.get_logger().warn(
                f"WP{self._mc_wp_idx} 정착 타임아웃 "
                f"{self._mc_wp_timeout:.0f}s 초과 "
                f"(dist={dist_to_wp:.2f}m speed={speed:.2f}m/s) → 강제 진행")
            if self._mc_wp_idx >= len(self._mc_wps) - 1:
                done = True
            else:
                next_idx = self._mc_wp_idx + 1
            self._mc_settled_ticks = 0
            self._mc_wp_elapsed = 0.0

        self._mc_wp_idx = next_idx
        return done



def main(args=None):
    rclpy.init(args=args)
    node = OffboardNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
