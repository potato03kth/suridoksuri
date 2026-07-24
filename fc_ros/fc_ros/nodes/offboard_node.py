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
"""
from __future__ import annotations
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
    trans_mc_trigger, vtol_is_mc, landing_done,
    override_mode, override_reached, override_fallback_due, wp1_land_ready,
    after_climb_state, after_following_state, takeoff_request_fields,
    home_amsl_confirmed,
)
from fc_bridge.comm.vehicle_state import VehicleState
from fc_bridge.utils.rotation import yaw_ned_to_quat_enu
import enum
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, ExtendedState, HomePosition
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
    LANDING = "landing"
    OVERRIDE = "override"
    DONE = "done"


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

        control_hz = self.get_parameter("control_hz").value
        self._dt = 1.0 / max(control_hz, 2.0)
        self._vehicle_type = str(
            self.get_parameter("vehicle_type").value).lower()
        self._is_mc = self._vehicle_type == "mc"
        self._entry_mode = self.get_parameter("entry_mode").value
        self._wp0_r = self.get_parameter("wp0_entry_radius").value
        self._wp0_htol = self.get_parameter("wp0_heading_tol").value
        self._v_approach = self.get_parameter("v_approach").value
        frame_id = self.get_parameter("cmd_vel_frame_id").value
        self._transition_alt = float(
            self.get_parameter("transition_alt").value)
        self._d_end_thresh = float(self.get_parameter("d_end_thresh").value)
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

        # ── 경로 계획 ─────────────────────────────────────────
        from fc_bridge.planning.planner_runner import run_planner, resolve_planner_name
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

        path = run_planner(planner_name, raw_wps, vehicle_params)
        path_pts = np.array([pt.pos[:2] for pt in path.points])
        v_profile = np.array([pt.v_ref for pt in path.points])
        s_arc = np.array([pt.s for pt in path.points])
        v_profile = apply_terminal_decel(
            v_profile, s_arc, v_terminal, decel_dist)

        # ── 경로 데이터 ──────────────────────────────────────
        self._pts = np.asarray(path_pts, dtype=float)
        self._v = np.asarray(v_profile, dtype=float)
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
        self._guidance = L1Guidance(
            self.get_parameter("l1_dist").value, self._pts, self._v)
        self._sm = _State.ARM_TAKEOFF
        self._offboard_requested = False
        self._stream_ticks = 0
        self._follow_ticks = 0
        self._current_mode = ""
        # MC STREAMING/FOLLOWING 위치 setpoint 슬루레이트 제한용 (2026-07-20 사고 대응).
        self._mc_pos_ramp = None
        # ARM_TAKEOFF 시퀀스 플래그
        self._arm_sent = False
        self._takeoff_sent = False
        # 이륙 지점 지면 AMSL (/mavros/home_position). CommandTOL 목표고도(절대) 기준.
        # 수신 즉시 단발 신뢰하지 않고 최근 N개가 tol 이내로 수렴해야 확정한다
        # (2026-07-23 실비행 사고: 막 재시작된 MAVROS가 첫 수신값을 그대로 썼다가
        # 26.7m 스테일 오차로 3m 상승명령이 29.7m 상승으로 실행됨 — state_logic.py
        # home_amsl_confirmed() 참조).
        self._home_amsl_samples: list[float] = []
        self._home_amsl = None
        # 이륙 순간 로컬 지면 높이 (h_up). CLIMBING AGL 판정의 지면 기준(2026-07-07).
        self._takeoff_ground_h = 0.0
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
        # OVERRIDE (긴급 수동 전환) 플래그
        self._override_target = "MANUAL"
        self._override_ticks = 0
        self._override_fallback_sent = False

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
            HomePosition,
            "/mavros/home_position/home",
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

    def _cb_twist(self, msg: TwistStamped) -> None:
        with self._vs_lock:
            update_from_twist(self._vehicle_state, msg)

    def _cb_state(self, msg: State) -> None:
        with self._vs_lock:
            update_from_mavros_state(self._vehicle_state, msg)
            self._current_mode = msg.mode

    def _cb_extended(self, msg: ExtendedState) -> None:
        with self._vs_lock:
            update_from_extended_state(self._vehicle_state, msg)

    def _cb_home(self, msg: HomePosition) -> None:
        # geo.altitude = 이륙 지점 지면 AMSL. CommandTOL 목표고도(절대)의 기준값.
        # 단발 스냅샷 대신 최근 샘플이 tol 이내로 수렴할 때만 확정한다(2026-07-23 대응).
        self._home_amsl_samples.append(float(msg.geo.altitude))
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
                self.get_logger().info("OFFBOARD 확인 → FOLLOWING")
                self._sm = (_State.ENTRY if self._entry_mode == "mid_flight"
                            else _State.FOLLOWING)
                self._follow_ticks = 0
                return

            # 폴백: OFFBOARD 미활성 상태에서 20 tick 후 재요청
            self._stream_ticks += 1
            if self._stream_ticks == 20 and not self._offboard_requested:
                self._request_offboard()
                self._offboard_requested = True
                self.get_logger().info("OFFBOARD 전환 요청 (폴백)")

        elif self._sm == _State.ENTRY:
            if self._step_entry(state):
                self.get_logger().info("ENTRY 완료 -> FOLLOWING")
                self._sm = _State.FOLLOWING

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

        elif self._sm == _State.LANDING:
            self._step_landing(state)

        elif self._sm == _State.OVERRIDE:
            self._step_override(state)

        elif self._sm == _State.DONE:
            self._setpoint.publish(np.zeros(3))

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
            # 이륙 순간 로컬 지면 높이 캡처 (로컬 원점≠지면 보정, CLIMBING AGL 판정용).
            self._takeoff_ground_h = float(state.pos_ned[2])
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

    # ── CLIMBING ─────────────────────────────────────────────

    def _step_climbing(self, state: VehicleState) -> None:
        """운용 고도 도달 확인 → 다음 상태 전환.

        VTOL: TRANSITION_FW(MC→FW 천이). MC: STREAMING(천이 생략, OFFBOARD 진입).
        """
        if climbing_reached(state.pos_ned[2], self._transition_alt,
                            self._takeoff_ground_h):
            nxt = _State(after_climb_state(self._is_mc))
            self.get_logger().info(
                f"운용 고도 {self._transition_alt:.1f}m 도달 → {nxt.value}"
                + (" (MC, 천이 생략)" if self._is_mc else ""))
            self._sm = nxt

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
                self.get_logger().warn(
                    f"천이 중 OFFBOARD 이탈 → 재요청 (mode={self._current_mode})")
            return

        # Phase 1: hover 세트포인트로 OFFBOARD 프라이밍 (HOLD 중에는 무시됨)
        if not self._fw_offboard_requested:
            self._setpoint.publish(np.zeros(3))  # hover — 방향 무관
            self._fw_prime_ticks += 1
            if self._fw_prime_ticks >= 20:
                if not self._set_mode_cli.service_is_ready():
                    self.get_logger().warn("/mavros/set_mode 서비스 없음")
                    return
                req = SetMode.Request()
                req.custom_mode = "OFFBOARD"
                self._set_mode_cli.call_async(req)
                self._fw_offboard_requested = True
                self.get_logger().info("천이 전 MC OFFBOARD 요청 (헤딩 정렬 대기)")
            return

        # OFFBOARD 미확인: keepalive 후 대기
        if self._current_mode != "OFFBOARD":
            self._setpoint.publish(np.zeros(3))
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
                # 부호 검증: 잘못 돌면 -0.3을 +0.3으로 바꿀 것
                yaw_rate = float(np.clip(-heading_err * 0.3, -0.5, 0.5))
                self._setpoint.publish(np.zeros(3), yaw_rate=yaw_rate)
                self.get_logger().debug(
                    f"헤딩 정렬 중 err={heading_err:.3f} rad yaw_rate={yaw_rate:.2f}")
                return

        # Phase 3: 헤딩 정렬 완료 — WP1 위치 setpoint(직선) + 천이 명령.
        # 위치 setpoint는 MC·FW 양쪽에서 작동: MC가 WP1 방향으로 가속하며 전이 →
        # FW가 동일 위치 setpoint로 직선 추종한다. (사전가속 불필요)
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

    def _request_offboard(self) -> None:
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

        if self._current_mode != "OFFBOARD":
            self._request_offboard()
            self.get_logger().warn(
                f"역천이 중 OFFBOARD 이탈 → 재요청 (mode={self._current_mode})")

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
        MC는 근접/정지 목표를 추종할 수 있어 끝점을 직접 목표로 발행한다.
        """
        wp1 = self._pts[-1]
        self._publish_pos_setpoint(
            np.array([wp1[0], wp1[1], self._cruise_alt]), state.yaw)

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
                self.get_logger().info(
                    f"WP1 도달·안정 → LANDING "
                    f"(dist={dist:.1f}m speed={speed:.1f}m/s)")
                self._sm = _State.LANDING
                return
        else:
            self._hold_stable_ticks = 0

        self._hold_elapsed += self._dt
        if self._hold_elapsed > self._hold_timeout:
            self.get_logger().warn(
                f"WP1 홀드 타임아웃 {self._hold_timeout:.0f}s 초과 "
                f"(dist={dist:.1f}m) → 강제 LANDING")
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

    def _step_following(self, state: VehicleState) -> bool:
        """경로 추종. 경로 끝 도달 시 True.

        FW·MC 둘 다 위치 setpoint를 발행한다(MC도 최종 VTOL 기체의 제어로직을
        그대로 검증해야 하므로 속도 setpoint로 전환하지 않음). FW는 lookahead
        위치를 그대로 발행 — 경로가 충분히 길어(선회반경보다 큼) 전방 목표가
        현재 위치와 크게 어긋나지 않는다. MC는 짧은 경로(선회반경 개념이
        없음)에서 같은 lookahead(70m)를 그대로 쓰면 목표점이 경로 끝점으로
        고정돼 현재 위치와 무관한 절대좌표가 될 수 있다(2026-07-20 실비행
        제어상실 원인). STREAMING에서 이어지는 `self._mc_pos_ramp`를 그 lookahead
        목표로 슬루레이트(≤`v_approach` m/s) 제한 하에 점진 접근시켜 순간점프를 막는다.
        cte는 진단용으로만 계산(조향엔 미사용).
        """
        pos = state.pos_ned

        if self._is_mc:
            tgt = self._guidance.target_point_ned(pos, _FW_LOOKAHEAD)
            _, _, cte = self._guidance.compute(pos, state.vel_ned)
            raw_target = np.array([tgt[0], tgt[1], self._cruise_alt])
            if self._mc_pos_ramp is None:
                self._mc_pos_ramp = np.array(pos, dtype=float)
            delta = raw_target - self._mc_pos_ramp
            dist = float(np.linalg.norm(delta))
            max_step = self._v_approach * self._dt
            if dist > max_step:
                self._mc_pos_ramp = self._mc_pos_ramp + delta * (max_step / dist)
            else:
                self._mc_pos_ramp = raw_target
            self._publish_pos_setpoint(self._mc_pos_ramp, state.yaw)
        else:
            tgt = self._guidance.target_point_ned(pos, _FW_LOOKAHEAD)
            chi_cmd, _, cte = self._guidance.compute(pos, state.vel_ned)
            self._publish_pos_setpoint(
                np.array([tgt[0], tgt[1], self._cruise_alt]), chi_cmd)

        # 진입 첫 틱 및 20틱마다 진단 로그 (경로 추종 / OFFBOARD 유지 확인)
        if self._follow_ticks == 0:
            self.get_logger().info(
                f"FOLLOWING 시작 pos=[{pos[0]:.1f},{pos[1]:.1f}] "
                f"tgt=[{tgt[0]:.1f},{tgt[1]:.1f}] cte={cte:.1f}m "
                f"mode={self._current_mode}")
        self._follow_ticks += 1
        if self._follow_ticks % 20 == 0:
            self.get_logger().info(
                f"FOLLOWING tick={self._follow_ticks} "
                f"mode={self._current_mode} "
                f"cte={cte:.1f}m "
                f"pos=[{pos[0]:.1f},{pos[1]:.1f}] "
                f"tgt=[{tgt[0]:.1f},{tgt[1]:.1f}]")
        if self._current_mode != "OFFBOARD":
            self.get_logger().warn(
                f"FOLLOWING 중 OFFBOARD 이탈 → 재요청 (mode={self._current_mode})")
            self._request_offboard()

        last_pt = self._pts[-1]
        dist_to_end = float(np.linalg.norm(pos[:2] - last_pt))
        return trans_mc_trigger(dist_to_end, self._d_end_thresh)



def main(args=None):
    rclpy.init(args=args)
    node = OffboardNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
