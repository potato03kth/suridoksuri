"""
OffboardNode: create_timer 기반 상태머신으로 Offboard 경로 추종.

기존 OffboardFollower의 while + time.sleep 루프를 ROS2 타이머 콜백으로 변환.

상태 머신:
  ARM_TAKEOFF   : ARM + AUTO.TAKEOFF 명령
  CLIMBING      : 천이 고도 도달 대기
  TRANSITION_FW : MC→FW 천이 명령 + vtol_state==FW 대기
  STREAMING     : FW 더미 세트포인트(전진속도) 발행 후 OFFBOARD 전환 요청
  ENTRY         : entry_mode="mid_flight" 시에만 통과; WP0 진입 + 헤딩 정렬 대기
  FOLLOWING     : L1Guidance 기반 경로 추종
  TRANSITION_MC : FW→MC 역천이 명령 + vtol_state==MC 대기
  LANDING       : AUTO.LAND 명령 + disarmed 대기
  DONE          : 착륙 완료 (타이머 계속 동작, 속도=0 유지)

MAVROS 토픽을 직접 구독해 TelemetryNode에 의존하지 않는다.
판정 순수 함수: fc_bridge.execution.state_logic (rclpy 없이 테스트 가능).
"""
from __future__ import annotations
import enum
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, ExtendedState
from mavros_msgs.srv import CommandBool, CommandLong, SetMode
from std_msgs.msg import Bool

_MAVROS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

from fc_bridge.comm.vehicle_state import VehicleState
from fc_bridge.execution.state_logic import (
    climbing_reached, vtol_is_fw,
    trans_mc_trigger, vtol_is_mc, landing_done,
    override_mode,
)
from fc_bridge.guidance.l1_guidance import L1Guidance
from fc_ros.adapters.vehicle_state_bridge import (
    update_from_pose,
    update_from_twist,
    update_from_mavros_state,
    update_from_extended_state,
)
from fc_ros.adapters.setpoint_publisher import SetpointPublisher


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


VTOL_STATE_MC = 3
VTOL_STATE_FW = 4


class _State(enum.Enum):
    IDLE          = "idle"
    ARM_TAKEOFF   = "arm_takeoff"
    CLIMBING      = "climbing"
    TRANSITION_FW = "transition_fw"
    STREAMING     = "streaming"
    ENTRY         = "entry"
    FOLLOWING     = "following"
    TRANSITION_MC = "transition_mc"
    LANDING       = "landing"
    DONE          = "done"


class OffboardNode(Node):
    """
    ROS2 파라미터:
      control_hz         (float, 10.0)  — 제어 루프 주파수 (≥2Hz)
      l1_dist            (float, 20.0)  — L1 lookahead 거리 (m)
      entry_mode         (str,  "pre_takeoff") — "pre_takeoff" | "mid_flight"
      wp0_entry_radius   (float, 5.0)   — WP0 도달 판정 반경 (m)
      wp0_heading_tol    (float, 0.2)   — 헤딩 허용 오차 (rad)
      v_approach         (float, 5.0)   — ENTRY 접근 속도 (m/s)
      a_max              (float, 2.94)  — 횡방향 가속도 상한 (m/s²)
      error_stall_steps  (int,   20)    — 오차 정체 판정 스텝 수
      accel_reduction    (float, 0.9)   — 오차 정체 시 a_max 감소 비율
      accel_min_frac     (float, 0.3)   — a_max 최솟값 비율 (원래 값 대비)
      cmd_vel_frame_id   (str,  "base_link") — TwistStamped frame_id (MAVROS 버전에 따라 다름)
      waypoints          (float[], [0,0,50, 100,0,50]) — flat 1D, 코드에서 reshape(-1,3)
      planner            (str,  "eta3") — "eta3" | "diterpin"
      v_cruise           (float, 15.0)  — 순항 속도 (m/s)
      a_max_g            (float, 0.3)   — 횡방향 가속도 상한 (g)
      gravity            (float, 9.81)  — 중력 가속도 (m/s²)
    """

    def __init__(self):
        super().__init__("offboard_node")

        # ── ROS2 파라미터 ────────────────────────────────────
        self.declare_parameter("control_hz",        10.0)
        self.declare_parameter("l1_dist",           20.0)
        self.declare_parameter("entry_mode",        "pre_takeoff")
        self.declare_parameter("wp0_entry_radius",  5.0)
        self.declare_parameter("wp0_heading_tol",   0.2)
        self.declare_parameter("v_approach",        5.0)
        self.declare_parameter("a_max",             2.94)
        self.declare_parameter("error_stall_steps", 20)
        self.declare_parameter("accel_reduction",   0.9)
        self.declare_parameter("accel_min_frac",    0.3)
        self.declare_parameter("cmd_vel_frame_id",  "base_link")
        self.declare_parameter("transition_alt",    50.0)
        self.declare_parameter("d_end_thresh",      10.0)
        self.declare_parameter("landing_timeout",   60.0)
        self.declare_parameter("v_terminal",        15.2)
        self.declare_parameter("decel_dist",        80.0)
        self.declare_parameter("waypoints",  [0.0, 0.0, 50.0, 100.0, 0.0, 50.0])
        self.declare_parameter("planner",    "eta3")
        self.declare_parameter("v_cruise",   15.0)
        self.declare_parameter("a_max_g",    0.3)
        self.declare_parameter("gravity",    9.81)

        control_hz        = self.get_parameter("control_hz").value
        self._dt          = 1.0 / max(control_hz, 2.0)
        self._entry_mode  = self.get_parameter("entry_mode").value
        self._wp0_r       = self.get_parameter("wp0_entry_radius").value
        self._wp0_htol    = self.get_parameter("wp0_heading_tol").value
        self._v_approach  = self.get_parameter("v_approach").value
        self._a_max       = self.get_parameter("a_max").value
        self._a_max_init  = self._a_max
        self._stall_steps = int(self.get_parameter("error_stall_steps").value)
        self._accel_red   = self.get_parameter("accel_reduction").value
        self._accel_min   = self._a_max * self.get_parameter("accel_min_frac").value
        frame_id          = self.get_parameter("cmd_vel_frame_id").value
        self._transition_alt  = float(self.get_parameter("transition_alt").value)
        self._d_end_thresh    = float(self.get_parameter("d_end_thresh").value)
        self._landing_timeout = float(self.get_parameter("landing_timeout").value)

        # ── 경로 계획 ─────────────────────────────────────────
        from fc_bridge.planning.planner_runner import run_planner
        from fc_bridge.planning.terminal_decel import apply_terminal_decel

        raw_wps = np.array(self.get_parameter("waypoints").value, dtype=float).reshape(-1, 3)
        planner_name   = self.get_parameter("planner").value
        vehicle_params = {
            "v_cruise": self.get_parameter("v_cruise").value,
            "a_max_g":  self.get_parameter("a_max_g").value,
            "gravity":  self.get_parameter("gravity").value,
        }
        v_terminal = float(self.get_parameter("v_terminal").value)
        decel_dist = float(self.get_parameter("decel_dist").value)

        path          = run_planner(planner_name, raw_wps, vehicle_params)
        path_pts      = np.array([pt.pos[:2]   for pt in path.points])
        v_profile     = np.array([pt.v_ref     for pt in path.points])
        s_arc         = np.array([pt.s         for pt in path.points])
        gamma_profile = np.array([pt.gamma_ref for pt in path.points])
        v_profile     = apply_terminal_decel(v_profile, s_arc, v_terminal, decel_dist)

        # ── 경로 데이터 ──────────────────────────────────────
        self._pts = np.asarray(path_pts, dtype=float)
        self._v   = np.asarray(v_profile, dtype=float)
        self._gamma = np.asarray(gamma_profile, dtype=float)

        # ── 내부 상태 ────────────────────────────────────────
        self._vehicle_state = VehicleState()
        self._vs_lock       = threading.Lock()
        self._guidance      = L1Guidance(
            self.get_parameter("l1_dist").value, self._pts, self._v)
        self._sm            = _State.ARM_TAKEOFF
        self._prev_errors: list[float] = []
        self._offboard_requested  = False
        self._stream_ticks        = 0
        self._current_mode        = ""
        # ARM_TAKEOFF 시퀀스 플래그
        self._arm_sent            = False
        self._takeoff_sent        = False
        # TRANSITION_FW 시퀀스 플래그
        self._fw_transition_sent  = False
        self._fw_prime_ticks        = 0     # OFFBOARD 프라이밍 틱 수
        self._fw_offboard_requested = False # 천이 전 MC OFFBOARD 요청 여부
        self._fw_heading_aligned    = False # WP 방향 헤딩 정렬 완료 여부
        # TRANSITION_MC / LANDING 시퀀스 플래그
        self._mc_transition_sent    = False
        self._landing_sent          = False
        self._landing_elapsed       = 0.0
        self._landing_timeout_warned = False

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
            Bool,
            "/fc_ros/override",
            self._cb_override, 10)

        # ── 발행 / 서비스 ────────────────────────────────────
        pub = self.create_publisher(
            TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10)
        self._setpoint      = SetpointPublisher(pub, frame_id=frame_id)
        self._set_mode_cli  = self.create_client(SetMode,      "/mavros/set_mode")
        self._arm_cli       = self.create_client(CommandBool,  "/mavros/cmd/arming")
        self._cmd_cli       = self.create_client(CommandLong,  "/mavros/cmd/command")

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
            # FW 상태이므로 속도 0 금지 — 첫 WP 방향 전진 세트포인트 발행
            if len(self._pts) > 1:
                seg = self._pts[1] - self._pts[0]
                seg /= (float(np.linalg.norm(seg)) + 1e-9)
            else:
                seg = np.array([1.0, 0.0])
            v_fwd = float(self._v[0]) if len(self._v) > 0 else 15.0
            fwd_vel = np.array([seg[0] * v_fwd, seg[1] * v_fwd, 0.0])
            self._setpoint.publish(fwd_vel)

            # 천이 전 MC OFFBOARD에서 유입된 경우: 이미 OFFBOARD → 즉시 전환
            if self._current_mode == "OFFBOARD":
                self.get_logger().info("OFFBOARD 확인 -> 경로 추종")
                self._sm = (_State.ENTRY if self._entry_mode == "mid_flight"
                            else _State.FOLLOWING)
                return

            # 폴백: OFFBOARD 미활성 상태에서 유입된 경우 20 tick 후 요청
            self._stream_ticks += 1
            if self._stream_ticks == 20 and not self._offboard_requested:
                self._request_offboard()
                self._offboard_requested = True
                self.get_logger().info("OFFBOARD 전환 요청 (폴백)")

            if self._offboard_requested and self._current_mode == "OFFBOARD":
                self.get_logger().info("OFFBOARD 전환 완료 -> 경로 추종")
                self._sm = (_State.ENTRY if self._entry_mode == "mid_flight"
                            else _State.FOLLOWING)

        elif self._sm == _State.ENTRY:
            if self._step_entry(state):
                self.get_logger().info("ENTRY 완료 -> FOLLOWING")
                self._sm = _State.FOLLOWING

        elif self._sm == _State.FOLLOWING:
            if self._step_following(state):
                self.get_logger().info("경로 추종 완료 -> TRANSITION_MC")
                self._sm = _State.TRANSITION_MC

        elif self._sm == _State.TRANSITION_MC:
            self._step_transition_mc(state)

        elif self._sm == _State.LANDING:
            self._step_landing(state)

        elif self._sm == _State.DONE:
            self._setpoint.publish(np.zeros(3))

    # ── ARM_TAKEOFF ──────────────────────────────────────────

    def _step_arm_takeoff(self, state: VehicleState) -> None:
        """ARM 요청 → armed 확인 → AUTO.TAKEOFF 요청 → CLIMBING 전환."""
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

        if not self._takeoff_sent:
            if not self._set_mode_cli.service_is_ready():
                self.get_logger().warn("/mavros/set_mode 서비스 없음")
                return
            req = SetMode.Request()
            req.custom_mode = "AUTO.TAKEOFF"
            self._set_mode_cli.call_async(req)
            self._takeoff_sent = True
            self.get_logger().info("AUTO.TAKEOFF 요청 -> CLIMBING")
            self._sm = _State.CLIMBING

    # ── CLIMBING ─────────────────────────────────────────────

    def _step_climbing(self, state: VehicleState) -> None:
        """천이 고도 도달 확인 → TRANSITION_FW 전환."""
        if climbing_reached(state.pos_ned[2], self._transition_alt):
            self.get_logger().info(
                f"천이 고도 {self._transition_alt:.1f}m 도달 → TRANSITION_FW")
            self._sm = _State.TRANSITION_FW

    # ── TRANSITION_FW ─────────────────────────────────────────

    def _step_transition_fw(self, state: VehicleState) -> None:
        """
        헤딩 정렬 후 MC→FW 직선 천이.

        Phase 1: MC+HOLD에서 OFFBOARD 프라이밍 (20 tick 세트포인트 발행)
        Phase 2: OFFBOARD 확인 + WP 방향 헤딩 정렬
        Phase 3: 헤딩 정렬 완료 → MC→FW 천이 명령
        Phase 4: vtol_state==FW 대기 → STREAMING
        """
        # Phase 4: FW 전환 완료 확인
        if vtol_is_fw(state.vtol_state):
            self.get_logger().info("FW 전환 완료 -> STREAMING")
            self._sm = _State.STREAMING
            return

        # 경로 시작 방향 (WP0→WP1 단위벡터)
        if len(self._pts) > 1:
            seg = self._pts[1] - self._pts[0]
            seg_norm = float(np.linalg.norm(seg))
            seg = seg / (seg_norm + 1e-9)
        else:
            seg = np.array([1.0, 0.0])
        chi_wp = float(np.arctan2(seg[1], seg[0]))

        v_align = float(self._v[0]) if len(self._v) > 0 else 15.0
        vel_cmd = np.array([seg[0] * v_align, seg[1] * v_align, 0.0])

        # Phase 1: OFFBOARD 프라이밍 (20 tick 동안 세트포인트 발행 후 요청)
        if not self._fw_offboard_requested:
            self._setpoint.publish(vel_cmd)
            self._fw_prime_ticks += 1
            if self._fw_prime_ticks >= 20:
                if not self._set_mode_cli.service_is_ready():
                    self.get_logger().warn("/mavros/set_mode 서비스 없음")
                    return
                req = SetMode.Request()
                req.custom_mode = "OFFBOARD"
                self._set_mode_cli.call_async(req)
                self._fw_offboard_requested = True
                self.get_logger().info("천이 전 MC OFFBOARD 요청 (헤딩 정렬 모드)")
            return

        # OFFBOARD keepalive (Phase 2, 3 공통)
        self._setpoint.publish(vel_cmd)

        # Phase 2: OFFBOARD 확인 + 헤딩 정렬
        if self._current_mode != "OFFBOARD":
            return  # OFFBOARD 미확인, 대기

        if not self._fw_heading_aligned:
            heading_err = abs(_wrap(chi_wp - state.yaw))
            if heading_err < self._wp0_htol:
                self._fw_heading_aligned = True
                self.get_logger().info(
                    f"헤딩 정렬 완료 err={heading_err:.3f} rad → 천이 명령 발행")
            else:
                self.get_logger().debug(f"헤딩 정렬 중 err={heading_err:.3f} rad")
                return

        # Phase 3: MC→FW 천이 명령 발행 (헤딩 정렬 완료 후 직선 천이)
        if not self._fw_transition_sent:
            if not self._cmd_cli.service_is_ready():
                self.get_logger().warn("/mavros/cmd/command 서비스 없음")
                return
            req = CommandLong.Request()
            req.command = 3000   # MAV_CMD_DO_VTOL_TRANSITION
            req.param1  = 4.0   # 목표 상태 FW(4)
            self._cmd_cli.call_async(req)
            self._fw_transition_sent = True
            self.get_logger().info("MC→FW 천이 명령 요청 (직선 천이)")

    # ── STREAMING ────────────────────────────────────────────

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
        state = self._get_state()
        req = SetMode.Request()
        req.custom_mode = override_mode(state.vtol_state)
        self._set_mode_cli.call_async(req)
        self._sm = _State.DONE
        self.get_logger().warn("긴급 수동 전환 실행")

    # ── TRANSITION_MC ─────────────────────────────────────────

    def _step_transition_mc(self, state: VehicleState) -> None:
        """FW→MC 역천이 명령 발행 → vtol_state==MC 확인 → LANDING 전환."""
        if vtol_is_mc(state.vtol_state):
            self.get_logger().info("MC 전환 완료 -> LANDING")
            self._sm = _State.LANDING
            return

        if not self._mc_transition_sent:
            if not self._cmd_cli.service_is_ready():
                self.get_logger().warn("/mavros/cmd/command 서비스 없음")
                return
            req = CommandLong.Request()
            req.command = 3000   # MAV_CMD_DO_VTOL_TRANSITION
            req.param1  = 3.0   # 목표 상태 MC(3)
            self._cmd_cli.call_async(req)
            self._mc_transition_sent = True
            self.get_logger().info("FW->MC 역천이 명령 요청")

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
        wp0  = self._pts[0]
        pos2 = state.pos_ned[:2]
        dist = float(np.linalg.norm(wp0 - pos2))

        to_wp0 = wp0 - pos2
        if np.linalg.norm(to_wp0) < 1e-3:
            to_wp0 = np.array([1.0, 0.0])
        to_wp0 /= np.linalg.norm(to_wp0)

        chi_to_wp0  = float(np.arctan2(to_wp0[1], to_wp0[0]))
        heading_err = abs(_wrap(chi_to_wp0 - state.yaw))

        if dist < self._wp0_r and heading_err < self._wp0_htol:
            return True

        v_cmd   = min(self._v_approach, dist * 0.5)
        vel_cmd = np.array([v_cmd * to_wp0[0], v_cmd * to_wp0[1], 0.0])
        self._setpoint.publish(vel_cmd)
        return False

    # ── FOLLOWING ────────────────────────────────────────────

    def _step_following(self, state: VehicleState) -> bool:
        """L1 guidance 경로 추종. 경로 끝 도달 시 True."""
        pos = state.pos_ned
        vel = state.vel_ned

        seg   = self._guidance.current_segment
        gamma = float(self._gamma[min(seg, len(self._gamma) - 1)])

        vel_cmd = self._guidance.ned_velocity_cmd(pos, vel, gamma_ref=gamma)

        # 오차 추적 및 a_max 적응 감소 (OffboardFollower 로직 그대로 이식)
        _, _, cte = self._guidance.compute(pos, vel)
        cross_err = abs(cte)
        self._prev_errors.append(cross_err)
        if len(self._prev_errors) > self._stall_steps:
            self._prev_errors.pop(0)
            if len(self._prev_errors) >= self._stall_steps:
                recent = self._prev_errors[-self._stall_steps // 2:]
                older  = self._prev_errors[:self._stall_steps // 2]
                if np.mean(recent) >= np.mean(older) - 0.05:
                    self._a_max = max(self._a_max * self._accel_red,
                                      self._accel_min)
                    self.get_logger().debug(
                        f"오차 정체 → a_max={self._a_max:.2f}")
                    self._prev_errors.clear()

        self._setpoint.publish(vel_cmd)

        last_pt      = self._pts[-1]
        dist_to_end  = float(np.linalg.norm(pos[:2] - last_pt))
        return trans_mc_trigger(dist_to_end, self._d_end_thresh)


def main(args=None):
    rclpy.init(args=args)
    node = OffboardNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
