"""
OffboardNode: create_timer 기반 상태머신으로 Offboard 경로 추종.

기존 OffboardFollower의 while + time.sleep 루프를 ROS2 타이머 콜백으로 변환.

상태 머신: IDLE → STREAMING → ENTRY → FOLLOWING → DONE
  STREAMING : 더미 세트포인트(속도=0) 발행 후 OFFBOARD 모드 전환 요청
  ENTRY     : entry_mode="mid_flight" 시에만 통과; WP0 진입 + 헤딩 정렬 대기
  FOLLOWING : L1Guidance 기반 경로 추종
  DONE      : 경로 끝 도달 시 진입 (타이머 계속 동작, 속도=0 유지)

MAVROS 토픽을 직접 구독해 TelemetryNode에 의존하지 않는다.
"""
from __future__ import annotations
import enum
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, ExtendedState
from mavros_msgs.srv import SetMode

from fc_bridge.comm.vehicle_state import VehicleState
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


class _State(enum.Enum):
    IDLE      = "idle"
    STREAMING = "streaming"
    ENTRY     = "entry"
    FOLLOWING = "following"
    DONE      = "done"


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

    생성자 파라미터 (런타임 주입):
      path_pts    : np.ndarray (N, 2)  — 2D NE 경로 점 [N, E]
      v_profile   : np.ndarray (N,)   — 각 경로점 목표 속도 (m/s)
      gamma_profile : np.ndarray (N,) 또는 None — 상승각 (rad)
    """

    def __init__(self,
                 path_pts: np.ndarray,
                 v_profile: np.ndarray,
                 gamma_profile: np.ndarray | None = None):
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

        control_hz       = self.get_parameter("control_hz").value
        self._dt         = 1.0 / max(control_hz, 2.0)
        self._entry_mode = self.get_parameter("entry_mode").value
        self._wp0_r      = self.get_parameter("wp0_entry_radius").value
        self._wp0_htol   = self.get_parameter("wp0_heading_tol").value
        self._v_approach = self.get_parameter("v_approach").value
        self._a_max      = self.get_parameter("a_max").value
        self._a_max_init = self._a_max
        self._stall_steps = int(self.get_parameter("error_stall_steps").value)
        self._accel_red  = self.get_parameter("accel_reduction").value
        self._accel_min  = self._a_max * self.get_parameter("accel_min_frac").value
        frame_id         = self.get_parameter("cmd_vel_frame_id").value

        # ── 경로 데이터 ──────────────────────────────────────
        self._pts = np.asarray(path_pts, dtype=float)
        self._v   = np.asarray(v_profile, dtype=float)
        self._gamma = (np.asarray(gamma_profile, dtype=float)
                       if gamma_profile is not None
                       else np.zeros(len(path_pts)))

        # ── 내부 상태 ────────────────────────────────────────
        self._vehicle_state = VehicleState()
        self._vs_lock       = threading.Lock()
        self._guidance      = L1Guidance(
            self.get_parameter("l1_dist").value, self._pts, self._v)
        self._sm            = _State.STREAMING
        self._prev_errors: list[float] = []
        self._offboard_requested = False

        # ── MAVROS 토픽 구독 ─────────────────────────────────
        self.create_subscription(
            PoseStamped,
            "/mavros/local_position/pose",
            self._cb_pose, 10)
        self.create_subscription(
            TwistStamped,
            "/mavros/local_position/velocity_local",
            self._cb_twist, 10)
        self.create_subscription(
            State,
            "/mavros/state",
            self._cb_state, 10)
        self.create_subscription(
            ExtendedState,
            "/mavros/extended_state",
            self._cb_extended, 10)

        # ── 발행 / 서비스 ────────────────────────────────────
        pub = self.create_publisher(
            TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10)
        self._setpoint      = SetpointPublisher(pub, frame_id=frame_id)
        self._set_mode_cli  = self.create_client(SetMode, "/mavros/set_mode")

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

    def _cb_extended(self, msg: ExtendedState) -> None:
        with self._vs_lock:
            update_from_extended_state(self._vehicle_state, msg)

    def _get_state(self) -> VehicleState:
        with self._vs_lock:
            return self._vehicle_state.copy()

    # ── 제어 루프 (타이머 콜백) ──────────────────────────────

    def _control_callback(self) -> None:
        state = self._get_state()

        if self._sm == _State.STREAMING:
            self._setpoint.publish(np.zeros(3))
            if not self._offboard_requested:
                self._request_offboard()
                self._offboard_requested = True
            self._sm = (_State.ENTRY if self._entry_mode == "mid_flight"
                        else _State.FOLLOWING)

        elif self._sm == _State.ENTRY:
            if self._step_entry(state):
                self.get_logger().info("ENTRY 완료 → FOLLOWING")
                self._sm = _State.FOLLOWING

        elif self._sm == _State.FOLLOWING:
            if self._step_following(state):
                self.get_logger().info("경로 추종 완료 → DONE")
                self._sm = _State.DONE

        elif self._sm == _State.DONE:
            self._setpoint.publish(np.zeros(3))

    # ── STREAMING ────────────────────────────────────────────

    def _request_offboard(self) -> None:
        if not self._set_mode_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("/mavros/set_mode 서비스 없음")
            return
        req = SetMode.Request()
        req.custom_mode = "OFFBOARD"
        self._set_mode_cli.call_async(req)

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
        return dist_to_end < 3.0


def main(args=None):
    """
    entry_point 실행 예시.

    실제 배포에서는 launch 파일에서 경로를 계획한 뒤 OffboardNode를 생성한다.
    여기서는 ROS2 파라미터로 waypoints를 받아 run_planner로 경로를 생성한다.

    필요 파라미터 (params YAML 또는 --ros-args -p):
      waypoints  : [[N, E, h], ...] 2D 리스트
      planner    : "eta3" | "diterpin"
      v_cruise   : float (m/s)
      a_max_g    : float (g)
      gravity    : float (m/s²)
    """
    rclpy.init(args=args)

    # 파라미터 읽기용 임시 노드
    tmp = rclpy.create_node("_offboard_param_reader")
    tmp.declare_parameter("waypoints", [[0.0, 0.0, 150.0], [500.0, 0.0, 150.0]])
    tmp.declare_parameter("planner",   "eta3")
    tmp.declare_parameter("v_cruise",  15.0)
    tmp.declare_parameter("a_max_g",   0.3)
    tmp.declare_parameter("gravity",   9.81)

    raw_wps      = tmp.get_parameter("waypoints").value
    planner_name = tmp.get_parameter("planner").value
    vehicle_params = {
        "v_cruise": tmp.get_parameter("v_cruise").value,
        "a_max_g":  tmp.get_parameter("a_max_g").value,
        "gravity":  tmp.get_parameter("gravity").value,
    }
    tmp.destroy_node()

    waypoints = np.array(raw_wps, dtype=float)

    from fc_bridge.planning.planner_runner import run_planner
    path = run_planner(planner_name, waypoints, vehicle_params)

    path_pts      = np.array([pt.pos[:2] for pt in path.points])
    v_profile     = np.array([pt.v_ref   for pt in path.points])
    gamma_profile = np.array([pt.gamma_ref for pt in path.points])

    node = OffboardNode(path_pts=path_pts,
                        v_profile=v_profile,
                        gamma_profile=gamma_profile)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
