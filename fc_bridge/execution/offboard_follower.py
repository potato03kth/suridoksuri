"""
Phase 2: Offboard 모드 실시간 속도 세트포인트 제어.

상태 머신:
  pre_takeoff 시나리오:
    IDLE → STREAMING → FOLLOWING

  mid_flight 시나리오:
    IDLE → STREAMING → ENTRY(WP0 진입) → FOLLOWING

STREAMING: Offboard 전환 전부터 더미 세트포인트 스트림을 보내 watchdog 유지.
ENTRY: WP0 방향으로 직선 접근 + 헤딩 정렬 완료 시 FOLLOWING 전환.
FOLLOWING: L1 guidance로 경로 추종. 위치 오차 정체 시 a_max 단계 감소.

SET_POSITION_TARGET_LOCAL_NED type_mask:
  velocity only: 0b0000_1111_1000_0111 = 0x0FC7
"""
from __future__ import annotations

import enum
import time
import numpy as np
from typing import Optional

from fc_bridge.comm.mavlink_conn import MavlinkConn
from fc_bridge.comm.telemetry import Telemetry
from fc_bridge.guidance.l1_guidance import L1Guidance
from fc_bridge.planning.speed_profile import compute_speed_profile


# MAVLink 상수
MAV_FRAME_LOCAL_NED = 1
_TYPE_MASK_VEL_ONLY = 0x0FC7   # ignore pos, acc, yaw; use vx,vy,vz


class _State(enum.Enum):
    IDLE = "idle"
    STREAMING = "streaming"   # 세트포인트 스트림 중, Offboard 전환 대기
    ENTRY = "entry"           # mid_flight: WP0 진입 기동
    FOLLOWING = "following"   # L1 경로 추종
    DONE = "done"


class OffboardFollower:
    """
    Offboard 모드 경로 추종기.

    Parameters
    ----------
    conn : MavlinkConn
    telemetry : Telemetry
    path_pts : np.ndarray, shape (N, 2)
        경로 2D 점 배열 [N, E].
    v_profile : np.ndarray, shape (N,)
        각 경로점의 목표 속도 (m/s).
    gamma_profile : np.ndarray, shape (N,), optional
        각 경로점의 상승각 (rad). None이면 0.
    entry_mode : str
        "pre_takeoff" 또는 "mid_flight".
    control_hz : float
        제어 루프 주파수 (Hz). ≥ 2 필수.
    l1_dist : float
        L1 lookahead 거리 (m).
    wp0_entry_radius : float
        ENTRY 상태에서 WP0 도달 판정 반경 (m).
    wp0_heading_tol : float
        ENTRY 상태에서 WP0 헤딩 허용 오차 (rad).
    v_approach : float
        ENTRY 상태에서 WP0로 접근하는 속도 (m/s).
    a_max : float
        횡방향 가속도 상한 (m/s²). 오차 정체 시 감소.
    error_stall_steps : int
        오차 개선 없음 판정 스텝 수.
    accel_reduction : float
        오차 정체 시 a_max 감소 비율.
    accel_min_frac : float
        a_max 최솟값 (원래 값 대비 비율).
    verbose : bool
    """

    def __init__(self,
                 conn: MavlinkConn,
                 telemetry: Telemetry,
                 path_pts: np.ndarray,
                 v_profile: np.ndarray,
                 gamma_profile: Optional[np.ndarray] = None,
                 entry_mode: str = "pre_takeoff",
                 control_hz: float = 10.0,
                 l1_dist: float = 20.0,
                 wp0_entry_radius: float = 5.0,
                 wp0_heading_tol: float = 0.2,
                 v_approach: float = 5.0,
                 a_max: float = 2.94,
                 error_stall_steps: int = 20,
                 accel_reduction: float = 0.9,
                 accel_min_frac: float = 0.3,
                 verbose: bool = False):
        self._conn = conn
        self._tel = telemetry
        self._pts = np.asarray(path_pts, dtype=float)
        self._v = np.asarray(v_profile, dtype=float)
        self._gamma = (np.asarray(gamma_profile, dtype=float)
                       if gamma_profile is not None
                       else np.zeros(len(path_pts)))
        self._entry_mode = entry_mode
        self._dt = 1.0 / max(control_hz, 2.0)
        self._l1 = l1_dist
        self._wp0_r = wp0_entry_radius
        self._wp0_htol = wp0_heading_tol
        self._v_approach = v_approach
        self._a_max = a_max
        self._a_max_init = a_max
        self._stall_steps = error_stall_steps
        self._accel_red = accel_reduction
        self._accel_min = a_max * accel_min_frac
        self._verbose = verbose

        self._guidance = L1Guidance(l1_dist, self._pts, self._v)
        self._state = _State.IDLE

        # 오차 추적
        self._prev_errors: list[float] = []

    # ── 공개 인터페이스 ──────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state.value

    def run(self, duration_s: float = 300.0) -> None:
        """
        Offboard 제어 루프를 실행한다.

        Parameters
        ----------
        duration_s : float
            최대 실행 시간 (s). 경로 끝 도달 또는 시간 초과 시 종료.
        """
        self._state = _State.STREAMING
        t_start = time.monotonic()

        while time.monotonic() - t_start < duration_s:
            t_loop = time.monotonic()
            state = self._tel.get_state()

            if self._state == _State.STREAMING:
                self._send_zero_vel(state)
                # Offboard 진입: 세트포인트 스트림 시작 후 모드 전환
                self._request_offboard()
                self._state = _State.ENTRY if self._entry_mode == "mid_flight" \
                              else _State.FOLLOWING

            elif self._state == _State.ENTRY:
                done = self._step_entry(state)
                if done:
                    if self._verbose:
                        print("[OffboardFollower] ENTRY 완료 → FOLLOWING")
                    self._state = _State.FOLLOWING

            elif self._state == _State.FOLLOWING:
                done = self._step_following(state)
                if done:
                    self._state = _State.DONE
                    break

            # 루프 주기 맞추기
            elapsed = time.monotonic() - t_loop
            sleep_t = self._dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        self._state = _State.DONE
        if self._verbose:
            print("[OffboardFollower] 종료")

    # ── STREAMING ───────────────────────────────────────────

    def _send_zero_vel(self, state) -> None:
        """Offboard 진입 전 더미 세트포인트 (속도=0)."""
        self._send_velocity(state, np.zeros(3))

    def _request_offboard(self) -> None:
        """PX4 Offboard 모드 전환 요청."""
        mav = self._conn.mav
        # MAV_CMD_DO_SET_MODE = 176, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED=1
        # PX4 custom mode for OFFBOARD = 6
        mav.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            176,    # MAV_CMD_DO_SET_MODE
            0,
            1,      # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            6,      # PX4 OFFBOARD custom mode
            0, 0, 0, 0, 0,
        )

    # ── ENTRY (mid_flight) ───────────────────────────────────

    def _step_entry(self, state) -> bool:
        """
        WP0 방향으로 접근. 도달 + 헤딩 정렬 시 True 반환.
        """
        wp0 = self._pts[0]
        pos2 = state.pos_ned[:2]
        dist = float(np.linalg.norm(wp0 - pos2))

        # WP0 방향 단위벡터
        to_wp0 = wp0 - pos2
        if np.linalg.norm(to_wp0) < 1e-3:
            to_wp0 = np.array([1.0, 0.0])
        to_wp0 /= np.linalg.norm(to_wp0)

        chi_to_wp0 = float(np.arctan2(to_wp0[1], to_wp0[0]))
        heading_err = abs(_wrap(chi_to_wp0 - state.yaw))

        if dist < self._wp0_r and heading_err < self._wp0_htol:
            return True

        # WP0 방향 속도 세트포인트
        v_cmd = min(self._v_approach, dist * 0.5)   # 가까워지면 감속
        vel_cmd = np.array([
            v_cmd * to_wp0[0],
            v_cmd * to_wp0[1],
            0.0,
        ])
        self._send_velocity(state, vel_cmd)
        return False

    # ── FOLLOWING ───────────────────────────────────────────

    def _step_following(self, state) -> bool:
        """
        L1 guidance로 경로 추종. 경로 끝 도달 시 True 반환.
        """
        pos = state.pos_ned
        vel = state.vel_ned

        # 현재 경로 세그먼트 인덱스의 gamma
        seg = self._guidance.current_segment
        gamma = float(self._gamma[min(seg, len(self._gamma) - 1)])

        vel_cmd = self._guidance.ned_velocity_cmd(pos, vel, gamma_ref=gamma)

        # 오차 추적 및 a_max 적응
        _, _, cte = self._guidance.compute(pos, vel)
        cross_err = abs(cte)
        self._prev_errors.append(cross_err)
        if len(self._prev_errors) > self._stall_steps:
            self._prev_errors.pop(0)
            if len(self._prev_errors) >= self._stall_steps:
                recent = self._prev_errors[-self._stall_steps // 2:]
                older  = self._prev_errors[:self._stall_steps // 2]
                if np.mean(recent) >= np.mean(older) - 0.05:
                    # 오차 개선 없음 → a_max 감소
                    self._a_max = max(self._a_max * self._accel_red,
                                      self._accel_min)
                    # 속도 프로필 재계산
                    kappa = np.array([
                        p.curvature if hasattr(p, "curvature") else 0.0
                        for p in [None] * len(self._v)
                    ])
                    if self._verbose:
                        print(f"[OffboardFollower] 오차 정체 → a_max={self._a_max:.2f}")
                    self._prev_errors.clear()

        self._send_velocity(state, vel_cmd)

        # 경로 끝 도달 판정
        last_pt = self._pts[-1]
        dist_to_end = float(np.linalg.norm(pos[:2] - last_pt))
        if dist_to_end < 3.0:
            return True
        return False

    # ── MAVLink 송신 ─────────────────────────────────────────

    def _send_velocity(self, state, vel_ned: np.ndarray) -> None:
        """SET_POSITION_TARGET_LOCAL_NED (velocity only) 전송."""
        vx, vy, vz = float(vel_ned[0]), float(vel_ned[1]), float(vel_ned[2])
        self._conn.mav.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,   # time_boot_ms
            self._conn.target_system,
            self._conn.target_component,
            MAV_FRAME_LOCAL_NED,
            _TYPE_MASK_VEL_ONLY,
            0.0, 0.0, 0.0,   # pos (ignored)
            vx, vy, vz,
            0.0, 0.0, 0.0,   # acc (ignored)
            0.0, 0.0,        # yaw, yaw_rate (ignored)
        )


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi
