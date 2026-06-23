"""OffboardNode 상태 전환 판정 순수 함수.

rclpy 의존 없이 단독 import·테스트 가능.
offboard_node.py 와 fc_ros/test/test_offboard_node.py 양쪽이 이 함수를 참조한다.
"""
import numpy as np


def climbing_reached(pos_ned_up: float, transition_alt: float) -> bool:
    """천이 고도 도달 여부. pos_ned_up = VehicleState.pos_ned[2] (h_up, 양수=위)."""
    return pos_ned_up >= transition_alt


def vtol_is_fw(vtol_state: int, FW: int = 4) -> bool:
    """vtol_state 가 FW(4) 인지 확인."""
    return vtol_state == FW


def trans_mc_trigger(dist_to_end: float, d_end_thresh: float) -> bool:
    """역천이 진입 조건: 경로 끝점까지 거리가 d_end_thresh 미만."""
    return dist_to_end < d_end_thresh


def vtol_is_mc(vtol_state: int, MC: int = 3) -> bool:
    """vtol_state 가 MC(3) 인지 확인."""
    return vtol_state == MC


def landing_done(armed: bool) -> bool:
    """착륙 완료 여부: disarmed 이면 True."""
    return not armed


def override_mode(vtol_state: int, MC: int = 3) -> str:
    """긴급 override 전환 모드 결정: MC 상태이면 POSCTL, FW 상태이면 MANUAL."""
    return "POSCTL" if vtol_state == MC else "MANUAL"


def vel_aligned_with_path(
    vel_ned,
    pts,
    min_speed: float = 1.0,
    cos_thresh: float = 0.5,
) -> bool:
    """속도 방향이 경로 첫 세그먼트와 cos_thresh(기본 0.5 = 60°) 이상 일치하면 True.

    vel_ned : array-like shape (2,) or (3,)  — NED 속도 [vN, vE, ...]
    pts     : array-like shape (N, 2)        — NE 경로점 (N >= 2 필요)
    """
    vel2 = np.asarray(vel_ned, dtype=float)[:2]
    speed = float(np.linalg.norm(vel2))
    if speed < min_speed:
        return False
    if len(pts) < 2:
        return True
    seg = np.asarray(pts[1], dtype=float)[:2] - np.asarray(pts[0], dtype=float)[:2]
    seg_norm = float(np.linalg.norm(seg))
    if seg_norm < 1e-9:
        return True
    return float(np.dot(vel2 / speed, seg / seg_norm)) > cos_thresh
