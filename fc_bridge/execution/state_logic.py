"""OffboardNode 상태 전환 판정 순수 함수.

rclpy 의존 없이 단독 import·테스트 가능.
offboard_node.py 와 fc_ros/test/test_offboard_node.py 양쪽이 이 함수를 참조한다.
"""


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
