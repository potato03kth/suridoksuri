"""OffboardNode 상태 전환 판정 순수 함수.

rclpy 의존 없이 단독 import·테스트 가능.
offboard_node.py 와 fc_ros/test/test_offboard_node.py 양쪽이 이 함수를 참조한다.
"""
import numpy as np


def climbing_reached(pos_ned_up: float, transition_alt: float,
                     ground_ref_up: float = 0.0) -> bool:
    """천이 고도 도달 여부 (이륙 지점 지면 기준 AGL).

    pos_ned_up = VehicleState.pos_ned[2] (h_up, 양수=위)는 EKF 로컬 원점 기준
    상대고도이며, 로컬 원점이 실제 지면과 일치하지 않을 수 있다
    (2026-07-07 실측: 로컬 원점이 지면보다 2.11m 높음 → 원점 기준 판정은 실제 AGL과
    어긋나 CLIMBING 무한대기를 유발한다). 이륙 순간 지면 높이(ground_ref_up)를 빼
    실제 AGL로 판정한다. ground_ref_up 기본 0.0은 원점≈지면일 때 기존 동작과 동일.
    """
    return (pos_ned_up - ground_ref_up) >= transition_alt


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


def after_climb_state(is_mc: bool) -> str:
    """CLIMBING 다음 상태 라벨.

    MC(순수 멀티콥터)는 FW 천이가 없으므로 STREAMING으로 직행하고,
    VTOL은 TRANSITION_FW(MC→FW 천이)를 거친다.
    반환값은 offboard_node._State 의 value 문자열과 일치한다.
    """
    return "streaming" if is_mc else "transition_fw"


def after_following_state(is_mc: bool) -> str:
    """FOLLOWING 다음 상태 라벨.

    MC는 역천이가 없으므로 HOLD(마지막 WP 복귀·정지·착륙)로 직행하고,
    VTOL은 TRANSITION_MC(FW→MC 역천이)를 거친다.
    반환값은 offboard_node._State 의 value 문자열과 일치한다.
    """
    return "hold" if is_mc else "transition_mc"


def override_mode(vtol_state: int, MC: int = 3) -> str:
    """긴급 override 전환 모드 결정: MC 상태이면 POSCTL, FW 상태이면 MANUAL."""
    return "POSCTL" if vtol_state == MC else "MANUAL"


def override_reached(current_mode: str, target_mode: str) -> bool:
    """override 종료 조건: manual 목표 모드 또는 AUTO.LOITER(안전 폴백) 진입."""
    return current_mode == target_mode or current_mode == "AUTO.LOITER"


def override_fallback_due(
    current_mode: str,
    target_mode: str,
    ticks: int,
    fallback_ticks: int,
    fallback_sent: bool,
) -> bool:
    """manual 모드 미진입 시 AUTO.LOITER 안전 폴백 발행 조건.

    headless SITL·RC 없음 등으로 PX4가 MANUAL/POSCTL을 거부하면 OFFBOARD가
    유지돼 기체가 폭주한다. fallback_ticks 경과 후에도 목표 모드 미진입이면
    AUTO.LOITER를 1회 발행해 자율 안전 홀드로 전환한다(경계값 strict >=).
    """
    if override_reached(current_mode, target_mode):
        return False
    return ticks >= fallback_ticks and not fallback_sent


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


def takeoff_request_fields(transition_alt: float, home_amsl: float) -> dict:
    """CommandTOL(/mavros/cmd/takeoff) 요청 필드.

    CommandTOL.altitude(→ MAVLink MAV_CMD_NAV_TAKEOFF param7)는 **AMSL 절대고도**다.
    transition_alt(지면 기준 상승분)에 이륙 지점 지면 AMSL(home_amsl,
    /mavros/home_position/home 의 geo.altitude)을 더해 절대고도로 변환한다.

    (2026-07-07 실측 확정: transition_alt(예 4.0)를 그대로 altitude에 실으면
    지면 AMSL(예 19.2m)보다 낮은 값이라 PX4 navigator가
    "Already higher than takeoff altitude"로 이륙을 취소, 모터 미가동 후
    COM_DISARM_PRFLT(10s) preflight auto-disarm으로 이륙 실패했다.
    근거: logs/2026-07-07_0217_last/notes.md. SITL은 transition_alt(50)>지면AMSL(≈0)
    이라 이 버그가 가려져 PASS했었다.)

    latitude/longitude/yaw=nan 이 MAVLink 관례상 "현재 위치/헤딩 사용"을 뜻한다
    (2026-07-06 SITL 실패로 확정: 0.0/0.0은 실제 좌표(위도 0, 경도 0)로 해석돼
    PX4가 유효한 이륙 목표를 만들지 못해 고도 미상승 후 preflight 안전 disarm됨).
    """
    return {
        "min_pitch": 0.0,
        "yaw": float("nan"),
        "latitude": float("nan"),
        "longitude": float("nan"),
        "altitude": float(home_amsl) + float(transition_alt),
    }


def wp1_land_ready(
    dist: float,
    speed: float,
    radius: float,
    speed_thresh: float,
) -> bool:
    """WP1 착륙 준비 판정: 수평거리 < radius 이고 수평속도 < speed_thresh.

    역천이 오버슈트 후 MC로 WP1에 복귀해 충분히 정착했는지 확인한다.
    (경계값은 strict < 이므로 미트리거)
    """
    return dist < radius and speed < speed_thresh
