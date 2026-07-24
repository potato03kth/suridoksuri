"""OffboardNode 상태 전환 판정 순수 함수.

rclpy 의존 없이 단독 import·테스트 가능.
offboard_node.py 와 fc_ros/test/test_offboard_node.py 양쪽이 이 함수를 참조한다.
"""
import numpy as np


def climbing_reached(pos_ned_up: float, transition_alt: float,
                     ground_ref_up: float = 0.0, alt_tol: float = 0.5,
                     vz_down: float = 0.0, vz_tol: float = 0.3) -> bool:
    """천이 고도 도달 여부 (이륙 지점 지면 기준 AGL, ±alt_tol 허용 + 수직속도 안정).

    pos_ned_up = VehicleState.pos_ned[2] (h_up, 양수=위)는 EKF 로컬 원점 기준
    상대고도이며, 로컬 원점이 실제 지면과 일치하지 않을 수 있다
    (2026-07-07 실측: 로컬 원점이 지면보다 2.11m 높음 → 원점 기준 판정은 실제 AGL과
    어긋나 CLIMBING 무한대기를 유발한다). 이륙 순간 지면 높이(ground_ref_up)를 빼
    실제 AGL로 판정한다. ground_ref_up 기본 0.0은 원점≈지면일 때 기존 동작과 동일.

    기존 "AGL >= transition_alt" 단측 판정은 실비행(2026-07-18 MC 오프보드 테스트)에서
    바로미터/EKF 잡음으로 목표고도 바로 아래(예 -0.1m)에 정착해 정확히 그 값을
    넘지 못하면 CLIMBING이 무한 대기하는 문제가 있었다. 목표고도를 점이 아닌
    반경 alt_tol 구간으로 취급해 |AGL - transition_alt| <= alt_tol 이면 도달로
    판정한다.

    N,E(수평)은 포함하지 않는다: CLIMBING 중 수평 위치는 PX4 AUTO.TAKEOFF가
    자체 관리하며 이 노드는 목표 N,E를 갖지 않는다. 표준 GPS(비-RTK) 수평 오차는
    통상 alt_tol(0.5m)보다 커, 수평 조건까지 추가하면 실비행에서 CLIMBING이
    영구 대기하는 더 심각한 회귀를 유발할 수 있어 의도적으로 제외했다.

    vz_down(VehicleState.vel_ned[2], NED 부호=하강 양수)이 vz_tol 이내여야
    "도달"로 인정한다 (2026-07-24 SITL 재현 대응). 위치조건만 있으면 아직
    상승 관성이 큰 상태(예 vz≈-1.3m/s)에서도 AGL이 transition_alt를 스쳐
    지나가는 순간 곧바로 STREAMING으로 넘어가버린다 — 이 시점엔 아직
    OFFBOARD 권한이 없어 PX4 AUTO.TAKEOFF가 그대로 계속 상승하고, 우리
    쪽 개입(OFFBOARD 요청)이 수 초 뒤에나 이뤄져 그사이 목표고도 대비
    +84%(3.0m→5.54m) 오버슈트가 실측됐다(`logs/2026-07-24_sitl_streaming_overshoot/`).
    vz_down 기본값 0.0·vz_tol 기본 0.3은 "속도조건 없음(항상 통과)"과 동일한
    기존 동작을 보존한다 — 이 조건을 실제로 활성화하려면 호출부가 반드시
    현재 수직속도를 vz_down에 넘겨야 한다.
    """
    pos_reached = abs((pos_ned_up - ground_ref_up) - transition_alt) <= alt_tol
    return pos_reached and abs(vz_down) <= vz_tol


def vtol_is_fw(vtol_state: int, FW: int = 4) -> bool:
    """vtol_state 가 FW(4) 인지 확인."""
    return vtol_state == FW


def trans_mc_trigger(dist_to_end: float, d_end_thresh: float,
                      current_segment: int | None = None,
                      n_segments: int | None = None) -> bool:
    """역천이 진입 조건: 경로 끝점까지 거리가 d_end_thresh 미만.

    current_segment(L1Guidance가 현재 추종 중인 구간 인덱스)와 n_segments(경로
    전체 구간 수)를 함께 주면, 유도기가 실제로 마지막 구간(n_segments-1)까지
    진행했을 때만 완료로 인정한다. 왕복(팰린드롬) 경로처럼 시작점과 끝점이
    같으면, `_find_segment()`가 위치 최근접으로만 구간을 고르기 때문에 이함
    직후 위치가 우연히 끝점 근처라는 이유만으로 첫 구간도 못 가본 채 "완료"로
    오판할 수 있다(2026-07-24 실비행 flight03/flight05, WP2 부호를 반대로
    바꿔도 둘 다 FOLLOWING 진입 즉시 완료 판정되는 것으로 재현 —
    `logs/2026-07-24_flight03/notes.md` 참조). 인자를 생략하면(호출부 미갱신)
    기존 동작과 동일해 회귀 없음.
    """
    if current_segment is not None and n_segments is not None:
        if current_segment < n_segments - 1:
            return False
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


def home_amsl_confirmed(samples, tol: float = 0.5, min_samples: int = 3):
    """최근 home_position AMSL 샘플들이 안정됐는지 확인, 안정 시 최신값 반환.

    samples는 수신 순서대로 쌓인 리스트(오래된 것이 앞). 마지막 min_samples개가
    서로 tol 이내로 수렴해야 신뢰할 수 있는 값으로 인정한다 — 미달이면 None.

    (2026-07-23 실비행 사고: `_cb_home`이 `/mavros/home_position/home`의 첫
    수신값을 그대로 단발 스냅샷해 `takeoff_request_fields()`에 넘기던 기존 방식이,
    막 재시작된 MAVROS가 PX4 부팅 초기(GPS 수직정확도 미수렴 시점)에 래치된
    오래된 home_position을 그대로 받아 26.7m 오차(실제 지면 366.93m AMSL vs
    수신값 393.6m AMSL)로 재현됨 — 이륙목표가 AGL 3.0m 대신 AGL 29.7m로 계산돼
    그대로 이륙. `docs/session_status.md` mc-실기체 트랙 "잔여 리스크" ③에서
    이미 권고됐던 대책 (a)를 구현한 것. 근거: `logs/2026-07-23_flight01/notes.md`.)
    """
    if len(samples) < min_samples:
        return None
    recent = samples[-min_samples:]
    if max(recent) - min(recent) > tol:
        return None
    return float(recent[-1])


def home_amsl_sample_fresh(age_s: float, max_age_s: float = 1.0) -> bool:
    """home_position 메시지 수신 지연(age_s)이 max_age_s 이내인지 판정.

    age_s = 처리 시각(rclpy 노드의 now()) - msg.header.stamp. `_cb_home`이
    이 결과가 False인 표본을 `home_amsl_confirmed()`의 수렴 표본에서 아예
    제외하는 데 쓴다.

    (2026-07-24 SITL 재현: 같은 세션 안에서 이전 PX4 인스턴스로 비행했을 때의
    home_position이 이번 비행에도 그대로 섞여 들어와, `min_samples`(3)개가
    우연히 서로 tol 이내로 일치해 `home_amsl_confirmed()`가 그 stale 값(약
    47.5m AMSL)을 그대로 확정 — 실제 이번 비행 지면은 0.25m AMSL이었는데도
    PX4에 이륙목표 50.47m AMSL(AGL 약 50m)을 요청함. `nav_state`가
    AUTO_TAKEOFF로 8초 넘게 머물며 계속 상승한 것도 이 때문으로 확인됨
    (`logs/2026-07-24_sitl_streaming_overshoot/`). 2026-07-23 사고
    (`home_amsl_confirmed()` 최초 도입 계기)도 동일하게 "래치된 오래된
    home_position"을 원인으로 지목했었다 — `min_samples` 수렴조건만으로는
    래치된 옛 값이 그 자체로 서로 tol 이내인 경우(정상적인 경우이므로 매우
    흔함)를 걸러내지 못한다는 뜻. 오래된 메시지를 표본에서 원천 배제해
    이 경로를 닫는다.
    """
    return age_s <= max_age_s


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
