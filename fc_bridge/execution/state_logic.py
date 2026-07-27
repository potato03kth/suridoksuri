"""OffboardNode 상태 전환 판정 순수 함수.

rclpy 의존 없이 단독 import·테스트 가능.
offboard_node.py 와 fc_ros/test/test_offboard_node.py 양쪽이 이 함수를 참조한다.
"""
import math

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


def trans_mc_trigger(dist_to_end: float, d_end_thresh: float) -> bool:
    """역천이 진입 조건: 경로 끝점까지 거리가 d_end_thresh 미만."""
    return dist_to_end < d_end_thresh


def mc_wp_advance(idx: int, dist_to_wp: float, end_thresh: float,
                   n_points: int, speed: float = 0.0,
                   settle_speed: float = float("inf"),
                   settled_ticks: int = 0,
                   settle_req: int = 0) -> tuple[int, int, bool]:
    """MC 이산 웨이포인트 순차추종: 현재 목표(idx)에 **정착**했으면 다음
    점으로 전진(마지막 점이면 경로완료).
    반환: (다음 tick에 쓸 idx, 다음 tick에 쓸 settled_ticks, 완료여부).

    **정착(settle) 판정** (2026-07-27, 사용자 지적): 도달 판정이 거리 조건
    하나뿐이면 기체가 반경(end_thresh) 경계를 스치는 순간 목표가 다음 점으로
    바뀌어, WP에 실제로 가보지도 않고 코너를 자르며 지나간다(2026-07-25
    flight04 실측: WP를 1.8~1.9m 지점에서 "통과" 판정). MC는 호버가 되므로
    각 WP에 멈춰 서는 게 자연스럽고, "경로를 찍으면 기체가 거기로 간다"는
    이 테스트기체의 목적에도 그쪽이 맞다. 그래서 거리에 더해
      ①수평속도 < settle_speed  ②그 상태가 settle_req 틱 연속 유지
    를 요구한다. 조건이 깨지면 카운터는 0으로 리셋된다(순간적으로 스쳐간
    표본 하나로 정착이 인정되지 않게).

    기본값 speed=0.0·settle_speed=inf·settle_req=0은 "속도조건 없음 +
    대기 없음"이라 종전 fly-by 동작과 완전히 동일하다 —
    `climbing_reached()`의 vz_down 기본값과 같은 규약이다. 이 조건을 실제로
    활성화하려면 호출부가 현재 수평속도와 정착 요구 틱수를 넘겨야 한다.

    타임아웃(정착을 못 해 한 WP에 갇히는 경우)은 이 함수가 아니라 호출부
    (`offboard_node._step_following`)가 경과시간으로 처리한다 —
    `hold_timeout`·`landing_timeout`과 같은 방식.

    MC는 L1Guidance(연속경로 투영+lookahead, FW 선회반경 회피용으로 설계된
    알고리즘)가 애초에 필요 없다 — 직선 구간을 순서대로 지나가기만 하면
    된다(2026-07-24, 사용자 지적). 이전엔 L1Guidance를 그대로 재사용하려다
    두 가지가 드러남: ①FW lookahead(70m)가 MC 짧은 경로보다 커 목표점이
    항상 경로 끝점으로 클램프됨 ②왕복(팰린드롬) 직선경로는 왕복 두 구간이
    기하학적으로 완전히 겹쳐 `_find_segment()`의 최근접 탐색 자체가 통째로
    무의미함(현재 위치와 무관하게 항상 첫 구간만 반환, 심지어 lookahead를
    줄여도 두 구간 사이 전환 지점에서 고정점에 수렴해 경로를 완주 못 하고
    영원히 멈춤 — SITL 시뮬레이션으로 실측). 이산 순차추종은 이 문제 자체가
    발생할 수 없다: 목표는 항상 배열의 다음 점이고, 도달 판정은 그 점까지의
    거리만 본다 — 경로가 왕복이든 직선이든 꺾여있든 상관없이 항상 배열
    순서대로 끝까지 간다.
    """
    if dist_to_wp < end_thresh and speed < settle_speed:
        settled = settled_ticks + 1
        if settled >= settle_req:
            if idx >= n_points - 1:
                return idx, settled, True
            return idx + 1, 0, False
        return idx, settled, False
    return idx, 0, False


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


def after_streaming_state(entry_mode: str) -> str:
    """STREAMING 에서 OFFBOARD 가 확정된 뒤 갈 상태 라벨.

    `entry_mode="mid_flight"` 면 WP0 진입·헤딩 정렬을 하는 ENTRY 를 거치고,
    기본값 `"pre_takeoff"` 면 곧장 FOLLOWING 이다.
    반환값은 offboard_node._State 의 value 문자열과 일치한다
    (`after_climb_state()`·`after_following_state()` 와 같은 규약).

    (2026-07-27 SITL-7 F-14: 호출부가 분기는 제대로 했는데 로그는 언제나
    `"OFFBOARD 확인 → FOLLOWING"` 이라 **실제로 ENTRY 로 갔을 때도 FOLLOWING
    이라고 찍혔다.** 그 결과 하니스 `tools/sitl/analyze_run.py` 의 상태창
    추정이 R1_c10_entry 런에서 ENTRY 구간을 통째로 FOLLOWING 으로 잡았고
    (`logs/2026-07-27_r1_timeouts_range/notes.md` §3-1 인용문에 그 오로그가
    그대로 남아 있다), ENTRY 무한대기라는 결함 자체가 로그만 봐서는
    "FOLLOWING 이 오래 걸린다"로 읽혔다. 상태 라벨을 이 함수 하나로 만들어
    분기와 로그가 갈라질 수 없게 한다.)
    """
    return "entry" if str(entry_mode) == "mid_flight" else "following"


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
    # NaN 방어 (2026-07-25 감사): NaN은 모든 비교가 False라 아래 수렴검사를 그냥
    # 통과한다 — 실측 `home_amsl_confirmed([19.2, nan, 19.3]) -> 19.3`으로,
    # 표본 하나만 NaN이어도 수렴판정이 통째로 무력화됐다. NaN이 확정되면
    # `_home_amsl is None`이 False라 이륙 시퀀스가 그대로 진행돼
    # `CommandTOL.altitude = NaN`이 나가고, PX4는 param7이 유한하지 않으면
    # `MIS_TAKEOFF_ALT`로 폴백한다 → 노드는 transition_alt를 지령했다고 믿는데
    # 기체는 전혀 다른 고도로 이륙하고 CLIMBING은 오지 않는 고도를 기다린다.
    # `/mavros/altitude`의 amsl은 PX4가 NaN을 보낼 수 있는 필드다(ALTITUDE #141,
    # z_global·air_data 둘 다 무효일 때) — 이론적 값이 아니다.
    if not all(math.isfinite(v) for v in recent):
        return None
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
    transition_alt(지면 기준 상승분)에 이륙 지점 지면 AMSL(home_amsl)을 더해
    절대고도로 변환한다.

    ⚠️ **home_amsl의 소스는 `/mavros/altitude` 의 `.amsl` 이다.**
    예전에 쓰던 `/mavros/home_position/home` 의 `geo.altitude`로 되돌리지 말 것 —
    그 값은 MAVROS가 ROS REP-103 관례(NavSatFix.altitude = WGS84 **타원체고**)를
    따라 EGM96 보정을 적용한 값이라 AMSL이 아니다(한국 기준 약 +24.8m 오차).
    2026-07-24에 이것 때문에 3m 상승명령이 실제 ~28.8m 상승으로 실행됐다(커밋 568fbe5).
    호출부는 `offboard_node._cb_home` — 함수명에 `home`이 남아 있으나 구독 토픽은
    `/mavros/altitude`다.

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


# ── 경로 기준계 (2026-07-25 사고 대응) ─────────────────────────

# PX4에서 조종사가 RC로 직접 조종하는 모드들. 이 모드로 빠졌다는 건
# "사람이 기체를 가져갔다"는 뜻이므로 소프트웨어가 OFFBOARD를 되찾으면 안 된다.
PILOT_MODES = frozenset({
    "MANUAL", "POSCTL", "ALTCTL", "STABILIZED", "ACRO",
    "RATTITUDE", "POSITION_SLOW",
})


def is_pilot_takeover(mode: str) -> bool:
    """현재 모드가 조종사 수동 인계 모드인지 판정.

    (2026-07-25 실비행 사고: FOLLOWING 중 `mode != "OFFBOARD"`이면 무조건
    OFFBOARD를 재요청하도록 돼 있어, 조종사가 POSCTL로 기체를 가져간 것을
    0.9초 만에 10회 재요청해 도로 뺏어왔다 — 조종사는 결국 KILL 스위치를
    써야 했다. 근거: `logs/2026-07-25_flight01/notes.md` §2.
    AUTO.LOITER/AUTO.RTL 등 자율 모드는 여기 포함하지 않는다 — 그건 PX4
    페일세이프가 잠깐 가져간 것일 수 있어 재요청이 정당하다.)
    """
    return str(mode).upper() in PILOT_MODES


def offboard_reacquire_allowed(mode: str) -> bool:
    """지금 OFFBOARD를 (재)요청해도 되는 모드인가.

    True 조건: 현재 OFFBOARD가 **아니고**(=요청할 이유가 있고), 조종사가 쥔
    수동모드도 **아닐** 때. 즉 AUTO.LOITER / AUTO.RTL / AUTO.TAKEOFF 같은
    자율·페일세이프 모드이거나 아직 모드를 못 받은 상태("")일 때만 허용한다.

    (2026-07-27 SITL-7 F-15 대응: `_step_transition_fw`의 헤딩 정렬 구간은
    `_fw_offboard_requested`가 이미 True라 OFFBOARD를 잃어도 재요청 경로가
    아예 없었다 — hover 세트포인트만 계속 뿜으며 무한 대기한다. 재요청 경로를
    여는 것이 수정인데, 그 경로가 **2026-07-25 조종사 인계 사고**(POSCTL로
    가져간 기체를 0.9초에 10회 재요청으로 되뺏어와 조종사가 KILL 스위치를
    써야 했던 건)를 되살리면 안 된다. 그래서 "재요청해도 되는가"를 이 함수
    하나로 못박고, 호출부는 False일 때 재요청이 아니라 `PILOT_TAKEOVER`로
    빠지게 한다. `is_pilot_takeover()`와 짝이며 그 정의를 그대로 상속한다.)
    """
    if str(mode).upper() == "OFFBOARD":
        return False
    return not is_pilot_takeover(mode)


# ── 상태 타임아웃 / 거리 상한 (2026-07-27 SITL-7 R1) ────────────

def state_timeout_due(elapsed_s: float, timeout_s: float) -> bool:
    """상태 체류시간이 타임아웃을 넘었는지. `timeout_s <= 0`이면 비활성.

    경계는 strict `>` — `hold_timeout`(`_step_hold`)·`mc_wp_timeout`
    (`_step_following`)·`landing_timeout`(`_step_landing`)이 쓰는 규약과
    동일하게 맞췄다.

    (2026-07-27 SITL-7 캠페인 F-1/F-2/F-3: 상태 12개 중 탈출 타임아웃이 있는
    것은 `HOLD`·`OVERRIDE`·`LANDING` 셋뿐이었고 나머지는 조건이 오지 않으면
    영구 대기했다. C10에서 `entry_mode:=mid_flight` 파라미터 하나로 ENTRY가
    432.67초 내내 탈출하지 못한 채 기체가 이륙지점에서 5.85km 이탈했다 —
    장애주입 없이. 스트림이 살아 있어 PX4 offboard-loss 페일세이프도 안 걸린다.
    근거: `logs/2026-07-27_sitl_vtol_campaign/campaign_report.md` §1, C10 metrics.json
    `completion.state_timeline`.)

    ⚠️ 이 타임아웃의 처리는 `mc_wp_timeout`류의 **강제 진행이 아니라 안전
    폴백**이다 — 호출부는 초과 시 이미 실증된 `_request_override()`
    (manual 시도 → AUTO.LOITER 폴백)로 떨어뜨린다. 새 폴백 경로를 만들면
    그 경로부터 다시 검증해야 하기 때문이다(S7에서 안전경로 3종 실증됨).
    """
    if timeout_s <= 0:
        return False
    return elapsed_s > timeout_s


def horiz_dist_from_origin(pos_ned, origin_ned) -> float:
    """이륙지점(origin_ned) 기준 **수평** 거리 (m). 고도는 무시한다.

    두 인자 모두 NED [N, E, h_up] (또는 앞 2원소만 있어도 된다).
    """
    p = np.asarray(pos_ned, dtype=float)[:2]
    o = np.asarray(origin_ned, dtype=float)[:2]
    return float(np.linalg.norm(p - o))


def range_limit_exceeded(horiz_dist_m: float, limit_m: float) -> bool:
    """수평거리 상한 초과 판정. `limit_m <= 0`이면 비활성(항상 False).

    NaN 방어: 거리 표본이 NaN이면 초과로 보지 **않는다**. NaN은 모든 비교가
    False라 `>`는 어차피 False지만, 이후 누가 부등호를 뒤집어도 안전측이
    유지되도록 명시한다 — `home_amsl_confirmed()`가 NaN 하나로 통째로
    무력화됐던 선례(2026-07-25 감사)와 같은 계열의 함정이다.

    (도입 근거: 하니스 `tools/sitl/run_scenario.py`의 `RangeGuard`가 C7에서
    1564m에 실제 발동해 7km 비행을 막았다. 그러나 그건 **하니스 밖에는 아무
    제동장치가 없다**는 뜻이기도 하다 — 실기체에는 하니스가 없다. 같은 감시를
    노드 안에 넣는다.)
    """
    if limit_m <= 0:
        return False
    if not math.isfinite(horiz_dist_m):
        return False
    return horiz_dist_m > limit_m


def path_max_range(pts, origin_ned) -> float:
    """경로점 전체 중 이륙지점 기준 최대 수평거리 (m). 빈 경로면 0.0.

    노드 기동/이륙 시점에 "이 경로는 애초에 거리 상한을 넘게 설계돼 있다"를
    미리 경고하기 위한 진단용 — 비행 중이 아니라 지상에서 잡으려는 것이다.
    """
    a = np.asarray(pts, dtype=float)
    if a.size == 0:
        return 0.0
    o = np.asarray(origin_ned, dtype=float)[:2]
    return float(np.max(np.linalg.norm(a[:, :2] - o, axis=1)))


def path_origin_ned(takeoff_pos_ned, waypoint_frame: str = "takeoff"):
    """`waypoints:=` 를 해석할 경로 원점(NED [N, E, h_up])을 결정한다.

    "takeoff" (기본) — 이륙 순간 기체 위치를 원점으로 삼는다. 즉 waypoints의
        [0,0,3]은 "이륙지점 상공 3m"를 뜻하게 된다.
    "local"          — 종전 동작. waypoints를 EKF 로컬 프레임 절대좌표로 본다.

    (2026-07-25 실비행 사고: EKF 로컬 원점은 이륙지점과 무관하다. flight01에서
    원점이 이륙지점으로부터 수평 10.94m·수직 10.55m 어긋나 있어 `[0,0,3]`이
    실제로는 "10.9m 옆 + 13.55m AGL"을 지령했고, 기체가 그리로 끌려가다
    조종사 KILL로 끝났다. "로컬 원점 ≠ 지면"은 2026-07-07에 이미 발견돼
    `climbing_reached()`의 `ground_ref_up`으로 보정됐으나, 같은 보정이
    경로추종 고도(`_cruise_alt`)와 수평 waypoints에는 적용되지 않았다.
    근거: `logs/2026-07-25_flight01/notes.md` §3.)

    주의: EKF가 비행 중 로컬 원점을 리셋하면(2026-07-25 실측 xy_reset 23회)
    여기서 캡처한 원점도 함께 어긋난다. 이 함수는 "이륙지점과 EKF 원점이
    처음부터 다르다"는 문제만 해결한다.
    """
    frame = str(waypoint_frame).lower()
    if frame == "local":
        return np.zeros(3)
    if frame != "takeoff":
        raise ValueError(
            f"waypoint_frame은 'takeoff' 또는 'local' — 받은 값: {waypoint_frame!r}")
    return np.asarray(takeoff_pos_ned, dtype=float)[:3].copy()


def slew_setpoint(current, target, max_step: float):
    """위치 setpoint 를 한 틱에 `max_step`(m) 이상 움직이지 않게 제한한다.

    current : 직전 틱에 발행한 setpoint (NED [N, E, h_up])
    target  : 이번 틱의 원래 목표 (같은 프레임)
    max_step: 한 틱 최대 이동량 (m). 보통 `v_approach * dt`.

    반환: 새 setpoint (np.ndarray, shape는 입력과 동일). 남은 거리가
    `max_step` 이하면 target 을 그대로 반환해 목표에 **정확히** 안착시킨다
    (점근 수렴으로 영원히 도달하지 못하는 것을 막는다).

    (2026-07-27 SITL-7 F-6. `_step_following` 의 MC 분기가 `_mc_pos_ramp` 로
    이미 쓰던 계산을 그대로 순수 함수로 뽑은 것이다 — 2026-07-20 실비행에서
    OFFBOARD 확정 순간 먼 절대좌표가 그대로 발행돼 PX4 가 급격한 자세보정을
    시도하다 제어상실·조종사 수동회수로 끝난 사고 대응으로 도입됐던 로직이다.
    같은 계단이 **FW→MC 역천이 직후 HOLD 진입 경계에도 그대로 남아 있었다** —
    캠페인 실측 60.2~143.6m(`logs/2026-07-27_sitl_vtol_campaign/*/metrics.json`
    의 `setpoint_jump` 에서 state=HOLD 항목). 두 곳이 같은 함수를 쓰게 해
    한쪽만 고쳐지는 일이 없게 한다.)
    """
    cur = np.asarray(current, dtype=float)
    tgt = np.asarray(target, dtype=float)
    delta = tgt - cur
    dist = float(np.linalg.norm(delta))
    step = float(max_step)
    if step <= 0.0 or dist <= step:
        return tgt.copy()
    return cur + delta * (step / dist)


def translate_path(pts, mc_wps, cruise_alt: float, origin_ned):
    """경로 데이터 전체를 `origin_ned` 만큼 평행이동한다.

    pts       : (N, 2) 보간 경로점 [N, E]
    mc_wps    : (M, 2) MC 순차추종용 원본 waypoint [N, E]
    cruise_alt: 순항고도 (h_up)
    origin_ned: (3,) 경로 원점 [N, E, h_up] — `path_origin_ned()` 반환값

    반환: (pts, mc_wps, cruise_alt) 새 배열/값. 입력은 변경하지 않는다.

    평행이동만 하므로 경로의 모양·길이·속도 프로파일·진행방향은 모두 보존된다
    (회전·스케일 없음). 호출부는 이 결과로 L1Guidance를 다시 만들어야 한다 —
    L1Guidance가 생성 시점의 경로 배열을 내부에 들고 있기 때문이다.
    """
    origin = np.asarray(origin_ned, dtype=float)
    if origin.shape[0] < 3:
        raise ValueError(f"origin_ned는 [N, E, h_up] 3원소 — 받은 shape: {origin.shape}")

    def _shift(arr):
        # 빈 리스트는 np.asarray가 shape (0,)로 만들어 (2,)와 broadcast 실패한다.
        # 이 함수는 이륙 순간에 호출되므로 여기서 죽으면 최악의 타이밍이다.
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return a.reshape(0, 2)
        return a + origin[:2]

    return (
        _shift(pts),
        _shift(mc_wps),
        float(cruise_alt) + float(origin[2]),
    )
