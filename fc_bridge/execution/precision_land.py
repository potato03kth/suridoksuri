"""
비전 정밀착륙 판정 로직 (순수 함수).

`fc_ros.nodes.offboard_node` 의 `VISION_SEARCH` / `PRECISION_LAND` 서브상태가
쓰는 판정만 모은다. **ROS·rclpy·numpy 외 의존 없음** — `state_logic.py` 와 같은
규율이고, 같은 이유다: 노드 메서드 안에 판정을 묻어 두면 rclpy 없는 랩탑에서
단위테스트가 불가능해 회귀 그물이 통째로 사라진다.

탐색 **궤적** 기하는 `search_pattern.py` 소관이다(나선·풋프린트·선회속도).
이 파일은 "언제 래치하는가 / 언제 내려가는가 / 언제 손을 떼는가"만 다룬다.

계약 배경은 `docs/fc_precision_land_handoff.md` §2. 이 파일이 소비하는 것은
그 문서가 정의한 vision 쪽 산출물(`landing_setpoint` + `target_status`)을
`fc_ros.adapters.vision_target_bridge` 가 정규화한 결과다.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

#: 🔴 정밀착륙 시한을 **래치 고도에 맞춰 늘릴 때**의 여유 배수(F2-b).
#:
#: `precision_land_timeout` 은 원래 "25m 탐색고도에서 래치한다"를 암묵적으로
#: 가정한 상수였다 — 그 경우 하강만 (25−3)/0.8 = 27.5s 라 60s 안에 정렬 여유가
#: 32.5s 남는다. 그런데 래치는 탐색고도 도달 **전**(HOLD 고도)에도 설 수 있고,
#: 2026-07-29 검증 R5 는 실제로 **AGL 50.0m 에서 래치**했다. 거기서는 하강만
#: (49.6−3.0)/0.8 = 58.3s 라 60s 를 정렬 몇 초로 넘겨 GPS 폴백이 된다
#: (R5 는 vision 의 `command_hint="land"` 가 42s 에 구해줬을 뿐이다).
#:
#: 배수 1.6 의 근거는 **같은 R5 런의 실측 하강 이력**이다:
#:   * 명목 0.8 m/s 스케줄 대비 진입~인계 전 구간 평균 **0.72 m/s**
#:     (AGL 50.0m@t=2s → 22.7m@t=38s), 즉 이상적 하강시간의 **1.11배**.
#:   * 그 1.11 배 위에 정렬 대기·유도 끊김의 여유로 약 44% 를 더 얹은 값이다.
#: 이 배수를 낮추면 "정상적으로 잘 내려오고 있는데 시계가 먼저 끝나" GPS 로
#: 떨어지는, F2 의 존재 이유를 무효화하는 실패 모드가 돌아온다.
DESCENT_BUDGET_FACTOR = 1.6


def latch_altitude_ok(agl_m: float, max_latch_agl_m: float) -> bool:
    """그 고도에서 잡은 좌표로 하강을 시작해도 되는가 (F2-a).

    🔴 **래치는 "여기로 내려간다"는 되돌릴 수 없는 결정이다.** 검출 신뢰구간
    밖(`search_pattern.max_detection_altitude_m()` = 기본 33.57m)에서 온
    setpoint 는 ①타겟이 `min_area` 미달이라 애초에 나올 수 없는 검출이거나
    ②오탐이다. 어느 쪽이든 그 좌표로 하강을 시작하면 엉뚱한 곳에 내린다.
    2026-07-29 검증 R5 가 **AGL 49.6m**(상한 33.57m 초과)에서 래치했다.

    래치를 막아도 손해가 없다는 것이 이 게이트의 핵심이다 — 탐색은 어차피
    탐색고도(25m)로 내려가는 중이고, 낮을수록 검출은 쉬워진다. 거른 프레임은
    "버린 기회"가 아니라 "몇 초 뒤 더 좋은 조건에서 다시 오는 것"이다.

    `max_latch_agl_m <= 0` 이면 상한 없음(비활성) — 이 저장소의 타임아웃·거리
    상한과 같은 규약이다. 호출부가 파라미터 0(=자동 산출)을 먼저 해석해
    양수로 바꿔 넘기므로, 여기서 0 을 보는 것은 **명시적 비활성**뿐이다.
    """
    lim = float(max_latch_agl_m)
    if lim <= 0.0:
        return True
    return float(agl_m) <= lim


def descent_budget_s(agl_m: float, handoff_agl_m: float,
                     descend_speed_mps: float) -> float:
    """래치 고도에서 인계고도까지 **이상적으로** 걸리는 하강시간 (s).

    "이상적"은 매 틱 정렬이 서서 `descend_allowed()` 가 항상 참인 경우다.
    실제로는 정렬 대기·유도 끊김으로 늘어나며, 그 여유는
    `DESCENT_BUDGET_FACTOR` 가 담당한다.

    `descend_speed_mps <= 0` 이면 하강이 성립하지 않으므로 0.0 을 준다 —
    0 나눗셈으로 `inf` 를 시한에 심는 것보다 **시한이 짧아지는 쪽**이 안전측이다
    (그 설정은 애초에 내려갈 수 없어 어차피 시한으로 끝나야 한다).
    """
    v = float(descend_speed_mps)
    if v <= 0.0:
        return 0.0
    drop = float(agl_m) - float(handoff_agl_m)
    return max(0.0, drop) / v


def precision_land_deadline_s(agl_latch_m: float, handoff_agl_m: float,
                              descend_speed_mps: float,
                              base_timeout_s: float,
                              factor: float = DESCENT_BUDGET_FACTOR) -> float:
    """이번 정밀착륙의 실제 시한 (s) — `precision_land_timeout` 을 **하한**으로 쓴다.

        시한 = max(base_timeout_s, factor × 이상적 하강시간)

    `base_timeout_s` 의 의미는 바뀌지 않는다(여전히 "정밀착륙 체류 상한"이고,
    낮은 고도에서 래치하는 정상 경로에서는 그 값이 그대로 쓰인다). 달라지는
    것은 **높은 고도에서 래치했을 때 시한이 같이 늘어난다**는 것뿐이다 —
    파라미터 하나를 새로 만들지 않고 F2-b 를 닫기 위해 이 형태를 골랐다.

    `base_timeout_s <= 0` 이면 비활성 규약이므로 **0 이하를 그대로 돌려준다**
    (호출부가 "0 이하 = 시한 없음"으로 읽는다). 여기서 배수 항을 살려버리면
    비활성이 조용히 활성으로 뒤집힌다.
    """
    base = float(base_timeout_s)
    if base <= 0.0:
        return base
    need = float(factor) * descent_budget_s(
        agl_latch_m, handoff_agl_m, descend_speed_mps)
    return max(base, need)


def latch_candidate(buf: Sequence,
                    min_ticks: int,
                    max_spread_m: float) -> Optional[np.ndarray]:
    """연속 관측 버퍼가 "같은 것을 계속 보고 있다"를 만족하면 착륙점을 돌려준다.

    `buf` 는 최신순이 아니라 **도착순** [N, E, h_up] 배열들이고, 마지막
    `min_ticks` 개만 본다. 반환값은 그 창의 평균 [N, E, h_up], 아니면 None.

    🔴 **단발 검출로 래치하지 않는 이유:** 탐색 중에는 기체가 계속 움직여 타겟이
    화각을 스쳐 지나간다. 한 프레임짜리 오탐 하나로 나선을 버리고 날아가면
    돌아올 방법이 없다(나선 상태를 이미 잃었다). 연속성 + 공간 일관성 두 가지를
    같이 요구하는 것이 "같은 물체"의 최소 증거다.

    🔴 **산포는 수평(N, E)만 본다.** 고도 성분은 카메라-타겟 거리 추정에서 오고
    그건 `nominal.yaml`(미검증 캘리브레이션) 정확도에 직접 걸려 있어, 수평보다
    훨씬 큰 분산을 갖는다. 여기에 같은 임계를 걸면 실제로는 잘 보고 있는데도
    영원히 래치가 안 선다.
    """
    n = int(min_ticks)
    if n <= 0 or len(buf) < n:
        return None
    win = np.array([np.asarray(b, dtype=float) for b in buf[-n:]])
    mean_ne = win[:, :2].mean(axis=0)
    spread = float(np.max(np.linalg.norm(win[:, :2] - mean_ne, axis=1)))
    if spread > float(max_spread_m):
        return None
    return np.array([mean_ne[0], mean_ne[1], float(win[:, 2].mean())])


def descend_allowed(guided: bool,
                    horiz_err_m: float,
                    align_tol_m: float,
                    veto: bool) -> bool:
    """수직 하강을 허가할 틱인가.

    🔴 **수평/수직 분리의 핵심.** `/vision/landing_setpoint` 를 통째로 위치
    setpoint 로 흘리면 `slew_setpoint` 가 3D 벡터 하나로 램프하므로 수평 정렬과
    수직 하강이 같은 속도 예산을 나눠 쓴다 — 25m 상공에서 15m 옆 타겟이면 3D
    거리 29m 를 `v_approach`(5m/s)로 걷느라 **수직 성분이 4m/s 하강**이 되고,
    정렬이 끝나기 전에 지면에 닿는다.

    그래서 "정렬됐고, 그 틱에 신선한 유도가 있고, 거부권이 없을 때"만 내려간다.
    vision 상태머신의 `CENTER_DESCEND`("중심 맞추며 하강") 의도를 FC 쪽에서
    구현한 것이다.

    `guided=False`(유도 상실)면 **고도를 붙든다** — 마지막으로 본 오차를 믿고
    추측 하강하면 그 오차가 그대로 착지 오차가 된다.
    """
    return bool(guided) and not bool(veto) and \
        float(horiz_err_m) < float(align_tol_m)


#: 인계고도 판정 허용대(m). `state_logic.climbing_reached()` 의 `alt_tol` 기본값과
#: **같은 숫자이고 같은 이유**다 — 고도 도달을 점으로 판정하면 기체가 목표 바로
#: 옆(±0.1m)에 정착했을 때 영영 성립하지 않는다.
HANDOFF_ALT_TOL_M = 0.5


def handoff_due(agl_m: float,
                handoff_agl_m: float,
                land_hint: bool,
                alt_tol_m: float = HANDOFF_ALT_TOL_M) -> bool:
    """AUTO.LAND 로 인계할 시점인가.

    두 경로다:
      ① `agl <= handoff_agl_m + alt_tol_m` — 계약상 바닥
         (`closed_loop_floor_agl_m` = vision 의 `terminal_agl_m` = 3.0m).
         그 밑은 비전 폐루프의 몫이 아니다.

         🔴 **허용대가 왜 필요한가 (2026-07-29 SITL 실측, F2-g).** 종전 판정은
         `agl <= handoff_agl_m` 단측이었는데, `_step_precision_land` 의 하강
         스케줄이 **정확히 같은 고도**(`_takeoff_ground_h + handoff_agl`)를
         하한으로 쓴다. 즉 FC 는 목표를 3.0m 에 놓고, 기체는 위치제어 정상
         오차만큼 그 **위에** 정착한다 → 조건이 원리적으로 성립하지 않는다.
         실측 `V2_vision_happy`: AGL 3.4m 까지 정상 하강한 뒤 **3.1m 에서
         22초간 정지**했고, 인계 대신 `precision_land_timeout` 이 걸려 GPS
         폴백으로 끝났다. 즉 해피패스의 정상 종료가 타임아웃이었다.
         (2026-07-29 검증 세션 `R5` 가 이걸 못 본 이유는 vision 의
         `command_hint="land"` 가 42s 에 먼저 구해줬기 때문이다 — ②번 경로.)

         이건 새로운 종류의 실수가 아니라 **이 저장소가 이미 한 번 고친 것**이다:
         `climbing_reached()` 가 "AGL >= transition_alt" 단측 판정으로 목표고도
         바로 아래에 정착해 CLIMBING 이 무한 대기하던 문제를 겪고 목표고도를
         점이 아닌 반경 `alt_tol` 구간으로 바꿨다. 기본값 0.5m 도 그쪽과 같다.

         ⚠️ 허용대는 인계를 **더 높은 고도에서** 일으킨다(3.0 → 3.5m). 그쪽이
         안전 방향이다 — 폐루프를 더 일찍 놓는 것이고, 캘리브레이션이
         `not_for_closed_loop_30cm` 인 현 상태에서 저고도 폐루프는 이득이 없다.
      ② vision 의 `command_hint == "land"` — TERMINAL 에서 블라인드가 2초를
         넘었다는 뜻이다. advisory 지만 **무시하면 설계 의도와 반대로 간다**:
         2초째 못 보는 추정으로 횡방향을 계속 물고 늘어지는 것은 오차를
         **키우는** 모드이고, 접지 순간의 횡속도가 0.105m 라이즈드 매트
         가장자리 전복으로 직결된다.
    """
    return float(agl_m) <= float(handoff_agl_m) + float(alt_tol_m) \
        or bool(land_hint)


def search_pass_next(pass_idx: int, max_passes: int = 2) -> Optional[int]:
    """탐색 회차 실패 시 다음 회차 번호. 더 없으면 None(= GPS 착륙 폴백).

    회차를 늘리는 것이 공짜가 아니라는 것이 이 함수의 존재 이유다 — 대회 성공
    판정이 *"재시도 없이"* 를 포함한 정성 기준이라(`vision_plan.md` §10),
    탐색을 오래 끄는 것 자체가 감점이다. 기본 2회에서 멈춘다.
    """
    nxt = int(pass_idx) + 1
    return nxt if nxt < int(max_passes) else None
