"""
`fc_bridge.execution.precision_land` — 비전 정밀착륙 판정 로직 테스트.

`offboard_node` 의 `VISION_SEARCH`/`PRECISION_LAND` 서브상태가 **같은 함수를
참조**하므로(노드 안에 복제본이 없다) 여기서 red 가 나면 실제 비행 경로가 깨진
것이다 — `test_offboard_node.py` 머리말이 밝힌 것과 같은 규율이다.
"""
import numpy as np
import pytest

from fc_bridge.execution.precision_land import (
    latch_candidate, descend_allowed, handoff_due, search_pass_next,
    latch_altitude_ok, descent_budget_s, precision_land_deadline_s,
)


# ── latch_candidate ─────────────────────────────────────────

def _pt(n, e, h=10.0):
    return np.array([n, e, h], dtype=float)


def test_latch_needs_min_ticks():
    """버퍼가 부족하면 아무리 일관돼도 래치하지 않는다."""
    buf = [_pt(5.0, 3.0), _pt(5.0, 3.0)]
    assert latch_candidate(buf, 3, 3.0) is None


def test_latch_succeeds_on_consistent_window():
    buf = [_pt(5.0, 3.0), _pt(5.1, 3.1), _pt(4.9, 2.9)]
    out = latch_candidate(buf, 3, 3.0)
    assert out is not None
    assert out[0] == pytest.approx(5.0, abs=1e-6)
    assert out[1] == pytest.approx(3.0, abs=1e-6)


def test_latch_rejects_scattered_window():
    """산포가 임계를 넘으면 래치하지 않는다 — 서로 다른 물체를 평균 내면
    실재하지 않는 좌표가 나온다."""
    buf = [_pt(0.0, 0.0), _pt(10.0, 0.0), _pt(0.0, 10.0)]
    assert latch_candidate(buf, 3, 3.0) is None


def test_latch_uses_only_last_n():
    """창 밖의 옛 관측은 산포 판정에 끼지 않는다."""
    buf = [_pt(100.0, 100.0),          # 옛날 것 — 무시돼야 한다
           _pt(5.0, 3.0), _pt(5.1, 3.1), _pt(4.9, 2.9)]
    assert latch_candidate(buf, 3, 3.0) is not None


def test_latch_spread_is_horizontal_only():
    """🔴 고도 성분은 산포 판정에 넣지 않는다 — 거리추정 분산이 수평보다
    훨씬 커서, 같은 임계를 걸면 잘 보고 있는데도 영원히 래치가 안 선다."""
    buf = [_pt(5.0, 3.0, 10.0), _pt(5.0, 3.0, 18.0), _pt(5.0, 3.0, 2.0)]
    out = latch_candidate(buf, 3, 3.0)
    assert out is not None
    assert out[2] == pytest.approx(10.0, abs=1e-6)


def test_latch_boundary_is_strict_greater():
    """정확히 임계면 통과한다(초과일 때만 거절)."""
    buf = [_pt(0.0, 0.0), _pt(0.0, 0.0), _pt(6.0, 0.0)]
    # 평균 N=2.0 → 최대이격 = 4.0
    assert latch_candidate(buf, 3, 4.0) is not None
    assert latch_candidate(buf, 3, 3.999) is None


def test_latch_zero_ticks_never_latches():
    assert latch_candidate([_pt(1.0, 1.0)], 0, 3.0) is None


# ── descend_allowed ─────────────────────────────────────────

def test_descend_allowed_when_aligned_and_guided():
    assert descend_allowed(True, 0.5, 1.0, False) is True


def test_descend_blocked_when_not_aligned():
    assert descend_allowed(True, 1.5, 1.0, False) is False


def test_descend_blocked_when_guidance_lost():
    """🔴 유도가 끊기면 정렬돼 있어도 내려가지 않는다 — 마지막으로 본 오차를
    믿고 추측 하강하면 그 오차가 그대로 착지 오차가 된다."""
    assert descend_allowed(False, 0.1, 1.0, False) is False


def test_descend_blocked_by_veto():
    assert descend_allowed(True, 0.1, 1.0, True) is False


def test_descend_boundary_strict():
    """임계와 정확히 같으면 하강하지 않는다(strict <)."""
    assert descend_allowed(True, 1.0, 1.0, False) is False
    assert descend_allowed(True, 0.999, 1.0, False) is True


# ── handoff_due ─────────────────────────────────────────────

def test_handoff_on_floor_altitude():
    assert handoff_due(3.0, 3.0, False) is True
    assert handoff_due(2.9, 3.0, False) is True
    assert handoff_due(3.1, 3.0, False) is False


def test_handoff_on_land_hint_at_any_altitude():
    """land 힌트는 고도와 무관하게 인계다 — TERMINAL 블라인드 2초 초과는
    '횡 유도를 놓으라'는 뜻이고, 물고 늘어지면 오차를 키운다."""
    assert handoff_due(20.0, 3.0, True) is True


def test_handoff_not_due_high_and_no_hint():
    assert handoff_due(20.0, 3.0, False) is False


# ── search_pass_next ────────────────────────────────────────

def test_search_pass_advances_once():
    assert search_pass_next(0) == 1


def test_search_pass_stops_after_max():
    """🔴 무한 재탐색 금지 — 대회 성공판정이 '재시도 없이'를 포함한다."""
    assert search_pass_next(1) is None


def test_search_pass_respects_custom_max():
    assert search_pass_next(1, max_passes=3) == 2
    assert search_pass_next(2, max_passes=3) is None


# ══════════════════════════════════════════════════════════════
# F2-a — 래치 고도 게이트 (2026-07-29)
# ══════════════════════════════════════════════════════════════
#
# 🔴 실측 근거: 검증 세션 R5(`logs/2026-07-29_f2_verify/R5_patched_handoff`)가
#   HOLD 고도 **AGL 49.6m** 에서 래치했다 — 검출 상한 33.57m 를 16m 초과한
#   고도의 좌표로 하강을 시작한 것이다.


def test_latch_altitude_gate_rejects_above_ceiling():
    """검출 상한 위에서는 래치를 허가하지 않는다 — R5 가 정확히 이 경우다."""
    assert latch_altitude_ok(49.6, 33.57) is False


def test_latch_altitude_gate_allows_search_altitude():
    """정상 경로는 막히지 않는다 — 기본 탐색고도 25m 는 상한 안이다."""
    assert latch_altitude_ok(25.0, 33.57) is True


def test_latch_altitude_gate_is_inclusive_at_the_boundary():
    """경계는 포함이다(`<=`) — 상한과 정확히 같은 고도에서 거절하면
    `max_detection_altitude_m()` 의 정의("이 값까지 검출 가능")와 어긋난다."""
    assert latch_altitude_ok(33.57, 33.57) is True


def test_latch_altitude_gate_disabled_by_non_positive_limit():
    """0 이하 = 비활성. 이 저장소의 타임아웃·거리 상한과 같은 규약이다."""
    assert latch_altitude_ok(1000.0, 0.0) is True
    assert latch_altitude_ok(1000.0, -1.0) is True


def test_latch_altitude_gate_matches_detection_ceiling():
    """게이트의 자동 기본값은 탐색 기하가 주는 검출 상한 그 자체다 —
    두 숫자가 갈라지면 "검출은 되는데 래치는 안 되는" 고도대가 생긴다."""
    from fc_bridge.execution.search_pattern import (
        max_detection_altitude_m, detection_altitude_ok as det_ok,
    )
    ceiling = max_detection_altitude_m()
    for agl in (10.0, 25.0, 33.0, 34.0, 40.0, 50.0):
        assert latch_altitude_ok(agl, ceiling) is det_ok(agl), agl


# ══════════════════════════════════════════════════════════════
# F2-b — 시한이 래치 고도와 정합해야 한다
# ══════════════════════════════════════════════════════════════


def test_descent_budget_is_the_plain_quotient():
    """(고도 − 인계고도) / 하강률. R5 조건 그대로."""
    assert descent_budget_s(49.6, 3.0, 0.8) == pytest.approx(58.25)


def test_descent_budget_is_zero_below_handoff():
    """인계고도 아래면 하강할 것이 없다 — 음수 예산을 만들지 않는다."""
    assert descent_budget_s(2.0, 3.0, 0.8) == 0.0


def test_descent_budget_zero_speed_does_not_produce_inf():
    """하강률 0 은 설정 오류다. `inf` 를 시한에 심으면 그 런은 영영 안 끝난다 —
    0.0 을 주어 파라미터 하한이 그대로 시한이 되게 한다(짧은 쪽 = 안전측)."""
    assert descent_budget_s(50.0, 3.0, 0.0) == 0.0


def test_deadline_covers_high_latch():
    """🔴 핵심 회귀. 49.6m 래치는 하강만 58.3s 라 60s 시한을 정렬 몇 초로
    넘긴다 — 시한이 하강예산보다 커야 한다."""
    d = precision_land_deadline_s(49.6, 3.0, 0.8, 60.0)
    assert d > descent_budget_s(49.6, 3.0, 0.8), \
        "시한이 이상적 하강시간보다 짧습니다 — 정상 하강 중에 GPS 폴백됩니다"
    assert d == pytest.approx(58.25 * 1.6)


def test_deadline_keeps_parameter_as_floor_at_low_latch():
    """정상 경로(탐색고도 25m 래치)에서는 파라미터 값이 그대로 쓰인다 —
    이 함수는 시한을 **늘리기만** 하고 줄이지 않는다."""
    assert precision_land_deadline_s(25.0, 3.0, 0.8, 60.0) == 60.0


def test_deadline_at_detection_ceiling_fits_measured_descent_rate():
    """F2-a 게이트를 켠 뒤의 최악 래치 고도(=검출 상한)에서도 시한이
    **실측 하강률**(R5: 0.72 m/s 평균)로 완주할 시간을 준다."""
    from fc_bridge.execution.search_pattern import max_detection_altitude_m
    ceiling = max_detection_altitude_m()
    d = precision_land_deadline_s(ceiling, 3.0, 0.8, 60.0)
    measured_s = (ceiling - 3.0) / 0.72        # R5 실측 평균 하강률
    assert d > measured_s, (
        f"시한 {d:.1f}s 가 실측 하강률 기준 소요 {measured_s:.1f}s 보다 짧습니다")


def test_deadline_preserves_disabled_convention():
    """🔴 `precision_land_timeout <= 0` 은 "시한 없음"이다. 배수 항이 살아나면
    비활성이 조용히 활성으로 뒤집혀, 끄려던 시한이 되레 걸린다."""
    assert precision_land_deadline_s(50.0, 3.0, 0.8, 0.0) == 0.0
    assert precision_land_deadline_s(50.0, 3.0, 0.8, -1.0) == -1.0
