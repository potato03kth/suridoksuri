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
