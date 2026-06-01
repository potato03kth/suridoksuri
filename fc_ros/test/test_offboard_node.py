"""
OffboardNode 순수 로직 테스트.

rclpy 없이 실행 가능한 수학적 로직만 검증한다.
"""
import numpy as np
import pytest


def _wrap(a: float) -> float:
    """[-π, π] 범위로 각도 정규화."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ── _wrap 단위 테스트 ────────────────────────────────────────

def test_wrap_pi_plus():
    assert _wrap(np.pi + 0.1) == pytest.approx(-np.pi + 0.1, abs=1e-9)


def test_wrap_neg_pi_minus():
    result = _wrap(-np.pi - 0.1)
    assert abs(result) == pytest.approx(np.pi - 0.1, abs=1e-9)


def test_wrap_zero():
    assert _wrap(0.0) == pytest.approx(0.0)


def test_wrap_two_pi():
    assert _wrap(2 * np.pi) == pytest.approx(0.0, abs=1e-9)


# ── ENTRY 조건 로직 테스트 ───────────────────────────────────

def _entry_done(wp0, pos2, yaw, wp0_r=5.0, wp0_htol=0.2) -> bool:
    """OffboardNode._step_entry() 도달 판정 로직 추출."""
    dist = float(np.linalg.norm(wp0 - pos2))
    to_wp0 = wp0 - pos2
    if np.linalg.norm(to_wp0) < 1e-3:
        to_wp0 = np.array([1.0, 0.0])
    to_wp0 /= np.linalg.norm(to_wp0)
    chi_to_wp0 = float(np.arctan2(to_wp0[1], to_wp0[0]))
    heading_err = abs(_wrap(chi_to_wp0 - yaw))
    return dist < wp0_r and heading_err < wp0_htol


def test_entry_done_when_close_and_aligned():
    wp0  = np.array([100.0, 0.0])
    pos2 = np.array([98.0,  0.0])   # dist=2 < 5
    yaw  = 0.0                      # 북쪽 헤딩 = WP0 방향
    assert _entry_done(wp0, pos2, yaw) is True


def test_entry_not_done_too_far():
    wp0  = np.array([100.0, 0.0])
    pos2 = np.array([0.0,   0.0])   # dist=100 >> 5
    yaw  = 0.0
    assert _entry_done(wp0, pos2, yaw) is False


def test_entry_not_done_heading_misaligned():
    wp0  = np.array([100.0, 0.0])
    pos2 = np.array([98.0,  0.0])   # dist=2 < 5
    yaw  = np.pi / 2                # 동쪽 헤딩 (WP0는 북쪽)
    assert _entry_done(wp0, pos2, yaw) is False
