"""
test_terminal_decel.py — apply_terminal_decel() 단위 테스트.
rclpy 불필요, Windows pytest로 실행 가능.
"""
import numpy as np
import pytest

from fc_bridge.planning.terminal_decel import apply_terminal_decel


def test_endpoint_reaches_v_terminal():
    """끝점 속도가 v_terminal이어야 한다."""
    s = np.linspace(0, 200, 201)
    v = np.full_like(s, 15.0)
    out = apply_terminal_decel(v, s, v_terminal=12.0, decel_dist=80.0)
    assert out[-1] == pytest.approx(12.0)


def test_outside_decel_zone_unchanged():
    """decel_dist 밖 구간은 원본 속도와 동일해야 한다."""
    s = np.linspace(0, 200, 201)
    v = np.full_like(s, 15.0)
    out = apply_terminal_decel(v, s, v_terminal=12.0, decel_dist=80.0)
    assert out[0] == pytest.approx(15.0)


def test_monotone_decreasing_in_zone():
    """감속 구간(끝 80m)이 단조 비증가해야 한다."""
    s = np.linspace(0, 200, 201)
    v = np.full_like(s, 15.0)
    out = apply_terminal_decel(v, s, v_terminal=12.0, decel_dist=80.0)
    zone = out[s >= (s[-1] - 80.0)]
    assert np.all(np.diff(zone) <= 1e-9)


def test_v_terminal_above_v_cruise_no_change():
    """v_terminal ≥ v_cruise 이면 속도 프로파일이 변하지 않아야 한다."""
    s = np.linspace(0, 200, 201)
    v = np.full_like(s, 15.0)
    out = apply_terminal_decel(v, s, v_terminal=16.0, decel_dist=80.0)
    np.testing.assert_array_almost_equal(out, v)


def test_short_path_decel_dist_covers_all():
    """decel_dist가 경로 전체보다 길면 처음부터 감속해야 한다."""
    s = np.linspace(0, 50, 51)
    v = np.full_like(s, 20.0)
    out = apply_terminal_decel(v, s, v_terminal=10.0, decel_dist=200.0)
    assert out[-1] == pytest.approx(10.0)
    assert out[0] == pytest.approx(10.0 + (20.0 - 10.0) * (50.0 / 200.0))
