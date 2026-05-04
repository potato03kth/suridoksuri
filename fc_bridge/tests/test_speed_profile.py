"""
speed_profile 단위 테스트.
"""
import numpy as np
import pytest
from fc_bridge.planning.speed_profile import compute_speed_profile


def test_zero_curvature_returns_cruise():
    """곡률 0 → 모든 점에서 v_cruise."""
    kappa = np.zeros(50)
    v = compute_speed_profile(kappa, v_cruise=15.0, a_max=3.0)
    assert np.all(v == pytest.approx(15.0))


def test_high_curvature_limits_speed():
    """곡률이 크면 속도가 v_cruise 미만으로 제한된다."""
    # v_lim = sqrt(a_max / kappa) = sqrt(3.0 / 0.3) = sqrt(10) ≈ 3.16
    kappa = np.full(20, 0.3)
    v = compute_speed_profile(kappa, v_cruise=15.0, a_max=3.0)
    assert np.all(v < 15.0)
    assert np.all(v == pytest.approx(np.sqrt(3.0 / 0.3), rel=0.01))


def test_never_exceeds_cruise():
    """어떤 경우에도 v_cruise 초과 없음."""
    rng = np.random.default_rng(0)
    kappa = rng.uniform(0.0, 0.5, 200)
    v = compute_speed_profile(kappa, v_cruise=12.0, a_max=2.5)
    assert np.all(v <= 12.0 + 1e-9)


def test_never_below_minimum():
    """최솟값 0.5 m/s 보장."""
    kappa = np.full(10, 1000.0)   # 극단적 곡률
    v = compute_speed_profile(kappa, v_cruise=15.0, a_max=3.0)
    assert np.all(v >= 0.5 - 1e-9)


def test_smoothing_propagates_backward():
    """고곡률 직전 구간에서 속도가 미리 감소한다 (역방향 smoothing)."""
    # 0~9: 직선(κ=0), 10~14: 고곡률(κ=0.3)
    kappa = np.zeros(20)
    kappa[10:15] = 0.3
    v = compute_speed_profile(kappa, v_cruise=15.0, a_max=3.0, smooth_window=5)
    # 인덱스 5~9는 고곡률 구간보다 5칸 앞 → smoothing 영향을 받아야 함
    v_high_kappa = np.sqrt(3.0 / 0.3)
    assert v[9] <= v_high_kappa + 1e-6   # 직전 점은 감속
    assert v[0] == pytest.approx(15.0)   # 충분히 멀리는 cruise 속도


def test_output_length_matches_input():
    """출력 배열 길이 = 입력 배열 길이."""
    kappa = np.linspace(0, 0.2, 73)
    v = compute_speed_profile(kappa, v_cruise=10.0, a_max=2.0)
    assert len(v) == 73
