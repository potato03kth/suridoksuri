"""
test_state_logic.py — offboard_node 상태 전환 판정 순수 함수 단위 테스트.
rclpy 불필요, Windows/WSL pytest로 실행 가능.
"""
import pytest

from fc_bridge.execution.state_logic import (
    climbing_reached, home_amsl_confirmed, home_amsl_sample_fresh,
)


# ── climbing_reached() ──────────────────────────────────────────

def test_climbing_reached_position_only_default_ignores_velocity():
    """기존 호출 형태(vz 인자 없음)는 속도와 무관하게 위치만으로 판정 — 회귀 없음."""
    assert climbing_reached(3.0, transition_alt=3.0)
    assert not climbing_reached(1.0, transition_alt=3.0)


def test_climbing_reached_true_when_settled():
    """목표고도 근방 + 수직속도 안정 시 도달로 판정."""
    assert climbing_reached(3.1, transition_alt=3.0, vz_down=0.05, vz_tol=0.3)


def test_climbing_reached_false_while_still_ascending():
    """2026-07-24 SITL 재현 시나리오: AGL은 목표를 스쳤지만 여전히 빠르게
    상승 중(vz_down=-1.3, NED 부호=하강 양수이므로 음수=상승)이면 미도달로
    판정해야 한다 — 이전에는 위치조건만으로 여기서 True가 나와 STREAMING으로
    조기 전환됐었다."""
    assert not climbing_reached(3.0, transition_alt=3.0, vz_down=-1.3, vz_tol=0.3)


def test_climbing_reached_position_ok_but_off_target():
    """수직속도가 안정돼도 위치조건 자체를 못 만족하면 미도달."""
    assert not climbing_reached(1.0, transition_alt=3.0, vz_down=0.0, vz_tol=0.3)


def test_climbing_reached_vz_boundary_inclusive():
    assert climbing_reached(3.0, transition_alt=3.0, vz_down=0.3, vz_tol=0.3)
    assert not climbing_reached(3.0, transition_alt=3.0, vz_down=0.31, vz_tol=0.3)


# ── home_amsl_confirmed() ───────────────────────────────────────

def test_home_amsl_confirmed_needs_min_samples():
    assert home_amsl_confirmed([10.0, 10.1], min_samples=3) is None


def test_home_amsl_confirmed_converges():
    assert home_amsl_confirmed([10.0, 10.1, 10.05], tol=0.5, min_samples=3) == pytest.approx(10.05)


def test_home_amsl_confirmed_rejects_spread():
    assert home_amsl_confirmed([10.0, 15.0, 10.2], tol=0.5, min_samples=3) is None


# ── home_amsl_sample_fresh() ────────────────────────────────────

def test_home_amsl_sample_fresh_within_age():
    assert home_amsl_sample_fresh(0.2, max_age_s=1.0)
    assert home_amsl_sample_fresh(1.0, max_age_s=1.0)  # 경계값 포함


def test_home_amsl_sample_fresh_rejects_stale():
    """2026-07-24 SITL 재현 시나리오: 이전 PX4 인스턴스에서 온 오래된
    home_position(예: 수 분 지연)은 신선하지 않다고 판정해야 한다."""
    assert not home_amsl_sample_fresh(65.0, max_age_s=1.0)
