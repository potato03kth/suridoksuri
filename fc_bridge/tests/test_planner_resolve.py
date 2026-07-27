"""
resolve_planner_name() 단위 테스트 — 기체 타입별 자동 선택 + 명시 우선.
rclpy 불필요, Windows/WSL pytest로 실행 가능.
"""
from fc_bridge.planning.planner_runner import (
    resolve_planner_name, planner_eta_s,
)


def test_auto_mc_selects_straight():
    assert resolve_planner_name("auto", "mc") == "straight"


def test_auto_vtol_selects_eta3():
    assert resolve_planner_name("auto", "vtol") == "eta3"


def test_empty_or_none_defaults_by_vehicle():
    assert resolve_planner_name("", "mc") == "straight"
    assert resolve_planner_name(None, "vtol") == "eta3"


def test_explicit_overrides_vehicle_type():
    # 명시 지정은 기체 타입과 무관하게 그대로 (파라미터 지정 시 지정대로)
    assert resolve_planner_name("eta3", "mc") == "eta3"
    assert resolve_planner_name("straight", "vtol") == "straight"
    assert resolve_planner_name("diterpin", "mc") == "diterpin"


def test_case_and_space_insensitive():
    assert resolve_planner_name(" AUTO ", "MC") == "straight"
    assert resolve_planner_name("Eta3", "vtol") == "eta3"


def test_non_mc_vehicle_falls_back_to_eta3():
    # mc가 아니면(빈 값·미지정 포함) 기존 vtol 기본(eta3) 유지
    assert resolve_planner_name("auto", "") == "eta3"
    assert resolve_planner_name("auto", "fixedwing") == "eta3"


# ── planner_eta_s() — F-12 블로킹 예상시간 (2026-07-27 SITL-7 R2) ──

def test_planner_eta_zero_for_two_waypoints():
    """2WP 는 실측상 즉시 반환 — 대기 안내를 띄울 이유가 없다."""
    assert planner_eta_s("eta3", 2) == 0.0
    assert planner_eta_s("eta3", 1) == 0.0


def test_planner_eta_zero_for_straight_planner():
    """MC 기본 플래너(straight)는 블로킹이 문제가 된 적이 없다."""
    assert planner_eta_s("straight", 5) == 0.0


def test_planner_eta_scales_with_corner_count():
    """실측: 3WP(코너 1개) 45~73s · 5WP 폐곡선(코너 3개) 263.5s."""
    one = planner_eta_s("eta3", 3)
    three = planner_eta_s("eta3", 5)
    assert 45.0 <= one <= 100.0
    assert three == 3 * one
    # 폐곡선 실측 263.5s 를 같은 자릿수로 예고해야 안내로서 쓸모가 있다
    assert 0.5 * 263.5 <= three <= 1.5 * 263.5
