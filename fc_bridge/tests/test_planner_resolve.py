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
    assert 45.0 <= one <= 120.0
    assert three == 3 * one
    # 폐곡선 실측 263.5s 를 같은 자릿수로 예고해야 안내로서 쓸모가 있다
    assert 0.5 * 263.5 <= three <= 1.5 * 263.5


# ── 예고는 실측보다 항상 커야 한다 (2026-07-28, v_cruise 18.0 정식값 확정) ──
#
# 이 로그의 유일한 존재 이유가 "죽은 게 아니라 계산 중"을 알리는 것이므로
# (F-12), 예고가 실측보다 작으면 정확히 그 오판을 유발한다.
# 아래 실측은 정식값 v_cruise=18.0, PX4 /root/PX4-vehicle @ c890d9db0a,
# 원본 logs/2026-07-28_vc18/<run>/node.log 의 "경로 계획 완료 — 소요 Xs".
_MEASURED_BLOCKING_S = {
    ("R2_a3_vc18", 3): 57.6,       # L자 90° 1회, 코너 1개
    ("R2_b5_vc18", 5): 221.6,      # 사각 폐곡선, 코너 3개
    ("R2_closed_vc18", 5): 223.6,  # 완전 폐회로(대회 경로 형상), 코너 3개
}


def test_planner_eta_never_underestimates_measured_blocking():
    """v_cruise=18.0 실측 전건에서 예고 ≥ 실측 이어야 한다.

    코너당 75s 였을 때 코너3 예고 225s 가 실측 223.6s 를 **1.4초 차이로**
    겨우 넘겼다 — 사실상 마진이 없었다. 90s 로 올려 여유를 되찾는다.
    """
    for (run, n_wp), measured in _MEASURED_BLOCKING_S.items():
        eta = planner_eta_s("eta3", n_wp)
        assert eta >= measured, f"{run}: 예고 {eta}s < 실측 {measured}s"


def test_planner_eta_keeps_useful_margin_over_measurement():
    """다만 여유가 10% 미만이면 런간 변동에 바로 뒤집힌다 — 최소 15% 확보."""
    for (run, n_wp), measured in _MEASURED_BLOCKING_S.items():
        eta = planner_eta_s("eta3", n_wp)
        assert eta >= measured * 1.15, (
            f"{run}: 예고 {eta}s 가 실측 {measured}s 대비 15% 여유 미달")
