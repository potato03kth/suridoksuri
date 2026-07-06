"""
resolve_planner_name() 단위 테스트 — 기체 타입별 자동 선택 + 명시 우선.
rclpy 불필요, Windows/WSL pytest로 실행 가능.
"""
from fc_bridge.planning.planner_runner import resolve_planner_name


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
