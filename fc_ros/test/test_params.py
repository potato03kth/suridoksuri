"""
작업 A 합격 기준 테스트: flat waypoints reshape + 신규 파라미터 기본값.
rclpy 불필요 — 순수 로직만 검증한다.
"""
import numpy as np
import pytest
import yaml
from pathlib import Path

_YAML_PATH = (
    Path(__file__).parent.parent / "fc_ros" / "params" / "fc_ros_params.yaml"
)


def _load_yaml():
    with open(_YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── flat waypoints reshape ─────────────────────────────────────────────────

def test_flat_waypoints_reshape():
    raw = [0.0, 0.0, 50.0, 100.0, 0.0, 50.0]
    wps = np.array(raw, dtype=float).reshape(-1, 3)
    assert wps.shape == (2, 3)
    assert wps[1, 0] == 100.0   # 북 100 m
    assert wps[0, 2] == 50.0    # 고도 50 m


def test_flat_waypoints_not_nested():
    """YAML에 중첩 리스트([[ ]]) 없이 flat 1D 리스트임을 검증한다."""
    data = _load_yaml()
    offboard_wps = data["offboard_node"]["ros__parameters"]["waypoints"]
    mission_wps  = data["mission_node"]["ros__parameters"]["waypoints"]

    # 값 자체가 숫자여야 함 (중첩 리스트면 list 타입)
    assert all(isinstance(v, (int, float)) for v in offboard_wps), \
        "offboard_node waypoints가 flat 1D 리스트가 아닙니다"
    assert all(isinstance(v, (int, float)) for v in mission_wps), \
        "mission_node waypoints가 flat 1D 리스트가 아닙니다"


def test_waypoints_altitude_50m():
    """운용 고도가 50 m로 통일되어 있음을 검증한다."""
    data = _load_yaml()
    offboard_wps = np.array(
        data["offboard_node"]["ros__parameters"]["waypoints"], dtype=float
    ).reshape(-1, 3)
    mission_wps = np.array(
        data["mission_node"]["ros__parameters"]["waypoints"], dtype=float
    ).reshape(-1, 3)

    assert np.all(offboard_wps[:, 2] == pytest.approx(50.0)), \
        "offboard_node waypoints 고도가 50 m가 아닙니다"
    assert np.all(mission_wps[:, 2] == pytest.approx(50.0)), \
        "mission_node waypoints 고도가 50 m가 아닙니다"


# ── 신규 파라미터 기본값 ───────────────────────────────────────────────────

def test_new_params_present_in_yaml():
    """신규 파라미터 5개가 YAML offboard_node 섹션에 존재하는지 확인한다."""
    data = _load_yaml()
    params = data["offboard_node"]["ros__parameters"]
    required = {
        "transition_alt":  50.0,
        "d_end_thresh":    10.0,
        "landing_timeout": 60.0,
        "v_terminal":      15.2,
        "decel_dist":      80.0,
    }
    for key, expected in required.items():
        assert key in params, f"YAML에 '{key}' 파라미터가 없습니다"
        assert params[key] == pytest.approx(expected), \
            f"'{key}' 기본값이 {expected}여야 합니다, 현재: {params[key]}"


def test_v_terminal_above_stall():
    """v_terminal(15.2) >= 스톨(13.8) × 1.1 = 15.18 조건 확인."""
    data = _load_yaml()
    v_terminal = data["offboard_node"]["ros__parameters"]["v_terminal"]
    stall_speed = 13.8
    assert v_terminal >= stall_speed * 1.1 - 1e-6, \
        f"v_terminal({v_terminal}) < 스톨×1.1({stall_speed * 1.1})"


# ── SITL-7 R1: 상태 타임아웃 4종 + 거리 상한 ──────────────────────────────
# 근거: logs/2026-07-27_sitl_vtol_campaign/campaign_report.md §4 (F-1/F-2/F-3)
# 실측 최대 체류시간은 각 런 metrics.json 의 state_windows_ulog_s 에서 산출.

_R1_DEFAULTS = {
    "climbing_timeout":      120.0,
    "transition_fw_timeout":  90.0,
    "transition_mc_timeout":  30.0,
    "entry_timeout":          60.0,
    "range_limit_m":         300.0,
}

# 실측 최대 체류시간 (s) — 이 값에 오발동하면 정상 미션이 죽는다.
# 캠페인 24런 + R1 검증 8런(logs/2026-07-27_r1_timeouts_range/)의 합집합 최대값.
# CLIMBING은 고도 의존이라 YAML 기본 transition_alt(50.0m) 조건의 최대값을 쓴다.
_MEASURED_MAX_DWELL_S = {
    "climbing_timeout":      31.91,   # A3_pxvehicle (t_alt=50m). R1 최대 29.42(R1_c10_entry)
    "transition_fw_timeout": 20.27,   # R1_c6a. 캠페인 최대는 20.08(A4), 최소 6.81(C2)
    "transition_mc_timeout":  7.87,   # R1_c6b. 캠페인 최대는 7.55(A2), 최소 4.64(B3)
}

# CLIMBING 체류 2점 회귀 (C1a 20m/16.98s, C1b 120m/52.11s):
#   dwell ≈ _CLIMB_FIXED_S + _CLIMB_S_PER_M × AGL   (상승률 ≈ 2.85 m/s)
_CLIMB_S_PER_M = (52.11 - 16.98) / (120.0 - 20.0)
_CLIMB_FIXED_S = 16.98 - 20.0 * _CLIMB_S_PER_M


def test_r1_guard_params_present_with_expected_defaults():
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    for key, expected in _R1_DEFAULTS.items():
        assert key in params, f"YAML에 '{key}' 파라미터가 없습니다"
        assert params[key] == pytest.approx(expected), \
            f"'{key}' 기본값이 {expected}여야 합니다, 현재: {params[key]}"


def test_r1_guards_are_enabled_by_default():
    """⚠️ 기본값이 '종전 동작 보존(0=비활성)'이면 실기체에 아무 효과가 없다.
    이 감시들은 켜져 있어야만 의미가 있으므로 기본값 활성을 고정한다."""
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    for key in _R1_DEFAULTS:
        assert params[key] > 0.0, \
            f"'{key}'가 비활성(<=0)으로 배포되면 안 됩니다: {params[key]}"


@pytest.mark.parametrize("key,measured_max", sorted(_MEASURED_MAX_DWELL_S.items()))
def test_r1_timeouts_have_margin_over_measured_maxima(key, measured_max):
    """정상 미션 오발동 방지 — 캠페인 실측 최대 체류시간 대비 최소 3배 여유."""
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    assert params[key] >= measured_max * 3.0, \
        (f"'{key}'={params[key]}s 가 실측 최대 체류 {measured_max}s 의 3배 미만 — "
         f"정상 미션에서 오발동할 수 있습니다")


def test_r1_climbing_timeout_covers_configured_altitude_with_margin():
    """CLIMBING 체류는 고도에 선형이다(2점 회귀, C1a 20m/16.98s · C1b 120m/52.11s).
    YAML 의 transition_alt 에서 예상되는 체류의 3배 이상이어야 오발동이 없다."""
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    predicted = _CLIMB_FIXED_S + _CLIMB_S_PER_M * params["transition_alt"]
    assert predicted == pytest.approx(27.5, abs=0.5)   # 50m 실측대(25.00~31.91s)와 일치
    assert params["climbing_timeout"] >= predicted * 3.0


def test_r1_climbing_timeout_still_covers_120m_climb():
    """실측 최장 상승(C1b, transition_alt=120m, 52.11s)에도 2배 이상 여유."""
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    assert params["climbing_timeout"] >= 52.11 * 2.0


def test_r1_entry_timeout_covers_full_default_path_at_approach_speed():
    """ENTRY는 성공 사례가 0건이라 실측 상한이 없다 → 설계값에서 유도한다:
    v_approach × entry_timeout 이 기본 경로 전장(300m) 이상이어야 한다."""
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    wps = np.array(params["waypoints"], dtype=float).reshape(-1, 3)
    path_len = float(np.sum(np.linalg.norm(np.diff(wps[:, :2], axis=0), axis=1)))
    assert params["v_approach"] * params["entry_timeout"] >= path_len


def test_r1_entry_timeout_would_have_caught_c10():
    """C10 실측 ENTRY 체류 432.67s (5.85km 이탈)."""
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    assert 432.6743 > params["entry_timeout"]


def test_r1_node_declare_defaults_match_yaml():
    """offboard_node.py 의 declare_parameter 기본값과 YAML 값이 어긋나면
    launch 없이 노드를 띄운 경우와 launch 경유가 서로 다르게 동작한다."""
    import re
    src = (Path(__file__).parent.parent / "fc_ros" / "nodes"
           / "offboard_node.py").read_text(encoding="utf-8")
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    for key, expected in _R1_DEFAULTS.items():
        m = re.search(
            r'declare_parameter\(\s*"%s"\s*,\s*([0-9.]+)\s*\)' % re.escape(key), src)
        assert m, f"offboard_node.py 에 '{key}' declare_parameter 가 없습니다"
        assert float(m.group(1)) == pytest.approx(expected)
        assert float(m.group(1)) == pytest.approx(params[key])


def test_r1_guard_params_exposed_as_launch_args():
    """현장 조정은 YAML 수정이 아니라 launch 인자로 한다(프로젝트 규율).
    launch 파일에 통로가 없으면 그 규율을 지킬 수 없다."""
    launch_src = (Path(__file__).parent.parent / "launch"
                  / "phase2.launch.py").read_text(encoding="utf-8")
    for key in _R1_DEFAULTS:
        assert f'"{key}"' in launch_src, \
            f"phase2.launch.py 에 '{key}' 오버라이드 통로가 없습니다"
