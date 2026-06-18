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
