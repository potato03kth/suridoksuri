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


# ── SITL-7 R5: 천이 고도 계단 램프 (F-9) ──────────────────────────────────
# 근거: campaign_report.md §2「고도 계단」·§4 F-9,
#       logs/2026-07-28_flight02/notes.md ② (실기체 재현 −29.6m)

_R5_ALT_SLEW_RATE = 3.0

# PX4 가 실제로 낸 수직속도 (m/s). 램프가 이보다 느리면 램프가 병목이 된다.
_MEASURED_VERTICAL_RATE_MPS = {
    "C1b_descent": 1.8,    # 119.8 → 50m 하강 실측
    "C1a_climb":   1.42,   # 22.25 → 48.96m 를 FOLLOWING 18.98s 에 걸쳐
}


def test_r5_alt_slew_rate_present_with_expected_default():
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    assert "alt_slew_rate" in params, "YAML에 'alt_slew_rate' 파라미터가 없습니다"
    assert params["alt_slew_rate"] == pytest.approx(_R5_ALT_SLEW_RATE)


def test_r5_alt_slew_rate_enabled_by_default():
    """0 이하로 배포하면 실기체에 아무 효과가 없다 — R1 감시와 같은 규율."""
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    assert params["alt_slew_rate"] > 0.0


@pytest.mark.parametrize("label,rate", sorted(_MEASURED_VERTICAL_RATE_MPS.items()))
def test_r5_alt_slew_rate_is_faster_than_airframe(label, rate):
    """램프가 기체 궤적의 병목이 되면 안 된다 — 실측 수직속도보다 빨라야 한다.
    (느리게 잡으면 C1a 처럼 경로 전체를 순항고도 미달로 나는 피해가 재현된다.)"""
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    assert params["alt_slew_rate"] > rate, \
        (f"alt_slew_rate={params['alt_slew_rate']} m/s 가 실측 {label}={rate} m/s "
         f"보다 느립니다 — 램프가 고도 수렴을 늦춰 F-10(종점 포착 실패)을 악화시킵니다")


def test_r5_tick_step_below_harness_jump_threshold():
    """틱당 이동이 하니스 setpoint_jump 임계(3×v_approach×dt)를 넘으면
    계단을 고친 뒤에도 판정이 FAIL 로 남는다."""
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    dt = 1.0 / params["control_hz"]
    step = params["alt_slew_rate"] * dt
    thresh = 3.0 * params["v_approach"] * dt
    assert step < thresh, f"틱당 {step}m 가 임계 {thresh}m 이상입니다"


def test_r5_alt_slew_rate_node_declare_default_matches_yaml():
    import re
    src = (Path(__file__).parent.parent / "fc_ros" / "nodes"
           / "offboard_node.py").read_text(encoding="utf-8")
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    m = re.search(r'declare_parameter\(\s*"alt_slew_rate"\s*,\s*([0-9.]+)\s*\)', src)
    assert m, "offboard_node.py 에 'alt_slew_rate' declare_parameter 가 없습니다"
    assert float(m.group(1)) == pytest.approx(params["alt_slew_rate"])


def test_r5_alt_slew_rate_exposed_as_launch_arg():
    launch_src = (Path(__file__).parent.parent / "launch"
                  / "phase2.launch.py").read_text(encoding="utf-8")
    assert '"alt_slew_rate"' in launch_src


def test_r5_fw_setpoint_sites_all_use_the_ramp():
    """FW 위치 setpoint 를 내는 자리에 `_cruise_alt` 절대값이 남아 있으면
    계단이 사라지는 게 아니라 다음 상태 경계로 옮겨갈 뿐이다.
    MC 분기·HOLD 는 이미 3D 슬루(`_mc_pos_ramp`)라 대상이 아니다."""
    src = (Path(__file__).parent.parent / "fc_ros" / "nodes"
           / "offboard_node.py").read_text(encoding="utf-8")
    offenders = [
        (n, line.strip())
        for n, line in enumerate(src.split("\n"), 1)
        if "self._cruise_alt]" in line and "_mc_pos_ramp" not in line
        and "raw_target" not in line
    ]
    assert not offenders, f"램프를 안 타는 setpoint 고도가 남아 있습니다: {offenders}"


# ── F2 비전 정밀착륙 파라미터 노출 (2026-07-29) ─────────────────────────────
#
# 🔴 2026-07-29 이전에는 이 18개 중 **하나도** YAML/launch 에 없었다. launch 인자
# 위생검사가 미선언 인자에 RuntimeError 를 던지므로
# `phase2.launch.py vision_landing:=true` 는 launch 단계에서 실패했다 —
# 즉 실기체에서 F2 를 켤 방법 자체가 존재하지 않았다.
# 값의 근거는 `docs/fc_precision_land_handoff.md` §8.

_VISION_PARAMS = {
    "vision_landing":          False,   # 🔴 기본 비활성이 계약이다
    "vision_search_alt":        25.0,
    "vision_search_radius":     30.0,
    "vision_search_spacing":     0.0,
    "vision_search_speed":       0.0,
    "vision_search_dwell":       3.0,
    "vision_search_timeout":   120.0,
    "vision_retry_alt":         15.0,
    "vision_retry_radius":      18.0,
    "vision_latch_frames":         3,
    "vision_latch_spread_m":     3.0,
    "vision_stale_timeout":      1.0,
    "vision_link_timeout":       3.0,
    "vision_veto_timeout":      10.0,
    "vision_align_tol":          1.0,
    "vision_descend_speed":      0.8,
    "vision_land_handoff_agl":   3.0,
    "precision_land_timeout":   60.0,
}

#: vision 실측 발행 주파수 (Hz). 🔴 10Hz 가 아니다 — 인수인계 §7.
_VISION_PUBLISH_HZ = 4.4
#: 탐색 1회차 실측 소요 (정렬 ~5s + dwell 3s + 나선 89s).
_SEARCH_PASS1_MEASURED_S = 97.0


@pytest.mark.parametrize("key,expected", sorted(_VISION_PARAMS.items(), key=str))
def test_f2_vision_param_present_in_yaml(key, expected):
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    assert key in params, f"YAML에 '{key}' 파라미터가 없습니다"
    if isinstance(expected, bool):
        assert params[key] is expected
    elif isinstance(expected, int):
        assert params[key] == expected and isinstance(params[key], int)
    else:
        assert params[key] == pytest.approx(expected)


def test_f2_vision_landing_defaults_to_false():
    """🔴 계약: 종전 경로(HOLD→LANDING)가 기본이다. true 로 배포하면
    `sitl_vtol_remediation_plan.md` §4-1 4번("첫 실비행에 미검증 변수 둘 금지")을
    깨뜨린다 — 실기체 FW+OFFBOARD 실적이 아직 0건이다."""
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    assert params["vision_landing"] is False


@pytest.mark.parametrize("key,expected", sorted(_VISION_PARAMS.items(), key=str))
def test_f2_node_declare_default_matches_yaml(key, expected):
    """노드 기본값과 YAML 이 어긋나면 launch 경유와 직접 기동이 다르게 돈다."""
    import re
    src = (Path(__file__).parent.parent / "fc_ros" / "nodes"
           / "offboard_node.py").read_text(encoding="utf-8")
    m = re.search(
        r'declare_parameter\(\s*"%s"\s*,\s*(True|False|[0-9.]+)\s*\)'
        % re.escape(key), src)
    assert m, f"offboard_node.py 에 '{key}' declare_parameter 가 없습니다"
    raw = m.group(1)
    got = {"True": True, "False": False}.get(raw)
    if got is None:
        got = float(raw)
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    if isinstance(expected, bool):
        assert got is expected and params[key] is expected
    else:
        assert got == pytest.approx(float(expected))
        assert got == pytest.approx(float(params[key]))


@pytest.mark.parametrize("key", sorted(_VISION_PARAMS))
def test_f2_vision_param_declared_as_launch_arg(key):
    """현장 조정은 YAML 수정이 아니라 launch 인자로 한다(프로젝트 규율).
    🔴 게다가 위생검사가 **미선언 인자를 거부**하므로, 선언이 없으면 인자를 준
    순간 launch 자체가 실패한다."""
    launch_src = (Path(__file__).parent.parent / "launch"
                  / "phase2.launch.py").read_text(encoding="utf-8")
    assert f'"{key}", default_value=""' in launch_src, \
        f"phase2.launch.py 에 '{key}' DeclareLaunchArgument 가 없습니다"


def _launch_forward_lists():
    """`phase2.launch.py` 의 `_VISION_*_ARGS` 세 목록을 이름별로 돌려준다."""
    import ast
    tree = ast.parse((Path(__file__).parent.parent / "launch"
                      / "phase2.launch.py").read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            name = getattr(t, "id", "")
            if not (name.startswith("_VISION_") and name.endswith("_ARGS")):
                continue
            if isinstance(node.value, (ast.Tuple, ast.List)):
                out[name] = [e.value for e in node.value.elts
                             if isinstance(e, ast.Constant)]
    return out


@pytest.mark.parametrize("key", sorted(_VISION_PARAMS))
def test_f2_vision_param_is_actually_forwarded(key):
    """선언만 하고 `_make_nodes` 의 overrides 에 안 넣으면 **인자를 줘도 무시**된다
    — launch 는 성공하고 값만 조용히 사라지는, 2026-07-28 flight02 형태의 사고다."""
    lists = _launch_forward_lists()
    assert lists, "phase2.launch.py 에서 _VISION_*_ARGS 목록을 찾지 못했습니다"
    forwarded = {k: n for n, keys in lists.items() for k in keys}
    assert key in forwarded, \
        f"'{key}' 가 overrides 로 전달되지 않습니다 (launch 인자가 조용히 무시됩니다)"


@pytest.mark.parametrize("key,expected", sorted(_VISION_PARAMS.items(), key=str))
def test_f2_vision_param_forwarded_with_matching_type(key, expected):
    """🔴 ROS2 파라미터는 타입이 엄격하다 — int 로 선언된 값에 float 을 덮어쓰면
    노드가 `ParameterTypeException` 으로 죽고, bool 을 `float()` 에 넣으면
    launch 가 죽는다. 목록 분류가 YAML 타입과 일치해야 한다."""
    lists = _launch_forward_lists()
    want = ("_VISION_BOOL_ARGS" if isinstance(expected, bool)
            else "_VISION_INT_ARGS" if isinstance(expected, int)
            else "_VISION_FLOAT_ARGS")
    assert key in lists.get(want, []), (
        f"'{key}' 는 {want} 에 있어야 합니다 (현재 분류: "
        f"{[n for n, keys in lists.items() if key in keys]})")


def test_f2_search_timeout_covers_measured_first_pass():
    """🔴 실측 역산값. 90s 로 잡으면 1회차를 **완주 직전에** 자른다(실측 97s)."""
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    assert params["vision_search_timeout"] > _SEARCH_PASS1_MEASURED_S


def test_f2_latch_window_is_meaningful_at_measured_publish_rate():
    """래치 단위는 **프레임**이다. 실측 4.4Hz 에서 창이 0.5s 미만이면 단발
    오탐 차단이라는 목적(§6-1 #5)을 못 하고, dwell(제자리 확인)보다 길면
    제자리에서 잡히는 싼 경로가 무의미해진다."""
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    window_s = params["vision_latch_frames"] / _VISION_PUBLISH_HZ
    assert params["vision_latch_frames"] >= 2, \
        "프레임 1건이면 단발 오탐 하나로 나선을 버린다"
    assert 0.5 <= window_s <= params["vision_search_dwell"], \
        f"래치 창 {window_s:.2f}s 가 범위를 벗어납니다"


def test_f2_stale_timeout_has_margin_over_measured_jitter():
    """실측 프레임 간격 p95 = 0.310s. 0.5s 근처로 잡으면 여유가 1.6배뿐이라
    헛경보가 난다(인수인계 §7)."""
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    assert params["vision_stale_timeout"] >= 3.0 * 0.310


def test_f2_handoff_agl_matches_vision_terminal_contract():
    """🔴 vision 의 `terminal_agl_m`(=`closed_loop_floor_agl_m`)과 같은 숫자여야
    한다 — 계약상 "3m 부터 AUTO.LAND 인계"다."""
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    assert params["vision_land_handoff_agl"] == pytest.approx(3.0)


def test_f2_latch_window_shorter_than_stale_timeout_is_not_required_but_recorded():
    """래치 창(프레임 기준)과 stale 창(시간 기준)은 서로 다른 축이다.
    다만 stale 창보다 래치 창이 훨씬 길면 유도가 끊기는 동안 래치가 영영 안
    서므로, 두 값이 같은 자릿수인지만 확인한다."""
    params = _load_yaml()["offboard_node"]["ros__parameters"]
    window_s = params["vision_latch_frames"] / _VISION_PUBLISH_HZ
    assert window_s <= 3.0 * params["vision_stale_timeout"]
