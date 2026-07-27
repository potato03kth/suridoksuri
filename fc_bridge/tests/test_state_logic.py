"""
test_state_logic.py — offboard_node 상태 전환 판정 순수 함수 단위 테스트.
rclpy 불필요, Windows/WSL pytest로 실행 가능.
"""
import math

import pytest

import numpy as np

from fc_bridge.execution.state_logic import (
    climbing_reached, home_amsl_confirmed, home_amsl_sample_fresh,
    is_pilot_takeover, path_origin_ned, translate_path,
    offboard_reacquire_allowed, state_timeout_due,
    horiz_dist_from_origin, range_limit_exceeded, path_max_range,
    after_streaming_state, slew_setpoint,
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


# ── is_pilot_takeover() ─────────────────────────────────────────

@pytest.mark.parametrize("mode", [
    "MANUAL", "POSCTL", "ALTCTL", "STABILIZED", "ACRO", "RATTITUDE",
    "POSITION_SLOW",
])
def test_is_pilot_takeover_manual_modes(mode):
    """조종사가 RC로 쥔 모드는 전부 인계로 판정해야 한다."""
    assert is_pilot_takeover(mode)


@pytest.mark.parametrize("mode", [
    "OFFBOARD", "AUTO.LOITER", "AUTO.LAND", "AUTO.RTL", "AUTO.TAKEOFF",
    "AUTO.MISSION", "",
])
def test_is_pilot_takeover_non_manual_modes(mode):
    """자율/페일세이프 모드는 인계가 아니다 — 재요청이 정당한 구간."""
    assert not is_pilot_takeover(mode)


def test_is_pilot_takeover_case_insensitive():
    assert is_pilot_takeover("posctl")


def test_is_pilot_takeover_2026_07_25_accident_mode():
    """2026-07-25 flight01 실사고: 조종사가 POSCTL로 인계했는데 노드가
    0.9초간 10회 OFFBOARD를 재요청해 도로 뺏어왔다."""
    assert is_pilot_takeover("POSCTL")


# ── path_origin_ned() ───────────────────────────────────────────

def test_path_origin_takeoff_frame_returns_takeoff_position():
    """기본 'takeoff' 프레임은 이륙지점을 그대로 경로 원점으로 쓴다."""
    origin = path_origin_ned([8.53, -6.84, -10.55])
    assert origin == pytest.approx([8.53, -6.84, -10.55])


def test_path_origin_local_frame_is_zero():
    """'local'은 종전 동작 — 원점 이동 없음."""
    assert path_origin_ned([8.53, -6.84, -10.55], "local") == pytest.approx([0.0, 0.0, 0.0])


def test_path_origin_rejects_unknown_frame():
    with pytest.raises(ValueError):
        path_origin_ned([0.0, 0.0, 0.0], "world")


def test_path_origin_does_not_alias_input():
    """반환값을 평행이동에 쓰므로 입력 배열과 메모리를 공유하면 안 된다."""
    src = np.array([1.0, 2.0, 3.0])
    origin = path_origin_ned(src)
    origin[0] = 99.0
    assert src[0] == pytest.approx(1.0)


def test_path_origin_2026_07_25_flight01_geometry():
    """실사고 수치 재현: waypoints [0,0,3]을 이륙지점 기준으로 옮기면
    '10.9m 옆 + 13.55m AGL'이 아니라 '제자리 + 3m AGL'이 된다."""
    takeoff = np.array([8.53, -6.84, -10.55])   # 실측 (N, E, h_up)
    origin = path_origin_ned(takeoff)
    wp0 = np.array([0.0, 0.0]) + origin[:2]
    cruise_alt = 3.0 + origin[2]

    # 수평: 이륙지점과 일치 (종전엔 10.94m 떨어진 점이었다)
    assert float(np.linalg.norm(wp0 - takeoff[:2])) == pytest.approx(0.0, abs=1e-9)
    # 고도: 지면 위 3m (종전엔 h_up=+3.0 = 지면 위 13.55m였다)
    assert cruise_alt - takeoff[2] == pytest.approx(3.0)
    assert cruise_alt == pytest.approx(-7.55)


# ── translate_path() ────────────────────────────────────────────

def test_translate_path_shifts_all_components():
    pts, mc_wps, alt = translate_path(
        [[0.0, 0.0], [1.0, 2.0]], [[0.0, 0.0], [-4.24, 4.24]], 3.0,
        [8.53, -6.84, -10.55])
    assert pts == pytest.approx(np.array([[8.53, -6.84], [9.53, -4.84]]))
    assert mc_wps == pytest.approx(np.array([[8.53, -6.84], [4.29, -2.60]]))
    assert alt == pytest.approx(-7.55)


def test_translate_path_zero_origin_is_identity():
    """'local' 프레임(원점 0)에서는 종전 동작과 완전히 동일해야 한다."""
    pts0 = [[0.0, 0.0], [1.0, 2.0]]
    wps0 = [[0.0, 0.0], [-4.24, 4.24]]
    pts, mc_wps, alt = translate_path(pts0, wps0, 3.0, [0.0, 0.0, 0.0])
    assert pts == pytest.approx(np.array(pts0))
    assert mc_wps == pytest.approx(np.array(wps0))
    assert alt == pytest.approx(3.0)


def test_translate_path_preserves_shape_and_heading():
    """평행이동이므로 구간 벡터(모양·길이·진행방향)는 변하지 않는다."""
    pts0 = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 9.0]])
    pts, _, _ = translate_path(pts0, [[0.0, 0.0]], 3.0, [100.0, -50.0, 7.0])
    assert np.diff(pts, axis=0) == pytest.approx(np.diff(pts0, axis=0))


def test_translate_path_does_not_mutate_input():
    pts0 = np.array([[0.0, 0.0], [1.0, 1.0]])
    wps0 = np.array([[0.0, 0.0]])
    translate_path(pts0, wps0, 3.0, [5.0, 5.0, 5.0])
    assert pts0 == pytest.approx(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert wps0 == pytest.approx(np.array([[0.0, 0.0]]))


def test_translate_path_rejects_short_origin():
    with pytest.raises(ValueError):
        translate_path([[0.0, 0.0]], [[0.0, 0.0]], 3.0, [1.0, 2.0])


def test_translate_path_2026_07_25_flight04_altitude():
    """flight04 실측: 지면 h_up=-2.09에서 3m를 요구했는데 종전 코드는
    h_up=+3.00(=5.09m AGL)을 발행했다. 보정 후엔 h_up=+0.91이어야 한다."""
    _, _, alt = translate_path([[0.0, 0.0]], [[0.0, 0.0]], 3.0,
                               [-3.36, 0.087, -2.09])
    assert alt == pytest.approx(0.91)
    assert alt - (-2.09) == pytest.approx(3.0)   # 지면 기준 정확히 3m


# ── home_amsl_confirmed() NaN 방어 (2026-07-25 감사) ────────────

def test_home_amsl_confirmed_rejects_all_nan():
    """PX4 ALTITUDE(#141)는 z_global·air_data 둘 다 무효면 amsl=NaN을 보낸다."""
    nan = float("nan")
    assert home_amsl_confirmed([nan, nan, nan]) is None


def test_home_amsl_confirmed_rejects_single_nan():
    """NaN은 모든 비교가 False라 수렴검사를 통째로 무력화했다 —
    종전 코드는 [19.2, nan, 19.3]에 대해 19.3을 '수렴했다'며 반환했다."""
    nan = float("nan")
    assert home_amsl_confirmed([19.2, nan, 19.3], tol=0.5, min_samples=3) is None


def test_home_amsl_confirmed_rejects_inf():
    assert home_amsl_confirmed([19.2, float("inf"), 19.3]) is None


def test_home_amsl_confirmed_nan_would_have_made_takeoff_altitude_nan():
    """NaN이 확정되면 CommandTOL.altitude=NaN이 나가고, PX4는 param7이 유한하지
    않으면 MIS_TAKEOFF_ALT로 폴백한다 → 지령한 고도와 다른 고도로 이륙."""
    from fc_bridge.execution.state_logic import takeoff_request_fields
    assert home_amsl_confirmed([float("nan")] * 3) is None
    # 방어가 없다면 이렇게 됐다는 것을 명시적으로 고정
    assert math.isnan(takeoff_request_fields(3.0, float("nan"))["altitude"])


def test_translate_path_empty_mc_wps_does_not_crash():
    """_mc_wps가 빈 리스트면 np.asarray가 shape (0,)를 만들어 (2,)와 broadcast
    실패한다. 이 함수는 이륙 순간에 호출되므로 여기서 죽으면 최악이다."""
    pts, mc_wps, alt = translate_path(
        np.zeros((3, 2)), [], 3.0, [1.0, 2.0, 3.0])
    assert mc_wps.shape == (0, 2)
    assert alt == pytest.approx(6.0)


# ── offboard_reacquire_allowed() — F-15 수정의 안전 가드 ─────────

def test_reacquire_not_needed_when_already_offboard():
    assert not offboard_reacquire_allowed("OFFBOARD")


@pytest.mark.parametrize("mode", ["AUTO.LOITER", "AUTO.RTL", "AUTO.TAKEOFF",
                                  "AUTO.MISSION", "HOLD", ""])
def test_reacquire_allowed_in_autonomous_modes(mode):
    """자율/페일세이프 모드거나 아직 모드를 못 받은 상태면 재요청이 정당하다 —
    F-15(정렬 구간 OFFBOARD 이탈)에서 복구해야 하는 경로가 바로 이것이다."""
    assert offboard_reacquire_allowed(mode)


@pytest.mark.parametrize("mode", ["MANUAL", "POSCTL", "ALTCTL", "STABILIZED",
                                  "ACRO", "RATTITUDE", "POSITION_SLOW"])
def test_reacquire_forbidden_in_pilot_modes(mode):
    """2026-07-25 flight01 회귀 방지: 조종사가 쥔 모드에서는 재요청 금지.
    F-15 수정으로 재요청 경로를 새로 여는 만큼 이 가드가 핵심이다."""
    assert not offboard_reacquire_allowed(mode)


def test_reacquire_forbidden_case_insensitive():
    assert not offboard_reacquire_allowed("posctl")


def test_reacquire_is_strict_complement_of_pilot_takeover():
    """OFFBOARD가 아닌 모든 모드에 대해 재요청 허용 == 조종사 인계 아님."""
    for mode in ("MANUAL", "POSCTL", "AUTO.LOITER", "AUTO.RTL", "ALTCTL", ""):
        assert offboard_reacquire_allowed(mode) == (not is_pilot_takeover(mode))


# ── state_timeout_due() — 타임아웃 4종 공통 판정 ─────────────────

def test_state_timeout_strictly_greater():
    """경계는 strict > — hold_timeout/mc_wp_timeout/landing_timeout 규약과 동일."""
    assert not state_timeout_due(30.0, 30.0)
    assert state_timeout_due(30.1, 30.0)


def test_state_timeout_disabled_when_zero_or_negative():
    """0 이하는 비활성 = 종전 무한대기 동작으로 되돌리는 탈출구."""
    assert not state_timeout_due(1e9, 0.0)
    assert not state_timeout_due(1e9, -1.0)


def test_state_timeout_c10_entry_would_have_fired():
    """C10 실측: ENTRY 체류 432.67s (5.85km 이탈). 기본값 60.0s면 잡힌다.
    근거: logs/2026-07-27_sitl_vtol_campaign/C10_pxvehicle/metrics.json
          completion.state_timeline[-1].dwell_s = 432.6743"""
    assert state_timeout_due(432.6743, 60.0)


@pytest.mark.parametrize("elapsed,timeout", [
    # 캠페인 실측 최대 체류 vs 신설 기본값 — 정상 미션에서 오발동하면 안 된다.
    (31.91, 120.0),   # CLIMBING 최대 (A3_pxvehicle, transition_alt=50)
    (52.11, 120.0),   # CLIMBING 최대 (C1b, transition_alt=120)
    (20.08,  90.0),   # TRANSITION_FW 최대 (A4)
    (7.55,   30.0),   # TRANSITION_MC 최대 (A2)
])
def test_state_timeout_does_not_fire_on_measured_maxima(elapsed, timeout):
    assert not state_timeout_due(elapsed, timeout)


# ── 거리 상한 ───────────────────────────────────────────────────

def test_horiz_dist_ignores_altitude():
    assert horiz_dist_from_origin([3.0, 4.0, 999.0], [0.0, 0.0, -50.0]) == pytest.approx(5.0)


def test_horiz_dist_relative_to_takeoff_not_local_origin():
    """기준점은 EKF 로컬 원점이 아니라 이륙지점이다 — 2026-07-25 flight01에서
    둘이 수평 10.94m 어긋나 있었다."""
    takeoff = [8.53, -6.84, -10.55]
    assert horiz_dist_from_origin(takeoff, takeoff) == pytest.approx(0.0)
    assert horiz_dist_from_origin([8.53, -6.84], [0.0, 0.0]) == pytest.approx(10.94, abs=0.01)


def test_range_limit_strictly_greater():
    assert not range_limit_exceeded(300.0, 300.0)
    assert range_limit_exceeded(300.1, 300.0)


def test_range_limit_disabled_when_zero_or_negative():
    assert not range_limit_exceeded(1e6, 0.0)
    assert not range_limit_exceeded(1e6, -1.0)


def test_range_limit_nan_is_not_a_breach():
    """NaN 표본으로 폴백이 발동하면 안 된다(안전측: 모르면 발동 안 함)."""
    assert not range_limit_exceeded(float("nan"), 300.0)


def test_range_limit_c10_excursion_would_have_fired():
    """C10 실측 이탈 5.85km (N=5038, E=-2979). 기본 상한 300m이면 잡힌다."""
    d = horiz_dist_from_origin([5038.0, -2979.0, 50.0], [0.0, 0.0, 0.0])
    assert d == pytest.approx(5852.0, abs=2.0)
    assert range_limit_exceeded(d, 300.0)


def test_range_limit_c7_harness_precedent():
    """하니스 RangeGuard가 C7에서 1564m에 발동한 전례 — 노드 상한(300m)이라면
    훨씬 일찍 잡았을 것이다."""
    assert range_limit_exceeded(1564.0, 1500.0)
    assert range_limit_exceeded(1564.0, 300.0)


# ── path_max_range() — 이륙 전 경로/상한 정합성 진단 ─────────────

def test_path_max_range_picks_farthest_point():
    pts = np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 200.0]])
    assert path_max_range(pts, [0.0, 0.0, 0.0]) == pytest.approx(282.84, abs=0.01)


def test_path_max_range_relative_to_takeoff():
    pts = np.array([[100.0, 0.0], [130.0, 0.0]])
    assert path_max_range(pts, [100.0, 0.0, 0.0]) == pytest.approx(30.0)


def test_path_max_range_empty_path_is_zero():
    assert path_max_range([], [0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_path_max_range_default_yaml_path_exceeds_default_limit():
    """yaml 기본 waypoints(정북 300m)는 기본 상한 300.0m와 맞물린다 —
    이륙 직전 경고가 나가야 하는 조건이며, 편도 300m 초과 경로를 시험할 땐
    range_limit_m 을 함께 키워야 한다는 근거."""
    pts = np.array([[0.0, 0.0], [300.0, 0.0]])
    far = path_max_range(pts, [0.0, 0.0, 0.0])
    assert far == pytest.approx(300.0)
    # 경로점 자체는 경계값이라 미발동이지만, 역천이 오버슈트(≈57-d_end_thresh,
    # 기본 10 → +46.8m)가 붙으면 실제 비행 최원점은 상한을 넘는다.
    assert not range_limit_exceeded(far, 300.0)
    assert range_limit_exceeded(far + 46.8, 300.0)


# ── after_streaming_state() — F-14 (2026-07-27 SITL-7 R2) ────────

def test_after_streaming_state_mid_flight_goes_to_entry():
    assert after_streaming_state("mid_flight") == "entry"


def test_after_streaming_state_default_goes_to_following():
    assert after_streaming_state("pre_takeoff") == "following"
    assert after_streaming_state("") == "following"
    assert after_streaming_state(None) == "following"


# ── slew_setpoint() — F-6 (2026-07-27 SITL-7 R2) ────────────────

def test_slew_setpoint_limits_step():
    out = slew_setpoint([0.0, 0.0, 50.0], [100.0, 0.0, 50.0], 0.5)
    assert out.tolist() == pytest.approx([0.5, 0.0, 50.0])


def test_slew_setpoint_snaps_when_within_step():
    out = slew_setpoint([0.0, 0.0, 50.0], [0.3, 0.0, 50.0], 0.5)
    assert out.tolist() == pytest.approx([0.3, 0.0, 50.0])


def test_slew_setpoint_moves_in_3d_including_altitude():
    out = slew_setpoint([0.0, 0.0, 0.0], [0.0, 0.0, 100.0], 2.0)
    assert out.tolist() == pytest.approx([0.0, 0.0, 2.0])


def test_slew_setpoint_zero_step_snaps_to_target():
    """max_step<=0 은 제한 없음(종전 동작)과 같다 — 램프 비활성 경로."""
    out = slew_setpoint([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], 0.0)
    assert out.tolist() == pytest.approx([10.0, 0.0, 0.0])


def test_slew_setpoint_does_not_mutate_input():
    cur = np.array([0.0, 0.0, 0.0])
    slew_setpoint(cur, [100.0, 0.0, 0.0], 1.0)
    assert cur.tolist() == [0.0, 0.0, 0.0]


def test_slew_setpoint_hold_entry_step_is_bounded_by_rate():
    """HOLD 진입 계단(캠페인 실측 92~144m)이 v_approach*dt 로 제한되는지.

    램프 시작점을 기체 현재 위치로 잡으므로(offboard_node._step_hold),
    끝점이 47m 떨어져 있어도 첫 틱 이동은 0.5m 다.
    """
    pos = np.array([347.0, 0.0, 50.0])      # 종점을 47m 지나친 지점
    wp1 = np.array([300.0, 0.0, 50.0])
    out = slew_setpoint(pos, wp1, 5.0 * 0.1)
    assert float(np.linalg.norm(out - pos)) == pytest.approx(0.5, abs=1e-9)
