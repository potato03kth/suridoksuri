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
