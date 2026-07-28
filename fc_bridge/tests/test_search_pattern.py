"""
test_search_pattern.py — 비전 착륙 탐색 비행 기하 순수 함수 단위 테스트.
rclpy/ROS 불필요, Windows/WSL pytest로 실행 가능.
"""
import math

import pytest

import numpy as np

from fc_bridge.execution.search_pattern import (
    DEFAULT_HALF_HFOV_TAN, DEFAULT_HALF_VFOV_TAN,
    OPTIMISTIC_HALF_HFOV_TAN, OPTIMISTIC_HALF_VFOV_TAN,
    DETECTION_HALF_HFOV_TAN,
    DISTRESS_MAT_SIZE_M, COARSE_MIN_AREA_PX, COARSE_FRAME_WIDTH_PX,
    SpiralState, camera_footprint_m, ring_spacing_from_footprint,
    max_turn_speed, spiral_advance, spiral_point_ne, spiral_complete,
    detection_altitude_ok, max_detection_altitude_m,
)


# ── 화각 상수 자체 ──────────────────────────────────────────────

def test_half_tan_constants_match_their_derivations():
    """상수가 문서화된 유도식(75°=수평 / 75°=대각·16:9 분해)의 반올림값인지.

    누가 값을 손으로 고치면 red — 이 파일의 모든 숫자가 여기 걸려 있다.
    """
    tan375 = math.tan(math.radians(37.5))
    diag = math.sqrt(16.0 ** 2 + 9.0 ** 2)          # √337
    assert OPTIMISTIC_HALF_HFOV_TAN == pytest.approx(tan375, abs=1e-5)
    assert OPTIMISTIC_HALF_VFOV_TAN == pytest.approx(tan375 * 9 / 16, abs=1e-5)
    assert DEFAULT_HALF_HFOV_TAN == pytest.approx(tan375 * 16 / diag, abs=1e-4)
    assert DEFAULT_HALF_VFOV_TAN == pytest.approx(tan375 * 9 / diag, abs=1e-4)


def test_conservative_constants_are_not_larger_than_exact_derivation():
    """반올림은 반드시 안전측(작은 쪽)이어야 한다 — 크면 커버리지 과대평가."""
    tan375 = math.tan(math.radians(37.5))
    diag = math.sqrt(337.0)
    assert DEFAULT_HALF_HFOV_TAN <= tan375 * 16 / diag
    assert DEFAULT_HALF_VFOV_TAN <= tan375 * 9 / diag


# ── camera_footprint_m() ────────────────────────────────────────

def test_camera_footprint_scales_linearly_with_agl():
    w10, h10 = camera_footprint_m(10.0)
    w20, h20 = camera_footprint_m(20.0)
    assert w20 == pytest.approx(2.0 * w10)
    assert h20 == pytest.approx(2.0 * h10)


def test_camera_footprint_conservative_default_matches_documented_formula():
    """보수 기본값 = 1.3375·h × 0.7524·h (파일 docstring 표와 대조)."""
    w, h = camera_footprint_m(1.0)
    assert w == pytest.approx(1.33752, abs=1e-5)
    assert h == pytest.approx(0.75236, abs=1e-5)


def test_camera_footprint_optimistic_matches_documented_formula():
    """낙관(75°=수평) = 1.5347·h × 0.8632·h."""
    w, h = camera_footprint_m(1.0, OPTIMISTIC_HALF_HFOV_TAN,
                              OPTIMISTIC_HALF_VFOV_TAN)
    assert w == pytest.approx(1.53466, abs=1e-5)
    assert h == pytest.approx(0.86324, abs=1e-5)


def test_camera_footprint_conservative_is_smaller_than_optimistic():
    """🔴 핵심 계약: 기본값(보수)이 낙관보다 **작아야** 한다.

    두 상수 세트를 뒤바꿔 넣으면 커버리지를 과대평가해 나선 링 사이에 탐색
    구멍이 생긴다 — 그 실수를 잡는 그물이 이것이다.
    """
    cons_w, cons_h = camera_footprint_m(25.0)
    opt_w, opt_h = camera_footprint_m(25.0, OPTIMISTIC_HALF_HFOV_TAN,
                                      OPTIMISTIC_HALF_VFOV_TAN)
    assert cons_w < opt_w
    assert cons_h < opt_h
    # 대각 해석은 수평 해석보다 정확히 16/√337 배 좁다.
    assert cons_w / opt_w == pytest.approx(16.0 / math.sqrt(337.0), abs=1e-4)


def test_camera_footprint_keeps_16_by_9_aspect():
    """어느 해석을 쓰든 프레임 종횡비는 16:9다(분해가 축을 안 섞었는지 확인)."""
    for ht, vt in ((DEFAULT_HALF_HFOV_TAN, DEFAULT_HALF_VFOV_TAN),
                   (OPTIMISTIC_HALF_HFOV_TAN, OPTIMISTIC_HALF_VFOV_TAN)):
        w, h = camera_footprint_m(30.0, ht, vt)
        assert w / h == pytest.approx(16.0 / 9.0, rel=1e-4)


def test_camera_footprint_short_side_is_the_vertical_one():
    """16:9 나디르에서 짧은 변은 세로다 — ring_spacing 에 넣을 값의 근거."""
    w, h = camera_footprint_m(25.0)
    assert h < w


# ── ring_spacing_from_footprint() ───────────────────────────────

def test_ring_spacing_default_overlap_is_65_percent_of_footprint():
    assert ring_spacing_from_footprint(10.0) == pytest.approx(6.5)


def test_ring_spacing_is_strictly_smaller_than_footprint():
    """🔴 겹침은 간격을 **줄여야** 한다.

    부호가 뒤집혀 1+overlap 이 되면 spacing > footprint 가 되어 이웃 링 사이에
    카메라가 한 번도 보지 않은 띠가 남는다(탐색 구멍).
    """
    for short in (2.0, 7.5, 18.8):
        for overlap in (0.0, 0.2, 0.35, 0.6):
            spacing = ring_spacing_from_footprint(short, overlap)
            assert spacing <= short
            if overlap > 0.0:
                assert spacing < short


def test_ring_spacing_zero_overlap_equals_footprint():
    assert ring_spacing_from_footprint(9.0, 0.0) == pytest.approx(9.0)


def test_ring_spacing_rejects_out_of_range_overlap():
    with pytest.raises(ValueError):
        ring_spacing_from_footprint(10.0, 1.0)
    with pytest.raises(ValueError):
        ring_spacing_from_footprint(10.0, -0.1)


def test_ring_spacing_at_25m_search_altitude():
    """실사용 조합 실측: 25m 보수 풋프린트의 짧은 변 → 링 간격."""
    w, h = camera_footprint_m(25.0)
    assert h == pytest.approx(18.809, abs=1e-3)
    assert ring_spacing_from_footprint(h) == pytest.approx(12.2257, abs=1e-3)


# ── max_turn_speed() ────────────────────────────────────────────

def test_max_turn_speed_matches_closed_form():
    """v = √(r·g·tan(tilt)) — 손계산값과 직접 대조."""
    v = max_turn_speed(10.0)
    assert v == pytest.approx(math.sqrt(10.0 * 9.80665 * math.tan(math.radians(6.0))),
                              rel=1e-12)
    assert v == pytest.approx(3.2105, abs=1e-4)


def test_max_turn_speed_grows_with_radius():
    """반경이 커지면 같은 tilt 로 더 빨리 돌 수 있다."""
    speeds = [max_turn_speed(r) for r in (5.0, 10.0, 20.0, 50.0)]
    assert speeds == sorted(speeds)
    assert all(a < b for a, b in zip(speeds, speeds[1:]))


def test_max_turn_speed_grows_with_tilt_limit():
    """tilt 상한을 키우면 허용속도가 커진다."""
    speeds = [max_turn_speed(15.0, math.radians(d)) for d in (2.0, 6.0, 12.0, 25.0)]
    assert all(a < b for a, b in zip(speeds, speeds[1:]))


def test_max_turn_speed_scales_as_square_root_of_radius():
    """🔴 sqrt 가 빠지면 여기서 걸린다.

    반경 4배 → 속도 2배여야 한다(sqrt 없으면 4배). 단조성 테스트만으로는
    sqrt 삭제를 못 잡는다 — v = r·g·tanθ 도 단조증가이기 때문이다.
    """
    assert max_turn_speed(40.0) / max_turn_speed(10.0) == pytest.approx(2.0, rel=1e-9)
    # 절대 크기도 못 박는다: sqrt 없이는 6°/10m 에서 10.3m/s 라는 비현실적 값이 나온다.
    assert max_turn_speed(10.0) < 5.0


def test_max_turn_speed_zero_radius_is_zero():
    assert max_turn_speed(0.0) == pytest.approx(0.0)


def test_max_turn_speed_rejects_bad_arguments():
    with pytest.raises(ValueError):
        max_turn_speed(-1.0)
    with pytest.raises(ValueError):
        max_turn_speed(10.0, math.pi / 2.0)


# ── spiral_advance() — 등속 계약 ────────────────────────────────

def _run_spiral(st, n, dt, speed, spacing, r_start, r_max):
    """n 스텝 전진하며 (상태, NE좌표) 이력을 돌려주는 헬퍼."""
    center = np.zeros(2)
    states = [st]
    points = [spiral_point_ne(st, center)]
    for _ in range(n):
        st = spiral_advance(st, dt, speed, spacing, r_start, r_max)
        states.append(st)
        points.append(spiral_point_ne(st, center))
    return states, points


def test_spiral_advance_keeps_constant_ground_speed():
    """🔴 등속 나선의 핵심 계약: 연속 두 점 사이 유클리드 거리 ≈ speed·dt.

    허용오차: **상대 5e-4**. 근거 — 전진 오일러 1스텝의 이산화 오차는
    `≈ b·speed·dt/(2r²)` 이고 여기 파라미터(b=3/2π=0.4775, speed·dt=0.015,
    r≥5)에서 1.4e-4 이다(관측 최대도 그 수준). 반면 √(r²+b²) 를 r 로 바꾸는
    파괴는 `b²/(2r²) = 4.6e-3` 의 오차를 내므로 이 허용오차 안에서 확실히
    red 가 된다(두 값이 9배 갈린다).
    """
    dt, speed, spacing = 0.005, 3.0, 3.0
    r_start, r_max = 5.0, 60.0
    st = SpiralState(theta_rad=0.0, radius_m=r_start, revolutions=0.0)
    _, points = _run_spiral(st, 400, dt, speed, spacing, r_start, r_max)

    step = speed * dt
    dists = [float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:])]
    assert len(dists) == 400
    for d in dists:
        assert d == pytest.approx(step, rel=5e-4)


def test_spiral_advance_constant_speed_from_zero_radius():
    """r=0 에서 출발해도 등속 계약이 유지된다.

    허용오차가 위 테스트(5e-4)보다 느슨한 **2e-3** 인 이유: 중심 근방
    r ~ b(=ring_spacing/2π) 구간에서는 한 스텝 동안 반경이 상대적으로 크게
    변해 전진 오일러의 이산화 오차가 일시적으로 커진다(실측 최대 rel 1.2e-3,
    r≈2.3m 부근). dt 에 정비례하는 순수 이산화 오차이며 dt→0 에서 사라진다
    (실측: dt 0.005→6.0e-3, 0.002→2.4e-3, 0.001→1.2e-3).
    """
    dt, speed, spacing = 0.001, 3.0, 3.0
    r_start, r_max = 0.0, 30.0
    st = SpiralState(theta_rad=0.0, radius_m=0.0, revolutions=0.0)
    _, points = _run_spiral(st, 2000, dt, speed, spacing, r_start, r_max)
    step = speed * dt
    for d in (float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:])):
        assert d == pytest.approx(step, rel=2e-3)


def test_spiral_advance_from_zero_radius_is_finite():
    """🔴 r=0 발산 회귀. √(r²+b²) 의 b² 를 지우면 첫 틱이 0 나눗셈이다.

    (파괴하면 ZeroDivisionError 로 죽거나 inf/NaN 이 상태에 실린다 — 어느
    쪽이든 이 테스트가 잡는다.)
    """
    st = SpiralState(theta_rad=0.0, radius_m=0.0, revolutions=0.0)
    center = np.array([100.0, -50.0])
    for _ in range(50):
        st = spiral_advance(st, 0.1, 2.0, 4.0, 0.0, 40.0)
        assert math.isfinite(st.theta_rad)
        assert math.isfinite(st.radius_m)
        assert math.isfinite(st.revolutions)
        assert not math.isnan(st.radius_m)
        p = spiral_point_ne(st, center)
        assert np.all(np.isfinite(p))
    assert st.radius_m > 0.0        # 실제로 바깥으로 벌어졌다


def test_spiral_advance_radius_grows_ring_spacing_per_revolution():
    """한 바퀴(=1 revolution)마다 반경이 정확히 ring_spacing 만큼 늘어난다."""
    spacing, r_start, r_max = 5.0, 1.0, 500.0
    st = SpiralState(theta_rad=0.0, radius_m=r_start, revolutions=0.0)
    for _ in range(4000):
        st = spiral_advance(st, 0.02, 6.0, spacing, r_start, r_max)
        if st.revolutions >= 3.0:
            break
    assert st.revolutions >= 3.0
    assert st.radius_m == pytest.approx(r_start + spacing * st.revolutions, rel=1e-9)


def test_spiral_advance_revolutions_track_theta():
    st = SpiralState(theta_rad=0.0, radius_m=2.0, revolutions=0.0)
    for _ in range(37):
        st = spiral_advance(st, 0.1, 4.0, 3.0, 2.0, 100.0)
        assert st.revolutions == pytest.approx(st.theta_rad / (2.0 * math.pi))


def test_spiral_advance_saturates_at_max_radius():
    """🔴 max_radius_m 포화: 넘지 않고, 도달 후에는 더 커지지 않는다."""
    r_max = 20.0
    st = SpiralState(theta_rad=0.0, radius_m=0.0, revolutions=0.0)
    radii = []
    for _ in range(3000):
        st = spiral_advance(st, 0.05, 5.0, 4.0, 0.0, r_max)
        radii.append(st.radius_m)
    assert max(radii) <= r_max + 1e-12
    assert radii[-1] == pytest.approx(r_max)
    # 포화 이후 구간에서는 반경이 단조 **비증가** (사실상 상수)여야 한다.
    tail = [r for r in radii if r >= r_max - 1e-9]
    assert len(tail) > 100
    assert all(r == pytest.approx(r_max) for r in tail)
    # 포화 후에도 각도는 계속 돈다 = 최대반경 원을 계속 그린다.
    assert st.theta_rad > 2.0 * math.pi


def test_spiral_advance_does_not_mutate_input_state():
    """frozen dataclass 라 원본이 변하면 안 된다(호출부가 이전 상태를 들고 있을 수 있다)."""
    st0 = SpiralState(theta_rad=0.3, radius_m=4.0, revolutions=0.3 / (2 * math.pi))
    st1 = spiral_advance(st0, 0.1, 3.0, 4.0, 3.0, 50.0)
    assert st0.theta_rad == pytest.approx(0.3)
    assert st0.radius_m == pytest.approx(4.0)
    assert st1 is not st0


def test_spiral_advance_degenerate_point_spiral_is_a_noop():
    """r=0 이면서 ring_spacing=0 은 나선이 아니라 점 — inf/NaN 대신 무동작."""
    st = SpiralState(theta_rad=0.0, radius_m=0.0, revolutions=0.0)
    out = spiral_advance(st, 0.1, 3.0, 0.0, 0.0, 10.0)
    assert out == st


def test_spiral_advance_is_deterministic():
    """같은 입력 → 같은 출력 (wall-clock/난수 없음)."""
    a = SpiralState(theta_rad=0.0, radius_m=0.0, revolutions=0.0)
    b = SpiralState(theta_rad=0.0, radius_m=0.0, revolutions=0.0)
    for _ in range(120):
        a = spiral_advance(a, 0.07, 3.3, 4.4, 0.0, 33.0)
        b = spiral_advance(b, 0.07, 3.3, 4.4, 0.0, 33.0)
    assert a == b


# ── spiral_point_ne() ───────────────────────────────────────────

def test_spiral_point_theta_zero_is_due_north():
    """🔴 θ=0 은 정북(+N). cos/sin 을 뒤바꾸면 여기서 걸린다.

    이 저장소의 수평 좌표는 NED [N, E] 다(ENU 아님) —
    `offboard_node._publish_pos_setpoint` 의 pos_ned=[N, E, h_up] 규약.
    """
    st = SpiralState(theta_rad=0.0, radius_m=7.0, revolutions=0.0)
    p = spiral_point_ne(st, [0.0, 0.0])
    assert p[0] == pytest.approx(7.0)       # N
    assert p[1] == pytest.approx(0.0, abs=1e-12)   # E


def test_spiral_point_theta_half_pi_is_due_east():
    st = SpiralState(theta_rad=math.pi / 2.0, radius_m=7.0, revolutions=0.25)
    p = spiral_point_ne(st, [0.0, 0.0])
    assert p[0] == pytest.approx(0.0, abs=1e-12)
    assert p[1] == pytest.approx(7.0)


def test_spiral_point_respects_center_offset():
    st = SpiralState(theta_rad=0.0, radius_m=3.0, revolutions=0.0)
    p = spiral_point_ne(st, [100.0, -40.0])
    assert p.tolist() == pytest.approx([103.0, -40.0])


def test_spiral_point_radius_is_distance_from_center():
    center = np.array([12.0, 34.0])
    for th in (0.0, 0.7, 2.9, 5.5, 11.0):
        st = SpiralState(theta_rad=th, radius_m=6.25, revolutions=th / (2 * math.pi))
        p = spiral_point_ne(st, center)
        assert float(np.linalg.norm(p - center)) == pytest.approx(6.25)


def test_spiral_point_returns_ndarray_of_two():
    p = spiral_point_ne(SpiralState(0.0, 1.0, 0.0), [0.0, 0.0])
    assert isinstance(p, np.ndarray)
    assert p.shape == (2,)


def test_spiral_point_accepts_three_component_center():
    """호출부가 pos_ned=[N, E, h_up] 을 그대로 넘겨도 앞 2원소만 쓴다."""
    st = SpiralState(theta_rad=0.0, radius_m=2.0, revolutions=0.0)
    p = spiral_point_ne(st, [5.0, 6.0, -7.0])
    assert p.tolist() == pytest.approx([7.0, 6.0])


# ── spiral_complete() ───────────────────────────────────────────

def _advance_until(pred, dt=0.05, speed=5.0, spacing=4.0, r_start=0.0,
                   r_max=20.0, limit=100000):
    """조건을 만족할 때까지 전진(테스트 헬퍼). 못 만족하면 pytest.fail."""
    st = SpiralState(theta_rad=0.0, radius_m=r_start, revolutions=0.0)
    for _ in range(limit):
        st = spiral_advance(st, dt, speed, spacing, r_start, r_max)
        if pred(st):
            return st
    pytest.fail("조건을 만족하기 전에 스텝 상한에 도달했다")


def test_spiral_complete_false_before_max_radius():
    st = _advance_until(lambda s: s.radius_m >= 10.0)
    assert st.radius_m < 20.0
    assert not spiral_complete(st, 20.0, 0.0)


def test_spiral_complete_false_just_after_reaching_max_radius():
    """🔴 최대반경에 닿은 순간은 **한 방위**만 본 것 — 아직 완주가 아니다."""
    st = _advance_until(lambda s: s.radius_m >= 20.0 - 1e-9)
    assert st.radius_m == pytest.approx(20.0)
    assert not spiral_complete(st, 20.0, 0.0)


def test_spiral_complete_true_after_one_more_revolution():
    """최대반경 도달 후 정확히 1바퀴를 더 돌면 True 로 바뀐다."""
    sat = _advance_until(lambda s: s.radius_m >= 20.0 - 1e-9)
    rev_sat = sat.revolutions_at_max_radius
    assert rev_sat is not None
    # 포화 직전(0.999바퀴 추가)까지는 여전히 False
    st = sat
    while st.revolutions < rev_sat + 0.999:
        st = spiral_advance(st, 0.05, 5.0, 4.0, 0.0, 20.0)
        if st.revolutions >= rev_sat + 0.999:
            break
        assert not spiral_complete(st, 20.0, 0.0)
    # 1바퀴를 채우면 True
    while st.revolutions < rev_sat + 1.0:
        st = spiral_advance(st, 0.05, 5.0, 4.0, 0.0, 20.0)
    assert spiral_complete(st, 20.0, 0.0)


def test_spiral_complete_saturation_marker_is_closed_form():
    """포화 회전수는 (max−r_start)/ring_spacing 로 정확히 박힌다(스텝 크기 무관)."""
    sat = _advance_until(lambda s: s.radius_m >= 20.0 - 1e-9,
                         dt=0.05, speed=5.0, spacing=4.0, r_start=0.0, r_max=20.0)
    assert sat.revolutions_at_max_radius == pytest.approx((20.0 - 0.0) / 4.0)


def test_spiral_complete_extra_revolutions_is_configurable():
    """'몇 바퀴 더'가 하드코딩이 아니다."""
    st = _advance_until(lambda s: s.radius_m >= 20.0 - 1e-9)
    while st.revolutions < st.revolutions_at_max_radius + 1.2:
        st = spiral_advance(st, 0.05, 5.0, 4.0, 0.0, 20.0)
    assert spiral_complete(st, 20.0, 0.0, extra_revolutions=1.0)
    assert not spiral_complete(st, 20.0, 0.0, extra_revolutions=2.0)


def test_spiral_complete_unknown_saturation_point_is_not_complete():
    """포화 시점을 모르는 손수 만든 상태는 **미완주**로 본다(안전측)."""
    st = SpiralState(theta_rad=50.0, radius_m=20.0, revolutions=50.0 / (2 * math.pi))
    assert st.revolutions_at_max_radius is None
    assert not spiral_complete(st, 20.0, r_start_m=0.0)
    # 반대로 처음부터 최대반경에서 시작하는 원 궤도라면 1바퀴로 완주다.
    assert spiral_complete(st, 20.0, r_start_m=20.0)


def test_spiral_complete_full_run_eventually_terminates():
    """종단간: 0에서 시작한 나선이 유한 스텝 안에 완주 판정에 도달한다."""
    r_start, r_max, spacing = 0.0, 25.0, 5.0
    st = SpiralState(theta_rad=0.0, radius_m=r_start, revolutions=0.0)
    steps = 0
    while not spiral_complete(st, r_max, r_start):
        st = spiral_advance(st, 0.1, 6.0, spacing, r_start, r_max)
        steps += 1
        if steps > 100000:
            pytest.fail("완주 판정이 오지 않는다")
    # 5m 간격으로 25m 까지 = 5바퀴 + 마지막 반경에서 1바퀴 = 6바퀴
    assert st.revolutions == pytest.approx(6.0, abs=0.05)
    assert st.radius_m == pytest.approx(r_max)


# ── detection_altitude_ok() / max_detection_altitude_m() ────────

def test_detection_altitude_matches_distress_coarse_yaml_table():
    """🔴 `vision/presets/distress_coarse.yaml` 헤더 고도표와의 정합.

    그 표는 gw(h)=1.535·h (=2·0.76733·h, **낙관/수평 75°**) 로 도출됐으므로
    낙관 half-tan 으로 검증한다: 10m/20m 검출, 40m 미검출.
    """
    ht = OPTIMISTIC_HALF_HFOV_TAN
    assert detection_altitude_ok(10.0, half_hfov_tan=ht)
    assert detection_altitude_ok(20.0, half_hfov_tan=ht)
    assert not detection_altitude_ok(40.0, half_hfov_tan=ht)


def test_detection_area_reproduces_yaml_derivation_numbers():
    """같은 공식이 yaml 주석의 면적값(~90,000 / ~22,500 / ~5,625)을 재현한다."""
    ht = OPTIMISTIC_HALF_HFOV_TAN

    def area(h):
        gw = 2.0 * h * ht
        gsd = gw / COARSE_FRAME_WIDTH_PX
        side = DISTRESS_MAT_SIZE_M / gsd
        return side * side

    assert area(10.0) == pytest.approx(90000.0, rel=0.01)
    assert area(20.0) == pytest.approx(22500.0, rel=0.01)
    assert area(40.0) == pytest.approx(5625.0, rel=0.01)
    assert area(10.0) >= COARSE_MIN_AREA_PX
    assert area(20.0) >= COARSE_MIN_AREA_PX
    assert area(40.0) < COARSE_MIN_AREA_PX


def test_max_detection_altitude_computed_values():
    """🔴 실제 계산값 고정 — 보수 38.518m / 낙관 33.570m."""
    assert max_detection_altitude_m(half_hfov_tan=DEFAULT_HALF_HFOV_TAN) == \
        pytest.approx(38.518, abs=1e-3)
    assert max_detection_altitude_m(half_hfov_tan=OPTIMISTIC_HALF_HFOV_TAN) == \
        pytest.approx(33.570, abs=1e-3)


def test_max_detection_altitude_default_is_the_safe_wide_fov_one():
    """🔴 검출의 기본 가정은 **넓은 화각**이어야 한다 (커버리지와 반대!).

    넓은 화각 = 거친 GSD = 검출에 불리 → 그쪽을 가정해야 상한을 과대평가하지
    않는다. 여기서 보수(좁은) 상수를 기본값으로 쓰면 상한이 38.5m로 4.9m
    부풀어 오른다.
    """
    assert DETECTION_HALF_HFOV_TAN == OPTIMISTIC_HALF_HFOV_TAN
    assert max_detection_altitude_m() == \
        pytest.approx(max_detection_altitude_m(half_hfov_tan=OPTIMISTIC_HALF_HFOV_TAN))
    assert max_detection_altitude_m() < \
        max_detection_altitude_m(half_hfov_tan=DEFAULT_HALF_HFOV_TAN)


def test_max_detection_altitude_is_the_exact_boundary():
    """상한 바로 아래는 통과, 바로 위는 불통 — 닫힌 해가 실제 경계인지."""
    h_max = max_detection_altitude_m()
    assert detection_altitude_ok(h_max - 1e-6)
    assert not detection_altitude_ok(h_max + 1e-6)
    assert detection_altitude_ok(h_max)      # 경계 포함(>= min_area)


def test_25m_search_altitude_is_within_detection_limit():
    """🔴 이번 작업의 최대 검증 포인트: 기본 탐색고도 25m 가 검출 상한 안인가.

    ✅ 안이다 — 넓은 화각(안전측) 가정에서도 33.570m 이라 8.57m 여유.
    다만 명목 면적은 14,425px² 로 min_area(8000)의 1.80배뿐이다(20m 는 2.82배)
    — 열화 마진이 얇아지는 것을 수치로 못 박아 둔다.
    """
    assert detection_altitude_ok(25.0)
    assert max_detection_altitude_m() > 25.0
    assert max_detection_altitude_m() - 25.0 == pytest.approx(8.570, abs=0.01)

    ht = DETECTION_HALF_HFOV_TAN
    side_px = DISTRESS_MAT_SIZE_M / ((2.0 * 25.0 * ht) / COARSE_FRAME_WIDTH_PX)
    area25 = side_px * side_px
    assert area25 == pytest.approx(14425.0, rel=0.005)
    assert area25 / COARSE_MIN_AREA_PX == pytest.approx(1.803, abs=0.01)


def test_detection_altitude_is_monotone_in_altitude():
    """고도가 높아질수록 어려워진다 — True 구간이 아래쪽 연속 구간이어야 한다."""
    oks = [detection_altitude_ok(h) for h in range(1, 60)]
    assert oks[0] is True
    assert oks[-1] is False
    # True 다음에 False 가 나온 뒤 다시 True 가 나오면 안 된다.
    assert oks == sorted(oks, reverse=True)


def test_max_detection_altitude_scales_with_target_size_and_min_area():
    """하드코딩이 아니라 인자에서 유도되는지."""
    base = max_detection_altitude_m()
    assert max_detection_altitude_m(target_size_m=6.0) == pytest.approx(2.0 * base)
    # min_area 4배 → 한 변 2배 요구 → 상한 절반
    assert max_detection_altitude_m(min_area_px=4.0 * COARSE_MIN_AREA_PX) == \
        pytest.approx(base / 2.0)
    assert max_detection_altitude_m(frame_width_px=3072.0) == pytest.approx(2.0 * base)


def test_max_detection_altitude_disabled_min_area_is_infinite():
    assert math.isinf(max_detection_altitude_m(min_area_px=0.0))
    assert detection_altitude_ok(10000.0, min_area_px=0.0)


def test_max_detection_altitude_rejects_nonpositive_fov():
    with pytest.raises(ValueError):
        max_detection_altitude_m(half_hfov_tan=0.0)


def test_module_constants_match_distress_coarse_yaml():
    """🔴 vision 프리셋에서 **복제**한 상수들 — 갈라지면 red.

    fc_bridge 는 vision 을 import 하지 않으므로(도메인 간 교차 import 0) 값을
    복제할 수밖에 없다. 그 복제가 조용히 어긋나는 것을 여기서 잡는다.
    `vision/presets/distress_coarse.yaml`: min_area=8000, size_m=3.0,
    다운스케일 폭 1536px.
    """
    assert DISTRESS_MAT_SIZE_M == 3.0
    assert COARSE_MIN_AREA_PX == 8000.0
    assert COARSE_FRAME_WIDTH_PX == 1536.0
