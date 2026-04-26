"""
체크포인트 1 단위 테스트
==========================

정책: 동역학은 가속도 한계를 강제하지 않는다 (클립 없음).
     알고리즘이 무리한 명령을 내면 그대로 적용되고 violation으로 기록된다.
     속도/자세/고도 등 물리적 한계는 여전히 강제됨.
"""
from __future__ import annotations
from utils.config_loader import load_aircraft_params
from utils.math_utils import (
    closest_point_on_polyline, look_ahead_point,
    signed_cross_track_error_2d,
)
from utils.delay_buffer import DelayBuffer
from utils.geodetic import geodetic_to_ned, ned_to_geodetic
from estimators.alpha_beta_filter import AlphaBetaFilter
from noise.gps_noise import GPSNoise
from dynamics.point_mass_3dof import PointMass3DoF
from dynamics.base_dynamics import (
    AircraftState, ControlInput, MODE_CRUISE
)
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass


def test_dynamics_straight_flight():
    print("\n[Test 1] 직진 비행 (모든 제어 입력 0)")
    params = load_aircraft_params()
    dyn = PointMass3DoF(params)
    state = AircraftState(
        pos=np.array([0.0, 0.0, 100.0]),
        v=18.0, chi=0.0, gamma=0.0, phi=0.0, mode=MODE_CRUISE,
    )
    u = ControlInput(bank_cmd=0.0, pitch_cmd=0.0, thrust_cmd=0.0)
    for _ in range(100):
        state = dyn.step(state, u, np.zeros(3), 0.01)
    print(f"  최종 위치: {state.pos}, 기대 x_N ≈ 18.0")
    assert abs(state.pos[0] - 18.0) < 0.5
    assert abs(state.pos[1]) < 0.1
    assert abs(state.pos[2] - 100.0) < 0.5
    assert abs(state.v - 18.0) < 0.5
    print("  ✓ 통과")


def test_dynamics_steady_turn():
    print("\n[Test 2] 정상 선회 (φ_cmd = 16.7°, 가속도 0.3g 한계 정확히 도달)")
    params = load_aircraft_params()
    dyn = PointMass3DoF(params)
    phi_cmd = np.deg2rad(16.7)
    state = AircraftState(
        pos=np.array([0.0, 0.0, 100.0]),
        v=18.0, chi=0.0, gamma=0.0, phi=phi_cmd, mode=MODE_CRUISE,
    )
    u = ControlInput(bank_cmd=phi_cmd, pitch_cmd=0.0, thrust_cmd=0.0)
    g = 9.81
    chi_rate_expected = g * np.tan(phi_cmd) / 18.0
    chi_history = [state.chi]
    a_body_history = []
    for _ in range(int(2.0 / 0.01)):
        state = dyn.step(state, u, np.zeros(3), 0.01)
        chi_history.append(state.chi)
        a_body_history.append(state.a_body.copy())
    chi_rate_actual = (chi_history[-1] - chi_history[0]) / 2.0
    a_y_mean = np.mean([a[1] for a in a_body_history])
    print(f"  기대 χ_rate ≈ {np.rad2deg(chi_rate_expected):.2f}°/s")
    print(f"  실제 χ_rate ≈ {np.rad2deg(chi_rate_actual):.2f}°/s")
    print(
        f"  평균 a_y_body ≈ {a_y_mean:.3f} m/s² (기대 {g * np.tan(phi_cmd):.3f})")
    assert abs(a_y_mean - g * np.tan(phi_cmd)) < 0.3
    assert abs(chi_rate_actual - chi_rate_expected) / chi_rate_expected < 0.05
    print("  ✓ 통과")


def test_dynamics_acceleration_violation():
    print("\n[Test 3] 가속도 한계 위반 — 60° 뱅크 명령 → 클립 없이 그대로 적용")
    params = load_aircraft_params()
    dyn = PointMass3DoF(params)
    state = AircraftState(
        pos=np.array([0.0, 0.0, 100.0]),
        v=18.0, chi=0.0, gamma=0.0, phi=0.0, mode=MODE_CRUISE,
    )
    # 60° 뱅크 명령: 구조 한계 69.7° 이내 → phi가 60°까지 도달
    # 정상선회 시 a_n = g·tan(60°) ≈ 1.732g, 한계 0.3g
    u = ControlInput(bank_cmd=np.deg2rad(60.0), pitch_cmd=0.0, thrust_cmd=0.0)
    n_violations = 0
    a_total_history = []
    violation_amounts = []
    for _ in range(200):
        state = dyn.step(state, u, np.zeros(3), 0.01)
        if state.accel_violation:
            n_violations += 1
            violation_amounts.append(state.accel_violation_amount)
        a_total_history.append(state.a_total_actual)

    a_max = params["a_max_g"] * 9.81
    print(f"  최종 phi: {np.rad2deg(state.phi):.2f}° (명령 60°)")
    print(
        f"  전체 구간 max a_total_actual: {max(a_total_history):.3f} m/s² (한계 {a_max:.3f})")
    print(f"  violation 발생: {n_violations}/200")
    print(
        f"  최대 violation amount: {max(violation_amounts):.3f} m/s² ({max(violation_amounts)/9.81:.3f}g)")

    # 클립이 없으므로 phi가 명령값 60°에 도달
    assert abs(np.rad2deg(state.phi) - 60.0) < 1.0, \
        f"클립 없어야 하는데 phi가 60°에 못 미침: {np.rad2deg(state.phi):.2f}°"
    # 가속도가 한계를 크게 초과
    assert max(a_total_history) > a_max * 3.0, \
        f"violation 발생해야 하는데 가속도 너무 작음: {max(a_total_history):.3f}"
    assert n_violations > 100
    print("  ✓ 통과")


def test_dynamics_no_violation():
    print("\n[Test 3b] 한계 내 명령 — 5° 뱅크: violation 없음, 전체 구간 검증")
    params = load_aircraft_params()
    dyn = PointMass3DoF(params)
    state = AircraftState(
        pos=np.array([0.0, 0.0, 100.0]),
        v=18.0, chi=0.0, gamma=0.0, phi=0.0, mode=MODE_CRUISE,
    )
    u = ControlInput(bank_cmd=np.deg2rad(5.0), pitch_cmd=0.0, thrust_cmd=0.0)
    n_violations = 0
    history = []
    for _ in range(200):
        state = dyn.step(state, u, np.zeros(3), 0.01)
        if state.accel_violation:
            n_violations += 1
        history.append(state)
    print(f"  violation 발생: {n_violations}/200 (기대 0)")
    print(f"  최종 a_total_actual = {state.a_total_actual:.3f}")
    assert n_violations == 0
    # 전체 구간 검사 — 클립 없으므로 cmd == actual
    for i, s in enumerate(history):
        assert abs(s.a_total_cmd - s.a_total_actual) < 1e-9, \
            f"step {i}: cmd != actual"
        assert s.accel_violation_amount == 0.0
    print("  ✓ 통과 (전체 200 step 검증)")


def test_gps_noise_statistics():
    print("\n[Test 4] GPS 노이즈 — 백색잡음 σ 검증")
    sigma = 0.5
    gps = GPSNoise(sigma_white=sigma, tau_bias=300.0,
                   sigma_bias_drive=0.0, seed=42)
    true_pos = np.array([100.0, 200.0, 150.0])
    samples = []
    for _ in range(10000):
        gps.step_bias(0.01)
        samples.append(gps.measure(true_pos))
    sigma_est = np.std(np.array(samples) - true_pos, axis=0)
    print(f"  추정 σ: {sigma_est} (기대 {sigma})")
    assert np.all(np.abs(sigma_est - sigma) < 0.05)
    print("  ✓ 통과")


def test_alpha_beta_filter():
    print("\n[Test 5] α-β 필터 — 노이즈 측정에서 진위치 추적")
    np.random.seed(0)
    filt = AlphaBetaFilter(alpha=0.7, beta=0.3)
    dt = 0.1
    v_true = np.array([18.0, 0.0, 0.0])
    pos_true = np.zeros(3)
    sigma = 0.5
    filt.initialize(pos_true, np.zeros(3), 0.0)
    errors_pos = []
    errors_vel = []
    for k in range(1, 200):
        pos_true = pos_true + v_true * dt
        z = pos_true + sigma * np.random.randn(3)
        p_hat, v_hat = filt.update(z, k * dt)
        errors_pos.append(np.linalg.norm(p_hat - pos_true))
        errors_vel.append(np.linalg.norm(v_hat - v_true))
    rms_pos = np.sqrt(np.mean(np.array(errors_pos[-100:])**2))
    rms_vel = np.sqrt(np.mean(np.array(errors_vel[-100:])**2))
    print(f"  후반 RMS 위치 오차: {rms_pos:.3f} m")
    print(f"  후반 RMS 속도 오차: {rms_vel:.3f} m/s")
    assert rms_pos < sigma * 1.5
    assert rms_vel < 3.0
    print("  ✓ 통과")


def test_geodetic_roundtrip():
    print("\n[Test 6] 위/경도 ↔ NED 변환 가역성")
    ref_lat, ref_lon, ref_alt = 35.1796, 126.8504, 0.0
    test_lat = ref_lat + 500.0 / 6378137.0 * (180 / np.pi)
    test_lon = ref_lon + 1000.0 / \
        (6378137.0 * np.cos(np.deg2rad(ref_lat))) * (180 / np.pi)
    ned = geodetic_to_ned(test_lat, test_lon, 100.0, ref_lat, ref_lon, ref_alt)
    print(f"  NED: {ned}")
    assert abs(ned[0] - 500.0) < 0.5
    assert abs(ned[1] - 1000.0) < 0.5
    lat2, lon2, _ = ned_to_geodetic(ned[0], ned[1], ned[2],
                                    ref_lat, ref_lon, ref_alt)
    assert abs(lat2 - test_lat) < 1e-7
    assert abs(lon2 - test_lon) < 1e-7
    print("  ✓ 통과")


def test_delay_buffer():
    print("\n[Test 7] 지연 버퍼 — 0.05s 지연")
    buf = DelayBuffer(delay=0.05, dt=0.01, init_value=0.0)
    outputs = [buf.update(v) for v in range(20)]
    print(f"  출력 첫 10개: {outputs[:10]}")
    assert outputs[0:5] == [0, 0, 0, 0, 0]
    assert outputs[5:10] == [0, 1, 2, 3, 4]
    print("  ✓ 통과")


def test_polyline_utils():
    print("\n[Test 8] 폴리라인 유틸 — closest point + look-ahead")
    polyline = np.array([
        [0.0, 0.0, 100.0], [100.0, 0.0, 100.0],
        [100.0, 100.0, 100.0], [0.0, 100.0, 100.0],
    ])
    point = np.array([50.0, 5.0, 100.0])
    seg, t, cp, d = closest_point_on_polyline(point, polyline)
    print(f"  closest dist: {d:.3f}")
    assert seg == 0
    assert abs(d - 5.0) < 0.01
    la, _, _ = look_ahead_point(polyline, seg, t, 50.0)
    assert abs(la[0] - 100.0) < 0.5
    err, _, _ = signed_cross_track_error_2d(point, polyline)
    assert abs(abs(err) - 5.0) < 0.01
    print("  ✓ 통과")


def test_scenario_override():
    print("\n[Test 9] 시나리오 override — a_max_g 변경")
    from utils.config_loader import (
        load_aircraft_params, merge_scenario_into_aircraft
    )
    aircraft = load_aircraft_params()
    merged = merge_scenario_into_aircraft(
        aircraft, {"a_max_g": 0.15, "v_cruise": 20.0})
    print(f"  원본: {aircraft['a_max_g']}, override 후: {merged['a_max_g']}")
    assert merged["a_max_g"] == 0.15
    assert aircraft["a_max_g"] == 0.3  # 원본 보존

    dyn = PointMass3DoF(merged)
    assert abs(dyn.a_max - 0.15 * 9.81) < 1e-6
    # 20° 뱅크 → 0.36g, 한계 0.15g → violation 발생
    state = AircraftState(
        pos=np.zeros(3), v=18.0, chi=0.0, gamma=0.0,
        phi=0.0, mode=MODE_CRUISE,
    )
    u = ControlInput(bank_cmd=np.deg2rad(20.0), pitch_cmd=0.0, thrust_cmd=0.0)
    n_violations = 0
    for _ in range(200):
        state = dyn.step(state, u, np.zeros(3), 0.01)
        if state.accel_violation:
            n_violations += 1
    print(f"  20° 뱅크 + 0.15g 한계 → violation {n_violations}/200")
    assert n_violations > 100
    assert abs(state.a_max_used - 0.15 * 9.81) < 1e-6
    print("  ✓ 통과")


def test_acceleration_limit_runtime_change():
    print("\n[Test 10] 가속도 한계 런타임 변경 — set_a_max_g()")
    params = load_aircraft_params()
    dyn = PointMass3DoF(params)
    print(f"  초기 a_max: {dyn.a_max:.3f}")
    dyn.set_a_max_g(0.15)
    print(f"  변경 후: {dyn.a_max:.3f}")
    assert abs(dyn.a_max - 0.15 * 9.81) < 1e-6

    state = AircraftState(
        pos=np.zeros(3), v=18.0, chi=0.0, gamma=0.0,
        phi=0.0, mode=MODE_CRUISE,
    )
    state = dyn.step(state, ControlInput(), np.zeros(3), 0.01)
    assert abs(state.a_max_used - 0.15 * 9.81) < 1e-6
    print(f"  step 후 a_max_used: {state.a_max_used:.3f}")
    print("  ✓ 통과")


def test_acceleration_recording_consistency():
    print("\n[Test 11] 가속도 기록 일관성 — 전체 구간 검증")
    params = load_aircraft_params()
    dyn = PointMass3DoF(params)
    state = AircraftState(
        pos=np.zeros(3), v=18.0, chi=0.0, gamma=0.0,
        phi=0.0, mode=MODE_CRUISE,
    )
    u = ControlInput(bank_cmd=np.deg2rad(60.0), pitch_cmd=0.0, thrust_cmd=0.0)
    history = []
    for _ in range(200):
        state = dyn.step(state, u, np.zeros(3), 0.01)
        history.append(state)

    print(f"  최종 a_total_cmd  = {state.a_total_cmd:.3f}")
    print(f"  최종 a_total_actual = {state.a_total_actual:.3f}")
    print(f"  최종 violation_amount = {state.accel_violation_amount:.3f}")

    # 전체 구간 — 클립 없으므로 cmd == actual
    for i, s in enumerate(history):
        assert abs(s.a_total_cmd - s.a_total_actual) < 1e-9, \
            f"step {i}: cmd != actual"
        # violation 플래그 ↔ a_total > a_max 일치
        expected_v = s.a_total_actual > s.a_max_used
        assert s.accel_violation == expected_v, f"step {i} violation 불일치"
        # violation_amount 정확
        expected_amt = max(0.0, s.a_total_actual - s.a_max_used)
        assert abs(s.accel_violation_amount - expected_amt) < 1e-6
    print("  ✓ 통과 (전체 200 step 검증)")


def test_sim_log_and_metrics():
    print("\n[Test 12] SimLog + compute_acceleration_metrics (violation 기반)")
    from utils.sim_log import SimLog, compute_acceleration_metrics

    params = load_aircraft_params()
    dyn = PointMass3DoF(params)
    state = AircraftState(
        pos=np.zeros(3), v=18.0, chi=0.0, gamma=0.0,
        phi=0.0, mode=MODE_CRUISE,
    )
    log = SimLog()
    for i in range(500):
        u = ControlInput(bank_cmd=np.deg2rad(5.0)) if i < 200 \
            else ControlInput(bank_cmd=np.deg2rad(60.0))
        state = dyn.step(state, u, np.zeros(3), 0.01)
        log.append_step(state, u, compute_time_ctrl=0.0001)

    m = compute_acceleration_metrics(log)
    print(f"  총 step: {len(log.t)}")
    print(
        f"  max a_total_actual: {m['max_a_total_actual']:.3f} ({m['max_a_total_actual_g']:.3f}g)")
    print(
        f"  max a_total_cmd:    {m['max_a_total_cmd']:.3f} ({m['max_a_total_cmd_g']:.3f}g)")
    print(
        f"  n_violations: {m['n_violations']} ({m['violation_time_ratio']*100:.1f}%)")
    print(f"  max violation amount: {m['max_violation_amount']:.3f} m/s²")

    assert m["max_a_total_actual_g"] > 1.0
    assert abs(m["max_a_total_cmd"] - m["max_a_total_actual"]) < 1e-6
    assert 200 < m["n_violations"] < 320
    assert m["max_violation_amount_g"] > 1.0
    print("  ✓ 통과")


def test_path_required_profile():
    """Test 13: 이상 가속도/뱅크각 프로파일 계산 검증."""
    print("\n[Test 13] 이상 프로파일 — 곡률에서 a_required, phi_required 도출")
    from path_planning.base_planner import Path, PathPoint
    from metrics import compute_path_required_profile

    # 인공 경로: 직선 + 곡률 일정 원호
    # κ = 0 (직선) → a_n = 0
    # κ = 0.01 (R=100m), v=18 → a_n = 18² × 0.01 = 3.24 m/s² ≈ 0.33g
    # → phi = arctan(3.24/9.81) ≈ 18.3°
    pts = []
    s = 0.0
    for i in range(10):
        pts.append(PathPoint(
            pos=np.array([float(i*5), 0.0, 100.0]),
            v_ref=18.0, chi_ref=0.0, gamma_ref=0.0,
            curvature=0.0, s=s,
        ))
        s += 5.0
    for i in range(10):
        pts.append(PathPoint(
            pos=np.array([0.0, float(i*5), 100.0]),
            v_ref=18.0, chi_ref=np.pi/2, gamma_ref=0.0,
            curvature=0.01, s=s,
        ))
        s += 5.0
    path = Path(points=pts)

    profile = compute_path_required_profile(path, gravity=9.81)

    # 첫 10점 (직선): a_n = 0
    assert np.all(profile["a_n_required"][:10] < 1e-9), \
        f"직선 구간 a_n != 0: {profile['a_n_required'][:10]}"
    # 후 10점 (κ=0.01): a_n ≈ 3.24
    expected_a_n = 18.0**2 * 0.01
    assert np.allclose(profile["a_n_required"][10:], expected_a_n, atol=0.01), \
        f"곡선 구간 a_n 오차: {profile['a_n_required'][10:]}"
    # phi_required
    expected_phi = np.arctan(expected_a_n / 9.81)
    assert np.allclose(profile["phi_required"][10:], expected_phi, atol=0.001)
    print(f"  직선 구간 a_n: {profile['a_n_required'][0]:.3f} (기대 0)")
    print(
        f"  곡선 구간 a_n: {profile['a_n_required'][15]:.3f} m/s² (기대 {expected_a_n:.3f})")
    print(f"  곡선 구간 φ:  {np.rad2deg(profile['phi_required'][15]):.2f}° "
          f"(기대 {np.rad2deg(expected_phi):.2f}°)")
    print("  ✓ 통과")


def test_composite_score_consistency():
    """Test 14: 종합 점수 — 더 나쁜 결과가 더 높은 점수를 받아야."""
    print("\n[Test 14] 종합 점수 — 일관성")
    from metrics import MetricsResult, compute_composite_score

    # 이상적 결과 (성공, 작은 오차)
    m_good = MetricsResult(
        n_wps=6, n_wps_arrived=6, success=True,
        mean_cpa=10.0, max_cpa=15.0,
        rms_crosstrack=5.0, rms_altitude_err=2.0,
        violation_time_ratio=0.0, excess_a_rms_g=0.05,
        efficiency_vs_planned=0.95, bank_rate_rms=0.5,
    )
    score_good = compute_composite_score(m_good)

    # 나쁜 결과 (오차 크고 위반 많음)
    m_bad = MetricsResult(
        n_wps=6, n_wps_arrived=6, success=True,
        mean_cpa=80.0, max_cpa=150.0,
        rms_crosstrack=50.0, rms_altitude_err=20.0,
        violation_time_ratio=0.3, excess_a_rms_g=0.3,
        efficiency_vs_planned=0.5, bank_rate_rms=2.0,
    )
    score_bad = compute_composite_score(m_bad)

    # 실패 결과
    m_fail = MetricsResult(
        n_wps=6, n_wps_arrived=2, success=False,
        mean_cpa=80.0, rms_crosstrack=50.0,
        violation_time_ratio=0.3,
    )
    score_fail = compute_composite_score(m_fail)

    print(f"  좋은 결과: {score_good:.2f}")
    print(f"  나쁜 결과: {score_bad:.2f}")
    print(f"  실패 결과: {score_fail:.2f}")
    assert score_good < score_bad, "좋은 결과의 점수가 더 작아야"
    assert score_bad < score_fail, "실패는 가장 큰 페널티"
    print("  ✓ 통과")
    print("\n[Test 12] SimLog + compute_acceleration_metrics (violation 기반)")
    from utils.sim_log import SimLog, compute_acceleration_metrics

    params = load_aircraft_params()
    dyn = PointMass3DoF(params)
    state = AircraftState(
        pos=np.zeros(3), v=18.0, chi=0.0, gamma=0.0,
        phi=0.0, mode=MODE_CRUISE,
    )
    log = SimLog()
    for i in range(500):
        u = ControlInput(bank_cmd=np.deg2rad(5.0)) if i < 200 \
            else ControlInput(bank_cmd=np.deg2rad(60.0))
        state = dyn.step(state, u, np.zeros(3), 0.01)
        log.append_step(state, u, compute_time_ctrl=0.0001)

    m = compute_acceleration_metrics(log)
    print(f"  총 step: {len(log.t)}")
    print(
        f"  max a_total_actual: {m['max_a_total_actual']:.3f} ({m['max_a_total_actual_g']:.3f}g)")
    print(
        f"  max a_total_cmd:    {m['max_a_total_cmd']:.3f} ({m['max_a_total_cmd_g']:.3f}g)")
    print(
        f"  n_violations: {m['n_violations']} ({m['violation_time_ratio']*100:.1f}%)")
    print(
        f"  max violation amount: {m['max_violation_amount']:.3f} m/s² ({m['max_violation_amount_g']:.3f}g)")
    print(f"  mean violation amount: {m['mean_violation_amount']:.3f} m/s²")

    # 클립 없는 정책 → max actual이 1g 이상
    assert m["max_a_total_actual_g"] > 1.0, \
        f"클립 없는 정책에서 actual이 1g 이하: {m['max_a_total_actual_g']}"
    # cmd == actual
    assert abs(m["max_a_total_cmd"] - m["max_a_total_actual"]) < 1e-6
    # 60° 구간(약 300 step)에서 위반, 5° 구간(200)은 위반 없음
    assert 200 < m["n_violations"] < 320
    assert m["max_violation_amount_g"] > 1.0
    print("  ✓ 통과")


if __name__ == "__main__":
    print("=" * 60)
    print("체크포인트 1 단위 테스트 (violation 정책)")
    print("=" * 60)

    tests = [
        test_dynamics_straight_flight,
        test_dynamics_steady_turn,
        test_dynamics_acceleration_violation,
        test_dynamics_no_violation,
        test_gps_noise_statistics,
        test_alpha_beta_filter,
        test_geodetic_roundtrip,
        test_delay_buffer,
        test_polyline_utils,
        test_scenario_override,
        test_acceleration_limit_runtime_change,
        test_acceleration_recording_consistency,
        test_sim_log_and_metrics,
        test_path_required_profile,
        test_composite_score_consistency,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  ✗ 실패: {e}")
            failed.append(t.__name__)
        except Exception as e:
            print(f"  ✗ 예외: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(t.__name__)

    print("\n" + "=" * 60)
    if not failed:
        print("✓ 모든 테스트 통과")
    else:
        print(f"✗ {len(failed)}개 실패: {failed}")
        sys.exit(1)
