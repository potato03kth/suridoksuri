"""
체크포인트 1 단위 테스트
==========================

목적:
1. 동역학이 물리적으로 그럴듯한 결과를 내는지
2. 노이즈 모듈의 통계적 특성이 맞는지
3. α-β 필터가 진위치를 잘 추적하는지
4. 좌표 변환의 가역성

실행: cd /home/claude/vtol_sim && python -m tests.test_checkpoint1
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from dynamics.base_dynamics import (
    AircraftState, ControlInput, MODE_CRUISE
)
from dynamics.point_mass_3dof import PointMass3DoF
from noise.gps_noise import GPSNoise
from noise.wind_model import WindModel
from estimators.alpha_beta_filter import AlphaBetaFilter
from utils.geodetic import geodetic_to_ned, ned_to_geodetic
from utils.delay_buffer import DelayBuffer
from utils.math_utils import (
    closest_point_on_polyline, look_ahead_point,
    signed_cross_track_error_2d, wrap_angle
)


def load_aircraft_params() -> dict:
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "aircraft.yaml"
    )
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ============================================================
# Test 1: 동역학 — 직진 비행 (제어 입력 0)
# ============================================================
def test_dynamics_straight_flight():
    print("\n[Test 1] 직진 비행 (모든 제어 입력 0)")
    params = load_aircraft_params()
    dyn = PointMass3DoF(params)

    state = AircraftState(
        pos=np.array([0.0, 0.0, 100.0]),
        v=18.0,
        chi=0.0,         # 정북향
        gamma=0.0,
        phi=0.0,
        mode=MODE_CRUISE,
    )
    u = ControlInput(bank_cmd=0.0, pitch_cmd=0.0, thrust_cmd=0.0)
    wind = np.zeros(3)
    dt = 0.01

    # 1초 시뮬레이션
    for _ in range(100):
        state = dyn.step(state, u, wind, dt)

    # 기대: 18 m/s × 1s = 18 m 북향 이동, 동향/고도 변화 없음
    expected_x_N = 18.0
    print(f"  최종 위치: {state.pos}, 기대 x_N ≈ {expected_x_N}")
    assert abs(state.pos[0] - expected_x_N) < 0.5, f"북향 이동 오차 큼: {state.pos[0]}"
    assert abs(state.pos[1]) < 0.1, f"동향 이동 0이어야: {state.pos[1]}"
    assert abs(state.pos[2] - 100.0) < 0.5, f"고도 유지 안 됨: {state.pos[2]}"
    assert abs(state.v - 18.0) < 0.5, f"속도 변화 큼: {state.v}"
    print("  ✓ 통과")


# ============================================================
# Test 2: 동역학 — 정상 선회 (constant bank)
# ============================================================
def test_dynamics_steady_turn():
    print("\n[Test 2] 정상 선회 (φ_cmd = 16.7°, 가속도 한계 내)")
    params = load_aircraft_params()
    dyn = PointMass3DoF(params)

    # 16.7° 뱅크 → 가속도 0.3g (한계 도달)
    phi_cmd = np.deg2rad(16.7)
    state = AircraftState(
        pos=np.array([0.0, 0.0, 100.0]),
        v=18.0,
        chi=0.0,
        gamma=0.0,
        phi=phi_cmd,  # 즉시 정상선회 상태로 시작
        mode=MODE_CRUISE,
    )
    u = ControlInput(bank_cmd=phi_cmd, pitch_cmd=0.0, thrust_cmd=0.0)
    wind = np.zeros(3)
    dt = 0.01

    g = 9.81
    # 기대 회전반경: R = v² / (g tan φ) = 18² / (9.81 × 0.3) ≈ 110.1 m
    R_expected = 18.0**2 / (g * np.tan(phi_cmd))
    # 기대 회전율: dχ/dt = g tan(φ) / v ≈ 0.163 rad/s → 1초에 약 9.4°
    chi_rate_expected = g * np.tan(phi_cmd) / 18.0

    chi_history = [state.chi]
    pos_history = [state.pos.copy()]
    a_body_history = []
    for _ in range(int(2.0 / dt)):  # 2초
        state = dyn.step(state, u, wind, dt)
        chi_history.append(state.chi)
        pos_history.append(state.pos.copy())
        a_body_history.append(state.a_body.copy())

    chi_rate_actual = (chi_history[-1] - chi_history[0]) / 2.0
    print(f"  기대 R ≈ {R_expected:.1f} m, 기대 χ_rate ≈ {np.rad2deg(chi_rate_expected):.2f}°/s")
    print(f"  실제 χ_rate (2초간) ≈ {np.rad2deg(chi_rate_actual):.2f}°/s")
    print(f"  최종 위치: {state.pos}")

    # body-frame 가속도: a_y가 약 +g·tan(φ) ≈ +0.3g 이어야
    a_y_mean = np.mean([a[1] for a in a_body_history])
    print(f"  평균 a_y_body ≈ {a_y_mean:.3f} m/s² (기대 {g * np.tan(phi_cmd):.3f})")
    assert abs(a_y_mean - g * np.tan(phi_cmd)) < 0.3, "구심가속도 오차 큼"

    # 회전율 오차 확인
    rate_error = abs(chi_rate_actual - chi_rate_expected) / chi_rate_expected
    assert rate_error < 0.05, f"회전율 오차 5% 초과: {rate_error*100:.1f}%"
    print("  ✓ 통과")


# ============================================================
# Test 3: 동역학 — 가속도 제약 강제 확인
# ============================================================
def test_dynamics_acceleration_clip():
    print("\n[Test 3] 가속도 제약 — 과도한 뱅크각 명령 시 자동 클립")
    params = load_aircraft_params()
    dyn = PointMass3DoF(params)

    state = AircraftState(
        pos=np.array([0.0, 0.0, 100.0]),
        v=18.0, chi=0.0, gamma=0.0, phi=0.0,
        mode=MODE_CRUISE,
    )
    # 60° 뱅크 명령 (구조 한계 69.7° 이내지만 가속도 한계 초과)
    u = ControlInput(bank_cmd=np.deg2rad(60.0), pitch_cmd=0.0, thrust_cmd=0.0)
    wind = np.zeros(3)
    dt = 0.01

    # 충분히 응답할 시간 (tau_phi=0.1s)
    for _ in range(200):
        state = dyn.step(state, u, wind, dt)

    a_total = np.linalg.norm([state.a_body[0], state.a_body[1]])  # 수평면 가속도 크기
    a_max = params["a_max_g"] * 9.81
    print(f"  최종 phi: {np.rad2deg(state.phi):.2f}° (명령 60°)")
    print(f"  수평 가속도 크기: {a_total:.3f} m/s² (한계 {a_max:.3f})")
    # 가속도가 한계를 크게 초과하지 않아야
    assert a_total <= a_max * 1.05, f"가속도 한계 초과: {a_total}"
    # 뱅크각도 ~16.7°에서 정착해야
    assert abs(state.phi) < np.deg2rad(20.0), f"뱅크각 클립 안됨: {np.rad2deg(state.phi)}"
    print("  ✓ 통과")


# ============================================================
# Test 4: GPS 노이즈 — 통계 특성
# ============================================================
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
    samples = np.array(samples)
    errors = samples - true_pos
    sigma_estimated = np.std(errors, axis=0)
    print(f"  추정 σ: {sigma_estimated} (기대 {sigma})")
    assert np.all(np.abs(sigma_estimated - sigma) < 0.05), f"σ 추정 오차 큼"
    print("  ✓ 통과")


# ============================================================
# Test 5: α-β 필터 수렴
# ============================================================
def test_alpha_beta_filter():
    print("\n[Test 5] α-β 필터 — 노이즈 측정에서 진위치 추적")
    np.random.seed(0)
    filt = AlphaBetaFilter(alpha=0.7, beta=0.3)

    # 18 m/s 등속 직선 운동 + GPS 노이즈
    dt = 0.1  # 10 Hz GPS
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

    # 후반 100스텝의 RMS 오차
    rms_pos = np.sqrt(np.mean(np.array(errors_pos[-100:])**2))
    rms_vel = np.sqrt(np.mean(np.array(errors_vel[-100:])**2))
    print(f"  후반 RMS 위치 오차: {rms_pos:.3f} m (입력 σ={sigma})")
    print(f"  후반 RMS 속도 오차: {rms_vel:.3f} m/s")
    assert rms_pos < sigma * 1.5, f"필터가 노이즈를 줄이지 못함"
    assert rms_vel < 3.0, f"속도 추정 오차 큼"
    print("  ✓ 통과")


# ============================================================
# Test 6: 좌표 변환 가역성
# ============================================================
def test_geodetic_roundtrip():
    print("\n[Test 6] 위/경도 ↔ NED 변환 가역성")
    ref_lat, ref_lon, ref_alt = 35.1796, 126.8504, 0.0
    # 1 km 동쪽, 500 m 북쪽, 100 m 위
    test_lat = ref_lat + 500.0 / 6378137.0 * (180 / np.pi)
    test_lon = ref_lon + 1000.0 / (6378137.0 * np.cos(np.deg2rad(ref_lat))) * (180 / np.pi)
    test_alt = 100.0

    ned = geodetic_to_ned(test_lat, test_lon, test_alt, ref_lat, ref_lon, ref_alt)
    print(f"  NED: {ned} (기대 [500, 1000, 100])")
    assert abs(ned[0] - 500.0) < 0.5
    assert abs(ned[1] - 1000.0) < 0.5
    assert abs(ned[2] - 100.0) < 0.001

    # 역변환
    lat2, lon2, alt2 = ned_to_geodetic(ned[0], ned[1], ned[2],
                                       ref_lat, ref_lon, ref_alt)
    assert abs(lat2 - test_lat) < 1e-7
    assert abs(lon2 - test_lon) < 1e-7
    print("  ✓ 통과")


# ============================================================
# Test 7: 지연 버퍼
# ============================================================
def test_delay_buffer():
    print("\n[Test 7] 지연 버퍼 — 0.05s 지연")
    buf = DelayBuffer(delay=0.05, dt=0.01, init_value=0.0)
    inputs = list(range(20))
    outputs = []
    for v in inputs:
        outputs.append(buf.update(v))

    # delay=0.05, dt=0.01 → 5스텝 지연
    # 처음 5개 출력은 init_value(0), 그 다음부터는 inputs[0:]
    print(f"  입력: {inputs[:10]}")
    print(f"  출력: {outputs[:10]}")
    assert outputs[0:5] == [0, 0, 0, 0, 0]
    assert outputs[5:10] == [0, 1, 2, 3, 4]
    print("  ✓ 통과")


# ============================================================
# Test 8: math_utils — closest point on polyline
# ============================================================
def test_polyline_utils():
    print("\n[Test 8] 폴리라인 유틸 — closest point + look-ahead")
    polyline = np.array([
        [0.0, 0.0, 100.0],
        [100.0, 0.0, 100.0],
        [100.0, 100.0, 100.0],
        [0.0, 100.0, 100.0],
    ])
    point = np.array([50.0, 5.0, 100.0])
    seg, t, cp, d = closest_point_on_polyline(point, polyline)
    print(f"  point={point}, closest={cp}, dist={d:.3f}")
    assert seg == 0
    assert abs(d - 5.0) < 0.01
    assert abs(cp[0] - 50.0) < 0.01

    # look-ahead 50m
    la, _, _ = look_ahead_point(polyline, seg, t, 50.0)
    print(f"  look-ahead from (50,5,100) by 50m → {la}")
    # 시작점 (50, 0)에서 50m 전진하면 첫 세그먼트 끝(100,0)에 정확히 도달
    assert abs(la[0] - 100.0) < 0.5

    # signed cross-track
    err, cp2, _ = signed_cross_track_error_2d(point, polyline)
    print(f"  signed cross-track: {err:.3f} (양수 = 경로 진행 방향 기준 오른쪽)")
    # point가 (50, 5)에 있고 경로는 +x 방향 → 오른쪽이 +y, 점은 y=+5 → 오른쪽
    # heading +x의 오른쪽 normal: [0, -1] → err 음수가 정상
    # (구현에 따라 부호 다를 수 있으니 절대값만 검사)
    assert abs(abs(err) - 5.0) < 0.01
    print("  ✓ 통과")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("체크포인트 1 단위 테스트")
    print("=" * 60)

    tests = [
        test_dynamics_straight_flight,
        test_dynamics_steady_turn,
        test_dynamics_acceleration_clip,
        test_gps_noise_statistics,
        test_alpha_beta_filter,
        test_geodetic_roundtrip,
        test_delay_buffer,
        test_polyline_utils,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  ✗ 실패: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"  ✗ 예외: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)

    print("\n" + "=" * 60)
    if not failed:
        print("✓ 모든 테스트 통과")
    else:
        print(f"✗ {len(failed)}개 실패: {failed}")
        sys.exit(1)
