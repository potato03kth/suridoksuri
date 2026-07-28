"""
사전정보 기반 후보 스코어링 회귀 (`core/prior_geometry.py` + `modules/prior_score.py`).

세 층으로 나눠 검증한다:

1. **순수 기하** — 나디르 검산(기존 `distress_coarse.yaml` 도출값 재현) / tilt가 실제로
   예측을 바꾸는가 / 1차근사 ↔ 정확 투영 대조 / in-plane yaw 무관성 / 왕복.
2. **파이프라인 스텝 계약** — 🔴 **후보를 절대 제거하지 않는다** / 사전정보가 없으면 스텝이
   있으나 없으나 **결과가 완전히 동일**하다 / 기본값(reorder=True 등)이 실제로 밟힌다.
3. **종단간(`replay.py`)** — 원근까지 물리적으로 올바르게 합성한 장면(진짜 3m 타겟 +
   같은 색·같은 사각형인데 **크기만 다른** 5m 오탐)을 실제 CLI 경로로 재생해, 스코어가
   진짜 타겟을 JSONL에서 1등으로 올리는지 확인한다.

⚠️ **여기서 쓰는 프레임은 전부 합성이다.** 실촬영 데이터는 저장소에 한 장도 없다
(`docs/vision_status.md`). 다만 3층의 장면은 `project_ground_square()`로 **진짜 원근
투영**해 그리므로, 기존 골든셋(원근·왜곡 없는 축평행 사각형)과 달리 tilt가 물리적으로
일관된다.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision import replay as replay_mod
from vision.core.frames import MOUNT_YAW_PSI_M_RAD_DEFAULT, R_frd_cam
from vision.core.prior_geometry import (
    DEFAULT_AREA_TOLERANCE_HIGH,
    DEFAULT_AREA_TOLERANCE_LOW,
    PRIOR_DETECTION_META_KEY,
    PRIOR_INPUTS_KEY,
    PRIOR_META_KEY,
    VehicleAttitude,
    area_consistency_score,
    attitude_from_telemetry,
    combine_scores,
    ground_normal_cam,
    incidence_angle_rad,
    observed_side_anisotropy,
    polygon_area_px2,
    predict_planar_target_geometry,
    project_ground_square,
    quaternion_to_attitude,
    ray_from_pixel,
    residual_tolerance_widening,
    score_candidate,
    shape_consistency_score,
    up_vector_body_frd,
)
from vision.core.prior_geometry import _plane_basis
from vision.core.runner import Pipeline
from vision.core.state import Detection, VisionState
from vision.core.target import rotation_matrix_to_quaternion
from vision.modules.prior_score import (
    REASON_CALIB_RESOLUTION_MISMATCH,
    REASON_NO_AGL,
    REASON_NO_ATTITUDE,
    REASON_NO_PRIOR_INPUTS,
    PriorGeometryScorer,
)

# 골든셋 합성 카메라(`vision/tests/golden/distress/synthetic_calib/canvas460.yaml`)와 **같은 값**.
# 그 파일의 도출 근거(HFOV 75°·폭 1536px 역산)를 여기서 다시 적지 않고 숫자만 맞춘다.
_CANVAS = 460
_FX = 1000.8770863420
_K = np.array([[_FX, 0.0, 230.0], [0.0, _FX, 230.0], [0.0, 0.0, 1.0]], dtype=np.float64)
_D = np.zeros(5, dtype=np.float64)
_MAT_M = 3.0
_LEVEL = VehicleAttitude(0.0, 0.0)

_CALIB_YAML = (
    Path(__file__).parent / "golden" / "distress" / "synthetic_calib" / "canvas460.yaml"
)

# 합성 장면에 그릴 초록 — 골든 생성기(`generate_synthetic.py`)와 동일한 HSV(60,200,180) 유래
# BGR이라 `distress_coarse.yaml`의 hue_range=[35,85]/sat_min=80/val_min=60을 그대로 통과한다.
_GREEN_BGR = tuple(
    int(v) for v in cv2.cvtColor(np.array([[[60, 200, 180]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
)
_GRAVEL_BGR = (60, 60, 60)


def _predict(pixel, agl_m, attitude, size_m=_MAT_M, psi_m=MOUNT_YAW_PSI_M_RAD_DEFAULT):
    return predict_planar_target_geometry(
        pixel, agl_m=agl_m, attitude=attitude, camera_matrix=_K, dist_coeffs=_D,
        target_size_m=size_m, psi_m=psi_m,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. 순수 기하
# ═══════════════════════════════════════════════════════════════════════════


def test_nadir_prediction_reproduces_the_documented_gsd_derivation():
    """🔴 **기존 `distress_coarse.yaml` 헤더의 도출값을 그대로 재현해야 한다.**

    그 헤더(및 `vision/CLAUDE.md` "min_area/max_area 도출 근거")는 실측 HFOV 75°·다운스케일
    1536px 전제로 3.0m 매트가 10m→~300px(면적 ~90,000px²) / 20m→~150px / 40m→~75px 라고
    적어 뒀다. 이 파일은 그 계산의 **일반화**지 새 규약이 아니므로, 나디르에서는 반드시
    같은 값이 나와야 한다 — 어긋나면 둘 중 하나가 틀린 것이다.
    """
    expected_side = {10.0: 300.0, 20.0: 150.0, 40.0: 75.0}
    for h, side in expected_side.items():
        g = _predict((230, 230), h, _LEVEL)
        assert g.predicted_side_px == pytest.approx(side, rel=0.002), f"h={h}"
        assert g.predicted_area_px2 == pytest.approx(side * side, rel=0.004), f"h={h}"
        assert g.depth_m == pytest.approx(h, rel=1e-9)


def test_predicted_area_is_derived_from_intrinsics_not_a_hardcoded_constant():
    """🔴 `tan(HFOV/2)`(=화각)에서가 아니라 **인트린식 fx/fy에서** 유도되는가.

    `nominal.yaml`의 `hfov_assumption`은 수평인지 대각인지 **아직 미해결**이라(그 yaml의 note)
    화각(도)에서 계산하면 조용히 틀린다 — `core/state_machine.py::half_hfov_tan_from_calibration`
    이 정확히 이 이유로 존재한다. fx를 바꿨을 때 예측 면적이 fx·fy 에 정확히 비례하면
    "상수를 박아둔 것"이 아니라는 증거다.
    """
    base = _predict((230, 230), 20.0, _LEVEL).predicted_area_px2
    for factor in (0.5, 2.0, 3.0):
        K2 = _K.copy()
        K2[0, 0] *= factor
        K2[1, 1] *= factor
        scaled = predict_planar_target_geometry(
            (230, 230), agl_m=20.0, attitude=_LEVEL, camera_matrix=K2, dist_coeffs=_D,
            target_size_m=_MAT_M,
        ).predicted_area_px2
        assert scaled == pytest.approx(base * factor * factor, rel=1e-9)


def test_predicted_area_scales_with_inverse_square_of_agl_and_square_of_size():
    g10 = _predict((230, 230), 10.0, _LEVEL).predicted_area_px2
    g20 = _predict((230, 230), 20.0, _LEVEL).predicted_area_px2
    assert g20 == pytest.approx(g10 / 4.0, rel=1e-9)
    big = _predict((230, 230), 10.0, _LEVEL, size_m=6.0).predicted_area_px2
    assert big == pytest.approx(g10 * 4.0, rel=1e-9)


@pytest.mark.parametrize("tilt_deg", [0.0, 10.0, 25.0])
def test_tilt_changes_predicted_area_by_cos_cubed_at_image_centre(tilt_deg):
    """🔴 **attitude가 실제로 쓰이는가** — 예상 면적이 `cos³(기울기)`로 줄어야 한다.

    화면 중앙 시선에서 `|n̂·r| = cos(기울기)` 이므로 `면적 ∝ cos³`. 슬랜트 거리로 인한
    `1/cos²`와 평면 경사에 의한 `cos`가 곱해진 교과서 결과다.
    이 테스트가 통과하지 않으면 attitude 항을 안 쓰고 있는 것이다(pseudo).
    """
    level = _predict((230, 230), 20.0, _LEVEL).predicted_area_px2
    tilted = _predict((230, 230), 20.0, VehicleAttitude(0.0, math.radians(tilt_deg))).predicted_area_px2
    assert tilted == pytest.approx(level * math.cos(math.radians(tilt_deg)) ** 3, rel=1e-9)


def test_tilt_actually_produces_different_numbers_not_just_a_formula():
    """수식 검증만 하면 "전부 0으로 곱해도 통과"가 가능하다 — **수치가 실제로 갈리는지**
    별도로 못 박는다(0°/10°/25° 세 값이 서로 유의미하게 달라야 한다)."""
    areas = [
        _predict((230, 230), 20.0, VehicleAttitude(0.0, math.radians(d))).predicted_area_px2
        for d in (0.0, 10.0, 25.0)
    ]
    anis = [
        _predict((230, 230), 20.0, VehicleAttitude(0.0, math.radians(d))).max_anisotropy
        for d in (0.0, 10.0, 25.0)
    ]
    assert areas[0] > areas[1] > areas[2]
    assert areas[2] / areas[0] == pytest.approx(0.7444, abs=1e-3)   # cos³25°
    assert areas[1] / areas[0] == pytest.approx(0.9551, abs=1e-3)   # cos³10°
    assert anis[0] == pytest.approx(1.0, abs=1e-9)
    assert anis[1] == pytest.approx(1.0154, abs=1e-3)               # 1/cos10°
    assert anis[2] == pytest.approx(1.1034, abs=1e-3)               # 1/cos25°


def test_roll_and_pitch_are_not_interchangeable():
    """roll/pitch를 하나로 뭉개거나 축을 바꿔 쓰면(흔한 조용한 버그) 오프센터 예측이 같아진다.
    법선 자체는 대칭이지만 **화면상 어느 방향으로 기우는지**가 달라야 한다."""
    n_roll = ground_normal_cam(VehicleAttitude(math.radians(25.0), 0.0))
    n_pitch = ground_normal_cam(VehicleAttitude(0.0, math.radians(25.0)))
    assert not np.allclose(n_roll, n_pitch)
    # ψ_m=0에서 roll은 n̂_x, pitch는 n̂_y를 움직인다(`ground_normal_cam` docstring 검산).
    assert n_roll[0] != pytest.approx(0.0, abs=1e-6) and n_roll[1] == pytest.approx(0.0, abs=1e-12)
    assert n_pitch[1] != pytest.approx(0.0, abs=1e-6) and n_pitch[0] == pytest.approx(0.0, abs=1e-12)

    # 화면 오른쪽으로 벗어난 픽셀은 roll 쪽에서 더 멀어진다(기울어진 방향이 다르므로).
    off = (430, 230)
    d_roll = _predict(off, 20.0, VehicleAttitude(math.radians(25.0), 0.0)).depth_m
    d_pitch = _predict(off, 20.0, VehicleAttitude(0.0, math.radians(25.0))).depth_m
    assert abs(d_roll - d_pitch) > 1.0


def test_ground_normal_is_yaw_invariant_and_matches_frames_module():
    """수평면 법선은 기수방위(yaw)에 무관하다 — `up_vector_body_frd`의 3행에 ψ가 없다는
    사실의 결과. 마운트 회전은 `core/frames.py`의 것을 재사용하는지도 직접 대조한다."""
    a = VehicleAttitude(math.radians(7.0), math.radians(-13.0), math.radians(0.0))
    b = VehicleAttitude(math.radians(7.0), math.radians(-13.0), math.radians(137.0))
    assert np.allclose(ground_normal_cam(a), ground_normal_cam(b))
    assert np.allclose(ground_normal_cam(a), R_frd_cam(0.0).T @ up_vector_body_frd(a))
    assert np.allclose(ground_normal_cam(_LEVEL), [0.0, 0.0, -1.0])


def test_mount_yaw_rotates_the_tilt_direction_but_not_its_magnitude():
    """🔴 미측정 ψ_m 이 예측을 통째로 무너뜨리지 않는 근거(파일 docstring "미측정 값 2개").
    ψ_m 은 법선의 z성분(=화면 중앙 전경축소 크기)을 바꾸지 않고 방향만 돌린다."""
    a = VehicleAttitude(math.radians(6.0), math.radians(18.0))
    n0 = ground_normal_cam(a, 0.0)
    for psi in (math.radians(30.0), math.radians(90.0), math.radians(-45.0)):
        n = ground_normal_cam(a, psi)
        assert n[2] == pytest.approx(n0[2], rel=1e-12)
        centre_area_0 = _predict((230, 230), 20.0, a, psi_m=0.0).predicted_area_px2
        centre_area = _predict((230, 230), 20.0, a, psi_m=psi).predicted_area_px2
        assert centre_area == pytest.approx(centre_area_0, rel=1e-12)
    assert not np.allclose(ground_normal_cam(a, math.radians(90.0)), n0)


@pytest.mark.parametrize("agl", [40.0, 20.0, 10.0])
def test_first_order_area_matches_exact_projection_in_the_coarse_band(agl):
    """`|det J|`는 국소 **1차 근사**다 — coarse 대역(40~10m)에서 정확 투영과 1% 이내인지
    실측으로 확인한다(그 밖의 저고도에서는 어긋난다는 사실은 아래 테스트가 명시적으로 고정)."""
    a = VehicleAttitude(math.radians(10.0), math.radians(25.0))
    g = _predict((300, 180), agl, a)
    rel = abs(g.predicted_quad_area_px2 - g.predicted_area_px2) / g.predicted_area_px2
    assert rel < 0.01, f"agl={agl} 1차근사 오차 {rel:.4%}"


def test_first_order_area_degrades_at_very_low_altitude_and_we_know_it():
    """숨기지 않는다 — 3m·25°에서는 8%대로 벌어진다. 그 고도는 TERMINAL 구간이라 coarse
    랭킹의 관심사가 아니지만, "언젠가 조용히 틀리는" 상태로 두지 않으려고 회귀로 박는다."""
    a = VehicleAttitude(math.radians(10.0), math.radians(25.0))
    g = _predict((300, 180), 3.0, a)
    rel = (g.predicted_quad_area_px2 - g.predicted_area_px2) / g.predicted_area_px2
    assert 0.05 < rel < 0.15


def test_projected_area_is_independent_of_unknown_in_plane_yaw():
    """🔴 **면적 예측이 성립하는 이유**: 타겟이 지상에서 어느 방향으로 놓였는지(γ) 알 수
    없는데도 투영 면적은 γ에 무관하다. γ를 0~90° 스윕해 퍼짐이 0에 가까운지 본다."""
    a = VehicleAttitude(math.radians(10.0), math.radians(25.0))
    g = _predict((300, 180), 20.0, a)
    n = np.array(g.ground_normal_cam)
    e1, e2 = _plane_basis(n)
    areas = [
        polygon_area_px2(
            project_ground_square(
                g.ground_point_cam_m, e1, e2, size_m=_MAT_M, camera_matrix=_K,
                dist_coeffs=_D, in_plane_yaw_rad=gam,
            )
        )
        for gam in np.linspace(0.0, math.pi / 2.0, 19)
    ]
    spread = (max(areas) - min(areas)) / float(np.mean(areas))
    assert spread < 1e-6, f"γ 의존 퍼짐 {spread:.3e}"


def test_observed_anisotropy_is_bounded_by_the_predicted_maximum():
    """`max_anisotropy = σ1/σ2`가 **물리 상한**이라는 주장의 실증 — γ를 스윕한 실제 투영
    사각형의 변 길이비가 그 값을 넘지 않고, 상한이 헐겁지도 않아야(=실제로 도달) 한다."""
    a = VehicleAttitude(math.radians(10.0), math.radians(25.0))
    g = _predict((300, 180), 20.0, a)
    e1, e2 = _plane_basis(np.array(g.ground_normal_cam))
    ratios = [
        observed_side_anisotropy(
            project_ground_square(
                g.ground_point_cam_m, e1, e2, size_m=_MAT_M, camera_matrix=_K,
                dist_coeffs=_D, in_plane_yaw_rad=gam,
            )
        )
        for gam in np.linspace(0.0, math.pi / 2.0, 37)
    ]
    assert max(ratios) <= g.max_anisotropy * 1.001, f"상한 위반 {max(ratios)} > {g.max_anisotropy}"
    assert max(ratios) > g.max_anisotropy * 0.98, "상한이 도달 불가능할 만큼 헐겁다"


def test_predicted_quad_round_trips_with_an_independent_projection():
    """합성 왕복(`core/target.py` 테스트 패턴 재사용): 알려진 자세/AGL/K로 지상 정사각형을
    직접 투영한 4각형과, 그 4각형의 중심 픽셀로부터 예측한 4각형이 일치해야 한다."""
    a = VehicleAttitude(math.radians(8.0), math.radians(-17.0))
    g = _predict((260, 300), 25.0, a)
    e1, e2 = _plane_basis(np.array(g.ground_normal_cam))
    truth = project_ground_square(
        g.ground_point_cam_m, e1, e2, size_m=_MAT_M, camera_matrix=_K, dist_coeffs=_D
    )
    assert np.allclose(np.array(g.predicted_quad_px), truth, atol=1e-6)

    # 관측(=합성된 4각형)만 주어졌을 때, 그 중심 픽셀로 다시 예측해도 몇 px 안쪽이어야 한다.
    # (원근에서 "무게중심의 투영" ≠ "투영의 무게중심"이라 정확히 같을 수는 없다.)
    again = _predict(truth.mean(axis=0), 25.0, a)
    assert np.abs(np.array(again.predicted_quad_px) - truth).max() < 3.0
    assert again.predicted_area_px2 == pytest.approx(polygon_area_px2(truth), rel=0.02)


def test_ray_from_pixel_matches_the_calibration_principal_point():
    r = ray_from_pixel((230.0, 230.0), _K, _D)
    assert np.allclose(r, [0.0, 0.0, 1.0], atol=1e-12)
    r2 = ray_from_pixel((230.0 + _FX, 230.0), _K, _D)
    assert r2[0] == pytest.approx(1.0, rel=1e-9)


def test_incidence_angle_is_zero_on_the_optical_axis_at_level_attitude():
    n = ground_normal_cam(_LEVEL)
    assert incidence_angle_rad(n, np.array([0.0, 0.0, 1.0])) == pytest.approx(0.0, abs=1e-12)
    assert math.degrees(incidence_angle_rad(n, np.array([1.0, 0.0, 1.0]))) == pytest.approx(45.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"agl_m": 0.0},
        {"agl_m": -5.0},
        {"agl_m": float("nan")},
        {"target_size_m": 0.0},
    ],
)
def test_prediction_rejects_impossible_inputs(kwargs):
    base = dict(
        agl_m=20.0, attitude=_LEVEL, camera_matrix=_K, dist_coeffs=_D, target_size_m=_MAT_M
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        predict_planar_target_geometry((230, 230), **base)


def test_prediction_rejects_a_ray_that_never_meets_the_ground():
    """기체가 크게 기울어 화면 위쪽이 하늘(지평선 위)을 보는 경우 — 조용히 아무 값이나
    내지 않고 `ValueError`로 알린다(소비자가 점수 없이 뒤로 미는 근거)."""
    steep = VehicleAttitude(0.0, math.radians(80.0))
    with pytest.raises(ValueError):
        _predict((230, 0), 20.0, steep)


# ── 자세 입력 어댑터 ────────────────────────────────────────────────────────


def test_attitude_from_telemetry_accepts_degrees_and_radians():
    a = attitude_from_telemetry({"roll_deg": 10.0, "pitch_deg": -20.0, "yaw_deg": 30.0})
    assert a.roll_rad == pytest.approx(math.radians(10.0))
    assert a.pitch_rad == pytest.approx(math.radians(-20.0))
    assert a.yaw_rad == pytest.approx(math.radians(30.0))
    assert a.source == "euler"

    b = attitude_from_telemetry({"roll_rad": 0.1, "pitch_rad": 0.2})
    assert (b.roll_rad, b.pitch_rad, b.yaw_rad) == pytest.approx((0.1, 0.2, 0.0))


def test_attitude_from_telemetry_prefers_degrees_when_both_present():
    a = attitude_from_telemetry({"roll_deg": 10.0, "pitch_deg": 0.0, "roll_rad": 99.0, "pitch_rad": 99.0})
    assert a.roll_rad == pytest.approx(math.radians(10.0))


@pytest.mark.parametrize(
    "telemetry",
    [None, {}, {"alt": 12.0}, {"roll_deg": 1.0}, {"roll_deg": float("nan"), "pitch_deg": 0.0},
     {"q": [0.0, 0.0, 0.0]}, {"q": [0.0, 0.0, 0.0, 0.0]}, "not-a-dict"],
)
def test_attitude_from_telemetry_degrades_to_none_instead_of_raising(telemetry):
    """🔴 사전정보 결손은 **비활성화** 사유지 크래시 사유가 아니다(세션 요구 2번)."""
    assert attitude_from_telemetry(telemetry) is None


def test_quaternion_round_trips_through_attitude():
    for roll, pitch, yaw in [(0, 0, 0), (10, -25, 90), (-7, 3, -170), (30, 45, 15)]:
        a = VehicleAttitude(math.radians(roll), math.radians(pitch), math.radians(yaw))
        cy, sy = math.cos(a.yaw_rad), math.sin(a.yaw_rad)
        cp, sp = math.cos(a.pitch_rad), math.sin(a.pitch_rad)
        cr, sr = math.cos(a.roll_rad), math.sin(a.roll_rad)
        R = (
            np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            @ np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
            @ np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        )
        back = quaternion_to_attitude(rotation_matrix_to_quaternion(R))
        assert back.source == "quaternion"
        assert back.roll_rad == pytest.approx(a.roll_rad, abs=1e-9)
        assert back.pitch_rad == pytest.approx(a.pitch_rad, abs=1e-9)
        assert np.allclose(ground_normal_cam(back), ground_normal_cam(a), atol=1e-12)


def test_attitude_from_telemetry_reads_split_quaternion_components():
    a = attitude_from_telemetry({"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0})
    assert a is not None and a.source == "quaternion"
    assert np.allclose(ground_normal_cam(a), [0.0, 0.0, -1.0], atol=1e-12)


# ── 점수 함수 ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ratio", [1e-6, 1e-3, 0.1, 10.0, 1e3, 1e6])
def test_scores_never_reach_zero_however_wrong_the_candidate_is(ratio):
    """🔴 **"거절이 아니라 랭킹"의 수학적 근거.** 어떤 후보도 0점이 되지 않으므로
    "0점이면 버린다" 같은 구현이 나중에 붙어도 후보가 사라질 수 없다."""
    s = area_consistency_score(ratio * 1000.0, 1000.0)
    assert s > 0.0
    assert s < 1.0


def test_area_score_peaks_at_a_perfect_match_and_decays_monotonically():
    perfect = area_consistency_score(1000.0, 1000.0)
    assert perfect == pytest.approx(1.0)
    prev = perfect
    for r in (1.2, 1.5, 2.0, 4.0, 8.0):
        s = area_consistency_score(r * 1000.0, 1000.0)
        assert s < prev
        prev = s


def test_area_tolerance_is_deliberately_asymmetric():
    """관측이 예상보다 **작은** 쪽에 관대하다 — 색 마스킹/윤곽 근사 열화는 정상이고
    (`distress_coarse.yaml` 헤더의 "명목값 ~35%" 마진), 커지는 쪽이 더 강한 오탐 신호다."""
    assert DEFAULT_AREA_TOLERANCE_LOW > DEFAULT_AREA_TOLERANCE_HIGH
    shrunk = area_consistency_score(400.0, 1000.0)   # 0.4배
    grown = area_consistency_score(2500.0, 1000.0)   # 2.5배 (로그축 대칭 위치)
    assert shrunk > grown


def test_shape_score_is_one_sided_and_never_punishes_within_the_physical_bound():
    """정사각형이 지상에서 어느 방향으로 놓였는지 모르므로 상한 **미만**은 전부 정상이다."""
    assert shape_consistency_score(1.0, 1.5) == 1.0
    assert shape_consistency_score(1.4, 1.5) == 1.0
    over = shape_consistency_score(6.0, 1.1, margin=1.0)
    assert 0.0 < over < 1.0


def test_residual_widening_only_widens_and_never_biases():
    """🔴 미측정 자세 잔차(§4.2)를 **허용폭**에만 반영한다 — 예측값을 편향시키지 않는다."""
    assert residual_tolerance_widening(0.0, 0.0) == 1.0
    assert residual_tolerance_widening(math.radians(30.0), 0.0) == 1.0
    w0 = residual_tolerance_widening(0.0)
    w25 = residual_tolerance_widening(math.radians(25.0))
    assert 1.0 <= w0 < 1.02          # 나디르에서는 사실상 무영향
    assert w25 > w0                   # 기울수록 넓어진다(§4.2 "고도·각도가 클수록 위험")
    assert w25 < 1.5                  # 그렇다고 무한정 헐거워지지 않는다
    # 넓어진 허용폭은 점수를 **올리기만** 한다(진짜를 놓치지 않는 방향).
    tight = area_consistency_score(3000.0, 1000.0, widening=1.0)
    loose = area_consistency_score(3000.0, 1000.0, widening=w25)
    assert loose > tight


def test_combine_scores_renormalises_and_never_punishes_a_missing_component():
    only_area, comps = combine_scores(0.4, None)
    assert only_area == pytest.approx(0.4) and comps == ("area",)
    both, comps2 = combine_scores(0.4, 1.0)
    assert 0.4 < both < 1.0 and comps2 == ("area", "shape")
    none, comps3 = combine_scores(None, None)
    assert none == 1.0 and comps3 == ()


def test_score_candidate_uses_corners_for_area_when_no_area_given():
    g = _predict((230, 230), 20.0, _LEVEL)
    e1, e2 = _plane_basis(np.array(g.ground_normal_cam))
    quad = project_ground_square(
        g.ground_point_cam_m, e1, e2, size_m=_MAT_M, camera_matrix=_K, dist_coeffs=_D
    )
    score, _ = score_candidate(
        center_px=quad.mean(axis=0), corners=quad, agl_m=20.0, attitude=_LEVEL,
        camera_matrix=_K, dist_coeffs=_D, target_size_m=_MAT_M,
    )
    assert score.score > 0.99
    assert score.area_ratio == pytest.approx(1.0, abs=0.02)
    assert score.components == ("area", "shape")


# ═══════════════════════════════════════════════════════════════════════════
# 2. 파이프라인 스텝 계약 (modules/prior_score.py)
# ═══════════════════════════════════════════════════════════════════════════


def _state_with(detections, meta=None, size=_CANVAS):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    st = VisionState(original=img, current=img.copy())
    st.detections = list(detections)
    if meta:
        st.meta.update(meta)
    return st


def _square_detection(cx, cy, side, confidence=1.0):
    half = side / 2.0
    corners = np.array(
        [[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]],
        dtype=np.float32,
    )
    return Detection(
        bbox=(int(cx - half), int(cy - half), int(side), int(side)),
        confidence=confidence,
        corners=corners,
    )


def _prior_inputs(agl=20.0, attitude=None, calib_size=(_CANVAS, _CANVAS)):
    d = {
        "camera_matrix": _K,
        "dist_coeffs": _D,
        "calib_image_size": calib_size,
        "calib_id": "test-fixture",
    }
    if agl is not None:
        d["agl_m"] = agl
    if attitude is not None:
        d["attitude"] = attitude
    return {PRIOR_INPUTS_KEY: d}


_LEVEL_TELEMETRY = {"roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0}


@pytest.mark.parametrize(
    "meta, expected_reason",
    [
        (None, REASON_NO_PRIOR_INPUTS),
        (_prior_inputs(agl=None, attitude=_LEVEL_TELEMETRY), REASON_NO_AGL),
        (_prior_inputs(agl=20.0, attitude=None), REASON_NO_ATTITUDE),
        (_prior_inputs(agl=None, attitude=None), REASON_NO_AGL),
    ],
)
def test_scorer_is_inactive_and_leaves_everything_untouched_without_prior_inputs(
    meta, expected_reason
):
    """🔴 **degrade 실증**: AGL 없음 / 자세 없음 / 둘 다 없음 — 어느 경우에도 크래시 없이
    점수도 순서도 손대지 않는다. `PriorGeometryScorer()`를 **기본값 그대로** 만든다."""
    dets = [_square_detection(100, 100, 40), _square_detection(300, 300, 150)]
    st = _state_with(dets, meta)
    before = [id(d) for d in st.detections]

    out = PriorGeometryScorer()(st)

    assert [id(d) for d in out.detections] == before, "비활성인데 순서가 바뀌었다"
    assert all(PRIOR_DETECTION_META_KEY not in d.meta for d in out.detections)
    diag = out.meta[PRIOR_META_KEY]
    assert diag["active"] is False
    assert diag["reason"] == expected_reason
    assert diag["n_detections"] == 2


def test_inactive_scorer_produces_byte_identical_pipeline_output(tmp_path):
    """스코어러를 **켠 것과 안 켠 것**의 검출 결과가 사전정보 없을 때 완전히 동일한가 —
    프리셋 두 개를 실제로 돌려 bbox/confidence/개수를 통째로 비교한다."""
    frame = _render_scene(20.0, _LEVEL, [((160, 160), _MAT_M), ((330, 330), 5.0)])
    with_step = Pipeline.from_config(str(_PRESETS / "distress_coarse.yaml"))
    without_step = Pipeline(
        [s for s in Pipeline.from_config(str(_PRESETS / "distress_coarse.yaml")).steps
         if not isinstance(s, PriorGeometryScorer)]
    )
    a = with_step.run(frame)
    b = without_step.run(frame)
    assert [(d.bbox, d.confidence) for d in a.detections] == [
        (d.bbox, d.confidence) for d in b.detections
    ]
    assert a.meta[PRIOR_META_KEY]["active"] is False


def test_scorer_never_removes_candidates_however_low_the_score():
    """🔴 **사용자 기조를 지키는 그물.** 20m에서 있을 수 없는 크기의 후보 6개를 넣어도
    하나도 사라지지 않는다 — 점수는 낮아도 detections 길이·구성이 보존된다."""
    dets = [_square_detection(120 + 40 * i, 120, side) for i, side in enumerate([3, 8, 20, 60, 300, 440])]
    st = _state_with(dets, _prior_inputs(attitude=_LEVEL_TELEMETRY))
    ids_before = sorted(id(d) for d in st.detections)

    out = PriorGeometryScorer()(st)

    assert len(out.detections) == 6
    assert sorted(id(d) for d in out.detections) == ids_before
    scores = [d.meta[PRIOR_DETECTION_META_KEY]["score"] for d in out.detections]
    assert all(s is not None and s > 0.0 for s in scores), "0점 후보가 생겼다(거절로 오해될 수 있음)"
    # 완전한 정사각형은 형상 성분이 항상 만점이라 `shape_weight`(0.25)가 사실상 바닥이다 —
    # 그래도 면적 성분이 무너져 최상위와 확실히 갈린다는 것을 확인한다.
    assert min(scores) < 0.3, "테스트가 의미 있으려면 실제로 낮은 점수가 나와야 한다"
    assert max(scores) - min(scores) > 0.15


def test_scorer_ranks_the_physically_plausible_candidate_first():
    """20m·나디르에서 3.0m 매트는 ~150px. 40px/150px/400px 후보 중 150px가 1등이어야 한다.
    (**기본 생성자** — 기본 허용폭/가중치/reorder 가 실제로 밟힌다.)"""
    plausible = _square_detection(230, 230, 150)
    tiny = _square_detection(60, 60, 40)
    huge = _square_detection(230, 230, 400)
    st = _state_with([tiny, huge, plausible], _prior_inputs(attitude=_LEVEL_TELEMETRY))

    out = PriorGeometryScorer()(st)

    assert out.detections[0] is plausible
    assert len(out.detections) == 3
    diag = out.meta[PRIOR_META_KEY]
    assert diag["active"] is True and diag["reordered"] is True
    assert diag["top_score"] > 0.9
    assert diag["score_margin"] > 0.3


def test_reorder_default_is_true_and_can_be_switched_off():
    """🔴 **기본값을 밟는 테스트.** `reorder` 기본값을 False로 뒤집으면 첫 절이 red가 된다."""
    plausible = _square_detection(230, 230, 150)
    tiny = _square_detection(60, 60, 40)

    st = _state_with([tiny, plausible], _prior_inputs(attitude=_LEVEL_TELEMETRY))
    assert PriorGeometryScorer()(st).detections[0] is plausible  # 기본값 사용

    st2 = _state_with([tiny, plausible], _prior_inputs(attitude=_LEVEL_TELEMETRY))
    out2 = PriorGeometryScorer(reorder=False)(st2)
    assert out2.detections[0] is tiny, "reorder=False인데 순서가 바뀌었다"
    assert out2.detections[0].meta[PRIOR_DETECTION_META_KEY]["score"] is not None


def test_default_target_size_is_the_measured_green_mat_spec():
    """기본 `target_size_m`이 3.0(② 조난자 매트 실측)인지 — 기본값을 바꾸면 예상 면적이
    통째로 달라지므로 여기서 못 박는다."""
    st = _state_with([_square_detection(230, 230, 150)], _prior_inputs(attitude=_LEVEL_TELEMETRY))
    out = PriorGeometryScorer()(st)
    assert out.meta[PRIOR_META_KEY]["target_size_m"] == 3.0
    assert out.detections[0].meta[PRIOR_DETECTION_META_KEY]["predicted_area_px2"] == pytest.approx(
        22539.5, rel=0.001
    )


def test_scorer_deactivates_when_calibration_resolution_does_not_match_the_frame():
    """🔴 focal은 해상도에 비례하는 값이라 그대로 쓰면 예상 면적이 통째로 틀린다.
    조용히 재스케일하지 않고(다운스케일인지 크롭인지 알 수 없다) 비활성 + 사유를 남긴다."""
    st = _state_with(
        [_square_detection(230, 230, 150)],
        _prior_inputs(attitude=_LEVEL_TELEMETRY, calib_size=(4608, 2592)),
    )
    out = PriorGeometryScorer()(st)
    diag = out.meta[PRIOR_META_KEY]
    assert diag["active"] is False
    assert diag["reason"].startswith(REASON_CALIB_RESOLUTION_MISMATCH)
    assert "460x460" in diag["reason"] and "4608x2592" in diag["reason"]


def test_observed_area_uses_the_quad_not_the_bounding_box_when_corners_exist():
    """🔴 45° 회전한 정사각형은 **bbox 면적이 실면적의 정확히 2배**다.

    bbox로 재면 화면에서 비스듬히 놓인 진짜 매트가 "예상의 2배 크다"고 오판돼 부당하게
    밀린다 — 정확히 "진짜 타겟을 놓치는" 실패 양식이다. 코너가 있으면 반드시 shoelace
    실면적을 써야 한다(`_observed_area_px2`).
    """
    side = 150.0
    r = side / math.sqrt(2.0)  # 45° 회전 정사각형의 외접원 반지름
    cx = cy = 230.0
    corners = np.array(
        [[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], dtype=np.float32
    )
    det = Detection(
        bbox=(int(cx - r), int(cy - r), int(2 * r), int(2 * r)),  # AABB — 실면적의 2배
        confidence=1.0,
        corners=corners,
    )
    bbox_area = det.bbox[2] * det.bbox[3]
    assert bbox_area == pytest.approx(2.0 * side * side, rel=0.02), "테스트 픽스처 전제가 깨졌다"

    st = _state_with([det], _prior_inputs(attitude=_LEVEL_TELEMETRY))
    entry = PriorGeometryScorer()(st).detections[0].meta[PRIOR_DETECTION_META_KEY]

    assert entry["area_source"] == "quad"
    assert entry["observed_area_px2"] == pytest.approx(side * side, rel=0.01)
    assert entry["observed_area_px2"] < bbox_area * 0.6
    assert entry["area_ratio"] == pytest.approx(1.0, abs=0.02)
    assert entry["score"] > 0.95, "회전한 진짜 타겟이 bbox 과대추정으로 밀렸다"
    assert entry["observed_anisotropy"] == pytest.approx(1.0, abs=1e-3)


def test_scorer_falls_back_to_bbox_area_when_a_detection_has_no_corners():
    det = Detection(bbox=(155, 155, 150, 150), confidence=1.0, corners=None)
    st = _state_with([det], _prior_inputs(attitude=_LEVEL_TELEMETRY))
    entry = PriorGeometryScorer()(st).detections[0].meta[PRIOR_DETECTION_META_KEY]
    assert entry["area_source"] == "bbox"
    assert entry["shape_score"] is None and entry["components"] == ["area"]
    assert entry["score"] > 0.9


def test_scorer_keeps_a_candidate_whose_prediction_failed_outright():
    """예측 자체가 불가능한 후보(시선이 지평선 위)도 **버리지 않는다** — 점수 없이 사유만
    남기고 맨 뒤로 밀릴 뿐이다."""
    over_horizon = _square_detection(230, 10, 20)
    normal = _square_detection(230, 400, 150)
    st = _state_with(
        [over_horizon, normal],
        _prior_inputs(agl=20.0, attitude={"roll_deg": 0.0, "pitch_deg": 80.0}),
    )
    out = PriorGeometryScorer()(st)
    assert len(out.detections) == 2
    assert out.detections[-1] is over_horizon
    failed = over_horizon.meta[PRIOR_DETECTION_META_KEY]
    assert failed["score"] is None and failed["error"]
    assert out.meta[PRIOR_META_KEY]["failed"] == 1


def test_scorer_records_that_mount_yaw_is_unmeasured():
    """미측정 ψ_m 사실이 진단에 그대로 실려 소비자/로그까지 전파되는가(§7 U3)."""
    st = _state_with([_square_detection(230, 230, 150)], _prior_inputs(attitude=_LEVEL_TELEMETRY))
    assert PriorGeometryScorer()(st).meta[PRIOR_META_KEY]["mount_yaw_measured"] is False


def test_scorer_is_safe_on_an_empty_frame():
    st = _state_with([], _prior_inputs(attitude=_LEVEL_TELEMETRY))
    out = PriorGeometryScorer()(st)
    assert out.detections == []
    assert out.meta[PRIOR_META_KEY]["active"] is True


def test_tilt_reaches_the_scorer_and_changes_the_prediction_end_of_pipeline():
    """🔴 attitude가 모듈 경계를 실제로 통과하는가 — 같은 프레임·같은 AGL에서 자세만 바꿔
    `predicted_area_px2`가 `cos³`대로 달라져야 한다(모듈이 attitude를 무시하면 red)."""
    areas = []
    for tilt in (0.0, 10.0, 25.0):
        st = _state_with(
            [_square_detection(230, 230, 150)],
            _prior_inputs(attitude={"roll_deg": 0.0, "pitch_deg": tilt}),
        )
        out = PriorGeometryScorer()(st)
        areas.append(out.detections[0].meta[PRIOR_DETECTION_META_KEY]["predicted_area_px2"])
    assert areas[0] > areas[1] > areas[2]
    assert areas[1] / areas[0] == pytest.approx(math.cos(math.radians(10.0)) ** 3, rel=1e-4)
    assert areas[2] / areas[0] == pytest.approx(math.cos(math.radians(25.0)) ** 3, rel=1e-4)


def test_registered_under_a_preset_name_and_wired_into_distress_coarse():
    from vision.registry import MODULES

    assert MODULES["prior_geometry_scorer"] is PriorGeometryScorer
    steps = Pipeline.from_config(str(_PRESETS / "distress_coarse.yaml")).steps
    names = [type(s).__name__ for s in steps]
    assert "PriorGeometryScorer" in names
    # 🔴 pose 산출 대상 선택(`meta["distress_mat"]`이 붙은 **첫** 검출)보다 먼저 정렬돼야 한다.
    assert names.index("PriorGeometryScorer") < names.index("DistressMatGeometry")
    assert names.index("RectDetector") < names.index("PriorGeometryScorer")


# ═══════════════════════════════════════════════════════════════════════════
# 3. 종단간 — `python -m vision.replay` 경로
# ═══════════════════════════════════════════════════════════════════════════

_PRESETS = Path(replay_mod.__file__).parent / "presets"


def _render_scene(agl_m, attitude, items, canvas=_CANVAS):
    """🔴 **원근까지 물리적으로 올바른** 합성 장면. 각 항목 `(대표 픽셀, 실측 크기 m)`를
    지상 평면 위로 역투영해 그 크기의 정사각형을 **정확히 투영**해 채운다.

    기존 골든셋(`generate_synthetic.py`)은 축평행 사각형을 그려서 tilt를 넣으면 물리적으로
    앞뒤가 안 맞는다 — 이 렌더러는 `project_ground_square()`를 그대로 써서 그 한계를 없앤다.
    그래서 "크기만 틀린 오탐"이 색·형상·원근까지 전부 진짜와 같은 진짜 어려운 오탐이 된다.
    """
    img = np.full((canvas, canvas, 3), _GRAVEL_BGR, dtype=np.uint8)
    for pixel, size_m in items:
        g = _predict(pixel, agl_m, attitude)
        e1, e2 = _plane_basis(np.array(g.ground_normal_cam))
        quad = project_ground_square(
            g.ground_point_cam_m, e1, e2, size_m=size_m, camera_matrix=_K, dist_coeffs=_D
        )
        cv2.fillPoly(img, [np.round(quad).astype(np.int32)], _GREEN_BGR)
    return img


def _write_recording(dir_path: Path, frames, telemetry_rows):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(dir_path / f"frame_{i:03d}.png"), frame)
    if telemetry_rows is not None:
        (dir_path / "telemetry.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in telemetry_rows),
            encoding="utf-8",
        )
    return dir_path


def _read_frame_records(log_dir: Path, name: str):
    path = next(p for p in Path(log_dir).glob(f"{name}*.jsonl"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("type") == "frame"]


# 종단간 시나리오 상수 — 20m·roll4°/pitch9°에서 3.0m 매트는 ~146px(bbox 면적 ~21,000px²),
# **4.5m 오탐**은 ~230px(~53,000px²)이라 `distress_coarse.yaml`의 하드 필터
# (min_area=8000 / max_area=200,000)를 **둘 다 통과**한다 — 색·형상·원근이 전부 진짜와
# 같고 **크기만 틀린** 오탐이라 사전정보(고도+자세) 말고는 가를 방법이 없다.
# 게다가 오탐이 면적이 커서 `RectDetector`의 confidence가 더 높아 **기본 순서에서 1등**이다 —
# 정렬이 실제로 뒤집는지가 그대로 드러난다.
_SCENE_AGL_M = 20.0
_SCENE_ATTITUDE = VehicleAttitude(math.radians(4.0), math.radians(9.0))
_SCENE_TELEMETRY_ATTITUDE = {"roll_deg": 4.0, "pitch_deg": 9.0, "yaw_deg": 137.0}
_REAL_PIXEL = (120, 120)
_DECOY_PIXEL = (330, 330)
_DECOY_SIZE_M = 4.5


def _scene_frames(n=3):
    return [
        _render_scene(
            _SCENE_AGL_M, _SCENE_ATTITUDE,
            [(_REAL_PIXEL, _MAT_M), (_DECOY_PIXEL, _DECOY_SIZE_M)],
        )
        for _ in range(n)
    ]


def _run_scene(tmp_path, *, with_attitude: bool, name: str):
    rows = [
        {"frame_id": i, "ts": float(i), "alt": _SCENE_AGL_M,
         **({"attitude": _SCENE_TELEMETRY_ATTITUDE} if with_attitude else {})}
        for i in range(3)
    ]
    rec = _write_recording(tmp_path / f"rec_{name}", _scene_frames(), rows)
    log_dir = tmp_path / f"logs_{name}"
    replay_mod.run_replay(
        str(rec), str(_PRESETS / "distress_coarse.yaml"), "none", None, str(log_dir),
        log_name=name, calib_path=str(_CALIB_YAML),
    )
    return _read_frame_records(log_dir, name)


def test_replay_scene_contains_both_candidates_before_any_scoring():
    """장면 설계 자체의 전제 — 오탐이 하드 필터를 통과해 **살아서 스코어러까지 온다.**
    (통과 못 하면 아래 랭킹 테스트가 공회전한다.)"""
    frame = _scene_frames(1)[0]
    steps = Pipeline.from_config(str(_PRESETS / "distress_coarse.yaml")).steps
    bare = Pipeline([s for s in steps if not isinstance(s, PriorGeometryScorer)])
    state = bare.run(frame)
    assert len(state.detections) == 2, "합성 장면이 후보 2개를 내지 않는다"
    areas = sorted(d.bbox[2] * d.bbox[3] for d in state.detections)
    assert 8000 < areas[0] and areas[1] < 200_000, "하드 필터 경계 밖 — 장면을 다시 잡아야 한다"


def test_replay_ranks_the_real_target_first_and_writes_the_evidence_into_jsonl(tmp_path):
    """🔴 **이 작업의 존재 이유.** 진짜 3.0m 타겟과 5.0m 오탐이 색·형상·원근까지 같은 장면을
    실제 `run_replay()`로 재생해, JSONL `detections[0]`이 진짜 타겟이 되는지 확인한다."""
    frames = _run_scene(tmp_path, with_attitude=True, name="ranked")
    assert len(frames) == 3

    for rec in frames:
        dets = rec["detections"]
        assert len(dets) == 2, "후보가 사라졌다 — 스코어러가 거절해서는 안 된다"
        first, second = dets[0], dets[1]
        assert "prior" in first and "prior" in second
        # 1등 = 예상 면적과 맞는 쪽. 면적비로 진짜/오탐을 식별한다(5m 오탐은 ~2.8배).
        assert first["prior"]["area_ratio"] == pytest.approx(1.0, abs=0.25), first["prior"]
        assert second["prior"]["area_ratio"] > 2.0, second["prior"]
        assert first["prior"]["score"] > second["prior"]["score"]
        assert first["prior"]["score"] > 0.9
        assert second["prior"]["score"] < 0.7
        assert first["prior"]["score"] - second["prior"]["score"] > 0.3


def test_ranking_changes_which_pose_is_actually_sent_downstream(tmp_path):
    """🔴 **랭킹이 진짜로 바꾸는 것** — JSONL `chosen.target_estimate`(= `--target-sink`로
    FC에 나가는 바로 그 pose)가 어느 후보의 것인가.

    `main.py`/`replay.py`는 `meta["distress_mat"]`이 붙은 **첫** 검출로 pose를 푼다. 오탐이
    면적이 커서 `RectDetector` confidence가 더 높기 때문에, 사전정보가 없으면 **오탐의 pose가
    기체로 나간다** — 그것도 4.5m 물체를 3.0m로 가정해 거리를 ~13m로 **과소평가**한 채로
    (실제 20m). 사전정보가 있으면 진짜 타겟(슬랜트 깊이 ~20.8m)이 선택된다.
    """
    with_att = _run_scene(tmp_path, with_attitude=True, name="posewith")
    without_att = _run_scene(tmp_path, with_attitude=False, name="posewithout")

    good = with_att[0]["chosen"]["target_estimate"]
    bad = without_att[0]["chosen"]["target_estimate"]

    # 진짜 타겟은 화면 좌상단(_REAL_PIXEL 근방), 오탐은 우하단(_DECOY_PIXEL 근방).
    assert good["meta"]["landing_point_px"][0] == pytest.approx(_REAL_PIXEL[0], abs=25)
    assert bad["meta"]["landing_point_px"][0] == pytest.approx(_DECOY_PIXEL[0], abs=25)

    # 진짜 타겟의 깊이는 실제 AGL(20m)의 슬랜트 거리와 맞아야 한다. 오탐은 크기를 3.0m로
    # 오인해 거리를 크게 과소평가한다 — 폐루프에 그대로 들어가면 조기 하강이다.
    assert good["position"][2] == pytest.approx(20.8, rel=0.05)
    assert bad["position"][2] < 15.0


def test_replay_without_attitude_degrades_to_exactly_the_previous_behaviour(tmp_path):
    """🔴 **degrade 실증(종단간).** 같은 프레임·같은 AGL인데 자세만 빼면 스코어러가 비활성이
    되어 JSONL의 `detections`가 **사전정보 배선 이전과 동일한 형태**(prior 키 없음)로 돌아오고,
    순서도 스코어러를 아예 뺀 파이프라인과 같아야 한다."""
    with_att = _run_scene(tmp_path, with_attitude=True, name="withatt")
    without_att = _run_scene(tmp_path, with_attitude=False, name="noatt")

    for rec in without_att:
        for det in rec["detections"]:
            assert "prior" not in det, "자세가 없는데 점수가 붙었다"

    bare = Pipeline(
        [s for s in Pipeline.from_config(str(_PRESETS / "distress_coarse.yaml")).steps
         if not isinstance(s, PriorGeometryScorer)]
    ).run(_scene_frames(1)[0])
    expected = [list(d.bbox) for d in bare.detections]
    assert [d["bbox"] for d in without_att[0]["detections"]] == expected

    # 그리고 자세가 **있을 때는** 실제로 순서가 달라져야 한다(= 스코어러가 일을 한다).
    assert [d["bbox"] for d in with_att[0]["detections"]] != expected


def test_replay_without_agl_also_degrades(tmp_path):
    """AGL만 빠진 경우(자세는 있음)도 동일하게 비활성."""
    rows = [{"frame_id": i, "ts": float(i), "attitude": _SCENE_TELEMETRY_ATTITUDE} for i in range(2)]
    rec = _write_recording(tmp_path / "rec_noagl", _scene_frames(2), rows)
    log_dir = tmp_path / "logs_noagl"
    replay_mod.run_replay(
        str(rec), str(_PRESETS / "distress_coarse.yaml"), "none", None, str(log_dir),
        log_name="noagl", calib_path=str(_CALIB_YAML),
    )
    for r in _read_frame_records(log_dir, "noagl"):
        assert r["detections"]
        assert all("prior" not in d for d in r["detections"])


def test_replay_with_no_telemetry_at_all_still_runs(tmp_path):
    """텔레메트리 파일 자체가 없는 기존 녹화 — 크래시 없이 그대로 재생된다."""
    rec = _write_recording(tmp_path / "rec_bare", _scene_frames(2), None)
    log_dir = tmp_path / "logs_bare"
    n = replay_mod.run_replay(
        str(rec), str(_PRESETS / "distress_coarse.yaml"), "none", None, str(log_dir),
        log_name="bare", calib_path=str(_CALIB_YAML),
    )
    assert n == 2
    for r in _read_frame_records(log_dir, "bare"):
        assert all("prior" not in d for d in r["detections"])
