"""`vision/core/target.py` 단위테스트 — 하드웨어 의존 없음.

ArUco 브랜치 Phase 3(`docs/vision_aruco_branch.md` §Phase 3). 핵심 검증 방법론은
`vision/tools/calib_analyze.py`의 "★합성 왕복" 패턴(진짜 K/dist·진짜 pose를 알고 합성
투영 → 복원값이 원래값과 허용오차 내 일치) 그대로 재사용한다 — 여기서는 "진짜 pose"를
알고 합성 투영한 4코너 → solvePnP 복원 pose가 원래 pose와 일치하는지 검증한다.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.core.target import (
    ARUCO_TARGET_SIZE_M,
    DISTRESS_MAT_PLATFORM_HEIGHT_M,
    DISTRESS_MAT_SIZE_M,
    TargetEstimate,
    marker_object_points,
    order_quad_corners_clockwise,
    project_pixel_onto_target_plane,
    rotation_matrix_to_quaternion,
    solve_target_pose,
)
from vision.utils.calibration_loader import load_camera_calibration

_REAL_NOMINAL_YAML = (
    Path(__file__).parent.parent / "calibration" / "cam109-imx708af75" / "nominal.yaml"
)


# ===========================================================================
# 헬퍼 (테스트 전용 — quaternion 정확성 자체검증에만 쓰임, 프로덕션 코드 아님)
# ===========================================================================


def _quat_to_matrix(q: tuple) -> np.ndarray:
    """quaternion (x,y,z,w) -> 3x3 회전행렬. 표준 공식(교차검증용, `rotation_matrix_to_quaternion`
    구현과 독립적으로 손으로 다시 적은 것 — 같은 코드로 자기 자신을 검증하지 않기 위함)."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quats_equal_up_to_sign(q1: tuple, q2: tuple, atol: float = 1e-6) -> bool:
    """quaternion q와 -q는 같은 회전을 나타낸다(이중피복) — 부호 이중성 고려한 비교."""
    a, b = np.asarray(q1, dtype=np.float64), np.asarray(q2, dtype=np.float64)
    return bool(np.allclose(a, b, atol=atol) or np.allclose(a, -b, atol=atol))


# ===========================================================================
# marker_object_points — 코너 순서/크기(확정 전제 §1)
# ===========================================================================


def test_marker_object_points_default_size_is_50cm_square():
    pts = marker_object_points()
    assert ARUCO_TARGET_SIZE_M == pytest.approx(0.50)
    expected = np.array(
        [[-0.25, -0.25, 0.0], [0.25, -0.25, 0.0], [0.25, 0.25, 0.0], [-0.25, 0.25, 0.0]]
    )
    assert pts.shape == (4, 3)
    assert np.allclose(pts, expected)


def test_marker_object_points_scales_with_size():
    pts = marker_object_points(size_m=1.0)
    expected = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
    assert np.allclose(pts, expected)


# ===========================================================================
# rotation_matrix_to_quaternion — 순수 numpy 구현 정확성(scipy 미사용)
# ===========================================================================


def test_identity_rotation_gives_identity_quaternion():
    q = rotation_matrix_to_quaternion(np.eye(3))
    assert q == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-9)


@pytest.mark.parametrize(
    "rvec",
    [
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [np.pi / 2, 0.0, 0.0],
        [0.0, np.pi / 2, 0.0],
        [0.0, 0.0, np.pi / 2],
        [1.0, 1.0, 1.0],          # 다축 복합 회전 (~99°)
        [3.0, 0.0, 0.0],          # 큰 각도(~171.9°) — trace가 작아지는(음수 근접) 분기 커버
        [2.0, -1.5, 0.5],
        [-0.3, 2.5, -1.1],
    ],
)
def test_rotation_matrix_to_quaternion_round_trips_for_various_rotations(rvec):
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
    q = rotation_matrix_to_quaternion(R)

    # 단위 quaternion
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)
    # 독립적으로 다시 구현한 quat->R 공식으로 원래 R을 복원하는지 (자기검증 회피)
    R_recovered = _quat_to_matrix(q)
    assert np.allclose(R, R_recovered, atol=1e-9)


# ===========================================================================
# ★ 합성 왕복(핵심) — 알려진 실제 pose로 투영 -> solvePnP 복원이 원래 pose와 일치
# ===========================================================================


def _project_marker_corners(
    rvec: np.ndarray, tvec: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
    object_points: np.ndarray,
) -> np.ndarray:
    image_points, _ = cv2.projectPoints(
        object_points.reshape(-1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    return image_points.reshape(-1, 2).astype(np.float32)


def test_synthetic_round_trip_recovers_known_pose_with_synthetic_intrinsics():
    camera_matrix = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    dist_coeffs = np.zeros(5)
    obj_points = marker_object_points()

    true_rvec = np.array([0.15, -0.25, 0.05])
    true_tvec = np.array([[0.08], [-0.05], [2.2]])
    image_points = _project_marker_corners(true_rvec, true_tvec, camera_matrix, dist_coeffs, obj_points)

    estimate = solve_target_pose(
        image_points, camera_matrix, dist_coeffs,
        object_points=obj_points, target_type="aruco_23", frame_id=42, timestamp=123.456,
        calib_accuracy="unverified", not_for_closed_loop_30cm=True,
        calib_id="synthetic-test",
    )

    true_position = tuple(true_tvec.reshape(-1).tolist())
    R_true, _ = cv2.Rodrigues(true_rvec)
    true_quat = rotation_matrix_to_quaternion(R_true)

    assert estimate.position == pytest.approx(true_position, abs=1e-4)
    assert _quats_equal_up_to_sign(estimate.orientation, true_quat, atol=1e-4)
    assert estimate.target_type == "aruco_23"
    assert estimate.frame_id == 42
    assert estimate.timestamp == pytest.approx(123.456)


def test_synthetic_round_trip_with_real_nominal_yaml_intrinsics():
    """실제 Phase 1 산출물(nominal.yaml)을 로더로 읽어 그대로 solvePnP에 넣는 end-to-end 확인."""
    assert _REAL_NOMINAL_YAML.exists(), f"Phase 1 산출물이 없음: {_REAL_NOMINAL_YAML}"
    calib = load_camera_calibration(_REAL_NOMINAL_YAML)
    obj_points = marker_object_points()

    true_rvec = np.array([0.05, 0.1, -0.2])
    true_tvec = np.array([[0.3], [-0.2], [8.0]])  # 8m 상공 근사
    image_points = _project_marker_corners(
        true_rvec, true_tvec, calib.camera_matrix, calib.dist_coeffs, obj_points
    )

    estimate = solve_target_pose(
        image_points, calib.camera_matrix, calib.dist_coeffs,
        object_points=obj_points, target_type="aruco_23", frame_id=1, timestamp=0.0,
        calib_accuracy=calib.accuracy, not_for_closed_loop_30cm=calib.not_for_closed_loop_30cm,
        calib_id=calib.calib_id,
    )

    true_position = tuple(true_tvec.reshape(-1).tolist())
    R_true, _ = cv2.Rodrigues(true_rvec)
    true_quat = rotation_matrix_to_quaternion(R_true)

    assert estimate.position == pytest.approx(true_position, abs=1e-3)
    assert _quats_equal_up_to_sign(estimate.orientation, true_quat, atol=1e-3)


# ===========================================================================
# provenance echo(§7.3) — calib_accuracy/not_for_closed_loop_30cm/calib_id가 실제로 채워지는지
# ===========================================================================


def _make_default_estimate(**overrides) -> TargetEstimate:
    camera_matrix = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    dist_coeffs = np.zeros(5)
    obj_points = marker_object_points()
    true_rvec = np.array([0.0, 0.0, 0.0])
    true_tvec = np.array([[0.0], [0.0], [2.0]])
    image_points = _project_marker_corners(true_rvec, true_tvec, camera_matrix, dist_coeffs, obj_points)
    kwargs = dict(
        target_type="aruco_23", frame_id=1, timestamp=0.0,
        calib_accuracy="unverified", not_for_closed_loop_30cm=True, calib_id="",
    )
    kwargs.update(overrides)
    return solve_target_pose(image_points, camera_matrix, dist_coeffs, object_points=obj_points, **kwargs)


def test_provenance_fields_are_echoed_from_caller():
    estimate = _make_default_estimate(
        calib_accuracy="unverified",
        not_for_closed_loop_30cm=True,
        calib_id="vision/calibration/cam109-imx708af75/nominal.yaml",
    )
    assert estimate.calib_accuracy == "unverified"
    assert estimate.not_for_closed_loop_30cm is True
    assert estimate.calib_id == "vision/calibration/cam109-imx708af75/nominal.yaml"


def test_provenance_fields_reflect_verified_calib_when_passed():
    """미래(실측 캘리브레이션 재개 후) accuracy="verified"/not_for_closed_loop_30cm=False가
    와도 하드코딩되지 않고 그대로 반영되는지."""
    estimate = _make_default_estimate(
        calib_accuracy="verified", not_for_closed_loop_30cm=False, calib_id="verified.yaml",
    )
    assert estimate.calib_accuracy == "verified"
    assert estimate.not_for_closed_loop_30cm is False
    assert estimate.calib_id == "verified.yaml"


# ===========================================================================
# uncertainty — 이번 Phase는 항상 None(자리만 확정)
# ===========================================================================


def test_uncertainty_is_always_none():
    estimate = _make_default_estimate()
    assert estimate.uncertainty is None

    estimate2 = _make_default_estimate(confidence=0.3)
    assert estimate2.uncertainty is None


# ===========================================================================
# 코너 순서 불일치 — 순서가 실제로 중요함을 회귀로 증명
# ===========================================================================


def test_corner_order_mismatch_breaks_round_trip():
    """코너 순서가 실제로 중요함을 두 가지 방식의 오배열로 증명한다.

    **실측으로 확인한 흥미로운 사실(둘 다 실제로 돌려서 확인함, 직관과 다름):**
    정사각형은 그 중심을 지나는 자기 법선축 기준 90도 회전에 대해 대칭이다 — 그래서 코너를
    "1칸 순환"시키는 오배열은 물리적으로 "마커를 자기 축으로 90도 돌린 것"과 동일한 합인
    (perfectly-fitting) 해로 수렴한다: **position(중심 위치)은 거의 완벽히 보존되고
    orientation만 깨진다.** 반대로 순환이 아닌 오배열(예: 인접 코너 한 쌍만 맞바꿈)은 어떤
    강체 회전으로도 설명할 수 없어 **position 자체가 크게 깨진다.** 두 실패 모드를 각각
    검증해 "순서가 중요하다"를 position/orientation 양쪽에서 증명한다.
    """
    camera_matrix = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    dist_coeffs = np.zeros(5)
    obj_points = marker_object_points()

    true_rvec = np.array([0.15, -0.25, 0.05])   # 비대칭 회전(90도 배수 아님)
    true_tvec = np.array([[0.08], [-0.05], [2.2]])
    correct_image_points = _project_marker_corners(true_rvec, true_tvec, camera_matrix, dist_coeffs, obj_points)
    true_position = tuple(true_tvec.reshape(-1).tolist())
    R_true, _ = cv2.Rodrigues(true_rvec)
    true_quat = rotation_matrix_to_quaternion(R_true)

    good_estimate = solve_target_pose(
        correct_image_points, camera_matrix, dist_coeffs, object_points=obj_points,
        target_type="aruco_23", frame_id=1, timestamp=0.0,
    )
    assert good_estimate.position == pytest.approx(true_position, abs=1e-4)
    assert _quats_equal_up_to_sign(good_estimate.orientation, true_quat, atol=1e-4)

    # (1) 1칸 순환 오배열 — position은 보존되지만 orientation이 깨진다(정사각형 90도 대칭 때문)
    rolled_image_points = np.roll(correct_image_points, shift=1, axis=0)
    rolled_estimate = solve_target_pose(
        rolled_image_points, camera_matrix, dist_coeffs, object_points=obj_points,
        target_type="aruco_23", frame_id=1, timestamp=0.0,
    )
    assert not _quats_equal_up_to_sign(rolled_estimate.orientation, true_quat, atol=0.05), (
        "1칸 순환 오배열인데도 orientation이 원래와 거의 일치함 - 순서 무관 회귀 우려"
    )

    # (2) 순환이 아닌 오배열(인접 코너 한 쌍만 맞바꿈) — 어떤 강체 회전으로도 안 맞아 position 자체가 깨진다
    swapped_image_points = correct_image_points[[1, 0, 2, 3]]
    swapped_estimate = solve_target_pose(
        swapped_image_points, camera_matrix, dist_coeffs, object_points=obj_points,
        target_type="aruco_23", frame_id=1, timestamp=0.0,
    )
    pos_error = np.linalg.norm(np.array(swapped_estimate.position) - np.array(true_position))
    assert pos_error > 0.05, f"코너를 맞바꿨는데도 원래 위치와 거의 일치함(오차 {pos_error}m) - 순서 무관 회귀 우려"


# ===========================================================================
# ② 조난자 구역(초록 매트) — 코너 순서 정규화 + 착륙점 역투영
# (`modules/distress_mat.py` / `main.py::_solve_distress_mat_estimate`가 쓰는 코어 기하)
# ===========================================================================


def _project_points(rvec, tvec, camera_matrix, dist_coeffs, object_points) -> np.ndarray:
    img, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3),
        rvec, tvec, camera_matrix, dist_coeffs,
    )
    return img.reshape(-1, 2).astype(np.float64)


_MAT_K = np.array([[1000.877086342046, 0.0, 230.0],
                   [0.0, 1000.877086342046, 230.0],
                   [0.0, 0.0, 1.0]])
_MAT_DIST = np.zeros(5)


def test_distress_mat_size_constant_is_the_measured_3m_spec():
    """실측 확정 스펙(3.0m×3.0m×0.105m 라이즈드 플랫폼). ArUco의 0.50과 혼동되면 산출 거리가
    정확히 6배 틀린다 — 상수 자체를 회귀로 박아 둔다."""
    assert DISTRESS_MAT_SIZE_M == 3.0
    assert DISTRESS_MAT_PLATFORM_HEIGHT_M == 0.105
    assert DISTRESS_MAT_SIZE_M != ARUCO_TARGET_SIZE_M


def test_mat_object_points_reuse_marker_object_points_with_3m_size():
    pts = marker_object_points(DISTRESS_MAT_SIZE_M)
    assert pts.shape == (4, 3)
    assert np.allclose(pts[:, 2], 0.0), "z=0 평면 가정이 깨졌다"
    # 시계방향, top-left부터 (marker_object_points 계약)
    assert np.allclose(pts, [[-1.5, -1.5, 0.0], [1.5, -1.5, 0.0], [1.5, 1.5, 0.0], [-1.5, 1.5, 0.0]])


def test_order_quad_corners_normalizes_the_real_approxpolydp_order():
    """🔴 실측 회귀: `RectDetector`(cv2.approxPolyDP)가 골든셋 distress/10m에서 실제로 낸 순서는
    `TL->BL->BR->TR`(**반시계**)였다. `marker_object_points()`는 시계방향이므로 그대로 넣으면
    감김이 반대(거울상 대응)가 된다. 정규화 후에는 시계방향·top-left 시작이어야 한다."""
    observed = np.array([[80, 80], [80, 380], [380, 380], [380, 80]], dtype=np.float64)
    ordered = order_quad_corners_clockwise(observed)
    assert np.allclose(ordered, [[80, 80], [380, 80], [380, 380], [80, 380]])


@pytest.mark.parametrize("shift", [0, 1, 2, 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_order_quad_corners_is_invariant_to_input_rotation_and_winding(shift, reverse):
    """approxPolyDP의 시작점/추적 방향이 어떻든 같은 정규화 결과가 나와야 한다 —
    "가끔 맞고 가끔 틀리는" pose의 원인을 구조적으로 없앤다."""
    base = np.array([[80, 80], [380, 80], [380, 380], [80, 380]], dtype=np.float64)
    perturbed = np.roll(base[::-1] if reverse else base, shift, axis=0)
    assert np.allclose(order_quad_corners_clockwise(perturbed), base)


def test_order_quad_corners_rejects_degenerate_quad():
    collapsed = np.array([[10, 10], [10, 10], [10, 10], [10, 10]], dtype=np.float64)
    with pytest.raises(ValueError):
        order_quad_corners_clockwise(collapsed)
    collinear = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.float64)
    with pytest.raises(ValueError):
        order_quad_corners_clockwise(collinear)


def test_order_quad_corners_requires_exactly_four_points():
    with pytest.raises(ValueError):
        order_quad_corners_clockwise(np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float64))


def test_known_mat_size_recovers_known_distance_without_agl():
    """★핵심: **알려진 실측 크기가 스케일을 준다** — AGL(라이다) 없이 거리가 나온다는 것이
    이 접근을 고른 이유다. 3.0m 매트를 알려진 거리에 합성 투영 -> 복원 z가 그 거리와 일치."""
    obj = marker_object_points(DISTRESS_MAT_SIZE_M)
    for true_z in (10.0, 20.0, 40.0):
        tvec = np.array([[0.0], [0.0], [true_z]])
        img = _project_points(np.zeros(3), tvec, _MAT_K, _MAT_DIST, obj)
        est = solve_target_pose(
            img, _MAT_K, _MAT_DIST, object_points=obj,
            target_type="distress_mat_center", frame_id=0, timestamp=0.0,
        )
        assert est.position[2] == pytest.approx(true_z, rel=1e-6)


def test_using_the_aruco_size_constant_for_the_mat_gives_a_6x_wrong_distance():
    """크기 상수 오용(3.0 대신 ArUco의 0.50)이 조용히 넘어가지 않는다는 것을 수치로 고정 —
    거리가 정확히 6배 짧게 나온다(0.5/3.0)."""
    obj = marker_object_points(DISTRESS_MAT_SIZE_M)
    tvec = np.array([[0.0], [0.0], [10.0]])
    img = _project_points(np.zeros(3), tvec, _MAT_K, _MAT_DIST, obj)
    wrong = solve_target_pose(
        img, _MAT_K, _MAT_DIST, object_points=marker_object_points(ARUCO_TARGET_SIZE_M),
        target_type="x", frame_id=0, timestamp=0.0,
    )
    assert wrong.position[2] == pytest.approx(10.0 * ARUCO_TARGET_SIZE_M / DISTRESS_MAT_SIZE_M, rel=1e-6)


def test_position_at_pixel_none_is_bit_identical_to_the_aruco_tvec_path():
    """🔴 ArUco 회귀: 새 `position_at_pixel` 인자를 안 주면 position은 예전 그대로 tvec이다."""
    obj = marker_object_points(ARUCO_TARGET_SIZE_M)
    rvec, tvec = np.array([0.1, -0.2, 0.05]), np.array([[0.3], [-0.1], [4.0]])
    img = _project_points(rvec, tvec, _MAT_K, _MAT_DIST, obj)
    est = solve_target_pose(img, _MAT_K, _MAT_DIST, object_points=obj,
                            target_type="aruco_23", frame_id=0, timestamp=0.0)
    # 진짜 pose가 아니라 **solvePnP 자신의 tvec**과 비트 단위로 같아야 한다 — 새 분기가
    # 기본 경로에 어떤 후처리도 끼워 넣지 않았다는 뜻(허용오차 비교로는 못 잡는다).
    ok, _, solver_tvec = cv2.solvePnP(
        obj.reshape(-1, 1, 3), np.asarray(img, dtype=np.float64).reshape(-1, 1, 2),
        _MAT_K, _MAT_DIST,
    )
    assert ok
    assert est.position == (
        float(solver_tvec[0, 0]), float(solver_tvec[1, 0]), float(solver_tvec[2, 0])
    )


def test_projecting_the_target_centre_pixel_reproduces_tvec_exactly():
    """역투영 경로의 자기검증 불변식: 타겟 원점이 찍힌 픽셀을 다시 평면으로 쏘면 tvec이 나온다."""
    obj = marker_object_points(DISTRESS_MAT_SIZE_M)
    rvec, tvec = np.array([0.08, -0.12, 0.03]), np.array([[0.6], [-0.4], [12.0]])
    img = _project_points(rvec, tvec, _MAT_K, _MAT_DIST, obj)
    centre_px = _project_points(rvec, tvec, _MAT_K, _MAT_DIST, np.zeros((1, 3)))[0]

    est = solve_target_pose(
        img, _MAT_K, _MAT_DIST, object_points=obj, position_at_pixel=tuple(centre_px),
        target_type="distress_mat_center", frame_id=0, timestamp=0.0,
    )
    assert est.position == pytest.approx(tuple(tvec.reshape(-1)), abs=1e-6)


def test_landing_pixel_gives_the_offset_point_not_the_mat_centre():
    """★핵심(§5.3): 착륙 목표는 매트 중심이 아니라 "박스 옆 빈 초록면"이다. 착륙점 픽셀을 주면
    position이 실제로 그 오프셋만큼 매트 중심에서 떨어져 나와야 한다."""
    obj = marker_object_points(DISTRESS_MAT_SIZE_M)
    tvec = np.array([[0.0], [0.0], [10.0]])
    img = _project_points(np.zeros(3), tvec, _MAT_K, _MAT_DIST, obj)
    # interior_margin_ratio=0.3 -> top-left 코너(-1.5,-1.5)를 중심 쪽으로 30% 당긴 점
    landing_obj = np.array([[-1.05, -1.05, 0.0]])
    landing_px = _project_points(np.zeros(3), tvec, _MAT_K, _MAT_DIST, landing_obj)[0]

    est = solve_target_pose(
        img, _MAT_K, _MAT_DIST, object_points=obj, position_at_pixel=tuple(landing_px),
        target_type="distress_landing_point", frame_id=0, timestamp=0.0,
    )
    assert est.position == pytest.approx((-1.05, -1.05, 10.0), abs=1e-6)
    assert np.linalg.norm(np.array(est.position[:2])) > 1.0, "착륙점이 매트 중심으로 되돌아갔다"


def test_square_90deg_rotation_ambiguity_does_not_move_the_landing_point():
    """🔴 정사각형은 90° 회전 4중 대칭이라 코너 라벨링이 4가지고 solvePnP 해도 4개다. 그런데도
    **착륙점 좌표는 4개 라벨링 전부에서 동일**해야 한다 — 광선-평면 교점은 평면에만 의존하고,
    자기 법선축 회전은 평면을 자기 자신으로 보내기 때문. 이게 성립하지 않으면 회전 모호성이
    유도 좌표로 새어나간다(피듀셜이 없어 회전을 풀 방법이 아예 없으므로 치명적)."""
    obj = marker_object_points(DISTRESS_MAT_SIZE_M)
    rvec, tvec = np.array([0.05, -0.09, 0.4]), np.array([[0.5], [0.2], [11.0]])
    img = _project_points(rvec, tvec, _MAT_K, _MAT_DIST, obj)
    landing_px = tuple(_project_points(rvec, tvec, _MAT_K, _MAT_DIST, np.array([[-1.05, -1.05, 0.0]]))[0])

    positions = []
    for shift in range(4):
        est = solve_target_pose(
            np.roll(img, shift, axis=0), _MAT_K, _MAT_DIST, object_points=obj,
            position_at_pixel=landing_px, target_type="distress_landing_point",
            frame_id=0, timestamp=0.0,
        )
        positions.append(est.position)
    for p in positions[1:]:
        assert p == pytest.approx(positions[0], abs=1e-6), (
            f"90° 회전 라벨링에 따라 착륙점이 움직였다: {positions}"
        )


def test_mirrored_corner_winding_produces_a_wrong_pose():
    """감김 방향이 반대(거울상 대응)면 pose가 조용히 틀린다 — `order_quad_corners_clockwise()`가
    필요한 이유 그 자체. 이 테스트가 green이면 정규화를 빼도 아무도 모른다."""
    obj = marker_object_points(DISTRESS_MAT_SIZE_M)
    rvec, tvec = np.array([0.2, -0.15, 0.1]), np.array([[0.4], [-0.3], [10.0]])
    img = _project_points(rvec, tvec, _MAT_K, _MAT_DIST, obj)

    good = solve_target_pose(img, _MAT_K, _MAT_DIST, object_points=obj,
                             target_type="x", frame_id=0, timestamp=0.0)
    mirrored = solve_target_pose(img[::-1], _MAT_K, _MAT_DIST, object_points=obj,
                                 target_type="x", frame_id=0, timestamp=0.0)
    R_true, _ = cv2.Rodrigues(rvec)
    assert _quats_equal_up_to_sign(good.orientation, rotation_matrix_to_quaternion(R_true), atol=1e-4)
    assert not _quats_equal_up_to_sign(
        mirrored.orientation, rotation_matrix_to_quaternion(R_true), atol=0.05
    ), "감김을 뒤집었는데 자세가 그대로 — 코너 순서 정규화가 무의미해진다"


def test_backprojection_rejects_a_ray_that_cannot_hit_the_plane():
    """수치 가드: 평면과 평행한 시선(교점 없음)은 조용히 이상한 좌표를 뱉지 말고 거절해야 한다."""
    R = np.eye(3)
    tvec = np.array([[0.0], [0.0], [10.0]])
    # 법선이 (0,0,1)인 평면에 대해 시선벡터의 z성분을 0으로 만드는 K는 만들 수 없으므로
    # 평면을 90° 눕혀 시선(0,0,1)과 평행하게 만든다.
    R_edge, _ = cv2.Rodrigues(np.array([np.pi / 2, 0.0, 0.0]))
    with pytest.raises(ValueError):
        project_pixel_onto_target_plane((230.0, 230.0), R_edge, tvec, _MAT_K, _MAT_DIST)
    # 정상 평면에서는 예외 없이 값이 나온다(가드가 항상 던지는 게 아님을 확인)
    assert project_pixel_onto_target_plane((230.0, 230.0), R, tvec, _MAT_K, _MAT_DIST)[2] > 0
