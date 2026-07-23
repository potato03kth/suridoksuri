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
    TargetEstimate,
    marker_object_points,
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
