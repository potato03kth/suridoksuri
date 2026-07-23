"""
ArUco 브랜치 Phase 3(`docs/vision_aruco_branch.md` §Phase 3): solvePnP + `TargetEstimate` 출력 계약.

**코어는 순수 기하 계산이다** — `vision/CLAUDE.md` import 규칙("core/ ← numpy, opencv만 허용")을
따라 이 파일은 파일 I/O(yaml 로드 등)를 하지 않는다. `nominal.yaml`(또는 향후 실측 캘리브레이션
아티팩트)을 읽는 어댑터는 `vision/utils/calibration_loader.py`에 분리돼 있다(§7.1 ports & adapters
원칙 — "카메라별 캘리브레이션 파일"은 어댑터, 코어는 camera_matrix/dist_coeffs 같은 순수 배열만
입력받는다). 호출 순서는 대략:

    calib = load_camera_calibration("vision/calibration/cam109-imx708af75/nominal.yaml")
    estimate = solve_target_pose(
        image_points=detection.corners, camera_matrix=calib.camera_matrix,
        dist_coeffs=calib.dist_coeffs, target_type="aruco_23", frame_id=..., timestamp=...,
        calib_accuracy=calib.accuracy, not_for_closed_loop_30cm=calib.not_for_closed_loop_30cm,
        calib_id=calib.calib_id,
    )

## 확정 전제 (재논의 대상 아님, `docs/vision_aruco_branch.md` §1 / 세션 지시 "확정 전제" 절)

- 타겟: `cv2.aruco.DICT_4X4_50` ID=23, 물리 크기 **50cm×50cm**(`ARUCO_TARGET_SIZE_M`).
- objectPoints 코너 순서는 `cv2.aruco.detectMarkers()`가 반환하는 순서(시계방향, top-left부터:
  top-left, top-right, bottom-right, bottom-left) — `vision/modules/aruco.py`의
  `Detection.corners`가 이 순서를 그대로 보존해 채운다. `marker_object_points()`가 만드는
  4코너도 반드시 이 순서와 대응해야 solvePnP가 옳은 pose를 낸다(순서가 틀리면 조용히 잘못된
  pose가 나온다 — `vision/tests/test_target.py`의 코너순서 회귀테스트 참조).
- 좌표계: 카메라 광학 프레임(OpenCV 표준 — X-우, Y-하, Z-전방/렌즈 바깥쪽). solvePnP raw 출력
  (rvec/tvec)을 그대로 이 프레임으로 해석한다. 별도 축 재매핑 없음.
- 단위: 미터(SI). objectPoints를 미터로 주면 tvec(=position)도 미터로 나온다.
- orientation 표현: quaternion (x, y, z, w). `cv2.Rodrigues(rvec)` → 3x3 회전행렬 →
  `rotation_matrix_to_quaternion()`(순수 numpy, scipy 미의존 — `vision/requirements.txt`에
  scipy가 없어 새 무거운 의존성 추가 대신 표준 Shepperd's method를 직접 구현했다).
- `uncertainty`: 스키마에 필드는 있으나 **이번 Phase는 항상 None**(자리만 만들어둠 — 실측
  캘리브레이션 이후 채울 대상).
- `calib_accuracy`/`not_for_closed_loop_30cm`/`calib_id`: `nominal.yaml`의 동명 필드가
  `TargetEstimate`까지 그대로 흘러야 한다는 요구(§7.1/§7.3 provenance echo, 리스크 절) —
  하위 소비자(향후 offboard 통합)가 이 pose를 폐루프에 바로 쓰면 안 된다는 걸 알 수 있어야
  한다. `solve_target_pose()` 호출자가 로드한 calib에서 그대로 넘겨줘야 실제로 채워진다(코어는
  yaml을 모르므로 스스로 채우지 않는다 — 호출자 책임).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# 확정 전제(§1) — 타겟 물리 크기. 재논의 대상 아님.
ARUCO_TARGET_SIZE_M = 0.50


@dataclass
class TargetEstimate:
    """코어가 뱉는 유일한 중립 dataclass(`docs/vision_plan.md` §7.1) — 상대 pose + 신뢰도 + 타입
    + frame_id + timestamp. 좌표계/단위/orientation 표현은 이 파일 docstring "확정 전제" 참조."""

    position: tuple[float, float, float]              # (x, y, z) 미터, 카메라 광학 프레임
    orientation: tuple[float, float, float, float]     # quaternion (x, y, z, w)
    confidence: float
    target_type: str                                   # 예: "aruco_23"
    frame_id: int
    timestamp: float
    # 불확실성(공분산/신뢰반경) — 이번 Phase는 항상 None. 실측 캘리브레이션 이후 채울 자리
    # (§7.1 "폐루프 게이팅·게인용 불확실성 필드 포함 여부" 결정 — 자리만 확정, 값 산출은 범위 밖).
    uncertainty: Optional[np.ndarray] = None
    # provenance echo(§7.3) — nominal.yaml의 동명 필드를 그대로 옮긴 것. 하위 소비자가 이 pose를
    # 폐루프 30cm 정밀착륙에 바로 쓰면 안 된다는 걸 판단하는 근거(§7.1 리스크 절).
    calib_accuracy: str = "unverified"
    not_for_closed_loop_30cm: bool = True
    calib_id: str = ""
    meta: dict = field(default_factory=dict)


def marker_object_points(size_m: float = ARUCO_TARGET_SIZE_M) -> np.ndarray:
    """마커 중심을 원점으로 하는 평면 4코너(z=0), (N,3) float64.

    순서는 `cv2.aruco.detectMarkers()` 코너 순서(시계방향, top-left부터: top-left, top-right,
    bottom-right, bottom-left)와 반드시 대응해야 한다 — `vision/modules/aruco.py`의
    `Detection.corners`가 이 순서로 채워진다."""
    h = float(size_m) / 2.0
    return np.array(
        [
            [-h, -h, 0.0],   # top-left
            [h, -h, 0.0],    # top-right
            [h, h, 0.0],     # bottom-right
            [-h, h, 0.0],    # bottom-left
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 회전행렬 -> quaternion (x, y, z, w). 표준 Shepperd's method(순수 numpy) —
    `vision/requirements.txt`에 scipy가 없어 새 의존성 추가 대신 직접 구현(세션 지시 우선순위).

    trace 부호에 따라 4갈래로 분기해 수치적으로 가장 안정적인 성분부터 계산한다(단순
    trace-based 공식은 trace가 작을 때 나눗셈이 불안정해짐 — 로보틱스 라이브러리들(ROS tf 등)의
    표준 구현과 동일한 분기 구조)."""
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0  # s = 4*qw
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0  # s = 4*qx
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0  # s = 4*qy
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0  # s = 4*qz
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm > 0.0:
        q = q / norm
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def solve_target_pose(
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    target_type: str,
    frame_id: int,
    timestamp: float,
    object_points: Optional[np.ndarray] = None,
    confidence: float = 1.0,
    calib_accuracy: str = "unverified",
    not_for_closed_loop_30cm: bool = True,
    calib_id: str = "",
    meta: Optional[dict] = None,
) -> TargetEstimate:
    """`cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)` -> rvec/tvec ->
    quaternion -> `TargetEstimate`.

    `image_points`는 `Detection.corners`(4x2 float32, 이미지 좌표, `vision/modules/aruco.py`)를
    그대로 받는다. `object_points` 미지정 시 `marker_object_points(ARUCO_TARGET_SIZE_M)`을 쓴다
    (코너 순서가 `image_points`와 대응한다는 전제 — 이 파일 docstring "확정 전제" 참조).

    calib_accuracy/not_for_closed_loop_30cm/calib_id는 **호출자가 로드한 calib에서 그대로
    넘겨줘야 한다** — 이 함수는 yaml을 모르므로(core는 파일 I/O 안 함) 기본값(`"unverified"`/
    `True`/`""`)으로만 채워진다."""
    if object_points is None:
        object_points = marker_object_points(ARUCO_TARGET_SIZE_M)

    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    K = np.asarray(camera_matrix, dtype=np.float64)
    dist = np.asarray(dist_coeffs, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
    if not ok:
        raise ValueError("solve_target_pose: cv2.solvePnP 실패(수렴하지 않음)")

    R, _ = cv2.Rodrigues(rvec)
    quat = rotation_matrix_to_quaternion(R)
    position = (float(tvec[0, 0]), float(tvec[1, 0]), float(tvec[2, 0]))

    return TargetEstimate(
        position=position,
        orientation=quat,
        confidence=confidence,
        target_type=target_type,
        frame_id=frame_id,
        timestamp=timestamp,
        uncertainty=None,
        calib_accuracy=calib_accuracy,
        not_for_closed_loop_30cm=not_for_closed_loop_30cm,
        calib_id=calib_id,
        meta=dict(meta) if meta else {},
    )
