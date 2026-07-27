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

# ② 조난자 구역(초록 매트) 실측 스펙 — `docs/vision_plan.md` §2/§5.3, `docs/vision_status.md`
# 대회 규정 절 "3.0m×3.0m×0.105m 라이즈드 플랫폼". ArUco와 달리 피듀셜이 없어 코너 4개 +
# **알려진 실측 크기**가 유일한 스케일 정보원이다(그래서 AGL 없이도 거리가 나온다).
DISTRESS_MAT_SIZE_M = 3.0
# 라이즈드 플랫폼의 물리 높이. **pose 자체엔 안 들어간다** — 아래 `MAT_PLANE_REFERENCE` 참조.
DISTRESS_MAT_PLATFORM_HEIGHT_M = 0.105

# 🔴 z=0 평면이 어디인가 (세션 지시 3번 — 명시 요구).
#
# `marker_object_points()`가 만드는 4코너는 전부 z=0이다. 초록 매트에서 검출되는 4코너는
# **위에서 본 초록 면의 윤곽**, 즉 라이즈드 플랫폼의 **윗면 가장자리**다(0.105m 라이저의
# 측면은 근-나디르 시야에서 거의 안 보이고, 초록색은 윗면에만 있다). 따라서:
#
#   z=0 평면 = 매트 **윗면**(주변 자갈 지면보다 0.105m 높다)
#   → `TargetEstimate.position`의 거리는 **윗면까지의 거리**다.
#
# 착륙 고도 해석의 결론: **윗면 기준이 맞다.** 기체는 매트 윗면에 내려앉으므로 접지 목표면이
# 곧 이 평면이다 — 착륙 자체엔 보정이 필요 없다. ⚠️ 단 소비자가 이 z를 **라이다 AGL**(주변
# 지면 기준)과 섞으면 정확히 `DISTRESS_MAT_PLATFORM_HEIGHT_M`만큼 어긋난다(비전 z가 라이다
# AGL보다 0.105m 작게 나온다). 그래서 두 값을 `TargetEstimate.meta`에 실어 소비자가 어느
# 기준인지 읽을 수 있게 한다(`modules/distress_mat.py` → `main.py`).
MAT_PLANE_REFERENCE = "mat_top_surface"

# 코너 4개가 실질적으로 한 점/한 직선으로 뭉개졌는지 판정하는 최소 면적(px²). solvePnP는 그런
# 입력에서도 조용히 아무 값이나 뱉으므로 여기서 먼저 거른다. 1.0은 "면적이 1픽셀 미만이면
# 사각형이라 부를 수 없다"는 하한일 뿐 튜닝 파라미터가 아니다.
_MIN_QUAD_AREA_PX2 = 1.0
# 시선벡터가 타겟 평면과 거의 평행일 때(=역투영 교점이 무한대) 거르는 하한. 순수 수치안정성
# 가드이며 물리 임계값이 아니다.
_MIN_RAY_PLANE_COS = 1e-9


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


def order_quad_corners_clockwise(corners) -> np.ndarray:
    """임의 순서의 사각형 4코너(이미지 좌표) -> `marker_object_points()`와 **대응되는 순서**로
    정규화한 (4,2) float64. 순서 = 화면상 시계방향, top-left부터.

    🔴 **이 정규화가 없으면 pose가 조용히 틀린다.** `cv2.aruco.detectMarkers()`는 코너 순서를
    보장하지만 `modules/detector.py::RectDetector`가 쓰는 `cv2.approxPolyDP()`는 **보장하지
    않는다** — 윤곽 추적 방향과 시작점에 따라 순서가 달라진다. 실측(골든셋 `distress/10m`)에서
    실제로 나온 순서는 `[(80,80),(80,380),(380,380),(380,80)]`로 **반시계(TL→BL→BR→TR)**였다.
    이걸 그대로 `solvePnP`에 넣으면 objectPoints(시계)와 **감김 방향이 반대**라 거울상 대응이
    되고, 복원되는 R/t는 물리적으로 불가능한 해가 된다(`tests/test_target.py`의 감김 회귀 참조).

    알고리즘: 무게중심 기준 `atan2(dy, dx)` 오름차순 정렬(이미지 좌표는 y가 아래로 증가하므로
    각도 증가 = **화면상 시계방향**) → `x+y`가 최소인 코너(top-left)가 맨 앞에 오도록 회전.
    타이(정사각형 등)에서도 `argmin`/`argsort`가 첫 항목을 고르므로 결정론이 깨지지 않는다(§7.5).

    ⚠️ 정사각 매트는 90° 회전 4겹 자기대칭이라 "어느 물리 코너가 top-left인가"는 **원리적으로
    결정할 수 없다**(피듀셜이 없으므로). 이 함수는 그 모호성을 푸는 게 아니라 **감김 방향과
    시작점을 결정론적으로 고정**할 뿐이다. 회전 모호성 자체가 착륙점 좌표에 영향을 주지 않는
    이유는 `project_pixel_onto_target_plane()` docstring 참조.
    """
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"order_quad_corners_clockwise: 코너 4개가 필요하다 (받은 개수: {pts.shape[0]})")

    centroid = pts.mean(axis=0)
    delta = pts - centroid
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    ordered = pts[np.argsort(angles, kind="stable")]

    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)

    # shoelace — 뭉개진 사각형(한 점/한 직선)을 solvePnP에 넘기기 전에 거른다.
    x, y = ordered[:, 0], ordered[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < _MIN_QUAD_AREA_PX2:
        raise ValueError(f"order_quad_corners_clockwise: 축퇴된 사각형(면적 {area:.4f}px²)")
    return ordered


def project_pixel_onto_target_plane(
    pixel_px,
    rotation: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[float, float, float]:
    """픽셀 하나를 **이미 푼 타겟 평면**(objectPoints의 z=0 평면) 위로 역투영해 카메라 광학
    프레임 3D 좌표를 돌려준다.

    쓰임: 초록 매트의 착륙점은 매트 **중심이 아니라** "박스 옆 빈 초록면"이다
    (`modules/distress_box.py`가 `landing_point_px`로 이미 확정해 둔 픽셀). 매트 4코너로 평면을
    풀어 두면 그 픽셀이 평면 위 어디인지는 광선-평면 교점 하나로 정해진다.

    🔴 **이 방식이 정사각형의 90° 회전 4중 모호성을 구조적으로 없앤다.** 정사각 매트는 코너
    라벨링이 4가지라 solvePnP 해도 4개(자기 중심축 90° 회전 차이)지만, 그 4개는 **전부 같은
    평면**을 준다(평면 법선을 축으로 도는 회전은 평면을 자기 자신으로 보낸다). 광선-평면 교점은
    평면에만 의존하므로 어느 해가 뽑히든 **착륙점 좌표가 완전히 동일**하다 — 흰 박스 같은
    비대칭 단서로 회전을 먼저 풀 필요가 없다(실측 스펙상 흰 박스는 매트 정중앙이라 애초에
    단서가 되지도 못한다). 남는 4중 모호성은 `orientation` 쿼터니언뿐이고, 그건 정사각형에
    대해 물리적으로 무의미한 자유도다.

    `dist_coeffs`를 무시하지 않는다 — `cv2.undistortPoints()`로 먼저 왜곡을 푼 뒤 시선벡터를
    만든다(nominal.yaml은 전부 0이라 현재는 항등이지만, 실측 캘리브레이션이 들어오면 자동 반영).
    """
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    # 타겟 평면의 법선(카메라 프레임) = 물체 프레임 z축의 상. 평면 위 한 점 = 물체 원점 = t.
    normal = R[:, 2]

    pt = np.asarray(pixel_px, dtype=np.float64).reshape(1, 1, 2)
    undistorted = cv2.undistortPoints(
        pt, np.asarray(camera_matrix, dtype=np.float64), np.asarray(dist_coeffs, dtype=np.float64)
    )
    ray = np.array([float(undistorted[0, 0, 0]), float(undistorted[0, 0, 1]), 1.0], dtype=np.float64)

    denom = float(normal @ ray)
    if abs(denom) < _MIN_RAY_PLANE_COS:
        raise ValueError("project_pixel_onto_target_plane: 시선이 타겟 평면과 평행 — 교점 없음")
    scale = float(normal @ t) / denom
    if scale <= 0.0:
        raise ValueError("project_pixel_onto_target_plane: 교점이 카메라 뒤쪽")
    point = scale * ray
    return (float(point[0]), float(point[1]), float(point[2]))


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
    position_at_pixel: Optional[tuple] = None,
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
    `True`/`""`)으로만 채워진다.

    `position_at_pixel`(기본 None): 주면 `position`이 **tvec(=타겟 원점)이 아니라 그 픽셀을
    타겟 평면 위로 역투영한 점**이 된다(`project_pixel_onto_target_plane()`). ② 조난자 구역은
    착륙 목표가 매트 중심이 아니라 "박스 옆 빈 초록면"이라 이 갈래가 필요하다.
    **None이면 기존 ArUco 경로와 한 줄도 다르지 않다**(position = tvec 그대로)."""
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
    if position_at_pixel is None:
        position = (float(tvec[0, 0]), float(tvec[1, 0]), float(tvec[2, 0]))
    else:
        position = project_pixel_onto_target_plane(position_at_pixel, R, tvec, K, dist)

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
