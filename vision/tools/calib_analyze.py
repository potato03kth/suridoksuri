"""
카메라 캘리브레이션 분석 도구 — `calib_capture.py`가 만든 촬영 세트를 캘리브레이션 아티팩트로.

**신설(2026-07-23).** `docs/vision_camera_bringup.md` 인수인계 3번("캘리브레이션 분석 스크립트
— 아직 없다. 새로 만들어야 한다")을 구현한다. 이 스크립트 실행 시점에는 **실촬영 사진이
아직 없다**(사용자가 촬영 진행 중) — 그래서 정확성은 합성 데이터 왕복 테스트로만 담보한다
(`vision/tests/test_calib_analyze.py`).

## 입력

`calib_capture.py`가 만드는 트리(그 파일 `_capture_locked()`/`run_focus_sweep_and_lock()`의
저장부, `ShotSpec.file_stem`/`build_shot_list()`로 실제 확인함):

```
<raw_root>/
  A/0.5m/_lock.json        <- 그 거리의 잠금 기록(스윕 표 포함, 이 스크립트는 안 읽음)
  A/0.5m/<stem>.png        <- 4608x2592 풀해상 촬영본(기본 미사용, --redetect 시에만)
  A/0.5m/<stem>.json       <- 사이드카(기본 입력 — corners_px가 이미 서브픽셀 정제됨)
  A/0.7m/... , A/1.2m/... , B/2.5m/...
```

사이드카 JSON 필드(코드로 확인, `tools/calib_capture.py::_capture_locked` 986~1017행 근방):
`timestamp, set, distance_m, seq, zone, u, v, yaw_deg, pitch_deg, roll_deg, label_ko,
pattern_size([cols,rows]), square_size_mm, locked_controls{lens_position, exposure_time,
analogue_gain, colour_gains}, corners_px(N×2), sharpness, reference_sharpness, target_zone_px,
image_size([w,h]), image_path`. 보드 치수(pattern_size/square_size_mm)는 **사이드카 값을
그대로 쓴다 — 하드코딩 안 함**(세트마다 A4/태블릿 보드가 다를 수 있음).

## 파이프라인

1. **로드** (`load_dataset`) — 트리 순회 → (set, distance_m)로 그룹핑. 기본은 사이드카
   `corners_px` 그대로(이미 서브픽셀 정제됨, 재검출 불필요). `--redetect`면 PNG에서
   `findChessboardCorners`+`cornerSubPix` 재실행(사이드카 누락/의심 시 대비).
2. **그룹별 캘리브레이션** (`calibrate_group_images`) — `cv2.calibrateCamera` → K, dist, RMS.
3. **불량컷 색출** (`detect_outliers`) — 사진별 재투영오차(`per_image_reprojection_errors`).
   `err > max(abs_threshold, mean+std_mult*std)`(전부 CLI 파라미터, 매직넘버 금지 —
   `docs/vision_plan.md` §7.3). 이상치 제외 재캘리브레이션 → 전후 RMS/K 비교.
4. **fx/fy vs LensPosition 직선적합 → 무한대(L=0) 외삽** (`linear_fit_lens_position`) —
   이 스크립트의 핵심 목적. 절편 a = 무한대 초점 fx/fy(기체 운용고도 10~40m는 사실상
   무한대). 물리 일관성 검사: 얇은렌즈 근사로 `fx(L)≈fx_inf*(1+f*L)`(f=미터 단위 초점거리)
   이므로 `b/a`[m]가 곧 f — mm로 환산해 IMX708급 통상 초점거리(~4.7mm)와 비교
   (`check_physical_consistency`). 디옵터 가정(`LensPosition≈1/distance_m`) 자체도
   그룹별로 직접 대조(`check_diopter_assumption`) — 이 대조가 촬영 세션의 명시적 목적.
5. **세트 B 대조** (`check_set_b_extrapolation`) — 세트 B(2.5m, L≈0.4, 40장, 최대커버리지)는
   모델 무관 무한대-근접 기준값. 외삽 예측 fx/fy vs 세트B 직접캘리브 fx/fy 비교, 왜곡계수도
   세트간 비교.
6. **정합성 검사** (`check_aspect_ratio`/`check_principal_point`/`check_hfov`/
   `check_group_image_counts`/`check_dataset_consistency`) — 실패해도 크래시 없이 보고.
   HFOV는 `recommended` camera_matrix의 fx로 계산해 이 카메라 실측 화각과 비교한다.
   **주의(코드로 확인한 배치 — `check_hfov` 참조):** 세션 지시 원문은 "75°"를 그대로
   기준값으로 쓰라고 명시하지만, `docs/vision_status.md`는 이 75°를 **대각(diagonal)**으로
   표기하고 있고 `calib_capture.py`의 `GROUND_WIDTH_COEFF_PER_M` 주석은 16:9 센서 환산
   실측 **수평화각 67.5°**를 채택했다 — 두 문서가 서로 다른 값을 "실측 화각"으로 쓰고
   있어 하나로 조용히 확정하지 않았다. 기본값은 세션 지시 원문(75°)을 따르되
   `--hfov-nominal-deg`로 즉시 바꿀 수 있게 열어뒀고, 이 배치 자체를 check의 detail에
   문장으로 남긴다(§7.3 매직넘버 금지 + 조용히 넘어가지 말 것 요구 반영).
7. **진단 플롯** (Agg, headless-safe) → `vision/results/`: 그룹별 코너 산포도, 사진별
   재투영오차 막대그래프(이상치 강조), fx-vs-LensPosition 산포+적합선+L=0 외삽.
8. **아티팩트** (`build_artifact`) → `vision/calibration/<camera_id>/<calib_id>.yaml`.
   `recommended`는 fx/fy=무한대외삽 절편, cx/cy/dist_coeffs=세트B(장수최다·커버리지최대·
   운용조건최근접) — `--recommended-source`로 다른 선택도 가능.
9. **견고성** (§9) — 부분 데이터(그룹 일부만, B 없음 등)에서 크래시 금지. 그룹<2면 직선적합
   생략+경고, `recommended`는 LensPosition 최저 그룹으로 폴백하고 사유를 yaml에 명시.

## 순수 함수 vs I/O

로드(`load_dataset`)와 저장(`save_artifact`/플롯 함수들)만 파일시스템을 만진다. 계산 로직
(`object_points_for_pattern`/`calibrate_group_images`/`per_image_reprojection_errors`/
`detect_outliers`/`analyze_group`/`linear_fit_lens_position`/각 `check_*`/
`determine_recommended`/`build_artifact`)은 순수 함수 — `vision/tests/test_calib_analyze.py`가
합성 데이터로 하드웨어 없이 검증한다.

## CLI

```bash
python vision/tools/calib_analyze.py --raw-root vision/data/calibration_raw
```

하드웨어 의존 없음(`cv2`/`numpy`/`PyYAML`/`matplotlib`뿐) — `vision/CLAUDE.md`의
`jsonl_view.py` 예외 규칙과 동일하게 `.venv` 설치 + pytest 대상이다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

import matplotlib

matplotlib.use("Agg")  # headless-safe — GUI 강제 호출 금지(드론/CI/RPi에 디스플레이 없음)
import matplotlib.pyplot as plt

from vision.utils.logging import get_git_commit_hash


def _configure_korean_font() -> None:
    """matplotlib 기본 폰트(DejaVu Sans)는 한글 글리프가 없어 플롯 라벨이 네모(tofu)로 깨진다
    (`calib_capture.py`가 cv2.putText Hershey 폰트에서 겪은 것과 같은 종류의 문제, 이 파일
    docstring 참조). 설치된 한글 폰트가 있으면 그걸 쓰고, 없으면(예: 폰트가 없는 RPi/CI)
    **조용히 기본 폰트로 남긴다** — 폰트 유무가 진단 플롯 생성 자체를 막아서는 안 된다
    (§9 견고성과 같은 원칙: 라벨이 깨지는 건 미관 문제일 뿐 크래시가 아니다)."""
    import matplotlib.font_manager as fm

    installed = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic"):
        if candidate in installed:
            matplotlib.rcParams["font.family"] = candidate
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


_configure_korean_font()

# ===========================================================================
# 상수 (매직넘버 금지 — 전부 이름+기본값+CLI로 노출, docs/vision_plan.md §7.3)
# ===========================================================================

DEFAULT_CAMERA_ID = "cam109-imx708af75"

DEFAULT_OUTLIER_ABS_PX = 1.0             # 재투영오차 이상치 절대 임계(px)
DEFAULT_OUTLIER_STD_MULT = 2.0           # mean + N*std 임계 배수
DEFAULT_MIN_IMAGES_FOR_RECAL = 4         # 이상치 제외 후 재캘리브레이션을 시도할 최소 잔여 장수

DEFAULT_NOMINAL_FOCAL_LENGTH_MM = 4.7    # IMX708급 모듈 통상 초점거리(물리 일관성 검사 기준)
DEFAULT_FOCAL_LENGTH_CHECK_RATIO = 3.0   # 통과 범위 = [nominal/ratio, nominal*ratio]

DEFAULT_DIOPTER_TOLERANCE_FRAC = 0.2     # LensPosition vs 1/distance_m 허용 상대오차

DEFAULT_ASPECT_RATIO_TOL = 0.02          # |fy/fx - 1| 허용
DEFAULT_PRINCIPAL_POINT_TOL_FRAC = 0.1   # (cx,cy)가 이미지 중심에서 벗어나도 되는 비율(치수 대비)
DEFAULT_HFOV_NOMINAL_DEG = 75.0          # 세션 지시 원문값 — 위 docstring "주의" 절 참조
DEFAULT_HFOV_TOL_DEG = 5.0
DEFAULT_SET_B_CONSISTENCY_TOL_PCT = 5.0  # 외삽 fx/fy vs 세트B 실측 fx/fy 허용 % 차이
DEFAULT_MIN_IMAGES_PER_GROUP = 3         # 그룹별 이미지 수 하한(정합성 검사용)


# ===========================================================================
# 데이터 모델
# ===========================================================================


@dataclass
class ImageRecord:
    """사이드카 JSON 1장 분량 — 순수 데이터."""

    json_path: Path
    image_path: Optional[Path]
    set_name: str
    distance_m: float
    seq: int
    zone: str
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    label_ko: str
    pattern_size: Tuple[int, int]
    square_size_mm: float
    lens_position: Optional[float]
    corners_px: np.ndarray          # (N,2) float64
    image_size: Tuple[int, int]     # (w,h)
    sharpness: Optional[float]
    corner_source: str              # "sidecar" | "redetect"


@dataclass
class GroupData:
    """(set, distance_m) 하나의 원본 이미지 묶음(캘리브레이션 전, 로드 직후)."""

    set_name: str
    distance_m: float
    images: List[ImageRecord]

    @property
    def key(self) -> Tuple[str, float]:
        return (self.set_name, self.distance_m)


@dataclass
class GroupCalibration:
    """`cv2.calibrateCamera` 결과 1회분."""

    K: np.ndarray
    dist: np.ndarray                # (5,) float64 — k1,k2,p1,p2,k3
    rms: float
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]
    object_points: List[np.ndarray]
    image_points: List[np.ndarray]
    images: List[ImageRecord]       # rvecs/tvecs/object_points/image_points와 같은 순서


@dataclass
class GroupAnalysis:
    """그룹 하나의 전체 분석 결과(캘리브레이션 전/후 + 이상치 + 메타)."""

    key: Tuple[str, float]
    pattern_size: Tuple[int, int]
    square_size_mm: float
    image_size: Tuple[int, int]
    lens_position: Optional[float]
    n_total_images: int
    calib_before: Optional[GroupCalibration]
    errors_before: List[float]
    outlier_mask: List[bool]
    outlier_threshold: float
    calib_final: Optional[GroupCalibration]   # 이상치 제외 재캘리브레이션 결과(불가 시 before와 동일 객체)
    errors_final: List[float]
    error_msg: Optional[str] = None            # 그룹 캘리브레이션 자체가 실패한 경우


# ===========================================================================
# 로드 (I/O) — 트리 순회 → (set, distance_m) 그룹핑
# ===========================================================================


def _redetect_corners(png_path: Path, pattern_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """PNG에서 코너 재검출(사이드카 corners_px 누락/의심 시 대비, --redetect).

    calib_capture.py `_capture_locked()`와 동일한 검출 절차(findChessboardCorners
    ADAPTIVE_THRESH|NORMALIZE_IMAGE → cornerSubPix)를 재사용한다."""
    bgr = cv2.imread(str(png_path))
    if bgr is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2)


def _load_image_record(json_path: Path, redetect: bool) -> Tuple[Optional[ImageRecord], List[str]]:
    """사이드카 JSON 1개 -> ImageRecord. 실패해도 크래시하지 않고 (None, 경고목록) 반환."""
    warnings: List[str] = []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return None, [f"{json_path}: JSON 로드 실패({e}) - 건너뜀"]

    try:
        set_name = str(data["set"])
        distance_m = float(data["distance_m"])
        pattern_size = (int(data["pattern_size"][0]), int(data["pattern_size"][1]))
        square_size_mm = float(data["square_size_mm"])
        image_size = (int(data["image_size"][0]), int(data["image_size"][1]))
    except (KeyError, TypeError, ValueError, IndexError) as e:
        return None, [f"{json_path}: 필수 필드 누락/형식오류({e}) - 건너뜀"]

    corners: Optional[np.ndarray] = None
    source = "sidecar"
    raw_corners = data.get("corners_px")
    if not redetect and raw_corners:
        arr = np.asarray(raw_corners, dtype=np.float64)
        if arr.size > 0:
            corners = arr.reshape(-1, 2)
    if corners is None:
        png_path = json_path.with_suffix(".png")
        if not png_path.exists():
            return None, [f"{json_path}: corners_px 없음(또는 --redetect)인데 PNG({png_path.name})도 없음 - 건너뜀"]
        corners = _redetect_corners(png_path, pattern_size)
        source = "redetect"
        if corners is None:
            return None, [f"{png_path}: findChessboardCorners 재검출 실패 - 건너뜀"]

    locked = data.get("locked_controls") or {}
    lens_position = locked.get("lens_position")

    image_path_raw = data.get("image_path")
    image_path = Path(image_path_raw) if image_path_raw else None

    rec = ImageRecord(
        json_path=json_path,
        image_path=image_path,
        set_name=set_name,
        distance_m=distance_m,
        seq=int(data.get("seq", 0)),
        zone=str(data.get("zone", "")),
        yaw_deg=float(data.get("yaw_deg", 0.0)),
        pitch_deg=float(data.get("pitch_deg", 0.0)),
        roll_deg=float(data.get("roll_deg", 0.0)),
        label_ko=str(data.get("label_ko", "")),
        pattern_size=pattern_size,
        square_size_mm=square_size_mm,
        lens_position=float(lens_position) if lens_position is not None else None,
        corners_px=corners,
        image_size=image_size,
        sharpness=data.get("sharpness"),
        corner_source=source,
    )
    return rec, warnings


def load_dataset(raw_root: Path, redetect: bool = False) -> Tuple[Dict[Tuple[str, float], GroupData], List[str]]:
    """raw_root 트리를 순회해 (set, distance_m)으로 그룹핑한다.

    디렉터리/파일 누락, 손상된 JSON, 코너 없는 이미지 — 전부 경고만 남기고 계속 진행한다
    (§9 견고성: 절대 크래시 금지). 그룹이 하나도 없으면 빈 dict를 반환한다(호출자가 판단)."""
    warnings: List[str] = []
    raw_root = Path(raw_root)
    if not raw_root.exists():
        return {}, [f"raw_root가 존재하지 않음: {raw_root}"]

    json_paths = sorted(p for p in raw_root.rglob("*.json") if p.name != "_lock.json")
    per_key: Dict[Tuple[str, float], List[ImageRecord]] = {}
    for jp in json_paths:
        rec, w = _load_image_record(jp, redetect)
        warnings.extend(w)
        if rec is None:
            continue
        per_key.setdefault((rec.set_name, rec.distance_m), []).append(rec)

    if not per_key:
        warnings.append(f"raw_root({raw_root})에서 유효한 사이드카를 하나도 읽지 못함")

    groups = {
        key: GroupData(key[0], key[1], sorted(records, key=lambda r: r.seq))
        for key, records in per_key.items()
    }
    return groups, warnings


def group_pattern_size(g: GroupData) -> Tuple[int, int]:
    return g.images[0].pattern_size


def group_square_size_mm(g: GroupData) -> float:
    return g.images[0].square_size_mm


def group_image_size(g: GroupData) -> Tuple[int, int]:
    return g.images[0].image_size


def group_lens_position(g: GroupData) -> Optional[float]:
    """그룹 내 이미지들의 locked_controls.lens_position 평균(잠금값이라 이론상 동일해야 함)."""
    vals = [im.lens_position for im in g.images if im.lens_position is not None]
    if not vals:
        return None
    return float(np.mean(vals))


# ===========================================================================
# 캘리브레이션 (순수 함수, cv2 필요)
# ===========================================================================


def object_points_for_pattern(pattern_size: Tuple[int, int], square_size_mm: float) -> np.ndarray:
    """체스보드 objectPoints(z=0 평면). pattern_size=(cols,rows)는 findChessboardCorners 관례."""
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    return objp


def calibrate_group_images(
    images: Sequence[ImageRecord], pattern_size: Tuple[int, int], square_size_mm: float, image_size: Tuple[int, int]
) -> GroupCalibration:
    """이미지 목록 하나로 cv2.calibrateCamera 1회 실행. 실패 시 그대로 예외를 던진다
    (호출자 `analyze_group`이 잡아서 그룹 단위로 우아하게 처리)."""
    if len(images) == 0:
        raise ValueError("calibrate_group_images: 이미지가 0장이라 캘리브레이션 불가")
    objp32 = object_points_for_pattern(pattern_size, square_size_mm).astype(np.float32)
    object_points = [objp32.copy() for _ in images]
    image_points = [im.corners_px.astype(np.float32).reshape(-1, 1, 2) for im in images]

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    return GroupCalibration(
        K=K,
        dist=np.asarray(dist, dtype=np.float64).reshape(-1),
        rms=float(rms),
        rvecs=list(rvecs),
        tvecs=list(tvecs),
        object_points=object_points,
        image_points=image_points,
        images=list(images),
    )


def per_image_reprojection_errors(calib: GroupCalibration) -> List[float]:
    """이미지별 RMS 재투영오차(px) — calib.images/rvecs/tvecs와 같은 순서로 반환."""
    errors = []
    for objp, imgp, rvec, tvec in zip(calib.object_points, calib.image_points, calib.rvecs, calib.tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, calib.K, calib.dist)
        proj = proj.reshape(-1, 2)
        actual = imgp.reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum((proj - actual) ** 2, axis=1))))
        errors.append(err)
    return errors


def detect_outliers(
    errors: Sequence[float], abs_threshold: float = DEFAULT_OUTLIER_ABS_PX, std_multiplier: float = DEFAULT_OUTLIER_STD_MULT
) -> Tuple[List[bool], float]:
    """이상치 규칙: err > max(abs_threshold, mean+std_multiplier*std). (mask, 실제사용임계값) 반환."""
    if not errors:
        return [], abs_threshold
    arr = np.asarray(errors, dtype=np.float64)
    threshold = max(float(abs_threshold), float(arr.mean() + std_multiplier * arr.std()))
    mask = [bool(e > threshold) for e in errors]
    return mask, threshold


def analyze_group(
    g: GroupData,
    outlier_abs_px: float = DEFAULT_OUTLIER_ABS_PX,
    outlier_std_mult: float = DEFAULT_OUTLIER_STD_MULT,
    min_images_for_recal: int = DEFAULT_MIN_IMAGES_FOR_RECAL,
) -> GroupAnalysis:
    """그룹 하나: 캘리브레이션 -> 이상치 검출 -> (있으면) 제외 후 재캘리브레이션.

    어떤 단계든 실패해도 예외를 밖으로 던지지 않는다 — `error_msg`에 담아 반환(§9)."""
    pattern_size = group_pattern_size(g)
    square_size_mm = group_square_size_mm(g)
    image_size = group_image_size(g)
    lens_position = group_lens_position(g)
    n_total = len(g.images)

    try:
        calib_before = calibrate_group_images(g.images, pattern_size, square_size_mm, image_size)
    except Exception as e:  # noqa: BLE001 - 그룹 단위 견고성(§9), 크래시 절대 금지
        return GroupAnalysis(
            key=g.key, pattern_size=pattern_size, square_size_mm=square_size_mm, image_size=image_size,
            lens_position=lens_position, n_total_images=n_total,
            calib_before=None, errors_before=[], outlier_mask=[], outlier_threshold=0.0,
            calib_final=None, errors_final=[], error_msg=f"캘리브레이션 실패: {e}",
        )

    errors_before = per_image_reprojection_errors(calib_before)
    outlier_mask, threshold = detect_outliers(errors_before, outlier_abs_px, outlier_std_mult)

    calib_final = calib_before
    errors_final = errors_before
    if any(outlier_mask):
        kept = [im for im, is_out in zip(g.images, outlier_mask) if not is_out]
        if len(kept) >= min_images_for_recal:
            try:
                calib_final = calibrate_group_images(kept, pattern_size, square_size_mm, image_size)
                errors_final = per_image_reprojection_errors(calib_final)
            except Exception:  # noqa: BLE001 - 재캘리브레이션 실패 시 원본 결과 유지
                calib_final = calib_before
                errors_final = errors_before
        # else: 제외 후 남는 장수가 너무 적음 - 재캘리브레이션 생략, calib_final은 before 그대로
        #       (outlier_mask는 보고용으로 그대로 남긴다 - "이상치로 보였으나 재캘리브 생략" 사실을
        #       구분할 수 있도록 calib_final.images 길이로 실제 반영 여부를 판단한다)

    return GroupAnalysis(
        key=g.key, pattern_size=pattern_size, square_size_mm=square_size_mm, image_size=image_size,
        lens_position=lens_position, n_total_images=n_total,
        calib_before=calib_before, errors_before=errors_before,
        outlier_mask=outlier_mask, outlier_threshold=threshold,
        calib_final=calib_final, errors_final=errors_final,
    )


# ===========================================================================
# fx/fy vs LensPosition 직선적합 (핵심 목적)
# ===========================================================================


def linear_fit_lens_position(points: Sequence[Tuple[Optional[float], Optional[float]]]) -> Optional[dict]:
    """(lens_position, fx_or_fy) 점들 -> 최소제곱 직선 y=a+b*L.

    유효 점(둘 다 None 아님)이 2개 미만이면 적합 불가 -> None(scipy 없음, numpy.polyfit 사용,
    저장소 규약)."""
    pts = [(float(l), float(y)) for l, y in points if l is not None and y is not None]
    if len(pts) < 2:
        return None
    L = np.array([p[0] for p in pts], dtype=np.float64)
    Y = np.array([p[1] for p in pts], dtype=np.float64)
    b, a = np.polyfit(L, Y, 1)  # deg=1 -> [기울기, 절편]
    a, b = float(a), float(b)
    y_pred = a + b * L
    ss_res = float(np.sum((Y - y_pred) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    residuals = [
        {"lens_position": float(l), "value": float(y), "predicted": float(yp), "residual": float(y - yp)}
        for l, y, yp in zip(L.tolist(), Y.tolist(), y_pred.tolist())
    ]
    return {"intercept": a, "slope": b, "r2": float(r2), "n_points": len(pts), "residuals": residuals}


def implied_focal_length_mm(fit: Optional[dict]) -> Optional[float]:
    """b/a[m] -> mm. 얇은렌즈 근사 fx(L)≈fx_inf*(1+f*L)(f=미터)에서 b/a=f 유도(모듈 docstring 참조)."""
    if fit is None:
        return None
    a, b = fit["intercept"], fit["slope"]
    if a == 0:
        return None
    return float((b / a) * 1000.0)


def check_physical_consistency(
    fx_fit: Optional[dict],
    nominal_focal_length_mm: float = DEFAULT_NOMINAL_FOCAL_LENGTH_MM,
    ratio: float = DEFAULT_FOCAL_LENGTH_CHECK_RATIO,
) -> dict:
    name = "physical_consistency_focal_length"
    implied = implied_focal_length_mm(fx_fit)
    if implied is None:
        return {"name": name, "ok": False, "detail": "fx-vs-LensPosition 직선적합 없음(그룹<2 또는 절편=0) - 검사 불가"}
    lo, hi = nominal_focal_length_mm / ratio, nominal_focal_length_mm * ratio
    ok = bool(implied > 0 and lo <= implied <= hi)
    verdict = "일관됨" if ok else "불일치 - LensPosition=1/거리(디옵터) 가정 또는 직선적합 자체가 깨졌을 수 있음"
    detail = (
        f"b/a={implied:.3f}mm (IMX708급 통상 초점거리 ~{nominal_focal_length_mm:.1f}mm, "
        f"허용범위 [{lo:.2f},{hi:.2f}]mm, ratio={ratio}) -> {verdict}"
    )
    return {"name": name, "ok": ok, "detail": detail}


def diopter_assumption_rows(group_analyses: Sequence[GroupAnalysis]) -> List[dict]:
    """그룹마다 locked_controls.lens_position vs 1/distance_m 대조 표(촬영 세션의 명시적 목적)."""
    rows = []
    for ga in group_analyses:
        if ga.lens_position is None or ga.key[1] <= 0:
            continue
        inv_d = 1.0 / ga.key[1]
        rel_err = abs(ga.lens_position - inv_d) / inv_d
        rows.append(
            {
                "set": ga.key[0], "distance_m": ga.key[1],
                "lens_position": ga.lens_position, "inverse_distance_m": inv_d,
                "relative_error": rel_err,
            }
        )
    return rows


def check_diopter_assumption(rows: Sequence[dict], tol_frac: float = DEFAULT_DIOPTER_TOLERANCE_FRAC) -> dict:
    name = "diopter_assumption_lens_position_vs_inverse_distance"
    if not rows:
        return {"name": name, "ok": False, "detail": "대조할 그룹 없음(lens_position 또는 distance_m 정보 부족)"}
    lines = [
        f"{r['set']}/{r['distance_m']:g}m: L={r['lens_position']:.3f} vs 1/d={r['inverse_distance_m']:.3f} "
        f"(상대오차 {r['relative_error'] * 100:.1f}%)"
        for r in rows
    ]
    max_rel = max(r["relative_error"] for r in rows)
    ok = bool(max_rel <= tol_frac)
    detail = f"최대 상대오차 {max_rel * 100:.1f}% (허용 {tol_frac * 100:.0f}%)\n" + "\n".join(lines)
    return {"name": name, "ok": ok, "detail": detail}


# ===========================================================================
# 세트 B 대조 (모델 무관 기준값)
# ===========================================================================


def find_set_b_group(group_analyses: Sequence[GroupAnalysis]) -> Optional[GroupAnalysis]:
    """set=='B'이고 캘리브레이션에 성공한 그룹 중 이미지 최다(스펙상 거리는 하나뿐이지만 방어적으로)."""
    candidates = [ga for ga in group_analyses if ga.key[0] == "B" and ga.calib_final is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda ga: len(ga.calib_final.images))


def check_set_b_extrapolation(
    fx_fit: Optional[dict],
    fy_fit: Optional[dict],
    set_b_ga: Optional[GroupAnalysis],
    all_valid_groups: Sequence[GroupAnalysis],
    tol_pct: float = DEFAULT_SET_B_CONSISTENCY_TOL_PCT,
) -> dict:
    name = "set_b_extrapolation_consistency"
    if set_b_ga is None or set_b_ga.calib_final is None:
        return {"name": name, "ok": False, "detail": "세트 B 그룹 없음 - 대조 불가"}
    if fx_fit is None or fy_fit is None or set_b_ga.lens_position is None:
        return {"name": name, "ok": False, "detail": "fx/fy-vs-LensPosition 적합 없음(그룹<2) 또는 세트B lens_position 없음 - 대조 불가"}

    L = set_b_ga.lens_position
    fx_pred = fx_fit["intercept"] + fx_fit["slope"] * L
    fy_pred = fy_fit["intercept"] + fy_fit["slope"] * L
    fx_actual = float(set_b_ga.calib_final.K[0, 0])
    fy_actual = float(set_b_ga.calib_final.K[1, 1])
    fx_pct = abs(fx_pred - fx_actual) / fx_actual * 100.0 if fx_actual else float("nan")
    fy_pct = abs(fy_pred - fy_actual) / fy_actual * 100.0 if fy_actual else float("nan")
    ok = bool(fx_pct <= tol_pct and fy_pct <= tol_pct)

    dist_b = set_b_ga.calib_final.dist
    dist_lines = []
    for ga in all_valid_groups:
        if ga is set_b_ga or ga.calib_final is None:
            continue
        diff = np.asarray(ga.calib_final.dist, dtype=np.float64) - np.asarray(dist_b, dtype=np.float64)
        dist_lines.append(f"  {ga.key[0]}/{ga.key[1]:g}m dist-distB = {np.array2string(diff, precision=4)}")

    detail = (
        f"fx: 외삽예측={fx_pred:.2f}px vs 세트B실측={fx_actual:.2f}px (차이 {fx_pct:.2f}%)\n"
        f"fy: 외삽예측={fy_pred:.2f}px vs 세트B실측={fy_actual:.2f}px (차이 {fy_pct:.2f}%)\n"
        f"허용 {tol_pct:.1f}% · 세트B distCoeffs={np.array2string(dist_b, precision=4)}\n"
        + ("왜곡계수 세트간 비교(대비 세트B):\n" + "\n".join(dist_lines) if dist_lines else "")
    )
    return {"name": name, "ok": ok, "detail": detail}


# ===========================================================================
# 정합성 검사 (실패해도 크래시 없이 보고)
# ===========================================================================


def compute_hfov_deg(fx: float, width_px: float) -> float:
    return math.degrees(2.0 * math.atan(width_px / (2.0 * fx)))


def check_aspect_ratio(K: np.ndarray, tol: float = DEFAULT_ASPECT_RATIO_TOL) -> dict:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    ratio = fy / fx if fx else float("nan")
    ok = bool(abs(ratio - 1.0) <= tol)
    return {"name": "aspect_ratio_fy_over_fx", "ok": ok, "detail": f"fy/fx={ratio:.4f} (허용 1±{tol})"}


def check_principal_point(K: np.ndarray, image_size: Tuple[int, int], tol_frac: float = DEFAULT_PRINCIPAL_POINT_TOL_FRAC) -> dict:
    w, h = image_size
    cx, cy = float(K[0, 2]), float(K[1, 2])
    dx = abs(cx - w / 2.0) / w if w else float("nan")
    dy = abs(cy - h / 2.0) / h if h else float("nan")
    ok = bool(dx <= tol_frac and dy <= tol_frac)
    detail = (
        f"(cx,cy)=({cx:.1f},{cy:.1f}), 이미지중심=({w / 2:.1f},{h / 2:.1f}), "
        f"편차=({dx * 100:.1f}%,{dy * 100:.1f}%) (허용 {tol_frac * 100:.0f}%)"
    )
    return {"name": "principal_point_near_center", "ok": ok, "detail": detail}


def check_hfov(
    K: np.ndarray, image_size: Tuple[int, int],
    nominal_deg: float = DEFAULT_HFOV_NOMINAL_DEG, tol_deg: float = DEFAULT_HFOV_TOL_DEG,
) -> dict:
    w, _h = image_size
    fx = float(K[0, 0])
    hfov = compute_hfov_deg(fx, w)
    ok = bool(abs(hfov - nominal_deg) <= tol_deg)
    detail = (
        f"계산 HFOV=2*atan(W/2fx)={hfov:.2f}° vs 기준 {nominal_deg:.1f}° (허용 ±{tol_deg:.1f}°). "
        f"주의: 세션 지시 원문의 75°를 기본값으로 쓰지만 docs/vision_status.md는 이 75°를 "
        f"대각(diagonal)로, calib_capture.py의 GROUND_WIDTH_COEFF_PER_M은 16:9 환산 실측 "
        f"수평화각 67.5°를 채택했다 — 두 문서가 다른 값을 '실측 화각'으로 쓰고 있어 하나로 "
        f"조용히 확정하지 않았다. --hfov-nominal-deg로 조정 가능."
    )
    return {"name": "hfov_vs_measured", "ok": ok, "detail": detail}


def check_group_image_counts(group_analyses: Sequence[GroupAnalysis], min_images: int = DEFAULT_MIN_IMAGES_PER_GROUP) -> dict:
    lines = []
    ok = True
    for ga in group_analyses:
        n = len(ga.calib_final.images) if ga.calib_final else 0
        lines.append(f"{ga.key[0]}/{ga.key[1]:g}m={n}장")
        if n < min_images:
            ok = False
    detail = f"최소 {min_images}장 기준 — " + ", ".join(lines)
    return {"name": "group_image_counts", "ok": bool(ok), "detail": detail}


def check_dataset_consistency(group_analyses: Sequence[GroupAnalysis]) -> dict:
    """보드/이미지 크기가 그룹 간에 일관되는지(다른 보드/해상도가 섞여 들어왔는지) 확인."""
    name = "dataset_consistency_board_and_image_size"
    valid = [ga for ga in group_analyses if ga.calib_final is not None]
    if not valid:
        return {"name": name, "ok": False, "detail": "유효 그룹 없음"}
    patterns = sorted({ga.pattern_size for ga in valid})
    squares = sorted({round(ga.square_size_mm, 3) for ga in valid})
    sizes = sorted({ga.image_size for ga in valid})
    ok = bool(len(patterns) == 1 and len(squares) == 1 and len(sizes) == 1)
    detail = f"pattern_size={patterns}, square_size_mm={squares}, image_size={sizes}"
    return {"name": name, "ok": ok, "detail": detail}


# ===========================================================================
# recommended 결정
# ===========================================================================


def determine_recommended(
    group_analyses: Sequence[GroupAnalysis],
    fx_fit: Optional[dict],
    fy_fit: Optional[dict],
    source: str = "auto",
) -> dict:
    """{"which","camera_matrix"(3x3 ndarray),"dist_coeffs"(ndarray),"note"} 반환.

    source:
      "auto"         — (기본, §8 요구대로) fx/fy=무한대외삽 절편, cx/cy/dist=세트B.
                        그룹<2(적합 불가)면 LensPosition 최저 그룹 전체로 폴백(§9).
      "B"            — 세트B 전체(fx,fy,cx,cy,dist 전부)를 강제 사용.
      "lowest-lens"  — LensPosition 최저 그룹 전체를 강제 사용.
      "SET:distance" — 예: "A:0.5" — 해당 그룹 전체를 강제 사용.
    """
    valid = [ga for ga in group_analyses if ga.calib_final is not None]
    if not valid:
        raise ValueError("determine_recommended: 캘리브레이션에 성공한 그룹이 하나도 없음")

    b_candidates = [ga for ga in valid if ga.key[0] == "B"]
    b_ga = max(b_candidates, key=lambda ga: len(ga.calib_final.images)) if b_candidates else None
    with_lens = [ga for ga in valid if ga.lens_position is not None]
    lowest_lens_ga = min(with_lens, key=lambda ga: ga.lens_position) if with_lens else valid[0]
    largest_ga = max(valid, key=lambda ga: len(ga.calib_final.images))

    def _whole_group(ga: GroupAnalysis) -> Tuple[float, float, float, float, np.ndarray]:
        K = ga.calib_final.K
        return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), ga.calib_final.dist

    def _explicit(spec: str) -> Optional[GroupAnalysis]:
        try:
            set_part, dist_part = spec.split(":")
            dist_val = float(dist_part)
        except ValueError:
            return None
        for ga in valid:
            if ga.key[0] == set_part and abs(ga.key[1] - dist_val) < 1e-6:
                return ga
        return None

    if source == "auto":
        if fx_fit is not None and fy_fit is not None:
            fx, fy = fx_fit["intercept"], fy_fit["intercept"]
            src_ga = b_ga if b_ga is not None else largest_ga
            cx, cy = float(src_ga.calib_final.K[0, 2]), float(src_ga.calib_final.K[1, 2])
            dist = src_ga.calib_final.dist
            b_desc = "세트B(장수최다·커버리지최대·운용조건최근접)" if b_ga is not None else "세트B 없음 — 이미지 최다 그룹으로 대체"
            which = f"fx/fy=lens_position_fit intercept(L=0,무한대외삽); cx/cy/dist_coeffs={src_ga.key[0]}/{src_ga.key[1]:g}m({b_desc})"
            note = (
                "fx,fy는 LensPosition->0(무한대 초점) 외삽 절편이다 — 기체는 10~40m 상공에서 "
                "사실상 무한대로 촬영하므로 이게 임무용 값이다. cx,cy,dist_coeffs는 "
                + b_desc + "에서 그대로 가져왔다(§8 recommended 결정 근거)."
            )
        else:
            ga = lowest_lens_ga
            fx, fy, cx, cy, dist = _whole_group(ga)
            which = f"직선적합 불가(유효 그룹<2) - LensPosition 최저 그룹({ga.key[0]}/{ga.key[1]:g}m) 전체 폴백"
            note = (
                "그룹이 2개 미만(또는 lens_position 정보 부족)이라 fx-vs-LensPosition 직선적합을 "
                "수행할 수 없었다. 무한대 초점에 가장 가까운, 즉 LensPosition이 가장 낮은 그룹의 "
                "camera_matrix/dist_coeffs를 그대로 recommended로 폴백했다(§9 견고성 요구)."
            )
    elif source == "B":
        if b_ga is None:
            raise ValueError("--recommended-source B: 세트 B 그룹이 없음")
        fx, fy, cx, cy, dist = _whole_group(b_ga)
        which = f"세트B({b_ga.key[0]}/{b_ga.key[1]:g}m) 전체 강제 사용(--recommended-source B)"
        note = "사용자가 --recommended-source B로 세트B 단독 캘리브레이션 결과를 명시적으로 지정했다."
    elif source == "lowest-lens":
        ga = lowest_lens_ga
        fx, fy, cx, cy, dist = _whole_group(ga)
        which = f"LensPosition 최저 그룹({ga.key[0]}/{ga.key[1]:g}m) 전체 강제 사용(--recommended-source lowest-lens)"
        note = "사용자가 --recommended-source lowest-lens로 명시적으로 지정했다."
    else:
        ga = _explicit(source)
        if ga is None:
            raise ValueError(f"--recommended-source {source!r}: 'SET:distance_m' 형식이거나 일치하는 그룹이 없음")
        fx, fy, cx, cy, dist = _whole_group(ga)
        which = f"명시 그룹({source}) 전체 강제 사용(--recommended-source)"
        note = f"사용자가 --recommended-source {source}로 명시적으로 지정했다."

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_arr = np.asarray(dist, dtype=np.float64).reshape(-1)
    return {"which": which, "camera_matrix": K, "dist_coeffs": dist_arr, "note": note}


# ===========================================================================
# 아티팩트 (build/save/next_calib_id)
# ===========================================================================


def next_calib_id(out_dir: Path, camera_id: str, date_str: str) -> str:
    """<camera_id>_<date_str><letter> — 같은 날짜에 이미 있으면 a,b,c...로 다음 글자."""
    out_dir = Path(out_dir)
    for i in range(26):
        letter = chr(ord("a") + i)
        candidate = f"{camera_id}_{date_str}{letter}"
        if not (out_dir / f"{candidate}.yaml").exists():
            return candidate
    raise RuntimeError(f"next_calib_id: {date_str}에 이미 a~z 26개가 있음 — 수동 확인 필요")


def _fit_to_yaml(fit: Optional[dict]) -> Optional[dict]:
    if fit is None:
        return None
    return {
        "intercept": fit["intercept"], "slope": fit["slope"], "r2": fit["r2"],
        "n_points": fit["n_points"], "residuals": fit["residuals"],
    }


def build_artifact(
    group_analyses: Sequence[GroupAnalysis],
    fx_fit: Optional[dict],
    fy_fit: Optional[dict],
    checks: Sequence[dict],
    recommended: dict,
    camera_id: str,
    calib_id: str,
    raw_root: str,
    git_commit: str,
    created_utc: str,
) -> dict:
    """yaml로 그대로 dump 가능한 순수 dict(전부 python native 타입)를 만든다."""
    valid = [ga for ga in group_analyses if ga.calib_final is not None]
    if not valid:
        raise ValueError("build_artifact: 유효한(캘리브레이션 성공) 그룹이 하나도 없음")

    pattern_size = valid[0].pattern_size
    square_size_mm = valid[0].square_size_mm
    image_size = valid[0].image_size

    n_used = sum(len(ga.calib_final.images) for ga in valid)
    n_rejected = sum(ga.n_total_images - len(ga.calib_final.images) for ga in valid)

    artifact = {
        "schema_version": 1,
        "calib_id": calib_id,
        "camera_id": camera_id,
        "created_utc": created_utc,
        "git_commit": git_commit,
        "source": {
            "raw_root": str(raw_root),
            "board": {"pattern_size": list(pattern_size), "square_size_mm": square_size_mm},
            "n_images_used": int(n_used),
            "n_images_rejected": int(n_rejected),
        },
        "image_size": list(image_size),
        "recommended": {
            "which": recommended["which"],
            "camera_matrix": recommended["camera_matrix"].tolist(),
            "dist_coeffs": recommended["dist_coeffs"].tolist(),
            "note": recommended["note"],
        },
        "groups": [
            {
                "set": ga.key[0],
                "distance_m": ga.key[1],
                "lens_position": ga.lens_position,
                "n_images": len(ga.calib_final.images),
                "rms_px": float(ga.calib_final.rms),
                "camera_matrix": ga.calib_final.K.tolist(),
                "dist_coeffs": ga.calib_final.dist.tolist(),
            }
            for ga in valid
        ],
        "lens_position_fit": {
            "fx": _fit_to_yaml(fx_fit),
            "fy": _fit_to_yaml(fy_fit),
            "implied_focal_length_mm": implied_focal_length_mm(fx_fit),
        },
        "checks": [dict(c) for c in checks],
    }
    return artifact


def save_artifact(artifact: dict, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(artifact, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return out_path


# ===========================================================================
# 진단 플롯 (Agg, headless-safe)
# ===========================================================================


def plot_corner_coverage(group_analyses: Sequence[GroupAnalysis], out_path: Path) -> Optional[Path]:
    """그룹별 코너 산포도 — "네 모서리를 채웠는가" 요구조건 검증용."""
    valid = [ga for ga in group_analyses if ga.calib_final is not None]
    if not valid:
        return None
    n = len(valid)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    for idx, ga in enumerate(valid):
        ax = axes[idx // cols][idx % cols]
        w, h = ga.image_size
        for im in ga.calib_final.images:
            pts = np.asarray(im.corners_px).reshape(-1, 2)
            ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.5, color="tab:blue")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # 이미지 좌표계(y 아래로 증가)
        ax.set_title(f"{ga.key[0]}/{ga.key[1]:g}m ({len(ga.calib_final.images)}장)", fontsize=9)
        ax.set_aspect("equal")
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle("그룹별 코너 산포도(프레임 커버리지)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_reprojection_errors(group_analyses: Sequence[GroupAnalysis], out_path: Path) -> Optional[Path]:
    """사진별 재투영오차 막대그래프(이상치 판정 시점=재캘리브레이션 전 기준, 이상치는 빨강)."""
    rows = []
    for ga in group_analyses:
        if ga.calib_before is None:
            continue
        for im, err, is_out in zip(ga.calib_before.images, ga.errors_before, ga.outlier_mask):
            rows.append((f"{ga.key[0]}/{ga.key[1]:g}m#{im.seq:03d}", err, is_out))
    if not rows:
        return None
    rows.sort(key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows]
    errs = [r[1] for r in rows]
    colors = ["tab:red" if r[2] else "tab:blue" for r in rows]

    fig, ax = plt.subplots(figsize=(max(8.0, len(rows) * 0.22), 5.0))
    ax.bar(range(len(rows)), errs, color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("재투영오차 RMS (px)")
    ax.set_title("사진별 재투영오차 (빨강=이상치)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_fx_vs_lens_position(fx_fit: Optional[dict], out_path: Path) -> Optional[Path]:
    """fx vs LensPosition 산포 + 적합선 + L=0 외삽 절편 표시."""
    if fx_fit is None:
        return None
    residuals = fx_fit["residuals"]
    L = [r["lens_position"] for r in residuals]
    Y = [r["value"] for r in residuals]
    a, b = fx_fit["intercept"], fx_fit["slope"]
    l_max = max(L + [0.0])
    xs = np.linspace(0.0, l_max * 1.1 if l_max > 0 else 1.0, 50)
    ys = a + b * xs

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(L, Y, color="tab:blue", zorder=3, label="그룹별 fx")
    ax.plot(xs, ys, color="tab:orange", label=f"적합: fx={a:.1f}+{b:.1f}*L (R²={fx_fit['r2']:.3f})")
    ax.scatter([0.0], [a], color="tab:red", marker="*", s=160, zorder=4, label=f"L=0 외삽: fx={a:.1f}px")
    ax.set_xlabel("LensPosition (diopter)")
    ax.set_ylabel("fx (px)")
    ax.set_title("fx vs LensPosition — 무한대 외삽")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ===========================================================================
# 사람이 읽는 리포트(stdout)
# ===========================================================================


def print_report(
    group_analyses: Sequence[GroupAnalysis],
    fx_fit: Optional[dict],
    fy_fit: Optional[dict],
    checks: Sequence[dict],
    recommended: dict,
    yaml_path: Path,
    plot_paths: Sequence[Path],
) -> None:
    print("=" * 78)
    print("그룹별 캘리브레이션 결과")
    print("=" * 78)
    for ga in group_analyses:
        if ga.calib_final is None:
            print(f"[{ga.key[0]}/{ga.key[1]:g}m] 실패: {ga.error_msg}")
            continue
        n_after = len(ga.calib_final.images)
        print(
            f"[{ga.key[0]}/{ga.key[1]:g}m] L={ga.lens_position} n={ga.n_total_images}->{n_after} "
            f"RMS(전)={ga.calib_before.rms:.4f}px RMS(후)={ga.calib_final.rms:.4f}px "
            f"fx={ga.calib_final.K[0, 0]:.2f} fy={ga.calib_final.K[1, 1]:.2f}"
        )
        outliers = [
            (im, err) for im, err, is_out in zip(ga.calib_before.images, ga.errors_before, ga.outlier_mask) if is_out
        ]
        if outliers:
            print("  불량컷(재투영오차 이상치, 촬영 기법 피드백용):")
            for im, err in sorted(outliers, key=lambda t: t[1], reverse=True):
                print(
                    f"    {im.json_path.name} err={err:.3f}px [{im.label_ko}] zone={im.zone} "
                    f"yaw={im.yaw_deg:+.0f} pitch={im.pitch_deg:+.0f} roll={im.roll_deg:+.0f}"
                )

    print("-" * 78)
    print("fx/fy vs LensPosition 직선적합(무한대 외삽)")
    for name, fit in (("fx", fx_fit), ("fy", fy_fit)):
        if fit is None:
            print(f"  {name}: 적합 생략(유효 그룹 < 2)")
            continue
        print(f"  {name}: a(절편,L=0)={fit['intercept']:.3f} b(기울기)={fit['slope']:.4f} R²={fit['r2']:.4f}")
        for r in fit["residuals"]:
            print(f"    L={r['lens_position']:.3f} 실측={r['value']:.3f} 예측={r['predicted']:.3f} 잔차={r['residual']:+.3f}")

    print("-" * 78)
    print("정합성 검사")
    for c in checks:
        mark = "OK  " if c["ok"] else "FAIL"
        print(f"  [{mark}] {c['name']}")
        for line in c["detail"].splitlines():
            print(f"          {line}")

    print("-" * 78)
    print(f"recommended: {recommended['which']}")
    print(f"  camera_matrix=\n{recommended['camera_matrix']}")
    print(f"  dist_coeffs={recommended['dist_coeffs']}")
    print(f"  note: {recommended['note']}")

    print("-" * 78)
    print(f"아티팩트: {yaml_path}")
    for p in plot_paths:
        print(f"플롯: {p}")


# ===========================================================================
# CLI
# ===========================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", default="vision/data/calibration_raw", help="calib_capture.py 촬영 트리 루트")
    parser.add_argument("--redetect", action="store_true", help="사이드카 corners_px 대신 PNG에서 재검출")
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID)
    parser.add_argument("--calib-id", default=None, help="미지정 시 <camera_id>_<date><letter> 자동 생성")
    parser.add_argument("--date", default=None, help="calib_id 자동생성용 날짜(YYYY-MM-DD), 기본 오늘(UTC)")
    parser.add_argument("--out-calib-dir", default="vision/calibration", help="아티팩트 yaml 루트(카메라별 하위폴더 자동)")
    parser.add_argument("--out-results-dir", default="vision/results", help="진단 플롯 PNG 저장 폴더")
    parser.add_argument("--recommended-source", default="auto", help="auto|B|lowest-lens|SET:distance_m")

    parser.add_argument("--outlier-abs-px", type=float, default=DEFAULT_OUTLIER_ABS_PX)
    parser.add_argument("--outlier-std-mult", type=float, default=DEFAULT_OUTLIER_STD_MULT)
    parser.add_argument("--min-images-for-recal", type=int, default=DEFAULT_MIN_IMAGES_FOR_RECAL)

    parser.add_argument("--nominal-focal-length-mm", type=float, default=DEFAULT_NOMINAL_FOCAL_LENGTH_MM)
    parser.add_argument("--focal-length-check-ratio", type=float, default=DEFAULT_FOCAL_LENGTH_CHECK_RATIO)
    parser.add_argument("--diopter-tolerance-frac", type=float, default=DEFAULT_DIOPTER_TOLERANCE_FRAC)

    parser.add_argument("--aspect-ratio-tol", type=float, default=DEFAULT_ASPECT_RATIO_TOL)
    parser.add_argument("--principal-point-tol-frac", type=float, default=DEFAULT_PRINCIPAL_POINT_TOL_FRAC)
    parser.add_argument("--hfov-nominal-deg", type=float, default=DEFAULT_HFOV_NOMINAL_DEG)
    parser.add_argument("--hfov-tol-deg", type=float, default=DEFAULT_HFOV_TOL_DEG)
    parser.add_argument("--set-b-consistency-tol-pct", type=float, default=DEFAULT_SET_B_CONSISTENCY_TOL_PCT)
    parser.add_argument("--min-images-per-group", type=int, default=DEFAULT_MIN_IMAGES_PER_GROUP)

    parser.add_argument("--no-plots", action="store_true", help="진단 플롯 생성 생략")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    raw_root = Path(args.raw_root)
    groups, warnings = load_dataset(raw_root, redetect=args.redetect)
    for w in warnings:
        print(f"경고: {w}", file=sys.stderr)

    if not groups:
        print(f"오류: '{raw_root}'에서 사용 가능한 캘리브레이션 데이터를 찾지 못함 — 종료", file=sys.stderr)
        return 1

    group_analyses = []
    for key in sorted(groups.keys()):
        ga = analyze_group(groups[key], args.outlier_abs_px, args.outlier_std_mult, args.min_images_for_recal)
        group_analyses.append(ga)
        if ga.error_msg:
            print(f"경고: 그룹 {key} 캘리브레이션 실패 - {ga.error_msg}", file=sys.stderr)

    valid = [ga for ga in group_analyses if ga.calib_final is not None]
    if not valid:
        print("오류: 모든 그룹의 캘리브레이션이 실패함 — 종료", file=sys.stderr)
        return 1

    fx_points = [(ga.lens_position, float(ga.calib_final.K[0, 0])) for ga in valid]
    fy_points = [(ga.lens_position, float(ga.calib_final.K[1, 1])) for ga in valid]
    fx_fit = linear_fit_lens_position(fx_points)
    fy_fit = linear_fit_lens_position(fy_points)
    if fx_fit is None:
        print("경고: fx-vs-LensPosition 직선적합 생략 — 유효 그룹(lens_position 보유) < 2", file=sys.stderr)

    try:
        recommended = determine_recommended(valid, fx_fit, fy_fit, source=args.recommended_source)
    except ValueError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1

    image_size = valid[0].image_size
    checks = [
        check_physical_consistency(fx_fit, args.nominal_focal_length_mm, args.focal_length_check_ratio),
        check_diopter_assumption(diopter_assumption_rows(valid), args.diopter_tolerance_frac),
        check_set_b_extrapolation(fx_fit, fy_fit, find_set_b_group(valid), valid, args.set_b_consistency_tol_pct),
        check_aspect_ratio(recommended["camera_matrix"], args.aspect_ratio_tol),
        check_principal_point(recommended["camera_matrix"], image_size, args.principal_point_tol_frac),
        check_hfov(recommended["camera_matrix"], image_size, args.hfov_nominal_deg, args.hfov_tol_deg),
        check_group_image_counts(valid, args.min_images_per_group),
        check_dataset_consistency(valid),
    ]

    date_str = args.date or datetime.now(timezone.utc).date().isoformat()
    out_calib_dir = Path(args.out_calib_dir) / args.camera_id
    calib_id = args.calib_id or next_calib_id(out_calib_dir, args.camera_id, date_str)
    created_utc = datetime.now(timezone.utc).isoformat()
    git_commit = get_git_commit_hash()

    artifact = build_artifact(
        valid, fx_fit, fy_fit, checks, recommended, args.camera_id, calib_id, str(raw_root), git_commit, created_utc
    )
    yaml_path = save_artifact(artifact, out_calib_dir / f"{calib_id}.yaml")

    plot_paths: List[Path] = []
    if not args.no_plots:
        results_dir = Path(args.out_results_dir)
        for p in (
            plot_corner_coverage(valid, results_dir / f"{calib_id}_corners.png"),
            plot_reprojection_errors(group_analyses, results_dir / f"{calib_id}_reproj_errors.png"),
            plot_fx_vs_lens_position(fx_fit, results_dir / f"{calib_id}_fx_vs_lens.png"),
        ):
            if p is not None:
                plot_paths.append(p)

    print_report(group_analyses, fx_fit, fy_fit, checks, recommended, yaml_path, plot_paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
