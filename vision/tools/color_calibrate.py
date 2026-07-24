"""
현장 색 캘리브레이터 (`docs/vision_plan.md` §5.5, §9 빌드순서 5번).

**지오메트리 캘리브레이션(카메라 인트린식, `calib_capture.py`/`calib_analyze.py`)과는
완전히 다른 개념이다** — §9 5번이 "여기서 '캘리브레이터'는 현장 조명 대응용 HSV 임계값
튜너로, 지오메트리 캘리브레이션과는 별개 개념"이라고 명시한다. 이 도구는 렌즈/센서를
전혀 다루지 않는다 — 오직 한 프레임의 ROI에서 뽑은 HSV 통계로 `ColorFilter`/`RedRingDetector`
생성자 인자를 제안할 뿐이다.

## 왜 필요한가

`docs/vision_plan.md` §5.5:
    "레이어드 HSV + 현장 색 캘리브레이터(라이브 ROI 샘플→임계값 자동 설정) — 규정·색이
    대회 직전 공개돼도 코드 수정 없이 대응."
§5.3이 ② 조난자 구역의 "핵심 난제 = 초록 색조 비동일"이라고 못박았다. coarse 색 파이프라인
(`ColorFilter` 등)은 이미 있지만, 지금까지는 사람이 preset yaml의 HSV 숫자를 손으로 추측해야
했다 — 대회 당일 조명/색이 예상과 다르면 이 도구로 현장에서 몇 분 만에 새 임계값을 뽑아
검토·반영할 수 있어야 한다는 게 이 도구의 존재 이유다.

## 흐름

1. **입력**: 프레임 이미지 파일 하나, 또는 녹화 폴더/영상 파일(`vision.utils.frame_source`의
   `open_dir_or_bag()`을 그대로 재사용 — 새 프레임 소스를 만들지 않는다). 정지 이미지 파일은
   `DirFrameSource`(디렉터리 전용)/`BagFrameSource`(영상 컨테이너 전용) 둘 다 원래 대상이
   아니라서 `vision.utils.image_loader.load_image()`로 직접 읽는다(가장 단순한 경로).
2. **ROI 지정**: `--roi x,y,w,h` 정수 좌표만 받는다. **GUI/마우스 인터랙션 없음** — 이 환경
   (WSL·RPi headless)엔 디스플레이가 없고, `cv2.selectROI`는 쓰지 않는다(세션 지시 금지 항목).
3. **통계 산출**: ROI를 HSV로 변환해 채널별 분포를 계산한다. **평균±N×표준편차 대신
   백분위수(percentile) 기반을 기본으로 쓴다** — 이유: 그림자/글레어/과노출 클리핑 같은
   국소 이상치 픽셀이 소수 섞여도(§5.5 "그림자·광택 글레어·과노출 클리핑 대비") 평균과
   표준편차는 그 이상치 쪽으로 쉽게 끌려가 임계값이 부풀거나(표준편차 팽창) 좁아진다
   (평균이 이상치 쪽으로 이동). 백분위수(기본 p5~p95)는 이상치가 전체의 일정 비율
   (기본 5% 미만)이면 그 값 자체를 사실상 무시하고 "주 색상 군집"의 실제 분포 경계를
   따른다 — `tests/test_color_calibrate.py::test_calibrate_roi_percentile_range_ignores_minority_outlier_pixels`가
   이 근거를 합성 이상치 픽셀로 직접 검증한다.
4. **⚠️ 빨강 Hue 랩어라운드 처리**: OpenCV HSV의 Hue는 0~179이고 빨강은 0 부근과 179 부근으로
   갈라진다. 이 코드베이스에 이미 관련 전례가 둘 있다 —
     - `vision/modules/color.py::ColorFilter` — 랩어라운드 **미지원**(§5.4 blind spot,
       `vision/CLAUDE.md` 테스트 규칙표에 회귀테스트로 기록됨). 단일 `hue_range=(low, high)`
       하나만 받으므로 애초에 두 구간을 표현할 수 없다.
     - `vision/modules/vertiport_ring.py::RedRingDetector` — **빨강 Hue 양끝 게이팅**으로
       랩어라운드에 대응하는 실제 구현(`low_hue_max`/`high_hue_min`/`sat_min`/`val_min`
       생성자 인자, `[0,low_hue_max]`∪`[high_hue_min,179]`를 OR로 게이팅). **이 도구가 따르는
       전례는 이쪽이다.**
   이 도구는 ROI Hue 표본을 `--wrap-split-hue`(기본 90) 기준 저/고 두 구간으로 나눠 각 구간의
   표본 비율이 둘 다 `--wrap-min-fraction`(기본 0.05) 이상이면 "랩어라운드"로 판정하고, 두
   구간(low/high)으로 나눠 `RedRingDetector` 호환 파라미터(`low_hue_max`/`high_hue_min`/
   `sat_min`/`val_min`)를 출력한다. 랩어라운드가 아니면 단일 `hue_range`로 `ColorFilter`
   호환 파라미터(`hue_range`/`sat_min`/`val_min`/`val_max`)를 출력한다. 어느 쪽이든 산출
   결과에 "호환 소비자"를 문자열로 명시한다.
5. **출력**: 산출된 임계값을 사람이 그대로 복붙할 수 있는 yaml 조각으로 stdout에 출력하고,
   `--output <path>`를 주면 같은 내용을 파일로도 저장한다. **기존 preset yaml을 자동으로
   덮어쓰지 않는다** — "제안"까지가 이 도구의 범위이고 사람이 검토 후 반영한다. 산출 근거
   (ROI 좌표·샘플 픽셀 수·백분위수·마진·랩어라운드 판정 근거·입력 파일)를 전부 yaml 주석으로
   함께 싣는다.
6. **진단 산출물**: `--diagnostic-dir <dir>`을 주면 (a) ROI 위치를 표시한 오버레이 이미지와
   (b) 채널별 HSV 히스토그램(제안된 임계값을 세로선으로 표시) PNG를 저장한다. 히스토그램은
   `vision/tools/jsonl_view.py`와 동일하게 matplotlib **Agg 백엔드**(headless-safe)를 쓴다.

## 임계값·마진 전부 CLI 파라미터 (매직넘버 금지, §7.3)

백분위수(`--low-percentile`/`--high-percentile`), 랩어라운드 판정 기준(`--wrap-split-hue`/
`--wrap-min-fraction`), 마진(`--hue-margin`/`--sat-margin`/`--val-margin`) 전부 CLI 인자로
노출된다 — 코드에 하드코딩된 숫자가 없다.

## 과설계 금지 (§5.5가 요구하지 않는 것은 넣지 않았다)

자동 색 추적·머신러닝 분류기·GUI(`cv2.selectROI` 등)·설정 자동 반영(preset yaml 자동 덮어쓰기)
전부 이 도구의 범위 밖이다. §5.5가 요구한 것은 "라이브 ROI 샘플 → 임계값 자동 설정" + "사람
검토" 두 가지뿐이다.

## 하드웨어 의존 없음 (`vision/CLAUDE.md` tools/ 예외 규칙)

`cv2`/`numpy`/`matplotlib`뿐 — `jsonl_view.py`/`calib_analyze.py`와 동일한 예외로 `.venv`
설치 + `tests/test_color_calibrate.py` pytest 대상이다.

## CLI

    python -m vision.tools.color_calibrate <이미지 파일|녹화폴더|영상파일> --roi x,y,w,h
    python -m vision.tools.color_calibrate frame.png --roi 100,100,60,60 \\
        --output vision/presets/_proposed_distress.yaml --diagnostic-dir vision/results/color_calib
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless-safe — GUI 강제 호출 금지(WSL/RPi에 디스플레이 없음)
import matplotlib.pyplot as plt

from vision.utils.frame_source import open_dir_or_bag
from vision.utils.image_loader import load_image

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

# --- 기본값(전부 CLI로 override 가능, §7.3 매직넘버 금지) ---
DEFAULT_LOW_PERCENTILE = 5.0
DEFAULT_HIGH_PERCENTILE = 95.0
DEFAULT_WRAP_SPLIT_HUE = 90
DEFAULT_WRAP_MIN_FRACTION = 0.05
DEFAULT_HUE_MARGIN = 0
DEFAULT_SAT_MARGIN = 0
DEFAULT_VAL_MARGIN = 0

HUE_MAX = 179
CHANNEL_MAX = 255


# ===========================================================================
# 프레임 로드 (frame_source 재사용 — 새 프레임 소스 구현 금지, 세션 지시)
# ===========================================================================


def load_frame(input_path: Path | str, frame_index: int = 0) -> np.ndarray:
    """이미지 파일 또는 녹화 폴더(디렉터리)/영상 파일에서 프레임 한 장을 읽는다.

    디렉터리·영상 파일은 `vision.utils.frame_source.open_dir_or_bag()`(DirFrameSource/
    BagFrameSource)을 그대로 재사용해 frame_index번째 프레임을 취한다. 단일 정지 이미지
    파일은 `DirFrameSource`(디렉터리 전용)도 `BagFrameSource`(영상 컨테이너 전용, cv2.VideoCapture
    기반이라 단일 정지 이미지에 대한 동작이 OpenCV 빌드마다 불안정함)도 원래 대상이 아니므로
    `vision.utils.image_loader.load_image()`로 직접 읽는다(가장 단순하고 확실한 경로).
    """
    input_path = Path(input_path)
    if input_path.is_dir():
        return _nth_frame_from_source(open_dir_or_bag(input_path), frame_index, str(input_path))
    if input_path.suffix.lower() in _IMAGE_SUFFIXES:
        return load_image(str(input_path))
    return _nth_frame_from_source(open_dir_or_bag(input_path), frame_index, str(input_path))


def _nth_frame_from_source(source, frame_index: int, label: str) -> np.ndarray:
    with source:
        for i, record in enumerate(source):
            if i == frame_index:
                return record.image
    raise ValueError(f"load_frame: frame_index={frame_index} — '{label}'에 그만큼의 프레임이 없음")


# ===========================================================================
# ROI 파싱/크롭
# ===========================================================================


def parse_roi(value: str) -> Tuple[int, int, int, int]:
    """`--roi x,y,w,h` 파싱. GUI 없음 — 좌표는 항상 사람이 숫자로 지정한다(세션 지시)."""
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"--roi는 x,y,w,h 4개 정수여야 합니다: {value!r}")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--roi는 정수만 허용합니다: {value!r}") from e
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"--roi의 w,h는 양수여야 합니다: {value!r}")
    return x, y, w, h


def crop_roi(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    img_h, img_w = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img_w, x + w), min(img_h, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"crop_roi: ROI가 프레임 범위를 벗어남 — roi={roi}, frame_size=({img_w}x{img_h})"
        )
    return image[y0:y1, x0:x1]


# ===========================================================================
# HSV 통계 + 임계값 산출 (순수 함수 — 하드웨어/파일 I/O 없음)
# ===========================================================================


def compute_hsv_channels(roi_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROI(BGR) → (hue, sat, val) 1차원 배열(각 픽셀 1개). int32로 반환(부호 걱정 없이 산술)."""
    if roi_bgr.size == 0:
        raise ValueError("compute_hsv_channels: ROI가 비어 있음(크기 0)")
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].reshape(-1).astype(np.int32)
    s = hsv[..., 1].reshape(-1).astype(np.int32)
    v = hsv[..., 2].reshape(-1).astype(np.int32)
    return h, s, v


def detect_hue_wraparound(
    hue: np.ndarray,
    wrap_split_hue: int = DEFAULT_WRAP_SPLIT_HUE,
    wrap_min_fraction: float = DEFAULT_WRAP_MIN_FRACTION,
) -> Tuple[bool, float, float]:
    """Hue 표본을 `wrap_split_hue` 기준 저/고 두 구간으로 나눠 둘 다 `wrap_min_fraction`
    이상의 비율을 차지하면 랩어라운드로 판정한다(`RedRingDetector`의 두 구간 게이팅과 같은
    분할 기준). 한쪽 구간이 노이즈 수준(< wrap_min_fraction)이면 단일 군집으로 본다 —
    그래야 순수 초록/파랑 등 원래 랩어라운드가 없는 색까지 오판하지 않는다."""
    n = int(hue.size)
    if n == 0:
        return False, 0.0, 0.0
    frac_low = float(np.count_nonzero(hue <= wrap_split_hue)) / n
    frac_high = float(np.count_nonzero(hue > wrap_split_hue)) / n
    wraparound = frac_low >= wrap_min_fraction and frac_high >= wrap_min_fraction
    return wraparound, frac_low, frac_high


def _clip_int(value: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(value))))


@dataclass
class CalibrationResult:
    """`calibrate_roi()`의 산출물 — 임계값 제안 + 산출 근거(전부 yaml 주석/리포트에 실림)."""

    roi: Tuple[int, int, int, int]
    n_pixels: int
    low_percentile: float
    high_percentile: float
    hue_margin: int
    sat_margin: int
    val_margin: int
    hue_median: float
    sat_median: float
    val_median: float
    hue_wraparound: bool
    wrap_split_hue: int
    wrap_min_fraction: float
    frac_hue_low: float
    frac_hue_high: float
    consumer: str
    params: dict


def calibrate_roi(
    roi_bgr: np.ndarray,
    *,
    roi_box: Tuple[int, int, int, int] = (0, 0, 0, 0),
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
    wrap_split_hue: int = DEFAULT_WRAP_SPLIT_HUE,
    wrap_min_fraction: float = DEFAULT_WRAP_MIN_FRACTION,
    hue_margin: int = DEFAULT_HUE_MARGIN,
    sat_margin: int = DEFAULT_SAT_MARGIN,
    val_margin: int = DEFAULT_VAL_MARGIN,
) -> CalibrationResult:
    """ROI(BGR) → `CalibrationResult`. 순수 함수(파일 I/O 없음) — 합성 패치로 직접 테스트 가능.

    랩어라운드면 `RedRingDetector` 호환(`low_hue_max`/`high_hue_min`/`sat_min`/`val_min`),
    아니면 `ColorFilter(mode=color)` 호환(`hue_range`/`sat_min`/`val_min`/`val_max`) 파라미터를
    `params`에 담는다.
    """
    h, s, v = compute_hsv_channels(roi_bgr)
    n = int(h.size)

    wraparound, frac_low, frac_high = detect_hue_wraparound(h, wrap_split_hue, wrap_min_fraction)

    sat_min = _clip_int(np.percentile(s, low_percentile) - sat_margin, 0, CHANNEL_MAX)
    val_min = _clip_int(np.percentile(v, low_percentile) - val_margin, 0, CHANNEL_MAX)
    val_max = _clip_int(np.percentile(v, high_percentile) + val_margin, 0, CHANNEL_MAX)

    if wraparound:
        low_side = h[h <= wrap_split_hue]
        high_side = h[h > wrap_split_hue]
        low_hue_max = _clip_int(np.percentile(low_side, high_percentile) + hue_margin, 0, HUE_MAX)
        high_hue_min = _clip_int(np.percentile(high_side, low_percentile) - hue_margin, 0, HUE_MAX)
        params = {
            "low_hue_max": low_hue_max,
            "high_hue_min": high_hue_min,
            "sat_min": sat_min,
            "val_min": val_min,
        }
        consumer = "RedRingDetector(vision/modules/vertiport_ring.py) — 빨강 Hue 양끝 게이팅 생성자 인자"
    else:
        hue_lo = _clip_int(np.percentile(h, low_percentile) - hue_margin, 0, HUE_MAX)
        hue_hi = _clip_int(np.percentile(h, high_percentile) + hue_margin, 0, HUE_MAX)
        params = {
            "hue_range": [hue_lo, hue_hi],
            "sat_min": sat_min,
            "val_min": val_min,
            "val_max": val_max,
        }
        consumer = "ColorFilter(vision/modules/color.py, mode=color) 생성자 인자"

    return CalibrationResult(
        roi=roi_box,
        n_pixels=n,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        hue_margin=hue_margin,
        sat_margin=sat_margin,
        val_margin=val_margin,
        hue_median=float(np.median(h)),
        sat_median=float(np.median(s)),
        val_median=float(np.median(v)),
        hue_wraparound=wraparound,
        wrap_split_hue=wrap_split_hue,
        wrap_min_fraction=wrap_min_fraction,
        frac_hue_low=frac_low,
        frac_hue_high=frac_high,
        consumer=consumer,
        params=params,
    )


# ===========================================================================
# 출력 — 사람이 복붙할 yaml 조각 (자동 반영 아님, "제안"까지가 범위)
# ===========================================================================


def format_yaml_snippet(result: CalibrationResult, *, input_label: str, frame_index: int) -> str:
    lines = [
        "# 현장 색 캘리브레이터(vision/tools/color_calibrate.py) 산출 — 사람이 검토 후 반영할 것.",
        "#   기존 preset yaml을 자동으로 덮어쓰지 않는다 — 이 조각을 복붙해 판단 후 넣는다.",
        f"# 입력: {input_label} (frame_index={frame_index})",
        f"# ROI(x,y,w,h)={list(result.roi)}  샘플 픽셀 수={result.n_pixels}",
        (
            f"# 백분위수 p{result.low_percentile:g}~p{result.high_percentile:g} 기반 "
            f"(평균±표준편차 대신 — 이상치에 강함, 도구 docstring 참조). "
            f"중앙값 H={result.hue_median:.1f} S={result.sat_median:.1f} V={result.val_median:.1f}"
        ),
        f"# 마진: hue±{result.hue_margin}  sat-{result.sat_margin}  val±{result.val_margin}",
    ]
    if result.hue_wraparound:
        lines.append(
            f"# Hue 랩어라운드 감지됨 (split={result.wrap_split_hue}, "
            f"저구간비율={result.frac_hue_low:.2f}, 고구간비율={result.frac_hue_high:.2f}) "
            "→ low/high 두 구간으로 출력"
        )
    else:
        lines.append(
            f"# Hue 랩어라운드 없음 (split={result.wrap_split_hue} 기준 한쪽 구간 비율이 "
            f"wrap_min_fraction={result.wrap_min_fraction:g} 미만: "
            f"저={result.frac_hue_low:.2f} 고={result.frac_hue_high:.2f}) → 단일 hue_range로 출력"
        )
    lines.append(f"# 호환 소비자: {result.consumer}")
    lines.append("")

    if result.hue_wraparound:
        lines.append("red_ring:  # RedRingDetector 생성자 인자와 동일한 키(vertiport_ring.py)")
        lines.append(f"  low_hue_max: {result.params['low_hue_max']}")
        lines.append(f"  high_hue_min: {result.params['high_hue_min']}")
        lines.append(f"  sat_min: {result.params['sat_min']}")
        lines.append(f"  val_min: {result.params['val_min']}")
    else:
        lines.append("color_filter:  # ColorFilter preset 키(color.py)")
        lines.append("  mode: color")
        lines.append(f"  hue_range: {result.params['hue_range']}")
        lines.append(f"  sat_min: {result.params['sat_min']}")
        lines.append(f"  val_min: {result.params['val_min']}")
        lines.append(f"  val_max: {result.params['val_max']}")

    return "\n".join(lines) + "\n"


# ===========================================================================
# 진단 산출물 (Agg, headless-safe — jsonl_view.py와 동일 패턴)
# ===========================================================================


def save_roi_overlay(frame_bgr: np.ndarray, roi: Tuple[int, int, int, int], out_path: Path) -> Path:
    """ROI 위치를 표시한 오버레이 이미지를 PNG로 저장(cv2, GUI 호출 없음 — imwrite만)."""
    x, y, w, h = roi
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    return out_path


def save_hsv_histogram(
    h: np.ndarray, s: np.ndarray, v: np.ndarray, result: CalibrationResult, out_path: Path
) -> Path:
    """채널별 HSV 히스토그램(제안된 임계값을 세로 점선으로 표시) — matplotlib Agg."""
    fig, (ax_h, ax_s, ax_v) = plt.subplots(3, 1, figsize=(7, 6.5))

    ax_h.hist(h, bins=60, range=(0, HUE_MAX), color="tab:blue", alpha=0.8)
    ax_h.set_title("Hue (0-179)", fontsize=9)
    if result.hue_wraparound:
        ax_h.axvline(result.params["low_hue_max"], color="tab:red", ls="--", lw=1)
        ax_h.axvline(result.params["high_hue_min"], color="tab:red", ls="--", lw=1)
    else:
        ax_h.axvline(result.params["hue_range"][0], color="tab:red", ls="--", lw=1)
        ax_h.axvline(result.params["hue_range"][1], color="tab:red", ls="--", lw=1)

    ax_s.hist(s, bins=64, range=(0, CHANNEL_MAX), color="tab:green", alpha=0.8)
    ax_s.set_title("Saturation (0-255)", fontsize=9)
    ax_s.axvline(result.params["sat_min"], color="tab:red", ls="--", lw=1)

    ax_v.hist(v, bins=64, range=(0, CHANNEL_MAX), color="tab:orange", alpha=0.8)
    ax_v.set_title("Value (0-255)", fontsize=9)
    ax_v.axvline(result.params["val_min"], color="tab:red", ls="--", lw=1)
    if not result.hue_wraparound:
        ax_v.axvline(result.params["val_max"], color="tab:red", ls="--", lw=1)

    fig.suptitle(f"color_calibrate — ROI={list(result.roi)} (n={result.n_pixels})")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ===========================================================================
# CLI
# ===========================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="프레임 이미지 파일, 또는 녹화 폴더(디렉터리)/영상 파일 경로")
    parser.add_argument(
        "--roi", required=True, type=parse_roi, metavar="x,y,w,h",
        help="샘플링할 ROI 정수 좌표 — GUI 없음(cv2.selectROI 사용 안 함), 반드시 CLI로 지정",
    )
    parser.add_argument(
        "--frame-index", type=int, default=0,
        help="녹화 폴더/영상 입력일 때 사용할 프레임 번호(기본 0)",
    )
    parser.add_argument("--low-percentile", type=float, default=DEFAULT_LOW_PERCENTILE)
    parser.add_argument("--high-percentile", type=float, default=DEFAULT_HIGH_PERCENTILE)
    parser.add_argument(
        "--wrap-split-hue", type=int, default=DEFAULT_WRAP_SPLIT_HUE,
        help="Hue 랩어라운드 판정용 저/고 분할 기준(기본 90, 0~179 중앙)",
    )
    parser.add_argument(
        "--wrap-min-fraction", type=float, default=DEFAULT_WRAP_MIN_FRACTION,
        help="양쪽 구간 모두 이 비율 이상이어야 랩어라운드로 판정(기본 0.05)",
    )
    parser.add_argument("--hue-margin", type=int, default=DEFAULT_HUE_MARGIN, help="hue 경계에 더할 여유(기본 0)")
    parser.add_argument("--sat-margin", type=int, default=DEFAULT_SAT_MARGIN, help="sat_min에서 뺄 여유(기본 0)")
    parser.add_argument("--val-margin", type=int, default=DEFAULT_VAL_MARGIN, help="val 경계에 더할 여유(기본 0)")
    parser.add_argument("--output", default=None, help="산출 yaml 조각을 저장할 파일 경로(선택)")
    parser.add_argument(
        "--diagnostic-dir", default=None,
        help="ROI 오버레이/HSV 히스토그램 PNG를 저장할 폴더(선택)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: 입력 경로 없음 — {input_path}", file=sys.stderr)
        return 1

    try:
        frame = load_frame(input_path, args.frame_index)
    except (FileNotFoundError, ValueError, NotADirectoryError) as e:
        print(f"Error: 프레임 로드 실패 — {e}", file=sys.stderr)
        return 1

    try:
        roi_bgr = crop_roi(frame, args.roi)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    result = calibrate_roi(
        roi_bgr,
        roi_box=args.roi,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        wrap_split_hue=args.wrap_split_hue,
        wrap_min_fraction=args.wrap_min_fraction,
        hue_margin=args.hue_margin,
        sat_margin=args.sat_margin,
        val_margin=args.val_margin,
    )

    snippet = format_yaml_snippet(result, input_label=str(input_path), frame_index=args.frame_index)
    print(snippet)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(snippet, encoding="utf-8")
        print(f"저장: {out_path}")

    if args.diagnostic_dir:
        h, s, v = compute_hsv_channels(roi_bgr)
        diag_dir = Path(args.diagnostic_dir)
        overlay_path = save_roi_overlay(frame, args.roi, diag_dir / "roi_overlay.png")
        hist_path = save_hsv_histogram(h, s, v, result, diag_dir / "hsv_histogram.png")
        print(f"진단 산출물: {overlay_path}")
        print(f"진단 산출물: {hist_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
