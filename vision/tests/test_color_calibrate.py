"""`vision/tools/color_calibrate.py` 테스트 (`docs/vision_plan.md` §5.5, §9 빌드순서 5번).

몽키패치 없음 — 전부 실제 함수 호출 + 실제 `ColorFilter`/`RedRingDetector`를 실제로 돌려서
산출된 임계값이 실제 파이프라인에서 동작하는지까지 확인한다("문자열만 맞는 게 아니라
파이프라인이 실제로 동작하는지"가 세션 지시의 핵심 검증 기준).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.core.state import Detection, VisionState
from vision.modules.color import ColorFilter
from vision.modules.vertiport_ring import RedRingDetector
from vision.tools.color_calibrate import (
    CHANNEL_MAX,
    DEFAULT_HUE_MARGIN,
    DEFAULT_SAT_MARGIN,
    DEFAULT_VAL_MARGIN,
    HUE_MAX,
    _clip_int,
    calibrate_roi,
    compute_hsv_channels,
    crop_roi,
    detect_hue_wraparound,
    format_yaml_snippet,
    load_frame,
    main,
    parse_roi,
    save_hsv_histogram,
    save_roi_overlay,
)

_GOLDEN_10M = (
    Path(__file__).parent / "golden" / "distress" / "10m" / "frame_000.png"
)


# ===========================================================================
# 합성 패치 헬퍼
# ===========================================================================


def _solid_hsv_patch(shape: tuple[int, int], hue: int, sat: int, val: int) -> np.ndarray:
    hsv = np.zeros((*shape, 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = sat
    hsv[..., 2] = val
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _noisy_wraparound_patch(
    shape: tuple[int, int], hue_center: int, hue_jitter: int, sat: int, val: int, seed: int = 0
) -> np.ndarray:
    """`hue_center`(보통 0) 주변으로 ±hue_jitter 지터를 줘 179/0 경계를 실제로 넘나드는 표본을
    만든다 — 실제 빨강 타겟에서 관측되는 Hue 랩어라운드를 흉내낸다."""
    rng = np.random.default_rng(seed)
    n = shape[0] * shape[1]
    hue = (hue_center + rng.integers(-hue_jitter, hue_jitter + 1, size=n)) % 180
    hsv = np.zeros((n, 3), dtype=np.uint8)
    hsv[:, 0] = hue.astype(np.uint8)
    hsv[:, 1] = sat
    hsv[:, 2] = val
    hsv = hsv.reshape(*shape, 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _noisy_green_patch(
    shape: tuple[int, int], hue_center: int, hue_sigma: float, sat: int, val: int, seed: int = 0
) -> np.ndarray:
    """초록 매트를 흉내낸 ROI — Hue만 정규분포로 흩뿌리고 sat/val은 고정한다.

    sat/val을 고정하는 이유: 마진 방향 테스트에서 **hue 마진 효과만 분리**하기 위함
    (S/V가 같이 흔들리면 미검출이 hue 때문인지 sat/val 때문인지 구분이 안 된다).
    """
    rng = np.random.default_rng(seed)
    n = shape[0] * shape[1]
    hue = np.clip(np.round(hue_center + rng.normal(0, hue_sigma, size=n)), 0, 179)
    hsv = np.zeros((n, 3), dtype=np.uint8)
    hsv[:, 0] = hue.astype(np.uint8)
    hsv[:, 1] = sat
    hsv[:, 2] = val
    hsv = hsv.reshape(*shape, 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _majority_minority_patch(
    shape: tuple[int, int], majority_hue: int, minority_hue: int, minority_fraction: float,
    sat: int, val: int, seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = shape[0] * shape[1]
    hue = np.full(n, majority_hue, dtype=np.int32)
    n_minority = int(n * minority_fraction)
    idx = rng.choice(n, size=n_minority, replace=False)
    hue[idx] = minority_hue
    hsv = np.zeros((n, 3), dtype=np.uint8)
    hsv[:, 0] = hue.astype(np.uint8)
    hsv[:, 1] = sat
    hsv[:, 2] = val
    hsv = hsv.reshape(*shape, 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _patch_with_glare_outliers(shape: tuple[int, int], outlier_fraction: float = 0.02, seed: int = 0) -> np.ndarray:
    """주 군집(초록 hue~60) + 소수(outlier_fraction) 과노출 글레어(hue=0,sat=0,val=255) 픽셀 —
    §5.5 "과노출 클리핑 대비"가 말하는 이상치 상황을 합성으로 재현."""
    rng = np.random.default_rng(seed)
    n = shape[0] * shape[1]
    hue = np.clip(60 + rng.normal(0, 2, size=n), 0, 179)
    sat = np.clip(200 + rng.normal(0, 5, size=n), 0, 255)
    val = np.clip(180 + rng.normal(0, 5, size=n), 0, 255)
    n_outliers = max(1, int(n * outlier_fraction))
    idx = rng.choice(n, size=n_outliers, replace=False)
    hue[idx] = 0
    sat[idx] = 0
    val[idx] = 255
    hsv = np.stack([hue, sat, val], axis=-1).reshape(*shape, 3).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ===========================================================================
# 1. ROI 파싱/크롭
# ===========================================================================


def test_parse_roi_valid():
    assert parse_roi("10,20,30,40") == (10, 20, 30, 40)


def test_parse_roi_wrong_field_count_raises():
    with pytest.raises(Exception):
        parse_roi("10,20,30")


def test_parse_roi_non_integer_raises():
    with pytest.raises(Exception):
        parse_roi("10,20,30,abc")


def test_parse_roi_non_positive_size_raises():
    with pytest.raises(Exception):
        parse_roi("10,20,0,40")


def test_crop_roi_within_bounds():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = crop_roi(img, (10, 10, 20, 20))
    assert roi.shape == (20, 20, 3)


def test_crop_roi_out_of_bounds_raises():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop_roi(img, (100, 100, 10, 10))


# ===========================================================================
# 2. HSV 채널 추출 + Hue 랩어라운드 판정
# ===========================================================================


def test_compute_hsv_channels_shapes():
    patch = _solid_hsv_patch((10, 10), hue=60, sat=200, val=180)
    h, s, v = compute_hsv_channels(patch)
    assert h.shape == (100,) and s.shape == (100,) and v.shape == (100,)


def test_compute_hsv_channels_empty_raises():
    with pytest.raises(ValueError):
        compute_hsv_channels(np.zeros((0, 0, 3), dtype=np.uint8))


def test_detect_hue_wraparound_single_cluster_is_false():
    patch = _solid_hsv_patch((20, 20), hue=60, sat=200, val=180)
    h, _, _ = compute_hsv_channels(patch)
    wrap, frac_low, frac_high = detect_hue_wraparound(h)
    assert wrap is False
    assert frac_low == 1.0 and frac_high == 0.0


def test_detect_hue_wraparound_split_cluster_is_true():
    patch = _noisy_wraparound_patch((30, 30), hue_center=0, hue_jitter=10, sat=200, val=200, seed=1)
    h, _, _ = compute_hsv_channels(patch)
    wrap, frac_low, frac_high = detect_hue_wraparound(h, wrap_split_hue=90, wrap_min_fraction=0.05)
    assert wrap is True
    assert frac_low > 0.05 and frac_high > 0.05


def test_detect_hue_wraparound_minority_below_threshold_is_false():
    """소수(3%) 이탈 픽셀은 wrap_min_fraction(5%) 미만이라 랩어라운드로 오판하면 안 된다."""
    patch = _majority_minority_patch(
        (40, 40), majority_hue=170, minority_hue=3, minority_fraction=0.03, sat=200, val=200, seed=2
    )
    h, _, _ = compute_hsv_channels(patch)
    wrap, frac_low, frac_high = detect_hue_wraparound(h, wrap_split_hue=90, wrap_min_fraction=0.05)
    assert wrap is False


# ===========================================================================
# 3. calibrate_roi — 백분위수 산출 + 이상치 강건성(핵심 근거)
# ===========================================================================


def test_calibrate_roi_percentile_range_ignores_minority_outlier_pixels():
    """§5.5 "그림자·글레어·과노출 클리핑 대비" 근거 — 평균±표준편차 대신 백분위수를 쓴
    이유를 직접 증명한다: 2% 글레어 이상치가 섞여도 산출된 hue/val 경계가 주 군집
    (hue~60, val~180) 근방에 그대로 남아야 한다(이상치 쪽으로 안 끌려감).

    **마진을 명시적으로 0으로 준다** — 이 테스트가 재는 것은 "백분위수가 이상치에 강한가"이지
    "마진이 얼마인가"가 아니다. 둘은 서로 다른 것을 덮으므로(도구 docstring "마진 정책" 절)
    여기서는 백분위수 효과만 분리해서 본다. 마진 기본값 자체의 회귀는 아래 10번 절이 맡는다.
    """
    patch = _patch_with_glare_outliers((60, 60), outlier_fraction=0.02, seed=5)
    h, s, v = compute_hsv_channels(patch)

    # 대조군: 이상치가 실제로 raw min/max를 오염시킨다는 사실 확인(그래서 순수 min/max로는 안 됨).
    assert int(np.min(h)) == 0
    assert int(np.max(v)) == 255

    result = calibrate_roi(patch, roi_box=(0, 0, 60, 60), hue_margin=0, sat_margin=0, val_margin=0)
    assert not result.hue_wraparound
    lo, hi = result.params["hue_range"]
    assert 50 <= lo <= 65 and 50 <= hi <= 70  # 주 군집(hue~60) 근방에 머무름 — 이상치(0)로 안 끌려감
    assert result.params["val_max"] < 230  # 이상치(255)로 안 끌려감(주 군집 val~180 근방 유지)


def test_calibrate_roi_non_wraparound_targets_colorfilter():
    patch = _solid_hsv_patch((40, 40), hue=60, sat=200, val=180)
    result = calibrate_roi(patch, roi_box=(0, 0, 40, 40))
    assert result.hue_wraparound is False
    assert "ColorFilter" in result.consumer
    assert set(result.params.keys()) == {"hue_range", "sat_min", "val_min", "val_max"}


def test_calibrate_roi_wraparound_targets_redringdetector():
    patch = _noisy_wraparound_patch((40, 40), hue_center=0, hue_jitter=10, sat=220, val=200, seed=3)
    result = calibrate_roi(patch, roi_box=(0, 0, 40, 40))
    assert result.hue_wraparound is True
    assert "RedRingDetector" in result.consumer
    assert set(result.params.keys()) == {"low_hue_max", "high_hue_min", "sat_min", "val_min"}
    # 두 구간 다 원점(0/179) 근방이어야 한다 — 전체 범위를 뭉뚱그린 "0~179 쓸모없는 범위" 아님.
    assert result.params["low_hue_max"] < 30
    assert result.params["high_hue_min"] > 150


def test_calibrate_roi_margins_and_percentiles_are_parameters_not_hardcoded():
    patch = _solid_hsv_patch((40, 40), hue=60, sat=200, val=180)
    tight = calibrate_roi(patch, roi_box=(0, 0, 40, 40), hue_margin=0, sat_margin=0, val_margin=0)
    loose = calibrate_roi(patch, roi_box=(0, 0, 40, 40), hue_margin=5, sat_margin=10, val_margin=10)
    assert loose.params["hue_range"][0] <= tight.params["hue_range"][0]
    assert loose.params["hue_range"][1] >= tight.params["hue_range"][1]
    assert loose.params["sat_min"] <= tight.params["sat_min"]


# ===========================================================================
# 4. 왕복 검증(핵심) — 산출된 임계값을 실제 ColorFilter/RedRingDetector에 먹인다
# ===========================================================================


def test_calibrate_roi_output_detected_by_real_colorfilter_and_rejects_background():
    canvas_size = 200
    bg_hsv = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    bg_hsv[..., 0] = 110  # 파랑 계열 — 초록과 뚜렷이 다른 배경
    bg_hsv[..., 1] = 200
    bg_hsv[..., 2] = 180
    canvas = cv2.cvtColor(bg_hsv, cv2.COLOR_HSV2BGR)

    x, y, w, h = 60, 60, 60, 60
    patch_bgr = _solid_hsv_patch((h, w), hue=60, sat=200, val=180)
    canvas[y:y + h, x:x + w] = patch_bgr

    roi = (x + 10, y + 10, w - 20, h - 20)  # 패치 내부(경계 여유)
    roi_bgr = crop_roi(canvas, roi)
    result = calibrate_roi(roi_bgr, roi_box=roi)
    assert not result.hue_wraparound

    color_filter = ColorFilter(mode="color", **result.params)
    state = VisionState(original=canvas.copy(), current=canvas.copy())
    state = color_filter(state)

    patch_mask = state.mask[y:y + h, x:x + w]
    bg_mask = state.mask.copy()
    bg_mask[y:y + h, x:x + w] = 0

    assert np.count_nonzero(patch_mask) / patch_mask.size > 0.95
    assert np.count_nonzero(bg_mask) == 0


def test_calibrate_roi_wraparound_output_detected_by_real_redringdetector_and_rejects_background():
    canvas_size = 200
    bg_hsv = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    bg_hsv[..., 0] = 100  # 시안 계열 — 빨강과 뚜렷이 다르고 랩어라운드 걱정도 없는 배경
    bg_hsv[..., 1] = 200
    bg_hsv[..., 2] = 180
    canvas = cv2.cvtColor(bg_hsv, cv2.COLOR_HSV2BGR)

    x, y, w, h = 60, 60, 60, 60
    patch_bgr = _noisy_wraparound_patch((h, w), hue_center=0, hue_jitter=10, sat=220, val=200, seed=4)
    canvas[y:y + h, x:x + w] = patch_bgr

    roi = (x + 10, y + 10, w - 20, h - 20)
    roi_bgr = crop_roi(canvas, roi)
    result = calibrate_roi(roi_bgr, roi_box=roi)
    assert result.hue_wraparound

    detector = RedRingDetector(**result.params)
    state = VisionState(
        original=canvas.copy(), current=canvas.copy(),
        detections=[Detection(bbox=(0, 0, canvas_size, canvas_size))],
    )
    state = detector(state)
    assert len(state.detections) == 1
    assert state.detections[0].meta["red_ring"]["gated_points"] > 0

    bg_only = cv2.cvtColor(bg_hsv, cv2.COLOR_HSV2BGR)
    bg_state = VisionState(
        original=bg_only, current=bg_only.copy(),
        detections=[Detection(bbox=(0, 0, canvas_size, canvas_size))],
    )
    bg_state = detector(bg_state)
    assert len(bg_state.detections) == 0


# ===========================================================================
# 5. yaml 조각 출력
# ===========================================================================


def test_format_yaml_snippet_non_wraparound_has_colorfilter_keys():
    patch = _solid_hsv_patch((30, 30), hue=60, sat=200, val=180)
    result = calibrate_roi(patch, roi_box=(0, 0, 30, 30))
    text = format_yaml_snippet(result, input_label="fake.png", frame_index=0)
    assert "color_filter:" in text
    assert "hue_range:" in text
    assert "red_ring:" not in text
    assert "사람이 검토" in text
    assert "fake.png" in text


def test_format_yaml_snippet_wraparound_has_redring_keys():
    patch = _noisy_wraparound_patch((30, 30), hue_center=0, hue_jitter=10, sat=220, val=200, seed=6)
    result = calibrate_roi(patch, roi_box=(0, 0, 30, 30))
    text = format_yaml_snippet(result, input_label="fake.png", frame_index=0)
    assert "red_ring:" in text
    assert "low_hue_max:" in text
    assert "color_filter:" not in text


# ===========================================================================
# 6. 진단 산출물(Agg, headless-safe)
# ===========================================================================


def test_save_roi_overlay_writes_real_png(tmp_path):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out_path = save_roi_overlay(frame, (10, 10, 20, 20), tmp_path / "overlay.png")
    assert out_path.exists()
    assert cv2.imread(str(out_path)) is not None


def test_save_hsv_histogram_writes_real_png(tmp_path):
    patch = _solid_hsv_patch((30, 30), hue=60, sat=200, val=180)
    h, s, v = compute_hsv_channels(patch)
    result = calibrate_roi(patch, roi_box=(0, 0, 30, 30))
    out_path = save_hsv_histogram(h, s, v, result, tmp_path / "hist.png")
    assert out_path.exists() and out_path.stat().st_size > 0


def test_save_hsv_histogram_wraparound_case_writes_real_png(tmp_path):
    patch = _noisy_wraparound_patch((30, 30), hue_center=0, hue_jitter=10, sat=220, val=200, seed=7)
    h, s, v = compute_hsv_channels(patch)
    result = calibrate_roi(patch, roi_box=(0, 0, 30, 30))
    out_path = save_hsv_histogram(h, s, v, result, tmp_path / "hist_wrap.png")
    assert out_path.exists() and out_path.stat().st_size > 0


# ===========================================================================
# 7. 프레임 로드(frame_source 재사용)
# ===========================================================================


def test_load_frame_single_image_file(tmp_path):
    img = np.full((20, 20, 3), 77, dtype=np.uint8)
    p = tmp_path / "frame.png"
    cv2.imwrite(str(p), img)
    loaded = load_frame(p)
    assert loaded.shape == (20, 20, 3)
    assert int(loaded[0, 0, 0]) == 77


def test_load_frame_directory_uses_dirframesource_and_frame_index(tmp_path):
    for i in range(3):
        img = np.full((10, 10, 3), i * 50, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"frame_{i:03d}.png"), img)
    loaded = load_frame(tmp_path, frame_index=1)
    assert int(loaded[0, 0, 0]) == 50  # frame_001.png


def test_load_frame_index_out_of_range_raises(tmp_path):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "frame_000.png"), img)
    with pytest.raises(ValueError):
        load_frame(tmp_path, frame_index=5)


# ===========================================================================
# 8. CLI end-to-end
# ===========================================================================


def test_cli_main_writes_output_and_diagnostics(tmp_path):
    canvas = _solid_hsv_patch((80, 80), hue=60, sat=200, val=180)
    input_path = tmp_path / "frame.png"
    cv2.imwrite(str(input_path), canvas)

    output_path = tmp_path / "proposed.yaml"
    diag_dir = tmp_path / "diag"

    rc = main([
        str(input_path), "--roi", "10,10,40,40",
        "--output", str(output_path), "--diagnostic-dir", str(diag_dir),
    ])
    assert rc == 0
    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "color_filter:" in text
    assert (diag_dir / "roi_overlay.png").exists()
    assert (diag_dir / "hsv_histogram.png").exists()


def test_cli_main_missing_input_returns_error(tmp_path):
    rc = main([str(tmp_path / "nope.png"), "--roi", "0,0,10,10"])
    assert rc == 1


def test_cli_main_roi_out_of_bounds_returns_error(tmp_path):
    canvas = np.zeros((50, 50, 3), dtype=np.uint8)
    input_path = tmp_path / "frame.png"
    cv2.imwrite(str(input_path), canvas)
    rc = main([str(input_path), "--roi", "100,100,10,10"])
    assert rc == 1


# ===========================================================================
# 9. 골든셋 교차확인(sanity) — ② 조난자 초록 매트 실제 프레임
# ===========================================================================


@pytest.mark.skipif(not _GOLDEN_10M.exists(), reason="골든셋 프레임 없음")
def test_golden_distress_mat_cross_check_is_in_reasonable_range():
    """`distress_coarse.yaml`의 손튜닝 값(hue_range=[35,85], sat_min=80, val_min=60)과 터무니없이
    다르지 않은지 sanity check — 완전 일치를 요구하지 않는다(합성 골든 프레임은 노이즈 없는
    단색이라 도구가 내는 범위가 손튜닝값보다 훨씬 좁은 게 정상)."""
    frame = cv2.imread(str(_GOLDEN_10M))
    assert frame is not None
    # 매트(초록, 좌상단 80~380 부근) 내부, 중앙 흰 박스(197~263 부근)를 피한 안전한 ROI.
    roi = (100, 100, 60, 60)
    roi_bgr = crop_roi(frame, roi)
    result = calibrate_roi(roi_bgr, roi_box=roi)

    assert not result.hue_wraparound
    lo, hi = result.params["hue_range"]
    # distress_coarse.yaml의 손튜닝 hue_range=[35,85] 안에 완전히 들어와야 한다(같은 초록을 보므로).
    assert 35 <= lo <= 85
    assert 35 <= hi <= 85
    assert result.params["sat_min"] > 0
    assert result.params["val_min"] > 0
    print(
        "\n[골든셋 교차확인] distress_coarse.yaml 손튜닝: hue_range=[35,85] sat_min=80 val_min=60\n"
        f"[골든셋 교차확인] color_calibrate 산출: hue_range={[lo, hi]} "
        f"sat_min={result.params['sat_min']} val_min={result.params['val_min']} "
        f"val_max={result.params['val_max']} (n_pixels={result.n_pixels})"
    )


# ===========================================================================
# 10. 마진 기본값 정책 회귀 (2026-07-28 확정 — 도구 docstring "마진 정책" 절)
#
#   백분위수 밴드와 마진은 서로 다른 것을 덮는다:
#     * 백분위수(p5~p95) = 캘리브레이션한 그 프레임 ROI 안의 공간 변동(그림자/글레어/얼룩)
#     * 마진               = 캘리브 시점 ↔ 비행 시점의 조건 변화(태양각/구름/WB 재수렴)
#   아래 테스트는 형상이 아니라 **실제 동작**을 본다 — 기본 마진이 없으면 놓치는 픽셀을
#   기본 마진이 실제로 잡아내는지를 진짜 `ColorFilter`/`RedRingDetector`로 확인한다.
# ===========================================================================


# 사용자 제공 실측 표본(2026-07-28): 초록구역 휴대폰 사진 랜덤 10점 중 이상치 2점 제외한 8점.
# 0~360° 스케일 — ÷2 하면 OpenCV Hue 68~84.5(청록 띤 초록)로 초록 매트에 물리적으로 타당하고
# distress_coarse.yaml 손튜닝값 hue_range=[35,85] 안에 들어온다. 0~255 가정(→192~238°, 파랑)과
# 이미-OpenCV 가정(→272~338°, 마젠타)은 초록과 모순이라 기각.
_FIELD_HUE_SAMPLES_DEG360 = [136, 162, 169, 146, 169, 158, 146, 163]


def test_default_hue_margin_matches_measured_sample_dispersion():
    """`DEFAULT_HUE_MARGIN`이 실측 표본 산포(1σ)에서 실제로 도출된 값인지 재계산으로 확인한다.

    상수를 그냥 읽어 비교하는 게 아니라 **원본 8점에서 σ를 다시 구해** 맞춰 본다 — 누가 이
    기본값을 근거 없이 바꾸면 여기서 걸린다. 절대 Hue 위치(mean)는 카메라/화이트밸런스
    미상이라 신뢰하지 않으므로 검증하지 않는다(산포만 근거로 쓴다는 결정 자체의 회귀).
    """
    samples_cv = np.array(_FIELD_HUE_SAMPLES_DEG360, dtype=np.float64) / 2.0  # 0~360° → OpenCV 0~179

    sample_sigma = float(np.std(samples_cv, ddof=1))  # 표본 표준편차
    population_sigma = float(np.std(samples_cv, ddof=0))  # 모집단 표준편차

    assert round(sample_sigma, 3) == 6.056
    assert round(population_sigma, 3) == 5.665
    # 두 추정량이 **둘 다** 6으로 반올림된다 → 어느 쪽을 쓰든 결론이 같다.
    assert round(sample_sigma) == DEFAULT_HUE_MARGIN
    assert round(population_sigma) == DEFAULT_HUE_MARGIN

    # 2σ 이상이 아닌 이유(이중계상 회피)의 회귀 — 기본값이 2σ 쪽으로 슬쩍 커지면 걸린다.
    assert DEFAULT_HUE_MARGIN < round(2 * sample_sigma)


def test_default_hue_margin_widens_band_relative_to_zero_margin():
    """기본 마진이 실제로 임계 밴드를 **넓히는지** — 마진 0과 직접 대조."""
    patch = _noisy_green_patch((60, 60), hue_center=78, hue_sigma=2.0, sat=200, val=180, seed=11)

    zero = calibrate_roi(patch, roi_box=(0, 0, 60, 60), hue_margin=0)
    default = calibrate_roi(patch, roi_box=(0, 0, 60, 60))  # DEFAULT_HUE_MARGIN 사용

    assert default.hue_margin == DEFAULT_HUE_MARGIN
    zero_lo, zero_hi = zero.params["hue_range"]
    def_lo, def_hi = default.params["hue_range"]

    # 양쪽 각각 정확히 DEFAULT_HUE_MARGIN만큼 벌어져야 한다(클리핑 영역에서 멀리 떨어진 표본).
    assert def_lo == zero_lo - DEFAULT_HUE_MARGIN
    assert def_hi == zero_hi + DEFAULT_HUE_MARGIN
    assert (def_hi - def_lo) == (zero_hi - zero_lo) + 2 * DEFAULT_HUE_MARGIN


def test_default_hue_margin_catches_lighting_shifted_target_that_zero_margin_misses():
    """**핵심(형상 아님, 실제 동작).** 캘리브레이션 뒤 조명이 바뀌어 매트 Hue가 통째로 이동한
    프레임을, 마진 0 임계값은 놓치고 기본 마진 임계값은 잡아내는지 **진짜 `ColorFilter`**로
    확인한다 — 마진이 존재하는 이유(캘리브 시점 ↔ 비행 시점 조건 변화) 그 자체의 검증.

    sat/val 마진은 양쪽 다 같은 값으로 고정해 **hue 마진 효과만 분리**한다(S/V 기본값이 0인
    것은 별도 테스트가 담당).
    """
    calib_frame = _noisy_green_patch((80, 80), hue_center=78, hue_sigma=2.0, sat=200, val=180, seed=12)
    roi = (10, 10, 60, 60)
    roi_bgr = crop_roi(calib_frame, roi)

    zero = calibrate_roi(roi_bgr, roi_box=roi, hue_margin=0, sat_margin=5, val_margin=5)
    default = calibrate_roi(roi_bgr, roi_box=roi, sat_margin=5, val_margin=5)

    # 비행 시점 프레임: 태양각/구름/WB 재수렴으로 매트 Hue 중심이 +6(=1σ)만큼 이동했다고 가정.
    shifted = _noisy_green_patch((80, 80), hue_center=84, hue_sigma=2.0, sat=200, val=180, seed=13)

    def _coverage(params: dict) -> float:
        state = VisionState(original=shifted.copy(), current=shifted.copy())
        state = ColorFilter(mode="color", **params)(state)
        return float(np.count_nonzero(state.mask)) / state.mask.size

    zero_cov = _coverage(zero.params)
    default_cov = _coverage(default.params)

    # 실측(cv2 5.0.0): zero_cov≈0.103, default_cov≈0.960 — 약 9배 차이. 임계는 cv2 버전별
    # HSV 왕복 오차를 감안해 넉넉히 잡되, 두 값이 뒤섞이지 않을 만큼은 떨어뜨려 둔다.
    assert default_cov > 0.90, f"기본 마진이 이동한 타겟을 잡지 못함 (coverage={default_cov:.3f})"
    assert zero_cov < 0.25, f"마진 0이 이동한 타겟을 놓치지 않음 — 대조군이 무의미 (coverage={zero_cov:.3f})"
    assert default_cov > zero_cov


def test_default_sat_val_margins_are_zero_and_leave_thresholds_at_raw_percentiles():
    """S/V 마진 기본값 0은 **의도된 결정**이다(표본이 Hue뿐이라 숫자를 지어내지 않는다 +
    내리면 저채도 자갈/저휘도 그림자가 그대로 들어온다). 누가 근거 없이 S/V 기본 쿠션을
    넣으면 여기서 걸린다 — 산출 임계값이 raw 백분위수와 **정확히** 같아야 한다."""
    assert DEFAULT_SAT_MARGIN == 0
    assert DEFAULT_VAL_MARGIN == 0

    patch = _noisy_green_patch((60, 60), hue_center=78, hue_sigma=2.0, sat=200, val=180, seed=14)
    _, s, v = compute_hsv_channels(patch)
    result = calibrate_roi(patch, roi_box=(0, 0, 60, 60))

    assert result.params["sat_min"] == _clip_int(np.percentile(s, 5.0), 0, CHANNEL_MAX)
    assert result.params["val_min"] == _clip_int(np.percentile(v, 5.0), 0, CHANNEL_MAX)
    assert result.params["val_max"] == _clip_int(np.percentile(v, 95.0), 0, CHANNEL_MAX)


# --- 경계 클리핑 ---


def test_clip_int_never_escapes_bounds():
    assert _clip_int(-100.0, 0, HUE_MAX) == 0
    assert _clip_int(1e6, 0, HUE_MAX) == HUE_MAX
    assert _clip_int(-0.4, 0, CHANNEL_MAX) == 0
    assert _clip_int(CHANNEL_MAX + 0.6, 0, CHANNEL_MAX) == CHANNEL_MAX
    assert _clip_int(179.4, 0, HUE_MAX) == HUE_MAX  # 반올림 후에도 상한 초과 안 함
    assert _clip_int(0.5, 0, HUE_MAX) in (0, 1)  # 반올림 규칙과 무관하게 범위 안


def test_hue_margin_clipped_at_hue_bounds_near_zero_and_179():
    """Hue 0/179 바로 옆 표본에 큰 마진을 줘도 산출값이 0~179를 벗어나면 안 된다.
    (랩어라운드로 오판되지 않도록 한쪽 군집만 있는 표본을 쓴다.)"""
    near_zero = _solid_hsv_patch((30, 30), hue=1, sat=200, val=180)
    result_lo = calibrate_roi(near_zero, roi_box=(0, 0, 30, 30), hue_margin=50)
    assert not result_lo.hue_wraparound
    lo, hi = result_lo.params["hue_range"]
    assert 0 <= lo <= HUE_MAX and 0 <= hi <= HUE_MAX
    assert lo == 0  # 음수로 새지 않고 정확히 하한에 걸린다

    near_max = _solid_hsv_patch((30, 30), hue=HUE_MAX, sat=200, val=180)
    result_hi = calibrate_roi(near_max, roi_box=(0, 0, 30, 30), hue_margin=50)
    assert not result_hi.hue_wraparound
    lo, hi = result_hi.params["hue_range"]
    assert 0 <= lo <= HUE_MAX and 0 <= hi <= HUE_MAX
    assert hi == HUE_MAX  # 180 이상으로 새지 않는다


def test_sat_val_margins_clipped_at_channel_bounds():
    """sat/val이 0/255 근처인 표본에 큰 마진을 줘도 0~255를 벗어나면 안 된다."""
    dark_dull = _solid_hsv_patch((30, 30), hue=78, sat=3, val=4)
    r_low = calibrate_roi(dark_dull, roi_box=(0, 0, 30, 30), sat_margin=200, val_margin=200)
    assert r_low.params["sat_min"] == 0
    assert r_low.params["val_min"] == 0
    assert 0 <= r_low.params["val_max"] <= CHANNEL_MAX

    bright = _solid_hsv_patch((30, 30), hue=78, sat=CHANNEL_MAX, val=CHANNEL_MAX)
    r_high = calibrate_roi(bright, roi_box=(0, 0, 30, 30), sat_margin=0, val_margin=200)
    assert r_high.params["val_max"] == CHANNEL_MAX  # 255를 넘지 않는다
    assert 0 <= r_high.params["val_min"] <= CHANNEL_MAX


# --- 랩어라운드 분기의 마진 **방향** (부호가 반대면 통과대역이 좁아진다 — 실제 위험한 버그) ---


def test_wraparound_margin_widens_both_passbands_not_narrows():
    """랩어라운드(빨강)에서 마진은 `low_hue_max`를 **키우고** `high_hue_min`을 **줄여야** 한다.
    즉 `[0, low_hue_max]` ∪ `[high_hue_min, 179]` 두 통과대역이 각각 **넓어지는** 방향.
    부호가 반대면 대역이 좁아져 빨강을 놓친다."""
    patch = _noisy_wraparound_patch((60, 60), hue_center=0, hue_jitter=5, sat=220, val=200, seed=21)

    zero = calibrate_roi(patch, roi_box=(0, 0, 60, 60), hue_margin=0)
    default = calibrate_roi(patch, roi_box=(0, 0, 60, 60))
    assert zero.hue_wraparound and default.hue_wraparound

    assert default.params["low_hue_max"] == zero.params["low_hue_max"] + DEFAULT_HUE_MARGIN
    assert default.params["high_hue_min"] == zero.params["high_hue_min"] - DEFAULT_HUE_MARGIN
    # 방향을 부등호로도 못박는다(위 등식이 값 변경으로 느슨해져도 방향은 남는다).
    assert default.params["low_hue_max"] > zero.params["low_hue_max"]
    assert default.params["high_hue_min"] < zero.params["high_hue_min"]


def test_wraparound_default_margin_gates_shifted_red_that_zero_margin_rejects():
    """**핵심(형상 아님, 실제 동작).** 위 방향 규칙을 진짜 `RedRingDetector`로 확인한다 —
    조명 변화로 이동한 빨강(양끝 각각)을 마진 0 임계값은 게이팅하지 못하고, 기본 마진
    임계값은 게이팅해야 한다. 부호가 뒤집히면 두 케이스 모두 실패한다."""
    patch = _noisy_wraparound_patch((60, 60), hue_center=0, hue_jitter=5, sat=220, val=200, seed=22)
    zero = calibrate_roi(patch, roi_box=(0, 0, 60, 60), hue_margin=0)
    default = calibrate_roi(patch, roi_box=(0, 0, 60, 60))

    # 마진 0 대역 바로 바깥, 기본 마진 대역 안쪽에 놓이는 이동된 빨강 Hue 두 개(저/고 양끝).
    shifted_low_hue = zero.params["low_hue_max"] + DEFAULT_HUE_MARGIN // 2
    shifted_high_hue = zero.params["high_hue_min"] - DEFAULT_HUE_MARGIN // 2
    assert shifted_low_hue < default.params["low_hue_max"]
    assert shifted_high_hue > default.params["high_hue_min"]

    def _gated(params: dict, hue: int) -> int:
        canvas = _solid_hsv_patch((100, 100), hue=hue, sat=240, val=230)
        state = VisionState(
            original=canvas, current=canvas.copy(),
            detections=[Detection(bbox=(0, 0, 100, 100))],
        )
        state = RedRingDetector(**params)(state)
        if not state.detections:
            return 0
        return int(state.detections[0].meta["red_ring"]["gated_points"])

    for hue in (shifted_low_hue, shifted_high_hue):
        assert _gated(zero.params, hue) == 0, f"대조군 무의미 — 마진 0이 hue={hue}를 이미 잡음"
        assert _gated(default.params, hue) > 0, f"기본 마진이 이동한 빨강 hue={hue}를 놓침"
