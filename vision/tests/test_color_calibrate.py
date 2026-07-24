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
    (hue~60, val~180) 근방에 그대로 남아야 한다(이상치 쪽으로 안 끌려감)."""
    patch = _patch_with_glare_outliers((60, 60), outlier_fraction=0.02, seed=5)
    h, s, v = compute_hsv_channels(patch)

    # 대조군: 이상치가 실제로 raw min/max를 오염시킨다는 사실 확인(그래서 순수 min/max로는 안 됨).
    assert int(np.min(h)) == 0
    assert int(np.max(v)) == 255

    result = calibrate_roi(patch, roi_box=(0, 0, 60, 60))
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
