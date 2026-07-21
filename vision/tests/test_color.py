import cv2
import numpy as np
import pytest
from vision.core.state import VisionState
from vision.modules.color import ColorFilter


def _gray_state(value: int = 200) -> VisionState:
    img = np.full((100, 100, 3), value, dtype=np.uint8)
    return VisionState(original=img.copy(), current=img.copy())


def _hsv_state(hue: int, sat: int = 200, val: int = 200) -> VisionState:
    hsv = np.zeros((100, 100, 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = sat
    hsv[..., 2] = val
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return VisionState(original=bgr.copy(), current=bgr.copy())


def test_gray_mode_produces_mask():
    state = ColorFilter(mode="gray", sat_max=50, val_min=100)(_gray_state(200))
    assert state.mask is not None
    assert state.mask.shape == (100, 100)


def test_gray_mode_captures_light_gray():
    state = ColorFilter(mode="gray", sat_max=50, val_min=100)(_gray_state(180))
    assert np.count_nonzero(state.mask) > 0


def test_gray_mode_rejects_dark():
    state = ColorFilter(mode="gray", sat_max=50, val_min=150)(_gray_state(50))
    assert np.count_nonzero(state.mask) == 0


def test_meta_recorded():
    state = ColorFilter()(_gray_state())
    assert "color_filter" in state.meta


def test_color_mode_green_captures_green_hue():
    state = ColorFilter(mode="color", hue_range=(35, 85), sat_min=100, val_min=100)(_hsv_state(hue=60))
    assert np.count_nonzero(state.mask) > 0


def test_color_mode_green_rejects_red_hue():
    state = ColorFilter(mode="color", hue_range=(35, 85), sat_min=100, val_min=100)(_hsv_state(hue=0))
    assert np.count_nonzero(state.mask) == 0


def test_color_mode_green_rejects_low_saturation():
    state = ColorFilter(mode="color", hue_range=(35, 85), sat_min=100, val_min=100)(_hsv_state(hue=60, sat=20))
    assert np.count_nonzero(state.mask) == 0


def test_color_mode_red_captures_low_hue_band():
    state = ColorFilter(mode="color", hue_range=(0, 10), sat_min=100, val_min=100)(_hsv_state(hue=0))
    assert np.count_nonzero(state.mask) > 0


def test_color_mode_red_captures_high_hue_band():
    state = ColorFilter(mode="color", hue_range=(170, 179), sat_min=100, val_min=100)(_hsv_state(hue=179))
    assert np.count_nonzero(state.mask) > 0


def test_color_mode_red_single_range_misses_hue_wraparound():
    """빨강은 Hue=0 부근과 179 부근 둘 다에 걸쳐있으나 ColorFilter는 단일 (lower,upper) 구간만 지원한다.
    양끝을 한 range로 걸면(lower>upper) cv2.inRange가 항상 빈 마스크를 낸다 — vision_plan.md §5.4 blind spot."""
    state = ColorFilter(mode="color", hue_range=(170, 10), sat_min=100, val_min=100)(_hsv_state(hue=0))
    assert np.count_nonzero(state.mask) == 0


def test_color_mode_meta_recorded():
    state = ColorFilter(mode="color", hue_range=(35, 85), sat_min=100, val_min=100)(_hsv_state(hue=60))
    assert state.meta["color_filter"]["mode"] == "color"
