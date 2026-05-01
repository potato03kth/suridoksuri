import numpy as np
import pytest
from vision.core.state import VisionState
from vision.modules.color import ColorFilter


def _gray_state(value: int = 200) -> VisionState:
    img = np.full((100, 100, 3), value, dtype=np.uint8)
    return VisionState(original=img.copy(), current=img.copy())


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
