import numpy as np
import cv2
import pytest
from vision.core.state import VisionState
from vision.modules.detector import RectDetector


def _state_with_rect_mask(size: int = 200, rect_size: int = 80) -> VisionState:
    mask = np.zeros((size, size), dtype=np.uint8)
    p = (size - rect_size) // 2
    cv2.rectangle(mask, (p, p), (p + rect_size, p + rect_size), 255, -1)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    return VisionState(original=img.copy(), current=img.copy(), mask=mask)


def test_detects_rectangle():
    state = RectDetector(min_area=100)(_state_with_rect_mask())
    assert len(state.detections) >= 1


def test_detection_has_bbox():
    state = RectDetector(min_area=100)(_state_with_rect_mask())
    det = state.detections[0]
    assert len(det.bbox) == 4
    x, y, w, h = det.bbox
    assert w > 0 and h > 0


def test_empty_mask_no_detections():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    state = VisionState(original=img, current=img.copy(), mask=mask)
    state = RectDetector()(state)
    assert len(state.detections) == 0


def test_meta_recorded():
    state = RectDetector(min_area=100)(_state_with_rect_mask())
    assert "rect_detector" in state.meta
