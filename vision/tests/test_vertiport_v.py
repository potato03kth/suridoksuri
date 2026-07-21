import cv2
import numpy as np
from vision.core.state import VisionState, Detection
from vision.modules.vertiport_v import BlackVMatcher

_BBOX = (50, 50, 100, 100)


def _state_with_shape(draw_fn) -> VisionState:
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    x, y, w, h = _BBOX
    roi = img[y:y + h, x:x + w]
    draw_fn(roi)
    return VisionState(original=img.copy(), current=img, detections=[Detection(bbox=_BBOX)])


def _draw_v(roi):
    h, w = roi.shape[:2]
    apex = (w // 2, h - 5)
    left = (5, 5)
    right = (w - 5, 5)
    cv2.line(roi, left, apex, (0, 0, 0), 15)
    cv2.line(roi, apex, right, (0, 0, 0), 15)


def _draw_solid_square(roi):
    h, w = roi.shape[:2]
    cv2.rectangle(roi, (10, 10), (w - 10, h - 10), (0, 0, 0), -1)


def test_confirms_v_shape():
    state = BlackVMatcher()(_state_with_shape(_draw_v))
    assert len(state.detections) == 1


def test_rejects_solid_blob_without_v_shape():
    state = BlackVMatcher()(_state_with_shape(_draw_solid_square))
    assert len(state.detections) == 0


def test_confirmed_detection_has_match_score_meta():
    state = BlackVMatcher()(_state_with_shape(_draw_v))
    assert "match_score" in state.detections[0].meta["black_v"]


def test_empty_detections_input_no_crash():
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    state = BlackVMatcher()(VisionState(original=img, current=img.copy(), detections=[]))
    assert state.detections == []


def test_meta_recorded():
    state = BlackVMatcher()(_state_with_shape(_draw_v))
    assert state.meta["black_v"] == {"confirmed": 1, "rejected": 0}


def test_deterministic():
    d1 = BlackVMatcher()(_state_with_shape(_draw_v)).detections
    d2 = BlackVMatcher()(_state_with_shape(_draw_v)).detections
    assert [d.meta["black_v"] for d in d1] == [d.meta["black_v"] for d in d2]
