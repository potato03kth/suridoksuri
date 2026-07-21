import cv2
import numpy as np
from vision.core.state import VisionState, Detection
from vision.modules.vertiport_ring import RedRingDetector

_BBOX = (20, 20, 160, 160)


def _bgr_for_hue(hue: int) -> tuple:
    px = np.uint8([[[hue, 255, 255]]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _state_with_ring(hue: int) -> VisionState:
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    x, y, w, h = _BBOX
    center = (x + w // 2, y + h // 2)
    cv2.circle(img, center, 60, _bgr_for_hue(hue), thickness=8)
    return VisionState(original=img.copy(), current=img, detections=[Detection(bbox=_BBOX)])


def test_detects_ring_low_hue_band():
    state = RedRingDetector()(_state_with_ring(hue=0))
    assert len(state.detections) == 1


def test_detects_ring_high_hue_band():
    state = RedRingDetector()(_state_with_ring(hue=179))
    assert len(state.detections) == 1


def test_ring_meta_has_center_and_radius():
    state = RedRingDetector()(_state_with_ring(hue=0))
    ring_meta = state.detections[0].meta["red_ring"]
    assert "center_px" in ring_meta and "radius_px" in ring_meta
    assert ring_meta["radius_px"] > 0


def test_no_red_no_detection():
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    state = RedRingDetector()(VisionState(original=img, current=img.copy(), detections=[Detection(bbox=_BBOX)]))
    assert len(state.detections) == 0


def test_empty_detections_input_no_crash():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    state = RedRingDetector()(VisionState(original=img, current=img.copy(), detections=[]))
    assert state.detections == []


def test_meta_recorded():
    state = RedRingDetector()(_state_with_ring(hue=0))
    assert state.meta["red_ring"] == {"confirmed": 1}


def test_deterministic():
    d1 = RedRingDetector()(_state_with_ring(hue=0)).detections
    d2 = RedRingDetector()(_state_with_ring(hue=0)).detections
    assert [d.meta["red_ring"] for d in d1] == [d.meta["red_ring"] for d in d2]
