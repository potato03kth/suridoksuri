import cv2
import numpy as np
from vision.core.state import VisionState
from vision.modules.vertiport_field import WhiteFieldDetector


def _mask_with_circle(size: int = 300, radius: int = 100) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), radius, 255, -1)
    return mask


def _state(mask) -> VisionState:
    img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    return VisionState(original=img.copy(), current=img.copy(), mask=mask)


def test_detects_circular_field():
    state = WhiteFieldDetector(min_area=100)(_state(_mask_with_circle()))
    assert len(state.detections) == 1


def test_high_confidence_for_near_perfect_circle():
    state = WhiteFieldDetector(min_area=100)(_state(_mask_with_circle()))
    # 픽셀 격자에서 그린 원은 컨투어 둘레가 계단식으로 과대추정되어 완전한 1.0은 안 나온다.
    assert state.detections[0].confidence > 0.85


def test_detection_meta_has_center_and_radius():
    state = WhiteFieldDetector(min_area=100)(_state(_mask_with_circle()))
    det_meta = state.detections[0].meta
    assert "center" in det_meta and "radius" in det_meta


def test_rejects_non_circular_blob():
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(mask, (50, 20), (70, 280), 255, -1)  # 가늘고 긴 막대 - 저원형도
    state = WhiteFieldDetector(min_area=1, min_circularity=0.6)(_state(mask))
    assert len(state.detections) == 0


def test_empty_mask_no_detections():
    state = WhiteFieldDetector()(_state(np.zeros((200, 200), dtype=np.uint8)))
    assert len(state.detections) == 0


def test_no_mask_no_crash():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    state = WhiteFieldDetector()(VisionState(original=img, current=img.copy()))
    assert state.detections == []


def test_meta_recorded():
    state = WhiteFieldDetector(min_area=100)(_state(_mask_with_circle()))
    assert "white_field" in state.meta
    assert state.meta["white_field"]["candidates"] == 1


def test_deterministic():
    mask = _mask_with_circle()
    d1 = WhiteFieldDetector(min_area=100)(_state(mask)).detections
    d2 = WhiteFieldDetector(min_area=100)(_state(mask)).detections
    assert [d.bbox for d in d1] == [d.bbox for d in d2]
