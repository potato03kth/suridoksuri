import cv2
import numpy as np
from vision.core.state import VisionState
from vision.modules.aruco import ArucoDetector

_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _canvas_with_marker(marker_id: int, x: int = 50, y: int = 50, marker_size: int = 150,
                         canvas_size: int = 300) -> np.ndarray:
    """단일 ArUco 마커를 흰 배경 위에 합성한 BGR 이미지(quiet zone 확보용 여백 포함)."""
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    marker_gray = cv2.aruco.generateImageMarker(_DICTIONARY, marker_id, marker_size)
    marker_bgr = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
    canvas[y:y + marker_size, x:x + marker_size] = marker_bgr
    return canvas


def _state_for(img: np.ndarray) -> VisionState:
    return VisionState(original=img, current=img.copy())


def test_decodes_valid_id_23():
    state = ArucoDetector()(_state_for(_canvas_with_marker(23)))
    assert len(state.detections) == 1
    assert state.detections[0].meta["aruco_id"] == 23


def test_confirmed_detection_has_corners():
    state = ArucoDetector()(_state_for(_canvas_with_marker(23)))
    corners = state.detections[0].corners
    assert corners is not None
    assert corners.shape == (4, 2)


def test_bbox_derived_from_corners():
    state = ArucoDetector()(_state_for(_canvas_with_marker(23, x=50, y=50, marker_size=150)))
    det = state.detections[0]
    corners = det.corners
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    expected_bbox = (
        int(round(x0)), int(round(y0)),
        int(round(x1 - x0)), int(round(y1 - y0)),
    )
    assert det.bbox == expected_bbox


def test_rejects_other_id():
    state = ArucoDetector()(_state_for(_canvas_with_marker(5)))
    assert state.detections == []
    assert state.meta["aruco"] == {"confirmed": 0, "rejected": 1}


def test_meta_recorded_on_success():
    state = ArucoDetector()(_state_for(_canvas_with_marker(23)))
    assert state.meta["aruco"] == {"confirmed": 1, "rejected": 0}


def test_mixed_valid_and_invalid_ids():
    canvas = np.full((300, 500, 3), 255, dtype=np.uint8)
    marker23 = cv2.cvtColor(cv2.aruco.generateImageMarker(_DICTIONARY, 23, 150), cv2.COLOR_GRAY2BGR)
    marker5 = cv2.cvtColor(cv2.aruco.generateImageMarker(_DICTIONARY, 5, 150), cv2.COLOR_GRAY2BGR)
    canvas[50:200, 20:170] = marker23
    canvas[50:200, 300:450] = marker5

    state = ArucoDetector()(_state_for(canvas))
    assert len(state.detections) == 1
    assert state.detections[0].meta["aruco_id"] == 23
    assert state.meta["aruco"] == {"confirmed": 1, "rejected": 1}


def test_empty_white_image_no_crash():
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    state = ArucoDetector()(_state_for(img))
    assert state.detections == []
    assert state.meta["aruco"] == {"confirmed": 0, "rejected": 0}


def test_empty_black_image_no_crash():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    state = ArucoDetector()(_state_for(img))
    assert state.detections == []
    assert state.meta["aruco"] == {"confirmed": 0, "rejected": 0}


def test_deterministic():
    img = _canvas_with_marker(23)
    d1 = ArucoDetector()(_state_for(img)).detections
    d2 = ArucoDetector()(_state_for(img)).detections
    assert [d.meta["aruco_id"] for d in d1] == [d.meta["aruco_id"] for d in d2]
    assert [d.bbox for d in d1] == [d.bbox for d in d2]


def test_does_not_mutate_original_or_current():
    img = _canvas_with_marker(23)
    original_copy = img.copy()
    current_copy = img.copy()
    state = VisionState(original=img, current=current_copy)
    ArucoDetector()(state)
    assert np.array_equal(state.original, original_copy)
    assert np.array_equal(state.current, current_copy)


def test_valid_ids_configurable():
    state = ArucoDetector(valid_ids=(5,))(_state_for(_canvas_with_marker(5)))
    assert len(state.detections) == 1
    assert state.detections[0].meta["aruco_id"] == 5
