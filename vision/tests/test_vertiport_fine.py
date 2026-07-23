"""presets/vertiport_fine.yaml — ArUco Phase 4 fine 단계 preset (docs/vision_aruco_branch.md).

버티포트 coarse 캐스케이드(test_vertiport_cascade.py)와 독립적으로 실행되는 preset이라는
설계 판단(vertiport_fine.yaml 헤더 주석 참조)을 그 자체로 검증한다 — Pipeline.from_config()로
실제로 로드되고, 합성 ArUco 마커 이미지에서 실제로 검출되는지 end-to-end로 확인한다.
"""
import cv2
import numpy as np
from vision.core.runner import Pipeline

_PRESET = "vision/presets/vertiport_fine.yaml"
_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _canvas_with_marker(marker_id: int, x: int = 50, y: int = 50, marker_size: int = 150,
                         canvas_size: int = 300) -> np.ndarray:
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    marker_gray = cv2.aruco.generateImageMarker(_DICTIONARY, marker_id, marker_size)
    canvas[y:y + marker_size, x:x + marker_size] = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
    return canvas


def test_preset_loads_and_confirms_id_23():
    pipeline = Pipeline.from_config(_PRESET)
    state = pipeline.run(_canvas_with_marker(23))

    assert len(state.detections) == 1
    det = state.detections[0]
    assert det.meta["aruco_id"] == 23
    assert det.corners is not None
    assert det.corners.shape == (4, 2)


def test_preset_rejects_other_id():
    pipeline = Pipeline.from_config(_PRESET)
    state = pipeline.run(_canvas_with_marker(7))
    assert state.detections == []
    assert state.meta["aruco"]["rejected"] == 1


def test_preset_empty_image_no_detections():
    pipeline = Pipeline.from_config(_PRESET)
    state = pipeline.run(np.full((300, 300, 3), 255, dtype=np.uint8))
    assert state.detections == []
    assert state.meta["aruco"] == {"confirmed": 0, "rejected": 0}
