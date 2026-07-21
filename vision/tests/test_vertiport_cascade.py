import cv2
import numpy as np
from vision.core.runner import Pipeline

_PRESET = "vision/presets/vertiport_coarse.yaml"


def _synthetic_vertiport(size: int = 400) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)

    cv2.circle(img, center, 150, (255, 255, 255), -1)  # 1차: 흰 필드

    apex = (center[0], center[1] + 90)
    left = (center[0] - 90, center[1] - 90)
    right = (center[0] + 90, center[1] - 90)
    cv2.line(img, left, apex, (0, 0, 0), 30)  # 2차: 검은 V
    cv2.line(img, apex, right, (0, 0, 0), 30)

    # V가 중심부까지 뻗어 3차 빨간 고리와 맞닿지 않도록, 고리를 그릴 중심 원판을 다시 흰색으로 비운다.
    # (V 팔이 중심에서 최소 ~25px까지 접근하므로 지움 반경은 그보다 확실히 작게 잡는다)
    cv2.circle(img, center, 22, (255, 255, 255), -1)
    cv2.circle(img, center, 13, (0, 0, 255), thickness=3)  # 3차: 빨간 고리
    return img


def test_cascade_confirms_all_three_stages():
    pipeline = Pipeline.from_config(_PRESET)
    state = pipeline.run(_synthetic_vertiport())

    assert len(state.detections) == 1
    det = state.detections[0]
    assert "black_v" in det.meta
    assert "red_ring" in det.meta


def test_cascade_meta_recorded_per_stage():
    pipeline = Pipeline.from_config(_PRESET)
    state = pipeline.run(_synthetic_vertiport())

    assert state.meta["white_field"]["candidates"] == 1
    assert state.meta["black_v"] == {"confirmed": 1, "rejected": 0}
    assert state.meta["red_ring"] == {"confirmed": 1}


def test_cascade_empty_image_no_detections():
    pipeline = Pipeline.from_config(_PRESET)
    state = pipeline.run(np.zeros((400, 400, 3), dtype=np.uint8))
    assert state.detections == []
