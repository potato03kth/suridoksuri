import cv2
import numpy as np
from vision.core.state import VisionState


class MorphologyModule:
    """
    형태학적 연산으로 마스크 노이즈 제거 및 구멍 메우기.
    ops: 적용할 연산 목록 — 'close', 'open', 'dilate', 'erode' 순서대로 적용.
    """

    _OP_MAP = {
        "close":  cv2.MORPH_CLOSE,
        "open":   cv2.MORPH_OPEN,
        "dilate": cv2.MORPH_DILATE,
        "erode":  cv2.MORPH_ERODE,
    }

    def __init__(self, ops: list[str] = None, kernel_size: int = 5):
        self.ops = ops or ["close", "open"]
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )

    def __call__(self, state: VisionState) -> VisionState:
        if state.mask is None:
            return state
        result = state.mask
        for op_name in self.ops:
            morph_type = self._OP_MAP[op_name]
            result = cv2.morphologyEx(result, morph_type, self.kernel)
        state.mask = result
        return state
