import cv2
import numpy as np
from vision.core.state import VisionState


class EdgeDetector:
    """Canny 엣지 검출 후 결과를 state.mask에 덮어씀."""

    def __init__(self, low_threshold: int = 50, high_threshold: int = 150):
        self.low = low_threshold
        self.high = high_threshold

    def __call__(self, state: VisionState) -> VisionState:
        gray = cv2.cvtColor(state.current, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.low, self.high)

        if state.mask is not None:
            edges = cv2.bitwise_and(edges, state.mask)

        state.mask = edges
        return state
