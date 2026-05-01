import cv2
import numpy as np
from vision.core.state import VisionState


class ColorFilter:
    """
    HSV 색공간에서 마스크 생성.
    mode='gray': 낮은 채도(무채색) 영역 추출 — 착륙 패드 탐지 기본 모드.
    mode='color': 특정 색상 범위 추출.
    """

    def __init__(
        self,
        mode: str = "gray",
        sat_max: int = 50,
        val_min: int = 80,
        val_max: int = 255,
        hue_range: tuple[int, int] = (0, 180),
        sat_min: int = 30,
    ):
        self.mode = mode
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self.hue_range = hue_range
        self.sat_min = sat_min

    def __call__(self, state: VisionState) -> VisionState:
        hsv = cv2.cvtColor(state.current, cv2.COLOR_BGR2HSV)

        if self.mode == "gray":
            lower = np.array([0, 0, self.val_min], dtype=np.uint8)
            upper = np.array([180, self.sat_max, self.val_max], dtype=np.uint8)
        else:
            lower = np.array([self.hue_range[0], self.sat_min, self.val_min], dtype=np.uint8)
            upper = np.array([self.hue_range[1], 255, self.val_max], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        state.mask = mask
        state.current = cv2.bitwise_and(state.current, state.current, mask=mask)
        state.meta["color_filter"] = {"mode": self.mode, "nonzero_ratio": float(np.count_nonzero(mask)) / mask.size}
        return state
