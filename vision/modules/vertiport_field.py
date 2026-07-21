import cv2
import numpy as np
from vision.core.state import VisionState, Detection


class WhiteFieldDetector:
    """
    버티포트 coarse 1차(§5.2): 흰색 채워진 원형 필드 탐지.
    ColorFilter(mode="gray")가 만든 mask에서 원형도 높은 큰 blob을 후보로 낸다.
    """

    def __init__(
        self,
        min_area: int = 500,
        max_area: int = 2_000_000,
        min_circularity: float = 0.6,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity

    def __call__(self, state: VisionState) -> VisionState:
        detections: list[Detection] = []
        if state.mask is not None:
            contours, _ = cv2.findContours(state.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (self.min_area <= area <= self.max_area):
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < self.min_circularity:
                    continue

                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append(Detection(
                    bbox=(x, y, w, h),
                    confidence=round(min(1.0, circularity), 3),
                    meta={"center": (float(cx), float(cy)), "radius": float(radius)},
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        state.detections = detections
        state.meta["white_field"] = {"candidates": len(detections)}
        return state
