import cv2
import numpy as np
from vision.core.state import VisionState, Detection


class RectDetector:
    """
    마스크에서 윤곽선을 추출하고 사각형 후보를 state.detections에 추가.
    approx_epsilon: 폴리곤 근사 정밀도 (윤곽선 둘레 대비 비율).
    """

    def __init__(
        self,
        min_area: int = 500,
        max_area: int = 500_000,
        approx_epsilon: float = 0.04,
        min_solidity: float = 0.7,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.approx_epsilon = approx_epsilon
        self.min_solidity = min_solidity

    def __call__(self, state: VisionState) -> VisionState:
        src = state.mask if state.mask is not None else cv2.cvtColor(state.current, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area <= area <= self.max_area):
                continue

            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < self.min_solidity:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.approx_epsilon * peri, True)
            x, y, w, h = cv2.boundingRect(approx)

            confidence = solidity * min(1.0, area / self.min_area)
            detections.append(Detection(
                bbox=(x, y, w, h),
                confidence=round(confidence, 3),
                corners=approx.reshape(-1, 2) if len(approx) == 4 else None,
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        state.detections = detections
        state.meta["rect_detector"] = {"candidates": len(detections)}
        return state
