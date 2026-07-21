import cv2
import numpy as np
from vision.core.state import VisionState


class RedRingDetector:
    """
    버티포트 coarse 3차(§5.2): 빨간 고리(직경2m·5cm) 색게이팅 + 원피팅.
    HSV 빨강은 Hue=0 부근과 179 부근 양끝에 걸쳐있어 두 구간을 OR로 게이팅한다
    (ColorFilter의 단일 range로는 못 잡는 경계 — vision_plan.md §5.4 blind spot 대응).
    게이팅된 픽셀에 최소외접원을 피팅해 중심/반지름을 낸다(면적 blob이 아니라 선이므로).
    color_filter가 mask 밖 픽셀을 지워버린 current 대신 원본 색상이 보존된 original을 읽는다.
    """

    def __init__(
        self,
        low_hue_max: int = 10,
        high_hue_min: int = 170,
        sat_min: int = 100,
        val_min: int = 80,
        min_points: int = 20,
        roi_margin: float = 0.1,
    ):
        self.low_hue_max = low_hue_max
        self.high_hue_min = high_hue_min
        self.sat_min = sat_min
        self.val_min = val_min
        self.min_points = min_points
        self.roi_margin = roi_margin

    def _red_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, self.sat_min, self.val_min], dtype=np.uint8)
        upper1 = np.array([self.low_hue_max, 255, 255], dtype=np.uint8)
        lower2 = np.array([self.high_hue_min, self.sat_min, self.val_min], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    def __call__(self, state: VisionState) -> VisionState:
        refined = []

        for det in state.detections:
            x, y, w, h = det.bbox
            mx, my = int(w * self.roi_margin), int(h * self.roi_margin)
            x0, y0 = max(0, x - mx), max(0, y - my)
            x1 = min(state.original.shape[1], x + w + mx)
            y1 = min(state.original.shape[0], y + h + my)
            roi = state.original[y0:y1, x0:x1]
            if roi.size == 0:
                continue

            mask = self._red_mask(roi)
            points = cv2.findNonZero(mask)
            if points is None or len(points) < self.min_points:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(points)
            det.meta["red_ring"] = {
                "center_px": (float(cx + x0), float(cy + y0)),
                "radius_px": float(radius),
                "gated_points": int(len(points)),
            }
            refined.append(det)

        state.detections = refined
        state.meta["red_ring"] = {"confirmed": len(refined)}
        return state
