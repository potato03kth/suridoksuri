import cv2
import numpy as np
from vision.core.state import VisionState


def _v_template_contour(size: int = 200, thickness: int = 30) -> np.ndarray:
    """matchShapes 비교용 참조 V(solid) 컨투어. 두꺼운 두 선분으로 쉐브론을 만든다."""
    canvas = np.zeros((size, size), dtype=np.uint8)
    apex = (size // 2, size - 10)
    left = (10, 10)
    right = (size - 10, 10)
    cv2.line(canvas, left, apex, 255, thickness)
    cv2.line(canvas, apex, right, 255, thickness)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)


class BlackVMatcher:
    """
    버티포트 coarse 2차(§5.2): 흰 필드 내부 검은 V 형상 확인.
    1차(white_field) bbox 안의 어두운 영역을 참조 V 컨투어와 matchShapes(Hu moments, 회전/스케일 불변)로 비교해
    형상이 맞지 않는 후보(다른 밝은 오탐)를 배제한다.
    color_filter가 mask 밖 픽셀을 지워버린 current 대신 원본 색상이 보존된 original을 읽는다.
    """

    def __init__(
        self,
        val_max: int = 80,
        min_area: int = 100,
        max_match_distance: float = 0.5,
        roi_margin: float = 0.1,
    ):
        self.val_max = val_max
        self.min_area = min_area
        self.max_match_distance = max_match_distance
        self.roi_margin = roi_margin
        self._template = _v_template_contour()

    def __call__(self, state: VisionState) -> VisionState:
        confirmed = []
        rejected = 0

        for det in state.detections:
            x, y, w, h = det.bbox
            mx, my = int(w * self.roi_margin), int(h * self.roi_margin)
            x0, y0 = max(0, x - mx), max(0, y - my)
            x1 = min(state.original.shape[1], x + w + mx)
            y1 = min(state.original.shape[0], y + h + my)
            roi = state.original[y0:y1, x0:x1]
            if roi.size == 0:
                rejected += 1
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, dark_mask = cv2.threshold(gray, self.val_max, 255, cv2.THRESH_BINARY_INV)

            # bbox는 정사각형이라 원형 필드 바깥 모서리(배경)도 같이 크롭된다 — 1차(white_field)가 준
            # 중심/반지름 meta로 원 안쪽만 남겨 배경이 어두운 오탐 컨투어로 잡히는 것을 막는다.
            if "center" in det.meta and "radius" in det.meta:
                cx, cy = det.meta["center"]
                radius = det.meta["radius"]
                field_mask = np.zeros(dark_mask.shape, dtype=np.uint8)
                cv2.circle(field_mask, (int(cx - x0), int(cy - y0)), int(radius * 0.95), 255, -1)
                dark_mask = cv2.bitwise_and(dark_mask, field_mask)

            contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = [c for c in contours if cv2.contourArea(c) >= self.min_area]

            best_score = min(
                (cv2.matchShapes(c, self._template, cv2.CONTOURS_MATCH_I1, 0.0) for c in candidates),
                default=None,
            )

            if best_score is not None and best_score <= self.max_match_distance:
                det.meta["black_v"] = {"match_score": round(float(best_score), 4)}
                confirmed.append(det)
            else:
                rejected += 1

        state.detections = confirmed
        state.meta["black_v"] = {"confirmed": len(confirmed), "rejected": rejected}
        return state
