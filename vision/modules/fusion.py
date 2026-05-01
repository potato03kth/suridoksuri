import numpy as np
from vision.core.state import VisionState, Detection


class TemporalFusion:
    """
    프레임 간 detection을 누적해 신뢰도를 높인다.
    같은 위치에 min_frames 회 이상 등장한 detection을 state.confirmed에 확정.
    """

    def __init__(self, min_frames: int = 5, iou_threshold: float = 0.4):
        self.min_frames = min_frames
        self.iou_threshold = iou_threshold
        self._candidates: list[dict] = []  # {detection, count}

    def __call__(self, state: VisionState) -> VisionState:
        if not state.detections:
            self._decay()
            return state

        for det in state.detections:
            matched = self._match(det)
            if matched is not None:
                matched["count"] += 1
                matched["detection"] = det
            else:
                self._candidates.append({"detection": det, "count": 1})

        self._decay()

        confirmed = [c for c in self._candidates if c["count"] >= self.min_frames]
        if confirmed:
            best = max(confirmed, key=lambda c: c["count"])
            state.confirmed = best["detection"]
            state.meta["fusion"] = {"confirmed_count": best["count"]}

        return state

    def _match(self, det: Detection) -> dict | None:
        for c in self._candidates:
            if self._iou(det.bbox, c["detection"].bbox) >= self.iou_threshold:
                return c
        return None

    def _decay(self):
        for c in self._candidates:
            c["count"] = max(0, c["count"] - 1)
        self._candidates = [c for c in self._candidates if c["count"] > 0]

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0
