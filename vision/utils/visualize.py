from pathlib import Path
import cv2
import numpy as np
from vision.core.state import Detection, VisionState


def draw_detections(image: np.ndarray, detections: list[Detection],
                    confirmed: Detection | None = None) -> np.ndarray:
    out = image.copy()
    for det in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
        label = f"{det.confidence:.2f}"
        cv2.putText(out, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    if confirmed is not None:
        x, y, w, h = confirmed.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(out, "CONFIRMED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return out


def draw_mask(mask: np.ndarray) -> np.ndarray:
    """단채널 마스크 → 시각화용 BGR 이미지."""
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def save_result(state: VisionState, output_path: str) -> None:
    annotated = draw_detections(state.original, state.detections, state.confirmed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, annotated)
