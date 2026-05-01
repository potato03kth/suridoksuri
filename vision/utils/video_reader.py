from pathlib import Path
import cv2
import numpy as np
from typing import Iterator


class VideoReader:
    """파일 경로 → 프레임 이터레이터. with 문 또는 for 문으로 사용."""

    def __init__(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"Video not found: {path}")
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._cap.release()
