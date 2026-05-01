from pathlib import Path
import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """파일 경로 → BGR numpy 배열. 파일이 없으면 FileNotFoundError."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"cv2 could not decode: {path}")
    return img
