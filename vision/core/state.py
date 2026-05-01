from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float = 1.0
    corners: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


@dataclass
class VisionState:
    original: np.ndarray            # BGR 원본 이미지 — 모듈이 수정하지 않음
    current: np.ndarray             # 처리 중인 BGR 작업 이미지
    mask: Optional[np.ndarray] = None           # 이진 마스크 (0/255)
    detections: list[Detection] = field(default_factory=list)
    confirmed: Optional[Detection] = None       # TemporalFusion 확정 결과
    meta: dict = field(default_factory=dict)    # 모듈별 진단 정보
