"""
노이즈 모듈 베이스
====================

각 노이즈 컴포넌트는 자체 RNG 인스턴스를 보유하여 시드 재현 가능.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseNoise(ABC):
    """모든 노이즈 컴포넌트의 베이스. 자체 RNG."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
