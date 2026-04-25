"""
GPS 측정 노이즈
================

위치 측정 = 진위치 + 백색잡음 + Gauss-Markov 바이어스

Gauss-Markov:
    db/dt = -b/τ + w(t),  w ~ N(0, σ_drive²)
    이산화: b_{k+1} = (1 - dt/τ) b_k + sqrt(dt) σ_drive ξ_k
"""
from __future__ import annotations
import numpy as np
from .base_noise import BaseNoise


class GPSNoise(BaseNoise):
    def __init__(self, sigma_white: float = 0.5,
                 tau_bias: float = 300.0,
                 sigma_bias_drive: float = 0.05,
                 seed: int | None = None):
        super().__init__(seed)
        self.sigma_white = sigma_white
        self.tau_bias = tau_bias
        self.sigma_bias_drive = sigma_bias_drive
        self._bias = np.zeros(3)

    def step_bias(self, dt: float) -> None:
        """매 dt마다 바이어스 업데이트."""
        decay = max(0.0, 1.0 - dt / self.tau_bias)
        drive = np.sqrt(dt) * self.sigma_bias_drive * self.rng.standard_normal(3)
        self._bias = decay * self._bias + drive

    def measure(self, true_pos: np.ndarray) -> np.ndarray:
        """진위치 → 측정값."""
        white = self.sigma_white * self.rng.standard_normal(3)
        return true_pos + white + self._bias

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._bias = np.zeros(3)
