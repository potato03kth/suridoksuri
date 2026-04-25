"""
바람 모델
==========

w(t) = w_steady + w_gust(t)

w_gust는 1차 Markov (Dryden 난류의 간이 버전):
    dw_g/dt = -(v/L) w_g + σ_g sqrt(2v/L) η(t)
    이산화: w_g_{k+1} = (1 - v dt/L) w_g_k + σ_g sqrt(2 v dt / L) ξ_k

L: 길이 스케일 (m). 보통 종축 ~200 m.
"""
from __future__ import annotations
import numpy as np
from .base_noise import BaseNoise


class WindModel(BaseNoise):
    def __init__(self, w_steady: tuple[float, float, float] = (0, 0, 0),
                 sigma_gust: float = 1.0,
                 length_scale: float = 200.0,
                 seed: int | None = None):
        super().__init__(seed)
        self.w_steady = np.array(w_steady, dtype=float)
        self.sigma = sigma_gust
        self.L = length_scale
        self._w_gust = np.zeros(3)

    def step(self, airspeed: float, dt: float) -> np.ndarray:
        """매 dt마다 바람 업데이트 후 현재 바람 벡터 반환 [w_N, w_E, w_D]."""
        v = max(airspeed, 1.0)  # 안전 floor
        decay = max(0.0, 1.0 - v * dt / self.L)
        drive_std = self.sigma * np.sqrt(2.0 * v * dt / self.L)
        drive = drive_std * self.rng.standard_normal(3)
        self._w_gust = decay * self._w_gust + drive
        return self.w_steady + self._w_gust

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._w_gust = np.zeros(3)
