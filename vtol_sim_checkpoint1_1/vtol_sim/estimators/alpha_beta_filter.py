"""
α-β 필터 (위치 + 속도 추정)
=============================

3차원 위치 측정 z_k가 들어올 때마다:
    예측: p̂_k⁻ = p̂_{k-1} + v̂_{k-1} Δt
         v̂_k⁻ = v̂_{k-1}
    잔차: r_k = z_k - p̂_k⁻
    갱신: p̂_k = p̂_k⁻ + α r_k
         v̂_k = v̂_k⁻ + (β/Δt) r_k

여기서는 위치만 추정하고 헤딩/롤은 측정 또는 진값 사용 (실제 IMU 모델 미포함).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EstimatedState:
    """추정 상태 — 측정 기반 위치/속도, 나머지는 동역학 진값에서 가져옴(또는 IMU 가정)."""
    pos_est: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel_est: np.ndarray = field(default_factory=lambda: np.zeros(3))  # NED 좌표 속도
    # 자세는 IMU가 정확히 알려준다고 가정 (Phase 1 단순화)
    chi: float = 0.0
    gamma: float = 0.0
    phi: float = 0.0
    v_air: float = 0.0  # 대기속도 (피토 등 별도 센서 가정)
    t: float = 0.0


class AlphaBetaFilter:
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        self.alpha = alpha
        self.beta = beta
        self._p_hat = np.zeros(3)
        self._v_hat = np.zeros(3)
        self._t_last = None

    def initialize(self, pos: np.ndarray, vel: np.ndarray, t: float) -> None:
        self._p_hat = pos.astype(float).copy()
        self._v_hat = vel.astype(float).copy()
        self._t_last = t

    def update(self, z: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """측정 z (위치)로 갱신. (p_hat, v_hat) 반환."""
        if self._t_last is None:
            self.initialize(z, np.zeros(3), t)
            return self._p_hat.copy(), self._v_hat.copy()

        dt = t - self._t_last
        if dt <= 0:
            return self._p_hat.copy(), self._v_hat.copy()

        # 예측
        p_pred = self._p_hat + self._v_hat * dt
        # 잔차
        r = z - p_pred
        # 갱신
        self._p_hat = p_pred + self.alpha * r
        self._v_hat = self._v_hat + (self.beta / dt) * r
        self._t_last = t

        return self._p_hat.copy(), self._v_hat.copy()

    def reset(self) -> None:
        self._p_hat = np.zeros(3)
        self._v_hat = np.zeros(3)
        self._t_last = None
