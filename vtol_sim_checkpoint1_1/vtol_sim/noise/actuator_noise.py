"""
액추에이터 노이즈
==================

조종면 명령 → 실제 명령 사이의 비이상성:
- 데드밴드: |φ_cmd - φ| < φ_db일 때 명령 무시
- 바이어스: 일정한 오프셋 (run마다 무작위)
- 1샷 화이트노이즈: 매 호출마다 작은 잡음
"""
from __future__ import annotations
import numpy as np
from .base_noise import BaseNoise
from dynamics.base_dynamics import ControlInput


class ActuatorNoise(BaseNoise):
    def __init__(self, deadband_rad: float = 0.0,
                 bias_sigma_rad: float = 0.0,
                 seed: int | None = None):
        super().__init__(seed)
        self.deadband = deadband_rad
        self.bias_sigma = bias_sigma_rad
        # 시드별로 한 번 추출되는 바이어스 (run-constant)
        self._bias_phi = float(self.bias_sigma * self.rng.standard_normal())

    def apply(self, u_cmd: ControlInput, current_phi: float) -> ControlInput:
        """제어 입력에 데드밴드 + 바이어스 적용."""
        u_out = u_cmd.copy()
        # 데드밴드: 현재 phi와 명령 phi의 차이가 deadband 이하면 명령 = 현재 phi (변화 없음)
        if abs(u_cmd.bank_cmd - current_phi) < self.deadband:
            u_out.bank_cmd = current_phi
        # 바이어스 적용
        u_out.bank_cmd = u_out.bank_cmd + self._bias_phi
        return u_out

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._bias_phi = float(self.bias_sigma * self.rng.standard_normal())
