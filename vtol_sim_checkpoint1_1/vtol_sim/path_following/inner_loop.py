"""
Inner Loop — 저수준 제어 (고도, 속도)
=======================================

Guidance loop가 출력한 (γ_cmd, v_cmd)를 (pitch_cmd, thrust_cmd)로 변환.
- 고도(γ) PI 제어 → pitch_cmd
- 속도 PI 제어 → thrust_cmd ([-1, 1])

뱅크각은 NLGL이 직접 출력하므로 inner loop를 거치지 않음.
"""
from __future__ import annotations
import numpy as np


class InnerLoopPI:
    """단순 PI 제어기 — 고도와 속도용."""

    def __init__(
        self,
        # 고도 추적용
        kp_h: float = 0.05,
        ki_h: float = 0.005,
        gamma_max_rad: float = np.deg2rad(15.0),
        gamma_min_rad: float = np.deg2rad(-10.0),
        # 속도 추적용
        kp_v: float = 0.3,
        ki_v: float = 0.05,
        thrust_min: float = -1.0,
        thrust_max: float = 1.0,
    ):
        self.kp_h = kp_h
        self.ki_h = ki_h
        self.gamma_max = gamma_max_rad
        self.gamma_min = gamma_min_rad

        self.kp_v = kp_v
        self.ki_v = ki_v
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max

        # 적분 누적 상태
        self._h_integ = 0.0
        self._v_integ = 0.0

    def compute_pitch(self, h_ref: float, h_curr: float, dt: float) -> float:
        """고도 PI → γ_cmd (rad)."""
        err = h_ref - h_curr
        # anti-windup: 출력 saturation 시 적분 정지 (단순 clamp)
        self._h_integ += err * dt
        gamma_cmd = self.kp_h * err + self.ki_h * self._h_integ
        # saturation
        gamma_cmd_clipped = float(np.clip(gamma_cmd, self.gamma_min, self.gamma_max))
        # 포화 시 적분 되돌림 (anti-windup back-calculation)
        if gamma_cmd != gamma_cmd_clipped:
            # 단순화: 적분 누적 차감
            self._h_integ -= err * dt
        return gamma_cmd_clipped

    def compute_thrust(self, v_ref: float, v_curr: float, dt: float) -> float:
        """속도 PI → thrust_cmd ([-1, 1])."""
        err = v_ref - v_curr
        self._v_integ += err * dt
        thrust_cmd = self.kp_v * err + self.ki_v * self._v_integ
        thrust_clipped = float(np.clip(thrust_cmd, self.thrust_min, self.thrust_max))
        if thrust_cmd != thrust_clipped:
            self._v_integ -= err * dt
        return thrust_clipped

    def reset(self) -> None:
        self._h_integ = 0.0
        self._v_integ = 0.0
