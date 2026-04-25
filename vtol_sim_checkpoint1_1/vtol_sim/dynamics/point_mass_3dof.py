"""
3-DoF Coordinated-Turn 점질량 동역학
======================================

상태: x = [x_N, x_E, h, v, chi, gamma, phi]
입력: u = (bank_cmd, pitch_cmd or gamma_cmd, thrust_cmd or a_T_cmd)

운동 방정식 (Cruise 모드):
    dx_N/dt   = v cos(γ) cos(χ) + w_N
    dx_E/dt   = v cos(γ) sin(χ) + w_E
    dh/dt     = v sin(γ) - w_D
    dv/dt     = a_T - g sin(γ)        # a_T: 접선 가속도 (추력에 의한)
    dχ/dt     = g tan(φ) / (v cos(γ)) # coordinated turn
    dγ/dt     = (a_z_cmd cos(φ) - g cos(γ)) / v  # 단순화
    dφ/dt     = (φ_cmd - φ) / τ_φ     # 1차 지연

** 단순화 가정 **:
- 추력 명령 thrust_cmd가 직접 접선 가속도 a_T (m/s²)로 매핑됨
  (또는 [0,1] 정규화 → a_T = thrust_cmd × a_max로 변환)
- pitch_cmd는 γ_cmd로 해석 → γ를 1차 지연으로 추적
- 항력은 명시적 모델 없음 (질문 Q-E2 답변에 따라 단순화)
"""
from __future__ import annotations
import numpy as np

from .base_dynamics import (
    BaseDynamics, AircraftState, ControlInput,
    MODE_HOVER, MODE_TRANSITION, MODE_CRUISE,
)


class PointMass3DoF(BaseDynamics):
    """3-DoF coordinated-turn 점질량 모델."""

    def __init__(self, params: dict):
        """
        Parameters
        ----------
        params : dict
            aircraft.yaml에서 로드한 파라미터
        """
        self.g = float(params["gravity"])
        self.mass = float(params["mass"])
        self.v_min = float(params["v_stall"])
        self.v_max = float(params["v_max"])
        self.a_max = float(params["a_max_g"]) * self.g
        self.phi_max = np.deg2rad(float(params["phi_max_deg"]))
        self.gamma_max = np.deg2rad(float(params["gamma_max_deg"]))
        self.gamma_min = np.deg2rad(float(params["gamma_min_deg"]))
        self.h_min = float(params["h_min"])
        self.h_max = float(params["h_max"])
        self.tau_phi = float(params["tau_phi"])
        self.tau_thrust = float(params["tau_thrust"])
        # 추력 명령 → 가속도 매핑 — thrust_cmd ∈ [-1, 1]일 때 a_T = thrust_cmd * a_max
        # (-1 = 최대감속, +1 = 최대가속). 실제 항공기는 비대칭이지만 단순화.
        self.thrust_to_accel = self.a_max

        # 내부 상태 (1차 지연용)
        self._a_T_actual: float = 0.0  # 현재 접선 가속도

    def reset(self, initial_state: AircraftState) -> None:
        self._a_T_actual = 0.0

    def step(self, state: AircraftState, u: ControlInput,
             wind: np.ndarray, dt: float) -> AircraftState:
        """4차 룽게-쿠타 적분 (RK4)."""
        if state.mode != MODE_CRUISE:
            # HOVER/TRANSITION 모드는 별도 처리. 여기서는 상태 그대로 + 시간만 진행.
            new_state = state.copy()
            new_state.t = state.t + dt
            return new_state

        # ---- 입력 처리 ----
        phi_cmd = float(np.clip(u.bank_cmd, -self.phi_max, self.phi_max))
        # pitch_cmd를 γ_cmd로 해석
        gamma_cmd = float(np.clip(u.pitch_cmd, self.gamma_min, self.gamma_max))
        # 추력 명령 → 목표 접선 가속도
        a_T_cmd = float(u.thrust_cmd) * self.thrust_to_accel
        # a_T 1차 지연 (간단히 1-step exponential)
        alpha_T = dt / max(self.tau_thrust, 1e-6)
        alpha_T = min(1.0, alpha_T)
        self._a_T_actual = (1 - alpha_T) * self._a_T_actual + alpha_T * a_T_cmd

        # ---- 가속도 제약 강제 (총 body-frame 가속도) ----
        # 구심가속도: a_n = g·tan(φ) (수평선회 가정)
        # 총 가속도 크기: sqrt(a_T² + a_n²)
        # 0.3g 초과 시 phi_cmd, a_T_cmd 비례 축소
        a_n_cmd = self.g * np.tan(phi_cmd)
        a_total = np.sqrt(self._a_T_actual**2 + a_n_cmd**2)
        if a_total > self.a_max:
            scale = self.a_max / max(a_total, 1e-9)
            self._a_T_actual *= scale
            # 뱅크각 명령도 축소
            a_n_cmd_scaled = a_n_cmd * scale
            phi_cmd_eff = float(np.arctan(a_n_cmd_scaled / self.g))
        else:
            phi_cmd_eff = phi_cmd

        # ---- RK4 적분 ----
        x0 = self._state_to_vec(state)
        k1 = self._derivative(x0, phi_cmd_eff, gamma_cmd, self._a_T_actual, wind)
        k2 = self._derivative(x0 + 0.5 * dt * k1, phi_cmd_eff, gamma_cmd, self._a_T_actual, wind)
        k3 = self._derivative(x0 + 0.5 * dt * k2, phi_cmd_eff, gamma_cmd, self._a_T_actual, wind)
        k4 = self._derivative(x0 + dt * k3, phi_cmd_eff, gamma_cmd, self._a_T_actual, wind)
        x_new = x0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # ---- 제약 강제 (post-integration) ----
        x_new = self._enforce_constraints(x_new)

        # ---- body-frame 가속도 계산 (측정용) ----
        a_body = self._compute_body_acceleration(x_new, self._a_T_actual)

        # ---- 새 상태 구성 ----
        new_state = AircraftState(
            pos=x_new[0:3].copy(),
            v=float(x_new[3]),
            chi=float(x_new[4]),
            gamma=float(x_new[5]),
            phi=float(x_new[6]),
            mode=state.mode,
            t=state.t + dt,
            a_body=a_body,
        )
        return new_state

    # ---- 내부 함수들 ----

    def _state_to_vec(self, state: AircraftState) -> np.ndarray:
        return np.array([
            state.pos[0], state.pos[1], state.pos[2],
            state.v, state.chi, state.gamma, state.phi
        ], dtype=float)

    def _derivative(self, x: np.ndarray, phi_cmd: float, gamma_cmd: float,
                    a_T: float, wind: np.ndarray) -> np.ndarray:
        """상태 미분 dx/dt."""
        x_N, x_E, h, v, chi, gamma, phi = x
        w_N, w_E, w_D = wind

        # 안전: v는 적분 중 일시적으로 v_min 이하가 될 수 있으므로 floor
        v_safe = max(v, 1e-3)
        cos_gamma = np.cos(gamma)
        # gamma가 ±π/2에 가까워지면 발산하므로 안전
        cos_gamma_safe = np.sign(cos_gamma) * max(abs(cos_gamma), 0.05)

        dx_N = v * cos_gamma * np.cos(chi) + w_N
        dx_E = v * cos_gamma * np.sin(chi) + w_E
        dh = v * np.sin(gamma) - w_D
        dv = a_T - self.g * np.sin(gamma)
        dchi = self.g * np.tan(phi) / (v_safe * cos_gamma_safe)

        # γ 1차 지연 (피치 명령 추적)
        # 더 단순하게: dγ/dt = (γ_cmd - γ) / τ_γ 형태로
        tau_gamma = 0.3  # s, 피치 응답 시정수
        dgamma = (gamma_cmd - gamma) / tau_gamma

        # φ 1차 지연
        dphi = (phi_cmd - phi) / self.tau_phi

        return np.array([dx_N, dx_E, dh, dv, dchi, dgamma, dphi])

    def _enforce_constraints(self, x: np.ndarray) -> np.ndarray:
        """제약 강제 — 적분 후 상태."""
        x_out = x.copy()
        # 속도
        x_out[3] = np.clip(x_out[3], self.v_min, self.v_max)
        # γ
        x_out[5] = np.clip(x_out[5], self.gamma_min, self.gamma_max)
        # φ
        x_out[6] = np.clip(x_out[6], -self.phi_max, self.phi_max)
        # 고도 (지오펜스 — 경고만, 강제 clip은 부드럽게)
        x_out[2] = np.clip(x_out[2], self.h_min, self.h_max)
        # χ wrap
        x_out[4] = np.mod(x_out[4] + np.pi, 2 * np.pi) - np.pi
        return x_out

    def _compute_body_acceleration(self, x: np.ndarray, a_T: float
                                   ) -> np.ndarray:
        """
        body-frame 가속도 [a_x, a_y, a_z].

        - a_x: 전방 (접선) — 추력에서 중력 보상 후 남은 분
        - a_y: 우측 (구심) — 선회 시 발생
        - a_z: 하방 — 중력의 body-z 성분 + 양력 분담
        """
        v, chi, gamma, phi = x[3], x[4], x[5], x[6]
        # 전방 가속도: a_T - g sin(γ) (이미 dv/dt 식)
        a_x = a_T - self.g * np.sin(gamma)
        # 측방 가속도: 구심가속도 = g·tan(φ)·cos(γ), body-y 양은 우측회전 시 +
        a_y = self.g * np.tan(phi) * np.cos(gamma)
        # 수직 가속도 (body-z): -g cos(γ) cos(φ) (수평비행 시 -g)
        # NED body-z는 down(+) 방향이지만 IMU가 측정하는 specific force는
        # body-z 방향으로 -양력. 단순히 -g·cos(γ)/cos(φ) 정도로 근사.
        # IMU specific force convention: a_z ≈ -g/cos(φ) (수평비행 선회 시).
        # 여기선 단순화: a_z = -g cos(γ) / cos(φ)
        cos_phi = max(abs(np.cos(phi)), 0.1) * np.sign(np.cos(phi))
        a_z = -self.g * np.cos(gamma) / cos_phi

        return np.array([a_x, a_y, a_z])
