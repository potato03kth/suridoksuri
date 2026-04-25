"""
동역학 추상 인터페이스 + 데이터 클래스
=========================================
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Optional
import numpy as np


# 모드 enum (간단히 문자열 사용)
MODE_HOVER = "HOVER"
MODE_TRANSITION = "TRANSITION"
MODE_CRUISE = "CRUISE"


@dataclass
class AircraftState:
    """
    항공기 상태.

    위치는 NED 지역 좌표 [x_N, x_E, h] (h = -z_NED, 양수 = 위쪽).
    각도는 모두 라디안.
    """
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [x_N, x_E, h]
    v: float = 0.0           # m/s, 대기속도 (≈ 대지속도, 무풍 시)
    chi: float = 0.0         # rad, 방위각 (True North 기준, 시계방향 양)
    gamma: float = 0.0       # rad, 비행경로각 (상승 양)
    phi: float = 0.0         # rad, 뱅크각 (우선회 양)
    mode: str = MODE_CRUISE
    t: float = 0.0           # s, 시뮬레이션 시간

    # 측정용 가속도 (body-frame, 매 step마다 동역학이 계산해줌)
    a_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # [a_x: 전방, a_y: 우측, a_z: 하방] — body 좌표계 (NED 컨벤션)

    def copy(self) -> "AircraftState":
        return replace(
            self,
            pos=self.pos.copy(),
            a_body=self.a_body.copy(),
        )

    def as_array(self) -> np.ndarray:
        """[x_N, x_E, h, v, chi, gamma, phi]"""
        return np.array([self.pos[0], self.pos[1], self.pos[2],
                         self.v, self.chi, self.gamma, self.phi])


@dataclass
class ControlInput:
    """
    제어 입력. Inner loop에서 출력하는 저수준 명령.
    """
    bank_cmd: float = 0.0         # rad, φ_cmd
    pitch_cmd: float = 0.0        # rad, θ_cmd (또는 γ_cmd 직접)
    thrust_cmd: float = 0.0       # 정규화 [0, 1] 또는 직접 가속도 (m/s²)

    # 이 형태는 Guidance에서 직접 출력하는 고수준 명령 (대안)
    # → inner_loop에서 위 3개로 변환
    chi_cmd: Optional[float] = None
    gamma_cmd: Optional[float] = None
    v_cmd: Optional[float] = None

    def copy(self) -> "ControlInput":
        return replace(self)


class BaseDynamics(ABC):
    """동역학 모델 추상 인터페이스."""

    @abstractmethod
    def step(self, state: AircraftState, u: ControlInput,
             wind: np.ndarray, dt: float) -> AircraftState:
        """
        한 스텝 진행. state는 변경하지 않고 새 상태를 반환.

        Parameters
        ----------
        state : 현재 상태
        u : 제어 입력 (저수준)
        wind : (3,) 바람 벡터 [w_N, w_E, w_D] (m/s)
        dt : 적분 스텝 (s)

        Returns
        -------
        new_state : 다음 상태
        """

    @abstractmethod
    def reset(self, initial_state: AircraftState) -> None:
        pass
