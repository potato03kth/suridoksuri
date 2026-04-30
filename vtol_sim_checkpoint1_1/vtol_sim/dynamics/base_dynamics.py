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
    pos: np.ndarray = field(
        default_factory=lambda: np.zeros(3))  # [x_N, x_E, h]
    v: float = 0.0           # m/s, 대기속도 (≈ 대지속도, 무풍 시)
    chi: float = 0.0         # rad, 방위각 (True North 기준, 시계방향 양)
    gamma: float = 0.0       # rad, 비행경로각 (상승 양)
    phi: float = 0.0         # rad, 뱅크각 (우선회 양)
    mode: str = MODE_CRUISE
    t: float = 0.0           # s, 시뮬레이션 시간

    # 측정용 가속도 (body-frame, 매 step마다 동역학이 계산해줌)
    a_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # [a_x: 전방, a_y: 우측, a_z: 하방] — body 좌표계 (NED 컨벤션)

    # === 알고리즘 평가용 가속도 기록 ===
    # 정책: 동역학은 가속도 한계를 강제하지 않는다.
    # 알고리즘이 무리한 명령을 내리면 그대로 적용되고, 그 결과로 발생하는
    # 한계 위반(violation)을 raw 데이터로 기록하여 평가에 사용한다.
    # (속도 한계, 구조적 자세 한계 등 물리적 한계는 여전히 강제됨)
    a_total_cmd: float = 0.0     # m/s², 명령된 총 가속도 크기 (= 실제 적용된 값)
    a_total_actual: float = 0.0  # m/s², 실제 총 가속도 크기 (수평면)
    # 이 모델에서는 a_total_cmd == a_total_actual
    # (속도/자세 saturation에 의해 미세 차이 가능)
    a_max_used: float = 0.0      # m/s², 평가 기준이 되는 가속도 한계
    # (시나리오 override 반영한 실제값)
    # True면 이 step에서 a_total > a_max_used (위반)
    accel_violation: bool = False
    # m/s², 한계 초과량 = max(0, a_total - a_max)
    accel_violation_amount: float = 0.0

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
