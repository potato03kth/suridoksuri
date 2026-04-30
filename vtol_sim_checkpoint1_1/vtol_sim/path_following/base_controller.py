"""
경로 추종 제어기 추상 인터페이스
=================================
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from dynamics.base_dynamics import ControlInput
from path_planning.base_planner import Path


class BaseController(ABC):
    """경로 추종 제어기 베이스."""

    @abstractmethod
    def compute(self, est_pos: np.ndarray, est_vel: np.ndarray,
                est_chi: float, est_gamma: float, est_phi: float, est_v: float,
                path: Path, t: float, dt: float) -> ControlInput:
        """
        측정/추정 상태 + 경로 → 제어 입력 계산.

        Parameters
        ----------
        est_pos : (3,) 추정 위치 [x_N, x_E, h]
        est_vel : (3,) 추정 속도 (NED) — α-β 필터 출력
        est_chi, est_gamma, est_phi : 측정/추정 자세 (rad)
        est_v : 추정 대기속도 (m/s)
        path : 추종할 경로
        t : 현재 시간 (s)
        dt : 제어 주기 (s)

        Returns
        -------
        ControlInput (bank_cmd, pitch_cmd, thrust_cmd)
        """

    @abstractmethod
    def reset(self) -> None:
        """내부 상태(적분기 등) 초기화."""
