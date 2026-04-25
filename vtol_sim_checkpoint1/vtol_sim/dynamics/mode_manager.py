"""
모드 관리 (HOVER ↔ TRANSITION ↔ CRUISE)
==========================================

순간 전환 모델:
- 이륙: HOVER 시작 → 고도 h_transition 도달 → CRUISE, v = v_cruise로 점프
- 착륙: 마지막 WP 도착 → HOVER, v = 0으로 점프

이 모듈은 시뮬레이션 시작 시 초기화 + 매 step 후 모드 체크.
"""
from __future__ import annotations
from .base_dynamics import (
    AircraftState, MODE_HOVER, MODE_TRANSITION, MODE_CRUISE
)


class ModeManager:
    def __init__(self, h_transition: float, v_cruise: float,
                 takeoff_climb_rate: float = 3.0):
        """
        Parameters
        ----------
        h_transition : 천이 발생 고도 (m)
        v_cruise : 천이 후 설정될 속도 (m/s)
        takeoff_climb_rate : HOVER 모드에서의 상승률 (m/s)
        """
        self.h_transition = h_transition
        self.v_cruise = v_cruise
        self.climb_rate = takeoff_climb_rate
        self._transition_done = False

    def initialize(self, takeoff_pos_xy: tuple[float, float],
                   start_altitude: float = 0.0) -> AircraftState:
        """
        이륙 시작 상태 — Phase 1에서는 단순화하여 곧장 CRUISE로 시작 가능.
        Phase 1 시연용으로 천이를 생략하려면 직접 CRUISE 상태를 만들면 됨.
        """
        import numpy as np
        return AircraftState(
            pos=np.array([takeoff_pos_xy[0], takeoff_pos_xy[1], start_altitude]),
            v=0.0,
            chi=0.0,
            gamma=0.0,
            phi=0.0,
            mode=MODE_HOVER,
            t=0.0,
        )

    def quick_start_cruise(self, pos: tuple[float, float, float],
                           heading: float = 0.0,
                           v: float | None = None) -> AircraftState:
        """Phase 1 통합 테스트용 — HOVER/TRANSITION 생략하고 CRUISE 즉시 시작."""
        import numpy as np
        return AircraftState(
            pos=np.array(pos, dtype=float),
            v=v if v is not None else self.v_cruise,
            chi=heading,
            gamma=0.0,
            phi=0.0,
            mode=MODE_CRUISE,
            t=0.0,
        )

    def update_mode(self, state: AircraftState,
                    last_wp_reached: bool = False) -> AircraftState:
        """
        매 step 후 호출. 모드 전환 조건 체크.
        """
        new_state = state.copy()

        if state.mode == MODE_HOVER and not self._transition_done:
            # 고도 상승만 진행 (단순 vertical climb 모델)
            # Phase 1에서는 quick_start_cruise를 권장하므로 이 경로는 부수적
            if state.pos[2] >= self.h_transition:
                # 천이!
                new_state.mode = MODE_CRUISE
                new_state.v = self.v_cruise
                self._transition_done = True

        elif state.mode == MODE_CRUISE and last_wp_reached:
            new_state.mode = MODE_HOVER
            new_state.v = 0.0

        return new_state
