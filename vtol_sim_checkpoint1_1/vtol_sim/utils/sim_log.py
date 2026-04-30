"""
시뮬레이션 로그 데이터 구조
==============================

각 시뮬레이션 step마다 시계열로 기록되는 데이터의 컨테이너.
가속도 기록(클립 이벤트, a_total 등)을 포함하여 알고리즘 평가에 활용.

이 모듈은 Phase 2의 metrics.py가 사용할 헬퍼 함수도 포함.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from dynamics.base_dynamics import AircraftState, ControlInput


@dataclass
class SimLog:
    """
    시뮬레이션 시계열 로그.

    리스트 기반으로 단순하게 — 종료 시 numpy 배열로 변환 가능.
    """
    t: list[float] = field(default_factory=list)            # s
    pos: list[np.ndarray] = field(default_factory=list)     # [x_N, x_E, h]
    v: list[float] = field(default_factory=list)            # m/s
    chi: list[float] = field(default_factory=list)          # rad
    gamma: list[float] = field(default_factory=list)        # rad
    phi: list[float] = field(default_factory=list)          # rad
    mode: list[str] = field(default_factory=list)

    # body-frame 가속도 (시뮬레이션 측정 가속도)
    a_body: list[np.ndarray] = field(default_factory=list)  # [a_x, a_y, a_z]

    # === 알고리즘 평가용 (가속도) ===
    # 정책: 동역학은 가속도 한계를 강제하지 않고 위반을 raw 데이터로 기록.
    a_total_cmd: list[float] = field(default_factory=list)
    a_total_actual: list[float] = field(default_factory=list)
    a_max_used: list[float] = field(default_factory=list)
    accel_violation: list[bool] = field(default_factory=list)
    accel_violation_amount: list[float] = field(default_factory=list)

    # 제어 입력
    bank_cmd: list[float] = field(default_factory=list)
    pitch_cmd: list[float] = field(default_factory=list)
    thrust_cmd: list[float] = field(default_factory=list)

    # 추정 상태 (위치만 — α-β 필터 출력)
    pos_est: list[np.ndarray] = field(default_factory=list)

    # 외란
    wind: list[np.ndarray] = field(default_factory=list)

    # 실연산시간 측정 (하이브리드 모드)
    compute_time_control: list[float] = field(default_factory=list)  # s, wall-clock
    compute_time_planning: float = 0.0  # 단발성, 초기 1회

    def append_step(self, state: AircraftState, u: ControlInput,
                    pos_est: np.ndarray | None = None,
                    wind: np.ndarray | None = None,
                    compute_time_ctrl: float = 0.0) -> None:
        self.t.append(state.t)
        self.pos.append(state.pos.copy())
        self.v.append(state.v)
        self.chi.append(state.chi)
        self.gamma.append(state.gamma)
        self.phi.append(state.phi)
        self.mode.append(state.mode)
        self.a_body.append(state.a_body.copy())
        self.a_total_cmd.append(state.a_total_cmd)
        self.a_total_actual.append(state.a_total_actual)
        self.a_max_used.append(state.a_max_used)
        self.accel_violation.append(state.accel_violation)
        self.accel_violation_amount.append(state.accel_violation_amount)
        self.bank_cmd.append(u.bank_cmd)
        self.pitch_cmd.append(u.pitch_cmd)
        self.thrust_cmd.append(u.thrust_cmd)
        self.pos_est.append(
            pos_est.copy() if pos_est is not None else state.pos.copy()
        )
        self.wind.append(
            wind.copy() if wind is not None else np.zeros(3)
        )
        self.compute_time_control.append(compute_time_ctrl)

    def to_arrays(self) -> dict:
        """모든 필드를 numpy 배열로 변환하여 dict 반환 (분석 편의)."""
        return {
            "t": np.array(self.t),
            "pos": np.array(self.pos),
            "v": np.array(self.v),
            "chi": np.array(self.chi),
            "gamma": np.array(self.gamma),
            "phi": np.array(self.phi),
            "mode": np.array(self.mode),
            "a_body": np.array(self.a_body),
            "a_total_cmd": np.array(self.a_total_cmd),
            "a_total_actual": np.array(self.a_total_actual),
            "a_max_used": np.array(self.a_max_used),
            "accel_violation": np.array(self.accel_violation),
            "accel_violation_amount": np.array(self.accel_violation_amount),
            "bank_cmd": np.array(self.bank_cmd),
            "pitch_cmd": np.array(self.pitch_cmd),
            "thrust_cmd": np.array(self.thrust_cmd),
            "pos_est": np.array(self.pos_est),
            "wind": np.array(self.wind),
            "compute_time_control": np.array(self.compute_time_control),
        }


# =============================================================================
# 가속도 통계 헬퍼 (metrics.py에서 import 예정)
# =============================================================================
def compute_acceleration_metrics(log: SimLog) -> dict:
    """
    로그에서 가속도 관련 통계를 계산.

    정책: 동역학이 가속도 한계를 강제하지 않으므로, 알고리즘이 한계를 초과한
    명령을 내면 그대로 적용되어 violation으로 기록된다. 이 함수는 그 violation
    들을 종합하여 알고리즘 품질의 raw 척도를 제공한다.

    Returns
    -------
    dict with keys:
        max_a_total_actual / _g  : 시뮬 중 도달한 가속도 최대값
        rms_a_total_actual       : 실제 가속도 RMS
        n_violations             : 한계 초과한 step 수
        violation_time_ratio     : 위반 시간 비율 (0~1)
        max_violation_amount     : 최대 초과량 (m/s²) — 가장 무리한 명령
        max_violation_amount_g   : g 단위
        mean_violation_amount    : 위반 step의 평균 초과량
        a_max_used_min/_max      : 가속도 한계 (시나리오 동적 변경 추적)
        max_a_body_xy            : 수평면 body 가속도 최대 크기
    """
    arr = log.to_arrays() if isinstance(log, SimLog) else log
    a_actual = arr["a_total_actual"]
    a_cmd = arr["a_total_cmd"]
    violations = arr["accel_violation"]
    violation_amounts = arr["accel_violation_amount"]
    a_body = arr["a_body"]
    a_max_used = arr["a_max_used"]
    g = 9.81

    n_total = len(a_actual)
    n_violations = int(np.sum(violations))

    # 위반 step의 평균 초과량
    if n_violations > 0:
        viol_amts_only = violation_amounts[violations]
        mean_viol_amount = float(np.mean(viol_amts_only))
        max_viol_amount = float(np.max(viol_amts_only))
    else:
        mean_viol_amount = 0.0
        max_viol_amount = 0.0

    # 수평면 body 가속도 크기
    a_body_xy = np.sqrt(a_body[:, 0]**2 + a_body[:, 1]**2)

    return {
        "max_a_total_actual": float(np.max(a_actual)) if n_total else 0.0,
        "max_a_total_actual_g": float(np.max(a_actual)) / g if n_total else 0.0,
        "max_a_total_cmd": float(np.max(a_cmd)) if n_total else 0.0,
        "max_a_total_cmd_g": float(np.max(a_cmd)) / g if n_total else 0.0,
        "rms_a_total_actual": float(np.sqrt(np.mean(a_actual**2))) if n_total else 0.0,
        "n_violations": n_violations,
        "violation_time_ratio": (n_violations / n_total) if n_total else 0.0,
        "max_violation_amount": max_viol_amount,
        "max_violation_amount_g": max_viol_amount / g,
        "mean_violation_amount": mean_viol_amount,
        "max_a_body_xy": float(np.max(a_body_xy)) if n_total else 0.0,
        "a_max_used_min": float(np.min(a_max_used)) if n_total else 0.0,
        "a_max_used_max": float(np.max(a_max_used)) if n_total else 0.0,
    }
