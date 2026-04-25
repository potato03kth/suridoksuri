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

    # === 알고리즘 평가용 ===
    a_total_cmd: list[float] = field(default_factory=list)
    a_total_actual: list[float] = field(default_factory=list)
    a_max_used: list[float] = field(default_factory=list)
    clip_event: list[bool] = field(default_factory=list)
    clip_factor: list[float] = field(default_factory=list)

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
        self.clip_event.append(state.clip_event)
        self.clip_factor.append(state.clip_factor)
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
            "clip_event": np.array(self.clip_event),
            "clip_factor": np.array(self.clip_factor),
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

    Returns
    -------
    dict with keys:
        max_a_total_actual   : 시뮬 중 실제 가속도 최대값 (m/s²)
        max_a_total_cmd      : 명령된 가속도 최대값 (포화 전, 알고리즘 공격성 척도)
        rms_a_total_actual   : 실제 가속도 RMS
        n_clip_events        : 클립 발생 step 수
        clip_time_ratio      : 클립 발생 시간 비율 (0~1)
        mean_clip_factor     : 클립 발생한 step의 평균 clip_factor
                               (1.0에 가까울수록 살짝 넘은 것, 0에 가까울수록 심하게 넘은 것)
        max_a_total_cmd_g    : g 단위
        max_a_body_xy        : 수평면 body 가속도 최대 크기
        a_max_used_min       : a_max_used 최솟값 (시나리오 동적 변경 시 변동)
        a_max_used_max       : a_max_used 최댓값
    """
    arr = log.to_arrays() if isinstance(log, SimLog) else log
    a_actual = arr["a_total_actual"]
    a_cmd = arr["a_total_cmd"]
    clip_events = arr["clip_event"]
    clip_factors = arr["clip_factor"]
    a_body = arr["a_body"]
    a_max_used = arr["a_max_used"]
    g = 9.81

    n_total = len(a_actual)
    n_clip = int(np.sum(clip_events))

    # 클립 발생한 step의 clip_factor만 평균
    if n_clip > 0:
        mean_clip_factor = float(np.mean(clip_factors[clip_events]))
    else:
        mean_clip_factor = 1.0

    # 수평면 body 가속도 크기
    a_body_xy = np.sqrt(a_body[:, 0]**2 + a_body[:, 1]**2)

    return {
        "max_a_total_actual": float(np.max(a_actual)) if n_total else 0.0,
        "max_a_total_actual_g": float(np.max(a_actual)) / g if n_total else 0.0,
        "max_a_total_cmd": float(np.max(a_cmd)) if n_total else 0.0,
        "max_a_total_cmd_g": float(np.max(a_cmd)) / g if n_total else 0.0,
        "rms_a_total_actual": float(np.sqrt(np.mean(a_actual**2))) if n_total else 0.0,
        "n_clip_events": n_clip,
        "clip_time_ratio": (n_clip / n_total) if n_total else 0.0,
        "mean_clip_factor": mean_clip_factor,
        "max_a_body_xy": float(np.max(a_body_xy)) if n_total else 0.0,
        "a_max_used_min": float(np.min(a_max_used)) if n_total else 0.0,
        "a_max_used_max": float(np.max(a_max_used)) if n_total else 0.0,
    }
