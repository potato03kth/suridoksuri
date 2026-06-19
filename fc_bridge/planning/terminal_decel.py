"""
경로 끝점 종단 감속 후처리 헬퍼.

run_planner()의 v_profile은 v_cruise로 고정된다.
이 헬퍼는 경로 끝 decel_dist 구간을 v_terminal로 선형 ramp-down한다.
"""
from __future__ import annotations
import numpy as np


def apply_terminal_decel(
    v_profile: np.ndarray,
    s_arc: np.ndarray,
    v_terminal: float,
    decel_dist: float,
) -> np.ndarray:
    """경로 끝 decel_dist 구간을 v_terminal로 수렴시킨다.

    Parameters
    ----------
    v_profile : np.ndarray, shape (N,)
        각 경로점의 목표 속도 (m/s).
    s_arc : np.ndarray, shape (N,)
        각 경로점의 누적 호길이 (m). 단조 증가해야 한다.
    v_terminal : float
        끝점 도달 목표 속도 (m/s). 스톨 × 1.1 이상 설정 권장.
    decel_dist : float
        감속 시작 거리 (m). 끝점으로부터 이 거리 안쪽에서만 ramp-down.

    Returns
    -------
    np.ndarray, shape (N,)
        후처리된 속도 프로파일. decel_dist 밖 구간은 원본과 동일.
    """
    v_profile = np.asarray(v_profile, dtype=float)
    s_arc = np.asarray(s_arc, dtype=float)

    s_end = s_arc[-1]
    d_remain = s_end - s_arc                         # 끝점까지 남은 거리
    in_zone = d_remain < decel_dist
    frac = np.clip(d_remain / decel_dist, 0.0, 1.0)  # 1(구간 시작)→0(끝점)
    v_ramp = v_terminal + (v_profile - v_terminal) * frac
    return np.where(in_zone, np.minimum(v_profile, v_ramp), v_profile)
