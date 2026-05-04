"""
곡률 기반 속도 프로필 계산.

PathPoint.v_ref = const 인 한계를 보완한다.
각 점에서 횡방향 가속도 제약을 만족하는 최대 속도를 계산하고,
역방향 smoothing으로 급감속 구간을 완화한다.
"""
from __future__ import annotations
import numpy as np


def compute_speed_profile(
    curvature: np.ndarray,
    v_cruise: float,
    a_max: float,
    smooth_window: int = 5,
) -> np.ndarray:
    """
    곡률 배열로부터 속도 프로필을 계산한다.

    Parameters
    ----------
    curvature : np.ndarray, shape (N,)
        각 경로점의 곡률 (1/m). 부호 무관, abs 처리.
    v_cruise : float
        순항 속도 (m/s). 속도 상한.
    a_max : float
        횡방향 가속도 상한 (m/s²).
    smooth_window : int
        전방 look-ahead smoothing 윈도우 크기.
        고곡률 이전 구간에서 미리 감속하도록 역방향 최솟값 전파.

    Returns
    -------
    np.ndarray, shape (N,)
        각 경로점의 목표 속도 (m/s).
    """
    kappa = np.abs(np.asarray(curvature, dtype=float))
    # 기본 제약: v = min(v_cruise, sqrt(a_max / |κ|))
    with np.errstate(divide="ignore", invalid="ignore"):
        v_lim = np.where(kappa > 1e-6, np.sqrt(a_max / kappa), v_cruise)
    v = np.minimum(v_cruise, v_lim)

    # 역방향 smoothing: 앞쪽에 저속 구간이 있으면 미리 감속
    # sliding minimum (window=smooth_window) 역방향 전파
    N = len(v)
    v_smooth = v.copy()
    for i in range(N - 2, -1, -1):
        end = min(i + smooth_window, N)
        v_smooth[i] = min(v_smooth[i], np.min(v[i:end]))

    return np.clip(v_smooth, 0.5, v_cruise)   # 최소 0.5 m/s
