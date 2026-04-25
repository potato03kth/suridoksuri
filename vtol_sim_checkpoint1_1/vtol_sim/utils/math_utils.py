"""
수학 유틸 함수
==============
"""
from __future__ import annotations
import numpy as np


def wrap_angle(angle: float) -> float:
    """각도를 [-π, π] 범위로 wrap."""
    return float(np.mod(angle + np.pi, 2 * np.pi) - np.pi)


def wrap_angle_array(angles: np.ndarray) -> np.ndarray:
    """벡터 버전."""
    return np.mod(angles + np.pi, 2 * np.pi) - np.pi


def saturate(x: float, lo: float, hi: float) -> float:
    """clip + 단일 값."""
    return float(np.clip(x, lo, hi))


def closest_point_on_polyline(point: np.ndarray, polyline: np.ndarray
                              ) -> tuple[int, float, np.ndarray, float]:
    """
    점에서 폴리라인 위 가장 가까운 점 찾기.

    Parameters
    ----------
    point : (3,) 또는 (2,) array
    polyline : (N, 3) 또는 (N, 2) array — 경로점 시퀀스

    Returns
    -------
    seg_idx : int
        가장 가까운 세그먼트 인덱스 (i, i+1 잇는 세그먼트)
    t : float
        세그먼트 내 매개변수 [0, 1]
    closest : array
        가장 가까운 점 (좌표)
    distance : float
    """
    n = len(polyline)
    if n < 2:
        d = float(np.linalg.norm(point - polyline[0]))
        return 0, 0.0, polyline[0], d

    best = (0, 0.0, polyline[0].copy(), np.inf)
    for i in range(n - 1):
        a = polyline[i]
        b = polyline[i + 1]
        ab = b - a
        ap = point - a
        ab_sq = float(np.dot(ab, ab))
        if ab_sq < 1e-12:
            t = 0.0
            cp = a.copy()
        else:
            t = float(np.dot(ap, ab) / ab_sq)
            t = max(0.0, min(1.0, t))
            cp = a + t * ab
        d = float(np.linalg.norm(point - cp))
        if d < best[3]:
            best = (i, t, cp, d)
    return best


def look_ahead_point(polyline: np.ndarray, seg_idx: int, t: float,
                     L: float) -> tuple[np.ndarray, int, float]:
    """
    폴리라인 위에서 (seg_idx, t) 위치로부터 호 길이 L 만큼 전진한 점.

    경로 끝에 도달하면 마지막 점 반환.
    """
    n = len(polyline)
    remaining = L
    cur_seg = seg_idx
    cur_t = t

    while cur_seg < n - 1 and remaining > 0:
        a = polyline[cur_seg]
        b = polyline[cur_seg + 1]
        seg_len = float(np.linalg.norm(b - a))
        seg_remain = seg_len * (1.0 - cur_t)
        if seg_remain >= remaining:
            # 이 세그먼트 안에서 완료
            cur_t += remaining / seg_len
            cur_t = min(1.0, cur_t)
            cp = a + cur_t * (b - a)
            return cp, cur_seg, cur_t
        else:
            # 다음 세그먼트로
            remaining -= seg_remain
            cur_seg += 1
            cur_t = 0.0

    # 끝점 반환
    return polyline[-1].copy(), n - 2, 1.0


def cross_track_error(point: np.ndarray, polyline: np.ndarray
                      ) -> tuple[float, np.ndarray, int]:
    """
    cross-track error (서명 없는 횡방향 오차).
    Returns: (distance, closest_point, segment_index)
    """
    seg, t, cp, d = closest_point_on_polyline(point, polyline)
    return d, cp, seg


def signed_cross_track_error_2d(point: np.ndarray, polyline: np.ndarray
                                ) -> tuple[float, np.ndarray, int]:
    """
    2D 서명 횡방향 오차. 경로 진행 방향 기준 오른쪽이 양(+), 왼쪽이 음(-).

    point, polyline 모두 (..., 2) 또는 (..., 3) — 첫 두 차원만 사용.
    """
    p2 = np.array(point[:2], dtype=float)
    pl2 = np.array(polyline[:, :2], dtype=float)
    seg, t, cp, d = closest_point_on_polyline(p2, pl2)

    # 세그먼트 진행 방향
    a = pl2[seg]
    b = pl2[seg + 1] if seg + 1 < len(pl2) else pl2[seg]
    tangent = b - a
    if np.linalg.norm(tangent) < 1e-12:
        return 0.0, np.array([cp[0], cp[1], 0.0]), seg
    tangent = tangent / np.linalg.norm(tangent)
    # 오른쪽 normal (heading 기준 +90°)
    normal = np.array([tangent[1], -tangent[0]])
    err_vec = p2 - cp
    sign = float(np.sign(np.dot(err_vec, normal)))
    return sign * d, np.array([cp[0], cp[1], 0.0]), seg


def deg2rad(d: float) -> float:
    return float(np.deg2rad(d))


def rad2deg(r: float) -> float:
    return float(np.rad2deg(r))
