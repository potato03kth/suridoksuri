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
    점에서 폴리라인 위 가장 가까운 점 찾기 (벡터화 구현).

    Parameters
    ----------
    point : (D,) array  (D=2 또는 3)
    polyline : (N, D) array — 경로점 시퀀스

    Returns
    -------
    seg_idx : int
        가장 가까운 세그먼트 인덱스
    t : float
        세그먼트 내 매개변수 [0, 1]
    closest : array (D,)
    distance : float
    """
    n = len(polyline)
    if n < 2:
        d = float(np.linalg.norm(point - polyline[0]))
        return 0, 0.0, polyline[0], d

    # 모든 세그먼트 [polyline[i], polyline[i+1]] 동시 처리
    a = polyline[:-1]                    # (N-1, D)
    b = polyline[1:]                     # (N-1, D)
    ab = b - a                           # (N-1, D)
    ap = point - a                       # (N-1, D)
    ab_sq = np.einsum("ij,ij->i", ab, ab)  # (N-1,)
    # 0인 segment 보호
    ab_sq_safe = np.where(ab_sq < 1e-12, 1.0, ab_sq)
    t_raw = np.einsum("ij,ij->i", ap, ab) / ab_sq_safe
    t = np.clip(t_raw, 0.0, 1.0)
    # 0-length segment는 t=0
    t = np.where(ab_sq < 1e-12, 0.0, t)
    # 가장 가까운 점들
    cps = a + t[:, None] * ab            # (N-1, D)
    diffs = cps - point                  # (N-1, D)
    d_sq = np.einsum("ij,ij->i", diffs, diffs)
    seg = int(np.argmin(d_sq))
    return seg, float(t[seg]), cps[seg].copy(), float(np.sqrt(d_sq[seg]))


def closest_point_on_polyline_local(point: np.ndarray, polyline: np.ndarray,
                                    last_seg: int, window: int = 50
                                    ) -> tuple[int, float, np.ndarray, float]:
    """
    국소 검색 — 이전 segment 근처의 ±window 범위만 검사.

    NLGL처럼 매 step 호출되는 경우, 기체 위치가 갑자기 점프하지 않으므로
    이전 검색 결과 근처만 보면 충분히 정확하면서 훨씬 빠르다.
    """
    n = len(polyline)
    lo = max(0, last_seg - window)
    hi = min(n - 1, last_seg + window + 1)
    sub = polyline[lo:hi + 1]
    seg_local, t, cp, d = closest_point_on_polyline(point, sub)
    return lo + seg_local, t, cp, d


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
