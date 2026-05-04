"""
L1 Guidance (Nonlinear Guidance Law).

Park, Deyst, How (2004) 기반. 경로 위의 lookahead point를 향하는 횡방향 가속도를
계산하고, NED 속도 세트포인트로 변환한다.

vtol_sim의 NLGLController 로직을 pymavlink 의존성 없이 재구현.
"""
from __future__ import annotations
import numpy as np


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class L1Guidance:
    """
    L1 guidance 계산기.

    Parameters
    ----------
    l1_dist : float
        Lookahead 거리 L1 (m). 크면 부드럽고, 작으면 공격적.
    path_pts : np.ndarray, shape (N, 2)
        경로점 2D 배열 [N, E] (NED 기준).
    v_profile : np.ndarray, shape (N,)
        각 경로점의 목표 속도 (m/s).
    gravity : float
        중력 가속도 (m/s²).
    """

    def __init__(self,
                 l1_dist: float,
                 path_pts: np.ndarray,
                 v_profile: np.ndarray,
                 gravity: float = 9.81):
        self._l1 = float(l1_dist)
        self._pts = np.asarray(path_pts, dtype=float)    # (N, 2) 2D
        self._v   = np.asarray(v_profile, dtype=float)
        self._g   = gravity
        self._seg_idx = 0   # 마지막으로 찾은 세그먼트 인덱스 캐시

    # ── 공개 인터페이스 ──────────────────────────────────────

    def compute(self,
                pos_ned: np.ndarray,
                vel_ned: np.ndarray,
                ) -> tuple[float, float, float]:
        """
        현재 위치·속도에서 L1 guidance를 계산한다.

        Parameters
        ----------
        pos_ned : np.ndarray, shape (3,) or (2,)
            현재 위치 [N, E, (h)].
        vel_ned : np.ndarray, shape (3,) or (2,)
            현재 속도 [vN, vE, (vD)].

        Returns
        -------
        (chi_cmd, v_cmd, cross_track_err)
            chi_cmd       : 목표 헤딩 (rad)
            v_cmd         : 목표 속도 (m/s)
            cross_track_err : 횡방향 경로 오차 (m, 좌측 + / 우측 -)
        """
        p2 = np.asarray(pos_ned[:2], dtype=float)
        v2 = np.asarray(vel_ned[:2], dtype=float)

        # 현재 속도 크기
        v_mag = float(np.linalg.norm(v2))
        if v_mag < 0.1:
            v_dir = np.array([1.0, 0.0])
        else:
            v_dir = v2 / v_mag

        # 가장 가까운 세그먼트 탐색
        seg = self._find_segment(p2)
        self._seg_idx = seg

        # lookahead point 계산
        lh_pt, lh_seg = self._lookahead_point(p2, seg)

        # 횡방향 오차 (세그먼트 법선 방향)
        cross_err = self._cross_track_error(p2, seg)

        # L1 벡터: 현재 위치 → lookahead
        vec_to_lh = lh_pt - p2
        dist_to_lh = float(np.linalg.norm(vec_to_lh))
        if dist_to_lh < 1e-3:
            # lookahead가 너무 가까우면 다음 세그먼트 방향 사용
            nxt = min(seg + 1, len(self._pts) - 1)
            vec_to_lh = self._pts[nxt] - p2
            dist_to_lh = float(np.linalg.norm(vec_to_lh))

        if dist_to_lh < 1e-3:
            return 0.0, float(self._v[min(seg, len(self._v) - 1)]), cross_err

        # 각도 에러 η (속도벡터와 lookahead 벡터 사이)
        lh_dir = vec_to_lh / dist_to_lh
        sin_eta = float(np.clip(
            v_dir[0] * lh_dir[1] - v_dir[1] * lh_dir[0],
            -1.0, 1.0
        ))
        cos_eta = float(np.clip(
            v_dir[0] * lh_dir[0] + v_dir[1] * lh_dir[1],
            -1.0, 1.0
        ))
        eta = float(np.arctan2(sin_eta, cos_eta))

        # 횡방향 가속도 명령
        a_lat = 2.0 * max(v_mag, 1.0) ** 2 * np.sin(eta) / self._l1

        # 목표 헤딩: lookahead 방향을 기반으로 a_lat 반영
        chi_lh = float(np.arctan2(lh_dir[1], lh_dir[0]))
        # 현재 헤딩에서 a_lat/v² 만큼 회전
        delta_chi = np.arctan2(a_lat, max(v_mag, 1.0) ** 2 / self._l1)
        chi_cmd = float(_wrap(chi_lh))

        # 속도 명령: lookahead 세그먼트 속도 사용
        v_idx = min(lh_seg, len(self._v) - 1)
        v_cmd = float(self._v[v_idx])

        return chi_cmd, v_cmd, float(cross_err)

    def ned_velocity_cmd(self,
                         pos_ned: np.ndarray,
                         vel_ned: np.ndarray,
                         gamma_ref: float = 0.0,
                         ) -> np.ndarray:
        """
        NED 속도벡터 세트포인트 [vN, vE, vD] 를 반환한다.

        Parameters
        ----------
        pos_ned, vel_ned : np.ndarray
        gamma_ref : float
            현재 경로점의 상승각 (rad). 고도 제어에 사용.

        Returns
        -------
        np.ndarray, shape (3,)  [vN, vE, vD]
        """
        chi_cmd, v_cmd, _ = self.compute(pos_ned, vel_ned)
        vN = v_cmd * np.cos(chi_cmd)
        vE = v_cmd * np.sin(chi_cmd)
        vD = -v_cmd * np.sin(gamma_ref)   # D축은 아래가 양수
        return np.array([vN, vE, vD])

    @property
    def current_segment(self) -> int:
        return self._seg_idx

    # ── 내부 유틸리티 ────────────────────────────────────────

    def _find_segment(self, p2: np.ndarray) -> int:
        """경로에서 가장 가까운 세그먼트 인덱스 반환."""
        N = len(self._pts)
        if N < 2:
            return 0

        best_d2 = np.inf
        best_i  = max(0, self._seg_idx - 2)   # 캐시 기반 로컬 탐색

        # 전체 탐색 (간단 구현; 경로가 매우 길 경우 최적화 가능)
        for i in range(N - 1):
            a = self._pts[i]
            b = self._pts[i + 1]
            ab = b - a
            ab2 = float(ab @ ab)
            if ab2 < 1e-12:
                proj = a
            else:
                t = float(np.clip((p2 - a) @ ab / ab2, 0.0, 1.0))
                proj = a + t * ab
            d2 = float((p2 - proj) @ (p2 - proj))
            if d2 < best_d2:
                best_d2 = d2
                best_i  = i

        return best_i

    def _lookahead_point(self,
                         p2: np.ndarray,
                         start_seg: int,
                         ) -> tuple[np.ndarray, int]:
        """
        start_seg 이후 경로 위에서 p2로부터 L1 거리만큼 떨어진 점을 반환.
        """
        N = len(self._pts)
        remain = self._l1

        for i in range(start_seg, N - 1):
            a = self._pts[i]
            b = self._pts[i + 1]
            seg_len = float(np.linalg.norm(b - a))
            if seg_len < 1e-9:
                continue

            # a에서 p2까지 투영 거리
            ab = b - a
            t = float(np.clip((p2 - a) @ ab / (seg_len ** 2), 0.0, 1.0))
            proj = a + t * ab
            dist_from_proj = float(np.linalg.norm(p2 - proj))

            # 이 세그먼트에서 남은 거리
            along_remain = seg_len * (1.0 - t)

            if along_remain >= remain:
                frac = t + remain / seg_len
                return a + frac * ab, i

            remain -= along_remain

        # 경로 끝 도달
        return self._pts[-1].copy(), N - 2

    def _cross_track_error(self, p2: np.ndarray, seg: int) -> float:
        """횡방향 경로 오차 (m). 좌측 + / 우측 -."""
        N = len(self._pts)
        a = self._pts[seg]
        b = self._pts[min(seg + 1, N - 1)]
        ab = b - a
        ab_len = float(np.linalg.norm(ab))
        if ab_len < 1e-9:
            return float(np.linalg.norm(p2 - a))
        # 부호 있는 횡방향 거리
        normal = np.array([-ab[1], ab[0]]) / ab_len
        return float((p2 - a) @ normal)
