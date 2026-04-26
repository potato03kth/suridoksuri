"""
B-Spline 경로 생성기
=====================
설계 원칙 (우선순위 순):
  1. WP 정확 통과        — make_interp_spline 보간 (모든 WP를 키포인트에 포함)
  2. 가속도 제약 준수    — R_min = v²/(a_max_g·g); 급선회 시 반경 R_min 원호 우회
  3. 짧은 경로 (최저)   — 우회로 인한 경로 연장 허용

키포인트 구조 (레그 i = WP_i → WP_{i+1}, 방향 u_i):
  ┌ WP_0
  ├ [A_1]          WP_1 직전, 고도 유지, u_0 방향 직선 접근 (d_straight 거리)
  ├  WP_1
  ├ [arc_pts | D_1] 급선회 → R_min 원호 중간점 / 완만한 선회 → 이탈 가이드
  ├ [A_2]          WP_2 직전 직선 접근
  ├  WP_2
  └  ...

곡률 부호 컨벤션: kappa > 0 → 우선회, kappa = d(chi)/ds
좌표계: NED 기반 [x_N, x_E, h] (h = 고도, 양수 위)
"""
from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import make_interp_spline

from .base_planner import BasePlanner, Path, PathPoint
from utils.math_utils import wrap_angle


class BSplinePlanner(BasePlanner):
    """
    Parameters
    ----------
    ds              : 경로 샘플 간격 (m)
    d_straight      : WP 직전 직선 접근 구간 (m)
    spline_degree   : B-spline 차수 (홀수 권장; 5 → C4 연속)
    wide_turn_deg   : 이 값 이상의 선회각에서 R_min 원호 우회 삽입 (deg)
    n_arc_pts       : 원호 중간 샘플 수 (끝점 포함)
    """

    def __init__(
        self,
        ds: float = 1.0,
        d_straight: float = 30.0,
        spline_degree: int = 5,
        wide_turn_deg: float = 100.0,
        n_arc_pts: int = 7,
    ):
        self.ds = ds
        self.d_straight = d_straight
        self.k = spline_degree
        self.wide_turn_rad = np.radians(wide_turn_deg)
        self.n_arc_pts = n_arc_pts

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    @staticmethod
    def _unit2d(v: np.ndarray) -> np.ndarray:
        n = float(np.hypot(v[0], v[1]))
        return v / max(n, 1e-9)

    def _arc_bypass(
        self,
        wp: np.ndarray,
        u_in2: np.ndarray,
        u_out2: np.ndarray,
        R: float,
    ) -> list[np.ndarray]:
        """
        WP를 지나며 u_in2 → u_out2 방향으로 우회하는 반경 R 원호 점 반환.

        - 원호 시작점은 WP (이미 시퀀스에 포함) → skip
        - 원호 끝점까지 n_arc_pts 점 반환
        - 고도는 WP 고도 유지

        수학:
          좌선회(CCW): 회전 중심 C = WP + R·n_left, n_left = 90° CCW of u_in2
                       시작각 a0 = atan2(u_in2) - π/2
                       끝각   a1 = a0 + theta
          우선회(CW):  회전 중심 C = WP + R·n_right, n_right = 90° CW of u_in2
                       시작각 a0 = atan2(u_in2) + π/2
                       끝각   a1 = a0 - theta
        """
        dot = float(np.clip(np.dot(u_in2, u_out2), -1.0, 1.0))
        theta = float(np.arccos(dot))
        if theta < 1e-6:
            return []

        cross = float(u_in2[0] * u_out2[1] - u_in2[1] * u_out2[0])
        alpha = float(np.arctan2(u_in2[1], u_in2[0]))
        alt = float(wp[2])

        if cross >= 0.0:
            # 좌선회 (반시계, CCW)
            C = wp[:2] + R * np.array([-np.sin(alpha), np.cos(alpha)])
            a0 = alpha - np.pi / 2.0
            a1 = a0 + theta
        else:
            # 우선회 (시계, CW)
            C = wp[:2] + R * np.array([np.sin(alpha), -np.cos(alpha)])
            a0 = alpha + np.pi / 2.0
            a1 = a0 - theta

        # linspace: 시작(WP)은 이미 시퀀스에 있으므로 제외, 끝점 포함
        angles = np.linspace(a0, a1, self.n_arc_pts + 2)[1:]
        return [
            np.array([C[0] + R * np.cos(a), C[1] + R * np.sin(a), alt])
            for a in angles
        ]

    # ------------------------------------------------------------------

    def _build_keypoints(
        self, wps: np.ndarray, R_min: float
    ) -> tuple[np.ndarray, dict[int, int]]:
        """
        B-spline 보간용 키포인트 시퀀스 생성.

        Returns
        -------
        pts    : (M, 3) 키포인트 배열
        wp_map : {원본 WP 인덱스: pts 내 인덱스}
        """
        N = len(wps)
        pts: list[np.ndarray] = []
        wp_map: dict[int, int] = {}

        # 레그별 단위벡터·길이
        u_legs: list[np.ndarray] = []
        leg_lens: list[float] = []
        for i in range(N - 1):
            v = wps[i + 1, :2] - wps[i, :2]
            ln = float(np.hypot(v[0], v[1]))
            u_legs.append(self._unit2d(v))
            leg_lens.append(max(ln, 1e-6))

        # ── WP_0 ──────────────────────────────────────────────────────
        wp_map[0] = len(pts)
        pts.append(wps[0].copy())

        # ── WP_1 ~ WP_{N-1} ──────────────────────────────────────────
        for i in range(1, N):
            u_in = u_legs[i - 1]
            leg_in = leg_lens[i - 1]

            # 직선 접근 거리 (레그 길이 40% 이하로 제한)
            d_app = min(self.d_straight, leg_in * 0.40)

            # 접근점 A_i: WP 직전, 고도 유지, u_in 방향 직선
            A = wps[i].copy()
            A[:2] -= d_app * u_in
            # A[2] = wps[i, 2]  — 이미 같음
            pts.append(A)

            # WP_i 자체
            wp_map[i] = len(pts)
            pts.append(wps[i].copy())

            if i == N - 1:
                break  # 마지막 WP는 이탈 처리 불필요

            u_out = u_legs[i]
            leg_out = leg_lens[i]

            dot_val = float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))
            turn_angle = float(np.arccos(dot_val))

            if turn_angle >= self.wide_turn_rad:
                # 급선회 → R_min 원호 우회점 삽입
                arc_pts = self._arc_bypass(wps[i], u_in, u_out, R_min)
                pts.extend(arc_pts)
            else:
                # 완만한 선회 → 이탈 가이드 점 1개
                # d_dep: 곡률이 R_min 이상이 되도록 충분한 거리 확보
                d_dep = float(np.clip(
                    R_min * np.sin(turn_angle) * 0.6 + 3.0,
                    3.0,
                    leg_out * 0.35,
                ))
                D = wps[i].copy()
                D[:2] += d_dep * u_out
                pts.append(D)

        return np.array(pts, dtype=float), wp_map

    # ------------------------------------------------------------------
    # 공개 인터페이스
    # ------------------------------------------------------------------

    def plan(
        self,
        waypoints_ned: np.ndarray,
        aircraft_params: dict,
        initial_state: dict | None = None,
    ) -> Path:
        t0 = time.perf_counter()

        v_cruise = float(aircraft_params["v_cruise"])
        g = float(aircraft_params.get("gravity", 9.81))
        a_max_g = float(aircraft_params["a_max_g"])

        # 순간 횡방향 가속도 제약 → 최소 회전 반경
        # a_lat = v²/R ≤ a_max_g·g  →  R_min = v²/(a_max_g·g)
        R_min = v_cruise ** 2 / (a_max_g * g)

        wps = np.asarray(waypoints_ned, dtype=float)
        N = len(wps)
        if N < 2:
            raise ValueError("waypoints_ned 는 최소 2개 필요")

        # ── 1. 키포인트 생성 ──────────────────────────────────────────
        key_pts, wp_map = self._build_keypoints(wps, R_min)
        M = len(key_pts)

        # ── 2. Chord-length 매개변수 ──────────────────────────────────
        t_arr = np.zeros(M)
        for j in range(1, M):
            t_arr[j] = t_arr[j - 1] + float(
                np.linalg.norm(key_pts[j] - key_pts[j - 1])
            )
        if t_arr[-1] < 1e-6:
            raise ValueError("WP들이 모두 같은 위치")

        # ── 3. B-spline 보간 ──────────────────────────────────────────
        k = min(self.k, M - 1)
        spl = make_interp_spline(t_arr, key_pts, k=k)

        # ── 4. 호 길이 계산 (고밀도 적분) ────────────────────────────
        n_dense = max(5000, int(t_arr[-1] / 0.05) + 1)
        t_dense = np.linspace(0.0, t_arr[-1], n_dense)
        dp_dense = spl(t_dense, 1)
        speed_dense = np.linalg.norm(dp_dense, axis=1)
        s_dense = np.zeros(n_dense)
        dt_d = np.diff(t_dense)
        s_dense[1:] = np.cumsum(0.5 * (speed_dense[:-1] + speed_dense[1:]) * dt_d)
        total_arc = float(s_dense[-1])

        # ── 5. 균등 호 길이 샘플링 ────────────────────────────────────
        n_pts = max(2, int(np.ceil(total_arc / self.ds)) + 1)
        s_unif = np.linspace(0.0, total_arc, n_pts)
        t_unif = np.interp(s_unif, s_dense, t_dense)

        # ── 6. 경로점 물리량 계산 ─────────────────────────────────────
        pos = spl(t_unif)       # (n_pts, 3)
        d1  = spl(t_unif, 1)   # 1차 도함수
        d2  = spl(t_unif, 2)   # 2차 도함수

        dN,  dE,  dh  = d1[:, 0], d1[:, 1], d1[:, 2]
        d2N, d2E       = d2[:, 0], d2[:, 1]
        horiz = np.sqrt(dN ** 2 + dE ** 2 + 1e-14)

        chi   = np.arctan2(dE, dN)
        kappa = (dN * d2E - dE * d2N) / horiz ** 3   # > 0 → 우선회
        gamma = np.arctan2(dh, horiz)

        # ── 7. WP 인덱스 마킹 ────────────────────────────────────────
        wp_path: dict[int, int] = {}
        for orig_i, ctrl_j in wp_map.items():
            wp_pos = key_pts[ctrl_j]
            dists = np.hypot(pos[:, 0] - wp_pos[0], pos[:, 1] - wp_pos[1])
            best = int(np.argmin(dists))
            if best not in wp_path:
                wp_path[best] = orig_i

        # ── 8. PathPoint 시퀀스 구성 ──────────────────────────────────
        path_points: list[PathPoint] = [
            PathPoint(
                pos=pos[idx],
                v_ref=v_cruise,
                chi_ref=float(wrap_angle(chi[idx])),
                gamma_ref=float(gamma[idx]),
                curvature=float(kappa[idx]),
                s=float(s_unif[idx]),
                wp_index=wp_path.get(idx),
            )
            for idx in range(n_pts)
        ]

        return Path(
            points=path_points,
            waypoints_ned=wps,
            total_length=total_arc,
            planning_time=time.perf_counter() - t0,
        )
