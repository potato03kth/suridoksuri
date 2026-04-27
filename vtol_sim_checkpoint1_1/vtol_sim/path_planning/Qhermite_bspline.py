"""
B-Spline (Quintic Hermite) 코너 라운딩 — G² 연속 버전
=====================================================

이전 Cubic Hermite 두 조각 구현은 G¹(접선 방향) 연속이지만
WP에서 2계 미분(곡률)이 점프하여 a_lat = v²·κ에 4~5g 스파이크가
발생했다. 또한 호 길이 매개변수 미분이 불연속이라 곡률 자체가
WP에서 디락 델타에 가까운 형태로 튀었다.

이번 구현은 코너를 두 개의 **5차 Hermite (Quintic Hermite)** 조각으로
만들고, 각 조각의 끝점에서 위치 / 1계 / 2계 미분을 모두 명시한다.

  segment1 : P_in -> WP    boundaries
      pos:   P_in,   WP
      vel:   d_in*L1, d_mid*L1     (Catmull-Rom 강도)
      acc:   0,      0             (곡률 0 보장)
  segment2 : WP -> P_out   boundaries
      pos:   WP,     P_out
      vel:   d_mid*L2, d_out*L2
      acc:   0,      0

성질
----
- P_in, WP, P_out 정확 통과 (Hermite 끝점 보간 성질)
- P_in/P_out/WP 모두에서 곡률 = 0   ⇒ G² 연속 + WP 곡률 점프 제거
- 직선 segment(곡률 0)와 매끄럽게 연결
- splprep / 매개변수화 의존 없음
"""
from __future__ import annotations

import time
import numpy as np

from .base_planner import BasePlanner, Path, PathPoint


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class BSplinePlanner(BasePlanner):
    """
    Parameters
    ----------
    ds : float                   샘플 간격 (m)
    straight_lead : float        WP 진입/진출 직선 유지 거리 기본값 (m)
    corner_lead_max_ratio : float
        코너 lead가 인접 segment 길이의 이 비율을 넘지 않도록 (기본 0.45)
    accel_tol : float            a_max 대비 허용 마진 (1.02 = 2% 허용)
    max_refine_iter : int        코너별 lead 확장 최대 반복
    lead_growth : float          위반 코너 lead 증가 배수
    tangent_scale : float
        Quintic Hermite 접선 크기 = tangent_scale * (조각 길이).
        Catmull-Rom 표준이 1.0. 작게 할수록 코너가 빠르게 휘고,
        크게 할수록 직선처럼 길게 뻗다가 안쪽에서 휜다.
    smooth_window : int          curvature 후처리 이동평균 폭 (홀수, 1=미사용)
    """

    def __init__(self,
                 ds: float = 1.0,
                 straight_lead: float = 30.0,
                 corner_lead_max_ratio: float = 0.45,
                 accel_tol: float = 1.02,
                 max_refine_iter: int = 20,
                 lead_growth: float = 1.35,
                 tangent_scale: float = 1.0,
                 smooth_window: int = 5):
        self.ds = float(ds)
        self.straight_lead = float(straight_lead)
        self.corner_lead_max_ratio = float(corner_lead_max_ratio)
        self.accel_tol = float(accel_tol)
        self.max_refine_iter = int(max_refine_iter)
        self.lead_growth = float(lead_growth)
        self.tangent_scale = float(tangent_scale)
        self.smooth_window = max(1, int(smooth_window) | 1)  # 홀수 강제

    # ------------------------------------------------------------------
    def plan(self, waypoints_ned, aircraft_params, initial_state=None):
        t0 = time.perf_counter()
        wps = np.asarray(waypoints_ned, dtype=float)
        if wps.ndim != 2 or wps.shape[1] != 3:
            raise ValueError("waypoints_ned는 (N, 3)이어야 합니다.")
        N = len(wps)
        if N < 2:
            raise ValueError("최소 2개의 WP가 필요합니다.")

        v_cruise = float(aircraft_params.get("v_cruise", 25.0))
        a_max_g = float(aircraft_params.get("a_max_g", 2.0))
        a_max = a_max_g * 9.80665
        v_min = float(aircraft_params.get("v_min", 0.5 * v_cruise))

        lead_in = np.full(N, self.straight_lead, dtype=float)
        lead_out = np.full(N, self.straight_lead, dtype=float)

        path = None
        meta_iter = 0
        meta_violations = []

        for it in range(self.max_refine_iter + 1):
            meta_iter = it + 1
            li, lo = self._cap_leads(wps, lead_in, lead_out)
            path = self._build_path(wps, li, lo, v_cruise, a_max, v_min)
            violations = self._validate_corners(path, a_max)
            meta_violations = violations
            if not violations or it == self.max_refine_iter:
                break
            for v in violations:
                wp_i = v["wp_index"]
                lead_in[wp_i] *= self.lead_growth
                lead_out[wp_i] *= self.lead_growth

        path.waypoints_ned = wps.copy()
        path.planning_time = time.perf_counter() - t0
        try:
            path.meta = {
                "iterations": meta_iter,
                "lead_in":  lead_in.tolist(),
                "lead_out": lead_out.tolist(),
                "violations_remaining": meta_violations,
            }
        except Exception:
            pass
        return path

    # ------------------------------------------------------------------
    def _cap_leads(self, wps, lead_in, lead_out):
        N = len(wps)
        li = lead_in.copy()
        lo = lead_out.copy()
        for i in range(N - 1):
            seg_len = float(np.linalg.norm(wps[i + 1] - wps[i]))
            cap = self.corner_lead_max_ratio * seg_len
            need = lo[i] + li[i + 1]
            allowed = 2.0 * cap
            if need > allowed:
                scale = allowed / need
                lo[i] *= scale
                li[i + 1] *= scale
            lo[i] = min(lo[i], cap)
            li[i + 1] = min(li[i + 1], cap)
        return li, lo

    # ------------------------------------------------------------------
    def _build_path(self, wps, lead_in, lead_out,
                    v_cruise, a_max, v_min) -> Path:
        N = len(wps)

        dirs_in = [None] * N
        dirs_out = [None] * N
        for i in range(N):
            if i > 0:
                dirs_in[i] = _unit(wps[i] - wps[i - 1])
            if i < N - 1:
                dirs_out[i] = _unit(wps[i + 1] - wps[i])

        P_in = [wps[i].copy() if i == 0 else wps[i] - dirs_in[i] * lead_in[i]
                for i in range(N)]
        P_out = [wps[i].copy() if i == N - 1 else wps[i] + dirs_out[i] * lead_out[i]
                 for i in range(N)]

        pts_all: list[np.ndarray] = []
        marks_all: list[int | None] = []

        def append_pts(pts, marks):
            for k in range(len(pts)):
                if pts_all and np.linalg.norm(pts[k] - pts_all[-1]) < 1e-6:
                    if marks[k] is not None:
                        marks_all[-1] = marks[k]
                    continue
                pts_all.append(pts[k].copy())
                marks_all.append(marks[k])

        for i in range(N):
            if 0 < i < N - 1:
                pts_corner, k_wp = self._build_corner_quintic(
                    P_in[i], wps[i], P_out[i],
                    dirs_in[i], dirs_out[i]
                )
                marks = [None] * len(pts_corner)
                marks[k_wp] = i
                append_pts(pts_corner, marks)

            if i < N - 1:
                A = wps[i] if i == 0 else P_out[i]
                B = wps[i + 1] if i == N - 2 else P_in[i + 1]
                seg_pts = self._sample_line(A, B)
                marks = [None] * len(seg_pts)
                if i == 0:
                    marks[0] = 0
                if i == N - 2:
                    marks[-1] = N - 1
                append_pts(seg_pts, marks)

        pts_all = np.array(pts_all)

        seg = np.diff(pts_all, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        s_cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_len = float(s_cum[-1])

        s_uniform = self._uniform_with_marks(s_cum, marks_all, self.ds)

        x = np.interp(s_uniform, s_cum, pts_all[:, 0])
        y = np.interp(s_uniform, s_cum, pts_all[:, 1])
        z = np.interp(s_uniform, s_cum, pts_all[:, 2])

        # 곡률/방위/상승각
        dx = np.gradient(x, s_uniform)
        dy = np.gradient(y, s_uniform)
        dz = np.gradient(z, s_uniform)
        ddx = np.gradient(dx, s_uniform)
        ddy = np.gradient(dy, s_uniform)
        ddz = np.gradient(dz, s_uniform)
        rp_norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-12
        cross = np.stack([dy*ddz - dz*ddy,
                          dz*ddx - dx*ddz,
                          dx*ddy - dy*ddx], axis=1)
        kappa = np.linalg.norm(cross, axis=1) / (rp_norm**3)
        sign = -np.sign(dx*ddy - dy*ddx)
        sign[sign == 0] = 1.0
        kappa_signed = kappa * sign

        # 후처리: 이산 미분 잡음 제거 (이동평균)
        if self.smooth_window > 1 and len(kappa_signed) > self.smooth_window:
            kappa_signed = self._moving_average(
                kappa_signed, self.smooth_window)

        chi = np.unwrap(np.arctan2(dy, dx))
        chi = np.array([_wrap_pi(c) for c in chi])
        gamma = np.arctan2(dz, np.sqrt(dx**2 + dy**2) + 1e-12)

        with np.errstate(divide="ignore"):
            v_max_curve = np.sqrt(
                a_max / np.maximum(np.abs(kappa_signed), 1e-9))
        v_ref = np.clip(np.minimum(v_cruise, v_max_curve), v_min, v_cruise)

        wp_to_s: dict[int, float] = {}
        for k, m in enumerate(marks_all):
            if m is not None:
                wp_to_s[m] = s_cum[k]
        wp_marks_uni: dict[int, int] = {}
        for wp_i, s_target in wp_to_s.items():
            k_uni = int(np.argmin(np.abs(s_uniform - s_target)))
            wp_marks_uni[k_uni] = wp_i

        points: list[PathPoint] = []
        for k_idx in range(len(s_uniform)):
            points.append(PathPoint(
                pos=np.array([x[k_idx], y[k_idx], z[k_idx]]),
                v_ref=float(v_ref[k_idx]),
                chi_ref=float(chi[k_idx]),
                gamma_ref=float(gamma[k_idx]),
                curvature=float(kappa_signed[k_idx]),
                s=float(s_uniform[k_idx]),
                wp_index=wp_marks_uni.get(k_idx, None),
            ))
        return Path(points=points, waypoints_ned=np.asarray(wps).copy(),
                    total_length=total_len)

    # ------------------------------------------------------------------
    # Quintic Hermite 코너: P_in -> WP -> P_out
    #   양 끝과 WP 모두에서 2계 미분 = 0  (곡률 0 강제)
    # ------------------------------------------------------------------
    def _build_corner_quintic(self,
                              P_in: np.ndarray, WP: np.ndarray, P_out: np.ndarray,
                              d_in: np.ndarray, d_out: np.ndarray
                              ) -> tuple[np.ndarray, int]:
        L1 = float(np.linalg.norm(WP - P_in))
        L2 = float(np.linalg.norm(P_out - WP))
        if L1 < 1e-9 or L2 < 1e-9:
            pts = self._sample_line(P_in, P_out)
            k = int(np.argmin(np.linalg.norm(pts - WP, axis=1)))
            return pts, k

        s = d_in + d_out
        if np.linalg.norm(s) < 1e-6:
            perp = np.array([-d_in[1], d_in[0], 0.0])
            if np.linalg.norm(perp) < 1e-6:
                perp = np.array([1.0, 0.0, 0.0])
            d_mid = _unit(perp)
        else:
            d_mid = _unit(s)

        # 접선 크기는 조각 길이에 비례 (Catmull-Rom 강도)
        T_in_1 = d_in * (L1 * self.tangent_scale)
        T_out_1 = d_mid * (L1 * self.tangent_scale)
        T_in_2 = d_mid * (L2 * self.tangent_scale)
        T_out_2 = d_out * (L2 * self.tangent_scale)

        # 2계 미분(가속도)은 0으로 강제 → 끝점 곡률 0
        Z = np.zeros(3)

        n1 = max(8, int(np.ceil(L1 * 2.0 / self.ds)))
        n2 = max(8, int(np.ceil(L2 * 2.0 / self.ds)))
        t1 = np.linspace(0.0, 1.0, n1)
        t2 = np.linspace(0.0, 1.0, n2)

        seg1 = self._quintic_hermite(P_in, WP,    T_in_1, T_out_1, Z, Z, t1)
        seg2 = self._quintic_hermite(WP,   P_out, T_in_2, T_out_2, Z, Z, t2)

        merged = np.vstack([seg1, seg2[1:]])

        sl = np.linalg.norm(np.diff(merged, axis=0), axis=1)
        s_cum = np.concatenate([[0.0], np.cumsum(sl)])
        total = s_cum[-1]
        if total < 1e-9:
            return P_in.reshape(1, 3), 0

        n_out = max(3, int(np.ceil(total / self.ds)) + 1)
        s_u = np.linspace(0.0, total, n_out)
        out = np.stack([np.interp(s_u, s_cum, merged[:, 0]),
                        np.interp(s_u, s_cum, merged[:, 1]),
                        np.interp(s_u, s_cum, merged[:, 2])], axis=1)
        s_wp = s_cum[len(seg1) - 1]
        k_wp = int(np.argmin(np.abs(s_u - s_wp)))
        out[k_wp] = WP   # 정확 통과 보장
        return out, k_wp

    @staticmethod
    def _quintic_hermite(P0: np.ndarray, P1: np.ndarray,
                         V0: np.ndarray, V1: np.ndarray,
                         A0: np.ndarray, A1: np.ndarray,
                         t: np.ndarray) -> np.ndarray:
        """
        5차 Hermite spline.
        끝점에서 위치(P), 1계 미분(V), 2계 미분(A) 모두 만족.

        H(t) = h0(t)P0 + h1(t)V0 + h2(t)A0/2 + h3(t)A1/2 + h4(t)V1 + h5(t)P1
        (표준 형태 정리; t∈[0,1])

        basis (t∈[0,1]):
          h0 = 1 - 10 t^3 + 15 t^4 - 6 t^5
          h1 = t - 6 t^3 + 8 t^4 - 3 t^5
          h2 = (1/2) t^2 - (3/2) t^3 + (3/2) t^4 - (1/2) t^5
          h3 = (1/2) t^3 - t^4 + (1/2) t^5
          h4 = -4 t^3 + 7 t^4 - 3 t^5
          h5 = 10 t^3 - 15 t^4 + 6 t^5
        """
        t = t.reshape(-1, 1)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t

        h0 = 1.0 - 10.0*t3 + 15.0*t4 - 6.0*t5
        h1 = t - 6.0*t3 + 8.0*t4 - 3.0*t5
        h2 = 0.5*t2 - 1.5*t3 + 1.5*t4 - 0.5*t5
        h3 = 0.5*t3 - t4 + 0.5*t5
        h4 = -4.0*t3 + 7.0*t4 - 3.0*t5
        h5 = 10.0*t3 - 15.0*t4 + 6.0*t5

        return (h0 * P0.reshape(1, 3) +
                h1 * V0.reshape(1, 3) +
                h2 * A0.reshape(1, 3) +
                h3 * A1.reshape(1, 3) +
                h4 * V1.reshape(1, 3) +
                h5 * P1.reshape(1, 3))

    # ------------------------------------------------------------------
    def _sample_line(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        L = float(np.linalg.norm(B - A))
        if L < 1e-9:
            return A.reshape(1, 3).copy()
        n = max(2, int(np.ceil(L / self.ds)) + 1)
        ts = np.linspace(0.0, 1.0, n).reshape(-1, 1)
        return A.reshape(1, 3) * (1 - ts) + B.reshape(1, 3) * ts

    @staticmethod
    def _uniform_with_marks(s_cum, marks, ds) -> np.ndarray:
        total = s_cum[-1]
        if total < 1e-9:
            return np.array([0.0])
        n = max(2, int(np.ceil(total / ds)) + 1)
        s_u = np.linspace(0.0, total, n)
        forced = [s_cum[k] for k, m in enumerate(marks) if m is not None]
        if forced:
            s_all = np.unique(np.concatenate([s_u, np.array(forced)]))
        else:
            s_all = s_u
        return s_all

    @staticmethod
    def _moving_average(arr: np.ndarray, w: int) -> np.ndarray:
        """홀수 폭 w의 대칭 이동평균 (양 끝은 reflect padding)."""
        if w <= 1 or len(arr) < w:
            return arr
        pad = w // 2
        a_pad = np.concatenate([arr[pad:0:-1], arr, arr[-2:-pad-2:-1]])
        kernel = np.ones(w) / w
        return np.convolve(a_pad, kernel, mode="valid")

    # ------------------------------------------------------------------
    def _validate_corners(self, path: Path, a_max: float) -> list[dict]:
        if len(path.points) < 3:
            return []
        kappa = np.array([p.curvature for p in path.points])
        v = np.array([p.v_ref for p in path.points])
        a_lat = (v ** 2) * np.abs(kappa)

        wp_idx_map: dict[int, int] = {}
        for k, p in enumerate(path.points):
            if p.wp_index is not None:
                wp_idx_map[p.wp_index] = k
        sorted_wps = sorted(wp_idx_map.keys())

        violations = []
        for j_pos, wp_i in enumerate(sorted_wps):
            if j_pos == 0 or j_pos == len(sorted_wps) - 1:
                continue
            k_wp = wp_idx_map[wp_i]
            k_prev = wp_idx_map[sorted_wps[j_pos - 1]]
            k_next = wp_idx_map[sorted_wps[j_pos + 1]]
            k_lo = (k_prev + k_wp) // 2
            k_hi = (k_wp + k_next) // 2
            if k_hi <= k_lo:
                continue
            seg_a = a_lat[k_lo:k_hi + 1]
            a_peak = float(np.max(seg_a))
            if a_peak > a_max * self.accel_tol:
                k_peak_local = int(np.argmax(seg_a))
                violations.append({
                    "wp_index": wp_i,
                    "k_peak": k_lo + k_peak_local,
                    "a_lat_peak": a_peak,
                    "a_max": a_max,
                })
        return violations
