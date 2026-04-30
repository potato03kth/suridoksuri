"""
B-Spline (Hermite) 코너 라운딩 기반 경로 생성기 — robust 버전
================================================================

이전 구현의 5점 spline 코너는 splprep의 chord-length 매개변수와
"거리상 가장 가까운 점 자르기"의 조합으로 인해 spline이 anchor를
통과하지 않고 엉뚱한 영역으로 부풀어 오르는 실패를 보였다.

이번 구현은 코너를 **두 개의 Cubic Hermite 조각**으로 명시적으로 구성한다:

  segment1: P_in -> WP   with tangents (d_in, d_mid)
  segment2: WP   -> P_out with tangents (d_mid, d_out)

성질
----
- Hermite는 끝점을 정확히 통과 → P_in, WP, P_out 모두 통과 보장
- 시작/끝 접선이 d_in/d_out과 정렬 → 직선 segment와 G1 연속
- WP에서 접선이 d_mid로 일치 → C1 연속
- splprep 의존 없음 → 매개변수화 문제 없음
"""
from __future__ import annotations

import time
import numpy as np

from .base_planner import BasePlanner, Path, PathPoint


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


def _wrap_pi(a): return (a + np.pi) % (2.0 * np.pi) - np.pi


class BSplinePlanner(BasePlanner):
    """
    Parameters
    ----------
    ds : float                 샘플 간격 (m)
    straight_lead : float      WP 진입/진출 직선 유지 거리 기본값 (m)
    corner_lead_max_ratio : float
        코너 lead가 인접 segment 길이의 이 비율을 넘지 않도록 (기본 0.45)
    accel_tol : float          a_max 대비 허용 마진 (1.02 = 2% 허용)
    max_refine_iter : int      코너별 lead 확장 최대 반복
    lead_growth : float        위반 코너 lead 증가 배수
    tangent_scale : float
        Hermite 접선 크기 = tangent_scale * (조각 길이).
        값이 클수록 접선이 강하게 작용해 직선처럼 길게 뻗고,
        작을수록 코너가 빨리 휘어진다. 기본 1.0.
    """

    def __init__(self,
                 ds: float = 1.0,
                 straight_lead: float = 30.0,
                 corner_lead_max_ratio: float = 0.45,
                 accel_tol: float = 1.02,
                 max_refine_iter: int = 20,
                 lead_growth: float = 1.35,
                 tangent_scale: float = 1.0):
        self.ds = float(ds)
        self.straight_lead = float(straight_lead)
        self.corner_lead_max_ratio = float(corner_lead_max_ratio)
        self.accel_tol = float(accel_tol)
        self.max_refine_iter = int(max_refine_iter)
        self.lead_growth = float(lead_growth)
        self.tangent_scale = float(tangent_scale)

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

        # 각 WP의 lead (in/out 별도)
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

        # 진입/진출 단위벡터 (3D: 고도 변화 포함)
        dirs_in = [None] * N
        dirs_out = [None] * N
        for i in range(N):
            if i > 0:
                dirs_in[i] = _unit(wps[i] - wps[i - 1])
            if i < N - 1:
                dirs_out[i] = _unit(wps[i + 1] - wps[i])

        # P_in[i], P_out[i]
        P_in = [wps[i].copy() if i == 0 else wps[i] - dirs_in[i] * lead_in[i]
                for i in range(N)]
        P_out = [wps[i].copy() if i == N - 1 else wps[i] + dirs_out[i] * lead_out[i]
                 for i in range(N)]

        # 조각 누적
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
            # 코너 (중간 WP)
            if 0 < i < N - 1:
                pts_corner, k_wp = self._build_corner_hermite(
                    P_in[i], wps[i], P_out[i],
                    dirs_in[i], dirs_out[i]
                )
                marks = [None] * len(pts_corner)
                marks[k_wp] = i
                append_pts(pts_corner, marks)

            # 직선 segment i -> i+1
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

        # 누적 호 길이
        seg = np.diff(pts_all, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        s_cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_len = float(s_cum[-1])

        # 균일 샘플 + 마킹 위치 강제 포함
        s_uniform = self._uniform_with_marks(s_cum, marks_all, self.ds)

        x = np.interp(s_uniform, s_cum, pts_all[:, 0])
        y = np.interp(s_uniform, s_cum, pts_all[:, 1])
        z = np.interp(s_uniform, s_cum, pts_all[:, 2])

        # 곡률/방위/상승각: 호 길이 1차/2차 미분
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

        chi = np.unwrap(np.arctan2(dy, dx))
        chi = np.array([_wrap_pi(c) for c in chi])
        gamma = np.arctan2(dz, np.sqrt(dx**2 + dy**2) + 1e-12)

        with np.errstate(divide="ignore"):
            v_max_curve = np.sqrt(
                a_max / np.maximum(np.abs(kappa_signed), 1e-9))
        v_ref = np.clip(np.minimum(v_cruise, v_max_curve), v_min, v_cruise)

        # WP 마킹을 균일 샘플로 매핑
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
    # Hermite 기반 코너: P_in -> WP -> P_out
    # ------------------------------------------------------------------
    def _build_corner_hermite(self,
                              P_in: np.ndarray, WP: np.ndarray, P_out: np.ndarray,
                              d_in: np.ndarray, d_out: np.ndarray
                              ) -> tuple[np.ndarray, int]:
        """
        두 Hermite 조각을 이어 P_in, WP, P_out을 정확히 통과하는 코너 생성.
        - 시작 접선 = d_in, 끝 접선 = d_out (정규화 단위벡터)
        - WP에서의 공통 접선 d_mid = unit(d_in + d_out)
          (d_in과 d_out이 거의 반대면 별도 처리)

        Returns
        -------
        pts : (K, 3) array — P_in부터 P_out까지 균일 호 길이 샘플
        k_wp : int         — pts에서 WP 위치 (조각 1의 끝 = 조각 2의 시작)
        """
        L1 = float(np.linalg.norm(WP - P_in))
        L2 = float(np.linalg.norm(P_out - WP))
        if L1 < 1e-9 or L2 < 1e-9:
            # 퇴화: 단순 직선
            pts = self._sample_line(P_in, P_out)
            k = int(np.argmin(np.linalg.norm(pts - WP, axis=1)))
            return pts, k

        # WP에서의 공통 접선
        s = d_in + d_out
        if np.linalg.norm(s) < 1e-6:
            # 거의 정반대 — d_in에 수직인 평면 내에서 적당한 방향 선택
            # 수평면 우측 perp을 우선 시도
            perp = np.array([-d_in[1], d_in[0], 0.0])
            if np.linalg.norm(perp) < 1e-6:
                perp = np.array([1.0, 0.0, 0.0])
            d_mid = _unit(perp)
        else:
            d_mid = _unit(s)

        # Hermite 접선 크기: 조각 길이에 비례 (Catmull-Rom 스타일)
        # tangent_scale=1.0 이면 표준 Catmull-Rom 강도
        T_in_1 = d_in * (L1 * self.tangent_scale)
        T_out_1 = d_mid * (L1 * self.tangent_scale)
        T_in_2 = d_mid * (L2 * self.tangent_scale)
        T_out_2 = d_out * (L2 * self.tangent_scale)

        # 조각별 dense 샘플 → 호길이 균일 재샘플
        n1 = max(2, int(np.ceil(L1 * 2.0 / self.ds)))   # 여유롭게
        n2 = max(2, int(np.ceil(L2 * 2.0 / self.ds)))
        t1 = np.linspace(0.0, 1.0, n1)
        t2 = np.linspace(0.0, 1.0, n2)

        seg1 = self._hermite(P_in,  WP,    T_in_1, T_out_1, t1)
        seg2 = self._hermite(WP,    P_out, T_in_2, T_out_2, t2)

        # 두 조각 합치기 (WP 중복 제거)
        merged = np.vstack([seg1, seg2[1:]])

        # 호 길이 균일 재샘플 (P_in ~ P_out)
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
        # WP 위치는 조각1의 끝 = s_cum[len(seg1)-1]
        s_wp = s_cum[len(seg1) - 1]
        k_wp = int(np.argmin(np.abs(s_u - s_wp)))
        # WP 정확 통과 보장: 가장 가까운 샘플 위치를 정확히 WP로 덮어쓰기
        out[k_wp] = WP
        return out, k_wp

    @staticmethod
    def _hermite(P0: np.ndarray, P1: np.ndarray,
                 T0: np.ndarray, T1: np.ndarray,
                 t: np.ndarray) -> np.ndarray:
        """
        3차 Hermite spline.
        H(t) = h00*P0 + h10*T0 + h01*P1 + h11*T1
        h00 = 2t^3 - 3t^2 + 1
        h10 = t^3 - 2t^2 + t
        h01 = -2t^3 + 3t^2
        h11 = t^3 - t^2
        """
        t = t.reshape(-1, 1)
        t2 = t * t
        t3 = t2 * t
        h00 = 2*t3 - 3*t2 + 1
        h10 = t3 - 2*t2 + t
        h01 = -2*t3 + 3*t2
        h11 = t3 - t2
        return (h00 * P0.reshape(1, 3) +
                h10 * T0.reshape(1, 3) +
                h01 * P1.reshape(1, 3) +
                h11 * T1.reshape(1, 3))

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
