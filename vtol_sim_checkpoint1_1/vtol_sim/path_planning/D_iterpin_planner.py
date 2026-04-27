"""
DIterativePinPlanner v1.0
=========================
가속도 제약 처리가 포함된 반복적 핀 기반 경로 생성기.
- 핀(Pin): 각 WP에 entry/exit 점, 역순 반복으로 방향 수렴
- 경로: 직선 구간 + Quintic Hermite 자유곡선
- 가속도 위반 시: 2D 파라미터 탐색 → 우회 WP 자동 삽입
- NED 좌표계 기반
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import warnings

from .base_planner import BasePlanner, Path, PathPoint


# ============================================================
# 모듈 레벨 유틸리티
# ============================================================
def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n < eps else v / n


def _rot90_ccw(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]])


@dataclass
class Pin:
    wp: np.ndarray
    direction: np.ndarray
    scale: float
    entry: np.ndarray = field(init=False)
    exit: np.ndarray = field(init=False)

    def __post_init__(self):
        self.entry = self.wp - self.direction * self.scale
        self.exit = self.wp + self.direction * self.scale


# ============================================================
# DIterativePinPlanner
# ============================================================
class DIterativePinPlanner(BasePlanner):
    """
    가속도 제약 처리가 포함된 반복적 핀 기반 경로 생성기.

    Parameters
    ----------
    num_iter : int, default 3
        핀 정제 반복 횟수.
    samples_per_segment : int, default 200
        Quintic Hermite 세그먼트 내부 샘플 수.
    end_pad : float, default 10.0
        마지막 WP 이후 직선 연장 길이 (m).
    straight_ratio0 : float, default 0.4
        초기 직선 구간 비율.
    alpha0 : float, default 0.6
        초기 자유곡선 접선 크기 계수.
    alpha_range : tuple, default (0.2, 1.5)
        2D 탐색 시 alpha 범위.
    sr_range : tuple, default (0.05, 0.6)
        2D 탐색 시 straight_ratio 범위.
    accel_tol : float, default 0.7
        가속도 허용 마진 (a_max = a_max_g * g * accel_tol).
    max_detours : int, default 5
        최대 우회 WP 삽입 횟수.
    search_steps : int, default 7
        2D 파라미터 탐색 그리드 크기.
    verbose : bool, default False
        디버그 출력 여부.
    """

    def __init__(self,
                 num_iter: int = 3,
                 samples_per_segment: int = 200,
                 end_pad: float = 10.0,
                 straight_ratio0: float = 0.4,
                 alpha0: float = 0.6,
                 alpha_range: Tuple[float, float] = (0.2, 1.5),
                 sr_range: Tuple[float, float] = (0.05, 0.6),
                 accel_tol: float = 0.7,
                 max_detours: int = 5,
                 search_steps: int = 7,
                 verbose: bool = False):
        self.num_iter = num_iter
        self.samples_per_segment = samples_per_segment
        self.end_pad = end_pad
        self.straight_ratio0 = straight_ratio0
        self.alpha0 = alpha0
        self.alpha_range = alpha_range
        self.sr_range = sr_range
        self.accel_tol = accel_tol
        self.max_detours = max_detours
        self.search_steps = search_steps
        self.verbose = verbose

    # ─── 공개 인터페이스 ────────────────────────────────────
    def plan(self, waypoints_ned, aircraft_params, initial_state=None) -> Path:
        wps = np.asarray(waypoints_ned, dtype=float)
        if wps.ndim != 2 or wps.shape[1] < 2 or wps.shape[0] < 2:
            raise ValueError(f"WP 배열 형식 오류: shape={wps.shape}, (N≥2, ≥2열) 필요")

        v_cruise = aircraft_params.get('v_cruise', 30.0)
        a_max_g = aircraft_params.get('a_max_g', 0.3)

        result = self._plan_2d(wps[:, :2], v_cruise, a_max_g)
        return self._to_path(result, wps, v_cruise)

    # ─── 1. 2D 경로 계획 ────────────────────────────────────
    def _plan_2d(self, wps_2d: np.ndarray, v_cruise: float, a_max_g: float) -> dict:
        """가속도 제약을 만족하는 2D 경로를 탐색·반환."""
        a_max = a_max_g * 9.81 * self.accel_tol
        kappa_max = a_max / (v_cruise ** 2)
        current_wps = wps_2d.copy()
        detour_idxs: List[int] = []
        final_alpha = self.alpha0
        final_sr = self.straight_ratio0
        final_P = final_kappa = final_alat = None
        feasible = False
        kmax_b = 0.0

        for d_iter in range(self.max_detours + 1):
            P, kappa, a_lat, kmax = self._evaluate_path(
                current_wps, self.alpha0, self.straight_ratio0, v_cruise)

            if kmax <= kappa_max:
                final_alpha, final_sr = self.alpha0, self.straight_ratio0
                final_P, final_kappa, final_alat = P, kappa, a_lat
                feasible = True
                break

            result, ok = self._2d_binary_search(current_wps, v_cruise, kappa_max)
            kmax_b, a_b, s_b, P_b, kappa_b, alat_b = result

            if ok:
                final_alpha, final_sr = a_b, s_b
                final_P, final_kappa, final_alat = P_b, kappa_b, alat_b
                feasible = True
                break

            worst = self._find_worst_wp(current_wps, kappa_b, P_b)
            if worst <= 0 or d_iter >= self.max_detours:
                final_alpha, final_sr = a_b, s_b
                final_P, final_kappa, final_alat = P_b, kappa_b, alat_b
                feasible = False
                break

            current_wps = self._insert_detour(current_wps, worst, kappa_max)
            detour_idxs.append(worst)

        if not feasible:
            warnings.warn(
                f"가속도 제약 미충족: max|kappa|={kmax_b:.4f} > kappa_max={kappa_max:.4f}. "
                f"alpha={final_alpha:.3f}, straight_ratio={final_sr:.3f}")

        return dict(path=final_P, kappa=final_kappa, a_lat=final_alat,
                    waypoints=current_wps, detours=detour_idxs,
                    alpha=final_alpha, straight_ratio=final_sr,
                    feasible=feasible, kappa_max=kappa_max)

    # ─── 2. 핀 계산 ─────────────────────────────────────────
    def _build_pins(self, wps_2d: np.ndarray, straight_ratio: float) -> List[Pin]:
        N = len(wps_2d)
        pins: List[Pin] = []
        for i in range(N):
            if i == 0:
                d_out = wps_2d[1] - wps_2d[0]
                d_avg = np.linalg.norm(d_out)
                direction = _unit(d_out)
            elif i == N - 1:
                d_in = wps_2d[-1] - wps_2d[-2]
                d_avg = np.linalg.norm(d_in)
                direction = _unit(d_in)
            else:
                d_in = wps_2d[i] - wps_2d[i - 1]
                d_out = wps_2d[i + 1] - wps_2d[i]
                d_avg = 0.5 * (np.linalg.norm(d_in) + np.linalg.norm(d_out))
                direction = _unit(_unit(d_in) + _unit(d_out))
                if np.linalg.norm(direction) < 1e-9:
                    direction = _unit(_rot90_ccw(_unit(d_in)))
            scale = d_avg * straight_ratio * 0.5
            pins.append(Pin(wp=wps_2d[i].copy(), direction=direction, scale=scale))
        return pins

    def _refine_pins(self, wps_2d: np.ndarray, straight_ratio: float) -> List[Pin]:
        """역순 반복으로 핀 방향을 수렴."""
        pins = self._build_pins(wps_2d, straight_ratio)
        N = len(pins)
        for _ in range(self.num_iter):
            for i in range(N - 2, 0, -1):
                d_in = pins[i].wp - pins[i - 1].exit
                d_out = pins[i + 1].entry - pins[i].wp
                new_dir = _unit(_unit(d_in) + _unit(d_out))
                if np.linalg.norm(new_dir) < 1e-9:
                    new_dir = _unit(_rot90_ccw(_unit(d_in)))
                pins[i].direction = new_dir
                pins[i].entry = pins[i].wp - new_dir * pins[i].scale
                pins[i].exit = pins[i].wp + new_dir * pins[i].scale
        return pins

    # ─── 3. 세그먼트 생성 ──────────────────────────────────
    def _assemble_path_pts(self, pins: List[Pin], alpha: float):
        """핀 목록에서 P, dP, ddP 배열을 조립."""
        N = len(pins)
        P_list, dP_list, ddP_list = [], [], []

        seg = self._line_segment(pins[0].wp, pins[0].exit,
                                 self.samples_per_segment // 4 + 2,
                                 pins[0].direction)
        P_list.append(seg[0]); dP_list.append(seg[1]); ddP_list.append(seg[2])

        for i in range(N - 1):
            Pc, dPc, ddPc = self._free_curve(pins[i], pins[i + 1], alpha)
            P_list.append(Pc); dP_list.append(dPc); ddP_list.append(ddPc)

            seg = self._line_segment(pins[i + 1].entry, pins[i + 1].wp,
                                     self.samples_per_segment // 4 + 2,
                                     pins[i + 1].direction)
            P_list.append(seg[0]); dP_list.append(seg[1]); ddP_list.append(seg[2])

            if i < N - 2:
                seg = self._line_segment(pins[i + 1].wp, pins[i + 1].exit,
                                         self.samples_per_segment // 4 + 2,
                                         pins[i + 1].direction)
                P_list.append(seg[0]); dP_list.append(seg[1]); ddP_list.append(seg[2])

        last = pins[-1]
        end_point = last.wp + last.direction * self.end_pad
        seg = self._line_segment(last.wp, end_point,
                                 self.samples_per_segment // 4 + 2,
                                 last.direction)
        P_list.append(seg[0]); dP_list.append(seg[1]); ddP_list.append(seg[2])

        return np.vstack(P_list), np.vstack(dP_list), np.vstack(ddP_list)

    def _free_curve(self, pin_a: Pin, pin_b: Pin, alpha: float):
        p0, p1 = pin_a.exit, pin_b.entry
        chord = np.linalg.norm(p1 - p0)
        if chord < 1e-9:
            P = np.tile(p0, (self.samples_per_segment, 1))
            dP = np.tile(pin_a.direction, (self.samples_per_segment, 1))
            ddP = np.zeros_like(P)
            return P, dP, ddP
        v0 = pin_a.direction * chord * alpha
        v1 = pin_b.direction * chord * alpha
        return self._quintic_hermite(p0, v0, np.zeros(2),
                                     p1, v1, np.zeros(2),
                                     n=self.samples_per_segment)

    # ─── 4. 경로 평가 ───────────────────────────────────────
    def _evaluate_path(self, wps_2d: np.ndarray, alpha: float,
                       straight_ratio: float, v_cruise: float):
        pins = self._refine_pins(wps_2d, straight_ratio)
        P, dP, ddP = self._assemble_path_pts(pins, alpha)
        kappa = self._signed_curvature(dP, ddP)
        a_lat = (v_cruise ** 2) * np.abs(kappa)
        return P, kappa, a_lat, float(np.max(np.abs(kappa)))

    def _2d_binary_search(self, wps_2d: np.ndarray, v_cruise: float,
                          kappa_max: float):
        """alpha × straight_ratio 그리드를 탐색해 제약 만족 파라미터를 반환."""
        a_lo, a_hi = self.alpha_range
        s_lo, s_hi = self.sr_range
        a_grid = np.linspace(a_lo, a_hi, self.search_steps)
        s_grid = np.linspace(s_hi, s_lo, self.search_steps)
        best = None
        feasible_best = None

        for s in s_grid:
            for a in a_grid:
                P, kappa, a_lat, kmax = self._evaluate_path(
                    wps_2d, alpha=a, straight_ratio=s, v_cruise=v_cruise)
                if kmax <= kappa_max and feasible_best is None:
                    feasible_best = (kmax, a, s, P, kappa, a_lat)
                if best is None or kmax < best[0]:
                    best = (kmax, a, s, P, kappa, a_lat)
            if feasible_best is not None:
                break

        if feasible_best is not None:
            return feasible_best, True
        return best, False

    # ─── 5. 가속도 위반 처리 ────────────────────────────────
    @staticmethod
    def _find_worst_wp(wps_2d: np.ndarray, kappa: np.ndarray,
                       P: np.ndarray) -> int:
        """곡률이 가장 큰 WP 인덱스를 반환 (첫/끝 WP 제외)."""
        N = len(wps_2d)
        if N < 3:
            return -1
        worst_idx, worst_val = -1, -1.0
        for i in range(1, N - 1):
            d_local = 0.25 * min(np.linalg.norm(wps_2d[i] - wps_2d[i - 1]),
                                 np.linalg.norm(wps_2d[i + 1] - wps_2d[i]))
            d2 = np.sum((P - wps_2d[i]) ** 2, axis=1)
            mask = d2 <= d_local ** 2
            if not np.any(mask):
                local_kmax = float(np.max(np.abs(kappa[np.argsort(d2)[:5]])))
            else:
                local_kmax = float(np.max(np.abs(kappa[mask])))
            if local_kmax > worst_val:
                worst_val = local_kmax
                worst_idx = i
        return worst_idx

    @staticmethod
    def _insert_detour(wps_2d: np.ndarray, idx: int, kappa_max: float,
                       margin: float = 1.25) -> np.ndarray:
        """곡률 과도 WP 주변에 우회점을 삽입."""
        R_min = 1.0 / max(kappa_max, 1e-9)
        d_off = R_min * margin
        WP_im1, WP_i, WP_ip1 = wps_2d[idx - 1], wps_2d[idx], wps_2d[idx + 1]
        v_in = _unit(WP_i - WP_im1)
        v_out = _unit(WP_ip1 - WP_i)
        bis = _unit(v_in + v_out)
        if np.linalg.norm(bis) < 1e-9:
            bis = _unit(_rot90_ccw(v_in))
        outward = -bis
        V_left = WP_i + outward * d_off - v_in * d_off
        V_right = WP_i + outward * d_off + v_out * d_off
        return np.vstack([wps_2d[:idx], V_left[None, :],
                          WP_i[None, :], V_right[None, :],
                          wps_2d[idx + 1:]])

    # ─── 6. Path 변환 ───────────────────────────────────────
    def _to_path(self, result: dict, wps_3d: np.ndarray, v_cruise: float) -> Path:
        sampled_2d = result['path']
        kappa_arr = result['kappa']

        seg_d = np.linalg.norm(np.diff(sampled_2d, axis=0), axis=1)
        s_arr = np.concatenate([[0.0], np.cumsum(seg_d)])

        # 고도 보간 (원본 3D WP 기준)
        wp_s = self._compute_wp_arclengths(wps_3d, sampled_2d, s_arr)
        h_arr = np.interp(s_arr, wp_s, wps_3d[:, 2])
        pos_3d = np.column_stack([sampled_2d, h_arr])

        chi_arr = self._compute_heading(sampled_2d, s_arr)

        points = [PathPoint(pos=pos_3d[k].copy(),
                            v_ref=v_cruise,
                            chi_ref=float(chi_arr[k]),
                            gamma_ref=0.0,
                            curvature=float(kappa_arr[k]),
                            s=float(s_arr[k]),
                            wp_index=None)
                  for k in range(len(sampled_2d))]

        # WP 마킹: 각 WP에 가장 가까운 단일 점
        for wi, wp in enumerate(wps_3d):
            d = np.linalg.norm(sampled_2d - wp[:2], axis=1)
            points[int(np.argmin(d))].wp_index = wi

        return Path(points=points,
                    waypoints_ned=np.asarray(wps_3d, dtype=float),
                    total_length=float(s_arr[-1]) if len(s_arr) else 0.0)

    # ─── 보조: 수학 ─────────────────────────────────────────
    @staticmethod
    def _quintic_hermite(p0, v0, a0, p1, v1, a1, n: int = 200):
        t = np.linspace(0.0, 1.0, n)
        t2, t3, t4, t5 = t*t, t**3, t**4, t**5
        H0  =  1 - 10*t3 + 15*t4 -  6*t5
        H1  =  t -  6*t3 +  8*t4 -  3*t5
        H2  =  0.5*t2 - 1.5*t3 + 1.5*t4 - 0.5*t5
        H3  = 10*t3 - 15*t4 +  6*t5
        H4  = - 4*t3 +  7*t4 -  3*t5
        H5  =  0.5*t3 -   t4 + 0.5*t5
        dH0 = -30*t2 + 60*t3 - 30*t4
        dH1 =  1 - 18*t2 + 32*t3 - 15*t4
        dH2 =  t - 4.5*t2 +  6*t3 - 2.5*t4
        dH3 = 30*t2 - 60*t3 + 30*t4
        dH4 = -12*t2 + 28*t3 - 15*t4
        dH5 = 1.5*t2 -  4*t3 + 2.5*t4
        ddH0 = -60*t + 180*t2 - 120*t3
        ddH1 = -36*t +  96*t2 -  60*t3
        ddH2 =   1 -   9*t +  18*t2 -  10*t3
        ddH3 =  60*t - 180*t2 + 120*t3
        ddH4 = -24*t +  84*t2 -  60*t3
        ddH5 =   3*t -  12*t2 +  10*t3
        P   = (H0[:,None]*p0  + H1[:,None]*v0  + H2[:,None]*a0 +
               H3[:,None]*p1  + H4[:,None]*v1  + H5[:,None]*a1)
        dP  = (dH0[:,None]*p0 + dH1[:,None]*v0 + dH2[:,None]*a0 +
               dH3[:,None]*p1 + dH4[:,None]*v1 + dH5[:,None]*a1)
        ddP = (ddH0[:,None]*p0 + ddH1[:,None]*v0 + ddH2[:,None]*a0 +
               ddH3[:,None]*p1 + ddH4[:,None]*v1 + ddH5[:,None]*a1)
        return P, dP, ddP

    @staticmethod
    def _signed_curvature(dP: np.ndarray, ddP: np.ndarray,
                          eps: float = 1e-12) -> np.ndarray:
        num = dP[:, 0]*ddP[:, 1] - dP[:, 1]*ddP[:, 0]
        den = (dP[:, 0]**2 + dP[:, 1]**2)**1.5 + eps
        return num / den

    @staticmethod
    def _line_segment(p0, p1, n: int, direction: np.ndarray):
        n = max(n, 2)
        t = np.linspace(0.0, 1.0, n)
        P = p0[None, :]*(1-t)[:, None] + p1[None, :]*t[:, None]
        chord = np.linalg.norm(p1 - p0)
        dP = np.tile(direction * max(chord, 1e-9), (n, 1))
        ddP = np.zeros_like(P)
        return P, dP, ddP

    @staticmethod
    def _compute_wp_arclengths(wps_3d: np.ndarray, sampled_2d: np.ndarray,
                               s_arr: np.ndarray) -> np.ndarray:
        ws = np.array([s_arr[int(np.argmin(np.linalg.norm(sampled_2d - w[:2], axis=1)))]
                       for w in wps_3d])
        for i in range(1, len(ws)):
            if ws[i] <= ws[i - 1]:
                ws[i] = ws[i - 1] + 1e-6
        return ws

    @staticmethod
    def _compute_heading(pts_2d: np.ndarray, s_arr: np.ndarray) -> np.ndarray:
        # Enforce strictly increasing s to avoid divide-by-zero in np.gradient
        # at segment join points where consecutive path samples coincide.
        s_safe = s_arr.copy().astype(float)
        for i in range(1, len(s_safe)):
            if s_safe[i] <= s_safe[i - 1]:
                s_safe[i] = s_safe[i - 1] + 1e-9
        dN = np.gradient(pts_2d[:, 0], s_safe)
        dE = np.gradient(pts_2d[:, 1], s_safe)
        chi = np.arctan2(dE, dN)
        # Forward-fill any remaining NaN (shouldn't occur after s_safe fix)
        for i in range(len(chi)):
            if not np.isfinite(chi[i]):
                chi[i] = chi[i - 1] if i > 0 else 0.0
        return chi
