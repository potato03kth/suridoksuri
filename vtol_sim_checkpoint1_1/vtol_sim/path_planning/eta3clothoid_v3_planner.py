# eta3_clothoid_planner_v3.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _wrap_arr(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _fresnel_endpoint(theta_i: float, kappa_i: float, kappa_j: float,
                      L: float, n_quad: int = 400) -> np.ndarray:
    if L < 1e-12:
        return np.zeros(2)
    s = np.linspace(0.0, L, n_quad + 1)
    dk = (kappa_j - kappa_i) / L
    th = theta_i + kappa_i * s + 0.5 * dk * s * s
    return np.array([np.trapz(np.cos(th), s), np.trapz(np.sin(th), s)])


def _clothoid_sample(theta_i: float, kappa_i: float, kappa_j: float,
                     L: float, ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = max(2, int(np.ceil(L / ds)) + 1)
    s = np.linspace(0.0, L, n)
    dk = (kappa_j - kappa_i) / L if L > 1e-12 else 0.0
    kappa_arr = kappa_i + dk * s
    theta_arr = theta_i + kappa_i * s + 0.5 * dk * s * s
    cos_h, sin_h = np.cos(theta_arr), np.sin(theta_arr)
    x = np.zeros(n)
    y = np.zeros(n)
    for k in range(1, n):
        h = s[k] - s[k - 1]
        x[k] = x[k - 1] + 0.5 * (cos_h[k - 1] + cos_h[k]) * h
        y[k] = y[k - 1] + 0.5 * (sin_h[k - 1] + sin_h[k]) * h
    return np.column_stack([x, y]), theta_arr, kappa_arr


def _menger_kappa(wps: np.ndarray, i: int, kappa_max: float) -> float:
    a = wps[i] - wps[i - 1]
    b = wps[i + 1] - wps[i]
    cross = a[0] * b[1] - a[1] * b[0]
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    chord = np.linalg.norm(wps[i + 1] - wps[i - 1])
    if la * lb * chord < 1e-9:
        return 0.0
    k_abs = abs(cross) / (la * lb * chord)
    return float(np.clip(np.sign(cross) * k_abs, -kappa_max * 0.9, kappa_max * 0.9))


def _initial_thetas(wps: np.ndarray, theta0: float, theta_N: float) -> np.ndarray:
    N = len(wps)
    th = np.zeros(N)
    th[0] = theta0
    th[-1] = theta_N
    for i in range(1, N - 1):
        d_in = wps[i] - wps[i - 1]
        d_out = wps[i + 1] - wps[i]
        L_in, L_out = np.linalg.norm(d_in), np.linalg.norm(d_out)
        if L_in < 1e-9 or L_out < 1e-9:
            th[i] = np.arctan2(d_out[1], d_out[0])
            continue
        w_in = 1.0 / np.sqrt(L_in)
        w_out = 1.0 / np.sqrt(L_out)
        bis = w_in * d_in / L_in + w_out * d_out / L_out
        if np.linalg.norm(bis) < 1e-9:
            bis = np.array([-d_in[1] / L_in, d_in[0] / L_in])
        th[i] = np.arctan2(bis[1], bis[0])
    return th


def _insert_wps_if_infeasible(wps: np.ndarray, thetas: np.ndarray,
                              kappa_max: float, max_insert: int = 4
                              ) -> tuple[np.ndarray, np.ndarray, list]:
    wps = np.array(wps, dtype=float)
    thetas = np.array(thetas, dtype=float)
    orig: list = list(range(len(wps)))

    for _ in range(max_insert):
        inserted = False
        i = 0
        while i < len(wps) - 1:
            chord = np.linalg.norm(wps[i + 1] - wps[i])
            if chord < 1e-6:
                i += 1
                continue
            dtheta = abs(_wrap(thetas[i + 1] - thetas[i]))
            k_need = 2.0 * dtheta / chord
            if k_need > kappa_max * 0.9:
                wp_mid = 0.5 * (wps[i] + wps[i + 1])
                th_mid = float(np.arctan2(
                    np.sin(thetas[i]) + np.sin(thetas[i + 1]),
                    np.cos(thetas[i]) + np.cos(thetas[i + 1])))
                wps = np.insert(wps, i + 1, wp_mid, axis=0)
                thetas = np.insert(thetas, i + 1, th_mid)
                orig.insert(i + 1, -1)
                inserted = True
                i += 2
            else:
                i += 1
        if not inserted:
            break
    return wps, thetas, orig


_V_CLIP_LO = -0.3
_V_CLIP_HI = 0.6


def _unpack(x: np.ndarray, N: int,
            theta_bc: tuple, kappa_bc: tuple,
            kappa_max: float, chords: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_inner = N - 2
    n_segs = N - 1
    th = np.empty(N)
    kp = np.empty(N)
    th[0], th[-1] = theta_bc
    kp[0], kp[-1] = kappa_bc
    if n_inner > 0:
        th[1:-1] = x[0:n_inner]
        kp[1:-1] = kappa_max * np.tanh(x[n_inner:2 * n_inner])
    v = x[2 * n_inner:2 * n_inner + n_segs]
    Ls = np.maximum(chords * np.exp(np.clip(v, _V_CLIP_LO, _V_CLIP_HI)), 1e-3)
    return th, kp, Ls


def _residual(x: np.ndarray, wps: np.ndarray,
              theta_bc: tuple, kappa_bc: tuple,
              kappa_max: float, chords: np.ndarray,
              mean_chord: float, n_quad: int = 100) -> np.ndarray:
    N = len(wps)
    n_segs = N - 1
    th, kp, Ls = _unpack(x, N, theta_bc, kappa_bc, kappa_max, chords)
    F = np.empty(3 * n_segs)
    for k in range(n_segs):
        p = _fresnel_endpoint(th[k], kp[k], kp[k + 1], Ls[k], n_quad)
        target = wps[k + 1] - wps[k]
        F[3 * k] = p[0] - target[0]
        F[3 * k + 1] = p[1] - target[1]
        th_end = th[k] + 0.5 * (kp[k] + kp[k + 1]) * Ls[k]
        F[3 * k + 2] = mean_chord * _wrap(th_end - th[k + 1])
    return F


def _solve_g2_nr(wps: np.ndarray, kappa_max: float,
                 theta0: float, kappa0: float,
                 theta_N: float, kappa_N: float,
                 max_iter: int = 60, tol: float = 1e-5,
                 eps_jac: float = 1e-6, verbose: bool = False
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(wps)
    n_segs = N - 1
    chords = np.array([np.linalg.norm(wps[k + 1] - wps[k])
                      for k in range(n_segs)])

    th_init = _initial_thetas(wps, theta0, theta_N)
    kp_init = np.zeros(N)
    for i in range(1, N - 1):
        kp_init[i] = _menger_kappa(wps, i, kappa_max)
    kp_init[0], kp_init[-1] = kappa0, kappa_N

    if N <= 2:
        return th_init, kp_init, np.maximum(chords, 1e-3)

    n_inner = N - 2
    mean_chord = float(np.mean(chords))
    theta_bc = (float(theta0), float(theta_N))
    kappa_bc = (float(kappa0), float(kappa_N))

    x = np.empty(2 * n_inner + n_segs)
    x[0:n_inner] = th_init[1:-1]
    x[n_inner:2 * n_inner] = np.arctanh(np.clip(
        kp_init[1:-1] / kappa_max, -0.9999, 0.9999))
    x[2 * n_inner:] = 0.0

    args = (wps, theta_bc, kappa_bc, kappa_max, chords, mean_chord)
    prev = np.inf

    for it in range(max_iter):
        F = _residual(x, *args)
        nF = float(np.linalg.norm(F))
        if verbose:
            _, kk, LL = _unpack(x, N, theta_bc, kappa_bc, kappa_max, chords)
            pos_max = max(np.max(np.abs(F[0::3])), np.max(np.abs(F[1::3])))
            head_max = np.max(np.abs(F[2::3])) / max(mean_chord, 1e-9)
            print(f"  [G2-NR it={it:02d}] |F|={nF:.3e}  pos_max={pos_max:.3e}m  "
                  f"head_max={head_max:.3e}rad  |κ|/κmax={np.max(np.abs(kk))/kappa_max:.3f}  "
                  f"L/chord∈[{np.min(LL/chords):.2f},{np.max(LL/chords):.2f}]")
        if nF < tol:
            break

        m, n = len(F), len(x)
        J = np.zeros((m, n))
        for j in range(n):
            xp = x.copy()
            xp[j] += eps_jac
            J[:, j] = (_residual(xp, *args) - F) / eps_jac

        try:
            dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
        except np.linalg.LinAlgError:
            break

        c1 = 1e-4
        step = 1.0
        ok = False
        for _ in range(8):
            Ftry = _residual(x + step * dx, *args)
            if np.linalg.norm(Ftry) <= (1.0 - c1 * step) * nF:
                ok = True
                break
            step *= 0.5
        x += step * dx

        if abs(prev - nF) < 1e-10 and nF < 10 * tol:
            break
        prev = nF

    th, kp, Ls = _unpack(x, N, theta_bc, kappa_bc, kappa_max, chords)
    kp = np.clip(kp, -kappa_max * 0.98, kappa_max * 0.98)
    kp[0], kp[-1] = kappa_bc
    return th, kp, Ls


def _terminal_decay(theta_end: float, kappa_end: float,
                    kappa_max: float, ds: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if abs(kappa_end) < 1e-6:
        return np.zeros((0, 2)), np.array([]), np.array([]), 0.0
    L = abs(kappa_end) / (kappa_max * 0.5)
    pts, th_arr, kp_arr = _clothoid_sample(theta_end, kappa_end, 0.0, L, ds)
    return pts, th_arr, kp_arr, L


class PathPoint:
    def __init__(self, pos, v_ref, chi_ref, gamma_ref, curvature, s, wp_index=None):
        self.pos = pos
        self.v_ref = v_ref
        self.chi_ref = chi_ref
        self.gamma_ref = gamma_ref
        self.curvature = curvature
        self.s = s
        self.wp_index = wp_index


class Path:
    def __init__(self, points, waypoints_ned, total_length, planning_time):
        self.points = points
        self.waypoints_ned = waypoints_ned
        self.total_length = total_length
        self.planning_time = planning_time


class BasePlanner:
    pass


class Eta3ClothoidPlannerV3(BasePlanner):
    def __init__(self,
                 ds: float = 1.0,
                 accel_tol: float = 0.9,
                 nr_tol: float = 1e-5,
                 nr_max_iter: int = 60,
                 end_extension: float = 15.0,
                 verbose: bool = False):
        self.ds = ds
        self.accel_tol = accel_tol
        self.nr_tol = nr_tol
        self.nr_max_iter = nr_max_iter
        self.end_extension = end_extension
        self.verbose = verbose

    def plan(self, waypoints_ned: np.ndarray, aircraft_params: dict,
             initial_state: dict | None = None) -> Path:
        t0 = time.perf_counter()
        wps_full = np.asarray(waypoints_ned, dtype=float)
        if len(wps_full) < 2:
            raise ValueError("waypoints는 최소 2개 필요")

        g = float(aircraft_params.get("gravity", 9.81))
        v_cruise = float(aircraft_params["v_cruise"])
        a_max_g = float(aircraft_params["a_max_g"])
        a_max = a_max_g * g * self.accel_tol
        kappa_max = a_max / (v_cruise ** 2)

        N0 = len(wps_full)
        wps_2d = wps_full[:, :2]

        theta0 = np.arctan2(wps_full[1, 1] - wps_full[0, 1],
                            wps_full[1, 0] - wps_full[0, 0])
        theta_N = np.arctan2(wps_full[-1, 1] - wps_full[-2, 1],
                             wps_full[-1, 0] - wps_full[-2, 0])
        if initial_state and "initial_heading" in initial_state:
            theta0 = float(initial_state["initial_heading"])
        kappa0 = 0.0
        kappa_N = 0.0

        th_pre = _initial_thetas(wps_2d, theta0, theta_N)
        wps_2d, _, orig_idx = _insert_wps_if_infeasible(
            wps_2d, th_pre, kappa_max)
        N = len(wps_2d)

        thetas, kappas, seg_Ls = _solve_g2_nr(
            wps_2d, kappa_max, theta0, kappa0, theta_N, kappa_N,
            max_iter=self.nr_max_iter, tol=self.nr_tol,
            verbose=self.verbose)

        all_pts: List[np.ndarray] = []
        all_kp: List[np.ndarray] = []
        wp_marks: dict[int, int] = {}

        for k in range(N - 1):
            seg_pts, _, seg_kp = _clothoid_sample(
                thetas[k], kappas[k], kappas[k + 1], seg_Ls[k], self.ds)
            seg_pts = seg_pts + wps_2d[k]
            if orig_idx[k] >= 0:
                wp_marks[sum(len(p) for p in all_pts)] = orig_idx[k]
            all_pts.append(seg_pts[:-1])
            all_kp.append(seg_kp[:-1])

        idx_last = sum(len(p) for p in all_pts)
        if orig_idx[N - 1] >= 0:
            wp_marks[idx_last] = orig_idx[N - 1]

        th_terminal = thetas[-2] + 0.5 * (kappas[-2] + kappas[-1]) * seg_Ls[-1]
        decay_pts, _, decay_kp, _ = _terminal_decay(
            th_terminal, kappas[-1], kappa_max, self.ds)

        all_pts.append(wps_2d[N - 1: N])
        all_kp.append(np.array([kappas[-1]]))

        if len(decay_pts) > 1:
            decay_pts = decay_pts + wps_2d[-1]
            all_pts.append(decay_pts[1:])
            all_kp.append(decay_kp[1:])
            terminal_pos = decay_pts[-1]
            terminal_th = th_terminal
        else:
            terminal_pos = wps_2d[-1]
            terminal_th = th_terminal

        last_dir = np.array([np.cos(terminal_th), np.sin(terminal_th)])
        n_ext = max(2, int(self.end_extension / self.ds))
        ext_pts = (terminal_pos + last_dir
                   * np.linspace(0.0, self.end_extension, n_ext)[:, None])
        all_pts.append(ext_pts[1:])
        all_kp.append(np.zeros(len(ext_pts) - 1))

        pts_arr = np.concatenate(all_pts, axis=0)
        kp_arr = np.concatenate(all_kp, axis=0)

        diffs = np.diff(pts_arr, axis=0)
        s_arr = np.concatenate([[0.0],
                                np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])

        sorted_marks = sorted(
            [(idx, wi) for idx, wi in wp_marks.items() if wi < N0],
            key=lambda t: t[1])
        wp_s = np.array([s_arr[idx] for idx, _ in sorted_marks]
                        ) if sorted_marks else np.array([0.0])
        wp_h = np.array([wps_full[wi, 2] for _, wi in sorted_marks]
                        ) if sorted_marks else np.array([wps_full[0, 2]])
        alt_arr = np.interp(s_arr, wp_s, wp_h)

        chi_arr = np.zeros(len(pts_arr))
        for i in range(len(pts_arr) - 1):
            d = pts_arr[i + 1] - pts_arr[i]
            chi_arr[i] = np.arctan2(d[1], d[0])
        chi_arr[-1] = chi_arr[-2]

        gamma_arr = np.zeros(len(pts_arr))
        for i in range(len(pts_arr) - 1):
            dh = alt_arr[i + 1] - alt_arr[i]
            ds_i = s_arr[i + 1] - s_arr[i]
            gamma_arr[i] = np.arctan2(dh, ds_i) if ds_i > 1e-9 else 0.0
        gamma_arr[-1] = gamma_arr[-2]

        points: list[PathPoint] = []
        for idx in range(len(pts_arr)):
            points.append(PathPoint(
                pos=np.array([pts_arr[idx, 0], pts_arr[idx, 1], alt_arr[idx]]),
                v_ref=v_cruise,
                chi_ref=float(chi_arr[idx]),
                gamma_ref=float(gamma_arr[idx]),
                curvature=float(kp_arr[idx]),
                s=float(s_arr[idx]),
                wp_index=wp_marks.get(idx, None),
            ))

        return Path(
            points=points,
            waypoints_ned=wps_full,
            total_length=float(s_arr[-1]),
            planning_time=time.perf_counter() - t0,
        )
