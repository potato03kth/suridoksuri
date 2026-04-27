from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _rot90_ccw(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]])


def _quintic_hermite(p0, v0, a0, p1, v1, a1, n: int = 200):
    t = np.linspace(0.0, 1.0, n)
    t2 = t*t
    t3 = t2*t
    t4 = t3*t
    t5 = t4*t
    H0 = 1 - 10*t3 + 15*t4 - 6*t5
    H1 = t - 6*t3 + 8*t4 - 3*t5
    H2 = 0.5*t2 - 1.5*t3 + 1.5*t4 - 0.5*t5
    H3 = 10*t3 - 15*t4 + 6*t5
    H4 = -  4*t3 + 7*t4 - 3*t5
    H5 = 0.5*t3 - t4 + 0.5*t5
    dH0 = -30*t2 + 60*t3 - 30*t4
    dH1 = 1 - 18*t2 + 32*t3 - 15*t4
    dH2 = t - 4.5*t2 + 6*t3 - 2.5*t4
    dH3 = 30*t2 - 60*t3 + 30*t4
    dH4 = -12*t2 + 28*t3 - 15*t4
    dH5 = 1.5*t2 - 4*t3 + 2.5*t4
    ddH0 = -60*t + 180*t2 - 120*t3
    ddH1 = -36*t + 96*t2 - 60*t3
    ddH2 = 1 - 9*t + 18*t2 - 10*t3
    ddH3 = 60*t - 180*t2 + 120*t3
    ddH4 = -24*t + 84*t2 - 60*t3
    ddH5 = 3*t - 12*t2 + 10*t3
    P = (H0[:, None]*p0 + H1[:, None]*v0 + H2[:, None]*a0 +
         H3[:, None]*p1 + H4[:, None]*v1 + H5[:, None]*a1)
    dP = (dH0[:, None]*p0 + dH1[:, None]*v0 + dH2[:, None]*a0 +
          dH3[:, None]*p1 + dH4[:, None]*v1 + dH5[:, None]*a1)
    ddP = (ddH0[:, None]*p0 + ddH1[:, None]*v0 + ddH2[:, None]*a0 +
           ddH3[:, None]*p1 + ddH4[:, None]*v1 + ddH5[:, None]*a1)
    return P, dP, ddP


def _signed_curvature(dP: np.ndarray, ddP: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    num = dP[:, 0]*ddP[:, 1] - dP[:, 1]*ddP[:, 0]
    den = (dP[:, 0]**2 + dP[:, 1]**2)**1.5 + eps
    return num/den


@dataclass
class Pin:
    wp: np.ndarray
    direction: np.ndarray
    scale: float
    entry: np.ndarray = field(init=False)
    exit:  np.ndarray = field(init=False)

    def __post_init__(self):
        self.entry = self.wp - self.direction * self.scale
        self.exit = self.wp + self.direction * self.scale


def _build_pins(WPs: np.ndarray, straight_ratio: float) -> List[Pin]:
    N = len(WPs)
    pins: List[Pin] = []
    for i in range(N):
        if i == 0:
            d_out = WPs[1] - WPs[0]
            d_avg = np.linalg.norm(d_out)
            direction = _unit(d_out)
        elif i == N - 1:
            d_in = WPs[-1] - WPs[-2]
            d_avg = np.linalg.norm(d_in)
            direction = _unit(d_in)
        else:
            d_in = WPs[i] - WPs[i-1]
            d_out = WPs[i+1] - WPs[i]
            d_avg = 0.5*(np.linalg.norm(d_in) + np.linalg.norm(d_out))
            direction = _unit(_unit(d_in) + _unit(d_out))
            if np.linalg.norm(direction) < 1e-9:
                direction = _unit(_rot90_ccw(_unit(d_in)))
        scale = d_avg * straight_ratio * 0.5
        pins.append(Pin(wp=WPs[i].copy(), direction=direction, scale=scale))
    return pins


def _refine_pins_iterative(WPs: np.ndarray, straight_ratio: float, num_iter: int) -> List[Pin]:
    pins = _build_pins(WPs, straight_ratio)
    N = len(pins)
    for _ in range(num_iter):
        for i in range(N - 2, 0, -1):
            d_in = pins[i].wp - pins[i-1].exit
            d_out = pins[i+1].entry - pins[i].wp
            new_dir = _unit(_unit(d_in) + _unit(d_out))
            if np.linalg.norm(new_dir) < 1e-9:
                new_dir = _unit(_rot90_ccw(_unit(d_in)))
            pins[i].direction = new_dir
            pins[i].entry = pins[i].wp - new_dir * pins[i].scale
            pins[i].exit = pins[i].wp + new_dir * pins[i].scale
    return pins


def _free_curve(pin_a: Pin, pin_b: Pin, alpha: float, n_samples: int):
    p0 = pin_a.exit
    p1 = pin_b.entry
    chord = np.linalg.norm(p1 - p0)
    if chord < 1e-9:
        P = np.tile(p0, (n_samples, 1))
        dP = np.tile(pin_a.direction, (n_samples, 1))
        ddP = np.zeros_like(P)
        return P, dP, ddP
    v0 = pin_a.direction * chord * alpha
    v1 = pin_b.direction * chord * alpha
    a0 = np.zeros(2)
    a1 = np.zeros(2)
    return _quintic_hermite(p0, v0, a0, p1, v1, a1, n=n_samples)


def _line_segment(p0, p1, n, direction):
    n = max(n, 2)
    t = np.linspace(0.0, 1.0, n)
    P = p0[None, :]*(1-t)[:, None] + p1[None, :]*t[:, None]
    chord = np.linalg.norm(p1 - p0)
    dP = np.tile(direction * max(chord, 1e-9), (n, 1))
    ddP = np.zeros_like(P)
    return P, dP, ddP


def _assemble_path(pins: List[Pin], alpha: float, samples_per_segment: int, end_pad: float):
    N = len(pins)
    P_list, dP_list, ddP_list = [], [], []
    seg = _line_segment(pins[0].wp, pins[0].exit,
                        samples_per_segment // 4 + 2, pins[0].direction)
    P_list.append(seg[0])
    dP_list.append(seg[1])
    ddP_list.append(seg[2])
    for i in range(N - 1):
        Pc, dPc, ddPc = _free_curve(
            pins[i], pins[i+1], alpha, samples_per_segment)
        P_list.append(Pc)
        dP_list.append(dPc)
        ddP_list.append(ddPc)
        seg = _line_segment(pins[i+1].entry, pins[i+1].wp,
                            samples_per_segment // 4 + 2, pins[i+1].direction)
        P_list.append(seg[0])
        dP_list.append(seg[1])
        ddP_list.append(seg[2])
        if i < N - 2:
            seg = _line_segment(
                pins[i+1].wp, pins[i+1].exit, samples_per_segment // 4 + 2, pins[i+1].direction)
            P_list.append(seg[0])
            dP_list.append(seg[1])
            ddP_list.append(seg[2])
    last = pins[-1]
    end_point = last.wp + last.direction * end_pad
    seg = _line_segment(last.wp, end_point,
                        samples_per_segment // 4 + 2, last.direction)
    P_list.append(seg[0])
    dP_list.append(seg[1])
    ddP_list.append(seg[2])
    return np.vstack(P_list), np.vstack(dP_list), np.vstack(ddP_list)


def _evaluate_path(WPs, alpha, straight_ratio, num_iter, samples_per_segment, end_pad, v_cruise):
    pins = _refine_pins_iterative(WPs, straight_ratio, num_iter)
    P, dP, ddP = _assemble_path(pins, alpha, samples_per_segment, end_pad)
    kappa = _signed_curvature(dP, ddP)
    a_lat = (v_cruise**2) * np.abs(kappa)
    return P, kappa, a_lat, float(np.max(np.abs(kappa)))


def _2d_binary_search(WPs, v_cruise, kappa_max, alpha_range, sr_range, num_iter, samples_per_segment, end_pad, max_steps=7, verbose=False):
    a_lo, a_hi = alpha_range
    s_lo, s_hi = sr_range
    a_grid = np.linspace(a_lo, a_hi, max_steps)
    s_grid = np.linspace(s_hi, s_lo, max_steps)
    best = None
    feasible_best = None
    for s in s_grid:
        for a in a_grid:
            P, kappa, a_lat, kmax = _evaluate_path(
                WPs, alpha=a, straight_ratio=s, num_iter=num_iter, samples_per_segment=samples_per_segment, end_pad=end_pad, v_cruise=v_cruise)
            if kmax <= kappa_max and feasible_best is None:
                feasible_best = (kmax, a, s, P, kappa, a_lat)
            if best is None or kmax < best[0]:
                best = (kmax, a, s, P, kappa, a_lat)
        if feasible_best is not None:
            break
    if feasible_best is not None:
        return feasible_best, True
    return best, False


def _find_worst_wp(WPs, kappa, P) -> int:
    N = len(WPs)
    if N < 3:
        return -1
    worst_idx = -1
    worst_val = -1.0
    for i in range(1, N - 1):
        d_local = 0.25 * \
            min(np.linalg.norm(WPs[i] - WPs[i-1]),
                np.linalg.norm(WPs[i+1] - WPs[i]))
        d2 = np.sum((P - WPs[i])**2, axis=1)
        mask = d2 <= d_local**2
        if not np.any(mask):
            nearest = np.argsort(d2)[:5]
            local_kmax = float(np.max(np.abs(kappa[nearest])))
        else:
            local_kmax = float(np.max(np.abs(kappa[mask])))
        if local_kmax > worst_val:
            worst_val = local_kmax
            worst_idx = i
    return worst_idx


def _insert_detour(WPs: np.ndarray, idx: int, kappa_max: float, margin: float = 1.25) -> np.ndarray:
    R_min = 1.0 / max(kappa_max, 1e-9)
    d_off = R_min * margin
    WP_im1 = WPs[idx - 1]
    WP_i = WPs[idx]
    WP_ip1 = WPs[idx + 1]
    v_in = _unit(WP_i - WP_im1)
    v_out = _unit(WP_ip1 - WP_i)
    bis = _unit(v_in + v_out)
    if np.linalg.norm(bis) < 1e-9:
        bis = _unit(_rot90_ccw(v_in))
    outward = -bis
    V_left = WP_i + outward*d_off - v_in * d_off
    V_right = WP_i + outward*d_off + v_out * d_off
    return np.vstack([WPs[:idx], V_left[None, :], WP_i[None, :], V_right[None, :], WPs[idx+1:]])


def plan_path(waypoints, v_cruise: float, a_max_g: float, accel_tol: float = 0.7, num_iter: int = 3, samples_per_segment: int = 200, end_pad: float = 10.0, straight_ratio0: float = 0.4, alpha0: float = 0.6, alpha_range: Tuple[float, float] = (0.2, 1.5), sr_range: Tuple[float, float] = (0.05, 0.6), max_detours: int = 5, search_steps: int = 7, verbose: bool = False) -> Dict[str, Any]:
    WPs = np.asarray(waypoints, dtype=float)
    if WPs.ndim != 2 or WPs.shape[1] != 2 or WPs.shape[0] < 2:
        raise ValueError("waypoints must be (N,2) array with N>=2.")
    a_max = a_max_g * 9.81 * accel_tol
    kappa_max = a_max / (v_cruise**2)
    detour_idxs: List[int] = []
    current_WPs = WPs.copy()
    final_alpha = alpha0
    final_sr = straight_ratio0
    final_P = final_kappa = final_alat = None
    feasible = False
    for d_iter in range(max_detours + 1):
        P, kappa, a_lat, kmax = _evaluate_path(current_WPs, alpha=alpha0, straight_ratio=straight_ratio0,
                                               num_iter=num_iter, samples_per_segment=samples_per_segment, end_pad=end_pad, v_cruise=v_cruise)
        if kmax <= kappa_max:
            final_alpha, final_sr = alpha0, straight_ratio0
            final_P, final_kappa, final_alat = P, kappa, a_lat
            feasible = True
            break
        result, ok = _2d_binary_search(current_WPs, v_cruise=v_cruise, kappa_max=kappa_max, alpha_range=alpha_range, sr_range=sr_range,
                                       num_iter=num_iter, samples_per_segment=samples_per_segment, end_pad=end_pad, max_steps=search_steps, verbose=verbose)
        kmax_b, a_b, s_b, P_b, kappa_b, alat_b = result
        if ok:
            final_alpha, final_sr = a_b, s_b
            final_P, final_kappa, final_alat = P_b, kappa_b, alat_b
            feasible = True
            break
        worst = _find_worst_wp(current_WPs, kappa_b, P_b)
        if worst <= 0 or d_iter >= max_detours:
            final_alpha, final_sr = a_b, s_b
            final_P, final_kappa, final_alat = P_b, kappa_b, alat_b
            feasible = False
            break
        current_WPs = _insert_detour(
            current_WPs, worst, kappa_max, margin=1.25)
        detour_idxs.append(worst)
    return dict(path=final_P, kappa=final_kappa, a_lat=final_alat, waypoints=current_WPs, detours=detour_idxs, alpha=final_alpha, straight_ratio=final_sr, feasible=feasible, kappa_max=kappa_max, a_max=a_max)
