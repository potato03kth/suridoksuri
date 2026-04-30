"""
η³-Clothoid Planner v3.2 — WP 통과 보장 + 자기 연속성 동시 만족
================================================================================
설계 문서: eta3_clothoid_planner_v3.md  (변경점은 CHANGES_v3_2.md 참조)

v3.1 → v3.2 핵심 변경:
  • [BUGFIX] 누적 좌표 연결로 인한 잔차 누적 → 후반 WP 빗나감 문제 해결
  • 각 segment 샘플링은 wps_2d[k] 원점 기준으로 복귀 (v3 방식)
  • 단, segment k의 끝점을 wps_2d[k+1]로 affine 보정해 자기 연속성 보장
    - 보정 방식: 끝점 잔차를 segment 길이를 따라 선형 분배
    - NR 잔차가 작으면 보정도 작아 곡선이 거의 영향 없음
    - NR 잔차가 커도 path는 매끈 + 모든 WP 정확 통과
  • 잔차 가중치 균형 개선: mean_chord 대신 헤딩에 별도 가중치 w_head
  • NR 위치 잔차가 큰 경우(특정 임계값 초과) 자동 경고 출력
"""
from __future__ import annotations
import time
import numpy as np
from .base_planner import BasePlanner, Path, PathPoint


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _wrap_arr(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _fresnel_endpoint(theta_i: float, kappa_i: float,
                      kappa_j: float, L: float,
                      n_quad: int = 400) -> np.ndarray:
    if L < 1e-12:
        return np.zeros(2)
    s = np.linspace(0.0, L, n_quad + 1)
    dk = (kappa_j - kappa_i) / L
    th = theta_i + kappa_i * s + 0.5 * dk * s * s
    return np.array([np.trapz(np.cos(th), s), np.trapz(np.sin(th), s)])


def _clothoid_sample(theta_i: float, kappa_i: float,
                     kappa_j: float, L: float,
                     ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return np.column_stack([x, y]), theta_arr, kappa_arr, s


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


def _initial_thetas(wps: np.ndarray,
                    theta0: float, theta_N: float) -> np.ndarray:
    N = len(wps)
    th = np.zeros(N)
    th[0] = theta0
    th[-1] = theta_N
    for i in range(1, N - 1):
        d_in = wps[i] - wps[i - 1]
        d_out = wps[i + 1] - wps[i]
        L_in, L_out = np.linalg.norm(d_in), np.linalg.norm(d_out)
        if L_in < 1e-9 or L_out < 1e-9:
            th[i] = float(np.arctan2(d_out[1], d_out[0]))
            continue
        w_in = 1.0 / np.sqrt(L_in)
        w_out = 1.0 / np.sqrt(L_out)
        bis = w_in * d_in / L_in + w_out * d_out / L_out
        if np.linalg.norm(bis) < 1e-9:
            bis = np.array([-d_in[1] / L_in, d_in[0] / L_in])
        th[i] = float(np.arctan2(bis[1], bis[0]))
    return th


def _insert_wps_if_infeasible(wps: np.ndarray,
                              thetas: np.ndarray,
                              kappa_max: float,
                              max_insert: int = 4
                              ) -> tuple[np.ndarray, np.ndarray, list]:
    wps = np.array(wps, dtype=float)
    thetas = np.array(thetas, dtype=float)
    orig_indices: list = list(range(len(wps)))

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
                orig_indices.insert(i + 1, -1)
                inserted = True
                i += 2
            else:
                i += 1
        if not inserted:
            break

    return wps, thetas, orig_indices


_V_CLIP_LO_DEFAULT = -0.5
_V_CLIP_HI_DEFAULT = 1.2


def _unpack(x: np.ndarray, N: int,
            theta_bc: tuple, kappa_bc: tuple,
            kappa_max: float, chords: np.ndarray,
            v_lo: float, v_hi: float
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
    Ls = np.maximum(chords * np.exp(np.clip(v, v_lo, v_hi)), 1e-3)
    return th, kp, Ls


def _residual(x: np.ndarray,
              wps: np.ndarray,
              theta_bc: tuple, kappa_bc: tuple,
              kappa_max: float,
              chords: np.ndarray,
              w_head: float,
              v_lo: float, v_hi: float,
              n_quad: int = 100) -> np.ndarray:
    N = len(wps)
    n_segs = N - 1
    th, kp, Ls = _unpack(x, N, theta_bc, kappa_bc,
                         kappa_max, chords, v_lo, v_hi)
    F = np.empty(3 * n_segs)
    for k in range(n_segs):
        p = _fresnel_endpoint(th[k], kp[k], kp[k + 1], Ls[k], n_quad)
        target = wps[k + 1] - wps[k]
        F[3 * k] = p[0] - target[0]
        F[3 * k + 1] = p[1] - target[1]
        th_end = th[k] + 0.5 * (kp[k] + kp[k + 1]) * Ls[k]
        F[3 * k + 2] = w_head * _wrap(th_end - th[k + 1])
    return F


def _solve_g2_nr(wps: np.ndarray,
                 kappa_max: float,
                 theta0: float, kappa0: float,
                 theta_N: float, kappa_N: float,
                 th_init_full: np.ndarray | None = None,
                 max_iter: int = 60, tol: float = 1e-5,
                 eps_jac: float = 1e-6,
                 verbose: bool = False
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    N = len(wps)
    n_segs = N - 1
    chords = np.array([np.linalg.norm(wps[k + 1] - wps[k])
                       for k in range(n_segs)])

    if th_init_full is not None and len(th_init_full) == N:
        th_init = np.array(th_init_full, dtype=float)
        th_init[0] = theta0
        th_init[-1] = theta_N
    else:
        th_init = _initial_thetas(wps, theta0, theta_N)

    kp_init = np.zeros(N)
    for i in range(1, N - 1):
        kp_init[i] = _menger_kappa(wps, i, kappa_max)
    kp_init[0], kp_init[-1] = kappa0, kappa_N

    if N <= 2:
        return th_init, kp_init, np.maximum(chords, 1e-3), 0.0, 0.0

    n_inner = N - 2
    mean_chord = float(np.mean(chords))
    theta_bc = (float(theta0), float(theta_N))
    kappa_bc = (float(kappa0), float(kappa_N))
    w_head = float(min(mean_chord, 50.0))

    clip_attempts = [
        (_V_CLIP_LO_DEFAULT, _V_CLIP_HI_DEFAULT),
        (-0.7, 1.8),
        (-1.0, 2.5),
    ]

    best = None

    for attempt_idx, (v_lo, v_hi) in enumerate(clip_attempts):
        x = np.empty(2 * n_inner + n_segs)
        x[0:n_inner] = th_init[1:-1]
        x[n_inner:2 * n_inner] = np.arctanh(
            np.clip(kp_init[1:-1] / kappa_max, -0.9999, 0.9999))
        x[2 * n_inner:] = 0.0

        args = (wps, theta_bc, kappa_bc, kappa_max, chords, w_head, v_lo, v_hi)
        prev_norm = np.inf

        for it in range(max_iter):
            F = _residual(x, *args)
            norm_F = float(np.linalg.norm(F))

            if verbose:
                _, kk, LL = _unpack(x, N, theta_bc, kappa_bc, kappa_max,
                                    chords, v_lo, v_hi)
                pos_max = max(np.max(np.abs(F[0::3])), np.max(np.abs(F[1::3])))
                head_max = np.max(np.abs(F[2::3])) / max(w_head, 1e-9)
                print(f"  [G2-NR a={attempt_idx} clip=({v_lo:.1f},{v_hi:.1f}) it={it:02d}] "
                      f"|F|={norm_F:.3e} pos_max={pos_max:.3e}m "
                      f"head_max={head_max:.3e}rad "
                      f"|κ|/κmax={np.max(np.abs(kk))/kappa_max:.3f} "
                      f"L/chord∈[{np.min(LL/chords):.2f},{np.max(LL/chords):.2f}]")

            if norm_F < tol:
                break

            m_j, n_j = len(F), len(x)
            J = np.zeros((m_j, n_j))
            for j in range(n_j):
                xp = x.copy()
                xp[j] += eps_jac
                J[:, j] = (_residual(xp, *args) - F) / eps_jac

            try:
                lam = 1e-8 * np.trace(J.T @ J) / max(n_j, 1)
                A = J.T @ J + lam * np.eye(n_j)
                b = -J.T @ F
                dx = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                try:
                    dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
                except np.linalg.LinAlgError:
                    break

            c_armijo = 1e-4
            step = 1.0
            for _bt in range(20):
                if np.linalg.norm(_residual(x + step * dx, *args)) \
                        <= (1.0 - c_armijo * step) * norm_F:
                    break
                step *= 0.5
            x += step * dx

            if abs(prev_norm - norm_F) < 1e-10 and norm_F < 10 * tol:
                break
            prev_norm = norm_F

        F_final = _residual(x, *args)
        norm_final = float(np.linalg.norm(F_final))
        if best is None or norm_final < best[0]:
            best = (norm_final, x.copy(), v_lo, v_hi)

        if norm_final < tol:
            break

    norm_best, x_best, v_lo_best, v_hi_best = best
    th, kp, Ls = _unpack(x_best, N, theta_bc, kappa_bc, kappa_max,
                         chords, v_lo_best, v_hi_best)
    kp = np.clip(kp, -kappa_max * 0.98, kappa_max * 0.98)
    kp[0], kp[-1] = kappa_bc

    F_best = _residual(x_best, wps, theta_bc, kappa_bc, kappa_max, chords,
                       w_head, v_lo_best, v_hi_best)
    pos_max_final = float(max(np.max(np.abs(F_best[0::3])),
                              np.max(np.abs(F_best[1::3]))))
    head_max_final = float(np.max(np.abs(F_best[2::3])) / max(w_head, 1e-9))

    return th, kp, Ls, pos_max_final, head_max_final


def _terminal_decay(theta_end: float, kappa_end: float,
                    kappa_max: float, ds: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if abs(kappa_end) < 1e-6:
        return np.zeros((0, 2)), np.array([]), np.array([]), 0.0
    L = abs(kappa_end) / (kappa_max * 0.5)
    pts, th_arr, kp_arr, _ = _clothoid_sample(theta_end, kappa_end, 0.0, L, ds)
    return pts, th_arr, kp_arr, L


class Eta3ClothoidPlannerV3(BasePlanner):
    """η³-Clothoid Planner v3.2."""

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

    def plan(self,
             waypoints_ned: np.ndarray,
             aircraft_params: dict,
             initial_state: dict | None = None) -> Path:
        t0 = time.perf_counter()
        wps = np.asarray(waypoints_ned, dtype=float)
        if len(wps) < 2:
            raise ValueError("waypoints는 최소 2개 필요")

        g = float(aircraft_params.get("gravity", 9.81))
        v_cruise = float(aircraft_params["v_cruise"])
        a_max_g = float(aircraft_params["a_max_g"])
        a_max = a_max_g * g * self.accel_tol
        kappa_max = a_max / (v_cruise ** 2)

        N_original = len(wps)
        wps_2d = wps[:, :2]

        theta0 = np.arctan2(wps[1, 1] - wps[0, 1], wps[1, 0] - wps[0, 0])
        theta_N = np.arctan2(wps[-1, 1] - wps[-2, 1], wps[-1, 0] - wps[-2, 0])
        if initial_state and "initial_heading" in initial_state:
            theta0 = float(initial_state["initial_heading"])
        kappa0 = 0.0
        kappa_N = 0.0

        th_pre = _initial_thetas(wps_2d, theta0, theta_N)
        wps_2d, th_after_insert, orig_indices = _insert_wps_if_infeasible(
            wps_2d, th_pre, kappa_max)
        N = len(wps_2d)
        if self.verbose:
            n_ins = sum(1 for x in orig_indices if x < 0)
            print(f"[Stage 0] WP 삽입 {n_ins}개 (총 {N}개)")

        if self.verbose:
            print("[Stage 1] 단일 통합 G2 NR")
        thetas, kappas, seg_Ls, pos_res, head_res = _solve_g2_nr(
            wps_2d, kappa_max, theta0, kappa0, theta_N, kappa_N,
            th_init_full=th_after_insert,
            max_iter=self.nr_max_iter, tol=self.nr_tol,
            verbose=self.verbose,
        )
        if self.verbose:
            print(
                f"[Stage 1] pos_max={pos_res:.3e}m head_max={head_res:.3e}rad")
        if pos_res > 0.5:
            print(f"[Eta3ClothoidPlannerV3] ⚠ NR 위치 잔차 {pos_res:.3f}m가 큽니다. "
                  f"affine 보정으로 WP 통과는 보장하지만 곡선이 변형될 수 있습니다.")

        all_pts:   list[np.ndarray] = []
        all_kappa: list[np.ndarray] = []
        wp_marks:  dict[int, int] = {}

        if orig_indices[0] >= 0:
            wp_marks[0] = orig_indices[0]

        for k in range(N - 1):
            seg_pts_local, seg_th, seg_kp, seg_s = _clothoid_sample(
                thetas[k], kappas[k], kappas[k + 1], seg_Ls[k], self.ds)

            seg_end_local = seg_pts_local[-1]
            target_end_local = wps_2d[k + 1] - wps_2d[k]
            err = target_end_local - seg_end_local
            L_total = max(seg_s[-1], 1e-9)
            correction = np.outer(seg_s / L_total, err)
            seg_pts_local_corrected = seg_pts_local + correction
            seg_pts_global = seg_pts_local_corrected + wps_2d[k]

            if k < N - 2:
                all_pts.append(seg_pts_global[:-1])
                all_kappa.append(seg_kp[:-1])
            else:
                all_pts.append(seg_pts_global)
                all_kappa.append(seg_kp)

            idx_next_wp = sum(len(p) for p in all_pts) - \
                (1 if k == N - 2 else 0)
            if orig_indices[k + 1] >= 0:
                wp_marks[idx_next_wp] = orig_indices[k + 1]

        th_terminal = thetas[-2] + 0.5 * (kappas[-2] + kappas[-1]) * seg_Ls[-1]

        last_global = all_pts[-1][-1].copy()
        decay_pts, decay_th_arr, decay_kp, _ = _terminal_decay(
            th_terminal, kappas[-1], kappa_max, self.ds)

        if len(decay_pts) > 1:
            decay_pts_global = decay_pts + last_global
            all_pts.append(decay_pts_global[1:])
            all_kappa.append(decay_kp[1:])
            terminal_pos = decay_pts_global[-1]
            terminal_th = float(decay_th_arr[-1])
        else:
            terminal_pos = last_global
            terminal_th = th_terminal

        last_dir = np.array([np.cos(terminal_th), np.sin(terminal_th)])
        n_ext = max(2, int(self.end_extension / self.ds))
        ext_pts = (terminal_pos
                   + last_dir * np.linspace(0.0, self.end_extension, n_ext)[:, None])
        all_pts.append(ext_pts[1:])
        all_kappa.append(np.zeros(len(ext_pts) - 1))

        pts_arr = np.concatenate(all_pts,   axis=0)
        kappa_arr = np.concatenate(all_kappa, axis=0)

        diffs = np.diff(pts_arr, axis=0)
        s_arr = np.concatenate(
            [[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])

        sorted_marks = sorted(
            [(idx, wi) for idx, wi in wp_marks.items() if wi < N_original],
            key=lambda t: t[1])
        if len(sorted_marks) >= 2:
            wp_s_arr = np.array([s_arr[idx] for idx, _ in sorted_marks])
            wp_h_arr = np.array([wps[wi, 2] for _, wi in sorted_marks])
            alt_arr = np.interp(s_arr, wp_s_arr, wp_h_arr)
        else:
            alt_arr = np.full(len(pts_arr), wps[0, 2])

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
                curvature=float(kappa_arr[idx]),
                s=float(s_arr[idx]),
                wp_index=wp_marks.get(idx, None),
            ))

        return Path(
            points=points,
            waypoints_ned=wps,
            total_length=float(s_arr[-1]),
            planning_time=time.perf_counter() - t0,
        )
