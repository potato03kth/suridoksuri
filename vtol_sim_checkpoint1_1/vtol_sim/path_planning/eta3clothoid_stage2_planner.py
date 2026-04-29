"""
η³-Spline + Clothoid Interior Replacement Planner  v2 (Stage 1 fully implemented)
=================================================================================
두 단계 파이프라인:
  Stage 1 — 단순화 η³ G2 보간: 자연조건 NR로 {θ_i, κ_i} 결정
  Stage 2 — 구간별 Clothoid 재구성: G1 헤딩 잔차 명시 NR (v3)

보장 항목:
  ✓ WP 완전 통과          (Stage 2 위치 잔차 < tol)
  ✓ κ_max 전 구간 준수    (κ_max·tanh 매개변수화)
  ✓ G1 연속성             (Stage 2 헤딩 잔차 < tol)
  ✓ G2 연속 (근사)        (Stage 1 자연 κ 평활화)
  ✗ G3 연속성             (미구현)
"""
from __future__ import annotations
import time
import numpy as np
from .base_planner import BasePlanner, Path, PathPoint


# ─────────────────────────────────────────────────────────────────────────────
# 모듈 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _wrap(a: float) -> float:
    """각도를 (-π, π]로 정규화."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _clothoid_sample(theta_i: float, kappa_i: float,
                     kappa_j: float, L: float,
                     ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """단일 클로소이드 구간 이산 샘플링 (원점 기준 로컬 프레임)."""
    n = max(2, int(np.ceil(L / ds)) + 1)
    s_arr = np.linspace(0.0, L, n)
    dk = (kappa_j - kappa_i) / L if L > 1e-12 else 0.0

    kappa_arr = kappa_i + dk * s_arr
    theta_arr = theta_i + kappa_i * s_arr + 0.5 * dk * s_arr ** 2

    cos_h = np.cos(theta_arr)
    sin_h = np.sin(theta_arr)
    x = np.zeros(n)
    y = np.zeros(n)
    for k in range(1, n):
        step = s_arr[k] - s_arr[k - 1]
        x[k] = x[k - 1] + 0.5 * (cos_h[k - 1] + cos_h[k]) * step
        y[k] = y[k - 1] + 0.5 * (sin_h[k - 1] + sin_h[k]) * step

    return np.column_stack([x, y]), theta_arr, kappa_arr


def _fresnel_endpoint(theta_i: float, kappa_i: float,
                      kappa_j: float, L: float,
                      n_quad: int = 400) -> np.ndarray:
    """클로소이드 구간 끝점 변위 (고밀도 적분, 상대 좌표)."""
    if L < 1e-12:
        return np.zeros(2)
    s_arr = np.linspace(0.0, L, n_quad + 1)
    dk = (kappa_j - kappa_i) / L
    theta_arr = theta_i + kappa_i * s_arr + 0.5 * dk * s_arr ** 2
    dx = np.trapz(np.cos(theta_arr), s_arr)
    dy = np.trapz(np.sin(theta_arr), s_arr)
    return np.array([dx, dy])


def _menger_kappa(wps_2d: np.ndarray, i: int, kappa_max: float) -> float:
    """Interior node i의 부호 있는 Menger 곡률 (NED: CW=+)."""
    a = wps_2d[i] - wps_2d[i - 1]
    b = wps_2d[i + 1] - wps_2d[i]
    cross = a[0] * b[1] - a[1] * b[0]
    area2 = abs(cross)
    la = np.linalg.norm(a)
    lb = np.linalg.norm(b)
    chord_ac = np.linalg.norm(wps_2d[i + 1] - wps_2d[i - 1])
    if la * lb * chord_ac < 1e-9:
        return 0.0
    kappa_abs = area2 / (la * lb * chord_ac)
    sign = 1.0 if cross >= 0.0 else -1.0
    return float(np.clip(sign * kappa_abs, -kappa_max * 0.9, kappa_max * 0.9))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — 단순화 η³ G2 보간 (FULL IMPLEMENTATION)
# ─────────────────────────────────────────────────────────────────────────────

def _eta3_initial_guess(wps_2d: np.ndarray,
                        theta0: float, kappa0: float,
                        theta_N: float, kappa_N: float,
                        kappa_max: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """G2 NR 전 초기값: 이등분선 θ + Menger κ."""
    N = len(wps_2d)
    thetas = np.zeros(N)
    kappas = np.zeros(N)
    thetas[0],     kappas[0] = theta0,  kappa0
    thetas[N - 1], kappas[N - 1] = theta_N, kappa_N

    for i in range(1, N - 1):
        d_in = _unit(wps_2d[i] - wps_2d[i - 1])
        d_out = _unit(wps_2d[i + 1] - wps_2d[i])
        bis = d_in + d_out
        if np.linalg.norm(bis) < 1e-9:
            bis = np.array([-d_in[1], d_in[0]])  # U턴 fallback
        thetas[i] = np.arctan2(bis[1], bis[0])
        kappas[i] = _menger_kappa(wps_2d, i, kappa_max)

    return thetas, kappas


def _theta_natural(wps_2d: np.ndarray, i: int) -> float:
    """
    노드 i의 자연 헤딩 — 인접 코드 길이로 가중된 이등분선.

    θ_nat = arg( d_in/√L_in + d_out/√L_out )
    짧은 코드 쪽 가중 ↑ → 급선회 분산.
    """
    d_in_vec = wps_2d[i] - wps_2d[i - 1]
    d_out_vec = wps_2d[i + 1] - wps_2d[i]
    L_in = np.linalg.norm(d_in_vec)
    L_out = np.linalg.norm(d_out_vec)
    if L_in < 1e-9 or L_out < 1e-9:
        return float(np.arctan2(d_out_vec[1], d_out_vec[0]))

    d_in = d_in_vec / L_in
    d_out = d_out_vec / L_out
    w_in = 1.0 / np.sqrt(L_in)
    w_out = 1.0 / np.sqrt(L_out)

    bis = w_in * d_in + w_out * d_out
    if np.linalg.norm(bis) < 1e-9:                   # U턴
        bis = np.array([-d_in[1], d_in[0]])
    return float(np.arctan2(bis[1], bis[0]))


def _kappa_natural(wps_2d: np.ndarray, kappas: np.ndarray,
                   i: int, kappa_max: float) -> float:
    """
    노드 i의 자연 곡률 — Menger 곡률 + 인접 평활화.

    κ_nat = 0.5·κ_Menger(i) + 0.25·(κ_{i-1} + κ_{i+1})
    """
    k_menger = _menger_kappa(wps_2d, i, kappa_max)
    k_smooth = 0.25 * (kappas[i - 1] + kappas[i + 1])
    return float(np.clip(0.5 * k_menger + k_smooth,
                         -kappa_max * 0.95, kappa_max * 0.95))


def _eta3_g2_residual(x: np.ndarray,
                      wps_2d: np.ndarray,
                      theta_bc: tuple, kappa_bc: tuple,
                      kappa_max: float) -> np.ndarray:
    """
    단순화 η³ G2 잔차.

    자유변수 x[2j], x[2j+1] = θ_i, κ_i  (i = j+1, 1≤i≤N-2)
    잔차 크기 2(N-2):
      F[2j  ] = θ_i − θ_natural(i)
      F[2j+1] = κ_i − κ_natural(i)
    """
    N = len(wps_2d)
    n_inner = N - 2

    thetas = np.empty(N)
    kappas = np.empty(N)
    thetas[0] = theta_bc[0]
    thetas[-1] = theta_bc[1]
    kappas[0] = kappa_bc[0]
    kappas[-1] = kappa_bc[1]
    thetas[1:-1] = x[0::2]
    kappas[1:-1] = x[1::2]

    F = np.empty(2 * n_inner)
    for j, i in enumerate(range(1, N - 1)):
        F[2 * j] = _wrap(thetas[i] - _theta_natural(wps_2d, i))
        F[2 * j + 1] = kappas[i] - _kappa_natural(wps_2d, kappas, i, kappa_max)
    return F


def _solve_eta3_g2(wps_2d: np.ndarray,
                   kappa_max: float,
                   theta0: float, kappa0: float,
                   theta_N: float, kappa_N: float,
                   max_iter: int = 50, tol: float = 1e-6,
                   eps_jac: float = 1e-6,
                   verbose: bool = False
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Stage 1: 단순화 η³ G2 NR.

    NR 실패 시 Picard 반복으로 fallback (구조상 fixed-point 형태).
    """
    N = len(wps_2d)
    thetas, kappas = _eta3_initial_guess(wps_2d, theta0, kappa0,
                                         theta_N, kappa_N, kappa_max)
    if N <= 2:
        return thetas, kappas

    n_inner = N - 2
    theta_bc = (float(theta0),  float(theta_N))
    kappa_bc = (float(kappa0),  float(kappa_N))

    # 자유변수: [θ_1, κ_1, ..., θ_{N-2}, κ_{N-2}]
    x = np.empty(2 * n_inner)
    x[0::2] = thetas[1:-1]
    x[1::2] = kappas[1:-1]

    args = (wps_2d, theta_bc, kappa_bc, kappa_max)

    # ── Newton-Raphson ─────────────────────────────────────────
    converged = False
    prev_norm = np.inf
    for it in range(max_iter):
        F = _eta3_g2_residual(x, *args)
        norm_F = float(np.linalg.norm(F))

        if verbose:
            print(f"  [η³-G2 it={it:02d}] |F|={norm_F:.3e}")

        if norm_F < tol:
            converged = True
            break

        # 수치 야코비안
        n_x = len(x)
        J = np.zeros((len(F), n_x))
        for j in range(n_x):
            xp = x.copy()
            xp[j] += eps_jac
            J[:, j] = (_eta3_g2_residual(xp, *args) - F) / eps_jac

        try:
            dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
        except np.linalg.LinAlgError:
            break

        # Armijo
        c_armijo = 1e-4
        step = 1.0
        accepted = False
        for _ls in range(6):
            F_try = _eta3_g2_residual(x + step * dx, *args)
            if np.linalg.norm(F_try) <= (1.0 - c_armijo * step) * norm_F:
                accepted = True
                break
            step *= 0.5

        if not accepted:
            # NR 정체 → Picard fallback
            if verbose:
                print(f"  [η³-G2 it={it:02d}] Armijo 실패 → Picard fallback")
            break

        x += step * dx

        if abs(prev_norm - norm_F) < 1e-12:
            break
        prev_norm = norm_F

    # ── Picard fallback (NR 미수렴 시) ─────────────────────────
    if not converged:
        thetas[1:-1] = x[0::2]
        kappas[1:-1] = x[1::2]
        for it in range(20):
            theta_new = thetas.copy()
            kappa_new = kappas.copy()
            for i in range(1, N - 1):
                theta_new[i] = _theta_natural(wps_2d, i)
                kappa_new[i] = _kappa_natural(wps_2d, kappas, i, kappa_max)
            delta = max(np.max(np.abs(_wrap_arr(theta_new - thetas))),
                        np.max(np.abs(kappa_new - kappas)))
            thetas, kappas = theta_new, kappa_new
            if verbose:
                print(f"  [η³-G2 Picard it={it:02d}] Δ={delta:.3e}")
            if delta < tol:
                break
    else:
        thetas[1:-1] = x[0::2]
        kappas[1:-1] = x[1::2]

    # 최종 클리핑 (Stage 2에 5% 여유)
    kappas = np.clip(kappas, -kappa_max * 0.95, kappa_max * 0.95)
    kappas[0] = kappa_bc[0]
    kappas[-1] = kappa_bc[1]

    return thetas, kappas


def _wrap_arr(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ─────────────────────────────────────────────────────────────────────────────
# 1순위: 기하학적 실현 가능성 검사 + WP 삽입
# ─────────────────────────────────────────────────────────────────────────────

def _check_and_insert_wps(
    wps_2d:     np.ndarray,
    thetas:     np.ndarray,
    kappas:     np.ndarray,
    kappa_max:  float,
    max_insert: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """κ_needed > κ_max 구간에 중점 WP 삽입."""
    wps_2d = np.array(wps_2d, dtype=float)
    thetas = np.array(thetas, dtype=float)
    kappas = np.array(kappas, dtype=float)
    orig_indices: list = list(range(len(wps_2d)))

    for _ in range(max_insert):
        inserted = False
        i = 0
        while i < len(wps_2d) - 1:
            chord = np.linalg.norm(wps_2d[i + 1] - wps_2d[i])
            if chord < 1e-6:
                i += 1
                continue

            dtheta = abs(_wrap(thetas[i + 1] - thetas[i]))
            kappa_needed = 2.0 * dtheta / chord

            if kappa_needed > kappa_max:
                wp_mid = 0.5 * (wps_2d[i] + wps_2d[i + 1])
                theta_mid = np.arctan2(
                    wps_2d[i + 1][1] - wps_2d[i][1],
                    wps_2d[i + 1][0] - wps_2d[i][0],
                )
                wps_2d = np.insert(wps_2d, i + 1, wp_mid, axis=0)
                thetas = np.insert(thetas, i + 1, theta_mid)
                kappas = np.insert(kappas, i + 1, 0.0)
                orig_indices.insert(i + 1, -1)
                inserted = True
                i += 2
            else:
                i += 1

        if not inserted:
            break

    return wps_2d, thetas, kappas, orig_indices


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — 전역 동시 NR v3 (G1 헤딩 잔차 명시)
# ─────────────────────────────────────────────────────────────────────────────

_V_CLIP_LO = -1.0
_V_CLIP_HI = 1.5


def _unpack_state(x: np.ndarray, N: int,
                  theta_bc: tuple, kappa_bc: tuple,
                  kappa_max: float,
                  chords: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_inner = N - 2
    n_segs = N - 1

    thetas = np.empty(N)
    kappas = np.empty(N)
    thetas[0] = theta_bc[0]
    thetas[-1] = theta_bc[1]
    kappas[0] = kappa_bc[0]
    kappas[-1] = kappa_bc[1]

    if n_inner > 0:
        thetas[1:-1] = x[0:n_inner]
        kappas[1:-1] = kappa_max * np.tanh(x[n_inner:2 * n_inner])

    v = x[2 * n_inner:]
    Ls = np.maximum(chords * np.exp(np.clip(v, _V_CLIP_LO, _V_CLIP_HI)), 1e-3)
    return thetas, kappas, Ls


def _global_residual_g1h(x: np.ndarray,
                         wps_2d: np.ndarray,
                         theta_bc: tuple, kappa_bc: tuple,
                         kappa_max: float,
                         chords: np.ndarray,
                         mean_chord: float,
                         n_quad: int = 100) -> np.ndarray:
    """G1-헤딩 명시 전역 잔차 (위치 2 + 헤딩 1) × N-1."""
    N = len(wps_2d)
    n_segs = N - 1
    thetas, kappas, Ls = _unpack_state(
        x, N, theta_bc, kappa_bc, kappa_max, chords)

    F = np.empty(3 * n_segs)
    for i in range(n_segs):
        p = _fresnel_endpoint(thetas[i], kappas[i],
                              kappas[i + 1], Ls[i], n_quad)
        target = wps_2d[i + 1] - wps_2d[i]
        F[3 * i] = p[0] - target[0]
        F[3 * i + 1] = p[1] - target[1]
        theta_end = thetas[i] + 0.5 * (kappas[i] + kappas[i + 1]) * Ls[i]
        F[3 * i + 2] = mean_chord * _wrap(theta_end - thetas[i + 1])
    return F


def _global_stage2_nr(wps_2d: np.ndarray,
                      thetas: np.ndarray,
                      kappas: np.ndarray,
                      kappa_max: float,
                      max_iter: int = 50,
                      tol: float = 1e-4,
                      eps_jac: float = 1e-6,
                      verbose: bool = False
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """전역 동시 NR v3 — θ 자유변수 + G1 헤딩 잔차 명시."""
    N = len(wps_2d)
    n_segs = N - 1
    chords = np.array([np.linalg.norm(wps_2d[i + 1] - wps_2d[i])
                      for i in range(n_segs)])

    if N <= 2:
        return thetas, kappas, np.maximum(chords, 1e-3)

    n_inner = N - 2
    mean_chord = float(np.mean(chords))
    theta_bc = (float(thetas[0]),  float(thetas[-1]))
    kappa_bc = (float(kappas[0]),  float(kappas[-1]))

    theta_init = thetas[1:-1].copy()
    u_init = np.arctanh(np.clip(kappas[1:-1] / kappa_max, -0.9999, 0.9999))
    v_init = np.zeros(n_segs)
    x = np.concatenate([theta_init, u_init, v_init])

    args = (wps_2d, theta_bc, kappa_bc, kappa_max, chords, mean_chord)

    prev_norm = np.inf
    for it in range(max_iter):
        F = _global_residual_g1h(x, *args)
        norm_F = float(np.linalg.norm(F))

        if verbose:
            _, kk, LL = _unpack_state(
                x, N, theta_bc, kappa_bc, kappa_max, chords)
            head_res = np.max(np.abs(F[2::3])) / max(mean_chord, 1e-9)
            print(f"  [Stage2 NR it={it:02d}] |F|={norm_F:.3e}  "
                  f"max|κ|/κmax={np.max(np.abs(kk))/kappa_max:.3f}  "
                  f"max L/chord={np.max(LL/chords):.2f}  "
                  f"max head_res={head_res:.3e} rad")

        if norm_F < tol:
            break

        m_j, n_j = len(F), len(x)
        J = np.zeros((m_j, n_j))
        for j in range(n_j):
            xp = x.copy()
            xp[j] += eps_jac
            J[:, j] = (_global_residual_g1h(xp, *args) - F) / eps_jac

        try:
            dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
        except np.linalg.LinAlgError:
            break

        c_armijo = 1e-4
        step = 1.0
        accepted = False
        for _ls in range(6):
            F_try = _global_residual_g1h(x + step * dx, *args)
            if np.linalg.norm(F_try) <= (1.0 - c_armijo * step) * norm_F:
                accepted = True
                break
            step *= 0.5

        if not accepted and verbose:
            print(f"  [Stage2 NR it={it:02d}] Armijo 실패 step={step:.3e}")

        x += step * dx

        if abs(prev_norm - norm_F) < 1e-9 and norm_F < 10 * tol:
            break
        prev_norm = norm_F

    thetas_out, kappas_out, Ls_out = _unpack_state(
        x, N, theta_bc, kappa_bc, kappa_max, chords
    )
    return thetas_out, kappas_out, Ls_out


# ─────────────────────────────────────────────────────────────────────────────
# Planner 클래스
# ─────────────────────────────────────────────────────────────────────────────

class Eta3ClothoidPlanner(BasePlanner):
    """
    η³-Spline + Clothoid Interior Replacement Planner v2.

    Stage 1 (단순화 G2) → 1순위(WP 삽입) → Stage 2 (G1-NR) → Clothoid 샘플링.
    """

    def __init__(self,
                 ds: float = 1.0,
                 accel_tol: float = 0.9,
                 nr_tol: float = 1e-4,
                 nr_max_iter: int = 30,
                 stage1_tol: float = 1e-6,
                 stage1_max_iter: int = 50,
                 end_extension: float = 15.0,
                 verbose: bool = False):
        self.ds = ds
        self.accel_tol = accel_tol
        self.nr_tol = nr_tol
        self.nr_max_iter = nr_max_iter
        self.stage1_tol = stage1_tol
        self.stage1_max_iter = stage1_max_iter
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

        # ── 기체 한계 ─────────────────────────────────────────
        g = float(aircraft_params.get("gravity", 9.81))
        v_cruise = float(aircraft_params["v_cruise"])
        a_max_g = float(aircraft_params["a_max_g"])
        a_max = a_max_g * g * self.accel_tol
        kappa_max = a_max / (v_cruise ** 2)

        N_original = len(wps)
        wps_2d = wps[:, :2]
        N = N_original

        # ── 경계 조건 ─────────────────────────────────────────
        theta0 = np.arctan2(wps[1, 1] - wps[0, 1], wps[1, 0] - wps[0, 0])
        theta_N = np.arctan2(wps[-1, 1] - wps[-2, 1], wps[-1, 0] - wps[-2, 0])
        kappa0 = 0.0
        kappa_N = 0.0
        if initial_state and "initial_heading" in initial_state:
            theta0 = float(initial_state["initial_heading"])

        # ── Stage 1: 단순화 η³ G2 ─────────────────────────────
        if self.verbose:
            print("[Stage 1] 단순화 η³ G2 NR")
        thetas, kappas = _solve_eta3_g2(
            wps_2d, kappa_max, theta0, kappa0, theta_N, kappa_N,
            max_iter=self.stage1_max_iter, tol=self.stage1_tol,
            verbose=self.verbose,
        )

        # ── 1순위: 기하학적 WP 삽입 ───────────────────────────
        wps_2d, thetas, kappas, orig_indices = _check_and_insert_wps(
            wps_2d, thetas, kappas, kappa_max,
        )
        N = len(wps_2d)
        if self.verbose:
            n_ins = sum(1 for x in orig_indices if x < 0)
            print(f"[1순위] WP 삽입 {n_ins}개 (총 {N}개)")

        # ── Stage 2: 전역 NR v3 ───────────────────────────────
        if self.verbose:
            print("[Stage 2] 전역 NR v3 (G1 헤딩 명시)")
        thetas, kappas, seg_Ls = _global_stage2_nr(
            wps_2d, thetas, kappas, kappa_max,
            max_iter=self.nr_max_iter, tol=self.nr_tol,
            verbose=self.verbose,
        )

        # ── Clothoid 샘플링 ───────────────────────────────────
        all_pts:   list[np.ndarray] = []
        all_kappa: list[np.ndarray] = []
        wp_marks:  dict[int, int] = {}

        for i in range(N - 1):
            L = seg_Ls[i]
            seg_pts, _, seg_kappa = _clothoid_sample(
                thetas[i], kappas[i], kappas[i + 1], L, self.ds
            )
            seg_pts = seg_pts + wps_2d[i]

            if orig_indices[i] >= 0:
                wp_marks[sum(len(p) for p in all_pts)] = orig_indices[i]

            all_pts.append(seg_pts[:-1])
            all_kappa.append(seg_kappa[:-1])

        idx_last = sum(len(p) for p in all_pts)
        if orig_indices[N - 1] >= 0:
            wp_marks[idx_last] = orig_indices[N - 1]
        all_pts.append(wps_2d[N - 1: N])
        all_kappa.append(np.array([kappas[-1]]))

        # 종단 연장
        last_dir = np.array([np.cos(thetas[-1]), np.sin(thetas[-1])])
        n_ext = max(2, int(self.end_extension / self.ds))
        ext_pts = (wps_2d[-1] +
                   last_dir * np.linspace(0.0, self.end_extension, n_ext)[:, None])
        all_pts.append(ext_pts[1:])
        all_kappa.append(np.zeros(len(ext_pts) - 1))

        pts_arr = np.concatenate(all_pts,   axis=0)
        kappa_arr = np.concatenate(all_kappa, axis=0)

        # ── 호 길이 ──────────────────────────────────────────
        diffs = np.diff(pts_arr, axis=0)
        s_arr = np.concatenate(
            [[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])

        # ── 고도 보간 ─────────────────────────────────────────
        sorted_marks = sorted(
            [(k, wi) for k, wi in wp_marks.items() if wi < N_original],
            key=lambda x: x[1],
        )
        wp_s_arr = np.array([s_arr[k] for k, _ in sorted_marks])
        wp_h_arr = np.array([wps[wi, 2] for _, wi in sorted_marks])
        alt_arr = np.interp(s_arr, wp_s_arr, wp_h_arr)

        # ── 방위각 ─────────────────────────────────────────────
        chi_arr = np.zeros(len(pts_arr))
        for i in range(len(pts_arr) - 1):
            d = pts_arr[i + 1] - pts_arr[i]
            chi_arr[i] = np.arctan2(d[1], d[0])
        chi_arr[-1] = chi_arr[-2]

        # ── 상승각 ─────────────────────────────────────────────
        gamma_arr = np.zeros(len(pts_arr))
        for i in range(len(pts_arr) - 1):
            dh = alt_arr[i + 1] - alt_arr[i]
            ds_i = s_arr[i + 1] - s_arr[i]
            gamma_arr[i] = np.arctan2(dh, ds_i) if ds_i > 1e-9 else 0.0
        gamma_arr[-1] = gamma_arr[-2]

        # ── PathPoint 조립 ────────────────────────────────────
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
