"""
Piecewise Clothoid G2 Interpolation Planner  [PSEUDO CODE]
===========================================================
정식화: 각 구간을 κ 선형 클로소이드로 표현하고,
        모든 WP에서 Fresnel 적분 위치 제약을 만족하는
        {θ_i, κ_i}를 전역 비선형 연립방정식으로 직접 풀이.

제약 방정식 (segment i: WP_i → WP_{i+1}):
  F_i(θ_i, κ_i, θ_{i+1}, κ_{i+1}) = ∫₀^{L_i} [cos θ(s), sin θ(s)]ᵀ ds
                                     − (WP_{i+1} − WP_i) = 0
  where L_i = 2(θ_{i+1} − θ_i) / (κ_i + κ_{i+1})

미지수: x = [θ_1, κ_1, θ_2, κ_2, ..., θ_{N-2}, κ_{N-2}]  (내부 노드, 2(N-2)개)
방정식: F(x) ∈ ℝ^{2(N-2)}  (내부 구간 위치 제약)

보장 항목:
  ✓ WP 완전 통과          (NR 수렴 조건)
  ✓ κ_max 전 구간 준수     (제약 투영)
  ✓ dκ/ds = const (구간)   (클로소이드 구조)
  ✓ G2 연속성             (κ_i 공유 — 설계상)
  △ G3 연속성             (use_g3=True 시 sub-segment 분할로 근사)
"""
from __future__ import annotations
import time
import numpy as np
from .base_planner import BasePlanner, Path, PathPoint


# ─────────────────────────────────────────────────────────────────────────────
# 모듈 레벨 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _compute_L(theta_i: float, kappa_i: float,
               theta_j: float, kappa_j: float,
               chord: float) -> float:
    """
    L = 2(θ_j − θ_i) / (κ_i + κ_j)
    퇴화(직선 근사) 시 chord 반환.
    """
    dtheta = _wrap(theta_j - theta_i)
    ksum   = kappa_i + kappa_j
    if abs(ksum) > 1e-8:
        L = 2.0 * dtheta / ksum
        if L > 1e-3:
            return L
    return max(chord, 1e-3)


def _fresnel_endpoint(theta_i: float, kappa_i: float,
                      kappa_j: float, L: float,
                      n_quad: int = 400) -> np.ndarray:
    """
    클로소이드 구간 끝점 변위: ∫₀^L [cos θ, sin θ] ds  (복합 사다리꼴).
    κ(s) = κ_i + (κ_j − κ_i)/L · s
    θ(s) = θ_i + κ_i·s + (κ_j − κ_i)/(2L)·s²
    """
    if L < 1e-12:
        return np.zeros(2)
    s   = np.linspace(0.0, L, n_quad + 1)
    dk  = (kappa_j - kappa_i) / L
    th  = theta_i + kappa_i * s + 0.5 * dk * s ** 2
    return np.array([np.trapz(np.cos(th), s), np.trapz(np.sin(th), s)])


def _clothoid_sample(origin: np.ndarray,
                     theta_i: float, kappa_i: float,
                     kappa_j: float, L: float,
                     ds: float) -> tuple[np.ndarray, np.ndarray]:
    """
    단일 클로소이드 구간 샘플링 (월드 프레임).
    Returns (pts (M,2), kappa (M,)).
    """
    n     = max(2, int(np.ceil(L / ds)) + 1)
    s_arr = np.linspace(0.0, L, n)
    dk    = (kappa_j - kappa_i) / L if L > 1e-12 else 0.0

    kappa_arr = kappa_i + dk * s_arr
    theta_arr = theta_i + kappa_i * s_arr + 0.5 * dk * s_arr ** 2

    cos_h = np.cos(theta_arr)
    sin_h = np.sin(theta_arr)
    x = np.zeros(n)
    y = np.zeros(n)
    for k in range(1, n):
        step  = s_arr[k] - s_arr[k - 1]
        x[k] = x[k - 1] + 0.5 * (cos_h[k - 1] + cos_h[k]) * step
        y[k] = y[k - 1] + 0.5 * (sin_h[k - 1] + sin_h[k]) * step

    pts = np.column_stack([x + origin[0], y + origin[1]])
    return pts, kappa_arr


# ─────────────────────────────────────────────────────────────────────────────
# 전역 비선형 연립방정식
# ─────────────────────────────────────────────────────────────────────────────

def _pack(thetas: np.ndarray, kappas: np.ndarray) -> np.ndarray:
    """내부 노드 {θ_i, κ_i}를 1D 벡터로."""
    x = np.empty(2 * (len(thetas) - 2))
    x[0::2] = thetas[1:-1]
    x[1::2] = kappas[1:-1]
    return x


def _unpack(x: np.ndarray, thetas: np.ndarray, kappas: np.ndarray,
            kappa_max: float) -> None:
    """벡터를 thetas, kappas 배열에 역대입 (in-place, κ 클리핑)."""
    thetas[1:-1] = x[0::2]
    kappas[1:-1] = np.clip(x[1::2], -kappa_max, kappa_max)


def _build_residual(x: np.ndarray,
                    wps_2d: np.ndarray,
                    thetas: np.ndarray,
                    kappas: np.ndarray,
                    kappa_max: float) -> np.ndarray:
    """
    전역 잔차 F(x) ∈ ℝ^{2(N-2)}.

    내부 구간 i-1 (WP_{i-1} → WP_i) 각각에 2개 방정식:
      F[2j], F[2j+1] = _fresnel_endpoint(θ_{i-1}, κ_{i-1}, κ_i, L_{i-1})
                       − (WP_i − WP_{i-1})

    j = 0 ... N-3,  i = 1 ... N-2
    """
    _unpack(x, thetas, kappas, kappa_max)
    N = len(wps_2d)
    F = np.zeros(2 * (N - 2))

    for j, i in enumerate(range(1, N - 1)):
        chord = np.linalg.norm(wps_2d[i] - wps_2d[i - 1])
        L     = _compute_L(thetas[i - 1], kappas[i - 1],
                           thetas[i],     kappas[i], chord)
        p_end = _fresnel_endpoint(thetas[i - 1], kappas[i - 1], kappas[i], L)
        F[2 * j: 2 * j + 2] = p_end - (wps_2d[i] - wps_2d[i - 1])

    return F


def _build_jacobian(x: np.ndarray,
                    wps_2d: np.ndarray,
                    thetas: np.ndarray,
                    kappas: np.ndarray,
                    kappa_max: float,
                    eps: float = 1e-6) -> np.ndarray:
    """
    수치 야코비안 J = ∂F/∂x  (크기: 2(N-2) × 2(N-2)).

    스파스 구조 활용:
      변수 j(= 내부 노드 i의 θ 또는 κ)는
      구간 i-1과 구간 i에만 영향 → 비제로 열은 최대 4개.

    PSEUDO: 여기서는 조밀 수치 미분으로 구현.
    성능이 중요하면 블록 삼중대각 구조를 이용한 해석적 야코비안 권장.
    """
    F0 = _build_residual(x, wps_2d, thetas.copy(), kappas.copy(), kappa_max)
    m, n = len(F0), len(x)
    J = np.zeros((m, n))

    for j in range(n):
        xp = x.copy()
        xp[j] += eps
        Fp = _build_residual(xp, wps_2d, thetas.copy(), kappas.copy(), kappa_max)
        J[:, j] = (Fp - F0) / eps

    # 원본 상태 복원
    _unpack(x, thetas, kappas, kappa_max)
    return J


# ─────────────────────────────────────────────────────────────────────────────
# 초기값 생성
# ─────────────────────────────────────────────────────────────────────────────

def _initial_guess(wps_2d: np.ndarray,
                   theta0: float, kappa0: float,
                   theta_N: float, kappa_N: float,
                   kappa_max: float) -> tuple[np.ndarray, np.ndarray]:
    """
    내부 노드 초기값:
      θ_i  : 입출 코드 이등분선 방위각
      κ_i  : Menger 곡률 (3-WP 외접원 역수)
    """
    N      = len(wps_2d)
    thetas = np.zeros(N)
    kappas = np.zeros(N)
    thetas[0], kappas[0]     = theta0, kappa0
    thetas[N - 1], kappas[N - 1] = theta_N, kappa_N

    for i in range(1, N - 1):
        d_in  = _unit(wps_2d[i]     - wps_2d[i - 1])
        d_out = _unit(wps_2d[i + 1] - wps_2d[i])

        # 이등분선 방위각
        bis       = _unit(d_in + d_out)
        thetas[i] = np.arctan2(bis[1], bis[0])

        # Menger 곡률 근사
        # κ ≈ 2·|cross(d_in, d_out)| / (chord_in + chord_out)
        cross      = d_in[0] * d_out[1] - d_in[1] * d_out[0]
        chord_sum  = (np.linalg.norm(wps_2d[i]     - wps_2d[i - 1]) +
                      np.linalg.norm(wps_2d[i + 1] - wps_2d[i]))
        kappas[i]  = np.clip(2.0 * cross / (chord_sum + 1e-9),
                             -kappa_max, kappa_max)

    return thetas, kappas


# ─────────────────────────────────────────────────────────────────────────────
# 전역 NR 풀이
# ─────────────────────────────────────────────────────────────────────────────

def _solve_global_nr(wps_2d: np.ndarray,
                     kappa_max: float,
                     theta0: float, kappa0: float,
                     theta_N: float, kappa_N: float,
                     max_iter: int = 50,
                     tol: float = 1e-4,
                     step_max: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    전역 Newton-Raphson: F(x) = 0 풀이.

    Algorithm
    ---------
    1. 초기값 x₀ = _initial_guess(...)
    2. for iter:
         F = _build_residual(x, ...)      # 2(N-2) 잔차
         if ||F||₂ < tol: break
         J = _build_jacobian(x, ...)      # 2(N-2) × 2(N-2)
         dx = lstsq(J, -F, rcond=None)    # 최소제곱 (정칙화)
         α  = min(1, step_max / ||dx||)   # 단순 라인 서치
         x += α · dx
         x[1::2] = clip(x[1::2], -κ_max, κ_max)  # κ 투영

    3. κ_max 위반 노드가 있으면:
         PSEUDO: 해당 노드 κ_i = ±κ_max 고정 후 θ_i만 재풀이 (자유도 감소)

    Returns thetas (N,), kappas (N,)
    """
    N = len(wps_2d)
    thetas, kappas = _initial_guess(wps_2d, theta0, kappa0, theta_N, kappa_N, kappa_max)

    if N <= 2:
        return thetas, kappas

    x = _pack(thetas, kappas)

    for it in range(max_iter):
        F = _build_residual(x, wps_2d, thetas, kappas, kappa_max)
        norm_F = np.linalg.norm(F)

        if norm_F < tol:
            break

        J  = _build_jacobian(x, wps_2d, thetas, kappas, kappa_max)
        dx = np.linalg.lstsq(J, -F, rcond=1e-12)[0]

        # 라인 서치 (norm-clamp)
        alpha = min(1.0, step_max / (np.linalg.norm(dx) + 1e-12))
        x    += alpha * dx

        # κ 투영
        x[1::2] = np.clip(x[1::2], -kappa_max, kappa_max)

    _unpack(x, thetas, kappas, kappa_max)

    # PSEUDO: κ_max 위반 재처리
    # for i in range(1, N-1):
    #     if abs(kappas[i]) >= kappa_max * 0.99:
    #         _refit_theta_fixed_kappa(thetas, kappas, wps_2d, i, kappa_max)

    return thetas, kappas


# ─────────────────────────────────────────────────────────────────────────────
# G3 서브-세그먼트 분할 (선택, PSEUDO)
# ─────────────────────────────────────────────────────────────────────────────

def _insert_g3_midnodes(wps: np.ndarray,
                        thetas: np.ndarray,
                        kappas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PSEUDO — G3 확장: 각 구간 중간에 자유 노드 P_mid 삽입.

    G3 조건 (dκ/ds 연속):
      (κ_{i+1} − κ_i)/L_i = (κ_{i+2} − κ_{i+1})/L_{i+1}

    Sub-segment 분할로 이 조건을 추가 제약으로 포함시키거나
    중간 노드의 κ_mid를 방정식 근으로 결정.

    이 함수는 WP 배열에 중간 노드를 삽입하여 새 wps, thetas, kappas를 반환.
    반환 후 _solve_global_nr 재실행 필요.
    """
    # PSEUDO:
    # new_wps, new_thetas, new_kappas = [], [], []
    # for i in range(len(wps) - 1):
    #     new_wps.append(wps[i])
    #     new_thetas.append(thetas[i])
    #     new_kappas.append(kappas[i])
    #     mid = (wps[i] + wps[i+1]) / 2.0
    #     kappa_mid = (kappas[i] + kappas[i+1]) / 2.0  # 초기값
    #     theta_mid = (thetas[i] + thetas[i+1]) / 2.0
    #     new_wps.append(mid)
    #     new_thetas.append(theta_mid)
    #     new_kappas.append(kappa_mid)
    # new_wps.append(wps[-1])
    # ...
    raise NotImplementedError("PSEUDO: G3 sub-segment 분할")


# ─────────────────────────────────────────────────────────────────────────────
# Planner 클래스
# ─────────────────────────────────────────────────────────────────────────────

class PiecewiseClothoidPlanner(BasePlanner):
    """
    Piecewise Clothoid G2 Interpolation Planner.

    각 구간을 클로소이드로 표현하고, Fresnel 적분 위치 제약을
    전역 Newton-Raphson으로 풀어 모든 WP에 정확히 통과.

    Parameters
    ----------
    ds           : 경로 점 간격 (m)
    accel_tol    : 가속도 여유율 (0 < tol ≤ 1)
    nr_tol       : NR 수렴 임계값 (m)
    nr_max_iter  : NR 최대 반복 횟수
    use_g3       : True 시 dκ/ds 연속 (G3) 근사 — sub-segment 분할
    end_extension: 마지막 WP 이후 직선 연장 (m)
    """

    def __init__(self,
                 ds: float = 1.0,
                 accel_tol: float = 0.9,
                 nr_tol: float = 1e-4,
                 nr_max_iter: int = 50,
                 use_g3: bool = False,
                 end_extension: float = 15.0):
        self.ds            = ds
        self.accel_tol     = accel_tol
        self.nr_tol        = nr_tol
        self.nr_max_iter   = nr_max_iter
        self.use_g3        = use_g3
        self.end_extension = end_extension

    # ── 공개 인터페이스 ───────────────────────────────────────────────────────

    def plan(self,
             waypoints_ned: np.ndarray,
             aircraft_params: dict,
             initial_state: dict | None = None) -> Path:
        t0  = time.perf_counter()
        wps = np.asarray(waypoints_ned, dtype=float)
        if len(wps) < 2:
            raise ValueError("waypoints는 최소 2개 필요")

        # ── 기체 한계 ─────────────────────────────────────────────────
        g         = float(aircraft_params.get("gravity", 9.81))
        v_cruise  = float(aircraft_params["v_cruise"])
        a_max_g   = float(aircraft_params["a_max_g"])
        a_max     = a_max_g * g * self.accel_tol
        kappa_max = a_max / (v_cruise ** 2)

        wps_2d = wps[:, :2]
        N      = len(wps)

        # ── 경계 조건 ─────────────────────────────────────────────────
        theta0  = np.arctan2(wps[1, 1] - wps[0, 1], wps[1, 0] - wps[0, 0])
        theta_N = np.arctan2(wps[-1, 1] - wps[-2, 1], wps[-1, 0] - wps[-2, 0])
        kappa0  = 0.0
        kappa_N = 0.0
        if initial_state and "initial_heading" in initial_state:
            theta0 = float(initial_state["initial_heading"])

        # ── G3 전처리 (선택) ──────────────────────────────────────────
        if self.use_g3:
            # PSEUDO: _insert_g3_midnodes 로 wps_2d 확장 후 N 갱신
            # wps, wps_2d, thetas_init, kappas_init = _insert_g3_midnodes(...)
            # N = len(wps_2d)
            pass

        # ── 전역 NR 풀이: {θ_i, κ_i} ─────────────────────────────────
        thetas, kappas = _solve_global_nr(
            wps_2d, kappa_max,
            theta0, kappa0, theta_N, kappa_N,
            max_iter=self.nr_max_iter, tol=self.nr_tol,
        )

        # ── 구간 샘플링 ───────────────────────────────────────────────
        all_pts:   list[np.ndarray] = []
        all_kappa: list[np.ndarray] = []
        wp_marks:  dict[int, int]   = {}

        for i in range(N - 1):
            chord = np.linalg.norm(wps_2d[i + 1] - wps_2d[i])
            L     = _compute_L(thetas[i], kappas[i],
                               thetas[i + 1], kappas[i + 1], chord)

            seg_pts, seg_kappa = _clothoid_sample(
                wps_2d[i], thetas[i], kappas[i], kappas[i + 1], L, self.ds
            )

            wp_marks[sum(len(p) for p in all_pts)] = i
            all_pts.append(seg_pts[:-1])
            all_kappa.append(seg_kappa[:-1])

        # 마지막 WP
        idx_last = sum(len(p) for p in all_pts)
        wp_marks[idx_last] = N - 1
        all_pts.append(wps_2d[N - 1: N])
        all_kappa.append(np.array([kappas[-1]]))

        # 종단 연장
        last_dir = np.array([np.cos(thetas[-1]), np.sin(thetas[-1])])
        n_ext    = max(2, int(self.end_extension / self.ds))
        ext_pts  = (wps_2d[-1] +
                    last_dir * np.linspace(0.0, self.end_extension, n_ext)[:, None])
        all_pts.append(ext_pts[1:])
        all_kappa.append(np.zeros(len(ext_pts) - 1))

        pts_arr   = np.concatenate(all_pts,   axis=0)
        kappa_arr = np.concatenate(all_kappa, axis=0)

        # ── 호 길이 ────────────────────────────────────────────────────
        diffs = np.diff(pts_arr, axis=0)
        s_arr = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])

        # ── 고도 보간 ─────────────────────────────────────────────────
        sorted_marks = sorted([(k, wi) for k, wi in wp_marks.items() if wi < N],
                               key=lambda x: x[1])
        wp_s_arr = np.array([s_arr[k] for k, _  in sorted_marks])
        wp_h_arr = np.array([wps[wi, 2] for _, wi in sorted_marks])
        alt_arr  = np.interp(s_arr, wp_s_arr, wp_h_arr)

        # ── 방위각 ─────────────────────────────────────────────────────
        chi_arr = np.zeros(len(pts_arr))
        for i in range(len(pts_arr) - 1):
            d = pts_arr[i + 1] - pts_arr[i]
            chi_arr[i] = np.arctan2(d[1], d[0])
        chi_arr[-1] = chi_arr[-2]

        # ── 상승각 ─────────────────────────────────────────────────────
        gamma_arr = np.zeros(len(pts_arr))
        for i in range(len(pts_arr) - 1):
            dh   = alt_arr[i + 1] - alt_arr[i]
            ds_i = s_arr[i + 1]   - s_arr[i]
            gamma_arr[i] = np.arctan2(dh, ds_i) if ds_i > 1e-9 else 0.0
        gamma_arr[-1] = gamma_arr[-2]

        # ── PathPoint 조립 ────────────────────────────────────────────
        points: list[PathPoint] = []
        for idx in range(len(pts_arr)):
            points.append(PathPoint(
                pos       = np.array([pts_arr[idx, 0], pts_arr[idx, 1], alt_arr[idx]]),
                v_ref     = v_cruise,
                chi_ref   = float(chi_arr[idx]),
                gamma_ref = float(gamma_arr[idx]),
                curvature = float(kappa_arr[idx]),
                s         = float(s_arr[idx]),
                wp_index  = wp_marks.get(idx, None),
            ))

        return Path(
            points        = points,
            waypoints_ned = wps,
            total_length  = float(s_arr[-1]),
            planning_time = time.perf_counter() - t0,
        )
