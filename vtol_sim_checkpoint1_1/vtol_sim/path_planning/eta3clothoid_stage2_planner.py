"""
η³-Spline + Clothoid Interior Replacement Planner  [PSEUDO CODE]
=================================================================
두 단계 파이프라인:
  Stage 1 — η³-spline G2 보간: 각 WP에서 {θ_i, κ_i} 결정
  Stage 2 — 구간별 Clothoid 재구성: NR 위치 보정으로 정확한 WP 통과 보장

보장 항목:
  ✓ WP 완전 통과          (NR 수렴 조건)
  ✓ κ_max 전 구간 준수     (클리핑 + 구조적 보장)
  ✓ dκ/ds = const (구간)   (클로소이드 구조)
  ✓ G2 연속성             (NR 보정 후)
  ✗ G3 연속성             (인접 구간 재협상 필요 — 미구현)

참고:
  Bertolazzi & Frego (2018), "G¹ and G² fitting with η³-splines"
  κ(s) = κ_i + (κ_{i+1} - κ_i)/L · s
  θ(s) = θ_i + κ_i·s + (κ_{i+1} - κ_i)/(2L)·s²
  p(s) = WP_i + ∫₀ˢ [cos θ, sin θ]ᵀ dt  (Fresnel 적분)
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
    """각도를 (-π, π]로 정규화."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _clothoid_sample(theta_i: float, kappa_i: float,
                     kappa_j: float, L: float,
                     ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    단일 클로소이드 구간 이산 샘플링 (원점 기준 로컬 프레임).

    κ(s) = κ_i + dk·s,          dk = (κ_j - κ_i)/L
    θ(s) = θ_i + κ_i·s + dk/2·s²
    p(s) = ∫₀ˢ [cos θ, sin θ] dt  (사다리꼴)

    Returns
    -------
    pts      : (M, 2)  위치
    theta_arr: (M,)    방위각
    kappa_arr: (M,)    곡률
    """
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
    """
    클로소이드 구간 끝점 변위 (고밀도 적분, 상대 좌표).
    Returns (dx, dy).
    """
    if L < 1e-12:
        return np.zeros(2)
    s_arr = np.linspace(0.0, L, n_quad + 1)
    dk = (kappa_j - kappa_i) / L
    theta_arr = theta_i + kappa_i * s_arr + 0.5 * dk * s_arr ** 2
    dx = np.trapz(np.cos(theta_arr), s_arr)
    dy = np.trapz(np.sin(theta_arr), s_arr)
    return np.array([dx, dy])


def _compute_L(theta_i: float, kappa_i: float,
               theta_j: float, kappa_j: float,
               chord: float) -> float:
    """
    헤딩 차 공식: L = 2(θ_j - θ_i)/(κ_i + κ_j)
    퇴화 시 chord 반환.
    """
    dtheta = _wrap(theta_j - theta_i)
    ksum = kappa_i + kappa_j
    if abs(ksum) > 1e-8:
        L_est = 2.0 * dtheta / ksum
        if L_est > 1e-3:
            return L_est
    return max(chord, 1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — η³-Spline G2 초기 풀이 (PSEUDO)
# ─────────────────────────────────────────────────────────────────────────────

def _menger_kappa(wps_2d: np.ndarray, i: int, kappa_max: float) -> float:
    """
    Interior node i 의 부호 있는 Menger 곡률.
    NED 컨벤션: 우선회(CW) = 양수.

    κ = 2·Area(a,b) / (|a|·|b|·|a-b_chord|)
    """
    a = wps_2d[i]     - wps_2d[i - 1]   # 진입 코드
    b = wps_2d[i + 1] - wps_2d[i]       # 탈출 코드
    cross = a[0] * b[1] - a[1] * b[0]   # >0 → NED 우선회
    area2 = abs(cross)
    la        = np.linalg.norm(a)
    lb        = np.linalg.norm(b)
    chord_ac  = np.linalg.norm(wps_2d[i + 1] - wps_2d[i - 1])
    if la * lb * chord_ac < 1e-9:
        return 0.0
    kappa_abs = area2 / (la * lb * chord_ac)
    sign = 1.0 if cross >= 0.0 else -1.0
    return float(np.clip(sign * kappa_abs, -kappa_max * 0.9, kappa_max * 0.9))


def _eta3_initial_guess(wps_2d: np.ndarray,
                        theta0: float, kappa0: float,
                        theta_N: float, kappa_N: float,
                        kappa_max: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    η³ G2 풀기 전 초기값 생성.

    θ_i : 인접 코드 방향의 이등분선 방위각 (U턴은 수직 방향으로 처리)
    κ_i : Menger 곡률 (내부 노드) — κ=0 대비 _compute_L 특이점 회피
    """
    N = len(wps_2d)
    thetas = np.zeros(N)
    kappas = np.zeros(N)
    thetas[0], kappas[0] = theta0, kappa0
    thetas[N - 1], kappas[N - 1] = theta_N, kappa_N

    for i in range(1, N - 1):
        d_in  = _unit(wps_2d[i] - wps_2d[i - 1])
        d_out = _unit(wps_2d[i + 1] - wps_2d[i])
        bis   = d_in + d_out
        if np.linalg.norm(bis) < 1e-9:
            bis = np.array([-d_in[1], d_in[0]])  # U턴: 진입 방향에 수직
        thetas[i] = np.arctan2(bis[1], bis[0])
        kappas[i] = _menger_kappa(wps_2d, i, kappa_max)

    return thetas, kappas


def _eta3_g2_residual(thetas: np.ndarray, kappas: np.ndarray,
                      wps_2d: np.ndarray) -> np.ndarray:
    """
    PSEUDO — η³ G2 연립 방정식 잔차.

    내부 노드 i (1 ≤ i ≤ N-2) 각각에 대해 2개 조건:
      F[2j]   = θ_i - θ_natural(chord_in, chord_out, κ_{i-1}, κ_i, κ_{i+1})
      F[2j+1] = κ_i - κ_natural(chord_in, chord_out, θ_i)

    θ_natural, κ_natural은 η³-spline의 G2 접합 조건에서 유도.
    (Bertolazzi-Frego 논문 Eq. 8–12 참조)

    Returns F of size 2(N-2).
    """
    # PSEUDO: 실제 η³ G2 조건 구현
    #
    # N = len(wps_2d)
    # F = np.zeros(2 * (N - 2))
    # for j, i in enumerate(range(1, N - 1)):
    #     chord_in  = wps_2d[i]     - wps_2d[i - 1]
    #     chord_out = wps_2d[i + 1] - wps_2d[i]
    #     eta_i = _eta3_node_params(thetas[i-1], kappas[i-1],
    #                               thetas[i],   kappas[i],
    #                               thetas[i+1], kappas[i+1],
    #                               chord_in, chord_out)
    #     F[2*j]   = thetas[i] - eta_i.theta_natural
    #     F[2*j+1] = kappas[i] - eta_i.kappa_natural
    # return F
    raise NotImplementedError("PSEUDO: η³ G2 조건 — Bertolazzi-Frego (2018) 참조하여 구현")


def _solve_eta3_g2(wps_2d: np.ndarray,
                   kappa_max: float,
                   theta0: float, kappa0: float,
                   theta_N: float, kappa_N: float,
                   max_iter: int = 50, tol: float = 1e-6
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Stage 1: η³ G2 보간으로 모든 WP의 {θ_i, κ_i} 결정.

    Algorithm
    ---------
    1. 초기값 생성 (_eta3_initial_guess)
    2. Newton-Raphson: x = [θ_1, κ_1, ..., θ_{N-2}, κ_{N-2}]
       for iter:
         F = _eta3_g2_residual(thetas, kappas, wps_2d)   # 2(N-2) 방정식
         if ||F|| < tol: break
         J = ∂F/∂x  (수치 미분, 5점 스텐실 권장)
         dx = lstsq(J, -F, rcond=None)[0]
         x += dx  (+ 라인 서치)
    3. κ_max 클리핑 후 θ 소폭 재조정 (G2 근사 유지)

    Returns thetas (N,), kappas (N,)  — 경계 포함, ±κ_max 클리핑됨
    """
    N = len(wps_2d)
    thetas, kappas = _eta3_initial_guess(wps_2d, theta0, kappa0, theta_N, kappa_N, kappa_max)

    if N <= 2:
        return thetas, kappas

    # x: 내부 노드만 최적화
    x = np.zeros(2 * (N - 2))
    x[0::2] = thetas[1:-1]
    x[1::2] = kappas[1:-1]

    # PSEUDO: Newton-Raphson 루프
    #
    # for it in range(max_iter):
    #     thetas[1:-1] = x[0::2]
    #     kappas[1:-1] = x[1::2]
    #     F = _eta3_g2_residual(thetas, kappas, wps_2d)
    #     if np.linalg.norm(F) < tol:
    #         break
    #     J = _numerical_jacobian(lambda v: _set_and_residual(v, thetas, kappas, wps_2d),
    #                             x, eps=1e-7)
    #     dx = np.linalg.lstsq(J, -F, rcond=None)[0]
    #     alpha = min(1.0, 0.5 / (np.linalg.norm(dx) + 1e-12))
    #     x += alpha * dx

    # κ_max 클리핑
    kappas = np.clip(kappas, -kappa_max, kappa_max)

    # PSEUDO: 클리핑 후 θ 재조정
    # for i in range(1, N-1):
    #     if abs(x[1::2][i-1]) >= kappa_max * 0.99:
    #         thetas[i] = _refit_theta_after_kclip(thetas, kappas, wps_2d, i)

    return thetas, kappas


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — 구간별 Clothoid 재구성 + NR 위치 보정
# ─────────────────────────────────────────────────────────────────────────────

def _segment_nr_correct(theta_i: float, kappa_i: float,
                        theta_j: float, kappa_j: float,
                        target_disp: np.ndarray,
                        kappa_max: float,
                        max_iter: int = 30, tol: float = 1e-4
                        ) -> tuple[float, float, float]:
    """
    NR 보정: 주어진 (θ_i, κ_i) → (θ_j, κ_j) 클로소이드 구간의 끝점이
    target_disp 를 통과하도록 θ_i, κ_i를 미세 조정하고 L을 결정.

    Algorithm
    ---------
    초기 L = _compute_L(θ_i, κ_i, θ_j, κ_j, chord)

    Newton-Raphson on x2 = [θ_i, κ_i]:
      for iter:
        L  = 2(θ_j - θ_i)/(κ_i + κ_j)
        p  = _fresnel_endpoint(θ_i, κ_i, κ_j, L)
        r  = p - target_disp
        if |r| < tol: break
        J  = [∂p/∂θ_i | ∂p/∂κ_i]  (수치 미분 2×2, L도 θ_i, κ_i에 종속)
        x2 -= J⁻¹ r
        κ_i = clip(κ_i, -κ_max, κ_max)

    Returns (L, theta_i_corrected, kappa_i_corrected)
    """
    chord = np.linalg.norm(target_disp)
    L = _compute_L(theta_i, kappa_i, theta_j, kappa_j, chord)
    theta_i_c = theta_i
    kappa_i_c = kappa_i

    eps = 1e-6
    for _ in range(max_iter):
        # 끝점 잔차
        p0  = _fresnel_endpoint(theta_i_c, kappa_i_c, kappa_j, L)
        res = p0 - target_disp
        if np.linalg.norm(res) < tol:
            break

        # 수치 야코비안 (θ_i, κ_i 각각 교란; L도 재계산)
        L_dT = _compute_L(theta_i_c + eps, kappa_i_c, theta_j, kappa_j, chord)
        pT   = _fresnel_endpoint(theta_i_c + eps, kappa_i_c, kappa_j, L_dT)
        L_dK = _compute_L(theta_i_c, kappa_i_c + eps, theta_j, kappa_j, chord)
        pK   = _fresnel_endpoint(theta_i_c, kappa_i_c + eps, kappa_j, L_dK)

        J = np.column_stack([(pT - p0) / eps, (pK - p0) / eps])  # 2×2

        try:
            delta = np.linalg.solve(J, -res)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, -res, rcond=None)[0]

        # 스텝 제한 (라인 서치 대신 단순 클램프)
        step = min(1.0, 0.3 / (np.linalg.norm(delta) + 1e-12))
        theta_i_c += step * delta[0]
        kappa_i_c  = np.clip(kappa_i_c + step * delta[1], -kappa_max, kappa_max)

        L = _compute_L(theta_i_c, kappa_i_c, theta_j, kappa_j, chord)

    return L, theta_i_c, kappa_i_c


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
    """
    κ_needed > κ_max 인 구간에 중점 WP를 삽입하여 기하학적 실현 가능성을 확보.

    κ_needed = 2·|Δθ| / chord  (균일 κ 근사)

    삽입 WP:
      θ_mid = 코드 방향각 (양 끝 코드가 동일 방향이므로 bisector = 코드 방향)
      κ_mid = 0  (중점 Menger 곡률은 구조적으로 0)

    max_insert 회 반복 후에도 불가 구간이 남으면 그대로 진행
    (후속 NR에서 κ_max 클리핑으로 최선 근사).

    Returns
    -------
    wps_2d_new   : (N_new, 2)
    thetas_new   : (N_new,)
    kappas_new   : (N_new,)
    orig_indices : list[int]  길이 N_new
        orig_indices[k] >= 0 → 원본 WP 인덱스
        orig_indices[k] == -1 → 삽입된 WP
    """
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
                wp_mid    = 0.5 * (wps_2d[i] + wps_2d[i + 1])
                theta_mid = np.arctan2(
                    wps_2d[i + 1][1] - wps_2d[i][1],
                    wps_2d[i + 1][0] - wps_2d[i][0],
                )
                wps_2d = np.insert(wps_2d, i + 1, wp_mid,    axis=0)
                thetas = np.insert(thetas, i + 1, theta_mid)
                kappas = np.insert(kappas, i + 1, 0.0)
                orig_indices.insert(i + 1, -1)
                inserted = True
                i += 2          # 삽입된 두 부분 구간을 이번 패스에서 재검사하지 않음
            else:
                i += 1

        if not inserted:
            break

    return wps_2d, thetas, kappas, orig_indices


# ─────────────────────────────────────────────────────────────────────────────
# Planner 클래스
# ─────────────────────────────────────────────────────────────────────────────

class Eta3ClothoidPlanner(BasePlanner):
    """
    η³-Spline + Clothoid Interior Replacement Planner.

    Parameters
    ----------
    ds           : 경로 점 간격 (m)
    accel_tol    : 가속도 여유율 (0 < tol ≤ 1)
    nr_tol       : NR 수렴 임계값 (m)
    nr_max_iter  : NR 최대 반복 횟수
    end_extension: 마지막 WP 이후 직선 연장 (m)
    """

    def __init__(self,
                 ds: float = 1.0,
                 accel_tol: float = 0.9,
                 nr_tol: float = 1e-4,
                 nr_max_iter: int = 30,
                 end_extension: float = 15.0):
        self.ds           = ds
        self.accel_tol    = accel_tol
        self.nr_tol       = nr_tol
        self.nr_max_iter  = nr_max_iter
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
        kappa_max = a_max / (v_cruise ** 2)   # = 1/R_min

        N_original = len(wps)          # 원본 WP 수 — 고도 보간, wp_index 마킹에 사용
        wps_2d     = wps[:, :2]
        N          = N_original

        # ── 경계 조건 ─────────────────────────────────────────────────
        theta0  = np.arctan2(wps[1, 1] - wps[0, 1], wps[1, 0] - wps[0, 0])
        theta_N = np.arctan2(wps[-1, 1] - wps[-2, 1], wps[-1, 0] - wps[-2, 0])
        kappa0  = 0.0
        kappa_N = 0.0
        if initial_state and "initial_heading" in initial_state:
            theta0 = float(initial_state["initial_heading"])

        # ── Stage 1: η³ G2 → {θ_i, κ_i} ─────────────────────────────
        # PSEUDO: _eta3_g2_residual 내부가 NotImplementedError를 발생시킴
        # 아래 호출은 완전 구현 시 활성화
        #
        # thetas, kappas = _solve_eta3_g2(
        #     wps_2d, kappa_max, theta0, kappa0, theta_N, kappa_N,
        #     max_iter=50, tol=1e-6,
        # )
        #
        # PSEUDO 대체: Menger κ를 초기값으로 사용하는 개선된 초기값
        thetas, kappas = _eta3_initial_guess(
            wps_2d, theta0, kappa0, theta_N, kappa_N, kappa_max,
        )
        kappas = np.clip(kappas, -kappa_max, kappa_max)

        # ── [1순위] 기하학적 실현 가능성 검사 + WP 삽입 ──────────────
        wps_2d, thetas, kappas, orig_indices = _check_and_insert_wps(
            wps_2d, thetas, kappas, kappa_max,
        )
        N = len(wps_2d)

        # ── Stage 2: 구간별 Clothoid + NR 위치 보정 ───────────────────
        all_pts:   list[np.ndarray] = []
        all_kappa: list[np.ndarray] = []
        wp_marks:  dict[int, int]   = {}   # path_idx → wp_index

        for i in range(N - 1):
            target_disp = wps_2d[i + 1] - wps_2d[i]

            L, th_c, kp_c = _segment_nr_correct(
                thetas[i], kappas[i],
                thetas[i + 1], kappas[i + 1],
                target_disp, kappa_max,
                max_iter=self.nr_max_iter, tol=self.nr_tol,
            )
            thetas[i] = th_c
            kappas[i] = kp_c

            seg_pts, _, seg_kappa = _clothoid_sample(
                thetas[i], kappas[i], kappas[i + 1], L, self.ds
            )
            seg_pts = seg_pts + wps_2d[i]  # 월드 프레임 평행 이동

            if orig_indices[i] >= 0:        # 삽입 WP는 wp_index 마킹 제외
                wp_marks[sum(len(p) for p in all_pts)] = orig_indices[i]

            all_pts.append(seg_pts[:-1])        # 마지막 점 = 다음 구간 첫 점
            all_kappa.append(seg_kappa[:-1])

        # 마지막 WP (orig_indices[-1] 은 항상 N_original-1)
        idx_last = sum(len(p) for p in all_pts)
        if orig_indices[N - 1] >= 0:
            wp_marks[idx_last] = orig_indices[N - 1]
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
        diffs    = np.diff(pts_arr, axis=0)
        s_arr    = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])

        # ── 고도 보간 ─────────────────────────────────────────────────
        sorted_marks = sorted(
            [(k, wi) for k, wi in wp_marks.items() if wi < N_original],
            key=lambda x: x[1],
        )
        wp_s_arr = np.array([s_arr[k] for k, _ in sorted_marks])
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
                pos      = np.array([pts_arr[idx, 0], pts_arr[idx, 1], alt_arr[idx]]),
                v_ref    = v_cruise,
                chi_ref  = float(chi_arr[idx]),
                gamma_ref= float(gamma_arr[idx]),
                curvature= float(kappa_arr[idx]),
                s        = float(s_arr[idx]),
                wp_index = wp_marks.get(idx, None),
            ))

        return Path(
            points        = points,
            waypoints_ned = wps,
            total_length  = float(s_arr[-1]),
            planning_time = time.perf_counter() - t0,
        )
