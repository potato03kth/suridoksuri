"""
η³-Clothoid Planner v3.1 — 단일 통합 G2 NR (수렴 안정성 + 연결 일관성 개선)
================================================================================
설계 문서: eta3_clothoid_planner_v3.md  (변경점은 CHANGES_v3_1.md 참조)

v3 → v3.1 핵심 변경:
  • Stage 0에서 계산한 θ 초기값을 Stage 1 NR로 전달 (이전: 폐기 후 재계산)
  • L 자유변수 클립 범위 확장: v∈[-0.5, 1.2] → L∈[0.61, 3.32]·chord
  • NR 비수렴 시 L 클립 한계를 더 풀어 재시도하는 fallback 루프
  • Clothoid 연결을 "WP 좌표 점프"가 아닌 "이전 segment 끝점 누적"으로 변경
    (NR 잔차에 의한 시각적 꺾임/직선 점프 제거)
  • 종단 마지막 WP 더미 점 추가 제거
  • Armijo line search 백트래킹 횟수 8 → 20

파이프라인:
  Stage 0 — κ_needed > κ_max 구간에 WP 사전 삽입 (θ는 원형 평균)
  Stage 1 — 단일 통합 NR: 위치(2) + 헤딩(1) 잔차 직접 해소
  종단    — κ 감쇠 클로소이드 1개 삽입 후 직선 연장
"""
from __future__ import annotations
import time
import numpy as np
from .base_planner import BasePlanner, Path, PathPoint


# ─────────────────────────────────────────────────────────────────────────────
# 모듈 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(a: float) -> float:
    """각도를 (-π, π]로 정규화."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _wrap_arr(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _fresnel_endpoint(theta_i: float, kappa_i: float,
                      kappa_j: float, L: float,
                      n_quad: int = 400) -> np.ndarray:
    """클로소이드 구간 끝점 변위 (사다리꼴 적분, 상대 좌표)."""
    if L < 1e-12:
        return np.zeros(2)
    s = np.linspace(0.0, L, n_quad + 1)
    dk = (kappa_j - kappa_i) / L
    th = theta_i + kappa_i * s + 0.5 * dk * s * s
    return np.array([np.trapz(np.cos(th), s), np.trapz(np.sin(th), s)])


def _clothoid_sample(theta_i: float, kappa_i: float,
                     kappa_j: float, L: float,
                     ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """단일 클로소이드 구간 이산 샘플링 (원점 기준 로컬 프레임)."""
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
    """Interior 노드 i의 부호 있는 Menger 곡률 (NED: CW=+)."""
    a = wps[i] - wps[i - 1]
    b = wps[i + 1] - wps[i]
    cross = a[0] * b[1] - a[1] * b[0]
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    chord = np.linalg.norm(wps[i + 1] - wps[i - 1])
    if la * lb * chord < 1e-9:
        return 0.0
    k_abs = abs(cross) / (la * lb * chord)
    return float(np.clip(np.sign(cross) * k_abs, -kappa_max * 0.9, kappa_max * 0.9))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — WP 사전 삽입
# ─────────────────────────────────────────────────────────────────────────────

def _initial_thetas(wps: np.ndarray,
                    theta0: float, theta_N: float) -> np.ndarray:
    """가중 이등분선으로 초기 θ 추정.
    θ_nat = arg( d_in/√L_in + d_out/√L_out ) — 짧은 코드 쪽 가중 ↑
    """
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
            bis = np.array([-d_in[1] / L_in, d_in[0] / L_in])  # U턴 fallback
        th[i] = float(np.arctan2(bis[1], bis[0]))
    return th


def _insert_wps_if_infeasible(wps: np.ndarray,
                              thetas: np.ndarray,
                              kappa_max: float,
                              max_insert: int = 4
                              ) -> tuple[np.ndarray, np.ndarray, list]:
    """κ_needed > κ_max 구간에 중점 WP 삽입.
    삽입된 WP의 θ: 양옆 θ의 원형 평균 (G1 부드러움 유지).
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — 단일 통합 G2 NR
# ─────────────────────────────────────────────────────────────────────────────

# v3.1: L 클립 범위 확장 — 큰 회전각/긴 chord 케이스 수렴 보장
_V_CLIP_LO_DEFAULT = -0.5   # exp(-0.5) ≈ 0.61
_V_CLIP_HI_DEFAULT = 1.2   # exp(1.2)  ≈ 3.32


def _unpack(x: np.ndarray, N: int,
            theta_bc: tuple, kappa_bc: tuple,
            kappa_max: float, chords: np.ndarray,
            v_lo: float, v_hi: float
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """자유변수 벡터 x → (θ, κ, L) 디코딩."""
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
              mean_chord: float,
              v_lo: float, v_hi: float,
              n_quad: int = 100) -> np.ndarray:
    """전역 잔차 (segment 당 3개: 위치 x, y, 헤딩)."""
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
        F[3 * k + 2] = mean_chord * _wrap(th_end - th[k + 1])
    return F


def _solve_g2_nr(wps: np.ndarray,
                 kappa_max: float,
                 theta0: float, kappa0: float,
                 theta_N: float, kappa_N: float,
                 th_init_full: np.ndarray | None = None,   # v3.1: Stage 0의 θ 전달
                 max_iter: int = 60, tol: float = 1e-5,
                 eps_jac: float = 1e-6,
                 verbose: bool = False
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """단일 통합 G2 NR. 반환에 최종 |F|를 추가하여 수렴 여부를 판단할 수 있게 함."""
    N = len(wps)
    n_segs = N - 1
    chords = np.array([np.linalg.norm(wps[k + 1] - wps[k])
                       for k in range(n_segs)])

    # v3.1: Stage 0의 θ가 있으면 사용, 없으면 가중 이등분선으로 다시 계산
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
        return th_init, kp_init, np.maximum(chords, 1e-3), 0.0

    n_inner = N - 2
    mean_chord = float(np.mean(chords))
    theta_bc = (float(theta0), float(theta_N))
    kappa_bc = (float(kappa0), float(kappa_N))

    # v3.1: 비수렴 시 L 클립 한계를 단계적으로 풀며 재시도
    clip_attempts = [
        (_V_CLIP_LO_DEFAULT, _V_CLIP_HI_DEFAULT),    # 1차: [-0.5, 1.2]
        (-0.7, 1.8),                                 # 2차: 더 풀어줌
        (-1.0, 2.5),                                 # 3차: 매우 관대
    ]

    best = None  # (norm_F, x, v_lo, v_hi)

    for v_lo, v_hi in clip_attempts:
        x = np.empty(2 * n_inner + n_segs)
        x[0:n_inner] = th_init[1:-1]
        x[n_inner:2 * n_inner] = np.arctanh(
            np.clip(kp_init[1:-1] / kappa_max, -0.9999, 0.9999))
        x[2 * n_inner:] = 0.0

        args = (wps, theta_bc, kappa_bc, kappa_max,
                chords, mean_chord, v_lo, v_hi)
        prev_norm = np.inf

        for it in range(max_iter):
            F = _residual(x, *args)
            norm_F = float(np.linalg.norm(F))

            if verbose:
                _, kk, LL = _unpack(x, N, theta_bc, kappa_bc, kappa_max,
                                    chords, v_lo, v_hi)
                pos_max = max(np.max(np.abs(F[0::3])), np.max(np.abs(F[1::3])))
                head_max = np.max(np.abs(F[2::3])) / max(mean_chord, 1e-9)
                print(f"  [G2-NR clip=({v_lo:.1f},{v_hi:.1f}) it={it:02d}] "
                      f"|F|={norm_F:.3e} pos_max={pos_max:.3e}m "
                      f"head_max={head_max:.3e}rad "
                      f"|κ|/κmax={np.max(np.abs(kk))/kappa_max:.3f} "
                      f"L/chord∈[{np.min(LL/chords):.2f},{np.max(LL/chords):.2f}]")

            if norm_F < tol:
                break

            # 수치 야코비안
            m_j, n_j = len(F), len(x)
            J = np.zeros((m_j, n_j))
            for j in range(n_j):
                xp = x.copy()
                xp[j] += eps_jac
                J[:, j] = (_residual(xp, *args) - F) / eps_jac

            try:
                dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
            except np.linalg.LinAlgError:
                break

            # v3.1: Armijo line search 백트래킹 횟수 8 → 20
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

        # 이번 시도 결과 평가
        F_final = _residual(x, *args)
        norm_final = float(np.linalg.norm(F_final))
        if best is None or norm_final < best[0]:
            best = (norm_final, x.copy(), v_lo, v_hi)

        if norm_final < tol:
            break  # 충분히 수렴 → 다음 클립 시도 불필요

        if verbose:
            print(f"  [G2-NR] clip=({v_lo:.1f},{v_hi:.1f}) 미수렴 "
                  f"(|F|={norm_final:.3e}). 더 넓은 클립으로 재시도.")

    # 최선 결과 사용
    norm_best, x_best, v_lo_best, v_hi_best = best
    th, kp, Ls = _unpack(x_best, N, theta_bc, kappa_bc, kappa_max,
                         chords, v_lo_best, v_hi_best)
    kp = np.clip(kp, -kappa_max * 0.98, kappa_max * 0.98)
    kp[0], kp[-1] = kappa_bc
    return th, kp, Ls, norm_best


# ─────────────────────────────────────────────────────────────────────────────
# 종단 κ 감쇠 클로소이드
# ─────────────────────────────────────────────────────────────────────────────

def _terminal_decay(theta_end: float, kappa_end: float,
                    kappa_max: float, ds: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """종단 κ를 0으로 부드럽게 감쇠시키는 단일 클로소이드."""
    if abs(kappa_end) < 1e-6:
        return np.zeros((0, 2)), np.array([]), np.array([]), 0.0
    L = abs(kappa_end) / (kappa_max * 0.5)
    pts, th_arr, kp_arr = _clothoid_sample(theta_end, kappa_end, 0.0, L, ds)
    return pts, th_arr, kp_arr, L


# ─────────────────────────────────────────────────────────────────────────────
# Planner 클래스
# ─────────────────────────────────────────────────────────────────────────────

class Eta3ClothoidPlannerV3(BasePlanner):
    """η³-Clothoid Planner v3.1."""

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

        # ── 기체 한계 ─────────────────────────────────────────
        g = float(aircraft_params.get("gravity", 9.81))
        v_cruise = float(aircraft_params["v_cruise"])
        a_max_g = float(aircraft_params["a_max_g"])
        a_max = a_max_g * g * self.accel_tol
        kappa_max = a_max / (v_cruise ** 2)

        N_original = len(wps)
        wps_2d = wps[:, :2]

        # ── 경계 조건 ─────────────────────────────────────────
        theta0 = np.arctan2(wps[1, 1] - wps[0, 1], wps[1, 0] - wps[0, 0])
        theta_N = np.arctan2(wps[-1, 1] - wps[-2, 1], wps[-1, 0] - wps[-2, 0])
        if initial_state and "initial_heading" in initial_state:
            theta0 = float(initial_state["initial_heading"])
        kappa0 = 0.0
        kappa_N = 0.0

        # ── Stage 0: WP 사전 삽입 ─────────────────────────────
        th_pre = _initial_thetas(wps_2d, theta0, theta_N)
        # v3.1: 두 번째 반환값(삽입 후 θ)을 받아 NR 초기값으로 사용
        wps_2d, th_after_insert, orig_indices = _insert_wps_if_infeasible(
            wps_2d, th_pre, kappa_max)
        N = len(wps_2d)
        if self.verbose:
            n_ins = sum(1 for x in orig_indices if x < 0)
            print(f"[Stage 0] WP 삽입 {n_ins}개 (총 {N}개)")

        # ── Stage 1: 단일 통합 G2 NR ──────────────────────────
        if self.verbose:
            print("[Stage 1] 단일 통합 G2 NR")
        thetas, kappas, seg_Ls, nr_norm_final = _solve_g2_nr(
            wps_2d, kappa_max, theta0, kappa0, theta_N, kappa_N,
            th_init_full=th_after_insert,            # v3.1
            max_iter=self.nr_max_iter, tol=self.nr_tol,
            verbose=self.verbose,
        )
        if self.verbose:
            print(f"[Stage 1] 최종 |F|={nr_norm_final:.3e} "
                  f"(tol={self.nr_tol:.0e}, "
                  f"{'수렴' if nr_norm_final < self.nr_tol else '⚠ 미수렴'})")

        # ── Clothoid 샘플링 (v3.1: 누적 좌표로 연결, WP 점프 제거) ──
        all_pts:   list[np.ndarray] = []
        all_kappa: list[np.ndarray] = []
        wp_marks:  dict[int, int] = {}

        # 시작점
        cur_origin = wps_2d[0].copy()
        if orig_indices[0] >= 0:
            wp_marks[0] = orig_indices[0]

        for k in range(N - 1):
            seg_pts, _seg_th, seg_kp = _clothoid_sample(
                thetas[k], kappas[k], kappas[k + 1], seg_Ls[k], self.ds)
            # 로컬 → 전역: 이전 segment의 끝점(=현재 origin)을 기준
            seg_pts_global = seg_pts + cur_origin

            if k == 0:
                all_pts.append(seg_pts_global[:-1])
                all_kappa.append(seg_kp[:-1])
            else:
                # 이전 segment 끝점은 이미 추가됨 → 중복 제거 위해 [1:-1] 사용 시
                # 시작점이 빠짐. 대신 [:-1]을 쓰되, 이전에 끝점을 안 넣었으므로
                # 여기 시작점이 바로 그 끝점 역할.
                all_pts.append(seg_pts_global[:-1])
                all_kappa.append(seg_kp[:-1])

            # 다음 segment의 origin = 이번 segment의 실제 끝점 (NR 잔차 포함, 일관됨)
            cur_origin = seg_pts_global[-1]

            # WP 마커: 다음 segment 시작 인덱스가 곧 wps_2d[k+1]에 대응
            idx_next_wp = sum(len(p) for p in all_pts)
            if orig_indices[k + 1] >= 0:
                wp_marks[idx_next_wp] = orig_indices[k + 1]

        # 마지막 segment 끝점을 path에 포함 (위에서 [:-1]로 잘랐으므로 추가)
        all_pts.append(cur_origin.reshape(1, 2))
        all_kappa.append(np.array([kappas[-1]]))

        # 마지막 segment 끝 헤딩 = 마지막 클로소이드의 종단 헤딩
        th_terminal = thetas[-2] + 0.5 * (kappas[-2] + kappas[-1]) * seg_Ls[-1]

        # ── 종단 κ 감쇠 클로소이드 ────────────────────────────
        decay_pts, decay_th_arr, decay_kp, _ = _terminal_decay(
            th_terminal, kappas[-1], kappa_max, self.ds)

        if len(decay_pts) > 1:
            # v3.1: decay도 cur_origin(실제 마지막 segment 끝점) 기준으로 평행이동
            decay_pts_global = decay_pts + cur_origin - decay_pts[0]
            all_pts.append(decay_pts_global[1:])
            all_kappa.append(decay_kp[1:])
            terminal_pos = decay_pts_global[-1]
            terminal_th = float(decay_th_arr[-1])
        else:
            terminal_pos = cur_origin
            terminal_th = th_terminal

        # ── 종단 직선 연장 ────────────────────────────────────
        last_dir = np.array([np.cos(terminal_th), np.sin(terminal_th)])
        n_ext = max(2, int(self.end_extension / self.ds))
        ext_pts = (terminal_pos
                   + last_dir * np.linspace(0.0, self.end_extension, n_ext)[:, None])
        all_pts.append(ext_pts[1:])
        all_kappa.append(np.zeros(len(ext_pts) - 1))

        pts_arr = np.concatenate(all_pts,    axis=0)
        kappa_arr = np.concatenate(all_kappa,  axis=0)

        # ── 호 길이 ───────────────────────────────────────────
        diffs = np.diff(pts_arr, axis=0)
        s_arr = np.concatenate(
            [[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])

        # ── 고도 보간 ─────────────────────────────────────────
        sorted_marks = sorted(
            [(idx, wi) for idx, wi in wp_marks.items() if wi < N_original],
            key=lambda t: t[1])
        if len(sorted_marks) >= 2:
            wp_s_arr = np.array([s_arr[idx] for idx, _ in sorted_marks])
            wp_h_arr = np.array([wps[wi, 2] for _, wi in sorted_marks])
            alt_arr = np.interp(s_arr, wp_s_arr, wp_h_arr)
        else:
            alt_arr = np.full(len(pts_arr), wps[0, 2])

        # ── 방위각 ────────────────────────────────────────────
        chi_arr = np.zeros(len(pts_arr))
        for i in range(len(pts_arr) - 1):
            d = pts_arr[i + 1] - pts_arr[i]
            chi_arr[i] = np.arctan2(d[1], d[0])
        chi_arr[-1] = chi_arr[-2]

        # ── 상승각 ────────────────────────────────────────────
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
