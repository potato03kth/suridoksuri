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

v3.2 → v3.3 핵심 변경:
  • [BUGFIX] 2D 퇴화 WP(같은 수평 위치·고도만 다른 WP — 이륙점 수직 상방의
    천이고도 WP)를 병합 + 출력 s를 strictly increasing으로 보장 →
    np.gradient divide-by-zero(NaN setpoint)로 offboard 경로추종이 시작되지
    못하던 실기체 결함의 근본 수정. min_wp_chord(기본 0.1 m)로 병합 임계.
  • 경고 print에 flush=True — ros2 launch 파이프 버퍼링에서도 실시간 출력.

v3.3 → v3.4 핵심 변경 (2026-07-29, F-12 플래너 블로킹):
  • [PERF] `_solve_g2_nr` 의 계산량 중 **결과에 영향을 주지 않는 부분**을
    제거했다. 출력 경로는 그대로다(10개 시나리오 최대 위치차 1.4e-6 m).
      L자 3WP  v_cruise=18 : 19.59 s → 0.126 s (155배)
      폐회로 5WP v_cruise=18 : 110.7 s → 0.404 s (274배)
    세 가지였다:
      1) 야코비안이 **대역폭 5의 띠행렬**인데 열마다 전체 잔차를 다시 계산했다
         (N=33 에서 9024칸 중 비영 403칸). → `_jacobian_fd` 로 희소 계산,
         조밀 계산과 **비트 단위 동일**(`_seg_deps` 로 의존 블록 판정).
      2) `max_iter=60` 을 매번 전량 소진했다. 이 잔차계는 방정식 3(N−1)개에
         미지수 3N−5개로 **2개 과결정**이라 정확해가 없고 `|F| < tol` 이
         원리적으로 도달 불가인데, 기존 정체 판정이 `norm_F < 10*tol` 을
         and 조건으로 걸고 있어 영원히 발동하지 않았다. 실측상 |F| 는 10회에서
         멈춘다(31.73 → … → 11.6303 이후 불변). → 상대개선율 기반 정체 판정 +
         선탐색 실패(하강방향 아님) 시 종료.
      3) `clip_attempts` 3회 재시도가 완전히 헛돌았다 — 세그먼트 길이배율 v 가
         경계 (-0.5, 1.2) 에 한 번도 닿지 않아(실측 v∈[-0.06, 0.004]) 세 번의
         |F| 가 소수점 15자리까지 같았다. → clip 이 실제로 걸린 적이 있을
         때만 재시도(`clip_hit`).
    회귀 테스트: `tests/test_eta3_v3_nr_cost.py` (판정 기준은 벽시계가 아니라
    결정적인 `_fresnel_endpoint` 호출 수).

  ⚠️ 아직 남은 **알고리즘 결함**(이번 변경 범위 밖 — 경로가 바뀌므로 별도 결정):
    • `_insert_wps_if_infeasible` 의 이등분 삽입은 판정식
      `k_need = 2Δθ/chord` 를 **바꾸지 못한다** — 중점을 넣으면 Δθ 와 chord 가
      함께 절반이 되어 k_need 가 불변이다(실측: 4패스 내내 0.007854 고정,
      위반 세그먼트만 2→4→8→16→32). 그래서 한 번 발동하면 항상 max_insert 를
      다 태워 N 이 16배가 되고, 삽입점이 폴리라인 **위에** 놓이므로 경로가
      폴리라인에 못박혀 선회가 조인트 한 곳의 꺾임으로 몰린다
      (v_cruise=18 L자: max_insert 0→4 에서 폴리라인 이탈 15.06 m → 0.15 m,
      |κ|max 0.490 → 1.247, 헤딩 점프 46.7° → 85.9°).
    • 위 2)의 2개 과결정 때문에 NR 은 원리적으로 잔차 0 에 못 간다
      (실측 pos 잔차 5~36 m). affine 보정이 WP 통과만 살려낼 뿐,
      세그먼트 조인트마다 접선이 꺾여 |κ| 가 한계의 50~161배로 튄다.
"""
from __future__ import annotations
import time
import numpy as np
from .base_planner import BasePlanner, Path, PathPoint


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _wrap_arr(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """np.trapz/np.trapezoid 대체 — numpy 버전에 무관하게 동작(2.0에서 trapz 제거됨)."""
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0)


def _fresnel_endpoint(theta_i: float, kappa_i: float,
                      kappa_j: float, L: float,
                      n_quad: int = 400) -> np.ndarray:
    if L < 1e-12:
        return np.zeros(2)
    s = np.linspace(0.0, L, n_quad + 1)
    dk = (kappa_j - kappa_i) / L
    th = theta_i + kappa_i * s + 0.5 * dk * s * s
    return np.array([_trapz(np.cos(th), s), _trapz(np.sin(th), s)])


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
            v_lo: float, v_hi: float,
            clip_hit: list | None = None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """설계변수 x 를 (θ, κ, L) 로 푼다.

    `clip_hit` 이 주어지면 세그먼트 길이배율 v 가 [v_lo, v_hi] 밖으로 나가
    `np.clip` 이 실제로 값을 바꾼 적이 있는지를 `clip_hit[0]` 에 기록한다.
    한 번도 걸리지 않았다면 **더 넓은 clip 으로 다시 푸는 것은 같은 함수를
    같은 초기값에서 다시 푸는 것**이라 결과가 비트 단위로 같다 —
    `_solve_g2_nr` 의 재시도 생략 판정에 쓴다.
    """
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
    if clip_hit is not None and not clip_hit[0]:
        if v.size and (float(np.min(v)) < v_lo or float(np.max(v)) > v_hi):
            clip_hit[0] = True
    Ls = np.maximum(chords * np.exp(np.clip(v, v_lo, v_hi)), 1e-3)
    return th, kp, Ls


def _seg_residual(th: np.ndarray, kp: np.ndarray, Ls: np.ndarray,
                  wps: np.ndarray, w_head: float, k: int,
                  n_quad: int = 100) -> tuple[float, float, float]:
    """세그먼트 k 의 잔차 3성분. `_residual` 이 블록마다 부르는 것과 같은 식."""
    p = _fresnel_endpoint(th[k], kp[k], kp[k + 1], Ls[k], n_quad)
    target = wps[k + 1] - wps[k]
    th_end = th[k] + 0.5 * (kp[k] + kp[k + 1]) * Ls[k]
    return (p[0] - target[0],
            p[1] - target[1],
            w_head * _wrap(th_end - th[k + 1]))


def _residual(x: np.ndarray,
              wps: np.ndarray,
              theta_bc: tuple, kappa_bc: tuple,
              kappa_max: float,
              chords: np.ndarray,
              w_head: float,
              v_lo: float, v_hi: float,
              n_quad: int = 100,
              clip_hit: list | None = None) -> np.ndarray:
    N = len(wps)
    n_segs = N - 1
    th, kp, Ls = _unpack(x, N, theta_bc, kappa_bc,
                         kappa_max, chords, v_lo, v_hi, clip_hit)
    F = np.empty(3 * n_segs)
    for k in range(n_segs):
        F[3 * k], F[3 * k + 1], F[3 * k + 2] = _seg_residual(
            th, kp, Ls, wps, w_head, k, n_quad)
    return F


def _seg_deps(j: int, n_inner: int, n_segs: int) -> tuple[int, ...]:
    """설계변수 x[j] 가 영향을 주는 세그먼트(잔차 블록) 인덱스.

    잔차 블록 k 는 (θ_k, θ_{k+1}, κ_k, κ_{k+1}, L_k) 에만 의존한다 —
    즉 야코비안은 **대역폭 5의 띠행렬**이고, x[j] 하나를 흔들었을 때
    변하는 행은 최대 2블록(6행)뿐이다. 나머지 행은 유한차분식
    `(f(x+εe_j) − f(x))/ε` 의 분자가 **비트 단위로 0** 이므로 정확히 0이다.
    따라서 이 희소성을 쓴 야코비안은 조밀 계산과 비트 단위로 같고,
    Fresnel 적분 호출을 O(N²) → O(N) 으로 줄인다.
    """
    if j < n_inner:                     # θ_{j+1}  → 블록 j, j+1
        cand = (j, j + 1)
    elif j < 2 * n_inner:               # κ_{i+1}  → 블록 i, i+1
        i = j - n_inner
        cand = (i, i + 1)
    else:                               # L_k      → 블록 k
        cand = (j - 2 * n_inner,)
    return tuple(k for k in cand if 0 <= k < n_segs)


def _jacobian_fd(x: np.ndarray, F: np.ndarray, args: tuple,
                 eps_jac: float, n_quad: int = 100) -> np.ndarray:
    """희소성을 이용한 전방 유한차분 야코비안 (조밀 계산과 비트 단위 동일)."""
    (wps, theta_bc, kappa_bc, kappa_max, chords, w_head, v_lo, v_hi) = args
    N = len(wps)
    n_segs = N - 1
    n_inner = N - 2
    n_j = len(x)
    J = np.zeros((3 * n_segs, n_j))
    for j in range(n_j):
        ks = _seg_deps(j, n_inner, n_segs)
        if not ks:
            continue
        xp = x.copy()
        xp[j] += eps_jac
        th, kp, Ls = _unpack(xp, N, theta_bc, kappa_bc,
                             kappa_max, chords, v_lo, v_hi)
        for k in ks:
            r = _seg_residual(th, kp, Ls, wps, w_head, k, n_quad)
            J[3 * k, j] = (r[0] - F[3 * k]) / eps_jac
            J[3 * k + 1, j] = (r[1] - F[3 * k + 1]) / eps_jac
            J[3 * k + 2, j] = (r[2] - F[3 * k + 2]) / eps_jac
    return J


def _solve_g2_nr(wps: np.ndarray,
                 kappa_max: float,
                 theta0: float, kappa0: float,
                 theta_N: float, kappa_N: float,
                 th_init_full: np.ndarray | None = None,
                 max_iter: int = 60, tol: float = 1e-5,
                 eps_jac: float = 1e-6,
                 verbose: bool = False,
                 stall_rtol: float = 1e-6,
                 stall_patience: int = 2
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
        clip_hit = [False]
        prev_norm = np.inf
        stall = 0

        for it in range(max_iter):
            F = _residual(x, *args, clip_hit=clip_hit)
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

            J = _jacobian_fd(x, F, args, eps_jac)
            n_j = len(x)

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
            ls_ok = False
            for _bt in range(20):
                if np.linalg.norm(_residual(x + step * dx, *args,
                                            clip_hit=clip_hit)) \
                        <= (1.0 - c_armijo * step) * norm_F:
                    ls_ok = True
                    break
                step *= 0.5
            if not ls_ok:
                # 20회 반감(step≈1e-6)해도 Armijo 감소를 못 얻었다 = dx 는 이
                # 지점에서 하강방향이 아니다. 원본은 그래도 step·dx 를 더하고
                # 남은 반복을 전부 소진했지만, 그 갱신량은 |dx|의 1e-6 배라
                # 잔차를 바꾸지 못한다(실측: it≥10 이후 |F| 가 1e-10 이내로
                # 고정). 여기서 끊는다.
                break
            x += step * dx

            # ── 정체 판정 ────────────────────────────────────────────
            # 원래 조건 `abs(prev-cur)<1e-10 and norm_F < 10*tol` 은 **해가
            # 존재할 때만** 성립한다. 이 문제는 방정식 3(N−1)개에 미지수
            # 3N−5개로 **2개 과결정**이라 일반적으로 정확해가 없고 |F| 가
            # tol 에 도달하지 못한다 → 두 번째 절이 영원히 거짓 → 매번
            # max_iter 를 다 태웠다. 최소제곱 최소점에 앉은 것도 수렴이므로
            # **상대 개선율**로 판정한다.
            if np.isfinite(prev_norm):
                denom = max(prev_norm, 1e-300)
                if (prev_norm - norm_F) / denom < stall_rtol:
                    stall += 1
                    if stall >= stall_patience:
                        break
                else:
                    stall = 0
            prev_norm = norm_F

        F_final = _residual(x, *args)
        norm_final = float(np.linalg.norm(F_final))
        if best is None or norm_final < best[0]:
            best = (norm_final, x.copy(), v_lo, v_hi)

        if norm_final < tol:
            break

        if not clip_hit[0]:
            # 세그먼트 길이배율이 clip 경계에 **한 번도** 닿지 않았다면 더 넓은
            # clip 은 같은 함수를 같은 초기값에서 다시 푸는 것이라 궤적이
            # 비트 단위로 같다. 실측(v_cruise 17/18, N=33): v∈[-0.06, 0.004]
            # 로 경계 (-0.5, 1.2) 근처에도 못 가고 attempt 0/1/2 의 |F| 가
            # 소수점 15자리까지 동일했다 → 재시도 2회는 순수 3배 낭비였다.
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
                 min_wp_chord: float = 0.1,
                 verbose: bool = False):
        self.ds = ds
        self.accel_tol = accel_tol
        self.nr_tol = nr_tol
        self.nr_max_iter = nr_max_iter
        self.end_extension = end_extension
        self.min_wp_chord = min_wp_chord
        self.verbose = verbose

    def plan(self,
             waypoints_ned: np.ndarray,
             aircraft_params: dict,
             initial_state: dict | None = None) -> Path:
        t0 = time.perf_counter()
        wps = np.asarray(waypoints_ned, dtype=float)
        if len(wps) < 2:
            raise ValueError("waypoints는 최소 2개 필요")

        # ── 2D 퇴화 WP 병합 ─────────────────────────────────────────
        # 같은 수평 위치에 고도만 다른 WP(수직 상승)는 2D clothoid로 표현할
        # 수 없고, 그대로 두면 s가 같은 중복 경로점이 생겨 하류 np.gradient에서
        # divide-by-zero → NaN setpoint가 발생한다. 병합 시 고도는 나중 WP 값을
        # 채택한다(순항은 상승 완료 고도에서 시작). 수직 상승 구간은 planner가
        # 아니라 이륙/천이 상태기계가 담당한다.
        keep = [0]
        merged: list[int] = []
        for i in range(1, len(wps)):
            if np.linalg.norm(wps[i, :2] - wps[keep[-1], :2]) < self.min_wp_chord:
                wps[keep[-1], 2] = wps[i, 2]
                merged.append(i)
            else:
                keep.append(i)
        if merged:
            print(f"[Eta3ClothoidPlannerV3] WARNING: merged consecutive WPs {merged} "
                  f"within {self.min_wp_chord}m horizontal distance into the previous WP "
                  f"(altitude takes the later value); vertical climb must be handled by "
                  f"the takeoff/transition state machine, not the planner.",
                  flush=True)
            wps = wps[keep]
            if len(wps) < 2:
                raise ValueError(
                    "2D 퇴화 WP 병합 후 waypoints가 2개 미만입니다. "
                    "WP 목록이 사실상 한 지점입니다 — 미션 WP를 확인하세요.")

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
            print(f"[Stage 0] WP inserted {n_ins} (total {N})")

        if self.verbose:
            print("[Stage 1] unified G2 NR")
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
            print(f"[Eta3ClothoidPlannerV3] WARNING: NR pos residual {pos_res:.3f}m is large. "
                  f"affine correction guarantees WP passage but curve may be deformed.",
                  flush=True)

        all_pts:  list[np.ndarray] = []
        wp_marks: dict[int, int] = {}

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
            else:
                all_pts.append(seg_pts_global)

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

        pts_arr = np.concatenate(all_pts, axis=0)

        diffs = np.diff(pts_arr, axis=0)
        s_arr = np.concatenate(
            [[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])

        # ── s strictly increasing 보장 (최종 안전망) ─────────────────
        # ds=0 중복점이 남아 있으면 아래 곡률 재계산의 np.gradient(f, s)가
        # divide-by-zero → NaN 궤적을 만들어 FC가 setpoint를 거부한다. 중복점을
        # 제거하고 wp 마크는 살아남는 점으로 재매핑한다.
        keep_mask = np.concatenate([[True], np.diff(s_arr) > 1e-9])
        if not np.all(keep_mask):
            old_to_new = np.cumsum(keep_mask) - 1
            wp_marks = {int(old_to_new[idx]): wi for idx, wi in wp_marks.items()}
            pts_arr = pts_arr[keep_mask]
            s_arr = s_arr[keep_mask]

        # affine 보정 후 실제 기하학적 곡률 재계산 (설계 κ 대신 실제 경로 위치 기준)
        # NR 잔차가 클수록 설계 κ와 실제 κ의 괴리가 크므로, 하위 소비자(속도 프로필,
        # MPC 피드포워드)에 정확한 값을 전달하기 위해 유한차분으로 재계산한다.
        # 공식: κ = (N′·E″ − E′·N″) / |v|³  (iterpin_planner._compute_curvature 동일)
        _dN = np.gradient(pts_arr[:, 0], s_arr)
        _dE = np.gradient(pts_arr[:, 1], s_arr)
        _d2N = np.gradient(_dN, s_arr)
        _d2E = np.gradient(_dE, s_arr)
        _spd = np.sqrt(np.maximum(_dN**2 + _dE**2, 1e-24))
        kappa_arr = (_dN * _d2E - _dE * _d2N) / _spd**3

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
