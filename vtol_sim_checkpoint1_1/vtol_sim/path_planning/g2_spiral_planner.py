"""G2 Polynomial-Spiral Planner — eta3 재설계 "안 C" 프로토타입 (1단계)
================================================================================
설계 문서: `docs/planner_g2_redesign.md`

**이 모듈은 아직 아무 데도 연결되지 않은 나란한 프로토타입이다.**
`eta3clothoid_v3_1_planner.py` 를 교체하지 않는다 — 지표 비교용으로만 존재한다.

--------------------------------------------------------------------------------
1. 기존 플래너가 왜 원리적으로 실패했는가
--------------------------------------------------------------------------------
`_solve_g2_nr` 은 **전역 결합** 잔차계를 푼다: 미지수 3N−5 (θ 내부 N−2, κ 내부
N−2, 세그먼트 길이 N−1) 대 방정식 3(N−1). 즉 **항상 정확히 2개 과결정**이라
정확해가 존재하지 않는다. `|F| < tol` 은 원리적으로 도달 불가고, 최소제곱
최소점에 앉은 뒤 affine 보정으로 WP 통과만 억지로 맞춘다 → 조인트마다 접선이
꺾여 실측 |κ| 가 한계의 161배가 된다.

근본 원인은 **자유도 예산**이다. 세그먼트당 clothoid 1개(κ 선형)는 자유도가
길이 L 하나뿐인데(양 끝 κ 는 이웃과 공유), 양 끝 (θ, κ) 4개와 위치 2개를
동시에 맞추라는 요구를 받았다.

--------------------------------------------------------------------------------
2. 안 C — 세그먼트별 완전 분리 + 자유도 예산 정확히 맞추기
--------------------------------------------------------------------------------
**모든 WP 에서 κ = 0 으로 고정한다.** 그러면 세그먼트 k 의 경계조건이
`(P_k, θ_k, 0)` → `(P_{k+1}, θ_{k+1}, 0)` 로 **이웃과 완전히 분리**된다.
전역 뉴턴이 사라지고 세그먼트마다 독립 문제가 된다 → O(N).

세그먼트 안의 헤딩을 σ = s/L 의 다항으로 둔다:

    ψ(σ) = α + Δθ·H(σ) + Σ_j c_j · Q(σ)·T(σ)^j
      H = 3σ² − 2σ³   (H(0)=0, H(1)=1)
      Q = σ²(1−σ)²    (Q(0)=Q(1)=0, Q′(0)=Q′(1)=0)
      T = 2σ − 1
      α = θ_k − φ_k   (현 방향 대비 시작 헤딩)

Q 인자 덕분에 **어떤 c 를 넣어도** κ(0)=κ(L)=0 이 유지되고 총 선회량이 Δθ 로
보존된다(Q(1)−Q(0)=0). 즉 Δθ 는 닫힌형으로 흡수됐고 c 는 순수 "형상" 자유도다.
곡률은 κ(σ) = ψ′(σ)/L.

  **최소 형태 (c 가 c₀ 하나): 미지수 2 (c₀, L) = 방정식 2 (Δx, Δy)** — 정확히 일치.
  (기존: 미지수 3N−5 대 방정식 3N−3 → 2개 과결정)

게다가 세그먼트를 현(chord) 방향으로 회전시키면 두 방정식이 **삼각화**된다:

    S(c₀) ≡ ∫₀¹ sin ψ dσ = 0        ← c₀ 만의 **스칼라** 방정식
    L     = D / ∫₀¹ cos ψ dσ        ← 닫힌형

세그먼트당 **1변수 뉴턴**이고 야코비안도 해석적이다:
    ∂S/∂c_j = ∫₀¹ Q·T^j · cos ψ dσ
그래서 반복이 3~6회에 끝나고, 실패하면 실패했다고 **명시적으로** 말할 수 있다.

**폴리시(polish, 선택)**: c₁, c₂ … 를 더 열면 방정식은 그대로 2개인데 미지수가
늘어 **부족결정**이 된다. 남는 자유도를 min-max |κ| 로 쓴다(c₀ 는 매번 다시
풀어 Δy=0 을 유지하므로 WP 통과는 항상 정확하다). 직접 측정한 이득:

    A3_L seg0 (α=0, Δθ=45°, D=200m)  자유도 1 → 1.80×κmax,  2 → 1.43,  3 → 1.36
    SHORT30   (α=0, Δθ=45°, D=30m)   자유도 1 → 12.0,       2 → 9.55,  3 → 9.09

기본값 `polish_dof=2`. 폴리시는 **정사각 해에서 출발해 개선될 때만 채택**하므로
실패해도 정사각 해로 되돌아간다 — 품질은 좋아지되 견고성은 잃지 않는다.

왜 clothoid(κ 선형) 조각 여러 개가 아니라 다항 κ 인가:
  • 조각 n 개의 자유도는 (길이 n + 내부 κ n−1) = 2n−1, 방정식은 3 (Δx, Δy, Δθ).
    n=2 면 3=3 으로 맞지만 **κ 부호가 안 바뀌어 S자를 못 만든다**(아래 3장).
    n=3 이면 5>3 으로 이번엔 2개 부족결정이라 어차피 임의 조건이 필요하다.
  • 다항 κ 는 부호 변경을 자연히 허용하고, 세그먼트 안에서 κ 가 C¹ 이라
    롤레이트 요구가 계단이 아니다(clothoid 조각 이음매는 계단이다).

--------------------------------------------------------------------------------
3. 왜 S자가 반드시 필요한가 (n=2 clothoid 쌍이 안 되는 이유)
--------------------------------------------------------------------------------
κ 부호가 일정하면 θ(s) 가 단조라서 **현 방향 φ 가 반드시 θ_k 와 θ_{k+1} 사이에
있어야** 한다. 그런데 첫 세그먼트는 θ_0 = φ_0 (기체가 첫 구간 방향으로 진입)
이라 경계에 걸리고, 마지막 세그먼트도 마찬가지다. L자 3WP 가 정확히 이 경우다:
θ_0=0°, φ_0=0°, θ_1=45° → 단조 κ 로는 끝점 (200,0) 에 도달할 수 없다.
∫ sin θ ds = 0 인데 θ 가 0→45° 로 단조면 적분이 양수이기 때문이다.
→ 경로는 **먼저 현 아래로 내려갔다가 올라와야** 한다. 이 "필연적 S자" 를
표현할 수 있어야 하고, c₀ 가 바로 그 자유도다.

--------------------------------------------------------------------------------
4. 해의 존재 조건 — 그리고 존재하지 않을 때
--------------------------------------------------------------------------------
(a) **기하 해의 존재**: S(c₀)=0 의 근. 실용 영역에서는 거의 항상 있다.
    없으면 `SegmentSolution.ok=False` 로 **조용히 넘어가지 않고** 보고한다.

(b) **곡률 제한의 만족은 별개 문제이고, 이 대회 경로에서는 만족되지 않는다.**
    "WP 정확 통과 + G2 + |κ|≤κmax" 셋이 동시에 성립하는지는 플래너 실력이
    아니라 **기하학**이 정한다. 코너를 되돌지 않고(loop 없이) 지나는 경로의
    이론 하한을 Pontryagin 으로 구할 수 있다 — 최적 제어가 bang-bang 이고
    특이(직선) 호가 없으므로 구조가 유일하게 정해진다:

        코너각 Δ 를 양쪽 세그먼트가 절반씩 나눠 갖는 대칭해에서
        한쪽 반문제(현 길이 D, 시작 헤딩 = 현 방향, 끝 헤딩 = 현+Δ/2)의
        최소 곡률은   κ_floor = f(Δ/2) / D,
        f(δ) = 2·sin(arccos((1+cos δ)/2)) + sin δ          ← 직접 유도·수치확인
        Δ=90° (δ=45°) 에서 f = 1.749118.

    직접 측정(v_cruise=18, a_max_g=0.3, accel_tol=0.85 → κmax=0.007721, R=129.5m,
    `tools/planner/compare_path_quality.py`):

        A3_L      200m 다리 90° : G1 이론하한 1.13×κmax,  본 플래너 1.42×
        R2_closed 200m 사각     : G1 이론하한 1.13×κmax,  본 플래너 1.25×
        S_L30     30m 다리 90°  : G1 이론하한 7.55×κmax,  본 플래너 9.43×

    **이론 하한부터 이미 1 을 넘는다.** 즉 이건 플래너의 결함이 아니라
    "18 m/s 로는 그 코너를 WP 를 밟으면서 돌 수 없다"는 물리다. 그래서 이
    플래너는 세 가지를 **명시**한다:
      • `plan()` 이 `self.last_report`(FeasibilityReport)에 세그먼트별 κ 비율을 남긴다.
      • 비율 > 1 이면 경고를 출력하고, **날 수 있는 속도**를 같이 알려준다.
        κmax = a_max/v² 이므로 v_feasible = sqrt(a_max / |κ|max).
      • `speed_policy="curvature"`(기본) 이면 v_ref 를 곡률에 맞춰 낮춰 내보낸다.
        하한은 `v_min_turn`. 하한으로도 모자라면 `speed_floor_violated=True` 다.

(c) **짧은 경로의 종단 감속**: 마지막 WP 에서 역천이하려면 미리 감속해야 하는데
    경로가 짧으면 거리가 안 나온다. `decel_dist`(기본 80 m) 대비 전장을 검사해
    `report.decel_margin_m` 로 보고한다. 음수면 경고한다.

--------------------------------------------------------------------------------
5. κ=0 을 WP 에 강제하는 대가 (정직하게)
--------------------------------------------------------------------------------
같은 방향으로 계속 도는 WP 열(원호 위의 WP 등)에서는 손해다. 반지름 R 원 위에
등간격으로 놓인 WP 를 지나는 이상적 경로는 κ≡1/R 인데, WP 마다 κ=0 을 강요하면
세그먼트마다 0→peak→0 이 되어

    peak = 1.5·Δθ / L = 1.5·Δθ / (R·Δθ) = **1.5 / R**   (간격 무관 상수 1.5배)

정사각 해(폴리시 없음)가 정확히 이 1.5배이고, 폴리시가 조금 깎아 **실측 1.37~1.46배**
다(`compare_path_quality.py --kappa-zero-penalty`: R=200/7WP 1.373, R=200/13WP 1.463,
R=400/7WP 1.374, R=150/9WP 1.377).
반대로 **직각 코너처럼 좌우 대칭인 꺾임에서는 κ=0 이 최적**이다 — 문제가 코너
이등분선에 대해 대칭이면 해도 대칭이고, 반사가 κ 의 부호를 뒤집으므로
κ(코너) = −κ(코너) ⇒ κ=0 이다. 이번 대회 경로(직사각/L자)는 후자다.
연속 동방향 선회가 많은 경로가 오면 κ 자유화가 필요하다(§7 확장 경로).

--------------------------------------------------------------------------------
6. 재사용
--------------------------------------------------------------------------------
`15f7fa4` 가 넣은 희소 야코비안·재시도 생략은 **전역 뉴턴을 전제**한 최적화라
여기엔 붙일 자리가 없다(전역 뉴턴 자체가 사라졌다). 대신 그 커밋의 교훈
— "결과에 영향 없는 반복을 태우지 말 것" — 을 구조로 가져왔다:
세그먼트당 해석 야코비안 1변수 뉴턴은 3~6회에 끝나고 정체 판정이 필요 없다.
numpy 만 쓰고 `np.trapz` 는 쓰지 않는다(2.x 제거됨, `0785777` 재발 방지).

--------------------------------------------------------------------------------
7. 아직 안 한 것 (2단계 후보)
--------------------------------------------------------------------------------
  • 내부 WP 의 κ 자유화 (동방향 연속 선회의 1.5배 손해 제거).
  • 곡률 한계를 못 지킬 때 **WP 통과를 포기하고 코너를 자르는** 대안 정책
    (`policy="relax"`). 지금은 v_ref 감속만 제공한다 — 어느 쪽을 택할지는
    사용자 판단 사안이라 구현하지 않았다.
  • `run_scenario.py`/`planner_runner.py` 등록 및 SITL 재검증.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .base_planner import BasePlanner, Path, PathPoint

# ─────────────────────────────────────────────────────────────────────────────
# 형상 기저 — σ ∈ [0, 1] 고정 격자 (모듈 로드 시 1회 계산)
# ─────────────────────────────────────────────────────────────────────────────
_NQ = 256                       # 짝수 필수 (합성 Simpson). 128~512 사이에서
                                # 품질(|κ| 비율)은 소수점 3자리까지 동일하고 비용만 는다 —
                                # 직접 측정(tools/planner/compare_path_quality.py --tune).
_MAX_DOF = 5                    # 미리 만들어 두는 형상 기저 개수
_SIG = np.linspace(0.0, 1.0, _NQ + 1)
_H = 3.0 * _SIG ** 2 - 2.0 * _SIG ** 3          # smoothstep
_dH = 6.0 * _SIG * (1.0 - _SIG)
_Q = _SIG ** 2 * (1.0 - _SIG) ** 2
_T = 2.0 * _SIG - 1.0

# 형상 기저 B_j = Q·T^j 와 그 도함수 (κ 계산용)
_B = np.array([_Q * _T ** j for j in range(_MAX_DOF)])
_dB = np.empty_like(_B)
_dQ = 2.0 * _SIG * (1.0 - _SIG) * (1.0 - 2.0 * _SIG)     # = -2σ(1−σ)T
for _j in range(_MAX_DOF):
    _dB[_j] = _dQ * _T ** _j
    if _j > 0:
        _dB[_j] = _dB[_j] + _Q * (2.0 * _j) * _T ** (_j - 1)

# 합성 Simpson 가중 (∫₀¹ f dσ ≈ _W·f)
_W = np.ones(_NQ + 1)
_W[1:-1:2] = 4.0
_W[2:-1:2] = 2.0
_W *= (1.0 / _NQ) / 3.0
_WB = _W * _B                    # ∂S/∂c_j 용


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _psi(alpha: float, dtheta: float, c: np.ndarray) -> np.ndarray:
    """세그먼트 국소 헤딩. ψ(0)=α, ψ(1)=α+Δθ — c 와 무관하게 성립."""
    return alpha + dtheta * _H + c @ _B[:len(c)]


def _S(alpha: float, dtheta: float, c: np.ndarray) -> float:
    return float(_W @ np.sin(_psi(alpha, dtheta, c)))


def _S_dS0(alpha: float, dtheta: float, c: np.ndarray) -> tuple[float, float]:
    """S(c) 와 ∂S/∂c₀ = ∫ Q cos ψ dσ."""
    p = _psi(alpha, dtheta, c)
    return float(_W @ np.sin(p)), float(_WB[0] @ np.cos(p))


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SegmentSolution:
    """세그먼트 하나의 해와 진단."""
    ok: bool = False
    reason: str = ""
    c: np.ndarray = field(default_factory=lambda: np.zeros(1))
    L: float = 0.0
    kappa_peak: float = 0.0     # |κ|max
    kappa_signed: float = 0.0   # 부호 있는 극값 (보고용)
    C: float = 1.0              # ∫cos ψ dσ (= D / L)
    iters: int = 0
    polished: bool = False


def _close_c0(alpha: float, dtheta: float, c: np.ndarray,
              tol: float = 1e-13, max_iter: int = 40) -> np.ndarray | None:
    """c₁… 를 고정한 채 c₀ 로 ∫sin ψ = 0 을 만족시킨다 (보호된 뉴턴 → 이분).

    이 함수가 성공하는 한 **WP 통과는 정확**하다 — 끝점 y 성분이 0 이고
    x 성분은 L = D/∫cos 로 닫아 주기 때문이다. affine 보정이 필요 없는 이유.
    """
    c = np.array(c, dtype=float)
    for _ in range(max_iter):
        S, dS = _S_dS0(alpha, dtheta, c)      # dS = ∂S/∂c₀
        if abs(S) < tol:
            return c
        if abs(dS) < 1e-14:
            break
        step = float(np.clip(-S / dS, -50.0, 50.0))
        c_try = c.copy()
        c_try[0] = c[0] + step
        S_new = _S(alpha, dtheta, c_try)
        bt = 0
        while abs(S_new) > abs(S) and bt < 40:   # backtracking
            step *= 0.5
            c_try[0] = c[0] + step
            S_new = _S(alpha, dtheta, c_try)
            bt += 1
        if bt >= 40:
            break
        c = c_try
    # 뉴턴 실패 → 확장 브래킷 + 이분
    return _bisect_c0(alpha, dtheta, c, tol)


def _bisect_c0(alpha: float, dtheta: float, c: np.ndarray,
               tol: float) -> np.ndarray | None:
    c = np.array(c, dtype=float)
    c0 = c.copy()
    c0[0] = 0.0
    S0 = _S(alpha, dtheta, c0)
    if abs(S0) < tol:
        return c0
    for sgn in ((-1.0, 1.0) if S0 > 0 else (1.0, -1.0)):
        lo, flo, h = 0.0, S0, 1.0
        for _ in range(45):
            hi = sgn * h
            ct = c0.copy()
            ct[0] = hi
            fhi = _S(alpha, dtheta, ct)
            if flo * fhi <= 0.0:
                for _ in range(120):
                    mid = 0.5 * (lo + hi)
                    ct[0] = mid
                    fm = _S(alpha, dtheta, ct)
                    if flo * fm <= 0.0:
                        hi = mid
                    else:
                        lo, flo = mid, fm
                    if abs(hi - lo) < 1e-14 * max(1.0, abs(hi)):
                        break
                ct[0] = 0.5 * (lo + hi)
                return ct
            lo, flo = hi, fhi
            h *= 1.6
    return None


def _finish(alpha: float, dtheta: float, D: float,
            c: np.ndarray, iters: int, polished: bool) -> SegmentSolution | None:
    """c 가 정해진 뒤 L 과 κ 를 닫는다."""
    p = _psi(alpha, dtheta, c)
    C = float(_W @ np.cos(p))
    if C <= 1e-6:
        return None
    L = D / C
    kap = (dtheta * _dH + c @ _dB[:len(c)]) / L
    i = int(np.argmax(np.abs(kap)))
    return SegmentSolution(ok=True, c=c, L=L, kappa_peak=float(abs(kap[i])),
                           kappa_signed=float(kap[i]), C=C, iters=iters,
                           polished=polished)


def solve_segment(alpha: float, dtheta: float, D: float,
                  polish_dof: int = 0,
                  polish_rounds: int = 3,
                  polish_grid: int = 7,
                  polish_span: float = 12.0) -> SegmentSolution:
    """세그먼트 해.

    Parameters
    ----------
    alpha      : 시작 헤딩 − 현 방향 (rad)
    dtheta     : 끝 헤딩 − 시작 헤딩 (rad), wrap 된 값
    D          : 현 길이 (m) > 0
    polish_dof : 추가 형상 자유도 개수 (0 이면 정사각 해 그대로).
                 c₁…c_{polish_dof} 를 좌표하강으로 흔들어 |κ|max 를 낮춘다.

    Returns
    -------
    SegmentSolution — `ok=False` 면 `reason` 에 왜 못 풀었는지가 들어 있다.
    **절대 조용히 근사해를 돌려주지 않는다.**
    """
    if D <= 1e-9:
        return SegmentSolution(ok=False, reason="chord<=0")

    n_dof = 1 + max(0, min(int(polish_dof), _MAX_DOF - 1))

    # 직선 특수해 — α=Δθ=0 이면 c=0 이 정확해다(S(0)=0).
    if abs(alpha) < 1e-14 and abs(dtheta) < 1e-14:
        return SegmentSolution(ok=True, reason="straight", c=np.zeros(n_dof),
                               L=D, kappa_peak=0.0, kappa_signed=0.0,
                               C=1.0, iters=0)

    # ── 1) 정사각 해 (c₀ 만) ──────────────────────────────────────────
    c = _close_c0(alpha, dtheta, np.zeros(n_dof))
    if c is None:
        return SegmentSolution(
            ok=False,
            reason="S(c₀)=0 의 근 없음 — 이 경계조건은 단일 스파이럴로 "
                   "표현할 수 없다(헤딩 배정을 바꾸거나 WP 를 나눠야 한다)")
    base = _finish(alpha, dtheta, D, c, 1, False)
    if base is None:
        return SegmentSolution(
            ok=False, c=c,
            reason="∫cos ψ ≤ 0 — 경로가 현 방향으로 전진하지 못한다"
                   "(끝점 도달 불가). 꺾임이 과도하거나 세그먼트가 너무 짧다.")
    if n_dof == 1:
        return base

    # ── 2) 폴리시 — 남는 자유도로 min-max |κ| ─────────────────────────
    best, span = base, float(polish_span)
    for _r in range(int(polish_rounds)):
        improved = True
        while improved:
            improved = False
            for j in range(1, n_dof):
                for v in best.c[j] + np.linspace(-span, span, int(polish_grid)):
                    trial = best.c.copy()
                    trial[j] = v
                    cc = _close_c0(alpha, dtheta, trial)
                    if cc is None:
                        continue
                    cand = _finish(alpha, dtheta, D, cc, 1, True)
                    if cand is not None and cand.kappa_peak < best.kappa_peak * (1 - 1e-9):
                        best, improved = cand, True
        span *= 0.35
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 샘플링 — 해를 실제 경로점으로
# ─────────────────────────────────────────────────────────────────────────────
def _cum_integrate(f: np.ndarray, h: float) -> np.ndarray:
    """누적 적분 (3점 Newton–Cotes, 국소 O(h⁴)).

    `np.trapz`/`np.trapezoid` 를 쓰지 않는다 — numpy 2.0 에서 `trapz` 가
    제거돼 실기체가 죽은 이력이 있다(`0785777`). 사다리꼴 누적은 끝점 오차가
    O(h²)라 WP 통과 오차가 mm 대로 남는데, 이 규칙은 μm 대로 줄인다.
    """
    n = len(f)
    inc = np.empty(n - 1)
    inc[:-1] = h / 12.0 * (5.0 * f[:-2] + 8.0 * f[1:-1] - f[2:])
    inc[-1] = h / 12.0 * (-f[-3] + 8.0 * f[-2] + 5.0 * f[-1])
    out = np.empty(n)
    out[0] = 0.0
    np.cumsum(inc, out=out[1:])
    return out


def sample_segment(theta_k: float, dtheta: float, sol: SegmentSolution,
                   ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """세그먼트를 ds 간격 경로점으로. 시작점 원점, 시작 헤딩 theta_k.

    Returns (xy(n,2), theta(n), kappa(n), s(n)) — s[0]=0, s[-1]=L.
    """
    L, c = sol.L, sol.c
    n_out0 = max(2, int(np.ceil(L / max(ds, 1e-6))) + 1)
    n_fine = max(1024, 8 * n_out0)      # 8배 과표본 — 중간점 선형보간 오차를 μm 대로
    if n_fine % 2:
        n_fine += 1
    sig = np.linspace(0.0, 1.0, n_fine + 1)
    q = sig ** 2 * (1.0 - sig) ** 2
    t = 2.0 * sig - 1.0
    hh = 3.0 * sig ** 2 - 2.0 * sig ** 3
    dhh = 6.0 * sig * (1.0 - sig)
    dq = 2.0 * sig * (1.0 - sig) * (1.0 - 2.0 * sig)

    th_f = theta_k + dtheta * hh
    dth_f = dtheta * dhh
    for j, cj in enumerate(c):
        th_f = th_f + cj * q * t ** j
        db = dq * t ** j
        if j > 0:
            db = db + q * (2.0 * j) * t ** (j - 1)
        dth_f = dth_f + cj * db

    h = L / n_fine
    x_f = _cum_integrate(np.cos(th_f), h)
    y_f = _cum_integrate(np.sin(th_f), h)
    s_f = sig * L

    n_out = max(2, int(np.ceil(L / ds)) + 1)
    s_out = np.linspace(0.0, L, n_out)
    x = np.interp(s_out, s_f, x_f)
    y = np.interp(s_out, s_f, y_f)
    x[-1], y[-1] = x_f[-1], y_f[-1]          # 끝점은 적분값 그대로 (WP 통과)
    th = np.interp(s_out, s_f, th_f)
    kap = np.interp(s_out, s_f, dth_f) / L
    return np.column_stack([x, y]), th, kap, s_out


# ─────────────────────────────────────────────────────────────────────────────
# 헤딩 배정
# ─────────────────────────────────────────────────────────────────────────────
def initial_headings(wps2d: np.ndarray) -> np.ndarray:
    """길이 가중 이등분선. 양 끝은 인접 구간 방향으로 고정한다."""
    N = len(wps2d)
    phi = np.array([np.arctan2(wps2d[k + 1, 1] - wps2d[k, 1],
                               wps2d[k + 1, 0] - wps2d[k, 0])
                    for k in range(N - 1)])
    th = np.empty(N)
    th[0], th[-1] = phi[0], phi[-1]
    for i in range(1, N - 1):
        L_in = np.linalg.norm(wps2d[i] - wps2d[i - 1])
        L_out = np.linalg.norm(wps2d[i + 1] - wps2d[i])
        w_in = 1.0 / np.sqrt(max(L_in, 1e-9))
        w_out = 1.0 / np.sqrt(max(L_out, 1e-9))
        vx = w_in * np.cos(phi[i - 1]) + w_out * np.cos(phi[i])
        vy = w_in * np.sin(phi[i - 1]) + w_out * np.sin(phi[i])
        th[i] = phi[i] if (vx * vx + vy * vy) < 1e-18 else np.arctan2(vy, vx)
    return th


def refine_headings(th: np.ndarray, phi: np.ndarray, D: np.ndarray,
                    sweeps: int = 3, span_deg: float = 45.0) -> np.ndarray:
    """내부 WP 헤딩을 좌표하강으로 다듬어 max|κ| 를 낮춘다.

    각 내부 WP 는 이웃 두 세그먼트에만 영향을 준다(κ=0 분리의 직접적 이득)
    → 국소 min-max 1차원 문제. 축소 탐색 3라운드면 충분하고 O(N)·상수배다.
    비용 절감을 위해 여기서는 **폴리시 없는 정사각 해**로 평가한다
    (순위는 거의 보존되고, 최종 경로는 폴리시된 해로 다시 만든다).
    """
    th = th.copy()
    N = len(th)
    if N < 3:
        return th

    def peak(i: int) -> float:
        s = solve_segment(_wrap(th[i] - phi[i]), _wrap(th[i + 1] - th[i]), D[i])
        return np.inf if not s.ok else s.kappa_peak

    for _ in range(int(sweeps)):
        for i in range(1, N - 1):
            base, span = th[i], np.radians(span_deg)
            for _round in range(3):
                best_c, best_v = base, None
                for cand in base + np.linspace(-span, span, 9):
                    th[i] = cand
                    a, b = peak(i - 1), peak(i)
                    v = (max(a, b), a + b)
                    if best_v is None or v < best_v:
                        best_v, best_c = v, cand
                base = best_c
                span *= 0.3
            th[i] = base
    return th


# ─────────────────────────────────────────────────────────────────────────────
# 진단 보고서
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FeasibilityReport:
    kappa_max: float = 0.0
    v_cruise: float = 0.0
    a_max: float = 0.0                     # m/s² (accel_tol 반영)
    seg_kappa_peak: list = field(default_factory=list)
    seg_ratio: list = field(default_factory=list)
    seg_L: list = field(default_factory=list)
    seg_ok: list = field(default_factory=list)
    seg_reason: list = field(default_factory=list)
    worst_ratio: float = 0.0
    worst_seg: int = -1
    v_feasible: float = 0.0                # 이 기하를 날 수 있는 최대 속도
    v_min_turn: float = 0.0
    speed_floor_violated: bool = False
    a_req_g_at_vmin: float = 0.0           # v_min_turn 으로 날 때 필요한 횡가속 (g)
    total_length: float = 0.0
    polyline_length: float = 0.0
    decel_dist_req: float = 0.0
    decel_margin_m: float = 0.0
    solver_iters: int = 0

    @property
    def curvature_feasible(self) -> bool:
        return bool(self.seg_ok) and all(self.seg_ok) and self.worst_ratio <= 1.0

    def lines(self) -> list[str]:
        out = [
            f"κmax={self.kappa_max:.6f} 1/m (R_min={1.0/max(self.kappa_max,1e-12):.1f} m)"
            f"  v_cruise={self.v_cruise:.1f} m/s  a_max={self.a_max:.3f} m/s²",
            f"전장 {self.total_length:.1f} m (폴리라인 {self.polyline_length:.1f} m)",
        ]
        for k, (pk, r, L, ok, why) in enumerate(zip(
                self.seg_kappa_peak, self.seg_ratio, self.seg_L,
                self.seg_ok, self.seg_reason)):
            mark = "OK " if (ok and r <= 1.0) else ("!! " if ok else "XX ")
            out.append(f"  {mark}seg{k}: L={L:8.2f} m  |κ|max={pk:.6f} "
                       f"({r:7.3f}× κmax){'  ' + why if why else ''}")
        out.append(f"최악 seg{self.worst_seg}: {self.worst_ratio:.3f}× κmax → "
                   f"이 기하를 날 수 있는 속도 v ≤ {self.v_feasible:.2f} m/s "
                   f"(v_min_turn={self.v_min_turn:.1f}"
                   f"{', 하한 미달!' if self.speed_floor_violated else ''})")
        out.append(f"v_min_turn={self.v_min_turn:.1f} m/s 로 날면 필요 횡가속 "
                   f"{self.a_req_g_at_vmin:.3f} g")
        out.append(f"종단 감속 여유 {self.decel_margin_m:+.1f} m "
                   f"(필요 {self.decel_dist_req:.1f} m)")
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 플래너
# ─────────────────────────────────────────────────────────────────────────────
class G2SpiralPlanner(BasePlanner):
    """WP 에서 κ=0 인 다항-곡률 스파이럴 스플라인. 세그먼트별 1변수 뉴턴.

    Parameters
    ----------
    ds : 경로점 간격 (m)
    accel_tol : κmax = a_max_g·g·accel_tol / v_cruise² 의 여유율
    heading_sweeps : 내부 WP 헤딩 좌표하강 반복 (0 이면 이등분선 그대로)
    polish_dof : 세그먼트 형상 추가 자유도 (0=정사각, 2=기본)
    speed_policy : "curvature"(기본) → v_ref = min(v_cruise, sqrt(a_max/|κ|)),
                   하한 v_min_turn. "constant" → v_ref = v_cruise (기존 동작).
    on_infeasible : "warn"(기본) | "raise" | "silent"
                    곡률 한계를 못 지킬 때의 행동. 기본은 **크게 출력**한다 —
                    기존 플래너가 조용히 변형된 곡선을 낸 것이 이번 재설계의
                    출발점이었다.
    end_extension : 마지막 WP 이후 직선 연장 (m). κ_N=0 이라 감쇠 구간이 없다.
    """

    def __init__(self,
                 ds: float = 1.0,
                 accel_tol: float = 0.9,
                 heading_sweeps: int = 2,
                 polish_dof: int = 2,
                 speed_policy: str = "curvature",
                 on_infeasible: str = "warn",
                 end_extension: float = 15.0,
                 min_wp_chord: float = 0.1,
                 verbose: bool = False):
        self.ds = float(ds)
        self.accel_tol = float(accel_tol)
        self.heading_sweeps = int(heading_sweeps)
        self.polish_dof = int(polish_dof)
        self.speed_policy = str(speed_policy)
        self.on_infeasible = str(on_infeasible)
        self.end_extension = float(end_extension)
        self.min_wp_chord = float(min_wp_chord)
        self.verbose = bool(verbose)
        self.last_report: FeasibilityReport | None = None
        self.last_headings: np.ndarray | None = None

    # ── 메인 ─────────────────────────────────────────────────────────
    def plan(self,
             waypoints_ned: np.ndarray,
             aircraft_params: dict,
             initial_state: dict | None = None) -> Path:
        t0 = time.perf_counter()
        wps = np.asarray(waypoints_ned, dtype=float).copy()
        if len(wps) < 2:
            raise ValueError("waypoints는 최소 2개 필요")
        wps = self._merge_degenerate(wps)

        g = float(aircraft_params.get("gravity", 9.81))
        v_cruise = float(aircraft_params["v_cruise"])
        a_max_g = float(aircraft_params["a_max_g"])
        a_max = a_max_g * g * self.accel_tol
        kappa_max = a_max / (v_cruise ** 2)
        v_min_turn = float(aircraft_params.get(
            "v_min_turn", aircraft_params.get("v_stall", 0.0) * 1.1))

        N = len(wps)
        wps2d = wps[:, :2]
        phi = np.array([np.arctan2(wps2d[k + 1, 1] - wps2d[k, 1],
                                   wps2d[k + 1, 0] - wps2d[k, 0])
                        for k in range(N - 1)])
        D = np.array([np.linalg.norm(wps2d[k + 1] - wps2d[k])
                      for k in range(N - 1)])

        th = initial_headings(wps2d)
        if initial_state and "initial_heading" in initial_state:
            th[0] = float(initial_state["initial_heading"])
        if self.heading_sweeps > 0:
            th = refine_headings(th, phi, D, sweeps=self.heading_sweeps)
        self.last_headings = th.copy()

        sols = [solve_segment(_wrap(th[k] - phi[k]), _wrap(th[k + 1] - th[k]),
                              D[k], polish_dof=self.polish_dof)
                for k in range(N - 1)]

        rep = FeasibilityReport(kappa_max=kappa_max, v_cruise=v_cruise,
                                a_max=a_max, v_min_turn=v_min_turn,
                                polyline_length=float(D.sum()))
        for s in sols:
            rep.seg_kappa_peak.append(s.kappa_peak)
            rep.seg_ratio.append(s.kappa_peak / kappa_max if s.ok else float("inf"))
            rep.seg_L.append(s.L)
            rep.seg_ok.append(s.ok)
            rep.seg_reason.append(s.reason)
            rep.solver_iters += s.iters
        rep.worst_seg = int(np.argmax(rep.seg_ratio))
        rep.worst_ratio = float(rep.seg_ratio[rep.worst_seg])
        peak_all = max((s.kappa_peak for s in sols if s.ok), default=0.0)
        rep.v_feasible = (float(np.sqrt(a_max / peak_all)) if peak_all > 1e-12
                          else float("inf"))
        rep.speed_floor_violated = bool(rep.v_feasible < v_min_turn)
        rep.a_req_g_at_vmin = float(peak_all * v_min_turn ** 2 / g)

        bad = [k for k, s in enumerate(sols) if not s.ok]
        if bad and self.on_infeasible != "silent":
            print("[G2SpiralPlanner] 해 없음 — 세그먼트 "
                  + ", ".join(f"{k}({sols[k].reason})" for k in bad), flush=True)

        # ── 경로점 생성 ─────────────────────────────────────────────
        pts_list, th_list, kap_list = [], [], []
        wp_marks: dict[int, int] = {0: 0}
        cursor = 0
        for k in range(N - 1):
            s = sols[k]
            if not s.ok:
                # 못 푼 세그먼트는 직선으로 잇되 **보고서에 XX 로 남긴다**.
                n = max(2, int(np.ceil(D[k] / self.ds)) + 1)
                t = np.linspace(0.0, 1.0, n)[:, None]
                xy = wps2d[k] + t * (wps2d[k + 1] - wps2d[k])
                thv, kpv = np.full(n, phi[k]), np.zeros(n)
            else:
                xy_loc, thv, kpv, _ = sample_segment(
                    th[k], _wrap(th[k + 1] - th[k]), s, self.ds)
                xy = xy_loc + wps2d[k]
            take = slice(None, -1) if k < N - 2 else slice(None)
            pts_list.append(xy[take])
            th_list.append(thv[take])
            kap_list.append(kpv[take])
            cursor += len(xy) - 1
            wp_marks[cursor] = k + 1

        if self.end_extension > 0.0:
            last = pts_list[-1][-1]
            d = np.array([np.cos(th[-1]), np.sin(th[-1])])
            n_ext = max(2, int(self.end_extension / self.ds) + 1)
            t = np.linspace(0.0, self.end_extension, n_ext)[1:, None]
            pts_list.append(last + t * d)
            th_list.append(np.full(n_ext - 1, th[-1]))
            kap_list.append(np.zeros(n_ext - 1))

        pts = np.concatenate(pts_list, axis=0)
        th_arr = np.concatenate(th_list)
        kap_arr = np.concatenate(kap_list)

        diffs = np.diff(pts, axis=0)
        s_arr = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
        keep = np.concatenate([[True], np.diff(s_arr) > 1e-9])
        if not np.all(keep):
            old_to_new = np.cumsum(keep) - 1
            wp_marks = {int(old_to_new[i]): wi for i, wi in wp_marks.items()}
            pts, s_arr = pts[keep], s_arr[keep]
            th_arr, kap_arr = th_arr[keep], kap_arr[keep]

        rep.total_length = float(s_arr[-1])
        rep.decel_dist_req = float(aircraft_params.get("decel_dist", 80.0))
        rep.decel_margin_m = rep.total_length - rep.decel_dist_req
        self.last_report = rep

        if self.on_infeasible != "silent" and (
                not rep.curvature_feasible or rep.decel_margin_m < 0.0):
            print("[G2SpiralPlanner] ⚠ 이 경로는 주어진 제약을 만족하지 않는다:",
                  flush=True)
            for ln in rep.lines():
                print("    " + ln, flush=True)
        elif self.verbose:
            for ln in rep.lines():
                print("[G2SpiralPlanner] " + ln, flush=True)
        if self.on_infeasible == "raise" and not rep.curvature_feasible:
            raise ValueError(f"[G2SpiralPlanner] 곡률 한계 초과 "
                             f"{rep.worst_ratio:.3f}× (seg{rep.worst_seg})")

        # ── 고도 / 속도 / 자세 ───────────────────────────────────────
        marks = sorted(wp_marks.items(), key=lambda t: t[1])
        if len(marks) >= 2:
            alt = np.interp(s_arr,
                            np.array([s_arr[i] for i, _ in marks]),
                            np.array([wps[wi, 2] for _, wi in marks]))
        else:
            alt = np.full(len(pts), wps[0, 2])

        if self.speed_policy == "curvature":
            ak = np.maximum(np.abs(kap_arr), 1e-12)
            v_lim = np.where(np.abs(kap_arr) > 1e-9, np.sqrt(a_max / ak), v_cruise)
            v_ref = np.clip(np.minimum(v_cruise, v_lim),
                            v_min_turn if v_min_turn > 0 else 0.5, v_cruise)
        else:
            v_ref = np.full(len(pts), v_cruise)

        gamma = np.zeros(len(pts))
        if len(pts) > 1:
            gamma[:-1] = np.arctan2(np.diff(alt), np.maximum(np.diff(s_arr), 1e-9))
            gamma[-1] = gamma[-2]

        points = [PathPoint(pos=np.array([pts[i, 0], pts[i, 1], alt[i]]),
                            v_ref=float(v_ref[i]),
                            chi_ref=float(_wrap(th_arr[i])),
                            gamma_ref=float(gamma[i]),
                            curvature=float(kap_arr[i]),
                            s=float(s_arr[i]),
                            wp_index=wp_marks.get(i))
                  for i in range(len(pts))]

        return Path(points=points, waypoints_ned=wps,
                    total_length=float(s_arr[-1]),
                    planning_time=time.perf_counter() - t0)

    # ── 보조 ─────────────────────────────────────────────────────────
    def _merge_degenerate(self, wps: np.ndarray) -> np.ndarray:
        """수평 위치가 같은 연속 WP(수직 상승) 병합 — 기존 플래너와 동일 규약."""
        keep, merged = [0], []
        for i in range(1, len(wps)):
            if np.linalg.norm(wps[i, :2] - wps[keep[-1], :2]) < self.min_wp_chord:
                wps[keep[-1], 2] = wps[i, 2]
                merged.append(i)
            else:
                keep.append(i)
        if merged:
            print(f"[G2SpiralPlanner] WARNING: merged WPs {merged} "
                  f"(<{self.min_wp_chord} m 수평거리). 수직 상승은 플래너가 아니라 "
                  f"이륙/천이 상태기계가 담당한다.", flush=True)
            wps = wps[keep]
            if len(wps) < 2:
                raise ValueError("퇴화 WP 병합 후 waypoints가 2개 미만입니다.")
        return wps


# ─────────────────────────────────────────────────────────────────────────────
# 이론 하한 (보고·검증용)
# ─────────────────────────────────────────────────────────────────────────────
def curvature_floor_g1(delta: float, chord: float) -> float:
    """**끝 세그먼트**(시작 헤딩 = 현 방향, α=0)의 곡률 이론 하한.

    현 길이 `chord`, 시작 헤딩 = 현 방향, 끝 헤딩 = 현 + `delta` 인 반문제에서
    |κ| ≤ k 로 끝점에 닿을 수 있는 최소 k.

    유도: 최대화 ∫cos θ ds  s.t. ∫sin θ ds = 0, θ(0)=0, θ(L)=δ, |θ′|≤k, L 자유.
    Pontryagin 에서 H ≡ 0 이고 특이(직선) 호는 H=0 과 모순이라 존재하지 않으므로
    최적은 순수 bang-bang, 그리고 α=0 이면 θ 가 반드시 한 번 음수로 내려가야
    하므로(∫sin θ ds = 0) 구조는 "−k 로 d 만큼 꺾었다가 +k 로 되꺾기" 하나다:

        cos d = (1 + cos δ)/2 ,   도달 가능 최대 현 = R·[2 sin d + sin δ]

    스위치를 3개로 늘려 수치 최적화하면 되돌아 도는(loop) 해가 나오는데 그건
    현 대비 곡률이 **더 큰** 해라 하한을 바꾸지 못한다(직접 확인).

    ⚠️ **α ≠ 0 인 중간 세그먼트에는 쓰면 안 된다.** 예컨대 α=−45°, β=+45° 는
    단일 원호로 y=0 이 자동 성립해 하한이 (sin β − sin α)/D 로 훨씬 낮다.
    이 함수는 경로 **첫/마지막** 세그먼트(기체가 첫 구간 방향으로 진입하고
    마지막 구간 방향으로 이탈하는, 실제로 가장 빡빡한 곳)에만 유효하다.

    **G2(곡률 연속)를 요구하면 이 값보다 반드시 커진다** — 이건 G1 하한이다.
    """
    d = float(np.arccos(np.clip((1.0 + np.cos(delta)) / 2.0, -1.0, 1.0)))
    f = 2.0 * np.sin(d) + np.sin(delta)
    return float(f / chord) if chord > 0 else float("inf")
