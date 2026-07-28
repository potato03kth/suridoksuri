"""Eta3ClothoidPlannerV3 — NR 반복 비용 회귀 테스트 (F-12)

배경 (2026-07-29 실측)
---------------------
`OffboardNode.__init__` 이 경로 계획에서 최대 263.5 s 동기 블로킹됐다. 원인을
계측한 결과 **계산량의 대부분이 결과에 영향을 주지 않는 반복**이었다:

  1. `clip_attempts` 3회 재시도 — 세그먼트 길이배율 v 가 clip 경계에 한 번도
     닿지 않아(실측 v∈[-0.06, 0.004] vs 경계 (-0.5, 1.2)) 세 attempt 의
     최종 |F| 가 소수점 15자리까지 동일했다. 순수 3배 낭비.
  2. `max_iter=60` 전량 소진 — 이 잔차계는 방정식 3(N−1)개에 미지수 3N−5개로
     **2개 과결정**이라 정확해가 없고 `|F| < tol` 이 원리적으로 도달 불가다.
     기존 정체 판정 `abs(prev−cur) < 1e-10 **and** norm_F < 10*tol` 의 둘째 절이
     영원히 거짓이라 절대 발동하지 않았다. 실측상 |F| 는 **10회**에서 멈춘다
     (31.73→20.58→14.60→13.47→11.89→11.66→…→11.6303 이후 불변).
  3. 조밀 유한차분 야코비안 — 잔차 블록 k 는 (θ_k, θ_{k+1}, κ_k, κ_{k+1}, L_k)
     에만 의존하는 **대역폭 5 띠행렬**인데 열마다 전체 잔차를 다시 계산했다.
     N=33 에서 9024칸 중 실제 비영 403칸(4.5%).

이 테스트들은 세 낭비가 되살아나면 실패한다. 판정 기준으로 **벽시계 시간이
아니라 `_fresnel_endpoint` 호출 수**를 쓴다 — CPU 성능과 무관하게 결정적이고,
바로 이 값이 블로킹 시간에 비례하기 때문이다.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from path_planning import eta3clothoid_v3_1_planner as P  # noqa: E402
from path_planning.eta3clothoid_v3_1_planner import Eta3ClothoidPlannerV3  # noqa: E402

AIRCRAFT = {"gravity": 9.81, "v_cruise": 18.0, "a_max_g": 0.3}

WPS_L = np.array([[0.0, 0.0, 50.0],
                  [200.0, 0.0, 50.0],
                  [200.0, 200.0, 50.0]])

WPS_CLOSED = np.array([[0.0, 0.0, 50.0],
                       [200.0, 0.0, 50.0],
                       [200.0, 200.0, 50.0],
                       [0.0, 200.0, 50.0],
                       [0.0, 0.0, 50.0]])


class _FresnelCounter:
    """`_fresnel_endpoint` 호출 수를 세는 컨텍스트 매니저."""

    def __init__(self) -> None:
        self.n = 0
        self._orig = None

    def __enter__(self):
        self._orig = P._fresnel_endpoint

        def counted(*a, **k):
            self.n += 1
            return self._orig(*a, **k)

        P._fresnel_endpoint = counted
        return self

    def __exit__(self, *exc):
        P._fresnel_endpoint = self._orig
        return False


def _plan(wps, **params):
    aircraft = dict(AIRCRAFT)
    aircraft.update(params)
    return Eta3ClothoidPlannerV3(ds=1.0, accel_tol=0.85,
                                 end_extension=0).plan(wps.copy(), aircraft)


# ─────────────────────────────────────────────────────────────────────────
# 1. 희소 야코비안 == 조밀 야코비안 (비트 단위)
# ─────────────────────────────────────────────────────────────────────────

def test_sparse_jacobian_bit_identical_to_dense():
    """`_jacobian_fd` 는 조밀 유한차분과 **비트 단위로** 같아야 한다.

    희소성 판정(`_seg_deps`)이 틀리면 빠뜨린 블록의 미분이 0으로 남아
    Newton 방향이 달라지고 경로가 조용히 바뀐다. 그 사고를 여기서 잡는다.
    """
    rng = np.random.default_rng(7)
    for N in (3, 4, 5, 9, 17, 33):
        wps = np.cumsum(rng.normal(0.0, 60.0, size=(N, 2)), axis=0)
        chords = np.array([np.linalg.norm(wps[k + 1] - wps[k])
                           for k in range(N - 1)])
        kappa_max = 0.008
        args = (wps, (0.1, -0.3), (0.0, 0.0), kappa_max, chords,
                float(min(chords.mean(), 50.0)), -0.5, 1.2)
        n_inner, n_segs = N - 2, N - 1
        x = rng.normal(0.0, 0.4, size=2 * n_inner + n_segs)
        F = P._residual(x, *args)

        eps = 1e-6
        J_dense = np.zeros((len(F), len(x)))
        for j in range(len(x)):
            xp = x.copy()
            xp[j] += eps
            J_dense[:, j] = (P._residual(xp, *args) - F) / eps

        J_sparse = P._jacobian_fd(x, F, args, eps)
        assert np.array_equal(J_dense, J_sparse), (
            f"N={N}: 희소 야코비안이 조밀과 다르다 "
            f"(maxdiff={np.abs(J_dense - J_sparse).max():.3e})")
        # 띠행렬 구조 자체도 확인 — 밀도가 높아졌다면 _seg_deps 가 과도하게
        # 넓어진 것이고 속도 이득이 사라진다.
        density = np.count_nonzero(J_sparse) / J_sparse.size
        if N >= 17:
            assert density < 0.12, f"N={N}: 야코비안 밀도 {density:.3f} — 띠 구조 상실"


def test_seg_deps_covers_every_nonzero_row():
    """`_seg_deps` 가 실제 의존 블록을 하나도 빠뜨리지 않는지 직접 확인."""
    rng = np.random.default_rng(11)
    N = 9
    wps = np.cumsum(rng.normal(0.0, 60.0, size=(N, 2)), axis=0)
    chords = np.array([np.linalg.norm(wps[k + 1] - wps[k]) for k in range(N - 1)])
    args = (wps, (0.2, 0.4), (0.0, 0.0), 0.008, chords,
            float(min(chords.mean(), 50.0)), -0.5, 1.2)
    n_inner, n_segs = N - 2, N - 1
    x = rng.normal(0.0, 0.4, size=2 * n_inner + n_segs)
    F = P._residual(x, *args)
    for j in range(len(x)):
        xp = x.copy()
        xp[j] += 1e-6
        changed = {k for k in range(n_segs)
                   if not np.array_equal(P._residual(xp, *args)[3 * k:3 * k + 3],
                                         F[3 * k:3 * k + 3])}
        declared = set(P._seg_deps(j, n_inner, n_segs))
        assert changed <= declared, (
            f"x[{j}]: 실제로 변한 블록 {sorted(changed)} 가 "
            f"선언된 의존 {sorted(declared)} 에 포함되지 않음")


# ─────────────────────────────────────────────────────────────────────────
# 2. 정체하면 멈춘다 / 무의미한 clip 재시도를 하지 않는다
# ─────────────────────────────────────────────────────────────────────────

def test_nr_stops_when_residual_stalls():
    """|F| 가 정체하면 max_iter 를 다 태우지 않아야 한다.

    이 잔차계는 2개 과결정이라 `|F| < tol` 에 도달할 수 없다. 정체 판정이
    없으면(=기존 코드) 반복 수가 항상 정확히 max_iter 가 된다.
    """
    wps2d = WPS_L[:, :2]
    theta0 = 0.0
    theta_N = np.pi / 2
    kappa_max = AIRCRAFT["a_max_g"] * AIRCRAFT["gravity"] * 0.85 / 18.0 ** 2

    calls = {"n": 0}
    orig = P._residual

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    P._residual = counted
    try:
        P._solve_g2_nr(wps2d, kappa_max, theta0, 0.0, theta_N, 0.0,
                       max_iter=60, tol=1e-5)
    finally:
        P._residual = orig

    # N=3 → 미지수 4개. 1반복당 잔차평가 = 야코비안 4 + |F| 1 + 선탐색 ≤20.
    # 60반복 × 3 attempt 를 다 태우면 4000회를 넘는다. 정체 판정이 살아 있으면
    # 수십 반복 안에 끝난다.
    assert calls["n"] < 2000, (
        f"잔차 평가 {calls['n']}회 — 정체 판정이 동작하지 않아 max_iter 를 "
        f"전부 소진한 것으로 보인다")


def test_clip_retry_skipped_when_bound_never_binds():
    """clip 경계에 한 번도 닿지 않으면 재시도(attempt 1,2)를 생략해야 한다.

    같은 초기값 + 더 넓은 경계 = 완전히 같은 문제이므로 재시도 결과가
    비트 단위로 같다. 실측에서 세 attempt 의 |F| 가 15자리까지 동일했다.
    """
    seen_clips = []
    orig = P._unpack

    def spy(x, N, theta_bc, kappa_bc, kappa_max, chords, v_lo, v_hi,
            clip_hit=None):
        seen_clips.append((v_lo, v_hi))
        return orig(x, N, theta_bc, kappa_bc, kappa_max, chords,
                    v_lo, v_hi, clip_hit)

    P._unpack = spy
    try:
        _plan(WPS_L)
    finally:
        P._unpack = orig

    used = set(seen_clips)
    assert used == {(-0.5, 1.2)}, (
        f"clip 경계가 안 걸리는 문제인데 재시도가 돌았다: {sorted(used)}")


# ─────────────────────────────────────────────────────────────────────────
# 3. 계산량 예산 — 이게 곧 블로킹 시간이다
# ─────────────────────────────────────────────────────────────────────────

def test_fresnel_call_budget_l_shape():
    """L자(3WP, v_cruise=18) 경로 계획의 Fresnel 적분 호출 수 상한.

    수정 전 실측 646,496회(≈19.6 s) → 수정 후 3,000회(0.11 s).
    상한 15,000회는 수정 후의 5배로, 정상적인 알고리즘 변경은 통과하고
    세 낭비(3배 재시도 / 60회 전량 소진 / 조밀 야코비안) 중 하나라도
    되살아나면 실패한다(파괴검증 D5·D6·D7 로 확인).
    """
    with _FresnelCounter() as c:
        _plan(WPS_L)
    assert c.n < 15_000, (
        f"Fresnel 호출 {c.n:,}회 — 수정 전 646,496회 수준으로 되돌아갔다")


def test_fresnel_call_budget_closed_loop():
    """폐회로(5WP, v_cruise=18) — F-12 최악 사례(문서값 263.5 s).

    수정 전 실측 2,399,488회 → 수정 후 6,232회(0.39 s). 상한 30,000회.
    """
    with _FresnelCounter() as c:
        _plan(WPS_CLOSED)
    assert c.n < 30_000, (
        f"Fresnel 호출 {c.n:,}회 — 수정 전 2,399,488회 수준으로 되돌아갔다")


# ─────────────────────────────────────────────────────────────────────────
# 4. 빨라지는 대신 경로가 나빠지지 않았는가
# ─────────────────────────────────────────────────────────────────────────

def test_path_quality_preserved_l_shape():
    """WP 통과·전장·단조 s — 수정 전 실측값과 같아야 한다."""
    path = _plan(WPS_L)
    pts = np.array([p.pos[:2] for p in path.points])
    for i, wp in enumerate(WPS_L):
        d = float(np.min(np.linalg.norm(pts - wp[:2], axis=1)))
        assert d < 1e-6, f"WP{i} 통과 오차 {d:.4f} m"
    # 수정 전 실측 400.0224 m
    assert abs(path.total_length - 400.0224) < 0.05, (
        f"전장 {path.total_length:.4f} m — 수정 전 400.0224 m 에서 벗어났다")
    s = np.array([p.s for p in path.points])
    assert np.all(np.diff(s) > 1e-9), "s 가 strictly increasing 이 아니다"


def test_path_quality_preserved_closed_loop():
    path = _plan(WPS_CLOSED)
    pts = np.array([p.pos[:2] for p in path.points])
    for i, wp in enumerate(WPS_CLOSED):
        d = float(np.min(np.linalg.norm(pts - wp[:2], axis=1)))
        assert d < 1e-6, f"WP{i} 통과 오차 {d:.4f} m"
    # 수정 전 실측 800.06 m
    assert abs(path.total_length - 800.06) < 0.1, (
        f"전장 {path.total_length:.4f} m — 수정 전 800.06 m 에서 벗어났다")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
