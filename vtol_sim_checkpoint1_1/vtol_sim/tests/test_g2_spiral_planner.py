"""G2SpiralPlanner — 재설계 "안 C" 의 **설계 전제**를 지키는 회귀 테스트.

여기서 지키려는 것은 성능이 아니라 **주장**이다. 기존 eta3 플래너가 무너진
경위가 "설계 전제(정확해 존재)가 성립하지 않는데 코드가 그 사실을 말하지 않은
것" 이었으므로, 이 테스트들은 하나하나가 문서의 문장에 대응한다:

  1. 자유도 예산이 맞는가 (미지수 2 = 방정식 2) → 잔차가 기계정밀도로 0인가
  2. WP 를 **보정 없이** 통과하는가
  3. WP 에서 κ=0 인가 (세그먼트 분리의 근거)
  4. G2(곡률 연속)인가 — 헤딩 점프가 없는가
  5. 곡률 한계를 못 지킬 때 **조용히 넘어가지 않고** 보고하는가
  6. 짧은 경로에서 죽지 않고 감속 여유 부족을 알려주는가
  7. numpy 2.x 에서 제거된 `np.trapz` 를 쓰지 않는가
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from path_planning.g2_spiral_planner import (  # noqa: E402
    G2SpiralPlanner, curvature_floor_g1, solve_segment, _psi, _W, _wrap,
)

AIRCRAFT = {"gravity": 9.81, "v_cruise": 18.0, "a_max_g": 0.3,
            "v_min_turn": 16.5, "v_stall": 13.8, "decel_dist": 80.0}
ACCEL_TOL = 0.85
KAPPA_MAX = AIRCRAFT["a_max_g"] * AIRCRAFT["gravity"] * ACCEL_TOL / 18.0 ** 2

WPS_L = np.array([[0.0, 0.0, 50.0], [200.0, 0.0, 50.0], [200.0, 200.0, 50.0]])
WPS_CLOSED = np.array([[0.0, 0.0, 50.0], [200.0, 0.0, 50.0], [200.0, 200.0, 50.0],
                       [0.0, 200.0, 50.0], [0.0, 0.0, 50.0]])
WPS_B5 = np.array([[0.0, 0.0, 50.0], [200.0, 0.0, 50.0], [200.0, 200.0, 50.0],
                   [0.0, 200.0, 50.0], [0.0, 20.0, 50.0]])
WPS_SHORT_L = np.array([[0.0, 0.0, 50.0], [30.0, 0.0, 50.0], [30.0, 30.0, 50.0]])
WPS_SHORT_LINE = np.array([[0.0, 0.0, 50.0], [40.0, 0.0, 50.0]])

ALL_WPS = {"L": WPS_L, "closed": WPS_CLOSED, "b5": WPS_B5,
           "short_L": WPS_SHORT_L, "short_line": WPS_SHORT_LINE}


def _plan(wps: np.ndarray, **kw):
    pl = G2SpiralPlanner(ds=1.0, accel_tol=ACCEL_TOL, end_extension=0.0,
                         on_infeasible="silent", **kw)
    return pl, pl.plan(wps.copy(), dict(AIRCRAFT))


# ─────────────────────────────────────────────────────────────────────────
# 1. 자유도 예산 — 잔차가 정말 0 인가
# ─────────────────────────────────────────────────────────────────────────
def test_segment_residual_is_machine_zero():
    """미지수 2 = 방정식 2 이므로 **정확해가 존재**한다.

    기존 플래너는 여기서 2개 과결정이라 |F| 가 11.6 에서 멈췄다. 이 테스트가
    깨지면 자유도 예산이 다시 어긋난 것이다.
    """
    rng = np.random.default_rng(3)
    for _ in range(200):
        alpha = rng.uniform(-1.2, 1.2)
        dtheta = rng.uniform(-2.0, 2.0)
        D = rng.uniform(20.0, 500.0)
        s = solve_segment(alpha, dtheta, D)
        if not s.ok:
            continue
        # Δy 잔차: ∫ sin ψ dσ 가 0 이어야 한다.
        assert abs(float(_W @ np.sin(_psi(alpha, dtheta, s.c)))) < 1e-11
        # Δx 잔차: L·∫cos ψ dσ = D 가 닫힌형으로 성립.
        assert abs(s.L * s.C - D) < 1e-9 * max(1.0, D)
        # Δθ 는 기저가 구조적으로 보장 — ψ(1) − ψ(0) = Δθ
        p = _psi(alpha, dtheta, s.c)
        assert abs((p[-1] - p[0]) - dtheta) < 1e-12


def test_polish_never_worse_and_keeps_constraints():
    """폴리시는 개선될 때만 채택 — 정사각 해보다 나빠질 수 없다."""
    for alpha, dtheta, D in [(0.0, np.pi / 4, 200.0), (0.0, np.pi / 4, 30.0),
                             (-np.pi / 4, np.pi / 2, 200.0), (0.3, -0.9, 120.0)]:
        base = solve_segment(alpha, dtheta, D, polish_dof=0)
        pol = solve_segment(alpha, dtheta, D, polish_dof=2)
        assert base.ok and pol.ok
        assert pol.kappa_peak <= base.kappa_peak * (1.0 + 1e-9)
        assert abs(float(_W @ np.sin(_psi(alpha, dtheta, pol.c)))) < 1e-11


# ─────────────────────────────────────────────────────────────────────────
# 2~4. 경로 불변식
# ─────────────────────────────────────────────────────────────────────────
def test_waypoints_passed_without_affine_correction():
    """WP 통과 오차가 μm 대여야 한다 — affine 보정을 쓰지 않기 때문이다.

    기존 플래너는 NR 잔차 5 m 를 affine 보정으로 덮어 WP 통과만 살렸다.
    여기서는 잔차 자체가 0 이라 보정이 없다.
    """
    for name, wps in ALL_WPS.items():
        _, path = _plan(wps)
        pos = path.positions_array()[:, :2]
        for w in wps[:, :2]:
            d = float(np.min(np.linalg.norm(pos - w, axis=1)))
            assert d < 1e-4, f"{name}: WP 통과 오차 {d:.3e} m"


def test_kappa_is_zero_at_waypoints():
    """WP 에서 κ=0 — 세그먼트 분리(전역 뉴턴 제거)의 근거 그 자체."""
    for name, wps in ALL_WPS.items():
        _, path = _plan(wps)
        for p in path.points:
            if p.wp_index is not None:
                assert abs(p.curvature) < 1e-9, f"{name}: WP{p.wp_index} κ={p.curvature}"


def test_heading_is_continuous():
    """이웃 경로점 헤딩 변화가 κ·ds 수준 — 폴리라인 조인트 꺾임이 없어야 한다.

    기존 플래너 실측 85.94°. 여기서는 |κ|max·ds 로 상계된다.
    """
    for name, wps in ALL_WPS.items():
        pl, path = _plan(wps)
        chi = np.array([p.chi_ref for p in path.points])
        dchi = np.abs((np.diff(chi) + np.pi) % (2 * np.pi) - np.pi)
        kmax = max(abs(p.curvature) for p in path.points)
        if len(dchi) == 0:
            continue
        assert dchi.max() <= kmax * pl.ds * 1.05 + 1e-9, (
            f"{name}: Δχ_max={np.degrees(dchi.max()):.3f}° > κmax·ds")
        assert np.degrees(dchi.max()) < 10.0, f"{name}: Δχ_max 가 10°를 넘는다"


def test_curvature_field_matches_geometry():
    """`PathPoint.curvature` 가 실제 경로 기하와 일치해야 한다.

    기존 플래너는 affine 보정 때문에 설계 κ 와 실제 κ 가 갈라져 하류 속도
    프로필·MPC 피드포워드가 허위 값을 받았다(tests/eta3_planner_issues.md §2).
    보정이 없으니 여기서는 일치해야 한다.
    """
    for name, wps in (("L", WPS_L), ("closed", WPS_CLOSED)):
        _, path = _plan(wps)
        pos = path.positions_array()[:, :2]
        s = np.array([p.s for p in path.points])
        dN, dE = np.gradient(pos[:, 0], s), np.gradient(pos[:, 1], s)
        d2N, d2E = np.gradient(dN, s), np.gradient(dE, s)
        spd = np.sqrt(np.maximum(dN ** 2 + dE ** 2, 1e-24))
        kap_geo = (dN * d2E - dE * d2N) / spd ** 3
        kap_fld = np.array([p.curvature for p in path.points])
        err = np.abs(kap_geo - kap_fld)
        # 마스크: (a) 양 끝은 np.gradient 의 편측 차분, (b) 세그먼트 이음매(=WP)는
        # κ 는 연속(=0)이지만 κ′ 가 불연속이라(G2 이고 G3 는 아니다) 중심차분이
        # 필연적으로 번진다. 실측 이음매 5.0e-4, 그 바깥 중앙값 2.0e-6.
        mask = np.ones(len(err), dtype=bool)
        mask[:3] = mask[-3:] = False
        for i, p in enumerate(path.points):
            if p.wp_index is not None:
                mask[max(0, i - 2):i + 3] = False
        assert err[mask].max() < 0.01 * KAPPA_MAX, (
            f"{name}: 이음매 밖 설계 κ vs 기하 κ 최대차 {err[mask].max():.3e}")
        assert err.max() < 0.10 * KAPPA_MAX, (
            f"{name}: 이음매 포함 최대차 {err.max():.3e} — κ′ 불연속 번짐치고 크다")


def test_path_invariants():
    """base_planner 규약: s 단조, wp_index 각 WP 정확히 하나, NaN 없음."""
    for name, wps in ALL_WPS.items():
        _, path = _plan(wps)
        s = np.array([p.s for p in path.points])
        assert np.all(np.diff(s) > 0), f"{name}: s 가 단조증가가 아니다"
        pos = path.positions_array()
        assert not np.isnan(pos).any(), f"{name}: NaN 경로점"
        marks = [p.wp_index for p in path.points if p.wp_index is not None]
        assert sorted(marks) == list(range(len(wps))), f"{name}: wp 마크 {marks}"
        assert abs(path.total_length - s[-1]) < 1e-9
        assert all(-np.pi <= p.chi_ref <= np.pi for p in path.points)


# ─────────────────────────────────────────────────────────────────────────
# 5~6. **조용히 뭉개지 않는가** — 이번 재설계의 핵심 요구
# ─────────────────────────────────────────────────────────────────────────
def test_infeasible_curvature_is_reported_not_hidden():
    """18 m/s 로는 200m 다리 90° 코너를 WP 를 밟으며 돌 수 없다 — 그렇다고 말해야."""
    pl, _ = _plan(WPS_L)
    rep = pl.last_report
    assert rep is not None
    assert not rep.curvature_feasible, "이 경로는 실현 불가여야 한다(기하학적 사실)"
    assert rep.worst_ratio > 1.0
    assert 10.0 < rep.v_feasible < AIRCRAFT["v_cruise"], (
        f"v_feasible={rep.v_feasible} — 날 수 있는 속도를 알려줘야 한다")
    assert rep.a_req_g_at_vmin > 0.0
    assert any("κmax" in ln for ln in rep.lines())


def test_raise_policy_actually_raises():
    pl = G2SpiralPlanner(ds=1.0, accel_tol=ACCEL_TOL, end_extension=0.0,
                         on_infeasible="raise")
    try:
        pl.plan(WPS_L.copy(), dict(AIRCRAFT))
    except ValueError as e:
        assert "곡률 한계 초과" in str(e)
    else:
        raise AssertionError("on_infeasible='raise' 인데 통과했다")


def test_short_path_decel_margin_reported():
    """짧은 경로 — 죽지 않고, 종단 감속 거리가 모자라다고 보고해야 한다."""
    pl, path = _plan(WPS_SHORT_LINE)
    assert pl.last_report.decel_margin_m < 0.0
    assert abs(path.total_length - 40.0) < 1e-6
    # 직선이므로 곡률은 정확히 0 이고 실현 가능해야 한다.
    assert pl.last_report.curvature_feasible

    pl2, _ = _plan(WPS_SHORT_L)
    assert pl2.last_report.decel_margin_m < 0.0
    assert pl2.last_report.worst_ratio > 5.0        # 30m 다리 90° 는 절망적
    assert pl2.last_report.speed_floor_violated


def test_two_waypoint_straight_is_exact():
    """2WP 직선은 정확히 직선이어야 한다(뉴턴을 아예 타지 않는 특수해)."""
    _, path = _plan(WPS_SHORT_LINE)
    assert max(abs(p.curvature) for p in path.points) == 0.0
    pos = path.positions_array()[:, :2]
    assert np.abs(pos[:, 1]).max() < 1e-12


def test_degenerate_vertical_waypoint_merged():
    """수평 위치가 같고 고도만 다른 WP — 기존 플래너와 같은 규약으로 병합."""
    wps = np.array([[0.0, 0.0, 20.0], [0.0, 0.0, 50.0],
                    [200.0, 0.0, 50.0], [200.0, 200.0, 50.0]])
    _, path = _plan(wps)
    s = np.array([p.s for p in path.points])
    assert np.all(np.diff(s) > 0)
    assert abs(path.points[0].pos[2] - 50.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────
# 7. 이론 하한 / 회귀 상한
# ─────────────────────────────────────────────────────────────────────────
def test_g1_floor_formula():
    """f(45°) = 1.749118 — 직접 유도값. 이 상수가 바뀌면 유도가 바뀐 것이다."""
    assert abs(curvature_floor_g1(np.radians(45.0), 1.0) - 1.749118) < 1e-5
    assert abs(curvature_floor_g1(np.radians(45.0), 200.0) * 200.0
               - curvature_floor_g1(np.radians(45.0), 1.0)) < 1e-12


def test_solution_stays_above_g1_floor():
    """G2 해는 곡률 불연속을 허용한 G1 하한보다 반드시 크다 — 검산."""
    for D in (100.0, 200.0, 400.0):
        s = solve_segment(0.0, np.radians(45.0), D, polish_dof=3)
        assert s.ok
        assert s.kappa_peak > curvature_floor_g1(np.radians(45.0), D) * 0.999


def test_quality_regression_bounds():
    """지금 달성한 품질이 조용히 나빠지면 실패한다 (2026-07-29 실측 기준)."""
    expect = {"L": 1.45, "closed": 1.30, "b5": 1.35}
    for name, bound in expect.items():
        pl, _ = _plan(ALL_WPS[name])
        assert pl.last_report.worst_ratio < bound, (
            f"{name}: worst_ratio {pl.last_report.worst_ratio:.3f} ≥ {bound}")


def test_no_numpy_trapz_usage():
    """numpy 2.0 에서 제거된 `np.trapz` 를 쓰면 실기체가 죽는다 (`0785777`)."""
    import path_planning.g2_spiral_planner as mod
    src = open(mod.__file__, encoding="utf-8").read()
    # 주석·독스트링의 언급이 아니라 **호출**을 잡는다.
    assert "np.trapz(" not in src and "np.trapezoid(" not in src


def test_planning_time_not_worse_than_eta3():
    """계산 시간이 기존 플래너보다 나빠지지 않아야 한다 (같은 실행 안에서 비교)."""
    import time
    from path_planning.eta3clothoid_v3_1_planner import Eta3ClothoidPlannerV3

    def best(fn, n=3):
        return min((lambda: (lambda t0: (fn(), time.perf_counter() - t0)[1])(
            time.perf_counter()))() for _ in range(n))

    for wps in (WPS_L, WPS_CLOSED, WPS_B5):
        old = Eta3ClothoidPlannerV3(ds=1.0, accel_tol=ACCEL_TOL, end_extension=0.0)
        new = G2SpiralPlanner(ds=1.0, accel_tol=ACCEL_TOL, end_extension=0.0,
                              on_infeasible="silent")
        t_old = best(lambda: old.plan(wps.copy(), dict(AIRCRAFT)))
        t_new = best(lambda: new.plan(wps.copy(), dict(AIRCRAFT)))
        assert t_new <= t_old * 1.2 + 0.02, (
            f"N={len(wps)}: 신규 {t_new:.4f}s > 기존 {t_old:.4f}s")


def test_wrap_range():
    for a in (-10.0, -np.pi, 0.0, np.pi, 7.0):
        assert -np.pi - 1e-12 <= _wrap(a) <= np.pi + 1e-12
