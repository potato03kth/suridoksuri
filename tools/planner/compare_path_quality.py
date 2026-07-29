#!/usr/bin/env python3
"""경로 **품질** 비교기 — 기존 eta3 vs 재설계 G2 스파이럴 (안 C, 1단계).

`tools/sitl/measure_planner_blocking.py` 가 *속도*를 재는 하니스였다면 이건
*품질*을 잰다. 속도는 `15f7fa4` 로 이미 해결됐고(L자 26.1s→0.098s), 남은
문제는 경로가 사실상 폴리라인이라는 것이다.

측정 지표 — **두 플래너에 완전히 같은 식**을 쓴다:
  kappa_ratio  : |κ|max / κmax.  κ 는 출력된 경로점에서 유한차분으로 다시
                 계산한다(플래너가 자기 필드에 뭘 써 넣었든 무관하게 실제
                 기하를 본다). 공식은 iterpin/_compute_curvature 와 동일.
  dchi_max     : 이웃 경로점 사이 헤딩 변화 최대 (deg). 곡선이면 κ·ds 수준이고
                 폴리라인 조인트면 코너각 그대로 튄다.
  dkappa_max   : 이웃 경로점 사이 κ 변화 최대 / κmax → G2(곡률 연속) 판정.
  wp_err       : 각 원본 WP 에 대한 경로 최근접 거리의 최대 (m).
  polydev      : 폴리라인 대비 이탈 (최대/평균, m).
  t_plan       : 3회 중 최소 (벽시계, s).

사용:
    python3 tools/planner/compare_path_quality.py
    python3 tools/planner/compare_path_quality.py --scenario A3_L --plot out.png
    python3 tools/planner/compare_path_quality.py --dof-study
    python3 tools/planner/compare_path_quality.py --json out.json

SITL 이 필요 없다(순수 Python). 절대 시간은 CPU 에 따라 달라지니 두 플래너를
**같은 실행 안에서** 비교한 값만 나란히 놓을 것.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path as FSPath

import numpy as np

_REPO = FSPath(__file__).resolve().parents[2]
_SIM = _REPO / "vtol_sim_checkpoint1_1" / "vtol_sim"
for _p in (str(_REPO), str(_SIM)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from path_planning.eta3clothoid_v3_1_planner import Eta3ClothoidPlannerV3  # noqa: E402
from path_planning.g2_spiral_planner import (  # noqa: E402
    G2SpiralPlanner, curvature_floor_g1,
)
from path_planning.straight_line_planner import StraightLinePlanner  # noqa: E402

# 경로는 tools/sitl/scenarios.yaml 의 해당 시나리오와 글자 그대로 같다.
SCENARIOS: dict[str, list[list[float]]] = {
    # ── 캠페인 경로 ────────────────────────────────────────────────
    "A3_L":       [[0, 0, 50], [200, 0, 50], [200, 200, 50]],
    "R2_closed":  [[0, 0, 50], [200, 0, 50], [200, 200, 50], [0, 200, 50], [0, 0, 50]],
    "B5_square":  [[0, 0, 50], [200, 0, 50], [200, 200, 50], [0, 200, 50], [0, 20, 50]],
    "B2_gentle":  [[0, 0, 50], [150, 0, 50], [300, 80, 50], [450, 200, 50]],
    "B4_uturn":   [[0, 0, 50], [250, 0, 50], [100, 150, 50]],
    # ── 짧은 경로 (사용자 지시 2026-07-29: 디버깅용 짧은 경로에서도 문제없어야) ──
    "S_line40":   [[0, 0, 50], [40, 0, 50]],
    "S_L30":      [[0, 0, 50], [30, 0, 50], [30, 30, 50]],
    "S_L50":      [[0, 0, 50], [50, 0, 50], [50, 50, 50]],
    "S_dogleg":   [[0, 0, 50], [40, 0, 50], [70, 20, 50]],
    "S_zigzag":   [[0, 0, 50], [30, 0, 50], [50, 25, 50], [80, 5, 50]],
    # ── κ=0@WP 강제의 대가를 드러내는 경로 (같은 방향 연속 선회) ──
    "ARC_R200":   None,        # 아래에서 생성
}


def _make_arc(R: float = 200.0, n: int = 7, sweep_deg: float = 120.0) -> list[list[float]]:
    """반지름 R 원호 위 등간격 WP — 이상적 경로는 κ≡1/R 인 케이스."""
    a = np.radians(np.linspace(0.0, sweep_deg, n))
    return [[float(R * np.sin(t)), float(R * (1 - np.cos(t))), 50.0] for t in a]


SCENARIOS["ARC_R200"] = _make_arc()

AIRCRAFT = {"gravity": 9.81, "v_cruise": 18.0, "a_max_g": 0.3,
            "v_min_turn": 16.5, "v_stall": 13.8, "decel_dist": 80.0}
ACCEL_TOL = 0.85          # fc_bridge/planning/planner_runner.py 가 쓰는 값


# ─────────────────────────────────────────────────────────────────────────────
# 지표
# ─────────────────────────────────────────────────────────────────────────────
def geometric_curvature(pos: np.ndarray, s: np.ndarray) -> np.ndarray:
    """κ = (N′E″ − E′N″)/|v|³ — 출력 경로점에서 직접. 두 플래너에 동일 적용."""
    dN = np.gradient(pos[:, 0], s)
    dE = np.gradient(pos[:, 1], s)
    d2N = np.gradient(dN, s)
    d2E = np.gradient(dE, s)
    spd = np.sqrt(np.maximum(dN ** 2 + dE ** 2, 1e-24))
    return (dN * d2E - dE * d2N) / spd ** 3


def polyline_deviation(pos: np.ndarray, wps2d: np.ndarray) -> tuple[float, float]:
    a = wps2d[:-1]
    b = wps2d[1:]
    ab = b - a
    L2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    d = np.empty(len(pos))
    for i, p in enumerate(pos):
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / L2, 0.0, 1.0)
        foot = a + t[:, None] * ab
        d[i] = np.min(np.linalg.norm(p - foot, axis=1))
    return float(d.max()), float(d.mean())


def measure(path, wps: np.ndarray, kappa_max: float, t_plan: float) -> dict:
    pos = path.positions_array()[:, :2]
    s = np.array([p.s for p in path.points])
    kap_geo = geometric_curvature(pos, s)
    kap_field = np.array([p.curvature for p in path.points])
    chi = np.array([p.chi_ref for p in path.points])
    dchi = np.abs((np.diff(chi) + np.pi) % (2 * np.pi) - np.pi)
    dev_max, dev_mean = polyline_deviation(pos, wps[:, :2])
    wp_err = max(float(np.min(np.linalg.norm(pos - w, axis=1))) for w in wps[:, :2])
    n_marks = sum(1 for p in path.points if p.wp_index is not None)
    return {
        "n_pts": int(len(pos)),
        "t_plan_s": float(t_plan),
        "kappa_geo_max": float(np.abs(kap_geo).max()),
        "kappa_ratio": float(np.abs(kap_geo).max() / kappa_max),
        "kappa_field_ratio": float(np.abs(kap_field).max() / kappa_max),
        "dchi_max_deg": float(np.degrees(dchi.max())) if len(dchi) else 0.0,
        "dkappa_max_ratio": (float(np.abs(np.diff(kap_geo)).max() / kappa_max)
                             if len(kap_geo) > 1 else 0.0),
        "wp_err_m": wp_err,
        "n_wp_marks": n_marks,
        "polydev_max_m": dev_max,
        "polydev_mean_m": dev_mean,
        "length_m": float(path.total_length),
        "v_ref_min": float(min(p.v_ref for p in path.points)),
    }


def _best_of(fn, repeat: int = 3):
    best_t, out = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best_t = min(best_t, time.perf_counter() - t0)
    return out, best_t


def run_one(name: str, wps_list, repeat: int = 3, quiet: bool = True) -> dict:
    wps = np.array(wps_list, dtype=float)
    kappa_max = (AIRCRAFT["a_max_g"] * AIRCRAFT["gravity"] * ACCEL_TOL
                 / AIRCRAFT["v_cruise"] ** 2)
    row: dict = {"scenario": name, "n_wp": int(len(wps)), "kappa_max": kappa_max}

    planners = {
        "eta3(현재)": lambda: Eta3ClothoidPlannerV3(
            ds=1.0, accel_tol=ACCEL_TOL, end_extension=0.0),
        "g2(안C)": lambda: G2SpiralPlanner(
            ds=1.0, accel_tol=ACCEL_TOL, end_extension=0.0,
            on_infeasible="silent" if quiet else "warn"),
        "straight": lambda: StraightLinePlanner(ds=1.0),
    }
    for label, make in planners.items():
        pl = make()
        try:
            path, t = _best_of(lambda: pl.plan(wps.copy(), dict(AIRCRAFT)), repeat)
        except Exception as e:                                   # noqa: BLE001
            row[label] = {"error": f"{type(e).__name__}: {e}"}
            continue
        row[label] = measure(path, wps, kappa_max, t)
        if label == "g2(안C)" and pl.last_report is not None:
            r = pl.last_report
            row[label].update({
                "v_feasible": r.v_feasible,
                "worst_seg_ratio_design": r.worst_ratio,
                "speed_floor_violated": r.speed_floor_violated,
                "decel_margin_m": r.decel_margin_m,
            })

    # G1 이론 하한 — **첫/마지막 세그먼트에만** 유효하다(α=0 인 곳).
    # 코너를 양쪽 세그먼트가 절반씩 나눠 갖는 대칭분할 기준.
    floors = []
    if len(wps) >= 3:
        for i, chord_ij in ((1, (0, 1)), (len(wps) - 2, (len(wps) - 2, len(wps) - 1))):
            a = wps[i, :2] - wps[i - 1, :2]
            b = wps[i + 1, :2] - wps[i, :2]
            turn = abs(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))
            chord = float(np.linalg.norm(wps[chord_ij[1], :2] - wps[chord_ij[0], :2]))
            if turn > 1e-9 and chord > 1e-9:
                floors.append(curvature_floor_g1(turn / 2.0, chord))
    row["g1_floor_ratio"] = (float(max(floors) / kappa_max) if floors else 0.0)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────────────────────────────────
_COLS = [
    ("|κ|max/κmax", "kappa_ratio", "{:>11.2f}"),
    ("Δχ_max(deg)", "dchi_max_deg", "{:>11.2f}"),
    ("Δκ_max/κmax", "dkappa_max_ratio", "{:>11.2f}"),
    ("WP오차(m)", "wp_err_m", "{:>11.2e}"),
    ("이탈max(m)", "polydev_max_m", "{:>11.2f}"),
    ("전장(m)", "length_m", "{:>11.1f}"),
    ("t(s)", "t_plan_s", "{:>11.4f}"),
]


def print_table(rows: list[dict]) -> None:
    hdr = f"{'시나리오':<12}{'플래너':<12}" + "".join(f"{h:>13}" for h, _, _ in _COLS)
    print(hdr)
    print("─" * len(hdr))
    for r in rows:
        for label in ("eta3(현재)", "g2(안C)", "straight"):
            m = r.get(label)
            if m is None:
                continue
            if "error" in m:
                print(f"{r['scenario']:<12}{label:<12}  {m['error']}")
                continue
            line = f"{r['scenario']:<12}{label:<12}"
            for _h, key, fmt in _COLS:
                line += f"{fmt.format(m[key]):>13}"
            print(line)
        extra = r.get("g2(안C)", {})
        if "v_feasible" in extra:
            vf = extra["v_feasible"]
            print(f"{'':<12}{'└ 진단':<12}  G1이론하한 {r['g1_floor_ratio']:.2f}×κmax | "
                  f"v_feasible {vf:.2f} m/s"
                  f"{' (v_min_turn 미달!)' if extra['speed_floor_violated'] else ''}"
                  f" | 감속여유 {extra['decel_margin_m']:+.0f} m")
        print()


def dof_study() -> None:
    """polish 자유도에 따른 |κ| 개선 — ansatz 한계 vs 기하 한계 분리."""
    from path_planning.g2_spiral_planner import solve_segment
    kappa_max = (AIRCRAFT["a_max_g"] * AIRCRAFT["gravity"] * ACCEL_TOL
                 / AIRCRAFT["v_cruise"] ** 2)
    print(f"κmax = {kappa_max:.6f} 1/m (R_min = {1/kappa_max:.1f} m)\n")
    print(f"{'세그먼트 (α, Δθ, D)':<34}" + "".join(f"{'dof=' + str(d):>10}"
                                                   for d in range(0, 4))
          + f"{'G1하한':>10}")
    cases = [
        ("끝세그 (α=0°, Δθ=45°, 200m)", 0.0, 45.0, 200.0),
        ("끝세그 (α=0°, Δθ=45°, 400m)", 0.0, 45.0, 400.0),
        ("중간세그 (α=−45°, Δθ=90°, 200m)", -45.0, 90.0, 200.0),
        ("짧은 L  (α=0°, Δθ=45°, 50m)", 0.0, 45.0, 50.0),
        ("짧은 L  (α=0°, Δθ=45°, 30m)", 0.0, 45.0, 30.0),
    ]
    for name, a_deg, d_deg, D in cases:
        line = f"{name:<34}"
        for dof in range(0, 4):
            s = solve_segment(np.radians(a_deg), np.radians(d_deg), D, polish_dof=dof)
            line += f"{(s.kappa_peak / kappa_max if s.ok else float('nan')):>10.3f}"
        if abs(a_deg) < 1e-9:
            line += f"{curvature_floor_g1(np.radians(d_deg), D) / kappa_max:>10.3f}"
        else:
            line += f"{'—(주)':>10}"
        print(line)
    print("\n  dof=0 은 정사각(미지수=방정식) 해, dof≥1 은 남는 자유도로 min-max |κ|.")
    print("  G1하한은 곡률 **불연속**을 허용한 bang-bang 이론 하한이라 G2 해는")
    print("  반드시 그보다 크다. (주) α≠0 인 중간 세그먼트에는 이 공식이 유효하지")
    print("  않다 — 단일 원호로 y=0 이 자동 성립해 하한이 훨씬 낮아진다.")


def kappa_zero_penalty() -> None:
    """WP 에서 κ=0 강제의 대가 — 원호 WP 열에서 1.5배가 나오는지 직접 확인."""
    kappa_max = (AIRCRAFT["a_max_g"] * AIRCRAFT["gravity"] * ACCEL_TOL
                 / AIRCRAFT["v_cruise"] ** 2)
    print("반지름 R 원호 위 등간격 WP — 이상적 경로는 κ≡1/R. κ=0@WP 강제의 대가:")
    print(f"{'R(m)':>8}{'WP수':>6}{'이상 κ':>12}{'실측 |κ|max':>14}{'배수':>8}")
    for R, n in ((200.0, 7), (200.0, 13), (400.0, 7), (150.0, 9)):
        wps = np.array(_make_arc(R, n), dtype=float)
        pl = G2SpiralPlanner(ds=1.0, accel_tol=ACCEL_TOL, end_extension=0.0,
                             on_infeasible="silent")
        path = pl.plan(wps.copy(), dict(AIRCRAFT))
        kap = geometric_curvature(path.positions_array()[:, :2],
                                  np.array([p.s for p in path.points]))
        ideal = 1.0 / R
        print(f"{R:>8.0f}{n:>6d}{ideal:>12.6f}{np.abs(kap).max():>14.6f}"
              f"{np.abs(kap).max()/ideal:>8.3f}")
    print(f"  (참고 κmax={kappa_max:.6f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", action="append",
                    help="시나리오 이름 (반복 가능). 생략하면 전부.")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--dof-study", action="store_true",
                    help="polish 자유도 대비 |κ| 표만 출력")
    ap.add_argument("--kappa-zero-penalty", action="store_true",
                    help="κ=0@WP 강제의 1.5배 대가 확인")
    ap.add_argument("--verbose-report", action="store_true",
                    help="G2 플래너의 실현가능성 경고를 그대로 출력")
    args = ap.parse_args()

    if args.dof_study:
        dof_study()
        return 0
    if args.kappa_zero_penalty:
        kappa_zero_penalty()
        return 0

    names = args.scenario or list(SCENARIOS)
    rows = []
    for nm in names:
        if nm not in SCENARIOS:
            print(f"unknown scenario: {nm}", file=sys.stderr)
            return 2
        rows.append(run_one(nm, SCENARIOS[nm], repeat=args.repeat,
                            quiet=not args.verbose_report))

    kmax = rows[0]["kappa_max"]
    print(f"\nv_cruise={AIRCRAFT['v_cruise']} a_max_g={AIRCRAFT['a_max_g']} "
          f"accel_tol={ACCEL_TOL} → κmax={kmax:.6f} 1/m (R_min={1/kmax:.1f} m)\n")
    print_table(rows)

    if args.json:
        FSPath(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"→ {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
