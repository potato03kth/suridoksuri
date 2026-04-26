"""
알고리즘 배치 비교 — Phase 2
==============================

4개 조합 (2 planner × 2 controller):
  Dubins + NLGL   Dubins + MPC
  Spline + NLGL   Spline + MPC

선택 시나리오에서 각 조합 실행 → MetricsResult 수집 → 비교표 출력.
Monte Carlo 옵션: --mc N  (기본 1회)

사용:
    python compare_algorithms.py basic
    python compare_algorithms.py basic --mc 10
    python compare_algorithms.py gusty --mc 30 --no-plot
"""
from __future__ import annotations
import sys
import os
import argparse
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
except AttributeError:
    pass

import numpy as np

from utils.config_loader import (
    load_aircraft_params, load_simulation_params, load_scenario,
    merge_scenario_into_aircraft,
)
from path_planning.dubins_planner import DubinsPlanner
from path_planning.spline_planner import SplinePlanner
from path_planning.waypoint_generator import waypoints_from_config
from path_following.nlgl_controller import NLGLController
from path_following.mpc_controller import MPCController
from path_following.inner_loop import InnerLoopPI
from simulator import Simulator
from metrics import compute_metrics, MetricsResult


# =============================================================================
# 조합 정의
# =============================================================================
COMBINATIONS = [
    ("dubins", "nlgl"),
    ("dubins", "mpc"),
    ("spline", "nlgl"),
    ("spline", "mpc"),
]


def build_planner(name: str):
    if name == "dubins":
        return DubinsPlanner(ds=2.0, R_factor=1.05, use_energy_climb=True)
    if name == "spline":
        return SplinePlanner(ds=2.0, bc_type="clamped")
    raise ValueError(f"Unknown planner: {name}")


def build_controller(name: str, aircraft: dict):
    v = aircraft["v_cruise"]
    g = aircraft["gravity"]
    a_max = aircraft["a_max_g"] * g
    R_min_op = (v ** 2) / a_max
    phi_max = np.deg2rad(aircraft["phi_max_deg"])
    tau_phi = aircraft["tau_phi"]

    if name == "nlgl":
        L1 = 2.0 * R_min_op
        return NLGLController(
            L1=L1,
            phi_max_rad=phi_max,
            inner_loop=InnerLoopPI(),
            gravity=g,
        )
    if name == "mpc":
        return MPCController(
            N_p=20,
            q_ey=1.0,
            q_echi=0.5,
            q_phi=0.05,
            q_term_factor=5.0,
            r_phi=2.0,
            phi_max_rad=np.deg2rad(45.0),
            tau_phi=tau_phi,
            gravity=g,
            inner_loop=InnerLoopPI(),
            look_ahead_m=2.0 * R_min_op,
        )
    raise ValueError(f"Unknown controller: {name}")


# =============================================================================
# 단일 실행
# =============================================================================
def run_one(planner_name: str, controller_name: str,
            aircraft: dict, sim_cfg: dict, scenario: dict,
            waypoints: np.ndarray, seed: int) -> tuple[MetricsResult, dict]:
    planner = build_planner(planner_name)
    controller = build_controller(controller_name, aircraft)
    sim = Simulator(
        aircraft_params=aircraft,
        sim_cfg=sim_cfg,
        scenario_cfg=scenario,
        planner=planner,
        controller=controller,
        seed=seed,
    )
    result = sim.run(waypoints, controller_name=controller_name)
    m = compute_metrics(result, aircraft_params=aircraft)
    return m, result


# =============================================================================
# Monte Carlo 배치
# =============================================================================
def run_batch(planner_name: str, controller_name: str,
              aircraft: dict, sim_cfg: dict, scenario: dict,
              waypoints: np.ndarray,
              n_runs: int, seed_base: int) -> list[MetricsResult]:
    results = []
    for i in range(n_runs):
        seed = seed_base + i
        try:
            m, _ = run_one(
                planner_name, controller_name,
                aircraft, sim_cfg, scenario, waypoints, seed
            )
            results.append(m)
        except Exception as e:
            print(f"    [WARN] {planner_name}+{controller_name} seed={seed} "
                  f"failed: {e}")
    return results


# =============================================================================
# 통계 집계
# =============================================================================
def aggregate(metrics_list: list[MetricsResult]) -> dict:
    """여러 MetricsResult → 평균/표준편차 dict."""
    if not metrics_list:
        return {}
    dicts = [asdict(m) for m in metrics_list]
    keys = [k for k, v in dicts[0].items()
            if isinstance(v, (int, float)) and k != "cpa_distances"]
    agg = {}
    for k in keys:
        vals = [d[k] for d in dicts if isinstance(d[k], (int, float))]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    agg["n_runs"] = len(metrics_list)
    agg["success_rate"] = float(np.mean([d["success"] for d in dicts]))
    return agg


# =============================================================================
# 비교 표 출력
# =============================================================================
_COL_METRICS = [
    ("composite_score",       "Score",       ".1f"),
    ("success_rate",          "Success%",    ".0%"),
    ("mean_cpa",              "CPA(m)",      ".1f"),
    ("rms_crosstrack",        "RMS-CT(m)",   ".1f"),
    ("rms_crosstrack_curved", "CT-Curve(m)", ".1f"),
    ("violation_time_ratio",  "Viol%",       ".1%"),
    ("excess_a_rms_g",        "ExcG",        ".3f"),
    ("path_feasibility_ratio","PathFeas",    ".2%"),
    ("bank_rate_rms",         "BkRate",      ".3f"),
    ("rms_altitude_err",      "AltErr(m)",   ".1f"),
    ("mean_compute_time_ms",  "Ctrl(ms)",    ".2f"),
    ("planning_time_ms",      "Plan(ms)",    ".1f"),
]


def print_comparison_table(combo_agg: dict[str, dict]) -> None:
    combo_names = list(combo_agg.keys())
    n_cols = len(combo_names)
    col_w = max(12, max(len(n) for n in combo_names) + 2)

    header = f"{'Metric':<18}" + "".join(
        f"{n:>{col_w}}" for n in combo_names
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for field, label, fmt in _COL_METRICS:
        row = f"{label:<18}"
        for cname in combo_names:
            agg = combo_agg[cname]
            # success_rate 직접, 나머지는 _mean suffix
            key = field if field in agg else f"{field}_mean"
            val = agg.get(key, float("nan"))
            if np.isnan(val):
                row += f"{'N/A':>{col_w}}"
            else:
                row += f"{format(val, fmt):>{col_w}}"
        print(row)

    print(sep)

    # 최고 성능 하이라이트
    best = min(combo_agg.items(),
               key=lambda kv: kv[1].get("composite_score_mean",
                                        kv[1].get("composite_score", 1e9)))
    n_runs = next(iter(combo_agg.values())).get("n_runs", 1)
    print(f"\n  최우수 조합: {best[0]}  "
          f"(Score {best[1].get('composite_score_mean', best[1].get('composite_score')):.2f})")
    if n_runs > 1:
        print(f"  Monte Carlo: {n_runs} runs per combination")


# =============================================================================
# 결과 JSON 저장
# =============================================================================
def save_results_json(combo_agg: dict, filepath: str) -> None:
    import json

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    clean = {}
    for k, v in combo_agg.items():
        clean[k] = {kk: _convert(vv) for kk, vv in v.items()}

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"  결과 저장: {filepath}")


# =============================================================================
# main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="VTOL 알고리즘 비교")
    parser.add_argument("scenario", help="시나리오 이름 (e.g., basic, gusty)")
    parser.add_argument("--mc", type=int, default=1,
                        help="Monte Carlo 반복 횟수 (기본 1)")
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--combos", nargs="+",
                        default=None,
                        help="실행할 조합만 선택. "
                             "형식: planner+controller (e.g. dubins+nlgl spline+mpc)")
    args = parser.parse_args()

    # 설정 로드
    aircraft_orig = load_aircraft_params()
    sim_cfg = load_simulation_params()
    scenario = load_scenario(args.scenario)
    aircraft = merge_scenario_into_aircraft(aircraft_orig, scenario)

    waypoints = waypoints_from_config(scenario)
    print(f"Scenario: {args.scenario}  WPs: {len(waypoints)}")
    for i, wp in enumerate(waypoints):
        print(f"  WP{i}: ({wp[0]:7.1f}, {wp[1]:7.1f}, h={wp[2]:.1f})")

    seed_base = (args.seed_base
                 if args.seed_base is not None
                 else scenario["monte_carlo"]["seed_base"])
    n_runs = args.mc
    print(f"\nMonte Carlo: {n_runs} run(s), seed_base={seed_base}\n")

    # 선택 조합 필터
    combos = COMBINATIONS
    if args.combos:
        selected = []
        for c in args.combos:
            parts = c.split("+")
            if len(parts) == 2:
                selected.append((parts[0], parts[1]))
        if selected:
            combos = selected

    os.makedirs(args.output_dir, exist_ok=True)

    # 조합별 실행
    combo_agg: dict[str, dict] = {}
    all_single: dict[str, tuple[MetricsResult, dict]] = {}  # 첫 번째 run만

    for planner_name, controller_name in combos:
        label = f"{planner_name}+{controller_name}"
        print(f"Running: {label} ...", flush=True)
        t0 = time.perf_counter()

        metrics_list = run_batch(
            planner_name, controller_name,
            aircraft, sim_cfg, scenario, waypoints,
            n_runs, seed_base,
        )

        elapsed = time.perf_counter() - t0
        print(f"  -> {len(metrics_list)} runs, {elapsed:.1f}s total")

        if not metrics_list:
            print(f"  [SKIP] 모든 실행 실패")
            continue

        agg = aggregate(metrics_list)
        combo_agg[label] = agg

        # 단일 결과도 보관 (시각화용)
        if n_runs == 1:
            _, res = run_one(
                planner_name, controller_name,
                aircraft, sim_cfg, scenario, waypoints, seed_base,
            )
            all_single[label] = (metrics_list[0], res)

    if not combo_agg:
        print("실행된 조합이 없습니다.")
        return

    print_comparison_table(combo_agg)

    # JSON 저장
    json_path = os.path.join(
        args.output_dir, f"compare_{args.scenario}_mc{n_runs}.json"
    )
    save_results_json(combo_agg, json_path)

    # 시각화 (단일 run만, mc=1 시)
    if not args.no_plot and n_runs == 1 and all_single:
        try:
            from visualization import plot_simulation_results
            for label, (m, res) in all_single.items():
                safe_label = label.replace("+", "_")
                plot_path = os.path.join(
                    args.output_dir,
                    f"compare_{args.scenario}_{safe_label}_s{seed_base}.png",
                )
                plot_simulation_results(
                    res, plot_path,
                    title=f"{args.scenario} | {label} | seed={seed_base}",
                    metrics_obj=m,
                    aircraft_params=aircraft,
                )
                print(f"  Plot: {plot_path}")
        except Exception as e:
            print(f"  [시각화 건너뜀: {e}]")


if __name__ == "__main__":
    main()
