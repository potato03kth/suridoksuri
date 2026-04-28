"""
시나리오 실행 스크립트
========================

사용:
    python run_scenario.py basic [--planner dubins] [--controller nlgl] [--seed 42]
    python run_scenario.py basic --no-plot

기본: planner=dubins, controller=nlgl, seed=시나리오 yaml의 monte_carlo.seed_base
"""
from __future__ import annotations
from visualization import plot_simulation_results
from metrics import compute_metrics, print_metrics
from simulator import Simulator
from path_following.inner_loop import InnerLoopPI
from path_following.mpc_controller import MPCController
from path_following.nlgl_controller import NLGLController
from path_planning.waypoint_generator import waypoints_from_config
from path_planning.spline_planner import SplinePlanner
from path_planning.dubins_planner import DubinsPlanner
from path_planning.bspline_planner import BSplinePlanner
from path_planning.bspline_2_planner import BSplinePlanner as BSpline2Planner
from path_planning.hermite_bspline import BSplinePlanner as HermitePlanner
from path_planning.Qhermite_bspline import BSplinePlanner as QHermitePlanner
from path_planning.iterpin_planner import IterativePinPlanner
from path_planning.D_iterpin_planner import DIterativePinPlanner
from path_planning.clothoid_planner import ClothoidPlanner
from path_planning.eta3clothoid_planner import Eta3ClothoidPlanner
from utils.config_loader import (
    load_simulation_params, load_scenario, merge_scenario_into_aircraft,
    load_aircraft_params,
)
import numpy as np
import sys
import os
import argparse
from time import localtime, strftime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass


def build_planner(name: str):
    if name == "dubins":
        return DubinsPlanner(ds=2.0, R_factor=1.05, use_energy_climb=True)
    if name == "spline":
        return SplinePlanner(ds=2.0, bc_type="clamped")
    if name == "bspline":
        return BSplinePlanner(ds=1.0, d_straight=30.0, spline_degree=5)
    if name == "bspline2":
        return BSpline2Planner(ds=1.0, straight_lead=50, spline_degree=3, max_refine_iter=5)
    if name == "hermite":
        return HermitePlanner(ds=1.0, straight_lead=50.0, max_refine_iter=30, accel_tol=0.9)
    if name == "qhermite":
        return QHermitePlanner(ds=1.0, straight_lead=50.0, max_refine_iter=30, accel_tol=0.9, tangent_scale=1.1)
    if name == "iterpin":
        return IterativePinPlanner(ds=1.0, num_iter=3, alpha=0.1, straight_ratio=0.05)
    if name == "diterpin":
        return DIterativePinPlanner(num_iter=4, alpha0=0.6, straight_ratio0=0.4, search_steps=50, max_detours=2, alpha_range=(0.1, 2.2), sr_range=(0.02, 0.8))
    if name == "clothoid":
        return ClothoidPlanner(ds=1.0, accel_tol=0.9, spiral_fraction=0.4)
    if name == "eta3clothoid":
        return Eta3ClothoidPlanner(ds=1.0, accel_tol=0.8)
    raise ValueError(f"Unknown planner: {name}")


def build_controller(name: str, aircraft_params: dict):
    v = aircraft_params["v_cruise"]
    g = aircraft_params["gravity"]
    a_max = aircraft_params["a_max_g"] * g
    R_min_op = (v ** 2) / a_max
    phi_max = np.deg2rad(aircraft_params["phi_max_deg"])

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
            phi_max_rad=np.deg2rad(45.0),  # 대형 오차 복원 허용 (구조한계 69.7° 미만)
            tau_phi=aircraft_params["tau_phi"],
            gravity=g,
            inner_loop=InnerLoopPI(),
            look_ahead_m=2.0 * R_min_op,
        )
    raise ValueError(f"Unknown controller: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", help="scenario name (e.g., 'basic')")
    parser.add_argument("--planner", default="dubins",
                        choices=["dubins", "spline", "bspline", "bspline2", "hermite", "qhermite", "iterpin", "diterpin", "clothoid", "eta3clothoid"])
    parser.add_argument("--controller", default="nlgl",
                        choices=["nlgl", "mpc"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    # 설정 로드
    aircraft_orig = load_aircraft_params()
    sim_cfg = load_simulation_params()
    scenario = load_scenario(args.scenario)

    aircraft = merge_scenario_into_aircraft(aircraft_orig, scenario)

    # 시드 결정 (WP 생성과 시뮬레이션이 동일한 시드를 사용)
    seed = args.seed if args.seed is not None else scenario["monte_carlo"]["seed_base"]
    print(f"Seed: {seed}")

    # 결정된 seed를 config에 주입한 뒤 WP 생성
    if scenario["waypoint"].get("source", "auto") == "auto":
        scenario["waypoint"]["auto"]["seed"] = seed
    waypoints = waypoints_from_config(scenario)
    print(f"Generated {len(waypoints)} waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"  WP{i}: ({wp[0]:7.1f}, {wp[1]:7.1f}, h={wp[2]:5.1f})")

    # Planner / Controller
    planner = build_planner(args.planner)
    controller = build_controller(args.controller, aircraft)
    print(f"\nUsing planner: {args.planner}")
    ctrl_info = f"L1={controller.L1:.1f}m" if hasattr(
        controller, "L1") else f"N_p={controller.N_p}"
    print(f"Using controller: {args.controller} ({ctrl_info})")

    # 시뮬레이션
    sim = Simulator(
        aircraft_params=aircraft,
        sim_cfg=sim_cfg,
        scenario_cfg=scenario,
        planner=planner,
        controller=controller,
        seed=seed,
    )
    print(f"\nRunning simulation (max duration={sim_cfg['duration_max']}s)...")
    result = sim.run(waypoints, controller_name=args.controller)

    # Metrics
    m = compute_metrics(result, aircraft_params=aircraft)
    label = f"{args.scenario} | {args.planner} + {args.controller} | seed={seed}"
    print_metrics(m, label=label)

    # Plot
    if not args.no_plot:
        os.makedirs(args.output_dir, exist_ok=True)
        time_now = strftime("%Y%m%d%H%M%S", localtime())
        plot_path = os.path.join(
            args.output_dir,
            f"{args.scenario}_{args.planner}_{args.controller}_s{seed}_{time_now}.png"
        )
        plot_simulation_results(
            result, plot_path, title=label,
            metrics_obj=m, aircraft_params=aircraft,
        )
        print(f"\nPlot saved to: {plot_path}")

    return result, m


if __name__ == "__main__":
    main()
