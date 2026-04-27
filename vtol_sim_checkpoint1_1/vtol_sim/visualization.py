"""
시뮬레이션 결과 시각화
=========================

생성하는 그래프 (3x3 = 9 panel):
1. Top-down 2D
2. 3D 궤적
3. 고도 vs 시간
4. Cross-track 오차 (직선/곡선 색 구분)
5. 제어 입력 (bank, thrust)
6. 가속도 — 실제 + 이상(경로 요구) + 한계 + 위반 음영
7. 뱅크각 — 명령 + 이상(경로 요구)
8. Excess 가속도 (실제 - 이상) — controller 비효율 시각화
9. 종합 점수 패널 (텍스트)
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")


def plot_simulation_results(sim_result: dict, save_path: str,
                            title: str = "VTOL Simulation Result",
                            metrics_obj=None,
                            aircraft_params: dict | None = None) -> None:
    """모든 plot을 하나의 PNG로 저장.

    Parameters
    ----------
    sim_result : Simulator.run() 반환 dict
    save_path : 저장 경로
    title : 그래프 제목
    metrics_obj : MetricsResult, 종합 점수 패널 표시용 (선택)
    aircraft_params : 한계선(a_max 등) 표시용 (선택)
    """
    log = sim_result["log"]
    path = sim_result["path"]
    waypoints = sim_result["waypoints"]
    wp_records = sim_result["wp_records"]

    arr = log.to_arrays()
    actual_pos = arr["pos"]
    planned_pos = path.positions_array()
    t_arr = arr["t"]

    # 이상 프로파일 + 시간축 매핑 — visualization 자체에서 계산
    from metrics import (
        compute_path_required_profile, map_path_profile_to_time
    )
    profile = compute_path_required_profile(path)
    time_profile = map_path_profile_to_time(actual_pos, path, profile)
    a_required_t = time_profile["a_required_t"]
    phi_required_t = time_profile["phi_required_t"]
    on_curve = time_profile["on_curve"]
    g = 9.81

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(title, fontsize=14)

    # === (1) Top-down 2D ===
    ax_2d = fig.add_subplot(3, 3, 1)
    ax_2d.plot(planned_pos[:, 1], planned_pos[:, 0], "b--",
               label="Planned", alpha=0.5, linewidth=1.0)
    # ax_2d.plot(actual_pos[:, 1], actual_pos[:, 0], "r-",
    #            label="Actual", linewidth=1.0)
    ax_2d.scatter(waypoints[:, 1], waypoints[:, 0],
                  c="green", s=100, marker="o", zorder=5,
                  edgecolor="black", label="WP")
    for i, wp in enumerate(waypoints):
        ax_2d.annotate(f"{i}", (wp[1], wp[0]),
                       textcoords="offset points", xytext=(5, 5))
    for rec in wp_records:
        if rec is not None and rec.cpa_distance < 50.0:
            ax_2d.plot(rec.cpa_position[1], rec.cpa_position[0], "rx",
                       markersize=10, markeredgewidth=2)
    ax_2d.set_xlabel("East (m)")
    ax_2d.set_ylabel("North (m)")
    ax_2d.set_title("Top-down (NE plane)")
    ax_2d.legend(fontsize=8)
    ax_2d.grid(True, alpha=0.3)
    ax_2d.set_aspect("equal")

    # === (2) 3D 궤적 ===
    ax_3d = fig.add_subplot(3, 3, 2, projection="3d")
    ax_3d.plot(planned_pos[:, 1], planned_pos[:, 0], planned_pos[:, 2],
               "b--", label="Planned", alpha=0.5)
    ax_3d.plot(actual_pos[:, 1], actual_pos[:, 0], actual_pos[:, 2],
               "r-", label="Actual", linewidth=1.0)
    ax_3d.scatter(waypoints[:, 1], waypoints[:, 0], waypoints[:, 2],
                  c="green", s=80, marker="o", edgecolor="black")
    ax_3d.set_xlabel("East (m)")
    ax_3d.set_ylabel("North (m)")
    ax_3d.set_zlabel("Altitude (m)")
    ax_3d.set_title("3D Trajectory")
    ax_3d.legend(fontsize=8)

    # === (3) 고도 시간 ===
    ax_h = fig.add_subplot(3, 3, 3)
    ax_h.plot(t_arr, actual_pos[:, 2], "r-", label="Actual altitude")
    ax_h.scatter([0], [waypoints[0, 2]], c="green", s=50, zorder=5)
    for rec in wp_records[1:]:
        if rec is not None:
            ax_h.scatter([rec.cpa_time], [rec.cpa_position[2]],
                         c="green", s=50, zorder=5)
    ax_h.set_xlabel("Time (s)")
    ax_h.set_ylabel("Altitude (m)")
    ax_h.set_title("Altitude vs Time")
    ax_h.legend(fontsize=8)
    ax_h.grid(True, alpha=0.3)

    # === (4) Cross-track 오차 (직선/곡선 색 구분) ===
    from utils.math_utils import closest_point_on_polyline_local
    ax_ct = fig.add_subplot(3, 3, 4)
    polyline_2d = planned_pos[:, :2]
    ct_errors = np.zeros(len(actual_pos))
    last_seg = 0
    for i in range(len(actual_pos)):
        seg, _, _, d = closest_point_on_polyline_local(
            actual_pos[i, :2], polyline_2d, last_seg, window=100
        )
        last_seg = seg
        ct_errors[i] = d
    # 직선 구간은 파랑, 곡선 구간은 주황 — 두 시리즈로 분리하여 그림
    ct_curve = np.where(on_curve, ct_errors, np.nan)
    ct_straight = np.where(~on_curve, ct_errors, np.nan)
    ax_ct.plot(t_arr, ct_straight, "b-", linewidth=1.0, label="Straight")
    ax_ct.plot(t_arr, ct_curve, color="orange", linewidth=1.0, label="Curved")
    ax_ct.set_xlabel("Time (s)")
    ax_ct.set_ylabel("Cross-track error (m)")
    ax_ct.set_title("Cross-track Error (split by segment type)")
    ax_ct.legend(fontsize=8)
    ax_ct.grid(True, alpha=0.3)

    # === (5) 제어 입력 ===
    ax_u = fig.add_subplot(3, 3, 5)
    bank_deg = np.rad2deg(arr["bank_cmd"])
    thrust = arr["thrust_cmd"]
    # ax_u.plot(t_arr, bank_deg, "b-", label="bank_cmd (deg)", alpha=0.7)
    ax_u2 = ax_u.twinx()
    ax_u2.plot(t_arr, thrust, "g-", label="thrust_cmd", alpha=0.7)
    ax_u.set_xlabel("Time (s)")
    ax_u.set_ylabel("Bank cmd (deg)", color="b")
    ax_u2.set_ylabel("Thrust cmd", color="g")
    ax_u.set_title("Control Inputs")
    ax_u.grid(True, alpha=0.3)

    # === (6) 가속도 — 실제 + 이상 + 한계 + 위반 음영 ===
    ax_a = fig.add_subplot(3, 3, 6)
    a_actual = arr["a_total_actual"] / g
    a_max = arr["a_max_used"] / g
    a_required_g = a_required_t / g
    violations = arr["accel_violation"]
    ax_a.plot(t_arr, a_required_g, color="green", linestyle=":",
              label="a_required (path-ideal)", linewidth=1.0)
    ax_a.plot(t_arr, a_actual, "r-", label="a_actual", linewidth=1.0)
    ax_a.plot(t_arr, a_max, "k--", label="a_max", linewidth=1.0)
    if np.any(violations):
        ax_a.fill_between(t_arr, 0, a_actual, where=violations,
                          color="red", alpha=0.2, label="violation")
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Acceleration (g)")
    ax_a.set_title("Acceleration: Actual vs Path-Required")
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.3)

    # === (7) 뱅크각 — 명령 + 이상 ===
    ax_phi = fig.add_subplot(3, 3, 7)
    phi_cmd_deg = np.rad2deg(arr["bank_cmd"])
    phi_req_deg = np.rad2deg(phi_required_t)
    ax_phi.plot(t_arr, phi_req_deg, color="green", linestyle=":",
                label="|φ_required| (path)", linewidth=1.0)
    ax_phi.plot(t_arr, -phi_req_deg, color="green", linestyle=":",
                linewidth=1.0)  # 음수도 표시 (좌선회 가능)
    # ax_phi.plot(t_arr, phi_cmd_deg, "b-", label="φ_cmd (controller)",
    #             linewidth=1.0, alpha=0.8)
    if aircraft_params is not None:
        a_max_g = aircraft_params["a_max_g"]
        phi_practical = np.rad2deg(np.arctan(a_max_g))
        ax_phi.axhline(phi_practical, color="red", linestyle="--",
                       alpha=0.4, label=f"φ at a_max={a_max_g}g")
        ax_phi.axhline(-phi_practical, color="red", linestyle="--", alpha=0.4)
    ax_phi.set_xlabel("Time (s)")
    ax_phi.set_ylabel("Bank angle (deg)")
    ax_phi.set_title("Bank Angle: Commanded vs Path-Required")
    ax_phi.legend(fontsize=8)
    ax_phi.grid(True, alpha=0.3)

    # === (8) Excess 가속도 — controller 비효율 ===
    ax_ex = fig.add_subplot(3, 3, 8)
    excess = (a_actual - a_required_g)  # signed g
    ax_ex.axhline(0, color="black", linewidth=0.5)
    ax_ex.fill_between(t_arr, 0, excess, where=(excess > 0),
                       color="red", alpha=0.5, label="extra (controller adds)")
    ax_ex.fill_between(t_arr, 0, excess, where=(excess <= 0),
                       color="blue", alpha=0.5, label="deficit (under-tracking)")
    ax_ex.plot(t_arr, excess, "k-", linewidth=0.5)
    ax_ex.set_xlabel("Time (s)")
    ax_ex.set_ylabel("Excess acceleration (g)")
    ax_ex.set_title("Excess Acceleration (Actual - Path-Required)")
    ax_ex.legend(fontsize=8)
    ax_ex.grid(True, alpha=0.3)

    # === (9) 종합 점수 패널 ===
    ax_summary = fig.add_subplot(3, 3, 9)
    ax_summary.axis("off")
    if metrics_obj is not None:
        m = metrics_obj
        text = (
            f"Composite Score: {m.composite_score:.1f}\n"
            f"  (lower is better)\n"
            f"\n"
            f"-- WP & Tracking --\n"
            f"  Arrived: {m.n_wps_arrived}/{m.n_wps}\n"
            f"  Mean CPA: {m.mean_cpa:.1f} m\n"
            f"  RMS CT: {m.rms_crosstrack:.1f} m\n"
            f"    straight: {m.rms_crosstrack_straight:.1f} m\n"
            f"    curved:   {m.rms_crosstrack_curved:.1f} m\n"
            f"\n"
            f"-- Acceleration --\n"
            f"  Path-required max: {m.path_max_a_required_g:.3f}g\n"
            f"  Path feasibility: {m.path_feasibility_ratio*100:.1f}%\n"
            f"  Actual max: {m.max_a_actual_g:.3f}g\n"
            f"  Violations: {m.violation_time_ratio*100:.1f}% time\n"
            f"  Excess RMS: {m.excess_a_rms_g:.3f}g\n"
            f"\n"
            f"-- Efficiency --\n"
            f"  Path eff: {m.efficiency_vs_planned:.3f}\n"
            f"  Bank rate RMS: {m.bank_rate_rms:.3f} rad/s\n"
            f"\n"
            f"-- Time --\n"
            f"  Total: {m.total_time:.1f} s\n"
            f"  Plan: {m.planning_time_ms:.2f} ms\n"
            f"  Ctrl: {m.mean_compute_time_ms:.3f} ms (avg)\n"
        )
        ax_summary.text(
            0.0, 1.0, text, fontsize=10, family="monospace",
            verticalalignment="top", transform=ax_summary.transAxes,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
