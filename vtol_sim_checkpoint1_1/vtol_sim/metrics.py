"""
성능 평가 지표
================

시뮬레이션 결과로부터 알고리즘 비교용 통합 metric 계산.
가속도 metrics는 utils/sim_log.py의 compute_acceleration_metrics 재사용.

추가된 분석 차원:
- 이상 프로파일 (경로 자체가 요구하는 가속도/뱅크각) — planner 평가
- Excess 가속도 (실제 - 이상) — controller 효율
- 경로 feasibility — 경로가 처음부터 한계를 어겼는지
- 직선/곡선 구간 분리 RMS — controller가 어떤 상황에서 약한지
- 종합 점수 — 한 줄 비교용 가중합
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

from utils.sim_log import SimLog, compute_acceleration_metrics
from utils.math_utils import (
    closest_point_on_polyline, signed_cross_track_error_2d,
    closest_point_on_polyline_local,
)
from path_planning.base_planner import Path


# =============================================================================
# 이상 프로파일 계산 — 경로가 요구하는 가속도/뱅크각
# =============================================================================
def compute_path_required_profile(path: Path, gravity: float = 9.81) -> dict:
    """
    경로의 각 점에서 "이상적으로 따랐을 때" 필요한 가속도/뱅크각 계산.

    구심가속도 (수평면): a_n = v² · |κ|
        (κ는 곡률 1/R, v는 그 점의 v_ref)
    종방향 가속도: a_t = v · dv/ds
        (단, 우리 planner는 v_ref가 거의 일정하므로 a_t ≈ 0)
    수직 가속도 변화 (gamma 변화에 의한): 단순화하여 무시
    이상 뱅크각: phi = arctan(a_n / g)   (수평선회 가정)

    Returns
    -------
    dict with arrays (길이 N = path 점 개수):
        a_required        : 총 가속도 크기 (m/s²) — sqrt(a_n² + a_t²)
        a_n_required      : 구심 성분
        a_t_required      : 종방향 성분
        phi_required      : 이상 뱅크각 (rad)
        v_ref             : 각 점의 속도
        s                 : 호 길이
    """
    n = len(path.points)
    a_n = np.zeros(n)
    a_t = np.zeros(n)
    phi_req = np.zeros(n)
    v_arr = np.zeros(n)
    s_arr = np.zeros(n)
    for i, p in enumerate(path.points):
        v = max(p.v_ref, 1e-3)
        v_arr[i] = v
        s_arr[i] = p.s
        # 구심 가속도 (곡률은 signed지만 크기만 사용)
        a_n[i] = (v ** 2) * abs(p.curvature)
        # 이상 뱅크각
        phi_req[i] = np.arctan(a_n[i] / gravity)

    # 종방향 가속도: dv/ds * v (호 길이 미분 사용)
    if n >= 2:
        ds = np.gradient(s_arr)
        # ds==0 보호
        ds_safe = np.where(np.abs(ds) < 1e-9, 1.0, ds)
        dv_ds = np.gradient(v_arr) / ds_safe
        a_t = v_arr * dv_ds
        # ds==0인 곳은 0으로
        a_t = np.where(np.abs(ds) < 1e-9, 0.0, a_t)

    a_total_req = np.sqrt(a_n ** 2 + a_t ** 2)

    return {
        "a_required": a_total_req,
        "a_n_required": a_n,
        "a_t_required": a_t,
        "phi_required": phi_req,
        "v_ref": v_arr,
        "s": s_arr,
        "curvature": np.array([p.curvature for p in path.points]),
    }


# =============================================================================
# 시간축으로 매핑된 이상 프로파일
# =============================================================================
def map_path_profile_to_time(
    actual_pos: np.ndarray, path: Path, profile: dict,
) -> dict:
    """
    매 시뮬레이션 step에서, 그 시점에 기체가 따라가야 했던 path 점의
    이상 가속도/뱅크각을 매핑.

    매핑 방식: 각 step의 actual_pos에서 가장 가까운 path 점 찾고 그 점의 값 사용.

    Returns
    -------
    dict with arrays (길이 T = step 개수):
        a_required_t     : 시간축 이상 가속도
        phi_required_t   : 시간축 이상 뱅크각
        on_curve         : bool 배열 — 그 시점이 곡선(|κ|>임계) 위인지
    """
    polyline_2d = path.positions_array()[:, :2]
    n_steps = len(actual_pos)
    a_req_t = np.zeros(n_steps)
    phi_req_t = np.zeros(n_steps)
    on_curve = np.zeros(n_steps, dtype=bool)

    last_seg = 0
    # 곡선 판정 임계값 — 곡률 0.001 이상이면 곡선 (R < 1000m)
    KAPPA_CURVE = 0.001

    for i in range(n_steps):
        seg, _, _, _ = closest_point_on_polyline_local(
            actual_pos[i, :2], polyline_2d, last_seg, window=100
        )
        last_seg = seg
        # path.points[seg]의 이상 값 사용
        a_req_t[i] = profile["a_required"][seg]
        phi_req_t[i] = profile["phi_required"][seg]
        on_curve[i] = abs(profile["curvature"][seg]) >= KAPPA_CURVE

    return {
        "a_required_t": a_req_t,
        "phi_required_t": phi_req_t,
        "on_curve": on_curve,
    }


# =============================================================================
# 종합 점수 (가중합)
# =============================================================================
def compute_composite_score(m: "MetricsResult",
                            weights: dict | None = None) -> float:
    """
    알고리즘 한 줄 비교용 가중합. 작을수록 좋음.

    구성:
      w_cpa     × mean CPA (m)              → WP 정확도
      w_ct      × RMS cross-track (m)        → 경로 추종
      w_viol    × violation_time_ratio × 100 → 가속도 한계 위반
      w_excess  × excess_a_rms_g × 10        → controller 비효율
      w_eff     × (1 - efficiency)            → 경로 효율
      w_smooth  × bank_rate_rms              → 제어 평활도
      w_alt     × RMS altitude error (m)     → 고도 추종

    Parameters
    ----------
    weights : dict, optional
        가중치 override. 기본값: 아래 W_DEFAULT.
    """
    W_DEFAULT = {
        "cpa": 1.0,
        "ct": 0.5,
        "viol": 5.0,
        "excess": 2.0,
        "eff": 30.0,
        "smooth": 5.0,
        "alt": 0.3,
    }
    w = {**W_DEFAULT, **(weights or {})}
    eff = m.efficiency_vs_planned if m.efficiency_vs_planned > 0 else 0.0
    score = (
        w["cpa"] * m.mean_cpa
        + w["ct"] * m.rms_crosstrack
        + w["viol"] * m.violation_time_ratio * 100.0
        + w["excess"] * m.excess_a_rms_g * 10.0
        + w["eff"] * max(0.0, 1.0 - eff)
        + w["smooth"] * m.bank_rate_rms
        + w["alt"] * m.rms_altitude_err
    )
    # 실패 페널티
    if not m.success:
        score += 1000.0 * (m.n_wps - m.n_wps_arrived)
    return float(score)


@dataclass
class MetricsResult:
    # === Waypoint 도달 ===
    n_wps: int = 0
    n_wps_arrived: int = 0
    cpa_distances: list = None        # 각 WP의 CPA distance
    max_cpa: float = 0.0
    mean_cpa: float = 0.0

    # === 경로 추종 ===
    rms_crosstrack: float = 0.0
    max_crosstrack: float = 0.0
    rms_altitude_err: float = 0.0
    max_altitude_err: float = 0.0
    # 직선/곡선 구간 분리 (사용자 제안 B 일부)
    rms_crosstrack_straight: float = 0.0  # 직선 구간만
    rms_crosstrack_curved: float = 0.0    # 곡선 구간만
    frac_time_curved: float = 0.0         # 전체 시간 대비 곡선 구간 비율

    # === 가속도 (위반 기반) ===
    max_a_actual_g: float = 0.0
    max_a_cmd_g: float = 0.0
    n_violations: int = 0
    violation_time_ratio: float = 0.0
    max_violation_g: float = 0.0
    mean_violation: float = 0.0

    # === 경로 자체의 이상 프로파일 (사용자 제안 1) ===
    # planner가 생성한 경로를 "이상적으로" 따랐을 때 필요한 가속도/뱅크각
    path_max_a_required_g: float = 0.0     # 경로가 요구하는 최대 구심가속도 (g)
    path_rms_a_required_g: float = 0.0     # RMS
    path_max_phi_required_deg: float = 0.0  # 경로가 요구하는 최대 뱅크각 (deg)
    path_rms_phi_required_deg: float = 0.0

    # === 경로 feasibility (제안 A) ===
    # 경로 자체가 한계를 넘는 비율 — planner의 잘못
    # a_required <= a_max인 점 비율 (1.0=완전 가능)
    path_feasibility_ratio: float = 1.0
    path_max_excess_g: float = 0.0         # 경로가 요구하는 가속도가 한계 초과한 양

    # === 추종 가속도 효율 (사용자 제안 3) ===
    # 실제 가속도 - 이상(경로 요구) 가속도 = controller가 추가로 만든 가속도
    excess_a_rms_g: float = 0.0            # RMS excess (controller 비효율)
    excess_a_max_g: float = 0.0            # 최대 excess
    excess_a_mean_g: float = 0.0           # 평균 excess (signed: 양수=추가 생성)

    # === 제어 평활도 ===
    bank_rate_rms: float = 0.0
    thrust_rate_rms: float = 0.0
    # 명령 뱅크각 vs 이상 뱅크각의 일탈 (사용자 제안 1과 연계)
    phi_cmd_rms_excess_deg: float = 0.0    # |phi_cmd - phi_path|의 RMS

    # === 경로 효율 ===
    actual_path_length: float = 0.0
    planned_path_length: float = 0.0
    straight_line_length: float = 0.0
    efficiency_vs_planned: float = 0.0
    efficiency_vs_straight: float = 0.0

    # === 시간 ===
    total_time: float = 0.0
    planning_time_ms: float = 0.0
    mean_compute_time_ms: float = 0.0
    max_compute_time_ms: float = 0.0

    # === 종합 점수 (제안 E) ===
    # 가중합. 작을수록 좋음. 알고리즘 한 줄 비교용.
    composite_score: float = 0.0

    # === 메타 ===
    success: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(sim_result: dict,
                    aircraft_params: dict | None = None,
                    composite_weights: dict | None = None) -> MetricsResult:
    """
    시뮬레이션 결과 dict에서 metrics 계산.

    Parameters
    ----------
    sim_result : Simulator.run() 반환 dict
    aircraft_params : 이상 프로파일 비교 시 한계값 (a_max, phi_max) 참조용
    composite_weights : 종합 점수 가중치 override
    """
    log: SimLog = sim_result["log"]
    path: Path = sim_result["path"]
    waypoints = sim_result["waypoints"]
    wp_records = sim_result["wp_records"]
    arrived_flags = sim_result["wp_arrived"]

    arr = log.to_arrays()
    pos = arr["pos"]                 # (T, 3)
    bank = arr["bank_cmd"]           # (T,)
    thrust = arr["thrust_cmd"]       # (T,)
    n_steps = len(pos)
    g = 9.81

    m = MetricsResult()

    # === WP ===
    m.n_wps = len(waypoints)
    m.n_wps_arrived = int(np.sum(arrived_flags))
    cpa_dists = [r.cpa_distance if r is not None else np.inf
                 for r in wp_records]
    m.cpa_distances = cpa_dists
    finite_cpa = [d for d in cpa_dists if np.isfinite(d)]
    if finite_cpa:
        m.max_cpa = float(np.max(finite_cpa))
        m.mean_cpa = float(np.mean(finite_cpa))

    # === 경로 추종 (cross-track) ===
    polyline_2d = path.positions_array()[:, :2]
    polyline_3d = path.positions_array()
    ct_errors = np.zeros(n_steps)
    alt_errors = np.zeros(n_steps)
    seg_at_step = np.zeros(n_steps, dtype=int)
    last_seg = 0
    for i in range(n_steps):
        seg, _, cp_2d, d_h = closest_point_on_polyline_local(
            pos[i, :2], polyline_2d, last_seg, window=100
        )
        last_seg = seg
        seg_at_step[i] = seg
        ct_errors[i] = d_h
        h_ref = polyline_3d[seg, 2]
        alt_errors[i] = abs(pos[i, 2] - h_ref)

    if n_steps > 0:
        m.rms_crosstrack = float(np.sqrt(np.mean(ct_errors ** 2)))
        m.max_crosstrack = float(np.max(ct_errors))
        m.rms_altitude_err = float(np.sqrt(np.mean(alt_errors ** 2)))
        m.max_altitude_err = float(np.max(alt_errors))

    # === 이상 프로파일 (planner 평가) ===
    profile = compute_path_required_profile(path, gravity=g)
    a_req = profile["a_required"]
    phi_req = profile["phi_required"]

    m.path_max_a_required_g = float(np.max(a_req)) / g
    m.path_rms_a_required_g = float(np.sqrt(np.mean(a_req ** 2))) / g
    m.path_max_phi_required_deg = float(np.rad2deg(np.max(np.abs(phi_req))))
    m.path_rms_phi_required_deg = float(
        np.rad2deg(np.sqrt(np.mean(phi_req ** 2)))
    )

    # === 경로 feasibility (제안 A) ===
    if aircraft_params is not None:
        a_max_path = float(aircraft_params["a_max_g"]) * g
        feasible_mask = a_req <= a_max_path
        if len(a_req) > 0:
            m.path_feasibility_ratio = float(np.mean(feasible_mask))
            excess_path = np.maximum(0.0, a_req - a_max_path)
            m.path_max_excess_g = float(np.max(excess_path)) / g
        else:
            m.path_feasibility_ratio = 1.0
            m.path_max_excess_g = 0.0

    # === Excess 가속도 (사용자 제안 3) ===
    # 시간축 이상 프로파일 매핑
    time_profile = map_path_profile_to_time(pos, path, profile)
    a_required_t = time_profile["a_required_t"]
    phi_required_t = time_profile["phi_required_t"]
    on_curve = time_profile["on_curve"]

    a_actual = arr["a_total_actual"]
    excess = a_actual - a_required_t
    if n_steps > 0:
        m.excess_a_rms_g = float(np.sqrt(np.mean(excess ** 2))) / g
        m.excess_a_max_g = float(np.max(np.abs(excess))) / g
        m.excess_a_mean_g = float(np.mean(excess)) / g

    # === 직선/곡선 분리 RMS (제안 B 일부) ===
    if n_steps > 0:
        n_curve = int(np.sum(on_curve))
        n_straight = int(np.sum(~on_curve))
        m.frac_time_curved = n_curve / n_steps
        if n_curve > 0:
            m.rms_crosstrack_curved = float(
                np.sqrt(np.mean(ct_errors[on_curve] ** 2))
            )
        if n_straight > 0:
            m.rms_crosstrack_straight = float(
                np.sqrt(np.mean(ct_errors[~on_curve] ** 2))
            )

    # === 가속도 위반 ===
    acc_metrics = compute_acceleration_metrics(log)
    m.max_a_actual_g = acc_metrics["max_a_total_actual_g"]
    m.max_a_cmd_g = acc_metrics["max_a_total_cmd_g"]
    m.n_violations = acc_metrics["n_violations"]
    m.violation_time_ratio = acc_metrics["violation_time_ratio"]
    m.max_violation_g = acc_metrics["max_violation_amount_g"]
    m.mean_violation = acc_metrics["mean_violation_amount"]

    # === 제어 평활도 ===
    if n_steps > 1:
        bank_rate = np.diff(bank) / np.diff(arr["t"])
        thrust_rate = np.diff(thrust) / np.diff(arr["t"])
        m.bank_rate_rms = float(np.sqrt(np.mean(bank_rate ** 2)))
        m.thrust_rate_rms = float(np.sqrt(np.mean(thrust_rate ** 2)))

    # 명령 뱅크각과 이상 뱅크각의 차이 — 부호 처리: |bank_cmd| vs |phi_required_t|
    # bank_cmd는 부호 있고, phi_required는 항상 양수(곡률 절댓값 사용했음)
    # 따라서 |bank_cmd|를 phi_required_t와 비교
    if n_steps > 0:
        phi_cmd_excess = np.abs(bank) - phi_required_t
        m.phi_cmd_rms_excess_deg = float(
            np.rad2deg(np.sqrt(np.mean(phi_cmd_excess ** 2)))
        )

    # === 경로 효율 ===
    actual_len = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    m.actual_path_length = actual_len
    m.planned_path_length = path.total_length
    straight_len = float(np.sum(
        np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    ))
    m.straight_line_length = straight_len
    if actual_len > 0:
        m.efficiency_vs_planned = path.total_length / actual_len
        m.efficiency_vs_straight = straight_len / actual_len

    # === 시간 ===
    m.total_time = sim_result["total_time"]
    m.planning_time_ms = sim_result["planning_time"] * 1000.0
    compute_times = arr["compute_time_control"]
    if len(compute_times) > 0:
        nonzero = compute_times[compute_times > 0]
        if len(nonzero) > 0:
            m.mean_compute_time_ms = float(np.mean(nonzero)) * 1000.0
            m.max_compute_time_ms = float(np.max(nonzero)) * 1000.0

    m.success = sim_result["success"]

    # === 종합 점수 ===
    m.composite_score = compute_composite_score(m, weights=composite_weights)

    return m


def print_metrics(m: MetricsResult, label: str = "") -> None:
    """metrics를 콘솔에 보기 좋게 출력."""
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"{'='*60}")
    print(f"  성공 여부: {'✓ 성공' if m.success else '✗ 실패'}")
    print(f"  도달 WP: {m.n_wps_arrived}/{m.n_wps}")
    print(f"  종합 점수 (낮을수록 좋음): {m.composite_score:.2f}")

    print(f"\n  [WP 도달 정확도 (CPA)]")
    if m.cpa_distances:
        for i, d in enumerate(m.cpa_distances):
            if np.isfinite(d):
                print(f"    WP {i}: {d:6.2f} m")
            else:
                print(f"    WP {i}: 미통과")
    print(f"    평균 CPA: {m.mean_cpa:.2f} m, 최대 CPA: {m.max_cpa:.2f} m")

    print(f"\n  [경로 추종 오차]")
    print(
        f"    RMS cross-track: {m.rms_crosstrack:.2f} m, max: {m.max_crosstrack:.2f} m")
    print(f"      └ 직선 구간 RMS: {m.rms_crosstrack_straight:.2f} m")
    print(f"      └ 곡선 구간 RMS: {m.rms_crosstrack_curved:.2f} m "
          f"(시간 비율 {m.frac_time_curved*100:.1f}%)")
    print(
        f"    RMS 고도 오차:   {m.rms_altitude_err:.2f} m, max: {m.max_altitude_err:.2f} m")

    print(f"\n  [경로 자체의 이상 프로파일 (planner 평가)]")
    print(f"    경로 요구 가속도: max {m.path_max_a_required_g:.3f}g, "
          f"RMS {m.path_rms_a_required_g:.3f}g")
    print(f"    경로 요구 뱅크각: max {m.path_max_phi_required_deg:.2f}°, "
          f"RMS {m.path_rms_phi_required_deg:.2f}°")
    print(f"    경로 feasibility: {m.path_feasibility_ratio*100:.1f}% "
          f"(한계 내 점 비율, 1=완전 가능)")
    if m.path_max_excess_g > 0.001:
        print(f"    ⚠ 경로 자체가 한계 초과 — 최대 {m.path_max_excess_g:.3f}g 만큼 초과")

    print(f"\n  [가속도 한계 위반 (controller 명령 결과)]")
    print(
        f"    max actual: {m.max_a_actual_g:.3f}g, max cmd: {m.max_a_cmd_g:.3f}g")
    print(
        f"    위반 횟수: {m.n_violations} ({m.violation_time_ratio*100:.1f}% of time)")
    print(
        f"    최대 위반량: {m.max_violation_g:.3f}g, 평균 위반량: {m.mean_violation:.3f} m/s²")

    print(f"\n  [Excess 가속도 (controller 비효율 = 실제 - 이상)]")
    print(f"    RMS excess: {m.excess_a_rms_g:.3f}g")
    print(f"    Max |excess|: {m.excess_a_max_g:.3f}g")
    print(f"    Mean excess: {m.excess_a_mean_g:+.3f}g (양수=추가 가속, 음수=부족 가속)")

    print(f"\n  [제어 평활도]")
    print(f"    bank_cmd 변화율 RMS: {m.bank_rate_rms:.3f} rad/s")
    print(f"    thrust_cmd 변화율 RMS: {m.thrust_rate_rms:.3f} /s")
    print(f"    |φ_cmd| − φ_required RMS: {m.phi_cmd_rms_excess_deg:.2f}° "
          f"(0에 가까울수록 이상 명령)")

    print(f"\n  [경로 효율]")
    print(f"    실제 비행: {m.actual_path_length:.1f} m")
    print(f"    계획 경로: {m.planned_path_length:.1f} m")
    print(f"    직선 거리: {m.straight_line_length:.1f} m")
    print(f"    계획 대비 효율: {m.efficiency_vs_planned:.3f}")
    print(f"    직선 대비 효율: {m.efficiency_vs_straight:.3f}")

    print(f"\n  [시간/연산]")
    print(f"    총 비행 시간: {m.total_time:.2f} s")
    print(f"    경로 생성: {m.planning_time_ms:.2f} ms")
    print(f"    평균 제어 연산: {m.mean_compute_time_ms:.3f} ms, "
          f"최대: {m.max_compute_time_ms:.3f} ms")
    print(f"{'='*60}")
