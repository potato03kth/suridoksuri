"""
시뮬레이션 메인 엔진
======================

기능:
1. 경로 생성 (초기 1회)
2. 메인 루프: GPS 측정 → 상태 추정 → 제어 계산 → 동역학 진행 → 로깅
3. 시간 지연 모델링:
   - 센싱 지연 (DelayBuffer)
   - 제어 계산 지연 (하이브리드 측정/모델 모드)
   - 통신 지연 (제어 컴퓨터 → FCC, DelayBuffer)
4. WP 도달 판정 (반경 + 통과평면 병용)
5. 종료 조건: 모든 WP 도달, 또는 duration 초과
"""
from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np

from dynamics.base_dynamics import (
    AircraftState, ControlInput, MODE_CRUISE
)
from dynamics.point_mass_3dof import PointMass3DoF
from dynamics.mode_manager import ModeManager
from noise.gps_noise import GPSNoise
from noise.wind_model import WindModel
from noise.actuator_noise import ActuatorNoise
from estimators.alpha_beta_filter import AlphaBetaFilter
from path_planning.base_planner import BasePlanner, Path
from path_following.base_controller import BaseController
from utils.delay_buffer import DelayBuffer
from utils.sim_log import SimLog
from utils.math_utils import wrap_angle


@dataclass
class WPArrivalRecord:
    """각 WP 도달 시 기록 — closest point of approach (CPA)."""
    wp_index: int
    cpa_distance: float          # m, 기체-WP 최단 거리
    cpa_time: float              # s, CPA 시점
    cpa_position: np.ndarray     # 기체 위치 (CPA 시)
    arrival_via: str             # 'radius' | 'plane' | 'cpa_only'
                                 # cpa_only는 도달 판정 미달이지만 가장 가까웠던 시점


class WaypointTracker:
    """
    각 WP에 대해:
      - 매 step마다 거리 측정, CPA 갱신
      - 반경 또는 통과평면 조건 충족 시 'arrived' 마킹
      - 도달 시점의 CPA 저장
    """

    def __init__(self, waypoints: np.ndarray,
                 R_accept: float, R_accept_plane_factor: float = 2.0):
        self.waypoints = waypoints
        self.R_accept = R_accept
        self.R_accept_plane = R_accept * R_accept_plane_factor
        self.n = len(waypoints)
        # 각 WP의 통과 평면 법선 (단위벡터, NED 평면) — WP 진입 방향
        self.tangents = self._compute_tangents()

        # 추적 상태
        self.arrived = [False] * self.n
        self.cpa: list[WPArrivalRecord | None] = [None] * self.n
        # 매 step에서 갱신: WP에 가장 가까운 거리, 시점, 위치
        self._cpa_dist = [np.inf] * self.n
        self._cpa_time = [0.0] * self.n
        self._cpa_pos: list[np.ndarray] = [np.zeros(3) for _ in range(self.n)]
        self._prev_dot_t: list[float] = [0.0] * self.n
        # ↑ 통과평면 dot product 이전 값 (부호 변화 감지용)

    def _compute_tangents(self) -> np.ndarray:
        """각 WP의 진입 방향 (단위벡터). 마지막 WP는 이전 방향 사용."""
        N = self.n
        tans = np.zeros((N, 3))
        for i in range(N):
            if i == 0:
                v = self.waypoints[1] - self.waypoints[0]
            elif i == N - 1:
                v = self.waypoints[i] - self.waypoints[i - 1]
            else:
                # 이전과 다음의 평균 방향
                v_in = self.waypoints[i] - self.waypoints[i - 1]
                v_out = self.waypoints[i + 1] - self.waypoints[i]
                v = v_in / max(np.linalg.norm(v_in), 1e-9) + \
                    v_out / max(np.linalg.norm(v_out), 1e-9)
            n = np.linalg.norm(v)
            tans[i] = v / max(n, 1e-9)
        return tans

    def update(self, pos: np.ndarray, t: float, target_idx: int) -> bool:
        """
        target_idx (현재 추적 중인 WP)에 대해 매 step 업데이트.
        도달 여부 반환.

        도달 판정 우선순위:
        1. 반경 조건: pos-WP < R_accept
        2. 통과평면 조건: tangent · (pos-wp) 부호 변화 + 수직거리 < R_accept_plane
        3. CPA-departure 조건: 최단거리 경신 멈춘 후 cpa_min*1.5 + 5m 이상 멀어짐
           (WP에서 빗나가더라도 가장 가까웠던 시점을 도달로 인정)
        """
        if self.arrived[target_idx]:
            return True

        wp = self.waypoints[target_idx]
        d = float(np.linalg.norm(pos - wp))

        # CPA 갱신
        if d < self._cpa_dist[target_idx]:
            self._cpa_dist[target_idx] = d
            self._cpa_time[target_idx] = t
            self._cpa_pos[target_idx] = pos.copy()

        # 1. 반경 조건
        arrived_radius = d < self.R_accept

        # 2. 통과평면 조건
        rel = pos - wp
        dot_t = float(self.tangents[target_idx] @ rel)
        passed_plane = (
            self._prev_dot_t[target_idx] <= 0 and dot_t > 0
        )
        perp = rel - dot_t * self.tangents[target_idx]
        d_perp = float(np.linalg.norm(perp))
        arrived_plane = passed_plane and d_perp < self.R_accept_plane
        self._prev_dot_t[target_idx] = dot_t

        # 3. CPA-departure 조건 (가장 가까운 점에서 충분히 멀어짐)
        cpa_min = self._cpa_dist[target_idx]
        # 최단거리 갱신 멈춘 상태에서 현재 거리가 cpa_min의 1.5배 + 5m 이상이면 통과
        # 즉 "이미 가장 가까웠고 이제 멀어지고 있다" 판정
        threshold_far = cpa_min * 1.5 + 5.0
        arrived_cpa_dep = (cpa_min < np.inf) and (d > threshold_far) and (d > cpa_min + 5.0)

        if arrived_radius or arrived_plane or arrived_cpa_dep:
            if arrived_radius:
                via = "radius"
            elif arrived_plane:
                via = "plane"
            else:
                via = "cpa_departure"
            self.arrived[target_idx] = True
            self.cpa[target_idx] = WPArrivalRecord(
                wp_index=target_idx,
                cpa_distance=self._cpa_dist[target_idx],
                cpa_time=self._cpa_time[target_idx],
                cpa_position=self._cpa_pos[target_idx],
                arrival_via=via,
            )
            return True
        return False

    def finalize(self) -> None:
        """시뮬 종료 시 미도달 WP들에 대해 CPA만 기록."""
        for i in range(self.n):
            if not self.arrived[i] and self._cpa_dist[i] < np.inf:
                self.cpa[i] = WPArrivalRecord(
                    wp_index=i,
                    cpa_distance=self._cpa_dist[i],
                    cpa_time=self._cpa_time[i],
                    cpa_position=self._cpa_pos[i],
                    arrival_via="cpa_only",
                )


class Simulator:
    """
    메인 시뮬레이션 엔진.

    사용 예시:
        sim = Simulator(aircraft_params, sim_cfg, scenario_cfg, planner, controller)
        result = sim.run(waypoints_ned)
        # result는 (log, wp_records, planning_time, total_time, success)
    """

    def __init__(self,
                 aircraft_params: dict,
                 sim_cfg: dict,
                 scenario_cfg: dict,
                 planner: BasePlanner,
                 controller: BaseController,
                 seed: int | None = None):
        self.aircraft_params = aircraft_params
        self.sim_cfg = sim_cfg
        self.scenario_cfg = scenario_cfg
        self.planner = planner
        self.controller = controller

        # === 동역학 ===
        self.dynamics = PointMass3DoF(aircraft_params)
        self.mode_mgr = ModeManager(
            h_transition=aircraft_params["h_transition"],
            v_cruise=aircraft_params["v_cruise"],
        )

        # === 노이즈 ===
        noise_cfg = scenario_cfg["noise"]
        self.gps = GPSNoise(
            sigma_white=noise_cfg["gps_sigma"],
            tau_bias=noise_cfg["gps_bias_tau"],
            sigma_bias_drive=noise_cfg["gps_bias_drive_sigma"],
            seed=seed,
        )
        self.wind = WindModel(
            w_steady=tuple(noise_cfg["wind_steady"]),
            sigma_gust=noise_cfg["wind_gust_sigma"],
            length_scale=noise_cfg["wind_gust_length_scale"],
            seed=(seed + 1) if seed is not None else None,
        )
        self.actuator = ActuatorNoise(
            deadband_rad=np.deg2rad(noise_cfg["actuator_deadband_deg"]),
            bias_sigma_rad=np.deg2rad(noise_cfg["actuator_bias_sigma_deg"]),
            seed=(seed + 2) if seed is not None else None,
        )

        # === 상태 추정 ===
        ab_cfg = sim_cfg["alpha_beta"]
        self.estimator = AlphaBetaFilter(
            alpha=ab_cfg["alpha"], beta=ab_cfg["beta"],
        )

        # === 시간 지연 ===
        # 시나리오에서 override 가능
        delay_overrides = scenario_cfg.get("delay_overrides", {})
        self.dt_physics = sim_cfg["dt_physics"]
        self.gps_period = sim_cfg["gps_period"]
        self.ctrl_period = sim_cfg["control_period"]
        self.delay_sensing = delay_overrides.get(
            "delay_sensing", sim_cfg["delay_sensing"])
        self.delay_comm = delay_overrides.get(
            "delay_comm_to_fcc", sim_cfg["delay_comm_to_fcc"])
        # 하이브리드 모드
        self.delay_mode = sim_cfg["delay_mode"]
        self.modeled_compute = sim_cfg.get("modeled_compute_time", {})

        # === 종료 조건 ===
        self.duration_max = sim_cfg["duration_max"]

        # === WP 도달 판정 ===
        wp_acc_cfg = sim_cfg["waypoint_acceptance"]
        self.R_accept = max(
            wp_acc_cfg["R_accept_min"],
            noise_cfg["gps_sigma"] * wp_acc_cfg["R_accept_factor"],
        )
        self.R_accept_plane_factor = wp_acc_cfg["R_accept_plane_factor"]

    def run(self, waypoints_ned: np.ndarray,
            controller_name: str = "nlgl") -> dict:
        """
        시뮬레이션 실행. controller_name은 modeled compute time lookup용.
        """
        # ===== 1. 경로 생성 =====
        t0 = time.perf_counter()
        path = self.planner.plan(waypoints_ned, self.aircraft_params)
        planning_wall = time.perf_counter() - t0

        # ===== 2. 초기 상태 =====
        start_pos = waypoints_ned[0].copy()
        # 초기 헤딩: WP0 → WP1
        v0 = waypoints_ned[1, :2] - waypoints_ned[0, :2]
        init_chi = float(np.arctan2(v0[1], v0[0]))
        v_cruise = self.aircraft_params["v_cruise"]
        state = AircraftState(
            pos=start_pos,
            v=v_cruise,
            chi=init_chi,
            gamma=0.0,
            phi=0.0,
            mode=MODE_CRUISE,
            t=0.0,
        )

        # 동역학 초기화
        self.dynamics.reset(state)
        self.controller.reset()

        # 상태 추정기 초기화 (NED 속도 = 초기 chi 방향 v_cruise)
        init_vel = np.array([v_cruise * np.cos(init_chi),
                             v_cruise * np.sin(init_chi),
                             0.0])
        self.estimator.initialize(start_pos, init_vel, 0.0)

        # 지연 버퍼 초기화 — 제어 명령용
        zero_ctrl = ControlInput()
        comm_buf = DelayBuffer(
            delay=self.delay_comm,
            dt=self.dt_physics,
            init_value=zero_ctrl,
        )
        # 센싱 지연 — GPS 측정 결과를 지연시킴
        # (센싱 지연이 매우 작을 수 있으므로 빈도는 GPS 주기 = 0.1s)
        sens_buf_dt = self.gps_period
        sens_buf = DelayBuffer(
            delay=self.delay_sensing,
            dt=max(sens_buf_dt, self.dt_physics),
            init_value=start_pos.copy(),
        )

        # ===== 3. 메인 루프 =====
        log = SimLog()
        wp_tracker = WaypointTracker(
            waypoints_ned, self.R_accept,
            R_accept_plane_factor=self.R_accept_plane_factor,
        )
        target_idx = 1  # 첫 WP는 시작점이므로 두 번째부터 추적
        wp_tracker.arrived[0] = True
        wp_tracker.cpa[0] = WPArrivalRecord(
            wp_index=0, cpa_distance=0.0, cpa_time=0.0,
            cpa_position=start_pos.copy(), arrival_via="radius",
        )

        # 추정 상태 캐시 (제어 주기 사이에 reuse)
        est_pos = start_pos.copy()
        est_vel = init_vel.copy()
        last_ctrl = zero_ctrl
        last_compute_time = 0.0

        # GPS / 제어 다음 트리거 시각
        next_gps_t = 0.0
        next_ctrl_t = 0.0
        success = False
        t = 0.0

        while t < self.duration_max:
            # ---- (1) GPS 측정 + 상태 추정 (주기적) ----
            if t >= next_gps_t:
                self.gps.step_bias(self.gps_period)
                z_raw = self.gps.measure(state.pos)
                # 센싱 지연 적용
                z = sens_buf.update(z_raw)
                est_pos, est_vel = self.estimator.update(z, t)
                next_gps_t += self.gps_period

            # ---- (2) 제어 계산 (주기적, 측정/모델 시간 기록) ----
            if t >= next_ctrl_t:
                t_ctrl_start = time.perf_counter()
                u_cmd = self.controller.compute(
                    est_pos=est_pos,
                    est_vel=est_vel,
                    est_chi=state.chi,    # 자세는 IMU가 알려준다 가정
                    est_gamma=state.gamma,
                    est_phi=state.phi,
                    est_v=state.v,
                    path=path,
                    t=t,
                    dt=self.ctrl_period,
                )
                t_ctrl_wall = time.perf_counter() - t_ctrl_start
                # 하이브리드: 측정 사용
                if self.delay_mode in ("measured", "hybrid"):
                    last_compute_time = t_ctrl_wall
                else:  # modeled
                    last_compute_time = self.modeled_compute.get(
                        controller_name, 0.001)
                last_ctrl = u_cmd
                next_ctrl_t += self.ctrl_period

            # ---- (3) 통신 지연 ----
            # 매 dt마다 update — 지연 시간만큼 buffer를 통과
            u_after_comm = comm_buf.update(last_ctrl)

            # ---- (4) 액추에이터 노이즈 적용 ----
            u_actual = self.actuator.apply(u_after_comm, state.phi)

            # ---- (5) 바람 + 동역학 ----
            wind = self.wind.step(state.v, self.dt_physics)
            state = self.dynamics.step(state, u_actual, wind, self.dt_physics)

            # ---- (6) 로깅 ----
            log.append_step(
                state, u_actual, pos_est=est_pos,
                wind=wind, compute_time_ctrl=last_compute_time,
            )

            # ---- (7) WP 추적 ----
            if target_idx < len(waypoints_ned):
                arrived = wp_tracker.update(state.pos, t, target_idx)
                if arrived:
                    target_idx += 1

            # 모든 WP 도달 시 종료
            if target_idx >= len(waypoints_ned):
                success = True
                break

            t += self.dt_physics

        # ===== 4. 마무리 =====
        wp_tracker.finalize()
        log.compute_time_planning = planning_wall

        return {
            "log": log,
            "path": path,
            "waypoints": waypoints_ned,
            "wp_records": wp_tracker.cpa,
            "wp_arrived": wp_tracker.arrived,
            "planning_time": planning_wall,
            "total_time": t,
            "success": success,
        }
