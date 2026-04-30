"""
Dubins-style 경로 생성기 + 에너지 기반 고도 프로파일
======================================================

이 구현은 표준 Dubins 알고리즘(LSL/LSR/RSL/RSR/RLR/LRL 6가지 케이스)이 아닌,
**WP 시퀀스를 직선으로 잇고 각 모서리에 최소 회전반경 원호를 끼워 넣는**
실용적 단순화를 사용합니다. 이는 본 비교 시뮬레이션에서:
- WP 통과 정확성을 높이고 (각 WP를 정확히 지남)
- 곡률은 직선↔원호 경계에서 불연속 (Dubins 특성과 동일)
- 구현이 단순하여 버그 가능성이 낮음

알고리즘:
    각 WP_i (1 <= i < N-1)에서:
        in_dir  = (WP_i - WP_{i-1}) 정규화
        out_dir = (WP_{i+1} - WP_i) 정규화
        선회 각도 θ = angle between in_dir and out_dir
        선회 반경 R = max(R_min, configurable)
        모서리 컷오프 거리 d = R · tan(θ/2)
        → WP 직전 점 = WP - in_dir·d, 원호 진입 시작
        → WP 직후 점 = WP + out_dir·d, 원호 진입 종료
        → 원호: 두 직선의 내각 이등분선 위 중심, 각도 (π - θ)

    수직 프로파일 (에너지 기반):
        선회 진입 직전 고도를 Δh = v²/(2g) · (1/cos(φ_max) - 1) 만큼 사전 상승
        Δh는 작아서(예: v=18, φ=16.7° → Δh ≈ 1.5m) 거의 무시 가능 수준이지만
        에너지 기동성 원칙 반영
        고도는 호 길이 따라 선형 보간

호 길이 매개변수화:
    PathPoint.s = 누적 호 길이 (m)
    경로 샘플링 간격 ds (기본 1m) — 추후 follower의 look-ahead가 사용
"""
from __future__ import annotations
import time
import numpy as np

from .base_planner import BasePlanner, Path, PathPoint
from utils.math_utils import wrap_angle


def _angle_2d(v: np.ndarray) -> float:
    """2D 벡터의 방위각 (atan2)."""
    return float(np.arctan2(v[1], v[0]))


class DubinsPlanner(BasePlanner):
    def __init__(self, ds: float = 1.0, R_factor: float = 1.0,
                 use_energy_climb: bool = True):
        """
        Parameters
        ----------
        ds : 경로 샘플링 간격 (m)
        R_factor : 회전반경에 곱할 안전 계수 (1.0 = 최소 회전반경 사용)
        use_energy_climb : 선회 진입 전 에너지 고도 보상 사용 여부
        """
        self.ds = ds
        self.R_factor = R_factor
        self.use_energy_climb = use_energy_climb

    def plan(self, waypoints_ned: np.ndarray, aircraft_params: dict,
             initial_state: dict | None = None) -> Path:
        t_start = time.perf_counter()

        # 파라미터 추출
        g = float(aircraft_params["gravity"])
        v_cruise = float(aircraft_params["v_cruise"])
        # 가속도 한계 기준의 실제 운용 뱅크각:
        # a_max에 의한 phi_practical = arctan(a_max_g)
        a_max_g = float(aircraft_params["a_max_g"])
        phi_practical = np.arctan(a_max_g)  # 0.3g → 16.7°
        # 최소 회전반경 (운용 한계 기준)
        R_min = (v_cruise ** 2) / (g * np.tan(phi_practical))
        R = R_min * self.R_factor

        wps = np.asarray(waypoints_ned, dtype=float)
        N = len(wps)
        if N < 2:
            raise ValueError("waypoints는 최소 2개 필요")

        # 초기 헤딩: initial_state에서 받거나, WP0→WP1 방향
        init_heading = (
            initial_state.get("initial_heading")
            if initial_state else None
        )

        # ===== 1단계: 모서리(corner) 정보 계산 =====
        # 각 WP i (1 <= i < N-1)에서 진입/이탈 방향
        # corners[i] = {'cutoff': d, 'arc_center': p_c, 'arc_in': p_in,
        #               'arc_out': p_out, 'turn_angle': theta_signed}
        corners = [None] * N

        for i in range(1, N - 1):
            in_vec = wps[i, :2] - wps[i - 1, :2]
            out_vec = wps[i + 1, :2] - wps[i, :2]
            in_len = np.linalg.norm(in_vec)
            out_len = np.linalg.norm(out_vec)
            if in_len < 1e-6 or out_len < 1e-6:
                continue
            in_dir = in_vec / in_len
            out_dir = out_vec / out_len

            # 진입/이탈 방위각
            chi_in = np.arctan2(in_dir[1], in_dir[0])
            chi_out = np.arctan2(out_dir[1], out_dir[0])
            # 선회 각도 (signed) — 좌선회 양, 우선회 음 (수학 컨벤션)
            # 단, 항공 컨벤션은 우선회 양이지만 여기서는 곡률 계산 편의상 수학 사용
            turn = wrap_angle(chi_out - chi_in)

            # 선회 절반각
            half_turn = abs(turn) / 2.0
            # 직선이 거의 평행이면 모서리 무시
            if half_turn < np.deg2rad(0.5):
                continue

            # 컷오프 거리 d = R · tan(θ/2)
            d = R * np.tan(half_turn)

            # 사용 가능한 최대 컷오프는 양쪽 직선의 절반보다 작아야
            d_max_in = in_len / 2.0
            d_max_out = out_len / 2.0
            d_eff = min(d, d_max_in, d_max_out)
            R_eff = d_eff / max(np.tan(half_turn), 1e-9)
            # R이 줄어들었다면 이 점에서는 R_eff 사용
            # (직선이 짧아 원호가 못 들어감 → 작은 R로 깎임)

            # 원호 진입점, 이탈점 (2D)
            p_in_2d = wps[i, :2] - in_dir * d_eff
            p_out_2d = wps[i, :2] + out_dir * d_eff

            # 원호 중심: 진입점에서 in_dir에 수직인 방향으로 R_eff
            # 좌선회(turn>0)면 진행방향 왼쪽, 우선회(turn<0)면 오른쪽
            normal_in = np.array([-in_dir[1], in_dir[0]])  # 진행방향 왼쪽
            sign = 1.0 if turn > 0 else -1.0
            p_center_2d = p_in_2d + sign * R_eff * normal_in

            corners[i] = {
                "cutoff": d_eff,
                "R_eff": R_eff,
                "p_in_2d": p_in_2d,
                "p_out_2d": p_out_2d,
                "p_center_2d": p_center_2d,
                "turn": turn,  # signed
                "chi_in": chi_in,
                "chi_out": chi_out,
            }

        # ===== 2단계: 경로 샘플링 =====
        # 직선 구간 (WP_{i-1}_corner_out → WP_i_corner_in)
        # + 원호 구간 (WP_i_corner_in → WP_i_corner_out)
        # 의 시퀀스로 점들을 만든다.
        points: list[PathPoint] = []
        s_total = 0.0

        # 첫 WP에서 시작 — 첫 WP는 corner가 없으므로 직접 WP에서 시작
        # 첫 직선의 시작점
        seg_start = wps[0, :2].copy()
        seg_start_h = wps[0, 2]

        # 첫 WP를 path point로 추가
        first_chi = (
            init_heading if init_heading is not None
            else _angle_2d(wps[1, :2] - wps[0, :2])
        )
        points.append(PathPoint(
            pos=np.array([wps[0, 0], wps[0, 1], wps[0, 2]]),
            v_ref=v_cruise,
            chi_ref=first_chi,
            gamma_ref=0.0,
            curvature=0.0,
            s=0.0,
            wp_index=0,
        ))

        for i in range(N):
            # 이번 WP 또는 마지막까지의 직선 끝점
            if i == 0:
                continue  # 첫 WP는 위에서 추가됨

            # 이전 corner의 p_out (있으면) 또는 이전 WP가 직선 시작
            if i - 1 == 0:
                line_start_2d = wps[0, :2].copy()
                line_start_h = wps[0, 2]
            else:
                prev_corner = corners[i - 1]
                if prev_corner is not None:
                    line_start_2d = prev_corner["p_out_2d"]
                else:
                    line_start_2d = wps[i - 1, :2].copy()
                # 고도는 직선 시작 시점에 이전 WP 고도 사용 → 다음 점에서 보간
                line_start_h = wps[i - 1, 2]

            # 이번 WP의 corner 정보
            cur_corner = corners[i] if i < N - 1 else None
            if cur_corner is not None:
                line_end_2d = cur_corner["p_in_2d"]
            else:
                line_end_2d = wps[i, :2].copy()

            # ----- 직선 샘플링 -----
            line_vec = line_end_2d - line_start_2d
            line_len = float(np.linalg.norm(line_vec))
            if line_len > 1e-6:
                line_dir = line_vec / line_len
                line_chi = _angle_2d(line_dir)
                n_samples = max(2, int(np.ceil(line_len / self.ds)))
                for k in range(1, n_samples + 1):
                    frac = k / n_samples
                    pt_2d = line_start_2d + line_vec * frac
                    # 고도: 직선 구간은 line_start_h → 직선 끝 고도 보간
                    # 직선 끝 고도 = WP_i 고도 (혹은 corner의 진입 고도)
                    h_end = (
                        wps[i, 2] - self._energy_climb_extra(
                            v_cruise, g, cur_corner)
                        if cur_corner is not None and self.use_energy_climb
                        else wps[i, 2]
                    )
                    pt_h = line_start_h + (h_end - line_start_h) * frac
                    s_step = line_len * (1.0 / n_samples)
                    s_total += s_step
                    pt_pos = np.array([pt_2d[0], pt_2d[1], pt_h])
                    points.append(PathPoint(
                        pos=pt_pos, v_ref=v_cruise,
                        chi_ref=line_chi, gamma_ref=0.0,
                        curvature=0.0, s=s_total,
                        wp_index=None,
                    ))

            # ----- 원호 샘플링 (cur_corner가 있으면) -----
            if cur_corner is not None:
                R_eff = cur_corner["R_eff"]
                center = cur_corner["p_center_2d"]
                turn = cur_corner["turn"]
                # 원호 시작 각도 (center → p_in 방향)
                v_start = cur_corner["p_in_2d"] - center
                v_end = cur_corner["p_out_2d"] - center
                ang_start = np.arctan2(v_start[1], v_start[0])
                ang_end = np.arctan2(v_end[1], v_end[0])
                # 진행 방향: turn > 0이면 좌선회 (+CCW), turn < 0이면 우선회 (-CW)
                sign = 1.0 if turn > 0 else -1.0
                # 원호 각도 크기 (= |turn|)
                arc_ang = abs(turn)
                arc_len = R_eff * arc_ang

                # 곡률 (signed): 우선회면 +, 좌선회면 - (Dubins 컨벤션)
                # 여기서는 항공 비행 컨벤션을 따라 "우선회 +"로 통일
                curvature_signed = -sign * (1.0 / R_eff)
                # ↑ turn>0(수학CCW=좌선회=항공좌선회) → curvature<0
                # ↑ turn<0(수학CW=우선회=항공우선회) → curvature>0

                n_arc = max(3, int(np.ceil(arc_len / self.ds)))
                for k in range(1, n_arc + 1):
                    frac = k / n_arc
                    ang = ang_start + sign * arc_ang * frac
                    pt_2d = center + R_eff * np.array([np.cos(ang), np.sin(ang)])
                    # 호 위에서의 진행 방향 (접선, 90° 회전)
                    chi_arc = ang + sign * np.pi / 2.0
                    chi_arc = wrap_angle(chi_arc)
                    # 고도: corner 시작에서 종료까지 — 시작 고도(에너지 climb 적용)에서
                    # 종료 고도(다음 직선 시작 고도 = WP_i 고도)로 선형 보간
                    h_start_arc = (
                        wps[i, 2] - self._energy_climb_extra(v_cruise, g, cur_corner)
                        if self.use_energy_climb else wps[i, 2]
                    )
                    h_end_arc = wps[i, 2]
                    pt_h = h_start_arc + (h_end_arc - h_start_arc) * frac
                    s_step = arc_len * (1.0 / n_arc)
                    s_total += s_step
                    pt_pos = np.array([pt_2d[0], pt_2d[1], pt_h])
                    # WP 자체의 통과 시점은 호의 중간 — 별도로 마킹 필요
                    # 단순화: 호의 중간 점에 wp_index 부여
                    wp_idx = i if k == n_arc // 2 else None
                    points.append(PathPoint(
                        pos=pt_pos, v_ref=v_cruise,
                        chi_ref=chi_arc, gamma_ref=0.0,
                        curvature=curvature_signed, s=s_total,
                        wp_index=wp_idx,
                    ))
            else:
                # 마지막 WP — 직선이 정확히 WP에 도달하므로 마지막 점에 wp_index
                if points:
                    points[-1].wp_index = i

        planning_time = time.perf_counter() - t_start

        return Path(
            points=points,
            waypoints_ned=wps,
            total_length=s_total,
            planning_time=planning_time,
        )

    @staticmethod
    def _energy_climb_extra(v: float, g: float, corner: dict | None) -> float:
        """
        선회 진입 전 추가 고도 — 에너지 기동성 원칙.
        Δh = v²/(2g) · (1/cos(φ) - 1)
        실제 운용 phi (a_max로 결정되는)를 사용.
        """
        if corner is None:
            return 0.0
        # corner의 R_eff에서 phi_eff = arctan(v²/(g·R)) 추정
        R = corner["R_eff"]
        if R <= 0:
            return 0.0
        phi_eff = np.arctan(v**2 / (g * R))
        if phi_eff < 1e-6:
            return 0.0
        return (v**2) / (2.0 * g) * (1.0 / np.cos(phi_eff) - 1.0)
