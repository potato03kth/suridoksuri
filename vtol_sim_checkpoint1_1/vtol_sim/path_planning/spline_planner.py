"""
Cubic Spline 경로 생성기 — C2 연속 (곡률 연속), WP 통과 보장
=================================================================

Dubins 대비 핵심 장점:
  - 직선↔선회 경계에서 곡률 불연속 없음 → controller 추종 개선
  - 같은 회전반경 조건에서 선행각 변화가 부드러움
  - 모든 WP를 보간점으로 통과 (위치 보장)

핵심 설계:
  1. 모든 WP를 aug_wps에 반드시 포함 (CubicSpline 보간점).
  2. 각 WP 직전/직후에 WP와 동일 고도의 guide point (G_in, G_out) 삽입:
       aug 구조: WP0, G_out0 | G_in1, WP1, G_out1 | G_in2, WP2, ...
     - G_in_i : WP_i 직전, incoming 방향으로 d_guide 거리, 고도 = WP_i
     - G_out_i: WP_i 직후, outgoing 방향으로 d_guide 거리, 고도 = WP_i
     → WP 근방 직선 구간 + 고도 편차 최소화
  3. G_out_i ~ G_in_{i+1} 구간은 자유 spline → 부드러운 곡률 전환
  4. 실제 호 길이 기반 재샘플링 (dense evaluate → uniform-s 역보간)
  5. 각 샘플에서 chi_ref, curvature, gamma_ref 계산

곡률 부호 컨벤션:
  kappa > 0 → 우선회 (항공: chi 증가 방향)
  kappa = d(chi)/ds
"""
from __future__ import annotations
import time
import numpy as np
from scipy.interpolate import CubicSpline

from .base_planner import BasePlanner, Path, PathPoint
from utils.math_utils import wrap_angle


class SplinePlanner(BasePlanner):
    def __init__(self, ds: float = 1.0, bc_type: str = "clamped",
                 R_factor: float = 1.0,
                 guide_factor: float = 0.35):
        """
        Parameters
        ----------
        ds           : 경로 샘플링 간격 (m)
        bc_type      : 'clamped' | 'natural' — 경로 양 끝단 BC
        R_factor     : 회전반경 보정 계수 (1.0 = a_max 기준 최소 반경)
        guide_factor : guide point 거리 = R_min * guide_factor
                       (단, 레그 길이의 40% 초과 불가 → G_in/G_out 비중복 보장)
        """
        self.ds = ds
        self.bc_type = bc_type
        self.R_factor = R_factor
        self.guide_factor = guide_factor

    # ------------------------------------------------------------------
    def _preprocess_waypoints(self, wps: np.ndarray,
                               R_min: float) -> tuple[np.ndarray, dict]:
        """
        모든 WP를 보간점으로 포함하며, WP 직전/직후에 guide point 삽입.

        aug 배열 구조 (중간 WP 예시):
          WP_0, G_out_0,  G_in_1, WP_1, G_out_1,  G_in_2, WP_2, ...

        G_in / G_out 은 WP와 같은 고도로 설정:
          - 고도 편차 최소화
          - G_in~WP, WP~G_out 각각 방향 일치 → WP 근방 직선 구간 형성

        G_out_i ~ G_in_{i+1} 사이 여백: 레그 길이의 최소 20%
        → 스플라인 자유 구간에서 부드러운 곡률 전환

        Returns
        -------
        aug_wps : (M, 3) 확장 WP 배열
        wp_map  : 원본 WP 인덱스 → aug 배열 내 WP 위치 매핑
        """
        N = len(wps)
        aug: list[np.ndarray] = []
        wp_map: dict[int, int] = {}

        d_base = R_min * self.guide_factor  # 기본 guide 거리

        for i in range(N):
            in_vec = wps[i, :2] - wps[i - 1, :2] if i > 0 else None
            out_vec = wps[i + 1, :2] - wps[i, :2] if i < N - 1 else None

            in_len = float(np.linalg.norm(in_vec)) if in_vec is not None else 0.0
            out_len = float(np.linalg.norm(out_vec)) if out_vec is not None else 0.0

            in_dir = (in_vec / in_len) if (in_vec is not None and in_len > 1e-6) else None
            out_dir = (out_vec / out_len) if (out_vec is not None and out_len > 1e-6) else None

            # G_in: WP 직전, incoming 방향, 고도 = WP (첫 WP 제외)
            if in_dir is not None:
                d_in = min(d_base, in_len * 0.40)
                g_in = np.array([
                    wps[i, 0] - in_dir[0] * d_in,
                    wps[i, 1] - in_dir[1] * d_in,
                    wps[i, 2],
                ])
                aug.append(g_in)

            # WP 자체 — 항상 보간점으로 포함
            aug.append(wps[i].copy())
            wp_map[i] = len(aug) - 1

            # G_out: WP 직후, outgoing 방향, 고도 = WP (마지막 WP 제외)
            if out_dir is not None:
                d_out = min(d_base, out_len * 0.40)
                g_out = np.array([
                    wps[i, 0] + out_dir[0] * d_out,
                    wps[i, 1] + out_dir[1] * d_out,
                    wps[i, 2],
                ])
                aug.append(g_out)

        return np.array(aug), wp_map

    # ------------------------------------------------------------------
    def plan(self, waypoints_ned: np.ndarray, aircraft_params: dict,
             initial_state: dict | None = None) -> Path:
        t_start = time.perf_counter()

        v_cruise = float(aircraft_params["v_cruise"])
        g = float(aircraft_params["gravity"])
        a_max_g = float(aircraft_params["a_max_g"])

        phi_practical = np.arctan(a_max_g)
        R_min = (v_cruise ** 2) / (g * np.tan(phi_practical)) * self.R_factor

        wps = np.asarray(waypoints_ned, dtype=float)
        N_orig = len(wps)
        if N_orig < 2:
            raise ValueError("waypoints는 최소 2개 필요")

        # ===== 1. WP + guide point 확장 배열 =====
        aug_wps, wp_map = self._preprocess_waypoints(wps, R_min)

        # ===== 2. Chord-length 매개변수 =====
        M = len(aug_wps)
        chords = np.zeros(M)
        for i in range(1, M):
            chords[i] = chords[i - 1] + float(
                np.linalg.norm(aug_wps[i] - aug_wps[i - 1])
            )
        total_chord = chords[-1]
        if total_chord < 1e-6:
            raise ValueError("WP들이 모두 같은 위치")

        # ===== 3. Cubic Spline 피팅 =====
        init_heading = (initial_state or {}).get("initial_heading")
        # aug_wps[0]=WP0, aug_wps[1]=G_out0 → WP0→WP1 방향이 자연스러운 시작 방향
        chi_start = (
            init_heading if init_heading is not None
            else float(np.arctan2(
                aug_wps[1, 1] - aug_wps[0, 1],
                aug_wps[1, 0] - aug_wps[0, 0],
            ))
        )
        # aug_wps[-1]=WP_{N-1}, aug_wps[-2]=G_in_{N-1} → 도착 방향
        chi_end = float(np.arctan2(
            aug_wps[-1, 1] - aug_wps[-2, 1],
            aug_wps[-1, 0] - aug_wps[-2, 0],
        ))

        if self.bc_type == "clamped" and M >= 3:
            bc_N = ((1, np.cos(chi_start)), (1, np.cos(chi_end)))
            bc_E = ((1, np.sin(chi_start)), (1, np.sin(chi_end)))
        else:
            bc_N = "natural"
            bc_E = "natural"

        cs_N = CubicSpline(chords, aug_wps[:, 0], bc_type=bc_N)
        cs_E = CubicSpline(chords, aug_wps[:, 1], bc_type=bc_E)
        cs_h = CubicSpline(chords, aug_wps[:, 2], bc_type="natural")

        # ===== 4. 실제 호 길이 계산 =====
        n_dense = max(3000, int(total_chord / 0.1) + 1)
        t_dense = np.linspace(0.0, chords[-1], n_dense)

        dN = cs_N(t_dense, 1)
        dE = cs_E(t_dense, 1)
        dh = cs_h(t_dense, 1)
        ds_dt = np.sqrt(dN ** 2 + dE ** 2 + dh ** 2)

        s_dense = np.zeros(n_dense)
        dt_arr = np.diff(t_dense)
        s_dense[1:] = np.cumsum(0.5 * (ds_dt[:-1] + ds_dt[1:]) * dt_arr)
        total_arc = s_dense[-1]

        # ===== 5. 균등 호 길이 → chord-param 역보간 =====
        n_pts = max(2, int(np.ceil(total_arc / self.ds)) + 1)
        s_uniform = np.linspace(0.0, total_arc, n_pts)
        t_uniform = np.interp(s_uniform, s_dense, t_dense)

        # ===== 6. 경로점 계산 =====
        pos_N = cs_N(t_uniform)
        pos_E = cs_E(t_uniform)
        pos_h = cs_h(t_uniform)

        dN_u = cs_N(t_uniform, 1)
        dE_u = cs_E(t_uniform, 1)
        d2N_u = cs_N(t_uniform, 2)
        d2E_u = cs_E(t_uniform, 2)
        dh_u = cs_h(t_uniform, 1)

        horiz_speed = np.sqrt(dN_u ** 2 + dE_u ** 2 + 1e-14)

        chi_arr = np.arctan2(dE_u, dN_u)
        kappa = (dN_u * d2E_u - dE_u * d2N_u) / (horiz_speed ** 3)
        gamma_arr = np.arctan2(dh_u, horiz_speed)

        # ===== 7. 원본 WP 인덱스 마킹 (aug 내 WP 위치 기준) =====
        wp_assigned: dict[int, int] = {}
        for orig_wi, aug_wi in wp_map.items():
            wp_pos = aug_wps[aug_wi]
            dists = np.hypot(pos_N - wp_pos[0], pos_E - wp_pos[1])
            best_k = int(np.argmin(dists))
            if best_k not in wp_assigned:
                wp_assigned[best_k] = orig_wi

        points: list[PathPoint] = []
        for k in range(n_pts):
            points.append(PathPoint(
                pos=np.array([pos_N[k], pos_E[k], pos_h[k]]),
                v_ref=v_cruise,
                chi_ref=float(wrap_angle(chi_arr[k])),
                gamma_ref=float(gamma_arr[k]),
                curvature=float(kappa[k]),
                s=float(s_uniform[k]),
                wp_index=wp_assigned.get(k),
            ))

        planning_time = time.perf_counter() - t_start

        return Path(
            points=points,
            waypoints_ned=wps,
            total_length=total_arc,
            planning_time=planning_time,
        )
