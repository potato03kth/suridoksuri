"""
Straight-Line Planner — 쿼드(멀티콥터) 검증 비행용 최소 planner
================================================================

목적: 실기체 브링업에서 planner 요인을 배제하고 offboard 경로추종을
빠르게 증명하기 위한 planner. WP 사이를 3D 직선으로 잇고 ds 간격으로
샘플링한다. NR/최적화가 전혀 없어 RPi5에서도 planning이 즉시 끝난다.

특성:
  • 3D 직선 연결 — 쿼드는 임의 방향 추종이 가능하므로 곡률 완화 불필요
  • 수직 상승 구간(같은 N,E, 고도만 다른 WP)도 표현 가능
    (s는 3D 호 길이라서 strictly increasing 유지 → np.gradient NaN 불가)
  • curvature = 0 고정, chi_ref는 수평 진행 방향(수직 구간은 직전 값 유지)
  • 완전히 동일한 연속 WP는 병합 + 경고

고정익(FW)에는 부적합 — 곡률 불연속(WP 꺾임)을 그대로 노출한다.
쿼드 전용 또는 planner 배제 검증용으로만 사용할 것.
"""
from __future__ import annotations
import time
import numpy as np
from .base_planner import BasePlanner, Path, PathPoint


class StraightLinePlanner(BasePlanner):
    """WP 간 3D 직선 + 등간격 샘플링. 쿼드/검증 비행용."""

    def __init__(self, ds: float = 1.0, min_wp_dist: float = 1e-6):
        self.ds = ds
        self.min_wp_dist = min_wp_dist

    def plan(self,
             waypoints_ned: np.ndarray,
             aircraft_params: dict,
             initial_state: dict | None = None) -> Path:
        t0 = time.perf_counter()
        wps = np.asarray(waypoints_ned, dtype=float)
        if len(wps) < 2:
            raise ValueError("waypoints는 최소 2개 필요")

        # 완전 중복 WP 병합 (3D 거리 기준 — 수직 구간은 유효하므로 유지)
        keep = [0]
        for i in range(1, len(wps)):
            if np.linalg.norm(wps[i] - wps[keep[-1]]) >= self.min_wp_dist:
                keep.append(i)
        if len(keep) != len(wps):
            dropped = sorted(set(range(len(wps))) - set(keep))
            print(f"[StraightLinePlanner] WARNING: merged fully-duplicate 3D WPs {dropped}",
                  flush=True)
            wps = wps[keep]
            if len(wps) < 2:
                raise ValueError("중복 병합 후 waypoints가 2개 미만입니다.")

        v_cruise = float(aircraft_params["v_cruise"])

        pts: list[np.ndarray] = []
        wp_marks: dict[int, int] = {}
        chi_prev = 0.0
        if initial_state and "initial_heading" in initial_state:
            chi_prev = float(initial_state["initial_heading"])

        chi_list: list[float] = []
        gamma_list: list[float] = []

        for k in range(len(wps) - 1):
            seg = wps[k + 1] - wps[k]
            L = float(np.linalg.norm(seg))
            L_xy = float(np.hypot(seg[0], seg[1]))
            chi_k = float(np.arctan2(seg[1], seg[0])) if L_xy > 1e-9 else chi_prev
            gamma_k = float(np.arctan2(seg[2], L_xy))
            chi_prev = chi_k

            n = max(2, int(np.ceil(L / self.ds)) + 1)
            t = np.linspace(0.0, 1.0, n)
            seg_pts = wps[k] + t[:, None] * seg

            last = k == len(wps) - 2
            if not last:
                seg_pts = seg_pts[:-1]          # 다음 segment 시작점과 중복 방지
                n_used = n - 1
            else:
                n_used = n

            wp_marks[len(pts)] = k              # segment 시작점 = WP k
            pts.extend(seg_pts)
            chi_list.extend([chi_k] * n_used)
            gamma_list.extend([gamma_k] * n_used)

        wp_marks[len(pts) - 1] = len(wps) - 1   # 마지막 점 = 마지막 WP

        pts_arr = np.asarray(pts)
        diffs = np.diff(pts_arr, axis=0)
        s_arr = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])

        points: list[PathPoint] = []
        for idx in range(len(pts_arr)):
            points.append(PathPoint(
                pos=pts_arr[idx].copy(),
                v_ref=v_cruise,
                chi_ref=chi_list[idx],
                gamma_ref=gamma_list[idx],
                curvature=0.0,
                s=float(s_arr[idx]),
                wp_index=wp_marks.get(idx, None),
            ))

        return Path(
            points=points,
            waypoints_ned=wps,
            total_length=float(s_arr[-1]),
            planning_time=time.perf_counter() - t0,
        )
