"""
IterativePinPlanner v1.1
========================
반복적 핀 기반 경로 생성기.
- 핀(Pin): 각 WP에 entry/exit 점, 역순 반복으로 수렴
- 경로 5구간: 직선 → 완화곡선 → 자유곡선 → 완화곡선 → 직선
- 모든 곡선: Quintic Hermite, 접선 크기 trans_len 통일 → C2 연속
- 마지막 WP만 endpoint_extension 만큼 직선 연장 → WP 통과 보장
- NED 좌표계: 우선회 양 곡률 / 좌선회 음 곡률
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import time
import warnings

from .base_planner import BasePlanner, Path, PathPoint


# ============================================================
# IterativePinPlanner
# ============================================================
def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n < eps else v / n


class IterativePinPlanner(BasePlanner):
    """
    반복적 핀 기반 경로 생성기.

    Parameters
    ----------
    num_iter : int, default 3
        핀 계산 반복 횟수.
    alpha : float, default 0.35
        핀 스케일 계수. scale = d_avg * alpha * f(angle)
    straight_ratio : float, default 0.20
        직선구간 비율. trans_len = seg_len * straight_ratio * 0.6
    ds : float, default 1.0
        경로 샘플링 간격 (m).
    endpoint_extension : float, default 10.0
        마지막 WP 통과 보장을 위한 직선 연장 길이 (m).
        첫 WP는 연장하지 않음.
    resample_uniform : bool, default True
        True  : (A) 균등 ds 호 길이 재샘플링
        False : (B) 세그먼트별 직접 샘플링
    segment_samples : int, default 50
        Quintic Hermite 세그먼트 내부 샘플 수.
    """

    def __init__(self,
                 num_iter: int = 3,
                 alpha: float = 0.35,
                 straight_ratio: float = 0.20,
                 ds: float = 1.0,
                 endpoint_extension: float = 10.0,
                 resample_uniform: bool = True,
                 segment_samples: int = 50):
        self.num_iter = num_iter
        self.alpha = alpha
        self.straight_ratio = straight_ratio
        self.ds = ds
        self.endpoint_extension = endpoint_extension
        self.resample_uniform = resample_uniform
        self.segment_samples = segment_samples

    # ─── 공개 인터페이스 ────────────────────────────────────
    def plan(self, waypoints_ned, aircraft_params, initial_state=None) -> Path:
        t0 = time.perf_counter()
        wps = np.asarray(waypoints_ned, dtype=float)
        N = len(wps)
        if N < 2:
            raise ValueError(f"WP 개수 부족: N={N}, 최소 2 이상 필요")

        wps_2d = wps[:, :2]
        pins = self._compute_pins(wps_2d)
        segs = self._build_segments(pins, wps_2d)
        path = self._to_path(segs, wps, aircraft_params)
        path.planning_time = time.perf_counter() - t0
        return path

    # ─── 1. 핀 계산 ─────────────────────────────────────────
    def _compute_pins(self, wps_2d: np.ndarray):
        N = len(wps_2d)

        if N == 2:
            d = _normalize(wps_2d[1] - wps_2d[0])
            entries = wps_2d.copy()
            exits = np.array([wps_2d[0],
                              wps_2d[1] + d * self.endpoint_extension])
            return entries, exits

        entries = wps_2d.copy()
        exits = wps_2d.copy()

        for it in range(self.num_iter):
            for i in range(N - 1, -1, -1):
                self._compute_pin_single(i, it, entries, exits, wps_2d)

        # iter 종료 후 마지막 WP만 연장
        dIn_last = _normalize(wps_2d[N - 1] - entries[N - 1])
        exits[N - 1] = wps_2d[N - 1] + dIn_last * self.endpoint_extension
        return entries, exits

    def _compute_pin_single(self, i, it, entries, exits, wps):
        """단일 WP 핀 갱신. it==0이면 WP 기준, it>=1이면 핀 기준."""
        N = len(wps)
        ref_prev = (lambda k: wps[k - 1]
                    ) if it == 0 else (lambda k: exits[k - 1])
        ref_next = (lambda k: wps[k + 1]
                    ) if it == 0 else (lambda k: entries[k + 1])
        ref_in = (lambda k: wps[k]) if it == 0 else (lambda k: entries[k])
        ref_out = (lambda k: wps[k]) if it == 0 else (lambda k: exits[k])

        # 첫 WP: 탈출만
        if i == 0:
            dOut = _normalize(ref_next(0) - ref_out(0))
            scale = np.linalg.norm(ref_next(0) - wps[0]) * self.alpha
            entries[0] = wps[0]
            exits[0] = wps[0] + dOut * scale
            return

        # 마지막 WP: 진입만 (연장은 iter 후 별도 적용)
        if i == N - 1:
            dIn = _normalize(ref_in(N - 1) - ref_prev(N - 1))
            scale = np.linalg.norm(wps[N - 1] - ref_prev(N - 1)) * self.alpha
            entries[N - 1] = wps[N - 1] - dIn * scale
            exits[N - 1] = wps[N - 1]
            return

        # 중간 WP
        dIn = _normalize(ref_in(i) - ref_prev(i))
        dOut = _normalize(ref_next(i) - ref_out(i))
        rvN = _normalize(dIn + dOut)
        if np.linalg.norm(rvN) < 1e-12:
            # 180도 U턴: dIn에 수직한 방향으로 임시 배치
            rvN = np.array([-dIn[1], dIn[0]])

        angle = np.arccos(np.clip(np.dot(dIn, dOut), -1.0, 1.0))
        d_avg = 0.5 * (np.linalg.norm(wps[i] - ref_prev(i)) +
                       np.linalg.norm(ref_next(i) - wps[i]))
        scale = d_avg * self.alpha * (0.2 + angle / np.pi * 0.8)

        entry_c = wps[i] - rvN * scale
        exit_c = wps[i] + rvN * scale

        # 레이블 추적 (뒤집힘 swap)
        if np.dot(wps[i] - entry_c, dIn) >= 0:
            entries[i] = entry_c
            exits[i] = exit_c
        else:
            entries[i] = exit_c
            exits[i] = entry_c

    # ─── 2. 세그먼트 생성 ──────────────────────────────────
    def _build_segments(self, pins, wps_2d):
        entries, exits = pins
        N = len(wps_2d)

        # 각 WP의 직선 방향
        lineDirs = np.zeros_like(entries)
        for i in range(N):
            v = exits[i] - entries[i]
            n = np.linalg.norm(v)
            if n < 1e-12:
                lineDirs[i] = (_normalize(wps_2d[i + 1] - wps_2d[i])
                               if i + 1 < N
                               else _normalize(wps_2d[i] - wps_2d[i - 1]))
            else:
                lineDirs[i] = v / n

        segments = [{'type': 'straight',
                     'p0': entries[0].copy(), 'p1': exits[0].copy(),
                     'wp_idx': 0}]

        for i in range(N - 1):
            self._add_inter_segments(exits[i], entries[i + 1],
                                     lineDirs[i], lineDirs[i + 1], segments)
            segments.append({'type': 'straight',
                             'p0': entries[i + 1].copy(),
                             'p1': exits[i + 1].copy(),
                             'wp_idx': i + 1})
        return segments

    def _add_inter_segments(self, exit_i, entry_next, lD_i, lD_n, segs):
        """두 직선 사이의 곡선 [완화 → 자유 → 완화] 또는 [완화 → 완화]."""
        L = np.linalg.norm(entry_next - exit_i)
        if L < 1e-9:
            return
        tl_nom = L * self.straight_ratio * 0.6

        if 2 * tl_nom < L:
            # 정상: 자유곡선 공간 있음
            tl = tl_nom
            tA = exit_i + lD_i * tl
            tB = entry_next - lD_n * tl

            segs.append({'type': 'transition_out',
                         'pts': self._qh(exit_i, lD_i*tl, np.zeros(2),
                                         tA, lD_i*tl, np.zeros(2),
                                         self.segment_samples),
                         'wp_idx': None})
            # 자유곡선: m 크기를 trans_len으로 통일 → C2 연속
            segs.append({'type': 'free',
                         'pts': self._qh(tA, lD_i*tl, np.zeros(2),
                                         tB, lD_n*tl, np.zeros(2),
                                         self.segment_samples),
                         'wp_idx': None})
            segs.append({'type': 'transition_in',
                         'pts': self._qh(tB, lD_n*tl, np.zeros(2),
                                         entry_next, lD_n*tl, np.zeros(2),
                                         self.segment_samples),
                         'wp_idx': None})
        else:
            # 자유곡선 생략: 완화 두 개 직접 연결
            tl = L * 0.5
            mid = 0.5 * (exit_i + entry_next)
            segs.append({'type': 'transition_out',
                         'pts': self._qh(exit_i, lD_i*tl, np.zeros(2),
                                         mid, lD_n*tl, np.zeros(2),
                                         self.segment_samples),
                         'wp_idx': None})
            segs.append({'type': 'transition_in',
                         'pts': self._qh(mid, lD_n*tl, np.zeros(2),
                                         entry_next, lD_n*tl, np.zeros(2),
                                         self.segment_samples),
                         'wp_idx': None})

    @staticmethod
    def _qh(p0, m0, a0, p1, m1, a1, n=50) -> np.ndarray:
        """Quintic Hermite. n+1개 점 반환."""
        t = np.linspace(0.0, 1.0, n + 1)
        t2, t3, t4, t5 = t * t, t**3, t**4, t**5
        h0 = 1 - 10*t3 + 15*t4 - 6*t5
        h1 = t - 6*t3 + 8*t4 - 3*t5
        h2 = (t2 - 3*t3 + 3*t4 - t5) * 0.5
        h3 = 10*t3 - 15*t4 + 6*t5
        h4 = -4*t3 + 7*t4 - 3*t5
        h5 = (t3 - 2*t4 + t5) * 0.5
        return (h0[:, None]*p0 + h1[:, None]*m0 + h2[:, None]*a0
                + h3[:, None]*p1 + h4[:, None]*m1 + h5[:, None]*a1)

    # ─── 3. Path 변환 ───────────────────────────────────────
    def _to_path(self, segments, wps_3d, aircraft_params) -> Path:
        v_cruise = aircraft_params.get('v_cruise', 30.0)

        # 모든 점 통합
        all_pts = []
        for seg in segments:
            pts = (np.array([seg['p0'], seg['p1']])
                   if seg['type'] == 'straight'
                   else seg['pts'])
            if all_pts and np.linalg.norm(all_pts[-1][-1] - pts[0]) < 1e-9:
                pts = pts[1:]
            if len(pts) > 0:
                all_pts.append(pts)
        raw_2d = np.vstack(all_pts)

        # 샘플링
        if self.resample_uniform:
            sampled_2d, s_arr = self._uniform_resample(raw_2d, self.ds)
        else:
            sampled_2d, s_arr = self._segment_resample(segments, self.ds)

        # 고도 보간
        wp_s = self._compute_wp_arclengths(wps_3d, sampled_2d, s_arr)
        h_arr = np.interp(s_arr, wp_s, wps_3d[:, 2])
        pos_3d = np.column_stack([sampled_2d, h_arr])

        # 방위각 / 곡률 (호 길이 기반 미분)
        chi_arr = self._compute_heading(sampled_2d, s_arr)
        kappa_arr = self._compute_curvature(sampled_2d, s_arr)

        # PathPoint 생성
        points = [PathPoint(pos=pos_3d[k].copy(),
                            v_ref=v_cruise,
                            chi_ref=float(chi_arr[k]),
                            gamma_ref=0.0,
                            curvature=float(kappa_arr[k]),
                            s=float(s_arr[k]),
                            wp_index=None)
                  for k in range(len(sampled_2d))]

        # WP 마킹: 각 WP에 가장 가까운 단일 점
        for wi, wp in enumerate(wps_3d):
            d = np.linalg.norm(sampled_2d - wp[:2], axis=1)
            points[int(np.argmin(d))].wp_index = wi

        # 곡률 한계 검사 (경고만)
        phi_max = aircraft_params.get('phi_max_deg', None)
        if phi_max is not None and v_cruise > 0:
            kappa_max = 9.81 * np.tan(np.radians(phi_max)) / v_cruise**2
            if np.any(np.abs(kappa_arr) > kappa_max):
                warnings.warn(
                    f"곡률 초과: max|kappa|={np.max(np.abs(kappa_arr)):.4f} "
                    f"> kappa_max={kappa_max:.4f}. "
                    f"alpha 또는 straight_ratio 조정 권장")

        return Path(points=points,
                    waypoints_ned=np.asarray(wps_3d, dtype=float),
                    total_length=float(s_arr[-1]) if len(s_arr) else 0.0)

    # ─── 보조: 샘플링 ───
    @staticmethod
    def _uniform_resample(pts_2d, ds):
        """(A) 모든 점을 이은 뒤 균등 호 길이로 재샘플링."""
        if len(pts_2d) < 2:
            return pts_2d.copy(), np.array([0.0])
        seg_d = np.linalg.norm(np.diff(pts_2d, axis=0), axis=1)
        s_raw = np.concatenate([[0.0], np.cumsum(seg_d)])
        total = s_raw[-1]
        if total < ds:
            return pts_2d.copy(), s_raw
        n = int(np.floor(total / ds)) + 1
        s_t = np.linspace(0.0, total, n)
        x = np.interp(s_t, s_raw, pts_2d[:, 0])
        y = np.interp(s_t, s_raw, pts_2d[:, 1])
        return np.column_stack([x, y]), s_t

    @staticmethod
    def _segment_resample(segments, ds):
        """(B) 세그먼트별 ds 간격으로 직접 샘플링."""
        pts_list, s_list = [], []
        s_offset = 0.0
        for seg in segments:
            if seg['type'] == 'straight':
                p0, p1 = seg['p0'], seg['p1']
                L = np.linalg.norm(p1 - p0)
                if L < 1e-9:
                    continue
                n = max(int(np.ceil(L / ds)), 1)
                t = np.linspace(0.0, 1.0, n + 1)
                pts = p0[None, :] + (p1 - p0)[None, :] * t[:, None]
            else:
                pts = seg['pts']
                d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                L = np.sum(d)
                if L < 1e-9:
                    continue
                n = max(int(np.ceil(L / ds)), 1)
                s_loc = np.concatenate([[0.0], np.cumsum(d)])
                s_t = np.linspace(0.0, L, n + 1)
                pts = np.column_stack([np.interp(s_t, s_loc, pts[:, 0]),
                                       np.interp(s_t, s_loc, pts[:, 1])])
            if pts_list and np.linalg.norm(pts_list[-1][-1] - pts[0]) < 1e-9:
                pts = pts[1:]
                if len(pts) == 0:
                    continue
            if pts_list:
                last = pts_list[-1][-1]
                d0 = np.linalg.norm(pts[0] - last)
                d_all = np.concatenate([[d0],
                                        np.linalg.norm(np.diff(pts, axis=0), axis=1)])
                s_loc = s_offset + np.cumsum(d_all)
            else:
                d_all = np.concatenate([[0.0],
                                        np.linalg.norm(np.diff(pts, axis=0), axis=1)])
                s_loc = np.cumsum(d_all)
            pts_list.append(pts)
            s_list.append(s_loc)
            s_offset = s_loc[-1]
        return np.vstack(pts_list), np.concatenate(s_list)

    @staticmethod
    def _compute_wp_arclengths(wps_3d, sampled_2d, s_arr):
        ws = np.array([s_arr[int(np.argmin(np.linalg.norm(sampled_2d - w[:2], axis=1)))]
                       for w in wps_3d])
        for i in range(1, len(ws)):
            if ws[i] <= ws[i - 1]:
                ws[i] = ws[i - 1] + 1e-6
        return ws

    @staticmethod
    def _compute_heading(pts_2d, s_arr):
        """방위각 chi (True North 기준 시계방향, rad)."""
        dN = np.gradient(pts_2d[:, 0], s_arr)
        dE = np.gradient(pts_2d[:, 1], s_arr)
        return np.arctan2(dE, dN)

    @staticmethod
    def _compute_curvature(pts_2d, s_arr):
        """부호 있는 2D 곡률 (호 길이 기반).
        NED 컨벤션: 우선회(시계방향) +, 좌선회(반시계) -.
        """
        if len(pts_2d) < 3:
            return np.zeros(len(pts_2d))
        dN = np.gradient(pts_2d[:, 0], s_arr)
        dE = np.gradient(pts_2d[:, 1], s_arr)
        d2N = np.gradient(dN, s_arr)
        d2E = np.gradient(dE, s_arr)
        speed = np.sqrt(np.maximum(dN**2 + dE**2, 1e-24))
        return (dN * d2E - dE * d2N) / speed**3
