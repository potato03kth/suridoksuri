"""
B-Spline 코너 라운딩 기반 경로 생성기
=====================================

이전의 "전체 anchor → 단일 spline fit" 방식은 lead/detour anchor가
서로 결합되어 spline이 WP에서 크게 벗어나는 문제(특히 180° 반전)에서
실패했다. 본 구현은 다음 구조로 대체한다:

  [직선] — [코너 spline (WP 통과 보장)] — [직선] — [코너 spline] — ...

- 직선 구간 : 두 WP를 잇는 직선의 일부 (양 끝의 lead 영역 제외)
- 코너 구간 : WP 진입 lead 지점 -> WP -> 진출 lead 지점을
              로컬 5점 B-spline으로 보간 (WP 정확 통과)
- 코너는 서로 격리 → 한 코너 수정이 다른 코너에 영향 없음
- 가속도 위반 시 "그 코너의 lead만" 확대
- 180° 반전은 코너 lead가 매우 커지면 자연스럽게 큰 원호로 변형됨
"""
from __future__ import annotations

import time
import numpy as np
from scipy.interpolate import splprep, splev

from .base_planner import BasePlanner, Path, PathPoint


# =============================================================================
# 유틸리티
# =============================================================================
def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


def _wrap_pi(a): return (a + np.pi) % (2.0 * np.pi) - np.pi


# =============================================================================
# B-Spline Planner
# =============================================================================
class BSplinePlanner(BasePlanner):
    """
    Parameters
    ----------
    ds : float                 샘플 간격 (m)
    straight_lead : float      WP 진입/진출 직선 유지 거리 기본값 (m)
    corner_lead_max_ratio : float
        코너 lead가 인접 segment 길이의 이 비율을 넘지 않게 캡 (기본 0.45)
    spline_degree : int        코너 spline 차수
    accel_tol : float          a_max 대비 허용 마진
    max_refine_iter : int      코너별 lead 확장 최대 반복
    lead_growth : float        위반 코너 lead 증가 배수
    """

    def __init__(self,
                 ds: float = 1.0,
                 straight_lead: float = 30.0,
                 corner_lead_max_ratio: float = 0.45,
                 spline_degree: int = 3,
                 accel_tol: float = 1.02,
                 max_refine_iter: int = 15,
                 lead_growth: float = 1.35):
        self.ds = float(ds)
        self.straight_lead = float(straight_lead)
        self.corner_lead_max_ratio = float(corner_lead_max_ratio)
        self.spline_degree = int(spline_degree)
        self.accel_tol = float(accel_tol)
        self.max_refine_iter = int(max_refine_iter)
        self.lead_growth = float(lead_growth)

    # ------------------------------------------------------------------
    def plan(self, waypoints_ned, aircraft_params, initial_state=None):
        t0 = time.perf_counter()
        wps = np.asarray(waypoints_ned, dtype=float)
        if wps.ndim != 2 or wps.shape[1] != 3:
            raise ValueError("waypoints_ned는 (N, 3)이어야 합니다.")
        N = len(wps)
        if N < 2:
            raise ValueError("최소 2개의 WP가 필요합니다.")

        v_cruise = float(aircraft_params.get("v_cruise", 25.0))
        a_max_g = float(aircraft_params.get("a_max_g", 2.0))
        a_max = a_max_g * 9.80665
        v_min = float(aircraft_params.get("v_min", 0.5 * v_cruise))

        # 코너 lead (각 중간 WP마다 in/out 별도)
        # WP0과 WP_{N-1}는 코너 없음(끝점)
        lead_in = np.full(N, self.straight_lead, dtype=float)
        lead_out = np.full(N, self.straight_lead, dtype=float)

        path: Path | None = None
        meta_iter = 0
        meta_violations = []

        for it in range(self.max_refine_iter + 1):
            meta_iter = it + 1
            # 인접 segment 길이로 lead 캡
            lead_in_use, lead_out_use = self._cap_leads(wps, lead_in, lead_out)

            path = self._build_path(wps, lead_in_use, lead_out_use,
                                    v_cruise, a_max, v_min)
            violations = self._validate_corners(path, a_max)
            meta_violations = violations
            if not violations or it == self.max_refine_iter:
                break

            # 위반 코너의 in/out lead 모두 증가
            for v in violations:
                wp_i = v["wp_index"]
                lead_in[wp_i] *= self.lead_growth
                lead_out[wp_i] *= self.lead_growth

        path.waypoints_ned = wps.copy()
        path.planning_time = time.perf_counter() - t0
        try:
            path.meta = {
                "iterations": meta_iter,
                "lead_in":  lead_in.tolist(),
                "lead_out": lead_out.tolist(),
                "violations_remaining": meta_violations,
            }
        except Exception:
            pass
        return path

    # ------------------------------------------------------------------
    # lead 캡: 두 코너의 lead 합이 segment 길이를 넘지 않게
    # ------------------------------------------------------------------
    def _cap_leads(self, wps, lead_in, lead_out):
        N = len(wps)
        li = lead_in.copy()
        lo = lead_out.copy()
        for i in range(N - 1):
            seg_len = float(np.linalg.norm(wps[i + 1] - wps[i]))
            cap = self.corner_lead_max_ratio * seg_len
            # 양쪽 코너가 segment를 나눠 가짐
            need = lo[i] + li[i + 1]
            allowed = 2.0 * cap  # 각각 cap까지 허용 (합 2*cap=0.9*seg_len)
            if need > allowed:
                # 비례 축소
                scale = allowed / need
                lo[i] *= scale
                li[i + 1] *= scale
            # 개별 cap도 적용
            lo[i] = min(lo[i], cap)
            li[i + 1] = min(li[i + 1], cap)
        return li, lo

    # ------------------------------------------------------------------
    # 직선 + 코너 합성으로 path 구성
    # ------------------------------------------------------------------
    def _build_path(self, wps, lead_in, lead_out,
                    v_cruise, a_max, v_min) -> Path:
        N = len(wps)

        # 각 WP의 진입/진출 단위벡터 (수평면 기준 — 고도는 직선 보간이 자연스럽도록 3D 그대로 사용)
        # 다만 lead anchor는 3D 직선 위에 둔다 (즉 in/out vector는 3D)
        dirs_in = [None] * N   # WP로 들어오는 방향 (이전 WP -> 현재 WP)
        dirs_out = [None] * N   # WP에서 나가는 방향 (현재 WP -> 다음 WP)
        for i in range(N):
            if i > 0:
                dirs_in[i] = _unit(wps[i] - wps[i - 1])
            if i < N - 1:
                dirs_out[i] = _unit(wps[i + 1] - wps[i])

        # 각 WP의 진입 lead 지점 P_in[i], 진출 lead 지점 P_out[i]
        # 끝점은 자기 자신
        P_in = [None] * N
        P_out = [None] * N
        for i in range(N):
            if i == 0:
                P_in[i] = wps[i].copy()
            else:
                P_in[i] = wps[i] - dirs_in[i] * lead_in[i]
            if i == N - 1:
                P_out[i] = wps[i].copy()
            else:
                P_out[i] = wps[i] + dirs_out[i] * lead_out[i]

        # 경로 조각 리스트: (points: (K,3), wp_index_at_each: list[int|None])
        pieces: list[tuple[np.ndarray, list[int | None]]] = []

        for i in range(N):
            # ---- 코너 i (중간 WP만) ----
            if 0 < i < N - 1:
                pts_corner, wp_marks = self._build_corner(
                    P_out[i - 1] if i - 1 == 0 else None,  # 미사용
                    P_in[i], wps[i], P_out[i],
                    dirs_in[i], dirs_out[i],
                    a_max, v_cruise
                )
                # 첫 점은 P_in[i], 마지막 점은 P_out[i]
                # WP 마킹 — wps[i]에 가장 가까운 점에 wp_index = i
                k_wp = int(np.argmin(np.linalg.norm(
                    pts_corner - wps[i], axis=1)))
                marks = [None] * len(pts_corner)
                marks[k_wp] = i
                pieces.append((pts_corner, marks))

            # ---- 직선 segment i -> i+1 ----
            #   from = (i==0 ? wps[0] : P_out[i])
            #   to   = (i==N-2 ? wps[N-1] : P_in[i+1])
            if i < N - 1:
                A = wps[i] if i == 0 else P_out[i]
                B = wps[i + 1] if i == N - 2 else P_in[i + 1]
                seg_pts = self._sample_line(A, B)
                marks = [None] * len(seg_pts)
                # 끝점 WP 마킹 (직선이 WP 자체에서 끝나는 경우)
                if i == 0:
                    marks[0] = 0
                if i == N - 2:
                    marks[-1] = N - 1
                pieces.append((seg_pts, marks))

        # 조각들 이어붙이기 (중복점 제거)
        pts_all: list[np.ndarray] = []
        marks_all: list[int | None] = []
        for pts, marks in pieces:
            for k in range(len(pts)):
                if pts_all and np.linalg.norm(pts[k] - pts_all[-1]) < 1e-6:
                    # 중복 — 마킹만 보존
                    if marks[k] is not None:
                        marks_all[-1] = marks[k]
                    continue
                pts_all.append(pts[k])
                marks_all.append(marks[k])

        pts_all = np.array(pts_all)

        # 호 길이 누적
        seg = np.diff(pts_all, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        s_cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_len = float(s_cum[-1])

        # 균일 ds로 재샘플링 — WP 마킹 보존을 위해, 마킹 있는 점은 강제 포함
        s_uniform = self._uniform_with_marks(s_cum, marks_all, self.ds)

        # 보간
        x = np.interp(s_uniform, s_cum, pts_all[:, 0])
        y = np.interp(s_uniform, s_cum, pts_all[:, 1])
        z = np.interp(s_uniform, s_cum, pts_all[:, 2])

        # 곡률 / chi / gamma는 차분으로 (이미 충분히 조밀한 점)
        dx = np.gradient(x, s_uniform)
        dy = np.gradient(y, s_uniform)
        dz = np.gradient(z, s_uniform)
        ddx = np.gradient(dx, s_uniform)
        ddy = np.gradient(dy, s_uniform)
        ddz = np.gradient(dz, s_uniform)
        rp_norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-12

        cross = np.stack([dy*ddz - dz*ddy,
                          dz*ddx - dx*ddz,
                          dx*ddy - dy*ddx], axis=1)
        kappa = np.linalg.norm(cross, axis=1) / (rp_norm**3)
        sign = -np.sign(dx*ddy - dy*ddx)
        sign[sign == 0] = 1.0
        kappa_signed = kappa * sign

        chi = np.unwrap(np.arctan2(dy, dx))
        chi = np.array([_wrap_pi(c) for c in chi])
        gamma = np.arctan2(dz, np.sqrt(dx**2 + dy**2) + 1e-12)

        # 속도 프로파일
        with np.errstate(divide="ignore"):
            v_max_curve = np.sqrt(
                a_max / np.maximum(np.abs(kappa_signed), 1e-9))
        v_ref = np.clip(np.minimum(v_cruise, v_max_curve), v_min, v_cruise)

        # WP 마킹을 s_uniform 위로 매핑
        wp_to_s: dict[int, float] = {}
        for k, m in enumerate(marks_all):
            if m is not None:
                wp_to_s[m] = s_cum[k]
        wp_marks_uni: dict[int, int] = {}
        for wp_i, s_target in wp_to_s.items():
            k_uni = int(np.argmin(np.abs(s_uniform - s_target)))
            wp_marks_uni[k_uni] = wp_i

        points: list[PathPoint] = []
        for k_idx in range(len(s_uniform)):
            points.append(PathPoint(
                pos=np.array([x[k_idx], y[k_idx], z[k_idx]]),
                v_ref=float(v_ref[k_idx]),
                chi_ref=float(chi[k_idx]),
                gamma_ref=float(gamma[k_idx]),
                curvature=float(kappa_signed[k_idx]),
                s=float(s_uniform[k_idx]),
                wp_index=wp_marks_uni.get(k_idx, None),
            ))

        return Path(points=points, waypoints_ned=np.asarray(wps).copy(),
                    total_length=total_len)

    # ------------------------------------------------------------------
    # 코너 빌드: P_in -> WP -> P_out 을 부드럽게 통과하는 5점 spline
    # ------------------------------------------------------------------
    def _build_corner(self, _unused_prev,
                      P_in: np.ndarray, WP: np.ndarray, P_out: np.ndarray,
                      d_in: np.ndarray, d_out: np.ndarray,
                      a_max: float, v_cruise: float) -> tuple[np.ndarray, list]:
        """
        5개의 anchor로 짧은 spline 코너를 만든다:
          A = P_in - 0.5 * lead_in_dir * (작은 buffer)  → in 방향 직선성 강화
          B = P_in
          C = WP                                        → 정확 통과 보장
          D = P_out
          E = P_out + 0.5 * lead_out_dir * buffer        → out 방향 직선성 강화

        실제로는 A, E를 짧게 두면 spline이 P_in/P_out에서 d_in/d_out 방향과
        잘 정렬된다 (즉, 직선 segment와 이어붙였을 때 부드러움).
        """
        # buffer 길이: lead 길이의 절반, 단 너무 길지 않게
        lead_len_in = float(np.linalg.norm(WP - P_in))
        lead_len_out = float(np.linalg.norm(P_out - WP))
        buf_in = 0.5 * lead_len_in
        buf_out = 0.5 * lead_len_out

        A = P_in - d_in * buf_in
        E = P_out + d_out * buf_out

        anchors = np.stack([A, P_in, WP, P_out, E], axis=0)

        # 차수
        k = min(self.spline_degree, len(anchors) - 1)
        try:
            tck, _ = splprep([anchors[:, 0], anchors[:, 1], anchors[:, 2]],
                             k=k, s=0.0)
        except Exception:
            # 보간 실패 시 단순 직선 라운딩
            return self._sample_line(P_in, P_out), [None]

        # 충분히 조밀하게 샘플 후, P_in~P_out 사이 호 길이 기반 ds 샘플
        u_d = np.linspace(0.0, 1.0, 400)
        x, y, z = splev(u_d, tck)
        pts = np.stack([x, y, z], axis=1)

        # P_in과 P_out에 해당하는 u 위치 찾기 (anchors[1], anchors[3])
        # splprep은 anchors의 u 값을 반환하지 않지만, 우리는 균등 매개변수가 아니므로
        # 거리상 가장 가까운 dense 인덱스 사용
        k_in = int(np.argmin(np.linalg.norm(pts - P_in,  axis=1)))
        k_out = int(np.argmin(np.linalg.norm(pts - P_out, axis=1)))
        if k_out <= k_in:
            k_out = len(pts) - 1
            k_in = 0
        sub = pts[k_in:k_out + 1]

        # 호길이 균일 재샘플
        seg = np.diff(sub, axis=0)
        sl = np.linalg.norm(seg, axis=1)
        s_cum = np.concatenate([[0.0], np.cumsum(sl)])
        if s_cum[-1] < 1e-9:
            return self._sample_line(P_in, P_out), [None]
        n = max(2, int(np.ceil(s_cum[-1] / self.ds)) + 1)
        s_u = np.linspace(0.0, s_cum[-1], n)
        out = np.stack([np.interp(s_u, s_cum, sub[:, 0]),
                        np.interp(s_u, s_cum, sub[:, 1]),
                        np.interp(s_u, s_cum, sub[:, 2])], axis=1)
        return out, [None] * len(out)

    # ------------------------------------------------------------------
    def _sample_line(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        L = float(np.linalg.norm(B - A))
        if L < 1e-9:
            return A.reshape(1, 3).copy()
        n = max(2, int(np.ceil(L / self.ds)) + 1)
        ts = np.linspace(0.0, 1.0, n).reshape(-1, 1)
        return A.reshape(1, 3) * (1 - ts) + B.reshape(1, 3) * ts

    @staticmethod
    def _uniform_with_marks(s_cum, marks, ds) -> np.ndarray:
        total = s_cum[-1]
        if total < 1e-9:
            return np.array([0.0])
        n = max(2, int(np.ceil(total / ds)) + 1)
        s_u = np.linspace(0.0, total, n)
        # 마킹된 위치 강제 포함
        forced = [s_cum[k] for k, m in enumerate(marks) if m is not None]
        s_all = np.unique(np.concatenate([s_u, np.array(forced)]))
        return s_all

    # ------------------------------------------------------------------
    # 코너별 validation: 각 중간 WP의 코너 영역에서 max a_lat 검사
    # ------------------------------------------------------------------
    def _validate_corners(self, path: Path, a_max: float) -> list[dict]:
        if len(path.points) < 3:
            return []
        kappa = np.array([p.curvature for p in path.points])
        v = np.array([p.v_ref for p in path.points])
        a_lat = (v ** 2) * np.abs(kappa)

        # WP path index 매핑
        wp_idx_map: dict[int, int] = {}
        for k, p in enumerate(path.points):
            if p.wp_index is not None:
                wp_idx_map[p.wp_index] = k
        wp_indices_sorted = sorted(wp_idx_map.keys())

        violations = []
        # 중간 WP만 검사 (코너가 있는 WP)
        for j_pos, wp_i in enumerate(wp_indices_sorted):
            if j_pos == 0 or j_pos == len(wp_indices_sorted) - 1:
                continue
            k_wp = wp_idx_map[wp_i]
            k_prev = wp_idx_map[wp_indices_sorted[j_pos - 1]]
            k_next = wp_idx_map[wp_indices_sorted[j_pos + 1]]
            # 코너 영역: 이전 WP와의 중점 ~ 다음 WP와의 중점
            k_lo = (k_prev + k_wp) // 2
            k_hi = (k_wp + k_next) // 2
            seg_a = a_lat[k_lo:k_hi + 1]
            if seg_a.size == 0:
                continue
            a_peak = float(np.max(seg_a))
            if a_peak > a_max * self.accel_tol:
                k_peak_local = int(np.argmax(seg_a))
                violations.append({
                    "wp_index": wp_i,
                    "k_peak": k_lo + k_peak_local,
                    "a_lat_peak": a_peak,
                    "a_max": a_max,
                })
        return violations
