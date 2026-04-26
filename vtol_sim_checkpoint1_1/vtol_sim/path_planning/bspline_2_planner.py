"""
B-Spline 기반 경로 생성기 (per-WP 적응형 refinement)
======================================================

BasePlanner를 상속하여 WP 시퀀스로부터 부드러운 3D 비행 경로를 생성한다.

설계 원칙
---------
1) WP 통과 보장: B-spline 보간(s=0)으로 모든 WP 정확 통과
2) WP 직전 직선 구간: 각 WP 전후 straight_lead*lead_scale[i] anchor 삽입
3) 가속도 제약: 곡선 구간(segment) 단위로 최대 측방가속도 검사 →
                위반 segment의 책임 WP의 lead/detour scale만 선택적 증가
4) 180도 반전: 진입/진출 각도가 임계 이상이면 측방 offset 우회점 삽입,
                위반 시 detour_scale로 offset 확대
5) 우선순위: WP 근접 > 가속도 준수 >> 경로 길이
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


def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _signed_angle_2d(u: np.ndarray, v: np.ndarray) -> float:
    return _wrap_pi(np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0]))


# =============================================================================
# B-Spline Planner
# =============================================================================
class BSplinePlanner(BasePlanner):
    """
    Parameters
    ----------
    ds : float
        샘플 간격 (m).
    straight_lead : float
        WP 전후 기본 직선 유지 거리 (m).
    spline_degree : int
        B-spline 차수 (기본 3).
    smoothing : float
        splprep s 파라미터 (0=정확 보간).
    reversal_angle_deg : float
        이 각도 이상이면 detour 보조점 삽입.
    max_refine_iter : int
        per-WP refinement 최대 반복.
    accel_tol : float
        a_max 대비 허용 마진 (1.0 = 정확히 a_max).
    lead_growth : float
        위반 시 lead_scale 증가 배수.
    detour_growth : float
        위반 시 detour_scale 증가 배수.
    lead_scale_max : float
        lead_scale 상한 (안전장치).
    """

    def __init__(self,
                 ds: float = 1.0,
                 straight_lead: float = 30.0,
                 spline_degree: int = 3,
                 smoothing: float = 0.0,
                 reversal_angle_deg: float = 150.0,
                 max_refine_iter: int = 12,
                 accel_tol: float = 1.02,
                 lead_growth: float = 1.4,
                 detour_growth: float = 1.5,
                 lead_scale_max: float = 20.0):
        self.ds = float(ds)
        self.straight_lead = float(straight_lead)
        self.spline_degree = int(spline_degree)
        self.smoothing = float(smoothing)
        self.reversal_angle_deg = float(reversal_angle_deg)
        self.max_refine_iter = int(max_refine_iter)
        self.accel_tol = float(accel_tol)
        self.lead_growth = float(lead_growth)
        self.detour_growth = float(detour_growth)
        self.lead_scale_max = float(lead_scale_max)

    # ---------------------------------------------------------------------
    # 메인
    # ---------------------------------------------------------------------
    def plan(self, waypoints_ned: np.ndarray, aircraft_params: dict,
             initial_state: dict | None = None) -> Path:
        t0 = time.perf_counter()

        wps = np.asarray(waypoints_ned, dtype=float)
        if wps.ndim != 2 or wps.shape[1] != 3:
            raise ValueError("waypoints_ned는 (N, 3)이어야 합니다.")
        if len(wps) < 2:
            raise ValueError("최소 2개 이상의 WP가 필요합니다.")

        v_cruise = float(aircraft_params.get("v_cruise", 25.0))
        a_max_g = float(aircraft_params.get("a_max_g", 2.0))
        a_max = a_max_g * 9.81
        v_min = float(aircraft_params.get("v_min", 0.5 * v_cruise))
        R_min = (v_cruise ** 2) / max(a_max, 1e-6)

        N = len(wps)
        # per-WP 스케일 (1.0에서 시작, 위반 시 해당 WP만 증가)
        lead_scale = np.ones(N)
        detour_scale = np.ones(N)

        path: Path | None = None
        last_violations: list[dict] = []

        for it in range(self.max_refine_iter + 1):
            anchors, anchor_wp_idx, has_detour = self._build_anchors(
                wps, lead_scale, detour_scale, R_min
            )
            try:
                path = self._fit_and_sample(
                    anchors, anchor_wp_idx, wps, v_cruise, a_max, v_min
                )
            except Exception:
                # 보간 실패 → 모든 lead 살짝 증가 후 재시도
                lead_scale = np.minimum(lead_scale * 1.2, self.lead_scale_max)
                continue

            # ---- 곡선 segment 단위 validation ----
            violations = self._find_violating_segments(path, a_max)
            last_violations = violations

            if not violations or it == self.max_refine_iter:
                break

            # ---- 위반 segment → 책임 WP 매핑 후 해당 WP의 scale만 증가 ----
            wp_path_indices = self._wp_path_indices(path)  # {wp_i: path_idx}
            for v in violations:
                wp_i = self._blame_wp(v, wp_path_indices, N)
                if wp_i is None:
                    continue
                # 해당 WP가 detour를 갖고 있으면 detour를 우선 키움
                if has_detour[wp_i]:
                    detour_scale[wp_i] = min(
                        detour_scale[wp_i] * self.detour_growth,
                        self.lead_scale_max
                    )
                # lead도 함께 키움 (직선 구간 충분히 확보)
                lead_scale[wp_i] = min(
                    lead_scale[wp_i] * self.lead_growth,
                    self.lead_scale_max
                )

        assert path is not None
        path.waypoints_ned = wps.copy()
        path.planning_time = time.perf_counter() - t0
        # 디버깅 편의를 위해 메타로 보관 (Path 정의에 없으면 무시됨)
        try:
            path.meta = {
                "lead_scale": lead_scale.tolist(),
                "detour_scale": detour_scale.tolist(),
                "violations_remaining": last_violations,
                "iterations": it + 1,
            }
        except Exception:
            pass
        return path

    # ---------------------------------------------------------------------
    # anchor 구성 (per-WP 스케일 적용)
    # ---------------------------------------------------------------------
    def _build_anchors(self,
                       wps: np.ndarray,
                       lead_scale: np.ndarray,
                       detour_scale: np.ndarray,
                       R_min: float
                       ) -> tuple[np.ndarray, list[int | None], np.ndarray]:
        N = len(wps)
        anchors: list[np.ndarray] = []
        wp_idx: list[int | None] = []
        has_detour = np.zeros(N, dtype=bool)

        base_lead = self.straight_lead

        for i in range(N):
            wp = wps[i]
            in_dir = np.zeros(3)
            out_dir = np.zeros(3)

            if i > 0:
                v_in = wps[i] - wps[i - 1]
                in_dir = _unit(np.array([v_in[0], v_in[1], 0.0]))
            if i < N - 1:
                v_out = wps[i + 1] - wps[i]
                out_dir = _unit(np.array([v_out[0], v_out[1], 0.0]))

            # 진입 lead (이 WP의 lead_scale 사용)
            if i > 0 and np.linalg.norm(in_dir) > 0:
                seg_len = np.linalg.norm(wps[i] - wps[i - 1])
                use_lead = min(base_lead * lead_scale[i], 0.45 * seg_len)
                pre = wp - in_dir * use_lead
                pre[2] = wp[2]
                anchors.append(pre)
                wp_idx.append(None)

            # WP 본체
            anchors.append(wp.copy())
            wp_idx.append(i)

            # 180° 반전 detour
            if 0 < i < N - 1 and np.linalg.norm(in_dir) > 0 and np.linalg.norm(out_dir) > 0:
                turn = abs(_signed_angle_2d(in_dir[:2], out_dir[:2]))
                if np.degrees(turn) >= self.reversal_angle_deg:
                    sgn = np.sign(_signed_angle_2d(in_dir[:2], out_dir[:2]))
                    if sgn == 0:
                        sgn = 1.0
                    perp = np.array([-in_dir[1], in_dir[0], 0.0]) * sgn
                    offset = max(2.0 * R_min, base_lead) * detour_scale[i]
                    detour = wp + perp * offset
                    detour[2] = wp[2]
                    anchors.append(detour)
                    wp_idx.append(None)
                    has_detour[i] = True

            # 진출 lead
            if i < N - 1 and np.linalg.norm(out_dir) > 0:
                seg_len = np.linalg.norm(wps[i + 1] - wps[i])
                use_lead = min(base_lead * lead_scale[i], 0.45 * seg_len)
                post = wp + out_dir * use_lead
                post[2] = wp[2]
                anchors.append(post)
                wp_idx.append(None)

        # 중복 제거
        anchors_np = np.array(anchors)
        keep = [0]
        wp_idx_filtered = [wp_idx[0]]
        for k in range(1, len(anchors_np)):
            if np.linalg.norm(anchors_np[k] - anchors_np[keep[-1]]) > 1e-3:
                keep.append(k)
                wp_idx_filtered.append(wp_idx[k])
            else:
                if wp_idx[k] is not None:
                    wp_idx_filtered[-1] = wp_idx[k]
        return anchors_np[keep], wp_idx_filtered, has_detour

    # ---------------------------------------------------------------------
    # B-spline 적합 + 균일 호 길이 샘플링
    # ---------------------------------------------------------------------
    def _fit_and_sample(self,
                        anchors: np.ndarray,
                        anchor_wp_idx: list[int | None],
                        wps: np.ndarray,
                        v_cruise: float,
                        a_max: float,
                        v_min: float) -> Path:
        M = len(anchors)
        k = min(self.spline_degree, M - 1)
        if k < 1:
            raise ValueError("anchor가 너무 적습니다.")

        tck, u_anchors = splprep(
            [anchors[:, 0], anchors[:, 1], anchors[:, 2]],
            k=k, s=self.smoothing
        )
        dense_n = max(2000, int(20 * M))
        u_dense = np.linspace(0.0, 1.0, dense_n)
        x, y, z = splev(u_dense, tck)
        dx, dy, dz = splev(u_dense, tck, der=1)
        ddx, ddy, ddz = splev(u_dense, tck, der=2)

        pts_dense = np.stack([x, y, z], axis=1)
        seg_len = np.linalg.norm(np.diff(pts_dense, axis=0), axis=1)
        s_dense = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_len = float(s_dense[-1])

        rp = np.stack([dx, dy, dz], axis=1)
        rpp = np.stack([ddx, ddy, ddz], axis=1)
        rp_norm = np.linalg.norm(rp, axis=1) + 1e-12
        kappa_dense = np.linalg.norm(
            np.cross(rp, rpp), axis=1) / (rp_norm ** 3)

        cross2d_z = dx * ddy - dy * ddx
        sign = -np.sign(cross2d_z)
        sign[sign == 0] = 1.0
        kappa_signed_dense = kappa_dense * sign

        chi_dense = np.arctan2(dy, dx)
        horiz = np.sqrt(dx ** 2 + dy ** 2) + 1e-12
        gamma_dense = np.arctan2(dz, horiz)

        n_pts = max(2, int(np.ceil(total_len / self.ds)) + 1)
        s_uniform = np.linspace(0.0, total_len, n_pts)

        def interp(arr): return np.interp(s_uniform, s_dense, arr)

        x_u = interp(x)
        y_u = interp(y)
        z_u = interp(z)
        kappa_u = interp(kappa_signed_dense)
        chi_u = np.array([_wrap_pi(c) for c in interp(np.unwrap(chi_dense))])
        gamma_u = interp(gamma_dense)

        with np.errstate(divide="ignore"):
            v_max_curve = np.sqrt(a_max / np.maximum(np.abs(kappa_u), 1e-9))
        v_ref = np.clip(np.minimum(v_cruise, v_max_curve), v_min, v_cruise)

        # WP 매핑
        u_to_s = np.interp(u_anchors, u_dense, s_dense)
        wp_marks: dict[int, int] = {}
        for a_i, wp_i in enumerate(anchor_wp_idx):
            if wp_i is None:
                continue
            s_target = u_to_s[a_i]
            kp = int(np.argmin(np.abs(s_uniform - s_target)))
            wp_marks[kp] = a_i

        points: list[PathPoint] = []
        for k_idx in range(n_pts):
            wp_index = None
            if k_idx in wp_marks:
                wp_index = anchor_wp_idx[wp_marks[k_idx]]
            points.append(PathPoint(
                pos=np.array([x_u[k_idx], y_u[k_idx], z_u[k_idx]]),
                v_ref=float(v_ref[k_idx]),
                chi_ref=float(chi_u[k_idx]),
                gamma_ref=float(gamma_u[k_idx]),
                curvature=float(kappa_u[k_idx]),
                s=float(s_uniform[k_idx]),
                wp_index=wp_index,
            ))

        return Path(points=points, waypoints_ned=wps.copy(), total_length=total_len)

    # ---------------------------------------------------------------------
    # 곡선 segment 단위 validation
    # ---------------------------------------------------------------------
    def _find_violating_segments(self, path: Path, a_max: float
                                 ) -> list[dict]:
        """
        곡률이 의미있게 큰 연속 구간(segment)을 찾고,
        그 안의 최대 측방가속도가 a_max를 초과하면 violation으로 보고.

        v_ref가 곡률에 따라 이미 줄어들었더라도, v_min 클립 때문에
        실제 a_lat = v_ref^2 * |kappa|가 a_max를 초과할 수 있다.
        """
        if len(path.points) < 3:
            return []

        kappa = np.array([p.curvature for p in path.points])
        v_ref = np.array([p.v_ref for p in path.points])
        a_lat = (v_ref ** 2) * np.abs(kappa)

        # 곡선 구간 정의: |kappa|가 잡음 임계 이상인 연속 영역
        # (직선부 잡음 제거)
        kmax = float(np.max(np.abs(kappa))) if len(kappa) else 0.0
        if kmax <= 1e-9:
            return []
        kappa_thresh = max(1e-4, 0.05 * kmax)
        in_curve = np.abs(kappa) > kappa_thresh

        violations: list[dict] = []
        n = len(in_curve)
        i = 0
        while i < n:
            if not in_curve[i]:
                i += 1
                continue
            j = i
            while j < n and in_curve[j]:
                j += 1
            # segment [i, j)
            seg_a_max = float(np.max(a_lat[i:j]))
            if seg_a_max > a_max * self.accel_tol:
                # 이 segment의 가장 위반이 큰 path index
                k_peak = i + int(np.argmax(a_lat[i:j]))
                violations.append({
                    "i_start": i, "i_end": j,
                    "k_peak": k_peak,
                    "a_lat_peak": seg_a_max,
                    "kappa_peak": float(np.abs(kappa[k_peak])),
                })
            i = j
        return violations

    # ---------------------------------------------------------------------
    # 위반 segment의 책임 WP 찾기
    # ---------------------------------------------------------------------
    def _wp_path_indices(self, path: Path) -> dict[int, int]:
        """wp_index -> path_point_index 매핑."""
        out: dict[int, int] = {}
        for k, p in enumerate(path.points):
            if p.wp_index is not None:
                out[p.wp_index] = k
        return out

    @staticmethod
    def _blame_wp(violation: dict,
                  wp_path_indices: dict[int, int],
                  N: int) -> int | None:
        """
        위반 segment(곡선)의 peak 위치에서 가장 가까운 WP를 책임자로 지목.
        곡선은 보통 어떤 WP를 '돌아가는' 부분에서 형성되므로 그 WP가 책임.
        """
        if not wp_path_indices:
            return None
        k_peak = violation["k_peak"]
        # 경로 인덱스 거리상 가장 가까운 WP
        best_wp = None
        best_d = 10**18
        for wp_i, k_wp in wp_path_indices.items():
            d = abs(k_wp - k_peak)
            if d < best_d:
                best_d = d
                best_wp = wp_i
        return best_wp
