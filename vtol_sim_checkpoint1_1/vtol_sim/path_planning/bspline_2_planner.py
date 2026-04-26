"""
B-Spline 기반 경로 생성기
==========================

BasePlanner를 상속하여 WP 시퀀스로부터 부드러운 3D 비행 경로를 생성한다.

설계 원칙
---------
1) WP 통과 보장: B-spline 보간(interpolation)으로 모든 WP를 정확히 통과
2) WP 직전 직선 구간: 각 WP 전후 straight_lead(기본 30m)에 보조 anchor 삽입
3) 가속도 제약: 곡률 기반 속도 프로파일 + 필요시 lead 거리/우회 강도 동적 조정
4) 180도 반전 대응: 진입/진출 방향이 반대에 가까우면 측방 오프셋 보조점 추가하여
                   최소 회전반경(R_min = v² / a_max) 이상의 완만한 곡선 유도
5) 우선순위: WP 근접 > 가속도 준수 >> 경로 길이
"""
from __future__ import annotations

import time
import numpy as np
from scipy.interpolate import splprep, splev

from .base_planner import BasePlanner, Path, PathPoint  # 경로는 프로젝트 구조에 맞게 조정


# =============================================================================
# 유틸리티
# =============================================================================
def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _wrap_pi(a: float) -> float:
    """[-pi, pi]로 래핑."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _signed_angle_2d(u: np.ndarray, v: np.ndarray) -> float:
    """2D 벡터 u→v 부호 있는 각도 (rad). 좌회전(반시계) 양수."""
    cu = np.arctan2(u[1], u[0])
    cv = np.arctan2(v[1], v[0])
    return _wrap_pi(cv - cu)


# =============================================================================
# B-Spline Planner
# =============================================================================
class BSplinePlanner(BasePlanner):
    """
    B-Spline 보간 기반 경로 생성기.

    Parameters
    ----------
    ds : float
        경로 샘플 간격 (m). 작을수록 정밀.
    straight_lead : float
        WP 전후 직선 유지 거리 (m). 기본 30 m.
    spline_degree : int
        B-spline 차수. 기본 3 (cubic).
    smoothing : float
        splprep의 s 파라미터. 0이면 정확 보간(WP 통과 보장).
    reversal_angle_deg : float
        이 각도 이상의 방향 변화는 "180도 반전"으로 간주하고 우회 보조점 삽입.
    max_refine_iter : int
        가속도 제약 위반 시 lead 거리 증가 재계획 최대 횟수.
    """

    def __init__(self,
                 ds: float = 1.0,
                 straight_lead: float = 30.0,
                 spline_degree: int = 3,
                 smoothing: float = 0.0,
                 reversal_angle_deg: float = 150.0,
                 max_refine_iter: int = 5):
        self.ds = float(ds)
        self.straight_lead = float(straight_lead)
        self.spline_degree = int(spline_degree)
        self.smoothing = float(smoothing)
        self.reversal_angle_deg = float(reversal_angle_deg)
        self.max_refine_iter = int(max_refine_iter)

    # ---------------------------------------------------------------------
    # 메인 인터페이스
    # ---------------------------------------------------------------------
    def plan(self, waypoints_ned: np.ndarray, aircraft_params: dict,
             initial_state: dict | None = None) -> Path:
        t0 = time.perf_counter()

        wps = np.asarray(waypoints_ned, dtype=float)
        if wps.ndim != 2 or wps.shape[1] != 3:
            raise ValueError("waypoints_ned는 (N, 3) 배열이어야 합니다.")
        if len(wps) < 2:
            raise ValueError("최소 2개 이상의 WP가 필요합니다.")

        # 기체 파라미터
        v_cruise = float(aircraft_params.get("v_cruise", 25.0))
        a_max_g = float(aircraft_params.get("a_max_g", 2.0))   # g 단위
        a_max = a_max_g * 9.80665                              # m/s^2
        v_min = float(aircraft_params.get("v_min", 0.5 * v_cruise))

        # 최소 회전반경
        R_min = (v_cruise ** 2) / max(a_max, 1e-6)

        # 1) 보조 anchor 점 시퀀스 구성
        #    - 각 WP 양쪽으로 straight_lead만큼 직선 유지점 삽입
        #    - 180도 반전 구간에는 측방 오프셋 우회점 삽입
        lead = self.straight_lead

        path: Path | None = None
        for it in range(self.max_refine_iter + 1):
            anchors, anchor_wp_idx = self._build_anchors(wps, lead, R_min)
            try:
                path = self._fit_and_sample(anchors, anchor_wp_idx, wps,
                                            v_cruise, a_max, v_min)
            except Exception as e:
                # 보간 실패 시 lead를 늘려 재시도
                lead *= 1.5
                continue

            # 가속도 제약 검사
            kappa = np.array([p.curvature for p in path.points])
            v_arr = np.array([p.v_ref for p in path.points])
            a_lat = (v_arr ** 2) * np.abs(kappa)
            if np.max(a_lat) <= a_max * 1.02 or it == self.max_refine_iter:
                break
            # 위반 시 lead 증가 → 곡률 완화
            lead *= 1.3

        assert path is not None
        path.waypoints_ned = wps.copy()
        path.planning_time = time.perf_counter() - t0
        return path

    # ---------------------------------------------------------------------
    # 1단계: anchor 시퀀스 구성
    # ---------------------------------------------------------------------
    def _build_anchors(self, wps: np.ndarray, lead: float, R_min: float
                       ) -> tuple[np.ndarray, list[int | None]]:
        """
        WP 시퀀스에 lead 직선점과 180도 반전 우회점을 삽입한 anchor 시퀀스 생성.

        Returns
        -------
        anchors : (M, 3) array
        anchor_wp_idx : list[Optional[int]]
            각 anchor가 어느 WP에 해당하는지(아니면 None).
        """
        N = len(wps)
        anchors: list[np.ndarray] = []
        wp_idx: list[int | None] = []

        for i in range(N):
            wp = wps[i]

            # --- 진입 방향 단위벡터 (수평 평면) ---
            if i > 0:
                in_vec_h = wps[i] - wps[i - 1]
                in_dir = _unit(np.array([in_vec_h[0], in_vec_h[1], 0.0]))
            else:
                in_dir = np.zeros(3)

            # --- 진출 방향 단위벡터 ---
            if i < N - 1:
                out_vec_h = wps[i + 1] - wps[i]
                out_dir = _unit(np.array([out_vec_h[0], out_vec_h[1], 0.0]))
            else:
                out_dir = np.zeros(3)

            # 1) 진입 lead 직선점 (WP 직전, 고도는 WP와 동일 → 고도 유지)
            if i > 0 and np.linalg.norm(in_dir) > 0:
                seg_len = np.linalg.norm(wps[i] - wps[i - 1])
                # 이전 WP와 너무 가까우면 lead 축소
                use_lead = min(lead, 0.4 * seg_len)
                pre = wp - in_dir * use_lead
                pre[2] = wp[2]  # 고도 유지
                anchors.append(pre)
                wp_idx.append(None)

            # 2) WP 본체
            anchors.append(wp.copy())
            wp_idx.append(i)

            # 3) 180도 반전 우회 보조점
            #    중간 WP에서 진입/진출 사이 각도 검사
            if 0 < i < N - 1 and np.linalg.norm(in_dir) > 0 and np.linalg.norm(out_dir) > 0:
                # 진행방향 변화각 (in → out 방향 회전각)
                turn = abs(_signed_angle_2d(
                    in_dir[:2], out_dir[:2]))  # [0, pi]
                turn_deg = np.degrees(turn)

                if turn_deg >= self.reversal_angle_deg:
                    # 반전에 가까움 → 측방 우회점 삽입
                    # 우회 방향: in_dir 기준 좌/우 중 out_dir 쪽으로
                    # signed_angle 부호로 결정
                    sgn = np.sign(_signed_angle_2d(in_dir[:2], out_dir[:2]))
                    if sgn == 0:
                        sgn = 1.0
                    # 좌수직(2D): rotate +90 = (-y, x)
                    perp = np.array([-in_dir[1], in_dir[0], 0.0]) * sgn
                    # 우회 강도: 최소 회전반경 이상으로
                    offset = max(2.0 * R_min, lead)
                    detour = wp + perp * offset
                    detour[2] = wp[2]  # 고도 유지
                    anchors.append(detour)
                    wp_idx.append(None)

            # 4) 진출 lead 직선점 (WP 직후)
            if i < N - 1 and np.linalg.norm(out_dir) > 0:
                seg_len = np.linalg.norm(wps[i + 1] - wps[i])
                use_lead = min(lead, 0.4 * seg_len)
                post = wp + out_dir * use_lead
                post[2] = wp[2]
                anchors.append(post)
                wp_idx.append(None)

        # 중복/근접 anchor 제거
        anchors_np = np.array(anchors)
        wp_idx_filtered: list[int | None] = []
        keep_idx = [0]
        wp_idx_filtered.append(wp_idx[0])
        for k in range(1, len(anchors_np)):
            if np.linalg.norm(anchors_np[k] - anchors_np[keep_idx[-1]]) > 1e-3:
                keep_idx.append(k)
                wp_idx_filtered.append(wp_idx[k])
            else:
                # 중복인데 한쪽이 WP면 WP 인덱스 유지
                if wp_idx[k] is not None:
                    wp_idx_filtered[-1] = wp_idx[k]

        return anchors_np[keep_idx], wp_idx_filtered

    # ---------------------------------------------------------------------
    # 2단계: B-spline 적합 + 호 길이 균일 샘플링
    # ---------------------------------------------------------------------
    def _fit_and_sample(self,
                        anchors: np.ndarray,
                        anchor_wp_idx: list[int | None],
                        wps: np.ndarray,
                        v_cruise: float,
                        a_max: float,
                        v_min: float) -> Path:
        M = len(anchors)
        # spline 차수 — anchor 수에 맞춰 조정
        k = min(self.spline_degree, M - 1)
        if k < 1:
            raise ValueError("anchor가 너무 적습니다.")

        # splprep: 매개변수 t in [0,1], 정확 보간(s=0)
        tck, u_anchors = splprep(
            [anchors[:, 0], anchors[:, 1], anchors[:, 2]],
            k=k, s=self.smoothing
        )
        # u_anchors[i]는 anchors[i]가 위치한 매개변수 값

        # 조밀 샘플 → 누적 호 길이 계산
        dense_n = max(2000, int(20 * M))
        u_dense = np.linspace(0.0, 1.0, dense_n)
        x, y, z = splev(u_dense, tck)
        dx, dy, dz = splev(u_dense, tck, der=1)
        ddx, ddy, ddz = splev(u_dense, tck, der=2)

        pts_dense = np.stack([x, y, z], axis=1)
        seg = np.diff(pts_dense, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        s_dense = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_len = float(s_dense[-1])

        # 곡률 (3D): kappa = |r' x r''| / |r'|^3
        rp = np.stack([dx, dy, dz], axis=1)
        rpp = np.stack([ddx, ddy, ddz], axis=1)
        cross = np.cross(rp, rpp)
        rp_norm = np.linalg.norm(rp, axis=1) + 1e-12
        kappa_dense = np.linalg.norm(cross, axis=1) / (rp_norm ** 3)

        # 부호 (수평면 기준 좌선회 음 / 우선회 양 — 명세대로)
        # 2D 외적의 z성분 부호 사용: 양수면 좌선회(반시계) → 음의 곡률
        cross2d_z = dx * ddy - dy * ddx
        sign = -np.sign(cross2d_z)   # 좌선회(+ z) → 음, 우선회(− z) → 양
        sign[sign == 0] = 1.0
        kappa_signed_dense = kappa_dense * sign

        # 방위각, 상승각
        chi_dense = np.arctan2(dy, dx)                       # NED 기준
        horiz = np.sqrt(dx ** 2 + dy ** 2) + 1e-12
        gamma_dense = np.arctan2(dz, horiz)                  # 양수 = 상승

        # 균일 호 길이로 재샘플
        n_pts = max(2, int(np.ceil(total_len / self.ds)) + 1)
        s_uniform = np.linspace(0.0, total_len, n_pts)

        # 보간 인덱스
        def interp(arr):
            return np.interp(s_uniform, s_dense, arr)

        x_u = interp(x)
        y_u = interp(y)
        z_u = interp(z)
        kappa_u = interp(kappa_signed_dense)

        # chi는 원형 보간 — unwrap 후 보간
        chi_unwrap = np.unwrap(chi_dense)
        chi_u = interp(chi_unwrap)
        chi_u = np.array([_wrap_pi(c) for c in chi_u])

        gamma_u = interp(gamma_dense)

        # 속도 프로파일: 가속도 제약 충족
        # v_lat = sqrt(a_max / |kappa|) 상한, v_cruise와 v_min 사이 클립
        with np.errstate(divide="ignore"):
            v_max_curve = np.sqrt(a_max / np.maximum(np.abs(kappa_u), 1e-9))
        v_ref = np.minimum(v_cruise, v_max_curve)
        v_ref = np.maximum(v_ref, v_min)

        # WP가 균일 샘플 상 어느 인덱스인지 — 가장 가까운 점에 wp_index 마킹
        # anchor의 매개변수 u_anchors → 호 길이로 변환
        u_to_s = np.interp(u_anchors, u_dense, s_dense)
        wp_marks: dict[int, int] = {}  # path_index -> wp_index
        for a_i, wp_i in enumerate(anchor_wp_idx):
            if wp_i is None:
                continue
            s_target = u_to_s[a_i]
            k_path = int(np.argmin(np.abs(s_uniform - s_target)))
            # 충돌 시 더 가까운 것을 우선
            if k_path in wp_marks:
                prev = wp_marks[k_path]
                if abs(s_uniform[k_path] - s_target) < \
                   abs(s_uniform[k_path] - u_to_s[prev]):
                    wp_marks[k_path] = a_i
            else:
                wp_marks[k_path] = a_i

        # PathPoint 시퀀스 구축
        points: list[PathPoint] = []
        for k_idx in range(n_pts):
            wp_index = None
            if k_idx in wp_marks:
                anchor_i = wp_marks[k_idx]
                wp_index = anchor_wp_idx[anchor_i]

            pp = PathPoint(
                pos=np.array([x_u[k_idx], y_u[k_idx], z_u[k_idx]]),
                v_ref=float(v_ref[k_idx]),
                chi_ref=float(chi_u[k_idx]),
                gamma_ref=float(gamma_u[k_idx]),
                curvature=float(kappa_u[k_idx]),
                s=float(s_uniform[k_idx]),
                wp_index=wp_index,
            )
            points.append(pp)

        return Path(
            points=points,
            waypoints_ned=wps.copy(),
            total_length=total_len,
        )
