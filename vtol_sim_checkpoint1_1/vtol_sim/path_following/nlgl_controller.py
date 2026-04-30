"""
Nonlinear Guidance Law (L1 Guidance) — Park et al. 2004
=========================================================

알고리즘:
1. 현재 위치에서 경로 위 가장 가까운 점 찾기
2. 그 점에서 호 길이 L1 만큼 전방의 점 = reference point
3. 현재 속도 벡터와 (ref - 현재 위치) 사이 각 η
4. 명령 횡가속도: a_cmd = 2 v² sin(η) / L1
5. 뱅크각 명령: φ_cmd = atan(a_cmd / g)

내부 루프 통합:
- 고도/속도는 InnerLoopPI 사용
- 속도 명령: 경로의 v_ref (look-ahead 점)
- 고도 명령: 경로의 h (look-ahead 점)
"""
from __future__ import annotations
import numpy as np

from .base_controller import BaseController
from .inner_loop import InnerLoopPI
from dynamics.base_dynamics import ControlInput
from path_planning.base_planner import Path
from utils.math_utils import (
    closest_point_on_polyline_local, look_ahead_point, wrap_angle
)


class NLGLController(BaseController):
    def __init__(
        self,
        L1: float = 60.0,                 # m, look-ahead 거리
        phi_max_rad: float = np.deg2rad(60.0),  # 명령 뱅크각 한계 (구조 한계와 별도)
        inner_loop: InnerLoopPI | None = None,
        gravity: float = 9.81,
    ):
        self.L1 = L1
        self.phi_max = phi_max_rad
        self.g = gravity
        self.inner = inner_loop or InnerLoopPI()

        # 캐시 — closest point 검색 가속화 (이전 idx 근처부터 시작)
        self._last_seg_idx: int = 0
        # 경로 폴리라인 캐시 (plan은 한 번만 계산되므로)
        self._cached_polyline: np.ndarray | None = None
        self._cached_path_id: int = -1

    def reset(self) -> None:
        self._last_seg_idx = 0
        self.inner.reset()

    def _get_polyline(self, path: Path) -> np.ndarray:
        """경로의 위치 배열 캐시."""
        path_id = id(path)
        if path_id != self._cached_path_id:
            self._cached_polyline = path.positions_array()
            self._cached_path_id = path_id
            self._last_seg_idx = 0
        return self._cached_polyline

    def compute(self, est_pos: np.ndarray, est_vel: np.ndarray,
                est_chi: float, est_gamma: float, est_phi: float, est_v: float,
                path: Path, t: float, dt: float) -> ControlInput:
        polyline = self._get_polyline(path)

        # 1. 가장 가까운 점 찾기 — 국소 검색 (이전 인덱스 근처)
        seg_idx, seg_t, closest, _ = closest_point_on_polyline_local(
            est_pos[:2], polyline[:, :2], self._last_seg_idx, window=80
        )
        self._last_seg_idx = seg_idx

        # 2. L1 전방 점 (수평면)
        ref_pt_2d, _, _ = look_ahead_point(
            polyline[:, :2], seg_idx, seg_t, self.L1
        )

        # 현재 위치(수평) → 참조점 벡터
        to_ref = np.array([ref_pt_2d[0] - est_pos[0],
                           ref_pt_2d[1] - est_pos[1]])
        dist_to_ref = float(np.linalg.norm(to_ref))
        if dist_to_ref < 1e-6:
            # 참조점이 거의 현재 위치와 같음 — 직진 유지
            phi_cmd = 0.0
        else:
            # 3. 속도 방향 (estimated heading 사용 — 무풍 가정)
            # est_chi 사용. (est_vel의 NED 성분 atan2도 가능하지만 noise 영향)
            v_dir = np.array([np.cos(est_chi), np.sin(est_chi)])
            # 4. η 각도 — to_ref와 v_dir 사이
            cross = v_dir[0] * to_ref[1] - v_dir[1] * to_ref[0]
            dot = v_dir[0] * to_ref[0] + v_dir[1] * to_ref[1]
            eta = float(np.arctan2(cross, dot))
            # 5. 명령 횡가속도
            v_used = max(est_v, 1.0)
            a_cmd = 2.0 * (v_used ** 2) * np.sin(eta) / self.L1
            # NED 좌표 컨벤션 부호 분석:
            #   v_dir = [cos(chi), sin(chi)] = [v_N, v_E]
            #   chi: 정북 0, 정동 π/2 (시계방향 양 — 항공 컨벤션)
            #   cross > 0 (v_N·r_E - v_E·r_N > 0) → ref가 v의 시계방향 = 우측
            #   우측에 ref가 있으면 우선회 필요 → phi > 0 (우선회 양)
            #   => phi_cmd = arctan(a_cmd/g) (부호 반전 없음)
            phi_cmd = float(np.arctan(a_cmd / self.g))
            phi_cmd = float(np.clip(phi_cmd, -self.phi_max, self.phi_max))

        # ---- 내부 루프: 고도, 속도 ----
        # look-ahead 점의 path 인덱스에서 v_ref, h_ref 추출
        # seg_idx가 폴리라인 세그먼트 i,i+1을 가리키므로
        # look-ahead 점의 정확한 인덱스 찾기 (단순화: seg 시작 인덱스 사용)
        # 더 정확히는 look_ahead_point가 반환한 (new_seg, new_t)에서 보간
        ref_seg, ref_t, _ = (None, None, None)
        # path.points의 v_ref / h를 그대로 사용 — seg_idx 기준
        ref_pt_idx = min(seg_idx + int(self.L1 / 2.0), len(path.points) - 1)
        # ↑ 단순화: 평균 점간 거리 ~ds 가정
        v_ref = path.points[ref_pt_idx].v_ref
        h_ref = path.points[ref_pt_idx].pos[2]

        gamma_cmd = self.inner.compute_pitch(h_ref, est_pos[2], dt)
        thrust_cmd = self.inner.compute_thrust(v_ref, est_v, dt)

        return ControlInput(
            bank_cmd=phi_cmd,
            pitch_cmd=gamma_cmd,
            thrust_cmd=thrust_cmd,
        )
