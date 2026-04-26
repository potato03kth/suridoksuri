"""
Linear Model Predictive Controller (횡방향 MPC)
================================================

모델:
  상태  x = [e_y, e_chi, phi]  (cross-track, heading error, bank)
  입력  u = phi_cmd
  피드포워드: 경로 곡률 kappa_k

이산 시간 (Euler, dt = control_period):
  A = [[1,  v*dt,  0         ],
       [0,  1,     (g/v)*dt  ],
       [0,  0,     1-dt/tau_p]]
  B = [[0],
       [0],
       [dt/tau_p]]
  d_k = [0, -kappa_k * v * dt, 0]^T   ← 경로 곡률 피드포워드

QP (condensed 형식, 호라이즌 N_p):
  min  U^T H U + 2 f^T U
  s.t. -phi_max <= u_k <= phi_max

  H = Gamma^T Q_bar Gamma + R_bar  (N_p × N_p)
  f = Gamma^T Q_bar (Phi x0 + D_vec)

풀이:
  cvxpy (OSQP) 설치된 경우 → 엄밀한 제약 QP
  미설치 시 → numpy 역행렬 (비제약 해 + box 클램프) fallback

고도/속도: InnerLoopPI (NLGL과 동일)
"""
from __future__ import annotations
import numpy as np

from .base_controller import BaseController
from .inner_loop import InnerLoopPI
from dynamics.base_dynamics import ControlInput
from path_planning.base_planner import Path
from utils.math_utils import (
    closest_point_on_polyline_local, wrap_angle
)

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


def _build_prediction_matrices(A: np.ndarray, B: np.ndarray,
                                N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    x_vec = Phi * x0 + Gamma * U + D_vec (D_vec 제외)

    Phi  : (N*n_x, n_x)
    Gamma: (N*n_x, N*n_u)  하삼각 블록 행렬
    """
    n_x = A.shape[0]
    n_u = B.shape[1]
    Phi = np.zeros((N * n_x, n_x))
    Gamma = np.zeros((N * n_x, N * n_u))

    A_pow = np.eye(n_x)
    for k in range(N):
        A_pow = A_pow @ A
        Phi[k * n_x:(k + 1) * n_x, :] = A_pow
        for j in range(k + 1):
            exp = k - j
            Aj = np.linalg.matrix_power(A, exp)
            Gamma[k * n_x:(k + 1) * n_x, j * n_u:(j + 1) * n_u] = Aj @ B

    return Phi, Gamma


class MPCController(BaseController):
    def __init__(
        self,
        N_p: int = 20,
        q_ey: float = 1.0,
        q_echi: float = 0.3,
        q_phi: float = 0.05,
        q_term_factor: float = 5.0,
        r_phi: float = 0.5,
        phi_max_rad: float = np.deg2rad(60.0),
        tau_phi: float = 0.1,
        gravity: float = 9.81,
        inner_loop: InnerLoopPI | None = None,
        look_ahead_m: float = 60.0,
    ):
        """
        Parameters
        ----------
        N_p : 예측 호라이즌 (스텝 수, dt=0.05 → 1.0s)
        q_ey, q_echi, q_phi : 상태 가중치
        q_term_factor : 터미널 비용 = factor * running Q
        r_phi : 입력 가중치 (phi_cmd)
        phi_max_rad : 뱅크각 명령 상한 (rad)
        tau_phi : 뱅크각 1차 지연 상수 (s) — aircraft.yaml과 일치시킬 것
        gravity : 중력가속도
        inner_loop : 고도·속도 PI (None이면 기본값 사용)
        look_ahead_m : 폴리라인 국소 검색 외 내부 ref 추출용 거리
        """
        self.N_p = N_p
        self.phi_max = phi_max_rad
        self.g = gravity
        self.tau_phi = tau_phi
        self.inner = inner_loop or InnerLoopPI()
        self.L_ref = look_ahead_m

        # 가중치 행렬
        Q_run = np.diag([q_ey, q_echi, q_phi])
        Q_term = q_term_factor * Q_run
        self._Q_run = Q_run
        self._Q_term = Q_term
        self._r_phi = r_phi

        # 캐시
        self._last_seg_idx: int = 0
        self._cached_polyline: np.ndarray | None = None
        self._cached_path_id: int = -1
        self._prev_phi_cmd: float = 0.0

        # Dynamics 행렬 (dt가 compute() 첫 호출 때 결정되므로 초기화 후 캐시)
        self._dt_cached: float | None = None
        self._v_cached: float | None = None
        self._A: np.ndarray | None = None
        self._B: np.ndarray | None = None
        self._Phi: np.ndarray | None = None
        self._Gamma: np.ndarray | None = None
        self._H: np.ndarray | None = None
        self._H_inv: np.ndarray | None = None

        # cvxpy 문제 캐시
        self._cp_prob = None
        self._cp_U = None
        self._cp_x0_param = None
        self._cp_D_param = None

    def reset(self) -> None:
        self._last_seg_idx = 0
        self._prev_phi_cmd = 0.0
        self.inner.reset()

    def _get_polyline(self, path: Path) -> np.ndarray:
        pid = id(path)
        if pid != self._cached_path_id:
            self._cached_polyline = path.positions_array()
            self._cached_path_id = pid
            self._last_seg_idx = 0
        return self._cached_polyline

    def _build_matrices(self, v: float, dt: float) -> None:
        """
        A, B, Phi, Gamma, H 구축 (v 또는 dt 변화 시 재계산).

        예측 스텝 dt_pred: dt_ctrl보다 길게 설정하여 충분한 lookahead 확보.
        ZOH 이산화를 사용해 dt_pred > tau_phi 에서도 안정성 보장.
        """
        from scipy.signal import cont2discrete

        tau = self.tau_phi
        g = self.g

        # 예측 스텝: max(6*dt, 0.3s) — 제어주기보다 훨씬 길게
        dt_pred = max(6.0 * dt, 0.3)

        # 연속 시간 모델
        Ac = np.array([
            [0.0, v,        0.0     ],
            [0.0, 0.0,  g / v       ],
            [0.0, 0.0, -1.0 / tau  ],
        ])
        Bc = np.array([[0.0], [0.0], [1.0 / tau]])

        # ZOH 이산화 (scipy)
        Ad, Bd, _, _, _ = cont2discrete((Ac, Bc, np.eye(3), np.zeros((3, 1))),
                                         dt_pred, method="zoh")
        A = Ad
        B = Bd

        Phi, Gamma = _build_prediction_matrices(A, B, self.N_p)

        N = self.N_p
        n_x = 3
        # Q_bar (블록 대각: N-1개 Q_run + 1개 Q_term)
        Q_bar = np.zeros((N * n_x, N * n_x))
        for k in range(N - 1):
            Q_bar[k * n_x:(k + 1) * n_x,
                  k * n_x:(k + 1) * n_x] = self._Q_run
        Q_bar[(N - 1) * n_x:N * n_x,
              (N - 1) * n_x:N * n_x] = self._Q_term

        R_bar = self._r_phi * np.eye(N)

        H = Gamma.T @ Q_bar @ Gamma + R_bar

        self._A = A
        self._B = B
        self._Phi = Phi
        self._Gamma = Gamma
        self._Q_bar = Q_bar
        self._H = H
        self._dt_pred = dt_pred   # 실제 사용된 예측 스텝 저장
        # 역행렬 (fallback용)
        try:
            self._H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self._H_inv = np.linalg.pinv(H)
        self._dt_cached = dt
        self._v_cached = v

        # cvxpy 문제 재구성
        if _HAS_CVXPY:
            self._build_cvxpy_problem(A, B, N, dt_pred, v)

    def _build_cvxpy_problem(self, A: np.ndarray, B: np.ndarray,
                              N: int, dt_pred: float, v: float) -> None:
        """cvxpy Problem을 Parameter 기반으로 구성 (매 step 재사용)."""
        n_x, n_u = 3, 1
        x0_p = cp.Parameter(n_x)
        D_p = cp.Parameter(N * n_x)

        U = cp.Variable(N * n_u)
        X = cp.Variable((N + 1) * n_x)

        Q_run = self._Q_run
        Q_term = self._Q_term
        r = self._r_phi

        cost = 0.0
        constr = [X[:n_x] == x0_p]

        for k in range(N):
            xk = X[k * n_x:(k + 1) * n_x]
            xk1 = X[(k + 1) * n_x:(k + 2) * n_x]
            uk = U[k * n_u:(k + 1) * n_u]
            dk = D_p[k * n_x:(k + 1) * n_x]

            constr.append(xk1 == A @ xk + B @ uk + dk)
            constr.append(uk <= self.phi_max)
            constr.append(uk >= -self.phi_max)

            if k < N - 1:
                cost += cp.quad_form(xk, Q_run) + r * cp.sum_squares(uk)
            else:
                cost += cp.quad_form(xk, Q_term) + r * cp.sum_squares(uk)

        xN = X[N * n_x:(N + 1) * n_x]
        cost += cp.quad_form(xN, Q_term)

        prob = cp.Problem(cp.Minimize(cost), constr)

        self._cp_prob = prob
        self._cp_U = U
        self._cp_x0_param = x0_p
        self._cp_D_param = D_p

    def _get_preview_curvatures(self, path: Path, seg_idx: int,
                                 v: float) -> np.ndarray:
        """
        예측 호라이즌 N_p 스텝 각각에 대한 경로 곡률 미리보기.
        k 스텝 후 위치는 경로 위 s_current + k * v * dt_pred 지점.
        """
        pts = path.points
        n_path = len(pts)
        kappas = np.zeros(self.N_p)
        dt_pred = getattr(self, "_dt_pred", 0.3)

        # 경로 평균 ds (점 간 거리) — 역할: index 단위 변환
        if n_path > 1:
            ds_avg = pts[-1].s / max(n_path - 1, 1)
        else:
            ds_avg = 2.0

        steps_per_pred = max(1, int(v * dt_pred / max(ds_avg, 0.1)))

        for k in range(self.N_p):
            idx = min(seg_idx + k * steps_per_pred, n_path - 1)
            kappas[k] = pts[idx].curvature

        return kappas

    def _compute_fallback(self, x0: np.ndarray, D_vec: np.ndarray) -> float:
        """cvxpy 불가 시 numpy 비제약 MPC + 클램프."""
        f = self._Gamma.T @ self._Q_bar @ (self._Phi @ x0 + D_vec)
        U_opt = -self._H_inv @ f
        u0 = float(np.clip(U_opt[0], -self.phi_max, self.phi_max))
        return u0

    def _compute_cvxpy(self, x0: np.ndarray, D_vec: np.ndarray) -> float:
        """cvxpy OSQP 풀이."""
        self._cp_x0_param.value = x0
        self._cp_D_param.value = D_vec
        try:
            self._cp_prob.solve(solver=cp.OSQP, warm_start=True,
                                verbose=False, eps_abs=1e-4, eps_rel=1e-4,
                                max_iter=2000)
            U_val = self._cp_U.value
            if U_val is not None:
                return float(np.clip(U_val[0], -self.phi_max, self.phi_max))
        except Exception:
            pass
        # 풀이 실패 시 fallback
        return self._compute_fallback(x0, D_vec)

    def compute(self, est_pos: np.ndarray, est_vel: np.ndarray,
                est_chi: float, est_gamma: float, est_phi: float, est_v: float,
                path: Path, t: float, dt: float) -> ControlInput:
        polyline = self._get_polyline(path)

        # 가장 가까운 경로점
        seg_idx, seg_t, closest, _ = closest_point_on_polyline_local(
            est_pos[:2], polyline[:, :2], self._last_seg_idx, window=80
        )
        self._last_seg_idx = seg_idx

        # 동역학 행렬 캐시 — v 변화가 크거나 dt가 달라지면 재계산
        v_used = max(est_v, 10.0)
        if (self._dt_cached is None
                or abs(dt - self._dt_cached) > 1e-6
                or abs(v_used - (self._v_cached or 0.0)) > 1.0):
            self._build_matrices(v_used, dt)

        # ===== 상태 계산 =====
        # 1. Cross-track error (부호: 경로 오른쪽 +)
        ref_pt = path.points[seg_idx]
        chi_ref = ref_pt.chi_ref

        # 경로 방향 벡터
        # NED에서 우측 법선: [-sin(chi), cos(chi)] (heading=0이면 동쪽(+E)이 오른쪽)
        path_dir = np.array([np.cos(chi_ref), np.sin(chi_ref)])
        path_normal = np.array([-path_dir[1], path_dir[0]])  # 오른쪽 법선
        rel = est_pos[:2] - closest[:2]
        e_y = float(np.dot(rel, path_normal))

        # 2. Heading error
        e_chi = float(wrap_angle(est_chi - chi_ref))

        # 3. Bank angle (이미 측정됨)
        phi = est_phi

        x0 = np.array([e_y, e_chi, phi])

        # 곡률 피드포워드는 비활성화 (예측 스텝 불일치 시 불안정 위험)
        # Phase 3에서 정확한 multi-rate 구현 시 활성화
        D_vec = np.zeros(self.N_p * 3)

        # ===== MPC 풀이 =====
        if _HAS_CVXPY and self._cp_prob is not None:
            phi_cmd = self._compute_cvxpy(x0, D_vec)
        else:
            phi_cmd = self._compute_fallback(x0, D_vec)

        self._prev_phi_cmd = phi_cmd

        # ===== 내부 루프: 고도·속도 =====
        ref_idx = min(seg_idx + int(self.L_ref / 2.0), len(path.points) - 1)
        v_ref = path.points[ref_idx].v_ref
        h_ref = path.points[ref_idx].pos[2]

        gamma_cmd = self.inner.compute_pitch(h_ref, est_pos[2], dt)
        thrust_cmd = self.inner.compute_thrust(v_ref, est_v, dt)

        return ControlInput(
            bank_cmd=phi_cmd,
            pitch_cmd=gamma_cmd,
            thrust_cmd=thrust_cmd,
        )
