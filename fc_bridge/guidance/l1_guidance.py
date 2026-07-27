"""
L1 Guidance (Nonlinear Guidance Law).

Park, Deyst, How (2004) 기반. 경로 위의 lookahead point를 향하는 횡방향 가속도를
계산하고, NED 속도 세트포인트로 변환한다.

vtol_sim의 NLGLController 로직을 pymavlink 의존성 없이 재구현.
"""
from __future__ import annotations
import numpy as np


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ── 세그먼트 탐색 (2026-07-27 SITL-7 R2) ──────────────────────────────
#
# 아래 세 함수는 `L1Guidance._find_segment()` 를 구성하는 순수 함수다.
# rclpy·ROS 없이 단독 import·테스트된다(`fc_bridge/tests/test_l1_guidance.py`).

_SEG_WINDOW_M = 150.0   # 전방 탐색 창 (m). 아래 segment_search_range() 참조
_SEG_BACK_M = 5.0       # 허용 역행 폭 (m). 아래 segment_search_range() 참조


def cumulative_arclen(pts) -> np.ndarray:
    """경로점 배열의 누적 호길이 (m). shape (N,), [0] = 0.0.

    경로점 간격은 플래너마다 다르다(eta3 는 `ds=1.0` 로 약 1m, straight 는
    그렇지 않다). 탐색 창을 **인덱스 개수**로 잡으면 그 간격 가정에 묶이므로
    미터로 잡고 여기서 인덱스로 환산한다.
    """
    a = np.asarray(pts, dtype=float)
    if a.size == 0:
        return np.zeros(0)
    a = a.reshape(len(a), -1)[:, :2]
    if len(a) < 2:
        return np.zeros(len(a))
    d = np.linalg.norm(np.diff(a, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def segment_search_range(cum_len, prev_idx: int,
                         window_m: float = _SEG_WINDOW_M,
                         back_m: float = _SEG_BACK_M) -> tuple[int, int]:
    """최근접 세그먼트를 찾을 인덱스 구간 `[lo, hi]`(양끝 포함)을 정한다.

    이전 틱의 세그먼트(`prev_idx`)에서 **경로를 따라** 앞으로 `window_m`,
    뒤로 `back_m` 만큼만 본다. 결과는 항상 `lo <= prev_idx <= hi` 를 만족한다.

    **왜 창을 두는가 (F: `_find_segment` 전역탐색, SITL-7 R2 최우선 항목)**

    종전 `_find_segment()` 는 캐시 `_seg_idx` 를 **갱신만 하고 탐색에는 쓰지
    않아** 매 틱 경로 전체를 훑어 최근접 세그먼트를 골랐다. 그래서 선택이
    "경로상 진행도"가 아니라 **순간 기하학적 거리**로만 결정된다 —
    폐회로·U턴·자기교차처럼 서로 다른 레그가 가까이 지나가는 지점에서는
    기체가 지금 타고 있는 레그가 아니라 **엉뚱한 레그를 잡을 수 있고**,
    그 순간 `_lookahead_point()` 가 그 레그를 기준으로 전방 점을 뽑으므로
    목표점이 통째로 다른 곳(최악의 경우 진행 반대방향)으로 튄다.
    **대회 경로가 폐회로로 확정(2026-07-27 사용자)** 되었으므로 이건 잠재
    위험이 아니라 발현 조건이다.

    같은 계열의 실패가 이미 실측된 적이 있다 — MC 왕복(팰린드롬) 경로에서
    왕복 두 구간이 기하학적으로 겹쳐 최근접 탐색이 통째로 무의미해졌고,
    기체가 전환점에서 고정점에 수렴해 경로를 완주하지 못하고 영원히
    멈췄다(`fc_bridge/execution/state_logic.py:mc_wp_advance` 독스트링,
    2026-07-24 SITL 실측). MC 는 이산 순차추종으로 도망쳤지만 FW 는
    연속경로 투영이 필요해 이 함수를 고치는 것이 유일한 길이다.

    **창 폭 150m 의 근거**
      - 창은 "한 틱 사이에 투영점이 전진할 수 있는 거리 + 여유" 여야 한다.
        순항 20 m/s · 제어틱 0.1s → 2m/틱. 캠페인 실측 제어틱 p95 는 104ms
        이고 최악의 재개 공백을 1s 로 봐도 20m 다. 150m 는 그 7.5배다.
      - `_FW_LOOKAHEAD`(70m)보다 넉넉히 커야 한다 — 목표점이 창 밖에 있으면
        `_lookahead_point()` 가 창 끝에서 잘린 세그먼트부터 걷게 된다.
      - 진입 시점의 초기 오차도 이 창으로 흡수된다. STREAMING 진입 시
        기체는 경로 시작점에서 수십~수백 m 앞서 있을 수 있는데(천이 중
        전진), 인덱스가 매 틱 최대 창 폭만큼 전진하므로 300m 앞서 있어도
        2~3틱(0.2~0.3s) 만에 따라잡는다. 되돌아가지 못해 갇히는 경로가 없다.
      - **더 키우면 안 되는 이유:** 창이 넓을수록 "가까이 지나가는 다른 레그"가
        창 안에 들어와 전역탐색과 같은 오선택이 되살아난다. 창은 회귀 위험을
        줄이는 만큼만 넓혀야 한다.

    **역행 5m 의 근거 (완전 금지가 아닌 이유)**
      - 완전 단조(back_m=0)로 두면 코너 오버슈트 후 안쪽으로 되돌아 붙을 때
        투영점이 몇 m 앞에 붙박여, `_lookahead_point()` 가 실제 투영보다
        그만큼 앞에서 걷기 시작한다(70m lookahead 대비 무해하지만 불필요).
      - 5m 는 EKF 잡음·코너 투영 흔들림을 흡수하기엔 충분하고, **다른 레그에
        닿기엔 턱없이 부족하다** — 이 기체의 선회반경이
        `v²/(g·a_max_g) = 20²/(9.81×0.3) ≈ 136m` 라 비행 가능한 폐회로의
        레그 간격은 그 2배 수준이고, 자기교차 경로에서도 교차각이 있어
        5m 안에 두 레그가 함께 들어오는 구간은 교차점 극근방뿐이다.
      - 즉 "역행 금지"의 목적(진행도 되감기 차단)은 유지하면서 수 m 의
        수치 흔들림만 허용한다.
    """
    c = np.asarray(cum_len, dtype=float)
    n_seg = len(c) - 1
    if n_seg <= 0:
        return 0, 0
    i = int(min(max(int(prev_idx), 0), n_seg - 1))
    s0 = float(c[i])
    lo = int(np.searchsorted(c, s0 - float(back_m), side="left"))
    hi = int(np.searchsorted(c, s0 + float(window_m), side="right")) - 1
    lo = max(0, min(lo, i))
    hi = min(n_seg - 1, max(hi, i))
    return lo, hi


def nearest_segment_in_range(pts, p2, lo: int, hi: int) -> int:
    """`[lo, hi]` 구간에서 점 `p2` 에 가장 가까운 세그먼트 인덱스.

    동점이면 **작은 인덱스**를 고른다 — 종전 전역탐색 루프가 `d2 < best_d2`
    (strict) 로 첫 최소를 유지했던 것과 같은 규약이다(`np.argmin` 동일).
    """
    a2 = np.asarray(pts, dtype=float)
    a2 = a2.reshape(len(a2), -1)[:, :2]
    n_seg = len(a2) - 1
    if n_seg <= 0:
        return 0
    lo = int(min(max(int(lo), 0), n_seg - 1))
    hi = int(min(max(int(hi), lo), n_seg - 1))
    p = np.asarray(p2, dtype=float)[:2]

    a = a2[lo:hi + 1]
    b = a2[lo + 1:hi + 2]
    ab = b - a
    ab2 = np.einsum("ij,ij->i", ab, ab)
    safe = ab2 > 1e-12
    t = np.zeros(len(a))
    t[safe] = np.einsum("ij,ij->i", p - a[safe], ab[safe]) / ab2[safe]
    np.clip(t, 0.0, 1.0, out=t)
    proj = a + t[:, None] * ab
    d = p - proj
    return lo + int(np.argmin(np.einsum("ij,ij->i", d, d)))


class L1Guidance:
    """
    L1 guidance 계산기.

    Parameters
    ----------
    l1_dist : float
        Lookahead 거리 L1 (m). 크면 부드럽고, 작으면 공격적.
    path_pts : np.ndarray, shape (N, 2)
        경로점 2D 배열 [N, E] (NED 기준).
    v_profile : np.ndarray, shape (N,)
        각 경로점의 목표 속도 (m/s).
    gravity : float
        중력 가속도 (m/s²).
    seg_window_m : float
        세그먼트 탐색 창 (전방, m). 근거는 `segment_search_range()` 독스트링.
    seg_back_m : float
        세그먼트 탐색 허용 역행 폭 (m). 근거는 `segment_search_range()`.
    """

    def __init__(self,
                 l1_dist: float,
                 path_pts: np.ndarray,
                 v_profile: np.ndarray,
                 gravity: float = 9.81,
                 seg_window_m: float = _SEG_WINDOW_M,
                 seg_back_m: float = _SEG_BACK_M):
        self._l1 = float(l1_dist)
        self._pts = np.asarray(path_pts, dtype=float)    # (N, 2) 2D
        self._v   = np.asarray(v_profile, dtype=float)
        self._g   = gravity
        self._seg_idx = 0   # 마지막으로 찾은 세그먼트 인덱스 캐시
        # 세그먼트 탐색 창 (2026-07-27 R2) — 인덱스가 아니라 호길이(m) 기준.
        self._seg_window_m = float(seg_window_m)
        self._seg_back_m = float(seg_back_m)
        self._cum_len = cumulative_arclen(self._pts)

    # ── 공개 인터페이스 ──────────────────────────────────────

    def compute(self,
                pos_ned: np.ndarray,
                vel_ned: np.ndarray,
                ) -> tuple[float, float, float]:
        """
        현재 위치·속도에서 L1 guidance를 계산한다.

        Parameters
        ----------
        pos_ned : np.ndarray, shape (3,) or (2,)
            현재 위치 [N, E, (h)].
        vel_ned : np.ndarray, shape (3,) or (2,)
            현재 속도 [vN, vE, (vD)].

        Returns
        -------
        (chi_cmd, v_cmd, cross_track_err)
            chi_cmd       : 목표 헤딩 (rad)
            v_cmd         : 목표 속도 (m/s)
            cross_track_err : 횡방향 경로 오차 (m, **우측 + / 좌측 −**)
        """
        p2 = np.asarray(pos_ned[:2], dtype=float)
        v2 = np.asarray(vel_ned[:2], dtype=float)

        # 현재 속도 크기
        v_mag = float(np.linalg.norm(v2))
        if v_mag < 0.1:
            v_dir = np.array([1.0, 0.0])
        else:
            v_dir = v2 / v_mag

        # 가장 가까운 세그먼트 탐색
        seg = self._find_segment(p2)
        self._seg_idx = seg

        # lookahead point 계산
        lh_pt, lh_seg = self._lookahead_point(p2, seg)

        # 횡방향 오차 (세그먼트 법선 방향)
        cross_err = self._cross_track_error(p2, seg)

        # L1 벡터: 현재 위치 → lookahead
        vec_to_lh = lh_pt - p2
        dist_to_lh = float(np.linalg.norm(vec_to_lh))
        if dist_to_lh < 1e-3:
            # lookahead가 너무 가까우면 다음 세그먼트 방향 사용
            nxt = min(seg + 1, len(self._pts) - 1)
            vec_to_lh = self._pts[nxt] - p2
            dist_to_lh = float(np.linalg.norm(vec_to_lh))

        if dist_to_lh < 1e-3:
            return 0.0, float(self._v[min(seg, len(self._v) - 1)]), cross_err

        # 각도 에러 η (속도벡터와 lookahead 벡터 사이)
        lh_dir = vec_to_lh / dist_to_lh
        sin_eta = float(np.clip(
            v_dir[0] * lh_dir[1] - v_dir[1] * lh_dir[0],
            -1.0, 1.0
        ))
        cos_eta = float(np.clip(
            v_dir[0] * lh_dir[0] + v_dir[1] * lh_dir[1],
            -1.0, 1.0
        ))
        eta = float(np.arctan2(sin_eta, cos_eta))

        # 횡방향 가속도 명령
        a_lat = 2.0 * max(v_mag, 1.0) ** 2 * np.sin(eta) / self._l1

        # 목표 헤딩: lookahead 방향을 기반으로 a_lat 반영
        chi_lh = float(np.arctan2(lh_dir[1], lh_dir[0]))
        # 현재 헤딩에서 a_lat/v² 만큼 회전
        delta_chi = np.arctan2(a_lat, max(v_mag, 1.0) ** 2 / self._l1)
        chi_cmd = float(_wrap(chi_lh))

        # 속도 명령: lookahead 세그먼트 속도 사용
        v_idx = min(lh_seg, len(self._v) - 1)
        v_cmd = float(self._v[v_idx])

        return chi_cmd, v_cmd, float(cross_err)

    def ned_velocity_cmd(self,
                         pos_ned: np.ndarray,
                         vel_ned: np.ndarray,
                         gamma_ref: float = 0.0,
                         ) -> np.ndarray:
        """
        NED 속도벡터 세트포인트 [vN, vE, vD] 를 반환한다.

        Parameters
        ----------
        pos_ned, vel_ned : np.ndarray
        gamma_ref : float
            현재 경로점의 상승각 (rad). 고도 제어에 사용.

        Returns
        -------
        np.ndarray, shape (3,)  [vN, vE, vD]
        """
        chi_cmd, v_cmd, _ = self.compute(pos_ned, vel_ned)
        vN = v_cmd * np.cos(chi_cmd)
        vE = v_cmd * np.sin(chi_cmd)
        vD = -v_cmd * np.sin(gamma_ref)   # D축은 아래가 양수
        return np.array([vN, vE, vD])

    def target_point_ned(self,
                         pos_ned: np.ndarray,
                         lookahead: float,
                         ) -> np.ndarray:
        """현재 위치에서 경로를 따라 lookahead(m) 전방의 위치 [N, E]를 반환.

        FW 오프보드는 위치 setpoint만 추종하므로, 이 점을 위치 목표로 발행한다.
        lookahead는 기체 선회반경보다 충분히 커야(목표점을 반경 밖에 둬야) FW가
        목표점을 orbit하는 flower-pattern을 피한다.

        Parameters
        ----------
        pos_ned : np.ndarray, shape (3,) or (2,)
            현재 위치 [N, E, (h)].
        lookahead : float
            경로 투영점으로부터의 전방 거리 (m).

        Returns
        -------
        np.ndarray, shape (2,)  [N, E]
        """
        p2 = np.asarray(pos_ned[:2], dtype=float)
        seg = self._find_segment(p2)
        self._seg_idx = seg
        lh_pt, _ = self._lookahead_point(p2, seg, lookahead)
        return lh_pt.copy()

    @property
    def current_segment(self) -> int:
        return self._seg_idx

    # ── 내부 유틸리티 ────────────────────────────────────────

    def _find_segment(self, p2: np.ndarray) -> int:
        """직전 세그먼트 근방(창) 안에서 가장 가까운 세그먼트 인덱스 반환.

        **2026-07-27 (SITL-7 R2) 이전에는 매 틱 경로 전체를 훑었다.** 캐시
        `_seg_idx` 를 갱신만 하고 탐색에는 쓰지 않아, 선택이 경로상 진행도가
        아니라 순간 거리로만 정해졌다 — 폐회로·U턴에서 엉뚱한 레그를 잡을 수
        있는 구조였다. 창 폭·역행 규칙과 그 근거는 `segment_search_range()`
        독스트링에 있다.

        ⚠️ 이 함수는 상태를 읽기만 한다. `_seg_idx` 갱신은 호출부
        (`compute()` / `target_point_ned()`)가 한다 — 종전과 동일한 규약이라
        호출 순서에 대한 가정이 바뀌지 않는다.
        """
        if len(self._pts) < 2:
            return 0
        lo, hi = segment_search_range(
            self._cum_len, self._seg_idx,
            self._seg_window_m, self._seg_back_m)
        return nearest_segment_in_range(self._pts, p2, lo, hi)

    def _lookahead_point(self,
                         p2: np.ndarray,
                         start_seg: int,
                         dist: float | None = None,
                         ) -> tuple[np.ndarray, int]:
        """
        start_seg 이후 경로 위에서 p2의 투영점으로부터 dist(기본 L1)만큼
        전방의 점을 반환.
        """
        N = len(self._pts)
        remain = self._l1 if dist is None else float(dist)

        for i in range(start_seg, N - 1):
            a = self._pts[i]
            b = self._pts[i + 1]
            seg_len = float(np.linalg.norm(b - a))
            if seg_len < 1e-9:
                continue

            # a에서 p2까지 투영 거리
            ab = b - a
            t = float(np.clip((p2 - a) @ ab / (seg_len ** 2), 0.0, 1.0))
            proj = a + t * ab
            dist_from_proj = float(np.linalg.norm(p2 - proj))

            # 이 세그먼트에서 남은 거리
            along_remain = seg_len * (1.0 - t)

            if along_remain >= remain:
                frac = t + remain / seg_len
                return a + frac * ab, i

            remain -= along_remain

        # 경로 끝 도달
        return self._pts[-1].copy(), N - 2

    def _cross_track_error(self, p2: np.ndarray, seg: int) -> float:
        """횡방향 경로 오차 (m). **우측 + / 좌측 −** (2026-07-25 실측 정정).

        normal = [-ab[1], ab[0]] 는 NED에서 진행방향의 **오른쪽** 법선이다
        (북진 ab=[1,0] → normal=[0,1]=동=오른쪽). 실측:
            동쪽 +5m(오른쪽) → cte=+5.000 / 서쪽 -5m(왼쪽) → cte=-5.000
        종전 docstring은 "좌측 +"라고 반대로 적혀 있었다. 현재 cte는 로그와
        abs() 비교에만 쓰여 거동 영향은 없었으나, 횡방향 보정에 쓰는 순간
        부호가 반대로 들어간다.
        """
        N = len(self._pts)
        a = self._pts[seg]
        b = self._pts[min(seg + 1, N - 1)]
        ab = b - a
        ab_len = float(np.linalg.norm(ab))
        if ab_len < 1e-9:
            return float(np.linalg.norm(p2 - a))
        # 부호 있는 횡방향 거리
        normal = np.array([-ab[1], ab[0]]) / ab_len
        return float((p2 - a) @ normal)
