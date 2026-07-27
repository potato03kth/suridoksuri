"""
경로 생성 알고리즘 실행기.

eta3 또는 diterpin 플래너를 선택해 Path 객체를 반환한다.
vtol_sim 패키지 경로를 sys.path에 추가해 import한다.
"""
from __future__ import annotations
import sys
from pathlib import Path as FSPath
import numpy as np

# vtol_sim 패키지 import 경로 설정
# parents[2] = suridoksuri-1/ (repo root) — vtol_sim_checkpoint1_1 의 부모가 필요
_REPO_ROOT = FSPath(__file__).parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vtol_sim_checkpoint1_1.vtol_sim.path_planning.base_planner import Path  # noqa: E402


_PLANNER_NAMES = ("eta3", "diterpin", "straight")


def resolve_planner_name(planner_param: str, vehicle_type: str) -> str:
    """planner 파라미터를 실제 플래너 이름으로 해석한다.

    - "auto"(또는 빈 값): 기체 타입으로 자동 선택 —
        mc(멀티콥터/쿼드) → "straight"(곡률 완화 불필요 + 퇴화 WP에 안전),
        vtol/그 외        → "eta3".
    - 그 외(명시 지정): 지정값을 그대로 사용 — 명시 지정이 항상 우선한다.
    """
    p = (planner_param or "").strip().lower()
    if p and p != "auto":
        return p
    vt = (vehicle_type or "").strip().lower()
    return "straight" if vt == "mc" else "eta3"


def planner_eta_s(planner_name: str, n_waypoints: int,
                  per_corner_s: float = 90.0) -> float:
    """경로 생성 블로킹 예상 소요시간 (s). 어림값이고 정밀도를 주장하지 않는다.

    `run_planner()` 는 `OffboardNode.__init__` 에서 **동기로** 돌아 그 동안
    노드가 콜백을 하나도 처리하지 못한다. 캠페인 실측(§1-4):

        2WP 즉시 · L자(3WP) 50~69s · U턴(3WP) 45~73s · 폐곡선(5WP) **263.5s**

    즉 비용은 **내부 코너 수(N-2)** 에 거의 비례한다.
    `straight` 플래너와 2WP 이하는 0.

    **이 값의 용도는 오직 "지금 멈춘 게 아니라 계산 중"을 현장에 알리는
    것**이다(F-12). 현장에서 launch 가 죽은 줄 알고 재기동하면 중복 인스턴스가
    되고, 그건 gz·MAVROS·OFFBOARD 스트림이 두 벌 도는 상태다.
    그래서 **예고는 실측보다 항상 커야 한다** — 작으면 "예고보다 오래 걸리네,
    죽었나"로 오판하게 만들어 이 로그의 존재 이유를 정면으로 배반한다.

    **코너당 75s → 90s 상향 (2026-07-28, `v_cruise` 정식값 18.0 확정 세션).**
    75s 는 `v_cruise=20.0` 실측(코너1 52.4~55.9s / 코너3 188.5~195.6s)에는
    충분했지만, 정식값 18.0 실측에서 마진이 사실상 소진됐다:

        v_cruise=18.0 SITL 실측 — R2_a3_vc18 57.6s(예고 75, 여유 23%)
                                  R2_b5_vc18 221.6s(예고 225, 여유 **1.5%**)
                                  R2_closed_vc18 223.6s(예고 225, 여유 **0.6%**)
        원본: logs/2026-07-28_vc18/*/node.log

    코너당 90s 면 코너3 예고 270s 로 실측 223.6s 대비 **21% 여유**가 되고,
    코너1 90s 는 실측 57.6s 대비 56% 여유다.

    ⚠️ 소요시간의 `v_cruise` 민감도는 **매끄러운 곡선이 아니라 16↔17 사이의
    계단**이다(2026-07-28 실측, `tools/sitl/measure_planner_blocking.py`):
    L자 16→0.46s / 17→23.4s, 폐회로 16→27.4s / 17→101.8s. ≤16 이 싼 이유는
    NR 이 일찍 포기해(잔차 8~36m) 변형된 곡선을 내놓기 때문이고, ≥17 은
    잔차 ≈5m 로 수렴한다. 즉 **18 과 20 은 같은 대역이라 비용이 사실상 같다** —
    이 어림이 코너 수만 보는 것이 정식값 대역에서는 오히려 타당하다.
    """
    if str(planner_name).strip().lower() == "straight":
        return 0.0
    n = int(n_waypoints)
    if n <= 2:
        return 0.0
    return float(per_corner_s) * (n - 2)


def run_planner(
    planner_name: str,
    waypoints_ned: np.ndarray,
    vehicle_params: dict,
    planner_kwargs: dict | None = None,
    initial_state: dict | None = None,
) -> Path:
    """
    지정 플래너로 경로를 생성해 반환한다.

    Parameters
    ----------
    planner_name : str
        "eta3" | "diterpin" | "straight" (해석된 실제 플래너 이름).
        "auto"는 resolve_planner_name()으로 먼저 해석해 넘길 것.
    waypoints_ned : np.ndarray, shape (N, 3)
        NED 좌표 웨이포인트 [N, E, h_up].
    vehicle_params : dict
        최소 키: "v_cruise" (m/s), "a_max_g" (g), "gravity" (m/s²).
    planner_kwargs : dict, optional
        플래너 생성자에 전달할 추가 파라미터.
    initial_state : dict, optional
        "initial_heading" (rad) 등.

    Returns
    -------
    Path
    """
    if planner_name not in _PLANNER_NAMES:
        raise ValueError(f"planner_name은 {_PLANNER_NAMES} 중 하나여야 합니다.")

    kwargs = planner_kwargs or {}
    wps = np.asarray(waypoints_ned, dtype=float)

    if planner_name == "eta3":
        from vtol_sim_checkpoint1_1.vtol_sim.path_planning.eta3clothoid_v3_1_planner import (
            Eta3ClothoidPlannerV3,
        )
        planner = Eta3ClothoidPlannerV3(
            ds=1.0, accel_tol=0.85, end_extension=0)

    elif planner_name == "straight":
        from vtol_sim_checkpoint1_1.vtol_sim.path_planning.straight_line_planner import (
            StraightLinePlanner,
        )
        planner = StraightLinePlanner(ds=1.0)

    else:  # diterpin
        from vtol_sim_checkpoint1_1.vtol_sim.path_planning.D_iterpin_planner import DIterativePinPlanner
        planner = DIterativePinPlanner(num_iter=4, alpha0=0.6, straight_ratio0=0.4,
                                       search_steps=50, max_detours=0, alpha_range=(0.1, 2.2), sr_range=(0.02, 0.8))

    return planner.plan(wps, vehicle_params, initial_state)
