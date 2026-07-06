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
