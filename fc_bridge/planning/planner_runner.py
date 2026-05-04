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
_VTOL_SIM_ROOT = FSPath(__file__).parents[2] / "vtol_sim_checkpoint1_1"
if str(_VTOL_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VTOL_SIM_ROOT))

from vtol_sim.path_planning.base_planner import Path  # noqa: E402


_PLANNER_NAMES = ("eta3", "diterpin")


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
        "eta3" 또는 "diterpin"
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
        from vtol_sim.path_planning.eta3clothoid_v3_1_planner import (
            Eta3ClothoidPlannerV3,
        )
        planner = Eta3ClothoidPlannerV3(**kwargs)

    else:  # diterpin
        from vtol_sim.path_planning.D_iterpin_planner import DIterativePinPlanner
        planner = DIterativePinPlanner(**kwargs)

    return planner.plan(wps, vehicle_params, initial_state)
