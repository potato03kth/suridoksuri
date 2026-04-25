"""
설정 파일 로더
================

YAML 파일을 항상 UTF-8로 읽어 Windows cp949 인코딩 충돌을 방지.
한글 주석이 포함된 설정 파일에 필수.
"""
from __future__ import annotations
import os
import yaml


def load_yaml(path: str) -> dict:
    """UTF-8 인코딩으로 YAML 파일 로드."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_dir() -> str:
    """vtol_sim/config 디렉토리 절대 경로."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
    )


def load_aircraft_params() -> dict:
    return load_yaml(os.path.join(get_config_dir(), "aircraft.yaml"))


def load_simulation_params() -> dict:
    return load_yaml(os.path.join(get_config_dir(), "simulation.yaml"))


def load_scenario(name: str) -> dict:
    """시나리오 이름으로 로드. 예: 'basic' → scenario_basic.yaml"""
    fname = f"scenario_{name}.yaml" if not name.startswith("scenario_") else f"{name}.yaml"
    return load_yaml(os.path.join(get_config_dir(), fname))


# =============================================================================
# 시나리오 override 메커니즘
# =============================================================================
# 시나리오 YAML에서 다음 키들을 정의하면 aircraft.yaml의 동일 키를 override한다.
# 이 목록은 의미론적 안전성을 위해 명시적으로 관리한다 (실수로 무관 키가
# 덮어써지는 것을 방지).
SCENARIO_OVERRIDABLE_AIRCRAFT_KEYS = {
    "a_max_g",       # 가속도 한계 (시나리오 3 등에서 변경)
    "v_cruise",      # 순항속도 (시나리오 3에서 변경)
    "v_max",
    "v_min_turn",
    "phi_max_deg",
    "h_min",
    "h_max",
}


def merge_scenario_into_aircraft(aircraft_params: dict,
                                 scenario: dict) -> dict:
    """
    시나리오의 override 값들을 aircraft 파라미터에 병합.

    원본은 변경하지 않고 새 dict 반환 (deepcopy 방식).
    SCENARIO_OVERRIDABLE_AIRCRAFT_KEYS에 있는 키만 처리.

    Returns
    -------
    merged : dict
        시나리오 override가 반영된 aircraft 파라미터
    overrides_applied : 출력은 안 하지만 로깅용으로 print 가능하도록 dict로 분리 가능
    """
    import copy
    merged = copy.deepcopy(aircraft_params)
    for key in SCENARIO_OVERRIDABLE_AIRCRAFT_KEYS:
        if key in scenario:
            merged[key] = scenario[key]
    return merged


def get_active_aircraft_params(scenario_name: str) -> dict:
    """편의 함수: 시나리오 적용된 aircraft 파라미터를 한 번에 로드."""
    aircraft = load_aircraft_params()
    scenario = load_scenario(scenario_name)
    return merge_scenario_into_aircraft(aircraft, scenario)
