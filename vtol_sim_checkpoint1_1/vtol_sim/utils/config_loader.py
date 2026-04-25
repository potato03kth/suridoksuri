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
