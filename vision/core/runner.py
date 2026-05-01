from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
import numpy as np

from .state import VisionState


class Pipeline:
    def __init__(self, steps: list[Any]):
        self.steps = steps

    def run(self, image: np.ndarray) -> VisionState:
        state = VisionState(original=image.copy(), current=image.copy())
        for step in self.steps:
            state = step(state)
        return state

    # 시나리오 전환: yaml 경로만 바꾸면 됨
    @classmethod
    def from_config(cls, config_path: str) -> "Pipeline":
        from vision.registry import MODULES
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        steps = []
        for name, params in cfg["pipeline"].items():
            params = params or {}
            if name not in MODULES:
                raise ValueError(f"Unknown module: '{name}'. Available: {list(MODULES)}")
            steps.append(MODULES[name](**params))
        return cls(steps)

    # 디버그용: steps 슬라이싱으로 부분 파이프라인 실행
    def partial(self, end: int) -> "Pipeline":
        return Pipeline(self.steps[:end])
