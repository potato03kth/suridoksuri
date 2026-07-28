from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
import numpy as np

from .state import VisionState


class Pipeline:
    def __init__(self, steps: list[Any]):
        self.steps = steps

    def run(self, image: np.ndarray, meta: dict | None = None) -> VisionState:
        """`meta`(선택, 기본 None): 파이프라인 시작 전에 `state.meta`에 심어 둘 **프레임 단위
        외부 입력**. 모듈은 `state`만 받으므로(`__call__(state)->state`), 프레임마다 달라지는
        외부 정보(라이다 AGL·기체 자세 등)를 스텝에 전달할 통로가 이것 말고 없다.

        `modules/prior_score.py::PriorGeometryScorer`가 이 통로로 사전정보를 받는다
        (`prior_inputs` 키). 캘리브레이션 로드는 여전히 `main.py`/`replay.py`가 하고 여기로는
        **객체만** 흘러온다 — import 규칙("modules/ ← vision.core 만")을 지키기 위함.

        **기본값 None이면 예전과 한 줄도 다르지 않다** — 기존 호출부는 전부 무변경."""
        state = VisionState(original=image.copy(), current=image.copy())
        if meta:
            state.meta.update(meta)
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
