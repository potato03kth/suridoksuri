import numpy as np
import pytest
from vision.core.runner import Pipeline
from vision.modules.color import ColorFilter
from vision.modules.edge import EdgeDetector
from vision.modules.detector import RectDetector


def _blank_image(h: int = 200, w: int = 200) -> np.ndarray:
    return np.full((h, w, 3), 180, dtype=np.uint8)


def test_pipeline_runs_all_steps():
    ran = []

    class Marker:
        def __init__(self, name):
            self.name = name
        def __call__(self, state):
            ran.append(self.name)
            return state

    pipeline = Pipeline([Marker("a"), Marker("b"), Marker("c")])
    pipeline.run(_blank_image())
    assert ran == ["a", "b", "c"]


def test_partial_pipeline():
    pipeline = Pipeline([ColorFilter(), EdgeDetector(), RectDetector()])
    partial = pipeline.partial(2)
    assert len(partial.steps) == 2


def test_from_config_single_frame(tmp_path):
    yaml_content = """
pipeline:
  color_filter:
    mode: gray
    sat_max: 50
    val_min: 80
  rect_detector:
    min_area: 100
"""
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)
    pipeline = Pipeline.from_config(str(config_file))
    state = pipeline.run(_blank_image())
    assert state is not None


def test_from_config_unknown_module_raises(tmp_path):
    yaml_content = "pipeline:\n  nonexistent_module:\n"
    config_file = tmp_path / "bad.yaml"
    config_file.write_text(yaml_content)
    with pytest.raises(ValueError, match="Unknown module"):
        Pipeline.from_config(str(config_file))
