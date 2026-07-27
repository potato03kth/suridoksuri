"""`registry.py` 회귀 테스트.

검증 대상은 `vision/CLAUDE.md` 테스트 규칙표의 계약 —
**"등록 이름 전부 실제 클래스 매핑·중복 없음"**.

`registry.MODULES`는 preset yaml의 문자열을 실제 클래스로 바꾸는 단 하나의 관문이라
(`core/runner.py::Pipeline.from_config`), 여기가 조용히 어긋나면 preset이 통째로
죽거나(ValueError) 엉뚱한 모듈이 실행된다. 그래서 "매핑이 존재한다"에서 멈추지 않고
**실제로 인스턴스화해 파이프라인 인터페이스로 호출되는지**까지 확인한다.

⚠️ **"중복 없음"은 런타임 dict로는 검증할 수 없다** — 파이썬 dict 리터럴은 키가
겹치면 조용히 뒤엣것으로 덮어쓰고 끝나서, 이미 만들어진 `MODULES`를 아무리 들여다봐도
중복이 있었다는 사실 자체가 남지 않는다. 그래서 소스를 `ast`로 파싱해 원문 키를 센다.
"""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

import vision.modules as modules_pkg
from vision import registry
from vision.core.state import VisionState

PRESET_DIR = Path(registry.__file__).parent / "presets"


def _blank_state() -> VisionState:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    return VisionState(original=img, current=img.copy())


# --------------------------------------------------------------------------
# 등록 이름 → 실제 클래스 매핑
# --------------------------------------------------------------------------

def test_registry_is_not_empty():
    """빈 dict가 조용히 통과하면 아래 매핑 테스트들이 전부 공회전한다."""
    assert len(registry.MODULES) >= 14


@pytest.mark.parametrize("name", sorted(registry.MODULES))
def test_every_registered_name_maps_to_a_real_class(name):
    """등록 값은 (문자열/None/인스턴스가 아니라) 실제 클래스여야 한다."""
    obj = registry.MODULES[name]
    assert inspect.isclass(obj), f"'{name}' 등록 값이 클래스가 아니다: {obj!r}"


@pytest.mark.parametrize("name", sorted(registry.MODULES))
def test_every_registered_class_lives_in_vision_modules(name):
    """등록된 클래스는 전부 `vision.modules` 패키지의 것이어야 한다.

    (오타로 엉뚱한 동명 클래스를 끌어다 등록하는 사고 방지)
    """
    cls = registry.MODULES[name]
    assert cls.__module__.startswith("vision.modules."), \
        f"'{name}' → {cls.__module__}.{cls.__name__} 는 vision.modules 소속이 아니다"
    assert getattr(modules_pkg, cls.__name__, None) is cls


@pytest.mark.parametrize("name", sorted(registry.MODULES))
def test_every_registered_class_implements_pipeline_interface(name):
    """모든 모듈 계약: 인자 없이 생성 가능하고 `__call__(state) -> VisionState`.

    매핑만 확인하고 끝내면 "클래스이긴 한데 파이프라인에선 못 쓰는 것"이 통과한다.
    실제로 만들어 빈 프레임을 한 번 통과시켜 본다(공통 규칙 3: 경계 입력 안전).
    """
    cls = registry.MODULES[name]
    instance = cls()                      # preset이 파라미터를 생략해도 살아야 한다
    assert callable(instance), f"'{name}' 인스턴스가 호출 가능하지 않다"

    out = instance(_blank_state())
    assert isinstance(out, VisionState), f"'{name}' 이 VisionState를 반환하지 않았다"


# --------------------------------------------------------------------------
# 중복 없음
# --------------------------------------------------------------------------

def _registry_literal_keys() -> list[str]:
    """registry.py 소스에서 MODULES dict 리터럴의 키를 **원문 그대로** 뽑는다."""
    source = Path(registry.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        targets = getattr(node, "targets", None) or (
            [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "MODULES":
                assert isinstance(node.value, ast.Dict), "MODULES가 dict 리터럴이 아니다"
                return [k.value for k in node.value.keys]
    raise AssertionError("registry.py 에서 MODULES 대입문을 찾지 못했다")


def test_source_has_no_duplicate_registration_names():
    """이름 중복 등록 금지 — dict 리터럴이 조용히 덮어쓰는 사고를 소스에서 잡는다."""
    keys = _registry_literal_keys()
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    assert duplicates == [], f"중복 등록된 이름: {duplicates}"
    # 소스 키 목록과 런타임 dict가 같아야 한다(= 조용히 덮어쓴 것이 없다)
    assert sorted(keys) == sorted(registry.MODULES)


def test_no_class_is_registered_under_two_names():
    """같은 클래스를 이름만 바꿔 두 번 등록하지 않는다 (preset 혼선 방지)."""
    seen: dict[type, str] = {}
    for name, cls in registry.MODULES.items():
        assert cls not in seen, f"{cls.__name__} 이 '{seen[cls]}' 와 '{name}' 로 중복 등록됐다"
        seen[cls] = name


# --------------------------------------------------------------------------
# 등록 누락 / preset 정합 — 레지스트리가 "단 하나의 관문"이라는 계약
# --------------------------------------------------------------------------

def test_every_exported_module_class_is_registered():
    """`vision.modules.__all__` 의 모듈 클래스는 전부 등록돼 있어야 한다.

    새 모듈을 만들고 `registry.py` 등록을 깜빡하는 것이 이 저장소의 실제 실수 경로다
    (`vision/CLAUDE.md`: "새 모듈 등록은 여기에만").
    """
    registered = set(registry.MODULES.values())
    missing = [
        n for n in modules_pkg.__all__
        if getattr(modules_pkg, n) not in registered
    ]
    assert missing == [], f"registry.py 에 등록되지 않은 모듈: {missing}"


@pytest.mark.parametrize(
    "preset_path", sorted(PRESET_DIR.glob("*.yaml")), ids=lambda p: p.name
)
def test_every_preset_step_name_is_registered(preset_path):
    """커밋된 preset이 참조하는 스텝 이름은 전부 등록돼 있어야 한다.

    `Pipeline.from_config`가 미등록 이름에 ValueError를 내므로, 이 정합이 깨지면
    해당 preset은 로드 자체가 불가능하다 — 실행 전에 여기서 잡는다.
    """
    cfg = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
    unknown = [n for n in cfg["pipeline"] if n not in registry.MODULES]
    assert unknown == [], f"{preset_path.name} 의 미등록 스텝: {unknown}"
