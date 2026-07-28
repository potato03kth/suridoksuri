"""골든셋 재생 회귀 테스트 (vision_plan.md §7.5 "골든셋 회귀 테스트", §7.9 항목7).

`vision/tests/golden/<타겟>/<고도>/`의 합성 프레임(§ generate_synthetic.py, 실촬영 아님 —
docs/vision_status.md 참조)을 **실제** `vision.replay.run_replay()`(DirFrameSource + 실제
Pipeline)로 재생시켜, 실제로 나온 JSONL 블랙박스 검출 결과를 `labels.json`의 기대값과
비교한다. 파이프라인을 몽키패치하지 않는다 — replay.py가 실제로 쓰는 것과 동일한
`Pipeline.from_config(preset).run(image)` 경로를 그대로 통과시킨다.

이 테스트가 실패하면: 검출기(color.py/detector.py/vertiport_*.py)를 고치라는 뜻이 아니라
(이번 세션 범위 밖, vision/CLAUDE.md 공통규칙), 파이프라인이 실제로 달라졌다는 뜻이므로
`labels.json`의 기대값을 새 실제 동작에 맞게 갱신하라는 신호다("X 고치다 Y 깨짐" 방지가
이 회귀 테스트의 존재 이유 그 자체).
"""
import json
from pathlib import Path

import cv2
import pytest

import vision.replay as replay_mod
from vision.core.runner import Pipeline

_GOLDEN_ROOT = Path(__file__).parent / "golden"
_PRESETS_DIR = Path(__file__).parent.parent / "presets"


def _discover_leaf_dirs() -> list[Path]:
    """labels.json을 직접 담고 있는 리프 디렉터리들(예: golden/vertiport/10m/)을 찾는다."""
    return sorted(p.parent for p in _GOLDEN_ROOT.rglob("labels.json"))


_LEAF_DIRS = _discover_leaf_dirs()


def _leaf_id(path: Path) -> str:
    return str(path.relative_to(_GOLDEN_ROOT))


def test_golden_set_scaffold_is_non_empty():
    """스캐폴드 자체가 비어있으면 이하 파라미터화 테스트가 전부 스킵되며 조용히 통과해버린다 —
    그 상황을 명시적으로 잡아낸다."""
    assert len(_LEAF_DIRS) > 0, "vision/tests/golden 아래 labels.json이 하나도 없다"
    targets = {p.relative_to(_GOLDEN_ROOT).parts[0] for p in _LEAF_DIRS}
    assert targets == {"vertiport", "distress", "no_target"}


def test_no_target_has_distress_coarse_regression_case():
    """커버리지 갭 회귀: no_target 오탐 방지가 vertiport_coarse.yaml 만이 아니라
    distress_coarse.yaml(초록색 필터, ② 조난자 구역 coarse)로도 검증되는지 —
    한 프리셋만 통과하고 다른 프리셋에서 오탐이 나는 경우를 놓치지 않기 위함."""
    no_target_presets = {
        json.loads((leaf / "labels.json").read_text(encoding="utf-8"))["preset"]
        for leaf in _LEAF_DIRS
        if leaf.relative_to(_GOLDEN_ROOT).parts[0] == "no_target"
    }
    assert "distress_coarse.yaml" in no_target_presets, (
        "no_target 오탐 회귀가 distress_coarse.yaml 프리셋으로 커버되지 않음"
    )


def test_no_target_has_distress_fine_regression_case():
    """[2026-07-25 신설] white_box_detector 캐스케이드(distress_fine.yaml)도 빈 장면에서
    엉뚱한 흰 박스를 만들어내지 않는지 — 위 distress_coarse 커버리지 갭 회귀와 동일 이유로
    새 모듈에 대해서도 반복한다."""
    no_target_presets = {
        json.loads((leaf / "labels.json").read_text(encoding="utf-8"))["preset"]
        for leaf in _LEAF_DIRS
        if leaf.relative_to(_GOLDEN_ROOT).parts[0] == "no_target"
    }
    assert "distress_fine.yaml" in no_target_presets, (
        "no_target 오탐 회귀가 distress_fine.yaml(white_box_detector) 프리셋으로 커버되지 않음"
    )


@pytest.mark.parametrize("leaf_dir", _LEAF_DIRS, ids=_leaf_id)
def test_replay_matches_golden_labels(leaf_dir, tmp_path):
    """§ 요구사항 핵심: vision/replay.py의 run_replay()로 골든 프레임을 실제로 재생하고,
    실제로 쓰인 JSONL 블랙박스의 detections 개수를 labels.json 기대값과 비교한다."""
    labels = json.loads((leaf_dir / "labels.json").read_text(encoding="utf-8"))
    preset_path = str(_PRESETS_DIR / labels["preset"])
    log_dir = tmp_path / "logs"

    frame_count = replay_mod.run_replay(
        str(leaf_dir), preset_path, display="none", output=None,
        log_dir=str(log_dir), log_name="golden",
    )
    assert frame_count == len(labels["frames"])

    records = [json.loads(line) for line in (log_dir / "golden.jsonl").read_text(encoding="utf-8").splitlines()]
    records.sort(key=lambda r: r["frame_id"])

    assert len(records) == len(labels["frames"])
    for record, expected in zip(records, labels["frames"]):
        n_expected = expected["expect_num_detections"]
        assert len(record["detections"]) == n_expected, (
            f"{leaf_dir} / {expected['file']}: replay 실제 검출 {len(record['detections'])}건, "
            f"골든 기대값 {n_expected}건 (known_limitation={expected.get('known_limitation')})"
        )


@pytest.mark.parametrize("leaf_dir", _LEAF_DIRS, ids=_leaf_id)
def test_pipeline_stage_meta_matches_golden_labels(leaf_dir):
    """detections 개수만으로는 "왜"가 안 보인다 — 캐스케이드 단계별 meta(§5.2)까지 실제
    Pipeline.run()으로 직접 검증해 어느 단계에서 확인/거절됐는지 회귀로 고정한다.
    (replay.py와 동일하게 실제 Pipeline.from_config(...).run(image)를 그대로 호출 — 우회 없음.)"""
    labels = json.loads((leaf_dir / "labels.json").read_text(encoding="utf-8"))
    preset_path = str(_PRESETS_DIR / labels["preset"])
    pipeline = Pipeline.from_config(preset_path)

    for expected in labels["frames"]:
        expect_meta = expected.get("expect_stage_meta")
        if expect_meta is None:
            continue
        image = cv2.imread(str(leaf_dir / expected["file"]))
        assert image is not None, f"이미지 로드 실패: {leaf_dir / expected['file']}"
        state = pipeline.run(image)
        for stage_name, expected_stage_meta in expect_meta.items():
            assert state.meta.get(stage_name) == expected_stage_meta, (
                f"{leaf_dir} / {expected['file']}: stage '{stage_name}' meta 불일치 — "
                f"실제={state.meta.get(stage_name)!r} 기대={expected_stage_meta!r}"
            )
        assert len(state.detections) == expected["expect_num_detections"]


# ---------------------------------------------------------------------------
# 재생성 경로 무결성 (2026-07-28)
#
# `no_target/distress_coarse/` 리프가 2026-07-21에 **손으로** 만들어져 `generate_synthetic.py`
# 에 들어가지 않았고, 그래서 골든셋을 재생성하면 그 리프만 조용히 사라지는 갭이 있었다.
# 아래 테스트가 그 갭 자체를 회귀로 고정한다 — 앞으로 새 리프를 손으로 추가하면 red 가 된다.
# ---------------------------------------------------------------------------


def _load_generator_module():
    """`generate_synthetic.py`는 패키지가 아닌 스크립트라(golden/ 에 `__init__.py` 없음)
    파일 경로로 직접 로드한다. 파일명이 `test_*` 가 아니라 pytest 수집 대상도 아니다."""
    import importlib.util

    path = _GOLDEN_ROOT / "generate_synthetic.py"
    spec = importlib.util.spec_from_file_location("_golden_generate_synthetic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_golden_leaf_is_reproducible_by_generate_synthetic(tmp_path, monkeypatch):
    """저장소에 커밋된 골든셋 전체가 `generate_synthetic.py` 로 **바이트 동일하게** 재생성되는가.

    리프 목록과 파일 내용을 모두 본다:
      - 생성기가 안 만드는 리프가 있으면(=손으로 만든 리프) red
      - 생성기 출력이 커밋된 골든과 한 바이트라도 다르면 red (조용한 드리프트 방지)
    """
    gen = _load_generator_module()
    monkeypatch.setattr(gen, "_ROOT", tmp_path)
    gen.generate()

    def _tree(root):
        return {
            str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in (".png", ".json")
        }

    regenerated = _tree(tmp_path)
    committed = _tree(_GOLDEN_ROOT)

    missing = sorted(set(committed) - set(regenerated))
    assert not missing, (
        f"generate_synthetic.py 가 재생성하지 못하는 골든 파일: {missing} — "
        "손으로 만든 리프는 생성기에도 반드시 넣어야 재생성 경로가 안 끊긴다."
    )
    extra = sorted(set(regenerated) - set(committed))
    assert not extra, f"생성기가 만들었는데 커밋 안 된 골든 파일: {extra}"

    differing = sorted(k for k in committed if committed[k] != regenerated[k])
    assert not differing, f"재생성 결과가 커밋된 골든과 다르다(조용한 드리프트): {differing}"


def test_no_target_distress_coarse_leaf_is_generated():
    """복구한 그 리프를 이름으로 못 박는다 — 위 전수 테스트가 통째로 지워져도 남는 최소 방어선."""
    leaf = _GOLDEN_ROOT / "no_target" / "distress_coarse"
    assert (leaf / "frame_000.png").exists()
    labels = json.loads((leaf / "labels.json").read_text(encoding="utf-8"))
    assert labels["preset"] == "distress_coarse.yaml"
    assert labels["frames"][0]["expect_num_detections"] == 0

    src = (_GOLDEN_ROOT / "generate_synthetic.py").read_text(encoding="utf-8")
    assert '"no_target" / "distress_coarse"' in src, "생성기에 이 리프가 없다(재생성 경로 단절)"
