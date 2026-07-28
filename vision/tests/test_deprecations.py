"""폐기된 모듈이 되살아나지 않는지 지키는 묘비(tombstone) 테스트.

폐기 결정은 문서에만 적어두면 반드시 되살아난다 — 특히 **다른 도메인 문서가 아직 그것을
확장하라고 적어두고 있을 때**는 더 그렇다. 여기서 코드로 못 박는다.
"""
from pathlib import Path

import pytest

_VISION_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# utils/geo_project.py — 픽셀 -> GPS 절대좌표 (2026-07-28 삭제)
#
# 폐기 사유(`docs/vision_plan.md` §12, §"측위", 루트 `CLAUDE.md` 의존관계 절):
# GPS 절대좌표 방식은 GPS 정확도 한계(~1~2m)로 30cm 요구에 미달한다. 30cm는 **카메라가
# 제어오차를 직접 측정하는 폐루프 비주얼 서보잉**으로만 달성되고, 그 대체 경로가 이제
# 실제로 존재한다 — `core/frames.py`(cam->body 프레임 체인) + `core/target.py::solve_target_pose`
# + `modules/distress_mat.py`(초록구역 상대 pose) + `utils/target_sink.py`(소켓 인터페이스).
#
# 🔴 **되살아날 위험이 실재한다:** `docs/pixhawk6c_rpi4_integration_guide.md`(FC 도메인 문서)가
# 아직 `pixel_to_gps_with_attitude()` 확장을 P2/P3 작업으로 적어두고 있다. 그 문서는 이번
# vision 세션의 소관이 아니라 손대지 않았다 — 대신 이 테스트가 코드 쪽 재발을 막는다.
# ---------------------------------------------------------------------------


def test_geo_project_module_stays_deleted():
    assert not (_VISION_ROOT / "utils" / "geo_project.py").exists(), (
        "vision/utils/geo_project.py 가 되살아났다. GPS 절대좌표 방식은 30cm 요구에 미달해 "
        "폐기됐다(docs/vision_plan.md §12) — 상대 pose 폐루프(core/frames.py + core/target.py + "
        "modules/distress_mat.py + utils/target_sink.py)를 쓸 것."
    )


def test_geo_project_is_not_importable():
    with pytest.raises(ModuleNotFoundError):
        __import__("vision.utils.geo_project")


def test_no_source_file_references_pixel_to_gps():
    """문자열/주석 수준의 부활(다른 파일에 같은 함수를 다시 심는 것)까지 막는다."""
    offenders = []
    for path in _VISION_ROOT.rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if "pixel_to_gps" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(_VISION_ROOT)))
    assert not offenders, f"폐기된 pixel_to_gps 가 다시 등장했다: {offenders}"
