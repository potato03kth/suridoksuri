"""`vision/modules/distress_mat.py::DistressMatGeometry` 단위테스트 — 하드웨어 의존 없음.

이 모듈은 **검출을 더하거나 빼지 않는다** — `rect_detector`가 낸 초록 매트 후보에 solvePnP용
기하(정규화된 4코너 + 착륙점 픽셀 + 평면 기준)를 태깅할 뿐이다. 그래서 테스트도 두 축이다:
① 태깅 내용이 정확한가(코너 순서·착륙점 출처·물리 상수), ② 선언 필드 계약을 지키는가.

`vision/CLAUDE.md` "공통 규칙 (모든 모듈 테스트)" 4개(선언 필드 계약 / meta 네임스페이스 /
빈·경계 입력 / 결정론)를 전부 포함한다.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from vision.core.state import Detection, VisionState
from vision.core.target import DISTRESS_MAT_PLATFORM_HEIGHT_M, DISTRESS_MAT_SIZE_M
from vision.modules.distress_mat import DistressMatGeometry

# `modules/detector.py::RectDetector`가 골든셋 distress/10m에서 **실제로** 낸 코너 순서
# (반시계 TL->BL->BR->TR). 이 순서를 그대로 solvePnP에 넣으면 감김이 반대라 pose가 틀린다.
_OBSERVED_APPROXPOLYDP_CORNERS = np.array([[80, 80], [80, 380], [380, 380], [380, 80]])
_NORMALIZED = [[80.0, 80.0], [380.0, 80.0], [380.0, 380.0], [80.0, 380.0]]


def _state(detections) -> VisionState:
    img = np.full((460, 460, 3), 60, dtype=np.uint8)
    return VisionState(original=img.copy(), current=img.copy(), detections=list(detections))


def _mat_detection(corners=_OBSERVED_APPROXPOLYDP_CORNERS, white_box: dict | None = None) -> Detection:
    det = Detection(bbox=(80, 80, 301, 301), confidence=0.9,
                    corners=None if corners is None else np.asarray(corners))
    if white_box is not None:
        det.meta["white_box_detector"] = white_box
    return det


# ===========================================================================
# 코너 순서 정규화 — 조용한 오작동 1순위
# ===========================================================================


def test_corners_are_normalized_to_the_solvepnp_order():
    """🔴 핵심: approxPolyDP의 반시계 순서를 `marker_object_points()`와 대응되는
    시계방향·top-left 시작으로 정규화해 실어야 한다. 원본 순서를 그대로 실으면 pose가 틀린다."""
    state = _state([_mat_detection()])
    out = DistressMatGeometry()(state)

    mat = out.detections[0].meta["distress_mat"]
    assert mat["corners_px"] == _NORMALIZED
    assert mat["corners_px"] != _OBSERVED_APPROXPOLYDP_CORNERS.tolist(), (
        "정규화가 사실상 항등이 됐다 — 이 테스트가 코너 순서 회귀를 못 잡는다"
    )


def test_detection_without_four_corners_is_skipped_with_a_specific_reason():
    """`RectDetector`는 approx가 4점이 아니면 `corners=None`을 남긴다 — 그건 pose를 못 푼다는
    뜻이지 "매트가 없다"는 뜻이 아니므로 사유를 구분해 남긴다(§5.4 사유 뭉개기 금지)."""
    state = _state([_mat_detection(corners=None)])
    out = DistressMatGeometry()(state)

    assert "distress_mat" not in out.detections[0].meta
    assert out.meta["distress_mat_geometry"] == {
        "tagged": 0, "skipped": 1, "skip_reasons": ["no_quad_corners"],
    }


def test_degenerate_quad_is_skipped_with_its_own_reason():
    state = _state([_mat_detection(corners=np.array([[10, 10], [10, 10], [10, 10], [10, 10]]))])
    out = DistressMatGeometry()(state)

    assert "distress_mat" not in out.detections[0].meta
    assert out.meta["distress_mat_geometry"]["skip_reasons"] == ["degenerate_quad"]


def test_five_point_polygon_is_skipped_not_silently_truncated():
    pentagon = np.array([[80, 80], [380, 80], [380, 380], [230, 420], [80, 380]])
    out = DistressMatGeometry()(_state([_mat_detection(corners=pentagon)]))
    assert "distress_mat" not in out.detections[0].meta
    assert out.meta["distress_mat_geometry"]["skip_reasons"] == ["no_quad_corners"]


# ===========================================================================
# 착륙점 — 매트 중심이 아니다(§5.3), 그리고 규칙을 재구현하지 않는다
# ===========================================================================


def test_coarse_without_white_box_degrades_to_mat_centre_and_says_so():
    """coarse 대역(40~15m)에는 흰 박스 단계가 없다 — 매트 중심으로 degrade하되 그 사실을
    `landing_point_source`로 밝혀야 한다(소비자가 "최종 착륙점"으로 오인하면 안 됨)."""
    out = DistressMatGeometry()(_state([_mat_detection()]))
    mat = out.detections[0].meta["distress_mat"]

    assert mat["landing_point_source"] == "mat_center"
    assert mat["landing_point_px"] == pytest.approx([230.0, 230.0])


def test_fine_consumes_white_box_landing_point_verbatim():
    """🔴 §5.3 기존 규약 준수: 착륙점 규칙("박스에서 가장 먼 매트 모서리를 안쪽으로 당김")은
    `modules/distress_box.py`가 이미 확정한다. 이 모듈은 그걸 **다시 구현하지 않고 그대로
    소비**해야 한다 — 규칙이 두 곳에 복사되면 한쪽만 고쳐졌을 때 조용히 어긋난다."""
    wb = {"box_center_px": [230.5, 230.5], "landing_point_px": [125.15, 125.15]}
    out = DistressMatGeometry()(_state([_mat_detection(white_box=wb)]))
    mat = out.detections[0].meta["distress_mat"]

    assert mat["landing_point_source"] == "white_box_landing_point"
    assert mat["landing_point_px"] == [125.15, 125.15]
    # 매트 중심으로 되돌아가지 않았다는 것을 명시적으로 고정한다.
    assert mat["landing_point_px"] != pytest.approx([230.0, 230.0])


def test_white_box_meta_without_landing_point_falls_back_to_mat_centre():
    """흰 박스 meta가 있어도 `landing_point_px`가 없으면(거절된 후보 등) 중심으로 degrade —
    KeyError로 파이프라인을 죽이지 않는다."""
    out = DistressMatGeometry()(_state([_mat_detection(white_box={"box_center_px": [1.0, 2.0]})]))
    assert out.detections[0].meta["distress_mat"]["landing_point_source"] == "mat_center"


# ===========================================================================
# 물리 상수 — 매직넘버 금지(§7.3) + provenance
# ===========================================================================


def test_physical_constants_default_to_the_measured_spec():
    mat = DistressMatGeometry()(_state([_mat_detection()])).detections[0].meta["distress_mat"]
    assert mat["size_m"] == DISTRESS_MAT_SIZE_M == 3.0
    assert mat["platform_height_m"] == DISTRESS_MAT_PLATFORM_HEIGHT_M == 0.105


def test_constructor_values_actually_reach_the_meta_not_a_hardcoded_constant():
    """preset yaml로 조정 가능해야 한다(매직넘버 금지) — 하드코딩이면 이 테스트가 red."""
    mat = DistressMatGeometry(size_m=2.0, platform_height_m=0.25)(
        _state([_mat_detection()])
    ).detections[0].meta["distress_mat"]
    assert mat["size_m"] == 2.0
    assert mat["platform_height_m"] == 0.25


def test_plane_reference_is_the_mat_top_surface_not_the_ground():
    """0.105m 라이즈드 구조물이라 z=0 평면이 **매트 윗면**이다(지면 아님). 소비자가 라이다
    AGL과 섞으면 정확히 platform_height_m 만큼 어긋나므로 그 기준을 명시해 전파한다."""
    mat = DistressMatGeometry()(_state([_mat_detection()])).detections[0].meta["distress_mat"]
    assert mat["plane_reference"] == "mat_top_surface"


# ===========================================================================
# 공통 규칙 — 선언 필드 계약 / meta 네임스페이스 / 빈 입력 / 결정론
# ===========================================================================


def test_declared_field_contract_only_touches_detections_and_meta():
    state = _state([_mat_detection()])
    state.mask = np.zeros((460, 460), dtype=np.uint8)
    original_before = state.original.copy()
    current_before = state.current.copy()
    mask_before = state.mask.copy()
    n_before = len(state.detections)

    out = DistressMatGeometry()(state)

    assert np.array_equal(out.original, original_before)
    assert np.array_equal(out.current, current_before)
    assert np.array_equal(out.mask, mask_before)
    assert out.confirmed is None
    assert len(out.detections) == n_before, "이 모듈은 검출을 더하거나 빼지 않는다"


def test_meta_namespace_is_the_module_name():
    out = DistressMatGeometry()(_state([_mat_detection()]))
    assert set(out.meta) == {"distress_mat_geometry"}
    assert set(out.meta["distress_mat_geometry"]) == {"tagged", "skipped", "skip_reasons"}


def test_empty_detections_is_safe_and_still_records_meta():
    out = DistressMatGeometry()(_state([]))
    assert out.detections == []
    assert out.meta["distress_mat_geometry"] == {"tagged": 0, "skipped": 0, "skip_reasons": []}


def test_mixed_detections_tag_only_the_usable_ones():
    out = DistressMatGeometry()(_state([_mat_detection(), _mat_detection(corners=None)]))
    assert "distress_mat" in out.detections[0].meta
    assert "distress_mat" not in out.detections[1].meta
    assert out.meta["distress_mat_geometry"]["tagged"] == 1
    assert out.meta["distress_mat_geometry"]["skipped"] == 1


def test_deterministic_for_the_same_input():
    module = DistressMatGeometry()
    runs = []
    for _ in range(3):
        wb = {"landing_point_px": [125.15, 125.15]}
        runs.append(copy.deepcopy(
            module(_state([_mat_detection(white_box=wb)])).detections[0].meta["distress_mat"]
        ))
    assert runs[0] == runs[1] == runs[2]


def test_meta_is_json_serializable_plain_python():
    """와이어 레코드(`core/wire.py`)와 JSONL 블랙박스로 그대로 흘러가므로 numpy 스칼라가
    새어 나가면 `json.dumps`가 터진다."""
    import json

    out = DistressMatGeometry()(_state([_mat_detection()]))
    json.dumps(out.detections[0].meta["distress_mat"])
    json.dumps(out.meta["distress_mat_geometry"])
