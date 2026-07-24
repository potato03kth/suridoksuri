"""WhiteBoxDetector 단위테스트 (vision/modules/distress_box.py, vision_plan.md §5.3 fine 단계).

`vision/CLAUDE.md` "공통 규칙(모든 모듈 테스트)" 4개를 담는다:
1. 선언 필드 계약 — original/detections만 읽고 detections/meta만 쓴다.
2. meta 네임스페이스 — state.meta["white_box_detector"].
3. 빈/경계 입력 — 빈 detections, 박스 없음, 너무 크거나/작거나/길쭉한 박스에서 크래시 없이
   합리적 거절.
4. 결정론 — 같은 입력 → 같은 출력(meta 전체 비교).

착륙점(landing_point_px)이 박스 반대편(가장 먼 매트 모서리 쪽으로 당겨진 지점)을 가리키는지,
그리고 항상 매트 bbox 내부에 있는지도 검증한다(§5.3 "박스 옆 빈 초록면" 설계 요건).
"""
import cv2
import numpy as np
from vision.core.state import VisionState, Detection
from vision.modules.distress_box import WhiteBoxDetector

_MAT_BBOX = (50, 50, 300, 300)


def _bgr_green() -> tuple:
    hsv_green = np.array([[[60, 200, 180]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(hsv_green, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _canvas_with_mat(bbox=_MAT_BBOX, canvas_size=(400, 400)) -> np.ndarray:
    canvas = np.full((canvas_size[0], canvas_size[1], 3), (60, 60, 60), dtype=np.uint8)
    x, y, w, h = bbox
    cv2.rectangle(canvas, (x, y), (x + w, y + h), _bgr_green(), -1)
    return canvas


def _state_with_mat(box_fn=None, bbox=_MAT_BBOX) -> VisionState:
    canvas = _canvas_with_mat(bbox)
    if box_fn is not None:
        box_fn(canvas, bbox)
    return VisionState(original=canvas.copy(), current=canvas.copy(), detections=[Detection(bbox=bbox)])


def _draw_centered_box(canvas, bbox, side=20):
    """실측 스펙 비율(20cm/3.0m ≈ 0.0667 선형, 면적비 ≈ 0.00444) — 매트 정중앙."""
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    half = side // 2
    cv2.rectangle(canvas, (cx - half, cy - half), (cx + half, cy + half), (255, 255, 255), -1)


def _draw_offcenter_box(canvas, bbox, side=20, margin=25):
    """박스를 매트 좌상단 쪽에 배치 — 착륙점이 반대편(우하단)으로 밀려나는지 검증용."""
    x, y, w, h = bbox
    bx0, by0 = x + margin, y + margin
    cv2.rectangle(canvas, (bx0, by0), (bx0 + side, by0 + side), (255, 255, 255), -1)


def _draw_huge_box(canvas, bbox):
    """매트 대부분을 채우는 흰 블롭 — area_ratio 상한 초과로 거절돼야 한다."""
    x, y, w, h = bbox
    margin = 5
    cv2.rectangle(canvas, (x + margin, y + margin), (x + w - margin, y + h - margin), (255, 255, 255), -1)


def _draw_tiny_speck(canvas, bbox):
    """area_ratio 하한 미만 — 거절돼야 한다."""
    x, y, w, h = bbox
    cv2.rectangle(canvas, (x + 5, y + 5), (x + 5 + 8, y + 5 + 8), (255, 255, 255), -1)


def _draw_elongated_box(canvas, bbox):
    """세로로 매우 긴 흰 막대 — 면적비는 통과해도 종횡비 초과로 거절돼야 한다."""
    x, y, w, h = bbox
    cx = x + w // 2
    cv2.rectangle(canvas, (cx - 3, y + 20), (cx + 3, y + h - 20), (255, 255, 255), -1)


# ---------------------------------------------------------------------------
# 정상 확인 + 착륙점
# ---------------------------------------------------------------------------


def test_confirms_white_box_in_mat():
    state = WhiteBoxDetector()(_state_with_mat(_draw_centered_box))
    assert len(state.detections) == 1


def test_confirmed_detection_has_landing_point_meta():
    state = WhiteBoxDetector()(_state_with_mat(_draw_centered_box))
    meta = state.detections[0].meta["white_box_detector"]
    assert "landing_point_px" in meta
    assert "box_bbox" in meta and "box_center_px" in meta
    assert "area_ratio" in meta and "solidity" in meta


def test_landing_point_stays_within_mat_bbox():
    state = WhiteBoxDetector()(_state_with_mat(_draw_centered_box))
    x, y, w, h = _MAT_BBOX
    lx, ly = state.detections[0].meta["white_box_detector"]["landing_point_px"]
    assert x <= lx <= x + w
    assert y <= ly <= y + h


def test_landing_point_pushed_away_from_offcenter_box():
    """박스가 매트 좌상단 쪽에 있으면 착륙점은 반대편(우하단, 중심보다 x/y 모두 큼)으로
    밀려나야 한다 — §5.3 "박스 옆 빈 초록면" 요건의 최소 검증."""
    state = WhiteBoxDetector()(_state_with_mat(lambda c, b: _draw_offcenter_box(c, b, margin=25)))
    assert len(state.detections) == 1
    x, y, w, h = _MAT_BBOX
    mat_cx, mat_cy = x + w / 2.0, y + h / 2.0
    lx, ly = state.detections[0].meta["white_box_detector"]["landing_point_px"]
    assert lx > mat_cx
    assert ly > mat_cy
    # 여전히 매트 내부
    assert x <= lx <= x + w
    assert y <= ly <= y + h


def test_meta_recorded_confirmed_case():
    state = WhiteBoxDetector()(_state_with_mat(_draw_centered_box))
    meta = state.meta["white_box_detector"]
    assert meta["confirmed"] == 1
    assert meta["rejected"] == 0
    assert meta["reject_reasons"] == []


# ---------------------------------------------------------------------------
# 거절 케이스
# ---------------------------------------------------------------------------


def test_rejects_mat_with_no_white_box():
    state = WhiteBoxDetector()(_state_with_mat(box_fn=None))
    assert len(state.detections) == 0
    meta = state.meta["white_box_detector"]
    assert meta["confirmed"] == 0
    assert meta["rejected"] == 1
    assert "no_white_pixels" in meta["reject_reasons"]


def test_rejects_box_too_large():
    state = WhiteBoxDetector()(_state_with_mat(_draw_huge_box))
    assert len(state.detections) == 0
    assert state.meta["white_box_detector"]["rejected"] == 1


def test_rejects_box_too_small():
    state = WhiteBoxDetector()(_state_with_mat(_draw_tiny_speck))
    assert len(state.detections) == 0
    assert state.meta["white_box_detector"]["rejected"] == 1


def test_rejects_elongated_non_square_box():
    state = WhiteBoxDetector()(_state_with_mat(_draw_elongated_box))
    assert len(state.detections) == 0
    assert state.meta["white_box_detector"]["rejected"] == 1


def test_two_candidates_one_confirmed_one_rejected():
    bbox1 = (20, 20, 300, 300)
    bbox2 = (380, 20, 300, 300)
    canvas = np.full((400, 700, 3), (60, 60, 60), dtype=np.uint8)
    for bbox in (bbox1, bbox2):
        x, y, w, h = bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), _bgr_green(), -1)
    _draw_centered_box(canvas, bbox1)
    # bbox2에는 박스를 그리지 않음 -> 거절 대상

    state = VisionState(
        original=canvas.copy(), current=canvas.copy(),
        detections=[Detection(bbox=bbox1), Detection(bbox=bbox2)],
    )
    result = WhiteBoxDetector()(state)

    assert len(result.detections) == 1
    assert result.detections[0].bbox == bbox1
    meta = result.meta["white_box_detector"]
    assert meta["confirmed"] == 1
    assert meta["rejected"] == 1
    assert "no_white_pixels" in meta["reject_reasons"]


# ---------------------------------------------------------------------------
# 빈/경계 입력
# ---------------------------------------------------------------------------


def test_empty_detections_input_no_crash():
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    state = WhiteBoxDetector()(VisionState(original=img, current=img.copy(), detections=[]))
    assert state.detections == []
    assert state.meta["white_box_detector"] == {"confirmed": 0, "rejected": 0, "reject_reasons": []}


def test_zero_size_bbox_no_crash():
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    state = WhiteBoxDetector()(
        VisionState(original=img, current=img.copy(), detections=[Detection(bbox=(10, 10, 0, 0))])
    )
    assert state.detections == []
    assert "empty_mat_bbox" in state.meta["white_box_detector"]["reject_reasons"]


# ---------------------------------------------------------------------------
# 선언 필드 계약 + 결정론
# ---------------------------------------------------------------------------


def test_does_not_mutate_original_current_or_mask():
    state_in = _state_with_mat(_draw_centered_box)
    original_before = state_in.original.copy()
    current_before = state_in.current.copy()
    state_out = WhiteBoxDetector()(state_in)
    assert np.array_equal(state_out.original, original_before)
    assert np.array_equal(state_out.current, current_before)
    assert state_out.mask is None


def test_deterministic():
    d1 = WhiteBoxDetector()(_state_with_mat(_draw_centered_box))
    d2 = WhiteBoxDetector()(_state_with_mat(_draw_centered_box))
    assert [d.meta["white_box_detector"] for d in d1.detections] == [
        d.meta["white_box_detector"] for d in d2.detections
    ]
    assert d1.meta["white_box_detector"] == d2.meta["white_box_detector"]
