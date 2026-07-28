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


# ---------------------------------------------------------------------------
# 등거리 축퇴 완화 (2026-07-28) — distress_box.py 클래스 docstring "등거리 축퇴와 그 완화"
#
# 실측 스펙상 흰 박스가 매트 정중앙이라 네 모서리가 등거리다. 1px 흔들림만으로 선택 모서리가
# 뒤집혀 착륙점이 매트 반대편(≈2.97m)으로 점프하던 것을 tie 허용오차 + 프레임간 히스테리시스로
# 잡는다. 이 착륙점은 `modules/distress_mat.py`를 거쳐 실제 유도 좌표로 나간다.
# ---------------------------------------------------------------------------

# 1px 흔들림 시퀀스(mp4 압축이 흰 박스 컨투어 중심을 흔드는 상황의 최소 모형)
_JITTER = [(0, 0), (1, 1), (0, 0), (-1, -1), (0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)]


def _draw_centered_box_jittered(canvas, bbox, dx, dy, side=20):
    x, y, w, h = bbox
    cx, cy = x + w // 2 + dx, y + h // 2 + dy
    half = side // 2
    cv2.rectangle(canvas, (cx - half, cy - half), (cx + half, cy + half), (255, 255, 255), -1)


def _landing_sequence(detector, jitter=_JITTER, bbox=_MAT_BBOX, draw=None):
    """흔들림 시퀀스를 한 detector 인스턴스에 순서대로 먹여 착륙점 열을 얻는다."""
    draw = draw or _draw_centered_box_jittered
    points = []
    for dx, dy in jitter:
        canvas = _canvas_with_mat(bbox)
        draw(canvas, bbox, dx, dy)
        state = VisionState(
            original=canvas.copy(), current=canvas.copy(), detections=[Detection(bbox=bbox)]
        )
        state = detector(state)
        assert len(state.detections) == 1, f"흔들림 ({dx},{dy})에서 확정 실패"
        points.append(tuple(state.detections[0].meta["white_box_detector"]["landing_point_px"]))
    return points


def _max_jump_px(points):
    return max(
        (float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(points, points[1:])), default=0.0
    )


def test_legacy_selection_jumps_across_the_mat_under_1px_jitter():
    """완화를 끈 옛 동작은 실제로 매트 반대편까지 점프한다 — 고친 대상이 무엇인지 못 박는 테스트.

    이게 red가 되면 축퇴 재현 자체가 사라진 것이므로 아래 완화 테스트의 의미도 재검토해야 한다.
    """
    legacy = WhiteBoxDetector(tie_tolerance_ratio=0.0, corner_hysteresis=False)
    points = _landing_sequence(legacy)
    # 매트가 300px(=3.0m)이므로 200px 초과 점프는 매트 반대편으로 건너뛴 것이다.
    assert _max_jump_px(points) > 200.0
    assert len(set(points)) > 1


def test_centered_box_landing_point_is_stable_under_1px_jitter():
    """기본값(완화 켜짐)에서는 같은 흔들림에도 착륙점이 전혀 움직이지 않아야 한다."""
    points = _landing_sequence(WhiteBoxDetector())
    assert len(set(points)) == 1, f"착륙점이 흔들렸다: {sorted(set(points))}"
    assert _max_jump_px(points) == 0.0


def test_landing_point_meta_reports_corner_and_tie_diagnostics():
    """축퇴 진단이 blackbox로 나가는지 — 정중앙 박스는 네 모서리 전부가 동률 후보여야 한다."""
    state = WhiteBoxDetector()(_state_with_mat(_draw_centered_box))
    meta = state.detections[0].meta["white_box_detector"]
    assert meta["landing_corner"] in ("tl", "tr", "br", "bl")
    assert meta["corner_tie_count"] == 4
    assert meta["corner_from_hysteresis"] is False  # 첫 프레임은 정규 순서로 결정


def test_offcenter_box_beyond_tie_band_still_picks_opposite_corner():
    """완화가 '박스 옆' 규약을 삼키지 않는지 — 확실히 편심된 박스는 여전히 반대편을 고른다."""
    detector = WhiteBoxDetector()
    state = detector(_state_with_mat(lambda c, b: _draw_offcenter_box(c, b, margin=25)))
    meta = state.detections[0].meta["white_box_detector"]
    assert meta["landing_corner"] == "br"      # 박스가 좌상단 -> 착륙점은 우하단
    assert meta["corner_tie_count"] == 1       # 동률 아님 = 완화가 개입하지 않았다


def test_offcenter_box_landing_point_also_stable_under_jitter():
    def draw(canvas, bbox, dx, dy):
        x, y, w, h = bbox
        cv2.rectangle(canvas, (x + 25 + dx, y + 25 + dy),
                      (x + 45 + dx, y + 45 + dy), (255, 255, 255), -1)

    points = _landing_sequence(WhiteBoxDetector(), draw=draw)
    assert len(set(points)) == 1


def test_tie_tolerance_zero_and_no_hysteresis_reproduces_legacy_exactly():
    """완화 파라미터를 끄면 옛 동작과 **정확히** 같아야 한다(파라미터가 실제로 살아 있다는 증거)."""
    off = WhiteBoxDetector(tie_tolerance_ratio=0.0, corner_hysteresis=False)
    on = WhiteBoxDetector()
    assert _landing_sequence(off) != _landing_sequence(on)


def test_hysteresis_is_a_schmitt_trigger_no_oscillation():
    """동률 허용오차 경계에 걸친 배치에서도 한 번 전환하면 눌러앉지, 왕복하지 않는다.

    `_select_far_corner`는 순수 함수라 이미지 렌더링 잡음 없이 슈미트 성질만 직접 검증한다.
    """
    x, y, w, h = _MAT_BBOX
    corners = [(float(x), float(y)), (float(x + w), float(y)),
               (float(x + w), float(y + h)), (float(x), float(y + h))]
    diag = float(np.hypot(w, h))
    mat_cx, mat_cy = x + w / 2.0, y + h / 2.0

    naive = WhiteBoxDetector(corner_hysteresis=False)

    def naive_choice(delta):
        # 박스를 TL 쪽(-delta)으로 밀면 실제 최원거리 모서리는 BR로 옮겨간다.
        return naive._select_far_corner(corners, mat_cx - delta, mat_cy - delta, diag, None)[0]

    # 전환 경계를 해석식으로 유도하지 않고 **실측 스캔으로 찾는다** — 경계를 정하는 경쟁 모서리가
    # 매트 종횡비에 따라 달라져(정사각 매트에서는 BR이 아니라 TR/BL이 먼저 동률에서 빠진다)
    # 손으로 유도한 식은 조용히 틀리기 쉽다.
    base = naive_choice(0.0)
    switch_delta = next(
        (d / 100.0 for d in range(1, 100 * int(diag)) if naive_choice(d / 100.0) != base), None
    )
    assert switch_delta is not None, "전환 경계를 못 찾았다 — 허용오차가 너무 크다"

    # 경계를 사이에 두고 오가는 시퀀스
    deltas = [switch_delta * 1.5, switch_delta * 0.8, switch_delta * 1.5,
              switch_delta * 0.8, switch_delta * 1.5, switch_delta * 0.8]

    naive_chosen = [naive_choice(d) for d in deltas]
    assert len(set(naive_chosen)) > 1, f"진동 대조군이 진동하지 않았다: {naive_chosen}"

    detector = WhiteBoxDetector()
    prev, chosen = None, []
    for delta in deltas:
        idx, _, _ = detector._select_far_corner(
            corners, mat_cx - delta, mat_cy - delta, diag, prev
        )
        chosen.append(idx)
        prev = idx
    # 첫 프레임에 확정된 뒤, 경계를 오가도 절대 다른 모서리로 넘어가지 않는다
    assert len(set(chosen)) == 1, f"경계에서 진동했다: {chosen}"


def test_hysteresis_state_is_per_instance_and_resettable():
    """상태가 인스턴스 단위라 새 인스턴스는 항상 같은 초기 선택에서 출발한다(§7.5 결정론)."""
    a = WhiteBoxDetector()
    b = WhiteBoxDetector()
    assert _landing_sequence(a) == _landing_sequence(b)

    # 히스테리시스 상태가 실제로 프레임 사이에 배선돼 있는지(안 쓰이면 슈미트 성질도 없다)
    assert a._prev_corner_choices, "프레임 처리 후에도 히스테리시스 상태가 비어 있다"
    assert a._prev_corner_choices[0][0] == _MAT_BBOX

    a.reset()
    assert a._prev_corner_choices == []
    assert _landing_sequence(a) == _landing_sequence(b)


def test_hysteresis_does_not_cross_contaminate_two_mats():
    """매트가 둘이면 IoU로 각자 짝을 찾아야 한다 — 순서로 섞이면 안 된다."""
    bbox1 = (20, 20, 300, 300)
    bbox2 = (380, 20, 300, 300)
    detector = WhiteBoxDetector()
    corners_seen = []
    for dx, dy in _JITTER:
        canvas = np.full((400, 700, 3), (60, 60, 60), dtype=np.uint8)
        for bbox in (bbox1, bbox2):
            x, y, w, h = bbox
            cv2.rectangle(canvas, (x, y), (x + w, y + h), _bgr_green(), -1)
        _draw_centered_box_jittered(canvas, bbox1, dx, dy)
        # 두 번째 매트는 박스를 좌상단에 편심 배치 -> 반대편(br)이 안정적으로 나와야 한다
        x2, y2, _, _ = bbox2
        cv2.rectangle(canvas, (x2 + 25 + dx, y2 + 25 + dy),
                      (x2 + 45 + dx, y2 + 45 + dy), (255, 255, 255), -1)
        state = detector(
            VisionState(original=canvas.copy(), current=canvas.copy(),
                        detections=[Detection(bbox=bbox1), Detection(bbox=bbox2)])
        )
        assert len(state.detections) == 2
        corners_seen.append(
            tuple(d.meta["white_box_detector"]["landing_corner"] for d in state.detections)
        )
    assert len(set(corners_seen)) == 1
    assert corners_seen[0][1] == "br"


def test_landing_point_meta_is_json_serializable():
    """blackbox JSONL로 나가므로 numpy 타입이 새면 안 된다."""
    import json

    state = WhiteBoxDetector()(_state_with_mat(_draw_centered_box))
    json.dumps(state.detections[0].meta["white_box_detector"])
