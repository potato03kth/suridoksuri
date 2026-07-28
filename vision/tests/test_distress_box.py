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
    # 매트가 300px(=3.0m)이므로 100px(=1.0m) 초과 점프는 매트 반대편으로 건너뛴 것이다.
    # (2026-07-28 기체 크기 기반 재설계로 착륙점이 중심에 더 가까워져 대각 점프 자체가
    #  2·d·√2 = 1.59m ≈ 159px로 줄었다 — 옛 임계 200px는 그 변화만으로 red가 됐었다.)
    assert _max_jump_px(points) > 100.0
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


# ===========================================================================
# 착륙점 기하 — 기체 크기 기반 재설계 (2026-07-28)
#
# 예전 착륙점 거리는 `interior_margin_ratio`(무차원 잠정값 0.3) 하나로 정해졌고 **기체 크기를
# 한 번도 고려하지 않았다**. 기체 최외곽 반경이 0.5m를 초과함이 확정되며 구조적 결함이 드러났다
# (중심에서 1.05m -> 매트 가장자리 여유 0.45m < R). 이제 거리 d를 물리량에서 도출한다:
#     하한(박스 회피, **대각선**으로 잰다)     : √2·d − b·√2 ≥ R  ⟺  d ≥ b + R/√2
#     상한(매트 이탈 회피, **축 방향**으로 잰다): M − d ≥ R + δ    ⟺  d ≤ M − R − δ
# 유도·전제는 `modules/distress_box.py::compute_landing_window` docstring.
# ===========================================================================

import math

import pytest

from vision.modules.distress_box import (
    AIRCRAFT_RADIUS_M_DEFAULT,
    AIRCRAFT_RADIUS_MEASURED,
    DISTRESS_WHITE_BOX_SIZE_M,
    LANDING_DRIFT_ALLOWANCE_M_DEFAULT,
    compute_landing_window,
)

_SQRT2 = math.sqrt(2.0)
_MAT_SIZE_M = 3.0          # 실측 확정 스펙 (vision_plan.md §2)


def _landing_distance_m(detector, mat_size_m=_MAT_SIZE_M, bbox=_MAT_BBOX):
    """**렌더링된 실제 파이프라인**으로 착륙점을 얻어 매트 중심 기준 축당 거리(m)로 환산한다.

    순수 함수(`compute_landing_window`)만 검증하면 "계산은 맞는데 착륙점에 안 쓰인다"는 배선
    결함을 통째로 놓친다 — 그래서 실제 `__call__` 산출물을 잰다.
    """
    state = detector(_state_with_mat(_draw_centered_box, bbox=bbox))
    assert len(state.detections) == 1, "안전창이 있는 설정인데 확정되지 않았다"
    x, y, w, h = bbox
    lx, ly = state.detections[0].meta["white_box_detector"]["landing_point_px"]
    px_per_m_x, px_per_m_y = w / mat_size_m, h / mat_size_m
    dx = abs(lx - (x + w / 2.0)) / px_per_m_x
    dy = abs(ly - (y + h / 2.0)) / px_per_m_y
    assert dx == pytest.approx(dy, abs=1e-6), "착륙점이 대각선 위에 있지 않다"
    return dx


@pytest.mark.parametrize("radius_m", [0.3, 0.5, 0.6])
def test_landing_point_satisfies_both_safety_inequalities(radius_m):
    """★핵심: 산출된 착륙점이 실제로 두 부등식을 **동시에** 만족하는가.

    하한을 대각선으로, 상한을 축 방향으로 재는 것이 이 테스트의 전부다 — 둘을 바꿔 재면
    (또는 √2를 빠뜨리면) 한쪽이 반드시 깨진다.
    """
    d = _landing_distance_m(WhiteBoxDetector(aircraft_radius_m=radius_m))
    half_box = DISTRESS_WHITE_BOX_SIZE_M / 2.0
    half_mat = _MAT_SIZE_M / 2.0

    # (1) 박스를 밟지 않는다 — 대각선 거리
    assert _SQRT2 * d >= half_box * _SQRT2 + radius_m - 1e-9, (
        f"R={radius_m}: 착륙점 d={d:.4f}m에서 기체가 흰 박스를 밟는다"
    )
    # (2) 매트를 벗어나지 않는다 — 축 방향 최단거리
    assert d <= half_mat - radius_m - LANDING_DRIFT_ALLOWANCE_M_DEFAULT + 1e-9, (
        f"R={radius_m}: 착륙점 d={d:.4f}m에서 기체가 0.105m 라이즈드 플랫폼 밖으로 나간다"
    )


def test_the_old_ratio_rule_would_have_overhung_the_mat():
    """고친 대상이 무엇인지 못 박는 테스트 — 옛 규칙(interior_margin_ratio=0.3)은
    **기체 반경의 알려진 하한(0.5m)에서조차** 매트를 넘는다. red가 되면 결함 재현이 사라진 것."""
    old_d = (1.0 - 0.3) * (_MAT_SIZE_M / 2.0)        # = 1.05 m
    known_radius_lower_bound = 0.5                    # 사용자 확정: "0.5m 초과"
    available = _MAT_SIZE_M / 2.0 - old_d             # = 0.45 m (착륙점 -> 매트 가장자리)

    # 유도 오차를 **0으로 쳐도** 기체 반경조차 못 담는다 — 완벽히 유도해도 삐져나간다.
    assert available < known_radius_lower_bound, (
        f"옛 착륙점이 매트를 안 넘는다(여유 {available:.3f}m) — 결함 재현이 깨졌다"
    )
    # 허용 드리프트까지 얹으면 격차가 더 벌어진다.
    assert available < known_radius_lower_bound + LANDING_DRIFT_ALLOWANCE_M_DEFAULT


@pytest.mark.parametrize(
    "radius_m, expect_feasible, expect_d_min, expect_d_max",
    [
        # 손계산: d_min = 0.10 + R/√2, d_max = 1.5 − R − 0.30 (M=1.5, b=0.10, δ=0.30)
        (0.3, True, 0.312132, 0.900000),
        (0.5, True, 0.453553, 0.700000),
        (0.7, False, 0.594975, 0.500000),
        (1.0, False, 0.807107, 0.200000),
    ],
)
def test_safety_window_bounds_are_the_hand_computed_values(
    radius_m, expect_feasible, expect_d_min, expect_d_max
):
    """수치 실증 표를 회귀로 고정한다 — 부등식 어느 한쪽이라도 형태가 바뀌면 red."""
    w = compute_landing_window(_MAT_SIZE_M, DISTRESS_WHITE_BOX_SIZE_M, radius_m,
                               LANDING_DRIFT_ALLOWANCE_M_DEFAULT, 0.5)
    assert w["d_min_m"] == pytest.approx(expect_d_min, abs=1e-6)
    assert w["d_max_m"] == pytest.approx(expect_d_max, abs=1e-6)
    assert w["feasible"] is expect_feasible
    # 절벽 위치(안전창이 존재하는 R의 상한)는 R과 무관한 상수다.
    assert w["aircraft_radius_max_feasible_m"] == pytest.approx(0.644365, abs=1e-6)


def test_empty_safety_window_rejects_instead_of_inventing_a_landing_point():
    """🔴 R이 커서 해가 없으면 **착륙점을 지어내지 않는다** — 검출을 거절하고 사유를 남긴다."""
    detector = WhiteBoxDetector(aircraft_radius_m=1.0)
    state = detector(_state_with_mat(_draw_centered_box))

    assert state.detections == [], "안전창이 비었는데 착륙점이 나왔다"
    meta = state.meta["white_box_detector"]
    assert meta["confirmed"] == 0
    assert meta["rejected"] == 1
    assert "landing_point_infeasible" in meta["reject_reasons"]
    # 사유가 "박스를 못 찾음"과 구분돼야 한다(§5.4 사유 뭉개기 금지).
    assert "no_white_pixels" not in meta["reject_reasons"]

    # 계산된 하한/상한/R이 사람이 볼 수 있게 남는다 — 안전창이 비었다는 것은 설계 정보다.
    detail = meta["landing_point_infeasible"]
    assert detail["feasible"] is False
    assert detail["d_min_m"] > detail["d_max_m"]
    assert detail["aircraft_radius_m"] == 1.0
    assert detail["aircraft_radius_measured"] is False
    assert detail["d_m"] is None
    import json

    json.dumps(meta)  # 상위가 그대로 실어 보낼 수 있어야 한다


def test_feasible_run_leaves_state_meta_shape_untouched():
    """정상 경로의 `state.meta` 형태는 안 바뀐다 — 골든셋 labels.json이 통째로 동등비교한다."""
    state = WhiteBoxDetector()(_state_with_mat(_draw_centered_box))
    assert state.meta["white_box_detector"] == {
        "confirmed": 1, "rejected": 0, "reject_reasons": [],
    }


def test_infeasible_does_not_silently_degrade_to_the_mat_centre():
    """🔴 거절이 **필수**인 이유: 착륙점만 빼고 검출을 남기면 `DistressMatGeometry`가
    매트 중심(=흰 박스 위)으로 우아하게 degrade한다 — "해가 없다"가 "박스 위에 내려라"로
    조용히 뒤집힌다. 실제 하위 모듈을 통과시켜 그 경로가 안 열리는지 확인한다."""
    from vision.modules.distress_mat import DistressMatGeometry

    x, y, w, h = _MAT_BBOX
    canvas = _canvas_with_mat(_MAT_BBOX)
    _draw_centered_box(canvas, _MAT_BBOX)
    det = Detection(
        bbox=_MAT_BBOX,
        corners=[(x, y), (x, y + h), (x + w, y + h), (x + w, y)],
    )
    state = VisionState(original=canvas.copy(), current=canvas.copy(), detections=[det])

    state = WhiteBoxDetector(aircraft_radius_m=1.0)(state)
    state = DistressMatGeometry()(state)

    assert state.meta["distress_mat_geometry"]["tagged"] == 0
    assert all("distress_mat" not in d.meta for d in state.detections)


def test_shipped_defaults_produce_the_documented_landing_distance():
    """🔴 **기본값 자체를 밟는 테스트.** 인자를 전부 명시로 넘기는 테스트만 있으면 기본값을
    뒤집어도 전건 통과한다(이 저장소에서 같은 사고가 두 번 났다). 그래서 여기서는 인자를
    하나도 주지 않고, 기대값은 상수를 재참조하지 않은 **손계산 리터럴**로 적는다."""
    assert AIRCRAFT_RADIUS_M_DEFAULT == 0.60
    assert LANDING_DRIFT_ALLOWANCE_M_DEFAULT == 0.30
    assert DISTRESS_WHITE_BOX_SIZE_M == 0.20

    detector = WhiteBoxDetector()          # ← 인자 0개
    assert detector.mat_size_m == 3.0      # core/target.py::DISTRESS_MAT_SIZE_M 경유
    assert detector.safety_window_bias == 0.5

    # d = 0.10 + 0.60/√2 = 0.524264 (하한), 1.5 − 0.60 − 0.30 = 0.600000 (상한), 중점 0.562132
    assert _landing_distance_m(detector) == pytest.approx(0.562132, abs=1e-4)


def test_aircraft_radius_default_is_flagged_unmeasured_and_above_the_known_bound():
    """🔴 R은 실측된 적이 없다(`core/frames.py`의 ψ_m 패턴). 실측 없이 이 플래그를 뒤집으면 red.

    아는 사실은 "0.5m 초과"뿐이므로 기본값은 그 하한보다 커야 한다 — 작게 잡으면 매트를 넘는다.
    """
    assert AIRCRAFT_RADIUS_MEASURED is False
    assert AIRCRAFT_RADIUS_M_DEFAULT > 0.5
    # 미측정 플래그가 진단 meta까지 전파되는지(소비자가 이 추정을 얼마나 믿을지 정하는 근거)
    state = WhiteBoxDetector()(_state_with_mat(_draw_centered_box))
    geom = state.detections[0].meta["white_box_detector"]["landing_geometry"]
    assert geom["aircraft_radius_measured"] is False
    assert geom["aircraft_radius_m"] == AIRCRAFT_RADIUS_M_DEFAULT


def test_landing_geometry_meta_carries_bounds_slack_and_is_json_serializable():
    """§7.4 포렌식: 착륙점이 "왜 거기인가"의 근거 전량이 검출 meta에 실린다."""
    import json

    state = WhiteBoxDetector()(_state_with_mat(_draw_centered_box))
    geom = state.detections[0].meta["white_box_detector"]["landing_geometry"]
    assert geom["feasible"] is True
    assert geom["d_min_m"] <= geom["d_m"] <= geom["d_max_m"]
    assert geom["box_clearance_m"] >= 0.0
    assert geom["mat_edge_margin_m"] >= LANDING_DRIFT_ALLOWANCE_M_DEFAULT - 1e-9
    assert geom["aircraft_radius_max_feasible_m"] > geom["aircraft_radius_m"]
    json.dumps(state.detections[0].meta["white_box_detector"])


def test_box_center_offset_is_exported_so_the_centred_box_premise_can_be_checked():
    """하한 유도는 "박스가 매트 정중앙"(§2)을 전제한다 — 전제가 깨졌는지 사후에 보여야 한다."""
    centred = WhiteBoxDetector()(_state_with_mat(_draw_centered_box))
    off = WhiteBoxDetector()(_state_with_mat(lambda c, b: _draw_offcenter_box(c, b, margin=25)))

    ox, oy = centred.detections[0].meta["white_box_detector"]["landing_geometry"][
        "box_center_offset_m"]
    assert abs(ox) < 0.05 and abs(oy) < 0.05

    ox2, oy2 = off.detections[0].meta["white_box_detector"]["landing_geometry"][
        "box_center_offset_m"]
    assert ox2 < -0.5 and oy2 < -0.5, "편심 박스인데 오프셋이 0 근처로 보고됐다"


def test_geometry_parameters_are_live_not_hardcoded():
    """매트/박스/드리프트/bias 각각이 실제로 착륙점을 움직이는가(하드코딩이면 red)."""
    base = _landing_distance_m(WhiteBoxDetector())

    # 매트가 커지면 안전창 상한이 멀어져 착륙점도 멀어진다
    bigger_mat = _landing_distance_m(WhiteBoxDetector(mat_size_m=6.0), mat_size_m=6.0)
    assert bigger_mat > base + 0.5

    # 드리프트 허용치를 줄이면 상한이 늘어 착륙점이 바깥으로 간다
    assert _landing_distance_m(WhiteBoxDetector(drift_allowance_m=0.0)) > base

    # 박스가 커지면 하한이 밀려 착륙점이 바깥으로 간다
    # (0.30m까지만 키운다 — 0.40m를 넘기면 하한이 상한을 추월해 안전창 자체가 비어버린다)
    assert _landing_distance_m(WhiteBoxDetector(white_box_size_m=0.30)) > base

    # bias=0 이면 하한, bias=1 이면 상한에 정확히 붙는다
    assert _landing_distance_m(WhiteBoxDetector(safety_window_bias=0.0)) == pytest.approx(
        0.524264, abs=1e-4)
    assert _landing_distance_m(WhiteBoxDetector(safety_window_bias=1.0)) == pytest.approx(
        0.600000, abs=1e-4)


def test_constructor_rejects_impossible_geometry_arguments():
    """인자 검증은 생성자에서 — 프레임을 만지기 전에 실패한다(`LiveFrameSource` AF와 같은 원칙)."""
    for kwargs in (
        {"mat_size_m": 0.0},
        {"mat_size_m": -3.0},
        {"white_box_size_m": -0.1},
        {"aircraft_radius_m": -0.1},
        {"drift_allowance_m": -0.1},
        {"safety_window_bias": -0.01},
        {"safety_window_bias": 1.01},
    ):
        with pytest.raises(ValueError):
            WhiteBoxDetector(**kwargs)


# ---------------------------------------------------------------------------
# interior_margin_ratio 폐기 (하위호환 no-op)
# ---------------------------------------------------------------------------


def test_interior_margin_ratio_is_accepted_but_ignored():
    """🔴 물리 도출값이 **항상** 이긴다. 둘 다 조용히 적용되면 어느 쪽이 이겼는지 모른 채
    같은 결함이 재발한다. 기존 preset yaml이 이 키를 줘도 로드가 죽지는 않아야 한다."""
    with pytest.warns(DeprecationWarning):
        legacy_kwarg = WhiteBoxDetector(interior_margin_ratio=0.3)

    # 옛 값(0.3)을 주든 말도 안 되는 값(0.95)을 주든 착륙점은 물리 도출값 그대로다.
    assert _landing_distance_m(legacy_kwarg) == pytest.approx(
        _landing_distance_m(WhiteBoxDetector()), abs=1e-9)
    with pytest.warns(DeprecationWarning):
        absurd = WhiteBoxDetector(interior_margin_ratio=0.95)
    assert _landing_distance_m(absurd) == pytest.approx(
        _landing_distance_m(WhiteBoxDetector()), abs=1e-9)


def test_ignoring_interior_margin_ratio_is_announced_not_silent():
    """§7.4 침묵 금지 — 무시했다는 사실이 meta로 나가야 조용한 오설정을 잡을 수 있다."""
    with pytest.warns(DeprecationWarning):
        detector = WhiteBoxDetector(interior_margin_ratio=0.3)
    state = detector(_state_with_mat(_draw_centered_box))
    assert state.meta["white_box_detector"]["interior_margin_ratio_ignored"] == 0.3

    # 안 주면 이 키 자체가 없다(정상 경로의 meta 형태 불변)
    clean = WhiteBoxDetector()(_state_with_mat(_draw_centered_box))
    assert "interior_margin_ratio_ignored" not in clean.meta["white_box_detector"]


def test_distress_fine_preset_values_match_the_module_defaults():
    """preset yaml이 값을 복제하므로 조용한 드리프트를 여기서 막는다
    (`test_state_machine.py`가 nominal.yaml과 모듈 상수를 대조하는 것과 같은 패턴)."""
    from pathlib import Path

    import yaml

    import vision

    preset = Path(vision.__file__).parent / "presets" / "distress_fine.yaml"
    cfg = yaml.safe_load(preset.read_text(encoding="utf-8"))["pipeline"]["white_box_detector"]

    assert cfg["aircraft_radius_m"] == AIRCRAFT_RADIUS_M_DEFAULT
    assert cfg["drift_allowance_m"] == LANDING_DRIFT_ALLOWANCE_M_DEFAULT
    assert cfg["white_box_size_m"] == DISTRESS_WHITE_BOX_SIZE_M
    assert cfg["mat_size_m"] == 3.0
    assert cfg["safety_window_bias"] == 0.5
    # 폐기된 키가 되살아나면 red
    assert "interior_margin_ratio" not in cfg
