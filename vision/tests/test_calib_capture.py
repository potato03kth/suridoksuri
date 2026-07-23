"""vision/tools/calib_capture.py 순수 함수 단위 테스트.

calib_capture.py 자체는 picamera2/libcamera가 필요한 RPi 하드웨어 전용 도구지만(vision/CLAUDE.md
tools/ 예외 규칙), 샷 목록 생성·좌표/픽셀 변환·커버리지 비닝·코너 안정성·블러 판정·컨트롤
고정 유지 검증은 numpy/cv2만 쓰는 순수 함수라 하드웨어 없이 여기서 검증한다.
"""
import numpy as np
import pytest

import cv2

from vision.tools.calib_capture import (
    COVERAGE_COLS,
    COVERAGE_ROWS,
    SET_A_DISTANCES_M,
    SET_B_DISTANCE_M,
    ShotSpec,
    _render_page,
    parse_autocapture_enabled,
    board_bounding_box,
    board_dimensions_mm,
    board_region_sharpness,
    build_shot_list,
    coverage_fill_fraction,
    corner_bin_indices,
    corners_are_stable,
    draw_coverage_inset,
    draw_zone_rect,
    format_instruction_ko,
    is_capture_blurry,
    locks_held_within_tolerance,
    max_corner_movement_px,
    new_coverage_grid,
    point_in_rect,
    render_preview_overlay,
    target_zone_size_px,
    update_coverage_grid,
    zone_label_ko,
    zone_to_pixel_rect,
)


# ---------------------------------------------------------------------------
# 샷 목록 생성
# ---------------------------------------------------------------------------


def test_build_shot_list_total_count_matches_spec():
    """세트A 3거리 x 15컷 = 45, 세트B 1거리 x 40컷 = 85 총합 — 세션 지시 수치와 정확히 일치."""
    shots = build_shot_list()
    assert len(shots) == 85


def test_build_shot_list_set_a_has_15_shots_per_distance():
    shots = build_shot_list()
    for d in SET_A_DISTANCES_M:
        subset = [s for s in shots if s.set_name == "A" and s.distance_m == d]
        assert len(subset) == 15
        assert all(s.total_in_distance == 15 for s in subset)
        # seq는 1..15 연속
        assert sorted(s.seq for s in subset) == list(range(1, 16))


def test_build_shot_list_set_b_has_40_shots():
    shots = build_shot_list()
    subset = [s for s in shots if s.set_name == "B"]
    assert len(subset) == 40
    assert all(s.distance_m == SET_B_DISTANCE_M for s in subset)
    assert all(s.total_in_distance == 40 for s in subset)
    assert sorted(s.seq for s in subset) == list(range(1, 41))


def test_build_shot_list_set_b_grid_covers_12_unique_positions_x_3_tilts():
    """격자 12위치 x 3틸트 = 36 + 코너롤 4 = 40. 격자 부분의 고유 (u,v) 개수가 12인지 확인."""
    shots = build_shot_list()
    grid_shots = [s for s in shots if s.set_name == "B" and s.zone == "grid"]
    assert len(grid_shots) == 36
    unique_positions = {(round(s.u, 3), round(s.v, 3)) for s in grid_shots}
    assert len(unique_positions) == 12
    # 각 위치마다 정확히 3틸트
    from collections import Counter

    counts = Counter((round(s.u, 3), round(s.v, 3)) for s in grid_shots)
    assert all(c == 3 for c in counts.values())


def test_build_shot_list_set_b_corner_roll_shots_are_roll_only():
    shots = build_shot_list()
    corner_shots = [s for s in shots if s.set_name == "B" and s.zone == "corner-roll"]
    assert len(corner_shots) == 4
    for s in corner_shots:
        assert s.yaw_deg == 0.0
        assert s.pitch_deg == 0.0
        assert 30.0 <= s.roll_deg <= 60.0


def test_build_shot_list_set_a_first_shot_is_centre_frontal():
    shots = build_shot_list()
    first = shots[0]
    assert first.set_name == "A"
    assert first.zone == "centre"
    assert first.u == pytest.approx(0.5)
    assert first.v == pytest.approx(0.5)
    assert first.yaw_deg == 0.0 and first.pitch_deg == 0.0 and first.roll_deg == 0.0


def test_build_shot_list_edge_zones_use_0p2_0p8_corners_use_both():
    shots = build_shot_list()
    a_shots = [s for s in shots if s.set_name == "A" and s.distance_m == SET_A_DISTANCES_M[0]]
    by_zone = {s.zone: s for s in a_shots if s.zone in ("left", "right", "top", "bottom", "TL", "TR", "BL", "BR")}
    assert by_zone["left"].u == pytest.approx(0.2) and by_zone["left"].v == pytest.approx(0.5)
    assert by_zone["right"].u == pytest.approx(0.8) and by_zone["right"].v == pytest.approx(0.5)
    assert by_zone["top"].v == pytest.approx(0.2) and by_zone["top"].u == pytest.approx(0.5)
    assert by_zone["bottom"].v == pytest.approx(0.8) and by_zone["bottom"].u == pytest.approx(0.5)
    for corner_zone, (eu, ev) in {
        "TL": (0.2, 0.2),
        "TR": (0.8, 0.2),
        "BL": (0.2, 0.8),
        "BR": (0.8, 0.8),
    }.items():
        # TL/BR도 각도변형 샷이 여러 개 있으므로 최소 하나는 이 위치를 가짐만 확인
        matches = [s for s in a_shots if s.zone == corner_zone]
        assert any(s.u == pytest.approx(eu) and s.v == pytest.approx(ev) for s in matches)


def test_zone_label_ko_known_positions():
    assert zone_label_ko(0.5, 0.5) == "정중앙"
    assert zone_label_ko(0.2, 0.2) == "좌상단(TL)"
    assert zone_label_ko(0.8, 0.2) == "우상단(TR)"
    assert zone_label_ko(0.2, 0.8) == "좌하단(BL)"
    assert zone_label_ko(0.8, 0.8) == "우하단(BR)"
    assert zone_label_ko(0.2, 0.5) == "좌측"
    assert "격자" in zone_label_ko(0.38, 0.5)


def test_format_instruction_ko_all_zero_is_frontal():
    assert format_instruction_ko(0.0, 0.0, 0.0) == "정면(회전 없음)"


def test_format_instruction_ko_includes_each_nonzero_axis():
    text = format_instruction_ko(35.0, -20.0, 15.0)
    assert "요잉 +35" in text
    assert "피칭 -20" in text
    assert "롤 +15" in text


# ---------------------------------------------------------------------------
# 보드 치수 / 타겟존 픽셀 변환
# ---------------------------------------------------------------------------


def test_board_dimensions_mm_a4_default():
    """(9,6) 패턴 + 28.0mm 사각형 -> 10x7 사각형 실치수 280x196mm (2026-07-23 재실측 정정값)."""
    w, h = board_dimensions_mm((9, 6), 28.0)
    assert w == pytest.approx(280.0)
    assert h == pytest.approx(196.0)


def test_board_dimensions_mm_tablet():
    w, h = board_dimensions_mm((9, 6), 20.0)
    assert w == pytest.approx(200.0)
    assert h == pytest.approx(140.0)


def test_target_zone_size_px_matches_spec_empirical_square_size_at_0p5m():
    """세션 지시 실측: 0.5m에서 정사각형 ~194px(풀해상도). 보드폭/10과 비교해 근사 검증."""
    board_w_mm, board_h_mm = board_dimensions_mm((9, 6), 28.0)
    width_px, _ = target_zone_size_px(0.5, frame_w=4608, board_width_mm=board_w_mm, board_height_mm=board_h_mm)
    square_px = width_px / 10.0  # 10개 사각형/가로
    assert square_px == pytest.approx(194, rel=0.03)


def test_target_zone_size_px_matches_spec_empirical_square_size_at_2p5m():
    """세션 지시 실측: 2.5m에서 정사각형 ~39px(풀해상도)."""
    board_w_mm, board_h_mm = board_dimensions_mm((9, 6), 28.0)
    width_px, _ = target_zone_size_px(2.5, frame_w=4608, board_width_mm=board_w_mm, board_height_mm=board_h_mm)
    square_px = width_px / 10.0
    assert square_px == pytest.approx(39, rel=0.03)


def test_target_zone_size_px_rejects_nonpositive_distance():
    with pytest.raises(ValueError):
        target_zone_size_px(0.0, 1280, 280.0, 196.0)
    with pytest.raises(ValueError):
        target_zone_size_px(-1.0, 1280, 280.0, 196.0)


def test_zone_to_pixel_rect_centre_is_symmetric():
    rect = zone_to_pixel_rect(0.5, 0.5, 1280, 720, 0.7, 280.0, 196.0)
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    assert cx == pytest.approx(640, abs=1)
    assert cy == pytest.approx(360, abs=1)


def test_zone_to_pixel_rect_clips_to_frame_bounds_near_edge():
    """가까운 거리(큰 박스) + 가장자리 위치 -> 박스가 프레임을 넘어가면 클립돼야 한다."""
    rect = zone_to_pixel_rect(0.2, 0.2, 640, 480, 0.3, 280.0, 196.0)
    x0, y0, x1, y1 = rect
    assert x0 >= 0 and y0 >= 0
    assert x1 <= 640 and y1 <= 480


def test_zone_to_pixel_rect_rejects_out_of_range_uv():
    with pytest.raises(ValueError):
        zone_to_pixel_rect(1.5, 0.5, 1280, 720, 1.0, 280.0, 196.0)
    with pytest.raises(ValueError):
        zone_to_pixel_rect(0.5, -0.1, 1280, 720, 1.0, 280.0, 196.0)


def test_point_in_rect():
    rect = (10, 10, 100, 100)
    assert point_in_rect(50, 50, rect) is True
    assert point_in_rect(5, 50, rect) is False
    assert point_in_rect(10, 10, rect) is True  # 경계 포함


# ---------------------------------------------------------------------------
# 커버리지 비닝
# ---------------------------------------------------------------------------


def test_new_coverage_grid_shape_and_all_false():
    grid = new_coverage_grid()
    assert grid.shape == (COVERAGE_ROWS, COVERAGE_COLS)
    assert not grid.any()


def test_corner_bin_indices_maps_to_expected_cell():
    # 1280x720 프레임, 12x8 격자 -> 셀 폭 106.67px, 셀 높이 90px
    corners = np.array([[50.0, 40.0]])  # 첫 열/첫 행
    bins = corner_bin_indices(corners, frame_w=1280, frame_h=720, cols=12, rows=8)
    assert bins == [(0, 0)]


def test_corner_bin_indices_clips_edge_point_into_last_bin():
    corners = np.array([[1280.0, 720.0]])  # 정확히 경계(프레임 밖 취급 안 되게 클립)
    bins = corner_bin_indices(corners, frame_w=1280, frame_h=720, cols=12, rows=8)
    assert bins == [(11, 7)]


def test_corner_bin_indices_dedups_multiple_points_in_same_cell():
    corners = np.array([[10.0, 10.0], [15.0, 12.0], [20.0, 8.0]])  # 다 같은 셀
    bins = corner_bin_indices(corners, frame_w=1280, frame_h=720, cols=12, rows=8)
    assert bins == [(0, 0)]


def test_update_coverage_grid_marks_bins_and_fill_fraction():
    grid = new_coverage_grid(cols=12, rows=8)
    corners = np.array([[0.0, 0.0], [1279.0, 719.0]])  # 두 대각 모서리 셀
    update_coverage_grid(grid, corners, frame_w=1280, frame_h=720)
    assert grid[0, 0] and grid[7, 11]
    assert coverage_fill_fraction(grid) == pytest.approx(2.0 / (12 * 8))


# ---------------------------------------------------------------------------
# 코너 안정성(자동촬영 게이팅)
# ---------------------------------------------------------------------------


def test_max_corner_movement_px_basic():
    prev = np.array([[0.0, 0.0], [10.0, 0.0]])
    curr = np.array([[0.0, 0.0], [13.0, 4.0]])  # 두번째 코너 5px 이동(3-4-5 삼각형)
    assert max_corner_movement_px(prev, curr) == pytest.approx(5.0)


def test_max_corner_movement_px_mismatched_shapes_raises():
    prev = np.array([[0.0, 0.0]])
    curr = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        max_corner_movement_px(prev, curr)


def test_corners_are_stable_true_when_movement_below_threshold():
    base = np.array([[float(i), float(i)] for i in range(6)])
    history = [base + np.array([0.1 * k, 0.0]) for k in range(5)]  # 최대 이동 0.4px
    assert corners_are_stable(history, max_movement_px=2.0) is True


def test_corners_are_stable_false_when_movement_exceeds_threshold():
    base = np.array([[float(i), float(i)] for i in range(6)])
    history = [base, base + np.array([5.0, 0.0])]  # 5px 이동 >= 2px 임계
    assert corners_are_stable(history, max_movement_px=2.0) is False


def test_corners_are_stable_false_with_fewer_than_2_frames():
    base = np.array([[0.0, 0.0]])
    assert corners_are_stable([base], max_movement_px=2.0) is False
    assert corners_are_stable([], max_movement_px=2.0) is False


def test_corners_are_stable_false_when_corner_count_changes():
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    b = np.array([[0.0, 0.0]])
    assert corners_are_stable([a, b], max_movement_px=2.0) is False


def test_corners_are_stable_catches_cumulative_drift_across_window():
    """각 스텝은 임계 미만이지만 누적으로는 임계를 넘는 경우도 불안정으로 잡아야 한다."""
    base = np.array([[0.0, 0.0]])
    history = [base + np.array([1.5 * k, 0.0]) for k in range(5)]  # 스텝간 1.5px, 전체 6px
    assert corners_are_stable(history, max_movement_px=2.0) is False


# ---------------------------------------------------------------------------
# 블러 판정
# ---------------------------------------------------------------------------


def _make_checkerboard(size=200, squares=8):
    board = np.zeros((size, size), dtype=np.uint8)
    step = size // squares
    for r in range(squares):
        for c in range(squares):
            if (r + c) % 2 == 0:
                board[r * step:(r + 1) * step, c * step:(c + 1) * step] = 255
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def test_board_bounding_box_adds_margin_and_clips():
    corners = np.array([[10.0, 10.0], [90.0, 10.0], [10.0, 90.0], [90.0, 90.0]])
    x0, y0, x1, y1 = board_bounding_box(corners, frame_w=100, frame_h=100, margin_frac=0.5)
    assert x0 == 0 and y0 == 0  # 마진 적용 후 음수 -> 0으로 클립
    assert x1 == 100 and y1 == 100


def test_board_bounding_box_rejects_empty_corners():
    with pytest.raises(ValueError):
        board_bounding_box(np.zeros((0, 2)), 100, 100)


def test_board_region_sharpness_sharp_vs_blurred():
    sharp = _make_checkerboard()
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)
    corners = np.array([[10.0, 10.0], [190.0, 10.0], [10.0, 190.0], [190.0, 190.0]])
    s_sharp = board_region_sharpness(sharp, corners)
    s_blur = board_region_sharpness(blurred, corners)
    assert s_sharp > s_blur


def test_is_capture_blurry_below_ratio_is_blurry():
    assert is_capture_blurry(sample_sharpness=30.0, reference_sharpness=100.0, ratio=0.4) is True


def test_is_capture_blurry_above_ratio_is_not_blurry():
    assert is_capture_blurry(sample_sharpness=90.0, reference_sharpness=100.0, ratio=0.4) is False


def test_is_capture_blurry_boundary_is_not_blurry():
    assert is_capture_blurry(sample_sharpness=40.0, reference_sharpness=100.0, ratio=0.4) is False


def test_is_capture_blurry_zero_reference_passes_through():
    assert is_capture_blurry(sample_sharpness=0.1, reference_sharpness=0.0, ratio=0.4) is False


# ---------------------------------------------------------------------------
# 컨트롤 고정 유지 검증
# ---------------------------------------------------------------------------


def _meta(lens, exp, gain, cg):
    return {"LensPosition": lens, "ExposureTime": exp, "AnalogueGain": gain, "ColourGains": cg}


def test_locks_held_within_tolerance_accepts_hardware_quantization():
    """세션 지시 실측 사례: exposure 8000 요청 -> 실제 7993, gain 1.0 -> 1.123. 이 정도 편차는
    통과해야 한다(정확 일치로 판정하지 않음)."""
    frames = [_meta(5.6, 7993, 1.123, (2.1, 1.8)) for _ in range(20)]
    ok, reasons = locks_held_within_tolerance(frames, lens_target=5.6, exposure_target=8000, gain_target=1.0,
                                                colour_gains_target=(2.1, 1.8))
    assert ok is True
    assert reasons == []


def test_locks_held_within_tolerance_detects_large_lens_drift():
    frames = [_meta(5.6, 8000, 1.0, (2.1, 1.8))] * 19 + [_meta(9.0, 8000, 1.0, (2.1, 1.8))]
    ok, reasons = locks_held_within_tolerance(frames, lens_target=5.6, exposure_target=8000, gain_target=1.0,
                                                colour_gains_target=(2.1, 1.8))
    assert ok is False
    assert any("LensPosition" in r for r in reasons)


def test_locks_held_within_tolerance_detects_large_exposure_drift():
    frames = [_meta(5.6, 2000, 1.0, (2.1, 1.8)) for _ in range(20)]  # 목표 8000인데 계속 2000
    ok, reasons = locks_held_within_tolerance(frames, lens_target=5.6, exposure_target=8000, gain_target=1.0,
                                                colour_gains_target=(2.1, 1.8))
    assert ok is False
    assert any("ExposureTime" in r for r in reasons)


def test_locks_held_within_tolerance_detects_colour_gains_drift():
    frames = [_meta(5.6, 8000, 1.0, (3.5, 1.8)) for _ in range(20)]  # R게인이 목표와 크게 다름
    ok, reasons = locks_held_within_tolerance(frames, lens_target=5.6, exposure_target=8000, gain_target=1.0,
                                                colour_gains_target=(2.1, 1.8))
    assert ok is False
    assert any("ColourGains" in r for r in reasons)


def test_locks_held_within_tolerance_ignores_missing_keys():
    frames = [{"LensPosition": 5.6}] * 20  # ExposureTime/AnalogueGain/ColourGains 없음
    ok, reasons = locks_held_within_tolerance(frames, lens_target=5.6, exposure_target=8000, gain_target=1.0,
                                                colour_gains_target=(2.1, 1.8))
    assert ok is True


# ---------------------------------------------------------------------------
# 오버레이 렌더링 — 크래시 없이 동작 + 대상 영역에 실제로 그려지는지
# ---------------------------------------------------------------------------


def test_draw_zone_rect_draws_nonzero_pixels():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_zone_rect(frame, (10, 10, 50, 50), (0, 255, 255), thickness=2)
    assert frame.any()


def test_draw_zone_rect_skips_degenerate_rect_without_crash():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_zone_rect(frame, (50, 50, 50, 50), (0, 255, 255), thickness=2)  # 폭/높이 0
    assert not frame.any()  # 아무것도 안 그려짐(크래시만 없으면 됨)


def test_draw_coverage_inset_renders_without_crash_and_marks_filled_bins():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    grid = new_coverage_grid()
    grid[0, 0] = True
    draw_coverage_inset(frame, grid, origin=(10, 10), cell_px=10)
    assert frame.any()


def test_render_preview_overlay_runs_end_to_end_without_hardware():
    shots = build_shot_list()
    shot = shots[0]
    specs = [s for s in shots if s.set_name == shot.set_name and s.distance_m == shot.distance_m]
    mask = [False] * len(specs)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    coverage = new_coverage_grid()
    out = render_preview_overlay(
        frame, specs, mask, shot, board_width_mm=280.0, board_height_mm=196.0,
        coverage=coverage, lock_ok=False, lock_values=None,
    )
    assert out.shape == (720, 1280, 3)
    assert out.any()  # 텍스트/박스가 실제로 그려짐


def test_render_preview_overlay_with_lock_and_corners_and_countdown():
    shots = build_shot_list()
    shot = shots[0]
    specs = [shot]
    mask = [True]
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    coverage = new_coverage_grid()
    coverage[0, 0] = True
    corners = np.array([[100.0, 100.0], [200.0, 100.0], [100.0, 200.0], [200.0, 200.0]])
    lock_values = {"lens_position": 5.6, "exposure_time": 8000, "analogue_gain": 1.1, "colour_gains": (2.1, 1.8)}
    out = render_preview_overlay(
        frame, specs, mask, shot, board_width_mm=280.0, board_height_mm=196.0,
        coverage=coverage, lock_ok=True, lock_values=lock_values,
        live_corners=corners, countdown_s=0.8, last_result_text="촬영 성공: 001.png",
    )
    assert out.shape == (720, 1280, 3)


# ---------------------------------------------------------------------------
# /autocapture 토글 — 2026-07-23 실기 버그 회귀
# ---------------------------------------------------------------------------


def test_parse_autocapture_enabled_explicit_values():
    assert parse_autocapture_enabled("enabled=1") is True
    assert parse_autocapture_enabled("enabled=0") is False


def test_parse_autocapture_enabled_defaults_to_off_when_param_missing():
    """파라미터가 없으면 **끈다**.

    과거 기본값이 켬("1")이라, GET 폼이 쿼리를 날려먹는 것과 맞물려 "끄기" 버튼이 매번
    자동촬영을 켰다(실기에서 사용자가 자동촬영을 끌 수 없었음). 켜지는 쪽이 사진을 멋대로
    찍어 더 해로우므로 기본값은 끔이어야 한다.
    """
    assert parse_autocapture_enabled("") is False
    assert parse_autocapture_enabled("other=1") is False


class _StubSession:
    """`_render_page`가 읽는 것은 `status_dict()` 하나뿐이라 그것만 흉내낸다."""

    def __init__(self, auto_capture_enabled: bool):
        self._auto = auto_capture_enabled

    def status_dict(self):
        return {"auto_capture_enabled": self._auto}


@pytest.mark.parametrize("enabled,expected_value", [(True, "0"), (False, "1")])
def test_render_page_autocapture_button_carries_enabled_as_hidden_input(enabled, expected_value):
    """토글 버튼은 `enabled`를 **hidden input**으로 실어야 한다.

    `action="/autocapture?enabled=N"`으로 쿼리에 박으면 안 된다 — method=GET 폼은 제출 시
    action의 쿼리스트링을 버리고 폼 입력값으로 교체하므로 그 값이 서버에 도달하지 않는다.
    """
    html = _render_page(_StubSession(enabled), "10.0.0.1", 8080)
    assert f'<input type="hidden" name="enabled" value="{expected_value}">' in html
    # 회귀 가드: action에 쿼리스트링을 되돌려 넣으면 다시 깨진다
    assert 'action="/autocapture?' not in html
    assert 'action="/autocapture"' in html
