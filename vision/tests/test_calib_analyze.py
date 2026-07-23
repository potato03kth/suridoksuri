"""vision/tools/calib_analyze.py 테스트 — 합성 데이터 자체검증.

**2026-07-23 시점 실촬영 사진이 아직 없다**(사용자가 촬영 진행 중, `docs/vision_camera_bringup.md`
인수인계 참조) — 그래서 정확성은 여기서 **진짜 K/dist를 아는 합성 체스보드 투영**으로만
담보한다. 실사진 확보 후 골든셋 회귀는 별도 세션에서 추가한다(`tests/golden/` 패턴 재사용 예정).

합성 사이드카는 `calib_capture.py::_capture_locked()`가 실제로 쓰는 필드명을 그대로 흉내낸다
(`vision/CLAUDE.md`/이 모듈 docstring이 코드로 확인한 스키마).
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from vision.tools.calib_analyze import (
    analyze_group,
    build_artifact,
    check_dataset_consistency,
    check_diopter_assumption,
    check_hfov,
    detect_outliers,
    determine_recommended,
    diopter_assumption_rows,
    find_set_b_group,
    implied_focal_length_mm,
    linear_fit_lens_position,
    load_dataset,
    main,
    next_calib_id,
    object_points_for_pattern,
    save_artifact,
)


# ===========================================================================
# 합성 데이터 생성 헬퍼
# ===========================================================================

PATTERN_SIZE = (9, 6)
SQUARE_SIZE_MM = 28.0
IMAGE_SIZE = (1600, 1200)  # (w,h) — 실제 4608x2592보다 작게(테스트 속도), 종횡비만 다름(문제 없음)


def _true_camera_matrix() -> np.ndarray:
    return np.array([[1400.0, 0.0, 812.0], [0.0, 1390.0, 588.0], [0.0, 0.0, 1.0]])


def _true_dist_coeffs() -> np.ndarray:
    return np.array([-0.08, 0.02, 0.0012, -0.0018, 0.006])


def _object_points() -> np.ndarray:
    return object_points_for_pattern(PATTERN_SIZE, SQUARE_SIZE_MM).astype(np.float32)


def _random_pose(rng: np.random.Generator, z_mm: float):
    """축퇴(전부 정면평행) 회피용 — 요/피치/롤을 충분히 섞는다."""
    yaw = np.radians(rng.uniform(-30, 30))
    pitch = np.radians(rng.uniform(-30, 30))
    roll = np.radians(rng.uniform(-20, 20))
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    R = rz @ ry @ rx
    rvec, _ = cv2.Rodrigues(R)
    tx = rng.uniform(-90, 90)
    ty = rng.uniform(-70, 70)
    tvec = np.array([[tx], [ty], [z_mm]], dtype=np.float64)
    return rvec, tvec


def _synthesize_corners(K: np.ndarray, dist: np.ndarray, n_images: int, z_mm: float, seed: int = 0):
    """n_images장 분량 (Nx2 px 코너) 목록 — 전부 이미지 경계 안에 들어오도록 포즈 재추첨."""
    rng = np.random.default_rng(seed)
    objp = _object_points()
    margin = 15.0
    out = []
    for i in range(n_images):
        for _attempt in range(300):
            rvec, tvec = _random_pose(rng, z_mm)
            proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
            proj = proj.reshape(-1, 2)
            if (
                np.all(proj[:, 0] >= margin) and np.all(proj[:, 0] <= IMAGE_SIZE[0] - margin)
                and np.all(proj[:, 1] >= margin) and np.all(proj[:, 1] <= IMAGE_SIZE[1] - margin)
            ):
                out.append(proj)
                break
        else:
            raise RuntimeError(f"이미지 {i}: 300회 시도해도 경계 안에 드는 포즈를 못 찾음(z_mm={z_mm})")
    return out


def _write_sidecar(
    root: Path, set_name: str, distance_m: float, seq: int, corners: np.ndarray, lens_position: float,
    image_size=IMAGE_SIZE, pattern_size=PATTERN_SIZE, square_size_mm=SQUARE_SIZE_MM,
) -> Path:
    d = root / set_name / f"{distance_m:g}m"
    d.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "timestamp": "2026-07-23T00:00:00+00:00",
        "set": set_name, "distance_m": distance_m, "seq": seq, "zone": "grid",
        "u": 0.5, "v": 0.5, "yaw_deg": 12.0, "pitch_deg": -8.0, "roll_deg": 0.0,
        "label_ko": f"합성테스트#{seq}", "pattern_size": list(pattern_size), "square_size_mm": square_size_mm,
        "locked_controls": {
            "lens_position": lens_position, "exposure_time": 8000, "analogue_gain": 1.5,
            "colour_gains": [1.8, 1.6],
        },
        "corners_px": corners.reshape(-1, 2).tolist(),
        "sharpness": 120.0, "reference_sharpness": 130.0,
        "target_zone_px": [0, 0, image_size[0], image_size[1]],
        "image_size": list(image_size), "image_path": str(d / f"{seq:03d}.png"),
    }
    json_path = d / f"{seq:03d}.json"
    json_path.write_text(json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")
    return json_path


def _build_group_tree(
    root: Path, set_name: str, distance_m: float, K: np.ndarray, dist: np.ndarray,
    n_images: int, lens_position: float, seed: int = 0,
) -> None:
    corners_list = _synthesize_corners(K, dist, n_images, z_mm=distance_m * 1000.0, seed=seed)
    for i, corners in enumerate(corners_list, start=1):
        _write_sidecar(root, set_name, distance_m, i, corners, lens_position)


# ===========================================================================
# ★ 합성 왕복 테스트(핵심) — 단일 그룹 캘리브레이션이 진짜 K/dist를 복원하는가
# ===========================================================================


def test_synthetic_round_trip_recovers_intrinsics_within_1_percent(tmp_path):
    K_true = _true_camera_matrix()
    dist_true = _true_dist_coeffs()
    root = tmp_path / "raw"
    _build_group_tree(root, "A", 0.5, K_true, dist_true, n_images=20, lens_position=2.0, seed=7)

    groups, warnings = load_dataset(root)
    assert warnings == []
    assert len(groups) == 1
    key = next(iter(groups))
    ga = analyze_group(groups[key])
    assert ga.calib_final is not None, ga.error_msg
    assert len(ga.calib_final.images) == 20  # 이상치 없이 전부 사용됐어야 함(합성=완전 무결)

    K_est = ga.calib_final.K
    dist_est = ga.calib_final.dist

    assert K_est[0, 0] == pytest.approx(K_true[0, 0], rel=0.01)
    assert K_est[1, 1] == pytest.approx(K_true[1, 1], rel=0.01)
    assert K_est[0, 2] == pytest.approx(K_true[0, 2], rel=0.01)
    assert K_est[1, 2] == pytest.approx(K_true[1, 2], rel=0.01)

    # 왜곡계수는 0 근방 값도 있어 상대오차가 부적절 -> 절대오차로 검증
    assert dist_est[0] == pytest.approx(dist_true[0], abs=0.01)
    assert dist_est[1] == pytest.approx(dist_true[1], abs=0.02)
    assert dist_est[2] == pytest.approx(dist_true[2], abs=0.002)
    assert dist_est[3] == pytest.approx(dist_true[3], abs=0.002)


# ===========================================================================
# fx-vs-LensPosition 직선적합 복원
# ===========================================================================


def test_linear_fit_recovers_known_line():
    a_true, b_true = 1400.0, -60.0
    lens_positions = [0.40, 0.83, 1.43, 2.03]
    points = [(L, a_true + b_true * L) for L in lens_positions]

    fit = linear_fit_lens_position(points)

    assert fit is not None
    assert fit["intercept"] == pytest.approx(a_true, abs=1e-6)
    assert fit["slope"] == pytest.approx(b_true, abs=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-6)
    assert fit["n_points"] == 4


def test_linear_fit_returns_none_when_fewer_than_2_points():
    assert linear_fit_lens_position([]) is None
    assert linear_fit_lens_position([(1.0, 1000.0)]) is None
    assert linear_fit_lens_position([(1.0, None)]) is None


def test_implied_focal_length_mm_matches_hand_derivation():
    # fx(L) = 1000*(1 + 0.0047*L) 형태로 직접 구성 -> a=1000, b=4.7 -> b/a[m]=0.0047 -> mm=4.7
    fit = {"intercept": 1000.0, "slope": 4.7}
    assert implied_focal_length_mm(fit) == pytest.approx(4.7, abs=1e-9)
    assert implied_focal_length_mm(None) is None


# ===========================================================================
# 이상치 검출
# ===========================================================================


def test_detect_outliers_flags_values_above_threshold():
    # 표본이 너무 작으면 극단값 하나가 mean+std 자체를 밀어올려 역설적으로 "이상치 아님"으로
    # 판정될 수 있다(breakdown point) — 정상값을 충분히 섞어 이 함정을 피한다.
    errors = [0.2, 0.3, 0.25, 0.4, 0.35, 0.28, 0.32, 5.0]  # 마지막이 뚜렷한 이상치
    mask, threshold = detect_outliers(errors, abs_threshold=1.0, std_multiplier=2.0)
    assert mask == [False, False, False, False, False, False, False, True]
    assert threshold >= 1.0


def test_outlier_detection_flags_corrupted_image_and_improves_rms(tmp_path):
    K_true = _true_camera_matrix()
    dist_true = _true_dist_coeffs()
    root = tmp_path / "raw"
    _build_group_tree(root, "A", 0.5, K_true, dist_true, n_images=15, lens_position=2.0, seed=11)

    groups, _warnings = load_dataset(root)
    key = next(iter(groups))
    group = groups[key]

    # 한 이미지의 코너 전부에 큰 독립 잡음을 줘 어떤 강체 포즈로도 설명 안 되게 오염시킨다
    # (균일 이동은 그 이미지의 tvec 재추정으로 흡수돼 오차가 안 드러날 수 있어 피한다).
    rng = np.random.default_rng(99)
    target = group.images[0]
    target.corners_px = target.corners_px + rng.normal(scale=35.0, size=target.corners_px.shape)

    ga = analyze_group(group, outlier_abs_px=1.0, outlier_std_mult=2.0, min_images_for_recal=4)

    assert ga.calib_before is not None
    assert ga.outlier_mask[0] is True
    assert sum(ga.outlier_mask) >= 1
    assert len(ga.calib_final.images) == 14  # 오염된 1장이 실제로 제외됐는지
    assert ga.calib_final.rms < ga.calib_before.rms  # 제외 후 RMS 개선


# ===========================================================================
# 부분 데이터(그룹 1개) — 크래시 없음·적합 생략·recommended 폴백
# ===========================================================================


def test_partial_data_single_group_no_crash_fit_skipped_and_fallback_recorded(tmp_path):
    K_true = _true_camera_matrix()
    dist_true = _true_dist_coeffs()
    root = tmp_path / "raw"
    _build_group_tree(root, "A", 0.7, K_true, dist_true, n_images=12, lens_position=1.43, seed=13)

    out_calib_dir = tmp_path / "calib"
    out_results_dir = tmp_path / "results"
    rc = main(
        [
            "--raw-root", str(root),
            "--out-calib-dir", str(out_calib_dir),
            "--out-results-dir", str(out_results_dir),
            "--camera-id", "test-cam",
            "--calib-id", "test-cam_2026-07-23a",
            "--no-plots",
        ]
    )
    assert rc == 0

    yaml_path = out_calib_dir / "test-cam" / "test-cam_2026-07-23a.yaml"
    assert yaml_path.exists()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    assert data["lens_position_fit"]["fx"] is None
    assert data["lens_position_fit"]["fy"] is None
    assert "폴백" in data["recommended"]["note"]
    assert "LensPosition" in data["recommended"]["which"] or "폴백" in data["recommended"]["which"]
    # 단일 그룹이라도 recommended camera_matrix/dist_coeffs는 채워져 있어야 함(크래시 없이 동작)
    assert len(data["recommended"]["camera_matrix"]) == 3
    assert len(data["recommended"]["dist_coeffs"]) == 5


def test_load_dataset_missing_raw_root_returns_empty_without_crash(tmp_path):
    groups, warnings = load_dataset(tmp_path / "does-not-exist")
    assert groups == {}
    assert any("존재하지 않음" in w for w in warnings)


def test_main_reports_error_without_crash_when_no_data_found(tmp_path, capsys):
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    rc = main(["--raw-root", str(empty_root), "--out-calib-dir", str(tmp_path / "calib")])
    assert rc == 1
    captured = capsys.readouterr()
    assert "오류" in captured.err


# ===========================================================================
# yaml 아티팩트 왕복 + 필수 키
# ===========================================================================

REQUIRED_TOP_KEYS = {
    "schema_version", "calib_id", "camera_id", "created_utc", "git_commit",
    "source", "image_size", "recommended", "groups", "lens_position_fit", "checks",
}
REQUIRED_GROUP_KEYS = {"set", "distance_m", "lens_position", "n_images", "rms_px", "camera_matrix", "dist_coeffs"}
REQUIRED_CHECK_KEYS = {"name", "ok", "detail"}


def test_yaml_artifact_round_trips_and_has_required_keys(tmp_path):
    K_true = _true_camera_matrix()
    dist_true = _true_dist_coeffs()
    root = tmp_path / "raw"
    _build_group_tree(root, "A", 0.5, K_true, dist_true, n_images=10, lens_position=2.03, seed=21)
    _build_group_tree(root, "B", 2.5, K_true, dist_true, n_images=10, lens_position=0.40, seed=22)

    out_calib_dir = tmp_path / "calib"
    out_results_dir = tmp_path / "results"
    rc = main(
        [
            "--raw-root", str(root),
            "--out-calib-dir", str(out_calib_dir),
            "--out-results-dir", str(out_results_dir),
            "--camera-id", "test-cam",
            "--calib-id", "test-cam_2026-07-23b",
            "--no-plots",
        ]
    )
    assert rc == 0

    yaml_path = out_calib_dir / "test-cam" / "test-cam_2026-07-23b.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    assert REQUIRED_TOP_KEYS <= data.keys()
    assert data["camera_id"] == "test-cam"
    assert data["calib_id"] == "test-cam_2026-07-23b"
    assert data["schema_version"] == 1
    assert data["lens_position_fit"]["fx"] is not None  # 그룹 2개 -> 적합 성공
    assert len(data["groups"]) == 2
    for g in data["groups"]:
        assert REQUIRED_GROUP_KEYS <= g.keys()
        assert len(g["camera_matrix"]) == 3 and len(g["camera_matrix"][0]) == 3
        assert len(g["dist_coeffs"]) == 5
    for c in data["checks"]:
        assert set(c.keys()) == REQUIRED_CHECK_KEYS
        assert isinstance(c["ok"], bool)  # numpy.bool_ 누출 방지 회귀
    # 세트B가 있으므로 recommended는 fit(fx/fy)+세트B(cx/cy/dist) 조합이어야 함
    assert "세트B" in data["recommended"]["which"] or "B" in data["recommended"]["which"]


def test_save_artifact_and_next_calib_id_avoid_collision(tmp_path):
    out_dir = tmp_path / "cam"
    out_dir.mkdir(parents=True)
    (out_dir / "cam_2026-07-23a.yaml").write_text("x: 1\n")
    next_id = next_calib_id(out_dir, "cam", "2026-07-23")
    assert next_id == "cam_2026-07-23b"


# ===========================================================================
# 세트 B 대조 / 디옵터 가정 / 정합성 검사 — 개별 순수 함수 단위 검증
# ===========================================================================


def test_find_set_b_group_picks_set_b_with_most_images(tmp_path):
    K_true = _true_camera_matrix()
    dist_true = _true_dist_coeffs()
    root = tmp_path / "raw"
    _build_group_tree(root, "A", 0.5, K_true, dist_true, n_images=8, lens_position=2.0, seed=31)
    _build_group_tree(root, "B", 2.5, K_true, dist_true, n_images=8, lens_position=0.4, seed=32)
    groups, _ = load_dataset(root)
    analyses = [analyze_group(g) for g in groups.values()]
    b_ga = find_set_b_group(analyses)
    assert b_ga is not None
    assert b_ga.key[0] == "B"


def test_diopter_assumption_rows_and_check_matches_bringup_predictions():
    # docs/vision_camera_bringup.md 예상치: 0.5m->2.0, 0.7m->1.43, 1.2m->0.83, 2.5m->0.40
    from vision.tools.calib_analyze import GroupAnalysis

    def fake_ga(set_name, distance_m, lens_position):
        return GroupAnalysis(
            key=(set_name, distance_m), pattern_size=(9, 6), square_size_mm=28.0, image_size=(100, 100),
            lens_position=lens_position, n_total_images=1,
            calib_before=None, errors_before=[], outlier_mask=[], outlier_threshold=0.0,
            calib_final=None, errors_final=[],
        )

    analyses = [
        fake_ga("A", 0.5, 2.0),
        fake_ga("A", 0.7, 1.43),
        fake_ga("A", 1.2, 0.83),
        fake_ga("B", 2.5, 0.40),
    ]
    rows = diopter_assumption_rows(analyses)
    assert len(rows) == 4
    check = check_diopter_assumption(rows, tol_frac=0.05)
    assert check["ok"] is True
    assert isinstance(check["ok"], bool)


def test_check_dataset_consistency_flags_mismatched_board(tmp_path):
    K_true = _true_camera_matrix()
    dist_true = _true_dist_coeffs()
    root = tmp_path / "raw"
    _build_group_tree(root, "A", 0.5, K_true, dist_true, n_images=6, lens_position=2.0, seed=41)
    # 두 번째 그룹은 다른 square_size_mm(태블릿 보드)로 기록
    _write_sidecar(
        root, "A", 0.7, 1,
        _synthesize_corners(K_true, dist_true, 1, z_mm=700.0, seed=42)[0],
        lens_position=1.43, square_size_mm=20.0,
    )
    groups, _ = load_dataset(root)
    analyses = [analyze_group(g) for g in groups.values()]
    check = check_dataset_consistency(analyses)
    assert check["ok"] is False
    assert isinstance(check["ok"], bool)


def test_determine_recommended_invalid_explicit_source_raises_value_error(tmp_path):
    K_true = _true_camera_matrix()
    dist_true = _true_dist_coeffs()
    root = tmp_path / "raw"
    _build_group_tree(root, "A", 0.5, K_true, dist_true, n_images=6, lens_position=2.0, seed=51)
    groups, _ = load_dataset(root)
    analyses = [analyze_group(g) for g in groups.values()]
    with pytest.raises(ValueError):
        determine_recommended(analyses, None, None, source="Z:99")


def test_main_invalid_recommended_source_reports_error_not_crash(tmp_path, capsys):
    K_true = _true_camera_matrix()
    dist_true = _true_dist_coeffs()
    root = tmp_path / "raw"
    _build_group_tree(root, "A", 0.5, K_true, dist_true, n_images=6, lens_position=2.0, seed=61)
    rc = main(
        [
            "--raw-root", str(root),
            "--out-calib-dir", str(tmp_path / "calib"),
            "--out-results-dir", str(tmp_path / "results"),
            "--recommended-source", "B",  # 세트 B 없음
            "--no-plots",
        ]
    )
    assert rc == 1
    assert "오류" in capsys.readouterr().err


# ===========================================================================
# 진단 플롯 — 실제로 PNG 파일이 생성되는지(스모크)
# ===========================================================================


def test_main_end_to_end_produces_plots(tmp_path):
    K_true = _true_camera_matrix()
    dist_true = _true_dist_coeffs()
    root = tmp_path / "raw"
    _build_group_tree(root, "A", 0.5, K_true, dist_true, n_images=8, lens_position=2.0, seed=71)
    _build_group_tree(root, "B", 2.5, K_true, dist_true, n_images=8, lens_position=0.4, seed=72)

    out_calib_dir = tmp_path / "calib"
    out_results_dir = tmp_path / "results"
    rc = main(
        [
            "--raw-root", str(root),
            "--out-calib-dir", str(out_calib_dir),
            "--out-results-dir", str(out_results_dir),
            "--camera-id", "test-cam",
            "--calib-id", "test-cam_2026-07-23c",
        ]
    )
    assert rc == 0
    assert (out_results_dir / "test-cam_2026-07-23c_corners.png").exists()
    assert (out_results_dir / "test-cam_2026-07-23c_reproj_errors.png").exists()
    assert (out_results_dir / "test-cam_2026-07-23c_fx_vs_lens.png").exists()


# ===========================================================================
# --redetect 경로 (PNG에서 재검출)
# ===========================================================================


def test_redetect_finds_real_chessboard_pattern_in_png(tmp_path):
    """실제 체스보드 이미지를 그려 PNG로 저장하고, corners_px 없는 사이드카로 --redetect가
    실제로 코너를 되찾는지 확인한다(코너 재검출 경로 자체의 정확성 검증)."""
    root = tmp_path / "raw"
    d = root / "A" / "0.5m"
    d.mkdir(parents=True)

    board_img = _render_flat_chessboard(PATTERN_SIZE, square_px=60, margin_px=60)
    png_path = d / "001.png"
    cv2.imwrite(str(png_path), board_img)

    sidecar = {
        "set": "A", "distance_m": 0.5, "seq": 1, "zone": "centre",
        "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0, "label_ko": "정면",
        "pattern_size": list(PATTERN_SIZE), "square_size_mm": SQUARE_SIZE_MM,
        "locked_controls": {"lens_position": 2.0, "exposure_time": 8000, "analogue_gain": 1.0, "colour_gains": [1.5, 1.5]},
        "corners_px": [],  # 일부러 비움 -> redetect 강제
        "image_size": [board_img.shape[1], board_img.shape[0]],
        "image_path": str(png_path),
    }
    (d / "001.json").write_text(json.dumps(sidecar), encoding="utf-8")

    groups, warnings = load_dataset(root, redetect=True)
    assert warnings == []
    assert len(groups) == 1
    ga = next(iter(groups.values()))
    assert ga.images[0].corner_source == "redetect"
    assert ga.images[0].corners_px.shape == (54, 2)  # 9*6


def _render_flat_chessboard(pattern_size, square_px: int, margin_px: int) -> np.ndarray:
    cols, rows = pattern_size
    sq_cols, sq_rows = cols + 1, rows + 1
    h = sq_rows * square_px + margin_px * 2
    w = sq_cols * square_px + margin_px * 2
    img = np.full((h, w), 255, dtype=np.uint8)
    for r in range(sq_rows):
        for c in range(sq_cols):
            if (r + c) % 2 == 0:
                y0 = margin_px + r * square_px
                x0 = margin_px + c * square_px
                img[y0 : y0 + square_px, x0 : x0 + square_px] = 0
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
