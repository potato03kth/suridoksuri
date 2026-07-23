"""`vision/tools/compute_nominal_intrinsics.py` 단위테스트 — 하드웨어 의존 없음.

`vision/tools/calib_analyze.py`의 검증 패턴(왕복 검증·yaml 스키마 round-trip)을 그대로
재사용한다(`docs/vision_aruco_branch.md` Phase 1 지시).
"""
from __future__ import annotations

import math

import pytest
import yaml

from vision.tools.calib_analyze import DEFAULT_CAMERA_ID as CALIB_ANALYZE_DEFAULT_CAMERA_ID
from vision.tools.compute_nominal_intrinsics import (
    DEFAULT_CAMERA_ID,
    build_nominal_artifact,
    compute_focal_length_px,
    compute_fov_deg,
    compute_nominal_intrinsics,
    main,
    save_nominal_artifact,
)


# ===========================================================================
# camera_id 관례 일치 (회귀) — calib_analyze.py가 나중에 같은 카메라의 실측
# 캘리브레이션을 같은 vision/calibration/<camera_id>/ 아래 쌓으려면 두 상수가
# 반드시 같아야 한다(모듈 docstring 참조).
# ===========================================================================


def test_default_camera_id_matches_calib_analyze():
    assert DEFAULT_CAMERA_ID == CALIB_ANALYZE_DEFAULT_CAMERA_ID


# ===========================================================================
# 왕복(round-trip) 검증 — 알려진 HFOV로 f 계산 후 atan2 역산해 원래 HFOV 복원
# ===========================================================================


@pytest.mark.parametrize("width_px,hfov_deg", [(4608, 75.0), (1920, 90.0), (640, 10.0), (100, 179.0)])
def test_focal_length_round_trips_to_original_hfov_horizontal(width_px, hfov_deg):
    f = compute_focal_length_px(width_px, hfov_deg)
    recovered = compute_fov_deg(f, width_px)
    assert recovered == pytest.approx(hfov_deg, abs=1e-9)


@pytest.mark.parametrize("width_px,height_px,hfov_deg", [(4608, 2592, 75.0), (1920, 1080, 102.0)])
def test_nominal_intrinsics_round_trips_diagonal_axis(width_px, height_px, hfov_deg):
    intr = compute_nominal_intrinsics(width_px, height_px, hfov_deg, hfov_axis="diagonal")
    diag = math.hypot(width_px, height_px)
    recovered = compute_fov_deg(intr.fx, diag)
    assert recovered == pytest.approx(hfov_deg, abs=1e-9)


def test_known_90deg_horizontal_gives_exact_half_width_focal_length():
    # fov=90deg -> tan(45deg)=1 -> f = (W/2)/1 = W/2 (닫힌 형태로 정확히 검증 가능한 케이스)
    intr = compute_nominal_intrinsics(4608, 2592, 90.0, hfov_axis="horizontal")
    assert intr.fx == pytest.approx(2304.0)
    assert intr.fy == pytest.approx(2304.0)
    assert intr.cx == pytest.approx(2304.0)
    assert intr.cy == pytest.approx(1296.0)
    assert list(intr.dist_coeffs) == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_narrower_hfov_gives_larger_focal_length():
    wide = compute_nominal_intrinsics(4608, 2592, 90.0, hfov_axis="horizontal")
    narrow = compute_nominal_intrinsics(4608, 2592, 30.0, hfov_axis="horizontal")
    assert narrow.fx > wide.fx


def test_invalid_hfov_axis_raises():
    with pytest.raises(ValueError):
        compute_nominal_intrinsics(4608, 2592, 75.0, hfov_axis="bogus")


@pytest.mark.parametrize("bad_fov", [0.0, -10.0, 180.0, 200.0])
def test_out_of_range_fov_raises(bad_fov):
    with pytest.raises(ValueError):
        compute_focal_length_px(1000.0, bad_fov)


# ===========================================================================
# yaml 아티팩트 round-trip + 필수 키(calib_analyze.py 패턴 재사용)
# ===========================================================================

REQUIRED_TOP_KEYS = {
    "schema_version", "camera_id", "created_utc", "git_commit", "source", "accuracy",
    "not_for_closed_loop_30cm", "image_size", "camera_matrix", "dist_coeffs", "hfov_assumption",
}


def test_yaml_artifact_round_trips_and_has_required_keys(tmp_path):
    intr = compute_nominal_intrinsics(4608, 2592, 75.0, hfov_axis="horizontal")
    artifact = build_nominal_artifact(intr, "test-cam", "deadbeef", "2026-07-24T00:00:00+00:00")
    out_path = save_nominal_artifact(artifact, tmp_path / "test-cam" / "nominal.yaml")

    data = yaml.safe_load(out_path.read_text(encoding="utf-8"))

    assert REQUIRED_TOP_KEYS <= data.keys()
    assert data["camera_id"] == "test-cam"
    assert data["schema_version"] == 1
    assert data["source"] == "nominal_datasheet"
    assert data["accuracy"] == "unverified"
    assert data["not_for_closed_loop_30cm"] is True
    assert data["image_size"] == [4608, 2592]
    assert len(data["camera_matrix"]) == 3 and len(data["camera_matrix"][0]) == 3
    assert data["camera_matrix"][0][0] == pytest.approx(intr.fx)
    assert data["camera_matrix"][1][1] == pytest.approx(intr.fy)
    assert data["camera_matrix"][0][2] == pytest.approx(2304.0)
    assert data["camera_matrix"][1][2] == pytest.approx(1296.0)
    assert len(data["dist_coeffs"]) == 5
    assert all(c == 0.0 for c in data["dist_coeffs"])
    assert data["hfov_assumption"]["axis"] == "horizontal"
    assert data["hfov_assumption"]["hfov_deg"] == pytest.approx(75.0)
    assert isinstance(data["hfov_assumption"]["note"], str) and len(data["hfov_assumption"]["note"]) > 0


def test_main_end_to_end_writes_nominal_yaml(tmp_path):
    out_calib_dir = tmp_path / "calib"
    rc = main(
        [
            "--sensor-width-px", "4608",
            "--sensor-height-px", "2592",
            "--hfov-deg", "75.0",
            "--hfov-axis", "horizontal",
            "--camera-id", "cam109-imx708af75",
            "--out-calib-dir", str(out_calib_dir),
        ]
    )
    assert rc == 0

    yaml_path = out_calib_dir / "cam109-imx708af75" / "nominal.yaml"
    assert yaml_path.exists()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["camera_id"] == "cam109-imx708af75"
    assert data["hfov_assumption"]["axis"] == "horizontal"


def test_main_invalid_hfov_reports_error_without_crash(tmp_path, capsys):
    rc = main(["--hfov-deg", "0.0", "--out-calib-dir", str(tmp_path / "calib")])
    assert rc == 1
    captured = capsys.readouterr()
    assert "오류" in captured.err
