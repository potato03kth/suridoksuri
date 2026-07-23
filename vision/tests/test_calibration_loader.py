"""`vision/utils/calibration_loader.py` 단위테스트 — 하드웨어 의존 없음.

ArUco 브랜치 Phase 3(`docs/vision_aruco_branch.md` §Phase 3) — `nominal.yaml` 로드 어댑터.
yaml round-trip 검증 패턴은 `vision/tools/calib_analyze.py`/`compute_nominal_intrinsics.py`의
관례(`tests/test_compute_nominal_intrinsics.py`)를 재사용한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from vision.utils.calibration_loader import CameraCalibration, load_camera_calibration

_REAL_NOMINAL_YAML = (
    Path(__file__).parent.parent / "calibration" / "cam109-imx708af75" / "nominal.yaml"
)


# ===========================================================================
# 실제 커밋된 nominal.yaml 로드(Phase 1 산출물, 회귀) — 이 파일이 실제로 존재하고
# 로더가 실제 스키마를 파싱할 수 있는지 확인.
# ===========================================================================


def test_loads_real_committed_nominal_yaml():
    assert _REAL_NOMINAL_YAML.exists(), f"Phase 1 산출물이 없음: {_REAL_NOMINAL_YAML}"
    calib = load_camera_calibration(_REAL_NOMINAL_YAML)

    assert isinstance(calib, CameraCalibration)
    assert calib.camera_id == "cam109-imx708af75"
    assert calib.camera_matrix.shape == (3, 3)
    assert calib.dist_coeffs.shape == (5,)
    assert calib.image_size == (4608, 2592)
    assert calib.source == "nominal_datasheet"
    assert calib.accuracy == "unverified"
    assert calib.not_for_closed_loop_30cm is True
    # provenance echo(§7.3) — calib_id는 로드에 쓴 경로 그대로
    assert calib.calib_id == str(_REAL_NOMINAL_YAML)
    assert calib.raw["schema_version"] == 1


def test_real_nominal_yaml_camera_matrix_values_match_file():
    calib = load_camera_calibration(_REAL_NOMINAL_YAML)
    data = yaml.safe_load(_REAL_NOMINAL_YAML.read_text(encoding="utf-8"))
    expected_K = np.asarray(data["camera_matrix"], dtype=np.float64)
    assert np.allclose(calib.camera_matrix, expected_K)


# ===========================================================================
# 합성 yaml round-trip(tmp_path) — calib_analyze.py/compute_nominal_intrinsics.py 관례 재사용
# ===========================================================================


def _write_synthetic_nominal_yaml(path: Path, **overrides) -> dict:
    artifact = {
        "schema_version": 1,
        "camera_id": "test-cam",
        "created_utc": "2026-07-24T00:00:00+00:00",
        "git_commit": "deadbeef",
        "source": "nominal_datasheet",
        "accuracy": "unverified",
        "not_for_closed_loop_30cm": True,
        "image_size": [640, 480],
        "camera_matrix": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "hfov_assumption": {"axis": "horizontal", "hfov_deg": 66.0, "note": "테스트용"},
    }
    artifact.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(artifact, f, allow_unicode=True, sort_keys=False)
    return artifact


def test_round_trip_synthetic_yaml_matches_written_values(tmp_path):
    out_path = tmp_path / "test-cam" / "nominal.yaml"
    _write_synthetic_nominal_yaml(out_path)

    calib = load_camera_calibration(out_path)

    assert calib.camera_id == "test-cam"
    assert np.allclose(calib.camera_matrix, [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    assert np.allclose(calib.dist_coeffs, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert calib.image_size == (640, 480)
    assert calib.accuracy == "unverified"
    assert calib.not_for_closed_loop_30cm is True
    assert calib.calib_id == str(out_path)


def test_round_trip_reflects_verified_accuracy_flags(tmp_path):
    """실측 캘리브레이션 재개 후를 대비 — accuracy/not_for_closed_loop_30cm이 값 그대로
    반영되는지(항상 unverified/True로 하드코딩되지 않았는지) 확인."""
    out_path = tmp_path / "test-cam" / "nominal.yaml"
    _write_synthetic_nominal_yaml(out_path, accuracy="verified", not_for_closed_loop_30cm=False)

    calib = load_camera_calibration(out_path)

    assert calib.accuracy == "verified"
    assert calib.not_for_closed_loop_30cm is False


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_camera_calibration(tmp_path / "does_not_exist" / "nominal.yaml")
