"""
`vision/calibration/<camera_id>/nominal.yaml` 로드 어댑터(ArUco 브랜치 Phase 3,
`docs/vision_aruco_branch.md` §Phase 3).

§7.1 ports & adapters 원칙대로 "카메라별 캘리브레이션 파일"은 어댑터다 — `vision/core/target.py`의
`solve_target_pose()`는 파일 I/O를 하지 않으므로(core 순수성, import 규칙), 이 파일이 yaml을 읽어
순수 배열(`camera_matrix`/`dist_coeffs`)과 provenance 필드(`accuracy`/`not_for_closed_loop_30cm`/
`calib_id`)로 변환해 넘긴다.

**스키마: `vision/tools/compute_nominal_intrinsics.py`가 만드는 `nominal.yaml` 전용**이다(
`vision/tools/calib_analyze.py`가 만드는 `<calib_id>.yaml`은 `camera_matrix`가 `recommended` 아래
중첩돼 있고 `accuracy`/`not_for_closed_loop_30cm` 필드 자체가 없는 다른 스키마라 이 로더의 대상이
아니다 — 필요해지면 그때 별도 로더나 스키마 통합을 검토, 지금은 과설계하지 않는다).

`calib_id`(provenance echo, §7.3): `nominal.yaml`에는 `calib_id` 키가 없으므로(그 키는
`calib_analyze.py` 산출물에만 있음) **로드에 쓴 파일 경로 문자열**을 그대로 `calib_id`로 쓴다
(세션 지시 "nominal.yaml 경로 또는 파일 해시" 중 경로 방식 채택 — 해시보다 사람이 바로 알아볼 수
있고, 어차피 `git_commit`이 yaml 안에 이미 있어 내용 추적은 그걸로 가능하다).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import yaml


@dataclass
class CameraCalibration:
    """`nominal.yaml` 1개를 로드한 결과 — `vision.core.target.solve_target_pose()` 입력으로
    바로 넘길 수 있는 순수 배열 + provenance 필드."""

    camera_id: str
    camera_matrix: np.ndarray        # (3,3) float64
    dist_coeffs: np.ndarray          # (5,) float64
    image_size: tuple[int, int]      # (width_px, height_px)
    source: str                      # 예: "nominal_datasheet"
    accuracy: str                    # 예: "unverified"
    not_for_closed_loop_30cm: bool
    calib_id: str                    # provenance echo — 이 yaml 파일 경로(문자열)
    raw: dict                        # yaml.safe_load 원본 그대로(디버깅/확장용)


def load_camera_calibration(path: Union[str, Path]) -> CameraCalibration:
    """`nominal.yaml` 경로 -> `CameraCalibration`. 파일 없음/필수 키 누락 시 그대로 예외
    전파(호출자가 판단할 문제 — 이 로더가 조용히 기본값으로 넘어가면 nominal 여부/정확도
    플래그가 소리 없이 사라질 위험이 있어 여기서는 관대하게 넘어가지 않는다)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"카메라 캘리브레이션 yaml을 찾을 수 없음: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64)
    image_size = tuple(int(v) for v in data["image_size"])

    return CameraCalibration(
        camera_id=str(data.get("camera_id", "")),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,  # type: ignore[assignment]
        source=str(data.get("source", "")),
        accuracy=str(data.get("accuracy", "unverified")),
        not_for_closed_loop_30cm=bool(data.get("not_for_closed_loop_30cm", True)),
        calib_id=str(p),
        raw=data,
    )
