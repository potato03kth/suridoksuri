"""
Nominal(데이터시트/실측 화각 근사) 카메라 intrinsics 산출 — 실측 체커보드 캘리브레이션이
2차예선 마감임박으로 보류된 동안(메모리 `project_vision_calibration_deferred`) ArUco
solvePnP 파이프라인(`docs/vision_aruco_branch.md` Phase 1)이 쓸 임시 K/dist를 만든다.

## 공식

핀홀 모델, 정사각 픽셀(fx=fy) 가정:

```
f  = (L/2) / tan(FOV/2)     # L=기준 축 길이(px), FOV=그 축의 화각(rad)
fx = fy = f
cx = W/2, cy = H/2
distCoeffs = [0, 0, 0, 0, 0]   # 왜곡 없음 가정(nominal이므로)
```

기준 축은 `--hfov-axis`로 고른다:
- `horizontal`(기본) — L=W(가로 픽셀), FOV=`--hfov-deg`를 수평화각으로 간주.
- `diagonal` — L=sqrt(W²+H²), FOV=`--hfov-deg`를 대각화각으로 간주.

**이번 스크립트의 기본 가정은 수평(horizontal)이다** — `docs/vision_aruco_branch.md` §1
지시대로. 이 카메라(CAM109-IMX708AF-75)의 HFOV 75°가 실제로 수평인지 대각인지는
`docs/vision_camera_bringup.md` "미해결로 남긴 판단" 절 기준 **아직 불일치가 안 풀렸다** —
이 스크립트는 그 논쟁을 해결하지 않고, 어느 쪽으로 가정했는지를 출력 yaml
`hfov_assumption`에 그대로 기록만 한다(실측 캘리브레이션 재개 시 검정 대상).

## 출력

`vision/calibration/<camera_id>/nominal.yaml` — `vision/tools/calib_analyze.py`가 만드는
`<calib_id>.yaml`과 같은 디렉터리를 공유한다(그 스크립트가 나중에 실측 캘리브레이션 결과를
같은 `vision/calibration/<camera_id>/` 아래 쓴다). 스키마는 그 파일의 관례(`schema_version`,
`camera_id`, `created_utc`, `git_commit`, `camera_matrix`, `dist_coeffs`)를 최대한 따르되,
nominal 고유 필드(`source`/`accuracy`/`not_for_closed_loop_30cm`/`hfov_assumption`)를 추가한다.
`camera_id` 기본값(`cam109-imx708af75`)은 `calib_analyze.py`의 `DEFAULT_CAMERA_ID`와 **반드시
동일하게 유지**한다(같은 카메라를 가리켜야 두 스크립트의 아티팩트가 같은 디렉터리에 쌓인다) —
`matplotlib` 의존 없이 상수만 참조하려고 그 모듈을 통째로 import하진 않았으니, 그 값을 바꾸면
여기도 같이 바꿀 것(`vision/tests/test_compute_nominal_intrinsics.py`가 두 값 일치를 회귀테스트로 고정).

## 검증

- 왕복(round-trip): 알려진 HFOV로 fx를 계산한 뒤 `atan2` 역산으로 원래 HFOV가 복원되는지
  (`vision/tests/test_compute_nominal_intrinsics.py`, `calib_analyze.py`의 `compute_hfov_deg`
  왕복 검증 패턴과 동일 방법론).
- yaml 스키마 round-trip(`yaml.safe_load` 후 필수 키 확인) — `calib_analyze.py`
  `test_yaml_artifact_round_trips_and_has_required_keys` 패턴 재사용.

## CLI

```bash
python -m vision.tools.compute_nominal_intrinsics \\
    --sensor-width-px 4608 --sensor-height-px 2592 \\
    --hfov-deg 75.0 --hfov-axis horizontal \\
    --camera-id cam109-imx708af75
```

(저장소 루트에서 `-m`으로 실행 — `vision.utils.logging`을 절대 import하므로 `python
vision/tools/compute_nominal_intrinsics.py` 직접 실행은 `ModuleNotFoundError: No module
named 'vision'`로 실패한다. `calib_analyze.py`도 같은 이유로 같은 제약이 있다.)

하드웨어 의존 없음(`numpy`/`PyYAML`뿐) — `vision/CLAUDE.md`의 `jsonl_view.py`/`calib_analyze.py`
예외 규칙과 동일하게 `.venv` 설치 + pytest 대상이다. **범위: Phase 1만.** ArUco 디코드
(Phase 2)·solvePnP(Phase 3)·파이프라인 통합(Phase 4)은 이 스크립트의 범위 밖이다.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import yaml

from vision.utils.logging import get_git_commit_hash

# ===========================================================================
# 상수 (매직넘버 금지 — 전부 이름+기본값+CLI로 노출, docs/vision_plan.md §7.3)
# ===========================================================================

# calib_analyze.py::DEFAULT_CAMERA_ID 와 동일하게 유지할 것(위 모듈 docstring 참조).
DEFAULT_CAMERA_ID = "cam109-imx708af75"

DEFAULT_SENSOR_WIDTH_PX = 4608
DEFAULT_SENSOR_HEIGHT_PX = 2592
DEFAULT_HFOV_DEG = 75.0
DEFAULT_HFOV_AXIS = "horizontal"  # "horizontal" | "diagonal" — docs/vision_aruco_branch.md §1 확정 가정
DEFAULT_OUT_CALIB_DIR = "vision/calibration"

_ZERO_DIST_COEFFS = (0.0, 0.0, 0.0, 0.0, 0.0)  # nominal(왜곡 미보정) 가정 — CLI로 바꿀 대상 아님


# ===========================================================================
# 순수 계산
# ===========================================================================


@dataclass
class NominalIntrinsics:
    """핀홀 모델, 정사각 픽셀 가정으로 산출한 nominal intrinsics."""

    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: tuple
    width_px: int
    height_px: int
    hfov_deg: float
    hfov_axis: str


def _reference_axis_length_px(width_px: int, height_px: int, axis: str) -> float:
    if axis == "horizontal":
        return float(width_px)
    if axis == "diagonal":
        return math.hypot(float(width_px), float(height_px))
    raise ValueError(f"hfov_axis는 'horizontal' 또는 'diagonal'이어야 함: {axis!r}")


def compute_focal_length_px(reference_length_px: float, fov_deg: float) -> float:
    """f = (L/2) / tan(FOV/2). L=기준 축 길이(px), FOV=그 축의 화각(deg)."""
    if fov_deg <= 0.0 or fov_deg >= 180.0:
        raise ValueError(f"fov_deg는 (0,180) 구간이어야 함: {fov_deg}")
    return (reference_length_px / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def compute_fov_deg(focal_length_px: float, reference_length_px: float) -> float:
    """`compute_focal_length_px`의 역함수 — atan2 역산으로 FOV 복원(왕복 검증용)."""
    return math.degrees(2.0 * math.atan2(reference_length_px / 2.0, focal_length_px))


def compute_nominal_intrinsics(
    width_px: int,
    height_px: int,
    hfov_deg: float,
    hfov_axis: str = DEFAULT_HFOV_AXIS,
) -> NominalIntrinsics:
    """핀홀 모델 nominal intrinsics: fx=fy=f(정사각 픽셀), cx=W/2, cy=H/2, dist=0."""
    ref_len = _reference_axis_length_px(width_px, height_px, hfov_axis)
    f = compute_focal_length_px(ref_len, hfov_deg)
    return NominalIntrinsics(
        fx=f,
        fy=f,
        cx=width_px / 2.0,
        cy=height_px / 2.0,
        dist_coeffs=_ZERO_DIST_COEFFS,
        width_px=width_px,
        height_px=height_px,
        hfov_deg=hfov_deg,
        hfov_axis=hfov_axis,
    )


# ===========================================================================
# 아티팩트 (build/save)
# ===========================================================================


def build_nominal_artifact(
    intr: NominalIntrinsics,
    camera_id: str,
    git_commit: str,
    created_utc: str,
) -> dict:
    """yaml로 그대로 dump 가능한 순수 dict(전부 python native 타입)를 만든다.

    필드는 `vision/tools/calib_analyze.py`가 만드는 `<calib_id>.yaml`과 최대한 같은
    이름/구조를 쓴다(`schema_version`/`camera_id`/`created_utc`/`git_commit`/
    `camera_matrix`/`dist_coeffs`/`image_size`) — 두 아티팩트가 같은
    `vision/calibration/<camera_id>/` 아래 공존할 것이므로."""
    K = [
        [intr.fx, 0.0, intr.cx],
        [0.0, intr.fy, intr.cy],
        [0.0, 0.0, 1.0],
    ]
    return {
        "schema_version": 1,
        "camera_id": camera_id,
        "created_utc": created_utc,
        "git_commit": git_commit,
        "source": "nominal_datasheet",
        "accuracy": "unverified",
        "not_for_closed_loop_30cm": True,
        "image_size": [int(intr.width_px), int(intr.height_px)],
        "camera_matrix": K,
        "dist_coeffs": list(intr.dist_coeffs),
        "hfov_assumption": {
            "axis": intr.hfov_axis,
            "hfov_deg": intr.hfov_deg,
            "note": (
                "실측 아님 — 데이터시트/실측 화각 근사(nominal). 이번 산출은 축="
                f"{intr.hfov_axis}로 가정하고 진행했다(docs/vision_aruco_branch.md §1 확정 "
                "지시). 이 카메라(CAM109-IMX708AF-75)의 HFOV 75°가 실제로 수평인지 대각인지는 "
                "docs/vision_camera_bringup.md '미해결로 남긴 판단' 절 기준 아직 불일치가 "
                "안 풀렸다 — 대각일 가능성이 있고, 실측 캘리브레이션 재개 시 검정 대상이다."
            ),
        },
    }


def save_nominal_artifact(artifact: dict, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(artifact, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return out_path


# ===========================================================================
# CLI
# ===========================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensor-width-px", type=int, default=DEFAULT_SENSOR_WIDTH_PX)
    parser.add_argument("--sensor-height-px", type=int, default=DEFAULT_SENSOR_HEIGHT_PX)
    parser.add_argument("--hfov-deg", type=float, default=DEFAULT_HFOV_DEG)
    parser.add_argument(
        "--hfov-axis", choices=["horizontal", "diagonal"], default=DEFAULT_HFOV_AXIS,
        help="HFOV 기준값이 어느 축인지(대각/수평 불일치, docs/vision_camera_bringup.md 참조). 기본 horizontal.",
    )
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID)
    parser.add_argument("--out-calib-dir", default=DEFAULT_OUT_CALIB_DIR, help="아티팩트 yaml 루트(카메라별 하위폴더 자동)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    try:
        intr = compute_nominal_intrinsics(
            args.sensor_width_px, args.sensor_height_px, args.hfov_deg, args.hfov_axis
        )
    except ValueError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1

    git_commit = get_git_commit_hash()
    created_utc = datetime.now(timezone.utc).isoformat()
    artifact = build_nominal_artifact(intr, args.camera_id, git_commit, created_utc)

    out_path = Path(args.out_calib_dir) / args.camera_id / "nominal.yaml"
    save_nominal_artifact(artifact, out_path)

    print(f"camera_id={args.camera_id} image_size=({intr.width_px}x{intr.height_px})")
    print(f"hfov_axis={intr.hfov_axis} hfov_deg={intr.hfov_deg}")
    print(f"fx=fy={intr.fx:.4f}px cx={intr.cx:.2f} cy={intr.cy:.2f} dist_coeffs={list(intr.dist_coeffs)}")
    print(f"아티팩트: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
