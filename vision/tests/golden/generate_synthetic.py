"""골든셋 합성 프레임 생성기 (vision_plan.md §7.9 "지금 당장 할 일" 7번).

⚠️ 실촬영 데이터 아님. 실기체/실카메라 데이터가 없는 상태(§ vision_status.md)에서
골든셋 **폴더 구조/스키마**가 실제로 동작함을 증명하기 위한 합성(synthetic) 플레이스홀더다.
`test_vertiport_cascade.py`가 이미 쓰는 방식과 동일하게 OpenCV 도형으로 타겟을 그린다.

이 스크립트는 pytest 대상이 아니다(파일명이 test_*가 아님) — 골든 프레임을 재생성하고
싶을 때(예: 타겟 스펙이 바뀌거나 altitude 티어를 늘릴 때) 수동으로 실행하는 도구다.

    python vision/tests/golden/generate_synthetic.py

실행하면 이 파일 옆의 <target>/<altitude>/frame_000.png + labels.json을 덮어쓴다.

**altitude 라벨(10m/20m/40m)은 스키마 자리표시자다.** vision_plan.md §4.1 GSD 표의 고도
구간 이름을 그대로 빌려 폴더명으로 쓰지만, 이 스크립트가 그리는 픽셀 크기는 그 표의
GSD 계산값을 그대로 옮긴 것이 **아니다** — §4.1 GSD 표는 화각 102° 가정인데 실제 장착된
카메라는 75°로 확인되어(docs/vision_status.md) 표 자체가 재검증 대기 중이다. 정밀 재현을
주장하면 거짓 정밀도가 되므로, 대신 "가까움/중간/멂"에 대응하는 스키마용 크기 계층만
잡았다. 실기체 데이터가 들어오면 이 자리에 진짜 촬영 프레임 + 실측 라벨로 교체한다.
"""
import json
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).parent


def _synthetic_vertiport(diameter_px: int, canvas: int) -> np.ndarray:
    """test_vertiport_cascade.py의 합성 패턴과 동일한 비율(흰 필드/검은 V/빨간 고리)을
    diameter_px로 스케일링. 흰 필드 반지름 r 기준 V 절반폭=0.6r, V 두께=0.2r,
    중앙 클리어 반지름=0.1467r, 빨간 고리 반지름=0.0867r·두께=0.02r (원 테스트의 r=150 비율 그대로)."""
    r = diameter_px / 2.0
    img = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    center = (canvas // 2, canvas // 2)

    cv2.circle(img, center, int(round(r)), (255, 255, 255), -1)

    v_offset = max(1, int(round(0.6 * r)))
    thickness = max(1, int(round(0.2 * r)))
    apex = (center[0], center[1] + v_offset)
    left = (center[0] - v_offset, center[1] - v_offset)
    right = (center[0] + v_offset, center[1] - v_offset)
    cv2.line(img, left, apex, (0, 0, 0), thickness)
    cv2.line(img, apex, right, (0, 0, 0), thickness)

    clear_r = max(1, int(round(0.1467 * r)))
    ring_r = max(1, int(round(0.0867 * r)))
    ring_th = max(1, int(round(0.02 * r)))
    cv2.circle(img, center, clear_r, (255, 255, 255), -1)
    cv2.circle(img, center, ring_r, (0, 0, 255), thickness=ring_th)
    return img


def _synthetic_distress(mat_size: int, canvas: int, box_ratio: float = 0.22) -> np.ndarray:
    """초록 사각 매트 + 중앙 흰 박스 (vision_plan.md §5.3). 배경은 자갈을 흉내 낸 무채색 회색."""
    img = np.full((canvas, canvas, 3), (60, 60, 60), dtype=np.uint8)

    hsv_green = np.array([[[60, 200, 180]]], dtype=np.uint8)
    bgr_green = tuple(int(v) for v in cv2.cvtColor(hsv_green, cv2.COLOR_HSV2BGR)[0, 0])

    c = canvas // 2
    half = mat_size // 2
    cv2.rectangle(img, (c - half, c - half), (c + half, c + half), bgr_green, -1)

    box_half = int(mat_size * box_ratio / 2)
    if box_half > 0:
        cv2.rectangle(img, (c - box_half, c - box_half), (c + box_half, c + box_half), (255, 255, 255), -1)
    return img


def _write_frame(target_dir: Path, image: np.ndarray, labels: dict) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target_dir / "frame_000.png"), image)
    (target_dir / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def generate() -> None:
    # --- vertiport (① — vertiport_coarse.yaml 3단 캐스케이드) ---
    # 10m/20m: 캐스케이드 3단 전부 확인 (실제 파이프라인으로 검증됨).
    # 40m: 흰 필드 자체는 걸리지만(candidates=1) black_v가 형상매칭에서 떨어져(rejected)
    #      최종 검출 0건 — 저해상(작은 픽셀 스케일)에서 morphology(kernel_size=5, 고정)가
    #      V 노치 주변 연결성을 깨서 생기는 현재 실제 동작이다. 검출기 파라미터는 이번 세션
    #      범위 밖(튜닝 금지)이므로 고치지 않고, 실측된 그대로를 "알려진 한계"로 골든셋에 고정한다.
    _write_frame(
        _ROOT / "vertiport" / "10m",
        _synthetic_vertiport(diameter_px=300, canvas=500),
        {
            "target": "vertiport",
            "altitude_label": "10m (근접, 스키마 자리표시자 — 주석 참조)",
            "preset": "vertiport_coarse.yaml",
            "note": "3단 캐스케이드(white_field->black_v->red_ring) 전부 확인되는 근접 케이스.",
            "frames": [
                {
                    "file": "frame_000.png",
                    "expect_num_detections": 1,
                    "expect_stage_meta": {
                        "white_field": {"candidates": 1},
                        "black_v": {"confirmed": 1, "rejected": 0},
                        "red_ring": {"confirmed": 1},
                    },
                    "known_limitation": False,
                }
            ],
        },
    )
    _write_frame(
        _ROOT / "vertiport" / "20m",
        _synthetic_vertiport(diameter_px=240, canvas=500),
        {
            "target": "vertiport",
            "altitude_label": "20m (중간, 스키마 자리표시자 — 주석 참조)",
            "preset": "vertiport_coarse.yaml",
            "note": "10m보다 작은 스케일에서도 3단 캐스케이드 전부 확인됨.",
            "frames": [
                {
                    "file": "frame_000.png",
                    "expect_num_detections": 1,
                    "expect_stage_meta": {
                        "white_field": {"candidates": 1},
                        "black_v": {"confirmed": 1, "rejected": 0},
                        "red_ring": {"confirmed": 1},
                    },
                    "known_limitation": False,
                }
            ],
        },
    )
    _write_frame(
        _ROOT / "vertiport" / "40m",
        _synthetic_vertiport(diameter_px=94, canvas=500),
        {
            "target": "vertiport",
            "altitude_label": "40m (원거리, 스키마 자리표시자 — 주석 참조)",
            "preset": "vertiport_coarse.yaml",
            "note": (
                "알려진 한계: white_field는 후보 1개를 내지만(경계 근처 원형도) black_v 형상매칭이 "
                "떨어져 최종 검출 0건. 저해상 스케일에서 고정 kernel_size=5 morphology가 V 노치 "
                "연결성을 깨는 것이 원인으로 보임(이 스크립트 상단 docstring 참조). 검출기 자체는 "
                "이번 세션에서 튜닝하지 않음 — 실기체 데이터 확보 후 재검증 대상."
            ),
            "frames": [
                {
                    "file": "frame_000.png",
                    "expect_num_detections": 0,
                    "expect_stage_meta": {
                        "white_field": {"candidates": 1},
                        "black_v": {"confirmed": 0, "rejected": 1},
                        "red_ring": {"confirmed": 0},
                    },
                    "known_limitation": True,
                }
            ],
        },
    )

    # --- distress (② — distress_coarse.yaml, 전용 모듈 없이 ColorFilter+RectDetector 조합) ---
    _write_frame(
        _ROOT / "distress" / "10m",
        _synthetic_distress(mat_size=260, canvas=320),
        {
            "target": "distress",
            "altitude_label": "10m (근접, 스키마 자리표시자)",
            "preset": "distress_coarse.yaml",
            "note": "초록 매트 + 중앙 흰 박스, 근접 스케일 — rect_detector 검출 확인.",
            "frames": [
                {
                    "file": "frame_000.png",
                    "expect_num_detections": 1,
                    "expect_stage_meta": {"rect_detector": {"candidates": 1}},
                    "known_limitation": False,
                }
            ],
        },
    )
    _write_frame(
        _ROOT / "distress" / "20m",
        _synthetic_distress(mat_size=180, canvas=320),
        {
            "target": "distress",
            "altitude_label": "20m (중간, 스키마 자리표시자)",
            "preset": "distress_coarse.yaml",
            "note": "중간 스케일에서도 검출 유지.",
            "frames": [
                {
                    "file": "frame_000.png",
                    "expect_num_detections": 1,
                    "expect_stage_meta": {"rect_detector": {"candidates": 1}},
                    "known_limitation": False,
                }
            ],
        },
    )
    _write_frame(
        _ROOT / "distress" / "40m",
        _synthetic_distress(mat_size=16, canvas=320),
        {
            "target": "distress",
            "altitude_label": "40m (원거리, 스키마 자리표시자)",
            "preset": "distress_coarse.yaml",
            "note": (
                "매트 픽셀 면적(16px 한 변, area=256)이 rect_detector min_area(300)보다 작아 "
                "검출 0건 — 원거리에서 타겟이 최소 검출 가능 크기 미만이 되는 물리적으로 타당한 "
                "케이스(§4.1 coarse 하한 고도 논지와 동일 방향). 파라미터 미변경."
            ),
            "frames": [
                {
                    "file": "frame_000.png",
                    "expect_num_detections": 0,
                    "expect_stage_meta": {"rect_detector": {"candidates": 0}},
                    "known_limitation": True,
                }
            ],
        },
    )

    # --- no_target (④ 단순착륙과 동일 조건 대응: 피듀셜 없는 평지 — 오탐 방지 회귀) ---
    blank = np.full((400, 400, 3), (70, 70, 70), dtype=np.uint8)
    rng = np.random.default_rng(seed=42)
    noise = rng.integers(-10, 10, size=blank.shape, dtype=np.int16)
    blank = np.clip(blank.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    _write_frame(
        _ROOT / "no_target",
        blank,
        {
            "target": "no_target",
            "altitude_label": "n/a — ④ 단순착륙(피듀셜 없음)과 동일 조건, vision_plan.md §5.6",
            "preset": "vertiport_coarse.yaml",
            "note": (
                "④ 단순 착륙은 비전 개입 없음(GPS+라이다만, §5.6)이라 전용 골든셋 대상이 아니다. "
                "대신 이 프레임은 '피듀셜 없는 평지에서 가장 정교한 캐스케이드(vertiport)조차 "
                "오탐하지 않는가'를 회귀로 고정한다."
            ),
            "frames": [
                {
                    "file": "frame_000.png",
                    "expect_num_detections": 0,
                    "expect_stage_meta": None,
                    "known_limitation": False,
                }
            ],
        },
    )

    print("골든셋 합성 프레임 생성 완료:", _ROOT)


if __name__ == "__main__":
    generate()
