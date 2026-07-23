"""
RPi5 체커보드 캘리브레이션 — 라이브 미리보기 + 촬영 도구 (libcamera/picamera2 정식 경로).

2026-07-23 카메라 브링업(`docs/vision_camera_bringup.md`) 성공으로 V4L2 raw 우회
(`vision/tools/rpi_capture.py`)가 아니라 libcamera/picamera2 정식 경로로 AF/AE/AWB를
실제로 제어할 수 있게 됐다. 이 도구는 그 경로 위에서 "초점/노출/화이트밸런스를 거리마다
한 번 확실히 고정하고, 고정이 실제로 유지되는지 검증한 뒤에만 촬영을 허용"하는
캘리브레이션 전용 촬영 흐름을 구현한다 — 초점/노출/WB가 샷마다 흔들리면 전체 촬영 세트가
서로 다른 인트린식을 가진 것이 돼 캘리브레이션 자체가 무의미해진다(과거 이미 겪은 실패).

## 아키텍처

- **단일 still 설정, 세션 내내 모드 전환 없음.** `Picamera2.create_still_configuration(
  main=..., lores=...)` 하나로 `main`(풀해상도, 기본 4608x2592)과 `lores`(미리보기, 기본
  1280x720)를 **동시에** 얻는다 — 모드 전환이 센서 크롭/비닝을 바꿔 인트린식이 흔들리는
  것을 원천 차단(세션 지시 사항).
- **포맷 "RGB888"을 요청해야 BGR 배열을 얻는다.** picamera2 자체 소스(`request.py`의
  `mode_lookup`/`FORMAT_TABLE`)로 실기체에서 직접 확인한 명명 관례 역전 — DRM fourcc
  "RGB888"의 실제 메모리 바이트 순서가 B,G,R이고 "BGR888"은 R,G,B다(널리 알려진 picamera2
  함정). cv2(BGR 기대)로 바로 쓰려면 반드시 `format="RGB888"`을 요청한다.
- **두 개의 HTTP 포트:** 영상은 기존에 테스트된 `vision/utils/stream.py`의 `MjpegStreamer`
  를 그대로 재사용(포트 8080, 오버레이가 그려진 lores 프레임을 push) — 새 MJPEG 서버를
  만들지 않는다(세션 지시). 이 파일 자체의 작은 컨트롤 서버는 포트 8081에서 UI 페이지 +
  촬영/잠금 엔드포인트만 서빙하고, 그 페이지는 `<img src="http://<host>:8080/stream">`로
  8080 스트림을 그대로 embed한다(순수 HTML, `vision/tools/rpi_capture.py`와 동일한
  http.server+폼 패턴, 새 JS 프레임워크 없음).
- **컨트롤 잠금 절차(거리가 바뀔 때마다 반복):**
  1. `LensPosition`(0~15 디옵터, 서드파티 클론 실측 하드클램프 — 32 상한 가정 금지)을
     스윕하며 중앙 크롭 라플라시안 분산이 최고인 지점을 찾아 `AfMode=Manual`로 고정.
  2. `AeEnable=True`/`AwbEnable=True`로 잠깐 자동 정착시킨 뒤 `ExposureTime`/
     `AnalogueGain`/`ColourGains`를 읽어 그 값 그대로 `AeEnable=False`/`AwbEnable=False`로
     고정.
  3. 고정 직후 **최소 20프레임 창**으로 값이 실제로 유지되는지 검증(`_locks_held_within_tolerance`)
     — 컨트롤 적용에 4~6프레임이 걸리고 값 자체가 하드웨어 양자화되므로(노출 8000→7993,
     게인 1.0→1.123) 정확 일치가 아니라 관용 범위로 판정한다. 유지 안 되면 잠금을
     실패 처리하고 그 거리의 촬영을 거부한다.
  4. 성공한 잠금값 전체(스윕 표 포함)를 `<out>/<set>/<distance>m/_lock.json`에, 매 샷의
     실제 적용값은 각 샷의 JSON 사이드카에 기록.
- **촬영 흐름:** `main` 프레임 → `findChessboardCorners`(ADAPTIVE_THRESH|NORMALIZE_IMAGE)
  → 성공 시 `cornerSubPix` → 보드 영역 라플라시안 분산으로 모션블러 판정(거리 잠금 시
  기록해 둔 기준 선명도 대비 비율 미달이면 블러로 거부) → 통과 시에만 PNG+JSON 저장,
  샷 인덱스 전진, 커버리지 맵 갱신. **실패(패턴 미검출/블러)면 인덱스를 전진하지 않고
  이유를 보여준다** — 사용자가 그냥 다시 찍으면 된다.
- **자동촬영:** lores 프레임에서 라이브 코너 검출이 되고(가까운 거리에서만 현실적으로
  성공), 최근 5프레임 코너 이동량이 다 2px 미만이고, 보드 중심이 타겟존 안에 있으면
  짧은 카운트다운 후 자동으로 `main` 프레임을 촬영한다. 원거리(2.5m, 정사각형이 lores
  기준 ~11px)에서는 라이브 검출이 사실상 항상 실패할 것으로 예상되므로 — 이 도구는
  **라이브 검출 성공 여부에 어떤 기능도 게이팅하지 않는다.** 타겟존 박스(항상 그려짐)가
  주된 조준 보조 수단이고, 진짜 판정은 촬영 시점의 풀해상도 프레임에서만 이뤄진다.
  수동 촬영은 라이브 검출 여부와 무관하게 항상 가능.
- **커버리지 맵:** 승인된(성공한) 촬영마다 코너들을 프레임 전체에 걸친 12×8 격자에
  비닝해 어느 빈이 아직 비어 있는지 미리보기 인셋으로 보여준다 — 왜곡계수는 주변부
  코너로 결정되므로 이게 가장 중요한 UI 요소(세션 지시).

## 실행

```bash
source /home/suri/local-libcamera/env.sh
$PICAM_PYTHON vision/tools/calib_capture.py --out-dir /home/suri/local-libcamera-src/calib
```
브라우저에서 `http://<라즈베리파이 tailscale IP>:8081/` 을 연다(영상은 페이지 안에
자동으로 embed됨, 8080에 따로 접속할 필요 없음).

## 순수 함수 vs 하드웨어 함수

이 파일 하단(`CalibSession` 이하)은 `Picamera2`/`libcamera.controls`가 필요한 하드웨어
전용 코드다(단위테스트 대상 아님 — `vision/CLAUDE.md`의 tools/ 예외 규칙). 그 위 절반
(샷 목록 생성·좌표/픽셀 변환·커버리지 비닝·코너 안정성·블러 판정)은 numpy/cv2만 쓰는
순수 함수라 하드웨어 없이 `vision/tests/test_calib_capture.py`에서 검증한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - 항상 있어야 하는 환경에서만 실행되는 도구
    cv2 = None

from vision.utils.stream import MjpegStreamer
from vision.tools.rpi_capture import (
    crop_center_fraction,
    laplacian_sharpness,
    pick_best_focus,
)

try:
    from picamera2 import Picamera2
    from libcamera import controls as libcamera_controls
except ImportError:  # pragma: no cover - picam-venv 밖(개발 호스트/CI)에서는 순수 함수만 씀
    Picamera2 = None
    libcamera_controls = None


# ===========================================================================
# 상수
# ===========================================================================

DEFAULT_PATTERN_SIZE = (9, 6)          # 10x7 정사각형 보드 -> 코너 9x6 (내부 교차점)
DEFAULT_SQUARE_SIZE_MM_A4 = 28.0       # A4 인쇄판 실측(2026-07-23 재실측 정정, 이전 28.1 오기)
DEFAULT_SQUARE_SIZE_MM_TABLET = 20.0   # 태블릿 화면 표시판

DEFAULT_MAIN_SIZE = (4608, 2592)
DEFAULT_LORES_SIZE = (1280, 720)

COVERAGE_COLS = 12
COVERAGE_ROWS = 8

# 지상폭 계수 — 카메라 실측 수평화각 67.5도에서 유도: gw(d)[m] = 2*d*tan(67.5/2 deg) ~= 1.338*d
# (세션 지시가 이 상수를 그대로 명시함 — 재계산 없이 그대로 채택)
GROUND_WIDTH_COEFF_PER_M = 1.338

DEFAULT_BLUR_RATIO = 0.4       # 거리 잠금 시 기준 선명도 대비 이 비율 미만이면 블러로 거부
DEFAULT_LOCK_VERIFY_FRAMES = 20  # 컨트롤 고정 검증 최소 프레임 창(세션 지시)
DEFAULT_AUTO_CAPTURE_DELAY_S = 1.2
DEFAULT_STABILITY_MAX_PX = 2.0
DEFAULT_STABILITY_WINDOW = 5

SET_A_DISTANCES_M: Tuple[float, ...] = (0.5, 0.7, 1.2)
SET_B_DISTANCE_M = 2.5


# ===========================================================================
# 순수 함수 — 샷 목록 생성
# ===========================================================================


@dataclass(frozen=True)
class ShotSpec:
    """샷 하나의 명세 — 순수 데이터, 하드웨어 의존 없음."""

    set_name: str        # "A" 또는 "B"
    distance_m: float
    seq: int              # 이 거리 안에서 1부터 시작하는 순번
    total_in_distance: int
    zone: str              # "centre"/"left"/...·"grid"·"corner-roll"
    u: float               # 0~1, 프레임 내 보드 중심 목표 위치(가로)
    v: float               # 0~1, 세로
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    label_ko: str          # HTML 컨트롤 페이지/JSON 사이드카용 한국어 지시문
    label_en: str          # 프레임에 굽는 cv2 오버레이용 ASCII 지시문(위 "영문 버전" 절 참조)

    @property
    def dir_name(self) -> str:
        return f"{self.distance_m:g}m"

    @property
    def file_stem(self) -> str:
        return f"{self.seq:03d}"


# (zone, u, v, yaw_deg, pitch_deg, roll_deg) — 거리마다 그대로 반복되는 15컷 세트A 목록.
# 세션 지시 원문의 순서·값을 그대로 따른다.
_SET_A_SHOTS: Tuple[Tuple[str, float, float, float, float, float], ...] = (
    ("centre", 0.5, 0.5, 0.0, 0.0, 0.0),      # centre[frontal]
    ("centre", 0.5, 0.5, 35.0, 0.0, 0.0),     # centre[yaw+35]
    ("centre", 0.5, 0.5, 0.0, 35.0, 0.0),     # centre[pitch+35]
    ("left", 0.2, 0.5, -30.0, 0.0, 0.0),      # left[yaw-30]
    ("right", 0.8, 0.5, 30.0, 0.0, 0.0),      # right[yaw+30]
    ("top", 0.5, 0.2, 0.0, -30.0, 0.0),       # top[pitch-30]
    ("bottom", 0.5, 0.8, 0.0, 30.0, 0.0),     # bottom[pitch+30]
    ("TL", 0.2, 0.2, -25.0, -25.0, 0.0),      # TL[yaw-25,pitch-25]
    ("TR", 0.8, 0.2, 25.0, -25.0, 0.0),       # TR[yaw+25,pitch-25]
    ("BL", 0.2, 0.8, -25.0, 25.0, 0.0),       # BL[yaw-25,pitch+25]
    ("BR", 0.8, 0.8, 25.0, 25.0, 0.0),        # BR[yaw+25,pitch+25]
    ("centre", 0.5, 0.5, 0.0, 0.0, 45.0),     # centre[frontal,roll45]
    ("TL", 0.2, 0.2, -40.0, 0.0, 30.0),       # TL[yaw-40,roll30]
    ("BR", 0.8, 0.8, 0.0, 40.0, -30.0),       # BR[pitch+40,roll-30]
    ("centre", 0.5, 0.5, 20.0, -20.0, 15.0),  # centre[yaw+20,pitch-20,roll15]
)

# 세트B(2.5m) 격자: u x v = 4 x 3 = 12위치, 위치마다 3틸트.
# "3 tilts (~=18°, yaw~=38°, pitch~=38°)" 해석: 1번째는 요/피치 각 ~18도의 완만한 대각
# 기울임(단일축 큰 각도 두 개와 구분되는 3번째 변형으로 채택), 2/3번째는 지시문 그대로
# 요잉/피칭 단일축 ~38도. (근거는 이 파일 docstring 및 vision/CLAUDE.md에 기록)
_SET_B_GRID_U: Tuple[float, ...] = (0.15, 0.38, 0.62, 0.85)
_SET_B_GRID_V: Tuple[float, ...] = (0.2, 0.5, 0.8)
_SET_B_TILTS: Tuple[Tuple[float, float, float], ...] = (
    (18.0, 18.0, 0.0),
    (38.0, 0.0, 0.0),
    (0.0, 38.0, 0.0),
)
# 격자 네 모서리에 롤 30~60도 변형 4컷 추가.
_SET_B_ROLL_CORNERS: Tuple[Tuple[float, float, float], ...] = (
    (_SET_B_GRID_U[0], _SET_B_GRID_V[0], 30.0),
    (_SET_B_GRID_U[-1], _SET_B_GRID_V[0], 40.0),
    (_SET_B_GRID_U[0], _SET_B_GRID_V[-1], 50.0),
    (_SET_B_GRID_U[-1], _SET_B_GRID_V[-1], 60.0),
)


def zone_label_ko(u: float, v: float) -> str:
    """(u,v) 위치 -> 한국어 위치 이름. edge=0.2/0.8, corner=둘 다. 그 외는 격자 좌표 그대로."""
    near = lambda a, b: abs(a - b) < 1e-6
    if near(u, 0.5) and near(v, 0.5):
        return "정중앙"
    if near(v, 0.2) and near(u, 0.2):
        return "좌상단(TL)"
    if near(v, 0.2) and near(u, 0.8):
        return "우상단(TR)"
    if near(v, 0.8) and near(u, 0.2):
        return "좌하단(BL)"
    if near(v, 0.8) and near(u, 0.8):
        return "우하단(BR)"
    if near(u, 0.2):
        return "좌측"
    if near(u, 0.8):
        return "우측"
    if near(v, 0.2):
        return "상단"
    if near(v, 0.8):
        return "하단"
    return f"격자(u={u:.2f},v={v:.2f})"


def format_instruction_ko(yaw_deg: float, pitch_deg: float, roll_deg: float) -> str:
    """(yaw,pitch,roll) -> 한국어 지시문. 전부 0이면 '정면(회전 없음)'."""
    parts = []
    if abs(yaw_deg) > 1e-9:
        parts.append(f"요잉 {yaw_deg:+.0f}°")
    if abs(pitch_deg) > 1e-9:
        parts.append(f"피칭 {pitch_deg:+.0f}°")
    if abs(roll_deg) > 1e-9:
        parts.append(f"롤 {roll_deg:+.0f}°")
    if not parts:
        return "정면(회전 없음)"
    return ", ".join(parts)


# --- 영문(ASCII) 버전 — 프레임에 직접 굽는 오버레이(cv2.putText) 전용 ---
#
# **실기체로 확인된 문제(2026-07-23):** cv2.putText는 Hershey 벌터 폰트만 쓰는데 이 폰트가
# 한글 글리프를 지원하지 않는다 — 실행해 보면 한글 문자열이 전부 "?"로 깨져 나온다. 이
# RPi에는 한글 지원 폰트가 시스템에 전혀 없고(`fc-list :lang=ko` 결과 0개, DejaVu만 설치됨),
# 이 세션의 하드 리밋상 새 apt 패키지 설치가 금지돼 있어 폰트를 새로 깔 수 없다. 그래서
# **프레임에 굽는 HUD 텍스트만** 영문으로 렌더링한다 — `label_ko`/`zone_label_ko`(HTML
# 컨트롤 페이지, JSON 사이드카)는 그대로 두고 안 건드린다. 브라우저는 자체 폰트로 UTF-8
# 한글을 정상 렌더링하므로(실측 확인됨) 사용자가 실제로 지시문을 읽는 주 경로(HTML 페이지)는
# 이 문제의 영향을 받지 않는다.
def zone_label_en(u: float, v: float) -> str:
    near = lambda a, b: abs(a - b) < 1e-6
    if near(u, 0.5) and near(v, 0.5):
        return "CENTER"
    if near(v, 0.2) and near(u, 0.2):
        return "TL"
    if near(v, 0.2) and near(u, 0.8):
        return "TR"
    if near(v, 0.8) and near(u, 0.2):
        return "BL"
    if near(v, 0.8) and near(u, 0.8):
        return "BR"
    if near(u, 0.2):
        return "LEFT"
    if near(u, 0.8):
        return "RIGHT"
    if near(v, 0.2):
        return "TOP"
    if near(v, 0.8):
        return "BOTTOM"
    return f"GRID(u={u:.2f},v={v:.2f})"


def format_instruction_en(yaw_deg: float, pitch_deg: float, roll_deg: float) -> str:
    parts = []
    if abs(yaw_deg) > 1e-9:
        parts.append(f"YAW{yaw_deg:+.0f}")
    if abs(pitch_deg) > 1e-9:
        parts.append(f"PITCH{pitch_deg:+.0f}")
    if abs(roll_deg) > 1e-9:
        parts.append(f"ROLL{roll_deg:+.0f}")
    if not parts:
        return "FRONTAL(no rotation)"
    return " ".join(parts)


def _make_shots_for_distance(
    set_name: str, distance_m: float, tuples: Sequence[Tuple[str, float, float, float, float, float]]
) -> List[ShotSpec]:
    total = len(tuples)
    shots = []
    for i, (zone, u, v, yaw, pitch, roll) in enumerate(tuples, start=1):
        label_ko = f"{zone_label_ko(u, v)} · {format_instruction_ko(yaw, pitch, roll)}"
        label_en = f"{zone_label_en(u, v)} {format_instruction_en(yaw, pitch, roll)}"
        shots.append(ShotSpec(set_name, distance_m, i, total, zone, u, v, yaw, pitch, roll, label_ko, label_en))
    return shots


def _build_set_b_tuples() -> List[Tuple[str, float, float, float, float, float]]:
    tuples: List[Tuple[str, float, float, float, float, float]] = []
    for u in _SET_B_GRID_U:
        for v in _SET_B_GRID_V:
            for yaw, pitch, roll in _SET_B_TILTS:
                tuples.append(("grid", u, v, yaw, pitch, roll))
    for u, v, roll in _SET_B_ROLL_CORNERS:
        tuples.append(("corner-roll", u, v, 0.0, 0.0, roll))
    return tuples


def build_shot_list(
    set_a_distances: Sequence[float] = SET_A_DISTANCES_M,
    set_b_distance: float = SET_B_DISTANCE_M,
) -> List[ShotSpec]:
    """전체 촬영 세션의 샷 목록(순서 고정) — 세트A(거리마다 15컷) + 세트B(40컷).

    순수 함수 — 하드웨어/파일시스템 접근 없음. UI가 이 목록의 순서 그대로 사용자를 이끈다.
    """
    shots: List[ShotSpec] = []
    for d in set_a_distances:
        shots.extend(_make_shots_for_distance("A", d, _SET_A_SHOTS))
    shots.extend(_make_shots_for_distance("B", set_b_distance, _build_set_b_tuples()))
    return shots


# ===========================================================================
# 순수 함수 — 보드 치수 / 타겟존 픽셀 변환
# ===========================================================================


def board_dimensions_mm(pattern_size: Tuple[int, int], square_size_mm: float) -> Tuple[float, float]:
    """체스보드 인쇄 영역 실치수(mm) — (가로, 세로). pattern_size=(cols,rows)는
    findChessboardCorners 관례(내부 교차점 개수) — 사각형 개수는 각 축 +1."""
    cols, rows = pattern_size
    return (cols + 1) * square_size_mm, (rows + 1) * square_size_mm


def target_zone_size_px(
    distance_m: float,
    frame_w: int,
    board_width_mm: float,
    board_height_mm: float,
    gw_coeff: float = GROUND_WIDTH_COEFF_PER_M,
) -> Tuple[float, float]:
    """이 거리에서 보드가 프레임에 차지할 것으로 기대되는 (폭,높이) px.

    폭 비율 = (board_width_mm/1000) / (gw_coeff * distance_m) — 프레임 폭에 대한 분수.
    높이는 보드 실측 종횡비를 그대로 적용(정사각 픽셀 가정 — 이 센서는 별도 픽셀종횡비
    보정이 필요하다는 근거가 없어 가장 단순한 가정 채택).
    """
    if distance_m <= 0:
        raise ValueError(f"target_zone_size_px: distance_m은 양수여야 한다 (입력: {distance_m})")
    if board_width_mm <= 0 or board_height_mm <= 0:
        raise ValueError("target_zone_size_px: board_width_mm/board_height_mm은 양수여야 한다")
    width_frac = (board_width_mm / 1000.0) / (gw_coeff * distance_m)
    width_px = width_frac * frame_w
    height_px = width_px * (board_height_mm / board_width_mm)
    return width_px, height_px


def zone_to_pixel_rect(
    u: float,
    v: float,
    frame_w: int,
    frame_h: int,
    distance_m: float,
    board_width_mm: float,
    board_height_mm: float,
) -> Tuple[int, int, int, int]:
    """(u,v) 중심 위치 + 거리 + 보드치수 -> 타겟존 박스 (x0,y0,x1,y1), 프레임 경계로 클립.

    u,v는 0~1(0.5,0.5=중앙). 순수 함수 — 실제 프레임 크기(main 4608x2592 또는 lores
    1280x720 어느 쪽에 대해서도 동일한 분수 비율로 정확히 스케일된다."""
    if not (0.0 <= u <= 1.0):
        raise ValueError(f"zone_to_pixel_rect: u는 0~1 사이여야 한다 (입력: {u})")
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"zone_to_pixel_rect: v는 0~1 사이여야 한다 (입력: {v})")
    w_px, h_px = target_zone_size_px(distance_m, frame_w, board_width_mm, board_height_mm)
    cx, cy = u * frame_w, v * frame_h
    x0, y0 = cx - w_px / 2.0, cy - h_px / 2.0
    x1, y1 = cx + w_px / 2.0, cy + h_px / 2.0
    x0c, y0c = max(0.0, x0), max(0.0, y0)
    x1c, y1c = min(float(frame_w), x1), min(float(frame_h), y1)
    return int(round(x0c)), int(round(y0c)), int(round(x1c)), int(round(y1c))


def point_in_rect(x: float, y: float, rect: Tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


# ===========================================================================
# 순수 함수 — 커버리지 비닝(12x8 격자)
# ===========================================================================


def new_coverage_grid(cols: int = COVERAGE_COLS, rows: int = COVERAGE_ROWS) -> np.ndarray:
    return np.zeros((rows, cols), dtype=bool)


def corner_bin_indices(
    corners: np.ndarray, frame_w: int, frame_h: int, cols: int = COVERAGE_COLS, rows: int = COVERAGE_ROWS
) -> List[Tuple[int, int]]:
    """코너 좌표들(Nx2, 픽셀) -> (col,row) 빈 인덱스 목록(중복 제거, 정렬됨)."""
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        return []
    cols_idx = np.clip((pts[:, 0] / frame_w * cols).astype(int), 0, cols - 1)
    rows_idx = np.clip((pts[:, 1] / frame_h * rows).astype(int), 0, rows - 1)
    return sorted(set(zip(cols_idx.tolist(), rows_idx.tolist())))


def update_coverage_grid(coverage: np.ndarray, corners: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """coverage(bool ndarray, shape (rows,cols))를 제자리 갱신하고 그대로 반환."""
    rows, cols = coverage.shape
    for c, r in corner_bin_indices(corners, frame_w, frame_h, cols, rows):
        coverage[r, c] = True
    return coverage


def coverage_fill_fraction(coverage: np.ndarray) -> float:
    return float(coverage.sum()) / float(coverage.size)


# ===========================================================================
# 순수 함수 — 코너 안정성(자동촬영 게이팅)
# ===========================================================================


def max_corner_movement_px(prev: np.ndarray, curr: np.ndarray) -> float:
    """두 코너 배열(같은 순서/개수 가정) 간 최대 유클리드 이동량. 개수가 다르면 ValueError."""
    p = np.asarray(prev, dtype=np.float64).reshape(-1, 2)
    c = np.asarray(curr, dtype=np.float64).reshape(-1, 2)
    if p.shape != c.shape:
        raise ValueError(f"max_corner_movement_px: 코너 개수 불일치 (prev={p.shape}, curr={c.shape})")
    if p.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(c - p, axis=1)))


def corners_are_stable(
    history: Sequence[np.ndarray], max_movement_px: float = DEFAULT_STABILITY_MAX_PX
) -> bool:
    """최근 N프레임 코너 이력이 전부 서로 max_movement_px 미만으로 움직였으면 안정.

    - 2프레임 미만이면 판단 불가 -> False(안정으로 오판하지 않음, 세션 지시대로 보수적).
    - 프레임 간 코너 개수가 다르면(검출 자체가 불안정) 불안정으로 판단.
    - 연속 프레임 쌍뿐 아니라 이력 전체 내 임의 두 프레임 간 최대 이동량을 본다 — 5프레임
      동안 완만히 표류(각 스텝은 2px 미만이지만 누적은 5px)하는 경우도 불안정으로 잡기 위함.
    """
    if len(history) < 2:
        return False
    shapes = {np.asarray(h).reshape(-1, 2).shape for h in history}
    if len(shapes) != 1:
        return False
    arrs = [np.asarray(h, dtype=np.float64).reshape(-1, 2) for h in history]
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            if np.max(np.linalg.norm(arrs[j] - arrs[i], axis=1)) >= max_movement_px:
                return False
    return True


# ===========================================================================
# 순수 함수 — 블러(모션블러) 판정
# ===========================================================================


def board_bounding_box(
    corners: np.ndarray, frame_w: int, frame_h: int, margin_frac: float = 0.15
) -> Tuple[int, int, int, int]:
    """검출된 코너들을 감싸는 바운딩박스(여유 margin_frac 추가), 프레임 경계로 클립."""
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        raise ValueError("board_bounding_box: 빈 코너 배열")
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    w, h = x1 - x0, y1 - y0
    x0 -= w * margin_frac
    x1 += w * margin_frac
    y0 -= h * margin_frac
    y1 += h * margin_frac
    x0c, y0c = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1c, y1c = min(frame_w, int(np.ceil(x1))), min(frame_h, int(np.ceil(y1)))
    return x0c, y0c, x1c, y1c


def board_region_sharpness(bgr: np.ndarray, corners: np.ndarray) -> float:
    """검출된 코너 바운딩박스 영역에 대한 라플라시안 분산(선명도) — 모션블러 판정용.
    rpi_capture.py의 laplacian_sharpness()를 그대로 재사용(재발명 안 함)."""
    if cv2 is None:
        raise RuntimeError("board_region_sharpness: cv2가 필요하다")
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = board_bounding_box(corners, w, h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("board_region_sharpness: 유효하지 않은 보드 영역(면적 0)")
    return laplacian_sharpness(bgr[y0:y1, x0:x1])


def is_capture_blurry(sample_sharpness: float, reference_sharpness: float, ratio: float = DEFAULT_BLUR_RATIO) -> bool:
    """샘플 선명도가 기준(거리 잠금 시점에 측정한 선명도) 대비 ratio 미만이면 블러로 판정.

    절대 임계값(조명/게인마다 스케일이 다름)이 아니라 **같은 세션의 자기 기준 대비 상대
    비율**을 쓴다 — 절대 매직넘버보다 조명 조건 변화에 강건하다(근거는 이 파일 docstring).
    reference_sharpness가 0 이하(측정 실패 등)면 판단 불가 -> 블러 아님으로 통과시킨다
    (기준 자체가 없으면 블러라고 단정할 근거도 없음).
    """
    if reference_sharpness <= 0:
        return False
    return sample_sharpness < ratio * reference_sharpness


# ===========================================================================
# 순수 함수 — 컨트롤 고정 유지 검증(메타데이터 프레임 목록만 입력, 하드웨어 호출 없음)
# ===========================================================================


def locks_held_within_tolerance(
    frames_meta: Sequence[Dict],
    lens_target: float,
    exposure_target: float,
    gain_target: float,
    colour_gains_target: Tuple[float, float],
    lens_tol: float = 0.5,
    exposure_tol_frac: float = 0.12,
    gain_tol_frac: float = 0.15,
    colour_gain_tol: float = 0.2,
) -> Tuple[bool, List[str]]:
    """캡처된 메타데이터 프레임 목록(각 dict — LensPosition/ExposureTime/AnalogueGain/
    ColourGains 키)이 목표값 근방에서 유지됐는지 확인.

    **정확 일치로 판정하지 않는다** — 컨트롤 적용에 4~6프레임이 걸리고 값 자체가 하드웨어
    양자화되므로(노출 8000 요청 -> 실제 7993, 게인 1.0 -> 1.123) 관용 범위(tol)로 비교한다
    (세션 지시 사실 반영). 순수 함수 — 하드웨어 호출 없이 합성 메타데이터 리스트로
    단위테스트 가능.
    """
    reasons: List[str] = []
    lens_vals = [f.get("LensPosition") for f in frames_meta if f.get("LensPosition") is not None]
    exp_vals = [f.get("ExposureTime") for f in frames_meta if f.get("ExposureTime") is not None]
    gain_vals = [f.get("AnalogueGain") for f in frames_meta if f.get("AnalogueGain") is not None]
    cg_vals = [f.get("ColourGains") for f in frames_meta if f.get("ColourGains") is not None]

    if any(abs(v - lens_target) > lens_tol for v in lens_vals):
        reasons.append(f"LensPosition 드리프트(목표={lens_target}, 허용={lens_tol})")
    if any(abs(v - exposure_target) > exposure_target * exposure_tol_frac for v in exp_vals):
        reasons.append(f"ExposureTime 드리프트(목표={exposure_target})")
    if any(abs(v - gain_target) > gain_target * gain_tol_frac for v in gain_vals):
        reasons.append(f"AnalogueGain 드리프트(목표={gain_target})")
    for cg in cg_vals:
        if abs(cg[0] - colour_gains_target[0]) > colour_gain_tol or abs(cg[1] - colour_gains_target[1]) > colour_gain_tol:
            reasons.append(f"ColourGains 드리프트(목표={colour_gains_target})")
            break
    return (len(reasons) == 0, reasons)


# ===========================================================================
# 오버레이 렌더링 (cv2 필요, 하드웨어 없이도 합성 프레임으로 동작 확인 가능)
# ===========================================================================


def draw_zone_rect(
    frame_bgr: np.ndarray, rect: Tuple[int, int, int, int], color: Tuple[int, int, int], thickness: int = 2
) -> np.ndarray:
    x0, y0, x1, y1 = rect
    if x1 > x0 and y1 > y0:
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, thickness)
    return frame_bgr


def draw_coverage_inset(
    frame_bgr: np.ndarray, coverage: np.ndarray, origin: Tuple[int, int] = (10, 10), cell_px: int = 12
) -> np.ndarray:
    rows, cols = coverage.shape
    x0, y0 = origin
    for r in range(rows):
        for c in range(cols):
            color = (0, 210, 0) if coverage[r, c] else (70, 70, 70)
            p0 = (x0 + c * cell_px, y0 + r * cell_px)
            p1 = (x0 + (c + 1) * cell_px - 1, y0 + (r + 1) * cell_px - 1)
            cv2.rectangle(frame_bgr, p0, p1, color, -1)
    cv2.rectangle(frame_bgr, (x0 - 2, y0 - 2), (x0 + cols * cell_px + 2, y0 + rows * cell_px + 2), (255, 255, 255), 1)
    return frame_bgr


def _put_lines(frame_bgr: np.ndarray, lines: Sequence[str], org: Tuple[int, int], color=(255, 255, 255), scale=0.5):
    x, y = org
    for i, line in enumerate(lines):
        yy = y + int(i * 20 * scale * 2)
        cv2.putText(frame_bgr, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return frame_bgr


def render_preview_overlay(
    lores_bgr: np.ndarray,
    shots_for_distance: Sequence[ShotSpec],
    completed_mask: Sequence[bool],
    current_shot: ShotSpec,
    board_width_mm: float,
    board_height_mm: float,
    coverage: np.ndarray,
    lock_ok: bool,
    lock_values: Optional[Dict],
    live_corners: Optional[np.ndarray] = None,
    countdown_s: Optional[float] = None,
    last_result_text: Optional[str] = None,
) -> np.ndarray:
    """lores 프레임 위에 전체 오버레이를 그린다. 반환값은 lores_bgr을 제자리 수정한 참조.

    **텍스트는 전부 ASCII로 그린다** — cv2.putText(Hershey 폰트)가 한글 글리프를 지원하지
    않아(실기체 확인, 이 파일 상단 "영문 버전" 절 참조) `current_shot.label_en`/영문 상태
    문구를 쓴다. `last_result_text`도 호출자가 ASCII로 넘겨야 한다(한국어 원문은 HTML
    컨트롤 페이지/JSON 사이드카가 따로 담당)."""
    h, w = lores_bgr.shape[:2]
    frame = lores_bgr

    # 1) 남은/완료 존 박스(얇게)
    for spec, done in zip(shots_for_distance, completed_mask):
        rect = zone_to_pixel_rect(spec.u, spec.v, w, h, spec.distance_m, board_width_mm, board_height_mm)
        color = (0, 160, 0) if done else (90, 90, 90)
        draw_zone_rect(frame, rect, color, 1)

    # 2) 현재 타겟존(굵게, 밝은 색)
    cur_rect = zone_to_pixel_rect(
        current_shot.u, current_shot.v, w, h, current_shot.distance_m, board_width_mm, board_height_mm
    )
    draw_zone_rect(frame, cur_rect, (0, 255, 255), 3)

    # 3) 라이브 검출 코너(성공 시)
    if live_corners is not None and len(live_corners) > 0:
        pts = np.asarray(live_corners, dtype=np.float64).reshape(-1, 2)
        for x, y in pts:
            cv2.circle(frame, (int(round(x)), int(round(y))), 3, (255, 0, 255), -1)

    # 4) 커버리지 인셋(우상단)
    inset_origin = (w - COVERAGE_COLS * 12 - 14, 14)
    draw_coverage_inset(frame, coverage, inset_origin, cell_px=12)

    # 5) 텍스트 (ASCII 전용 — docstring 참조)
    lock_line = "LOCK: none - must lock before capture"
    if lock_ok and lock_values:
        lock_line = (
            f"LOCK: lens={lock_values['lens_position']:.2f} "
            f"exp={lock_values['exposure_time']} gain={lock_values['analogue_gain']:.2f} "
            f"wb=({lock_values['colour_gains'][0]:.2f},{lock_values['colour_gains'][1]:.2f})"
        )
    lines = [
        f"[{current_shot.set_name}] {current_shot.distance_m:g}m  SHOT {current_shot.seq}/{current_shot.total_in_distance}",
        current_shot.label_en,
        lock_line,
        f"COVERAGE {coverage_fill_fraction(coverage) * 100:.0f}%",
    ]
    if countdown_s is not None:
        lines.append(f"AUTO-CAPTURE in {countdown_s:.1f}s")
    if last_result_text:
        lines.append(last_result_text)
    _put_lines(frame, lines, (10, h - 20 * len(lines) - 6))
    return frame


# ===========================================================================
# 하드웨어 세션 (Picamera2) — 단위테스트 대상 아님
# ===========================================================================


class CalibSession:
    """Picamera2 카메라 + 세션 상태(샷 진행/잠금/커버리지)를 묶는다.

    모든 카메라 접근(`_lock`으로 직렬화)은 미리보기 루프 스레드와 HTTP 핸들러 스레드
    양쪽에서 올 수 있어 `vision/tools/rpi_capture.py`의 `_CaptureSession`과 동일하게
    단일 락으로 직렬화한다(새 패턴 발명 안 함)."""

    def __init__(
        self,
        out_root: Path,
        pattern_size: Tuple[int, int] = DEFAULT_PATTERN_SIZE,
        square_size_mm: float = DEFAULT_SQUARE_SIZE_MM_A4,
        main_size: Tuple[int, int] = DEFAULT_MAIN_SIZE,
        lores_size: Tuple[int, int] = DEFAULT_LORES_SIZE,
        blur_ratio: float = DEFAULT_BLUR_RATIO,
        lock_verify_frames: int = DEFAULT_LOCK_VERIFY_FRAMES,
        auto_capture_enabled: bool = True,
        auto_capture_delay_s: float = DEFAULT_AUTO_CAPTURE_DELAY_S,
        lens_sweep_values: Optional[List[float]] = None,
    ):
        if Picamera2 is None:
            raise RuntimeError(
                "CalibSession: picamera2/libcamera를 import할 수 없다 — "
                "'source /home/suri/local-libcamera/env.sh' 후 $PICAM_PYTHON으로 실행해라."
            )
        self.out_root = out_root
        self.pattern_size = pattern_size
        self.square_size_mm = square_size_mm
        self.board_width_mm, self.board_height_mm = board_dimensions_mm(pattern_size, square_size_mm)
        self.main_size = main_size
        self.lores_size = lores_size
        self.blur_ratio = blur_ratio
        self.lock_verify_frames = max(20, lock_verify_frames)
        self.auto_capture_enabled = auto_capture_enabled
        self.auto_capture_delay_s = auto_capture_delay_s
        self.lens_sweep_values = lens_sweep_values or [round(x, 1) for x in np.arange(0.0, 15.01, 1.0)]

        self.shots = build_shot_list()
        self.completed = [False] * len(self.shots)
        self.shot_idx = 0
        self.coverage = new_coverage_grid()

        self.lock_ok = False
        self.lock_distance_m: Optional[float] = None
        self.lock_values: Optional[Dict] = None
        self.reference_sharpness: Optional[float] = None
        self._last_lock_reasons: List[str] = []

        self.corner_history: List[np.ndarray] = []
        self.autocapture_pending_since: Optional[float] = None
        self.last_result_text: Optional[str] = None      # 한국어 — HTML 컨트롤 페이지/JSON용
        self.last_result_text_en: Optional[str] = None    # ASCII — 프레임에 굽는 오버레이용
        self._last_live_corners: Optional[np.ndarray] = None
        self._last_countdown: Optional[float] = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self.picam = Picamera2()
        cfg = self.picam.create_still_configuration(
            main={"format": "RGB888", "size": main_size},  # 명명 역전 주의 — docstring 참조. 실제로는 BGR 배열이 나온다.
            lores={"format": "RGB888", "size": lores_size},
        )
        self.picam.configure(cfg)
        self.picam.start()
        time.sleep(1.0)

    # ---------- 진행 상태 ----------

    def current_shot(self) -> ShotSpec:
        return self.shots[self.shot_idx]

    def shots_for_current_distance(self) -> Tuple[List[ShotSpec], List[bool]]:
        shot = self.current_shot()
        specs, mask = [], []
        for s, done in zip(self.shots, self.completed):
            if s.set_name == shot.set_name and s.distance_m == shot.distance_m:
                specs.append(s)
                mask.append(done)
        return specs, mask

    def status_dict(self) -> Dict:
        shot = self.current_shot()
        return {
            "set": shot.set_name,
            "distance_m": shot.distance_m,
            "seq": shot.seq,
            "total_in_distance": shot.total_in_distance,
            "label_ko": shot.label_ko,
            "shot_index_overall": self.shot_idx,
            "total_shots_overall": len(self.shots),
            "done_overall": sum(self.completed),
            "lock_ok": self.lock_ok,
            "lock_distance_m": self.lock_distance_m,
            "lock_values": self.lock_values,
            "coverage_fill_fraction": coverage_fill_fraction(self.coverage),
            "auto_capture_enabled": self.auto_capture_enabled,
            "last_result": self.last_result_text,
        }

    # ---------- 잠금 ----------

    def run_focus_sweep_and_lock(self, roi_fraction: float = 0.6) -> Dict:
        """현재 샷 거리에 대해 초점 스윕 -> AF/AE/AWB 고정 -> 유지 검증.

        실패(유지 검증 미달)면 RuntimeError — 호출자가 그 사유를 사용자에게 보여준다.
        """
        with self._lock:
            shot = self.current_shot()
            results = []
            for lp in self.lens_sweep_values:
                self.picam.set_controls({"AfMode": libcamera_controls.AfModeEnum.Manual, "LensPosition": float(lp)})
                for _ in range(6):
                    self.picam.capture_metadata()
                bgr = self.picam.capture_array("main")
                roi = crop_center_fraction(bgr, roi_fraction)
                sharp = laplacian_sharpness(roi)
                results.append((float(lp), sharp))

            best_lens = pick_best_focus(results)
            self.picam.set_controls({"AfMode": libcamera_controls.AfModeEnum.Manual, "LensPosition": float(best_lens)})
            for _ in range(6):
                self.picam.capture_metadata()

            self.picam.set_controls({"AeEnable": True, "AwbEnable": True})
            m = {}
            for _ in range(25):
                m = self.picam.capture_metadata()
            exposure = int(m["ExposureTime"])
            gain = float(m["AnalogueGain"])
            colour_gains = (float(m["ColourGains"][0]), float(m["ColourGains"][1]))

            self.picam.set_controls(
                {
                    "AeEnable": False,
                    "ExposureTime": exposure,
                    "AnalogueGain": gain,
                    "AwbEnable": False,
                    "ColourGains": colour_gains,
                }
            )
            for _ in range(6):
                self.picam.capture_metadata()

            verify_frames = []
            for _ in range(self.lock_verify_frames):
                mm = self.picam.capture_metadata()
                verify_frames.append(
                    {
                        "LensPosition": mm.get("LensPosition"),
                        "ExposureTime": mm.get("ExposureTime"),
                        "AnalogueGain": mm.get("AnalogueGain"),
                        "ColourGains": tuple(mm.get("ColourGains", (None, None))),
                    }
                )

            ok, reasons = locks_held_within_tolerance(
                verify_frames, best_lens, exposure, gain, colour_gains
            )
            self._last_lock_reasons = reasons
            if not ok:
                self.lock_ok = False
                raise RuntimeError("컨트롤 고정이 유지되지 않았다: " + "; ".join(reasons))

            self.lock_ok = True
            self.lock_distance_m = shot.distance_m
            self.lock_values = {
                "lens_position": best_lens,
                "exposure_time": exposure,
                "analogue_gain": gain,
                "colour_gains": colour_gains,
            }

            ref_bgr = self.picam.capture_array("main")
            rect = zone_to_pixel_rect(
                shot.u, shot.v, ref_bgr.shape[1], ref_bgr.shape[0], shot.distance_m,
                self.board_width_mm, self.board_height_mm,
            )
            x0, y0, x1, y1 = rect
            if x1 > x0 and y1 > y0:
                self.reference_sharpness = laplacian_sharpness(ref_bgr[y0:y1, x0:x1])
            else:
                self.reference_sharpness = laplacian_sharpness(crop_center_fraction(ref_bgr, roi_fraction))

            lock_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "set": shot.set_name,
                "distance_m": shot.distance_m,
                "sweep": results,
                "chosen_lens_position": best_lens,
                "locked": self.lock_values,
                "verify_frames": verify_frames,
                "verify_ok": ok,
                "reference_sharpness": self.reference_sharpness,
                "pattern_size": list(self.pattern_size),
                "square_size_mm": self.square_size_mm,
            }
            out_dir = self.out_root / shot.set_name / shot.dir_name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "_lock.json").write_text(json.dumps(lock_record, ensure_ascii=False, indent=2))
            self.last_result_text = f"잠금 성공: lens={best_lens:.2f} exp={exposure} gain={gain:.2f}"
            self.last_result_text_en = f"LOCK OK: lens={best_lens:.2f} exp={exposure} gain={gain:.2f}"
            return self.lock_values

    # ---------- 미리보기 틱 ----------

    def preview_tick(self) -> np.ndarray:
        """lores 프레임 1장을 얻어 오버레이까지 그린 결과를 반환. 라이브 검출/자동촬영도
        여기서 처리한다(모든 카메라 접근은 이 세션의 락 안에서 직렬화)."""
        with self._lock:
            lores = self.picam.capture_array("lores")
            shot = self.current_shot()

            found = False
            corners_lores = None
            # 세션 지시: 원거리(약 2.5m)는 lores 기준 정사각형이 ~11px라 라이브 검출이
            # 사실상 항상 실패할 것으로 예상됨 — 실패해도 아무 기능도 이것에 의존하지
            # 않지만(게이팅 금지), 매 틱마다 무거운 실패 탐색을 반복하는 성능 낭비를 줄이려
            # 원거리에서는 아예 시도하지 않는다(정확성 게이팅이 아니라 순수 성능 최적화).
            if cv2 is not None and shot.distance_m <= 1.5:
                gray = cv2.cvtColor(lores, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(
                    gray, self.pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                if found:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    corners_lores = corners.reshape(-1, 2)

            if found and corners_lores is not None:
                self.corner_history.append(corners_lores)
                if len(self.corner_history) > DEFAULT_STABILITY_WINDOW:
                    self.corner_history.pop(0)
            else:
                self.corner_history.clear()
                self.autocapture_pending_since = None

            countdown = None
            if self.auto_capture_enabled and found and corners_lores is not None:
                stable = corners_are_stable(self.corner_history)
                centre = corners_lores.mean(axis=0)
                rect = zone_to_pixel_rect(
                    shot.u, shot.v, lores.shape[1], lores.shape[0], shot.distance_m,
                    self.board_width_mm, self.board_height_mm,
                )
                in_zone = point_in_rect(float(centre[0]), float(centre[1]), rect)
                if stable and in_zone and self.lock_ok and self.lock_distance_m == shot.distance_m:
                    now = time.time()
                    if self.autocapture_pending_since is None:
                        self.autocapture_pending_since = now
                    elapsed = now - self.autocapture_pending_since
                    countdown = max(0.0, self.auto_capture_delay_s - elapsed)
                    if elapsed >= self.auto_capture_delay_s:
                        self.autocapture_pending_since = None
                        # capture_locked()이 같은 락을 재획득하려 하면 교착되므로,
                        # 락을 쥔 채로 내부 구현을 직접 호출한다.
                        self._capture_locked()
                else:
                    self.autocapture_pending_since = None

            specs, mask = self.shots_for_current_distance()
            overlay = render_preview_overlay(
                lores,
                specs,
                mask,
                shot,
                self.board_width_mm,
                self.board_height_mm,
                self.coverage,
                self.lock_ok and self.lock_distance_m == shot.distance_m,
                self.lock_values,
                live_corners=corners_lores,
                countdown_s=countdown,
                last_result_text=self.last_result_text_en,
            )
            return overlay

    # ---------- 촬영 ----------

    def capture(self) -> Dict:
        with self._lock:
            return self._capture_locked()

    def _capture_locked(self) -> Dict:
        """`_lock`을 이미 쥔 상태에서 호출 — 실제 촬영 로직."""
        shot = self.current_shot()
        if not (self.lock_ok and self.lock_distance_m == shot.distance_m):
            result = {"ok": False, "reason": f"{shot.distance_m:g}m 거리의 컨트롤이 아직 잠기지 않음 — 먼저 잠금을 실행하라"}
            self.last_result_text = result["reason"]
            self.last_result_text_en = f"NOT LOCKED for {shot.distance_m:g}m - run lock first"
            return result

        bgr = self.picam.capture_array("main")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, self.pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            result = {"ok": False, "reason": "체스보드 패턴을 찾지 못함(findChessboardCorners 실패) — 각도/거리/조명을 확인하고 다시 촬영"}
            self.last_result_text = result["reason"]
            self.last_result_text_en = "PATTERN NOT FOUND - check angle/distance/light, retry"
            return result

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corners_flat = corners.reshape(-1, 2)

        sharpness = board_region_sharpness(bgr, corners_flat)
        ref = self.reference_sharpness if self.reference_sharpness else sharpness
        if is_capture_blurry(sharpness, ref, self.blur_ratio):
            result = {
                "ok": False,
                "reason": f"흔들림(모션블러) 의심 — sharpness={sharpness:.1f} (기준 {ref:.1f}의 {self.blur_ratio*100:.0f}% 미만) — 손을 고정하고 다시 촬영",
            }
            self.last_result_text = result["reason"]
            self.last_result_text_en = f"BLURRY - sharpness={sharpness:.1f} (< {self.blur_ratio*100:.0f}% of ref {ref:.1f}) - hold still, retry"
            return result

        out_dir = self.out_root / shot.set_name / shot.dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / f"{shot.file_stem}.png"
        json_path = out_dir / f"{shot.file_stem}.json"
        cv2.imwrite(str(png_path), bgr)

        rect = zone_to_pixel_rect(
            shot.u, shot.v, bgr.shape[1], bgr.shape[0], shot.distance_m, self.board_width_mm, self.board_height_mm
        )
        sidecar = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "set": shot.set_name,
            "distance_m": shot.distance_m,
            "seq": shot.seq,
            "zone": shot.zone,
            "u": shot.u,
            "v": shot.v,
            "yaw_deg": shot.yaw_deg,
            "pitch_deg": shot.pitch_deg,
            "roll_deg": shot.roll_deg,
            "label_ko": shot.label_ko,
            "pattern_size": list(self.pattern_size),
            "square_size_mm": self.square_size_mm,
            "locked_controls": dict(self.lock_values) if self.lock_values else None,
            "corners_px": corners_flat.tolist(),
            "sharpness": sharpness,
            "reference_sharpness": self.reference_sharpness,
            "target_zone_px": list(rect),
            "image_size": [bgr.shape[1], bgr.shape[0]],
            "image_path": str(png_path),
        }
        json_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2))

        update_coverage_grid(self.coverage, corners_flat, bgr.shape[1], bgr.shape[0])
        self.completed[self.shot_idx] = True
        if self.shot_idx < len(self.shots) - 1:
            self.shot_idx += 1

        result = {"ok": True, "path": str(png_path), "sharpness": sharpness, "num_corners": int(len(corners_flat))}
        self.last_result_text = f"촬영 성공: {png_path.name} (sharpness={sharpness:.1f})"
        self.last_result_text_en = f"CAPTURED {png_path.name} (sharpness={sharpness:.1f})"
        return result

    def skip(self) -> None:
        with self._lock:
            if self.shot_idx < len(self.shots) - 1:
                self.shot_idx += 1
            self.last_result_text = "건너뜀(저장 안 됨)"
            self.last_result_text_en = "SKIPPED (not saved)"

    def prev(self) -> None:
        with self._lock:
            if self.shot_idx > 0:
                self.shot_idx -= 1
            self.last_result_text = "이전 샷으로 이동"
            self.last_result_text_en = "MOVED TO PREVIOUS SHOT"

    def set_auto_capture(self, enabled: bool) -> None:
        with self._lock:
            self.auto_capture_enabled = enabled
            self.autocapture_pending_since = None

    def close(self) -> None:
        self._stop_event.set()
        try:
            self.picam.stop()
        finally:
            self.picam.close()


def _preview_loop(session: CalibSession, streamer: MjpegStreamer, interval: float) -> None:
    while not session._stop_event.is_set():
        try:
            frame = session.preview_tick()
            streamer.push_frame(frame)
        except Exception as e:  # noqa: BLE001 - 미리보기 루프는 죽으면 안 됨, 로그만 남김
            print(f"[미리보기 실패] {e}", file=sys.stderr)
        time.sleep(interval)


# ===========================================================================
# 컨트롤 HTTP 서버 (포트 8081) — 순수 HTML 폼, JS 프레임워크 없음
# ===========================================================================


def parse_autocapture_enabled(query: str) -> bool:
    """`/autocapture` 쿼리스트링 -> 자동촬영 on/off. **`enabled`가 없으면 끈다.**

    2026-07-23 실기 버그의 수정본이다. 원래 이 기본값이 `"1"`(켬)이었고 조준 페이지의
    토글 버튼이 이렇게 생겼었다::

        <form method="GET" action="/autocapture?enabled=0">...</form>

    HTML 사양상 **`method="GET"` 폼은 제출할 때 action URL의 쿼리스트링을 버리고 폼 입력값을
    직렬화한 것으로 교체한다.** 이 폼엔 입력 요소가 하나도 없어서 빈 쿼리가 되고, 결국
    브라우저는 `/autocapture`로만 이동해 서버가 기본값 `"1"`을 읽었다 — **"자동촬영 끄기"
    버튼이 누를 때마다 오히려 켜서, 사용자가 자동촬영을 끌 수 없었다**(보드가 존에 들어올
    때마다 멋대로 촬영돼 촬영 진행이 막힘). 버튼은 hidden input으로 고쳤고(`_render_page`),
    기본값은 여기서 뒤집는다 — 자동촬영이 의도치 않게 **켜지는** 쪽이 사진을 멋대로 찍으므로
    더 해롭기 때문이다.
    """
    return parse_qs(query).get("enabled", ["0"])[0] == "1"


def _render_page(session: CalibSession, video_host: str, video_port: int) -> str:
    """조준용 메인 페이지 — **새로고침 없음**(2026-07-23 수정).

    과거 `<meta http-equiv="refresh" content="2">`가 2초마다 문서 전체를 새로 불러와
    `<img>` MJPEG 연결을 매번 끊고 재연결시켜, 85컷 촬영 내내 조준 중 화면이 계속
    끊겼다(사용자 실측 불만). 존/샷 번호/직전 촬영 성공·거절(사유)/커버리지 맵은 이미
    `render_preview_overlay()`가 프레임에 실시간으로 구워 넣으므로 이 페이지의 텍스트는
    중복이었다 — 그 중복 텍스트(새로고침 없이는 오히려 고정된 채 낡아갈 정보)는 지우고,
    필요하면 볼 수 있게 별도 `/status` 탭(그 페이지는 영상이 없어 새로고침해도 무해함)으로
    옮겼다. 버튼(잠금/촬영/이전/건너뛰기/자동촬영 토글)은 그대로 둔다 — 전부 사용자
    액션이 있을 때만 303 리다이렉트로 자기 자신을 갱신하므로 새로고침 없이도 낡지 않는다.
    """
    status = session.status_dict()
    ac_label = "자동촬영 끄기" if status["auto_capture_enabled"] else "자동촬영 켜기"
    ac_next = 0 if status["auto_capture_enabled"] else 1
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>캘리브레이션 촬영</title></head>
<body style="margin:0;background:#111;color:#eee;font-family:sans-serif">
<img src="http://{video_host}:{video_port}/stream" style="width:100%;display:block;background:#000">
<div style="padding:10px 14px">
  <div style="margin-bottom:8px;color:#999;font-size:13px">존/샷 번호/직전 촬영 결과/커버리지는 영상 위에 실시간으로 표시됩니다 &middot; <a href="/status" target="_blank" style="color:#9cf">상태 텍스트만 보기(새 탭, 2초마다 자동갱신)</a></div>

  <form method="POST" action="/lock" style="margin-bottom:8px">
    <button style="width:100%;padding:14px;font-size:18px" type="submit">이 거리 초점/노출/WB 잠금 시작</button>
  </form>
  <form method="POST" action="/capture" style="margin-bottom:8px">
    <button style="width:100%;padding:22px;font-size:22px;background:#2a6;color:#fff;border:none" type="submit">촬영</button>
  </form>
  <div style="display:flex;gap:8px">
    <form method="GET" action="/prev" style="flex:1"><button style="width:100%;padding:12px">이전 샷</button></form>
    <form method="GET" action="/skip" style="flex:1"><button style="width:100%;padding:12px">건너뛰기</button></form>
    <form method="GET" action="/autocapture" style="flex:1"><input type="hidden" name="enabled" value="{ac_next}"><button style="width:100%;padding:12px">{ac_label}</button></form>
  </div>
</div>
</body></html>"""


def _render_status_page(session: CalibSession) -> str:
    """`/status` — 영상이 없는 별도 상태 페이지. 여기는 새로고침해도 아무것도 끊기지
    않으므로(임베드된 `<img>` 스트림이 없음) 2초 자동 새로고침을 그대로 둔다. 두 번째
    브라우저 탭으로 열어 텍스트로 진행 상황을 보고 싶을 때 쓴다 — 조준 페이지(`/`)
    자체에는 더 이상 이 텍스트를 넣지 않는다(위 `_render_page` docstring 참조)."""
    status = session.status_dict()
    lock_html = "미잠금"
    if status["lock_ok"] and status["lock_distance_m"] == status["distance_m"]:
        lv = status["lock_values"]
        lock_html = f"잠김: lens={lv['lens_position']:.2f} exp={lv['exposure_time']} gain={lv['analogue_gain']:.2f}"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="2">
<title>캘리브레이션 상태</title></head>
<body style="margin:0;background:#111;color:#eee;font-family:sans-serif">
<div style="padding:14px">
  <div style="font-size:18px;margin-bottom:4px">[{status['set']}] {status['distance_m']:g}m &middot; 샷 {status['seq']}/{status['total_in_distance']} &middot; 전체 {status['shot_index_overall']+1}/{status['total_shots_overall']} (완료 {status['done_overall']})</div>
  <div style="margin-bottom:4px">{status['label_ko']}</div>
  <div style="margin-bottom:8px;color:#9f9">{lock_html}</div>
  <div style="margin-bottom:8px">커버리지: {status['coverage_fill_fraction']*100:.0f}%</div>
  {'<div style="margin-bottom:8px;color:#ff8">' + status['last_result'] + '</div>' if status['last_result'] else ''}
  <div style="margin-top:14px;color:#999;font-size:13px">이 페이지는 영상을 포함하지 않아 2초마다 새로고침돼도 무해합니다. 영상은 <a href="/" style="color:#9cf">촬영 페이지</a>에서.</div>
</div>
</body></html>"""


class _ControlHandler(BaseHTTPRequestHandler):
    session: CalibSession
    video_port: int

    def log_message(self, *args):
        pass

    def _video_host(self) -> str:
        host_hdr = self.headers.get("Host", "localhost")
        return host_hdr.split(":")[0] if ":" in host_hdr else host_hdr

    def _send(self, body: bytes, content_type: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = _render_page(self.session, self._video_host(), self.video_port).encode("utf-8")
            self._send(body, "text/html; charset=utf-8")
        elif parsed.path == "/status":
            body = _render_status_page(self.session).encode("utf-8")
            self._send(body, "text/html; charset=utf-8")
        elif parsed.path == "/status.json":
            self._send(json.dumps(self.session.status_dict(), ensure_ascii=False).encode("utf-8"), "application/json")
        elif parsed.path == "/skip":
            self.session.skip()
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
        elif parsed.path == "/prev":
            self.session.prev()
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
        elif parsed.path == "/autocapture":
            self.session.set_auto_capture(parse_autocapture_enabled(parsed.query))
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/capture":
            result = self.session.capture()
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
        elif self.path == "/lock":
            try:
                self.session.run_focus_sweep_and_lock()
            except RuntimeError as e:
                self.session.last_result_text = f"잠금 실패: {e}"
                self.session.last_result_text_en = "LOCK FAILED - open /status tab for reason"
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def _check_deps() -> None:
    if cv2 is None:
        print("cv2(OpenCV)가 없다 — picam-venv에 opencv가 설치돼 있는지 확인해라", file=sys.stderr)
        sys.exit(1)
    if Picamera2 is None:
        print(
            "picamera2/libcamera를 import할 수 없다 — "
            "'source /home/suri/local-libcamera/env.sh' 후 $PICAM_PYTHON으로 실행해라.",
            file=sys.stderr,
        )
        sys.exit(1)


def _parse_wh(s: str) -> Tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def _parse_pattern_size(s: str) -> Tuple[int, int]:
    c, r = s.lower().split("x")
    return int(c), int(r)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="/home/suri/local-libcamera-src/calib", help="촬영 저장 루트 폴더")
    parser.add_argument("--pattern-size", default="9x6", help="체스보드 내부 코너 개수 WxH (기본 9x6 = 10x7 사각형)")
    parser.add_argument("--board-preset", choices=["a4", "tablet"], default="a4", help="--square-size-mm 미지정 시 기본값 선택")
    parser.add_argument("--square-size-mm", type=float, default=None, help=f"사각형 한 변 mm (기본: a4={DEFAULT_SQUARE_SIZE_MM_A4}, tablet={DEFAULT_SQUARE_SIZE_MM_TABLET})")
    parser.add_argument("--main-size", default="4608x2592", help="풀해상도 촬영 크기 WxH")
    parser.add_argument("--lores-size", default="1280x720", help="미리보기 크기 WxH")
    parser.add_argument("--video-port", type=int, default=8080, help="MjpegStreamer 포트")
    parser.add_argument("--control-port", type=int, default=8081, help="컨트롤 UI 포트")
    parser.add_argument("--stream-width", type=int, default=960, help="MjpegStreamer 다운스케일 목표 폭")
    parser.add_argument("--stream-height", type=int, default=540, help="MjpegStreamer 다운스케일 목표 높이")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--preview-interval", type=float, default=0.2, help="미리보기 틱 주기(초)")
    parser.add_argument("--auto-capture", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-capture-delay-s", type=float, default=DEFAULT_AUTO_CAPTURE_DELAY_S)
    parser.add_argument("--blur-ratio", type=float, default=DEFAULT_BLUR_RATIO)
    parser.add_argument("--lock-verify-frames", type=int, default=DEFAULT_LOCK_VERIFY_FRAMES)
    args = parser.parse_args()

    _check_deps()

    square_size_mm = args.square_size_mm
    if square_size_mm is None:
        square_size_mm = DEFAULT_SQUARE_SIZE_MM_TABLET if args.board_preset == "tablet" else DEFAULT_SQUARE_SIZE_MM_A4

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    session = CalibSession(
        out_root=out_root,
        pattern_size=_parse_pattern_size(args.pattern_size),
        square_size_mm=square_size_mm,
        main_size=_parse_wh(args.main_size),
        lores_size=_parse_wh(args.lores_size),
        blur_ratio=args.blur_ratio,
        lock_verify_frames=args.lock_verify_frames,
        auto_capture_enabled=args.auto_capture,
        auto_capture_delay_s=args.auto_capture_delay_s,
    )

    streamer = MjpegStreamer(
        host=args.host, port=args.video_port, target_width=args.stream_width, target_height=args.stream_height
    )

    server = None
    preview_thread = None
    try:
        streamer.start()
        preview_thread = threading.Thread(
            target=_preview_loop, args=(session, streamer, args.preview_interval), daemon=True
        )
        preview_thread.start()

        handler = type("_BoundControlHandler", (_ControlHandler,), {"session": session, "video_port": args.video_port})
        server = ThreadingHTTPServer((args.host, args.control_port), handler)
        print(f"컨트롤 UI: http://<이 라즈베리파이의 tailscale IP>:{args.control_port}/", flush=True)
        print(f"영상 스트림(직접 확인용): http://<이 라즈베리파이의 tailscale IP>:{args.video_port}/stream", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        streamer.stop()
        session.close()


if __name__ == "__main__":
    main()
