from pathlib import Path
import cv2
import numpy as np
from vision.core.state import Detection, VisionState

# ---------------------------------------------------------------------------
# 유도(정밀착륙) 발행 상태 오버레이 — `draw_sink_status()` 상수 (매직넘버 금지, §7.3)
#
# 🔴 **왜 필요한가(2026-07-28 사용자 결정):** `--display`(창/MJPEG 스트림)와 `--target-sink`
# (유도 좌표 발행)는 **완전히 독립**이다. 그래서 "화면은 멀쩡히 뜨고 검출도 되는데 유도
# 좌표는 아무 데도 안 나가는" 상태가 실제로 가능하다 — 디버깅 중에 이걸 놓치면 최악이다.
# bind 실패는 기동 시점에 즉사시킬 수 있지만(main.py `_make_target_sink`), **bind는 됐는데
# 소비자가 아무도 안 붙은 상태**는 죽일 수 없다(시작 직후엔 소비자가 아직 안 붙는 게 정상).
# 그 사각지대를 화면에 상시 노출하는 것이 이 오버레이의 존재 이유다.
# ---------------------------------------------------------------------------
SINK_OVERLAY_OK_COLOR = (0, 200, 0)        # 소비자 있음 — `draw_detections`의 초록과 동일 톤
SINK_OVERLAY_ALERT_COLOR = (0, 0, 255)     # 소비자 0명 — 빨강(가장 눈에 띄는 경고색)
SINK_OVERLAY_OFF_COLOR = (0, 165, 255)     # sink 자체가 꺼짐 — 주황(운영자의 명시적 선택)
# 폰트 크기는 프레임 폭에 비례시킨다 — 실기체 4608px과 테스트용 300px에서 둘 다 읽혀야 한다.
SINK_OVERLAY_FONT_REF_WIDTH_PX = 1400.0
SINK_OVERLAY_MIN_FONT_SCALE = 0.35
SINK_OVERLAY_HEADLINE_FACTOR = 1.4         # 1행(경고행)만 키운다 — "눈에 확 띄게"
SINK_OVERLAY_PANEL_ALPHA = 0.35            # 패널을 **반투명**으로 어둡게(밑의 화면을 안 가림)

_SINK_OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------------------------------------------------------------------------
# 착륙 판단 오버레이 — `draw_landing_overlay()` 상수 (매직넘버 금지, §7.3)
#
# 🔴 **왜 필요한가(2026-07-29, 2차예선 보고 제출 영상):** `draw_detections()`가 그리는 것은
# **초록(0,200,0) 얇은 bbox + 신뢰도 숫자**뿐이다. 우선 타겟인 ② 조난자 구역은 **초록 매트**라
# 초록 선이 배경에 묻혀, 실제로 뽑아 본 녹화 영상에서 "인식이 되고 있다"가 눈으로 안 보였다.
# 게다가 이 파이프라인이 실제로 판단하는 것들 — **착륙점**(박스 옆 빈 초록면), 흰 박스,
# 상태머신의 상태/명령, 추정 거리 — 은 JSONL에만 있고 **화면에는 한 픽셀도 그려지지 않았다.**
# 이 오버레이는 그 값들을 그리기만 한다(계산하지 않는다 — 착륙점 규칙 재구현 금지,
# `modules/distress_mat.py`가 `landing_point_px`를 그대로 소비하는 것과 같은 원칙).
#
# **색은 초록을 피한다** — 배경(초록 매트)·기존 검출 박스(초록)와 구분돼야 한다.
# ---------------------------------------------------------------------------
LANDING_OVERLAY_TARGET_COLOR = (0, 255, 255)     # 노랑 — 확정 타겟(매트/마커) bbox
LANDING_OVERLAY_BOX_COLOR = (255, 160, 0)        # 하늘색 — 매트 안 흰 박스(랜드마크)
# 🔴 착륙점은 **마젠타**다 — 빨강(0,0,255)은 `SINK_OVERLAY_ALERT_COLOR`("소비자 0명" 경고)가
# 이미 쓰고 있어서, 두 오버레이를 같이 켠 프레임에서 같은 색이 **다른 뜻**을 갖게 된다.
# 마젠타는 초록 매트·회색 자갈 어느 배경에서도 튀고 두 경고색과 겹치지 않는다.
LANDING_OVERLAY_POINT_COLOR = (255, 0, 255)      # 마젠타 — 착륙점(기체가 실제로 갈 곳)
LANDING_OVERLAY_TEXT_COLOR = (255, 255, 255)     # 흰색 — 하단 패널 글씨
# 폰트/선 굵기는 `draw_sink_status`와 같은 이유로 프레임 폭에 비례시킨다(4608px~300px 공용).
LANDING_OVERLAY_FONT_REF_WIDTH_PX = 1400.0
LANDING_OVERLAY_MIN_FONT_SCALE = 0.35
LANDING_OVERLAY_PANEL_ALPHA = 0.35               # 하단 패널도 반투명(밑 영상을 안 가림)
LANDING_OVERLAY_CROSS_RATIO = 0.02               # 착륙점 십자 반길이 = 프레임 폭 × 이 비율
LANDING_OVERLAY_RING_RATIO = 0.012               # 착륙점 원 반지름 = 프레임 폭 × 이 비율

_LANDING_OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_sink_status(
    image: np.ndarray,
    *,
    enabled: bool,
    consumers: int = 0,
    seq: int = 0,
    dropped: int = 0,
    endpoint: str | None = None,
) -> np.ndarray:
    """유도 발행 상태(소비자 수 / 마지막 발행 seq / 드롭 수)를 프레임 좌상단에 그린다.

    - **`image`를 제자리에서(in-place) 고치고 같은 배열을 돌려준다.** 호출자는
      `draw_detections()`가 이미 만들어 준 사본을 넘긴다 — 프레임 사본을 한 번 더 만들면
      매 프레임 전체 복사가 한 번 더 늘어나므로. 검출 그리기 **뒤에** 부르는 것이 계약이라
      오버레이가 검출 결과를 훼손하지 않는다(패널도 반투명이라 밑이 비친다).
    - **ASCII만 쓴다.** `cv2.putText`의 Hershey 폰트는 한글을 못 그린다(글자가 깨져 나온다).
    - **호출 자체를 `--display != none`으로 게이팅하는 것은 호출자 책임**이다 — 드론 기본
      경로(`--display none`)에서 오버레이 비용이 0이어야 하기 때문(§7.9 헤드리스 전제).

    `enabled=False`(= `NullSink`, `--target-sink` 미지정)도 "유도 좌표가 아무 데도 안 나간다"는
    같은 사실이므로 조용히 넘어가지 않고 주황색으로 명시한다.
    """
    h, w = image.shape[:2]
    scale = max(SINK_OVERLAY_MIN_FONT_SCALE, w / SINK_OVERLAY_FONT_REF_WIDTH_PX)

    if not enabled:
        colour = SINK_OVERLAY_OFF_COLOR
        headline = "SINK OFF - NO GUIDANCE OUT"
        detail = "(--target-sink not given)"
    elif consumers <= 0:
        colour = SINK_OVERLAY_ALERT_COLOR
        headline = "CONSUMERS 0 - GUIDANCE GOES NOWHERE"
        detail = f"sink {endpoint or '?'}  seq {seq}  dropped {dropped}"
    else:
        colour = SINK_OVERLAY_OK_COLOR
        headline = f"CONSUMERS {consumers}"
        detail = f"sink {endpoint or '?'}  seq {seq}  dropped {dropped}"

    lines = [(headline, scale * SINK_OVERLAY_HEADLINE_FACTOR), (detail, scale)]
    pad = max(2, int(round(6 * scale)))
    metrics = []
    for text, font_scale in lines:
        thickness = max(1, int(round(2 * font_scale)))
        (tw, th), baseline = cv2.getTextSize(text, _SINK_OVERLAY_FONT, font_scale, thickness)
        metrics.append((text, font_scale, thickness, tw, th, baseline))

    box_w = max(m[3] for m in metrics) + 2 * pad
    box_h = sum(m[4] + m[5] for m in metrics) + pad * (len(metrics) + 1)
    x1, y1 = min(w, pad + box_w), min(h, pad + box_h)

    # 반투명 어둡게 — 밑의 영상이 비쳐 보이므로 오버레이가 화면을 "가리지" 않는다.
    roi = image[pad:y1, pad:x1]
    if roi.size:
        cv2.addWeighted(roi, 1.0 - SINK_OVERLAY_PANEL_ALPHA, roi, 0.0, 0.0, dst=roi)

    y = pad
    for text, font_scale, thickness, _tw, th, baseline in metrics:
        y += pad + th
        cv2.putText(image, text, (2 * pad, y), _SINK_OVERLAY_FONT, font_scale, colour, thickness)
        y += baseline
    return image


def _landing_overlay_panel(
    image: np.ndarray, lines: list[str], scale: float, thickness: int
) -> None:
    """하단 좌측에 반투명 패널 + 여러 줄 텍스트. `draw_sink_status`의 패널과 같은 방식이되
    위치만 아래다 — 두 오버레이를 같이 켜도 서로 안 가린다(sink=좌상단 / 이쪽=좌하단)."""
    h, w = image.shape[:2]
    pad = max(2, int(round(6 * scale)))
    metrics = []
    for text in lines:
        (tw, th), baseline = cv2.getTextSize(text, _LANDING_OVERLAY_FONT, scale, thickness)
        metrics.append((text, tw, th, baseline))

    box_w = max(m[1] for m in metrics) + 2 * pad
    box_h = sum(m[2] + m[3] for m in metrics) + pad * (len(metrics) + 1)
    y0 = max(0, h - pad - box_h)
    x1, y1 = min(w, pad + box_w), min(h, y0 + box_h)

    roi = image[y0:y1, pad:x1]
    if roi.size:
        cv2.addWeighted(roi, 1.0 - LANDING_OVERLAY_PANEL_ALPHA, roi, 0.0, 0.0, dst=roi)

    y = y0
    for text, _tw, th, baseline in metrics:
        y += pad + th
        cv2.putText(
            image, text, (2 * pad, y), _LANDING_OVERLAY_FONT, scale,
            LANDING_OVERLAY_TEXT_COLOR, thickness,
        )
        y += baseline


def draw_landing_overlay(
    image: np.ndarray,
    detections: list[Detection],
    *,
    state: str | None = None,
    command: str | None = None,
    estimate=None,
    frame_id: int | None = None,
) -> np.ndarray:
    """착륙 판단(착륙점·흰 박스·상태머신·추정거리)을 프레임에 그린다 — **제자리 수정**.

    - **`image`를 제자리에서 고치고 같은 배열을 돌려준다.** 호출자는 `draw_detections()`가
      이미 만들어 준 사본을 넘긴다(프레임 사본을 한 번 더 만들지 않는다 — `draw_sink_status`와
      동일 계약). `draw_detections()` **뒤에** 부르는 것이 계약이다.
    - **아무것도 계산하지 않는다.** 착륙점은 `modules/distress_box.py`가 이미 확정해
      `det.meta["white_box_detector"]["landing_point_px"]`에 넣어 둔 값을 그대로 읽는다 —
      규칙을 두 곳에 복사하면 한쪽만 고쳐졌을 때 조용히 어긋난다(`modules/distress_mat.py`가
      같은 이유로 재구현을 거부하는 것과 같은 원칙).
    - **호출 게이팅은 호출자 책임**이다 — 드론 기본 경로(`--display none` + sink 없음)에서
      비용이 0이어야 하므로 `main.py`/`replay.py`가 `--report-overlay`로만 부른다(§7.9).
    - **ASCII만 쓴다.** `cv2.putText`의 Hershey 폰트는 한글을 못 그린다(글자가 깨져 나온다).

    `estimate`는 `core/target.py::TargetEstimate`(없으면 None) — `position[2]`(광학 프레임 z,
    미터)를 거리로, `target_type`을 타겟 종류로 표시한다. 값이 없으면 그 줄만 빠진다
    (**0이나 추측값을 채우지 않는다** — 이 저장소가 `core/wire.py`에서 "가장 위험한
    거짓말"이라 부른 그것).
    """
    h, w = image.shape[:2]
    scale = max(LANDING_OVERLAY_MIN_FONT_SCALE, w / LANDING_OVERLAY_FONT_REF_WIDTH_PX)
    thickness = max(1, int(round(2 * scale)))

    for det in detections:
        wb = det.meta.get("white_box_detector") if det.meta else None
        x, y, bw, bh = det.bbox
        cv2.rectangle(
            image, (x, y), (x + bw, y + bh), LANDING_OVERLAY_TARGET_COLOR, thickness + 1
        )
        if not wb:
            continue

        box_bbox = wb.get("box_bbox")
        if box_bbox is not None:
            bx, by, bbw, bbh = (int(v) for v in box_bbox)
            cv2.rectangle(
                image, (bx, by), (bx + bbw, by + bbh), LANDING_OVERLAY_BOX_COLOR, thickness
            )

        point = wb.get("landing_point_px")
        if point is not None:
            px, py = int(round(float(point[0]))), int(round(float(point[1])))
            arm = max(3, int(round(w * LANDING_OVERLAY_CROSS_RATIO)))
            ring = max(2, int(round(w * LANDING_OVERLAY_RING_RATIO)))
            cv2.line(image, (px - arm, py), (px + arm, py), LANDING_OVERLAY_POINT_COLOR, thickness)
            cv2.line(image, (px, py - arm), (px, py + arm), LANDING_OVERLAY_POINT_COLOR, thickness)
            cv2.circle(image, (px, py), ring, LANDING_OVERLAY_POINT_COLOR, thickness)
            cv2.putText(
                image, "LANDING PT", (px + arm + 4, py - 4), _LANDING_OVERLAY_FONT,
                scale, LANDING_OVERLAY_POINT_COLOR, thickness,
            )

    lines = [f"DET {len(detections)}"]
    if frame_id is not None:
        lines[0] = f"FRAME {frame_id}  DET {len(detections)}"
    if state is not None:
        lines.append(f"STATE {state}" + (f"  CMD {command}" if command is not None else ""))
    if estimate is not None:
        lines.append(f"TARGET {estimate.target_type}  DIST {float(estimate.position[2]):.2f} m")
    _landing_overlay_panel(image, lines, scale, thickness)
    return image


def draw_detections(image: np.ndarray, detections: list[Detection],
                    confirmed: Detection | None = None) -> np.ndarray:
    out = image.copy()
    for det in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
        label = f"{det.confidence:.2f}"
        cv2.putText(out, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    if confirmed is not None:
        x, y, w, h = confirmed.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(out, "CONFIRMED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return out


def draw_mask(mask: np.ndarray) -> np.ndarray:
    """단채널 마스크 → 시각화용 BGR 이미지."""
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def save_result(state: VisionState, output_path: str) -> None:
    annotated = draw_detections(state.original, state.detections, state.confirmed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, annotated)
