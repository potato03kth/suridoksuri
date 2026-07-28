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
