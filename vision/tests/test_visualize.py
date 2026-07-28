"""`vision/utils/visualize.py::draw_sink_status` — 유도 발행 상태 오버레이 단위 테스트.

**왜 이 오버레이가 있는가(2026-07-28 사용자 결정):** `--display`(창/MJPEG 스트림)와
`--target-sink`(유도 좌표 발행)는 **완전히 독립**이다 — 그래서 "화면은 멀쩡히 뜨고 검출도
되는데 유도 좌표는 아무 데도 안 나가는" 상태가 실제로 가능하다. 기동(bind) 실패는 즉사로
막지만(`main.py::_make_target_sink`), **bind는 됐는데 소비자가 0명**인 경우는 죽일 수 없다
(시작 직후엔 소비자가 아직 안 붙는 게 정상). 그 사각지대를 화면에 상시 노출하는 것이 이
함수의 유일한 목적이다.

이 파일은 **그리기 자체**만 본다. `main.py`/`replay.py`가 실제로 이걸 부르는지(그리고
`--display none`에서 안 부르는지)는 `test_main.py`/`test_replay.py`가 담당한다.

⚠️ `draw_detections`/`save_result`의 일반 테스트는 여전히 TODO다 — 이 파일이 그걸 대신하지
않는다(vision/CLAUDE.md 단위별 테스트 표 참조).
"""
import cv2
import numpy as np

from vision.utils.visualize import (
    SINK_OVERLAY_ALERT_COLOR,
    SINK_OVERLAY_OFF_COLOR,
    SINK_OVERLAY_OK_COLOR,
    draw_detections,
    draw_sink_status,
)
from vision.core.state import Detection


def _blank(w: int = 640, h: int = 480) -> np.ndarray:
    return np.full((h, w, 3), 40, dtype=np.uint8)


def _count_colour(img: np.ndarray, bgr) -> int:
    return int(np.count_nonzero(np.all(img == np.array(bgr, dtype=np.uint8), axis=2)))


def test_no_consumer_is_drawn_in_the_alert_colour():
    """🔴 핵심 요구("소비자 0명이면 눈에 확 띄게"): 소비자가 없으면 경고색이 실제 픽셀로
    찍혀야 한다. 경고색을 정상색과 같게 만들면(=경고를 지우면) 이 테스트가 red가 된다."""
    img = draw_sink_status(_blank(), enabled=True, consumers=0, seq=10, dropped=0,
                           endpoint="127.0.0.1:8091")
    assert _count_colour(img, SINK_OVERLAY_ALERT_COLOR) > 0
    assert _count_colour(img, SINK_OVERLAY_OK_COLOR) == 0


def test_attached_consumer_is_drawn_in_the_ok_colour_and_not_the_alert_colour():
    img = draw_sink_status(_blank(), enabled=True, consumers=2, seq=10, dropped=0,
                           endpoint="127.0.0.1:8091")
    assert _count_colour(img, SINK_OVERLAY_OK_COLOR) > 0
    assert _count_colour(img, SINK_OVERLAY_ALERT_COLOR) == 0


def test_alert_headline_is_bigger_than_the_ok_headline():
    """"눈에 확 띄게"는 색만이 아니라 크기다 — 경고 상태의 잉크량이 정상 상태보다 많아야
    한다(같은 폰트 크기로 그리면 red)."""
    alert = draw_sink_status(_blank(), enabled=True, consumers=0, seq=10, dropped=0,
                             endpoint="127.0.0.1:8091")
    ok = draw_sink_status(_blank(), enabled=True, consumers=1, seq=10, dropped=0,
                          endpoint="127.0.0.1:8091")
    alert_ink = _count_colour(alert, SINK_OVERLAY_ALERT_COLOR)
    ok_ink = _count_colour(ok, SINK_OVERLAY_OK_COLOR)
    assert alert_ink > ok_ink


def test_sink_off_uses_its_own_colour_distinct_from_both():
    """`--target-sink` 미지정도 "유도가 안 나간다"는 같은 사실이라 표시하지만, 운영자의
    명시적 선택이므로 경고(빨강)와 정상(초록) 어느 쪽과도 구분돼야 한다."""
    img = draw_sink_status(_blank(), enabled=False)
    assert _count_colour(img, SINK_OVERLAY_OFF_COLOR) > 0
    assert _count_colour(img, SINK_OVERLAY_ALERT_COLOR) == 0
    assert _count_colour(img, SINK_OVERLAY_OK_COLOR) == 0


def test_counters_actually_reach_the_pixels():
    """seq/dropped가 실제로 그려지는지 — 값이 달라지면 픽셀도 달라져야 한다(라벨만 그리고
    숫자를 버리면 red)."""
    a = draw_sink_status(_blank(), enabled=True, consumers=1, seq=1, dropped=0, endpoint="h:1")
    b = draw_sink_status(_blank(), enabled=True, consumers=1, seq=999, dropped=0, endpoint="h:1")
    c = draw_sink_status(_blank(), enabled=True, consumers=1, seq=1, dropped=77, endpoint="h:1")
    assert not np.array_equal(a, b), "seq가 화면에 반영되지 않는다"
    assert not np.array_equal(a, c), "dropped가 화면에 반영되지 않는다"


def test_overlay_does_not_destroy_the_detection_drawing():
    """🔴 "오버레이가 검출 결과 그리기를 가리거나 망가뜨리면 안 된다": 오버레이 패널 **밖**의
    검출 박스 픽셀은 한 점도 달라지면 안 되고, 패널 안쪽도 (반투명이라) 완전히 덮이지 않는다."""
    base = _blank()
    det = Detection(bbox=(300, 300, 120, 100), confidence=0.9)
    annotated = draw_detections(base, [det], confirmed=det)
    before = annotated.copy()

    after = draw_sink_status(annotated, enabled=True, consumers=0, seq=3, dropped=0,
                             endpoint="127.0.0.1:8091")
    # 오버레이는 좌상단 패널에만 그린다 — 검출 박스가 있는 우하단은 비트 단위로 동일해야 한다.
    assert np.array_equal(after[200:, 200:], before[200:, 200:]), (
        "오버레이가 패널 밖(검출 결과 영역)을 건드렸다"
    )
    changed = np.count_nonzero(np.any(after != before, axis=2))
    assert changed > 0, "오버레이가 아무것도 안 그렸다"


def test_overlay_draws_in_place_and_preserves_shape_and_dtype():
    """`draw_detections()`가 이미 만든 사본 위에 덧그린다 — 사본을 또 만들면 매 프레임 전체
    복사가 한 번 더 늘어난다. 같은 배열 객체를 돌려주는 것이 그 계약의 표현이다."""
    img = _blank(320, 240)
    out = draw_sink_status(img, enabled=True, consumers=0, seq=1, dropped=0, endpoint="h:1")
    assert out is img
    assert out.shape == (240, 320, 3) and out.dtype == np.uint8


def test_overlay_is_deterministic():
    a = draw_sink_status(_blank(), enabled=True, consumers=0, seq=42, dropped=1, endpoint="h:1")
    b = draw_sink_status(_blank(), enabled=True, consumers=0, seq=42, dropped=1, endpoint="h:1")
    assert np.array_equal(a, b)


def test_overlay_survives_tiny_and_huge_frames_without_crashing():
    """실기체 4608px과 테스트용 소형 프레임 둘 다에서 크래시 없이 그려져야 한다(패널이 프레임
    보다 커지는 경우 포함 — cv2가 알아서 자르되 예외는 나면 안 된다)."""
    for w, h in [(64, 48), (120, 120), (4608, 2592)]:
        img = draw_sink_status(
            np.full((h, w, 3), 40, dtype=np.uint8),
            enabled=True, consumers=0, seq=7, dropped=0, endpoint="127.0.0.1:8091",
        )
        assert img.shape == (h, w, 3)


def test_overlay_text_is_ascii_only():
    """`cv2.putText`의 Hershey 폰트는 한글을 못 그린다(글자가 깨진다) — 오버레이 문자열이
    ASCII임을 소스 레벨에서 고정한다. 한글을 넣으면 화면에서 조용히 깨진다."""
    import inspect

    import vision.utils.visualize as vis

    src = inspect.getsource(vis.draw_sink_status)
    literals = [
        line for line in src.splitlines()
        if ("headline" in line or "detail" in line) and "=" in line
    ]
    assert literals, "오버레이 문자열 리터럴을 찾지 못했다(테스트가 낡았다)"
    for line in literals:
        assert line.isascii(), f"오버레이 문자열에 비-ASCII가 섞였다: {line.strip()}"
