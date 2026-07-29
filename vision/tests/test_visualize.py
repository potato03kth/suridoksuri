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


# ---------------------------------------------------------------------------
# `draw_landing_overlay` — 착륙 판단 오버레이 (2026-07-29, 2차예선 보고 제출 영상)
#
# **왜 있는가:** `draw_detections()`가 그리는 것은 초록 얇은 bbox + 신뢰도 숫자뿐이라, 우선
# 타겟인 ② 조난자 구역(**초록 매트**) 위에서는 배경에 묻혀 "인식되고 있다"가 눈에 안 보였다
# (실제로 뽑아 본 녹화 영상에서 확인). 게다가 파이프라인이 실제로 판단하는 값들 — 착륙점,
# 흰 박스, 상태머신 상태/명령, 추정거리 — 은 JSONL에만 있고 화면엔 한 픽셀도 없었다.
#
# 이 함수의 계약은 **"그리기만 한다"** 이다 — 착륙점을 다시 계산하지 않고
# `det.meta["white_box_detector"]["landing_point_px"]`를 그대로 읽는다. 아래
# `test_landing_point_follows_the_meta_value_and_is_not_recomputed`가 그걸 고정한다.
# ---------------------------------------------------------------------------
from vision.utils.visualize import (  # noqa: E402 - 절 구분을 위해 여기서 import
    LANDING_OVERLAY_BOX_COLOR,
    LANDING_OVERLAY_POINT_COLOR,
    LANDING_OVERLAY_TARGET_COLOR,
    draw_landing_overlay,
)


def _mat_detection(landing_px=(300.0, 200.0), box_bbox=(280, 180, 40, 40)) -> Detection:
    """`modules/distress_box.py::WhiteBoxDetector`가 확정 검출에 붙이는 meta 형태 그대로."""
    return Detection(
        bbox=(200, 120, 200, 160),
        confidence=1.0,
        meta={
            "white_box_detector": {
                "box_bbox": list(box_bbox),
                "landing_point_px": list(landing_px),
            }
        },
    )


def test_landing_point_follows_the_meta_value_and_is_not_recomputed():
    """🔴 이 오버레이의 핵심 계약 — 착륙점을 **재계산하지 않고** meta 값을 그대로 그린다.
    (규칙을 두 곳에 복사하면 한쪽만 고쳐졌을 때 조용히 어긋난다 — `modules/distress_mat.py`가
    같은 이유로 재구현을 거부하는 것과 같은 원칙.) meta의 픽셀을 옮기면 십자도 따라 움직여야
    하고, bbox 중심 같은 자체 계산으로 그리면 두 이미지가 같아져 red가 된다."""
    a = draw_landing_overlay(_blank(), [_mat_detection(landing_px=(260.0, 160.0))])
    b = draw_landing_overlay(_blank(), [_mat_detection(landing_px=(340.0, 240.0))])
    assert _count_colour(a, LANDING_OVERLAY_POINT_COLOR) > 0
    assert not np.array_equal(a, b), "meta의 착륙점을 안 읽고 자체 계산으로 그리고 있다"

    # 십자 중심이 실제로 meta 픽셀 근처인지 — 좌표를 통째로 무시하면 여기서 잡힌다.
    ys, xs = np.where(np.all(a == np.array(LANDING_OVERLAY_POINT_COLOR, dtype=np.uint8), axis=2))
    # "LANDING PT" 라벨도 같은 색이라 오른쪽으로 퍼진다 — 십자 세로선이 있는 x 근방만 본다.
    assert abs(int(np.median(ys)) - 160) <= 20
    assert xs.min() <= 260 <= xs.max()


def test_no_landing_point_drawn_when_the_detection_has_no_white_box_meta():
    """coarse 전용 프리셋(흰 박스 단계 없음)에서는 착륙점이 아직 없다 — 없는 것을 지어내지
    않는다(`core/wire.py`가 "가장 위험한 거짓말"이라 부른 그것). 타겟 bbox는 그린다."""
    img = draw_landing_overlay(_blank(), [Detection(bbox=(200, 120, 200, 160))])
    assert _count_colour(img, LANDING_OVERLAY_TARGET_COLOR) > 0
    assert _count_colour(img, LANDING_OVERLAY_POINT_COLOR) == 0
    assert _count_colour(img, LANDING_OVERLAY_BOX_COLOR) == 0


def test_target_box_colour_is_not_the_detection_green_that_vanishes_on_a_green_mat():
    """🔴 이 오버레이가 만들어진 이유 자체 — 초록 매트 위 초록 선은 안 보인다. 타겟 박스
    색이 `draw_detections`의 초록(0,200,0)으로 되돌아가면 red가 된다."""
    assert LANDING_OVERLAY_TARGET_COLOR != (0, 200, 0)
    green_mat = np.zeros((480, 640, 3), dtype=np.uint8)
    green_mat[:, :] = (60, 180, 60)
    img = draw_landing_overlay(green_mat.copy(), [_mat_detection()])
    assert _count_colour(img, LANDING_OVERLAY_TARGET_COLOR) > 0


def test_state_and_command_actually_reach_the_pixels():
    """상태/명령 문자열이 실제로 그려지는지 — 라벨만 그리고 값을 버리면 두 이미지가 같아진다."""
    servo = draw_landing_overlay(_blank(), [_mat_detection()], state="PRECISION_SERVO",
                                 command="center")
    hold = draw_landing_overlay(_blank(), [_mat_detection()], state="HOLD", command="hold")
    assert not np.array_equal(servo, hold)


def test_estimate_distance_reaches_the_pixels_and_is_omitted_when_absent():
    """추정거리(z)도 값이 실제로 그려져야 하고, `estimate`가 없으면 그 줄 자체가 없어야 한다
    (0을 채워 "기체 바로 아래" 같은 거짓말을 만들지 않는다)."""
    class _Est:
        target_type = "distress_landing_point"
        position = (0.1, 0.2, 12.5)

    class _Far(_Est):
        position = (0.1, 0.2, 31.0)

    near = draw_landing_overlay(_blank(), [_mat_detection()], estimate=_Est())
    far = draw_landing_overlay(_blank(), [_mat_detection()], estimate=_Far())
    none = draw_landing_overlay(_blank(), [_mat_detection()], estimate=None)
    assert not np.array_equal(near, far), "거리 값이 픽셀에 반영되지 않는다"
    assert not np.array_equal(near, none)


def test_landing_point_colour_does_not_collide_with_the_sink_alert_colour():
    """🔴 두 오버레이를 같이 켠 프레임에서 같은 색이 다른 뜻을 가지면 안 된다 — 착륙점 색을
    sink 경고색(빨강)으로 되돌리면 red가 된다."""
    assert LANDING_OVERLAY_POINT_COLOR != SINK_OVERLAY_ALERT_COLOR
    assert LANDING_OVERLAY_POINT_COLOR != SINK_OVERLAY_OK_COLOR
    assert LANDING_OVERLAY_POINT_COLOR != SINK_OVERLAY_OFF_COLOR


def test_landing_overlay_panel_is_at_the_bottom_and_does_not_hide_the_sink_panel():
    """sink 패널(좌상단)과 이 패널(좌하단)을 같이 켜도 서로 안 가려야 한다 — 두 오버레이를
    동시에 쓰는 것이 보고 영상의 실제 사용 형태다. 패널을 상단으로 옮기면 red가 된다."""
    img = draw_sink_status(_blank(), enabled=True, consumers=0, seq=1, dropped=0,
                           endpoint="127.0.0.1:8091")
    top_before = img[:120].copy()          # sink 패널이 사는 영역
    # 검출 그리기는 프레임 어디에나 갈 수 있으므로(그건 이 계약이 아니다) 패널만 그리게 한다.
    draw_landing_overlay(img, [], state="HOLD", command="hold")
    assert np.array_equal(img[:120], top_before), "착륙 패널이 sink 패널 영역을 침범했다"
    assert _count_colour(img, SINK_OVERLAY_ALERT_COLOR) > 0
    assert not np.array_equal(img[-120:], _blank()[-120:]), "하단에 패널이 안 그려졌다"


def test_landing_overlay_draws_in_place_and_preserves_shape_and_dtype():
    """`draw_sink_status`와 같은 계약 — 사본을 또 만들지 않는다(매 프레임 전체 복사 금지)."""
    img = _blank()
    out = draw_landing_overlay(img, [_mat_detection()], state="LOCK")
    assert out is img
    assert out.shape == (480, 640, 3) and out.dtype == np.uint8


def test_landing_overlay_is_deterministic():
    a = draw_landing_overlay(_blank(), [_mat_detection()], state="LOCK", command="hold",
                             frame_id=7)
    b = draw_landing_overlay(_blank(), [_mat_detection()], state="LOCK", command="hold",
                             frame_id=7)
    assert np.array_equal(a, b)


def test_landing_overlay_survives_tiny_and_huge_frames_without_crashing():
    """실기체 4608px과 테스트용 초소형 프레임 양쪽에서 크래시 없이 그려져야 한다."""
    for w, h in ((64, 48), (4608, 2592)):
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        draw_landing_overlay(img, [_mat_detection()], state="TERMINAL", command="land",
                             frame_id=1)


def test_landing_overlay_handles_empty_detections():
    img = draw_landing_overlay(_blank(), [], state="ACQUIRE", command="scan")
    assert _count_colour(img, LANDING_OVERLAY_POINT_COLOR) == 0


def test_landing_overlay_text_is_ascii_only():
    """Hershey 폰트는 한글을 못 그린다 — 화면 문자열이 ASCII임을 소스에서 고정한다."""
    import inspect

    import vision.utils.visualize as vis

    src = inspect.getsource(vis.draw_landing_overlay)
    literals = [line for line in src.splitlines() if 'f"' in line or '"LANDING' in line]
    assert literals, "오버레이 문자열 리터럴을 찾지 못했다(테스트가 낡았다)"
    for line in literals:
        assert line.isascii(), f"오버레이 문자열에 비-ASCII가 섞였다: {line.strip()}"
