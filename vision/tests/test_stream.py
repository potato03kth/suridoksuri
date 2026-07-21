"""vision/utils/stream.py — MJPEG-over-HTTP 라이브 스트림 어댑터 테스트 (vision_plan.md §7.9 항목5).

실제 HTTP 서버를 띄우고, 실제 프레임(합성 이미지 + DirFrameSource로 조달한 골든셋 프레임)을
실제로 흘려보낸 뒤, 실제 HTTP 클라이언트로 /stream 에 접속해 진짜 MJPEG 바이트를 받아
cv2.imdecode로 디코드해서 유효한 이미지인지 검증한다. 비차단/드롭 동작도 실제 느린 컨슈머
상황을 만들어 검증한다(pseudo 테스트 금지 — vision_status.md 세션 지시).

RPi/실카메라 접속 없음 — DirFrameSource로 골든셋 폴더에서 프레임을 조달한다(허용된 경로).
"""
import http.client
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.utils.frame_source import DirFrameSource
from vision.utils.stream import MjpegStreamer, _DropOldestFrameQueue

_GOLDEN_DIR = Path(__file__).parent / "golden" / "vertiport" / "10m"


def _read_one_mjpeg_frame(response: http.client.HTTPResponse, timeout: float = 5.0) -> bytes:
    """multipart/x-mixed-replace 스트림에서 파트 하나(JPEG bytes)를 읽어낸다."""
    deadline = time.monotonic() + timeout
    # boundary 라인까지 스킵
    while True:
        line = response.readline()
        if not line:
            raise AssertionError("스트림이 boundary 전에 닫힘")
        if line.strip().startswith(b"--"):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("boundary 대기 시간 초과")
    # 파트 헤더 읽기
    content_length = None
    while True:
        line = response.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        if line.lower().startswith(b"content-length"):
            content_length = int(line.split(b":", 1)[1].strip())
    assert content_length, "MJPEG 파트에 Content-Length 헤더가 없음"
    data = response.read(content_length)
    assert len(data) == content_length
    return data


def _start_producer(streamer: MjpegStreamer, frames, interval: float = 0.02):
    """frames(리스트, 순환)를 별도 스레드에서 계속 push_frame 한다. (stop_flag, thread) 반환."""
    stop_flag = threading.Event()

    def _loop():
        i = 0
        while not stop_flag.is_set():
            streamer.push_frame(frames[i % len(frames)])
            i += 1
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return stop_flag, t


@pytest.fixture
def golden_frames():
    assert _GOLDEN_DIR.exists(), f"골든셋 디렉터리 없음 — {_GOLDEN_DIR}"
    with DirFrameSource(_GOLDEN_DIR) as source:
        frames = [r.image for r in source]
    assert frames, "골든셋에서 프레임을 하나도 못 읽음"
    return frames


def test_stream_serves_real_decodable_mjpeg_frame(golden_frames):
    """실제 서버 기동 -> 실제 골든셋 프레임 흘려보내기 -> 실제 HTTP GET /stream -> 실제 디코드."""
    streamer = MjpegStreamer(host="127.0.0.1", port=0, target_width=640, target_height=480)
    streamer.start()
    stop_flag, producer = _start_producer(streamer, golden_frames)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", streamer.port, timeout=5)
        conn.request("GET", "/stream")
        response = conn.getresponse()
        assert response.status == 200
        assert "multipart/x-mixed-replace" in response.getheader("Content-Type", "")

        jpeg_bytes = _read_one_mjpeg_frame(response)
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        assert decoded is not None, "받은 바이트가 유효한 JPEG로 디코드되지 않음"
        assert decoded.size > 0
        assert decoded.shape[0] <= 480 and decoded.shape[1] <= 640

        conn.close()
    finally:
        stop_flag.set()
        producer.join(timeout=2)
        streamer.stop()


def test_index_page_serves_html(golden_frames):
    streamer = MjpegStreamer(host="127.0.0.1", port=0)
    streamer.start()
    stop_flag, producer = _start_producer(streamer, golden_frames)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", streamer.port, timeout=5)
        conn.request("GET", "/")
        response = conn.getresponse()
        body = response.read()
        assert response.status == 200
        assert b"<img" in body
        assert b"/stream" in body
        conn.close()
    finally:
        stop_flag.set()
        producer.join(timeout=2)
        streamer.stop()


def test_unknown_path_returns_404():
    streamer = MjpegStreamer(host="127.0.0.1", port=0)
    streamer.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", streamer.port, timeout=5)
        conn.request("GET", "/nope")
        response = conn.getresponse()
        response.read()
        assert response.status == 404
        conn.close()
    finally:
        streamer.stop()


def test_downscale_preserves_aspect_and_fits_vga_box():
    """VGA보다 큰 프레임을 흘리면 target box 안으로(종횡비 유지) 축소돼 나와야 한다."""
    big_frame = np.random.randint(0, 255, (960, 1280, 3), dtype=np.uint8)  # 4:3, VGA의 2배
    streamer = MjpegStreamer(host="127.0.0.1", port=0, target_width=640, target_height=480)
    streamer.start()
    stop_flag, producer = _start_producer(streamer, [big_frame], interval=0.02)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", streamer.port, timeout=5)
        conn.request("GET", "/stream")
        response = conn.getresponse()
        jpeg_bytes = _read_one_mjpeg_frame(response)
        decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        # 1280x960(4:3)은 정확히 640x480으로 딱 맞게 축소된다(종횡비 동일)
        assert decoded.shape[:2] == (480, 640)
        conn.close()
    finally:
        stop_flag.set()
        producer.join(timeout=2)
        streamer.stop()


def test_smaller_than_vga_frame_not_upscaled():
    small_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    streamer = MjpegStreamer(host="127.0.0.1", port=0, target_width=640, target_height=480)
    streamer.start()
    stop_flag, producer = _start_producer(streamer, [small_frame], interval=0.02)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", streamer.port, timeout=5)
        conn.request("GET", "/stream")
        response = conn.getresponse()
        jpeg_bytes = _read_one_mjpeg_frame(response)
        decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape[:2] == (240, 320)  # 업스케일 안 됨
        conn.close()
    finally:
        stop_flag.set()
        producer.join(timeout=2)
        streamer.stop()


# ---------- 비침습(non-blocking) 검증 ----------

def test_drop_oldest_frame_queue_never_grows_past_maxsize():
    """primitive 자체를 화이트박스로 검증 — 인코더 소비 속도와 무관하게 상한이 지켜지는지."""
    q = _DropOldestFrameQueue(maxsize=2)
    for i in range(200):
        q.put(np.full((4, 4, 3), i % 256, dtype=np.uint8))
    assert q._q.qsize() <= 2


def test_push_frame_is_non_blocking_and_fast_with_no_consumer():
    """클라이언트가 하나도 없어도(인코더 큐만 도는 상황) push_frame은 즉시 반환해야 한다."""
    streamer = MjpegStreamer(host="127.0.0.1", port=0, max_queue=2)
    streamer.start()
    try:
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(500):
            streamer.push_frame(frame)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"push_frame이 500회에 {elapsed:.3f}s — 블로킹 의심"
    finally:
        streamer.stop()


def test_push_frame_before_start_is_a_safe_noop():
    streamer = MjpegStreamer(host="127.0.0.1", port=0)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    streamer.push_frame(frame)  # start() 전 — 예외 없이 조용히 무시돼야 한다


def test_push_frame_non_blocking_with_real_slow_consumer_connected():
    """실제 느린(응답 body를 안 읽는) HTTP 클라이언트를 붙여 놓고도 push_frame이
    막히지 않는지 검증한다 — §7.9 비침습 전제의 핵심 요구사항."""
    streamer = MjpegStreamer(host="127.0.0.1", port=0, max_queue=2, target_width=640, target_height=480)
    streamer.start()

    # 실제 클라이언트를 연결해 응답 헤더까지만 받고, body(프레임들)는 절대 read()하지 않는다.
    slow_conn = http.client.HTTPConnection("127.0.0.1", streamer.port, timeout=5)
    slow_conn.request("GET", "/stream")
    slow_response = slow_conn.getresponse()
    assert slow_response.status == 200

    stop_flag = threading.Event()

    def _feed():
        big_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        while not stop_flag.is_set():
            streamer.push_frame(big_frame)

    feeder = threading.Thread(target=_feed, daemon=True)
    feeder.start()
    time.sleep(0.5)  # 핸들러 스레드의 소켓 송신 버퍼를 채울 시간을 준다(느린 클라이언트가 안 읽으므로)

    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    t0 = time.perf_counter()
    for _ in range(300):
        streamer.push_frame(frame)
    elapsed = time.perf_counter() - t0

    stop_flag.set()
    feeder.join(timeout=2)
    slow_conn.close()
    streamer.stop()

    assert elapsed < 1.0, (
        f"느린 컨슈머가 붙어있는 상태에서 push_frame 300회에 {elapsed:.3f}s "
        "— 파이프라인 스레드가 HTTP 송신에 의해 블로킹된 것으로 의심됨"
    )


def test_stop_and_restart_is_clean():
    streamer = MjpegStreamer(host="127.0.0.1", port=0)
    streamer.start()
    streamer.push_frame(np.zeros((10, 10, 3), dtype=np.uint8))
    streamer.stop()
    # 재시작 가능해야 한다(idempotent 설계 확인)
    streamer.start()
    conn = http.client.HTTPConnection("127.0.0.1", streamer.port, timeout=5)
    conn.request("GET", "/")
    response = conn.getresponse()
    response.read()
    assert response.status == 200
    conn.close()
    streamer.stop()
