"""
라이브 저해상 MJPEG-over-HTTP 스트리밍 어댑터.

vision_plan.md §7.9 "지금 당장 할 일" 5번, "작동영상 피드백 3경로" 표의 (b)행:
파이프라인이 처리 중인 프레임(compute_tap)을 VGA급으로 다운스케일해 MJPEG-over-HTTP로
내보낸다. 같은 네트워크의 랩탑 브라우저에서 http://<host>:<port>/stream 으로 실시간 관찰.
표준 라이브러리 http.server만 사용(§7.9 (b) "MJPEG-over-HTTP 또는 ROS2 image_transport" 중
이번 세션은 MJPEG-over-HTTP만 — ROS2 경로는 도메인 간 import 금지 원칙상 이번 범위 밖).

비침습 설계(§7.9 "비침습 전제: dev 계측이 제어루프를 지연시키면 안 된다"):
  - push_frame()은 절대 블로킹하지 않는다 — bounded queue + drop-oldest.
    vision/utils/blackbox.py의 _DropOldestQueueHandler와 동일한 패턴을 그대로 재사용한다
    (새 패턴을 발명하지 않는다).
  - 다운스케일·JPEG 인코딩·HTTP 서빙은 전부 별도 스레드에서 일어난다. 클라이언트가 없거나
    느려도(HTTP 핸들러 스레드만 영향받음) push_frame()을 호출하는 파이프라인 스레드는 절대
    영향받지 않는다.

기본값(vision/CLAUDE.md에 근거 기록): 640x480(VGA), 포트 8080, JPEG quality 80,
큐 길이 2(최신 프레임 위주 관찰이 목적이라 지연 누적보다 최신성 우선).
다운스케일은 종횡비를 유지한 채 target 박스 안에 맞추는 방식(letterbox 패딩 없음) —
관찰용 브라우저 <img> 태그가 알아서 크기를 맞추므로 최소본으로 충분하다고 판단.
"""
from __future__ import annotations

import queue
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

_BOUNDARY = "visionframe"


class _DropOldestFrameQueue:
    """큐가 가득 차면 가장 오래된 프레임을 버리고 최신 프레임을 넣는다 — 절대 블로킹하지 않는다.

    vision/utils/blackbox.py의 _DropOldestQueueHandler.enqueue와 동일한 패턴(일관성 유지).
    """

    def __init__(self, maxsize: int):
        self._q: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._put_lock = threading.Lock()  # 여러 producer 스레드가 동시에 push할 수 있어 evict+insert를 원자화한다

    def put(self, item) -> None:
        with self._put_lock:
            while True:
                try:
                    self._q.put_nowait(item)
                    return
                except queue.Full:
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass

    def get(self, timeout: Optional[float] = None):
        return self._q.get(timeout=timeout)


def _fit_within(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """종횡비를 유지한 채 (max_w, max_h) 박스 안에 들어가도록 축소한다. 업스케일은 하지 않는다."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return frame
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class _StreamHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class MjpegStreamer:
    """VGA급 다운스케일 + MJPEG-over-HTTP 라이브 스트림.

    사용법:
        streamer = MjpegStreamer(port=8080)
        streamer.start()
        ...
        streamer.push_frame(annotated_bgr)   # 파이프라인 루프 안에서, 매 프레임마다(비차단)
        ...
        streamer.stop()

    브라우저에서 http://<host>:<port>/stream (또는 / 에서 <img> 미리보기 페이지).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        target_width: int = 640,
        target_height: int = 480,
        jpeg_quality: int = 80,
        max_queue: int = 2,
    ):
        self.host = host
        self.port = port
        self.target_width = target_width
        self.target_height = target_height
        self.jpeg_quality = jpeg_quality

        self._frame_queue = _DropOldestFrameQueue(max_queue)
        self._latest_jpeg: Optional[bytes] = None
        self._frame_seq = 0
        self._cond = threading.Condition()

        self._stop_event = threading.Event()
        self._encoder_thread: Optional[threading.Thread] = None
        self._httpd: Optional[_StreamHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

    # ---------- 파이프라인 스레드에서 호출 ----------

    def start(self) -> None:
        """HTTP 서버 + 인코더 스레드를 띄운다. 이미 시작돼 있으면 아무 것도 안 한다(idempotent)."""
        if self._server_thread is not None:
            return
        self._stop_event.clear()

        streamer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # 콘솔에 접속 로그 스팸 안 냄
                pass

            def do_GET(self):
                if self.path == "/stream":
                    streamer._serve_stream(self)
                elif self.path in ("/", "/index.html"):
                    streamer._serve_index(self)
                else:
                    self.send_error(404)

        self._httpd = _StreamHTTPServer((self.host, self.port), _Handler)
        self.port = self._httpd.server_address[1]  # port=0(임시 포트) 요청 시 실제 바인딩 포트로 갱신

        self._server_thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True, name="mjpeg-http"
        )
        self._server_thread.start()

        self._encoder_thread = threading.Thread(
            target=self._encode_loop, daemon=True, name="mjpeg-encoder"
        )
        self._encoder_thread.start()

    def push_frame(self, frame_bgr: Optional[np.ndarray]) -> None:
        """파이프라인 루프에서 매 프레임 호출. 절대 블로킹하지 않는다(bounded queue+drop-oldest).

        start() 전에 호출하면 조용히 무시한다(스트림이 opt-in이므로 안전한 기본 동작).
        """
        if self._server_thread is None or frame_bgr is None:
            return
        self._frame_queue.put(frame_bgr)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._encoder_thread is not None:
            self._encoder_thread.join(timeout=timeout)
        if self._server_thread is not None:
            self._server_thread.join(timeout=timeout)
        self._encoder_thread = None
        self._server_thread = None
        self._httpd = None

    @property
    def url(self) -> str:
        host = self.host if self.host not in ("0.0.0.0", "") else "localhost"
        return f"http://{host}:{self.port}/stream"

    # ---------- 인코더 스레드 ----------

    def _encode_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            resized = _fit_within(frame, self.target_width, self.target_height)
            ok, buf = cv2.imencode(
                ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not ok:
                continue
            with self._cond:
                self._latest_jpeg = buf.tobytes()
                self._frame_seq += 1
                self._cond.notify_all()

    # ---------- HTTP 핸들러 스레드(클라이언트당 1개) ----------

    def _serve_index(self, handler: BaseHTTPRequestHandler) -> None:
        body = (
            b"<html><body style='margin:0;background:#111'>"
            b"<img src='/stream' style='max-width:100%;height:auto' /></body></html>"
        )
        handler.send_response(200)
        handler.send_header("Content-Type", "text/html")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _serve_stream(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            handler.send_response(200)
            handler.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}"
            )
            handler.send_header("Cache-Control", "no-cache")
            handler.send_header("Pragma", "no-cache")
            handler.end_headers()

            last_seq = -1
            while not self._stop_event.is_set():
                with self._cond:
                    self._cond.wait_for(
                        lambda: self._stop_event.is_set() or self._frame_seq != last_seq,
                        timeout=1.0,
                    )
                    jpeg = self._latest_jpeg
                    last_seq = self._frame_seq
                if self._stop_event.is_set():
                    break
                if jpeg is None:
                    continue
                handler.wfile.write(f"--{_BOUNDARY}\r\n".encode())
                handler.wfile.write(b"Content-Type: image/jpeg\r\n")
                handler.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                handler.wfile.write(jpeg)
                handler.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, socket.error):
            pass  # 클라이언트 연결 끊김 — 정상 종료, 파이프라인에는 영향 없음
