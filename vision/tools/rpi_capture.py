"""
RPi5 + Camera Module 3 헤드리스 캘리브레이션 촬영 도구.

RPi에 디스플레이가 없어서 프레이밍(각도/거리)을 눈으로 볼 수 없다는 문제와,
셔터를 누를 물리 버튼이 없다는 문제를 함께 푼다:

- 저해상도 미리보기를 MJPEG로 네트워크에 뿌려 노트북 브라우저에서 실시간으로 본다.
- 같은 화면의 "촬영" 버튼(HTTP) 또는 SSH 터미널에서 Enter로 풀해상도 정지샷을 찍는다.

RPi에서 실행 (picamera2가 이미 RPi OS에 포함돼 있음, 이 저장소 .venv와 무관):
    python3 vision/tools/rpi_capture.py --out vision/data/calibration_raw

노트북에서: http://<rpi-ip>:8000/ 열어서 위치 잡고 버튼으로 촬영, 또는 SSH 터미널에서 Enter.
"""
import argparse
import io
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from picamera2 import Picamera2
except ImportError:
    print("picamera2가 없다 — 이 스크립트는 RPi OS(Bookworm+)에서만 동작한다.", file=sys.stderr)
    sys.exit(1)

_PAGE = """<!doctype html><html><body style="margin:0;background:#111">
<img src="/stream" style="width:100%;display:block">
<form method="POST" action="/capture">
<button style="width:100%;padding:24px;font-size:24px" type="submit">촬영</button>
</form>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    camera: "_CaptureSession"

    def log_message(self, *args):
        pass  # 콘솔에 요청 로그 스팸 안 냄

    def do_GET(self):
        if self.path == "/":
            body = _PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
            self._stream_mjpeg()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/capture":
            path = self.camera.capture()
            body = f"saved: {path}".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def _stream_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                jpeg = self.camera.next_preview_jpeg()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(0.15)  # 프레이밍용이라 초당 몇 장이면 충분
        except (BrokenPipeError, ConnectionResetError):
            pass  # 브라우저가 탭을 닫은 것 — 정상


class _CaptureSession:
    def __init__(self, picam2: Picamera2, out_dir: Path):
        self._picam2 = picam2
        self._out_dir = out_dir
        self._lock = threading.Lock()
        self._count = 0

    def next_preview_jpeg(self) -> bytes:
        import cv2
        frame = self._picam2.capture_array("lores")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()

    def capture(self) -> str:
        with self._lock:
            self._count += 1
            path = self._out_dir / f"calib_{self._count:03d}.jpg"
            self._picam2.capture_file(str(path))
            print(f"[촬영] {path}")
            return str(path)


def _enter_key_trigger(session: _CaptureSession) -> None:
    """SSH 터미널에서 Enter만 눌러도 촬영 — 브라우저 버튼과 동등한 경로."""
    while True:
        try:
            input()
        except EOFError:
            return
        session.capture()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="vision/data/calibration_raw", help="촬영 저장 폴더")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--main-size", default="4608x2592", help="풀해상도 촬영 크기 WxH")
    parser.add_argument("--preview-size", default="640x480", help="미리보기 스트림 크기 WxH")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_w, main_h = (int(v) for v in args.main_size.split("x"))
    prev_w, prev_h = (int(v) for v in args.preview_size.split("x"))

    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (main_w, main_h)},
        lores={"size": (prev_w, prev_h), "format": "YUV420"},
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)  # 노출/화밸 안정화

    session = _CaptureSession(picam2, out_dir)
    threading.Thread(target=_enter_key_trigger, args=(session,), daemon=True).start()

    handler = type("_BoundHandler", (_Handler,), {"camera": session})
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(f"미리보기: http://<이 라즈베리파이의 IP>:{args.port}/  (SSH 터미널에서 Enter로도 촬영)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        picam2.stop()


if __name__ == "__main__":
    main()
