"""
RPi5(Ubuntu, libcamera/GStreamer 스택) 헤드리스 캘리브레이션 촬영 도구.

RPi에 디스플레이가 없어서 프레이밍(각도/거리)을 눈으로 볼 수 없다는 문제와,
셔터를 누를 물리 버튼이 없다는 문제를 함께 푼다:

- 저해상도 스냅샷을 ~1초 간격으로 찍어 브라우저에서 자동갱신 이미지로 본다
  (진짜 MJPEG 스트림이 아니라 정지샷 갱신 — 체커보드 위치 잡는 용도라 이걸로 충분).
- 같은 화면의 "촬영" 버튼(HTTP) 또는 SSH 터미널에서 Enter로 풀해상도 정지샷을 찍는다.

**주의: 이 RPi는 Raspberry Pi OS가 아니라 Ubuntu — picamera2/rpicam-apps 없음.**
대신 libcamera의 GStreamer 플러그인(`libcamerasrc`)을 서브프로세스로 호출한다.
사전 설치 (사용자가 sudo로 직접):
    sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-base \\
        gstreamer1.0-plugins-good gstreamer1.0-libcamera

RPi에서 실행:
    python3 vision/tools/rpi_capture.py --out vision/data/calibration_raw

노트북에서: http://<rpi-ip>:8000/ 열어서 위치 잡고 버튼으로 촬영, 또는 SSH 터미널에서 Enter.
"""
import argparse
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_GST_LAUNCH = "gst-launch-1.0"

_PAGE = """<!doctype html><html><body style="margin:0;background:#111">
<img id="p" src="/preview.jpg" style="width:100%;display:block"
     onerror="setTimeout(()=>{this.src='/preview.jpg?t='+Date.now()}, 500)"
     onload="setTimeout(()=>{this.src='/preview.jpg?t='+Date.now()}, 400)">
<form method="POST" action="/capture">
<button style="width:100%;padding:24px;font-size:24px" type="submit">촬영</button>
</form>
</body></html>"""


def _gst_capture_jpeg(width: int, height: int, out_path: Path, timeout: float = 10.0) -> None:
    """libcamerasrc로 프레임 1장을 잡아 out_path에 JPEG로 저장. 카메라는 동시에 1개 세션만 열 수 있어
    호출 측(_CaptureSession)이 락으로 직렬화해야 한다."""
    cmd = [
        _GST_LAUNCH, "-e", "libcamerasrc", "num-buffers=1", "!",
        f"video/x-raw,width={width},height={height}", "!",
        "videoconvert", "!", "jpegenc", "!",
        "filesink", f"location={out_path}",
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)


class _Handler(BaseHTTPRequestHandler):
    camera: "_CaptureSession"

    def log_message(self, *args):
        pass  # 콘솔에 요청 로그 스팸 안 냄

    def do_GET(self):
        if self.path == "/":
            self._send_bytes(_PAGE.encode(), "text/html; charset=utf-8")
        elif self.path.startswith("/preview.jpg"):
            jpeg = self.camera.latest_preview_jpeg()
            if jpeg is None:
                self.send_response(503)
                self.end_headers()
            else:
                self._send_bytes(jpeg, "image/jpeg")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/capture":
            path = self.camera.capture()
            self._send_bytes(f"saved: {path}".encode(), "text/plain; charset=utf-8")
        else:
            self.send_response(404)
            self.end_headers()

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class _CaptureSession:
    """카메라는 한 번에 한 프로세스만 열 수 있어, 미리보기 루프와 촬영 트리거가
    같은 락을 공유해 gst-launch 호출을 직렬화한다."""

    def __init__(self, out_dir: Path, preview_size: tuple[int, int], main_size: tuple[int, int]):
        self._out_dir = out_dir
        self._preview_size = preview_size
        self._main_size = main_size
        self._lock = threading.Lock()
        self._count = 0
        self._preview_path = out_dir / "_preview.jpg"
        self._preview_jpeg: bytes | None = None

    def preview_tick(self) -> None:
        w, h = self._preview_size
        with self._lock:
            try:
                _gst_capture_jpeg(w, h, self._preview_path, timeout=5.0)
                self._preview_jpeg = self._preview_path.read_bytes()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"[미리보기 실패] {e}", file=sys.stderr)

    def latest_preview_jpeg(self) -> bytes | None:
        return self._preview_jpeg

    def capture(self) -> str:
        w, h = self._main_size
        with self._lock:
            self._count += 1
            path = self._out_dir / f"calib_{self._count:03d}.jpg"
            _gst_capture_jpeg(w, h, path, timeout=15.0)
            print(f"[촬영] {path}")
            return str(path)


def _preview_loop(session: _CaptureSession, interval: float) -> None:
    while True:
        session.preview_tick()
        time.sleep(interval)


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
    parser.add_argument("--preview-size", default="640x480", help="미리보기 스냅샷 크기 WxH")
    parser.add_argument("--preview-interval", type=float, default=1.0, help="미리보기 갱신 주기(초)")
    args = parser.parse_args()

    if shutil.which(_GST_LAUNCH) is None:
        print(
            f"{_GST_LAUNCH}가 없다 — 이 RPi는 Ubuntu라 picamera2/rpicam-apps가 아니라 GStreamer가 필요하다.\n"
            "  sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-base "
            "gstreamer1.0-plugins-good gstreamer1.0-libcamera",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_w, main_h = (int(v) for v in args.main_size.split("x"))
    prev_w, prev_h = (int(v) for v in args.preview_size.split("x"))

    session = _CaptureSession(out_dir, (prev_w, prev_h), (main_w, main_h))
    threading.Thread(target=_preview_loop, args=(session, args.preview_interval), daemon=True).start()
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


if __name__ == "__main__":
    main()
