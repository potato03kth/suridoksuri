"""
RPi5(Ubuntu, rp1-cfe V4L2) 헤드리스 캘리브레이션 촬영 도구 — V4L2 RAW 직접 캡처 + 수동 디베이어.

이 RPi는 libcamera가 PiSP IPA 모듈(ipa_rpi_pisp.so) 없이 빌드돼 있어 libcamera 상위 레벨
(picamera2/GStreamer libcamerasrc 포함)이 전부 막혀 있다(메모리 project_rpi5_ubuntu_camera_stack.md
참고). 커널 V4L2/media-controller 레벨은 정상이므로 libcamera를 완전히 우회해 V4L2 RAW를 직접
캡처하고 베이어를 수동으로 디베이어한다.

장착 카메라: 서드파티 클론 CAM109-IMX708AF-75 (IMX708 센서). 커널에 다음으로 잡힌다
(실기체 media-ctl -p -d /dev/media1 로 확인, 2026-07-22):
    /dev/video0        = rp1-cfe-csi2_ch0 (캡처 노드)
    /dev/v4l-subdev0   = csi2 (CSI-2 수신기)
    /dev/v4l-subdev2   = imx708 (센서)
픽셀포맷: 'pRAA' (V4L2_PIX_FMT_SRGGB10P, MIPI RAW10 패킹) — v4l2-ctl -d /dev/video0
--list-formats-ext 로 확인. imx708 소스 패드 mbus code는 SRGGB10_1X10 → 베이어 패턴 RGGB.

## media-controller 파이프라인이 필요한 이유 (2026-07-22 실기체로 확정)

/dev/video0는 "MC-centric" 캡처 노드라 단순히 --set-fmt-video만으로는 VIDIOC_STREAMON이 안
된다. 아래 3가지를 media-ctl/v4l2-ctl로 명시적으로 맞춰야 media_pipeline_start()가 통과한다
(각각 실제로 막혔던 지점 — dmesg dynamic_debug로 확인):

  1. **링크 활성화** — `"csi2":4 -> "rp1-cfe-csi2_ch0":0` 링크가 기본 비활성 상태
     (커널 로그: "csi2_ch0 node link is not enabled."). media-ctl -l 로 활성화해야 한다.
  2. **필드(field) 일치** — imx708 소스 패드는 field=None인데 csi2 싱크 패드0 기본값은
     field=Any. `v4l2_subdev_link_validate_default`가 이 둘을 다르다고 보고 링크 검증을
     거부한다(커널 로그: "field does not match", VIDIOC_STREAMON이 -EPIPE/-32로 실패).
     media-ctl -V 로 csi2 패드0도 명시적으로 field:none 으로 맞춰야 한다.
  3. **임베디드 메타데이터 패드 폭 일치** — imx708 패드1(임베디드 데이터, IMMUTABLE 링크라
     항상 파이프라인 그래프에 포함됨)의 실제 폭은 28800인데 csi2 싱크 패드1 기본값은 16384
     (커널 로그: "width does not match", 마찬가지로 -EPIPE). imx708 패드1의 *실제* 폭을
     읽어(하드코딩 금지 — 모드에 따라 달라질 수 있음) csi2 싱크 패드1에 그대로 맞춰야 한다.
     media-ctl -V는 SENSOR_DATA 심볼릭 코드명을 못 받아들여 실패하므로 v4l2-ctl
     --set-subdev-fmt로 mbus code 0x7002(MEDIA_BUS_FMT_SENSOR_DATA)를 숫자로 직접 지정한다.

이 셋을 다 맞추기 전엔 VIDIOC_STREAMON이 "Broken pipe"(-EPIPE)로 실패한다 — 재현 명령·근본
원인 전체 경과는 `docs/vision_status.md` 2026-07-22 항목 참고.

## 센서 모드 스냅

imx708는 임의 해상도를 그대로 받지 않고 가장 가까운 지원 모드로 스냅한다(실기체 확인:
640x480 요청 시 1536x864로 스냅됨). 그래서 `configure_pipeline()`은 센서에 설정을 요청한
직후 실제 적용된 크기를 다시 읽어(readback) 그 값을 나머지 파이프라인(csi2 싱크/소스 패드,
video0 포맷)에 그대로 전파한다 — 요청값을 그대로 믿지 않는다.

RPi에서 실행 (HTTP 미리보기+촬영 서버):
    python3 vision/tools/rpi_capture.py --out vision/data/calibration_raw
노트북에서: http://<rpi-ip>:8000/ 열어서 위치 잡고 버튼으로 촬영, 또는 SSH 터미널에서 Enter.

RPi에서 논-인터랙티브 단발 캡처(스크립트/원격 검증용):
    python3 vision/tools/rpi_capture.py --single-shot --out vision/data/calibration_raw
"""
import argparse
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # RPi에는 항상 있음. 순수 unpack_raw10()만 쓰는 개발 호스트는 cv2 없이도 import 가능.
    cv2 = None

_V4L2_CTL = "v4l2-ctl"
_MEDIA_CTL = "media-ctl"

_VIDEO_DEVICE = "/dev/video0"
_CSI2_SUBDEV = "/dev/v4l-subdev0"
_SENSOR_SUBDEV = "/dev/v4l-subdev2"
_MEDIA_DEVICE = "/dev/media1"

_EMBEDDED_DATA_CODE = 0x7002  # MEDIA_BUS_FMT_SENSOR_DATA
_BAYER_MBUS_CODE = "SRGGB10_1X10"
_PIXELFORMAT = "pRAA"  # V4L2_PIX_FMT_SRGGB10P (10-bit Bayer RGRG/GBGB Packed)

# 센서가 보고하는 베이어 패턴 -> OpenCV cvtColor 코드.
# OpenCV의 Bayer 코드명은 픽셀(0,0)이 아니라 (1,1) 기준으로 패턴을 기술하는 관례라, 센서가
# "RGGB"로 보고해도 OpenCV 코드로는 한 칸 밀린 "BG"가 대응된다(IMX219/IMX708류 라즈베리파이
# raw 캡처에서 흔히 나오는 실수 지점 — 여기서 표로 명시적으로 고정해 둔다).
_BAYER_CV_CODE = {
    "rggb": "COLOR_BayerBG2BGR",
    "bggr": "COLOR_BayerRG2BGR",
    "grbg": "COLOR_BayerGB2BGR",
    "gbrg": "COLOR_BayerGR2BGR",
}


def unpack_raw10(packed: np.ndarray, width: int, height: int) -> np.ndarray:
    """MIPI RAW10 패킹된 바이트 배열(1D uint8) -> (height, width) uint16 배열(값 범위 0~1023).

    4픽셀이 5바이트로 패킹된다(V4L2_PIX_FMT_SRGGB10P/'pRAA'): 앞 4바이트는 각 픽셀의 상위
    8비트, 5번째 바이트는 4픽셀의 하위 2비트씩을 모아 담는다(픽셀0이 최하위 2비트).
    width는 4의 배수여야 한다 — 한 줄 바이트수 = width/4*5 (실기체 4608폭에서 5760바이트/줄로
    실측 확인, video0 --get-fmt-video의 Bytes per Line과 정확히 일치).
    """
    if width % 4 != 0:
        raise ValueError(f"unpack_raw10: width는 4의 배수여야 한다 (width={width})")
    bytes_per_row = width // 4 * 5
    expected = bytes_per_row * height
    if packed.size != expected:
        raise ValueError(
            f"unpack_raw10: 크기 불일치 — 입력 {packed.size}바이트, "
            f"기대값 {expected}바이트 (width={width}, height={height}, "
            f"bytes_per_row={bytes_per_row})"
        )
    rows = packed.reshape(height, bytes_per_row)
    b0 = rows[:, 0::5].astype(np.uint16)
    b1 = rows[:, 1::5].astype(np.uint16)
    b2 = rows[:, 2::5].astype(np.uint16)
    b3 = rows[:, 3::5].astype(np.uint16)
    b4 = rows[:, 4::5].astype(np.uint16)
    p0 = (b0 << 2) | (b4 & 0x03)
    p1 = (b1 << 2) | ((b4 >> 2) & 0x03)
    p2 = (b2 << 2) | ((b4 >> 4) & 0x03)
    p3 = (b3 << 2) | ((b4 >> 6) & 0x03)
    out = np.empty((height, width), dtype=np.uint16)
    out[:, 0::4] = p0
    out[:, 1::4] = p1
    out[:, 2::4] = p2
    out[:, 3::4] = p3
    return out


def apply_gray_world_white_balance(bgr8: np.ndarray) -> np.ndarray:
    """Gray-world 가정 기반 화이트밸런스 보정: BGR uint8 이미지 -> BGR uint8 이미지.

    "전체 이미지의 R/G/B 채널 평균이 같아야 한다"는 gray-world 가정에 따라, 채널별 평균의
    평균(회색 기준값)에 각 채널 평균이 맞춰지도록 채널별 게인(gray/mean_c)을 곱한다. 픽셀별
    보정이 아니라 프레임 전체 통계 1개(채널당 평균)만 쓰는 가장 단순한 변형 — 체커보드 등
    특정 기준 패치나 조명 스펙트럼 추정 없이 numpy만으로 계산 가능해 이 raw 캡처 후처리
    단계에 적합하다고 판단(근거는 vision/CLAUDE.md 참조). 채널 평균이 0인 완전 검은 이미지는
    0으로 나누는 것을 피하기 위해 원본을 그대로 반환한다.
    """
    if bgr8.dtype != np.uint8 or bgr8.ndim != 3 or bgr8.shape[2] != 3:
        raise ValueError(
            f"apply_gray_world_white_balance: (H,W,3) uint8 BGR 배열이 필요하다 "
            f"(입력 shape={bgr8.shape}, dtype={bgr8.dtype})"
        )
    bgr = bgr8.astype(np.float64)
    channel_means = bgr.reshape(-1, 3).mean(axis=0)  # [mean_b, mean_g, mean_r]
    if np.any(channel_means == 0):
        return bgr8.copy()
    gray = channel_means.mean()
    gains = gray / channel_means
    out = bgr * gains  # 브로드캐스트: 마지막 축(채널)별로 게인 적용
    return np.clip(out, 0, 255).astype(np.uint8)


def debayer_to_bgr8(bayer10: np.ndarray, pattern: str = "rggb", white_balance: bool = True) -> np.ndarray:
    """unpack_raw10() 출력(10비트 값을 16비트 컨테이너에 담은 베이어 평면) -> 8비트 BGR.

    cv2.cvtColor의 Bayer 디모자이킹은 8비트 입력을 기대하므로, 먼저 10비트(0~1023)를
    8비트(0~255)로 다운시프트(>>2)한 뒤 디베이어한다. 캘리브레이션 정밀도가 더 필요하면
    unpack_raw10()의 16비트(실질 10비트) 출력을 직접 쓰는 것도 가능 — 이 함수는 미리보기/
    시각 확인/체커보드 코너검출(8비트 입력 요구)용.

    white_balance=True(기본값)면 디베이어 직후 apply_gray_world_white_balance()로 화이트밸런스를
    보정한다 — raw 베이어 경로는 ISP(libcamera)를 완전히 우회하므로 화이트밸런스가 전혀 적용되지
    않은 채 강한 색편향(이 카메라는 초록 편향)이 나타난다(근거는 vision/CLAUDE.md 참조).
    """
    if cv2 is None:
        raise RuntimeError("debayer_to_bgr8: cv2(OpenCV)가 설치돼 있지 않다")
    pattern = pattern.lower()
    if pattern not in _BAYER_CV_CODE:
        raise ValueError(
            f"debayer_to_bgr8: 알 수 없는 베이어 패턴 {pattern!r} "
            f"(지원: {sorted(_BAYER_CV_CODE)})"
        )
    bayer8 = (bayer10 >> 2).astype(np.uint8)
    cv_code = getattr(cv2, _BAYER_CV_CODE[pattern])
    bgr = cv2.cvtColor(bayer8, cv_code)
    if white_balance:
        bgr = apply_gray_world_white_balance(bgr)
    return bgr


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def _parse_subdev_size(output: str) -> tuple:
    """v4l2-ctl --get-subdev-fmt 출력에서 "Width/Height : W/H" 줄을 파싱."""
    for line in output.splitlines():
        if "Width/Height" in line:
            w, h = (int(v) for v in line.split(":", 1)[1].strip().split("/"))
            return w, h
    raise RuntimeError(f"_parse_subdev_size: Width/Height 줄을 못 찾음:\n{output}")


def _sensor_active_size(pad: int) -> tuple:
    """imx708(v4l-subdev2)의 pad 현재 활성 크기를 읽는다(모드 스냅 이후 실제값)."""
    out = _run([_V4L2_CTL, "-d", _SENSOR_SUBDEV, "--get-subdev-fmt", f"pad={pad}"]).stdout
    return _parse_subdev_size(out)


def configure_pipeline(width: int, height: int) -> tuple:
    """미디어 파이프라인 전체(링크+포맷)를 요청 해상도로 맞춘다.

    센서가 요청값을 가장 가까운 지원 모드로 스냅할 수 있으므로, 센서에 설정한 뒤 실제
    적용값을 다시 읽어 나머지 파이프라인에 그 실제값을 전파한다.
    반환값: (실제 폭, 실제 높이) — 요청값과 다를 수 있다.
    """
    # 1. 링크: csi2 소스 패드4 -> rp1-cfe-csi2_ch0 활성화. pisp-fe(ISP) 경로는 이 도구가
    #    쓰지 않으므로 명시적으로 비활성 유지(파이프라인 검증 시 불필요한 경로가 끼어들지
    #    않도록 — 실기체에서는 원래도 비활성 상태였지만 명시적으로 고정해 둔다).
    _run([_MEDIA_CTL, "-d", _MEDIA_DEVICE, "-l", '"csi2":4 -> "rp1-cfe-csi2_ch0":0 [1]'])
    _run([_MEDIA_CTL, "-d", _MEDIA_DEVICE, "-l", '"csi2":4 -> "pisp-fe":0 [0]'])

    # 2. 센서 이미지 소스 패드(pad0)에 요청 후 실제 스냅값 readback
    _run([_MEDIA_CTL, "-d", _MEDIA_DEVICE, "-V",
          f'"imx708":0 [fmt:{_BAYER_MBUS_CODE}/{width}x{height} field:none]'])
    actual_w, actual_h = _sensor_active_size(pad=0)

    # 3. csi2 이미지 싱크(패드0)/소스(패드4)도 실제값+field:none으로 맞춤 — field 불일치가
    #    실기체에서 실제로 -EPIPE 원인이었다(csi2 싱크 기본값 field:any).
    _run([_MEDIA_CTL, "-d", _MEDIA_DEVICE, "-V",
          f'"csi2":0 [fmt:{_BAYER_MBUS_CODE}/{actual_w}x{actual_h} field:none]'])
    _run([_MEDIA_CTL, "-d", _MEDIA_DEVICE, "-V",
          f'"csi2":4 [fmt:{_BAYER_MBUS_CODE}/{actual_w}x{actual_h} field:none]'])

    # 4. 임베디드 메타데이터 패드(imx708 패드1, IMMUTABLE 링크라 파이프라인 그래프에 항상
    #    포함됨) — 실제 폭을 읽어 csi2 싱크 패드1에 그대로 맞춤.
    embed_w, embed_h = _sensor_active_size(pad=1)
    _run([_V4L2_CTL, "-d", _CSI2_SUBDEV, "--set-subdev-fmt",
          f"pad=1,width={embed_w},height={embed_h},code={_EMBEDDED_DATA_CODE:#x},field=none"])

    # 5. 비디오 캡처 노드 포맷 (실제 해상도로)
    _run([_V4L2_CTL, "-d", _VIDEO_DEVICE, "--set-fmt-video",
          f"width={actual_w},height={actual_h},pixelformat={_PIXELFORMAT}"])

    return actual_w, actual_h


def capture_raw_frame(width: int, height: int, out_path: Path) -> tuple:
    """파이프라인을 (width,height)로 구성하고 프레임 1장을 raw 파일로 캡처한다.
    반환값: 실제 적용된 (width, height) — 센서 모드 스냅으로 요청값과 다를 수 있다."""
    actual_w, actual_h = configure_pipeline(width, height)
    _run([_V4L2_CTL, "-d", _VIDEO_DEVICE, "--stream-mmap", "--stream-count=1",
          f"--stream-to={out_path}"])
    return actual_w, actual_h


def capture_frame_bgr(width: int, height: int, raw_tmp_path: Path,
                       bayer_pattern: str = "rggb", white_balance: bool = True) -> np.ndarray:
    """capture_raw_frame + unpack_raw10 + debayer_to_bgr8을 묶은 헬퍼.
    raw_tmp_path에 중간 raw 파일을 남긴다(디버깅용, 매 호출 덮어씀)."""
    actual_w, actual_h = capture_raw_frame(width, height, raw_tmp_path)
    packed = np.fromfile(raw_tmp_path, dtype=np.uint8)
    bayer16 = unpack_raw10(packed, actual_w, actual_h)
    return debayer_to_bgr8(bayer16, pattern=bayer_pattern, white_balance=white_balance)


_PAGE = """<!doctype html><html><body style="margin:0;background:#111">
<img id="p" src="/preview.jpg" style="width:100%;display:block"
     onerror="setTimeout(()=>{this.src='/preview.jpg?t='+Date.now()}, 500)"
     onload="setTimeout(()=>{this.src='/preview.jpg?t='+Date.now()}, 400)">
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
    """카메라(미디어 파이프라인)는 한 번에 한 스트림만 열 수 있어, 미리보기 루프와 촬영
    트리거가 같은 락을 공유해 v4l2-ctl 스트리밍 호출을 직렬화한다."""

    def __init__(self, out_dir: Path, preview_size: tuple, main_size: tuple,
                 bayer_pattern: str = "rggb", white_balance: bool = True):
        self._out_dir = out_dir
        self._preview_size = preview_size
        self._main_size = main_size
        self._bayer_pattern = bayer_pattern
        self._white_balance = white_balance
        self._lock = threading.Lock()
        self._count = 0
        self._raw_tmp = out_dir / "_last_capture.raw"
        self._preview_jpeg = None

    def preview_tick(self) -> None:
        w, h = self._preview_size
        with self._lock:
            try:
                bgr = capture_frame_bgr(w, h, self._raw_tmp, self._bayer_pattern, self._white_balance)
                ok, jpeg = cv2.imencode(".jpg", bgr)
                if ok:
                    self._preview_jpeg = jpeg.tobytes()
            except (subprocess.CalledProcessError, ValueError, RuntimeError) as e:
                print(f"[미리보기 실패] {e}", file=sys.stderr)

    def latest_preview_jpeg(self):
        return self._preview_jpeg

    def capture(self) -> str:
        w, h = self._main_size
        with self._lock:
            self._count += 1
            path = self._out_dir / f"calib_{self._count:03d}.png"
            bgr = capture_frame_bgr(w, h, self._raw_tmp, self._bayer_pattern, self._white_balance)
            cv2.imwrite(str(path), bgr)
            print(f"[촬영] {path} shape={bgr.shape} mean={bgr.mean():.1f} std={bgr.std():.1f}")
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


def _check_tools_available() -> None:
    missing = [t for t in (_V4L2_CTL, _MEDIA_CTL) if shutil.which(t) is None]
    if missing:
        print(
            f"다음 도구가 없다: {', '.join(missing)}\n"
            "  sudo apt install -y v4l-utils media-ctl",
            file=sys.stderr,
        )
        sys.exit(1)
    if cv2 is None:
        print("cv2(OpenCV)가 없다 — RPi에는 python3-opencv(또는 opencv-python-headless) 필요",
              file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="vision/data/calibration_raw", help="촬영 저장 폴더")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--main-size", default="4608x2592", help="풀해상도 촬영 크기 WxH")
    parser.add_argument("--preview-size", default="1536x864",
                         help="미리보기 스냅샷 크기 WxH (센서 지원 모드로 스냅될 수 있음)")
    parser.add_argument("--preview-interval", type=float, default=1.5, help="미리보기 갱신 주기(초)")
    parser.add_argument("--bayer-pattern", default="rggb",
                         choices=sorted(_BAYER_CV_CODE), help="센서 베이어 패턴 (IMX708=rggb)")
    parser.add_argument("--single-shot", action="store_true",
                         help="HTTP 서버 없이 프레임 1장만 main-size로 캡처해 저장하고 종료"
                              "(원격/스크립트 검증용)")
    parser.add_argument("--white-balance", action=argparse.BooleanOptionalAction, default=True,
                         help="Gray-world 가정 기반 화이트밸런스 보정(기본 켜짐). raw 베이어 경로는"
                              " ISP를 우회하므로 끄면 강한 색편향(이 카메라는 초록)이 그대로 남는다."
                              " --no-white-balance로 끌 수 있다.")
    args = parser.parse_args()

    _check_tools_available()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_w, main_h = (int(v) for v in args.main_size.split("x"))
    prev_w, prev_h = (int(v) for v in args.preview_size.split("x"))

    if args.single_shot:
        raw_tmp = out_dir / "_single_shot.raw"
        bgr = capture_frame_bgr(main_w, main_h, raw_tmp, args.bayer_pattern, args.white_balance)
        path = out_dir / "single_shot.png"
        cv2.imwrite(str(path), bgr)
        b_mean, g_mean, r_mean = (float(bgr[:, :, i].mean()) for i in range(3))
        print(f"[단발촬영] {path} shape={bgr.shape} dtype={bgr.dtype} "
              f"mean={bgr.mean():.2f} std={bgr.std():.2f} "
              f"min={int(bgr.min())} max={int(bgr.max())} "
              f"white_balance={args.white_balance} "
              f"B={b_mean:.2f} G={g_mean:.2f} R={r_mean:.2f}")
        return

    session = _CaptureSession(out_dir, (prev_w, prev_h), (main_w, main_h),
                               args.bayer_pattern, args.white_balance)
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
