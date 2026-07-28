"""
RPi H.264 라이브 디버그 스트림 (docs/vision_camera_bringup.md Phase 3 정식 도구화).

**목적:** 카메라 원본(초점/노출/프레이밍 등)을 저지연으로 관찰하기 위한 디버그 스트림.
`vision/utils/stream.py`(MjpegStreamer, 검출결과 오버레이 관찰용)와 용도가 다르며 병행
운용한다 — 어느 한쪽을 폐기하지 않는다. **검출 박스는 얹지 않는다. 카메라 원본만 송출한다.**

1A 세션이 최소 스모크로 실증한 조합을 그대로 채택한다(재조사 금지):
    Picamera2.create_video_configuration(main={"size": (W,H), "format": "RGB888"},
                                          controls={"FrameRate": FPS})
    + H264Encoder(bitrate=BITRATE) + FfmpegOutput("-f mpegts -listen 1 tcp://HOST:PORT")
오케스트레이터가 직접 재현 검증(150/150 프레임, steady-state 30.02fps, 라이브니스 0/75,
지연 25~41ms, 랩탑 cv2.VideoCapture(..., cv2.CAP_FFMPEG)로 수신 성공) — 이 조합 자체는 이미
증명됐으므로 여기서 다시 검증하지 않는다.

1A가 찾은 함정 3가지에 대한 대응(재발 금지):
  1. SIGINT는 비대화형 SSH 백그라운드 자식에서 SIG_IGN이 될 수 있어 못 믿는다(`/proc/PID/status`
     SigIgn 비트로 실측 확인됨, nohup/setsid로도 회피 안 됨) → SIGTERM 핸들러를 명시적으로 달아
     graceful shutdown(카메라 release, 인코더 정지)을 보장한다(`_install_sigterm_handler`).
  2. `-listen 1`은 1회성이라 클라이언트가 끊으면 ffmpeg 프로세스가 종료되고 재접속이 불가능
     하다 → 클라이언트 종료(ffmpeg 프로세스 종료)를 감지해 FfmpegOutput을 재기동하는 루프가
     기본 동작이다(`--once`로 단발 모드 선택 가능). 카메라/인코더는 재기동하지 않고 계속
     돌아가며 FfmpegOutput 객체만 교체한다(Encoder.output 세터가 이미 실행 중인 인코더에
     새 output을 등록하는 것을 재사용 — picamera2 소스 확인, encoder.stop()/start() 불필요).
  3. 연결 직후 인코더가 밀어둔 프레임이 한꺼번에 쏟아지는 버스트(최초 ~1~1.5초) →
     `compute_fps_stats()`가 기본으로 앞 N프레임을 건너뛴다(§ 기본값 DEFAULT_WARMUP_FRAMES).

CLI(RPi에서 실행, picam-venv):
    source /home/suri/local-libcamera/env.sh
    "$PICAM_PYTHON" -m vision.tools.h264_stream --af-mode continuous
    "$PICAM_PYTHON" -m vision.tools.h264_stream --af-mode manual --lens-position 5.0
    "$PICAM_PYTHON" -m vision.tools.h264_stream --once   # 1A와 동일한 단발 모드

라이브러리로 사용(순수 함수만, picamera2 불필요 — 랩탑 .venv에서도 그대로 동작):
    from vision.tools.h264_stream import compute_fps_stats, count_identical_frame_pairs
    stats = compute_fps_stats(frame_arrival_timestamps)
    n_dupes = count_identical_frame_pairs(decoded_frames)

⚠️ picamera2/libcamera는 이 노트북 .venv에 없는 RPi 전용 하드웨어 라이브러리다 — 그래서
`run_server()`/`_make_af_controls()` 호출 시점(지연 import)에서만 import한다. 이 모듈을
import하는 것 자체, 그리고 위 CLI 인자 파싱·순수 함수 전부는 picamera2 없이도 항상 성공해야
한다(`vision/utils/frame_source.py::LiveFrameSource`와 동일한 지연 import 격리 패턴 +
회귀테스트, `vision/tests/test_h264_stream.py`).
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from typing import Any, Optional, Sequence, Tuple

import numpy as np

# AF 제어 순수 로직의 **단일 출처는 `vision/utils/frame_source.py`** 다(2026-07-28에 그리로
# 옮겼다 — 라이브 파이프라인 경로에도 AF가 필요해졌는데, import 규칙상 `utils/`가 `tools/`를
# import할 수는 없어 방향을 뒤집었다. 두 벌로 복제하면 LENS_POSITION_MAX 같은 실측 물리값이
# 한쪽만 갱신돼 조용히 갈라진다). 아래 재노출 이름들은 이 모듈의 기존 공개 API 유지용이다.
from vision.utils.frame_source import (  # noqa: E402
    AF_MODES,
    DEFAULT_AF_MODE,
    LENS_POSITION_MAX,
    LENS_POSITION_MIN,
    make_af_controls as _make_af_controls,
    validate_af_args,
    validate_lens_position,
)


# ===========================================================================
# 매직넘버 금지(§7.3) — 전부 이름 붙은 상수
# ===========================================================================

DEFAULT_WIDTH = 1536
DEFAULT_HEIGHT = 864
DEFAULT_FPS = 30
DEFAULT_BITRATE = 5_000_000
DEFAULT_PORT = 8082
DEFAULT_HOST = "0.0.0.0"

# 재접속 감지 폴링 주기 — 너무 짧으면 CPU 낭비, 너무 길면 재접속 반응이 느려진다.
RECONNECT_POLL_INTERVAL_S = 0.2

# 연결 직후 버스트(1A 실측 최초 ~1~1.5초) 제외 기본값 — 30fps 기준 45프레임 ≈ 1.5초.
DEFAULT_WARMUP_FRAMES = 45

# FfmpegOutput.stop()이 무기한 대기(picamera2 라이브러리 자체 동작 — audio=False면
# self.timeout=None 고정이라 ffmpeg가 stdin EOF 이후에도 스스로 안 끝나면 영원히 블록됨,
# 1B 실측으로 재현됨: 재접속 시 SPS/PPS 없는 지점부터 파싱을 시작한 ffmpeg가 다음 키프레임을
# 못 만나 "no frame!"을 무한 반복하며 종료하지 않음) → 안전망으로 이 시간 안에 안 끝나면
# 서브프로세스를 강제 kill한다.
OUTPUT_STOP_TIMEOUT_S = 2.0


# ===========================================================================
# 순수 로직 (하드웨어 비의존 — picamera2/libcamera import 없음, pytest 대상)
# ===========================================================================

def parse_resolution(spec: str) -> Tuple[int, int]:
    """'1536x864' 같은 'WIDTHxHEIGHT' 문자열을 (width, height) 정수쌍으로 파싱.

    구분자는 'x'(대소문자 무관) 하나만 허용. 둘 다 양의 정수여야 한다.
    """
    parts = spec.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"해상도 형식 오류(WIDTHxHEIGHT 필요, 예: 1536x864): {spec!r}")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"해상도 정수 파싱 실패: {spec!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"해상도는 양의 정수여야 함: {spec!r}")
    return width, height


def build_ffmpeg_listen_spec(host: str, port: int) -> str:
    """FfmpegOutput에 넘길 출력 스펙 문자열 조립 — 1A가 실증한 조합 그대로:
    '-f mpegts -listen 1 tcp://HOST:PORT'.
    """
    if not (0 < port <= 65535):
        raise ValueError(f"포트는 1~65535 범위여야 함: {port}")
    return f"-f mpegts -listen 1 tcp://{host}:{port}"


def compute_fps_stats(
    timestamps: Sequence[float], warmup_frames: int = DEFAULT_WARMUP_FRAMES
) -> dict:
    """프레임 도착 timestamps(초, 단조증가)로 steady-state fps 계산.

    연결 직후 버스트(1A 실측 최초 ~1~1.5초)를 배제하기 위해 앞 `warmup_frames`개를 기본으로
    건너뛴다 — 이 프로젝트에서 짧은 측정창으로 "기능 없다"고 오판한 사고가 반복돼(브리핑 확정
    사실) 워밍업 스킵을 항상 켜둔 채로 기본 동작하게 한다.
    """
    n = len(timestamps)
    if n < 2:
        raise ValueError(f"timestamps가 최소 2개 이상이어야 fps 계산 가능(n={n})")
    steady = list(timestamps[warmup_frames:])
    if len(steady) < 2:
        raise ValueError(
            f"워밍업({warmup_frames}) 제외 후 남은 프레임이 2개 미만(n={n}) — 표본 부족. "
            "측정창을 늘리거나 warmup_frames를 줄여라."
        )
    elapsed = steady[-1] - steady[0]
    if elapsed <= 0:
        raise ValueError("timestamps가 단조증가하지 않음(elapsed<=0) — 입력을 확인하라")
    steady_frames = len(steady) - 1  # 구간 수 = 표본수 - 1
    return {
        "n_frames": n,
        "warmup_skipped": n - len(steady),
        "steady_frames": len(steady),
        "elapsed_s": elapsed,
        "fps": steady_frames / elapsed,
    }


def count_identical_frame_pairs(frames: Sequence[np.ndarray]) -> int:
    """인접 프레임쌍 중 완전 동일(np.array_equal)한 쌍의 개수 — 라이브니스 검증.

    0이면 프레임이 실제로 매 캡처마다 갱신되고 있다는 뜻(정지 버퍼로 같은 프레임을 반복
    송출하는 게 아님) — 1A/1B 검증에서 쓰는 지표를 그대로 재사용 가능하게 분리했다.
    """
    count = 0
    for a, b in zip(frames, frames[1:]):
        if np.array_equal(a, b):
            count += 1
    return count


def _is_output_dead(output: Any) -> bool:
    """FfmpegOutput이 죽었는지(클라이언트 종료로 ffmpeg 프로세스가 종료됨) 판정.

    두 신호 중 하나만 있어도 죽은 것으로 본다 — `output_broken`(다음 프레임 쓰기 시도가
    BrokenPipeError로 실패하면 picamera2가 세팅)과 ffmpeg 서브프로세스 자체의 `poll()`
    반환값(None이 아니면 이미 종료됨). 이 함수는 duck-typing만 쓰므로 실제 FfmpegOutput
    없이도(getattr 기본값으로) 테스트 가능하다.
    """
    if getattr(output, "output_broken", False):
        return True
    ffmpeg_proc = getattr(output, "ffmpeg", None)
    if ffmpeg_proc is not None and ffmpeg_proc.poll() is not None:
        return True
    return False


def _stop_output_with_timeout(output: Any, timeout_s: float = OUTPUT_STOP_TIMEOUT_S) -> None:
    """`output.stop()`을 별도 스레드에서 실행하고 `timeout_s` 안에 안 끝나면 ffmpeg 서브프로세스를
    강제 kill한다.

    picamera2의 `FfmpegOutput.stop()`은 `self.ffmpeg.wait(timeout=self.timeout)`을 호출하는데
    `self.timeout`은 `audio=False`(이 도구는 항상 그렇다)면 `None`으로 고정돼 있어 — ffmpeg
    프로세스가 stdin을 닫아도 스스로 끝나지 않으면 무기한 블록된다(1B 실측: 재접속 직후
    SPS/PPS 없는 지점부터 파싱을 시작한 ffmpeg가 다음 자연 키프레임 전까지 "no frame!"을
    반복하는 구간에 딱 걸리면 이 상태로 SIGTERM이 와도 stop_recording()이 영원히 안 끝난다.
    `encoder.force_key_frame()`으로 이 대기 자체를 없애려 했으나 이 RPi5(LibavH264Encoder)의
    picamera2/PyAV 버전 조합에서 크래시하는 별개 버그가 있어 호출하지 않기로 함(위 run_server
    재접속 분기 주석 참고) — 그래서 이 안전망이 유일한 방어선이다). 이 함수는 `output.ffmpeg`/
    `output.stop`만 duck-typing으로 쓰므로 실제 FfmpegOutput 없이도 테스트 가능하다.
    """
    proc = getattr(output, "ffmpeg", None)

    def _run_stop():
        try:
            output.stop()
        except Exception:
            pass

    t = threading.Thread(target=_run_stop, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive() and proc is not None:
        print(
            f"[h264_stream] output.stop()이 {timeout_s}s 내 완료되지 않음 — "
            "ffmpeg 서브프로세스 강제 kill.",
            file=sys.stderr,
            flush=True,
        )
        try:
            proc.kill()
        except Exception:
            pass
        t.join(timeout_s)  # kill로 wait()가 풀려 스레드가 마저 정리될 시간을 조금 더 준다


def _install_sigterm_handler(stop_event: threading.Event) -> None:
    """SIGTERM(+가능하면 SIGINT)을 받으면 stop_event를 세팅하는 핸들러를 설치.

    SIGINT는 비대화형 SSH 백그라운드 자식에서 bash가 SIG_IGN으로 만들어버리는 게 실측
    확인됐다(1A, `/proc/PID/status`의 SigIgn 비트) — nohup/setsid로도 회피 안 됨. 그래서
    graceful shutdown을 보장하는 유일한 신호는 SIGTERM이고, SIGINT는 "있으면 좋은" 보너스
    (직접 터미널에서 Ctrl-C로 띄운 경우)일 뿐 이 계약이 SIGINT에 의존하지 않는다.
    """

    def _handler(signum, frame):  # noqa: ARG001 - signal handler 표준 시그니처
        stop_event.set()

    signal.signal(signal.SIGTERM, _handler)
    try:
        signal.signal(signal.SIGINT, _handler)
    except ValueError:
        # 메인 스레드가 아니면 signal.signal 자체가 실패할 수 있다 — SIGTERM 등록만으로
        # graceful shutdown 계약은 이미 충족되므로 조용히 넘어간다.
        pass


# ===========================================================================
# 하드웨어 연동 (picamera2/libcamera — 함수 내부 지연 import로 격리)
# ===========================================================================

def _build_video_config(picam2: Any, width: int, height: int, fps: int) -> Any:
    """1A가 실증한 조합 그대로: still이 아니라 video config, RGB888(실제 바이트순서는 BGR —
    LiveFrameSource와 동일한 picamera2 명명 역전, calib_capture.py에서 이미 확인된 사실)."""
    return picam2.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
        controls={"FrameRate": fps},
    )


def run_server(
    args: argparse.Namespace,
    *,
    stop_event: Optional[threading.Event] = None,
    install_signal_handler: bool = True,
    poll_interval_s: float = RECONNECT_POLL_INTERVAL_S,
) -> int:
    """RPi에서 실제로 실행되는 서버 루프. picamera2/libcamera는 여기서만 import된다.

    카메라를 한 번만 열고 인코더도 한 번만 시작한다 — 클라이언트가 끊겨 ffmpeg 프로세스가
    죽으면(`-listen 1`의 1회성) FfmpegOutput 객체만 새로 만들어 인코더에 재등록한다
    (`Encoder.output` 세터가 실행 중인 인코더에도 새 output을 등록하는 것을 재사용, picamera2
    소스 확인됨 — encoder.stop()/start()나 카메라 재시작이 불필요해 스트림 재개가 더 매끄럽다).

    `stop_event`/`install_signal_handler`/`poll_interval_s`는 테스트 주입용 — 기본 동작(RPi
    실행)은 내부에서 이벤트를 만들고 SIGTERM 핸들러를 설치한다.
    """
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import FfmpegOutput
    from libcamera import controls as libcamera_controls

    if stop_event is None:
        stop_event = threading.Event()
    if install_signal_handler:
        _install_sigterm_handler(stop_event)

    picam2 = Picamera2()
    config = _build_video_config(picam2, args.width, args.height, args.fps)
    picam2.configure(config)

    initial_controls, trigger_controls = _make_af_controls(
        args.af_mode, args.lens_position, libcamera_controls
    )
    picam2.set_controls(initial_controls)

    encoder = H264Encoder(bitrate=args.bitrate)

    def _make_output() -> Any:
        spec = build_ffmpeg_listen_spec(args.host, args.port)
        out = FfmpegOutput(spec)

        def _err_cb(exc):
            print(f"[h264_stream] FFMPEG ERROR CALLBACK: {exc}", file=sys.stderr, flush=True)

        out.error_callback = _err_cb
        return out

    output = _make_output()
    print(
        f"[h264_stream] {args.width}x{args.height}@{args.fps}fps bitrate={args.bitrate} "
        f"-> tcp listen {args.host}:{args.port} (af_mode={args.af_mode}"
        + (f", lens_position={args.lens_position}" if args.lens_position is not None else "")
        + ")",
        flush=True,
    )
    picam2.start_recording(encoder, output)

    if trigger_controls is not None:
        picam2.set_controls(trigger_controls)

    try:
        while not stop_event.is_set():
            time.sleep(poll_interval_s)
            if _is_output_dead(output):
                print(
                    "[h264_stream] 클라이언트 연결 종료 감지(ffmpeg 프로세스 종료).",
                    flush=True,
                )
                _stop_output_with_timeout(output)
                if args.once:
                    print("[h264_stream] --once 지정 — 재접속하지 않고 종료.", flush=True)
                    break
                output = _make_output()
                encoder.output = output
                output.start()
                # 인코더/카메라를 재시작하지 않고 output만 교체하므로, 새 클라이언트는 인코더가
                # 마침 GOP 중간 어디에 있든 그 지점부터 H264 바이트를 받는다 — SPS/PPS 없는
                # P-슬라이스부터 시작하면 ffmpeg가 다음 키프레임까지 "non-existing PPS 0
                # referenced"/"no frame!"로 잠깐 실패한다(실측, docs/vision_camera_bringup.md
                # Phase 3 절 참고). ⚠️ `encoder.force_key_frame()`으로 즉시 IDR을 강제해 이
                # 대기를 없애려 시도했으나 **이 RPi5(비-VC4 플랫폼)에서 실제로 쓰이는
                # picamera2.encoders.LibavH264Encoder의 force_key_frame()이 설치된 PyAV 버전과
                # 안 맞아 인코더 백그라운드 스레드가 크래시한다**(`frame.pict_type = "I"`가
                # `TypeError: an integer is required`로 실패, picamera2/av 버전 불일치 버그 —
                # 실측 확인, RPi 시스템 변경 범위 밖이라 여기서 고치지 않는다). 그래서
                # force_key_frame()은 호출하지 않는다 — 대신 `H264Encoder`(→ 이 플랫폼에서는
                # LibavH264Encoder) 기본값 `iperiod=30`(30fps 기준 약 1초)이 스스로 주기적으로
                # 키프레임을 넣어주는 것에 기대어 재접속 후 최대 1GOP(~1초) 안에 자연 복구된다
                # (실측 확인 — 아래 docs 절 참고). 급히 강제 키프레임이 필요해지면 picamera2/av
                # 버전을 맞춘 뒤 재도입을 검토할 것.
                print(
                    f"[h264_stream] 재기동 완료 — 재접속 대기 중 (tcp {args.host}:{args.port}), "
                    "다음 자연 키프레임(최대 iperiod 프레임)까지 디코드 에러가 짧게 보일 수 있음.",
                    flush=True,
                )
    finally:
        print("[h264_stream] 종료 처리 중(encoder stop, camera release)...", flush=True)
        # picam2.stop_recording()이 내부에서 output.stop()을 다시 부르는데(Encoder.stop()),
        # 그 output이 살아있으면서도 안 죽은 상태(예: 재접속 실패로 멈춘 ffmpeg)일 수 있어
        # 먼저 bounded timeout으로 정리해둔다 — stop_recording()이 만나는 output은 이미
        # ffmpeg=None이 되어 있어 즉시 반환된다(§ _stop_output_with_timeout docstring).
        _stop_output_with_timeout(output)
        try:
            picam2.stop_recording()
        except Exception as exc:
            print(f"[h264_stream] stop_recording 중 예외(무시하고 계속): {exc}", file=sys.stderr, flush=True)
        try:
            picam2.close()
        except Exception as exc:
            print(f"[h264_stream] picam2.close() 중 예외(무시): {exc}", file=sys.stderr, flush=True)
        print("[h264_stream] 종료 완료.", flush=True)

    return 0


# ===========================================================================
# CLI
# ===========================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--resolution",
        default=f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}",
        help=f"WIDTHxHEIGHT (기본 {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}, 1A 실증값)",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"기본 {DEFAULT_FPS}")
    parser.add_argument(
        "--bitrate", type=int, default=DEFAULT_BITRATE, help=f"H.264 비트레이트, 기본 {DEFAULT_BITRATE}"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"기본 {DEFAULT_PORT}")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"바인딩 주소, 기본 {DEFAULT_HOST}")
    parser.add_argument(
        "--af-mode", choices=list(AF_MODES), default=DEFAULT_AF_MODE, help=f"기본 {DEFAULT_AF_MODE}"
    )
    parser.add_argument(
        "--lens-position",
        type=float,
        default=None,
        help=f"디옵터, --af-mode manual 전용 ({LENS_POSITION_MIN}~{LENS_POSITION_MAX})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="클라이언트 연결 종료 시 재기동하지 않고 종료(기본은 재기동 루프)",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        args.width, args.height = parse_resolution(args.resolution)
        validate_af_args(args.af_mode, args.lens_position)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return run_server(args)


if __name__ == "__main__":
    sys.exit(main())
