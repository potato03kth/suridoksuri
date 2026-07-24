"""vision/tools/h264_stream.py 순수 로직 단위 테스트 (docs/vision_camera_bringup.md Phase 3).

h264_stream.py 자체는 RPi 하드웨어 전용 도구(picamera2/libcamera)라 vision/CLAUDE.md import
규칙상 tools/의 "CI/pytest 대상 아님" 예외가 적용되지만, 그 안의 순수 로직(해상도/ffmpeg 스펙
파싱, 렌즈위치 범위 검증, AF 모드 매핑, fps/라이브니스 통계, output 사망 판정, SIGTERM 핸들러)은
picamera2 없이도 검증 가능하다(jsonl_view.py/calib_analyze.py/rpi_capture.py와 동일한 예외
패턴). picamera2 지연 import 격리 자체는 `vision/utils/frame_source.py`의 LiveFrameSource
회귀테스트와 동일한 방식으로 검증한다.
"""
import os
import signal
import sys
import threading
import time

import numpy as np
import pytest

from vision.tools.h264_stream import (
    AF_MODES,
    DEFAULT_BITRATE,
    DEFAULT_FPS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_WARMUP_FRAMES,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    LENS_POSITION_MAX,
    LENS_POSITION_MIN,
    _install_sigterm_handler,
    _is_output_dead,
    _make_af_controls,
    _stop_output_with_timeout,
    build_arg_parser,
    build_ffmpeg_listen_spec,
    compute_fps_stats,
    count_identical_frame_pairs,
    parse_args,
    parse_resolution,
    validate_af_args,
    validate_lens_position,
)


# ---------- parse_resolution ----------

def test_parse_resolution_valid():
    assert parse_resolution("1536x864") == (1536, 864)


def test_parse_resolution_case_insensitive():
    assert parse_resolution("1280X720") == (1280, 720)


@pytest.mark.parametrize(
    "spec",
    ["1536", "1536x864x1", "abcxdef", "0x864", "1536x0", "-100x864", ""],
)
def test_parse_resolution_rejects_invalid(spec):
    with pytest.raises(ValueError):
        parse_resolution(spec)


# ---------- build_ffmpeg_listen_spec ----------

def test_build_ffmpeg_listen_spec_matches_1a_contract():
    # 1A가 실증한 정확한 조합 — 여기서 문자열이 흔들리면 재현성이 깨진다.
    assert (
        build_ffmpeg_listen_spec("0.0.0.0", 8082)
        == "-f mpegts -listen 1 tcp://0.0.0.0:8082"
    )


@pytest.mark.parametrize("port", [0, -1, 65536, 100000])
def test_build_ffmpeg_listen_spec_rejects_invalid_port(port):
    with pytest.raises(ValueError):
        build_ffmpeg_listen_spec("0.0.0.0", port)


# ---------- validate_lens_position ----------

@pytest.mark.parametrize("value", [0.0, 7.5, 15.0, LENS_POSITION_MIN, LENS_POSITION_MAX])
def test_validate_lens_position_accepts_in_range(value):
    assert validate_lens_position(value) == value


@pytest.mark.parametrize("value", [-0.01, 15.01, 32.0, -5.0, 100.0])
def test_validate_lens_position_rejects_out_of_range(value):
    with pytest.raises(ValueError):
        validate_lens_position(value)


def test_validate_lens_position_error_does_not_endorse_32_as_upper_bound():
    """32.0을 상한으로 오해하는 회귀를 막는다 — 에러 메시지가 실측 하드클램프(15.0)를 명시해야 함."""
    with pytest.raises(ValueError, match="15.0"):
        validate_lens_position(32.0)


# ---------- validate_af_args ----------

def test_validate_af_args_continuous_without_lens_position_ok():
    validate_af_args("continuous", None)


def test_validate_af_args_auto_without_lens_position_ok():
    validate_af_args("auto", None)


def test_validate_af_args_manual_requires_lens_position():
    with pytest.raises(ValueError):
        validate_af_args("manual", None)


def test_validate_af_args_manual_with_valid_lens_position_ok():
    validate_af_args("manual", 5.0)


def test_validate_af_args_manual_with_out_of_range_lens_position_rejected():
    with pytest.raises(ValueError):
        validate_af_args("manual", 32.0)


def test_validate_af_args_rejects_lens_position_outside_manual():
    with pytest.raises(ValueError):
        validate_af_args("continuous", 5.0)
    with pytest.raises(ValueError):
        validate_af_args("auto", 5.0)


def test_validate_af_args_rejects_unknown_mode():
    with pytest.raises(ValueError):
        validate_af_args("nonexistent", None)


def test_af_modes_constant_matches_cli_choices():
    assert set(AF_MODES) == {"continuous", "auto", "manual"}


# ---------- compute_fps_stats ----------

def test_compute_fps_stats_basic_30fps():
    # 정확히 30fps 간격의 100프레임, warmup=45 기본값 적용
    timestamps = [i / 30.0 for i in range(100)]
    stats = compute_fps_stats(timestamps)
    assert stats["n_frames"] == 100
    assert stats["warmup_skipped"] == DEFAULT_WARMUP_FRAMES
    assert stats["steady_frames"] == 100 - DEFAULT_WARMUP_FRAMES
    assert stats["fps"] == pytest.approx(30.0, abs=0.01)


def test_compute_fps_stats_warmup_zero_uses_all_frames():
    timestamps = [i / 30.0 for i in range(10)]
    stats = compute_fps_stats(timestamps, warmup_frames=0)
    assert stats["warmup_skipped"] == 0
    assert stats["steady_frames"] == 10
    assert stats["fps"] == pytest.approx(30.0, abs=0.01)


def test_compute_fps_stats_rejects_too_few_frames():
    with pytest.raises(ValueError):
        compute_fps_stats([1.0])


def test_compute_fps_stats_rejects_insufficient_after_warmup():
    # 5프레임인데 warmup=45 요구 -> steady가 2개 미만
    timestamps = [i / 30.0 for i in range(5)]
    with pytest.raises(ValueError, match="표본 부족"):
        compute_fps_stats(timestamps, warmup_frames=45)


def test_compute_fps_stats_rejects_non_monotonic():
    timestamps = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        compute_fps_stats(timestamps, warmup_frames=0)


# ---------- count_identical_frame_pairs ----------

def test_count_identical_frame_pairs_all_distinct():
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(5)]
    assert count_identical_frame_pairs(frames) == 0


def test_count_identical_frame_pairs_detects_duplicates():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.ones((4, 4, 3), dtype=np.uint8)
    frames = [a, a, b, b, b]  # (a,a) 1쌍 + (b,b),(b,b) 2쌍 = 3쌍
    assert count_identical_frame_pairs(frames) == 3


def test_count_identical_frame_pairs_empty_and_single():
    assert count_identical_frame_pairs([]) == 0
    assert count_identical_frame_pairs([np.zeros((2, 2))]) == 0


# ---------- _make_af_controls (controls 모듈은 가짜 주입 — libcamera import 없음) ----------

class _FakeAfModeEnum:
    Continuous = "Continuous"
    Auto = "Auto"
    Manual = "Manual"


class _FakeAfTriggerEnum:
    Start = "Start"
    Cancel = "Cancel"


class _FakeControlsModule:
    AfModeEnum = _FakeAfModeEnum
    AfTriggerEnum = _FakeAfTriggerEnum


def test_make_af_controls_continuous_no_trigger():
    initial, trigger = _make_af_controls("continuous", None, _FakeControlsModule)
    assert initial == {"AfMode": "Continuous"}
    assert trigger is None


def test_make_af_controls_auto_includes_trigger_start():
    initial, trigger = _make_af_controls("auto", None, _FakeControlsModule)
    assert initial == {"AfMode": "Auto"}
    assert trigger == {"AfTrigger": "Start"}


def test_make_af_controls_manual_sets_lens_position_no_trigger():
    initial, trigger = _make_af_controls("manual", 5.5, _FakeControlsModule)
    assert initial == {"AfMode": "Manual", "LensPosition": 5.5}
    assert trigger is None


def test_make_af_controls_rejects_unknown_mode():
    with pytest.raises(ValueError):
        _make_af_controls("nonexistent", None, _FakeControlsModule)


# ---------- _is_output_dead (duck-typing — 실제 FfmpegOutput 불필요) ----------

class _FakeFfmpegProc:
    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


class _FakeOutput:
    def __init__(self, output_broken=False, ffmpeg=None):
        self.output_broken = output_broken
        self.ffmpeg = ffmpeg


def test_is_output_dead_false_when_healthy():
    out = _FakeOutput(output_broken=False, ffmpeg=_FakeFfmpegProc(returncode=None))
    assert _is_output_dead(out) is False


def test_is_output_dead_true_when_output_broken_flag_set():
    out = _FakeOutput(output_broken=True, ffmpeg=_FakeFfmpegProc(returncode=None))
    assert _is_output_dead(out) is True


def test_is_output_dead_true_when_ffmpeg_process_exited():
    out = _FakeOutput(output_broken=False, ffmpeg=_FakeFfmpegProc(returncode=0))
    assert _is_output_dead(out) is True


def test_is_output_dead_handles_missing_attrs_gracefully():
    class _Bare:
        pass

    assert _is_output_dead(_Bare()) is False


# ---------- _stop_output_with_timeout (duck-typing — 1B 실측 재현: 무기한 대기 안전망) ----------

class _FakeQuickOutput:
    """정상적으로 즉시 끝나는 output — kill 개입이 없어야 한다."""

    def __init__(self):
        self.ffmpeg = _FakeFfmpegProc(returncode=None)
        self.stop_called = False

    def stop(self):
        self.stop_called = True
        self.ffmpeg.returncode = 0  # 정상 종료 시늉


class _FakeHangingOutput:
    """1B 실측 재현 — stop()이 끝나지 않고 블록되는 output(재접속 실패로 멈춘 ffmpeg 시늉).

    kill()이 호출되면 block을 풀어준다(테스트가 실제로 멈추지 않게).
    """

    def __init__(self):
        self.ffmpeg = _FakeFfmpegProc(returncode=None)
        self._release = threading.Event()

    def stop(self):
        self._release.wait(timeout=5.0)  # kill() 콜백이 안 오더라도 테스트 자체는 5s 후 끝남


def test_stop_output_with_timeout_normal_completion_does_not_kill():
    out = _FakeQuickOutput()
    killed = {"v": False}
    out.ffmpeg.kill = lambda: killed.__setitem__("v", True)

    _stop_output_with_timeout(out, timeout_s=1.0)

    assert out.stop_called is True
    assert killed["v"] is False


def test_stop_output_with_timeout_kills_hanging_process():
    out = _FakeHangingOutput()
    killed = {"v": False}

    def _kill():
        killed["v"] = True
        out._release.set()  # kill 이후엔 stop() 스레드가 마저 끝나게 놓아줌

    out.ffmpeg.kill = _kill

    _stop_output_with_timeout(out, timeout_s=0.05)

    assert killed["v"] is True


def test_stop_output_with_timeout_noop_when_ffmpeg_already_none():
    out = _FakeQuickOutput()
    out.ffmpeg = None  # 이미 정리된 output(재호출 시나리오)
    _stop_output_with_timeout(out, timeout_s=1.0)  # 크래시 없이 끝나야 함
    assert out.stop_called is True


# ---------- _install_sigterm_handler (실제 SIGTERM — picamera2 불필요, 전역 핸들러는 복원) ----------

def test_install_sigterm_handler_sets_stop_event_on_sigterm():
    original_term = signal.getsignal(signal.SIGTERM)
    original_int = signal.getsignal(signal.SIGINT)
    stop_event = threading.Event()
    try:
        _install_sigterm_handler(stop_event)
        assert not stop_event.is_set()
        os.kill(os.getpid(), signal.SIGTERM)
        deadline = time.time() + 2.0
        while not stop_event.is_set() and time.time() < deadline:
            time.sleep(0.01)
        assert stop_event.is_set(), "SIGTERM 핸들러가 stop_event를 세팅하지 않음"
    finally:
        signal.signal(signal.SIGTERM, original_term)
        signal.signal(signal.SIGINT, original_int)


# ---------- CLI 인자 파싱 ----------

def test_parse_args_defaults_match_1a_contract():
    args = parse_args([])
    assert (args.width, args.height) == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    assert args.fps == DEFAULT_FPS
    assert args.bitrate == DEFAULT_BITRATE
    assert args.port == DEFAULT_PORT
    assert args.host == DEFAULT_HOST
    assert args.af_mode == "continuous"
    assert args.lens_position is None
    assert args.once is False


def test_parse_args_custom_resolution_and_manual_af():
    args = parse_args(
        ["--resolution", "1280x720", "--af-mode", "manual", "--lens-position", "5.0", "--once"]
    )
    assert (args.width, args.height) == (1280, 720)
    assert args.af_mode == "manual"
    assert args.lens_position == 5.0
    assert args.once is True


def test_parse_args_manual_without_lens_position_exits(capsys):
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--af-mode", "manual"])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "lens-position" in captured.err


def test_parse_args_invalid_resolution_exits(capsys):
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--resolution", "not-a-resolution"])
    assert exc_info.value.code == 2


def test_parse_args_rejects_unknown_af_mode(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--af-mode", "bogus"])


def test_build_arg_parser_has_no_required_positional_args():
    # 원격 SSH 백그라운드 기동 시 인자 없이도 바로 실행 가능해야 한다(기본값만으로 동작).
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.resolution == f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}"


# ---------- 지연 import 격리 회귀테스트 (frame_source.py LiveFrameSource와 동일 패턴) ----------

def test_h264_stream_module_has_no_top_level_picamera2_or_libcamera_import():
    """모듈 최상단(들여쓰기 없는 줄)에 picamera2/libcamera import가 없어야 한다 — 있으면
    picamera2가 없는 이 .venv에서 이 모듈을 import하는 모든 코드(CLI 인자 파싱 포함)까지
    import 시점에 깨진다."""
    import vision.tools.h264_stream as mod

    src = open(mod.__file__, encoding="utf-8").read()
    for line in src.splitlines():
        if line.startswith("import picamera2") or line.startswith("from picamera2"):
            pytest.fail(f"picamera2 import가 모듈 최상단(들여쓰기 없음)에 있음: {line!r}")
        if line.startswith("import libcamera") or line.startswith("from libcamera"):
            pytest.fail(f"libcamera import가 모듈 최상단(들여쓰기 없음)에 있음: {line!r}")


def test_h264_stream_imports_without_picamera2_or_libcamera_installed(monkeypatch):
    """picamera2/libcamera가 sys.modules에서 import 불가능한 상태를 강제해도(=미설치 시뮬레이션)
    vision.tools.h264_stream 모듈 자체의 import와 CLI 인자 파싱은 성공해야 한다(지연 import
    격리 증명)."""
    import importlib

    monkeypatch.setitem(sys.modules, "picamera2", None)
    monkeypatch.setitem(sys.modules, "libcamera", None)
    monkeypatch.delitem(sys.modules, "vision.tools.h264_stream", raising=False)

    reloaded = importlib.import_module("vision.tools.h264_stream")
    args = reloaded.parse_args(["--af-mode", "continuous"])
    assert args.af_mode == "continuous"

    # run_server 호출(지연 import 지점)은 picamera2가 실제로 막혀 있으니 ImportError가 나야 정상.
    with pytest.raises(ImportError):
        reloaded.run_server(args)

    monkeypatch.delitem(sys.modules, "vision.tools.h264_stream", raising=False)
    importlib.import_module("vision.tools.h264_stream")
