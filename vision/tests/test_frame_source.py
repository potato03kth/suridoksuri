"""FrameSource 어댑터(Live/Dir/Bag) 테스트 (vision_plan.md §7.5/§7.9 항목4).

Dir/Bag은 실제 로컬 파일(진짜 png/ mp4 + telemetry.jsonl)로 실제 프레임 디코딩을 검증한다.
Live는 실카메라가 없으므로(이 노트북 .venv엔 picamera2 자체가 설치돼 있지 않다 — RPi 전용
하드웨어 라이브러리) `picamera2.Picamera2`를 가짜 모듈로 `sys.modules`에 주입해 몽키패치하는
방식으로 재시도/에러 계약을 검증한다. `LiveFrameSource.open()`이 `from picamera2 import
Picamera2`를 지연 import하므로 이 가짜 모듈 주입만으로 실제 picamera2 없이도 전체 계약을
검증할 수 있다.
"""
import importlib
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.utils import frame_source as frame_source_module
from vision.utils.frame_source import (
    AF_MODES,
    BagFrameSource,
    DEFAULT_AF_MODE,
    DirFrameSource,
    FrameRecord,
    LENS_POSITION_MAX,
    LENS_POSITION_MIN,
    LiveFrameSource,
    UvcFrameSource,
    make_af_controls,
    open_dir_or_bag,
    parse_fourcc,
    validate_af_args,
    validate_lens_position,
)


# ---------- DirFrameSource ----------

def _write_frames(dir_path, n, color_step=20):
    for i in range(n):
        img = np.full((10, 10, 3), min(255, i * color_step), dtype=np.uint8)
        cv2.imwrite(str(dir_path / f"frame_{i:04d}.png"), img)


def test_dir_frame_source_reads_real_frames_in_order(tmp_path):
    _write_frames(tmp_path, 3)
    source = DirFrameSource(tmp_path)
    records = list(source)
    assert [r.frame_id for r in records] == [0, 1, 2]
    # 실제로 디코딩된 픽셀값이 기록 순서와 일치하는지 (색상단조증가로 순서 검증)
    means = [r.image.mean() for r in records]
    assert means == sorted(means)
    assert records[0].image.shape == (10, 10, 3)


def test_dir_frame_source_len(tmp_path):
    _write_frames(tmp_path, 5)
    assert len(DirFrameSource(tmp_path)) == 5


def test_dir_frame_source_missing_dir_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        DirFrameSource(tmp_path / "does_not_exist")


def test_dir_frame_source_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DirFrameSource(tmp_path)


def test_dir_frame_source_matches_real_telemetry_jsonl(tmp_path):
    _write_frames(tmp_path, 3)
    telemetry_lines = [
        json.dumps({"frame_id": 0, "ts": 100.0, "alt": 12.5}),
        json.dumps({"frame_id": 1, "ts": 100.1, "alt": 12.4}),
        json.dumps({"frame_id": 2, "ts": 100.2, "alt": 12.3}),
    ]
    (tmp_path / "telemetry.jsonl").write_text("\n".join(telemetry_lines) + "\n")

    records = list(DirFrameSource(tmp_path))
    assert [r.ts for r in records] == [100.0, 100.1, 100.2]
    assert records[1].telemetry["alt"] == 12.4


def test_dir_frame_source_without_telemetry_uses_index_placeholder_ts(tmp_path):
    _write_frames(tmp_path, 2)
    records = list(DirFrameSource(tmp_path))
    assert records[0].ts == 0.0
    assert records[1].ts == 1.0
    assert records[0].telemetry == {}


def test_dir_frame_source_deterministic_across_runs(tmp_path):
    _write_frames(tmp_path, 4)
    ids_a = [r.frame_id for r in DirFrameSource(tmp_path)]
    ids_b = [r.frame_id for r in DirFrameSource(tmp_path)]
    assert ids_a == ids_b


# ---------- BagFrameSource ----------

def _write_bag_video(path, n_frames, size=(20, 16)):
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_bag_frame_source_reads_real_video_frames(tmp_path):
    video_path = tmp_path / "flight.mp4"
    _write_bag_video(video_path, 4)

    records = list(BagFrameSource(video_path))
    assert [r.frame_id for r in records] == [0, 1, 2, 3]
    assert records[0].image.shape[:2] == (16, 20)


def test_bag_frame_source_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        BagFrameSource(tmp_path / "nope.mp4")


def test_bag_frame_source_matches_real_sidecar_telemetry(tmp_path):
    video_path = tmp_path / "flight.mp4"
    _write_bag_video(video_path, 3)
    sidecar = tmp_path / "flight.jsonl"
    sidecar.write_text(
        "\n".join(
            json.dumps({"frame_id": i, "ts": 50.0 + i, "alt": 5.0 - i * 0.1}) for i in range(3)
        )
        + "\n"
    )

    records = list(BagFrameSource(video_path))
    assert [r.ts for r in records] == [50.0, 51.0, 52.0]
    assert records[2].telemetry["alt"] == pytest.approx(4.8)


def test_bag_frame_source_without_sidecar_uses_fps_derived_ts(tmp_path):
    video_path = tmp_path / "flight.mp4"
    _write_bag_video(video_path, 2)
    records = list(BagFrameSource(video_path))
    assert records[0].ts == pytest.approx(0.0)
    assert records[1].ts > records[0].ts


# ---------- open_dir_or_bag factory ----------

def test_open_dir_or_bag_picks_dir_for_directory(tmp_path):
    _write_frames(tmp_path, 2)
    assert isinstance(open_dir_or_bag(tmp_path), DirFrameSource)


def test_open_dir_or_bag_picks_bag_for_file(tmp_path):
    video_path = tmp_path / "flight.mp4"
    _write_bag_video(video_path, 2)
    assert isinstance(open_dir_or_bag(video_path), BagFrameSource)


def test_open_dir_or_bag_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_dir_or_bag(tmp_path / "nowhere")


# ---------- LiveFrameSource — 인터페이스 계약 (실카메라 없음, picamera2를 가짜 모듈로 몽키패치) ----------

class _AlwaysFailsPicam2:
    """Picamera2() 생성 자체가 실패 — "카메라 없음"류 시나리오."""

    def __init__(self, *_a, **_k):
        raise RuntimeError("no cameras available")


class _StartFailsPicam2:
    """생성/설정은 되지만 start()에서 실패 — "열리긴 했지만 스트림을 못 켬"류 시나리오."""

    def __init__(self, *_a, **_k):
        self.closed = False

    def create_still_configuration(self, main=None, **_k):
        return {"main": main}

    def configure(self, _cfg):
        pass

    def start(self):
        raise RuntimeError("failed to start camera")

    def close(self):
        self.closed = True


class _OneFramePicam2:
    """정상 연결 + 프레임 1장까지는 성공, 그 다음 capture_array가 실패(연결 끊김 시뮬레이션)."""

    def __init__(self, *_a, **_k):
        self._served = False
        self.started = False
        self.stopped = False
        self.closed = False
        self.configured_with = None

    def create_still_configuration(self, main=None, **_k):
        return {"main": main}

    def configure(self, cfg):
        self.configured_with = cfg

    def start(self):
        self.started = True

    def capture_array(self, name="main"):
        assert name == "main"
        if self._served:
            raise RuntimeError("camera disconnected")
        self._served = True
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def _inject_fake_picamera2(monkeypatch, picam_cls):
    """`from picamera2 import Picamera2`가 실제 picamera2 없이 성공하도록 가짜 모듈을 주입."""
    fake_module = types.ModuleType("picamera2")
    fake_module.Picamera2 = picam_cls
    monkeypatch.setitem(sys.modules, "picamera2", fake_module)


def test_live_frame_source_retries_then_raises_connection_error(monkeypatch):
    attempts = []

    def _make(*_a, **_k):
        attempts.append(1)
        return _AlwaysFailsPicam2()

    _inject_fake_picamera2(monkeypatch, _make)
    source = LiveFrameSource(camera_num=0, retries=3, retry_delay=0)
    with pytest.raises(ConnectionError, match="카메라 연결 실패"):
        source.open()
    assert len(attempts) == 3


def test_live_frame_source_start_failure_retries_then_raises_connection_error(monkeypatch):
    _inject_fake_picamera2(monkeypatch, _StartFailsPicam2)
    source = LiveFrameSource(camera_num=0, retries=2, retry_delay=0)
    with pytest.raises(ConnectionError, match="카메라 연결 실패"):
        source.open()


def test_live_frame_source_read_failure_raises_connection_error(monkeypatch):
    _inject_fake_picamera2(monkeypatch, _OneFramePicam2)
    source = LiveFrameSource(camera_num=0, retries=1, retry_delay=0)
    it = iter(source)
    next(it)  # 첫 프레임은 성공
    with pytest.raises(ConnectionError, match="프레임 읽기 실패"):
        next(it)


def test_live_frame_source_yields_frame_record_when_open_succeeds(monkeypatch):
    _inject_fake_picamera2(monkeypatch, _OneFramePicam2)
    source = LiveFrameSource(camera_num=0, retries=1, retry_delay=0)
    record = next(iter(source))
    assert isinstance(record, FrameRecord)
    assert record.frame_id == 0
    assert record.image.shape == (4, 4, 3)


def test_live_frame_source_requests_rgb888_format_and_configured_resolution(monkeypatch):
    """picamera2 명명 역전(RGB888 요청 → 실제 BGR 바이트순서, calib_capture.py에서 실기체로
    확인된 사실)을 그대로 재사용하므로, 별도 cv2.cvtColor 없이 create_still_configuration에
    "RGB888" 포맷을 요청하는지와 생성자에 준 해상도가 그대로 전달되는지 확인한다."""
    _inject_fake_picamera2(monkeypatch, _OneFramePicam2)
    source = LiveFrameSource(camera_num=0, resolution=(64, 48), retries=1, retry_delay=0)
    source.open()
    assert source._picam.configured_with == {"main": {"format": "RGB888", "size": (64, 48)}}


def test_live_frame_source_close_stops_and_closes_picam(monkeypatch):
    _inject_fake_picamera2(monkeypatch, _OneFramePicam2)
    with LiveFrameSource(camera_num=0, retries=1, retry_delay=0) as source:
        picam = source._picam
        assert picam.started
    assert picam.stopped
    assert picam.closed


def test_live_frame_source_rejects_invalid_retries():
    with pytest.raises(ValueError):
        LiveFrameSource(camera_num=0, retries=0)


# ---------- LiveFrameSource — 지연 import 격리 회귀테스트 ----------

def test_frame_source_module_has_no_top_level_picamera2_import():
    """모듈 최상단(들여쓰기 없는 줄)에 picamera2 import가 없어야 한다 — 있으면 picamera2가 없는
    이 .venv에서 DirFrameSource/BagFrameSource를 쓰는 모든 코드까지 import 시점에 깨진다."""
    import vision.utils.frame_source as mod

    source_text = Path(mod.__file__).read_text(encoding="utf-8")
    for line in source_text.splitlines():
        if line.startswith("import picamera2") or line.startswith("from picamera2"):
            pytest.fail(f"picamera2 import가 모듈 최상단(들여쓰기 없음)에 있음: {line!r}")


def test_vision_utils_frame_source_imports_without_picamera2_installed(monkeypatch):
    """picamera2가 sys.modules에서 import 불가능한 상태를 강제해도(=미설치 시뮬레이션)
    vision.utils.frame_source 모듈 자체의 import는 성공해야 한다(지연 import 격리 증명)."""
    monkeypatch.setitem(sys.modules, "picamera2", None)  # None 캐시 → import picamera2가 ImportError
    monkeypatch.delitem(sys.modules, "vision.utils.frame_source", raising=False)

    reloaded = importlib.import_module("vision.utils.frame_source")
    assert hasattr(reloaded, "LiveFrameSource")

    # picamera2가 실제로 막혀 있으니 open()을 부르면(지연 import 지점) ImportError가 나야 정상 —
    # 즉 모듈 import 자체와 open() 호출은 별개 시점이라는 것까지 확인한다.
    source = reloaded.LiveFrameSource(camera_num=0, retries=1, retry_delay=0)
    with pytest.raises(ImportError):
        source.open()


# ---------- LiveFrameSource — AF(오토포커스) 제어 (2026-07-28) ----------
#
# 🔴 **실기체 미검증.** RPi 접속이 금지된 세션에서 작성됐다 — 아래는 전부 가짜 picamera2/
# libcamera 주입 기반 단위테스트이고, "실제로 초점이 이동했는지(선명도 지표)"는 확인하지
# 못했다. 실기체 검증 절차는 vision/CLAUDE.md "LiveFrameSource AF 제어" 절 참조.


class _AfRecordingPicam2(_OneFramePicam2):
    """`set_controls` 호출을 순서대로 기록하는 가짜 — AF 컨트롤이 실제로 걸리는지 검증용."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.control_calls = []

    def set_controls(self, controls):
        self.control_calls.append(controls)


class _AfRejectingPicam2(_AfRecordingPicam2):
    """드라이버가 AF 컨트롤을 거부하는 상황 — 카메라 자체는 살아 있어야 한다."""

    def set_controls(self, controls):
        raise RuntimeError("control not supported")


class _FakeAfModeEnum:
    Continuous = "ENUM_CONTINUOUS"
    Auto = "ENUM_AUTO"
    Manual = "ENUM_MANUAL"


class _FakeAfTriggerEnum:
    Start = "ENUM_TRIGGER_START"


def _inject_fake_libcamera(monkeypatch):
    """`from libcamera import controls`가 실제 libcamera 없이 성공하도록 가짜 모듈 주입."""
    controls_mod = types.ModuleType("libcamera.controls")
    controls_mod.AfModeEnum = _FakeAfModeEnum
    controls_mod.AfTriggerEnum = _FakeAfTriggerEnum
    libcamera_mod = types.ModuleType("libcamera")
    libcamera_mod.controls = controls_mod
    monkeypatch.setitem(sys.modules, "libcamera", libcamera_mod)
    monkeypatch.setitem(sys.modules, "libcamera.controls", controls_mod)


# --- 순수 로직(하드웨어 무관) ---


def test_af_pure_logic_make_controls_per_mode():
    controls = types.SimpleNamespace(AfModeEnum=_FakeAfModeEnum, AfTriggerEnum=_FakeAfTriggerEnum)

    initial, trigger = make_af_controls("continuous", None, controls)
    assert initial == {"AfMode": "ENUM_CONTINUOUS"} and trigger is None

    # auto 는 트리거 없이는 렌즈가 마지막 위치에 머문다 — 2차 set_controls 가 반드시 있어야 한다
    initial, trigger = make_af_controls("auto", None, controls)
    assert initial == {"AfMode": "ENUM_AUTO"}
    assert trigger == {"AfTrigger": "ENUM_TRIGGER_START"}

    initial, trigger = make_af_controls("manual", 3.5, controls)
    assert initial == {"AfMode": "ENUM_MANUAL", "LensPosition": 3.5}
    assert trigger is None


def test_af_lens_position_range_is_the_measured_hard_clamp():
    """드라이버가 광고하는 32.0이 아니라 실측 하드클램프 15.0이 상한이다(실기체 확정 사실)."""
    assert (LENS_POSITION_MIN, LENS_POSITION_MAX) == (0.0, 15.0)
    assert validate_lens_position(0.0) == 0.0
    assert validate_lens_position(15.0) == 15.0
    with pytest.raises(ValueError):
        validate_lens_position(15.1)
    with pytest.raises(ValueError):
        validate_lens_position(32.0)   # 광고 상한을 그대로 쓰면 안 된다
    with pytest.raises(ValueError):
        validate_lens_position(-0.1)


def test_af_arg_combinations_validated():
    validate_af_args("continuous", None)
    validate_af_args("manual", 5.0)
    with pytest.raises(ValueError):
        validate_af_args("manual", None)          # manual 은 lens_position 필수
    with pytest.raises(ValueError):
        validate_af_args("continuous", 5.0)       # 조용히 무시되는 인자 금지
    with pytest.raises(ValueError):
        validate_af_args("bogus", None)


def test_live_frame_source_validates_af_args_in_constructor():
    """하드웨어를 만지기 **전에** 실패해야 한다 — open()까지 가서야 터지면 현장에서 늦다."""
    with pytest.raises(ValueError):
        LiveFrameSource(af_mode="manual")                      # lens_position 없음
    with pytest.raises(ValueError):
        LiveFrameSource(af_mode="continuous", lens_position=3.0)
    with pytest.raises(ValueError):
        LiveFrameSource(af_mode="bogus")
    with pytest.raises(ValueError):
        LiveFrameSource(af_mode=None, lens_position=3.0)       # AF 미개입인데 렌즈위치?


# --- open() 배선 ---


def test_live_frame_source_defaults_to_continuous_af(monkeypatch):
    """기본값이 연속 AF — 기체가 10~40m를 오르내리므로. (이 변경 전에는 AF를 아예 안 건드렸다.)"""
    _inject_fake_picamera2(monkeypatch, _AfRecordingPicam2)
    _inject_fake_libcamera(monkeypatch)
    source = LiveFrameSource(camera_num=0, retries=1, retry_delay=0)
    assert source.af_mode == DEFAULT_AF_MODE == "continuous"
    source.open()
    assert source._picam.control_calls == [{"AfMode": "ENUM_CONTINUOUS"}]
    assert source.af_applied is True
    assert source.af_error is None


def test_live_frame_source_auto_mode_also_sends_trigger(monkeypatch):
    _inject_fake_picamera2(monkeypatch, _AfRecordingPicam2)
    _inject_fake_libcamera(monkeypatch)
    source = LiveFrameSource(camera_num=0, retries=1, retry_delay=0, af_mode="auto")
    source.open()
    assert source._picam.control_calls == [
        {"AfMode": "ENUM_AUTO"},
        {"AfTrigger": "ENUM_TRIGGER_START"},
    ]


def test_live_frame_source_manual_mode_sets_lens_position(monkeypatch):
    _inject_fake_picamera2(monkeypatch, _AfRecordingPicam2)
    _inject_fake_libcamera(monkeypatch)
    source = LiveFrameSource(
        camera_num=0, retries=1, retry_delay=0, af_mode="manual", lens_position=4.25
    )
    source.open()
    assert source._picam.control_calls == [{"AfMode": "ENUM_MANUAL", "LensPosition": 4.25}]


def test_live_frame_source_af_none_leaves_driver_defaults(monkeypatch):
    """escape hatch — `af_mode=None`이면 AF에 손을 대지 않는다(2026-07-28 이전 동작)."""
    _inject_fake_picamera2(monkeypatch, _AfRecordingPicam2)
    _inject_fake_libcamera(monkeypatch)
    source = LiveFrameSource(camera_num=0, retries=1, retry_delay=0, af_mode=None)
    source.open()
    assert source._picam.control_calls == []
    assert source.af_applied is False
    assert source.af_error is None


def test_live_frame_source_af_failure_does_not_kill_capture(monkeypatch):
    """AF를 못 걸어도 카메라는 계속 돌아야 한다 — 대신 사유를 af_error에 남긴다(침묵 금지)."""
    _inject_fake_picamera2(monkeypatch, _AfRejectingPicam2)
    _inject_fake_libcamera(monkeypatch)
    source = LiveFrameSource(camera_num=0, retries=1, retry_delay=0)
    source.open()
    assert source.af_applied is False
    assert source.af_error is not None and "control not supported" in source.af_error
    # 그럼에도 프레임은 정상적으로 나온다
    record = next(iter(source))
    assert record.image.shape == (4, 4, 3)


def test_live_frame_source_af_survives_missing_libcamera(monkeypatch):
    """libcamera 자체가 없어도(비-RPi 환경) 크래시 없이 사유만 남기고 진행."""
    _inject_fake_picamera2(monkeypatch, _AfRecordingPicam2)
    monkeypatch.setitem(sys.modules, "libcamera", None)  # import libcamera -> ImportError
    source = LiveFrameSource(camera_num=0, retries=1, retry_delay=0)
    source.open()
    assert source.af_applied is False
    assert source.af_error is not None
    assert source._picam.control_calls == []


def test_frame_source_module_has_no_top_level_libcamera_import():
    """picamera2와 같은 이유 — libcamera도 RPi 전용이라 최상단 import면 이 .venv가 깨진다."""
    import vision.utils.frame_source as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    for line in src.splitlines():
        if line.startswith("import libcamera") or line.startswith("from libcamera"):
            pytest.fail(f"libcamera import가 모듈 최상단(들여쓰기 없음)에 있음: {line!r}")


def test_h264_stream_reuses_the_same_af_implementation():
    """AF 제어 단일 출처 계약 — `tools/h264_stream.py`가 자기 사본을 갖지 않고 여기서 가져다 쓴다.
    복제되면 LENS_POSITION_MAX 같은 실측 물리값이 한쪽만 갱신돼 조용히 갈라진다."""
    import vision.tools.h264_stream as h264

    assert h264._make_af_controls is make_af_controls
    assert h264.validate_af_args is validate_af_args
    assert h264.validate_lens_position is validate_lens_position
    assert h264.AF_MODES is AF_MODES
    assert h264.LENS_POSITION_MAX == LENS_POSITION_MAX


# ---------- UvcFrameSource — USB 웹캠 (실웹캠 없음, cv2.VideoCapture를 가짜로 몽키패치) ----------
#
# 2026-07-30 CSI 카메라 물리적 사망 대응 경로. `LiveFrameSource`와 달리 picamera2/libcamera에
# 의존하지 않으므로 sys.modules 주입이 아니라 `frame_source.cv2.VideoCapture` 속성을 직접
# 몽키패치한다(monkeypatch가 테스트 종료 시 원복).

class _FakeCapture:
    """가짜 cv2.VideoCapture. set/get 호출을 순서까지 기록해 설정 순서 계약을 검증할 수 있다.

    actual: 실제로 적용되는 해상도 — 요청과 다르게 줘서 "싸구려 웹캠이 조용히 다른 해상도를
        돌려주는" 실제 동작을 재현한다.
    """

    def __init__(self, *, opened=True, actual=(1920, 1080), fps=30.0, frames=1,
                 reject_props=()):
        self.opened = opened
        self._actual = actual
        self._fps = fps
        self._frames_left = frames
        self._reject_props = set(reject_props)
        self.set_calls = []          # [(prop, value), ...] 순서 보존
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return prop not in self._reject_props

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._actual[0])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._actual[1])
        if prop == cv2.CAP_PROP_FPS:
            return float(self._fps)
        return 0.0

    def read(self):
        if self._frames_left <= 0:
            return False, None
        self._frames_left -= 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def _patch_videocapture(monkeypatch, cap_or_factory):
    """`frame_source` 모듈이 보는 cv2.VideoCapture를 교체하고, 생성 인자를 기록해 돌려준다."""
    created = []

    def _factory(device, backend=None):
        created.append((device, backend))
        cap = cap_or_factory(device, backend) if callable(cap_or_factory) else cap_or_factory
        return cap

    monkeypatch.setattr(frame_source_module.cv2, "VideoCapture", _factory)
    return created


# --- 🔴 핵심 안전장치: 해상도 불일치 ---

def test_uvc_rejects_silent_resolution_fallback(monkeypatch):
    """요청 1920x1080인데 웹캠이 640x480을 돌려주면 하드 실패해야 한다.

    이대로 통과시키면 nominal.yaml의 camera_matrix가 해상도와 어긋나 solvePnP 거리·pose가
    소리 없이 3배 틀린다 — 이 클래스가 존재하는 가장 큰 이유다.
    """
    cap = _FakeCapture(actual=(640, 480))
    _patch_videocapture(monkeypatch, cap)
    src = UvcFrameSource(resolution=(1920, 1080))
    with pytest.raises(ValueError) as e:
        src.open()
    msg = str(e.value)
    assert "1920x1080" in msg and "640x480" in msg
    assert cap.released, "실패 시 카메라를 release 해야 한다"


def test_uvc_resolution_mismatch_message_gives_the_fix(monkeypatch):
    """에러가 '뭐가 틀렸다'만 말하고 끝나면 현장에서 못 고친다 — 조치 명령까지 담아야 한다."""
    _patch_videocapture(monkeypatch, _FakeCapture(actual=(640, 480)))
    with pytest.raises(ValueError) as e:
        UvcFrameSource(resolution=(1920, 1080)).open()
    msg = str(e.value)
    assert "--list-formats-ext" in msg, "지원 조합 확인 명령이 있어야 한다"
    assert "compute_nominal_intrinsics" in msg, "캘리브레이션 재생성 명령이 있어야 한다"
    assert "--uvc-allow-resolution-mismatch" in msg, "진행 방법이 있어야 한다"


def test_uvc_resolution_mismatch_is_not_retried(monkeypatch):
    """몇 번을 다시 열어도 지원 안 하는 해상도는 계속 지원 안 한다 — 재시도는 낭비다."""
    created = _patch_videocapture(monkeypatch, lambda *_a: _FakeCapture(actual=(640, 480)))
    with pytest.raises(ValueError):
        UvcFrameSource(resolution=(1920, 1080), retries=5, retry_delay=0).open()
    assert len(created) == 1, f"재시도 없이 1회만 열어야 함 (실제 {len(created)}회)"


def test_uvc_allow_resolution_mismatch_proceeds_and_records_actual(monkeypatch):
    """명시적으로 허용하면 진행하되, 실제 해상도를 관측 가능하게 남긴다."""
    _patch_videocapture(monkeypatch, _FakeCapture(actual=(640, 480)))
    src = UvcFrameSource(resolution=(1920, 1080), allow_resolution_mismatch=True)
    src.open()
    assert src.actual_resolution == (640, 480)
    src.close()


def test_uvc_records_actual_resolution_and_fps_on_success(monkeypatch):
    _patch_videocapture(monkeypatch, _FakeCapture(actual=(1280, 720), fps=15.0))
    with UvcFrameSource(resolution=(1280, 720)) as src:
        assert src.actual_resolution == (1280, 720)
        assert src.actual_fps == 15.0


# --- 설정 순서/내용 ---

def test_uvc_sets_fourcc_before_resolution(monkeypatch):
    """V4L2는 FOURCC를 해상도보다 먼저 걸어야 한다 — 순서가 뒤집히면 해상도가 되돌려진다."""
    cap = _FakeCapture()
    _patch_videocapture(monkeypatch, cap)
    UvcFrameSource(resolution=(1920, 1080)).open()
    props = [p for p, _ in cap.set_calls]
    assert props.index(cv2.CAP_PROP_FOURCC) < props.index(cv2.CAP_PROP_FRAME_WIDTH)
    assert props.index(cv2.CAP_PROP_FOURCC) < props.index(cv2.CAP_PROP_FRAME_HEIGHT)


def test_uvc_defaults_to_mjpg_for_usb2_bandwidth(monkeypatch):
    """무압축 YUYV로 FHD를 요청하면 카메라가 조용히 5fps급으로 떨어진다 — 기본이 MJPG여야 한다."""
    cap = _FakeCapture()
    _patch_videocapture(monkeypatch, cap)
    UvcFrameSource().open()
    fourcc_values = [v for p, v in cap.set_calls if p == cv2.CAP_PROP_FOURCC]
    assert fourcc_values == [parse_fourcc("MJPG")]


def test_uvc_requests_buffersize_one_to_avoid_stale_frames(monkeypatch):
    """파이프라인이 카메라보다 느리면 V4L2 큐에 프레임이 쌓여 낡은 그림을 본다."""
    cap = _FakeCapture()
    _patch_videocapture(monkeypatch, cap)
    UvcFrameSource().open()
    assert (cv2.CAP_PROP_BUFFERSIZE, 1) in cap.set_calls


def test_uvc_buffersize_rejection_does_not_break_open(monkeypatch):
    """BUFFERSIZE를 지원 안 하는 백엔드에서도 열려야 한다(best-effort)."""
    _patch_videocapture(monkeypatch, _FakeCapture(reject_props=(cv2.CAP_PROP_BUFFERSIZE,)))
    with UvcFrameSource() as src:
        assert src.actual_resolution == (1920, 1080)


def test_uvc_uses_v4l2_backend_by_default(monkeypatch):
    created = _patch_videocapture(monkeypatch, _FakeCapture())
    UvcFrameSource().open()
    assert created[0][1] == cv2.CAP_V4L2


def test_uvc_accepts_stable_device_path(monkeypatch):
    """/dev/video* 인덱스는 USB 재연결·부팅 순서에 따라 흔들린다(이 저장소의 /dev/mediaN 전례).
    by-id 안정 경로를 그대로 받을 수 있어야 한다."""
    path = "/dev/v4l/by-id/usb-Generic_Webcam-video-index0"
    created = _patch_videocapture(monkeypatch, _FakeCapture())
    UvcFrameSource(device=path).open()
    assert created[0][0] == path


# --- 연결/에러 계약 (LiveFrameSource와 동일) ---

def test_uvc_retries_then_raises_connection_error(monkeypatch):
    created = _patch_videocapture(monkeypatch, lambda *_a: _FakeCapture(opened=False))
    with pytest.raises(ConnectionError):
        UvcFrameSource(retries=3, retry_delay=0).open()
    assert len(created) == 3


def test_uvc_read_failure_raises_connection_error(monkeypatch):
    _patch_videocapture(monkeypatch, _FakeCapture(frames=1))
    src = UvcFrameSource()
    it = iter(src)
    assert isinstance(next(it), FrameRecord)
    with pytest.raises(ConnectionError):
        next(it)


def test_uvc_yields_frame_records_with_increasing_ids(monkeypatch):
    _patch_videocapture(monkeypatch, _FakeCapture(frames=3))
    records = []
    with UvcFrameSource() as src:
        for rec in src:
            records.append(rec)
            if len(records) == 3:
                break
    assert [r.frame_id for r in records] == [0, 1, 2]
    assert all(r.image.shape == (4, 4, 3) for r in records)
    assert all(r.telemetry == {} for r in records)


def test_uvc_close_releases_capture(monkeypatch):
    cap = _FakeCapture()
    _patch_videocapture(monkeypatch, cap)
    src = UvcFrameSource()
    src.open()
    src.close()
    assert cap.released


def test_uvc_context_manager_releases_on_exception(monkeypatch):
    cap = _FakeCapture(frames=0)
    _patch_videocapture(monkeypatch, cap)
    with pytest.raises(ConnectionError):
        with UvcFrameSource() as src:
            next(iter(src))
    assert cap.released


# --- 인자 검증 ---

def test_uvc_rejects_invalid_retries():
    with pytest.raises(ValueError):
        UvcFrameSource(retries=0)


def test_uvc_rejects_bad_fourcc_length():
    with pytest.raises(ValueError):
        UvcFrameSource(fourcc="MJP")
    with pytest.raises(ValueError):
        parse_fourcc("MJPEG")


def test_uvc_focus_requires_autofocus_off():
    """오토포커스가 켜진 채 수동 초점값을 주면 드라이버가 곧바로 덮어쓴다 — 조용히 무시되는
    인자를 남기지 않는다(LiveFrameSource의 lens_position 규칙과 같은 철학)."""
    with pytest.raises(ValueError):
        UvcFrameSource(focus=0.0)
    with pytest.raises(ValueError):
        UvcFrameSource(autofocus=True, focus=0.0)
    UvcFrameSource(autofocus=False, focus=0.0)  # 이 조합만 허용


# --- 초점 컨트롤 best-effort ---

def test_uvc_does_not_touch_focus_by_default(monkeypatch):
    cap = _FakeCapture()
    _patch_videocapture(monkeypatch, cap)
    src = UvcFrameSource()
    src.open()
    props = [p for p, _ in cap.set_calls]
    assert cv2.CAP_PROP_AUTOFOCUS not in props
    assert src.control_error is None


def test_uvc_fixed_focus_webcam_rejection_is_not_fatal(monkeypatch):
    """싸구려 웹캠은 대개 고정초점이라 이 컨트롤이 아예 없다 — 정상 상황으로 취급해야 한다."""
    _patch_videocapture(monkeypatch, _FakeCapture(reject_props=(cv2.CAP_PROP_AUTOFOCUS,)))
    src = UvcFrameSource(autofocus=False)
    src.open()  # 예외 없이 열려야 한다
    assert src.control_error is not None
    assert "고정초점" in src.control_error


def test_uvc_applies_autofocus_off_and_focus_value(monkeypatch):
    cap = _FakeCapture()
    _patch_videocapture(monkeypatch, cap)
    src = UvcFrameSource(autofocus=False, focus=0.0)
    src.open()
    assert (cv2.CAP_PROP_AUTOFOCUS, 0) in cap.set_calls
    assert (cv2.CAP_PROP_FOCUS, 0.0) in cap.set_calls
    assert src.control_error is None
