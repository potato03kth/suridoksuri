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

from vision.utils.frame_source import (
    BagFrameSource,
    DirFrameSource,
    FrameRecord,
    LiveFrameSource,
    open_dir_or_bag,
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
