"""FrameSource 어댑터(Live/Dir/Bag) 테스트 (vision_plan.md §7.5/§7.9 항목4).

Dir/Bag은 실제 로컬 파일(진짜 png/ mp4 + telemetry.jsonl)로 실제 프레임 디코딩을 검증한다.
Live는 실카메라가 없으므로 cv2.VideoCapture만 몽키패치해 재시도/에러 계약을 검증한다
(이 세션은 RPi 실카메라 작업 금지 — docs/vision_status.md).
"""
import json

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


# ---------- LiveFrameSource — 인터페이스 계약 (실카메라 없음, cv2.VideoCapture만 몽키패치) ----------

class _AlwaysClosedCapture:
    def __init__(self, *_a, **_k):
        pass

    def isOpened(self):
        return False

    def release(self):
        pass


class _OpensButNeverReadsCapture:
    def __init__(self, *_a, **_k):
        pass

    def isOpened(self):
        return True

    def read(self):
        return False, None

    def release(self):
        pass


class _OneFrameCapture:
    def __init__(self, *_a, **_k):
        self._served = False

    def isOpened(self):
        return True

    def read(self):
        if self._served:
            return False, None
        self._served = True
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        pass


def test_live_frame_source_retries_then_raises_connection_error(monkeypatch):
    attempts = []
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda *a, **k: (attempts.append(1), _AlwaysClosedCapture())[1]
    )
    source = LiveFrameSource(device=0, retries=3, retry_delay=0)
    with pytest.raises(ConnectionError, match="카메라 연결 실패"):
        source.open()
    assert len(attempts) == 3


def test_live_frame_source_read_failure_raises_connection_error(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: _OpensButNeverReadsCapture())
    source = LiveFrameSource(device=0, retries=1, retry_delay=0)
    with pytest.raises(ConnectionError, match="프레임 읽기 실패"):
        next(iter(source))


def test_live_frame_source_yields_frame_record_when_open_succeeds(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: _OneFrameCapture())
    source = LiveFrameSource(device=0, retries=1, retry_delay=0)
    record = next(iter(source))
    assert isinstance(record, FrameRecord)
    assert record.frame_id == 0
    assert record.image.shape == (4, 4, 3)


def test_live_frame_source_rejects_invalid_retries():
    with pytest.raises(ValueError):
        LiveFrameSource(device=0, retries=0)
