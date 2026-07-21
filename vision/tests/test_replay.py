"""vision/replay.py — 오프라인 재생 CLI 테스트 (vision_plan.md §7.5/§7.9 항목4).

실제 녹화 폴더(진짜 png 프레임 + telemetry.jsonl)를 실제 파이프라인으로 재생시켜
JSONL 블랙박스가 실제로 생성되고 텔레메트리·latency가 올바르게 들어가는지 검증한다.
"""
import http.client
import json
import sys
import threading
import time

import cv2
import numpy as np
import pytest

import vision.replay as replay_mod
from vision.utils.frame_source import BagFrameSource, DirFrameSource, open_dir_or_bag
from vision.utils.stream import MjpegStreamer


def _make_recording_dir(tmp_path, n=3):
    rec_dir = tmp_path / "recording"
    rec_dir.mkdir()
    for i in range(n):
        img = np.full((40, 40, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(rec_dir / f"frame_{i:04d}.png"), img)
    telemetry = "\n".join(
        json.dumps({"frame_id": i, "ts": float(i), "alt": 10.0 - i, "attitude": {"yaw": 0.0}})
        for i in range(n)
    )
    (rec_dir / "telemetry.jsonl").write_text(telemetry + "\n")
    return rec_dir


def test_run_replay_on_real_dir_writes_real_jsonl_with_telemetry(tmp_path):
    rec_dir = _make_recording_dir(tmp_path, n=3)
    log_dir = tmp_path / "logs"

    from pathlib import Path
    preset_path = str(Path(replay_mod.__file__).parent / "presets" / "single_frame.yaml")

    frame_count = replay_mod.run_replay(
        str(rec_dir), preset_path, display="none", output=None, log_dir=str(log_dir), log_name="rep"
    )
    assert frame_count == 3

    jsonl_path = log_dir / "rep.jsonl"
    assert jsonl_path.exists()
    records = [json.loads(l) for l in jsonl_path.read_text().splitlines()]
    assert [r["frame_id"] for r in records] == [0, 1, 2]
    assert records[1]["alt"] == 9.0
    assert records[1]["attitude"] == {"yaw": 0.0}
    assert all(r["latency"] >= 0 for r in records)

    log_text = (log_dir / "rep.log").read_text()
    assert "provenance" in log_text


def test_run_replay_writes_real_output_video(tmp_path):
    from pathlib import Path

    rec_dir = _make_recording_dir(tmp_path, n=2)
    preset_path = str(Path(replay_mod.__file__).parent / "presets" / "single_frame.yaml")
    output_path = tmp_path / "out.mp4"
    log_dir = tmp_path / "logs"

    frame_count = replay_mod.run_replay(
        str(rec_dir), preset_path, display="none", output=str(output_path),
        log_dir=str(log_dir), log_name="rep2",
    )
    assert frame_count == 2
    assert output_path.exists()

    cap = cv2.VideoCapture(str(output_path))
    read_count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        read_count += 1
    assert read_count == 2


def test_cli_main_replays_bag_file(tmp_path, monkeypatch):
    from pathlib import Path

    video_path = tmp_path / "flight.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (20, 16))
    for i in range(2):
        writer.write(np.full((16, 20, 3), i * 10, dtype=np.uint8))
    writer.release()

    preset_path = str(Path(replay_mod.__file__).parent / "presets" / "single_frame.yaml")
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vision.replay", str(video_path),
            "--preset", preset_path,
            "--log-dir", str(log_dir),
            "--log-name", "clibag",
        ],
    )
    replay_mod.main()

    records = [json.loads(l) for l in (log_dir / "clibag.jsonl").read_text().splitlines()]
    assert [r["frame_id"] for r in records] == [0, 1]


def test_open_dir_or_bag_used_by_replay_dispatches_correctly(tmp_path):
    rec_dir = _make_recording_dir(tmp_path, n=1)
    assert isinstance(open_dir_or_bag(rec_dir), DirFrameSource)

    video_path = tmp_path / "b.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (10, 10))
    writer.write(np.zeros((10, 10, 3), dtype=np.uint8))
    writer.release()
    assert isinstance(open_dir_or_bag(video_path), BagFrameSource)


def test_display_stream_serves_real_frames_via_replay(tmp_path, monkeypatch):
    """§7.9 항목5 replay.py 통합: display="stream" 이 실제 MjpegStreamer로 재생 프레임을 흘리는지
    -> 실제 HTTP GET /stream 으로 접속해 진짜 프레임 하나를 디코드해서 검증한다(pseudo 테스트 금지).

    재생 자체는 결정론(§7.5)을 지키는 run_replay 그대로 쓰되, 테스트에서만 파이프라인 실행
    사이에 약간의 지연을 줘서(Pipeline.run monkeypatch) 실제 HTTP 클라이언트가 재생이 끝나기
    전에 접속할 시간을 확보한다 — 스트리밍 배관 자체는 실제로 동작한다.
    """
    from pathlib import Path

    from vision.core.runner import Pipeline
    from vision.tests.test_stream import _read_one_mjpeg_frame

    rec_dir = _make_recording_dir(tmp_path, n=15)
    log_dir = tmp_path / "logs"
    preset_path = str(Path(replay_mod.__file__).parent / "presets" / "single_frame.yaml")

    real_run = Pipeline.run

    def _slow_run(self, *a, **kw):
        time.sleep(0.03)
        return real_run(self, *a, **kw)

    monkeypatch.setattr(Pipeline, "run", _slow_run)

    started = {}
    real_start = MjpegStreamer.start

    def _capturing_start(self):
        real_start(self)
        started["port"] = self.port

    monkeypatch.setattr(MjpegStreamer, "start", _capturing_start)

    result = {}

    def _run():
        result["frame_count"] = replay_mod.run_replay(
            str(rec_dir), preset_path, display="stream", output=None,
            log_dir=str(log_dir), log_name="streamrep", stream_host="127.0.0.1", stream_port=0,
        )

    worker = threading.Thread(target=_run)
    worker.start()
    try:
        deadline = time.monotonic() + 5.0
        while "port" not in started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "port" in started, "MjpegStreamer가 재생 중 뜨지 않음"

        conn = http.client.HTTPConnection("127.0.0.1", started["port"], timeout=5)
        conn.request("GET", "/stream")
        response = conn.getresponse()
        assert response.status == 200

        jpeg_bytes = _read_one_mjpeg_frame(response)
        decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.size > 0
        conn.close()
    finally:
        worker.join(timeout=10)

    assert result.get("frame_count") == 15
