import json
import time
from vision.utils.blackbox import BlackBoxLogger


def _read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_log_frame_writes_valid_jsonl(tmp_path):
    bb = BlackBoxLogger(str(tmp_path), name="frames", max_queue=100)
    bb.log_frame(frame_id=1, ts=1.0, detections=[{"score": 0.9, "bbox": [1, 2, 3, 4]}], state="ACQUIRE")
    bb.close()

    records = _read_jsonl(tmp_path / "frames.jsonl")
    assert len(records) == 1
    assert records[0]["type"] == "frame"
    assert records[0]["frame_id"] == 1
    assert records[0]["state"] == "ACQUIRE"


def test_log_frame_allows_missing_telemetry(tmp_path):
    bb = BlackBoxLogger(str(tmp_path), name="frames_notelem", max_queue=100)
    bb.log_frame(frame_id=1, ts=1.0, detections=[])
    bb.close()

    records = _read_jsonl(tmp_path / "frames_notelem.jsonl")
    assert records[0]["alt"] is None
    assert records[0]["attitude"] is None
    assert records[0]["command"] is None


def test_log_rejection_records_reason(tmp_path):
    bb = BlackBoxLogger(str(tmp_path), name="rejections", max_queue=100)
    bb.log_rejection(frame_id=5, ts=2.0, reason="shape_mismatch", meta={"score": 0.1})
    bb.close()

    records = _read_jsonl(tmp_path / "rejections.jsonl")
    assert records[0]["type"] == "rejection"
    assert records[0]["reason"] == "shape_mismatch"
    assert records[0]["meta"] == {"score": 0.1}


def test_bounded_queue_drops_oldest_under_burst(tmp_path):
    bb = BlackBoxLogger(str(tmp_path), name="burst", max_queue=5)
    for i in range(50):
        bb.log_frame(frame_id=i, ts=float(i), detections=[])
    bb.close()

    records = _read_jsonl(tmp_path / "burst.jsonl")
    assert len(records) <= 5
    assert records[-1]["frame_id"] == 49  # 최신 프레임은 절대 드랍되지 않는다


def test_close_does_not_hang_or_raise_when_queue_full(tmp_path):
    bb = BlackBoxLogger(str(tmp_path), name="close_safety", max_queue=2)
    for i in range(30):
        bb.log_frame(frame_id=i, ts=float(i), detections=[])
    bb.close(drain_timeout=0.5)  # 예외 없이 끝나야 한다


def test_all_frames_kept_when_under_capacity(tmp_path):
    bb = BlackBoxLogger(str(tmp_path), name="normal_rate", max_queue=100)
    for i in range(10):
        bb.log_frame(frame_id=i, ts=float(i), detections=[])
        time.sleep(0.001)
    bb.close()

    records = _read_jsonl(tmp_path / "normal_rate.jsonl")
    assert [r["frame_id"] for r in records] == list(range(10))
