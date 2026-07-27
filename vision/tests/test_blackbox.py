import json
import logging
import queue
import time

from vision.utils.blackbox import BlackBoxLogger, _DropOldestQueueHandler


def _read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def _log_record(msg: str) -> logging.LogRecord:
    return logging.LogRecord("t", logging.INFO, __file__, 1, msg, None, None)


def _drain(q: "queue.Queue") -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


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


# ── bounded queue 드롭 의미론 ────────────────────────────────────────────────
#
# ⚠️ **이 3건은 원래 한 개의 플래키 테스트였다.** 옛 테스트는
# `BlackBoxLogger(max_queue=5)`에 50건을 넣고 **파일에 남은 레코드가 `<= 5`**라고 단언했는데,
# `BlackBoxLogger.__init__`이 `QueueListener`를 이미 **start()** 해 둔 상태라 넣는 동안
# 리스너 스레드가 **동시에 큐를 비워 파일로 흘려보낸다.** 파일 건수는 생산 루프와 리스너
# 스레드의 스케줄링에 달린 값이다 — 실측(같은 커밋, blackbox.py 무변경):
#
#   | 생산 루프 조건        | 파일 건수 | 마지막 frame_id | `len(records) <= 5` |
#   | 양보 없음             | 5         | 49              | 통과 |
#   | 틱당 0.2ms 양보       | 50        | 49              | **실패** |
#
# 즉 **drop-oldest는 정상**이고 틀린 것은 단언이었다: drop-oldest가 보장하는 것은 "**큐에**
# 최대 N건"이지 "**싱크에** 최대 N건"이 아니다. 그래서 아래처럼 쪼갰다 —
#   ① 큐 상한 + drop-oldest 의미론은 **소비 스레드 없이** 결정론적으로 검증(부하 무관),
#   ② 그 상한이 `BlackBoxLogger`에 실제로 배선됐는지는 구조로 검증,
#   ③ end-to-end에서는 소비 스레드가 돌아도 **항상 참인 성질**만 단언.
# (`tests/test_target_sink.py`가 `_DropOldestQueue`에 쓴 패턴과 같은 수준이다.)


def test_drop_oldest_queue_handler_bounds_queue_and_keeps_newest():
    """① 결정론: 리스너(소비 스레드) 없이 핸들러만 두고 50건을 밀어 넣는다.

    두 가지를 **매 스텝** 못박는다:
      - 큐 크기가 정확히 `min(넣은 개수, 상한)` — 상한을 넘지도 않고, 가득 찬 뒤에는 상한을
        **유지**한다(drop-oldest는 한 건 버리고 한 건 넣으므로 크기가 줄지 않는다).
        ⚠️ 최종 상태만 보면 안 된다 — "가득 차면 큐를 통째로 비우고 넣는" 구현도 50/5에서는
        우연히 같은 최종 상태(45~49, size 5)에 도달한다(파괴검증 D-B3에서 실측).
      - 남는 것은 **가장 최근 5건** — 최신이 버려지면(drop-newest) 여기서 걸린다.
    """
    maxsize = 5
    q: "queue.Queue" = queue.Queue(maxsize=maxsize)
    handler = _DropOldestQueueHandler(q)
    for i in range(50):
        handler.emit(_log_record(str(i)))       # 실제 logging 경로(emit→prepare→enqueue)
        assert q.qsize() == min(i + 1, maxsize), (
            f"{i}번째 put 후 큐 크기가 {q.qsize()} — 상한 유지(가득 차면 한 건만 버림)가 깨졌다"
        )

    assert [r.getMessage() for r in _drain(q)] == [str(i) for i in range(45, 50)], (
        "가장 오래된 것이 아니라 최신 것이 버려졌다(drop-newest)"
    )


def test_blackbox_wires_a_bounded_drop_oldest_queue(tmp_path):
    """② 위 의미론이 실제로 `BlackBoxLogger`에 배선됐는지 — 큐가 `max_queue`로 유계이고
    핸들러가 drop-oldest 구현인지. (①은 핸들러 클래스만 보므로 이 배선이 풀려도 못 잡는다.)"""
    bb = BlackBoxLogger(str(tmp_path), name="wiring", max_queue=7)
    try:
        assert bb._queue.maxsize == 7, "큐가 max_queue로 유계가 아니다(무한 큐면 메모리 폭주)"
        assert isinstance(bb._logger.handlers[0], _DropOldestQueueHandler)
    finally:
        bb.close()


def test_burst_never_loses_the_newest_frame_at_the_sink(tmp_path):
    """③ end-to-end에서 **부하와 무관하게 항상 참인 것**만 단언한다: 최신 프레임은 절대
    드랍되지 않고, 남은 것들은 순서·유일성이 보존된 원본의 부분열이다. 파일 건수 자체는
    리스너 스케줄링에 달린 값이라 단언하지 않는다(그게 옛 플래키의 원인이었다)."""
    bb = BlackBoxLogger(str(tmp_path), name="burst", max_queue=5)
    for i in range(50):
        bb.log_frame(frame_id=i, ts=float(i), detections=[])
    bb.close()

    records = _read_jsonl(tmp_path / "burst.jsonl")
    frame_ids = [r["frame_id"] for r in records]
    assert frame_ids, "버스트에서 한 건도 안 남았다"
    assert frame_ids[-1] == 49, "최신 프레임은 절대 드랍되지 않는다"
    assert frame_ids == sorted(frame_ids), "기록 순서가 뒤바뀌었다"
    assert len(set(frame_ids)) == len(frame_ids), "같은 프레임이 두 번 기록됐다"
    assert set(frame_ids) <= set(range(50)), "넣지 않은 프레임이 기록됐다"


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
