import json
import logging
import logging.handlers
import queue
import time
from pathlib import Path
from typing import Optional


class _DropOldestQueueHandler(logging.handlers.QueueHandler):
    """큐가 가득 차면 가장 오래된 레코드를 버리고 새 걸 넣는다 — 절대 블로킹하지 않는다."""

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(record)


class _JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


class BlackBoxLogger:
    """
    프레임별 기계로그 JSONL 블랙박스 (vision_plan.md §7.4/§7.9).
    비차단(bounded queue + drop-oldest, 별도 스레드) — 제어 루프 핫패스를 막지 않는다.
    사람 로그(setup_dual_sink_logger)와 분리된 별도 채널.
    """

    def __init__(
        self,
        log_dir: str,
        name: str = "blackbox",
        max_queue: int = 1000,
        max_bytes: int = 20_000_000,
        backup_count: int = 10,
    ):
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        self._file_handler = logging.handlers.RotatingFileHandler(
            Path(log_dir) / f"{name}.jsonl", maxBytes=max_bytes, backupCount=backup_count
        )
        self._file_handler.setFormatter(_JsonlFormatter())

        self._logger = logging.getLogger(f"vision.blackbox.{name}")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()
        self._logger.propagate = False

        self._queue: "queue.Queue[logging.LogRecord]" = queue.Queue(maxsize=max_queue)
        self._logger.addHandler(_DropOldestQueueHandler(self._queue))

        self._listener = logging.handlers.QueueListener(self._queue, self._file_handler)
        self._listener.start()

    def log_frame(
        self,
        frame_id: int,
        ts: float,
        detections: list[dict],
        chosen: Optional[dict] = None,
        command: Optional[str] = None,
        state: Optional[str] = None,
        alt: Optional[float] = None,
        attitude: Optional[dict] = None,
        latency: Optional[float] = None,
    ) -> None:
        """§5.1 상태머신 프레임별 기록. FC 연동 전에는 alt/attitude/command가 None일 수 있다."""
        record = {
            "type": "frame",
            "ts": ts,
            "frame_id": frame_id,
            "alt": alt,
            "attitude": attitude,
            "detections": detections,
            "chosen": chosen,
            "command": command,
            "state": state,
            "latency": latency,
        }
        self._logger.info(json.dumps(record, ensure_ascii=False))

    def log_rejection(self, frame_id: int, ts: float, reason: str, meta: Optional[dict] = None) -> None:
        """왜 후보를 버렸나 — false-negative 디버깅의 금맥(§7.4)."""
        record = {
            "type": "rejection",
            "ts": ts,
            "frame_id": frame_id,
            "reason": reason,
            "meta": meta or {},
        }
        self._logger.info(json.dumps(record, ensure_ascii=False))

    def close(self, drain_timeout: float = 2.0) -> None:
        """리스너 스레드가 큐를 비울 시간을 준 뒤 정지한다 —
        QueueListener.stop()의 sentinel put_nowait는 큐가 꽉 차 있으면 queue.Full로 죽는다."""
        deadline = time.monotonic() + drain_timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.005)
        self._listener.stop()
        self._file_handler.close()
