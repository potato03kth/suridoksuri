"""main.py CLI 헤드리스 안전성 회귀 테스트 + 관측성(로거/블랙박스) 연결 테스트.

불변식: 기본값 --display none 은 어떤 GUI 함수(cv2.imshow 등)도 호출하지 않는다.
드론(디스플레이 없음)에서의 크래시를 방지하는 계약이므로 절대 깨지면 안 된다.

로그 출력은 실제 리포지토리 vision/results/를 더럽히지 않도록 모든 테스트에서
--log-dir 을 tmp_path 하위로 명시한다.
"""
import json
import sys

import cv2
import numpy as np
import pytest

import vision.main as main_mod


def _write_image(tmp_path) -> str:
    img = np.full((120, 120, 3), 180, dtype=np.uint8)
    p = tmp_path / "frame.png"
    cv2.imwrite(str(p), img)
    return str(p)


def _log_dir_args(tmp_path) -> list[str]:
    return ["--log-dir", str(tmp_path / "logs")]


def test_display_none_never_calls_imshow(tmp_path, monkeypatch):
    img_path = _write_image(tmp_path)
    calls = []
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(sys, "argv", ["vision.main", img_path, *_log_dir_args(tmp_path)])  # 기본 --display none
    main_mod.main()
    assert calls == [], "--display none 에서 imshow가 호출되면 헤드리스 크래시 위험"


def test_display_window_calls_imshow(tmp_path, monkeypatch):
    img_path = _write_image(tmp_path)
    calls = []
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(cv2, "waitKey", lambda *a, **k: ord("q"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(
        sys, "argv", ["vision.main", img_path, "--display", "window", *_log_dir_args(tmp_path)]
    )
    main_mod.main()
    assert len(calls) == 1


def test_display_file_requires_output(tmp_path, monkeypatch):
    img_path = _write_image(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vision.main", img_path, "--display", "file"])
    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 2


def test_display_stream_starts_real_server_and_pushes_frame(tmp_path, monkeypatch):
    """§7.9 항목5: --display stream이 실제 MjpegStreamer를 기동하고(실 소켓 bind+스레드),
    처리된 프레임을 실제로 push하는지 — 딥한 HTTP/MJPEG 바이트 검증은 test_stream.py가 맡는다."""
    img_path = _write_image(tmp_path)
    pushed = []
    real_push_frame = main_mod.MjpegStreamer.push_frame

    def _spy_push_frame(self, frame):
        pushed.append(frame)
        return real_push_frame(self, frame)

    monkeypatch.setattr(main_mod.MjpegStreamer, "push_frame", _spy_push_frame)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vision.main", img_path, "--display", "stream", "--stream-port", "0",
            *_log_dir_args(tmp_path),
        ],
    )
    main_mod.main()

    assert len(pushed) == 1
    assert pushed[0] is not None and pushed[0].size > 0


def test_streamer_start_failure_still_closes_blackbox(tmp_path, monkeypatch):
    """리소스 leak 회귀: streamer.start()가 (포트 충돌 등으로) OSError를 던져도
    blackbox.close()는 반드시 불려야 한다 — 안 그러면 큐스레드/파일핸들이 leak된다."""
    img_path = _write_image(tmp_path)
    closed = []
    real_close = main_mod.BlackBoxLogger.close

    def _spy_close(self, *a, **kw):
        closed.append(True)
        return real_close(self, *a, **kw)

    monkeypatch.setattr(main_mod.BlackBoxLogger, "close", _spy_close)
    monkeypatch.setattr(
        main_mod.MjpegStreamer, "start", lambda self: (_ for _ in ()).throw(OSError("port in use"))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["vision.main", img_path, "--display", "stream", "--stream-port", "0", *_log_dir_args(tmp_path)],
    )

    with pytest.raises(OSError):
        main_mod.main()

    assert closed == [True], "streamer.start() 실패해도 blackbox.close()가 호출돼야 함(leak 방지)"


def test_display_none_never_starts_streamer(tmp_path, monkeypatch):
    """opt-in 불변식: --display를 stream으로 켜지 않으면 스트리머가 아예 안 뜬다(오버헤드 없음)."""
    img_path = _write_image(tmp_path)
    started = []
    real_start = main_mod.MjpegStreamer.start
    monkeypatch.setattr(
        main_mod.MjpegStreamer, "start", lambda self: (started.append(True), real_start(self))
    )
    monkeypatch.setattr(sys, "argv", ["vision.main", img_path, *_log_dir_args(tmp_path)])
    main_mod.main()
    assert started == []


def test_image_run_writes_real_jsonl_blackbox_and_human_log(tmp_path, monkeypatch):
    """§7.9 항목4: blackbox/logger가 main.py 파이프라인에 실제로 연결됐는지 —
    모킹이 아니라 실제 이미지를 실제로 처리시켜 디스크에 실제 JSONL/.log가 남는지 검증."""
    img_path = _write_image(tmp_path)
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(
        sys, "argv", ["vision.main", img_path, "--log-dir", str(log_dir), "--log-name", "imgrun"]
    )
    main_mod.main()

    jsonl_path = log_dir / "imgrun.jsonl"
    log_path = log_dir / "imgrun.log"
    assert jsonl_path.exists(), "블랙박스 JSONL이 실제로 생성되지 않음"
    assert log_path.exists(), "사람로그 .log가 실제로 생성되지 않음"

    records = [json.loads(l) for l in jsonl_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["type"] == "frame"
    assert records[0]["frame_id"] == 0
    assert isinstance(records[0]["detections"], list)
    assert records[0]["latency"] >= 0

    log_text = log_path.read_text()
    assert "provenance" in log_text
    header_line = next(l for l in log_text.splitlines() if "provenance" in l)
    header = json.loads(header_line.split("provenance ", 1)[1])
    assert header["config"]["input"] == img_path
    assert "git_commit" in header


def test_video_run_writes_one_jsonl_record_per_frame(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    frame = np.full((60, 80, 3), 150, dtype=np.uint8)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (80, 60)
    )
    for _ in range(3):
        writer.write(frame)
    writer.release()

    log_dir = tmp_path / "logs"
    monkeypatch.setattr(
        sys,
        "argv",
        ["vision.main", str(video_path), "--log-dir", str(log_dir), "--log-name", "vidrun"],
    )
    main_mod.main()

    records = [json.loads(l) for l in (log_dir / "vidrun.jsonl").read_text().splitlines()]
    assert [r["frame_id"] for r in records] == [0, 1, 2]
    assert all(r["type"] == "frame" for r in records)
