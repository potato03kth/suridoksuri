"""vision/replay.py — 오프라인 재생 CLI 테스트 (vision_plan.md §7.5/§7.9 항목4).

실제 녹화 폴더(진짜 png 프레임 + telemetry.jsonl)를 실제 파이프라인으로 재생시켜
JSONL 블랙박스가 실제로 생성되고 텔레메트리·latency가 올바르게 들어가는지 검증한다.
"""
import http.client
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import vision.replay as replay_mod
from vision.utils.frame_source import BagFrameSource, DirFrameSource, open_dir_or_bag
from vision.utils.stream import MjpegStreamer

_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_VERTIPORT_FINE_PRESET = str(Path(replay_mod.__file__).parent / "presets" / "vertiport_fine.yaml")


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


def _make_aruco_recording_dir(tmp_path, n=2, marker_id: int = 23):
    """ArUco Phase 4 — ID=23 마커가 있는 합성 프레임으로 이뤄진 녹화 폴더."""
    rec_dir = tmp_path / "aruco_recording"
    rec_dir.mkdir()
    marker_gray = cv2.aruco.generateImageMarker(_DICTIONARY, marker_id, 150)
    marker_bgr = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
    for i in range(n):
        canvas = np.full((300, 300, 3), 255, dtype=np.uint8)
        canvas[50:200, 50:200] = marker_bgr
        cv2.imwrite(str(rec_dir / f"frame_{i:04d}.png"), canvas)
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


def test_streamer_start_failure_still_closes_blackbox(tmp_path, monkeypatch):
    """리소스 leak 회귀(main.py와 동일 패턴): streamer.start()가 OSError를 던져도
    blackbox.close()는 반드시 불려야 한다."""
    from pathlib import Path

    rec_dir = _make_recording_dir(tmp_path, n=2)
    preset_path = str(Path(replay_mod.__file__).parent / "presets" / "single_frame.yaml")
    log_dir = tmp_path / "logs"

    closed = []
    real_close = replay_mod.BlackBoxLogger.close

    def _spy_close(self, *a, **kw):
        closed.append(True)
        return real_close(self, *a, **kw)

    monkeypatch.setattr(replay_mod.BlackBoxLogger, "close", _spy_close)
    monkeypatch.setattr(
        replay_mod.MjpegStreamer, "start", lambda self: (_ for _ in ()).throw(OSError("port in use"))
    )

    with pytest.raises(OSError):
        replay_mod.run_replay(
            str(rec_dir), preset_path, display="stream", output=None,
            log_dir=str(log_dir), log_name="leaktest", stream_host="127.0.0.1", stream_port=0,
        )

    assert closed == [True], "streamer.start() 실패해도 blackbox.close()가 호출돼야 함(leak 방지)"


class _EmptySource:
    """0프레임 재생을 결정론적으로 재현하기 위한 가짜 FrameSource.

    실제 0프레임 mp4는 컨테이너/코덱에 따라 cv2.VideoCapture가 아예 못 여는 경우가 있어
    (isOpened()==False) BagFrameSource 생성 자체가 실패한다 — run_replay 루프에 진입하기도
    전에 에러가 나므로 "저장 로그 게이팅" 검증에 쓸 수 없다. open_dir_or_bag을 이 가짜로
    바꿔치기해 run_replay 루프 자체(실제 blackbox/logger/writer 로직)는 그대로 실행시키고
    입력 소스만 0프레임으로 고정한다.
    """

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def __iter__(self):
        return iter(())


def test_zero_frames_with_output_does_not_log_saved(tmp_path, monkeypatch):
    """거짓 '저장' 로그 회귀: 0프레임 재생이면 cv2.VideoWriter가 끝내 생성되지 않는다
    (§본문 replay.py 프레임 루프 안에서만 writer가 만들어짐) — output이 주어졌다는
    이유만으로 '저장' 로그를 찍으면 안 된다."""
    from pathlib import Path

    monkeypatch.setattr(replay_mod, "open_dir_or_bag", lambda _p: _EmptySource())
    preset_path = str(Path(replay_mod.__file__).parent / "presets" / "single_frame.yaml")
    output_path = tmp_path / "out.mp4"
    log_dir = tmp_path / "logs"

    frame_count = replay_mod.run_replay(
        "unused", preset_path, display="none", output=str(output_path),
        log_dir=str(log_dir), log_name="zerorun",
    )
    assert frame_count == 0
    assert not output_path.exists(), "0프레임이면 VideoWriter가 만들어지지 않아 파일도 없어야 함"

    log_text = (log_dir / "zerorun.log").read_text()
    assert "저장" not in log_text, "writer가 실제로 안 만들어졌는데 '저장' 로그가 찍히면 거짓 로그"


def test_nonzero_frames_with_output_logs_saved(tmp_path):
    """대조군: 실제로 프레임이 처리되고 writer가 만들어지는 정상 케이스에선 '저장' 로그가 찍혀야 함."""
    from pathlib import Path

    rec_dir = _make_recording_dir(tmp_path, n=2)
    preset_path = str(Path(replay_mod.__file__).parent / "presets" / "single_frame.yaml")
    output_path = tmp_path / "out.mp4"
    log_dir = tmp_path / "logs"

    frame_count = replay_mod.run_replay(
        str(rec_dir), preset_path, display="none", output=str(output_path),
        log_dir=str(log_dir), log_name="saverun",
    )
    assert frame_count == 2
    assert output_path.exists()

    log_text = (log_dir / "saverun.log").read_text()
    assert "저장" in log_text


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


# ===========================================================================
# ArUco Phase 4(docs/vision_aruco_branch.md) — TargetEstimate가 실제로 JSONL에 실리는지
# ===========================================================================


def test_aruco_preset_writes_target_estimate_into_jsonl_chosen(tmp_path):
    """§Phase4 핵심 요구: vertiport_fine.yaml로 실제 녹화 폴더(ArUco 마커 포함)를 재생하면
    JSONL의 chosen.target_estimate 안에 calib_id/calib_accuracy/not_for_closed_loop_30cm/
    position/orientation이 실제로 찍히는지 — 몽키패치 없이 실제 파이프라인+블랙박스."""
    rec_dir = _make_aruco_recording_dir(tmp_path, n=2)
    log_dir = tmp_path / "logs"

    frame_count = replay_mod.run_replay(
        str(rec_dir), _VERTIPORT_FINE_PRESET, display="none", output=None,
        log_dir=str(log_dir), log_name="arucorep",
    )
    assert frame_count == 2

    records = [json.loads(l) for l in (log_dir / "arucorep.jsonl").read_text().splitlines()]
    assert len(records) == 2
    for record in records:
        chosen = record["chosen"]
        assert chosen is not None
        estimate = chosen["target_estimate"]
        assert estimate["target_type"] == "aruco_23"
        assert len(estimate["position"]) == 3
        assert len(estimate["orientation"]) == 4
        assert estimate["calib_accuracy"] == "unverified"
        assert estimate["not_for_closed_loop_30cm"] is True
        assert estimate["calib_id"].endswith("nominal.yaml")


def test_no_aruco_marker_frame_has_no_target_estimate_and_does_not_crash(tmp_path):
    """§Phase4 요구: ArUco 마커가 없는 프레임(마커 없는 녹화)은 크래시 없이
    target_estimate가 없어야 한다."""
    rec_dir = _make_recording_dir(tmp_path, n=2)  # 마커 없는 단색 프레임
    log_dir = tmp_path / "logs"

    frame_count = replay_mod.run_replay(
        str(rec_dir), _VERTIPORT_FINE_PRESET, display="none", output=None,
        log_dir=str(log_dir), log_name="noarucorep",
    )
    assert frame_count == 2

    records = [json.loads(l) for l in (log_dir / "noarucorep.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert all(r["chosen"] is None for r in records)


def test_missing_calib_file_logs_warning_and_still_runs(tmp_path):
    """calib 파일이 없어도(calib_path 오지정) 재생 자체는 죽지 않고 target_estimate만 생략."""
    rec_dir = _make_aruco_recording_dir(tmp_path, n=1)
    log_dir = tmp_path / "logs"

    frame_count = replay_mod.run_replay(
        str(rec_dir), _VERTIPORT_FINE_PRESET, display="none", output=None,
        log_dir=str(log_dir), log_name="nocalibrep",
        calib_path=str(tmp_path / "does_not_exist.yaml"),
    )
    assert frame_count == 1

    records = [json.loads(l) for l in (log_dir / "nocalibrep.jsonl").read_text().splitlines()]
    assert records[0]["chosen"] is None

    log_text = (log_dir / "nocalibrep.log").read_text()
    assert "캘리브레이션 로드 실패" in log_text


# ===========================================================================
# §9 6번(공통 상태머신) 배선 — JSONL `state` 필드가 실제로 채워지는지(더 이상 전부 null 아님)
# ===========================================================================


def _make_aruco_recording_dir_with_telemetry(tmp_path, n: int, alts: list[float | None], marker_id: int = 23):
    """`_make_aruco_recording_dir`에 telemetry.jsonl(alt)을 얹은 변형 — replay.py가 AGL을
    실제로 읽어 상태머신에 먹이는지(§9 6번 배선 3) 검증하는 데 쓴다."""
    rec_dir = tmp_path / "aruco_recording_telem"
    rec_dir.mkdir()
    marker_gray = cv2.aruco.generateImageMarker(_DICTIONARY, marker_id, 150)
    marker_bgr = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
    for i in range(n):
        canvas = np.full((300, 300, 3), 255, dtype=np.uint8)
        canvas[50:200, 50:200] = marker_bgr
        cv2.imwrite(str(rec_dir / f"frame_{i:04d}.png"), canvas)
    lines = []
    for i in range(n):
        rec = {"frame_id": i, "ts": float(i) * 0.1}
        if alts[i] is not None:
            rec["alt"] = alts[i]
        lines.append(json.dumps(rec))
    (rec_dir / "telemetry.jsonl").write_text("\n".join(lines) + "\n")
    return rec_dir


def test_state_field_is_populated_and_progresses_through_real_pipeline(tmp_path):
    """§9 6번 요구: log_frame의 `state` 파라미터에 실제 상태머신 결과가 실려야 한다 —
    몽키패치 없이 실제 ArUco 검출이 반복되는 실제 녹화 폴더를 실제로 재생해 JSONL에
    null이 아닌 실제 상태 문자열이(ACQUIRE에 머물지 않고 진행하며) 찍히는지 확인."""
    rec_dir = _make_aruco_recording_dir(tmp_path, n=6)
    log_dir = tmp_path / "logs"

    frame_count = replay_mod.run_replay(
        str(rec_dir), _VERTIPORT_FINE_PRESET, display="none", output=None,
        log_dir=str(log_dir), log_name="staterep",
    )
    assert frame_count == 6

    records = [json.loads(l) for l in (log_dir / "staterep.jsonl").read_text().splitlines()]
    assert len(records) == 6
    states = [r["state"] for r in records]

    assert all(s is not None for s in states), "state 필드가 여전히 전부 null — 배선 안 됨"
    assert states[0] != "ACQUIRE"
    assert "PRECISION_SERVO" in states or "LOCK" in states
    assert all(r["command"] is not None for r in records)


def test_agl_from_telemetry_drives_state_machine_into_terminal(tmp_path):
    """§9 6번 배선 3 요구: telemetry.jsonl의 alt(라이다 AGL)가 있으면 상태머신에 실제로
    쓰여야 한다 — AGL이 낮아지는 실제 녹화를 재생해 TERMINAL까지 실제로 도달하는지 확인
    (없으면 그냥 None으로 흘려보내는 것과 구분이 안 되므로 이 테스트가 진짜 배선을 담보)."""
    alts = [None, None, None, 2.0, 2.0, 2.0, 2.0, 2.0]
    rec_dir = _make_aruco_recording_dir_with_telemetry(tmp_path, n=8, alts=alts)
    log_dir = tmp_path / "logs"

    frame_count = replay_mod.run_replay(
        str(rec_dir), _VERTIPORT_FINE_PRESET, display="none", output=None,
        log_dir=str(log_dir), log_name="aglrep",
    )
    assert frame_count == 8

    records = [json.loads(l) for l in (log_dir / "aglrep.jsonl").read_text().splitlines()]
    assert [r["alt"] for r in records] == alts
    states = [r["state"] for r in records]
    assert "TERMINAL" in states, f"AGL<=terminal_agl_m인데 TERMINAL 미도달 — 실제 상태열: {states}"
