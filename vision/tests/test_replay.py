"""vision/replay.py — 오프라인 재생 CLI 테스트 (vision_plan.md §7.5/§7.9 항목4).

실제 녹화 폴더(진짜 png 프레임 + telemetry.jsonl)를 실제 파이프라인으로 재생시켜
JSONL 블랙박스가 실제로 생성되고 텔레메트리·latency가 올바르게 들어가는지 검증한다.
"""
import http.client
import json
import math
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import vision.replay as replay_mod
from vision.core.wire import (
    RECORD_TYPE_STATE_HINT,
    RECORD_TYPE_TARGET,
    REQUIRED_STATE_HINT_KEYS,
    REQUIRED_TARGET_KEYS,
)
from vision.utils.frame_source import BagFrameSource, DirFrameSource, open_dir_or_bag
from vision.utils.stream import MjpegStreamer
from vision.utils.target_sink import SocketTargetSink

_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_VERTIPORT_FINE_PRESET = str(Path(replay_mod.__file__).parent / "presets" / "vertiport_fine.yaml")
_DISTRESS_FINE_PRESET = str(Path(replay_mod.__file__).parent / "presets" / "distress_fine.yaml")


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


# ===========================================================================
# ② 조난자 fine(§5.3, white_box_detector) 배선 — ArUco와 별개 경로로 fine_locked/체인이
# 실제로 이어지는지 (§9 "끊어진 체인을 잇는 작업")
# ===========================================================================


def _distress_fine_frame(mat_size: int = 300, canvas: int = 460, box_ratio: float = 0.0667) -> np.ndarray:
    """distress_fine.yaml 캐스케이드 검증용 최소 합성 프레임 — `vision/tests/golden/
    generate_synthetic.py`의 `_synthetic_distress`와 동일 패턴(실측 스펙 비율, vision/CLAUDE.md
    참조). 테스트 파일 간 상호 의존을 피하려 이 파일 안에 얇게 중복(test_main.py와 동일 중복,
    프로젝트 "각자 얇게 중복" 관례)."""
    img = np.full((canvas, canvas, 3), (60, 60, 60), dtype=np.uint8)
    hsv_green = np.array([[[60, 200, 180]]], dtype=np.uint8)
    bgr_green = tuple(int(v) for v in cv2.cvtColor(hsv_green, cv2.COLOR_HSV2BGR)[0, 0])
    c = canvas // 2
    half = mat_size // 2
    cv2.rectangle(img, (c - half, c - half), (c + half, c + half), bgr_green, -1)
    box_half = int(mat_size * box_ratio / 2)
    if box_half > 0:
        cv2.rectangle(img, (c - box_half, c - box_half), (c + box_half, c + box_half), (255, 255, 255), -1)
    return img


def _make_distress_fine_recording_dir(tmp_path, n: int) -> Path:
    rec_dir = tmp_path / "distress_fine_recording"
    rec_dir.mkdir()
    frame = _distress_fine_frame()
    for i in range(n):
        cv2.imwrite(str(rec_dir / f"frame_{i:04d}.png"), frame)
    return rec_dir


def _make_distress_fine_recording_dir_with_telemetry(tmp_path, n: int, alts: list) -> Path:
    rec_dir = _make_distress_fine_recording_dir(tmp_path, n)
    lines = []
    for i in range(n):
        rec = {"frame_id": i, "ts": float(i)}
        if alts[i] is not None:
            rec["alt"] = alts[i]
        lines.append(json.dumps(rec))
    (rec_dir / "telemetry.jsonl").write_text("\n".join(lines) + "\n")
    return rec_dir


def test_distress_fine_state_progresses_through_real_pipeline(tmp_path):
    """ArUco가 아니라 흰 박스 확정(white_box_detector) 경로로도 상태머신이 CENTER_DESCEND를
    넘어 진행하는지 — 몽키패치 없이 실제 재생으로 확인."""
    rec_dir = _make_distress_fine_recording_dir(tmp_path, n=6)
    log_dir = tmp_path / "logs"

    frame_count = replay_mod.run_replay(
        str(rec_dir), _DISTRESS_FINE_PRESET, display="none", output=None,
        log_dir=str(log_dir), log_name="distressrep",
    )
    assert frame_count == 6

    records = [json.loads(l) for l in (log_dir / "distressrep.jsonl").read_text().splitlines()]
    assert len(records) == 6
    states = [r["state"] for r in records]

    assert all(s is not None for s in states), "state 필드가 여전히 전부 null — 배선 안 됨"
    assert states[0] != "ACQUIRE"
    assert "PRECISION_SERVO" in states or "LOCK" in states


def test_distress_fine_agl_from_telemetry_drives_state_machine_into_terminal(tmp_path):
    """§9 6번 배선(AGL) + 이번 세션(② fine 체인 연결)의 결합 — telemetry.jsonl의 alt가
    white_box_detector 경로에서도 상태머신을 실제로 TERMINAL까지 진행시키는지 확인."""
    alts = [None, None, None, 2.0, 2.0, 2.0, 2.0, 2.0]
    rec_dir = _make_distress_fine_recording_dir_with_telemetry(tmp_path, n=8, alts=alts)
    log_dir = tmp_path / "logs"

    frame_count = replay_mod.run_replay(
        str(rec_dir), _DISTRESS_FINE_PRESET, display="none", output=None,
        log_dir=str(log_dir), log_name="distressaglrep",
    )
    assert frame_count == 8

    records = [json.loads(l) for l in (log_dir / "distressaglrep.jsonl").read_text().splitlines()]
    assert [r["alt"] for r in records] == alts
    states = [r["state"] for r in records]
    assert "TERMINAL" in states, f"AGL<=terminal_agl_m인데 TERMINAL 미도달 — 실제 상태열: {states}"


def test_distress_fine_preset_confirmed_detection_carries_landing_point_meta():
    """§5.3 설계 포인트 배선 증거 — `distress_fine.yaml`을 실제 `Pipeline.from_config`로 로드해
    돌렸을 때(레지스트리/preset 경로 전부 경유, 클래스 직접 호출 아님) 확정 detection에
    `landing_point_px`가 실제로 실리는지. **JSONL 자체(`_detections_to_list`)는 bbox/confidence만
    담고 meta를 싣지 않아 이 확인 불가**(로그 스키마 변경 금지 원칙) — 그래서 파이프라인
    출력을 직접 본다(replay.py가 내부적으로 쓰는 것과 동일한 실제 `Pipeline.run()` 호출,
    `test_golden_regression.py`의 이차 테스트와 동일 원칙)."""
    from vision.core.runner import Pipeline

    pipeline = Pipeline.from_config(_DISTRESS_FINE_PRESET)
    state = pipeline.run(_distress_fine_frame())

    assert len(state.detections) == 1
    wb_meta = state.detections[0].meta["white_box_detector"]
    assert "landing_point_px" in wb_meta
    x, y, w, h = state.detections[0].bbox
    lx, ly = wb_meta["landing_point_px"]
    assert x <= lx <= x + w
    assert y <= ly <= y + h


# ===========================================================================
# ② 조난자 구역(초록 매트) 상대 pose 배선 — 재생 경로
#
# §7.5가 "같은 파이프라인으로 오프라인 재생"을 회귀검증의 최대 레버로 못박고 있어 main.py와
# 동일한 pose 산출이 replay.py에도 붙어 있어야 한다(둘이 갈라지면 골든셋 재생으로 초록구역
# pose 회귀를 잡을 수 없다).
# ===========================================================================

_DISTRESS_COARSE_PRESET = str(Path(replay_mod.__file__).parent / "presets" / "distress_coarse.yaml")
_GOLDEN_DISTRESS = Path(replay_mod.__file__).parent / "tests" / "golden" / "distress"


def _golden_distress_recording(tmp_path, tier: str, n: int = 3):
    """골든 PNG를 그대로 복사해 만든 녹화 폴더(재인코딩 없음)."""
    rec_dir = tmp_path / f"rec_{tier}"
    rec_dir.mkdir()
    src = (_GOLDEN_DISTRESS / tier / "frame_000.png").read_bytes()
    for i in range(n):
        (rec_dir / f"frame_{i:03d}.png").write_bytes(src)
    return rec_dir


def _golden_calib(canvas: int) -> str:
    return str(_GOLDEN_DISTRESS / "synthetic_calib" / f"canvas{canvas}.yaml")


def test_replay_distress_coarse_writes_mat_pose_into_jsonl_chosen(tmp_path):
    """재생 경로에서도 초록구역 상대 pose가 실제로 `chosen.target_estimate`에 실려야 한다 —
    배선 전에는 ArUco 프리셋에서만 실렸다."""
    rec_dir = _golden_distress_recording(tmp_path, "10m", n=3)
    log_dir = tmp_path / "logs"

    replay_mod.run_replay(
        str(rec_dir), _DISTRESS_COARSE_PRESET, display="none", output=None,
        log_dir=str(log_dir), log_name="matpose", calib_path=_golden_calib(460),
    )

    records = [json.loads(l) for l in (log_dir / "matpose.jsonl").read_text().splitlines()
               if json.loads(l).get("type") == "frame"]
    assert records
    est = records[0]["chosen"]["target_estimate"]
    assert est["target_type"] == "distress_mat_center"
    assert est["position"][2] == pytest.approx(10.0, rel=0.05), (
        f"10m 골든 프레임인데 산출 거리가 {est['position'][2]:.3f}m"
    )
    assert est["calib_accuracy"] == "unverified"
    assert est["not_for_closed_loop_30cm"] is True
    assert est["meta"]["plane_reference"] == "mat_top_surface"


def test_replay_distress_fine_pose_is_the_landing_point_not_the_mat_centre(tmp_path):
    """§5.3: fine 재생에서 나오는 position은 매트 중심이 아니라 "박스 옆 빈 초록면"이다."""
    log_dir = tmp_path / "logs"
    replay_mod.run_replay(
        str(_golden_distress_recording(tmp_path, "fine", n=2)), _DISTRESS_FINE_PRESET,
        display="none", output=None, log_dir=str(log_dir), log_name="fpose",
        calib_path=_golden_calib(460),
    )
    fine = [json.loads(l) for l in (log_dir / "fpose.jsonl").read_text().splitlines()
            if json.loads(l).get("type") == "frame"][0]["chosen"]["target_estimate"]

    log_dir2 = tmp_path / "logs2"
    replay_mod.run_replay(
        str(_golden_distress_recording(tmp_path, "10m", n=2)), _DISTRESS_COARSE_PRESET,
        display="none", output=None, log_dir=str(log_dir2), log_name="cpose",
        calib_path=_golden_calib(460),
    )
    centre = [json.loads(l) for l in (log_dir2 / "cpose.jsonl").read_text().splitlines()
              if json.loads(l).get("type") == "frame"][0]["chosen"]["target_estimate"]

    assert fine["target_type"] == "distress_landing_point"
    lateral = math.hypot(fine["position"][0] - centre["position"][0],
                         fine["position"][1] - centre["position"][1])
    assert lateral == pytest.approx(1.05 * math.sqrt(2), rel=0.05), (
        f"착륙점이 매트 중심에서 밀려나지 않았다(수평차 {lateral:.3f}m)"
    )


def test_replay_aruco_chosen_shape_is_unchanged_by_the_distress_wiring(tmp_path):
    """ArUco 회귀: 재생 경로의 `chosen.target_estimate` 키 집합이 달라지면 안 된다."""
    rec_dir = _make_aruco_recording_dir(tmp_path, n=2)
    log_dir = tmp_path / "logs"
    replay_mod.run_replay(
        str(rec_dir), _VERTIPORT_FINE_PRESET, display="none", output=None,
        log_dir=str(log_dir), log_name="arucorep2",
    )
    rec = [json.loads(l) for l in (log_dir / "arucorep2.jsonl").read_text().splitlines()
           if json.loads(l).get("type") == "frame"][0]
    est = rec["chosen"]["target_estimate"]
    assert set(est) == {
        "position", "orientation", "confidence", "target_type",
        "calib_accuracy", "not_for_closed_loop_30cm", "calib_id",
    }, f"ArUco chosen 형태가 달라졌다: {sorted(est)}"
    assert est["target_type"] == "aruco_23"


def test_replay_landing_sm_config_derives_half_hfov_tan_from_calibration():
    """`replay.py`에도 같은 배선이 얇게 중복돼 있다 — 한쪽만 고쳐지는 것을 막는 회귀테스트."""
    import logging

    from vision.core.state_machine import NOMINAL_HALF_HFOV_TAN
    from vision.replay import _landing_sm_config

    class _FakeCalib:
        camera_matrix = [[500.0, 0.0, 500.0], [0.0, 500.0, 250.0], [0.0, 0.0, 1.0]]
        image_size = (1000, 500)

    logger = logging.getLogger("test_replay_landing_sm_config")
    assert _landing_sm_config(_FakeCalib(), logger).half_hfov_tan == pytest.approx(1.0)
    assert _landing_sm_config(None, logger).half_hfov_tan == pytest.approx(NOMINAL_HALF_HFOV_TAN)


# ===========================================================================
# `--target-sink` 배선 (2026-07-28) — Phase 1 후속 세션이 명시적으로 남긴 갭
#
# 🔴 **왜 재생 경로에 반드시 있어야 하는가:** `replay.py`는 Dir/Bag + `telemetry.jsonl`(AGL)
# 재생이라 **`agl_m`이 실린 `state_hint`를 회귀로 잡을 수 있는 유일한 경로**다 — `main.py`는
# AGL을 받는 경로가 아예 없어서(`_build_observation(agl_m=None)`) 상태머신이 구조적으로
# `TERMINAL`까지 가지 못한다. 즉 "착륙 최종단계의 유도 힌트가 소비자에게 실제로 흘러가는가"는
# 여기서만 검증된다.
#
# 정책은 main.py와 **문자 그대로 동일**하다: 같은 CLI 인자 이름 / 기본 꺼짐 / bind 실패는
# 강등 없는 하드 페일(exit 3).
# ===========================================================================


class _RecordingSocketTargetSink(SocketTargetSink):
    """발행된 레코드를 그대로 모아두는 진짜 sink(진짜 서버가 뜬다, port=0 임시포트).
    test_main.py와 같은 수법 — 테스트 파일 간 상호 import를 피해 얇게 중복한다."""

    instances: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: list[dict] = []
        _RecordingSocketTargetSink.instances.append(self)

    def publish(self, record: dict) -> None:
        self.records.append(record)
        super().publish(record)


@pytest.fixture
def recording_sink_cls(monkeypatch):
    _RecordingSocketTargetSink.instances = []
    monkeypatch.setattr(replay_mod, "SocketTargetSink", _RecordingSocketTargetSink)
    return _RecordingSocketTargetSink


def test_replay_target_sink_is_off_by_default_and_never_binds_a_socket(tmp_path, monkeypatch):
    """opt-in 불변식(main.py와 동일): 켜지 않으면 소켓도 스레드도 안 뜨고 발행 시도조차 없다."""
    started = []
    monkeypatch.setattr(
        replay_mod.SocketTargetSink, "start", lambda self: started.append(True)
    )
    rec_dir = _make_aruco_recording_dir(tmp_path, n=2)
    log_dir = tmp_path / "logs"
    replay_mod.run_replay(
        str(rec_dir), _VERTIPORT_FINE_PRESET, display="none", output=None,
        log_dir=str(log_dir), log_name="nosinkrep",
    )
    assert started == [], "--target-sink 미지정인데 소켓 sink가 기동됐다"
    assert "target sink 발행 실패" not in (log_dir / "nosinkrep.log").read_text()


def test_replay_sink_bind_failure_is_a_hard_failure_with_nonzero_exit(
    tmp_path, monkeypatch, capsys
):
    """bind 실패는 강등이 아니라 즉사 — main.py와 **같은 종료코드**여야 한다(정책이 갈리면
    호출 스크립트가 두 CLI를 다르게 분기해야 하고, 회귀검증 경로로서의 가치가 사라진다).
    몽키패치 없이 진짜로 포트를 점유해서 확인한다."""
    squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    squatter.bind(("127.0.0.1", 0))
    squatter.listen(1)
    port = squatter.getsockname()[1]
    try:
        rec_dir = _make_aruco_recording_dir(tmp_path, n=2)
        log_dir = tmp_path / "logs"
        with pytest.raises(SystemExit) as exc:
            replay_mod.run_replay(
                str(rec_dir), _VERTIPORT_FINE_PRESET, display="none", output=None,
                log_dir=str(log_dir), log_name="bindfailrep", target_sink=True,
                target_sink_port=port,
            )
        assert exc.value.code == replay_mod.EXIT_SINK_BIND_FAILED != 0
        assert str(port) in capsys.readouterr().err, "stderr에 포트 번호가 없다"
        # 프레임은 한 장도 처리되지 않아야 한다(강등 동작이면 2건이 남는다).
        jsonl = log_dir / "bindfailrep.jsonl"
        frames = [json.loads(l) for l in jsonl.read_text().splitlines()] if jsonl.exists() else []
        assert [r for r in frames if r.get("type") == "frame"] == []
    finally:
        squatter.close()


def test_replay_agl_drives_state_hints_all_the_way_to_terminal(
    tmp_path, recording_sink_cls
):
    """🔴 이 배선의 합격 기준: **AGL이 실린 재생에서 `state_hint`가 `TERMINAL`까지 진행**한다.
    (`main.py`는 AGL 경로가 없어 이 상태를 절대 못 만든다 — 이 테스트가 유일하다.)
    발행 레코드를 직접 관측해 소켓 타이밍에 기대지 않고 결정론적으로 고정한다."""
    alts = [None, None, None] + [2.0] * 9
    rec_dir = _make_aruco_recording_dir_with_telemetry(tmp_path, n=12, alts=alts)
    log_dir = tmp_path / "logs"

    frame_count = replay_mod.run_replay(
        str(rec_dir), _VERTIPORT_FINE_PRESET, display="none", output=None,
        log_dir=str(log_dir), log_name="sinkagl", target_sink=True, target_sink_port=0,
    )
    assert frame_count == 12

    sink = recording_sink_cls.instances[-1]
    targets = [r for r in sink.records if r["type"] == RECORD_TYPE_TARGET]
    hints = [r for r in sink.records if r["type"] == RECORD_TYPE_STATE_HINT]
    assert len(targets) == 12 and len(hints) == 12, "매 프레임 target 1건 + state_hint 1건"
    for rec in targets:
        assert all(k in rec for k in REQUIRED_TARGET_KEYS)
    for rec in hints:
        assert all(k in rec for k in REQUIRED_STATE_HINT_KEYS)
        assert rec["command_is_advisory"] is True
        assert "command" not in rec       # §6.3 — command는 명령이 아니다

    states = [h["state"] for h in hints]
    assert "TERMINAL" in states, f"AGL이 실렸는데 TERMINAL 미도달 — 실제 상태열: {states}"
    assert all(t["valid"] is True for t in targets)
    seqs = [r["seq"] for r in sink.records]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


class _GatedSource:
    """소비자가 붙을 때까지 첫 프레임을 미루는 래퍼 — 실소켓 종단간 테스트를 타이밍이 아니라
    **결정론**으로 만든다(test_main.py `_GatedVideoReader`와 같은 수법)."""

    gate: threading.Event = threading.Event()

    def __init__(self, inner):
        self._inner = inner

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def __iter__(self):
        _GatedSource.gate.wait(20.0)
        yield from self._inner


def test_replay_sink_streams_terminal_state_hints_to_a_stdlib_consumer(tmp_path, monkeypatch):
    """종단간: 순수 stdlib 소비자(`socket` + `makefile().readline()`, vision 미import)가 실제
    JSON Lines를 읽고, 그 안에 **AGL로 도달한 `TERMINAL` state_hint**가 실제로 들어 있다.
    재생 종료 시 EOF까지 확인(§5.4 "프로세스 사망 → 소비자가 EOF 즉시 수신")."""
    real_open = replay_mod.open_dir_or_bag
    _GatedSource.gate = threading.Event()
    monkeypatch.setattr(replay_mod, "open_dir_or_bag", lambda p: _GatedSource(real_open(p)))

    alts = [None, None, None] + [2.0] * 9
    rec_dir = _make_aruco_recording_dir_with_telemetry(tmp_path, n=12, alts=alts)
    port = 18101
    errors: list = []
    result: dict = {}

    def _run():
        try:
            result["n"] = replay_mod.run_replay(
                str(rec_dir), _VERTIPORT_FINE_PRESET, display="none", output=None,
                log_dir=str(tmp_path / "logs"), log_name="e2erep",
                target_sink=True, target_sink_port=port,
            )
        except BaseException as e:      # noqa: BLE001 - 스레드 예외를 테스트로 전달
            errors.append(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    try:
        conn = None
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and conn is None:
            try:
                conn = socket.create_connection(("127.0.0.1", port), timeout=5.0)
            except OSError:
                time.sleep(0.01)
        assert conn is not None, "sink 서버에 접속하지 못했다"

        with conn:
            _GatedSource.gate.set()          # 이제 프레임이 흐른다
            conn.settimeout(20.0)
            f = conn.makefile("r", encoding="utf-8")
            lines = []
            while True:                      # EOF까지 읽는다(=재생 종료)
                line = f.readline()
                if line == "":
                    break
                lines.append(line)

        t.join(timeout=20.0)
        assert not t.is_alive() and not errors, f"run_replay가 정상 종료하지 않음: {errors}"
        assert result.get("n") == 12

        assert all(l.endswith("\n") for l in lines), "JSONL 프레이밍(한 줄=한 레코드) 위반"
        records = [json.loads(l) for l in lines]
        hints = [r for r in records if r["type"] == RECORD_TYPE_STATE_HINT]
        targets = [r for r in records if r["type"] == RECORD_TYPE_TARGET]
        assert targets and hints
        assert "TERMINAL" in [h["state"] for h in hints], (
            f"실소켓으로 받은 state_hint에 TERMINAL이 없다: {[h['state'] for h in hints]}"
        )
        assert all(len(t_["position_frd"]) == 3 for t_ in targets)
        seqs = [r["seq"] for r in records]
        assert seqs == sorted(seqs), "seq가 단조증가가 아니다(순서가 뒤바뀌었다)"
    finally:
        _GatedSource.gate.set()
        t.join(timeout=10.0)


def test_replay_display_none_never_draws_the_sink_overlay(tmp_path, monkeypatch):
    """헤드리스 기본 경로에서 오버레이 비용은 0(main.py와 동일 계약)."""
    calls = []
    monkeypatch.setattr(replay_mod, "draw_sink_status", lambda *a, **k: calls.append(k))
    rec_dir = _make_aruco_recording_dir(tmp_path, n=2)
    replay_mod.run_replay(
        str(rec_dir), _VERTIPORT_FINE_PRESET, display="none", output=None,
        log_dir=str(tmp_path / "logs"), log_name="ovnonerep",
        target_sink=True, target_sink_port=0,
    )
    assert calls == [], "--display none 인데 오버레이를 그렸다"


def test_replay_display_stream_draws_the_sink_overlay_with_live_counters(tmp_path, monkeypatch):
    """`--display`가 켜지면 매 프레임 오버레이를 그리고, 실제 sink 카운터가 실려야 한다."""
    seen = []
    real = replay_mod.draw_sink_status

    def _spy(image, **kwargs):
        seen.append(kwargs)
        return real(image, **kwargs)

    monkeypatch.setattr(replay_mod, "draw_sink_status", _spy)
    rec_dir = _make_aruco_recording_dir(tmp_path, n=3)
    replay_mod.run_replay(
        str(rec_dir), _VERTIPORT_FINE_PRESET, display="stream", output=None,
        log_dir=str(tmp_path / "logs"), log_name="ovstreamrep", stream_port=0,
        target_sink=True, target_sink_port=0,
    )
    assert len(seen) == 3
    assert all(k["enabled"] is True and k["consumers"] == 0 for k in seen)
    assert [k["seq"] for k in seen] == [2, 4, 6], "발행 seq가 화면에 반영되지 않는다"
