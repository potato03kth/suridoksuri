"""main.py CLI 헤드리스 안전성 회귀 테스트 + 관측성(로거/블랙박스) 연결 테스트.

불변식: 기본값 --display none 은 어떤 GUI 함수(cv2.imshow 등)도 호출하지 않는다.
드론(디스플레이 없음)에서의 크래시를 방지하는 계약이므로 절대 깨지면 안 된다.

로그 출력은 실제 리포지토리 vision/results/를 더럽히지 않도록 모든 테스트에서
--log-dir 을 tmp_path 하위로 명시한다.
"""
import json
import math
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import vision.main as main_mod
from vision.core.wire import (
    RECORD_TYPE_STATE_HINT,
    RECORD_TYPE_TARGET,
    REQUIRED_STATE_HINT_KEYS,
    REQUIRED_TARGET_KEYS,
)
from vision.utils.frame_source import FrameRecord
from vision.utils.target_sink import SocketTargetSink

_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_VERTIPORT_FINE_PRESET = str(Path(main_mod.__file__).parent / "presets" / "vertiport_fine.yaml")
_DISTRESS_FINE_PRESET = str(Path(main_mod.__file__).parent / "presets" / "distress_fine.yaml")


def _write_image(tmp_path) -> str:
    img = np.full((120, 120, 3), 180, dtype=np.uint8)
    p = tmp_path / "frame.png"
    cv2.imwrite(str(p), img)
    return str(p)


def _write_aruco_image(tmp_path, marker_id: int = 23, name: str = "aruco.png") -> str:
    """ArUco Phase 4 — ID=23 마커를 합성한 이미지(vision/tests/test_aruco.py와 동일 패턴 재사용)."""
    canvas = np.full((300, 300, 3), 255, dtype=np.uint8)
    marker_gray = cv2.aruco.generateImageMarker(_DICTIONARY, marker_id, 150)
    canvas[50:200, 50:200] = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
    p = tmp_path / name
    cv2.imwrite(str(p), canvas)
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


# ===========================================================================
# ArUco Phase 4(docs/vision_aruco_branch.md) — TargetEstimate가 실제로 JSONL에 실리는지
# ===========================================================================


def test_aruco_preset_writes_target_estimate_into_jsonl_chosen(tmp_path, monkeypatch):
    """§Phase4 핵심 요구: vertiport_fine.yaml로 실제 ArUco 마커 이미지를 실제로 처리하면
    JSONL의 chosen.target_estimate 안에 calib_id/calib_accuracy/not_for_closed_loop_30cm/
    position/orientation이 실제로 찍히는지 — 몽키패치로 때우지 않고 실제 파이프라인+블랙박스."""
    img_path = _write_aruco_image(tmp_path)
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vision.main", img_path,
            "--preset", _VERTIPORT_FINE_PRESET,
            "--log-dir", str(log_dir), "--log-name", "arucorun",
        ],
    )
    main_mod.main()

    records = [json.loads(l) for l in (log_dir / "arucorun.jsonl").read_text().splitlines()]
    assert len(records) == 1
    chosen = records[0]["chosen"]
    assert chosen is not None
    estimate = chosen["target_estimate"]

    assert estimate["target_type"] == "aruco_23"
    assert len(estimate["position"]) == 3
    assert len(estimate["orientation"]) == 4
    # provenance echo(§7.3) — nominal.yaml 그대로 반영됐는지(하드코딩 아님)
    assert estimate["calib_accuracy"] == "unverified"
    assert estimate["not_for_closed_loop_30cm"] is True
    assert estimate["calib_id"].endswith("nominal.yaml")


def test_no_aruco_marker_frame_has_no_target_estimate_and_does_not_crash(tmp_path, monkeypatch):
    """§Phase4 요구: ArUco 마커가 없는 프레임은 크래시 없이 target_estimate가 없어야 한다."""
    img_path = _write_image(tmp_path)  # 마커 없는 단색 이미지
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vision.main", img_path,
            "--preset", _VERTIPORT_FINE_PRESET,
            "--log-dir", str(log_dir), "--log-name", "noarucorun",
        ],
    )
    main_mod.main()  # 크래시 없이 끝나야 함

    records = [json.loads(l) for l in (log_dir / "noarucorun.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["chosen"] is None


def test_missing_calib_file_logs_warning_and_still_runs(tmp_path, monkeypatch):
    """calib 파일이 없어도(--calib 오지정) 파이프라인 전체가 죽지 않고 target_estimate만 생략."""
    img_path = _write_aruco_image(tmp_path)
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vision.main", img_path,
            "--preset", _VERTIPORT_FINE_PRESET,
            "--calib", str(tmp_path / "does_not_exist.yaml"),
            "--log-dir", str(log_dir), "--log-name", "nocalibrun",
        ],
    )
    main_mod.main()

    records = [json.loads(l) for l in (log_dir / "nocalibrun.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["chosen"] is None

    log_text = (log_dir / "nocalibrun.log").read_text()
    assert "캘리브레이션 로드 실패" in log_text


# ===========================================================================
# §9 6번(공통 상태머신) 배선 — JSONL `state` 필드가 실제로 채워지는지(더 이상 전부 null 아님)
# ===========================================================================


def _write_aruco_video(tmp_path, n_frames: int = 6, marker_id: int = 23) -> str:
    """같은 ArUco ID=23 마커 프레임을 n_frames번 반복한 영상 — 상태머신이 실제로
    ACQUIRE 이후 단계까지 전이하려면 연속 프레임(lock_confirm_frames 기본 3)이 필요하다."""
    video_path = tmp_path / "aruco_clip.mp4"
    canvas = np.full((300, 300, 3), 255, dtype=np.uint8)
    marker_gray = cv2.aruco.generateImageMarker(_DICTIONARY, marker_id, 150)
    canvas[50:200, 50:200] = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (300, 300)
    )
    for _ in range(n_frames):
        writer.write(canvas)
    writer.release()
    return str(video_path)


def test_state_field_is_populated_and_progresses_through_real_pipeline(tmp_path, monkeypatch):
    """§9 6번 요구: log_frame의 `state` 파라미터에 실제 상태머신 결과가 실려야 한다 —
    몽키패치로 값을 주입하지 않고, 실제 ArUco 검출이 반복되는 실제 영상을 실제로 돌려
    JSONL에 null이 아닌 실제 상태 문자열이(그것도 ACQUIRE에 머물지 않고 진행하며) 찍히는지 확인."""
    video_path = _write_aruco_video(tmp_path, n_frames=6)
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vision.main", video_path,
            "--preset", _VERTIPORT_FINE_PRESET,
            "--log-dir", str(log_dir), "--log-name", "staterun",
        ],
    )
    main_mod.main()

    records = [json.loads(l) for l in (log_dir / "staterun.jsonl").read_text().splitlines()]
    assert len(records) == 6
    states = [r["state"] for r in records]

    assert all(s is not None for s in states), "state 필드가 여전히 전부 null — 배선 안 됨"
    assert all(isinstance(s, str) and s for s in states)
    # 반복되는 실제 검출만으로 ACQUIRE에 머물지 않고 실제로 진행해야 한다(커밋 게이트를
    # 실제로 통과한다는 증거) — 정확한 프레임별 값은 core/state_machine 단위테스트가 담당.
    assert states[0] != "ACQUIRE"
    assert "PRECISION_SERVO" in states or "LOCK" in states
    # command 힌트도 함께 실린다(같은 log_frame 파라미터 재사용).
    assert all(r["command"] is not None for r in records)


def _distress_fine_frame(mat_size: int = 300, canvas: int = 460, box_ratio: float = 0.0667) -> np.ndarray:
    """distress_fine.yaml 캐스케이드(초록 매트+흰 박스) 검증용 최소 합성 프레임 —
    vision/tests/golden/generate_synthetic.py의 `_synthetic_distress`와 동일 패턴(실측 스펙
    비율 20cm/3.0m≈0.0667, vision/CLAUDE.md 참조). 테스트 파일 간 상호 의존을 피하려 이
    파일 안에 얇게 중복(프로젝트 "각자 얇게 중복" 관례, 기존 `_write_aruco_image`도 동일)."""
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


def _write_distress_fine_video(tmp_path, n_frames: int = 6) -> str:
    """같은 초록 매트+흰 박스 프레임을 n_frames번 반복한 영상 — ArUco 경로와 별개로
    white_box_detector가 실제로 fine_locked를 True로 만들어 상태머신이 진행하는지 검증."""
    video_path = tmp_path / "distress_fine_clip.mp4"
    frame = _distress_fine_frame()
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()
    return str(video_path)


def test_distress_fine_state_progresses_through_real_pipeline(tmp_path, monkeypatch):
    """§9 "끊어진 체인을 잇는 작업" 요구: ArUco가 아니라 ② 조난자 fine(흰 박스 확정, §5.3)
    경로로도 `fine_locked`가 실제로 True가 되어 상태머신이 CENTER_DESCEND를 넘어 진행하는지 —
    몽키패치 없이 실제 영상+실제 파이프라인+실제 white_box_detector로 확인."""
    video_path = _write_distress_fine_video(tmp_path, n_frames=6)
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vision.main", video_path,
            "--preset", _DISTRESS_FINE_PRESET,
            "--log-dir", str(log_dir), "--log-name", "distressrun",
        ],
    )
    main_mod.main()

    records = [json.loads(l) for l in (log_dir / "distressrun.jsonl").read_text().splitlines()]
    assert len(records) == 6
    states = [r["state"] for r in records]

    assert all(s is not None for s in states), "state 필드가 여전히 전부 null — 배선 안 됨"
    assert states[0] != "ACQUIRE"
    assert "PRECISION_SERVO" in states or "LOCK" in states
    assert all(r["command"] is not None for r in records)

    # 착륙점(landing_point_px) 기준 center_error_norm이 실제로 detections에 실렸는지도 확인
    # (박스 중심이 아니라 착륙점 기준이어야 한다는 §5.3 설계 포인트의 배선 증거).
    for r in records:
        assert r["detections"], "white_box_detector 확정 detection이 실려야 한다"


# ===========================================================================
# 라이브 모드(LiveFrameSource 배선) — 실카메라/picamera2 없이 `main_mod.LiveFrameSource`를
# 몽키패치해 검증(vision.utils.frame_source의 LiveFrameSource 몽키패치 대신 이 방식을 택한
# 이유: main.py가 top-level에서 `from vision.utils.frame_source import LiveFrameSource`로
# 이름을 자기 네임스페이스에 들여왔으므로, main_mod.LiveFrameSource를 바꿔치는 쪽이 실제
# picamera2 가짜 모듈 주입보다 더 직접적이고 다른 프레임소스 테스트(test_frame_source.py)와
# 책임이 겹치지 않는다). 무한 이터레이터라는 라이브 소스의 특성 때문에 실제 카메라 대신
# **유한하게 끝나는 가짜**(N개 프레임 후 종료 또는 KeyboardInterrupt)로 무한루프 없이 검증한다.
# ===========================================================================


def _live_frame(value: int = 180):
    return np.full((120, 120, 3), value, dtype=np.uint8)


class _FakeLiveSourceFiniteFrames:
    """실카메라 대신 N개(3개) 프레임만 내고 정상 종료하는 가짜 — 무한루프에 테스트가 걸리지
    않으면서 `_run_live()` 경로 자체(파이프라인 실행→블랙박스 기록)를 검증하기 위함."""

    def __init__(self, camera_num=0, resolution=None, retries=3, retry_delay=1.0):
        self.camera_num = camera_num
        self.resolution = resolution
        self.retries = retries
        self.retry_delay = retry_delay
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *_exc):
        self.exited = True
        return False

    def __iter__(self):
        for i in range(3):
            yield FrameRecord(frame_id=i, ts=float(i), image=_live_frame(), telemetry={})


class _FakeLiveSourceInterruptsAfter2Frames:
    """2프레임을 낸 뒤 KeyboardInterrupt를 던져 실기체 Ctrl+C를 시뮬레이션."""

    def __init__(self, camera_num=0, resolution=None, retries=3, retry_delay=1.0):
        self.camera_num = camera_num
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *_exc):
        self.exited = True
        return False  # 예외를 삼키지 않음 — 실제 컨텍스트매니저(LiveFrameSource)와 동일 계약

    def __iter__(self):
        yield FrameRecord(frame_id=0, ts=0.0, image=_live_frame(), telemetry={})
        yield FrameRecord(frame_id=1, ts=1.0, image=_live_frame(), telemetry={})
        raise KeyboardInterrupt()


def test_live_input_spec_dispatches_to_run_live_and_writes_jsonl(tmp_path, monkeypatch):
    """`input`에 특수값 `live`를 주면 실제로 `_run_live()` 경로를 타 파이프라인+블랙박스가
    실제로 동작하는지 — 유한 가짜 소스로 무한루프 없이 검증."""
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(main_mod, "LiveFrameSource", _FakeLiveSourceFiniteFrames)
    monkeypatch.setattr(
        sys, "argv",
        ["vision.main", "live", "--log-dir", str(log_dir), "--log-name", "liverun"],
    )
    main_mod.main()

    jsonl_path = log_dir / "liverun.jsonl"
    assert jsonl_path.exists(), "라이브 모드에서도 블랙박스 JSONL이 실제로 생성돼야 함"
    records = [json.loads(l) for l in jsonl_path.read_text().splitlines()]
    assert [r["frame_id"] for r in records] == [0, 1, 2]
    assert all(r["type"] == "frame" for r in records)
    assert all(r["latency"] >= 0 for r in records)


def test_live_input_spec_with_camera_num_parses_and_passes_through(tmp_path, monkeypatch):
    """`live:N` 스펙에서 N이 실제로 LiveFrameSource(camera_num=N)에 전달되는지."""
    captured: dict = {}

    class _Capturing(_FakeLiveSourceFiniteFrames):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(main_mod, "LiveFrameSource", _Capturing)
    monkeypatch.setattr(
        sys, "argv",
        ["vision.main", "live:2", "--log-dir", str(tmp_path / "logs")],
    )
    main_mod.main()

    assert captured["camera_num"] == 2


def test_live_resolution_flag_passed_through_to_live_frame_source(tmp_path, monkeypatch):
    """--live-resolution WxH가 LiveFrameSource(resolution=(W, H))로 전달되는지."""
    captured: dict = {}

    class _Capturing(_FakeLiveSourceFiniteFrames):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(main_mod, "LiveFrameSource", _Capturing)
    monkeypatch.setattr(
        sys, "argv",
        [
            "vision.main", "live", "--live-resolution", "640x480",
            "--log-dir", str(tmp_path / "logs"),
        ],
    )
    main_mod.main()

    assert captured["resolution"] == (640, 480)


def test_live_mode_keyboard_interrupt_closes_blackbox_cleanly_no_crash(tmp_path, monkeypatch):
    """§확정 전제: Ctrl+C(KeyboardInterrupt)로 라이브 모드가 스택트레이스 없이 정상 종료돼야
    하고, 그 전에 기록된 프레임은 JSONL에 남아야 하며, blackbox.close()가 반드시 불려야 한다."""
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(main_mod, "LiveFrameSource", _FakeLiveSourceInterruptsAfter2Frames)

    closed = []
    real_close = main_mod.BlackBoxLogger.close

    def _spy_close(self, *a, **kw):
        closed.append(True)
        return real_close(self, *a, **kw)

    monkeypatch.setattr(main_mod.BlackBoxLogger, "close", _spy_close)
    monkeypatch.setattr(
        sys, "argv",
        ["vision.main", "live", "--log-dir", str(log_dir), "--log-name", "interruptrun"],
    )

    main_mod.main()  # KeyboardInterrupt가 main() 밖으로 전파되면 안 됨(크래시 스택트레이스 금지)

    assert closed == [True], "KeyboardInterrupt 발생해도 blackbox.close()가 호출돼야 함(리소스 leak 방지)"

    records = [json.loads(l) for l in (log_dir / "interruptrun.jsonl").read_text().splitlines()]
    assert [r["frame_id"] for r in records] == [0, 1], "interrupt 전에 낸 프레임은 기록돼야 함"

    log_text = (log_dir / "interruptrun.log").read_text()
    assert "Ctrl+C" in log_text, "정상 종료 로그가 남아야 함(스택트레이스로 죽으면 안 됨)"


class _FakeLiveSourceSigtermAfter2Frames:
    """2프레임을 낸 뒤 자기 프로세스에 SIGTERM을 보내 비대화형 배포 환경(systemd stop 등)의
    표준 종료 신호를 시뮬레이션한다. SIGINT(KeyboardInterrupt)와 별개 경로 검증 —
    비대화형 SSH 백그라운드 자식은 SIGINT가 SIG_IGN일 수 있어 못 믿는다(§h264_stream.py 실측과
    동일 근거, `vision/main.py::_install_sigterm_handler`)."""

    def __init__(self, camera_num=0, resolution=None, retries=3, retry_delay=1.0):
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *_exc):
        self.exited = True
        return False  # 예외를 삼키지 않음 — 실제 컨텍스트매니저(LiveFrameSource)와 동일 계약

    def __iter__(self):
        yield FrameRecord(frame_id=0, ts=0.0, image=_live_frame(), telemetry={})
        yield FrameRecord(frame_id=1, ts=1.0, image=_live_frame(), telemetry={})
        os.kill(os.getpid(), signal.SIGTERM)
        # 핸들러가 stop_event를 세팅한 뒤에도 제너레이터는 (실카메라처럼) 다음 프레임을 계속
        # 낼 수 있다 — `_run_live`가 이 프레임을 실제로 처리하지 않고 다음 프레임 경계에서
        # 즉시 버려야 한다(아래 테스트가 frame_id=2 부재로 검증).
        yield FrameRecord(frame_id=2, ts=2.0, image=_live_frame(), telemetry={})


def test_live_mode_sigterm_closes_blackbox_cleanly_no_crash(tmp_path, monkeypatch):
    """SIGTERM(비대화형 배포 환경 표준 종료 신호)도 KeyboardInterrupt와 동일하게 스택트레이스
    없이 정상 종료 + blackbox.close() 보장돼야 한다. 전역 SIGTERM 핸들러는 테스트 종료 후
    반드시 원복한다(§test_h264_stream.py::test_install_sigterm_handler_sets_stop_event_on_sigterm
    과 동일한 이유 — pytest 프로세스 전체에 영향 주지 않기 위함)."""
    original_term = signal.getsignal(signal.SIGTERM)
    try:
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(main_mod, "LiveFrameSource", _FakeLiveSourceSigtermAfter2Frames)

        closed = []
        real_close = main_mod.BlackBoxLogger.close

        def _spy_close(self, *a, **kw):
            closed.append(True)
            return real_close(self, *a, **kw)

        monkeypatch.setattr(main_mod.BlackBoxLogger, "close", _spy_close)
        monkeypatch.setattr(
            sys, "argv",
            ["vision.main", "live", "--log-dir", str(log_dir), "--log-name", "sigtermrun"],
        )

        main_mod.main()  # SIGTERM이 main() 밖으로 전파되면 안 됨(크래시 없이 정상 종료)

        assert closed == [True], "SIGTERM 수신해도 blackbox.close()가 호출돼야 함(리소스 leak 방지)"

        records = [json.loads(l) for l in (log_dir / "sigtermrun.jsonl").read_text().splitlines()]
        assert [r["frame_id"] for r in records] == [0, 1], (
            "SIGTERM 이후 낸 프레임(frame_id=2)은 처리되면 안 됨 — 다음 프레임 경계에서 즉시 종료"
        )

        log_text = (log_dir / "sigtermrun.log").read_text()
        assert "SIGTERM" in log_text, "SIGTERM으로 인한 정상 종료임이 로그에 남아야 함"
    finally:
        signal.signal(signal.SIGTERM, original_term)


def test_live_invalid_camera_num_spec_exits_with_usage_error(monkeypatch):
    """`live:notanumber`처럼 잘못된 라이브 스펙은 (실제 카메라 시도 없이) 사용법 에러로 종료."""
    monkeypatch.setattr(sys, "argv", ["vision.main", "live:notanumber"])
    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 2


def test_non_live_input_still_requires_existing_file(tmp_path, monkeypatch):
    """회귀: 라이브 스펙이 아닌 일반 입력은 여전히 input_path.exists() 체크를 받는다(기존
    이미지/영상 흐름을 절대 깨지 않는다는 확정 전제)."""
    monkeypatch.setattr(
        sys, "argv", ["vision.main", str(tmp_path / "does_not_exist.png")]
    )
    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 1


# ===========================================================================
# `--target-sink` 배선 (vision↔fc 인터페이스, docs/vision_fc_interface.md §9 작업 V5)
#
# 배선 지점: `_run_image`/`_run_video`/`_run_live` **세 경로 전부**. 라이브가 정밀착륙의
# 본령이지만, §7.5(기록·재생)가 "같은 파이프라인으로 오프라인 재생"을 회귀검증의 최대
# 레버로 못박고 있어 영상 경로에도 붙어 있어야 sink 계약을 책상에서 회귀로 잡을 수 있다.
# opt-in이라 꺼져 있으면 세 경로 모두 비용 0이다.
#
# 아래 테스트들은 `main_mod.SocketTargetSink`를 **진짜 클래스의 서브클래스**로 바꿔치기해
# (진짜 소켓 bind + 진짜 레코드 조립은 그대로 두고) 발행된 레코드를 관측한다 — `_publish_to_sink`
# 의 `isinstance(sink, SocketTargetSink)` 게이트가 서브클래스에서도 참이라 실제 경로가 돈다.
# 진짜 바이트가 진짜 소비자에게 흐르는지는 맨 아래 `..._streams_real_jsonl_...`가 실소켓으로,
# 소켓 계층 자체는 tests/test_target_sink.py가 담당한다.
# ===========================================================================


class _RecordingSocketTargetSink(SocketTargetSink):
    """발행된 레코드를 그대로 모아두는 진짜 sink(진짜 서버가 뜬다, port=0 임시포트)."""

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
    monkeypatch.setattr(main_mod, "SocketTargetSink", _RecordingSocketTargetSink)
    return _RecordingSocketTargetSink


def _sink_args(port: str = "0") -> list[str]:
    return ["--target-sink", "--target-sink-port", port]


def _write_blank_video(tmp_path, n_frames: int = 4) -> str:
    """마커가 하나도 없는 영상 — vertiport_fine.yaml로 돌리면 매 프레임 검출 0이 된다."""
    video_path = tmp_path / "blank_clip.mp4"
    frame = np.full((300, 300, 3), 255, dtype=np.uint8)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (300, 300))
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()
    return str(video_path)


def test_target_sink_is_off_by_default_and_never_binds_a_socket(tmp_path, monkeypatch):
    """opt-in 불변식(`--display stream`과 동일): 플래그를 안 주면 소켓도 스레드도 안 뜨고
    발행 시도조차 없다 — 기존 실행 경로가 조금도 달라지면 안 된다."""
    started = []
    monkeypatch.setattr(
        main_mod.SocketTargetSink, "start", lambda self: started.append(True)
    )
    img_path = _write_image(tmp_path)
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(
        sys, "argv",
        ["vision.main", img_path, "--log-dir", str(log_dir), "--log-name", "nosink"],
    )
    main_mod.main()

    assert started == [], "--target-sink 미지정인데 소켓 sink가 기동됐다"
    # NullSink 경로에서 레코드 조립을 시도하면 AttributeError가 잡혀 경고가 남는다 —
    # 그 경고가 없다는 것이 "기본 경로에서 조립 비용조차 안 든다"의 증거.
    assert "target sink 발행 실패" not in (log_dir / "nosink.log").read_text()


def test_target_sink_publishes_target_and_state_hint_every_frame(
    tmp_path, monkeypatch, recording_sink_cls
):
    """§9 V5 핵심: 실제 ArUco 영상을 실제 파이프라인으로 돌리면 매 프레임 `target` 1건 +
    `state_hint` 1건이 실제로 발행되고, 계약 상수(`REQUIRED_*_KEYS`)를 전부 만족해야 한다."""
    video_path = _write_aruco_video(tmp_path, n_frames=6)
    monkeypatch.setattr(
        sys, "argv",
        [
            "vision.main", video_path, "--preset", _VERTIPORT_FINE_PRESET,
            "--log-dir", str(tmp_path / "logs"), *_sink_args(),
        ],
    )
    main_mod.main()

    sink = recording_sink_cls.instances[-1]
    targets = [r for r in sink.records if r["type"] == RECORD_TYPE_TARGET]
    hints = [r for r in sink.records if r["type"] == RECORD_TYPE_STATE_HINT]
    assert len(targets) == 6, "프레임마다 target 레코드가 정확히 1건씩 나가야 한다"
    assert len(hints) == 6, "상태머신 힌트도 매 프레임 함께 나가야 한다(§6.3 advisory)"

    for rec in targets:
        assert all(k in rec for k in REQUIRED_TARGET_KEYS)
    for rec in hints:
        assert all(k in rec for k in REQUIRED_STATE_HINT_KEYS)

    # 실제 검출이 실린 유효 레코드인지 — 3프레임 좌표가 다 있고 provenance가 nominal.yaml
    assert all(r["valid"] is True for r in targets)
    assert all(len(r["position_cam"]) == 3 for r in targets)
    assert all(len(r["position_frd"]) == 3 for r in targets)
    assert all(r["target_type"] == "aruco_23" for r in targets)
    assert all(r["calib_id"].endswith("nominal.yaml") for r in targets)

    # §6.3 — command는 명령이 아니다. 맨 이름 `command` 키는 있으면 안 된다.
    assert all(h["command_is_advisory"] is True for h in hints)
    assert all("command" not in h for h in hints)
    assert any(h["state"] in ("LOCK", "PRECISION_SERVO") for h in hints), (
        "상태머신 Decision이 실제로 실려야 한다(ACQUIRE에 머물면 배선이 죽은 것)"
    )
    # seq는 단조증가 — 소비자가 유실을 감지하는 유일한 수단
    seqs = [r["seq"] for r in sink.records]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_target_sink_publishes_invalid_records_instead_of_going_silent(
    tmp_path, monkeypatch, recording_sink_cls
):
    """§5.4 계약 2번(페일세이프의 핵심): 검출이 없어도 **침묵하지 않는다** — 침묵은 "노드 사망"
    으로 예약돼 있으므로 "안 보임"은 valid=false + 사유로 명시해야 소비자가 구분할 수 있다."""
    video_path = _write_blank_video(tmp_path, n_frames=4)
    monkeypatch.setattr(
        sys, "argv",
        [
            "vision.main", video_path, "--preset", _VERTIPORT_FINE_PRESET,
            "--log-dir", str(tmp_path / "logs"), *_sink_args(),
        ],
    )
    main_mod.main()

    sink = recording_sink_cls.instances[-1]
    targets = [r for r in sink.records if r["type"] == RECORD_TYPE_TARGET]
    assert len(targets) == 4, "검출이 없다고 발행을 멈추면 안 된다(침묵=사망 신호)"
    assert all(r["valid"] is False for r in targets)
    assert all(r["reason"] == main_mod.SINK_REASON_NO_DETECTION for r in targets)
    # 포즈는 0이 아니라 null이어야 한다 — 0이면 "기체 바로 아래에 타겟"이라는 최악의 거짓말
    assert all(r["position_frd"] is None for r in targets)
    # 검출이 없어도 상태머신 힌트는 계속 나간다(소비자가 HOLD/ABORT를 볼 수 있어야 한다)
    assert len([r for r in sink.records if r["type"] == RECORD_TYPE_STATE_HINT]) == 4


def test_missing_calib_makes_sink_say_no_calibration_not_go_silent(
    tmp_path, monkeypatch, recording_sink_cls
):
    """calib 미로드도 침묵이 아니라 사유가 붙은 valid=false로 나가야 한다(사후분석 가능)."""
    img_path = _write_aruco_image(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        [
            "vision.main", img_path, "--preset", _VERTIPORT_FINE_PRESET,
            "--calib", str(tmp_path / "nope.yaml"),
            "--log-dir", str(tmp_path / "logs"), *_sink_args(),
        ],
    )
    main_mod.main()

    targets = [r for r in recording_sink_cls.instances[-1].records if r["type"] == RECORD_TYPE_TARGET]
    assert len(targets) == 1
    assert targets[0]["valid"] is False
    assert targets[0]["reason"] == main_mod.SINK_REASON_NO_CALIB


class _ExplodingSink(_RecordingSocketTargetSink):
    """발행 시도마다 예외를 던지는 sink — 최악의 소비자/조립 실패를 시뮬레이션."""

    def publish_target(self, estimate):
        raise RuntimeError("boom-target")

    def publish_invalid(self, **kwargs):
        raise RuntimeError("boom-invalid")

    def publish_state_hint(self, decision, **kwargs):
        raise RuntimeError("boom-hint")


def test_sink_exceptions_never_stop_the_detection_pipeline(tmp_path, monkeypatch):
    """🔴 절대 요구: sink가 어떤 이유로 실패해도 검출은 계속된다 — 착륙 유도보다 검출이 먼저
    죽는 일은 없어야 한다. 매 프레임 예외가 터져도 6프레임이 전부 JSONL에 남아야 한다."""
    _ExplodingSink.instances = []
    monkeypatch.setattr(main_mod, "SocketTargetSink", _ExplodingSink)
    video_path = _write_aruco_video(tmp_path, n_frames=6)
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(
        sys, "argv",
        [
            "vision.main", video_path, "--preset", _VERTIPORT_FINE_PRESET,
            "--log-dir", str(log_dir), "--log-name", "boomrun", *_sink_args(),
        ],
    )
    main_mod.main()  # 예외가 밖으로 새면 여기서 죽는다

    records = [json.loads(l) for l in (log_dir / "boomrun.jsonl").read_text().splitlines()]
    assert [r["frame_id"] for r in records] == list(range(6)), (
        "sink 예외 때문에 검출 파이프라인이 멈췄다"
    )
    assert "target sink 발행 실패" in (log_dir / "boomrun.log").read_text(), (
        "삼키되 조용히 삼키면 안 된다 — 경고 로그는 남아야 한다"
    )


def test_sink_bind_failure_degrades_to_nullsink_and_keeps_detecting(tmp_path, monkeypatch):
    """기동(bind) 실패도 검출을 죽이지 않는다 — NullSink로 강등하고 계속 간다(시끄럽게)."""
    monkeypatch.setattr(
        main_mod.SocketTargetSink,
        "start",
        lambda self: (_ for _ in ()).throw(OSError("address already in use")),
    )
    img_path = _write_image(tmp_path)
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(
        sys, "argv",
        [
            "vision.main", img_path, "--log-dir", str(log_dir),
            "--log-name", "bindfail", *_sink_args("8091"),
        ],
    )
    main_mod.main()  # SystemExit/OSError로 죽으면 안 된다

    records = [json.loads(l) for l in (log_dir / "bindfail.jsonl").read_text().splitlines()]
    assert len(records) == 1, "sink 기동 실패로 검출 파이프라인이 죽었다"
    assert "target sink 기동 실패" in (log_dir / "bindfail.log").read_text()


class _FakeLiveSourceSigtermAfter2FramesWithSink(_FakeLiveSourceSigtermAfter2Frames):
    """sink가 켜진 상태의 SIGTERM 경로 — 클래스 재사용(동작 동일)."""


def test_sink_does_not_hijack_the_live_sigterm_handler(tmp_path, monkeypatch, recording_sink_cls):
    """🔴 회귀: `signal.signal`은 신호당 핸들러가 하나뿐이라, sink 핸들러를 나중에 걸면
    `_install_sigterm_handler`가 등록한 라이브 모드 graceful shutdown을 **덮어써서** 죽인다
    (실기체에서만 드러났던 버그의 재발 경로). sink를 켠 채로도 SIGTERM 이후 프레임(frame_id=2)이
    처리되면 안 된다."""
    original_term = signal.getsignal(signal.SIGTERM)
    try:
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(
            main_mod, "LiveFrameSource", _FakeLiveSourceSigtermAfter2FramesWithSink
        )
        monkeypatch.setattr(
            sys, "argv",
            [
                "vision.main", "live", "--log-dir", str(log_dir),
                "--log-name", "sigtermsink", *_sink_args(),
            ],
        )
        main_mod.main()

        records = [json.loads(l) for l in (log_dir / "sigtermsink.jsonl").read_text().splitlines()]
        assert [r["frame_id"] for r in records] == [0, 1], (
            "sink가 SIGTERM 핸들러를 가로챘다 — 라이브 graceful shutdown이 깨졌다"
        )
        assert "SIGTERM" in (log_dir / "sigtermsink.log").read_text()
        # sink도 함께 정리돼야 한다(§5.4 EOF 경로)
        assert recording_sink_cls.instances[-1].stopping is True
    finally:
        signal.signal(signal.SIGTERM, original_term)


class _GatedVideoReader:
    """소비자가 붙을 때까지 첫 프레임을 미루는 가짜 VideoReader — 실소켓 종단간 테스트를
    타이밍에 기대지 않고 **결정론적으로** 만들기 위한 것(main.py가 서버라 소비자가 늦게
    붙으면 레코드를 놓칠 수 있다)."""

    gate: threading.Event = threading.Event()
    n_frames = 4

    def __init__(self, path):
        self.fps = 10

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def __iter__(self):
        _GatedVideoReader.gate.wait(20.0)
        canvas = np.full((300, 300, 3), 255, dtype=np.uint8)
        marker = cv2.aruco.generateImageMarker(_DICTIONARY, 23, 150)
        canvas[50:200, 50:200] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        for _ in range(_GatedVideoReader.n_frames):
            yield canvas.copy()


def test_target_sink_streams_real_jsonl_to_a_stdlib_consumer_and_eofs_on_exit(
    tmp_path, monkeypatch
):
    """종단간: 순수 stdlib 소비자(`socket` + `makefile().readline()`)가 실제로 붙어 실제 JSON
    Lines를 읽는다 — Phase 2 shim이 하게 될 그대로. main() 종료 시 **EOF**를 받는 것까지
    확인한다(§5.4 "프로세스 사망 → 소비자가 EOF 즉시 수신"의 배선 증거)."""
    import vision.utils.video_reader as vr_mod

    _GatedVideoReader.gate = threading.Event()
    monkeypatch.setattr(vr_mod, "VideoReader", _GatedVideoReader)

    port = 18099
    clip = tmp_path / "gated.mp4"
    clip.write_bytes(b"")  # 내용은 안 읽힌다(_GatedVideoReader가 대체)
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(
        sys, "argv",
        [
            "vision.main", str(clip), "--preset", _VERTIPORT_FINE_PRESET,
            "--log-dir", str(log_dir), "--log-name", "e2esink",
            "--target-sink", "--target-sink-port", str(port),
        ],
    )

    errors: list = []

    def _run():
        try:
            main_mod.main()
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
            _GatedVideoReader.gate.set()      # 이제 프레임이 흐른다
            f = conn.makefile("r", encoding="utf-8")
            lines = [f.readline() for _ in range(4)]
            records = [json.loads(l) for l in lines]

            assert all(l.endswith("\n") for l in lines), "JSONL 프레이밍(한 줄=한 레코드) 위반"
            types = [r["type"] for r in records]
            assert RECORD_TYPE_TARGET in types and RECORD_TYPE_STATE_HINT in types
            tgt = next(r for r in records if r["type"] == RECORD_TYPE_TARGET)
            assert all(k in tgt for k in REQUIRED_TARGET_KEYS)
            assert tgt["valid"] is True and len(tgt["position_frd"]) == 3

            t.join(timeout=20.0)
            assert not t.is_alive() and not errors, f"main()이 정상 종료하지 않음: {errors}"
            # main() 종료 = sink.close() = 클라이언트 소켓 close → 소비자는 EOF를 받는다
            conn.settimeout(5.0)
            while True:
                rest = f.readline()
                if rest == "":
                    break
    finally:
        _GatedVideoReader.gate.set()
        t.join(timeout=10.0)


# ===========================================================================
# ② 조난자 구역(초록 매트) 상대 pose 배선 — 갭 메우기
#
# 배선 전에는 초록구역이 **검출은 되는데 기체에 보낼 좌표가 없었다**(`_solve_aruco_estimate`가
# ArUco 전용이라 `position_flu`가 항상 null). 아래 테스트들은 골든셋(합성) 실프레임 +
# 실제 프리셋 + 실제 sink 레코드 조립으로 그 빈칸이 실제로 채워졌는지 확인한다.
#
# 캘리브레이션은 `vision/tests/golden/distress/synthetic_calib/`(골든 프레임을 만든 전제를
# 역산한 가상 카메라)를 쓴다 — 실장착 `nominal.yaml`은 4608px 센서 기준이라 460px 골든
# 캔버스에 그대로 쓰면 focal도 주점도 안 맞는다.
# ===========================================================================

_GOLDEN_DISTRESS = Path(main_mod.__file__).parent / "tests" / "golden" / "distress"
_DISTRESS_COARSE_PRESET = str(Path(main_mod.__file__).parent / "presets" / "distress_coarse.yaml")


def _golden_calib(canvas: int) -> str:
    return str(_GOLDEN_DISTRESS / "synthetic_calib" / f"canvas{canvas}.yaml")


def _golden_frame_copy(tmp_path, tier: str) -> str:
    """골든 PNG를 그대로 복사해 쓴다(재인코딩 없음 — mp4 압축이 코너를 1px 흔드는 것을 피해
    거리 정확도를 정직하게 잰다)."""
    dst = tmp_path / f"{tier}.png"
    dst.write_bytes((_GOLDEN_DISTRESS / tier / "frame_000.png").read_bytes())
    return str(dst)


def _run_distress(tmp_path, monkeypatch, *, tier, preset, canvas, name):
    monkeypatch.setattr(
        sys, "argv",
        ["vision.main", _golden_frame_copy(tmp_path, tier), "--preset", preset,
         "--calib", _golden_calib(canvas), "--log-dir", str(tmp_path / "logs"),
         "--log-name", name, *_sink_args()],
    )
    main_mod.main()


def test_distress_coarse_preset_emits_a_valid_target_with_real_coordinates(
    tmp_path, monkeypatch, recording_sink_cls
):
    """이 세션이 메운 갭 그 자체: 초록구역 coarse 프레임에서 `valid=true` + 3프레임 좌표가
    실제 숫자로 나가야 한다(예전엔 `valid=false, position_flu=null` 뿐이었다)."""
    _run_distress(tmp_path, monkeypatch, tier="10m", preset=_DISTRESS_COARSE_PRESET,
                  canvas=460, name="dcoarse")

    targets = [r for r in recording_sink_cls.instances[-1].records
               if r["type"] == RECORD_TYPE_TARGET]
    assert len(targets) == 1
    rec = targets[0]
    assert all(k in rec for k in REQUIRED_TARGET_KEYS)
    assert rec["valid"] is True, f"초록구역에서 여전히 무효 레코드가 나간다: {rec['reason']}"
    assert rec["target_type"] == "distress_mat_center"
    for key in ("position_cam", "position_flu", "position_frd"):
        assert rec[key] is not None and len(rec[key]) == 3
        assert all(isinstance(v, float) for v in rec[key])
    # coarse는 매트 중심이 목표라 화면 중앙 매트는 x,y가 0 근처, z가 실제 거리다.
    assert abs(rec["position_cam"][0]) < 0.1 and abs(rec["position_cam"][1]) < 0.1
    assert rec["position_cam"][2] > 1.0


@pytest.mark.parametrize(
    "tier, canvas, expected_z_m", [("10m", 460, 10.0), ("20m", 320, 20.0)]
)
def test_distress_pose_distance_matches_the_golden_altitude_label(
    tmp_path, monkeypatch, recording_sink_cls, tier, canvas, expected_z_m
):
    """거리 정확도 sanity — **알려진 실측 크기(3.0m)가 스케일을 준다**는 접근의 핵심 주장.
    크기 상수를 ArUco의 0.50으로 오용하면 거리가 6배 짧아져 즉시 red가 된다.

    ⚠️ 골든셋은 합성이고 고도 라벨은 "실측 스펙에서 GSD로 역산한 픽셀 크기"라는 뜻이다
    (`tests/golden/generate_synthetic.py`) — 실촬영 검증이 아니다. 여기서 확인하는 것은
    "산출 거리가 그 라벨과 정합한다"까지다."""
    _run_distress(tmp_path, monkeypatch, tier=tier, preset=_DISTRESS_COARSE_PRESET,
                  canvas=canvas, name=f"dist{tier}")

    rec = next(r for r in recording_sink_cls.instances[-1].records
               if r["type"] == RECORD_TYPE_TARGET)
    assert rec["valid"] is True
    z = rec["position_cam"][2]
    assert z == pytest.approx(expected_z_m, rel=0.05), (
        f"{tier} 라벨인데 산출 거리가 {z:.3f}m — 크기 상수/코너 순서/캘리브 불일치를 의심"
    )


def test_distress_fine_landing_point_is_offset_from_the_mat_centre(
    tmp_path, monkeypatch, recording_sink_cls
):
    """§5.3 핵심: 착륙 목표는 매트 중심이 아니라 "박스 옆 빈 초록면"이다. fine 프리셋의
    position은 coarse(매트 중심)의 position과 **눈에 띄게 달라야** 한다. 착륙점을 매트 중심으로
    되돌리는 회귀가 나면 두 값이 같아져 red."""
    _run_distress(tmp_path, monkeypatch, tier="fine", preset=_DISTRESS_FINE_PRESET,
                  canvas=460, name="dfine")
    fine_rec = next(r for r in recording_sink_cls.instances[-1].records
                    if r["type"] == RECORD_TYPE_TARGET)

    _run_distress(tmp_path, monkeypatch, tier="10m", preset=_DISTRESS_COARSE_PRESET,
                  canvas=460, name="dcentre")
    centre_rec = next(r for r in recording_sink_cls.instances[-1].records
                      if r["type"] == RECORD_TYPE_TARGET)

    assert fine_rec["valid"] is True
    assert fine_rec["target_type"] == "distress_landing_point"
    assert centre_rec["target_type"] == "distress_mat_center"

    lateral = math.hypot(fine_rec["position_cam"][0] - centre_rec["position_cam"][0],
                         fine_rec["position_cam"][1] - centre_rec["position_cam"][1])
    # 3.0m 매트 + interior_margin_ratio=0.3 -> 축당 0.7*1.5=1.05m, 대각 약 1.485m
    assert lateral == pytest.approx(1.05 * math.sqrt(2), rel=0.05), (
        f"착륙점이 매트 중심에서 밀려나지 않았다(수평차 {lateral:.3f}m)"
    )
    # 같은 매트를 보고 있으므로 거리(z)는 같아야 한다 — 오프셋이 z로 새면 안 된다.
    assert fine_rec["position_cam"][2] == pytest.approx(centre_rec["position_cam"][2], rel=1e-3)
    assert fine_rec["meta"]["landing_point_source"] == "white_box_landing_point"


def test_distress_estimate_carries_the_calibration_provenance(
    tmp_path, monkeypatch, recording_sink_cls
):
    """§7.3 provenance echo: nominal(미검증) intrinsics로 낸 pose라는 사실이 소비자까지
    그대로 가야 한다 — 이게 빠지면 소비자가 30cm 폐루프에 바로 써버릴 수 있다."""
    _run_distress(tmp_path, monkeypatch, tier="10m", preset=_DISTRESS_COARSE_PRESET,
                  canvas=460, name="dprov")
    rec = next(r for r in recording_sink_cls.instances[-1].records
               if r["type"] == RECORD_TYPE_TARGET)

    assert rec["calib_accuracy"] == "unverified"
    assert rec["not_for_closed_loop_30cm"] is True
    assert rec["calib_id"].endswith("canvas460.yaml")
    # 플래그가 소비자용 바닥 고도로 번역돼 나가는지까지(§5.5)
    assert rec["closed_loop_floor_agl_m"] == 3.0
    # z=0 평면이 지면이 아니라 매트 윗면이라는 사실도 전파돼야 한다(0.105m 라이즈드)
    assert rec["meta"]["plane_reference"] == "mat_top_surface"
    assert rec["meta"]["platform_height_m"] == 0.105


def test_distress_pose_uses_the_measured_3m_spec_not_the_aruco_size(
    tmp_path, monkeypatch, recording_sink_cls
):
    _run_distress(tmp_path, monkeypatch, tier="10m", preset=_DISTRESS_COARSE_PRESET,
                  canvas=460, name="dsize")
    rec = next(r for r in recording_sink_cls.instances[-1].records
               if r["type"] == RECORD_TYPE_TARGET)
    assert rec["meta"]["mat_size_m"] == 3.0


def test_mat_seen_but_pose_unsolvable_reports_its_own_reason(
    tmp_path, monkeypatch, recording_sink_cls
):
    """§5.4 사유 뭉개기 금지: "매트를 아예 못 봤다"와 "매트는 봤는데 4코너가 못 쓴다"는
    현장에서 다른 조치를 요구한다 — `no_target_detection` 하나로 합치면 원인을 못 가린다."""
    real_run = main_mod.Pipeline.run

    def _blank_corners(self, image):
        state = real_run(self, image)
        for d in state.detections:
            d.corners = None
            d.meta.pop("distress_mat", None)
        state.meta["distress_mat_geometry"] = {"tagged": 0, "skipped": 1,
                                               "skip_reasons": ["no_quad_corners"]}
        return state

    monkeypatch.setattr(main_mod.Pipeline, "run", _blank_corners)
    _run_distress(tmp_path, monkeypatch, tier="10m", preset=_DISTRESS_COARSE_PRESET,
                  canvas=460, name="dnogeom")

    rec = next(r for r in recording_sink_cls.instances[-1].records
               if r["type"] == RECORD_TYPE_TARGET)
    assert rec["valid"] is False
    assert rec["reason"] == main_mod.SINK_REASON_MAT_GEOMETRY_UNAVAILABLE
    assert rec["reason"] != main_mod.SINK_REASON_NO_DETECTION
    assert rec["position_frd"] is None, "무효 레코드의 포즈는 0이 아니라 null이어야 한다"


def test_a_preset_without_the_mat_geometry_step_still_says_no_target_detection(
    tmp_path, monkeypatch, recording_sink_cls
):
    """프리셋 주도 선택의 반대편 회귀: `distress_mat_geometry` 스텝이 없는 프리셋(범용
    `rect_detector`를 쓰는 `video.yaml` 등)은 초록구역 경로를 타면 안 된다 — 3m 매트 크기가
    엉뚱한 사각형에 적용되면 완전히 틀린 좌표가 유도로 나간다.

    ⚠️ 픽스처가 **실제로 4코너 검출을 만들어야** 이 회귀에 이빨이 생긴다. `video.yaml`을
    골든 프레임에 돌리면 검출이 0이라 어떤 구현이든 통과해 버린다(파괴검증 D7에서 실측).
    그래서 `distress_coarse.yaml`에서 **`distress_mat_geometry` 스텝만 뺀** 프리셋을 만들어
    쓴다 — 매트는 확실히 4코너로 검출되고, 다른 건 아무것도 다르지 않다."""
    import yaml

    cfg = yaml.safe_load(Path(_DISTRESS_COARSE_PRESET).read_text(encoding="utf-8"))
    removed = cfg["pipeline"].pop("distress_mat_geometry")
    assert removed is not None, "프리셋에 distress_mat_geometry 스텝이 없다 — 배선 자체가 빠졌다"
    stripped = tmp_path / "coarse_without_geometry.yaml"
    stripped.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # 이 프리셋이 매트를 4코너로 실제 검출한다는 것을 먼저 확인(픽스처가 무력하지 않다는 증거)
    frame_path = _golden_frame_copy(tmp_path, "10m")
    probe = main_mod.Pipeline.from_config(str(stripped)).run(cv2.imread(frame_path))
    assert probe.detections and probe.detections[0].corners is not None
    assert len(probe.detections[0].corners) == 4
    assert "distress_mat" not in probe.detections[0].meta

    monkeypatch.setattr(
        sys, "argv",
        ["vision.main", frame_path, "--preset", str(stripped),
         "--calib", _golden_calib(460), "--log-dir", str(tmp_path / "logs"),
         "--log-name", "generic", *_sink_args()],
    )
    main_mod.main()

    rec = next(r for r in recording_sink_cls.instances[-1].records
               if r["type"] == RECORD_TYPE_TARGET)
    assert rec["valid"] is False, "매트 기하 스텝이 없는 프리셋인데 pose가 나갔다"
    assert rec["reason"] == main_mod.SINK_REASON_NO_DETECTION
    assert rec["target_type"] is None


def test_aruco_path_is_untouched_by_the_distress_wiring(tmp_path, monkeypatch):
    """ArUco 회귀: 초록구역 경로를 얹어도 ArUco JSONL `chosen.target_estimate`의 **키 집합과
    값**이 조금도 달라지면 안 된다(특히 새 `meta` 키가 ArUco 쪽에 새어 들어가면 red)."""
    img_path = _write_aruco_image(tmp_path)
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(
        sys, "argv",
        ["vision.main", img_path, "--preset", _VERTIPORT_FINE_PRESET,
         "--log-dir", str(log_dir), "--log-name", "arucoregress"],
    )
    main_mod.main()

    rec = json.loads((log_dir / "arucoregress.jsonl").read_text().splitlines()[-1])
    est = rec["chosen"]["target_estimate"]
    assert set(est) == {
        "position", "orientation", "confidence", "target_type",
        "calib_accuracy", "not_for_closed_loop_30cm", "calib_id",
    }, f"ArUco chosen 형태가 달라졌다: {sorted(est)}"
    assert est["target_type"] == "aruco_23"
    assert est["calib_id"].endswith("nominal.yaml")
