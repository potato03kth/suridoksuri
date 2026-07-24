"""main.py CLI 헤드리스 안전성 회귀 테스트 + 관측성(로거/블랙박스) 연결 테스트.

불변식: 기본값 --display none 은 어떤 GUI 함수(cv2.imshow 등)도 호출하지 않는다.
드론(디스플레이 없음)에서의 크래시를 방지하는 계약이므로 절대 깨지면 안 된다.

로그 출력은 실제 리포지토리 vision/results/를 더럽히지 않도록 모든 테스트에서
--log-dir 을 tmp_path 하위로 명시한다.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import vision.main as main_mod
from vision.utils.frame_source import FrameRecord

_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_VERTIPORT_FINE_PRESET = str(Path(main_mod.__file__).parent / "presets" / "vertiport_fine.yaml")


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
