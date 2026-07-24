"""jsonl_view.py — JSONL 블랙박스 뷰어/플롯 최소본 테스트 (vision_plan.md §7.9 항목6).

핵심 검증: vision.main을 실제로 한 번 돌려 디스크에 진짜 JSONL 블랙박스를 만든 뒤,
그 실제 파일을 뷰어에 먹여서 (a) 읽은 행 수가 실제 JSONL의 type=frame 행 수와 정확히
일치하고, (b) 만든 Figure의 각 서브플롯 데이터 포인트 수가 그 행 수(또는 score가 있는
행 수)와 일치하며, (c) PNG 플롯 파일이 실제로 디스크에 생성되는지 확인한다.

§5.1 상태머신(`core/state_machine.py`)이 main.py/replay.py에 배선된 뒤로 `real_jsonl`
픽스처(vision.main 실제 실행)도 매 프레임 실제 `state` 문자열을 남긴다(아래
"state는 이제 실제로 채워진다" 절 참조). 다만 이 픽스처는 짧은 4프레임 합성 시퀀스라
상태가 한 값(`CENTER_DESCEND`)에 머무는데, 거절(rejection)·**다중** state 값이 시간축에
걸쳐 어떻게 그려지는지(안내 텍스트 대신 step 라인)는 실제 `BlackBoxLogger.log_frame`/
`log_rejection`을 직접 호출해(수기로 JSON을 쓰지 않음 — 실제 BlackBoxLogger 산출물) 별도로
검증한다(아래 "거절/다중 state 경계 케이스" 절). main.py는 여전히 거절 사유를 기록하지
않는다(§7.4 거절이유 로깅은 이번 §9 6번 범위 밖).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

import vision.main as main_mod
from vision.tools.jsonl_view import build_figure, load_records, save_figure
from vision.utils.blackbox import BlackBoxLogger


def _write_direct_rect_preset(path: Path) -> str:
    """color_filter -> rect_detector 직결 임시 preset.

    single_frame.yaml의 edge/morphology 조합은 얇은 Canny 엣지를 5x5 open 연산이
    지워버려 이 테스트용 단순 도형에서는 검출이 0이 된다(파이프라인 자체 버그 아님 —
    실제 착륙지점 규모/텍스처를 가정한 튜닝이라 합성 테스트 도형과 안 맞을 뿐).
    검출기 로직(color.py/detector.py)은 건드리지 않고, 뷰어 테스트 전용 임시 preset
    yaml만 새로 만들어 우회한다 — 기존 presets/*.yaml은 수정하지 않는다.
    """
    cfg = {
        "pipeline": {
            "color_filter": {"mode": "gray", "sat_max": 60, "val_min": 80},
            "rect_detector": {
                "min_area": 100, "max_area": 500000,
                "approx_epsilon": 0.04, "min_solidity": 0.5,
            },
        }
    }
    preset_path = path / "rect_direct.yaml"
    preset_path.write_text(yaml.dump(cfg))
    return str(preset_path)


def _write_video_with_squares(path: Path, sizes: list[int | None], frame_size: int = 200) -> str:
    """sizes[i] = i번째 프레임에 그릴 정사각형 한 변 길이. None이면 빈 프레임(검출 0 기대)."""
    video_path = path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (frame_size, frame_size)
    )
    for size in sizes:
        img = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
        if size is not None:
            p = (frame_size - size) // 2
            cv2.rectangle(img, (p, p), (p + size, p + size), (200, 200, 200), -1)
        writer.write(img)
    writer.release()
    return str(video_path)


@pytest.fixture
def real_jsonl(tmp_path, monkeypatch):
    """vision.main을 실제로 실행해 디스크에 진짜 JSONL 블랙박스를 만든다."""
    preset_path = _write_direct_rect_preset(tmp_path)
    video_path = _write_video_with_squares(tmp_path, sizes=[80, 60, None, 100])
    log_dir = tmp_path / "logs"

    monkeypatch.setattr(
        sys, "argv",
        ["vision.main", video_path, "--preset", preset_path,
         "--log-dir", str(log_dir), "--log-name", "viewtest"],
    )
    main_mod.main()

    jsonl_path = log_dir / "viewtest.jsonl"
    assert jsonl_path.exists()
    return jsonl_path


def test_real_run_produces_jsonl_with_some_real_detections(real_jsonl):
    """전제 확인: 합성 파이프라인이 실제로 detection/score를 만들어내는지 (뷰어 검증의 전제)."""
    raw = [json.loads(l) for l in real_jsonl.read_text().splitlines()]
    assert len(raw) == 4
    assert [r["frame_id"] for r in raw] == [0, 1, 2, 3]
    # 프레임 0,1,3 은 사각형이 있어 detection>=1, 프레임 2(빈 프레임)는 0검출이어야 함
    assert len(raw[0]["detections"]) >= 1
    assert len(raw[1]["detections"]) >= 1
    assert len(raw[2]["detections"]) == 0
    assert len(raw[3]["detections"]) >= 1
    assert raw[0]["detections"][0]["confidence"] > 0


def test_load_records_row_count_matches_real_jsonl_line_count(real_jsonl):
    raw_frame_lines = [
        json.loads(l) for l in real_jsonl.read_text().splitlines() if json.loads(l)["type"] == "frame"
    ]
    frame_rows, rejection_ts = load_records(real_jsonl)

    assert len(frame_rows) == len(raw_frame_lines) == 4
    assert rejection_ts == []  # main.py는 아직 거절 사유를 기록하지 않음
    assert [r.frame_id for r in frame_rows] == [row["frame_id"] for row in raw_frame_lines]
    assert [r.latency for r in frame_rows] == [row["latency"] for row in raw_frame_lines]
    # score = chosen 없으면 detections 중 최고 confidence. chosen은 아직 채워지지 않음(TemporalFusion 미사용 preset).
    for row, raw in zip(frame_rows, raw_frame_lines):
        if raw["detections"]:
            expected = max(d["confidence"] for d in raw["detections"])
            assert row.score == expected
        else:
            assert row.score is None
    # 빈 프레임(frame_id=2)만 score가 None이어야 함
    assert [r.score is None for r in frame_rows] == [False, False, True, False]


def test_build_figure_point_counts_match_scored_rows(real_jsonl):
    import math

    frame_rows, rejection_ts = load_records(real_jsonl)
    n_scored = sum(1 for r in frame_rows if r.score is not None)
    assert n_scored == 3  # 프레임 0,1,3 (프레임 2는 검출 0 -> score None)

    fig = build_figure(frame_rows, rejection_ts, x_field="frame_id", title="test")
    ax_score, ax_latency, ax_state = fig.axes

    # 결측은 nan으로 채워 라인이 끊기게 그린다(옆 프레임과 이어붙여 오해를 주지 않기 위해) —
    # 그 결과 라인의 포인트 수는 항상 JSONL의 type=frame 행 수(len(frame_rows))와 정확히 같다.
    score_line = ax_score.lines[0]
    assert len(score_line.get_xdata()) == len(frame_rows) == 4
    assert len(score_line.get_ydata()) == len(frame_rows) == 4
    score_ydata = list(score_line.get_ydata())
    assert math.isnan(score_ydata[2])  # 검출 0인 프레임 2는 구멍(nan)
    assert not any(math.isnan(y) for i, y in enumerate(score_ydata) if i != 2)

    latency_line = ax_latency.lines[0]
    assert len(latency_line.get_xdata()) == len(frame_rows) == 4

    # §9 6번 배선 이후: main.py가 매 프레임 실제 state 문자열을 남긴다 — 이 4프레임 합성
    # 시퀀스는 첫 프레임부터 이미 coarse 후보가 있어(rect_direct.yaml) 곧장 CENTER_DESCEND로
    # 전이되고(ACQUIRE는 후보 0인 프레임에서만 관측됨) 그 뒤로도 계속 그 상태에 머문다
    # (fine_locked 신호가 없는 프리셋이라 LOCK 이상으로는 못 감 — 커밋 게이트 불변식과 일치).
    # 값이 하나뿐이라도 실제 상태 데이터이므로 안내 텍스트가 아니라 step 라인이 그려져야 한다.
    assert [r.state for r in frame_rows] == ["CENTER_DESCEND"] * 4
    assert len(ax_state.lines) == 1
    assert len(ax_state.texts) == 0
    state_line = ax_state.lines[0]
    assert len(state_line.get_xdata()) == len(frame_rows) == 4
    assert ax_state.get_yticklabels()[0].get_text() == "CENTER_DESCEND"

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_save_figure_writes_real_png(tmp_path, real_jsonl):
    frame_rows, rejection_ts = load_records(real_jsonl)
    fig = build_figure(frame_rows, rejection_ts, x_field="ts")
    out_path = tmp_path / "view_out.png"
    result_path = save_figure(fig, out_path)

    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_cli_main_end_to_end_writes_png(tmp_path, real_jsonl):
    from vision.tools.jsonl_view import main as cli_main

    out_path = tmp_path / "cli_view.png"
    rc = cli_main([str(real_jsonl), "--output", str(out_path)])

    assert rc == 0
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_cli_main_errors_on_missing_file(tmp_path):
    from vision.tools.jsonl_view import main as cli_main

    rc = cli_main([str(tmp_path / "nope.jsonl")])
    assert rc == 1


def test_load_records_handles_no_frame_records(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text(json.dumps({"type": "rejection", "ts": 1.0, "frame_id": 0, "reason": "x"}) + "\n")
    frame_rows, rejection_ts = load_records(p)
    assert frame_rows == []
    assert rejection_ts == [1.0]

    from vision.tools.jsonl_view import main as cli_main
    rc = cli_main([str(p)])
    assert rc == 1


def test_x_axis_ts_gap_does_not_mix_scale_with_frame_id(tmp_path):
    """x축 스케일 혼용 회귀: x_field='ts'(기본값)인데 일부 행만 ts가 없으면, 그 행만
    frame_id(정수, 보통 0~N)로 새고 나머지는 ts-t0(경과초, 보통 훨씬 작은 소수)로 찍혀
    x축 스케일이 행마다 섞이면 안 된다. 결측 ts 행은 score/latency와 같은 nan-gap
    철학으로 x좌표도 nan이어야 한다(라인이 끊기되 엉뚱한 스케일로 튀지 않음).
    t0 자체가 None(전체 ts 결측)인 경우는 별개 케이스 — 그때는 전부 frame_id로 폴백해도
    일관되므로 혼용이 아니다(이 테스트는 그 경우를 다루지 않는다)."""
    import math

    bb = BlackBoxLogger(str(tmp_path), name="tsgap", max_queue=100)
    bb.log_frame(frame_id=0, ts=1000.0, detections=[{"bbox": [0, 0, 1, 1], "confidence": 0.5}], latency=0.01)
    bb.log_frame(frame_id=1, ts=None, detections=[{"bbox": [0, 0, 1, 1], "confidence": 0.6}], latency=0.02)
    bb.log_frame(frame_id=2, ts=1000.5, detections=[{"bbox": [0, 0, 1, 1], "confidence": 0.7}], latency=0.03)
    bb.close()

    jsonl_path = tmp_path / "tsgap.jsonl"
    frame_rows, rejection_ts = load_records(jsonl_path)
    assert [r.ts for r in frame_rows] == [1000.0, None, 1000.5]

    fig = build_figure(frame_rows, rejection_ts, x_field="ts")
    ax_score = fig.axes[0]
    xdata = list(ax_score.lines[0].get_xdata())

    assert xdata[0] == pytest.approx(0.0)
    assert math.isnan(xdata[1]), "ts 결측 행이 frame_id(=1)로 새면 옆 프레임들의 경과초 스케일과 섞인다"
    assert xdata[2] == pytest.approx(0.5)

    import matplotlib.pyplot as plt
    plt.close(fig)


# --- 거절/다중 state 경계 케이스: 실제 BlackBoxLogger로 산출한 JSONL(수기 JSON 아님) ---

def test_rejection_lines_and_multi_state_from_real_blackbox_logger(tmp_path):
    bb = BlackBoxLogger(str(tmp_path), name="edgecase", max_queue=100)
    bb.log_frame(frame_id=0, ts=100.0, detections=[{"bbox": [0, 0, 1, 1], "confidence": 0.4}], state="SEARCH", latency=0.01)
    bb.log_rejection(frame_id=0, ts=100.2, reason="shape_mismatch")
    bb.log_frame(frame_id=1, ts=101.0, detections=[{"bbox": [0, 0, 1, 1], "confidence": 0.9}], state="LOCKED", latency=0.02)
    bb.close()

    jsonl_path = tmp_path / "edgecase.jsonl"
    assert jsonl_path.exists()

    frame_rows, rejection_ts = load_records(jsonl_path)
    assert len(frame_rows) == 2
    assert rejection_ts == [100.2]
    assert [r.state for r in frame_rows] == ["SEARCH", "LOCKED"]

    fig = build_figure(frame_rows, rejection_ts, x_field="ts")
    ax_score, ax_latency, ax_state = fig.axes
    # 거절 1건 -> score 서브플롯에 세로선(axvline) 1개
    assert len(ax_score.lines) >= 2  # score 라인 1 + axvline 1
    # state 2종 -> step 라인 존재, 안내 텍스트 없음
    assert len(ax_state.lines) == 1
    assert len(ax_state.texts) == 0
    assert list(ax_state.get_yticklabels()[i].get_text() for i in range(2)) == ["LOCKED", "SEARCH"]

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_no_state_data_still_shows_guidance_text(tmp_path):
    """§9 6번 배선 이후로도 main.py/replay.py가 항상 state를 채운다고 이 파일(jsonl_view.py)이
    가정하면 안 된다 — state가 전혀 없는(예: 상태머신 배선 전 구 JSONL) 입력도 크래시 없이
    안내 텍스트로 우아하게 처리돼야 한다는 회귀를 여기서 직접 만든 JSONL로 계속 지킨다."""
    bb = BlackBoxLogger(str(tmp_path), name="nostate", max_queue=100)
    bb.log_frame(frame_id=0, ts=0.0, detections=[], latency=0.01)  # state 인자 생략 -> None
    bb.log_frame(frame_id=1, ts=0.1, detections=[], latency=0.01)
    bb.close()

    jsonl_path = tmp_path / "nostate.jsonl"
    frame_rows, rejection_ts = load_records(jsonl_path)
    assert all(r.state is None for r in frame_rows)

    fig = build_figure(frame_rows, rejection_ts, x_field="ts")
    ax_score, ax_latency, ax_state = fig.axes
    assert len(ax_state.lines) == 0
    assert len(ax_state.texts) == 1
    assert "no state data" in ax_state.texts[0].get_text()

    import matplotlib.pyplot as plt
    plt.close(fig)
