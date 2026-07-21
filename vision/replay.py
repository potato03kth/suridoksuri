"""
오프라인 재생 CLI (vision_plan.md §7.5 기록·재생, §7.9 (a) "재생 오버레이 뷰어" — 데스크 주력 디버깅 경로).

벤치/비행 런에서 녹화한 원본 프레임(+텔레메트리)을 **동일 파이프라인**으로 책상에서 재생한다.
결정론적: 같은 입력+config → 같은 출력(§7.5). 입력이 디렉터리면 DirFrameSource(프레임 파일들
+ 선택적 telemetry.jsonl), 파일이면 BagFrameSource(비디오 + 선택적 사이드카 telemetry.jsonl)로
자동 판별한다.

사용 예:
  # 녹화 폴더 재생, 창으로 확인
  python -m vision.replay recordings/flight02 --preset vision/presets/video.yaml --display window
  # bag(비디오 파일) 재생, 결과 mp4 + JSONL 블랙박스 기록
  python -m vision.replay recordings/flight02.mp4 --preset vision/presets/video.yaml \
      --output vision/results/replay.mp4 --log-dir vision/results/replay_logs
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

from vision.core.runner import Pipeline
from vision.utils.blackbox import BlackBoxLogger
from vision.utils.frame_source import open_dir_or_bag
from vision.utils.logging import log_provenance_header, setup_dual_sink_logger
from vision.utils.stream import MjpegStreamer
from vision.utils.visualize import draw_detections

_WINDOW_NAME = "Landing Zone Detector — Replay"


def _show_window(annotated, *, wait: int) -> bool:
    """§7.9 (a) 재생 오버레이 뷰어. 헤드리스 OpenCV에서는 imshow가 예외를 던지므로 방어한다."""
    try:
        cv2.imshow(_WINDOW_NAME, annotated)
        return (cv2.waitKey(wait) & 0xFF) != ord("q")
    except cv2.error as e:
        print(
            f"Error: --display window 사용 불가 (디스플레이 없음/headless OpenCV): {e}\n"
            "       헤드리스 환경에서는 --display none (기본) 또는 --output 을 쓰세요.",
            file=sys.stderr,
        )
        sys.exit(2)


def _confirmed_to_dict(confirmed) -> dict | None:
    if confirmed is None:
        return None
    return {"bbox": list(confirmed.bbox), "confidence": confirmed.confidence}


def _detections_to_list(detections) -> list[dict]:
    return [{"bbox": list(d.bbox), "confidence": d.confidence} for d in detections]


def run_replay(
    input_path: str,
    preset: str,
    display: str,
    output: str | None,
    log_dir: str,
    log_name: str = "replay",
    stream_host: str = "0.0.0.0",
    stream_port: int = 8080,
) -> int:
    """실제 재생 루프. main()에서 분리해 프로그램적으로도 호출/테스트 가능하게 한다.

    display="stream" (§7.9 항목5, opt-in): 재생 중인(annotated) 프레임을 MjpegStreamer로
    실시간 흘려보낸다 — 비차단(§7.9 비침습 전제), 켤 때만 오버헤드 발생.
    """
    source = open_dir_or_bag(input_path)
    pipeline = Pipeline.from_config(preset)

    logger = setup_dual_sink_logger(log_name, log_dir)
    log_provenance_header(logger, {"preset": preset, "input": input_path})
    blackbox = BlackBoxLogger(log_dir, name=log_name)

    streamer = None
    writer = None
    frame_count = 0
    try:
        if display == "stream":
            streamer = MjpegStreamer(host=stream_host, port=stream_port)
            streamer.start()
            logger.info("라이브 스트림 시작: %s", streamer.url)
            print(f"라이브 스트림: {streamer.url}")

        with source:
            for record in source:
                t0 = time.perf_counter()
                state = pipeline.run(record.image)
                latency = time.perf_counter() - t0

                annotated = draw_detections(state.original, state.detections, state.confirmed)
                frame_count += 1

                blackbox.log_frame(
                    frame_id=record.frame_id,
                    ts=record.ts,
                    detections=_detections_to_list(state.detections),
                    chosen=_confirmed_to_dict(state.confirmed),
                    alt=record.telemetry.get("alt"),
                    attitude=record.telemetry.get("attitude"),
                    latency=latency,
                )
                logger.debug(
                    "frame %d: %d detections, confirmed=%s, latency=%.4fs",
                    record.frame_id, len(state.detections), state.confirmed is not None, latency,
                )

                if streamer is not None:
                    streamer.push_frame(annotated)  # 비차단 — 재생 루프를 지연시키지 않음

                if output:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output, fourcc, 30, (w, h))
                    writer.write(annotated)

                if display == "window" and not _show_window(annotated, wait=1):
                    break
    finally:
        if writer:
            writer.release()
        if display == "window":
            cv2.destroyAllWindows()
        blackbox.close()
        if streamer is not None:
            streamer.stop()

    logger.info("replay 종료: %d 프레임 처리", frame_count)
    if writer:
        # writer는 프레임 루프 안에서 첫 프레임 처리 시에만 생성된다(main.py:165-167과 동일 게이팅) —
        # output이 주어졌어도 0프레임 재생이면 writer가 끝내 안 만들어지므로 output만으로는 게이팅 안 함.
        logger.info("저장: %s", output)
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Landing zone detector — offline replay (vision_plan.md §7.5/§7.9)")
    parser.add_argument("input", help="녹화 폴더(DirFrameSource) 또는 bag 파일(BagFrameSource, 비디오)")
    parser.add_argument(
        "--preset",
        default=str(Path(__file__).parent / "presets" / "video.yaml"),
        help="Pipeline preset yaml (default: presets/video.yaml)",
    )
    parser.add_argument("--output", default=None, help="주석 처리된 결과 mp4 저장 경로 (선택)")
    parser.add_argument(
        "--display",
        choices=["none", "window", "stream"],
        default="none",
        help="none=헤드리스(기본) · window=데스크톱 GUI 재생 뷰어 · "
             "stream=라이브 MJPEG-over-HTTP(§7.9 항목5, opt-in)",
    )
    parser.add_argument(
        "--stream-host", default="0.0.0.0", help="--display stream 바인딩 주소 (기본 0.0.0.0)"
    )
    parser.add_argument(
        "--stream-port", type=int, default=8080, help="--display stream 포트 (기본 8080)"
    )
    parser.add_argument(
        "--log-dir",
        default=str(Path(__file__).parent / "results" / "replay_logs"),
        help="사람로그(.log)+JSONL 블랙박스 출력 디렉터리",
    )
    parser.add_argument("--log-name", default="replay", help="로그 파일 basename")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input not found — {input_path}", file=sys.stderr)
        sys.exit(1)

    frame_count = run_replay(
        str(input_path), args.preset, args.display, args.output, args.log_dir, args.log_name,
        stream_host=args.stream_host, stream_port=args.stream_port,
    )
    print(f"Replayed {frame_count} frames from {input_path}")


if __name__ == "__main__":
    main()
