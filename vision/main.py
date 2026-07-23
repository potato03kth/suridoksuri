"""
Landing zone detector — CLI 진입점.

--display 모드 (헤드리스 안전 — vision_plan.md §7.9 항목 1):
  none    창/스트림 없음. 파이프라인 + 로그(+ --output 파일)만. **드론 기본값**.
  window  cv2 GUI 창 표시. 디스플레이 있는 데스크톱 전용.
  file    창 없음, --output 필수 — 주석(annotated) 프레임을 파일로 기록.
  stream  라이브 저해상 MJPEG-over-HTTP 스트림 (§7.9 항목 5). opt-in — 켤 때만 오버헤드 발생.
          --stream-host/--stream-port로 바인딩 주소 지정(기본 0.0.0.0:8080).
          http://<host>:<port>/stream 으로 브라우저 접속.

⚠️ 드론은 디스플레이가 없다. GUI 호출(imshow/waitKey)은 반드시 --display window
   뒤로만 실행된다. 기본값 none 은 어떤 GUI 함수도 호출하지 않는다(헤드리스 크래시 방지).

관측성 (vision_plan.md §7.4/§7.9, 항상 on): 실행할 때마다 이중싱크 사람로그(.log)와
프레임별 JSONL 블랙박스를 --log-dir(기본 results/logs)에 남긴다. --log-name으로 파일
basename 지정.

사용 예:
  # 데스크톱에서 영상 보며 디버깅
  python -m vision.main flight.mp4 --preset presets/video.yaml --display window
  # 드론/헤드리스 배치 (기본값 none — 창 없음, 파일만)
  python -m vision.main flight.mp4 --preset presets/video.yaml --output results/out.mp4
  # 정지 이미지
  python -m vision.main image.jpg --preset presets/low_light.yaml --output results/out.jpg
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from vision.core.runner import Pipeline
from vision.core.target import solve_target_pose
from vision.utils.blackbox import BlackBoxLogger
from vision.utils.calibration_loader import CameraCalibration, load_camera_calibration
from vision.utils.image_loader import load_image
from vision.utils.logging import log_provenance_header, setup_dual_sink_logger
from vision.utils.stream import MjpegStreamer
from vision.utils.visualize import save_result, draw_detections


_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
_WINDOW_NAME = "Landing Zone Detector"
# ArUco Phase 4(docs/vision_aruco_branch.md) — 기본 카메라 캘리브레이션. --calib로 override 가능.
_DEFAULT_CALIB_PATH = str(Path(__file__).parent / "calibration" / "cam109-imx708af75" / "nominal.yaml")


def _show_window(annotated, *, wait: int) -> bool:
    """cv2 GUI 창에 한 프레임 표시.

    헤드리스 OpenCV(디스플레이 없음)에서는 imshow가 예외를 던지므로 방어한다.
    반환: 계속 진행하면 True, 사용자가 'q'로 종료 요청하면 False.
    """
    try:
        cv2.imshow(_WINDOW_NAME, annotated)
        return (cv2.waitKey(wait) & 0xFF) != ord("q")
    except cv2.error as e:  # 헤드리스 빌드/디스플레이 부재
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


def _find_aruco_detection(detections):
    """ArUco Phase 4 — state.detections에서 코너를 가진 확정 ArUco 검출을 찾는다.
    modules/aruco.py는 ID 화이트리스트를 통과한 것만 detections에 넣으므로(meta["aruco_id"]
    존재) 첫 항목을 그대로 쓴다(단일 타겟 전제, §1 확정 — 다중 타겟 대비 금지)."""
    for d in detections:
        if d.corners is not None and d.meta.get("aruco_id") is not None:
            return d
    return None


def _target_estimate_to_dict(estimate) -> dict:
    return {
        "position": list(estimate.position),
        "orientation": list(estimate.orientation),
        "confidence": estimate.confidence,
        "target_type": estimate.target_type,
        "calib_accuracy": estimate.calib_accuracy,
        "not_for_closed_loop_30cm": estimate.not_for_closed_loop_30cm,
        "calib_id": estimate.calib_id,
    }


def _solve_aruco_chosen(
    state, calib: Optional[CameraCalibration], frame_id: int, ts: float, logger,
) -> Optional[dict]:
    """확정 ArUco 검출 + 로드된 calib이 둘 다 있을 때만 solvePnP를 시도한다(§Phase4 전제 —
    코너 없으면 solvePnP 자체가 실패하므로 매 프레임 무조건 시도하지 않음). 마커가 없거나
    calib 미로드거나 solvePnP가 수렴하지 않아도 크래시 없이 None을 돌려준다."""
    if calib is None:
        return None
    det = _find_aruco_detection(state.detections)
    if det is None:
        return None
    try:
        estimate = solve_target_pose(
            image_points=det.corners,
            camera_matrix=calib.camera_matrix,
            dist_coeffs=calib.dist_coeffs,
            target_type=f"aruco_{det.meta['aruco_id']}",
            frame_id=frame_id,
            timestamp=ts,
            confidence=det.confidence,
            calib_accuracy=calib.accuracy,
            not_for_closed_loop_30cm=calib.not_for_closed_loop_30cm,
            calib_id=calib.calib_id,
        )
    except ValueError as e:
        logger.warning("frame %d: ArUco solvePnP 실패 — TargetEstimate 생략: %s", frame_id, e)
        return None
    return {"target_estimate": _target_estimate_to_dict(estimate)}


def _run_image(
    pipeline: Pipeline,
    input_path: Path,
    output: str | None,
    display: str,
    logger,
    blackbox: BlackBoxLogger,
    streamer: MjpegStreamer | None = None,
    calib: Optional[CameraCalibration] = None,
) -> None:
    image = load_image(str(input_path))
    t0 = time.perf_counter()
    state = pipeline.run(image)
    latency = time.perf_counter() - t0
    ts = time.time()

    print(f"Detections: {len(state.detections)}")
    for i, d in enumerate(state.detections):
        print(f"  [{i}] bbox={d.bbox}  confidence={d.confidence:.3f}")
    if state.confirmed:
        print(f"Confirmed: bbox={state.confirmed.bbox}")

    chosen = _confirmed_to_dict(state.confirmed)
    aruco_chosen = _solve_aruco_chosen(state, calib, 0, ts, logger)
    if aruco_chosen is not None:
        chosen = {**(chosen or {}), **aruco_chosen}

    logger.info(
        "image %s: %d detections, confirmed=%s, latency=%.4fs",
        input_path.name, len(state.detections), state.confirmed is not None, latency,
    )
    blackbox.log_frame(
        frame_id=0,
        ts=ts,
        detections=_detections_to_list(state.detections),
        chosen=chosen,
        latency=latency,
    )

    if output:
        save_result(state, output)
        print(f"Saved: {output}")

    if display in ("window", "stream"):
        annotated = draw_detections(state.original, state.detections, state.confirmed)
        if streamer is not None:
            streamer.push_frame(annotated)  # 비차단(§7.9 비침습 전제)
        if display == "window":
            _show_window(annotated, wait=0)  # 키 입력까지 대기
            cv2.destroyAllWindows()


def _run_video(
    pipeline: Pipeline,
    input_path: Path,
    output: str | None,
    display: str,
    logger,
    blackbox: BlackBoxLogger,
    streamer: MjpegStreamer | None = None,
    calib: Optional[CameraCalibration] = None,
) -> None:
    from vision.utils.video_reader import VideoReader

    writer = None
    frame_count = 0
    with VideoReader(str(input_path)) as reader:
        for frame in reader:
            t0 = time.perf_counter()
            state = pipeline.run(frame)
            latency = time.perf_counter() - t0
            ts = time.time()
            annotated = draw_detections(state.original, state.detections, state.confirmed)

            chosen = _confirmed_to_dict(state.confirmed)
            aruco_chosen = _solve_aruco_chosen(state, calib, frame_count, ts, logger)
            if aruco_chosen is not None:
                chosen = {**(chosen or {}), **aruco_chosen}

            blackbox.log_frame(
                frame_id=frame_count,
                ts=ts,
                detections=_detections_to_list(state.detections),
                chosen=chosen,
                latency=latency,
            )
            logger.debug(
                "frame %d: %d detections, confirmed=%s, latency=%.4fs",
                frame_count, len(state.detections), state.confirmed is not None, latency,
            )
            frame_count += 1

            if streamer is not None:
                streamer.push_frame(annotated)  # 비차단(§7.9 비침습 전제) — 파이프라인 루프를 지연시키지 않음

            if output:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output, fourcc, reader.fps or 30, (w, h))
                writer.write(annotated)

            if display == "window" and not _show_window(annotated, wait=1):
                break

    if writer:
        writer.release()
        print(f"Saved: {output}  ({frame_count} frames)")
    logger.info("video %s 종료: %d 프레임 처리", input_path.name, frame_count)
    if display == "window":
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Landing zone object detector")
    parser.add_argument("input", help="Input image or video file path")
    parser.add_argument(
        "--preset",
        default=str(Path(__file__).parent / "presets" / "single_frame.yaml"),
        help="Pipeline preset yaml (default: presets/single_frame.yaml)",
    )
    parser.add_argument("--output", default=None, help="Output file path (optional)")
    parser.add_argument(
        "--display",
        choices=["none", "window", "file", "stream"],
        default="none",
        help="라이브 뷰 모드. none=헤드리스 안전(기본) · window=데스크톱 GUI · "
             "file=--output 필수 · stream=라이브 MJPEG-over-HTTP(§7.9 항목5, opt-in)",
    )
    parser.add_argument(
        "--stream-host", default="0.0.0.0", help="--display stream 바인딩 주소 (기본 0.0.0.0)"
    )
    parser.add_argument(
        "--stream-port", type=int, default=8080, help="--display stream 포트 (기본 8080)"
    )
    parser.add_argument(
        "--log-dir",
        default=str(Path(__file__).parent / "results" / "logs"),
        help="이중싱크 사람로그(.log)+JSONL 블랙박스 출력 디렉터리 (§7.4/§7.9)",
    )
    parser.add_argument("--log-name", default="vision", help="로그 파일 basename")
    parser.add_argument(
        "--calib",
        default=_DEFAULT_CALIB_PATH,
        help="ArUco TargetEstimate 계산용 카메라 캘리브레이션 yaml (기본: nominal.yaml, "
             "docs/vision_aruco_branch.md Phase 4). 프리셋에 aruco_detector가 없으면 로드만 "
             "되고 쓰이지 않는다.",
    )
    args = parser.parse_args()

    if args.display == "file" and not args.output:
        print("Error: --display file 은 --output 경로가 필요합니다.", file=sys.stderr)
        sys.exit(2)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found — {input_path}", file=sys.stderr)
        sys.exit(1)

    pipeline = Pipeline.from_config(args.preset)

    logger = setup_dual_sink_logger(args.log_name, args.log_dir)
    log_provenance_header(
        logger,
        {"preset": args.preset, "input": args.input, "display": args.display, "output": args.output},
    )
    blackbox = BlackBoxLogger(args.log_dir, name=args.log_name)

    # ArUco Phase 4 — calib은 매 프레임이 아니라 여기서 1회만 로드해 재사용한다(§7.1 config
    # 레이어드, 파일 I/O 낭비 방지). 프리셋이 aruco_detector를 안 쓰면 로드 결과는 그냥 안 쓰인다.
    # 파일이 없어도(nominal.yaml 미준비 환경 등) 전체 파이프라인을 죽이지 않고 경고만 남긴다 —
    # ArUco와 무관한 프리셋(버티포트/조난자 coarse 등) 실행까지 이 로드 실패로 막으면 안 된다.
    calib: Optional[CameraCalibration] = None
    try:
        calib = load_camera_calibration(args.calib)
    except FileNotFoundError as e:
        logger.warning("캘리브레이션 로드 실패(%s) — ArUco TargetEstimate 계산 생략: %s", args.calib, e)

    streamer = None
    try:
        if args.display == "stream":
            streamer = MjpegStreamer(host=args.stream_host, port=args.stream_port)
            streamer.start()
            logger.info("라이브 스트림 시작: %s", streamer.url)
            print(f"라이브 스트림: {streamer.url}")

        if input_path.suffix.lower() in _VIDEO_SUFFIXES:
            _run_video(pipeline, input_path, args.output, args.display, logger, blackbox, streamer, calib)
        else:
            _run_image(pipeline, input_path, args.output, args.display, logger, blackbox, streamer, calib)
    finally:
        blackbox.close()
        if streamer is not None:
            streamer.stop()


if __name__ == "__main__":
    main()
