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
  # 실카메라 라이브 모드 (RPi + picamera2 필요, opt-in — `input`에 특수값 `live`/`live:<camera_num>`)
  python -m vision.main live --preset presets/vertiport_fine.yaml --display none
  # 라이브 모드 + 카메라 번호 지정 + 해상도 override + 영상 기록
  python -m vision.main live:1 --live-resolution 1920x1080 --output results/live.mp4
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2

from vision.core.runner import Pipeline
from vision.core.state_machine import LandingStateMachine, Observation
from vision.core.target import solve_target_pose
from vision.utils.blackbox import BlackBoxLogger
from vision.utils.calibration_loader import CameraCalibration, load_camera_calibration
from vision.utils.frame_source import LiveFrameSource
from vision.utils.image_loader import load_image
from vision.utils.logging import log_provenance_header, setup_dual_sink_logger
from vision.utils.stream import MjpegStreamer
from vision.utils.visualize import save_result, draw_detections


_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
_WINDOW_NAME = "Landing Zone Detector"
# ArUco Phase 4(docs/vision_aruco_branch.md) — 기본 카메라 캘리브레이션. --calib로 override 가능.
_DEFAULT_CALIB_PATH = str(Path(__file__).parent / "calibration" / "cam109-imx708af75" / "nominal.yaml")
# 라이브 모드(LiveFrameSource 배선) — `input` 위치인자 특수값 접두사. import 자체는 picamera2 없이도
# 항상 성공한다(LiveFrameSource.open() 내부 지연 import 덕분, vision/CLAUDE.md "frame_source.py" 절).
_LIVE_INPUT_SPEC = "live"
# --output 지정 시 라이브 모드 VideoWriter가 쓸 fps. VideoReader처럼 소스가 fps를 알려주지 않으므로
# (무한 실시간 스트림) 고정값을 쓴다 — 정밀한 재생속도가 목적이 아니라 결과 확인용 기록이라 충분.
_LIVE_DEFAULT_OUTPUT_FPS = 20.0


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


def _find_white_box_detection(detections):
    """② 조난자 fine(§5.3, `modules/distress_box.py`) — state.detections에서 착륙점이 확정된
    흰 박스 검출을 찾는다. `WhiteBoxDetector`는 확정한 것만 `meta["white_box_detector"]`에
    `landing_point_px`를 실어 남기므로(거절된 매트 후보는 detections에서 아예 제거됨) 첫
    항목을 그대로 쓴다(ArUco와 동일 단일 타겟 전제)."""
    for d in detections:
        wb = d.meta.get("white_box_detector")
        if wb is not None and wb.get("landing_point_px") is not None:
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


def _build_observation(state, frame_id: int, ts: float, agl_m: Optional[float] = None) -> Observation:
    """§9 6번(공통 상태머신) 배선 — 현재 파이프라인 산출물에서 `Observation` 최소 필드만 뽑는다.

    `fine_locked`은 지금 코드베이스에 구현된 두 fine 검증 중 하나라도 있으면 True다:
    ArUco ID 확정 검출(`_find_aruco_detection`) 또는 ② 조난자 fine 흰 박스 확정
    (`_find_white_box_detection`, §5.3). 버티포트/조난자 coarse 전용 프리셋은 아직 fine 검증
    모듈이 없어 항상 False로 degrade한다 — 상태머신은 이 사실을 모르는 채 그대로 안전하게
    ACQUIRE/CENTER_DESCEND에 머문다(타겟 종류 무관 공통 골격 + 커밋 게이트 불변식과 일치).

    `center_error_norm`은 **착륙점 기준**으로 계산한다(§5.3 설계 포인트 — 착륙 목표는 흰 박스가
    아니라 "박스 옆 빈 초록면"이므로 화면중심 정렬 오차도 박스 중심이 아니라 착륙점 기준이어야
    한다). 흰 박스 fine lock이 없으면(ArUco 등 기존 경로) `state.confirmed`/첫 detection의
    bbox 중심으로 폴백 — 기존 동작 그대로 보존. **이름 주의**: `_norm`은 정규화(dx/dy를
    화면 절반폭/절반높이로 나눈 값의 노름, 0~약1.41)라는 뜻이지 픽셀이 아니다 — 예전 이름
    `center_error_px`는 단위를 거짓으로 암시해 정정했다(`core/state_machine.py` 참조).

    `scale_source`(§5.1 "blob 타겟 스케일 융합 규칙")는 흰 박스 blob 확정 시에만 채운다 —
    ArUco는 solvePnP로 자체 스케일이 나오므로 이 융합 규칙 대상이 아니다. AGL(라이다) 유효
    시 "agl", 없으면 (기지 매트 물리크기 기반) "known_size"로 대체 추정."""
    aruco_det = _find_aruco_detection(state.detections)
    white_box_det = _find_white_box_detection(state.detections)
    fine_locked = aruco_det is not None or white_box_det is not None

    center_px = None
    if white_box_det is not None:
        center_px = tuple(white_box_det.meta["white_box_detector"]["landing_point_px"])
    else:
        det = state.confirmed if state.confirmed is not None else (
            state.detections[0] if state.detections else None
        )
        if det is not None:
            center_px = det.center

    center_error_norm = None
    if center_px is not None:
        h, w = state.original.shape[:2]
        cx, cy = center_px
        dx = (cx - w / 2.0) / (w / 2.0)
        dy = (cy - h / 2.0) / (h / 2.0)
        center_error_norm = float((dx ** 2 + dy ** 2) ** 0.5)

    scale_source = None
    if white_box_det is not None:
        scale_source = "agl" if agl_m is not None else "known_size"

    return Observation(
        ts=ts,
        frame_id=frame_id,
        n_candidates=len(state.detections),
        center_error_norm=center_error_norm,
        fine_locked=fine_locked,
        agl_m=agl_m,
        scale_source=scale_source,
    )


def _parse_live_camera_num(input_arg: str) -> Optional[int]:
    """`input` 위치인자가 라이브 모드 스펙(`live` 또는 `live:<camera_num>`)이면 camera_num을
    반환하고, 일반 파일 경로면 None을 반환한다(기존 이미지/영상 분기로 폴백, 옵션 A —
    `input`을 여전히 필수 위치인자로 두어 argparse 구조를 최소로 건드린다)."""
    if input_arg == _LIVE_INPUT_SPEC:
        return 0
    prefix = _LIVE_INPUT_SPEC + ":"
    if input_arg.startswith(prefix):
        suffix = input_arg[len(prefix):]
        try:
            return int(suffix)
        except ValueError as e:
            raise ValueError(
                f"잘못된 라이브 모드 스펙 — {input_arg!r} (예: {_LIVE_INPUT_SPEC} 또는 "
                f"{_LIVE_INPUT_SPEC}:0)"
            ) from e
    return None


def _parse_resolution(value: str) -> Tuple[int, int]:
    """`--live-resolution WxH` 파싱. 잘못된 형식은 argparse가 사용법과 함께 exit(2)하게 둔다."""
    try:
        w_str, h_str = value.lower().split("x")
        return (int(w_str), int(h_str))
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"해상도는 WxH 형식이어야 합니다 (예: 1920x1080): {value!r}"
        ) from e


def _run_image(
    pipeline: Pipeline,
    input_path: Path,
    output: str | None,
    display: str,
    logger,
    blackbox: BlackBoxLogger,
    streamer: MjpegStreamer | None = None,
    calib: Optional[CameraCalibration] = None,
    state_machine: Optional[LandingStateMachine] = None,
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

    decision = None
    if state_machine is not None:
        decision = state_machine.update(_build_observation(state, 0, ts))

    logger.info(
        "image %s: %d detections, confirmed=%s, latency=%.4fs",
        input_path.name, len(state.detections), state.confirmed is not None, latency,
    )
    blackbox.log_frame(
        frame_id=0,
        ts=ts,
        detections=_detections_to_list(state.detections),
        chosen=chosen,
        state=decision.state.value if decision is not None else None,
        command=decision.command if decision is not None else None,
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
    state_machine: Optional[LandingStateMachine] = None,
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

            decision = None
            if state_machine is not None:
                decision = state_machine.update(_build_observation(state, frame_count, ts))

            blackbox.log_frame(
                frame_id=frame_count,
                ts=ts,
                detections=_detections_to_list(state.detections),
                chosen=chosen,
                state=decision.state.value if decision is not None else None,
                command=decision.command if decision is not None else None,
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


def _run_live(
    pipeline: Pipeline,
    camera_num: int,
    resolution: Optional[Tuple[int, int]],
    retries: int,
    retry_delay: float,
    output: str | None,
    display: str,
    logger,
    blackbox: BlackBoxLogger,
    streamer: MjpegStreamer | None = None,
    calib: Optional[CameraCalibration] = None,
    state_machine: Optional[LandingStateMachine] = None,
) -> None:
    """실카메라 라이브 모드 — `_run_video`와 거의 동일한 프레임 루프(무한 이터레이터라는 점만
    다름). 헤드리스(`--display none`)에서는 무한정 도는 게 정상 동작이라 Ctrl+C(KeyboardInterrupt)가
    유일한 종료 수단이다 — `with LiveFrameSource(...)`가 예외 전파 중에도 카메라를 release하고,
    바깥 try/except가 스택트레이스 없이 조용히 종료시킨다(로그만 남김)."""
    live_kwargs: dict = {"camera_num": camera_num, "retries": retries, "retry_delay": retry_delay}
    if resolution is not None:
        live_kwargs["resolution"] = resolution

    writer = None
    frame_count = 0
    try:
        with LiveFrameSource(**live_kwargs) as source:
            for record in source:
                t0 = time.perf_counter()
                state = pipeline.run(record.image)
                latency = time.perf_counter() - t0
                ts = record.ts
                annotated = draw_detections(state.original, state.detections, state.confirmed)

                chosen = _confirmed_to_dict(state.confirmed)
                aruco_chosen = _solve_aruco_chosen(state, calib, frame_count, ts, logger)
                if aruco_chosen is not None:
                    chosen = {**(chosen or {}), **aruco_chosen}

                decision = None
                if state_machine is not None:
                    decision = state_machine.update(
                        _build_observation(state, frame_count, ts, agl_m=record.telemetry.get("alt"))
                    )

                blackbox.log_frame(
                    frame_id=frame_count,
                    ts=ts,
                    detections=_detections_to_list(state.detections),
                    chosen=chosen,
                    state=decision.state.value if decision is not None else None,
                    command=decision.command if decision is not None else None,
                    latency=latency,
                )
                logger.debug(
                    "live frame %d: %d detections, confirmed=%s, latency=%.4fs",
                    frame_count, len(state.detections), state.confirmed is not None, latency,
                )
                frame_count += 1

                if streamer is not None:
                    streamer.push_frame(annotated)  # 비차단(§7.9 비침습 전제)

                if output:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output, fourcc, _LIVE_DEFAULT_OUTPUT_FPS, (w, h))
                    writer.write(annotated)

                if display == "window" and not _show_window(annotated, wait=1):
                    break
    except KeyboardInterrupt:
        logger.info(
            "라이브 모드 Ctrl+C로 종료 요청 — %d 프레임 처리 후 정상 종료", frame_count,
        )
    finally:
        if writer:
            writer.release()
            print(f"Saved: {output}  ({frame_count} frames)")
        logger.info("live camera_num=%s 종료: %d 프레임 처리", camera_num, frame_count)
        if display == "window":
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Landing zone object detector")
    parser.add_argument(
        "input",
        help="Input image or video file path, or 'live'/'live:<camera_num>' for a real-time "
             "camera stream (opt-in — requires picamera2, RPi only, see --live-* flags below)",
    )
    parser.add_argument(
        "--preset",
        default=str(Path(__file__).parent / "presets" / "single_frame.yaml"),
        help="Pipeline preset yaml (default: presets/single_frame.yaml)",
    )
    parser.add_argument("--output", default=None, help="Output file path (optional)")
    parser.add_argument(
        "--live-resolution",
        type=_parse_resolution,
        default=None,
        metavar="WxH",
        help="라이브 모드(`input`이 live/live:N) 해상도. 기본: LiveFrameSource 기본값 "
             "(4608x2592, nominal.yaml image_size와 일치 — solvePnP 캘리브레이션 정합 유지).",
    )
    parser.add_argument(
        "--live-retries", type=int, default=3,
        help="라이브 모드 카메라 연결 재시도 횟수 (기본 3)",
    )
    parser.add_argument(
        "--live-retry-delay", type=float, default=1.0,
        help="라이브 모드 카메라 연결 재시도 간격(초, 기본 1.0)",
    )
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

    try:
        live_camera_num = _parse_live_camera_num(args.input)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    input_path: Optional[Path] = None
    if live_camera_num is None:
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

    # §9 6번 — 공통 상태머신. 실행 전체에 걸쳐 인스턴스 하나 재사용(프레임 간 상태 누적이
    # 상태머신의 본질이므로 매 프레임 새로 만들면 안 됨). 단일 이미지 경로에서도 그대로
    # 통과시킨다 — 관측 1개짜리 시퀀스로 취급해도 크래시 없이 동작한다.
    state_machine = LandingStateMachine()

    streamer = None
    try:
        if args.display == "stream":
            streamer = MjpegStreamer(host=args.stream_host, port=args.stream_port)
            streamer.start()
            logger.info("라이브 스트림 시작: %s", streamer.url)
            print(f"라이브 스트림: {streamer.url}")

        if live_camera_num is not None:
            _run_live(
                pipeline, live_camera_num, args.live_resolution, args.live_retries,
                args.live_retry_delay, args.output, args.display, logger, blackbox, streamer, calib,
                state_machine,
            )
        elif input_path.suffix.lower() in _VIDEO_SUFFIXES:
            _run_video(
                pipeline, input_path, args.output, args.display, logger, blackbox, streamer, calib,
                state_machine,
            )
        else:
            _run_image(
                pipeline, input_path, args.output, args.display, logger, blackbox, streamer, calib,
                state_machine,
            )
    finally:
        blackbox.close()
        if streamer is not None:
            streamer.stop()


if __name__ == "__main__":
    main()
