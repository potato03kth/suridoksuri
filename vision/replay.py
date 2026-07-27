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
from typing import Optional

import cv2
import numpy as np

from vision.core.runner import Pipeline
from vision.core.state_machine import LandingStateMachine, Observation
from vision.core.target import marker_object_points, solve_target_pose
from vision.utils.blackbox import BlackBoxLogger
from vision.utils.calibration_loader import CameraCalibration, load_camera_calibration
from vision.utils.frame_source import open_dir_or_bag
from vision.utils.logging import log_provenance_header, setup_dual_sink_logger
from vision.utils.stream import MjpegStreamer
from vision.utils.visualize import draw_detections

_WINDOW_NAME = "Landing Zone Detector — Replay"
# ArUco Phase 4(docs/vision_aruco_branch.md) — 기본 카메라 캘리브레이션. main.py와 동일 기본값
# (main.py와 헬퍼 상호 import 안 함 원칙 — vision/CLAUDE.md import 규칙, 각자 얇게 중복 허용).
_DEFAULT_CALIB_PATH = str(Path(__file__).parent / "calibration" / "cam109-imx708af75" / "nominal.yaml")


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


def _find_aruco_detection(detections):
    """ArUco Phase 4 — state.detections에서 코너를 가진 확정 ArUco 검출을 찾는다(main.py와
    동일 로직, 상호 import 안 함 원칙에 따라 얇게 중복— vision/CLAUDE.md import 규칙)."""
    for d in detections:
        if d.corners is not None and d.meta.get("aruco_id") is not None:
            return d
    return None


def _find_white_box_detection(detections):
    """② 조난자 fine(§5.3, `modules/distress_box.py`) — state.detections에서 착륙점이 확정된
    흰 박스 검출을 찾는다(main.py와 동일 로직, 상호 import 안 함 원칙에 따라 얇게 중복)."""
    for d in detections:
        wb = d.meta.get("white_box_detector")
        if wb is not None and wb.get("landing_point_px") is not None:
            return d
    return None


def _find_distress_mat_detection(detections):
    """② 조난자 구역(초록 매트) — `modules/distress_mat.py::DistressMatGeometry`가 pose 산출용
    기하를 태깅해 둔 검출을 찾는다(main.py와 동일 로직, 얇게 중복)."""
    for d in detections:
        if d.meta.get("distress_mat") is not None:
            return d
    return None


def _target_estimate_to_dict(estimate) -> dict:
    d = {
        "position": list(estimate.position),
        "orientation": list(estimate.orientation),
        "confidence": estimate.confidence,
        "target_type": estimate.target_type,
        "calib_accuracy": estimate.calib_accuracy,
        "not_for_closed_loop_30cm": estimate.not_for_closed_loop_30cm,
        "calib_id": estimate.calib_id,
    }
    # ArUco 경로는 meta를 안 채우므로 이 키가 생기지 않는다 — 기존 JSONL `chosen` 형태 무변경.
    if estimate.meta:
        d["meta"] = dict(estimate.meta)
    return d


def _solve_aruco_chosen(
    state, calib: Optional[CameraCalibration], frame_id: int, ts: float, logger,
) -> Optional[dict]:
    """확정 ArUco 검출 + 로드된 calib이 둘 다 있을 때만 solvePnP를 시도한다. 마커가 없거나
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


def _solve_distress_mat_chosen(
    state, calib: Optional[CameraCalibration], frame_id: int, ts: float, logger,
) -> Optional[dict]:
    """초록 매트 4코너 + 알려진 실측 크기(3.0m) -> solvePnP -> ArUco와 같은 형식의
    `TargetEstimate`(main.py `_solve_distress_mat_estimate`와 동일 로직, 얇게 중복).

    §7.5가 "같은 파이프라인으로 오프라인 재생"을 회귀검증의 최대 레버로 못박고 있어, 재생
    경로에도 붙어 있어야 초록구역 pose 계약을 책상에서 회귀로 잡을 수 있다.
    **position은 매트 중심이 아니라 착륙점**(§5.3 "박스 옆 빈 초록면")이다."""
    if calib is None:
        return None
    det = _find_distress_mat_detection(state.detections)
    if det is None:
        return None
    mat = det.meta["distress_mat"]
    coarse = mat["landing_point_source"] == "mat_center"
    try:
        estimate = solve_target_pose(
            image_points=np.asarray(mat["corners_px"], dtype=np.float64),
            camera_matrix=calib.camera_matrix,
            dist_coeffs=calib.dist_coeffs,
            object_points=marker_object_points(mat["size_m"]),
            position_at_pixel=None if coarse else tuple(mat["landing_point_px"]),
            target_type="distress_mat_center" if coarse else "distress_landing_point",
            frame_id=frame_id,
            timestamp=ts,
            confidence=det.confidence,
            calib_accuracy=calib.accuracy,
            not_for_closed_loop_30cm=calib.not_for_closed_loop_30cm,
            calib_id=calib.calib_id,
            meta={
                "mat_size_m": mat["size_m"],
                "landing_point_px": mat["landing_point_px"],
                "landing_point_source": mat["landing_point_source"],
                "plane_reference": mat["plane_reference"],
                "platform_height_m": mat["platform_height_m"],
                "frame_size_px": [int(state.original.shape[1]), int(state.original.shape[0])],
                "calib_image_size_px": [int(calib.image_size[0]), int(calib.image_size[1])],
            },
        )
    except ValueError as e:
        logger.warning("frame %d: 초록 매트 pose 산출 실패 — TargetEstimate 생략: %s", frame_id, e)
        return None
    return {"target_estimate": _target_estimate_to_dict(estimate)}


def _solve_target_chosen(
    state, calib: Optional[CameraCalibration], frame_id: int, ts: float, logger,
) -> Optional[dict]:
    """어느 pose 산출기를 쓸지는 **프리셋이 남긴 meta**가 정한다(main.py와 동일 판단).
    🔴 ArUco 검출이 있으면 그 결과가 그대로 나가고, 초록 매트는 ArUco가 없을 때만 시도된다."""
    aruco_chosen = _solve_aruco_chosen(state, calib, frame_id, ts, logger)
    if aruco_chosen is not None or _find_aruco_detection(state.detections) is not None:
        return aruco_chosen
    return _solve_distress_mat_chosen(state, calib, frame_id, ts, logger)


def _build_observation(state, frame_id: int, ts: float, agl_m: Optional[float] = None) -> Observation:
    """§9 6번(공통 상태머신) 배선 — main.py와 동일 로직(상호 import 안 함 원칙에 따라 얇게
    중복, vision/CLAUDE.md import 규칙).

    `fine_locked`은 지금 구현된 두 fine 검증 중 하나라도 있으면 True다: ArUco ID 확정 검출
    또는 ② 조난자 fine 흰 박스 확정(§5.3). coarse 전용 프리셋은 둘 다 없어 항상 False로 degrade.

    `center_error_norm`은 착륙점 기준(§5.3 "박스 옆 빈 초록면" — 박스 중심이 아님)으로 계산하고,
    흰 박스 lock이 없으면 기존처럼 confirmed/첫 detection bbox 중심으로 폴백한다. **이름 주의**:
    `_norm`은 정규화(0~약1.41)라는 뜻이지 픽셀이 아니다 — 예전 이름 `center_error_px`는 단위를
    거짓으로 암시해 정정했다(`core/state_machine.py` 참조).

    `scale_source`(§5.1 blob 스케일 융합 규칙)는 흰 박스 blob 확정 시에만 채운다 — AGL 유효
    시 "agl", 없으면 "known_size". ArUco는 solvePnP 자체 스케일이라 대상 아님."""
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


def run_replay(
    input_path: str,
    preset: str,
    display: str,
    output: str | None,
    log_dir: str,
    log_name: str = "replay",
    stream_host: str = "0.0.0.0",
    stream_port: int = 8080,
    calib_path: str = _DEFAULT_CALIB_PATH,
) -> int:
    """실제 재생 루프. main()에서 분리해 프로그램적으로도 호출/테스트 가능하게 한다.

    display="stream" (§7.9 항목5, opt-in): 재생 중인(annotated) 프레임을 MjpegStreamer로
    실시간 흘려보낸다 — 비차단(§7.9 비침습 전제), 켤 때만 오버헤드 발생.

    calib_path(ArUco Phase 4): 카메라 캘리브레이션 yaml, 루프 시작 전 1회만 로드해 재사용한다
    (§7.1 config 레이어드, 매 프레임 파일 I/O 금지). 없어도(FileNotFoundError) 재생 자체는
    막지 않고 경고만 남긴다 — ArUco와 무관한 프리셋 재생까지 이 로드 실패로 막으면 안 된다.
    """
    source = open_dir_or_bag(input_path)
    pipeline = Pipeline.from_config(preset)

    logger = setup_dual_sink_logger(log_name, log_dir)
    log_provenance_header(logger, {"preset": preset, "input": input_path})
    blackbox = BlackBoxLogger(log_dir, name=log_name)

    calib: Optional[CameraCalibration] = None
    try:
        calib = load_camera_calibration(calib_path)
    except FileNotFoundError as e:
        logger.warning("캘리브레이션 로드 실패(%s) — ArUco TargetEstimate 계산 생략: %s", calib_path, e)

    # §9 6번 — 공통 상태머신. 재생 전체에 걸쳐 인스턴스 하나 재사용(결정론적 재생, §7.5 —
    # 매 프레임 새로 만들면 상태 누적 자체가 사라진다).
    state_machine = LandingStateMachine()

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

                chosen = _confirmed_to_dict(state.confirmed)
                aruco_chosen = _solve_target_chosen(state, calib, record.frame_id, record.ts, logger)
                if aruco_chosen is not None:
                    chosen = {**(chosen or {}), **aruco_chosen}

                # telemetry.jsonl에 alt(라이다 AGL)가 있으면 쓰고, 없으면 None으로 우아하게
                # degrade한다(실기체 telemetry 아직 없음 — AGL 없이도 상태머신은 동작해야 함).
                decision = state_machine.update(
                    _build_observation(
                        state, record.frame_id, record.ts, agl_m=record.telemetry.get("alt")
                    )
                )

                blackbox.log_frame(
                    frame_id=record.frame_id,
                    ts=record.ts,
                    detections=_detections_to_list(state.detections),
                    chosen=chosen,
                    state=decision.state.value,
                    command=decision.command,
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
    parser.add_argument(
        "--calib",
        default=_DEFAULT_CALIB_PATH,
        help="ArUco TargetEstimate 계산용 카메라 캘리브레이션 yaml (기본: nominal.yaml, "
             "docs/vision_aruco_branch.md Phase 4). 프리셋에 aruco_detector가 없으면 로드만 "
             "되고 쓰이지 않는다.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input not found — {input_path}", file=sys.stderr)
        sys.exit(1)

    frame_count = run_replay(
        str(input_path), args.preset, args.display, args.output, args.log_dir, args.log_name,
        stream_host=args.stream_host, stream_port=args.stream_port, calib_path=args.calib,
    )
    print(f"Replayed {frame_count} frames from {input_path}")


if __name__ == "__main__":
    main()
