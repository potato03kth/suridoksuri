"""
오프라인 재생 CLI (vision_plan.md §7.5 기록·재생, §7.9 (a) "재생 오버레이 뷰어" — 데스크 주력 디버깅 경로).

벤치/비행 런에서 녹화한 원본 프레임(+텔레메트리)을 **동일 파이프라인**으로 책상에서 재생한다.
결정론적: 같은 입력+config → 같은 출력(§7.5). 입력이 디렉터리면 DirFrameSource(프레임 파일들
+ 선택적 telemetry.jsonl), 파일이면 BagFrameSource(비디오 + 선택적 사이드카 telemetry.jsonl)로
자동 판별한다.

--target-sink (vision↔fc 인터페이스, `docs/vision_fc_interface.md` §9 작업 V5): `main.py`와
**완전히 같은 CLI 이름·같은 기본 꺼짐·같은 하드페일 정책**으로 배선돼 있다(헬퍼는 상호 import
안 함 원칙에 따라 얇게 중복). 🔴 **재생 경로에 있어야 하는 결정적 이유**: `replay.py`는
Dir/Bag + `telemetry.jsonl`(AGL) 재생이라 **`agl_m`이 실린 `state_hint`를 회귀로 잡을 수 있는
유일한 경로**다 — `main.py`는 AGL을 받는 경로가 아예 없어서 상태머신이 `TERMINAL`까지 가지
못한다. 즉 "AGL이 실린 유도 힌트가 소비자에게 실제로 흘러가는가"는 여기서만 검증된다.

사용 예:
  # 녹화 폴더 재생, 창으로 확인
  python -m vision.replay recordings/flight02 --preset vision/presets/video.yaml --display window
  # bag(비디오 파일) 재생, 결과 mp4 + JSONL 블랙박스 기록
  python -m vision.replay recordings/flight02.mp4 --preset vision/presets/video.yaml \
      --output vision/results/replay.mp4 --log-dir vision/results/replay_logs
  # 정밀착륙 인터페이스 발행 켜기(기본 꺼짐) — 소비자는 nc 127.0.0.1 8091 로 확인
  python -m vision.replay recordings/flight02 --preset vision/presets/distress_fine.yaml --target-sink
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from vision.core.prior_geometry import (
    PRIOR_DETECTION_META_KEY,
    PRIOR_INPUTS_KEY,
    PRIOR_META_KEY,
)
from vision.core.runner import Pipeline
from vision.core.state_machine import (
    Decision,
    LandingSMConfig,
    LandingStateMachine,
    Observation,
    half_hfov_tan_from_calibration,
)
from vision.core.target import TargetEstimate, marker_object_points, solve_target_pose
from vision.utils.blackbox import BlackBoxLogger
from vision.utils.calibration_loader import CameraCalibration, load_camera_calibration
from vision.utils.frame_source import open_dir_or_bag
from vision.utils.logging import log_provenance_header, setup_dual_sink_logger
from vision.utils.stream import MjpegStreamer
from vision.utils.target_sink import (
    DEFAULT_HOST as _SINK_DEFAULT_HOST,
    DEFAULT_PORT as _SINK_DEFAULT_PORT,
    NullSink,
    SocketTargetSink,
    TargetSink,
)
from vision.utils.visualize import (
    draw_detections, draw_sink_status, draw_landing_overlay,
)

_WINDOW_NAME = "Landing Zone Detector — Replay"
# ArUco Phase 4(docs/vision_aruco_branch.md) — 기본 카메라 캘리브레이션. main.py와 동일 기본값
# (main.py와 헬퍼 상호 import 안 함 원칙 — vision/CLAUDE.md import 규칙, 각자 얇게 중복 허용).
_DEFAULT_CALIB_PATH = str(Path(__file__).parent / "calibration" / "cam109-imx708af75" / "nominal.yaml")
# `--output` mp4 fps 기본값. 재생 입력(녹화 폴더/bag)은 프레임 시각을 갖고 있지만 재생 속도는
# 실시간이 아니라 **처리 속도**라 원본 촬영 fps를 그대로 안다고 볼 수 없다 — 그래서 예전부터
# 이 고정값(30)을 썼고 그 동작을 그대로 보존한다. 실제 촬영 fps를 아는 사람이 `--output-fps`로
# 명시하면 그 값이 이긴다(2026-07-29 — main.py의 같은 이름 인자와 의미를 맞춤).
_REPLAY_DEFAULT_OUTPUT_FPS = 30.0

# `--target-sink` 발행의 `valid=false` 사유 문자열(§5.4 계약 2번) — main.py와 동일 값·동일 의미
# (얇게 중복). 소비자가 두 경로를 구분 없이 같은 문자열로 읽을 수 있어야 한다.
SINK_REASON_OK = "ok"
SINK_REASON_NO_CALIB = "no_calibration"
SINK_REASON_NO_DETECTION = "no_target_detection"
SINK_REASON_SOLVE_FAILED = "pose_solve_failed"
SINK_REASON_MAT_GEOMETRY_UNAVAILABLE = "mat_geometry_unavailable"
SINK_REASON_LANDING_POINT_UNPROJECTABLE = "landing_point_unprojectable"

# main.py와 **같은 종료코드**를 쓴다 — 두 CLI의 실패 신호가 갈리면 스크립트가 분기해야 한다.
EXIT_SINK_BIND_FAILED = 3


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
    """JSONL 블랙박스용 축약. 사전정보 스코어러가 켜졌을 때만 `prior` 키가 추가된다 —
    꺼져 있으면(=대부분의 경우) 기존 JSONL 형태와 **한 글자도 다르지 않다**
    (main.py와 동일 로직, 상호 import 안 함 원칙에 따라 얇게 중복)."""
    out = []
    for d in detections:
        entry = {"bbox": list(d.bbox), "confidence": d.confidence}
        prior = d.meta.get(PRIOR_DETECTION_META_KEY)
        if isinstance(prior, dict):
            entry["prior"] = {
                k: prior.get(k)
                for k in ("score", "area_score", "shape_score", "observed_area_px2",
                          "predicted_area_px2", "area_ratio", "observed_anisotropy",
                          "max_anisotropy", "incidence_deg", "error")
                if k in prior
            }
        out.append(entry)
    return out


def _build_prior_inputs(calib: Optional[CameraCalibration], telemetry: dict) -> dict:
    """`modules/prior_score.py::PriorGeometryScorer`가 소비할 프레임 단위 사전정보를 조립한다
    (`Pipeline.run(image, meta=...)` 통로).

    🔴 **재생 경로가 이 기능의 유일한 실증 경로다** — `main.py` 라이브에는 AGL/자세 역방향
    채널이 아직 없어(`docs/vision_fc_interface.md`, 다른 세션 작업 중) 항상 비활성으로
    degrade한다. `telemetry.jsonl`에 `alt`(라이다 AGL)와 `attitude`(자세)가 실려야 활성된다.

    값이 없으면 **키를 아예 안 넣는다** — 스코어러가 결손을 사유와 함께 비활성으로 처리한다
    (`main.py`와 동일 로직, 얇게 중복)."""
    if calib is None:
        return {}
    inputs: dict = {
        "camera_matrix": calib.camera_matrix,
        "dist_coeffs": calib.dist_coeffs,
        "calib_image_size": tuple(calib.image_size),
        "calib_id": calib.calib_id,
    }
    alt = telemetry.get("alt") if isinstance(telemetry, dict) else None
    if alt is not None:
        inputs["agl_m"] = alt
    attitude = telemetry.get("attitude") if isinstance(telemetry, dict) else None
    if attitude is not None:
        inputs["attitude"] = attitude
    return {PRIOR_INPUTS_KEY: inputs}


def _find_aruco_detection(detections):
    """ArUco Phase 4 — state.detections에서 코너를 가진 확정 ArUco 검출을 찾는다(main.py와
    동일 로직, 상호 import 안 함 원칙에 따라 얇게 중복— vision/CLAUDE.md import 규칙)."""
    for d in detections:
        if d.corners is not None and d.meta.get("aruco_id") is not None:
            return d
    return None


def _landing_sm_config(calib, logger) -> LandingSMConfig:
    """상태머신 config — 로드된 캘리브레이션이 있으면 `tan(HFOV/2)`를 인트린식에서 유도해 넣는다.
    `main.py::_landing_sm_config`와 같은 헬퍼의 얇은 중복(두 CLI는 서로 import하지 않는다 —
    기존 원칙). 근거는 `core/state_machine.py`의 `NOMINAL_HALF_HFOV_TAN` 주석 참조.
    """
    if calib is None:
        return LandingSMConfig()
    try:
        return LandingSMConfig(
            half_hfov_tan=half_hfov_tan_from_calibration(calib.camera_matrix, calib.image_size)
        )
    except (ValueError, IndexError, TypeError) as e:
        logger.warning("캘리브레이션에서 tan(HFOV/2) 유도 실패 — nominal 기본값 사용: %s", e)
        return LandingSMConfig()


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


def _solve_aruco_estimate(
    state, calib: Optional[CameraCalibration], frame_id: int, ts: float, logger,
) -> Tuple[Optional[TargetEstimate], str]:
    """확정 ArUco 검출 + 로드된 calib이 둘 다 있을 때만 solvePnP를 시도한다. 마커가 없거나
    calib 미로드거나 solvePnP가 수렴하지 않아도 크래시 없이 `(None, 사유)`를 돌려준다.

    **`TargetEstimate` 객체를 그대로 돌려준다**(예전엔 여기서 곧바로 JSONL용 dict로 눌러
    담았다) — `--target-sink` 발행(`core/wire.py::build_target_record`)은 dict가 아니라
    dataclass를 요구하기 때문이다. dict 변환은 호출부가 한다(JSONL `chosen` 형태 무변경).
    **사유를 함께 돌려주는 이유**(§5.4 계약 2번): 무효 레코드에 사유를 실으려면 여기서 이미
    알고 있는 구분(calib 미로드 / 마커 없음 / solvePnP 실패)을 버리면 안 된다.
    (전부 main.py `_solve_aruco_estimate`와 동일 — 상호 import 안 함 원칙에 따라 얇게 중복.)
    """
    if calib is None:
        return None, SINK_REASON_NO_CALIB
    det = _find_aruco_detection(state.detections)
    if det is None:
        return None, SINK_REASON_NO_DETECTION
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
        return None, SINK_REASON_SOLVE_FAILED
    return estimate, SINK_REASON_OK


def _solve_distress_mat_estimate(
    state, calib: Optional[CameraCalibration], frame_id: int, ts: float, logger,
) -> Tuple[Optional[TargetEstimate], str]:
    """초록 매트 4코너 + 알려진 실측 크기(3.0m) -> solvePnP -> ArUco와 같은 형식의
    `TargetEstimate`(main.py `_solve_distress_mat_estimate`와 동일 로직, 얇게 중복).

    §7.5가 "같은 파이프라인으로 오프라인 재생"을 회귀검증의 최대 레버로 못박고 있어, 재생
    경로에도 붙어 있어야 초록구역 pose 계약을 책상에서 회귀로 잡을 수 있다.
    **position은 매트 중심이 아니라 착륙점**(§5.3 "박스 옆 빈 초록면")이다.

    실패 사유를 뭉개지 않는다: 매트 미검출 / 매트는 봤지만 4코너가 쓸모없음 / 착륙점 역투영
    실패 / solvePnP 미수렴을 각각 다른 문자열로 내보낸다(§5.4)."""
    if calib is None:
        return None, SINK_REASON_NO_CALIB
    det = _find_distress_mat_detection(state.detections)
    if det is None:
        # 매트 후보는 있었는데 기하 태깅이 전부 실패했다면 "아예 못 봤다"와 구분해 준다.
        geom_meta = state.meta.get("distress_mat_geometry")
        if geom_meta is not None and geom_meta.get("skipped"):
            return None, SINK_REASON_MAT_GEOMETRY_UNAVAILABLE
        return None, SINK_REASON_NO_DETECTION
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
        # 역투영 실패와 solvePnP 실패가 둘 다 ValueError로 온다 — 메시지가 아니라 "착륙점을
        # 넘겼는가"로 갈라야 사유가 안전하게 구분된다(main.py와 동일 판단).
        reason = SINK_REASON_SOLVE_FAILED if coarse else SINK_REASON_LANDING_POINT_UNPROJECTABLE
        logger.warning(
            "frame %d: 초록 매트 pose 산출 실패(%s) — TargetEstimate 생략: %s", frame_id, reason, e
        )
        return None, reason
    return estimate, SINK_REASON_OK


def _solve_target_estimate(
    state, calib: Optional[CameraCalibration], frame_id: int, ts: float, logger,
) -> Tuple[Optional[TargetEstimate], str]:
    """어느 pose 산출기를 쓸지는 **프리셋이 남긴 meta**가 정한다(main.py와 동일 판단).
    🔴 ArUco 검출이 있으면 그 결과가 그대로 나가고, 초록 매트는 ArUco가 없을 때만 시도된다
    (`no_target_detection` 이외의 사유는 ArUco 결과를 그대로 돌려준다 — 배선 전과 동일 값)."""
    estimate, reason = _solve_aruco_estimate(state, calib, frame_id, ts, logger)
    if reason != SINK_REASON_NO_DETECTION:
        return estimate, reason
    return _solve_distress_mat_estimate(state, calib, frame_id, ts, logger)


def _merge_target_estimate_into_chosen(chosen: Optional[dict], estimate) -> Optional[dict]:
    """ArUco Phase 4가 정한 JSONL `chosen` 형태를 그대로 유지한다(main.py와 동일, 얇게 중복)."""
    if estimate is None:
        return chosen
    return {**(chosen or {}), "target_estimate": _target_estimate_to_dict(estimate)}


def _publish_to_sink(
    sink: TargetSink,
    *,
    frame_id: int,
    estimate: Optional[TargetEstimate],
    reason: str,
    decision: Optional[Decision],
    logger,
) -> None:
    """`--target-sink` 발행 — 매 프레임 `target` 1건 + `state_hint` 1건(main.py와 동일, 얇게 중복).

    **§5.4 페일세이프 계약**: 검출이 없어도 **발행을 멈추지 않는다.** 침묵은 "노드가 죽었다"는
    뜻으로 예약돼 있으므로 "안 보임"은 `valid=false` + 사유로 명시해야 구분이 선다.

    🔴 **어떤 예외도 재생 루프로 새어 나가지 않는다** — bind(기동)와 정책이 정반대인 이유는
    `main.py::_publish_to_sink` docstring 참조(기동은 1회/지상, 발행은 매 프레임/비행 중).
    """
    if not isinstance(sink, SocketTargetSink):
        return
    try:
        if estimate is not None:
            sink.publish_target(estimate)
        else:
            sink.publish_invalid(frame_id=frame_id, reason=reason)
        if decision is not None:
            sink.publish_state_hint(decision, frame_id=frame_id)
    except Exception as e:  # noqa: BLE001 — 비차단·무크래시 계약이 예외 종류보다 우선한다
        logger.warning("frame %d: target sink 발행 실패(무시하고 계속): %s", frame_id, e)


def _make_target_sink(
    target_sink: bool, host: str, port: int, logger,
) -> TargetSink:
    """`main.py::_make_target_sink`와 **같은 정책**(얇게 중복): 미지정이면 `NullSink`,
    지정이면 기동된 `SocketTargetSink`, **기동(bind) 실패는 종료코드 3 하드 페일**.

    두 CLI의 정책이 갈리면 "재생에서는 조용히 넘어갔는데 라이브에서 죽는" 식의 차이가 생겨
    회귀검증 경로(§7.5)로서의 가치가 사라진다 — 근거 전문은 main.py 쪽 docstring 참조.
    """
    if not target_sink:
        return NullSink()
    sink = SocketTargetSink(host=host, port=port)
    try:
        sink.start()
    except OSError as e:
        endpoint = f"{host}:{port}"
        logger.error("target sink 기동 실패 (%s) — 하드 페일로 종료합니다: %s", endpoint, e)
        print(
            f"Error: --target-sink 기동 실패 — {endpoint} 에 bind 하지 못했습니다: {e}\n"
            f"       십중팔구 포트 {port} 충돌입니다. 점유 프로세스 확인: ss -ltnp | grep {port}\n"
            f"       --target-sink 를 명시했으므로 조용히 강등하지 않고 즉시 종료합니다 "
            f"(exit {EXIT_SINK_BIND_FAILED}). 다른 포트: --target-sink-port <N>",
            file=sys.stderr,
        )
        sys.exit(EXIT_SINK_BIND_FAILED)
    logger.info("target sink 시작: %s:%d (JSON Lines)", sink.host, sink.port)
    print(f"target sink: {sink.host}:{sink.port}")
    return sink


def _draw_sink_overlay(annotated, sink: TargetSink) -> None:
    """유도 발행 상태 오버레이(main.py와 동일, 얇게 중복) — **제자리 수정**.
    호출자는 `display != "none"`일 때만 부른다(헤드리스 경로 비용 0)."""
    if isinstance(sink, SocketTargetSink):
        draw_sink_status(
            annotated,
            enabled=True,
            consumers=sink.client_count,
            seq=sink.last_seq,
            dropped=sink.dropped,
            endpoint=f"{sink.host}:{sink.port}",
        )
    else:
        draw_sink_status(annotated, enabled=False)


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
    target_sink: bool = False,
    target_sink_host: str = _SINK_DEFAULT_HOST,
    target_sink_port: int = _SINK_DEFAULT_PORT,
    report_overlay: bool = False,
    output_fps: float | None = None,
) -> int:
    """실제 재생 루프. main()에서 분리해 프로그램적으로도 호출/테스트 가능하게 한다.

    display="stream" (§7.9 항목5, opt-in): 재생 중인(annotated) 프레임을 MjpegStreamer로
    실시간 흘려보낸다 — 비차단(§7.9 비침습 전제), 켤 때만 오버헤드 발생.

    calib_path(ArUco Phase 4): 카메라 캘리브레이션 yaml, 루프 시작 전 1회만 로드해 재사용한다
    (§7.1 config 레이어드, 매 프레임 파일 I/O 금지). 없어도(FileNotFoundError) 재생 자체는
    막지 않고 경고만 남긴다 — ArUco와 무관한 프리셋 재생까지 이 로드 실패로 막으면 안 된다.

    target_sink(기본 **꺼짐**, main.py와 동일 opt-in 관례): 켜면 `SocketTargetSink`를 띄우고
    매 프레임 `target` + `state_hint`를 발행한다. **재생 경로에만 있는 가치**: 여기서만
    `telemetry.jsonl`의 AGL이 상태머신에 흘러 `state_hint`가 `TERMINAL`까지 진행한다
    (`main.py`는 AGL을 받는 경로 자체가 없다) — 파일 docstring 참조.
    ⚠️ 기동(bind) 실패는 main.py와 같은 **하드 페일**(`SystemExit(3)`)이다.
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
    state_machine = LandingStateMachine(_landing_sm_config(calib, logger))

    # vision↔fc 인터페이스. **기본은 NullSink** — 켜지 않으면 소켓도 스레드도 안 뜬다.
    sink: TargetSink = NullSink()
    streamer = None
    writer = None
    frame_count = 0
    try:
        # 🔴 try **안**에서 만든다 — bind 실패는 SystemExit로 즉사하는데, 그때도 finally가
        # blackbox 큐 스레드/파일 핸들을 정리해야 한다(main.py와 동일 구조).
        sink = _make_target_sink(target_sink, target_sink_host, target_sink_port, logger)

        if display == "stream":
            streamer = MjpegStreamer(host=stream_host, port=stream_port)
            streamer.start()
            logger.info("라이브 스트림 시작: %s", streamer.url)
            print(f"라이브 스트림: {streamer.url}")

        with source:
            for record in source:
                t0 = time.perf_counter()
                # 사전정보(AGL·자세)를 파이프라인 시작 전에 심는다 — 없으면 빈 dict라
                # `Pipeline.run`이 예전과 동일하게 동작하고 스코어러도 비활성으로 degrade한다.
                state = pipeline.run(
                    record.image, meta=_build_prior_inputs(calib, record.telemetry)
                )
                latency = time.perf_counter() - t0

                annotated = draw_detections(state.original, state.detections, state.confirmed)
                frame_count += 1

                estimate, sink_reason = _solve_target_estimate(
                    state, calib, record.frame_id, record.ts, logger
                )
                chosen = _merge_target_estimate_into_chosen(
                    _confirmed_to_dict(state.confirmed), estimate
                )

                # telemetry.jsonl에 alt(라이다 AGL)가 있으면 쓰고, 없으면 None으로 우아하게
                # degrade한다(실기체 telemetry 아직 없음 — AGL 없이도 상태머신은 동작해야 함).
                decision = state_machine.update(
                    _build_observation(
                        state, record.frame_id, record.ts, agl_m=record.telemetry.get("alt")
                    )
                )

                _publish_to_sink(
                    sink, frame_id=record.frame_id, estimate=estimate, reason=sink_reason,
                    decision=decision, logger=logger,
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
                prior_meta = state.meta.get(PRIOR_META_KEY)
                if prior_meta is not None:
                    # 비활성 사유가 조용히 사라지면 "왜 점수가 안 붙지"를 현장에서 못 쫓는다.
                    logger.debug("frame %d: prior_geometry=%s", record.frame_id, prior_meta)

                # 발행 뒤에 그려야 이 프레임의 seq까지 반영된다. `none`이면 비용 0.
                if display != "none":
                    _draw_sink_overlay(annotated, sink)
                if report_overlay:
                    # main.py::_draw_report_overlay와 같은 내용(헬퍼 상호 import 금지 관례에 따라
                    # 얇게 중복). 착륙점/상태를 화면에 그리기만 한다 — 계산은 하지 않는다.
                    draw_landing_overlay(
                        annotated,
                        state.detections,
                        state=decision.state.value if decision is not None else None,
                        command=decision.command if decision is not None else None,
                        estimate=estimate,
                        frame_id=record.frame_id,
                    )

                if streamer is not None:
                    streamer.push_frame(annotated)  # 비차단 — 재생 루프를 지연시키지 않음

                if output:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            output, fourcc, output_fps or _REPLAY_DEFAULT_OUTPUT_FPS, (w, h)
                        )
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
        # sink 정리 = 클라이언트 소켓 close = 소비자가 **EOF**를 받는 경로(§5.4).
        # 정리 자체가 실패해도 다른 리소스 정리를 막지 않는다.
        try:
            sink.close()
        except Exception as e:  # noqa: BLE001 — 종료 경로에서 예외를 새로 만들지 않는다
            logger.warning("target sink 종료 실패(무시): %s", e)

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
    # vision↔fc 인터페이스 — main.py와 **문자 그대로 같은 인자 이름/기본값**(같은 opt-in 관례).
    parser.add_argument(
        "--target-sink",
        action="store_true",
        help="정밀착륙 인터페이스 발행 켜기(기본 꺼짐). localhost TCP 서버를 띄우고 매 프레임 "
             "JSON Lines(target + state_hint)를 발행한다. 재생 경로는 telemetry.jsonl의 AGL이 "
             "실려 state_hint가 TERMINAL까지 진행하는 유일한 경로다. 확인: nc 127.0.0.1 8091. "
             "⚠️ bind 실패 시 강등 없이 즉시 종료(exit 3).",
    )
    parser.add_argument(
        "--target-sink-host",
        default=_SINK_DEFAULT_HOST,
        help=f"--target-sink 바인딩 주소 (기본 {_SINK_DEFAULT_HOST}). ⚠️ 0.0.0.0으로 열면 "
             "유도 스트림이 주변 네트워크에 노출된다(utils/target_sink.py 참조).",
    )
    parser.add_argument(
        "--target-sink-port",
        type=int,
        default=_SINK_DEFAULT_PORT,
        help=f"--target-sink 포트 (기본 {_SINK_DEFAULT_PORT}). 0을 주면 OS가 임시 포트를 고른다.",
    )
    parser.add_argument(
        "--report-overlay",
        action="store_true",
        help="착륙 판단(착륙점/흰 박스/상태머신 상태·명령/추정거리)을 프레임에 그린다(기본 꺼짐). "
             "사람에게 보여줄 영상(보고/발표)용 — main.py와 문자 그대로 같은 인자다.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help=f"--output mp4에 기록할 fps (미지정 시 {_REPLAY_DEFAULT_OUTPUT_FPS}). "
             "원본 촬영 fps를 알면 명시할 것 — 아니면 저장 영상이 배속으로 재생된다.",
    )
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
        target_sink=args.target_sink, target_sink_host=args.target_sink_host,
        target_sink_port=args.target_sink_port,
        report_overlay=args.report_overlay, output_fps=args.output_fps,
    )
    print(f"Replayed {frame_count} frames from {input_path}")


if __name__ == "__main__":
    main()
