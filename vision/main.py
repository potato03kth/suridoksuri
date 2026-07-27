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

--target-sink (vision↔fc 인터페이스, `docs/vision_fc_interface.md` §9 작업 V5): **opt-in,
기본 꺼짐**. 켜면 `SocketTargetSink`(localhost TCP 서버, 기본 127.0.0.1:8091)를 띄우고 매
프레임 JSON Lines 레코드를 발행한다 — `target`(TargetEstimate 또는 valid=false) 1건 +
`state_hint`(상태머신 Decision) 1건. 끄면 `NullSink`라 레코드 조립 비용조차 들지 않는다
(기존 실행 경로 무변경). 확인: `nc 127.0.0.1 8091`.

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
  # 정밀착륙 인터페이스 발행 켜기(기본 꺼짐) — 소비자는 nc 127.0.0.1 8091 로 확인
  python -m vision.main live --preset presets/vertiport_fine.yaml --target-sink
"""
import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from vision.core.runner import Pipeline
from vision.core.state_machine import Decision, LandingStateMachine, Observation
from vision.core.target import (
    TargetEstimate,
    marker_object_points,
    solve_target_pose,
)
from vision.utils.blackbox import BlackBoxLogger
from vision.utils.calibration_loader import CameraCalibration, load_camera_calibration
from vision.utils.frame_source import LiveFrameSource
from vision.utils.image_loader import load_image
from vision.utils.logging import log_provenance_header, setup_dual_sink_logger
from vision.utils.stream import MjpegStreamer
from vision.utils.target_sink import (
    DEFAULT_HOST as _SINK_DEFAULT_HOST,
    DEFAULT_PORT as _SINK_DEFAULT_PORT,
    NullSink,
    SocketTargetSink,
    TargetSink,
)
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

# `--target-sink` 발행의 `valid=false` 사유 문자열(§5.4 계약 2번 — 침묵 대신 **사유와 함께**
# "안 보임"을 알린다). 매직 문자열을 호출부에 흩뿌리지 않도록 상수로 모은다(§7.3).
SINK_REASON_OK = "ok"
SINK_REASON_NO_CALIB = "no_calibration"
SINK_REASON_NO_DETECTION = "no_target_detection"
SINK_REASON_SOLVE_FAILED = "pose_solve_failed"
# ② 조난자 구역(초록 매트) 전용 사유 — `no_target_detection` 하나로 뭉뚱그리면 현장에서
# "매트를 아예 못 봤다"와 "매트는 봤는데 pose를 못 냈다"가 구분되지 않는다(§5.4 계약 2번의
# 취지가 사유를 남기는 것인데, 사유가 뭉개지면 있으나 마나다).
SINK_REASON_MAT_GEOMETRY_UNAVAILABLE = "mat_geometry_unavailable"
SINK_REASON_LANDING_POINT_UNPROJECTABLE = "landing_point_unprojectable"


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
    d = {
        "position": list(estimate.position),
        "orientation": list(estimate.orientation),
        "confidence": estimate.confidence,
        "target_type": estimate.target_type,
        "calib_accuracy": estimate.calib_accuracy,
        "not_for_closed_loop_30cm": estimate.not_for_closed_loop_30cm,
        "calib_id": estimate.calib_id,
    }
    # ArUco 경로는 meta를 안 채우므로(`solve_target_pose` 기본값 {}) 이 키가 생기지 않는다 —
    # 기존 JSONL `chosen` 형태 무변경. 초록 매트 경로만 착륙점/평면기준을 추가로 남긴다.
    if estimate.meta:
        d["meta"] = dict(estimate.meta)
    return d


def _solve_aruco_estimate(
    state, calib: Optional[CameraCalibration], frame_id: int, ts: float, logger,
) -> Tuple[Optional[TargetEstimate], str]:
    """확정 ArUco 검출 + 로드된 calib이 둘 다 있을 때만 solvePnP를 시도한다(§Phase4 전제 —
    코너 없으면 solvePnP 자체가 실패하므로 매 프레임 무조건 시도하지 않음). 마커가 없거나
    calib 미로드거나 solvePnP가 수렴하지 않아도 크래시 없이 `(None, 사유)`를 돌려준다.

    **`TargetEstimate` 객체를 그대로 돌려준다**(예전엔 여기서 곧바로 dict로 눌러 담았다) —
    `--target-sink` 발행(`core/wire.py::build_target_record`)은 dict가 아니라 dataclass를
    요구하기 때문이다. JSONL 블랙박스용 dict 변환은 호출부가 `_target_estimate_to_dict()`로
    한다(기존 `chosen` 형태·내용 무변경).

    **실패 사유를 함께 돌려주는 이유**(§5.4 계약 2번): 검출이 없을 때 sink가 침묵하면 소비자는
    "노드 사망"과 구분할 수 없다. 무효 레코드에 사유를 실으려면 여기서 이미 알고 있는 구분
    (calib 미로드 / 마커 없음 / solvePnP 실패)을 버리지 않고 밖으로 내보내야 한다.
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


def _find_distress_mat_detection(detections):
    """② 조난자 구역(초록 매트) — `modules/distress_mat.py::DistressMatGeometry`가 pose 산출용
    기하를 태깅해 둔 검출을 찾는다. 그 모듈은 코너 정규화에 성공한 것만 `meta["distress_mat"]`을
    남기므로 첫 항목을 그대로 쓴다(ArUco와 동일 단일 타겟 전제)."""
    for d in detections:
        if d.meta.get("distress_mat") is not None:
            return d
    return None


def _solve_distress_mat_estimate(
    state, calib: Optional[CameraCalibration], frame_id: int, ts: float, logger,
) -> Tuple[Optional[TargetEstimate], str]:
    """초록 매트 4코너 + **알려진 실측 크기(3.0m)** -> `solvePnP` -> ArUco와 **완전히 같은 형식**의
    `TargetEstimate`. 이 경로가 없으면 초록구역은 검출은 되는데 기체에 보낼 좌표가 없다.

    - **objectPoints는 `marker_object_points(size_m)`을 그대로 재사용한다** — 그 함수가 애초에
      크기를 인자로 받으므로(ArUco는 0.50) 새 기계장치가 필요 없다. 코너 순서 대응은
      `DistressMatGeometry`가 `order_quad_corners_clockwise()`로 이미 맞춰 뒀다.
    - **AGL이 필요 없다** — 알려진 크기가 스케일을 준다(`main.py`는 AGL을 받는 경로 자체가 없다).
    - **position은 매트 중심이 아니라 착륙점이다**(§5.3). `position_at_pixel`로 착륙점 픽셀을
      넘기면 `core/target.py`가 매트 평면 위로 역투영해 그 점을 `position`으로 쓴다.
    - **provenance는 ArUco 경로와 한 글자도 다르지 않게** 붙인다(nominal intrinsics라
      30cm 폐루프 미검증이라는 사실이 소비자까지 전파돼야 한다, §7.3).

    실패 사유를 뭉개지 않는다: 매트 미검출(`no_target_detection`) / 매트는 봤지만 4코너가
    쓸모없음(`mat_geometry_unavailable`) / 착륙점 역투영 실패(`landing_point_unprojectable`) /
    solvePnP 미수렴(`pose_solve_failed`)을 각각 다른 문자열로 내보낸다.
    """
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
            # coarse(매트 중심)는 tvec이 곧 그 점이라 역투영을 거치지 않는다 — 같은 값을
            # 두 경로로 구해 미세한 수치차를 만들 이유가 없다.
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
                # 🔴 z=0 평면이 지면이 아니라 매트 윗면(0.105m 라이즈드)이라는 사실 —
                # 소비자가 라이다 AGL과 섞을 때 이 차이를 알아야 한다.
                "plane_reference": mat["plane_reference"],
                "platform_height_m": mat["platform_height_m"],
                # 프레임 크기 ↔ 캘리브레이션 해상도 불일치는 pose를 그 비율만큼 통째로
                # 틀리게 한다(스케일이 focal에 그대로 실리므로). 조용히 재스케일하지 않고
                # (다운스케일인지 크롭인지 알 수 없다) 사실만 실어 보낸다.
                "frame_size_px": [int(state.original.shape[1]), int(state.original.shape[0])],
                "calib_image_size_px": [int(calib.image_size[0]), int(calib.image_size[1])],
            },
        )
    except ValueError as e:
        # `project_pixel_onto_target_plane()`(역투영)과 `cv2.solvePnP` 실패가 둘 다 ValueError로
        # 온다 — 메시지가 아니라 "착륙점을 넘겼는가"로 갈라야 사유가 안전하게 구분된다.
        reason = SINK_REASON_SOLVE_FAILED if coarse else SINK_REASON_LANDING_POINT_UNPROJECTABLE
        logger.warning(
            "frame %d: 초록 매트 pose 산출 실패(%s) — TargetEstimate 생략: %s", frame_id, reason, e
        )
        return None, reason
    return estimate, SINK_REASON_OK


def _solve_target_estimate(
    state, calib: Optional[CameraCalibration], frame_id: int, ts: float, logger,
) -> Tuple[Optional[TargetEstimate], str]:
    """프레임 하나에서 상대 pose를 산출한다 — **어느 산출기를 쓸지는 프리셋이 정한다.**

    선택 기전은 preset 경로 문자열 파싱이 아니라 **파이프라인이 남긴 meta**다: ArUco 프리셋은
    `meta["aruco_id"]`를, 초록구역 프리셋은 `meta["distress_mat"]`(=`distress_mat_geometry`
    스텝이 있을 때만 생김)을 남긴다. 경로 문자열 관례는 깨지기 쉽고, 같은 `rect_detector`를
    쓰는 범용 프리셋(`video.yaml`)에 3m 매트 크기가 실수로 적용되는 것도 막아야 한다.

    🔴 **ArUco 경로는 조금도 달라지지 않는다.** ArUco가 `no_target_detection` 이외의 사유를
    내면(성공/calib 없음/solvePnP 실패) 그대로 돌려주고, 초록 매트 경로는 **ArUco 검출이
    없을 때만** 시도된다. 초록 매트도 없으면 `no_target_detection` — 배선 전과 같은 값이다.
    """
    estimate, reason = _solve_aruco_estimate(state, calib, frame_id, ts, logger)
    if reason != SINK_REASON_NO_DETECTION:
        return estimate, reason
    return _solve_distress_mat_estimate(state, calib, frame_id, ts, logger)


def _merge_target_estimate_into_chosen(chosen: Optional[dict], estimate) -> Optional[dict]:
    """ArUco Phase 4가 정한 JSONL `chosen` 형태를 그대로 유지한다 — 기존 버티포트/조난자 경로의
    `{"bbox":..., "confidence":...}`와 키가 겹치지 않아 병합돼 공존한다."""
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
    """`--target-sink` 발행 — 매 프레임 `target` 1건 + (상태머신이 있으면) `state_hint` 1건.

    **§5.4 페일세이프 계약의 핵심이 여기 있다**: 검출이 없어도 **발행을 멈추지 않는다.**
    침묵은 "노드가 죽었다"는 뜻으로 예약돼 있으므로(소비자는 EOF/타임아웃으로 사망을 잰다),
    "안 보임"은 반드시 `valid=false` + 사유로 **명시적으로** 말해야 구분이 선다.

    🔴 **어떤 예외도 파이프라인으로 새어 나가지 않는다.** 소비자가 없든/끊기든/느리든/
    레코드 조립이 터지든 vision은 계속 돈다 — 착륙 유도보다 검출이 먼저 죽는 일은 없어야 한다.
    그래서 `except Exception`이 의도적으로 넓다(좁히면 예상 못 한 예외가 그대로 루프를 죽인다).
    `BaseException`(KeyboardInterrupt/SystemExit)은 **일부러 안 잡는다** — 그건 정상 종료 경로다.

    `NullSink`(기본값, `--target-sink` 미지정)면 즉시 돌아온다 — 레코드 조립 비용조차 들이지
    않아 기존 실행 경로가 조금도 달라지지 않는다(`if streamer is not None` opt-in 전례와 동일).
    """
    if not isinstance(sink, SocketTargetSink):
        return
    try:
        if estimate is not None:
            sink.publish_target(estimate)
        else:
            sink.publish_invalid(frame_id=frame_id, reason=reason)
        if decision is not None:
            # `command`는 `command_hint`(+`command_is_advisory`)로 나간다 — 명령이 아니라
            # 권고다(`core/wire.py` "🔴 command는 명령이 아니다").
            sink.publish_state_hint(decision, frame_id=frame_id)
    except Exception as e:  # noqa: BLE001 — 비차단·무크래시 계약이 예외 종류보다 우선한다
        logger.warning("frame %d: target sink 발행 실패(무시하고 계속): %s", frame_id, e)


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
    sink: Optional[TargetSink] = None,
) -> None:
    sink = sink if sink is not None else NullSink()
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

    estimate, sink_reason = _solve_target_estimate(state, calib, 0, ts, logger)
    chosen = _merge_target_estimate_into_chosen(_confirmed_to_dict(state.confirmed), estimate)

    decision = None
    if state_machine is not None:
        decision = state_machine.update(_build_observation(state, 0, ts))

    _publish_to_sink(
        sink, frame_id=0, estimate=estimate, reason=sink_reason, decision=decision, logger=logger,
    )

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
    sink: Optional[TargetSink] = None,
) -> None:
    from vision.utils.video_reader import VideoReader

    sink = sink if sink is not None else NullSink()
    writer = None
    frame_count = 0
    with VideoReader(str(input_path)) as reader:
        for frame in reader:
            t0 = time.perf_counter()
            state = pipeline.run(frame)
            latency = time.perf_counter() - t0
            ts = time.time()
            annotated = draw_detections(state.original, state.detections, state.confirmed)

            estimate, sink_reason = _solve_target_estimate(state, calib, frame_count, ts, logger)
            chosen = _merge_target_estimate_into_chosen(
                _confirmed_to_dict(state.confirmed), estimate
            )

            decision = None
            if state_machine is not None:
                decision = state_machine.update(_build_observation(state, frame_count, ts))

            _publish_to_sink(
                sink, frame_id=frame_count, estimate=estimate, reason=sink_reason,
                decision=decision, logger=logger,
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


def _install_sigterm_handler(stop_event: threading.Event) -> None:
    """SIGTERM 수신 시 stop_event를 세팅한다(`tools/h264_stream.py::_install_sigterm_handler`와
    동일 패턴 재사용). 비대화형 SSH 백그라운드 자식/systemd 등에서는 SIGINT가 SIG_IGN으로 막혀
    있을 수 있어(1A 실측, `docs/vision_camera_bringup.md`) graceful shutdown을 보장하는 유일한
    신호가 SIGTERM이다. **기존 Ctrl+C(KeyboardInterrupt/SIGINT) 경로는 건드리지 않는다** — 여기서는
    SIGTERM만 새로 등록해 추가한다(둘 다 등록돼도 서로 배타적이지 않음)."""

    def _handler(signum, frame):  # noqa: ARG001 - signal handler 표준 시그니처
        stop_event.set()

    signal.signal(signal.SIGTERM, _handler)


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
    sink: Optional[TargetSink] = None,
) -> None:
    """실카메라 라이브 모드 — `_run_video`와 거의 동일한 프레임 루프(무한 이터레이터라는 점만
    다름). 헤드리스(`--display none`)에서는 무한정 도는 게 정상 동작이라 Ctrl+C(KeyboardInterrupt)와
    SIGTERM 둘 다 종료 수단이다 — `with LiveFrameSource(...)`가 예외 전파 중에도 카메라를
    release하고, 바깥 try/except가 스택트레이스 없이 조용히 종료시킨다(로그만 남김). SIGTERM은
    비대화형 배포(systemd 등) 표준 종료 신호라 별도로 `stop_event`를 두고 다음 프레임 경계에서
    루프를 빠져나간다(§h264_stream.py와 동일 근거 — SIGINT는 비대화형 자식에서 못 믿음).

    🔴 **`sink.install_signal_handlers()`를 여기서 부르면 안 된다.** `signal.signal`은 신호당
    핸들러가 **하나**뿐이라, sink 핸들러를 나중에 걸면 바로 위 `_install_sigterm_handler`가
    등록한 핸들러를 **덮어써서** `stop_event`가 영영 세팅되지 않는다 — 실기체에서만 드러났던
    SIGTERM graceful shutdown 버그가 그대로 재발한다. sink 정리는 `main()`의 `finally`에서
    `sink.close()`로 하면 충분하다(루프를 빠져나오면 반드시 거기로 간다)."""
    sink = sink if sink is not None else NullSink()
    live_kwargs: dict = {"camera_num": camera_num, "retries": retries, "retry_delay": retry_delay}
    if resolution is not None:
        live_kwargs["resolution"] = resolution

    stop_event = threading.Event()
    _install_sigterm_handler(stop_event)

    writer = None
    frame_count = 0
    try:
        with LiveFrameSource(**live_kwargs) as source:
            for record in source:
                if stop_event.is_set():
                    logger.info(
                        "라이브 모드 SIGTERM으로 종료 요청 — %d 프레임 처리 후 정상 종료",
                        frame_count,
                    )
                    break
                t0 = time.perf_counter()
                state = pipeline.run(record.image)
                latency = time.perf_counter() - t0
                ts = record.ts
                annotated = draw_detections(state.original, state.detections, state.confirmed)

                estimate, sink_reason = _solve_target_estimate(
                    state, calib, frame_count, ts, logger
                )
                chosen = _merge_target_estimate_into_chosen(
                    _confirmed_to_dict(state.confirmed), estimate
                )

                decision = None
                if state_machine is not None:
                    decision = state_machine.update(
                        _build_observation(state, frame_count, ts, agl_m=record.telemetry.get("alt"))
                    )

                _publish_to_sink(
                    sink, frame_id=frame_count, estimate=estimate, reason=sink_reason,
                    decision=decision, logger=logger,
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


def _make_target_sink(args, logger) -> TargetSink:
    """`--target-sink` 미지정이면 `NullSink`, 지정이면 기동된 `SocketTargetSink`.

    🔴 **기동(bind)에 실패해도 `NullSink`로 강등하고 계속 간다.** `utils/target_sink.py`의
    `start()`는 "포트 충돌을 조용히 삼키면 안 된다"는 이유로 예외를 올리도록 설계돼 있고 그
    판단은 옳지만, `main.py` 레벨의 요구는 **검출 파이프라인이 sink 때문에 죽지 않는 것**이
    더 강하다(착륙 유도보다 검출이 먼저 죽을 수는 없다). 그래서 "조용히"가 아니라 **시끄럽게**
    — ERROR 로그 + stderr 출력 — 알린 뒤 강등한다. `--display stream`(MjpegStreamer)은 반대로
    예외를 그대로 올리는데, 그건 디버그용 스트림이라 실패가 곧 운용 중단 사유이기 때문이다.
    """
    if not args.target_sink:
        return NullSink()
    sink = SocketTargetSink(host=args.target_sink_host, port=args.target_sink_port)
    try:
        sink.start()
    except OSError as e:
        msg = (
            f"target sink 기동 실패 ({args.target_sink_host}:{args.target_sink_port}) — "
            f"발행 없이 검출만 계속합니다: {e}"
        )
        logger.error(msg)
        print(f"Error: {msg}", file=sys.stderr)
        return NullSink()
    logger.info("target sink 시작: %s:%d (JSON Lines)", sink.host, sink.port)
    print(f"target sink: {sink.host}:{sink.port}")
    return sink


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
    # vision↔fc 인터페이스(§9 작업 V5). `--display stream`과 **동일한 opt-in 관례** —
    # 켜는 플래그 1개 + host/port 2개. 기본 꺼짐(NullSink)이라 기존 실행 경로는 무변경.
    parser.add_argument(
        "--target-sink",
        action="store_true",
        help="정밀착륙 인터페이스 발행 켜기(기본 꺼짐). localhost TCP 서버를 띄우고 매 프레임 "
             "JSON Lines(target + state_hint)를 발행한다(docs/vision_fc_interface.md §9 V5). "
             "확인: nc 127.0.0.1 8091",
    )
    parser.add_argument(
        "--target-sink-host",
        default=_SINK_DEFAULT_HOST,
        help=f"--target-sink 바인딩 주소 (기본 {_SINK_DEFAULT_HOST}). ⚠️ 0.0.0.0으로 열면 "
             "비행 중 유도 스트림이 주변 네트워크에 노출된다(utils/target_sink.py 참조).",
    )
    parser.add_argument(
        "--target-sink-port",
        type=int,
        default=_SINK_DEFAULT_PORT,
        help=f"--target-sink 포트 (기본 {_SINK_DEFAULT_PORT}). 0을 주면 OS가 임시 포트를 고른다.",
    )
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

    # vision↔fc 인터페이스(§9 V5). **기본은 NullSink** — `--target-sink`를 명시하지 않으면
    # 소켓도 스레드도 뜨지 않고 레코드 조립도 일어나지 않는다(기존 실행 경로 완전 무변경).
    sink: TargetSink = _make_target_sink(args, logger)

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
                state_machine, sink,
            )
        elif input_path.suffix.lower() in _VIDEO_SUFFIXES:
            _run_video(
                pipeline, input_path, args.output, args.display, logger, blackbox, streamer, calib,
                state_machine, sink,
            )
        else:
            _run_image(
                pipeline, input_path, args.output, args.display, logger, blackbox, streamer, calib,
                state_machine, sink,
            )
    finally:
        blackbox.close()
        if streamer is not None:
            streamer.stop()
        # sink 정리 = 클라이언트 소켓 close = 소비자가 **EOF**를 받는 경로(§5.4). SIGTERM/
        # Ctrl+C로 루프를 빠져나와도 반드시 여기를 지난다. 정리 자체가 실패해도 다른 리소스
        # 정리를 막지 않는다.
        try:
            sink.close()
        except Exception as e:  # noqa: BLE001 — 종료 경로에서 예외를 새로 만들지 않는다
            logger.warning("target sink 종료 실패(무시): %s", e)


if __name__ == "__main__":
    main()
