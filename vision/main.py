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

🔴 **기동(bind) 실패는 하드 페일이다 — 종료코드 3으로 즉시 죽는다**(2026-07-28 사용자 결정).
`--target-sink`를 **명시적으로 준** 실행에서 "켜달라고 했는데 조용히 안 켜지는" 상태로 계속
도는 것은 디버깅 중 최악이다. 십중팔구 포트 충돌이므로 stderr에 포트 번호와 함께 사유를 낸다.
(발행 **도중**의 예외는 반대로 절대 죽이지 않는다 — `_publish_to_sink` docstring 참조.)

또한 `--display`가 켜져 있으면(`window`/`stream`/`file`) 화면에 **유도 발행 상태 오버레이**
(소비자 수 / 마지막 seq / 드롭 수)를 상시 표시한다 — `--display`와 `--target-sink`가 완전히
독립이라 "화면은 뜨는데 유도 좌표는 아무 데도 안 나가는" 상태가 실제로 가능하기 때문이다.
소비자가 0명이면 빨간 큰 글씨로 경고한다. `--display none`이면 오버레이 비용은 0이다.

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
  # USB 웹캠(UVC) 모드 — CSI 카메라 사망 대응 임시 경로. picamera2/libcamera 불필요(랩탑에서도 돈다)
  python -m vision.main uvc --preset presets/distress_fine.yaml --calib <웹캠용 nominal.yaml>
  # 안정 장치경로 권장(/dev/video* 인덱스는 USB 재연결·부팅 순서에 따라 흔들린다)
  python -m vision.main uvc:/dev/v4l/by-id/usb-XXXX-video-index0 --uvc-resolution 1280x720
  절차 정본: docs/vision_webcam_fallback.md (화각 실측 → nominal.yaml 생성 → 실행)
"""
import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from vision.core.prior_geometry import (
    PRIOR_DETECTION_META_KEY,
    PRIOR_INPUTS_KEY,
)
from vision.core.runner import Pipeline
from vision.core.state_machine import (
    Decision,
    LandingSMConfig,
    LandingStateMachine,
    Observation,
    half_hfov_tan_from_calibration,
)
from vision.core.target import (
    TargetEstimate,
    marker_object_points,
    solve_target_pose,
)
from vision.utils.blackbox import BlackBoxLogger
from vision.utils.calibration_loader import CameraCalibration, load_camera_calibration
from vision.utils.frame_source import (
    AF_MODES,
    DEFAULT_AF_MODE,
    LENS_POSITION_MAX,
    LENS_POSITION_MIN,
    LiveFrameSource,
    UvcFrameSource,
)
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
from vision.utils.visualize import (
    save_result, draw_detections, draw_sink_status, draw_landing_overlay,
)


_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
_WINDOW_NAME = "Landing Zone Detector"
# ArUco Phase 4(docs/vision_aruco_branch.md) — 기본 카메라 캘리브레이션. --calib로 override 가능.
_DEFAULT_CALIB_PATH = str(Path(__file__).parent / "calibration" / "cam109-imx708af75" / "nominal.yaml")
# 라이브 모드(LiveFrameSource 배선) — `input` 위치인자 특수값 접두사. import 자체는 picamera2 없이도
# 항상 성공한다(LiveFrameSource.open() 내부 지연 import 덕분, vision/CLAUDE.md "frame_source.py" 절).
_LIVE_INPUT_SPEC = "live"
# USB 웹캠(UVC) 모드 — `input` 위치인자 특수값 접두사. 2026-07-30 CSI 카메라 하드웨어 사망
# (docs/vision_report_video.md §1) 대응 임시 경로. `live`와 달리 picamera2/libcamera를 전혀
# 쓰지 않으므로 랩탑 .venv에서도 그대로 돈다.
#   uvc            → /dev/video0 (인덱스 0)
#   uvc:2          → /dev/video2
#   uvc:/dev/v4l/by-id/usb-...-video-index0  → 안정 경로(권장, 인덱스는 재부팅마다 흔들린다)
_UVC_INPUT_SPEC = "uvc"
# --output 지정 시 라이브 모드 VideoWriter가 쓸 fps. VideoReader처럼 소스가 fps를 알려주지 않으므로
# (무한 실시간 스트림) 고정값을 쓴다 — 정밀한 재생속도가 목적이 아니라 결과 확인용 기록이라 충분.
#
# 🔴 **이 값이 실제 캡처 속도와 다르면 저장된 mp4가 배속으로 재생된다**(2026-07-29, 2차예선 제출
# 영상 준비 중 확인). 실측 캡처 속도는 해상도에 크게 좌우된다 — 1536x864에서 ~13.6fps,
# 4608x2592(기본)에서 **~4.4Hz**(`vision/CLAUDE.md` U5 실측)라 20.0을 그대로 쓰면 각각 약 1.5배·
# 4.5배 빨라진다. 그래서 (a) `--output-fps`로 명시할 수 있게 하고 (b) 종료 시 **실측 평균 fps를
# 계산해 기록 fps와 10% 이상 어긋나면 경고**한다(아래 `_LIVE_OUTPUT_FPS_WARN_RATIO`).
# 기본값 자체는 바꾸지 않았다 — 기존 녹화물/스크립트의 재생속도가 조용히 달라지면 안 되므로.
_LIVE_DEFAULT_OUTPUT_FPS = 20.0
# 실측 평균 fps가 기록 fps와 이 비율 이상 어긋나면 경고한다(0.10 = 10%).
_LIVE_OUTPUT_FPS_WARN_RATIO = 0.10

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

# `--target-sink` bind 실패 전용 종료코드. 2(argparse 사용법 오류)/1(입력 파일 없음)과 겹치지
# 않게 새로 뒀다 — 스크립트가 "포트 충돌로 못 떴다"를 다른 실패와 구분할 수 있어야 한다.
EXIT_SINK_BIND_FAILED = 3


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
    """JSONL 블랙박스용 축약. 사전정보 스코어러(`modules/prior_score.py`)가 켜졌을 때만
    `prior` 키가 추가된다 — 꺼져 있으면(=현재 라이브 경로 전부) 기존 JSONL 형태와 **한 글자도
    다르지 않다**(회귀테스트로 고정)."""
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


def _build_prior_inputs(calib: Optional[CameraCalibration], telemetry: Optional[dict] = None) -> dict:
    """`modules/prior_score.py::PriorGeometryScorer`가 소비할 프레임 단위 사전정보를 조립한다
    (`Pipeline.run(image, meta=...)` 통로. `replay.py`에 같은 헬퍼가 얇게 중복 — 두 CLI는 서로
    import하지 않는다는 기존 원칙).

    🔴 **`main.py` 라이브 경로에는 AGL/자세 역방향 채널이 아직 없다**(`_build_observation`이
    `agl_m=None`인 것과 같은 이유 — `docs/vision_fc_interface.md`, 다른 세션 작업 중). 그래서
    여기서는 캘리브레이션만 실리고 스코어러는 **항상 비활성으로 degrade**한다. 채널이 열리면
    `telemetry`에 `alt`/`attitude`가 실리는 것만으로 그대로 활성된다 — 이 함수는 안 고쳐도 된다.
    """
    if calib is None:
        return {}
    inputs: dict = {
        "camera_matrix": calib.camera_matrix,
        "dist_coeffs": calib.dist_coeffs,
        "calib_image_size": tuple(calib.image_size),
        "calib_id": calib.calib_id,
    }
    if isinstance(telemetry, dict):
        if telemetry.get("alt") is not None:
            inputs["agl_m"] = telemetry["alt"]
        if telemetry.get("attitude") is not None:
            inputs["attitude"] = telemetry["attitude"]
    return {PRIOR_INPUTS_KEY: inputs}


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


def _landing_sm_config(calib, logger) -> LandingSMConfig:
    """상태머신 config — 로드된 캘리브레이션이 있으면 `tan(HFOV/2)`를 인트린식에서 유도해 넣는다.

    `TERMINAL` 블라인드 하강의 이탈 추정(`drift_estimate`)이 정규화 오차를 미터로 환산할 때 쓰는
    계수다(`core/state_machine.py` 참조). 캘리브 로드 실패 시엔 nominal 폴백을 그대로 쓴다 —
    ArUco와 무관한 프리셋 실행까지 막지 않는다는 기존 판단과 같은 이유.
    `replay.py`에도 같은 헬퍼가 얇게 중복돼 있다(두 CLI는 서로 import하지 않는다 — 기존 원칙).
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

    🔴 **여기(발행)와 기동(bind)의 정책이 정반대인 이유 — 헷갈리면 안 된다.**
    `_make_target_sink()`의 bind 실패는 **즉사**(종료코드 3)인데 여기 발행 실패는 **무시하고
    계속**이다. 모순이 아니라 시점이 다르다:
      - **bind는 기동 시점 1회**다. 이때 죽으면 기체는 아직 지상에 있고, 실패의 뜻은 "켜달라고
        명시한 발행이 처음부터 안 켜졌다"는 설정 오류다 → 조용히 계속 도는 게 더 위험하다.
      - **발행은 비행 중 매 프레임**이다. 소비자가 끊겼다는 이유로 여기서 죽으면 **비행 중에
        검출이 통째로 멈춘다** → 착륙 유도보다 검출이 먼저 죽는, 정확히 막아야 할 일이다.
    즉 "기동 실패는 못 뜬 것이고, 발행 실패는 소비자 사정"이다.

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


def _draw_sink_overlay(annotated, sink: TargetSink) -> None:
    """유도 발행 상태를 화면(annotated 프레임)에 덧그린다 — **제자리 수정**.

    🔴 **호출자는 `display != "none"`일 때만 부른다.** 드론 기본 경로(`--display none`)에서
    오버레이 비용이 정확히 0이어야 하기 때문이다(§7.9 헤드리스 전제, 회귀테스트로 고정).

    **왜 화면에 띄우나(2026-07-28 사용자 결정):** `--display`와 `--target-sink`는 완전히
    독립이라 **화면은 멀쩡히 뜨고 검출도 되는데 유도 좌표는 아무 데도 안 나가는** 상태가 실제로
    가능하다. bind 실패는 기동 시점에 즉사시켜 막지만(`_make_target_sink`), **bind는 됐는데
    소비자가 0명**인 경우는 죽일 수 없다 — 시작 직후엔 소비자가 아직 안 붙는 게 정상이므로.
    그 사각지대를 화면에 상시 노출하는 것이 이 함수의 유일한 목적이다.

    `draw_detections()` **뒤에** 부르는 것이 계약이라 검출 결과를 훼손하지 않고(패널도 반투명),
    `_publish_to_sink()` **뒤에** 부르므로 이 프레임의 발행까지 반영된 seq가 보인다.
    """
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
        # `NullSink`(= `--target-sink` 미지정)도 "유도 좌표가 아무 데도 안 나간다"는 같은
        # 사실이므로 조용히 넘어가지 않는다 — 다만 운영자의 명시적 선택이라 색만 다르다.
        draw_sink_status(annotated, enabled=False)


def _draw_report_overlay(annotated, state, decision, estimate, frame_id: int) -> None:
    """착륙 판단(착륙점/흰 박스/상태머신/추정거리)을 화면에 덧그린다 — **제자리 수정**.

    🔴 **호출자는 `--report-overlay`가 켜졌을 때만 부른다.** 기본 경로에서는 비용이 정확히 0이고
    저장/스트림 산출물도 이 플래그 이전과 한 픽셀도 다르지 않다(회귀테스트로 고정) —
    `_draw_sink_overlay`가 `--display none`에서 비용 0인 것과 같은 원칙(§7.9).

    **왜 opt-in인가:** 이건 비행 중 필요한 정보가 아니라 **사람에게 보여줄 영상**을 위한 것이다
    (2차예선 보고 제출). 드론 기본 경로에 상시 비용을 얹을 이유가 없다.

    `draw_detections()`/`_draw_sink_overlay()` **뒤에** 부르는 것이 계약이다(패널 위치가 좌하단
    이라 sink 패널(좌상단)과 겹치지 않는다).
    """
    draw_landing_overlay(
        annotated,
        state.detections,
        state=decision.state.value if decision is not None else None,
        command=decision.command if decision is not None else None,
        estimate=estimate,
        frame_id=frame_id,
    )


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


def _parse_uvc_device(input_arg: str) -> Optional[Union[int, str]]:
    """`input` 위치인자가 USB 웹캠 스펙(`uvc` / `uvc:<index>` / `uvc:<경로>`)이면 device를
    반환하고, 아니면 None(다른 분기로 폴백).

    `uvc:` 뒤가 정수면 인덱스, 아니면 장치 경로 문자열로 넘긴다 — `_parse_live_camera_num`은
    정수만 받지만 여기서는 **경로를 허용해야 한다**. `/dev/video*` 인덱스는 USB 재연결·부팅
    순서에 따라 흔들리는데(이 저장소의 `/dev/mediaN` 전례, `rpi_capture.py` 2026-07-22d),
    `/dev/v4l/by-id/...` 안정 경로를 쓰면 그 문제가 원천 차단되기 때문이다.
    """
    if input_arg == _UVC_INPUT_SPEC:
        return 0
    prefix = _UVC_INPUT_SPEC + ":"
    if input_arg.startswith(prefix):
        suffix = input_arg[len(prefix):]
        if not suffix:
            raise ValueError(
                f"잘못된 웹캠 모드 스펙 — {input_arg!r} "
                f"(예: {_UVC_INPUT_SPEC}, {_UVC_INPUT_SPEC}:2, {_UVC_INPUT_SPEC}:/dev/video2)"
            )
        try:
            return int(suffix)
        except ValueError:
            return suffix  # 장치 경로
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
    report_overlay: bool = False,
) -> None:
    sink = sink if sink is not None else NullSink()
    image = load_image(str(input_path))
    t0 = time.perf_counter()
    state = pipeline.run(image, meta=_build_prior_inputs(calib))
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

    # `--display none`이면 주석 프레임 자체를 만들지 않는다(기존 동작) → 오버레이 비용도 0.
    # 단 `--report-overlay`는 "사람에게 보여줄 산출물"이 목적이라 `--display none + --output`
    # (헤드리스 녹화)에서도 그려야 의미가 있다 — 그래서 프레임 생성 조건에 함께 들어간다.
    annotated = None
    if display != "none" or report_overlay:
        annotated = draw_detections(state.original, state.detections, state.confirmed)
        if display != "none":
            _draw_sink_overlay(annotated, sink)
        if report_overlay:
            _draw_report_overlay(annotated, state, decision, estimate, 0)

    if output:
        if annotated is not None:
            # 화면에 보이는 것과 **같은 프레임**을 파일로 남긴다 — `--display file`은 창이
            # 없어서 이 파일이 곧 "화면"이고, 유도 발행 상태가 거기 안 보이면 의미가 없다.
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output, annotated)
        else:
            save_result(state, output)  # --display none: 기존 경로 그대로(오버레이 없음)
        print(f"Saved: {output}")

    if annotated is not None:
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
    report_overlay: bool = False,
    output_fps: Optional[float] = None,
) -> None:
    from vision.utils.video_reader import VideoReader

    sink = sink if sink is not None else NullSink()
    writer = None
    frame_count = 0
    with VideoReader(str(input_path)) as reader:
        for frame in reader:
            t0 = time.perf_counter()
            state = pipeline.run(frame, meta=_build_prior_inputs(calib))
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

            # 발행 뒤에 그려야 이 프레임의 seq까지 반영된다. `none`이면 비용 0.
            if display != "none":
                _draw_sink_overlay(annotated, sink)
            if report_overlay:
                # frame_count는 바로 위에서 증가했으므로 이 프레임의 id는 -1이다.
                _draw_report_overlay(annotated, state, decision, estimate, frame_count - 1)

            if streamer is not None:
                streamer.push_frame(annotated)  # 비차단(§7.9 비침습 전제) — 파이프라인 루프를 지연시키지 않음

            if output:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        output, fourcc, output_fps or reader.fps or 30, (w, h)
                    )
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


def _measure_capture_fps(
    first_ts: Optional[float], last_ts: Optional[float], frame_count: int
) -> Optional[float]:
    """기록된 프레임들의 **실제** 평균 캡처 fps. 못 재면 None(경고를 지어내지 않는다).

    `frame_count`개 프레임의 시각 스팬은 **간격이 `frame_count - 1`개**다 — `frame_count`로
    나누면 fps를 체계적으로 과소평가한다(그러면 "느리다"는 거짓 경고가 상시 뜬다).
    프레임이 2개 미만이거나 시각이 단조증가하지 않으면(같은 ts 반복 등) 잴 수 없으므로 None.
    """
    if first_ts is None or last_ts is None or frame_count < 2:
        return None
    span = last_ts - first_ts
    if span <= 0:
        return None
    return (frame_count - 1) / span


def _run_source_loop(
    pipeline: Pipeline,
    source_factory,
    mode_label: str,
    device_label: str,
    output: str | None,
    display: str,
    logger,
    blackbox: BlackBoxLogger,
    streamer: MjpegStreamer | None,
    calib: Optional[CameraCalibration],
    state_machine: Optional[LandingStateMachine],
    sink: Optional[TargetSink],
    stop_event: threading.Event,
    on_open=None,
    report_overlay: bool = False,
    output_fps: Optional[float] = None,
) -> None:
    """무한 프레임 소스(실카메라 CSI / USB 웹캠) 공용 프레임 루프.

    `_run_video`와 거의 동일하되 소스가 무한 이터레이터라는 점만 다르다. 헤드리스
    (`--display none`)에서는 무한정 도는 게 정상 동작이라 Ctrl+C(KeyboardInterrupt)와 SIGTERM
    둘 다 종료 수단이다 — `with source_factory()`가 예외 전파 중에도 카메라를 release하고,
    바깥 try/except가 스택트레이스 없이 조용히 종료시킨다(로그만 남김). SIGTERM은 비대화형
    배포(systemd 등) 표준 종료 신호라 호출자가 넘긴 `stop_event`를 다음 프레임 경계에서 확인해
    루프를 빠져나간다(§h264_stream.py와 동일 근거 — SIGINT는 비대화형 자식에서 못 믿음).

    2026-07-30 `_run_live`에서 추출됐다 — CSI 카메라 하드웨어 사망으로 USB 웹캠 경로
    (`_run_uvc`)가 생기면서, 검증된 이 루프를 복제하는 대신 공유한다. 소스 종류에 따라 다른
    건 **생성 방법과 열린 직후 로깅뿐**이라 그 둘만 `source_factory`/`on_open`으로 뺐다.

    🔴 **`sink.install_signal_handlers()`를 여기서 부르면 안 된다.** `signal.signal`은 신호당
    핸들러가 **하나**뿐이라, sink 핸들러를 나중에 걸면 호출자의 `_install_sigterm_handler`가
    등록한 핸들러를 **덮어써서** `stop_event`가 영영 세팅되지 않는다 — 실기체에서만 드러났던
    SIGTERM graceful shutdown 버그가 그대로 재발한다. sink 정리는 `main()`의 `finally`에서
    `sink.close()`로 하면 충분하다(루프를 빠져나오면 반드시 거기로 간다).
    """
    sink = sink if sink is not None else NullSink()
    writer = None
    frame_count = 0
    # mp4에 기록할 fps. `--output-fps` 미지정이면 기존 고정값 그대로(동작 무변경).
    written_fps = output_fps or _LIVE_DEFAULT_OUTPUT_FPS
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    try:
        with source_factory() as source:
            if on_open is not None:
                on_open(source)
            for record in source:
                if stop_event.is_set():
                    logger.info(
                        "라이브 모드 SIGTERM으로 종료 요청 — %d 프레임 처리 후 정상 종료",
                        frame_count,
                    )
                    break
                t0 = time.perf_counter()
                state = pipeline.run(
                    record.image, meta=_build_prior_inputs(calib, record.telemetry)
                )
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

                # 발행 뒤에 그려야 이 프레임의 seq까지 반영된다. `none`이면 비용 0.
                if display != "none":
                    _draw_sink_overlay(annotated, sink)
                if report_overlay:
                    # frame_count는 바로 위에서 증가했으므로 이 프레임의 id는 -1이다.
                    _draw_report_overlay(annotated, state, decision, estimate, frame_count - 1)

                if streamer is not None:
                    streamer.push_frame(annotated)  # 비차단(§7.9 비침습 전제)

                if output:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output, fourcc, written_fps, (w, h))
                    writer.write(annotated)
                    # 실측 재생속도 검증용 — 첫/마지막 프레임 시각만 들고 있으면 충분하다.
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts

                if display == "window" and not _show_window(annotated, wait=1):
                    break
    except KeyboardInterrupt:
        logger.info(
            "%s Ctrl+C로 종료 요청 — %d 프레임 처리 후 정상 종료", mode_label, frame_count,
        )
    finally:
        if writer:
            writer.release()
            print(f"Saved: {output}  ({frame_count} frames)")
            measured = _measure_capture_fps(first_ts, last_ts, frame_count)
            if measured is not None:
                msg = (
                    f"기록 fps={written_fps:.2f} / 실측 캡처 fps={measured:.2f}"
                )
                if abs(measured - written_fps) > written_fps * _LIVE_OUTPUT_FPS_WARN_RATIO:
                    # 🔴 이걸 모르고 제출하면 영상이 배속으로 재생된다. 다시 돌릴 명령까지 준다.
                    logger.warning(
                        "%s — 저장된 mp4가 실제보다 %.2f배 빠르게 재생된다. "
                        "정확한 속도가 필요하면 `--output-fps %.2f`로 다시 녹화할 것.",
                        msg, written_fps / measured, measured,
                    )
                    print(f"WARNING: {msg} → --output-fps {measured:.2f} 권장")
                else:
                    logger.info("%s (일치)", msg)
        logger.info("%s 종료: %d 프레임 처리", device_label, frame_count)
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
    sink: Optional[TargetSink] = None,
    af_mode: Optional[str] = DEFAULT_AF_MODE,
    lens_position: Optional[float] = None,
    report_overlay: bool = False,
    output_fps: Optional[float] = None,
) -> None:
    """CSI 실카메라 라이브 모드(picamera2). 프레임 루프는 `_run_source_loop` 공유."""
    live_kwargs: dict = {
        "camera_num": camera_num,
        "retries": retries,
        "retry_delay": retry_delay,
        "af_mode": af_mode,
        "lens_position": lens_position,
    }
    if resolution is not None:
        live_kwargs["resolution"] = resolution

    stop_event = threading.Event()
    _install_sigterm_handler(stop_event)

    def _log_af(source) -> None:
        # AF 적용 결과를 사람로그에 남긴다 — 실패해도 카메라는 드라이버 기본 초점으로 계속
        # 돌기 때문에(§LiveFrameSource._apply_af), 로그가 없으면 "초점이 왜 이러지"를
        # 현장에서 추적할 방법이 없다.
        if source.af_error is not None:
            logger.warning(
                "AF 설정 실패(드라이버 기본 초점으로 계속 진행) af_mode=%s: %s",
                af_mode, source.af_error,
            )
        elif source.af_applied:
            logger.info("AF 설정 완료 af_mode=%s lens_position=%s", af_mode, lens_position)
        else:
            logger.info("AF 미개입(af_mode=None) — 드라이버 기본 동작 유지")

    _run_source_loop(
        pipeline,
        lambda: LiveFrameSource(**live_kwargs),
        "라이브 모드",
        f"live camera_num={camera_num}",
        output, display, logger, blackbox, streamer, calib, state_machine, sink,
        stop_event,
        on_open=_log_af,
        report_overlay=report_overlay,
        output_fps=output_fps,
    )


def _run_uvc(
    pipeline: Pipeline,
    device: Union[int, str],
    resolution: Optional[Tuple[int, int]],
    fps: Optional[float],
    fourcc: Optional[str],
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
    allow_resolution_mismatch: bool = False,
    autofocus: Optional[bool] = None,
    focus: Optional[float] = None,
    report_overlay: bool = False,
    output_fps: Optional[float] = None,
) -> None:
    """USB 웹캠(UVC) 라이브 모드 — 2026-07-30 CSI 카메라 하드웨어 사망 대응 임시 경로.

    프레임 루프는 `_run_live`와 완전히 동일한 `_run_source_loop`를 쓴다(검증된 루프 재사용).
    다른 건 소스 생성과 열린 직후 로깅뿐이다.
    """
    uvc_kwargs: dict = {
        "device": device,
        "retries": retries,
        "retry_delay": retry_delay,
        "allow_resolution_mismatch": allow_resolution_mismatch,
        "autofocus": autofocus,
        "focus": focus,
    }
    if resolution is not None:
        uvc_kwargs["resolution"] = resolution
    if fps is not None:
        uvc_kwargs["fps"] = fps
    if fourcc is not None:
        uvc_kwargs["fourcc"] = fourcc

    stop_event = threading.Event()
    _install_sigterm_handler(stop_event)

    def _log_uvc(source) -> None:
        # 실제로 무엇이 걸렸는지가 진실이다 — 요청값을 로그에 남기면 현장에서 오독한다.
        logger.info(
            "웹캠 열림 device=%s 실제해상도=%s 실제fps=%s fourcc=%s",
            device, source.actual_resolution, source.actual_fps, fourcc,
        )
        if source.actual_resolution != tuple(source.resolution):
            # allow_resolution_mismatch로 통과한 경우 — 조용히 넘어가면 안 된다.
            logger.warning(
                "요청 해상도 %s 와 실제 %s 가 다르다(--uvc-allow-resolution-mismatch로 허용됨) "
                "— 캘리브레이션이 실제 해상도 기준인지 반드시 확인할 것.",
                tuple(source.resolution), source.actual_resolution,
            )
        if source.control_error is not None:
            logger.warning("웹캠 초점 컨트롤 미적용(그대로 진행): %s", source.control_error)

    _run_source_loop(
        pipeline,
        lambda: UvcFrameSource(**uvc_kwargs),
        "USB 웹캠 모드",
        f"uvc device={device}",
        output, display, logger, blackbox, streamer, calib, state_machine, sink,
        stop_event,
        on_open=_log_uvc,
        report_overlay=report_overlay,
        output_fps=output_fps,
    )


def _make_target_sink(args, logger) -> TargetSink:
    """`--target-sink` 미지정이면 `NullSink`, 지정이면 기동된 `SocketTargetSink`.

    🔴 **기동(bind) 실패는 하드 페일이다 — 종료코드 `EXIT_SINK_BIND_FAILED`(3)로 즉시 죽는다.**
    (2026-07-28 사용자 결정. 이전 구현은 `NullSink`로 강등하고 검출을 계속했다 — 뒤집혔다.)

    근거:
    - `--target-sink`는 **명시적으로 켜달라고 준 플래그**다. "켰는데 조용히 안 켜진 채로 계속
      도는" 상태는 디버깅 중 최악이다 — 화면도 뜨고 로그도 돌고 검출도 되는데 정작 유도
      좌표만 아무 데도 안 나간다(이 비대칭이 바로 사용자가 지목한 위험이다).
    - `utils/target_sink.py::start()`가 이미 "포트 충돌을 조용히 삼키면 안 된다"고 판단해
      예외를 올린다 — 이제 `main.py`가 그 판단을 **뒤집지 않고 그대로 존중**한다.
    - `--display stream`(MjpegStreamer)이 예외를 그대로 올리는 기존 관례와도 일치한다.

    ⚠️ **이것은 발행(publish) 실패와 정책이 정반대이며, 그래야 한다** — bind는 기동 시점 1회뿐
    이라 죽어도 지상이지만, 발행은 비행 중 매 프레임이라 거기서 죽으면 검출이 통째로 멈춘다
    (`_publish_to_sink` docstring "여기(발행)와 기동(bind)의 정책이 정반대인 이유" 참조).

    ⚠️ **`--target-sink` 미지정 기본 경로는 조금도 달라지지 않는다** — 소켓을 바인드하지도,
    이 하드 페일 경로를 지나지도 않는다(회귀테스트로 고정).
    """
    if not args.target_sink:
        return NullSink()
    sink = SocketTargetSink(host=args.target_sink_host, port=args.target_sink_port)
    try:
        sink.start()
    except OSError as e:
        endpoint = f"{args.target_sink_host}:{args.target_sink_port}"
        logger.error("target sink 기동 실패 (%s) — 하드 페일로 종료합니다: %s", endpoint, e)
        # stderr에 **원인이 한눈에 보이게** 남긴다. 십중팔구 포트 충돌이므로 포트 번호와
        # 확인 명령까지 같이 준다(원격 세션이 재조사 없이 바로 진단할 수 있게).
        print(
            f"Error: --target-sink 기동 실패 — {endpoint} 에 bind 하지 못했습니다: {e}\n"
            f"       십중팔구 포트 {args.target_sink_port} 충돌입니다. "
            f"점유 프로세스 확인: ss -ltnp | grep {args.target_sink_port}\n"
            f"       --target-sink 를 명시했으므로 조용히 강등하지 않고 즉시 종료합니다 "
            f"(exit {EXIT_SINK_BIND_FAILED}). 다른 포트: --target-sink-port <N>",
            file=sys.stderr,
        )
        sys.exit(EXIT_SINK_BIND_FAILED)
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
    # USB 웹캠(UVC) 모드 — 2026-07-30 CSI 카메라 하드웨어 사망 대응. `--live-retries`/
    # `--live-retry-delay`는 의미가 같아 웹캠 모드에서도 그대로 재사용한다(같은 노브를 두 벌로
    # 나누면 현장에서 어느 쪽이 먹는지 헷갈린다).
    parser.add_argument(
        "--uvc-resolution",
        type=_parse_resolution,
        default=None,
        metavar="WxH",
        help="USB 웹캠 모드(`input`이 uvc/uvc:N/uvc:경로) 해상도. 기본 1920x1080. "
             "🔴 요청과 실제가 다르면 하드 실패한다(캘리브레이션이 조용히 어긋나는 것 방지) — "
             "지원 조합은 `v4l2-ctl -d <device> --list-formats-ext`로 확인할 것.",
    )
    parser.add_argument(
        "--uvc-fps", type=float, default=None,
        help="USB 웹캠 요청 fps (기본 30). 카메라가 광고만 하고 안 지키는 경우가 흔해 "
             "검증하지 않고 실제값을 로그에 남긴다.",
    )
    parser.add_argument(
        "--uvc-fourcc", default=None, metavar="MJPG",
        help="USB 웹캠 픽셀 포맷 4글자 코드 (기본 MJPG). USB 2.0에서 FHD 30fps는 MJPEG "
             "압축이 있어야만 나온다 — 무압축 YUYV면 카메라가 조용히 저fps로 떨어진다.",
    )
    parser.add_argument(
        "--uvc-allow-resolution-mismatch", action="store_true",
        help="요청 해상도가 안 걸려도 진행한다. 🔴 그 실제 해상도용 nominal.yaml을 먼저 만든 "
             "뒤에만 쓸 것 — 아니면 solvePnP 거리·pose가 소리 없이 틀린다.",
    )
    parser.add_argument(
        "--uvc-autofocus", choices=["on", "off"], default=None,
        help="USB 웹캠 오토포커스. 미지정이면 손대지 않는다(기본). 기체는 10~40m를 보므로 "
             "초점이 무한대 근처에 고정되는 편이 낫지만, 싸구려 웹캠은 대개 고정초점이라 "
             "이 컨트롤 자체가 없다(그 경우 경고만 남기고 계속 진행).",
    )
    parser.add_argument(
        "--uvc-focus", type=float, default=None,
        help="수동 초점값, --uvc-autofocus off 전용. 드라이버마다 단위가 달라 실측으로 찾는다.",
    )
    # AF 제어(2026-07-28) — `tools/h264_stream.py`와 같은 인자 이름/의미를 쓴다(현장에서 두
    # 도구를 번갈아 쓰므로 이름이 갈리면 헷갈린다). 🔴 실기체 미검증.
    parser.add_argument(
        "--af-mode", choices=[*AF_MODES, "none"], default=DEFAULT_AF_MODE,
        help=f"라이브 모드 오토포커스 (기본 {DEFAULT_AF_MODE}). "
             "none = AF에 손대지 않음(드라이버 기본 동작 — 2026-07-28 이전 동작)",
    )
    parser.add_argument(
        "--lens-position", type=float, default=None,
        help=f"디옵터, --af-mode manual 전용 ({LENS_POSITION_MIN}~{LENS_POSITION_MAX}). "
             "드라이버가 광고하는 상한 32.0은 실측 하드클램프로 무효",
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
        "--report-overlay",
        action="store_true",
        help="착륙 판단(착륙점/흰 박스/상태머신 상태·명령/추정거리)을 프레임에 그린다(기본 꺼짐). "
             "사람에게 보여줄 영상(보고/발표)용 — `--output`과 함께 쓰면 저장 파일에 그대로 남는다. "
             "드론 기본 경로에서는 켜지 않는다(비용 0).",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="--output mp4에 기록할 fps. 미지정 시 라이브는 고정값 "
             f"{_LIVE_DEFAULT_OUTPUT_FPS}, 영상 입력은 원본 fps. 실측 캡처 속도와 다르면 "
             "저장 영상이 배속으로 재생되므로, 라이브 녹화 종료 시 실측값과 어긋나면 경고가 뜬다.",
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

    if args.uvc_focus is not None and args.uvc_autofocus != "off":
        # 오토포커스가 켜진 채 수동 초점값을 주면 드라이버가 곧바로 덮어쓴다 — 조용히 무시되는
        # 인자를 남기지 않는다(UvcFrameSource 생성자와 같은 규칙, 여기서 먼저 걸어 CLI 오류로).
        print(
            "Error: --uvc-focus 는 --uvc-autofocus off 와 함께 써야 합니다.", file=sys.stderr
        )
        sys.exit(2)

    try:
        live_camera_num = _parse_live_camera_num(args.input)
        uvc_device = _parse_uvc_device(args.input)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    input_path: Optional[Path] = None
    if live_camera_num is None and uvc_device is None:
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
    state_machine = LandingStateMachine(_landing_sm_config(calib, logger))

    # vision↔fc 인터페이스(§9 V5). **기본은 NullSink** — `--target-sink`를 명시하지 않으면
    # 소켓도 스레드도 뜨지 않고 레코드 조립도 일어나지 않는다(기존 실행 경로 완전 무변경).
    sink: TargetSink = NullSink()

    streamer = None
    try:
        # 🔴 이 호출은 반드시 try **안**이어야 한다 — bind 실패는 `SystemExit`로 즉사하는데
        # (하드 페일), 그때도 아래 finally가 blackbox 큐 스레드/파일 핸들을 정리해야 한다
        # (`test_streamer_start_failure_still_closes_blackbox`가 세운 리소스 leak 계약).
        # `sink`는 위에서 이미 `NullSink`로 초기화돼 있어 finally의 `sink.close()`도 안전하다.
        sink = _make_target_sink(args, logger)

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
                af_mode=(None if args.af_mode == "none" else args.af_mode),
                lens_position=args.lens_position,
                report_overlay=args.report_overlay,
                output_fps=args.output_fps,
            )
        elif uvc_device is not None:
            _run_uvc(
                pipeline, uvc_device, args.uvc_resolution, args.uvc_fps, args.uvc_fourcc,
                args.live_retries, args.live_retry_delay, args.output, args.display, logger,
                blackbox, streamer, calib, state_machine, sink,
                allow_resolution_mismatch=args.uvc_allow_resolution_mismatch,
                autofocus=(None if args.uvc_autofocus is None
                           else args.uvc_autofocus == "on"),
                focus=args.uvc_focus,
                report_overlay=args.report_overlay,
                output_fps=args.output_fps,
            )
        elif input_path.suffix.lower() in _VIDEO_SUFFIXES:
            _run_video(
                pipeline, input_path, args.output, args.display, logger, blackbox, streamer, calib,
                state_machine, sink,
                report_overlay=args.report_overlay,
                output_fps=args.output_fps,
            )
        else:
            _run_image(
                pipeline, input_path, args.output, args.display, logger, blackbox, streamer, calib,
                state_machine, sink,
                report_overlay=args.report_overlay,
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
