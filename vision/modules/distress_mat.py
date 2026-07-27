import numpy as np
from vision.core.state import VisionState
from vision.core.target import (
    DISTRESS_MAT_PLATFORM_HEIGHT_M,
    DISTRESS_MAT_SIZE_M,
    MAT_PLANE_REFERENCE,
    order_quad_corners_clockwise,
)


class DistressMatGeometry:
    """
    ② 조난자 구역(초록 매트) — **상대 pose 산출에 필요한 기하 정보를 detections에 태깅**한다.
    `docs/vision_plan.md` §5.3 / `docs/vision_fc_interface.md`(정밀착륙 인터페이스).

    ## 이 모듈이 존재하는 이유 (해결하는 갭)

    정밀착륙 인터페이스(`--target-sink`)는 `TargetEstimate`(기체 기준 상대 pose)를 요구하는데,
    그걸 만드는 경로가 **ArUco 전용**이었다: 버티포트는 50cm 마커가 있어 `solvePnP`가 돌지만,
    초록구역은 피듀셜이 없어 **검출은 되는데 기체에 보낼 좌표가 없었다**(상태머신은
    `CENTER_DESCEND`를 내는데 `position_flu`는 항상 null). 이 모듈이 그 빈칸을 채운다.

    ## 왜 `solvePnP`인가 (고른 접근과 버린 안)

    - ✅ **매트 4코너 + 알려진 실측 크기(3.0m×3.0m) → `solvePnP`.** 채택.
      `core/target.py::marker_object_points(size_m)`가 애초에 크기를 인자로 받으므로 ArUco와
      **똑같은 기계장치**를 그대로 재사용한다 — 새 pose 파이프라인을 발명하지 않는다.
      결정적 이점: **AGL(라이다 고도)이 필요 없다.** 알려진 크기가 스케일을 준다.
    - ❌ **"픽셀 → AGL로 역투영"**(고도를 알면 GSD가 나오니 픽셀 오프셋을 미터로 환산). 기각 —
      `main.py`는 지금 AGL을 **받는 경로 자체가 없다**(`_build_observation(agl_m=None)`,
      `replay.py`만 telemetry에서 읽는다). 라이브에서 항상 None이라 산출이 아예 불가능하다.
    - ❌ **호모그래피로 매트 평면만 풀고 자세는 포기.** 기각 — `solvePnP`가 같은 입력에서
      회전까지 주는데 정보를 일부러 버릴 이유가 없고, 산출물 형식(`TargetEstimate`)이
      quaternion 자리를 이미 요구한다.

    ## 왜 `modules/`이고, 왜 pose를 여기서 안 푸는가

    `solve_target_pose()`를 부르려면 `utils/calibration_loader.py`가 필요한데 import 규칙
    (`vision/CLAUDE.md`)이 **`modules/ ← vision.core 만`** 이라 위반이 된다(ArUco Phase 4가
    같은 이유로 배선을 `main.py` 레벨에 둔 전례). 그래서 역할을 쪼갠다:

        modules/distress_mat.py  : 프레임 안에서 끝나는 순수 기하(코너 정규화·착륙점 픽셀 확정)
        main.py                  : calib 로드 + solvePnP 호출 + TargetEstimate 조립

    **이 모듈이 preset yaml의 스텝인 것 자체가 "pose 산출기 선택" 기전이다.** `main.py`가 preset
    경로 문자열을 파싱하지 않고도(그건 깨지기 쉬운 관례다) `meta["distress_mat"]` 존재 여부만
    보고 초록구역 경로를 고를 수 있다. 덕분에 같은 `rect_detector`를 쓰는 범용 프리셋
    (`video.yaml`)이 실수로 "3m 매트" 크기를 적용받는 일이 구조적으로 불가능하다.

    ## 착륙점은 매트 중심이 아니다 (§5.3 기존 규약 준수)

    흰 박스는 *랜드마크*고 착륙 목표는 **"박스 옆 빈 초록면"** 이다. 그 픽셀은
    `modules/distress_box.py::WhiteBoxDetector`가 이미 `meta["white_box_detector"]
    ["landing_point_px"]`로 확정해 둔다(매트 bbox 네 모서리 중 박스 중심에서 가장 먼 것을
    매트 중심 쪽으로 `interior_margin_ratio`만큼 당긴 점). **이 모듈은 그 규약을 다시 구현하지
    않고 산출물을 그대로 소비한다** — 규칙이 두 곳에 복사되면 한쪽만 고쳐졌을 때 조용히
    어긋나고, `interior_margin_ratio` 기본값이 이중으로 존재하게 된다.

    - fine(`distress_fine.yaml`, 흰 박스 확정됨) → `landing_point_px`를 그대로 쓴다.
    - coarse(`distress_coarse.yaml`, 흰 박스 단계 없음) → **매트 중심**으로 우아하게 degrade하고
      `landing_point_source="mat_center"`로 그 사실을 밝힌다. coarse 대역(40~15m)에서는 박스가
      애초에 해상되지 않으므로 이게 물리적으로 옳고, 상태머신도 `fine_locked=False`라
      `CENTER_DESCEND` 위로 못 올라간다(=매트 중심으로 접근만 하는 단계). **소비자가
      "coarse 중심"과 "fine 착륙점"을 구분할 수 있어야** 하므로 `main.py`가 `target_type`을
      각각 `distress_mat_center` / `distress_landing_point`로 다르게 붙인다.

    부수 효과로 이 착륙점 픽셀은 `main.py::_build_observation()`이 `center_error_norm`을 재는
    바로 그 픽셀과 **같다** — 상태머신이 "중앙에 맞췄다"고 판단한 점과 기체에 보내는 좌표가
    가리키는 점이 어긋나지 않는다.

    ## 코너 순서 (조용한 오작동 1순위)

    `RectDetector`의 코너는 `cv2.approxPolyDP()` 출력이라 **순서가 보장되지 않는다** — 실측
    (골든셋 `distress/10m`)에서 실제로 `TL→BL→BR→TR`(반시계)로 나왔고, 이걸 그대로
    `marker_object_points()`(시계)와 짝지으면 감김이 반대라 거울상 대응이 된다.
    `order_quad_corners_clockwise()`가 감김과 시작점을 고정한다. 회전 4중 모호성이 착륙점
    좌표에 영향을 주지 않는 이유는 `core/target.py::project_pixel_onto_target_plane()` 참조.

    매직넘버 금지 — 임계값은 전부 `__init__` 파라미터(→ preset yaml에서 조정 가능).
    """

    def __init__(
        self,
        size_m: float = DISTRESS_MAT_SIZE_M,
        platform_height_m: float = DISTRESS_MAT_PLATFORM_HEIGHT_M,
    ):
        self.size_m = float(size_m)
        self.platform_height_m = float(platform_height_m)

    def __call__(self, state: VisionState) -> VisionState:
        tagged = 0
        skipped = 0
        skip_reasons: list[str] = []

        for det in state.detections:
            if det.corners is None or len(det.corners) != 4:
                skipped += 1
                skip_reasons.append("no_quad_corners")
                continue
            try:
                ordered = order_quad_corners_clockwise(det.corners)
            except ValueError:
                skipped += 1
                skip_reasons.append("degenerate_quad")
                continue

            white_box = det.meta.get("white_box_detector") or {}
            landing_px = white_box.get("landing_point_px")
            if landing_px is not None:
                landing_point_px = [float(landing_px[0]), float(landing_px[1])]
                landing_point_source = "white_box_landing_point"
            else:
                # coarse degrade — 매트 중심(정규화된 4코너의 무게중심). solvePnP의 tvec과 같은
                # 점이라 main.py는 역투영 없이 tvec을 그대로 쓸 수 있다.
                landing_point_px = [float(v) for v in np.asarray(ordered).mean(axis=0)]
                landing_point_source = "mat_center"

            det.meta["distress_mat"] = {
                "size_m": self.size_m,
                "corners_px": [[float(x), float(y)] for x, y in ordered],
                "landing_point_px": landing_point_px,
                "landing_point_source": landing_point_source,
                # z=0 평면이 지면이 아니라 매트 **윗면**이라는 사실을 소비자까지 전파한다
                # (`core/target.py::MAT_PLANE_REFERENCE` docstring — 라이다 AGL과 섞으면
                # 정확히 platform_height_m 만큼 어긋난다).
                "plane_reference": MAT_PLANE_REFERENCE,
                "platform_height_m": self.platform_height_m,
            }
            tagged += 1

        state.meta["distress_mat_geometry"] = {
            "tagged": tagged,
            "skipped": skipped,
            "skip_reasons": skip_reasons,
        }
        return state
