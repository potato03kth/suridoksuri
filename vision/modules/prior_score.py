"""
사전정보 기반 후보 랭킹 파이프라인 스텝 (`core/prior_geometry.py`의 순수 기하를 소비).

## 계약 — 🔴 **거절하지 않는다**

- `state.detections`에서 **항목을 절대 제거하지 않는다.** 점수를 붙이고(`Detection.meta`)
  **순서를 재정렬**하는 것까지가 최대다. 개수는 입출력이 항상 같다.
- 기존 `rect_detector`의 `min_area`/`max_area` 하드 필터를 **대체하지 않는다** — 그 뒤에
  랭킹만 얹는다.
- 점수가 아무리 낮아도 후보는 살아남는다(점수 함수가 구조적으로 0에 닿지 않는다,
  `core/prior_geometry.py::_log_gaussian`).

근거는 사용자 기조 **"좋은 비행 시퀀스는 실패하지 않는 시퀀스"** — 진짜 타겟을 놓치는 것이
오탐을 하나 더 보는 것보다 훨씬 나쁘다. 여기에 하드 게이트를 걸면 AGL/자세가 틀렸을 때
**검출률이 통째로 0**이 될 수 있다.

## 자동 비활성 (우아한 degrade)

사전정보(AGL·자세·인트린식) 중 **하나라도 없으면 아무 일도 하지 않는다** — 점수도 안 붙이고
정렬도 안 한다. 그래서 스텝이 프리셋에 있든 없든 **입력이 없을 때의 검출 결과는 완전히 동일**
하다(세션 요구 2번, 회귀테스트로 고정). 비활성 사유는 `state.meta`에 남겨 조용히 사라지지
않게 한다(§7.4 "거절 이유 로깅").

현재 저장소에서 자세 텔레메트리를 채우는 경로는 **`replay.py`의 `telemetry.jsonl` 하나뿐**
이다(`main.py` 라이브 경로에는 AGL/자세 역방향 채널이 아직 없다 — 다른 세션 작업 중). 즉
**기본값은 비활성**이고, 이것이 의도된 안전 상태다.

## 왜 `modules/`이고 왜 순수 기하는 `core/`인가 (배치 근거)

`vision/CLAUDE.md` import 규칙:

    core/     ← numpy, opencv만. 다른 vision 서브패키지 import 금지.
    modules/  ← vision.core 만 import. 다른 modules 파일 import 금지.

- **순수 기하(`core/prior_geometry.py`)**: 파일 I/O·상태·프레임 개념 없는 함수 묶음 →
  `core/target.py`·`core/frames.py`와 정확히 같은 성격이라 `core/`가 맞다. 덕분에 "AGL과
  자세를 인자로 받는 순수 함수"라는 요구가 구조적으로 강제된다 — 나중에 라이브 경로가
  열리면 입력 소스만 바꿔 그대로 꽂힌다.
- **파이프라인 스텝(이 파일)**: `__call__(VisionState) -> VisionState` 인터페이스 + preset
  yaml 파라미터 + registry 등록이 필요한 어댑터 → `modules/`.
- **캘리브레이션 로드는 여기서 하지 않는다.** `utils/calibration_loader`를 `modules/`에서
  import하면 규칙 위반이다 — `main.py`/`replay.py`가 로드해 `state.meta[PRIOR_INPUTS_KEY]`로
  **객체만** 넘긴다(ArUco Phase 4·`distress_mat.py`가 같은 이유로 같은 배선을 쓴 전례).

## `state.meta[PRIOR_INPUTS_KEY]` 스키마 (호출자가 심는다)

    {
      "agl_m": 12.0,                        # 라이다 AGL [m], 필수
      "attitude": VehicleAttitude | dict,   # 필수. dict면 attitude_from_telemetry 스키마
      "camera_matrix": (3,3) array-like,    # 필수
      "dist_coeffs": (5,) array-like,       # 필수
      "calib_image_size": (width, height),  # 필수 — 프레임 해상도 정합 확인용
      "calib_id": "…",                      # 선택(진단)
    }

🔴 **캘리브레이션 해상도 ≠ 프레임 해상도면 비활성**한다. focal은 해상도에 비례하는 값이라
그대로 쓰면 예상 면적이 통째로 틀린다. 조용히 재스케일하지 않는 것은 기존 판단과 같다
(다운스케일인지 크롭인지 알 수 없다 — `main.py::_solve_distress_mat_estimate` 주석 참조).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from vision.core.frames import MOUNT_YAW_PSI_M_MEASURED, MOUNT_YAW_PSI_M_RAD_DEFAULT
from vision.core.prior_geometry import (
    DEFAULT_ANISOTROPY_MARGIN,
    DEFAULT_AREA_TOLERANCE_HIGH,
    DEFAULT_AREA_TOLERANCE_LOW,
    DEFAULT_AREA_WEIGHT,
    DEFAULT_ATTITUDE_RESIDUAL_RAD,
    DEFAULT_SHAPE_TOLERANCE,
    DEFAULT_SHAPE_WEIGHT,
    PRIOR_DETECTION_META_KEY,
    PRIOR_INPUTS_KEY,
    PRIOR_META_KEY,
    VehicleAttitude,
    attitude_from_telemetry,
    polygon_area_px2,
    score_candidate,
)
from vision.core.state import VisionState
from vision.core.target import DISTRESS_MAT_SIZE_M

# 🔴 키 3종은 `core/prior_geometry.py`가 **단일 출처**다(main.py/replay.py는 modules를
# import할 수 없으므로 — 그쪽 주석 참조). 여기서는 재노출만 한다.
DETECTION_META_KEY = PRIOR_DETECTION_META_KEY

# 비활성 사유 문자열 — 매직 문자열을 흩뿌리지 않는다(§7.3).
REASON_OK = "ok"
REASON_NO_PRIOR_INPUTS = "no_prior_inputs"
REASON_NO_AGL = "no_agl"
REASON_NO_ATTITUDE = "no_attitude"
REASON_NO_CALIBRATION = "no_calibration"
REASON_CALIB_RESOLUTION_MISMATCH = "calib_resolution_mismatch"
REASON_INVALID_AGL = "invalid_agl"
REASON_NO_DETECTIONS = "no_detections"


class PriorGeometryScorer:
    """후보에 "고도·자세상 이 크기·이 모양이어야 한다" 일치도를 붙이고 정렬한다.

    매직넘버 금지(§7.3) — 임계값은 전부 `__init__` 파라미터(→ preset yaml에서 조정 가능).
    기본값의 근거는 전부 `core/prior_geometry.py`의 동명 상수 주석에 있다.

    ⚠️ **타겟 형상은 한 변 `target_size_m`의 정사각형으로 가정한다**(② 조난자 초록 매트
    3.0m×3.0m / ArUco 50cm — 실측 스펙이 실제로 정사각이다). ① 버티포트의 **원형** 흰 필드
    (직경 3m)는 면적이 `π·(d/2)²`라 이 가정이 그대로는 안 맞는다 — 그래서 지금 이 스텝은
    `distress_coarse.yaml`에만 배선돼 있고 `vertiport_*.yaml`에는 넣지 않았다. 원형 타겟까지
    확장하려면 `target_size_m` 하나가 아니라 형상까지 파라미터로 받아야 한다(그때 하면 된다,
    지금 과설계하지 않는다).
    """

    def __init__(
        self,
        target_size_m: float = DISTRESS_MAT_SIZE_M,
        area_tolerance_low: float = DEFAULT_AREA_TOLERANCE_LOW,
        area_tolerance_high: float = DEFAULT_AREA_TOLERANCE_HIGH,
        shape_tolerance: float = DEFAULT_SHAPE_TOLERANCE,
        anisotropy_margin: float = DEFAULT_ANISOTROPY_MARGIN,
        attitude_residual_deg: float = math.degrees(DEFAULT_ATTITUDE_RESIDUAL_RAD),
        area_weight: float = DEFAULT_AREA_WEIGHT,
        shape_weight: float = DEFAULT_SHAPE_WEIGHT,
        mount_yaw_psi_m_rad: float = MOUNT_YAW_PSI_M_RAD_DEFAULT,
        reorder: bool = True,
    ):
        self.target_size_m = float(target_size_m)
        self.area_tolerance_low = float(area_tolerance_low)
        self.area_tolerance_high = float(area_tolerance_high)
        self.shape_tolerance = float(shape_tolerance)
        self.anisotropy_margin = float(anisotropy_margin)
        self.attitude_residual_rad = math.radians(float(attitude_residual_deg))
        self.area_weight = float(area_weight)
        self.shape_weight = float(shape_weight)
        self.mount_yaw_psi_m_rad = float(mount_yaw_psi_m_rad)
        self.reorder = bool(reorder)

    # -- 입력 해석 ---------------------------------------------------------

    def _resolve_inputs(self, state: VisionState) -> tuple[Optional[dict], str]:
        """`state.meta[PRIOR_INPUTS_KEY]` → 검증된 입력 dict 또는 `(None, 사유)`.

        **어떤 결손에도 예외를 올리지 않는다** — 사전정보 부재는 정상 상태다.
        """
        raw = state.meta.get(PRIOR_INPUTS_KEY)
        if not isinstance(raw, dict):
            return None, REASON_NO_PRIOR_INPUTS

        agl = raw.get("agl_m")
        if agl is None:
            return None, REASON_NO_AGL
        try:
            agl_m = float(agl)
        except (TypeError, ValueError):
            return None, REASON_INVALID_AGL
        if not math.isfinite(agl_m) or agl_m <= 0.0:
            return None, REASON_INVALID_AGL

        attitude = raw.get("attitude")
        if isinstance(attitude, dict):
            attitude = attitude_from_telemetry(attitude)
        if not isinstance(attitude, VehicleAttitude):
            return None, REASON_NO_ATTITUDE

        camera_matrix = raw.get("camera_matrix")
        dist_coeffs = raw.get("dist_coeffs")
        calib_size = raw.get("calib_image_size")
        if camera_matrix is None or dist_coeffs is None or calib_size is None:
            return None, REASON_NO_CALIBRATION
        try:
            K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
            dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
            calib_wh = (int(calib_size[0]), int(calib_size[1]))
        except (TypeError, ValueError, IndexError):
            return None, REASON_NO_CALIBRATION
        if not np.all(np.isfinite(K)) or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
            return None, REASON_NO_CALIBRATION

        frame_h, frame_w = state.original.shape[:2]
        if (int(frame_w), int(frame_h)) != calib_wh:
            return (
                None,
                f"{REASON_CALIB_RESOLUTION_MISMATCH}:frame={frame_w}x{frame_h},"
                f"calib={calib_wh[0]}x{calib_wh[1]}",
            )

        return (
            {
                "agl_m": agl_m,
                "attitude": attitude,
                "camera_matrix": K,
                "dist_coeffs": dist,
                "calib_id": str(raw.get("calib_id", "")),
            },
            REASON_OK,
        )

    # -- 파이프라인 스텝 ---------------------------------------------------

    def __call__(self, state: VisionState) -> VisionState:
        n_in = len(state.detections)

        inputs, reason = self._resolve_inputs(state)
        if inputs is None:
            state.meta[PRIOR_META_KEY] = {
                "active": False,
                "reason": reason,
                "scored": 0,
                "reordered": False,
                "n_detections": n_in,
            }
            return state

        if n_in == 0:
            state.meta[PRIOR_META_KEY] = {
                "active": True,
                "reason": REASON_NO_DETECTIONS,
                "scored": 0,
                "reordered": False,
                "n_detections": 0,
                "agl_m": inputs["agl_m"],
                "attitude_deg": self._attitude_deg(inputs["attitude"]),
                "mount_yaw_measured": MOUNT_YAW_PSI_M_MEASURED,
            }
            return state

        scored = 0
        failures: list[str] = []
        for det in state.detections:
            entry = self._score_detection(det, inputs)
            if entry.get("score") is None:
                failures.append(str(entry.get("error", "unknown")))
            else:
                scored += 1
            det.meta[DETECTION_META_KEY] = entry

        # 🔴 정렬만 한다 — `sorted`는 같은 객체들의 순열이라 개수가 변할 수 없다.
        #    점수 없는 후보(예측 실패)는 **뒤로 밀리되 사라지지 않는다**.
        reordered = False
        if self.reorder:
            before = list(state.detections)
            state.detections = sorted(
                state.detections,
                key=lambda d: -self._score_of(d),  # 안정 정렬 → 동점은 기존 순서(confidence) 유지
            )
            reordered = [id(d) for d in state.detections] != [id(d) for d in before]

        ranked = [self._score_of(d) for d in state.detections]
        state.meta[PRIOR_META_KEY] = {
            "active": True,
            "reason": REASON_OK,
            "scored": scored,
            "failed": len(failures),
            "failure_reasons": failures,
            "reordered": reordered,
            "n_detections": len(state.detections),
            "ranked_scores": [round(s, 6) for s in ranked],
            # 소비자(향후 상태머신 세션)가 "사전정보로 모호성이 풀렸는가"를 판단할 수 있는 훅.
            # 여기서 후보를 지우지 않으므로 `n_candidates`는 그대로다 — 판단은 소비자 몫이다.
            "top_score": round(ranked[0], 6) if ranked else None,
            "runner_up_score": round(ranked[1], 6) if len(ranked) > 1 else None,
            "score_margin": round(ranked[0] - ranked[1], 6) if len(ranked) > 1 else None,
            "agl_m": inputs["agl_m"],
            "attitude_deg": self._attitude_deg(inputs["attitude"]),
            "target_size_m": self.target_size_m,
            "calib_id": inputs["calib_id"],
            # ψ_m 미측정 사실을 진단에 그대로 태워 보낸다(`core/frames.py` §7 U3).
            "mount_yaw_measured": MOUNT_YAW_PSI_M_MEASURED,
            "mount_yaw_psi_m_rad": self.mount_yaw_psi_m_rad,
        }

        # 계약 불변식 — 개수 보존. 여기서 깨지면 코드 버그다(입력 결손이 아니라).
        assert len(state.detections) == n_in, "PriorGeometryScorer가 후보 개수를 바꿨다"
        return state

    # -- 내부 헬퍼 ---------------------------------------------------------

    @staticmethod
    def _attitude_deg(attitude: VehicleAttitude) -> dict:
        return {
            "roll": round(math.degrees(attitude.roll_rad), 4),
            "pitch": round(math.degrees(attitude.pitch_rad), 4),
            "yaw": round(math.degrees(attitude.yaw_rad), 4),
            "source": attitude.source,
        }

    @staticmethod
    def _score_of(det) -> float:
        """정렬 키. 점수가 없는 후보는 -1.0으로 **맨 뒤**에 놓되 제거하지 않는다."""
        entry = det.meta.get(DETECTION_META_KEY)
        if not isinstance(entry, dict):
            return -1.0
        value = entry.get("score")
        return float(value) if isinstance(value, (int, float)) else -1.0

    def _observed_area_px2(self, det) -> float:
        """관측 면적 — 코너 4개면 shoelace(윤곽 실면적), 없으면 bbox 면적으로 폴백.

        bbox는 회전된 사각형에서 과대추정되지만(최대 2배) 허용폭 안이고, 폴백이 없으면
        코너가 없는 검출기(예: `white_field_detector`)에서 점수를 아예 못 낸다.
        """
        corners = det.corners
        if corners is not None and len(np.asarray(corners).reshape(-1, 2)) == 4:
            return polygon_area_px2(corners)
        return float(det.bbox[2] * det.bbox[3])

    def _score_detection(self, det, inputs: dict) -> dict:
        corners = det.corners
        has_quad = corners is not None and len(np.asarray(corners).reshape(-1, 2)) == 4
        try:
            score, predicted = score_candidate(
                center_px=det.center,
                corners=corners if has_quad else None,
                observed_area_px2=self._observed_area_px2(det),
                agl_m=inputs["agl_m"],
                attitude=inputs["attitude"],
                camera_matrix=inputs["camera_matrix"],
                dist_coeffs=inputs["dist_coeffs"],
                target_size_m=self.target_size_m,
                psi_m=self.mount_yaw_psi_m_rad,
                area_tolerance_low=self.area_tolerance_low,
                area_tolerance_high=self.area_tolerance_high,
                shape_tolerance=self.shape_tolerance,
                anisotropy_margin=self.anisotropy_margin,
                attitude_residual_rad=self.attitude_residual_rad,
                area_weight=self.area_weight,
                shape_weight=self.shape_weight,
            )
        except ValueError as e:
            # 예측 자체가 불가능한 후보(시선이 지평선 위 등) — **버리지 않는다.**
            # 점수 없이 사유만 남기고, 정렬에서 뒤로 밀릴 뿐이다.
            return {"score": None, "error": str(e), "area_source": "quad" if has_quad else "bbox"}

        return {
            "score": round(score.score, 6),
            "area_score": round(score.area_score, 6) if score.area_score is not None else None,
            "shape_score": round(score.shape_score, 6) if score.shape_score is not None else None,
            "components": list(score.components),
            "observed_area_px2": round(score.observed_area_px2, 3),
            "predicted_area_px2": round(score.predicted_area_px2, 3),
            "area_ratio": round(score.area_ratio, 6),
            "observed_anisotropy": (
                round(score.observed_anisotropy, 6)
                if score.observed_anisotropy is not None
                else None
            ),
            "max_anisotropy": round(score.max_anisotropy, 6),
            "incidence_deg": round(score.incidence_deg, 4),
            "depth_m": round(score.depth_m, 4),
            "tolerance_widening": round(score.tolerance_widening, 6),
            "area_source": "quad" if has_quad else "bbox",
            # 예상 4각형 — 오버레이/사후 대조용(§7.4 "왜 이렇게 봤나"를 눈으로 보게).
            "predicted_quad_px": [[round(x, 3), round(y, 3)] for x, y in predicted.predicted_quad_px],
            "predicted_quad_area_px2": round(predicted.predicted_quad_area_px2, 3),
            "ground_normal_cam": [round(v, 6) for v in predicted.ground_normal_cam],
        }
