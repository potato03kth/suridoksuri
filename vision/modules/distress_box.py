import math
import warnings

import cv2
import numpy as np
from vision.core.state import VisionState
from vision.core.target import DISTRESS_MAT_SIZE_M

# 매트 bbox 모서리의 정규 순회 순서 — 이 순서가 곧 등거리 축퇴 시의 tie-break 우선순위다.
# (기존 구현의 `max()`가 첫 최댓값을 돌려주던 동작과 동일한 순서를 유지한다.)
CORNER_NAMES = ("tl", "tr", "br", "bl")

_SQRT2 = math.sqrt(2.0)

# ──────────────────────────────────────────────────────────────────────────────
# 착륙점 안전창의 물리 상수 (2026-07-28 재설계 — 아래 클래스 docstring "착륙점 기하" 절)
# ──────────────────────────────────────────────────────────────────────────────
# 🔴 기체 최외곽(다리/프롭 끝) 반경 R — **미측정**.
#
# `core/frames.py`의 `MOUNT_YAW_PSI_M_MEASURED = False` 패턴을 그대로 따른다: 값을 지어내지
# 않고 파라미터로 빼두되, "이건 실측값이 아니다"를 코드가 플래그로 들고 다니며 진단 meta까지
# 전파한다. 2026-07-28 사용자와의 설계 논의에서 확정된 **유일한 사실은 "0.5m를 초과한다"** 뿐이고
# 정확한 값은 아무도 모른다(저장소 어디에도 기체 치수가 없다 — `CA_ROTOR*_PX/PY`는 ±1.0 정규화
# 값이라 팔 길이가 아니다).
#
# 기본값 0.60은 **알려진 하한(0.5m) 위의 보수적 공칭값**이다. 방향의 비대칭이 이 선택을 정한다:
#   - R을 **작게** 잡으면 → 착륙점이 매트 가장자리에 너무 가까워져 **기체가 0.105m 라이즈드
#     플랫폼 밖으로 삐져나가 넘어진다**(진짜 사고).
#   - R을 **크게** 잡으면 → 안전창이 좁아지거나 비어 착륙점이 안 나온다(임무 실패, 그러나 안전).
# 후자가 낫다.
#
# 🔴 **이 매트에는 절벽이 있다.** 3.0m 매트 + 20cm 정중앙 박스 + δ=0.30m 조합에서
# 안전창이 존재하는 R의 상한은 **0.6444m**다(`compute_landing_window()`의
# `aircraft_radius_max_feasible_m`, 유도는 클래스 docstring). 실측 R이 이 값을 넘으면
# "박스 옆 빈 초록면" 규약 자체에 해가 없다 — 기본값 0.60은 그 절벽에서 4.4cm 앞이다.
# **실측이 최우선 미해결 항목이다.**
AIRCRAFT_RADIUS_M_DEFAULT = 0.60
AIRCRAFT_RADIUS_MEASURED = False

# 허용 드리프트 δ — 착륙 시점에 유도가 착륙점에서 벗어나도 좋은 거리.
# 근거는 지어낸 값이 아니라 `docs/vision_plan.md` §1의 요구사항 **"최종 정확도 <30cm"** 이다
# (저장소에 기록된 유일한 착륙 위치정확도 수치). ⚠️ 이건 "달성이 확인된 오차"가 아니라
# **요구 오차**다 — 실기체 폐루프 실측이 나오면 재검토 대상.
LANDING_DRIFT_ALLOWANCE_M_DEFAULT = 0.30

# 흰 사무박스 한 변 — `docs/vision_plan.md` §2 실측 확정("정중앙 20cm 흰 사무박스").
# 매트 대비 선형비 0.2/3.0 ≈ 0.0667(= `presets/distress_fine.yaml`의 면적비 0.00444 근거와 동일).
DISTRESS_WHITE_BOX_SIZE_M = 0.20


def compute_landing_window(
    mat_size_m: float,
    white_box_size_m: float,
    aircraft_radius_m: float,
    drift_allowance_m: float,
    safety_window_bias: float,
) -> dict:
    """착륙점의 **매트 중심으로부터의 거리 d**(축당, 미터)를 물리량에서 도출한다.

    프레임과 무관한 순수 함수다 — 입력이 전부 생성자 파라미터라 `__init__`에서 한 번만 계산한다.

    ## 기하 (🔴 하한과 상한을 재는 방향이 서로 다르다)

    착륙점은 매트 bbox **모서리 방향(대각선)** 에 있으므로 매트 중심 기준 좌표가 `(±d, ±d)`다.
    기체는 반경 `R`의 원반으로 근사한다. `M = 매트반변`, `b = 흰 박스 반변`, `δ = 허용 드리프트`.

    1. **하한 — 박스를 밟지 않는다.** 박스는 매트 정중앙의 정사각형이므로 착륙점에서 가장 가까운
       박스 점은 **대각선 방향의 박스 모서리**(중심에서 `b·√2`)다. 즉 거리를 **대각선으로** 잰다:

           √2·d − b·√2 ≥ R      ⟺      d ≥ b + R/√2                      … d_min

    2. **상한 — 매트를 벗어나지 않는다.** 매트 가장자리까지의 최단거리는 대각선이 아니라
       **축 방향**이다(착륙점 `(d,d)`에서 x=M 모서리까지 `M − d`). 대각선으로 재면 거리를
       √2배 과대평가해 **실제보다 안전하다고 착각**한다:

           M − d ≥ R + δ        ⟺      d ≤ M − R − δ                     … d_max

    ## 안전창이 빌 수 있다 (그리고 그건 설계 정보다)

    `d_min > d_max`면 **해가 없다**. R에 대해 풀면 절벽의 위치가 나온다:

           b + R/√2 = M − R − δ   ⟺   R_max = (M − δ − b) / (1 + 1/√2)

    기본값(M=1.5, b=0.10, δ=0.30)에서 `R_max = 1.1/1.7071 = 0.6444 m`.
    해가 없으면 `feasible=False`를 돌려주고 **착륙점을 지어내지 않는다**(호출자가 거절한다).

    ## 정중앙 박스가 하한의 최악 경우다

    하한은 "박스가 매트 정중앙"(§2 확정 스펙)을 전제로 유도했다. 이 전제는 **보수적 방향**이다 —
    모서리 선택 규칙이 "박스에서 가장 먼 모서리"라, 박스가 중심을 벗어나면 착륙점은 자동으로 더
    멀어진다(정중앙에서만 네 모서리가 등거리 = 고를 수 있는 최원거리가 최소). 실제 검출된 박스의
    중심 오프셋은 `box_center_offset_m`으로 meta에 실어 전제를 사후 검증할 수 있게 한다.
    """
    if not (mat_size_m > 0.0):
        raise ValueError(f"mat_size_m는 양수여야 한다: {mat_size_m}")
    if not (white_box_size_m >= 0.0):
        raise ValueError(f"white_box_size_m는 음수일 수 없다: {white_box_size_m}")
    if not (aircraft_radius_m >= 0.0):
        raise ValueError(f"aircraft_radius_m는 음수일 수 없다: {aircraft_radius_m}")
    if not (drift_allowance_m >= 0.0):
        raise ValueError(f"drift_allowance_m는 음수일 수 없다: {drift_allowance_m}")
    if not (0.0 <= safety_window_bias <= 1.0):
        raise ValueError(f"safety_window_bias는 0.0~1.0이어야 한다: {safety_window_bias}")

    half_mat = mat_size_m / 2.0
    half_box = white_box_size_m / 2.0
    d_min = half_box + aircraft_radius_m / _SQRT2
    d_max = half_mat - aircraft_radius_m - drift_allowance_m
    r_max = (half_mat - drift_allowance_m - half_box) / (1.0 + 1.0 / _SQRT2)
    feasible = d_min <= d_max

    window = {
        "feasible": bool(feasible),
        "d_min_m": round(d_min, 6),
        "d_max_m": round(d_max, 6),
        "mat_size_m": float(mat_size_m),
        "white_box_size_m": float(white_box_size_m),
        "aircraft_radius_m": float(aircraft_radius_m),
        "aircraft_radius_measured": bool(AIRCRAFT_RADIUS_MEASURED),
        "drift_allowance_m": float(drift_allowance_m),
        "safety_window_bias": float(safety_window_bias),
        # 이 매트/박스/δ 조합에서 안전창이 존재하는 R의 상한 — "절벽이 어디인가"를 사람에게
        # 보여주는 값이다(§7.4 관측성). 실측 R이 이걸 넘으면 규약 자체를 재검토해야 한다.
        "aircraft_radius_max_feasible_m": round(r_max, 6),
    }
    if not feasible:
        window["d_m"] = None
        window["pull_fraction"] = None
        window["box_clearance_m"] = None
        window["mat_edge_margin_m"] = None
        return window

    d = d_min + safety_window_bias * (d_max - d_min)
    window["d_m"] = round(d, 6)
    # 착륙점 = 매트 중심 + pull_fraction × (고른 모서리 − 매트 중심). 모서리는 매트 반변(=M)
    # 위치이므로 fraction은 곧 d/M이다. (폐기된 `interior_margin_ratio`와는 1 − ratio 관계.)
    window["pull_fraction"] = round(d / half_mat, 6)
    # 여유(슬랙) — 0에 가까울수록 절벽에 붙어 있다는 뜻.
    window["box_clearance_m"] = round(_SQRT2 * (d - half_box) - aircraft_radius_m, 6)
    window["mat_edge_margin_m"] = round(half_mat - d - aircraft_radius_m, 6)
    return window


class WhiteBoxDetector:
    """
    조난자 구역(② `docs/vision_plan.md` §5.3) fine 단계: 초록 매트 내부 정중앙 흰 박스(고명도
    blob, 실측 20cm — §2 "정중앙 20cm 흰 사무박스")를 확인한다.

    **핵심 설계 포인트: 착륙 목표는 박스가 아니라 "박스 옆 빈 초록면"이다**(§5.3 "fine(≤~15m):
    중앙 흰 박스 → 박스 옆 빈 초록면 착륙"). 박스는 랜드마크일 뿐이므로, 이 모듈은 박스를
    확인하는 데서 그치지 않고 착륙점(landing point)까지 계산해 `detection.meta`에 싣는다.

    coarse 단계(`distress_coarse.yaml` — `ColorFilter(mode=color)` + `RectDetector`)가 낸 초록
    매트 `detections`를 ROI로 읽어 그 안에서 흰 박스를 찾는 캐스케이드 패턴이다
    (`vertiport_v.py`/`vertiport_ring.py`와 동일 구조 — 이전 단계 detections를 ROI로 읽고
    자기 결과로 덮어쓴다). `color_filter`가 mask 밖 픽셀을 지워버린 `current` 대신 원본 색상이
    보존된 `original`을 읽는다(`vision/CLAUDE.md` "주의" 절 — ColorFilter 함정).

    **착륙점 산출 방법("가장 먼 모서리를 안전창까지 당기기"):**
    1. 매트 bbox의 네 모서리 중 흰 박스 중심에서 유클리드 거리가 가장 먼 모서리를 고른다
       (박스에서 최대한 멀어지는 방향 — "박스 옆"의 최소 요건).
    2. 그 모서리 **방향으로** 매트 중심에서 거리 `d`(미터)만큼 나간 점을 착륙점으로 삼는다.
       `d`는 **기체 크기에서 도출한 물리량**이다 — 아래 "착륙점 기하" 절 참조.
    이 방법을 고른 이유: 박스가 매트 중심에서 벗어나 배치되는 경우(실측이 아직 확정 전인 다른
    시나리오)에도 별도 분기 없이 동일 공식이 자연스럽게 "박스 반대편"을 가리킨다.
    **"박스 옆"의 정확한 방향·거리는 대회측 미회신 상태**(`vision_plan.md` §9 "중요" 각주)이므로
    "가장 먼 모서리 방향"이라는 *방향* 규약은 여전히 잠정값이다 — 규정이 확정되면 재검토 대상.
    반면 *거리*는 2026-07-28부터 잠정값이 아니라 기체 크기에서 도출된다.

    ## 🔴 착륙점 기하 — 기체 크기 기반 (2026-07-28 재설계)

    **고친 결함:** 예전에는 착륙점 거리가 `interior_margin_ratio`(무차원 비율, 기본 0.3)라는
    **잠정 상수** 하나로 정해졌고 **기체 크기를 한 번도 고려하지 않았다.** 2026-07-28에
    기체 최외곽(다리/프롭 끝) 반경이 **0.5m를 초과**한다는 것이 확정되며 구조적 결함이 드러났다:

        착륙점 = 중심에서 (1 − 0.3)·1.5 = 1.05 m  → 매트 가장자리까지 여유 0.45 m < R
        ⇒ 유도 오차가 정확히 0이어도 기체가 0.105m 라이즈드 플랫폼 밖으로 삐져나간다.

    매트는 페인트 선이 아니라 **물리적 구조물**이라 가장자리에 반쯤 걸치면 기울어져 넘어진다 —
    "표면적 실패"가 아니라 진짜 사고다.

    **대체:** 착륙점 거리 `d`를 비율이 아니라 **물리량에서 계산**한다. 부등식 두 개와 그 유도,
    "하한은 대각선으로 / 상한은 축 방향으로 잰다"는 기하, 안전창이 비는 조건은 전부
    모듈 함수 `compute_landing_window()` docstring에 있다(중복 서술하지 않는다).

    - 파라미터: `mat_size_m` · `white_box_size_m` · `aircraft_radius_m` · `drift_allowance_m`
      · `safety_window_bias`(안전창 안에서 어디를 고를지, 0=하한 / 1=상한 / 기본 0.5=중점).
    - 🔴 `aircraft_radius_m` 기본값은 **미측정 공칭값**이다(`AIRCRAFT_RADIUS_MEASURED = False`,
      `core/frames.py`의 ψ_m 패턴). 그 플래그는 `landing_geometry` meta로 그대로 전파된다.
    - 🔴 **안전창이 비면 착륙점을 내지 않는다.** 조용히 아무 값을 내는 대신 그 검출을 거절하고
      (`reject_reasons`에 `landing_point_infeasible`) 계산된 하한/상한/R을
      `state.meta["white_box_detector"]["landing_point_infeasible"]`에 남긴다. 상위
      (`main.py`/`replay.py`)가 이걸 읽어 `valid=false` + 사유로 발행하는 배선은 아직 없다.
      **거절이 필수인 이유:** 착륙점만 빼고 검출을 남기면 `modules/distress_mat.py`가
      `landing_point_px` 부재를 보고 **매트 중심으로 우아하게 degrade**한다 — 매트 중심은 바로
      그 흰 박스 위다. 즉 "안전창이 비었다"가 "박스 위에 착륙하라"로 조용히 뒤집힌다.

    **`interior_margin_ratio`는 폐기됐다(하위호환 no-op).** 물리 도출값과 잠정 비율이 둘 다
    적용되면 어느 쪽이 이겼는지 알 수 없어 정확히 이 결함이 재발한다 → **물리 도출값이 항상
    이긴다.** 기존 preset yaml이 이 키를 줘도 `Pipeline.from_config`가 죽지 않도록 인자는 계속
    받되(비행 직전 프리셋 로드가 통째로 실패하는 것이 더 나쁘다) **값은 무시하고**
    `DeprecationWarning`을 내며 `state.meta["white_box_detector"]["interior_margin_ratio_ignored"]`
    로 무시 사실을 blackbox까지 알린다(§7.4 침묵 금지).

    ## ⚠️ 등거리 축퇴와 그 완화 (2026-07-28)

    **실측 스펙상 흰 박스는 매트 정중앙**이라, 매트 bbox 네 모서리가 박스 중심에서 **이론상
    등거리**가 된다. 정확히 등거리이면 `max()`가 첫 최댓값을 돌려줘 결정론(§7.5) 자체는 안
    깨지지만, **실전 입력은 절대 정확히 등거리가 아니다** — mp4 압축/컨투어 양자화로 박스 중심이
    1px만 흔들려도 "가장 먼 모서리"가 TL↔BR로 뒤집힌다. 실측 재현(2026-07-28): 1px 흔들림만으로
    착륙점이 **2.97m 점프**하고 한 시퀀스 안에서 네 모서리를 전부 방문했다(정지 이미지=TL,
    같은 장면 mp4=BR). 이 착륙점은 `modules/distress_mat.py`를 거쳐 **실제 유도 좌표로 나가므로**
    프레임마다 매트 반대편을 지시하면 기체가 진동한다.

    완화는 **선택의 안정성만** 손댄다(`interior_margin_ratio`나 "박스 옆" 규약은 무변경):

    1. **동률 허용오차(`tie_tolerance_ratio`)** — 최대거리에서 `tie_tolerance_ratio × 매트 bbox
       대각선` 이내인 모서리를 전부 "동률 후보"로 보고, 그 안에서 `CORNER_NAMES` 정규 순서의
       첫 번째를 고른다. 픽셀 잡음이 허용오차보다 작으면 선택이 아예 흔들리지 않는다.
       `0.0`을 주면 **기존 동작과 정확히 동일**(정확 동률만 첫 모서리로 해소)하다.
    2. **프레임 간 히스테리시스(`corner_hysteresis`)** — 직전 프레임에 고른 모서리가 아직 동률
       후보 안에 있으면 그대로 유지한다. 이 둘의 조합은 **폭 `2 × tol`의 슈미트 트리거**가 된다
       (직전 모서리를 버리려면 tol 이상 뒤처져야 하고, 되돌아오려면 반대로 tol 이상 앞서야 한다).
       덕분에 허용오차 경계에 딱 걸친 배치에서도 **한 번 전환한 뒤 눌러앉지, 왕복 진동하지
       않는다.** 직전 프레임과의 대응은 매트 bbox IoU로 찾는다(`modules/fusion.py::TemporalFusion`
       과 같은 패턴 — 새 추적기를 발명하지 않는다).

    **결정론은 유지된다**(§7.5): 히스테리시스는 wall-clock/난수가 아니라 *입력 시퀀스*만의
    함수라, 같은 프레임열은 같은 착륙점열을 낸다. 상태는 인스턴스 단위이므로 새 인스턴스는 항상
    같은 초기 선택(정규 순서 첫 동률 후보)에서 출발한다 — `modules/tracker.py`/`fusion.py`와
    동일한 stateful 모듈 관례.

    매직넘버 금지 — 임계값은 전부 `__init__` 파라미터(→ preset yaml에서 조정 가능).
    """

    def __init__(
        self,
        val_min: int = 180,
        sat_max: int = 60,
        min_area_ratio: float = 0.0015,
        max_area_ratio: float = 0.02,
        min_solidity: float = 0.85,
        max_aspect_ratio: float = 1.4,
        roi_margin: float = 0.05,
        mat_size_m: float = DISTRESS_MAT_SIZE_M,
        white_box_size_m: float = DISTRESS_WHITE_BOX_SIZE_M,
        aircraft_radius_m: float = AIRCRAFT_RADIUS_M_DEFAULT,
        drift_allowance_m: float = LANDING_DRIFT_ALLOWANCE_M_DEFAULT,
        safety_window_bias: float = 0.5,
        tie_tolerance_ratio: float = 0.02,
        corner_hysteresis: bool = True,
        hysteresis_iou_min: float = 0.3,
        interior_margin_ratio: float | None = None,
    ):
        self.val_min = val_min
        self.sat_max = sat_max
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_solidity = min_solidity
        self.max_aspect_ratio = max_aspect_ratio
        self.roi_margin = roi_margin
        self.mat_size_m = float(mat_size_m)
        self.white_box_size_m = float(white_box_size_m)
        self.aircraft_radius_m = float(aircraft_radius_m)
        self.drift_allowance_m = float(drift_allowance_m)
        self.safety_window_bias = float(safety_window_bias)
        # 안전창은 프레임과 무관한 상수라 여기서 한 번만 계산한다(인자 검증도 여기서 —
        # `utils/frame_source.py`의 AF 인자 검증과 같은 "하드웨어/프레임을 만지기 전에 실패"
        # 원칙). 클래스 docstring "착륙점 기하" 절 + `compute_landing_window()` 참조.
        self._window = compute_landing_window(
            mat_size_m=self.mat_size_m,
            white_box_size_m=self.white_box_size_m,
            aircraft_radius_m=self.aircraft_radius_m,
            drift_allowance_m=self.drift_allowance_m,
            safety_window_bias=self.safety_window_bias,
        )
        # 폐기된 잠정 비율 — 받되 쓰지 않는다(클래스 docstring "착륙점 기하" 마지막 문단).
        self.interior_margin_ratio_ignored = (
            None if interior_margin_ratio is None else float(interior_margin_ratio)
        )
        if self.interior_margin_ratio_ignored is not None:
            warnings.warn(
                "interior_margin_ratio는 2026-07-28에 폐기됐다(무시됨). 착륙점 거리는 이제 "
                "기체 반경(aircraft_radius_m)·허용 드리프트(drift_allowance_m)·매트/박스 크기에서 "
                "도출된다 — WhiteBoxDetector docstring '착륙점 기하' 절 참조.",
                DeprecationWarning,
                stacklevel=2,
            )
        # 기본값 0.02 = 매트 bbox 대각선의 2%. 300px 매트(=3.0m)에서 tol≈8.5px, 슈미트 폭
        # 2·tol≈17px(≈17cm)로 실측 잡음 1px 대비 약 17배 여유. 반대로 이 대역을 벗어나려면
        # 박스가 대각선 방향으로 약 3px(≈3cm)만 치우치면 되므로, 규정이 확정돼 박스가 실제로
        # 편심 배치되는 시나리오의 기존 동작(=박스 반대편 지시)은 그대로 살아 있다.
        self.tie_tolerance_ratio = float(tie_tolerance_ratio)
        self.corner_hysteresis = bool(corner_hysteresis)
        self.hysteresis_iou_min = float(hysteresis_iou_min)
        # 프레임 간 히스테리시스 상태: [(매트 bbox, 선택한 모서리 인덱스), ...]
        self._prev_corner_choices: list[tuple[tuple[int, int, int, int], int]] = []

    def reset(self) -> None:
        """프레임 간 히스테리시스 상태를 지운다(새 시퀀스 시작 시)."""
        self._prev_corner_choices = []

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        """`modules/fusion.py::TemporalFusion._iou`와 동일한 bbox IoU (얇게 재사용)."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _previous_corner_index(self, mat_bbox: tuple) -> int | None:
        """직전 프레임에서 같은 매트로 볼 수 있는 항목의 모서리 인덱스(없으면 None)."""
        if not self.corner_hysteresis:
            return None
        best_iou, best_idx = 0.0, None
        for prev_bbox, corner_idx in self._prev_corner_choices:
            iou = self._iou(mat_bbox, prev_bbox)
            if iou >= self.hysteresis_iou_min and iou > best_iou:
                best_iou, best_idx = iou, corner_idx
        return best_idx

    def _select_far_corner(
        self, corners: list, box_cx: float, box_cy: float, diag: float, prev_idx: int | None
    ) -> tuple[int, int, bool]:
        """박스에서 가장 먼 매트 모서리를 고른다 → (모서리 인덱스, 동률 후보 수, 히스테리시스 사용 여부).

        축퇴 완화의 전부가 여기 있다 — 클래스 docstring "등거리 축퇴와 그 완화" 참조.
        """
        dists = [math.hypot(cx - box_cx, cy - box_cy) for cx, cy in corners]
        d_max = max(dists)
        tol = self.tie_tolerance_ratio * diag
        tied = [i for i, d in enumerate(dists) if d >= d_max - tol]
        if prev_idx is not None and prev_idx in tied:
            return prev_idx, len(tied), True
        # 정규 순서(CORNER_NAMES)의 첫 동률 후보 — tol=0이면 기존 `max()` 동작과 동일하다.
        return tied[0], len(tied), False

    def _white_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        """고명도·저채도(흰색) 게이팅. 초록 매트는 채도가 높아(합성 기준 sat=200) sat_max로
        자연히 배제된다 — 별도 색상 배제 로직 불필요."""
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, self.val_min], dtype=np.uint8)
        upper = np.array([180, self.sat_max, 255], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    def __call__(self, state: VisionState) -> VisionState:
        confirmed = []
        rejected = 0
        reject_reasons: list[str] = []
        corner_choices: list[tuple[tuple[int, int, int, int], int]] = []

        for det in state.detections:
            x, y, w, h = det.bbox
            mat_area = float(w * h)
            if mat_area <= 0:
                rejected += 1
                reject_reasons.append("empty_mat_bbox")
                continue

            mx, my = int(w * self.roi_margin), int(h * self.roi_margin)
            x0, y0 = max(0, x - mx), max(0, y - my)
            x1 = min(state.original.shape[1], x + w + mx)
            y1 = min(state.original.shape[0], y + h + my)
            roi = state.original[y0:y1, x0:x1]
            if roi.size == 0:
                rejected += 1
                reject_reasons.append("empty_roi")
                continue

            mask = self._white_mask(roi)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                rejected += 1
                reject_reasons.append("no_white_pixels")
                continue

            best = None  # (area, bx, by, bw, bh, solidity)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                ratio = area / mat_area
                if not (self.min_area_ratio <= ratio <= self.max_area_ratio):
                    continue
                hull_area = cv2.contourArea(cv2.convexHull(cnt))
                solidity = area / hull_area if hull_area > 0 else 0.0
                if solidity < self.min_solidity:
                    continue
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect = max(bw, bh) / max(1, min(bw, bh))
                if aspect > self.max_aspect_ratio:
                    continue
                if best is None or area > best[0]:
                    best = (area, bx, by, bw, bh, solidity)

            if best is None:
                rejected += 1
                reject_reasons.append("no_qualifying_contour")
                continue

            if not self._window["feasible"]:
                # 🔴 안전창이 비었다 — 착륙점을 지어내지 않는다. **박스는 찾았지만 내려앉을
                # 곳이 없다**는 뜻이라 `no_white_pixels`와 다른 사유를 쓴다(§5.4 사유 뭉개기 금지).
                # 검출 자체를 거절해야 한다: 착륙점만 빼고 남기면 `modules/distress_mat.py`가
                # 매트 **중심**(=흰 박스 위)으로 조용히 degrade한다. 클래스 docstring 참조.
                rejected += 1
                reject_reasons.append("landing_point_infeasible")
                continue

            area, bx, by, bw, bh, solidity = best
            # ROI 좌표 -> 원본 이미지 좌표
            box_x0, box_y0 = x0 + bx, y0 + by
            box_cx = box_x0 + bw / 2.0
            box_cy = box_y0 + bh / 2.0

            mat_cx, mat_cy = x + w / 2.0, y + h / 2.0
            corners = [
                (float(x), float(y)),
                (float(x + w), float(y)),
                (float(x + w), float(y + h)),
                (float(x), float(y + h)),
            ]
            corner_idx, tie_count, from_hysteresis = self._select_far_corner(
                corners,
                box_cx,
                box_cy,
                math.hypot(float(w), float(h)),
                self._previous_corner_index(det.bbox),
            )
            far_corner = corners[corner_idx]
            corner_choices.append((det.bbox, corner_idx))

            # 착륙점 = 매트 중심 + pull_fraction × (고른 모서리 − 매트 중심).
            # 모서리는 매트 중심에서 반변(=mat_size_m/2)만큼 떨어진 점이므로, 이 fraction은
            # 곧 d/(mat_size_m/2)이고 결과 변위는 정확히 d 미터가 된다. bbox가 원근으로
            # 정사각이 아니어도 축별 스케일이 각각 반영되므로 별도 분기가 필요 없다.
            pull = self._window["pull_fraction"]
            landing_x = mat_cx + pull * (far_corner[0] - mat_cx)
            landing_y = mat_cy + pull * (far_corner[1] - mat_cy)

            # 하한 유도가 전제한 "박스는 매트 정중앙"(§2 확정 스펙)을 사후 검증할 수 있게
            # 실제 검출된 박스 중심의 오프셋을 미터로 환산해 남긴다(전제가 깨지면 보이도록).
            px_per_m_x = w / self.mat_size_m
            px_per_m_y = h / self.mat_size_m
            landing_geometry = dict(self._window)
            landing_geometry["box_center_offset_m"] = [
                round(float((box_cx - mat_cx) / px_per_m_x), 6),
                round(float((box_cy - mat_cy) / px_per_m_y), 6),
            ]
            landing_geometry["px_per_m"] = [round(float(px_per_m_x), 6), round(float(px_per_m_y), 6)]

            det.meta["white_box_detector"] = {
                "box_bbox": [int(box_x0), int(box_y0), int(bw), int(bh)],
                "box_center_px": [float(box_cx), float(box_cy)],
                "area_ratio": round(float(area / mat_area), 5),
                "solidity": round(float(solidity), 4),
                "landing_point_px": [float(landing_x), float(landing_y)],
                # 축퇴 진단(§7.4 블랙박스 포렌식) — tie_count>1이면 그 프레임이 실제로 등거리
                # 축퇴 상황이었다는 뜻이고, corner_from_hysteresis가 True면 정규 순서 대신
                # 직전 선택이 유지된 것이다.
                "landing_corner": CORNER_NAMES[corner_idx],
                "corner_tie_count": int(tie_count),
                "corner_from_hysteresis": bool(from_hysteresis),
                # 기체 크기 기반 안전창(2026-07-28) — 착륙점이 "왜 거기인가"의 전량 근거.
                # `aircraft_radius_measured=False`가 여기까지 따라온다.
                "landing_geometry": landing_geometry,
            }
            confirmed.append(det)

        self._prev_corner_choices = corner_choices
        state.detections = confirmed
        state.meta["white_box_detector"] = {
            "confirmed": len(confirmed),
            "rejected": rejected,
            "reject_reasons": reject_reasons,
        }
        # ⚠️ 안전창 상수를 **정상 경로의 state.meta에는 싣지 않는다** — 이 dict는 골든셋
        # `labels.json`이 통째로 동등비교하는 대상이라(§ tests/test_golden_regression.py), 여기에
        # 물리 파라미터를 넣으면 R 기본값이 골든 픽스처에까지 복제돼 "단일 출처" 원칙이 깨진다.
        # 정상 경로의 근거 전량은 검출별 `det.meta[...]["landing_geometry"]`에 있다.
        if not self._window["feasible"]:
            # 🔴 안전창이 비었다는 것은 설계 정보다 — 사람에게 보여야 한다.
            # 다음 세션이 상위(main.py/replay.py)에서 valid=false 사유로 배선할 지점.
            state.meta["white_box_detector"]["landing_point_infeasible"] = dict(self._window)
        if self.interior_margin_ratio_ignored is not None:
            state.meta["white_box_detector"]["interior_margin_ratio_ignored"] = (
                self.interior_margin_ratio_ignored
            )
        return state
