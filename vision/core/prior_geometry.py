"""
사전정보 기반 후보 기하 예측 — **순수 기하** (`docs/vision_plan.md` §4.1 GSD / §4.2 tilt).

## 무엇을 하는가

"타겟이 지상 평면에 놓여 있고 실측 크기를 안다"(초록 매트 3.0m 정사각 / 버티포트 흰 필드
직경 3m / ArUco 50cm)는 **이미 확정된 사실**을, 기체 AGL + 자세와 묶으면 **"진짜 타겟이라면
화면에 어떤 크기·어떤 찌그러짐으로 보여야 하는가"**가 하나로 정해진다. 이 파일은 그 예측을
계산하고, 관측된 후보가 그 예측과 얼마나 맞는지를 **0~1 점수**로 환산한다.

## 🔴 이것은 게이트가 아니라 랭킹이다 (설계의 심장)

사용자 기조 = **"좋은 비행 시퀀스는 실패하지 않는 시퀀스"** → *"진짜 타겟을 놓치는 것"이
"오탐을 하나 더 보는 것"보다 훨씬 나쁘다.* 그래서:

- 이 파일의 모든 점수 함수는 **결코 0이 되지 않는다**(로그-가우시안 감쇠). "임계값 미만 =
  탈락" 같은 판정을 여기서 만들지 않는다.
- 소비자(`modules/prior_score.py`)는 점수를 붙이고 **정렬만** 한다 — `state.detections`에서
  항목을 절대 제거하지 않는다.
- 기존 `presets/distress_coarse.yaml`의 `min_area`/`max_area` 하드 필터를 **대체하지 않는다.**
  그건 그대로 두고, 통과한 후보들 사이의 **순위**만 이 점수가 정한다.

## 🔴 미측정 값 2개에 대해 이 설계가 왜 견디는가

1. **마운트 요각 ψ_m 미측정**(`core/frames.py::MOUNT_YAW_PSI_M_MEASURED = False`).
   ψ_m은 지상 평면의 법선을 카메라 프레임 안에서 **광축 둘레로 회전시킬 뿐** z성분을 바꾸지
   않는다(`ground_normal_cam()` 참조). 즉 **화면 중앙에서의 전경축소(foreshortening) 크기는
   ψ_m과 무관**하고, ψ_m 오차는 "기울어진 방향"만 틀리게 한다. 면적 예측은 화면 중앙 근처에서
   ψ_m에 사실상 둔감하다.
2. **고무마운트 자세 잔차 θ_res 미측정**(§4.2 "IMU 자세 ≠ 실제 카메라 자세", 실측 이력 없음).
   §4.2가 정량화한 `지상오차 = 고도 × tan(θ_res)`는 **위치**를 틀리게 하는 항인데, 이 파일은
   위치를 예측하지 않는다 — **관측된 후보 자신의 픽셀**을 지상 평면에 역투영해 거기서 예측을
   세운다. 그래서 θ_res는 위치오차로 새지 않고 **입사각을 θ_res 만큼 흔드는 2차 효과**로만
   남는다. 그 흔들림은 `attitude_residual_rad` 파라미터로 **허용폭을 넓히는 데** 반영된다
   (`residual_tolerance_widening()`), 절대 예측값을 편향시키지 않는다.

## 기하 유도 (나디르 특수화 → tilt 일반화)

카메라 원점, 광학 프레임(OpenCV: X-우, Y-하, Z-전방). 지상 평면은 **수평**이므로 카메라에서
평면까지의 수직거리 = 수선거리 = AGL `h` 다(기울어도 변하지 않는다).

    평면:  n̂ · P = -h        (n̂ = 카메라 프레임에서 본 지면 법선(위쪽), 단위벡터)

`n̂`은 기체 자세에서 나온다(`ground_normal_cam()`), 나디르·수평에서 `n̂ = (0, 0, -1)`.

픽셀 `(u,v)` → 정규화 시선 `r = K⁻¹[u,v,1]`(`r_z = 1`, 왜곡은 `undistortPoints`로 선처리).
광선-평면 교점:

    s = -h / (n̂·r),   P = s·r,   깊이 P_z = s        (n̂·r < 0 이어야 앞쪽·아래쪽)

**지상 평면(미터) → 이미지(픽셀) 국소 야코비안 J(2x2)의 행렬식**을 평면상 정규직교 기저
`e1,e2`(e1×e2 = n̂)로 직접 전개하면 중간항이 전부 `n̂·P`로 접혀 다음이 나온다:

    |det J| = fx·fy·h / P_z³ = fx·fy·|n̂·r|³ / h²

    ⇒ 예상 면적[px²] = (실측 크기)² · fx·fy·|n̂·r|³ / h²

- **나디르 검산:** `r=(0,0,1)`, `n̂=(0,0,-1)` → `|det J| = fx·fy/h²`. GSD = h/fx [m/px]와 정확히
  일치한다. 실제로 `presets/distress_coarse.yaml` 헤더의 도출값을 그대로 재현한다:
  `fx=fy=1000.877`(폭 1536px·HFOV 75° 역산), h=10m, 3.0m 매트 → 9·1000.877²/100 =
  **90,158px²** (그 헤더의 "~90,000px²"). 즉 이 파일은 기존 `min_area` 도출을 **일반화**한
  것이지 새 규약이 아니다.
- **tilt 검산:** 화면 중앙(`r=(0,0,1)`)에서 `|n̂·r| = |n̂_z| = cosφ·cosθ` →
  `예상 면적 ∝ cos³(기울기)`. 슬랜트 거리 `h/cos` 로 인한 `1/cos²`와 평면 경사에 의한 `cos`가
  곱해진 교과서 결과와 일치한다. 25°에서 0.744배(-26%)로 **실제로 달라진다**.
- **비등방성(찌그러짐):** `J = (1/s)·A·E`, `A = [[fx,0,-fx·r_x],[0,fy,-fy·r_y]]`(2x3),
  `E = [e1 e2]`(3x2). J의 특이값 비 `σ1/σ2`는 `1/s` 스케일과 기저 선택(우측 2x2 회전)에
  **불변**이라 잘 정의된다. 화면 중앙에서 `σ1/σ2 = 1/cos(기울기)` (0°→1.000, 10°→1.015,
  25°→1.103). 정사각형은 지상에서 어느 방향으로 놓였든(in-plane yaw γ 미지) 투영된 인접
  변 길이비가 **`[1/(σ1/σ2), σ1/σ2]` 안**에 반드시 들어오므로, 이 값이 관측 비등방성의
  **물리적 상한**이 된다.
- **면적은 in-plane yaw γ에 무관**하다(`|det J|`가 기저 선택에 불변) — 타겟이 지상에서 어느
  방향을 보고 있는지 **모르는 채로도** 면적 예측이 성립하는 이유다. 이것이 이 스코어러가
  피듀셜 없는 blob 타겟에도 그대로 먹히는 근거다.

## 한계 (숨기지 않는다)

- `|det J|`는 **국소 1차 근사**다. 타겟 각크기가 커지면(저고도 근접) 실제 투영 4각형 면적과
  어긋난다 — `predicted_quad_px`(정확한 투영)와 `predicted_area_px2`(1차 근사)를 **둘 다**
  내보내 그 차이를 진단으로 볼 수 있게 했다. coarse 대역(40~15m, 각반경 ≤ 6°)에서는 1% 미만.
- 야코비안에 **왜곡의 야코비안은 넣지 않았다**(시선벡터 산출에만 `undistortPoints`를 쓴다).
  현재 캘리브레이션은 `dist_coeffs = 0`이라 정확하고, 실측 캘리브레이션이 들어오면 재검토 대상.
- 골든셋이 전부 **합성**(원근·왜곡 없는 축평행 사각형)이라 여기 계산된 예측을 **실촬영과
  대조한 적이 없다**. 이 파일이 점수를 게이트가 아니라 랭킹으로만 쓰는 결정적 이유.

## import 규칙

`vision/CLAUDE.md`: **core/ ← numpy, opencv만**. 이 파일은 파일 I/O·로깅·wall-clock·난수를
쓰지 않는다(`core/target.py`/`core/state_machine.py`/`core/frames.py`와 동일 패턴 — 같은 입력
이면 항상 같은 출력, §7.5 결정론). 같은 `core` 안의 `frames`/`target`만 상대 import 한다
(`core/runner.py`가 `.state`를 import하는 전례와 동일).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from .frames import MOUNT_YAW_PSI_M_RAD_DEFAULT, R_frd_cam
from .target import order_quad_corners_clockwise

# ──────────────────────────────────────────────────────────────────────────────
# `state.meta` 키 — 호출자(main.py/replay.py) ↔ 모듈(modules/prior_score.py) 계약
# ──────────────────────────────────────────────────────────────────────────────
# 🔴 **키 이름이 `core/`에 있는 이유**: import 규칙상 `main.py`/`replay.py`는 `modules/`를
# import할 수 없다(`presets 경로 + utils + core 만`). 그런데 사전정보를 심는 쪽은 그 둘이고
# 읽는 쪽은 모듈이라, 키 문자열을 양쪽에 손으로 적으면 한쪽만 고쳐졌을 때 조용히 어긋난다
# (스코어러가 영영 비활성인데 아무도 모르는 상태). 그래서 **양쪽이 공유할 수 있는 유일한
# 장소인 `core/`에 단일 출처로 둔다** — `core/wire.py`가 레코드 키를 상수로 모아둔 것과 같은 판단.

# 호출자가 프레임 단위 사전정보를 심는 `state.meta` 키.
PRIOR_INPUTS_KEY = "prior_inputs"
# 스코어러가 프레임 단위 진단을 남기는 `state.meta` 키(모듈 이름 네임스페이스 규칙).
PRIOR_META_KEY = "prior_geometry_scorer"
# 후보별 점수·근거가 붙는 `Detection.meta` 키.
PRIOR_DETECTION_META_KEY = "prior_geometry"


# ──────────────────────────────────────────────────────────────────────────────
# 기본값 (매직넘버 금지, §7.3) — 전부 이름·근거를 달아 둔다.
# ──────────────────────────────────────────────────────────────────────────────

# 관측 면적이 예상보다 **작을 때**의 허용 배수. 색 마스킹/윤곽 근사로 검출 면적이 명목값의
# ~35%까지 열화될 수 있다는 실측 판단이 이미 `presets/distress_coarse.yaml` 헤더에 있고
# (min_area=8000이 그 마진), 라이다 AGL 오차(면적은 1/h²)도 여기에 함께 흡수된다.
# 3.0 = "관측이 예상의 1/3이어도 e^-0.5 ≈ 0.61점은 준다"는 뜻.
DEFAULT_AREA_TOLERANCE_LOW = 3.0
# 관측 면적이 예상보다 **클 때**의 허용 배수. 이쪽을 더 빡빡하게 두는 것은 비대칭이 의도된
# 것이다 — 윤곽이 배경과 붙어 커지는 것은 "타겟이 아니다"의 더 강한 신호인 반면, 작아지는
# 것은 정상 열화다. 절대 거절하지 않으므로 빡빡해도 후보는 살아남는다.
DEFAULT_AREA_TOLERANCE_HIGH = 1.8
# 관측 비등방성이 물리 상한(σ1/σ2)을 넘었을 때의 허용 배수.
DEFAULT_SHAPE_TOLERANCE = 1.5
# 물리 상한 σ1/σ2에 곱하는 여유. 윤곽 근사(approxPolyDP)가 만드는 코너 지터와, 타겟이 완전한
# 정사각형이 아닐 가능성을 흡수한다. 1.35 = "35% 더 찌그러져 보여도 정상으로 본다".
DEFAULT_ANISOTROPY_MARGIN = 1.35
# 🔴 고무마운트 자세 잔차(IMU 자세 ≠ 실제 카메라 자세, §4.2 4번). **아무도 측정한 적이 없다.**
# 값을 지어내지 않고 "허용폭을 넓히는 용도로만" 쓰는 보수적 공칭값 3°를 기본으로 둔다
# (§4.2가 예시로 든 2°보다 크게 — 미측정값은 넉넉히 잡는 쪽이 "진짜를 놓치지 않는다"에 맞다).
DEFAULT_ATTITUDE_RESIDUAL_RAD = math.radians(3.0)

# 점수 결합 가중치. 면적이 주력(in-plane yaw에 무관해 항상 유효), 형상은 보조(코너 4개가
# 있을 때만 계산 가능하고 상한 비교라 신호가 한쪽뿐).
DEFAULT_AREA_WEIGHT = 0.75
DEFAULT_SHAPE_WEIGHT = 0.25

# 시선이 지상 평면과 거의 평행할 때(교점이 무한대) 거르는 하한 — 순수 수치안정성 가드다
# (`core/target.py::_MIN_RAY_PLANE_COS`와 같은 취지·같은 값).
_MIN_RAY_PLANE_COS = 1e-9
# 지상 평면 기저를 만들 때 시드 벡터가 법선과 거의 평행이면 다른 축으로 갈아탄다.
_BASIS_SEED_PARALLEL_COS = 0.9
# 잔차를 더해도 입사각이 90°에 닿지 않게 막는 상한(수치안정성).
_MAX_INCIDENCE_RAD = math.radians(89.0)


@dataclass(frozen=True)
class VehicleAttitude:
    """기체 자세 — **지상 평면 법선 계산에 필요한 최소 필드만**(과설계 금지).

    규약(§4.3 body FRD 기준, `core/frames.py`와 같은 프레임):
      `roll_rad`  : 우현 하강(right-wing-down)이 양수
      `pitch_rad` : 기수 상승(nose-up)이 양수
      `yaw_rad`   : **수평면 법선은 yaw 불변**이라 계산에 쓰이지 않는다. 진단·기록용으로만
                    싣는다(로그에서 "그때 어디를 보고 있었나"를 읽을 수 있어야 하므로).
    """

    roll_rad: float
    pitch_rad: float
    yaw_rad: float = 0.0
    source: str = "euler"  # "euler" | "quaternion" — 어느 경로로 들어왔는지 진단용


def attitude_from_telemetry(telemetry: Optional[dict]) -> Optional[VehicleAttitude]:
    """`telemetry.jsonl`의 `attitude` 필드 → `VehicleAttitude`. **없거나 못 읽으면 None**.

    `utils/frame_source.py::FrameRecord.telemetry`가 이미 `attitude` 자리를 들고 있었지만
    (`blackbox.log_frame(attitude=...)`) 스키마가 정의된 적이 없어 아무도 채우지 않았다 —
    여기서 스키마를 확정한다. 받아들이는 형태(먼저 매칭되는 것 하나):

        {"roll_deg": 0.0, "pitch_deg": -5.0, "yaw_deg": 90.0}    # 권장
        {"roll_rad": 0.0, "pitch_rad": -0.087}
        {"q": [x, y, z, w]}  또는  {"qx":…, "qy":…, "qz":…, "qw":…}

    🔴 **쿼터니언 경로는 "world NED ← body FRD" 회전이라고 가정한다**(성분순서 (x,y,z,w) —
    `TargetEstimate.orientation`/ROS 규약, `core/frames.py` 어댑터 절 참조). 이 저장소에는
    실제 자세 텔레메트리가 아직 **한 건도 없어** 그 가정을 실측으로 검정한 적이 없다. 규약이
    다른 소스를 물리면 조용히 부호가 뒤집히므로, **가능하면 오일러(deg) 형태를 쓸 것.**

    ⚠️ 값이 유한하지 않거나(NaN/inf) 형식이 안 맞으면 **예외를 올리지 않고 None**을 준다 —
    사전정보 결손은 "비활성화"로 우아하게 degrade해야 하는 것이지 파이프라인을 죽일 사유가
    아니다(세션 요구 2번).
    """
    if not isinstance(telemetry, dict):
        return None

    def _finite(value) -> Optional[float]:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    # (1) 오일러 — deg 우선, 없으면 rad.
    for suffix, conv in (("_deg", math.radians), ("_rad", lambda v: v)):
        roll = _finite(telemetry.get(f"roll{suffix}"))
        pitch = _finite(telemetry.get(f"pitch{suffix}"))
        if roll is None or pitch is None:
            continue
        yaw = _finite(telemetry.get(f"yaw{suffix}"))
        return VehicleAttitude(
            roll_rad=conv(roll),
            pitch_rad=conv(pitch),
            yaw_rad=conv(yaw) if yaw is not None else 0.0,
            source="euler",
        )

    # (2) 쿼터니언 (x, y, z, w).
    quat = telemetry.get("q")
    if quat is None and all(k in telemetry for k in ("qx", "qy", "qz", "qw")):
        quat = [telemetry["qx"], telemetry["qy"], telemetry["qz"], telemetry["qw"]]
    if quat is None:
        return None
    comps = [_finite(v) for v in list(quat)[:4]]
    if len(comps) != 4 or any(c is None for c in comps):
        return None
    return quaternion_to_attitude(comps)  # type: ignore[arg-type]


def quaternion_to_attitude(q_xyzw: Sequence[float]) -> Optional[VehicleAttitude]:
    """`(x, y, z, w)` 쿼터니언(world NED ← body FRD) → `VehicleAttitude`. 노름 0이면 None.

    표준 ZYX(yaw-pitch-roll) 분해. `R = R_{ned←frd}` 의 3행이 곧
    `(-sinθ, cosθ·sinφ, cosθ·cosφ)` 이므로 거기서 φ/θ를, 1열에서 ψ를 뽑는다.
    (`ground_normal_cam()`이 쓰는 것과 **같은 3행**이라 두 경로가 구조적으로 일관된다.)
    """
    x, y, z, w = (float(v) for v in q_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + w * z)

    pitch = math.asin(max(-1.0, min(1.0, -r20)))
    roll = math.atan2(r21, r22)
    yaw = math.atan2(r10, r00)
    return VehicleAttitude(roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw, source="quaternion")


def up_vector_body_frd(attitude: VehicleAttitude) -> np.ndarray:
    """기체 body FRD 프레임에서 본 **월드 위쪽(up)** 단위벡터.

    `R_{ned←frd} = Rz(ψ)Ry(θ)Rx(φ)`의 3행이 "월드 아래(down)를 body에서 본 것" =
    `(-sinθ, cosθ·sinφ, cosθ·cosφ)` 이므로 up 은 그 부호를 뒤집은 것이다. **ψ(yaw)는 3행에
    나타나지 않는다** — 수평면 법선이 yaw 불변이라는 사실이 여기서 구조적으로 드러난다.

    검산: 수평(φ=θ=0) → `(0, 0, -1)` = body 위쪽. ✓
    """
    sr, cr = math.sin(attitude.roll_rad), math.cos(attitude.roll_rad)
    sp, cp = math.sin(attitude.pitch_rad), math.cos(attitude.pitch_rad)
    return np.array([sp, -cp * sr, -cp * cr], dtype=np.float64)


def ground_normal_cam(
    attitude: VehicleAttitude, psi_m: float = MOUNT_YAW_PSI_M_RAD_DEFAULT
) -> np.ndarray:
    """카메라 광학 프레임에서 본 **지상 평면 법선(위쪽 방향)** 단위벡터.

    `n̂_cam = R_{frd←cam}(ψ_m)ᵀ · up_frd` — 마운트 회전은 `core/frames.py`의 것을 **그대로
    재사용**한다(두 곳에 회전을 적어두면 한쪽만 고쳐졌을 때 조용히 어긋난다).

    검산(ψ_m=0, 수평): `R_{frd←cam}(0)ᵀ·(0,0,-1) = (0,0,-1)` — 나디르 카메라에서 지면은
    광축(+Z_cam) 방향으로 h만큼 떨어져 있고 그 법선은 카메라를 향한다. ✓
    ψ_m=0, pitch θ: `(0, -sinθ, -cosθ)`.

    🔴 **ψ_m은 이 벡터의 z성분을 바꾸지 않는다**(Rz는 z를 보존). 화면 중앙의 전경축소 크기가
    미측정 ψ_m에 둔감한 이유가 이것이다 — 파일 docstring "미측정 값 2개" 참조.
    """
    n = R_frd_cam(psi_m).T @ up_vector_body_frd(attitude)
    norm = float(np.linalg.norm(n))
    if norm <= 0.0:
        raise ValueError("ground_normal_cam: 법선 노름이 0이다")
    return n / norm


def incidence_angle_rad(normal_cam: np.ndarray, ray_cam: np.ndarray) -> float:
    """시선 `r`과 지상 평면 법선 사이의 **입사각**(0 = 정면으로 평면을 내려다봄).

    `cos(입사각) = |n̂·r̂|`. 전경축소가 `cos³`(면적)/`1/cos`(비등방성)으로 들어오는 그 각이다.
    """
    r = np.asarray(ray_cam, dtype=np.float64).reshape(3)
    n = np.asarray(normal_cam, dtype=np.float64).reshape(3)
    r_norm = float(np.linalg.norm(r))
    if r_norm <= 0.0:
        raise ValueError("incidence_angle_rad: 시선 벡터 노름이 0이다")
    cos_inc = abs(float(n @ r) / r_norm)
    return math.acos(max(0.0, min(1.0, cos_inc)))


def _plane_basis(normal_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """지상 평면 위 **결정론적** 정규직교 기저 `(e1, e2)`, `e1 × e2 = n̂`.

    실제 타겟이 지상에서 어느 방향을 보고 있는지(in-plane yaw γ)는 **알 수 없다** — 이 기저는
    물리적 의미가 아니라 "예상 4각형을 그릴 기준 방향"일 뿐이다. 면적·비등방성 예측은 이 기저
    선택에 **불변**이므로(파일 docstring 참조) 임의로 골라도 점수가 달라지지 않는다.
    """
    n = np.asarray(normal_cam, dtype=np.float64).reshape(3)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(n @ seed)) > _BASIS_SEED_PARALLEL_COS:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - float(n @ seed) * n
    e1 /= float(np.linalg.norm(e1))
    e2 = np.cross(n, e1)
    e2 /= float(np.linalg.norm(e2))
    return e1, e2


@dataclass(frozen=True)
class PredictedGeometry:
    """한 픽셀 위치에서 "진짜 타겟이라면 이렇게 보여야 한다"의 전량.

    **거절 사유가 아니라 진단이다** — 소비자는 이걸 그대로 `Detection.meta`에 실어
    JSONL 블랙박스까지 흘려보낸다(§7.4 "왜 이 후보를 이렇게 봤나"를 사후에 읽을 수 있게).
    """

    ground_point_cam_m: tuple[float, float, float]  # 광선-평면 교점(카메라 광학 프레임, m)
    depth_m: float                                   # 그 점의 카메라 Z(=슬랜트 깊이)
    incidence_rad: float                             # 시선-법선 입사각
    predicted_area_px2: float                        # |det J|·(크기)² — **1차 근사**
    predicted_side_px: float                         # sqrt(예상 면적) — 사람이 읽기 쉬운 형태
    max_anisotropy: float                            # σ1/σ2 — 관측 비등방성의 물리 상한
    predicted_quad_px: tuple                         # 기준 기저(γ=0)에서의 **정확한** 투영 4각형
    predicted_quad_area_px2: float                   # 위 4각형의 shoelace 면적(1차근사 검산용)
    ground_normal_cam: tuple[float, float, float]    # 쓰인 법선(진단 — attitude가 실제로 반영됐나)


def ray_from_pixel(
    pixel_px, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
) -> np.ndarray:
    """픽셀 → 정규화 시선벡터 `(x_n, y_n, 1)`. 왜곡은 `cv2.undistortPoints`로 먼저 푼다
    (`core/target.py::project_pixel_onto_target_plane`과 **같은 관례** — 두 경로가 서로 다른
    왜곡 처리를 하면 같은 픽셀이 두 개의 3D 점을 가리키게 된다)."""
    pt = np.asarray(pixel_px, dtype=np.float64).reshape(1, 1, 2)
    und = cv2.undistortPoints(
        pt,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
    )
    return np.array([float(und[0, 0, 0]), float(und[0, 0, 1]), 1.0], dtype=np.float64)


def predict_planar_target_geometry(
    pixel_px,
    *,
    agl_m: float,
    attitude: VehicleAttitude,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    target_size_m: float,
    psi_m: float = MOUNT_YAW_PSI_M_RAD_DEFAULT,
) -> PredictedGeometry:
    """**핵심 함수** — 입력 소스와 무관한 순수 함수다(라이브/재생 어느 경로가 AGL·자세를
    가져오든 그대로 꽂힌다, 세션 지시 2번).

    `pixel_px`는 후보의 **관측된** 대표 픽셀(중심)이다 — 예측 위치를 따로 만들지 않고 관측
    픽셀에 예측을 앵커링하는 것이 고무마운트 잔차에 견디는 이유다(파일 docstring 참조).

    AGL이 0 이하거나 시선이 평면과 평행하거나 교점이 카메라 뒤면 `ValueError`.
    """
    h = float(agl_m)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError(f"predict_planar_target_geometry: AGL이 양수가 아니다: {agl_m!r}")
    size = float(target_size_m)
    if not math.isfinite(size) or size <= 0.0:
        raise ValueError(f"predict_planar_target_geometry: 타겟 크기가 양수가 아니다: {target_size_m!r}")

    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist_coeffs, dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"predict_planar_target_geometry: fx/fy가 양수가 아니다: {fx}, {fy}")

    n = ground_normal_cam(attitude, psi_m)
    r = ray_from_pixel(pixel_px, K, dist)

    denom = float(n @ r)
    if abs(denom) < _MIN_RAY_PLANE_COS:
        raise ValueError("predict_planar_target_geometry: 시선이 지상 평면과 평행 — 교점 없음")
    s = -h / denom
    if s <= 0.0:
        raise ValueError("predict_planar_target_geometry: 지상 평면 교점이 카메라 뒤쪽")
    point = s * r

    # 면적: |det J| = fx·fy·|n̂·r|³ / h²  (파일 docstring 유도 참조)
    area_scale_px2_per_m2 = fx * fy * abs(denom) ** 3 / (h * h)
    predicted_area = size * size * area_scale_px2_per_m2

    # 비등방성: J = (1/s)·A·E 의 특이값 비. 1/s 는 비에 영향이 없어 생략한다.
    e1, e2 = _plane_basis(n)
    A = np.array(
        [[fx, 0.0, -fx * float(r[0])], [0.0, fy, -fy * float(r[1])]],
        dtype=np.float64,
    )
    J_shape = A @ np.column_stack((e1, e2))
    sv = np.linalg.svd(J_shape, compute_uv=False)
    sigma_max, sigma_min = float(sv[0]), float(sv[-1])
    max_anisotropy = sigma_max / sigma_min if sigma_min > 0.0 else float("inf")

    quad = project_ground_square(
        point, e1, e2, size_m=size, camera_matrix=K, dist_coeffs=dist
    )

    return PredictedGeometry(
        ground_point_cam_m=(float(point[0]), float(point[1]), float(point[2])),
        depth_m=float(point[2]),
        incidence_rad=incidence_angle_rad(n, r),
        predicted_area_px2=float(predicted_area),
        predicted_side_px=float(math.sqrt(max(0.0, predicted_area))),
        max_anisotropy=float(max_anisotropy),
        predicted_quad_px=tuple(tuple(float(v) for v in p) for p in quad),
        predicted_quad_area_px2=float(polygon_area_px2(quad)),
        ground_normal_cam=(float(n[0]), float(n[1]), float(n[2])),
    )


def project_ground_square(
    center_cam_m,
    e1: np.ndarray,
    e2: np.ndarray,
    *,
    size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    in_plane_yaw_rad: float = 0.0,
) -> np.ndarray:
    """지상 평면 위 정사각형(중심 `center_cam_m`, 한 변 `size_m`, 평면 내 회전 γ) →
    **정확한** 투영 4각형 (4,2) float64.

    `predicted_area_px2`(1차 근사)의 검산용이자, "예상 4각형"을 눈으로 보고 오버레이할 수 있게
    하는 산출물이다. 코너 순서는 `(+e1'+e2', -e1'+e2', -e1'-e2', +e1'-e2')`로 결정론적이다.
    """
    c, s = math.cos(float(in_plane_yaw_rad)), math.sin(float(in_plane_yaw_rad))
    a1 = c * np.asarray(e1, dtype=np.float64) + s * np.asarray(e2, dtype=np.float64)
    a2 = -s * np.asarray(e1, dtype=np.float64) + c * np.asarray(e2, dtype=np.float64)
    half = float(size_m) / 2.0
    center = np.asarray(center_cam_m, dtype=np.float64).reshape(3)
    pts3d = np.array(
        [
            center + half * a1 + half * a2,
            center - half * a1 + half * a2,
            center - half * a1 - half * a2,
            center + half * a1 - half * a2,
        ],
        dtype=np.float64,
    )
    zero = np.zeros(3, dtype=np.float64)
    projected, _ = cv2.projectPoints(
        pts3d,
        zero,
        zero,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
    )
    return projected.reshape(-1, 2).astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# 관측량 (Detection에서 뽑는 것)
# ──────────────────────────────────────────────────────────────────────────────


def polygon_area_px2(points) -> float:
    """shoelace 면적(px²). 감김 방향 무관(절댓값)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def observed_side_anisotropy(corners) -> float:
    """정규화된 4각형의 **인접 변 길이비**(≥ 1). 정사각형이 등방 투영되면 1.0.

    `order_quad_corners_clockwise()`로 감김·시작점을 먼저 고정한다 — `approxPolyDP` 출력은
    순서가 보장되지 않아(`core/target.py` 참조) 그냥 쓰면 "인접 변"이 인접이 아니게 된다.
    마주보는 두 변을 평균 내 윤곽 지터를 완화한다.
    """
    q = order_quad_corners_clockwise(corners)
    sides = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    l1 = 0.5 * (sides[0] + sides[2])
    l2 = 0.5 * (sides[1] + sides[3])
    lo, hi = min(l1, l2), max(l1, l2)
    if lo <= 0.0:
        return float("inf")
    return hi / lo


# ──────────────────────────────────────────────────────────────────────────────
# 점수 (0 < score ≤ 1, **절대 0이 되지 않는다**)
# ──────────────────────────────────────────────────────────────────────────────


def _log_gaussian(ratio: float, tolerance_factor: float) -> float:
    """`exp(-½·(ln(ratio)/ln(tol))²)` — 로그축 대칭 감쇠.

    - `ratio == 1` → 1.0 (완전 일치)
    - `ratio == tol` 또는 `1/tol` → `e^-0.5 ≈ 0.607`
    - 아무리 어긋나도 **0이 되지 않는다** — 이 성질이 "거절이 아니라 랭킹" 계약의 수학적 근거다.
    """
    # ratio가 0/음수/NaN이면 애초에 "면적"이 아니다 — 점수를 지어내지 않고 0을 준다.
    # (정상 경로에서는 도달할 수 없다: 관측·예상 면적이 둘 다 양수임을 호출자가 이미 확인한다.)
    if not math.isfinite(ratio) or ratio <= 0.0:
        return 0.0
    tol = float(tolerance_factor)
    if not math.isfinite(tol) or tol <= 1.0:
        raise ValueError(f"tolerance_factor는 1보다 커야 한다: {tolerance_factor!r}")
    z = math.log(ratio) / math.log(tol)
    return math.exp(-0.5 * z * z)


def residual_tolerance_widening(
    incidence_rad: float, attitude_residual_rad: float = DEFAULT_ATTITUDE_RESIDUAL_RAD
) -> float:
    """🔴 **미측정 자세 잔차 θ_res를 허용폭으로 흡수하는 계수**(≥ 1.0).

    §4.2 4번 "고무 마운트의 역설"이 정량화한 잔차는 실측된 적이 없다. 예측 면적이
    `cos³(입사각)`에 비례하므로, 입사각이 ±θ_res 만큼 틀릴 수 있다면 예측 면적은

        [cos³(inc+θ_res),  cos³(inc-θ_res)]

    범위 안에서 흔들린다. 그 폭의 비를 그대로 허용 배수에 곱한다 — **예측값을 편향시키지
    않고 허용폭만 넓히는** 방식이라, 잔차를 모르는 채로도 "진짜 타겟을 놓치는" 쪽으로는
    절대 기울지 않는다.

    수평·나디르(inc=0, θ_res=3°)에서는 1.004로 사실상 무영향, 25° 기울기에서 1.16 — 즉
    **고고도 대각선 시야에서만 눈에 띄게 넓어진다**. 이것이 §4.2의 "고도가 높을수록 위험"과
    같은 방향이다.
    """
    res = abs(float(attitude_residual_rad))
    inc = abs(float(incidence_rad))
    if res <= 0.0:
        return 1.0
    lo = max(0.0, inc - res)
    hi = min(_MAX_INCIDENCE_RAD, inc + res)
    return float((math.cos(lo) / math.cos(hi)) ** 3)


def area_consistency_score(
    observed_area_px2: float,
    predicted_area_px2: float,
    *,
    tolerance_low: float = DEFAULT_AREA_TOLERANCE_LOW,
    tolerance_high: float = DEFAULT_AREA_TOLERANCE_HIGH,
    widening: float = 1.0,
) -> float:
    """관측 면적 ↔ 예상 면적 일치도. **비대칭 허용폭**(작아지는 쪽에 관대) — 근거는
    `DEFAULT_AREA_TOLERANCE_LOW`/`_HIGH` 주석.
    """
    obs, pred = float(observed_area_px2), float(predicted_area_px2)
    if not (math.isfinite(obs) and math.isfinite(pred)) or obs <= 0.0 or pred <= 0.0:
        return 0.0
    ratio = obs / pred
    w = max(1.0, float(widening))
    tol = (tolerance_high if ratio >= 1.0 else tolerance_low) * w
    return _log_gaussian(ratio, tol)


def shape_consistency_score(
    observed_anisotropy: float,
    max_anisotropy: float,
    *,
    tolerance: float = DEFAULT_SHAPE_TOLERANCE,
    margin: float = DEFAULT_ANISOTROPY_MARGIN,
    widening: float = 1.0,
) -> float:
    """관측 비등방성이 **물리 상한**을 넘는 정도. 상한 안이면 1.0(패널티 없음) — 정사각형이
    지상에서 어느 방향으로 놓였는지 모르므로 상한 미만은 전부 정상이다(한쪽 방향 패널티).
    """
    obs = float(observed_anisotropy)
    limit = float(max_anisotropy) * max(1.0, float(margin)) * max(1.0, float(widening))
    if not math.isfinite(obs) or obs <= 0.0:
        return 0.0
    if not math.isfinite(limit) or limit <= 0.0:
        return 1.0
    excess = obs / limit
    if excess <= 1.0:
        return 1.0
    return _log_gaussian(excess, tolerance)


@dataclass(frozen=True)
class PriorScore:
    """후보 하나의 최종 점수 + **그 근거 전량**(§7.4 거절이유 로깅 철학 — 점수만 남기면
    현장에서 "왜 이 후보가 밀렸나"를 되짚을 수 없다)."""

    score: float
    area_score: Optional[float]
    shape_score: Optional[float]
    observed_area_px2: float
    predicted_area_px2: float
    area_ratio: float
    observed_anisotropy: Optional[float]
    max_anisotropy: float
    incidence_deg: float
    depth_m: float
    tolerance_widening: float
    components: tuple = field(default=())  # 실제로 쓰인 성분 이름들


def combine_scores(
    area_score: Optional[float],
    shape_score: Optional[float],
    *,
    area_weight: float = DEFAULT_AREA_WEIGHT,
    shape_weight: float = DEFAULT_SHAPE_WEIGHT,
) -> tuple[float, tuple]:
    """가용한 성분만 가중평균하고 가중치를 재정규화한다 — 코너가 없어 형상 점수를 못 내는
    후보가 그 사실만으로 불리해지면 안 되기 때문이다(면적만으로도 온전한 점수가 나와야 한다).
    성분이 하나도 없으면 `(1.0, ())` — **점수를 못 매기는 것은 감점 사유가 아니다.**
    """
    pairs = []
    if area_score is not None:
        pairs.append(("area", float(area_score), float(area_weight)))
    if shape_score is not None:
        pairs.append(("shape", float(shape_score), float(shape_weight)))
    total_w = sum(w for _, _, w in pairs)
    if not pairs or total_w <= 0.0:
        return 1.0, ()
    value = sum(s * w for _, s, w in pairs) / total_w
    return float(value), tuple(name for name, _, _ in pairs)


def score_candidate(
    *,
    center_px,
    corners=None,
    observed_area_px2: Optional[float] = None,
    agl_m: float,
    attitude: VehicleAttitude,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    target_size_m: float,
    psi_m: float = MOUNT_YAW_PSI_M_RAD_DEFAULT,
    area_tolerance_low: float = DEFAULT_AREA_TOLERANCE_LOW,
    area_tolerance_high: float = DEFAULT_AREA_TOLERANCE_HIGH,
    shape_tolerance: float = DEFAULT_SHAPE_TOLERANCE,
    anisotropy_margin: float = DEFAULT_ANISOTROPY_MARGIN,
    attitude_residual_rad: float = DEFAULT_ATTITUDE_RESIDUAL_RAD,
    area_weight: float = DEFAULT_AREA_WEIGHT,
    shape_weight: float = DEFAULT_SHAPE_WEIGHT,
) -> tuple[PriorScore, PredictedGeometry]:
    """후보 하나 → `(PriorScore, PredictedGeometry)`. **순수 함수** — 입력 소스를 모른다.

    `observed_area_px2`를 주면 그걸 쓰고, 안 주면 `corners`의 shoelace 면적을 쓴다(둘 다 없으면
    `ValueError`). 형상 점수는 `corners`가 4개일 때만 산출되고 없으면 `None`이다.
    """
    predicted = predict_planar_target_geometry(
        center_px,
        agl_m=agl_m,
        attitude=attitude,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        target_size_m=target_size_m,
        psi_m=psi_m,
    )
    widening = residual_tolerance_widening(predicted.incidence_rad, attitude_residual_rad)

    if observed_area_px2 is None:
        if corners is None:
            raise ValueError("score_candidate: observed_area_px2 또는 corners 중 하나는 필요하다")
        observed_area_px2 = polygon_area_px2(corners)
    obs_area = float(observed_area_px2)

    area_score = area_consistency_score(
        obs_area,
        predicted.predicted_area_px2,
        tolerance_low=area_tolerance_low,
        tolerance_high=area_tolerance_high,
        widening=widening,
    )

    obs_aniso: Optional[float] = None
    shape_score: Optional[float] = None
    if corners is not None and len(np.asarray(corners).reshape(-1, 2)) == 4:
        try:
            obs_aniso = observed_side_anisotropy(corners)
        except ValueError:
            obs_aniso = None  # 축퇴 사각형 — 형상 성분만 포기하고 면적 점수는 살린다
        if obs_aniso is not None:
            shape_score = shape_consistency_score(
                obs_aniso,
                predicted.max_anisotropy,
                tolerance=shape_tolerance,
                margin=anisotropy_margin,
                widening=widening,
            )

    value, components = combine_scores(
        area_score, shape_score, area_weight=area_weight, shape_weight=shape_weight
    )
    score = PriorScore(
        score=value,
        area_score=area_score,
        shape_score=shape_score,
        observed_area_px2=obs_area,
        predicted_area_px2=predicted.predicted_area_px2,
        area_ratio=(
            obs_area / predicted.predicted_area_px2
            if predicted.predicted_area_px2 > 0.0
            else float("inf")
        ),
        observed_anisotropy=obs_aniso,
        max_anisotropy=predicted.max_anisotropy,
        incidence_deg=math.degrees(predicted.incidence_rad),
        depth_m=predicted.depth_m,
        tolerance_widening=widening,
        components=components,
    )
    return score, predicted
