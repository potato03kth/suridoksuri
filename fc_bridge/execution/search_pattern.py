"""비전 착륙 탐색 비행의 순수 기하 로직.

마지막 WP(WP1) 상공 MC 상태에서 초록 매트(3.0m×3.0m)를 카메라로 찾는 국면의
계산만 담는다 — 카메라 풋프린트, 나선 탐색 궤적, 선회속도 상한, 검출 고도 상한.

`state_logic.py` 와 같은 규율이다: **numpy/math 만 쓴다.** ROS·rclpy·파일 I/O·
로깅·wall-clock·난수 없음 → 단독 import·단위테스트 가능하고, 같은 입력이면
항상 같은 출력이다.

--------------------------------------------------------------------------
🔴 화각(FOV) 가정 — 이 파일의 모든 숫자가 여기에 걸려 있다
--------------------------------------------------------------------------
장착 카메라(CAM109-IMX708AF-75)의 실측 화각은 **75°** 인데, 그 75°가 **수평인지
대각인지가 미해결**이다. `vision/calibration/cam109-imx708af75/nominal.yaml` 의
`hfov_assumption.note` 가 "대각일 가능성이 있고, 실측 캘리브레이션 재개 시 검정
대상" 이라고 명시한 그 미해결 항목이다. 16:9 프레임에서 두 해석은 이렇게 갈린다.

  ① 낙관(75° = **수평**):
        half_tan_h = tan(37.5°)              = 0.7673269880
        half_tan_v = tan(37.5°) · (9/16)     = 0.4316214307
        footprint  = 1.5347·h  ×  0.8632·h
  ② 보수(75° = **대각**): 대각 half-tan 0.7673269880 을 16:9 로 분해한다.
        대각 화소 길이 ∝ √(16² + 9²) = √337 = 18.3575597507
        half_tan_h = tan(37.5°) · 16/√337    = 0.6687834317
        half_tan_v = tan(37.5°) ·  9/√337    = 0.3761906803
        footprint  = 1.3375·h  ×  0.7524·h

**기본값(`DEFAULT_HALF_*FOV_TAN`)은 ②(보수)를 쓴다.** 커버리지를 과대평가하면
탐색 구멍이 생기기 때문이다 — 실제 화각이 대각 75°(좁음)인데 수평 75°(넓음)로
믿고 링 간격을 벌리면, 나선 이웃 링 사이에 카메라가 한 번도 보지 않은 띠가
남는다. 그 띠에 매트가 있으면 탐색은 "다 훑었는데 없다"로 끝난다.

⚠️ 상수 리터럴은 세션 지시가 준 반올림값(0.66876 / 0.37618)을 그대로 쓴다.
   위 정확한 유도값보다 각각 2.3e-5 / 1.1e-5 **작다**(상대 3.5e-5 / 2.8e-5).
   작은 쪽 = 풋프린트 과소평가 = 링 간격이 더 좁아짐 = 안전측이라 그대로 둔다.

--------------------------------------------------------------------------
🔴 "보수적"이 커버리지와 검출에서 **반대 상수를 가리킨다** (이 파일 최대의 함정)
--------------------------------------------------------------------------
같은 "안전한 가정"이 두 계산에서 서로 다른 half-tan 을 고른다:

  * **커버리지(링 간격)** — 화각이 **좁을수록** 위험(구멍) → 보수 = **좁은 쪽**
    (`DEFAULT_HALF_HFOV_TAN` = 0.66876).
  * **검출 고도 상한** — 화각이 **넓을수록** 위험. 넓으면 지상폭이 커지고
    GSD가 거칠어져 타겟이 차지하는 픽셀 수가 줄어든다 → 같은 고도에서 검출이
    더 어렵다 → 보수 = **넓은 쪽**(`DETECTION_HALF_HFOV_TAN` = 0.76733).

그래서 `camera_footprint_m()` 의 기본값과 `detection_altitude_ok()` /
`max_detection_altitude_m()` 의 기본값이 **일부러 다른 상수**다. 이 둘을
"통일"하면 한쪽이 반드시 위험한 방향으로 뒤집힌다.

--------------------------------------------------------------------------
🔴 검출 고도 상한 실측 계산값 (이 파일에서 직접 계산해 확인한 숫자)
--------------------------------------------------------------------------
`vision/presets/distress_coarse.yaml` 의 `rect_detector.min_area = 8000` px²,
매트 한 변 3.0m, coarse 다운스케일 프레임 폭 1536px 기준:

    max_detection_altitude_m = (3.0 · 1536) / (2 · half_hfov_tan · √8000)

  * 보수 half-tan 0.66876(좁은 화각 가정) → **38.518 m**
  * 낙관 half-tan 0.76733(넓은 화각 가정) → **33.570 m**   ← 채택(안전측)

**✅ 판정: 기본 탐색고도 25m 는 검출 상한 안이다.** 넓은 화각 가정(33.570m)
에서도 8.57m 여유가 있다. 25m 에서 매트 명목 면적은 다음과 같다(낙관 기준):

    지상폭 2·25·0.76733 = 38.366 m → GSD 2.498 cm/px
    한 변 120.1 px → 면적 **14,425 px²**  (min_area 8000 의 **1.80배**)

⚠️ 다만 `distress_coarse.yaml` 헤더가 `min_area=8000` 을 정한 근거는
   "20m 명목 22,500px² 가 색상 마스킹/윤곽 근사로 **명목의 ~35%까지 열화돼도**
   통과" 였다. 25m 에서는 그 열화 허용폭이 **55.5%(=8000/14425)** 로 줄어든다 —
   즉 25m 는 "검출 가능"이지만 **열화 마진은 20m 대비 1.56배 얇다.**
   현장에서 검출률이 떨어지면 탐색고도를 20m 쪽으로 내리는 것이 첫 번째 카드다.

(참고: 40m 는 명목 5,635px² 로 min_area 미달 — coarse 프리셋이 **의도적으로**
배제하는 고도다. 그 배제가 위 33.570m 상한과 정합한다.)
"""
import math
from dataclasses import dataclass, replace

import numpy as np

# ── 화각 상수 (파일 docstring "화각 가정" 절이 유도 전문) ──────────

# 보수(75° = 대각) — 커버리지/링 간격의 기본값. 좁은 화각을 가정해야 탐색
# 구멍이 안 생긴다.
DEFAULT_HALF_HFOV_TAN = 0.66876      # tan(37.5°)·16/√337 = 0.6687834317
DEFAULT_HALF_VFOV_TAN = 0.37618      # tan(37.5°)· 9/√337 = 0.3761906803

# 낙관(75° = 수평) — 화각이 이만큼 넓다면. 커버리지 계산에 쓰면 위험하지만
# **검출 고도 상한에는 이쪽이 안전측**이다(파일 docstring 참조).
OPTIMISTIC_HALF_HFOV_TAN = 0.76733   # tan(37.5°)       = 0.7673269880
OPTIMISTIC_HALF_VFOV_TAN = 0.43162   # tan(37.5°)·(9/16)= 0.4316214307

# 검출(GSD) 계산의 기본 half-tan. **일부러 낙관(넓은) 쪽이다** — 넓은 화각이
# 검출에는 불리하므로 그쪽을 가정해야 고도 상한을 과대평가하지 않는다.
# `vision/presets/distress_coarse.yaml` 의 도출 주석이 쓴 "gw(h) ≈ 1.535·h"
# 공식과도 같은 상수라(2·0.76733 = 1.53466) 그 표와 직접 대조된다.
DETECTION_HALF_HFOV_TAN = OPTIMISTIC_HALF_HFOV_TAN

# `vision/presets/distress_coarse.yaml` 실측 스펙 — 여기 기본값은 그 파일의
# 값을 **읽어오는 게 아니라 복제한** 것이다(fc_bridge 는 vision 을 import 하지
# 않는다, 도메인 간 교차 import 0 규칙). 두 값이 갈라지면 테스트가 red 가 된다.
DISTRESS_MAT_SIZE_M = 3.0            # 매트 한 변 (실측 3.0m×3.0m×0.105m)
COARSE_MIN_AREA_PX = 8000.0          # rect_detector.min_area
COARSE_FRAME_WIDTH_PX = 1536.0       # coarse 다운스케일 프레임 폭

_TWO_PI = 2.0 * math.pi


def camera_footprint_m(agl_m: float,
                       half_hfov_tan: float = DEFAULT_HALF_HFOV_TAN,
                       half_vfov_tan: float = DEFAULT_HALF_VFOV_TAN
                       ) -> "tuple[float, float]":
    """(폭_m, 높이_m). 나디르 가정 — tilt 보정 없음.

    폭 = 이미지 가로축(수평 화각) 방향 지상폭, 높이 = 세로축 방향 지상폭.
    지상폭 = 2 · AGL · tan(화각/2) 라는 핀홀 관계 그대로다.

    **tilt 보정을 일부러 넣지 않았다.** 탐색 국면이 보는 것은 "이 링을 돌 때
    매트가 화각 안에 한 번이라도 들어오느냐(있냐/없냐)" 뿐이고, 정밀 위치는
    그 뒤 정밀착륙 단계(vision `TargetEstimate` 상대 pose)가 담당한다.
    기울면 풋프린트가 사다리꼴로 찌그러지면서 **면적은 오히려 늘어나므로**
    (`vision/core/prior_geometry.py`: 면적 ∝ cos³θ 로 픽셀당 면적이 줄고
    지상 커버는 늘어난다) 나디르 가정은 커버리지 측면에서 과소평가 = 안전측이다.
    선회 중 tilt 자체는 `max_turn_speed()` 로 애초에 작게 묶는다.

    기본 half-tan 이 보수(대각 75°) 쪽인 이유는 파일 docstring 참조.
    """
    h = float(agl_m)
    return (2.0 * h * float(half_hfov_tan), 2.0 * h * float(half_vfov_tan))


def ring_spacing_from_footprint(footprint_short_m: float,
                                overlap_ratio: float = 0.35) -> float:
    """겹침 비율을 뺀 유효 진행폭. overlap 35% → spacing = short × 0.65.

    footprint_short_m 에는 `camera_footprint_m()` 이 준 두 값 중 **짧은 쪽**을
    넣는다(16:9 나디르에서는 세로 = 0.7524·h). 긴 쪽을 넣으면 기체가 어느
    방향으로 진행하느냐에 따라 커버가 안 되는 방위가 생긴다 — 짧은 쪽으로
    묶는 것이 방위와 무관하게 성립하는 유일한 값이다.

    겹침이 필요한 이유(왜 spacing = footprint 가 아닌가): ①이웃 링 사이 이음매
    에서 매트가 **절반만** 걸치면 `min_area` 하드 필터에 미달해 검출이 통째로
    사라진다(면적 필터는 부분 blob 을 봐주지 않는다) ②나선 궤적 자체가 위치
    오차·바람·EKF 드리프트로 이상궤적에서 벗어난다 ③나디르 가정이 실제 tilt 로
    깨지면 풋프린트가 옆으로 밀린다. 35%는 이 셋에 대한 잠정 여유이며 실기체
    미검증이다 — 좁힐 근거가 생기기 전까지 벌리지 말 것.
    """
    ratio = float(overlap_ratio)
    if not (0.0 <= ratio < 1.0):
        raise ValueError(f"overlap_ratio 는 [0, 1) — 받은 값: {overlap_ratio!r}")
    return float(footprint_short_m) * (1.0 - ratio)


def max_turn_speed(radius_m: float,
                   max_tilt_rad: float = math.radians(6.0),
                   g: float = 9.80665) -> float:
    """원 선회 시 구심가속도가 만드는 기체 tilt 상한에서 역산한 속도 상한.

        a = v²/r,  tilt = atan(a/g)  →  v = √(r · g · tan(tilt_max))

    **왜 tilt 를 묶는가:** 나디르 하드마운트 카메라가 기체와 함께 기울면 시선이
    `AGL · tan(tilt)` 만큼 지상에서 어긋난다(`vision/core/frames.py` §4.5 가
    쓰는 그 식). 25m AGL·6° 면 2.63m — 링 간격(수 m 급)과 같은 자릿수라
    탐색 커버리지가 통째로 틀어진다. 즉 이 상한은 승차감이 아니라 **커버리지
    기하를 지키기 위한** 제약이다.

    tilt 6°는 잠정값이다(실기체 미검증). 기울기가 커질수록 위 오프셋이
    비선형으로 커지므로 벌릴 때는 반드시 `AGL·tan(tilt)` 를 링 간격과 함께
    다시 볼 것.

    반경 0(제자리 선회)이면 0.0 을 준다 — 구심가속도가 없으므로 tilt 제약이
    속도를 묶지 않지만, 이 함수의 반환값은 "원 궤도 속도"라 0이 옳다.
    호출부는 이 값을 하한(예: 최소 탐색속도)과 함께 clamp 해야 한다.
    """
    r = float(radius_m)
    if r < 0.0:
        raise ValueError(f"radius_m 은 음수일 수 없다 — 받은 값: {radius_m!r}")
    tilt = float(max_tilt_rad)
    if not (0.0 <= tilt < math.pi / 2.0):
        raise ValueError(f"max_tilt_rad 는 [0, π/2) — 받은 값: {max_tilt_rad!r}")
    return math.sqrt(r * float(g) * math.tan(tilt))


@dataclass(frozen=True)
class SpiralState:
    """아르키메데스 나선 탐색의 진행 상태.

    theta_rad   : 누적 극각 (rad). 포화 후에도 계속 증가한다.
    radius_m    : 현재 반경 (m). `max_radius_m` 에서 **포화**한다.
    revolutions : 누적 회전수 = theta_rad / 2π (진행도 판정용).

    revolutions_at_max_radius:
        최대반경에 처음 도달한 순간의 누적 회전수. 아직 도달 전이면 None.
        🔴 **기본값이 있는 4번째 필드라 `SpiralState(theta, radius, revs)` 3인자
        생성은 그대로 동작한다.** 이 필드를 둔 이유는 `spiral_complete()` 가
        "최대반경에서 1바퀴 더" 를 판정하려면 *언제* 포화됐는지를 알아야 하는데,
        포화 후에는 `radius_m` 이 상수(max)라 (theta, radius, revolutions)
        셋만으로는 포화 시점을 복원할 방법이 **수학적으로 없기** 때문이다
        (포화 전에는 radius = r_start + ring_spacing·revolutions 로 역산되지만
        포화 후에는 그 관계가 끊긴다). 포화 시점을 상태에 명시적으로 남기는
        것이 "모르는 값을 추정해 채우지 않는다"는 이 저장소의 규율과도 맞다.
    """
    theta_rad: float
    radius_m: float
    revolutions: float
    revolutions_at_max_radius: "float | None" = None


def spiral_advance(st: SpiralState, dt_s: float, speed_mps: float,
                   ring_spacing_m: float, r_start_m: float,
                   max_radius_m: float) -> SpiralState:
    """아르키메데스 나선을 등속(속도 speed_mps)으로 dt만큼 전진시킨다.

        r(θ) = r_start + b·θ,        b = ring_spacing / 2π

    한 바퀴(θ +2π)마다 반경이 정확히 `ring_spacing_m` 늘어나는 나선이다 —
    즉 `ring_spacing_from_footprint()` 가 정한 "이웃 링 간 거리"가 그대로
    나선 링 간격이 된다.

    **등속 조건:** 접선속도 |dP/dt| = √(r² + (dr/dθ)²)·dθ/dt 이고 dr/dθ = b 이므로

        dθ/dt = speed / √(r² + b²)

    🔴 **`√(r² + b²)` 의 b² 를 빼고 `r` 로 쓰면 r→0 에서 dθ/dt 가 발산한다**
    (r=0 이면 0 나눗셈). 나선 중심에서 출발하는 것이 이 탐색의 정상 시나리오
    (마지막 WP 바로 위에서 시작)이므로 그 발산은 이론적 걱정이 아니라 첫 틱에
    바로 터지는 경로다. b² 항이 있으면 r=0 에서도 dθ/dt = speed/b 로 유한하다
    — 물리적으로도 옳다(중심 근처에서 궤적은 거의 직선이고, 속도는 각속도가
    아니라 접선속도로 묶여 있다).

    적분은 전진 오일러 1스텝이다. 등속 계약의 이산화 오차는
    `≈ b·speed·dt / (2r²)` (상대) 로, 제어틱(10Hz·수 m/s)에서 무시 가능하고
    **항상 speed·dt 를 살짝 넘는 방향**이다(반경이 스텝 도중 커지므로).
    2차 보정(중점법)을 넣지 않은 것은 의도적이다 — 위 식이 코드에 딱 한 번만
    나타나야 "sqrt 의 b² 를 지웠다" 같은 파괴가 한 곳만 고쳐도 재현되고,
    회귀테스트의 그물이 성기게 찢어지지 않는다.

    radius 는 `max_radius_m` 에서 **포화**시킨다(넘어가지 않음) — 포화 후에는
    그냥 최대반경 원을 계속 돈다. **완주 판정은 호출자가 한다**
    (`spiral_complete()` 참조).

    ⚠️ 포화 후에는 dr/dθ 가 0(원)이라 엄밀한 등속 스텝은 dθ = speed·dt/r 인데,
    이 함수는 분기 없이 계속 √(r²+b²) 를 쓴다 → 실제 속도가 계수
    `r/√(r²+b²) = 1 − b²/(2r²)` 만큼 **느려진다**. 포화 반경은 정의상 가장 큰
    반경이므로(예 r=50m·ring_spacing=3m → b=0.477, 계수 1−4.6e-5) 오차가
    무시 가능하고 방향도 안전측(느림)이다. 분기를 안 넣은 이유는 위와 같다 —
    핵심 식이 코드에 두 번 나타나면 파괴 한 번이 한쪽만 고쳐 그물이 찢어진다.
    """
    b = float(ring_spacing_m) / _TWO_PI
    r = float(st.radius_m)
    denom = math.hypot(r, b)          # = √(r² + b²), r=0 에서도 유한

    if denom <= 0.0:
        # r=0 이면서 ring_spacing=0 — 나선이 아니라 점이다. 각도를 지어내지
        # 않고 상태를 그대로 돌려준다(무한대/NaN 을 상태에 심는 것보다 낫다).
        return st

    dtheta = float(speed_mps) * float(dt_s) / denom
    theta = float(st.theta_rad) + dtheta
    revs = theta / _TWO_PI

    r_unsat = float(r_start_m) + b * theta
    r_max = float(max_radius_m)
    radius = min(r_unsat, r_max)

    marker = st.revolutions_at_max_radius
    if marker is None and r_unsat >= r_max:
        # 포화 시점은 근사하지 않고 **닫힌 형태로** 박는다:
        # r_start + ring_spacing·rev = max_radius  →  rev = (max−r_start)/spacing.
        # (이번 틱에 스텝 도중 넘었으므로 이번 틱의 revs 를 그대로 쓰면
        #  스텝 크기만큼 늦게 잡혀 "1바퀴 더"가 짧아진다.)
        if float(ring_spacing_m) > 0.0:
            marker = max(0.0, (r_max - float(r_start_m)) / float(ring_spacing_m))
        else:
            marker = revs

    return replace(st, theta_rad=theta, radius_m=radius, revolutions=revs,
                   revolutions_at_max_radius=marker)


def spiral_point_ne(st: SpiralState, center_ne) -> np.ndarray:
    """중심 기준 NE 평면 좌표 [N, E].

        N = c_n + r·cos(θ)      E = c_e + r·sin(θ)

    🔴 **이 저장소의 수평 좌표는 NED 의 [N, E] 다 (ENU 아님).**
    `offboard_node._publish_pos_setpoint` 의 `pos_ned=[N, E, h_up]` 규약과
    일치시킨다 — 같은 배열을 그대로 세트포인트 앞 두 성분에 실을 수 있다.
    따라서 θ=0 은 **정북(+N)**, θ=+π/2 는 **정동(+E)** 이고 θ 증가는
    (위에서 내려다볼 때) 시계방향이다. cos/sin 을 뒤바꾸면 축이 통째로
    거울상이 되어 탐색 궤적이 조용히 다른 곳을 훑는다.
    """
    c = np.asarray(center_ne, dtype=float)[:2]
    r = float(st.radius_m)
    th = float(st.theta_rad)
    return np.array([c[0] + r * math.cos(th), c[1] + r * math.sin(th)])


def spiral_complete(st: SpiralState, max_radius_m: float, r_start_m: float,
                    extra_revolutions: float = 1.0,
                    radius_tol_m: float = 1e-9) -> bool:
    """최대반경에 도달하고 그 반경에서 최소 1바퀴를 더 돌았으면 True.

    "1바퀴 더"를 요구하는 이유: 나선이 최대반경에 닿는 순간은 **한 방위**에서만
    닿은 것이라, 그 반경대의 나머지 방위는 아직 한 번도 안 봤다. 최대반경
    원을 한 바퀴 채워야 탐색 영역 가장자리가 전부 커버된다.

    포화 시점(`SpiralState.revolutions_at_max_radius`)이 None 인데 반경 조건은
    만족하는 상태 — 즉 `spiral_advance()` 를 한 번도 안 거친 손수 만든 상태 —
    는 `r_start_m` 으로 가른다: 시작반경이 이미 최대반경 이상이면 "처음부터
    포화된 원 궤도"로 보고 0회전 지점에서 포화된 것으로 취급하고, 그렇지
    않으면 **미완주로 본다**(모르면 계속 찾는다 = 안전측).
    """
    if float(st.radius_m) < float(max_radius_m) - float(radius_tol_m):
        return False
    marker = st.revolutions_at_max_radius
    if marker is None:
        if float(r_start_m) < float(max_radius_m):
            return False
        marker = 0.0
    return float(st.revolutions) >= float(marker) + float(extra_revolutions)


def detection_altitude_ok(agl_m: float,
                          target_size_m: float = DISTRESS_MAT_SIZE_M,
                          min_area_px: float = COARSE_MIN_AREA_PX,
                          frame_width_px: float = COARSE_FRAME_WIDTH_PX,
                          half_hfov_tan: float = DETECTION_HALF_HFOV_TAN) -> bool:
    """그 고도에서 타겟이 coarse 검출 `min_area` 를 넘는가.

        지상폭 gw = 2·agl·half_hfov_tan
        GSD       = gw / frame_width_px            [m/px]
        한 변 px  = target_size_m / GSD
        면적      = 한 변²      ≥ min_area_px  →  True

    `vision/presets/distress_coarse.yaml` 헤더의 도출 표와 **같은 공식**이다
    (그 표는 낙관 half-tan 으로 계산돼 있다: gw ≈ 1.535·h = 2·0.76733·h).
    10m→~90,000px² / 20m→~22,500px² / 40m→~5,625px² 이 그 표의 값이고
    이 함수가 그대로 재현한다.

    기본 half-tan 이 **낙관(넓은) 쪽**인 이유는 파일 docstring 참조 — 검출에는
    넓은 화각이 불리하므로 그쪽을 가정해야 고도 상한을 과대평가하지 않는다.
    """
    return float(agl_m) <= max_detection_altitude_m(
        target_size_m=target_size_m, min_area_px=min_area_px,
        frame_width_px=frame_width_px, half_hfov_tan=half_hfov_tan)


def max_detection_altitude_m(target_size_m: float = DISTRESS_MAT_SIZE_M,
                             min_area_px: float = COARSE_MIN_AREA_PX,
                             frame_width_px: float = COARSE_FRAME_WIDTH_PX,
                             half_hfov_tan: float = DETECTION_HALF_HFOV_TAN
                             ) -> float:
    """`detection_altitude_ok()` 가 참인 **최대 고도** (m). 닫힌 형태.

        면적(h) = (target_size_m · frame_width_px)² / (2·h·half_hfov_tan)²
        면적(h) ≥ min_area_px
          ⟺  h ≤ (target_size_m · frame_width_px)
                 / (2 · half_hfov_tan · √min_area_px)

    면적이 h² 에 반비례하는 단조감소 함수라 부등식이 한 번에 풀린다 —
    이분탐색을 쓸 이유가 없다(수치 오차·반복 횟수·경계 애매함만 늘어난다).

    🔴 **실제 계산값** (target 3.0m, min_area 8000px², frame 1536px):
        보수 half-tan 0.66876 → **38.518 m**
        낙관 half-tan 0.76733 → **33.570 m**   ← 기본값(안전측)
    기본 탐색고도 25m 는 둘 다에서 상한 안이다(파일 docstring "판정" 절).

    `min_area_px <= 0` 이면 상한이 없으므로 +inf 를 준다(면적 조건 비활성).
    """
    ht = float(half_hfov_tan)
    if ht <= 0.0:
        raise ValueError(f"half_hfov_tan 은 양수 — 받은 값: {half_hfov_tan!r}")
    area = float(min_area_px)
    if area <= 0.0:
        return float("inf")
    return (float(target_size_m) * float(frame_width_px)) / (
        2.0 * ht * math.sqrt(area))
