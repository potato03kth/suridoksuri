"""
공통 상태머신 + 안전 폴백 (`docs/vision_plan.md` §5.1, §9 빌드순서 6번).

```
ACQUIRE(대형 coarse 피처) → CENTER_DESCEND(중심 맞추며 하강)
  → LOCK(fine 피처 확정·검증) → PRECISION_SERVO(폐루프) → TERMINAL(AGL/블라인드 하강)
```
검출 상실·후보 모호는 `HOLD`로 빠진다. **TERMINAL(AGL ≤ `terminal_agl_m`)에서는 재상승하지
않는다** — 2026-07-28 사용자 확정 "밀어붙이기" 기조 전환(§5.1 갱신 서술 참조). 마지막 유효
추정이 착륙 기하가 이미 예산으로 잡아둔 드리프트(`max_terminal_commit_drift_m`) 안이면 블라인드
데드레코닝을 계속하고, 블라인드가 길어지면 **재상승이 아니라 최종 커밋**(`command="land"` =
AUTO.LAND 인계)으로 끝낸다. 예산을 넘으면 `HOLD`로 거절한다 — 밀어붙이면 0.105m 라이즈드
매트 밖에 걸쳐 전복되기 때문이다. 옛 재상승 동작은 `allow_terminal_abort_ascend=True` 한 줄로
정확히 복원된다.

**타겟 종류 무관 공통 골격** — 버티포트/조난자/십자 전용 분기 없음(§9 6번 핵심 요구). 타겟별
특수성은 `Observation`에 실려 들어오는 값(`fine_locked` 등)으로만 표현되고, 그 값을 어떻게
계산하는지는 호출자(`main.py`/`replay.py`)의 책임이다.

**코어는 순수 로직이다** — `vision/CLAUDE.md` import 규칙("core/ ← numpy, opencv만 허용")을
따라 파일 I/O·로깅·yaml 로드를 하지 않는다(`core/target.py`와 동일 패턴). wall-clock/난수도
쓰지 않는다 — `Observation.ts`는 호출자가 넘겨준 값을 그대로 쓸 뿐이라 같은 관측열이면 항상
같은 상태열이 나온다(§7.5 결정론).

이 파일이 뱉는 `Decision.command`는 문자열 힌트일 뿐이다 — 이걸 소비해 실제 기체를 움직이는
쪽(`fc_ros`/`fc_bridge`)은 이번 범위 밖(§9 7번, 다른 세션).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# 정규화 중심오차 -> 지상거리 환산 계수 `tan(HFOV/2)`의 nominal 기본값.
#
# **화각(도)이 아니라 인트린식에서 유도한다**: 핀홀 모델에서 화면 절반폭이 곧
# `fx · tan(HFOV/2)` 이므로 `tan(HFOV/2) = (width/2) / fx` 다. 이렇게 두면
# `calibration/cam109-imx708af75/nominal.yaml`의 `hfov_assumption`이 **수평인지 대각인지 아직
# 미해결**(그 yaml의 note)이라는 문제를 아예 우회한다 — `fx`는 가정이 아니라 실제로 solvePnP가
# 쓰는 값이고, 두 값은 서로 모순될 수 없다.
#
#   nominal.yaml: width=4608, fx=3002.6312590261377 -> 2304/3002.63 = 0.767327...
#   (참고: tan(37.5°) = 0.767327... 로 실제로 일치한다 — HFOV 75° 가정과 자기무모순)
#
# 실행 경로(`main.py`/`replay.py`)는 **로드한 캘리브레이션에서 직접 계산해 덮어쓴다**
# (`half_hfov_tan_from_calibration()`), 그래서 이 상수는 캘리브 로드 실패 시의 폴백일 뿐이다.
NOMINAL_HALF_HFOV_TAN = 0.7673269879789604


# TERMINAL 최종 커밋("밀어붙이기")을 허용하는 이탈 추정 상한(미터).
#
# **지어낸 값이 아니라 착륙점 기하가 이미 예산으로 잡아둔 바로 그 δ다.**
# `modules/distress_box.py::compute_landing_window()`는 착륙점 거리 `d`를 다음 상한으로 묶는다:
#
#     d ≤ M − R − δ        (M=매트 반변 1.5m, R=기체 최외곽 반경, δ=허용 드리프트)
#     ⟹  매트 가장자리 여유 = M − d − R ≥ δ
#
# 즉 **δ 이내의 드리프트로 착지하면 기체 최외곽이 매트 안에 남는다는 것이 착륙점 선정의 전제**다
# (`d`는 창의 중점을 고르므로 실제 여유는 더 크다 — R=0.60 기본값에서 `d=0.5621`,
# 실제 여유 `1.5−0.5621−0.60 = 0.3379m` > δ. 그래서 이 기본값은 실제 여유의 **보수적 하한**이다).
# δ 자체는 `docs/vision_plan.md` §1 요구사항 "최종 정확도 <30cm"이고
# `modules/distress_box.py::LANDING_DRIFT_ALLOWANCE_M_DEFAULT`와 **같은 숫자**다
# (`core/`는 `modules/`를 import할 수 없어 — import 규칙 — 값을 복제한다.
#  두 값이 갈라지면 `tests/test_state_machine.py`가 red가 된다).
#
# 🔴 **`max_drift_estimate_m`(0.75)와 일부러 다른 값이다.** 그쪽은 "데드레코닝을 믿을 수 있는가"를
# 보는 안전 폴백 게이트고, 이쪽은 "빗나간 채로 내려앉아도 매트 위인가"를 보는 **기하 게이트**다.
# 하나로 합치면 0.75m 드리프트로 커밋해 매트 밖(여유 0.3379m)에 내려앉는다.
TERMINAL_COMMIT_DRIFT_M_DEFAULT = 0.30

# 🔴 **이 임계는 실측 근거가 없다**(`core/frames.py`의 `MOUNT_YAW_PSI_M_MEASURED` 패턴).
# 위 유도는 `d ≤ M − R − δ`에 기대는데, 그 `R`(기체 최외곽 반경)이 **미측정**이다 —
# 저장소가 아는 사실은 "0.5m 초과"뿐이고 `AIRCRAFT_RADIUS_M_DEFAULT = 0.60`은 공칭값이다.
# 실측 R이 0.60을 넘으면 실제 매트 여유가 `0.3379 − (R_실측 − 0.60)`으로 줄어 δ=0.30을 깎아먹는다.
#
# **내일 R 실측이 들어오면 고칠 곳(딱 두 줄):**
#   1) `modules/distress_box.py`의 `AIRCRAFT_RADIUS_M_DEFAULT` ← 실측 R (트랙 A 소관)
#   2) 이 파일의 `TERMINAL_COMMIT_DRIFT_MEASURED = False` → `True`
#      (δ=0.30은 그대로 둔다 — `compute_landing_window`가 그 δ를 다시 예산으로 잡아
#       `mat_edge_margin_m ≥ δ`를 재확립하므로 임계 숫자 자체는 바뀔 이유가 없다.
#       실제 여유를 그대로 쓰고 싶으면 그때 `TERMINAL_COMMIT_DRIFT_M_DEFAULT`를 올린다.)
#   ⚠️ R_실측 > 0.6444m면 안전창 자체가 비어(`aircraft_radius_max_feasible_m`) 착륙점이 안 나온다 —
#      그건 이 임계의 문제가 아니라 "박스 옆" 규약 자체의 문제다.
TERMINAL_COMMIT_DRIFT_MEASURED = False


def half_hfov_tan_from_calibration(camera_matrix, image_size) -> float:
    """`(width/2) / fx` — 카메라 인트린식에서 정규화오차→지상거리 환산 계수를 유도한다.

    `core/`는 yaml을 못 읽으므로(import 규칙) 배열/튜플만 받는다 —
    `utils/calibration_loader.py::CameraCalibration`의 `camera_matrix`/`image_size`를 그대로 넘기면 된다.
    """
    fx = float(camera_matrix[0][0])
    if fx <= 0:
        raise ValueError(f"fx가 양수가 아니다: {fx}")
    return (float(image_size[0]) / 2.0) / fx


class LandingState(str, Enum):
    """`str` 서브클래싱 — JSONL(`blackbox.log_frame`의 `state` 필드)에 그대로 문자열로 실린다."""

    ACQUIRE = "ACQUIRE"
    CENTER_DESCEND = "CENTER_DESCEND"
    LOCK = "LOCK"
    PRECISION_SERVO = "PRECISION_SERVO"
    TERMINAL = "TERMINAL"
    HOLD = "HOLD"
    ABORT_ASCEND = "ABORT_ASCEND"


@dataclass
class LandingSMConfig:
    """모든 임계값을 이름 있는 필드로 뺀다 — 매직넘버 금지(§7.3).

    최소 요구 6개(§9 6번 세션 지시) + `terminal_agl_m`(TERMINAL 진입 판단에 반드시 필요해
    추가 — "근접 하강 시 마커가 화각을 이탈"하는 그 근접 여부를 판단할 기준이 없으면 TERMINAL에
    아예 진입할 조건을 못 만든다. 과설계가 아니라 §5.1 자체가 요구하는 최소 조건).
    """

    # TERMINAL 데드레코닝 최대 블라인드 지속시간(초).
    # ⚠️ **초과 시 동작이 2026-07-28에 뒤집혔다** — 예전엔 재상승(`ABORT_ASCEND`),
    # 지금은 **최종 커밋**(`command="land"`, AUTO.LAND 인계). 값의 뜻은 그대로다:
    # "마지막 유효 추정으로 데드레코닝을 계속해도 되는 시간".
    max_blind_duration_s: float = 2.0
    # 블라인드 하강 중 예상 착륙점 이탈 임계(미터 — §5.1 "예상 착륙점 이탈 임계").
    #
    # ⚠️ **2026-07-28에 1.0 -> 0.75로 함께 내렸다.** 이유는 값 튜닝이 아니라 **단위 정정**이다:
    # 예전 `drift_estimate = |정규화오차| × AGL`은 미터가 아니었고(차원 불일치), 그래서 1.0이라는
    # "미터" 임계와 비교하는 것 자체가 어긋나 있었다. `× tan(HFOV/2)` 항을 채워 진짜 미터가 된
    # 지금, 같은 임계 1.0을 그대로 두면 게이트가 **1/0.767 ≈ 1.30배 헐거워진다**(정확해지는 변경이
    # 안전 게이트를 조용히 푸는 것 — 금지). 0.75는 nominal 계수 0.7673에 맞춰 **기존 동작점을
    # 그대로(오히려 2.3% 더 빡빡하게) 재현**하는 값이다:
    #     옛 발동조건: 오차×AGL > 1.0        새 발동조건: 오차×AGL > 0.75/0.7673 = 0.977
    # 🔀 **이 값은 여전히 실기체 미검증 잠정값이다.** 30cm 정밀착륙 요구보다 2.5배 크지만, 이건
    # 정밀도 게이트가 아니라 "마지막으로 본 오차가 데드레코닝을 믿을 수 없을 만큼 컸는가"를 보는
    # **안전 폴백 게이트**라 요구정밀도까지 조이면 상시 ABORT가 난다. 실기체 데이터 확보 후 재튜닝 대상.
    #
    # ⚠️ **[2026-07-28] 기본 경로에서는 더 이상 소비되지 않는다** — 죽은 값이 아니라
    # `allow_terminal_abort_ascend=True`(옛 동작 복원)일 때만 쓰이는 **레거시 게이트**다.
    # 지우면 되돌리기 스위치가 옛 동작을 재현하지 못한다(등가성 회귀테스트가 이 사실을 고정한다).
    max_drift_estimate_m: float = 0.75
    # 🔴 **밀어붙이기(최종 커밋) 임계 — `max_drift_estimate_m`과 의도적으로 분리된 별개 값.**
    # 유도·실측 상태·내일 고칠 한 줄은 위 `TERMINAL_COMMIT_DRIFT_M_DEFAULT` 주석 참조.
    max_terminal_commit_drift_m: float = TERMINAL_COMMIT_DRIFT_M_DEFAULT
    # 위 임계에 실측 근거가 있는지(= 기체 반경 R이 측정됐는지). `Decision`까지 에코된다.
    terminal_commit_drift_measured: bool = TERMINAL_COMMIT_DRIFT_MEASURED
    # 🔴 되돌리기 한 줄 — `True`면 TERMINAL 폴백이 2026-07-28 이전 동작(`ABORT_ASCEND`)으로
    # **정확히** 복원된다(상태·명령·사유 전부. 등가성 자체가 회귀테스트 대상이다).
    # 기본 `False`인 이유는 §5.1 갱신 서술 참조 — 대회 성공 판정이 정성이라 재상승은 그 자체로
    # 실패 신호다. 삭제하지 않고 남겨둔 두 번째 이유: FC/운영자가 명시적으로 요구할 수 있는
    # 탈출 경로다(§8 페일세이프 계약의 옛 문구가 가리키던 그 동작).
    allow_terminal_abort_ascend: bool = False
    # 정규화 중심오차 -> 지상거리(m) 환산 계수 = tan(HFOV/2). 위 NOMINAL_HALF_HFOV_TAN 참조.
    # main.py/replay.py가 로드한 캘리브레이션에서 계산해 덮어쓴다.
    half_hfov_tan: float = NOMINAL_HALF_HFOV_TAN
    # LOCK 확정(커밋 게이트 통과)에 필요한 연속 fine_locked(비모호) 프레임 수.
    lock_confirm_frames: int = 3
    # 검출 상실 허용 연속 프레임 수 — 이 이하는 일시적 흔들림으로 보고 넘어간다.
    loss_tolerance_frames: int = 5
    # 중심정렬 허용오차 — 단위 없음(정규화, dx=(cx-w/2)/(w/2), dy 동일 → 노름 0~약1.41).
    # ⚠️ 이름에 _norm이 붙어있지 픽셀이 아니다 — main.py/replay.py의 _build_observation()이
    # 화면 절반폭/절반높이로 나눠 정규화한 값을 그대로 넣는다(픽셀값을 넣으면 즉시 오작동).
    center_tolerance_norm: float = 0.05
    # 이 개수를 넘는 동시 후보는 "모호"로 취급 — 커밋 게이트가 락을 거절한다(§5.1).
    max_candidates_for_lock: int = 1
    # 이 고도(AGL, m) 이하에서만 PRECISION_SERVO -> TERMINAL 진입(§5.1 "근접 하강").
    terminal_agl_m: float = 3.0


@dataclass
class Observation:
    """프레임 단위 관측 — 타겟 종류 무관 최소 필드만(과설계 금지, §9 6번 세션 지시)."""

    ts: float                                   # 초
    frame_id: int
    n_candidates: int = 0                       # 이번 프레임 후보 개수(coarse/fine 통틀어)
    # 화면중심 대비 정규화 오차(없으면 None). ⚠️ 이름에 _norm이 붙어있지 픽셀이 아니다 —
    # dx=(cx-w/2)/(w/2), dy=(cy-h/2)/(h/2)의 노름(0~약1.41 근사, main.py/replay.py 참조).
    # 픽셀 단위 값을 그대로 넣으면 center_tolerance_norm(0.05 기본)과 비교가 즉시 어긋난다.
    center_error_norm: Optional[float] = None
    fine_locked: bool = False                   # fine 피처 확정·검증(ArUco ID 등)
    agl_m: Optional[float] = None                # 라이다 AGL(없으면 None)
    target_estimate: Optional[Any] = None       # 있으면 TargetEstimate류(dict/객체), 없으면 None
    scale_source: Optional[str] = None          # "agl" | "known_size" | None — §5.1 blob 스케일 융합


@dataclass
class Decision:
    """상태머신 출력 — `command`는 소비자(FC 세션)를 위한 문자열 힌트일 뿐, 여기선 뱉기만 한다."""

    state: LandingState
    # "scan"/"center"/"descend"/"hold"/"ascend"/"land".
    # 🔴 `"land"`는 2026-07-28 추가 — **"비전 유도를 끝내고 지금 위치에 내려앉아라"**(AUTO.LAND
    # 인계). `"descend"`(마지막 유효 추정으로 횡방향을 계속 물고 늘어지며 하강)와 다르다.
    command: str
    reason: str                 # 전이/거절 사유(§7.4 거절이유 로깅 철학과 일치)
    blind_duration_s: float = 0.0     # TERMINAL 블라인드 경과시간(그 외 상태는 0.0)
    scale_source: Optional[str] = None  # 관측 그대로 에코 — 어느 소스를 썼는지 기록(§5.1)
    # 밀어붙이기 임계에 실측 근거가 있는지 — `LandingSMConfig.terminal_commit_drift_measured`를
    # 그대로 에코한다(`core/frames.py`의 `mount_yaw_measured`가 와이어까지 전파되는 것과 같은
    # 패턴). 지금은 항상 False다: 기체 반경 R이 미측정이라 매트 여유 보장이 조건부다.
    commit_drift_measured: bool = TERMINAL_COMMIT_DRIFT_MEASURED


class LandingStateMachine:
    """§5.1 공통 상태머신. `update()`를 프레임마다 호출 — 내부에 진행상황(연속 프레임 카운터,
    마지막 유효 관측)을 들고 있는 게 유일한 상태다(wall-clock/난수 없음, §7.5 결정론)."""

    def __init__(self, config: Optional[LandingSMConfig] = None):
        self.config = config or LandingSMConfig()
        self._state: LandingState = LandingState.ACQUIRE
        self._consecutive_lost = 0
        self._consecutive_fine_locked = 0
        self._blind_since_ts: Optional[float] = None
        self._last_center_error_norm: Optional[float] = None
        self._last_agl_m: Optional[float] = None

    @property
    def state(self) -> LandingState:
        return self._state

    def update(self, obs: Observation) -> Decision:
        cfg = self.config
        current = self._state

        has_candidate = obs.n_candidates >= 1
        ambiguous = obs.n_candidates > cfg.max_candidates_for_lock
        near_ground = obs.agl_m is not None and obs.agl_m <= cfg.terminal_agl_m
        centered = (
            obs.center_error_norm is not None
            and abs(obs.center_error_norm) <= cfg.center_tolerance_norm
        )

        # 최근 유효 관측 갱신 — TERMINAL 블라인드 데드레코닝/이탈 추정에만 쓴다.
        if has_candidate:
            self._consecutive_lost = 0
            if obs.center_error_norm is not None:
                self._last_center_error_norm = obs.center_error_norm
            if obs.agl_m is not None:
                self._last_agl_m = obs.agl_m
        else:
            self._consecutive_lost += 1

        if obs.fine_locked and not ambiguous:
            self._consecutive_fine_locked += 1
        else:
            self._consecutive_fine_locked = 0

        next_state = current
        command = "hold"
        reason = "unhandled"

        if current is LandingState.ACQUIRE:
            if has_candidate:
                next_state, reason, command = LandingState.CENTER_DESCEND, "coarse_candidate_found", "center"
            else:
                command, reason = "scan", "no_candidate"

        elif current is LandingState.CENTER_DESCEND:
            if not has_candidate:
                command, reason = "scan", "candidate_lost"
                if self._consecutive_lost > cfg.loss_tolerance_frames:
                    next_state, reason = LandingState.HOLD, "candidate_loss_exceeds_tolerance"
            elif obs.fine_locked and ambiguous:
                # 커밋 게이트: 락을 시도했으나(fine_locked) 후보가 모호 -> 거절(§5.1).
                next_state, reason, command = LandingState.HOLD, "lock_rejected_ambiguous_candidates", "hold"
            elif obs.fine_locked:
                next_state, reason, command = LandingState.LOCK, "fine_lock_signal_first_seen", "hold"
            else:
                command = "descend" if centered else "center"
                reason = "centered_descending" if centered else "centering"

        elif current is LandingState.LOCK:
            if obs.fine_locked and ambiguous:
                next_state, reason = LandingState.HOLD, "lock_rejected_ambiguous_candidates"
            elif not has_candidate or not obs.fine_locked:
                next_state, reason, command = LandingState.CENTER_DESCEND, "fine_lock_dropped_before_confirm", "center"
            elif self._consecutive_fine_locked >= cfg.lock_confirm_frames:
                next_state, reason, command = LandingState.PRECISION_SERVO, "lock_confirmed", "descend"
            else:
                command, reason = "hold", "lock_confirming"

        elif current is LandingState.PRECISION_SERVO:
            if near_ground:
                next_state, reason, command = LandingState.TERMINAL, "near_ground_enter_terminal", "descend"
                self._blind_since_ts = None
            elif obs.fine_locked and ambiguous:
                next_state, reason = LandingState.HOLD, "ambiguous_candidates_during_servo"
            elif not has_candidate or not obs.fine_locked:
                if self._consecutive_lost > cfg.loss_tolerance_frames:
                    next_state, reason = LandingState.HOLD, "lock_lost_exceeds_tolerance"
                else:
                    command, reason = "hold", "transient_lock_loss_within_tolerance"
            else:
                command = "descend" if centered else "center"
                reason = "servoing"

        elif current is LandingState.TERMINAL:
            if has_candidate and obs.fine_locked:
                self._blind_since_ts = None
                command, reason = "descend", "terminal_visual_descend"
            else:
                if self._blind_since_ts is None:
                    self._blind_since_ts = obs.ts
                blind_duration = max(0.0, obs.ts - self._blind_since_ts)
                # 이탈 추정(미터) — 마지막 유효 정규화 중심오차 × 마지막 유효 AGL × tan(HFOV/2)
                # (§5.1 "예상 착륙점 이탈 임계").
                #
                # **[2026-07-28] `tan(HFOV/2)` 항을 채웠다.** 예전 식은 이 항이 빠져 있어
                # "정규화 오차 × 미터"라는 차원 불일치 양이었고(미터가 아니었다), 그래서 미터
                # 임계값과 비교하는 것 자체가 어긋나 있었다. 핀홀 기하에서 화면 절반폭이
                # `fx·tan(HFOV/2)`이므로 정규화오차 e에 대응하는 광선각은 `tan(θ) = e·tan(HFOV/2)`,
                # 나디르 카메라의 지상거리는 `AGL·tan(θ)` — 근사가 아니라 **핀홀에서는 정확**하다.
                #
                # 다만 `center_error_norm`은 x(폭 정규화)와 y(높이 정규화)를 **섞은 노름**이라,
                # y 성분에는 원래 `tan(VFOV/2)`(< tan(HFOV/2), 가로가 긴 프레임)가 붙어야 한다.
                # 둘 다 `tan(HFOV/2)`로 환산하는 이 식은 그래서 **참값의 상한**이다 — 안전 게이트가
                # 원하는 방향(과소평가 없음)이라 의도적으로 이 근사를 고른다. 정확히 하려면
                # 상태머신이 dx/dy를 따로 받아야 하는데, 그건 "타겟 무관 최소 관측" 원칙을 깬다.
                drift_estimate = 0.0
                if self._last_center_error_norm is not None and self._last_agl_m is not None:
                    drift_estimate = (
                        abs(self._last_center_error_norm) * self._last_agl_m * cfg.half_hfov_tan
                    )
                if cfg.allow_terminal_abort_ascend:
                    # 🔴 되돌리기 경로 — 2026-07-28 이전 동작을 **한 글자도 바꾸지 않고** 복원한다.
                    # `command`가 여기서 대입되지 않아 초기값 "hold"로 남는 것까지 옛 동작이다
                    # (등가성 회귀테스트가 이 세부까지 고정한다 — 고치지 마라).
                    if blind_duration > cfg.max_blind_duration_s:
                        next_state, reason = LandingState.ABORT_ASCEND, "blind_duration_exceeded"
                    elif drift_estimate > cfg.max_drift_estimate_m:
                        next_state, reason = LandingState.ABORT_ASCEND, "drift_estimate_exceeded"
                    else:
                        command, reason = "descend", "terminal_blind_deadreckoning"
                elif drift_estimate > cfg.max_terminal_commit_drift_m:
                    # 🔴 **기하 게이트가 타이머를 이긴다 — 순서를 뒤집지 마라.**
                    # 마지막으로 본 이탈이 착륙 기하의 드리프트 예산(δ)을 넘었다. 여기서
                    # 밀어붙이면 0.105m 라이즈드 매트 **밖/가장자리**에 걸쳐 전복된다 —
                    # 표면적 실패가 아니라 진짜 사고다. 블라인드 시간 초과를 먼저 보면
                    # 그 사고 경로로 조용히 커밋된다.
                    #
                    # 재상승이 아니라 `HOLD`다: 3m 상공 제자리 정지는 "신중하게 보는 것"으로
                    # 보이지 실패로 보이지 않고, 재포착 시 `CENTER_DESCEND`로 복귀하는 살아있는
                    # 상태다(막다른 상태 아님). **상태를 TERMINAL로 두고 command만 "hold"로
                    # 두는 안은 기각했다** — 이 저장소는 `state`를 유일한 거부권 채널로
                    # 못박았고(`core/wire.py` "command는 명령이 아니다",
                    # `ros/shim_core.py`의 veto 집합 `{HOLD, ABORT_ASCEND}`), state만 읽는
                    # 소비자에게는 거부가 통째로 안 보여 바로 그 사고 경로가 열린다.
                    #
                    # ⚠️ **이 HOLD의 종결 시한은 vision이 정하지 않는다.** `core/wire.py`
                    # `FailsafeContract` docstring이 `hold_before_reascend_s`를 "FC가 자기
                    # 제어틱에서 재는 값, vision이 정해 보낼 값이 아니다"로 이미 확정했다.
                    # vision은 사유를 실어 보내고, 종결(AUTO.LAND 인계)은 FC가 한다.
                    next_state, reason = LandingState.HOLD, "terminal_hold_drift_exceeds_commit"
                elif blind_duration > cfg.max_blind_duration_s:
                    # 데드레코닝을 믿을 수 있는 시간이 다 됐다. **재상승하지 않고 커밋한다**
                    # (2026-07-28 기조 전환). 드리프트가 예산 안이라는 것은 바로 위 분기에서
                    # 이미 확인됐으므로 지금 내려앉으면 매트 위다.
                    #
                    # `"descend"`가 아니라 `"land"`인 이유: 2초째 못 보고 있는 추정으로 횡방향을
                    # 계속 물고 늘어지는 것이 오차를 **키우는** 모드다(과도응답·접지 순간 횡속도 →
                    # 라이즈드 가장자리에서 전복). 횡 유도를 놓고 수직으로 내려앉는 편이 안전하고,
                    # 이건 `core/wire.py::closed_loop_floor_agl_m()`이 이미 규정한 바로 그
                    # 인계다(`not_for_closed_loop_30cm=True` → "3.0m까지 정렬 후 AUTO.LAND 인계",
                    # 그리고 그 3.0m은 `terminal_agl_m` = TERMINAL 진입 고도와 같은 숫자다).
                    # 즉 이 분기는 기존 계약과 모순되기는커녕 **계약을 상태머신 쪽에서 명시적으로
                    # 발화**하는 것이다.
                    #
                    # 래치하지 않는다(다음 프레임에 다시 보이면 `terminal_visual_descend`로
                    # 돌아간다) — 인계 이후 모드를 되찾을지는 FC 소관이고, 상태머신이 그 결정을
                    # 대신 내리는 것은 스스로 부인한 권한을 행사하는 것이다.
                    command, reason = "land", "terminal_commit_handoff_blind_timeout"
                else:
                    command, reason = "descend", "terminal_blind_deadreckoning"

        elif current is LandingState.HOLD:
            if has_candidate and not ambiguous:
                next_state, reason, command = LandingState.CENTER_DESCEND, "recovered_from_hold", "center"
                self._consecutive_fine_locked = 0
            else:
                command, reason = "hold", "holding"

        elif current is LandingState.ABORT_ASCEND:
            command = "ascend"
            if has_candidate and not ambiguous:
                next_state, reason = LandingState.CENTER_DESCEND, "recovered_after_ascend"
                self._consecutive_fine_locked = 0
            else:
                reason = "ascending"

        blind_duration_out = 0.0
        if next_state is LandingState.TERMINAL and self._blind_since_ts is not None:
            blind_duration_out = max(0.0, obs.ts - self._blind_since_ts)

        self._state = next_state
        return Decision(
            state=next_state,
            command=command,
            reason=reason,
            blind_duration_s=blind_duration_out,
            scale_source=obs.scale_source,
            commit_drift_measured=cfg.terminal_commit_drift_measured,
        )
