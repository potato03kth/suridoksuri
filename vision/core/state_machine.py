"""
공통 상태머신 + 안전 폴백 (`docs/vision_plan.md` §5.1, §9 빌드순서 6번).

```
ACQUIRE(대형 coarse 피처) → CENTER_DESCEND(중심 맞추며 하강)
  → LOCK(fine 피처 확정·검증) → PRECISION_SERVO(폐루프) → TERMINAL(AGL/블라인드 하강)
```
검출 상실·후보 모호·TERMINAL 블라인드 지속시간 초과는 전부 `HOLD`/`ABORT_ASCEND`로 빠진다
(§5.1 "안 보이는데 계속 내려간다"를 금지하는 게 핵심 / §8 "추측 후 커밋 금지").

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

    # TERMINAL 데드레코닝 최대 블라인드 지속시간(초) — 초과 시 재상승(§5.1 "핵심").
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
    max_drift_estimate_m: float = 0.75
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
    command: str                # "scan"/"center"/"descend"/"hold"/"ascend"
    reason: str                 # 전이/거절 사유(§7.4 거절이유 로깅 철학과 일치)
    blind_duration_s: float = 0.0     # TERMINAL 블라인드 경과시간(그 외 상태는 0.0)
    scale_source: Optional[str] = None  # 관측 그대로 에코 — 어느 소스를 썼는지 기록(§5.1)


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
                if blind_duration > cfg.max_blind_duration_s:
                    next_state, reason = LandingState.ABORT_ASCEND, "blind_duration_exceeded"
                elif drift_estimate > cfg.max_drift_estimate_m:
                    next_state, reason = LandingState.ABORT_ASCEND, "drift_estimate_exceeded"
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
        )
