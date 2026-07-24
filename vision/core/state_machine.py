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
    # 블라인드 하강 중 예상 착륙점 이탈 임계(미터, 근사 추정치 — §5.1 "예상 착륙점 이탈 임계").
    max_drift_estimate_m: float = 1.0
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
                # 근사 이탈 추정 — 마지막 유효 정규화 중심오차 x 마지막 유효 AGL(§5.1 "예상
                # 착륙점 이탈 임계"). 정밀 기하가 아니라 안전 폴백 게이트용 근사치임을 의도적으로
                # 유지한다(세션 지시 "정밀도·물리 튜닝에 시간을 쓰지 말 것").
                # ⚠️ 단위 주의: 이 곱은 미터가 아니다 — "정규화 오차 × 미터"는 차원상 미터가
                # 아니다. 실제 지상거리 근사는 `정규화오차 × agl × tan(HFOV/2)`이고, 실측
                # HFOV 75°면 tan(37.5°)≈0.767이라 이 tan 항을 생략한 현재 근사값은 실제
                # 지상 이탈거리보다 약 1/0.767 ≈ 1.3배 크게(=보수적으로) 나온다. 안전 방향
                # (더 쉽게 ABORT_ASCEND로 빠짐)이라 급하지 않음 — 계산식은 바꾸지 않는다.
                # 임계값(`max_drift_estimate_m`) 재튜닝은 실기체 데이터 확보 후에 할 일.
                drift_estimate = 0.0
                if self._last_center_error_norm is not None and self._last_agl_m is not None:
                    drift_estimate = abs(self._last_center_error_norm) * self._last_agl_m
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
