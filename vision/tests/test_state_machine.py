"""`vision/core/state_machine.py` 단위테스트 (`docs/vision_plan.md` §5.1, §9 빌드순서 6번).

순수 로직이라 하드웨어/실측 데이터 불필요 — `Observation` 시퀀스를 직접 만들어 실제
`LandingStateMachine.update()`를 반복 호출하고, 실제로 반환되는 `Decision.state` 시퀀스를
검증한다(몽키패치 없음).
"""
from __future__ import annotations

import math

import pytest

from vision.core.state_machine import (
    Decision,
    LandingSMConfig,
    LandingState,
    LandingStateMachine,
    NOMINAL_HALF_HFOV_TAN,
    Observation,
    half_hfov_tan_from_calibration,
)


def _obs(
    ts: float,
    frame_id: int,
    n_candidates: int = 0,
    center_error_norm=None,
    fine_locked: bool = False,
    agl_m=None,
    scale_source=None,
) -> Observation:
    return Observation(
        ts=ts,
        frame_id=frame_id,
        n_candidates=n_candidates,
        center_error_norm=center_error_norm,
        fine_locked=fine_locked,
        agl_m=agl_m,
        scale_source=scale_source,
    )


# ===========================================================================
# 1. 정상 시퀀스 전이 — ACQUIRE→CENTER_DESCEND→LOCK→PRECISION_SERVO→TERMINAL
# ===========================================================================


def _normal_sequence_observations():
    """§5.1 순서를 그대로 밟는 관측열 — 기본 config(lock_confirm_frames=3,
    terminal_agl_m=3.0) 그대로 트리거되도록 값을 골랐다."""
    return [
        _obs(0.0, 0, n_candidates=0),                                              # ACQUIRE 유지
        _obs(0.1, 1, n_candidates=1, center_error_norm=0.5, fine_locked=False),      # coarse 발견
        _obs(0.2, 2, n_candidates=1, center_error_norm=0.5, fine_locked=False),      # 센터링 중
        _obs(0.3, 3, n_candidates=1, center_error_norm=0.02, fine_locked=True),      # fine lock 신호 최초
        _obs(0.4, 4, n_candidates=1, center_error_norm=0.02, fine_locked=True),      # 확정 카운트 2
        _obs(0.5, 5, n_candidates=1, center_error_norm=0.02, fine_locked=True),      # 확정 카운트 3 → 커밋
        _obs(0.6, 6, n_candidates=1, center_error_norm=0.01, fine_locked=True),      # servo 유지
        _obs(0.7, 7, n_candidates=1, center_error_norm=0.01, fine_locked=True, agl_m=2.0),  # 근접 → TERMINAL
    ]


def test_normal_sequence_transitions_through_all_five_states():
    sm = LandingStateMachine()
    observations = _normal_sequence_observations()
    states = [sm.update(o).state for o in observations]

    assert states == [
        LandingState.ACQUIRE,
        LandingState.CENTER_DESCEND,
        LandingState.CENTER_DESCEND,
        LandingState.LOCK,
        LandingState.LOCK,
        LandingState.PRECISION_SERVO,
        LandingState.PRECISION_SERVO,
        LandingState.TERMINAL,
    ]
    # 순서 자체도 재확인(§5.1 문구 그대로) — 각 상태가 처음 등장하는 인덱스가 단조증가.
    order = [LandingState.ACQUIRE, LandingState.CENTER_DESCEND, LandingState.LOCK,
             LandingState.PRECISION_SERVO, LandingState.TERMINAL]
    first_seen = [states.index(s) for s in order]
    assert first_seen == sorted(first_seen)


def test_command_hints_follow_state_semantics():
    """command는 소비자(FC)를 위한 힌트 문자열 — 상태별로 말이 되는 값이 나오는지."""
    sm = LandingStateMachine()
    decisions = [sm.update(o) for o in _normal_sequence_observations()]

    assert decisions[0].command == "scan"  # ACQUIRE, 후보 없음
    assert decisions[5].command == "descend"  # 방금 커밋(lock_confirmed)
    assert decisions[7].command == "descend"  # TERMINAL 진입 프레임


# ===========================================================================
# 2. 안전 폴백 — 검출 상실 → 허용 프레임 초과 → HOLD
# ===========================================================================


def _advance_to_precision_servo(sm: LandingStateMachine) -> float:
    """정상 시퀀스로 PRECISION_SERVO까지 진행시키고 마지막 ts를 반환(헬퍼, 테스트 전용)."""
    last_ts = 0.0
    for o in _normal_sequence_observations()[:-1]:  # TERMINAL 직전(agl_m 없는 servo 프레임)까지
        d = sm.update(o)
        last_ts = o.ts
    assert d.state == LandingState.PRECISION_SERVO
    return last_ts


def test_detection_loss_beyond_tolerance_falls_back_to_hold():
    """§9 6번 핵심 안전 폴백: 검출이 사라진 채 loss_tolerance_frames를 넘기면 반드시 HOLD로
    빠진다 — "안 보이는데 계속 서보한다"를 허용하면 안 된다."""
    cfg = LandingSMConfig(loss_tolerance_frames=5)
    sm = LandingStateMachine(cfg)
    last_ts = _advance_to_precision_servo(sm)
    assert sm.state == LandingState.PRECISION_SERVO

    states = []
    for i in range(cfg.loss_tolerance_frames + 1):
        last_ts += 0.1
        d = sm.update(_obs(last_ts, 100 + i, n_candidates=0))
        states.append(d.state)

    # 허용치 이내(처음 loss_tolerance_frames 프레임)까지는 아직 서보를 유지해야 한다
    # (일시적 흔들림까지 즉시 홀드로 빠지면 매끄러운 흐름이 깨진다).
    assert all(s == LandingState.PRECISION_SERVO for s in states[:-1])
    # 허용치를 넘긴 마지막 프레임에서 반드시 HOLD.
    assert states[-1] == LandingState.HOLD


def test_hold_recovers_when_candidate_reacquired():
    """HOLD가 막다른 상태가 아니라 재포착 시 CENTER_DESCEND로 복귀해야 전체 흐름이 유기적이다."""
    cfg = LandingSMConfig(loss_tolerance_frames=2)
    sm = LandingStateMachine(cfg)
    _advance_to_precision_servo(sm)
    ts = 100.0
    for i in range(cfg.loss_tolerance_frames + 1):
        ts += 0.1
        d = sm.update(_obs(ts, 200 + i, n_candidates=0))
    assert d.state == LandingState.HOLD

    d = sm.update(_obs(ts + 0.1, 210, n_candidates=1, center_error_norm=0.3, fine_locked=False))
    assert d.state == LandingState.CENTER_DESCEND


# ===========================================================================
# 3. TERMINAL 블라인드 지속시간 초과 → 재상승(ABORT_ASCEND) — §5.1이 "핵심"이라 명시
# ===========================================================================


def test_terminal_blind_duration_exceeded_triggers_abort_ascend():
    cfg = LandingSMConfig(max_blind_duration_s=2.0)
    sm = LandingStateMachine(cfg)
    for o in _normal_sequence_observations():
        d = sm.update(o)
    assert d.state == LandingState.TERMINAL  # 사전조건: TERMINAL 진입 확인

    last_ts = _normal_sequence_observations()[-1].ts
    ts_steps = [0.5, 0.7, 0.7, 0.7]  # 누적 0.5/1.2/1.9/2.6 → 3번째까지 2.0초 미만, 4번째에 초과
    states = []
    for i, dt in enumerate(ts_steps):
        last_ts += dt
        d = sm.update(_obs(last_ts, 300 + i, n_candidates=0))
        states.append(d.state)

    assert states[:-1] == [LandingState.TERMINAL, LandingState.TERMINAL, LandingState.TERMINAL]
    assert states[-1] == LandingState.ABORT_ASCEND
    assert d.reason == "blind_duration_exceeded"


def test_terminal_visual_reacquisition_resets_blind_timer():
    """블라인드 도중 다시 보이면 타이머가 리셋돼야 한다 — 그렇지 않으면 흔들리는 검출이
    누적 시간으로 억울하게 재상승을 유발한다."""
    cfg = LandingSMConfig(max_blind_duration_s=1.0)
    sm = LandingStateMachine(cfg)
    for o in _normal_sequence_observations():
        d = sm.update(o)
    assert d.state == LandingState.TERMINAL
    last_ts = _normal_sequence_observations()[-1].ts

    last_ts += 0.8
    d = sm.update(_obs(last_ts, 400, n_candidates=0))
    assert d.state == LandingState.TERMINAL  # 아직 1.0초 안 됨

    last_ts += 0.1
    d = sm.update(_obs(last_ts, 401, n_candidates=1, center_error_norm=0.01, fine_locked=True))
    assert d.state == LandingState.TERMINAL  # 재포착 — 타이머 리셋

    last_ts += 0.8
    d = sm.update(_obs(last_ts, 402, n_candidates=0))
    assert d.state == LandingState.TERMINAL  # 리셋 안 됐으면 누적 1.6초로 여기서 이미 초과했을 것


def test_terminal_drift_estimate_exceeded_triggers_abort_ascend():
    """§5.1 "예상 착륙점 이탈 임계" — 블라인드 지속시간이 아직 안 찼어도 마지막 유효
    중심오차x고도로 추정한 이탈이 임계를 넘으면 즉시 재상승해야 한다."""
    cfg = LandingSMConfig(max_blind_duration_s=100.0, max_drift_estimate_m=0.5)
    sm = LandingStateMachine(cfg)
    # 마지막 유효 관측이 중심오차 크고 고도 낮은 상태로 TERMINAL 진입하도록 직접 구성.
    sm.update(_obs(0.0, 0, n_candidates=1, center_error_norm=0.5, fine_locked=True))
    for _ in range(3):
        sm.update(_obs(0.1, 1, n_candidates=1, center_error_norm=0.5, fine_locked=True))
    d = sm.update(_obs(0.4, 4, n_candidates=1, center_error_norm=0.5, fine_locked=True, agl_m=2.0))
    assert d.state == LandingState.TERMINAL

    # drift_estimate = 0.5 * 2.0 * tan(HFOV/2)=0.7673 ≈ 0.767 m > 0.5 임계
    d = sm.update(_obs(0.5, 5, n_candidates=0))
    assert d.state == LandingState.ABORT_ASCEND
    assert d.reason == "drift_estimate_exceeded"


# ===========================================================================
# 4. 커밋 게이트 — 후보 2개(모호) → 락 거절(HOLD), LOCK/PRECISION_SERVO/TERMINAL 도달 불가
# ===========================================================================


def test_ambiguous_candidates_rejects_lock_and_falls_to_hold():
    sm = LandingStateMachine()
    states = [
        sm.update(_obs(0.0, 0, n_candidates=0)).state,
        sm.update(_obs(0.1, 1, n_candidates=2, fine_locked=True)).state,   # coarse 발견(모호 여부 무관)
        sm.update(_obs(0.2, 2, n_candidates=2, fine_locked=True)).state,   # 락 시도 → 모호 → 거절
    ]
    assert states == [LandingState.ACQUIRE, LandingState.CENTER_DESCEND, LandingState.HOLD]
    assert LandingState.LOCK not in states
    assert LandingState.PRECISION_SERVO not in states
    assert LandingState.TERMINAL not in states


def test_ambiguous_candidates_during_lock_confirmation_also_rejects():
    """LOCK 확정 카운트 도중에 후보가 모호해져도 커밋 게이트가 막아야 한다."""
    sm = LandingStateMachine(LandingSMConfig(lock_confirm_frames=3))
    sm.update(_obs(0.0, 0, n_candidates=1))
    d = sm.update(_obs(0.1, 1, n_candidates=1, fine_locked=True))
    assert d.state == LandingState.LOCK
    d = sm.update(_obs(0.2, 2, n_candidates=2, fine_locked=True))  # 갑자기 후보 2개로 모호해짐
    assert d.state == LandingState.HOLD
    assert d.reason == "lock_rejected_ambiguous_candidates"


# ===========================================================================
# 5. 불변식 — fine_locked=False인 채로는 TERMINAL/PRECISION_SERVO/LOCK에 도달 불가
#    (agl_m이 아무리 낮아도 커밋 게이트를 우회할 수 없다)
# ===========================================================================


def test_never_reaches_lock_states_without_fine_locked_even_with_low_agl():
    sm = LandingStateMachine()
    states = []
    for i in range(60):
        # 후보는 있다 갔다 하지만 fine_locked은 한 번도 True가 아니다. agl_m은 항상 매우 낮다
        # (근접 하강 상황을 흉내) — 그럼에도 커밋 게이트를 절대 통과할 수 없어야 한다.
        n = 1 if i % 3 != 0 else 0
        d = sm.update(_obs(float(i) * 0.1, i, n_candidates=n, center_error_norm=0.02,
                            fine_locked=False, agl_m=0.2))
        states.append(d.state)

    assert LandingState.LOCK not in states
    assert LandingState.PRECISION_SERVO not in states
    assert LandingState.TERMINAL not in states
    assert set(states) <= {LandingState.ACQUIRE, LandingState.CENTER_DESCEND, LandingState.HOLD}


# ===========================================================================
# 6. AGL 전부 None — 크래시 없이 동작(실기체 telemetry 아직 없음, 필수 요구사항)
# ===========================================================================


def test_runs_without_crashing_when_agl_always_none():
    sm = LandingStateMachine()
    observations = [
        _obs(o.ts, o.frame_id, n_candidates=o.n_candidates, center_error_norm=o.center_error_norm,
             fine_locked=o.fine_locked, agl_m=None)
        for o in _normal_sequence_observations()
    ]
    states = [sm.update(o).state for o in observations]  # 크래시하면 안 됨

    for s in states:
        assert isinstance(s, LandingState)
    # agl_m이 전혀 없으면 TERMINAL(AGL 기반 진입 조건)에 도달할 근거가 없다 —
    # 폐루프 서보에 계속 머무는 것이 안전한 축퇴 동작이다(추측 후 커밋 금지).
    assert LandingState.TERMINAL not in states
    assert states[-1] == LandingState.PRECISION_SERVO


# ===========================================================================
# 7. 결정론(§7.5) — 같은 관측열 → 같은 상태열
# ===========================================================================


def test_deterministic_same_observations_same_states():
    observations = _normal_sequence_observations() + [
        _obs(0.8, 8, n_candidates=0),
        _obs(1.6, 9, n_candidates=0),
        _obs(2.4, 10, n_candidates=0),
        _obs(3.2, 11, n_candidates=0),
    ]

    sm1 = LandingStateMachine()
    sm2 = LandingStateMachine()
    decisions1 = [sm1.update(o) for o in observations]
    decisions2 = [sm2.update(o) for o in observations]

    assert [d.state for d in decisions1] == [d.state for d in decisions2]
    assert [d.command for d in decisions1] == [d.command for d in decisions2]
    assert [d.reason for d in decisions1] == [d.reason for d in decisions2]


# ===========================================================================
# 기타 — config 매직넘버 금지(§7.3): 필드로 override 가능해야 함
# ===========================================================================


def test_config_thresholds_are_named_fields_not_magic_numbers():
    cfg = LandingSMConfig(
        max_blind_duration_s=5.0,
        max_drift_estimate_m=2.0,
        lock_confirm_frames=1,
        loss_tolerance_frames=0,
        center_tolerance_norm=0.1,
        max_candidates_for_lock=2,
        terminal_agl_m=1.0,
    )
    sm = LandingStateMachine(cfg)
    assert sm.config.lock_confirm_frames == 1
    assert sm.config.terminal_agl_m == 1.0

    # lock_confirm_frames=1이면 fine_locked 신호 첫 프레임에 바로 LOCK 진입,
    # 그 다음 프레임에 곧장 PRECISION_SERVO로 커밋돼야 한다(카운트 즉시 충족).
    sm.update(_obs(0.0, 0, n_candidates=1))
    d = sm.update(_obs(0.1, 1, n_candidates=1, fine_locked=True))
    assert d.state == LandingState.LOCK
    d = sm.update(_obs(0.2, 2, n_candidates=1, fine_locked=True))
    assert d.state == LandingState.PRECISION_SERVO


def test_scale_source_is_echoed_into_decision():
    """§5.1 blob 스케일 융합 규칙 — 어느 소스를 썼는지 Decision에 그대로 실려야 관측 가능하다."""
    sm = LandingStateMachine()
    d = sm.update(_obs(0.0, 0, n_candidates=1, scale_source="agl"))
    assert d.scale_source == "agl"
    d = sm.update(_obs(0.1, 1, n_candidates=1, scale_source="known_size"))
    assert d.scale_source == "known_size"


# ===========================================================================
# drift_estimate 의 tan(HFOV/2) 항 (2026-07-28)
#
# 예전 식 `|정규화오차| × AGL` 은 tan 항이 빠져 차원상 미터가 아니었다(약 1.303배 보수적).
# 항을 채워 진짜 미터가 됐고, 그만큼 게이트가 헐거워지지 않도록 max_drift_estimate_m 을
# 1.0 -> 0.75 로 함께 내렸다. vision/CLAUDE.md "drift_estimate 의 tan(HFOV/2) 항" 절 참조.
# ===========================================================================


def _terminal_sm_with_last_obs(cfg, center_error_norm, agl_m):
    """마지막 유효 관측(오차·AGL)을 남긴 채 TERMINAL 에 진입한 상태머신을 만든다."""
    sm = LandingStateMachine(cfg)
    sm.update(_obs(0.0, 0, n_candidates=1, center_error_norm=center_error_norm, fine_locked=True))
    for _ in range(3):
        sm.update(_obs(0.1, 1, n_candidates=1, center_error_norm=center_error_norm, fine_locked=True))
    d = sm.update(
        _obs(0.4, 4, n_candidates=1, center_error_norm=center_error_norm,
             fine_locked=True, agl_m=agl_m)
    )
    assert d.state == LandingState.TERMINAL
    return sm


def test_drift_estimate_includes_tan_half_hfov_term():
    """tan 항이 실제로 곱해지는지 — 옛 식이면 발동하고 새 식이면 발동하지 않는 임계로 가른다.

    e=0.5, AGL=2.0 -> 옛 식 1.000 / 새 식 0.767. 임계를 그 사이(0.9)에 두면
    tan 항이 빠지는 순간 red 가 된다.
    """
    cfg = LandingSMConfig(max_blind_duration_s=100.0, max_drift_estimate_m=0.9)
    sm = _terminal_sm_with_last_obs(cfg, center_error_norm=0.5, agl_m=2.0)
    d = sm.update(_obs(0.5, 5, n_candidates=0))
    assert d.state == LandingState.TERMINAL, "tan(HFOV/2) 항이 빠져 게이트가 과발동했다"
    assert d.reason == "terminal_blind_deadreckoning"


def test_drift_estimate_actually_consumes_half_hfov_tan_config():
    """`half_hfov_tan` 이 하드코딩이 아니라 config 에서 읽히는지 — 같은 관측·같은 임계에서
    계수만 바꿔 결과가 뒤집혀야 한다."""
    obs_kwargs = dict(center_error_norm=0.5, agl_m=2.0)

    wide = LandingSMConfig(max_blind_duration_s=100.0, max_drift_estimate_m=0.9, half_hfov_tan=2.0)
    d = _terminal_sm_with_last_obs(wide, **obs_kwargs).update(_obs(0.5, 5, n_candidates=0))
    assert d.state == LandingState.ABORT_ASCEND  # 0.5*2.0*2.0 = 2.0 > 0.9

    narrow = LandingSMConfig(max_blind_duration_s=100.0, max_drift_estimate_m=0.9, half_hfov_tan=0.1)
    d = _terminal_sm_with_last_obs(narrow, **obs_kwargs).update(_obs(0.5, 5, n_candidates=0))
    assert d.state == LandingState.TERMINAL      # 0.5*2.0*0.1 = 0.1 < 0.9


def test_half_hfov_tan_from_calibration_derives_from_intrinsics():
    """`tan(HFOV/2) = (width/2)/fx` — 화각(도) 가정을 안 거치고 인트린식에서 곧바로 나온다."""
    # HFOV 90° 인 가상 카메라: fx = (w/2)/tan(45°) = w/2
    k = half_hfov_tan_from_calibration([[500.0, 0, 500.0], [0, 500.0, 250.0], [0, 0, 1.0]], (1000, 500))
    assert k == pytest.approx(math.tan(math.radians(45.0)))

    with pytest.raises(ValueError):
        half_hfov_tan_from_calibration([[0.0, 0, 0], [0, 0, 0], [0, 0, 1.0]], (1000, 500))


def test_nominal_default_matches_shipped_calibration_file():
    """모듈 상수가 실제 배포된 nominal.yaml 과 어긋나면 red — 둘이 조용히 갈라지는 걸 막는다."""
    from pathlib import Path

    from vision.utils.calibration_loader import load_camera_calibration

    calib_path = Path(__file__).resolve().parents[1] / "calibration" / "cam109-imx708af75" / "nominal.yaml"
    calib = load_camera_calibration(calib_path)
    derived = half_hfov_tan_from_calibration(calib.camera_matrix, calib.image_size)
    assert derived == pytest.approx(NOMINAL_HALF_HFOV_TAN)
    # HFOV 75° 가정과도 자기무모순인지(대각/수평 미해결 이슈의 sanity check)
    assert derived == pytest.approx(math.tan(math.radians(37.5)), rel=1e-6)


def test_fix_did_not_loosen_the_safety_gate():
    """정확해지는 변경이 안전 게이트를 조용히 푸는 것을 금지한다 — 기본값의 발동점이
    옛 동작점(e*AGL > 1.0)보다 헐거우면 red."""
    cfg = LandingSMConfig()
    trip_at = cfg.max_drift_estimate_m / cfg.half_hfov_tan   # 발동에 필요한 e*AGL
    assert trip_at <= 1.0, (
        f"게이트가 헐거워졌다: 새 발동점 e*AGL>{trip_at:.4f} > 옛 발동점 1.0. "
        "식을 고쳤으면 max_drift_estimate_m 도 같이 내려야 한다."
    )
