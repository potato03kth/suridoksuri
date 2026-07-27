"""`vision/core/wire.py` — 와이어 포맷(JSON Lines) 왕복검증 + 페일세이프 계약.

핵심은 **왕복(round-trip)** 이다(`fc_bridge/tests/test_rotation.py` 관례): `TargetEstimate`/
`Decision` → 레코드 → JSON 한 줄 → 파싱 → dataclass 복원이 **무손실**이어야 한다. 소켓 단계에서
필드를 하나라도 잃으면 Phase 2 shim이 되살릴 방법이 없다.
"""
import json

import numpy as np
import pytest

from vision.core import frames, wire
from vision.core.state_machine import Decision, LandingState
from vision.core.target import TargetEstimate


def _estimate(**over) -> TargetEstimate:
    kw = dict(
        position=(1.5, -2.25, 12.75),
        orientation=(0.1, -0.2, 0.3, 0.9273618),
        confidence=1.0,
        target_type="aruco_23",
        frame_id=42,
        timestamp=1769000000.125,
        uncertainty=None,
        calib_accuracy="unverified",
        not_for_closed_loop_30cm=True,
        calib_id="vision/calibration/cam109-imx708af75/nominal.yaml",
    )
    kw.update(over)
    return TargetEstimate(**kw)


def _decision(**over) -> Decision:
    kw = dict(
        state=LandingState.PRECISION_SERVO,
        command="descend",
        reason="servoing",
        blind_duration_s=0.0,
        scale_source="agl",
    )
    kw.update(over)
    return Decision(**kw)


def _round_trip(record: dict) -> dict:
    return wire.decode_line(wire.encode_line(record))


# ── 왕복: TargetEstimate ───────────────────────────────────────────────────────

def test_target_estimate_round_trip_is_lossless():
    est = _estimate()
    rec = wire.build_target_record(est, seq=1, stamp_monotonic_ns=111, stamp_wall_ns=222)
    back = wire.target_estimate_from_record(_round_trip(rec))

    assert back.position == pytest.approx(est.position, abs=1e-12)
    assert back.orientation == pytest.approx(est.orientation, abs=1e-12)
    assert back.confidence == est.confidence
    assert back.target_type == est.target_type
    assert back.frame_id == est.frame_id
    assert back.timestamp == pytest.approx(est.timestamp, abs=1e-9)
    assert back.uncertainty is None
    assert back.calib_accuracy == est.calib_accuracy
    assert back.not_for_closed_loop_30cm == est.not_for_closed_loop_30cm
    assert back.calib_id == est.calib_id


def test_round_trip_preserves_uncertainty_matrix_when_present():
    """`uncertainty`는 이번 Phase 항상 None이지만, 실측 캘리브 후 공분산이 채워질 때를 대비해
    ndarray 경로도 지금 왕복이 되는지 확인해 둔다(그때 이 파일을 다시 안 열도록)."""
    cov = np.arange(36, dtype=np.float64).reshape(6, 6)
    rec = wire.build_target_record(
        _estimate(uncertainty=cov), seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2
    )
    back = wire.target_estimate_from_record(_round_trip(rec))
    np.testing.assert_allclose(back.uncertainty, cov)


def test_round_trip_preserves_meta():
    rec = wire.build_target_record(
        _estimate(meta={"aruco_id": 23, "note": "한글"}),
        seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2,
    )
    assert wire.target_estimate_from_record(_round_trip(rec)).meta == {"aruco_id": 23, "note": "한글"}


def test_all_required_target_keys_are_present():
    """세션 지시가 "반드시 실어야 한다"고 못박은 필드가 실제로 다 실리는가.

    (`position`/`timestamp`는 프레임·클록 모호성 때문에 `position_cam`/`position_flu`/
    `position_frd`와 명시 클록 필드로 확장돼 있다 — `core/wire.py` docstring 참조.)"""
    rec = wire.build_target_record(_estimate(), seq=7, stamp_monotonic_ns=1, stamp_wall_ns=2)
    for key in wire.REQUIRED_TARGET_KEYS:
        assert key in rec, f"필수 키 누락: {key}"
    ok, reason = wire.validate_record(_round_trip(rec))
    assert ok, reason


def test_position_is_carried_in_all_three_frames_and_they_are_consistent():
    """`position_flu`/`position_frd`가 진짜 변환 결과인가 — `position_cam`을 그대로 베껴 넣는
    구현이면 red가 된다."""
    est = _estimate(position=(0.0, -3.0, 12.5))
    rec = wire.build_target_record(est, seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2)
    np.testing.assert_allclose(rec["position_cam"], est.position, atol=1e-12)
    np.testing.assert_allclose(rec["position_flu"], frames.cam_to_flu(est.position), atol=1e-12)
    np.testing.assert_allclose(rec["position_frd"], frames.cam_to_frd(est.position), atol=1e-12)
    # 이미지 위쪽 타겟 → 기체 전방 (RT-5). 세 프레임이 서로 다른 값이어야 한다.
    np.testing.assert_allclose(rec["position_frd"], [3.0, 0.0, 12.5], atol=1e-12)


def test_mount_yaw_from_contract_actually_changes_the_body_frame_position():
    """계약의 ψ_m이 실제로 좌표변환에 먹히는가(하드코딩 0이 아닌가)."""
    est = _estimate(position=(0.0, -3.0, 12.5))
    rec = wire.build_target_record(
        est, seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2,
        contract=wire.FailsafeContract(mount_yaw_psi_m_rad=np.pi / 2),
    )
    np.testing.assert_allclose(rec["position_frd"], [0.0, 3.0, 12.5], atol=1e-12)
    assert rec["mount_yaw_psi_m_rad"] == pytest.approx(np.pi / 2)


def test_orientation_frame_is_labelled_as_camera_optical():
    """orientation은 변환하지 않고 카메라 프레임 그대로 싣는다 — Phase 2가 body 자세로
    오인하지 않도록 프레임이 반드시 명시돼야 한다."""
    rec = wire.build_target_record(_estimate(), seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2)
    assert rec["orientation_frame"] == wire.ORIENTATION_FRAME_CAM == "cam_optical"


# ── 왕복: Decision / command 힌트 ──────────────────────────────────────────────

def test_decision_round_trip_is_lossless():
    dec = _decision()
    rec = wire.build_state_hint_record(dec, seq=2, frame_id=42, stamp_monotonic_ns=1, stamp_wall_ns=2)
    back = wire.decision_from_record(_round_trip(rec))
    assert back == dec


@pytest.mark.parametrize("state", list(LandingState))
def test_every_landing_state_round_trips(state):
    """상태 enum이 전부 문자열 왕복되는가 — 새 상태가 추가돼도 조용히 깨지지 않게."""
    rec = wire.build_state_hint_record(
        _decision(state=state), seq=1, frame_id=1, stamp_monotonic_ns=1, stamp_wall_ns=2
    )
    assert wire.decision_from_record(_round_trip(rec)).state is state


def test_command_is_carried_as_an_explicitly_advisory_field():
    """🔴 §6.3 — `command`는 명령이 아니라 힌트다. 그 사실이 **형식에** 드러나야 한다.

    키 이름이 `command`로 되돌아가거나 advisory 플래그가 사라지면 red가 된다."""
    rec = wire.build_state_hint_record(
        _decision(command="descend"), seq=1, frame_id=1, stamp_monotonic_ns=1, stamp_wall_ns=2
    )
    assert rec["command_hint"] == "descend"
    assert rec["command_is_advisory"] is True
    assert "command" not in rec, "명령으로 오해될 수 있는 맨 키 이름 'command'가 있으면 안 된다"
    assert rec["type"] == "state_hint"


def test_state_hint_carries_reason_which_the_jsonl_blackbox_does_not():
    """§6.3 — `Decision.reason`은 JSONL에 안 실리는 값이라 인터페이스가 유일한 전달 경로다."""
    rec = wire.build_state_hint_record(
        _decision(reason="lock_rejected_ambiguous_candidates"),
        seq=1, frame_id=1, stamp_monotonic_ns=1, stamp_wall_ns=2,
    )
    assert rec["reason"] == "lock_rejected_ambiguous_candidates"
    for key in wire.REQUIRED_STATE_HINT_KEYS:
        assert key in rec, f"필수 키 누락: {key}"


# ── JSON Lines 프레이밍 ────────────────────────────────────────────────────────

def test_encoded_record_is_exactly_one_line():
    rec = wire.build_target_record(_estimate(), seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2)
    line = wire.encode_line(rec)
    assert line.endswith("\n")
    assert line.count("\n") == 1


def test_embedded_newlines_and_unicode_do_not_break_line_framing():
    """🔴 JSONL을 고른 근거 3번(자기 프레이밍)의 회귀. 사유 문자열에 실제 개행이 섞여도
    레코드는 여전히 정확히 한 줄이어야 하고, `readline()`으로 원값이 복원돼야 한다."""
    nasty = "줄바꿈\n포함\r\n그리고 탭\t + 유니코드 ✅"
    rec = wire.build_state_hint_record(
        _decision(reason=nasty), seq=1, frame_id=1, stamp_monotonic_ns=1, stamp_wall_ns=2
    )
    line = wire.encode_line(rec)
    assert line.count("\n") == 1
    assert wire.decode_line(line)["reason"] == nasty


def test_two_records_concatenated_split_cleanly_by_readline():
    """소비자가 실제로 하는 일(스트림을 readline으로 자르기)을 그대로 재현."""
    import io
    a = wire.build_target_record(_estimate(), seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2)
    b = wire.build_state_hint_record(
        _decision(reason="a\nb"), seq=2, frame_id=1, stamp_monotonic_ns=3, stamp_wall_ns=4
    )
    stream = io.StringIO(wire.encode_line(a) + wire.encode_line(b))
    assert wire.decode_line(stream.readline())["seq"] == 1
    assert wire.decode_line(stream.readline())["seq"] == 2
    assert stream.readline() == ""


def test_encoding_is_deterministic():
    """§7.5 — 같은 입력이면 같은 바이트."""
    est = _estimate()
    a = wire.encode_line(wire.build_target_record(est, seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2))
    b = wire.encode_line(wire.build_target_record(est, seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2))
    assert a == b


def test_decode_line_rejects_blank_and_non_object():
    with pytest.raises(ValueError):
        wire.decode_line("   \n")
    with pytest.raises(ValueError):
        wire.decode_line("[1, 2, 3]")


# ── 시계 계약 (§4.6) ───────────────────────────────────────────────────────────

def test_both_clocks_and_their_offset_are_carried():
    """vision은 `time.time()`, fc는 `time.monotonic()`을 쓴다 — 둘 다 싣고 오프셋도 싣는다."""
    rec = wire.build_target_record(
        _estimate(), seq=1, stamp_monotonic_ns=1_000, stamp_wall_ns=1_769_000_000_000_000_000
    )
    assert rec["stamp_monotonic_ns"] == 1_000
    assert rec["stamp_wall_ns"] == 1_769_000_000_000_000_000
    assert rec["clock_offset_ns"] == rec["stamp_wall_ns"] - rec["stamp_monotonic_ns"]


def test_source_timestamp_is_preserved_verbatim():
    """`TargetEstimate.timestamp` 원본이 클록 필드에 덮이지 않고 그대로 남아야 한다."""
    rec = wire.build_target_record(
        _estimate(timestamp=1769000000.125), seq=1, stamp_monotonic_ns=5, stamp_wall_ns=9
    )
    assert rec["timestamp"] == pytest.approx(1769000000.125, abs=1e-9)


# ── 페일세이프 계약 §5.3 — confidence 게이트 ──────────────────────────────────

def test_confidence_gate_is_disabled_by_default():
    """🔴 §5.3 — 기본값 0.0(비활성). 0.9 같은 값이 기본이 되면 "동작하는 것처럼 보이지만
    아무것도 거르지 않는 가짜 안전장치"가 된다."""
    c = wire.FailsafeContract()
    assert c.min_confidence == 0.0
    ok, reason = wire.gate_confidence(0.0, c)
    assert ok and reason == "confidence_gate_disabled"


def test_confidence_gate_is_a_noop_for_the_aruco_path_today():
    """🔴 §5.3의 측정 사실: ArUco `confidence`는 **항상 정확히 1.0**이라 어떤 하한을 켜도
    1.0은 통과한다 — 즉 이 게이트는 지금 no-op다. 그 사실 자체를 테스트로 고정해 둔다."""
    for threshold in (0.5, 0.9, 1.0):
        ok, _ = wire.gate_confidence(1.0, wire.FailsafeContract(min_confidence=threshold))
        assert ok, "confidence=1.0이 거절되면 ArUco 경로가 통째로 막힌다"


def test_confidence_gate_actually_gates_when_enabled():
    """게이트가 껍데기가 아님을 확인 — 켜고 미달값을 주면 거절되고 레코드가 valid=false가 된다."""
    c = wire.FailsafeContract(min_confidence=0.9)
    ok, reason = wire.gate_confidence(0.5, c)
    assert not ok and "confidence_below_min" in reason

    rec = wire.build_target_record(
        _estimate(confidence=0.5), seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2, contract=c
    )
    assert rec["valid"] is False
    assert "confidence_below_min" in rec["reason"]
    # 거절돼도 포즈는 그대로 실려 있다(로깅·사후분석용). 소비자가 valid를 보고 판단한다.
    assert rec["position_frd"] is not None


# ── 페일세이프 계약 §5.5 — not_for_closed_loop_30cm ───────────────────────────

def test_flag_true_means_floor_not_prohibition():
    """🔴 §5.5 — "True면 폐루프 금지"로 계약하면 통합이 착수 즉시 죽는다(플래그는 지금 100% True).
    "최종 커밋 금지"로 해석해 **바닥 고도**를 준다."""
    c = wire.FailsafeContract()
    assert wire.closed_loop_floor_agl_m(True, c) == 3.0
    rec = wire.build_target_record(
        _estimate(not_for_closed_loop_30cm=True), seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2
    )
    assert rec["valid"] is True, "플래그가 True라고 추정치를 무효화하면 안 된다(=착수 즉시 사망)"
    assert rec["closed_loop_floor_agl_m"] == 3.0
    assert rec["not_for_closed_loop_30cm"] is True


def test_flag_false_unlocks_terminal_without_code_change():
    """실측 캘리브 후 플래그가 False가 되면 바닥이 0이 되어 같은 코드가 TERMINAL까지 간다."""
    rec = wire.build_target_record(
        _estimate(not_for_closed_loop_30cm=False), seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2
    )
    assert rec["closed_loop_floor_agl_m"] == 0.0


def test_floor_default_matches_state_machine_terminal_agl():
    """§5.6 — 바닥 고도 기본값은 상태머신 `terminal_agl_m` 기본값과 정렬돼야 한다."""
    from vision.core.state_machine import LandingSMConfig
    assert wire.FailsafeContract().unverified_calib_floor_agl_m == LandingSMConfig().terminal_agl_m


def test_floor_is_configurable_not_hardcoded():
    c = wire.FailsafeContract(unverified_calib_floor_agl_m=5.0, verified_calib_floor_agl_m=0.3)
    assert wire.closed_loop_floor_agl_m(True, c) == 5.0
    assert wire.closed_loop_floor_agl_m(False, c) == 0.3


def test_provenance_echo_survives_the_wire():
    """§7.3 — "어느 캘리브로 난 비행인가"를 사후에 특정할 수 있어야 한다."""
    rec = _round_trip(wire.build_target_record(
        _estimate(calib_accuracy="measured", calib_id="calib/2026-07-30.yaml"),
        seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2,
    ))
    assert rec["calib_accuracy"] == "measured"
    assert rec["calib_id"] == "calib/2026-07-30.yaml"


def test_unmeasured_mount_yaw_is_propagated_to_the_consumer():
    """🔴 §7 U3 — ψ_m이 미측정이라는 사실이 소비자까지 가야 한다."""
    rec = wire.build_target_record(_estimate(), seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2)
    assert rec["mount_yaw_measured"] is False


# ── 페일세이프 계약 §5.4 — valid=false 레코드 ─────────────────────────────────

def test_invalid_record_reports_loss_without_going_silent():
    """§5.4 계약 2번 — 검출을 잃어도 발행을 멈추지 않는다. 침묵은 "죽음"으로 예약돼 있다."""
    rec = _round_trip(wire.build_invalid_target_record(
        seq=9, frame_id=100, reason="no_detection", stamp_monotonic_ns=1, stamp_wall_ns=2
    ))
    ok, why = wire.validate_record(rec)
    assert ok, why
    assert rec["valid"] is False
    assert rec["reason"] == "no_detection"
    assert rec["seq"] == 9 and rec["frame_id"] == 100


def test_invalid_record_has_null_pose_not_zeros():
    """🔴 0.0으로 채우면 소비자가 valid를 안 볼 때 "정확히 기체 바로 아래에 타겟이 있다"는
    가장 위험한 거짓말이 된다."""
    rec = wire.build_invalid_target_record(
        seq=1, frame_id=1, reason="x", stamp_monotonic_ns=1, stamp_wall_ns=2
    )
    for key in ("position_cam", "position_flu", "position_frd", "orientation", "confidence"):
        assert rec[key] is None, f"{key}가 None이 아니다 — 무효 레코드가 거짓 포즈를 실었다"


def test_invalid_record_stays_conservative_on_the_calib_flag():
    rec = wire.build_invalid_target_record(
        seq=1, frame_id=1, reason="x", stamp_monotonic_ns=1, stamp_wall_ns=2
    )
    assert rec["not_for_closed_loop_30cm"] is True


def test_invalid_record_cannot_be_restored_as_a_target_estimate():
    rec = wire.build_invalid_target_record(
        seq=1, frame_id=1, reason="no_detection", stamp_monotonic_ns=1, stamp_wall_ns=2
    )
    with pytest.raises(ValueError):
        wire.target_estimate_from_record(rec)


# ── 스키마 버전 / 검증 ────────────────────────────────────────────────────────

def test_schema_version_mismatch_is_rejected():
    """§7.2가 `schema_version`을 요구한 이유 — 모르는 버전을 조용히 먹으면 안 된다."""
    rec = wire.build_target_record(_estimate(), seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2)
    rec["schema_version"] = 999
    with pytest.raises(ValueError):
        wire.target_estimate_from_record(rec)
    ok, reason = wire.validate_record(rec)
    assert not ok and "schema_version" in reason


def test_validate_record_detects_missing_keys_and_unknown_types():
    rec = wire.build_target_record(_estimate(), seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2)
    del rec["position_frd"]
    ok, reason = wire.validate_record(rec)
    assert not ok and "position_frd" in reason

    ok, reason = wire.validate_record({"type": "nope", "schema_version": wire.SCHEMA_VERSION})
    assert not ok and "unknown_record_type" in reason


def test_wrong_record_type_is_rejected_by_the_typed_decoders():
    target = wire.build_target_record(_estimate(), seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2)
    hint = wire.build_state_hint_record(
        _decision(), seq=1, frame_id=1, stamp_monotonic_ns=1, stamp_wall_ns=2
    )
    with pytest.raises(ValueError):
        wire.decision_from_record(target)
    with pytest.raises(ValueError):
        wire.target_estimate_from_record(hint)


def test_record_is_plain_json_serialisable_without_numpy_leakage():
    """numpy 스칼라/배열이 새어 나가면 `json.dumps`가 죽는다(`calib_analyze.py`의
    numpy.bool_ 누출 회귀와 같은 종류의 함정)."""
    est = _estimate(
        position=tuple(np.array([1.0, 2.0, 3.0])),
        confidence=np.float64(1.0),
        not_for_closed_loop_30cm=np.bool_(True),
    )
    rec = wire.build_target_record(est, seq=1, stamp_monotonic_ns=1, stamp_wall_ns=2)
    json.dumps(rec)  # 예외 없이 통과해야 한다
    assert isinstance(rec["not_for_closed_loop_30cm"], bool)
    assert isinstance(rec["confidence"], float)
