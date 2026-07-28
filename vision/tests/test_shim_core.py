"""`vision/ros/shim_core.py` — 컨테이너 shim의 순수 변환 로직.

## 이 테스트가 rclpy 없이 도는 방법

`shim_core.py`는 rclpy도 vision도 import하지 않는 stdlib 전용 모듈이라 그냥 import된다.
rclpy를 import하는 것은 `shim_node.py` 하나뿐이고, 이 파일은 그것을 **import하지 않고
소스 원문(텍스트)으로만** 읽는다(`test_frame_source.py`가 picamera2 최상단 import 부재를
소스 텍스트로 검사하는 것과 같은 전례). 랩탑 `.venv`에 rclpy가 없어도 전부 통과한다.

## 입력을 지어내지 않는다

레코드는 손으로 쓴 dict가 아니라 **진짜 생산자 코드**(`vision.core.wire`)로 만든다 — 그래야
"shim이 상상한 스키마"가 아니라 "실제로 흘러오는 바이트"를 검증한다. 랩탑엔 cv2가 있어
`vision.core.wire` import가 되므로 **양쪽 계약을 직접 대조**할 수도 있다(런타임은 격리,
검증은 대조 — `shim_core.py` docstring 참조).
"""
import ast
import json
import pathlib

import numpy as np
import pytest

from vision.core import wire
from vision.core.state_machine import Decision, LandingState
from vision.core.target import TargetEstimate
from vision.ros import shim_core
from vision.ros.shim_core import ShimConfig, ShimRouter

_SHIM_NODE_PATH = pathlib.Path(shim_core.__file__).with_name("shim_node.py")

_MONO = 1_000_000_000_000
_WALL = 1_769_000_000_000_000_000


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


def _target_line(estimate=None, *, seq=1, mono=_MONO, wall=_WALL) -> str:
    rec = wire.build_target_record(
        estimate if estimate is not None else _estimate(),
        seq=seq,
        stamp_monotonic_ns=mono,
        stamp_wall_ns=wall,
    )
    return wire.encode_line(rec)


def _invalid_line(*, seq=1, frame_id=42, reason="no_target_detection") -> str:
    rec = wire.build_invalid_target_record(
        seq=seq, frame_id=frame_id, reason=reason,
        stamp_monotonic_ns=_MONO, stamp_wall_ns=_WALL,
    )
    return wire.encode_line(rec)


def _state_hint_line(*, seq=2, frame_id=42, state=LandingState.PRECISION_SERVO) -> str:
    rec = wire.build_state_hint_record(
        Decision(state=state, command="descend", reason="locked", blind_duration_s=0.0,
                 scale_source="known_size"),
        seq=seq, frame_id=frame_id, stamp_monotonic_ns=_MONO, stamp_wall_ns=_WALL,
    )
    return wire.encode_line(rec)


def _push(router, line, *, mono=_MONO, wall=_WALL):
    return router.push(line, recv_monotonic_ns=mono, recv_wall_ns=wall)


def _emit(router, target_line, *, mono=_MONO, wall=_WALL, hint_frame_id=42):
    """target을 넣고 짝이 되는 state_hint로 **밀어내** 완성된 `ShimOutput` 하나를 받는다.

    라우터는 stamp 동기를 위해 target을 잠깐 들고 있으므로(파일 docstring "짝짓기"),
    target만 넣으면 출력이 비는 것이 **정상**이다.
    """
    assert _push(router, target_line, mono=mono, wall=wall) == []
    outs = _push(router, _state_hint_line(frame_id=hint_frame_id), mono=mono, wall=wall)
    assert len(outs) == 1
    return outs[0]


def _kv(status) -> dict:
    return dict(status.values)


def _find(statuses, name):
    for s in statuses:
        if s.name == name:
            return s
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 계약 대조 — 복제한 상수가 생산자와 어긋나면 red (조용한 드리프트 방지)
# ──────────────────────────────────────────────────────────────────────────────


def test_schema_version_matches_producer():
    """shim은 wire.py를 import할 수 없어 상수를 복제한다 — 그 복제가 틀어지면 여기서 잡힌다."""
    assert shim_core.SCHEMA_VERSION == wire.SCHEMA_VERSION


def test_required_keys_match_producer():
    assert set(shim_core.REQUIRED_TARGET_KEYS) == set(wire.REQUIRED_TARGET_KEYS)
    assert set(shim_core.REQUIRED_STATE_HINT_KEYS) == set(wire.REQUIRED_STATE_HINT_KEYS)


def test_record_type_and_orientation_frame_names_match_producer():
    assert shim_core.RECORD_TYPE_TARGET == wire.RECORD_TYPE_TARGET
    assert shim_core.RECORD_TYPE_STATE_HINT == wire.RECORD_TYPE_STATE_HINT
    assert shim_core.ORIENTATION_FRAME_CAM == wire.ORIENTATION_FRAME_CAM


def test_shim_core_imports_no_vision_and_no_rclpy():
    """🔴 컨테이너엔 cv2가 없다 — `vision.core.wire`를 import하면 실기체에서 즉시 죽는다.

    (`wire.py → core/target.py → import cv2` 체인.) 소스 원문으로 고정한다.
    """
    src = pathlib.Path(shim_core.__file__).read_text(encoding="utf-8")
    code = "\n".join(
        ln for ln in src.splitlines() if ln.startswith(("import ", "from "))
    )
    assert "rclpy" not in code
    assert "vision." not in code
    assert "numpy" not in code
    assert "cv2" not in code


# ──────────────────────────────────────────────────────────────────────────────
# 🔴 schema_version 거절 (파괴검증 S-D2 대상)
# ──────────────────────────────────────────────────────────────────────────────


def test_unknown_schema_version_is_rejected_and_publishes_no_pose():
    rec = json.loads(_target_line())
    rec["schema_version"] = 2
    router = ShimRouter()
    outs = _push(router, json.dumps(rec))
    assert len(outs) == 1
    assert outs[0].pose is None, "모르는 스키마로 pose를 만들면 틀린 좌표로 유도한다"
    link = _find(outs[0].statuses, shim_core.STATUS_NAME_LINK)
    assert link is not None and link.level == shim_core.LEVEL_ERROR
    assert "schema_version_mismatch" in link.message
    assert router.rejected == 1


def test_validate_record_rejects_missing_keys_and_unknown_type():
    rec = json.loads(_target_line())
    del rec["position_flu"]
    ok, reason = shim_core.validate_record(rec)
    assert not ok and "missing_keys" in reason and "position_flu" in reason
    ok, reason = shim_core.validate_record({"type": "nope", "schema_version": 1})
    assert not ok and "unknown_record_type" in reason


def test_valid_record_passes_validation():
    ok, reason = shim_core.validate_record(json.loads(_target_line()))
    assert ok and reason == "ok"
    ok, reason = shim_core.validate_record(json.loads(_state_hint_line()))
    assert ok and reason == "ok"


# ──────────────────────────────────────────────────────────────────────────────
# 🔴 position_flu가 유도 입력이다 (파괴검증 S-D3 대상)
# ──────────────────────────────────────────────────────────────────────────────


def test_pose_position_is_position_flu_not_frd():
    """FLU와 FRD는 y·z 부호가 반대다 — 잘못 쓰면 좌우·상하가 뒤집힌 채 유도한다."""
    line = _target_line()
    rec = json.loads(line)
    flu, frd = rec["position_flu"], rec["position_frd"]
    # 전제 확인: 두 값이 실제로 달라야 이 테스트에 이빨이 있다.
    assert flu != frd
    assert flu[0] == pytest.approx(frd[0])
    assert flu[1] == pytest.approx(-frd[1])
    assert flu[2] == pytest.approx(-frd[2])

    router = ShimRouter()
    out = _emit(router, line)
    assert out.pose is not None
    assert list(out.pose.position) == pytest.approx(flu)
    assert list(out.pose.position) != pytest.approx(frd)


def test_pose_frame_id_is_base_link_flu():
    out = _emit(ShimRouter(), _target_line())
    assert out.pose.frame_id == shim_core.DEFAULT_POSE_FRAME_ID == "base_link"


def test_all_three_position_frames_survive_into_keyvalues():
    """§7.3 — 한 번 버린 provenance는 되살릴 수 없다. 손실 압축은 shim 책임이지 폐기가 아니다."""
    rec = json.loads(_target_line())
    out = _emit(ShimRouter(), json.dumps(rec))
    kv = _kv(_find(out.statuses, shim_core.STATUS_NAME_TARGET))
    for key in ("position_cam", "position_flu", "position_frd"):
        assert json.loads(kv[key]) == pytest.approx(rec[key])


# ──────────────────────────────────────────────────────────────────────────────
# 🔴 EOF = 생산자 사망 (파괴검증 S-D4 대상)
# ──────────────────────────────────────────────────────────────────────────────


def test_producer_eof_emits_error_status_and_never_a_pose():
    router = ShimRouter()
    outs = router.on_producer_gone(recv_wall_ns=_WALL, reason="eof")
    assert len(outs) == 1
    assert outs[0].pose is None
    assert outs[0].landing_target is None
    link = _find(outs[0].statuses, shim_core.STATUS_NAME_LINK)
    assert link.level == shim_core.LEVEL_ERROR
    assert _kv(link)["event"] == "producer_eof"


def test_producer_eof_discards_pending_target_instead_of_publishing_it_later():
    """죽은 생산자의 좌표를 사후에 발행하는 것은 '죽은 데이터로 유도'다."""
    router = ShimRouter()
    assert _push(router, _target_line()) == []  # 짝을 기다리며 보류 중
    outs = router.on_producer_gone(recv_wall_ns=_WALL, reason="eof")
    assert all(o.pose is None for o in outs)
    # EOF 이후 새 state_hint가 와도 보류분이 되살아나지 않는다.
    outs2 = _push(router, _state_hint_line())
    assert all(o.pose is None for o in outs2)


def test_disconnected_heartbeat_separates_producer_death_from_shim_death():
    """둘 다 pose는 침묵한다 — status가 계속 나오는가가 유일한 구분자다."""
    outs = ShimRouter().on_disconnected_heartbeat(recv_wall_ns=_WALL, endpoint="127.0.0.1:8091")
    link = _find(outs[0].statuses, shim_core.STATUS_NAME_LINK)
    assert link.level == shim_core.LEVEL_ERROR
    assert _kv(link)["event"] == "disconnected"
    assert outs[0].pose is None


# ──────────────────────────────────────────────────────────────────────────────
# 🔴 LandingTarget frame 정수 리터럴 (파괴검증 S-D1 대상)
# ──────────────────────────────────────────────────────────────────────────────


def test_mav_frame_constants_are_the_real_enum_not_the_msg_constants():
    """`LandingTarget.msg` 상수는 실제 `MAV_FRAME`과 어긋나 있다(실기체에서 두 파일 대조).

    msg: LOCAL_NED=2, MISSION=3, BODY_FRD 없음 / 실제: LOCAL_NED=1, MISSION=2, BODY_FRD=12.
    """
    assert shim_core.MAV_FRAME_BODY_FRD == 12
    assert shim_core.MAV_FRAME_LOCAL_NED == 1
    # msg 상수를 그대로 쓴 값들 — 어느 것도 우리 상수가 되면 안 된다.
    msg_local_ned, msg_mission = 2, 3
    assert shim_core.MAV_FRAME_LOCAL_NED != msg_local_ned
    assert shim_core.MAV_FRAME_BODY_FRD not in (msg_local_ned, msg_mission)


def test_landing_target_plan_uses_frame_12_and_flu_position():
    cfg = ShimConfig(publish_landing_target=True)
    rec = json.loads(_target_line())
    out = _emit(ShimRouter(cfg), json.dumps(rec))
    lt = out.landing_target
    assert lt is not None
    assert lt.frame == 12, "msg 상수(2)면 MAVLink가 MISSION으로 읽어 position_valid=false"
    # 🔴 frame=12여도 ROS 쪽 입력은 FLU다 — 플러그인이 baselink→aircraft로 바꾼다.
    assert list(lt.position) == pytest.approx(rec["position_flu"])
    assert list(lt.position) != pytest.approx(rec["position_frd"])


def test_landing_target_is_off_by_default_because_listen_lt_is_false():
    """`px4_config.yaml:214` `listen_lt: false` — 구독자가 없는 토픽을 기본으로 켜지 않는다."""
    assert ShimConfig().publish_landing_target is False
    out = _emit(ShimRouter(), _target_line())
    assert out.landing_target is None


def test_landing_target_type_maps_fiducial_vs_other():
    cfg = ShimConfig(publish_landing_target=True)
    out = _emit(ShimRouter(cfg), _target_line(_estimate(target_type="aruco_23")))
    assert out.landing_target.type == shim_core.LANDING_TARGET_TYPE_VISION_FIDUCIAL == 2
    out = _emit(
        ShimRouter(cfg), _target_line(_estimate(target_type="distress_landing_point"))
    )
    assert out.landing_target.type == shim_core.LANDING_TARGET_TYPE_VISION_OTHER == 3


def test_landing_target_distance_and_angles_are_projections_of_the_same_vector():
    cfg = ShimConfig(publish_landing_target=True)
    rec = json.loads(_target_line())
    lt = _emit(ShimRouter(cfg), json.dumps(rec)).landing_target
    x, y, z = rec["position_flu"]
    assert lt.distance == pytest.approx((x * x + y * y + z * z) ** 0.5)
    # 나디르(타겟이 아래) — z_flu < 0 이어야 각이 정의된다.
    assert z < 0
    assert lt.angle[0] == pytest.approx(np.arctan2(x, -z))
    assert lt.angle[1] == pytest.approx(np.arctan2(-y, -z))
    assert lt.size == (0.0, 0.0), "물리 크기는 와이어에 없다 — 지어내지 않는다"


def test_angles_are_zero_when_target_is_above_the_vehicle():
    assert shim_core._angles_from_flu((1.0, 1.0, 5.0)) == (0.0, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# §5.4 — valid=false는 "안 보임", 침묵은 "죽음"
# ──────────────────────────────────────────────────────────────────────────────


def test_invalid_record_publishes_warn_status_but_no_pose():
    router = ShimRouter()
    outs = _push(router, _invalid_line(reason="no_target_detection"))
    outs += _push(router, _state_hint_line())
    out = outs[-1]
    assert out.pose is None, "실을 좌표가 없다 — 0으로 채우면 '기체 바로 아래' 라는 거짓말"
    assert out.landing_target is None
    target = _find(out.statuses, shim_core.STATUS_NAME_TARGET)
    assert target.level == shim_core.LEVEL_WARN
    assert "no_target_detection" in target.message
    assert json.loads(_kv(target)["valid"]) is False


def test_invalid_record_keeps_conservative_closed_loop_floor():
    router = ShimRouter()
    _push(router, _invalid_line())
    out = _push(router, _state_hint_line())[0]
    kv = _kv(_find(out.statuses, shim_core.STATUS_NAME_TARGET))
    assert json.loads(kv["not_for_closed_loop_30cm"]) is True
    assert json.loads(kv["closed_loop_floor_agl_m"]) == pytest.approx(3.0)


def test_malformed_line_does_not_crash_and_reports_error():
    router = ShimRouter()
    outs = _push(router, "{not json at all")
    assert outs[0].pose is None
    link = _find(outs[0].statuses, shim_core.STATUS_NAME_LINK)
    assert link.level == shim_core.LEVEL_ERROR
    assert _kv(link)["event"] == "malformed_line"
    # 다음 정상 줄은 정상 처리된다 — 한 줄이 깨져도 스트림이 죽지 않는다.
    router.push(_target_line(), recv_monotonic_ns=_MONO, recv_wall_ns=_WALL)
    out = _push(router, _state_hint_line())[0]
    assert out.pose is not None


# ──────────────────────────────────────────────────────────────────────────────
# 🔴 not_for_closed_loop_30cm은 level에 태우지 않는다
# ──────────────────────────────────────────────────────────────────────────────


def test_unverified_calibration_does_not_permanently_raise_level():
    """지금 100% True다 — level에 태우면 level이 항상 WARN이라 진짜 고장이 묻힌다."""
    router = ShimRouter()
    _push(router, _target_line(_estimate(not_for_closed_loop_30cm=True)))
    out = _push(router, _state_hint_line())[0]
    target = _find(out.statuses, shim_core.STATUS_NAME_TARGET)
    assert target.level == shim_core.LEVEL_OK
    kv = _kv(target)
    assert json.loads(kv["not_for_closed_loop_30cm"]) is True
    # 대신 "최종 커밋 금지" 해석이 숫자로 나간다(3.0m까지 정렬 후 인계).
    assert json.loads(kv["closed_loop_floor_agl_m"]) == pytest.approx(3.0)


def test_verified_calibration_unlocks_floor_to_zero():
    router = ShimRouter()
    _push(router, _target_line(_estimate(not_for_closed_loop_30cm=False)))
    out = _push(router, _state_hint_line())[0]
    kv = _kv(_find(out.statuses, shim_core.STATUS_NAME_TARGET))
    assert json.loads(kv["closed_loop_floor_agl_m"]) == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 🔴 orientation은 cam_optical — Pose에 싣지 않는다
# ──────────────────────────────────────────────────────────────────────────────


def test_pose_orientation_is_identity_not_the_camera_frame_quaternion():
    rec = json.loads(_target_line())
    out = _emit(ShimRouter(), json.dumps(rec))
    assert out.pose.orientation == (0.0, 0.0, 0.0, 1.0)
    assert list(out.pose.orientation) != pytest.approx(rec["orientation"])


def test_camera_frame_quaternion_survives_in_keyvalues_with_its_frame_name():
    rec = json.loads(_target_line())
    out = _emit(ShimRouter(), json.dumps(rec))
    kv = _kv(_find(out.statuses, shim_core.STATUS_NAME_TARGET))
    assert json.loads(kv["orientation_cam_optical_xyzw"]) == pytest.approx(rec["orientation"])
    assert kv["orientation_frame"] == "cam_optical"


def test_rotation_covariance_is_unknown_not_zero():
    cfg = ShimConfig(unknown_covariance=1234.5)
    out = _emit(ShimRouter(cfg), _target_line())
    cov = out.pose.covariance
    assert len(cov) == 36
    for i in (21, 28, 35):
        assert cov[i] == pytest.approx(1234.5)


def test_default_unknown_covariance_is_not_zero():
    """🔴 **기본값 자체**를 고정한다.

    이 테스트가 없어서 파괴검증 D6(`DEFAULT_UNKNOWN_COVARIANCE = 0.0`)이 조용히 green으로
    통과했다 — 다른 공분산 테스트가 전부 `ShimConfig(unknown_covariance=...)`로 값을 **명시**해
    기본값을 한 번도 밟지 않았기 때문이다. 기본값이 0이면 `--unknown-covariance`를 안 준
    실제 운용에서 "분산 0 = 완벽히 정확"이 나간다.
    """
    assert shim_core.DEFAULT_UNKNOWN_COVARIANCE > 0.0
    assert ShimConfig().unknown_covariance > 0.0
    out = _emit(ShimRouter(), _target_line())  # 기본 config
    for i in (0, 7, 14, 21, 28, 35):
        assert out.pose.covariance[i] > 0.0


def test_invalid_record_with_a_position_still_publishes_no_pose():
    """🔴 `valid`는 **독립적인** 게이트다 — 좌표가 있어도 무효면 안 나간다.

    이 테스트가 없어서 파괴검증 D7(`if valid and ...` → `if ...`)이 green으로 통과했다:
    정상 생산자의 무효 레코드는 `position_flu=null`이라 좌표 존재 검사만으로도 걸러졌기
    때문이다. 하지만 그건 **생산자가 착해서** 통과한 것이고, 게이트가 사라진 사실은 못 잡는다.
    confidence 게이트가 켜지면(§5.3) "좌표는 있는데 valid=false"가 실제로 발생한다.
    """
    rec = json.loads(_target_line())
    rec["valid"] = False
    rec["reason"] = "confidence_below_min(0.100<0.900)"
    assert rec["position_flu"] is not None  # 전제: 좌표는 멀쩡히 들어 있다
    router = ShimRouter(ShimConfig(publish_landing_target=True))
    out = _emit(router, json.dumps(rec))
    assert out.pose is None, "valid=false인데 pose가 나가면 거절된 추정치로 유도한다"
    assert out.landing_target is None
    target = _find(out.statuses, shim_core.STATUS_NAME_TARGET)
    assert target.level == shim_core.LEVEL_WARN
    assert "confidence_below_min" in target.message


def test_position_covariance_is_unknown_while_uncertainty_is_none():
    """0을 채우면 '분산 0 = 완벽히 정확'이라 기성 필터가 절대 신뢰로 받아들인다."""
    rec = json.loads(_target_line())
    assert rec["uncertainty"] is None  # 이번 Phase 확정 전제
    cfg = ShimConfig(unknown_covariance=9e5)
    out = _emit(ShimRouter(cfg), json.dumps(rec))
    for i in (0, 7, 14):
        assert out.pose.covariance[i] == pytest.approx(9e5)
    kv = _kv(_find(out.statuses, shim_core.STATUS_NAME_TARGET))
    assert kv["uncertainty_source"] == "none"


def test_real_covariance_is_used_when_wire_carries_one():
    """실측 캘리브 후 공분산이 채워지면 이 파일을 다시 안 열어도 되게 지금 처리해 둔다."""
    unc = np.arange(36, dtype=np.float64).reshape(6, 6)
    out = _emit(ShimRouter(), _target_line(_estimate(uncertainty=unc)))
    assert out.pose.covariance[0] == pytest.approx(0.0)
    assert out.pose.covariance[14] == pytest.approx(14.0)
    # 회전 블록만은 여전히 '모름'이다 — Pose orientation이 단위 쿼터니언이므로.
    assert out.pose.covariance[35] == pytest.approx(shim_core.DEFAULT_UNKNOWN_COVARIANCE)


# ──────────────────────────────────────────────────────────────────────────────
# §6.3 — command는 advisory, 거부권은 state
# ──────────────────────────────────────────────────────────────────────────────


def test_state_status_carries_command_hint_and_never_a_bare_command_key():
    router = ShimRouter()
    _push(router, _target_line())
    out = _push(router, _state_hint_line())[0]
    kv = _kv(_find(out.statuses, shim_core.STATUS_NAME_STATE))
    assert kv["command_hint"] == "descend"
    assert json.loads(kv["command_is_advisory"]) is True
    assert "command" not in kv, "맨 이름 command 키는 '명령'으로 오해된다"


@pytest.mark.parametrize("state", [LandingState.HOLD, LandingState.ABORT_ASCEND])
def test_veto_states_raise_level(state):
    router = ShimRouter()
    _push(router, _target_line())
    out = _push(router, _state_hint_line(state=state))[0]
    st = _find(out.statuses, shim_core.STATUS_NAME_STATE)
    assert st.level == shim_core.LEVEL_WARN
    assert "veto" in st.message


def test_non_veto_states_stay_ok():
    router = ShimRouter()
    _push(router, _target_line())
    out = _push(router, _state_hint_line(state=LandingState.PRECISION_SERVO))[0]
    assert _find(out.statuses, shim_core.STATUS_NAME_STATE).level == shim_core.LEVEL_OK


# ──────────────────────────────────────────────────────────────────────────────
# stamp 동기 / 짝짓기
# ──────────────────────────────────────────────────────────────────────────────


def test_target_and_state_hint_leave_together_with_one_shared_stamp():
    """§3.2 계약 — 두 토픽을 나누면 반드시 같은 `header.stamp`여야 한다."""
    router = ShimRouter()
    assert _push(router, _target_line(seq=1)) == []
    outs = _push(router, _state_hint_line(seq=2))
    assert len(outs) == 1
    out = outs[0]
    assert out.pose is not None
    assert {s.name for s in out.statuses} == {
        shim_core.STATUS_NAME_TARGET, shim_core.STATUS_NAME_STATE
    }
    assert out.stamp_ns == _WALL
    assert out.pose.stamp_ns == out.stamp_ns


def test_header_stamp_uses_wall_clock_not_monotonic():
    """ROS 기본 클록은 CLOCK_REALTIME이다 — monotonic을 넣으면 now()와 비교가 무의미해진다."""
    out = _emit(ShimRouter(), _target_line(mono=_MONO, wall=_WALL))
    assert out.stamp_ns == _WALL != _MONO


def test_unpaired_target_is_flushed_when_next_target_arrives():
    """state_hint가 없다고 유도를 볼모로 잡지 않는다."""
    router = ShimRouter()
    _push(router, _target_line(seq=1))
    outs = _push(router, _target_line(seq=3))
    assert len(outs) == 1 and outs[0].pose is not None
    st = _find(outs[0].statuses, shim_core.STATUS_NAME_STATE)
    assert st.level == shim_core.LEVEL_WARN and st.message == "state_hint_missing"
    assert router.unpaired_targets == 1


def test_state_hint_with_mismatched_frame_id_is_not_mispaired():
    router = ShimRouter()
    _push(router, _target_line(seq=1))  # frame_id=42
    outs = _push(router, _state_hint_line(seq=2, frame_id=99))
    assert len(outs) == 2
    # 보류 target은 짝 없이 나가고, 엉뚱한 state_hint는 단독 status로 나간다.
    assert outs[0].pose is not None
    assert _find(outs[0].statuses, shim_core.STATUS_NAME_STATE).message == "state_hint_missing"
    assert outs[1].pose is None
    assert router.orphan_state_hints == 1


def test_stale_record_raises_level_but_still_publishes_pose():
    """stale 판정의 최종 거부권은 FC의 몫이다(§5.4) — shim은 재서 알리기만 한다."""
    cfg = ShimConfig(stale_warn_s=0.1)
    router = ShimRouter(cfg)
    _push(router, _target_line(mono=_MONO), mono=_MONO + 500_000_000)
    out = _push(router, _state_hint_line())[0]
    target = _find(out.statuses, shim_core.STATUS_NAME_TARGET)
    assert target.level == shim_core.LEVEL_WARN
    assert "stale" in target.message
    assert out.pose is not None, "shim이 데이터를 감추면 FC가 스스로 판단할 수 없다"
    assert json.loads(_kv(target)["age_s"]) == pytest.approx(0.5)


#: U5 실측(2026-07-28 실기체 라이브, 4608x2592 + distress_fine.yaml) — 프레임 간격 p95.
_MEASURED_FRAME_INTERVAL_P95_S = 0.310


def test_stale_warn_default_has_margin_over_the_measured_frame_interval():
    """🔴 기본값이 **실측**보다 여유가 있어야 한다 — 아니면 정상 지터에 헛경보가 뜬다.

    실측 발행 주파수는 약 4.4Hz(median 간격 0.221s / p95 0.310s)이지 문서가 가정해 온
    10Hz가 아니다. 헛경보는 진짜 경보를 무디게 만들므로 여유를 2배 이상 요구한다.
    """
    assert ShimConfig().stale_warn_s >= 2.0 * _MEASURED_FRAME_INTERVAL_P95_S
    # 그렇다고 무한정 크면 진짜 정지를 못 잡는다 — 상한도 같이 건다.
    assert ShimConfig().stale_warn_s <= 10.0 * _MEASURED_FRAME_INTERVAL_P95_S


def test_normal_frame_interval_does_not_trip_the_stale_warning():
    """실측 p95 간격만큼 늦은 레코드는 **OK**여야 한다(정상 동작이므로)."""
    router = ShimRouter()
    late_ns = int(_MEASURED_FRAME_INTERVAL_P95_S * 1e9)
    _push(router, _target_line(mono=_MONO), mono=_MONO + late_ns)
    out = _push(router, _state_hint_line())
    assert _find(out[0].statuses, shim_core.STATUS_NAME_TARGET).level == shim_core.LEVEL_OK


def test_age_is_measured_on_monotonic_clock():
    """컨테이너·호스트 CLOCK_MONOTONIC이 같은 기준임을 실측했으므로 이 뺄셈이 성립한다."""
    router = ShimRouter()
    _push(router, _target_line(mono=_MONO, wall=_WALL), mono=_MONO + 250_000_000, wall=0)
    out = _push(router, _state_hint_line())[0]
    kv = _kv(_find(out.statuses, shim_core.STATUS_NAME_TARGET))
    assert json.loads(kv["age_s"]) == pytest.approx(0.25)
    assert json.loads(kv["recv_monotonic_ns"]) == _MONO + 250_000_000


# ──────────────────────────────────────────────────────────────────────────────
# 🔴 초록구역 0.105m — 고치지 않고 명시적으로 수출한다
# ──────────────────────────────────────────────────────────────────────────────


def _distress_estimate():
    return _estimate(
        target_type="distress_landing_point",
        meta={
            "plane_reference": "mat_top_surface",
            "platform_height_m": 0.105,
            "landing_point_source": "white_box_landing_point",
        },
    )


def test_shim_does_not_silently_adjust_z_for_the_raised_platform():
    rec = json.loads(_target_line(_distress_estimate()))
    out = _emit(ShimRouter(), json.dumps(rec))
    assert out.pose.position[2] == pytest.approx(rec["position_flu"][2])


def test_platform_offset_is_exported_as_an_explicit_number():
    """라이다 AGL과 섞을 때 어긋나는 양을 FC가 스스로 정할 수 있어야 한다."""
    out = _emit(ShimRouter(), _target_line(_distress_estimate()))
    kv = _kv(_find(out.statuses, shim_core.STATUS_NAME_TARGET))
    assert kv["plane_reference"] == "mat_top_surface"
    assert json.loads(kv["platform_height_m"]) == pytest.approx(0.105)
    assert json.loads(kv["ground_agl_minus_vision_z_m"]) == pytest.approx(0.105)
    assert kv["landing_point_source"] == "white_box_landing_point"


def test_aruco_record_has_no_platform_offset_key():
    """ArUco 경로엔 매트 평면 개념이 없다 — 없는 키를 지어내지 않는다."""
    out = _emit(ShimRouter(), _target_line())
    kv = _kv(_find(out.statuses, shim_core.STATUS_NAME_TARGET))
    assert "ground_agl_minus_vision_z_m" not in kv
    assert "plane_reference" not in kv


# ──────────────────────────────────────────────────────────────────────────────
# KeyValue 직렬화 / 결정론
# ──────────────────────────────────────────────────────────────────────────────


def test_keyvalues_round_trip_through_json_loads():
    """`str(True)`는 `"True"`가 되어 파이썬 밖에서 파싱 불가다 — json.dumps여야 한다."""
    assert json.loads(shim_core.kv_value(True)) is True
    assert json.loads(shim_core.kv_value(None)) is None
    assert json.loads(shim_core.kv_value(1.5)) == pytest.approx(1.5)
    assert json.loads(shim_core.kv_value([1, 2, 3])) == [1, 2, 3]
    assert shim_core.kv_value("no_target_detection") == "no_target_detection"


def test_all_keyvalues_are_strings():
    """`diagnostic_msgs/KeyValue`는 key/value 둘 다 string이다 — 다른 타입이면 발행이 죽는다."""
    router = ShimRouter(ShimConfig(publish_landing_target=True))
    _push(router, _target_line(_distress_estimate()))
    out = _push(router, _state_hint_line())[0]
    for st in out.statuses:
        for k, v in st.values:
            assert isinstance(k, str) and isinstance(v, str)


def test_same_input_gives_identical_plans():
    line, hint = _target_line(), _state_hint_line()
    a = ShimRouter(); _push(a, line); out_a = _push(a, hint)[0]
    b = ShimRouter(); _push(b, line); out_b = _push(b, hint)[0]
    assert out_a == out_b


def test_blank_and_whitespace_lines_are_rejected_without_pose():
    outs = _push(ShimRouter(), "   \n")
    assert outs[0].pose is None
    assert _find(outs[0].statuses, shim_core.STATUS_NAME_LINK).level == shim_core.LEVEL_ERROR


# ──────────────────────────────────────────────────────────────────────────────
# shim_node.py 소스 원문 회귀 — rclpy 없이 "얇음"을 강제한다
# ──────────────────────────────────────────────────────────────────────────────


def _shim_node_src() -> str:
    return _SHIM_NODE_PATH.read_text(encoding="utf-8")


def _shim_node_ast() -> ast.Module:
    """**소스 문자열이 아니라 AST로 본다.**

    문자열 검색은 docstring/주석의 산문까지 잡아 "설명을 쓰면 red가 되는" 쓸모없는 테스트가
    된다(이 파일들은 함정을 길게 설명하는 것이 관례다). `test_registry.py`가 dict 리터럴
    중복키를 `ast`로 세는 것과 같은 도구 선택이다.
    """
    return ast.parse(_shim_node_src())


#: `mavros_msgs/msg/LandingTarget`의 프레임 상수 이름들. 값이 실제 `MAV_FRAME`과 어긋난다.
_FORBIDDEN_MSG_FRAME_CONSTANTS = (
    "LOCAL_NED", "BODY_NED", "MISSION", "GLOBAL_TERRAIN_ALT_INT", "LOCAL_ENU",
    "BODY_OFFSET_NED", "LOCAL_OFFSET_NED",
)


def test_shim_node_never_reads_a_landing_target_msg_frame_constant():
    """🔴 msg 상수를 쓰면 MAVLink가 다른 프레임으로 읽어 조용히 position_valid=false가 된다."""
    used = set()
    for node in ast.walk(_shim_node_ast()):
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_MSG_FRAME_CONSTANTS:
            used.add(node.attr)
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_MSG_FRAME_CONSTANTS:
            used.add(node.id)
    assert used == set(), f"shim_node.py가 msg 프레임 상수를 참조한다: {sorted(used)}"


def test_shim_node_takes_frame_only_from_the_plan():
    """`lt.frame`의 오른편이 `plan.frame` 경유가 아니면 red — 리터럴을 직접 박아도 잡힌다."""
    assigns = []
    for node in ast.walk(_shim_node_ast()):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Attribute) and tgt.attr == "frame":
                assigns.append(ast.dump(node.value))
    assert len(assigns) == 1, f"lt.frame 대입이 정확히 1곳이어야 한다 — {len(assigns)}곳"
    assert "attr='frame'" in assigns[0] and "id='plan'" in assigns[0]


def test_shim_node_never_reads_wire_keys_directly():
    """레코드 키 해석은 전부 shim_core의 몫이다(얇음 계약) — 그래야 랩탑에서 검증된다."""
    wire_keys = set(shim_core.REQUIRED_TARGET_KEYS) | set(shim_core.REQUIRED_STATE_HINT_KEYS)
    found = set()
    for node in ast.walk(_shim_node_ast()):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if node.slice.value in wire_keys:
                found.add(node.slice.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and arg.value in wire_keys:
                        found.add(arg.value)
    assert found == set(), f"shim_node.py가 와이어 키를 직접 읽는다: {sorted(found)}"


def test_shim_node_installs_its_sigterm_handler_after_rclpy_init():
    """🔴 순서 회귀 — `rclpy.init()`이 자기 SIGTERM 핸들러로 우리 것을 **덮어쓴다.**

    먼저 걸면 SIGTERM에 `stop_event`가 안 서고 종료 경로가
    `RCLError: rcl_shutdown already called`로 터진다(실기체에서 실제로 재현, 2026-07-28).
    이 저장소가 이미 한 번 밟은 "신호당 핸들러는 하나" 함정(`main.py` D-A5)과 같은 것이라
    순서 자체를 고정한다.
    """
    main_fn = next(
        n for n in ast.walk(_shim_node_ast())
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    # main() 본문을 **순서대로** 훑어 두 호출의 등장 위치를 찾는다
    # (ast.walk는 순서를 보장하지 않으므로 body를 직접 순회해야 한다).
    init_idx = handler_idx = None
    for idx, stmt in enumerate(main_fn.body):
        for sub in ast.walk(stmt):
            if not isinstance(sub, ast.Call):
                continue
            fn = sub.func
            if isinstance(fn, ast.Attribute) and fn.attr == "init":
                if isinstance(fn.value, ast.Name) and fn.value.id == "rclpy":
                    init_idx = idx if init_idx is None else init_idx
            if isinstance(fn, ast.Name) and fn.id == "_install_sigterm_handler":
                handler_idx = idx if handler_idx is None else handler_idx
    assert init_idx is not None, "main()에 rclpy.init() 호출이 없다"
    assert handler_idx is not None, "main()에 _install_sigterm_handler() 호출이 없다"
    assert init_idx < handler_idx, (
        f"rclpy.init()(문 {init_idx})이 _install_sigterm_handler()(문 {handler_idx})보다 "
        "뒤에 있다 — rclpy가 우리 SIGTERM 핸들러를 덮어쓴다"
    )


def test_shim_node_guards_rclpy_shutdown():
    """rclpy가 이미 컨텍스트를 내렸을 때 무조건 shutdown()을 부르면 종료 경로가 터진다."""
    assert "if rclpy.ok():" in _shim_node_src()


def test_shim_node_is_the_only_vision_file_importing_rclpy():
    vision_root = pathlib.Path(shim_core.__file__).parents[1]
    offenders = []
    for path in vision_root.rglob("*.py"):
        if path == _SHIM_NODE_PATH:
            continue
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            s = line.strip()
            if s.startswith(("import rclpy", "from rclpy")):
                offenders.append(str(path))
    assert offenders == []


def test_vision_ros_package_init_imports_nothing():
    """`__init__.py`가 shim_node를 import하면 rclpy 없는 환경에서 shim_core까지 죽는다."""
    src = (_SHIM_NODE_PATH.parent / "__init__.py").read_text(encoding="utf-8")
    code = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert code == []


def test_shim_node_imports_are_deferred_enough_to_keep_shim_core_importable():
    """이 테스트 파일 자체가 증거다 — rclpy 없는 랩탑에서 shim_core를 import해 여기까지 왔다."""
    assert "rclpy" not in [m for m in dir(shim_core)]
    import importlib
    assert importlib.util.find_spec("vision.ros.shim_core") is not None
