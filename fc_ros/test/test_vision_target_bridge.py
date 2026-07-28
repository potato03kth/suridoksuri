"""
VisionTargetBridge 순수 로직 테스트.

rclpy/vision 없이 실행 가능한 변환·판정 로직만 검증한다(`test_telemetry_node.py`와
같은 원칙 — ROS2 메시지는 `types.SimpleNamespace` 모의 객체로 주입하고 필드 이름은
실제 msg 정의와 맞춘다). 이 저장소 `.venv`에는 rclpy도 geometry_msgs도 없다.

계약 정본: `docs/fc_precision_land_handoff.md` §2·§4.
"""
import math
import types

import numpy as np
import pytest

from fc_ros.adapters.vision_target_bridge import (
    COMMAND_HINT_LAND,
    DEFAULT_STALE_TIMEOUT_S,
    LEVEL_ERROR,
    LEVEL_OK,
    LEVEL_WARN,
    STATUS_NAME_LINK,
    STATUS_NAME_SETPOINT,
    STATUS_NAME_STATE,
    STATUS_NAME_TARGET,
    VisionTargetBridge,
    enu_to_pos_ned_n_e_hup,
    enu_yaw_to_ned_yaw,
    level_to_int,
)


# ── 헬퍼: ROS2 메시지 모의 객체 ─────────────────────────────────

def _yaw_quat_enu(yaw_enu):
    """ENU z축 yaw-only 쿼터니언 (w, x, y, z)."""
    return math.cos(yaw_enu / 2.0), 0.0, 0.0, math.sin(yaw_enu / 2.0)


def _setpoint_msg(x_enu, y_enu, z_enu, yaw_enu=0.0):
    """geometry_msgs/PoseStamped (ENU, frame_id="map")."""
    w, x, y, z = _yaw_quat_enu(yaw_enu)
    return types.SimpleNamespace(
        header=types.SimpleNamespace(frame_id="map", stamp=None),
        pose=types.SimpleNamespace(
            position=types.SimpleNamespace(x=x_enu, y=y_enu, z=z_enu),
            orientation=types.SimpleNamespace(w=w, x=x, y=y, z=z),
        ),
    )


def _kv(key, value):
    return types.SimpleNamespace(key=key, value=value)


def _status(name, level, message="", values=()):
    """diagnostic_msgs/DiagnosticStatus.

    level 기본은 int로 준다. rclpy 실물은 `byte`(1바이트 bytes)를 주므로 그 경로는
    `test_level_accepts_rclpy_byte_field`가 따로 본다.
    """
    return types.SimpleNamespace(
        level=level, name=name, message=message,
        hardware_id="vision", values=list(values),
    )


def _status_array(*statuses):
    return types.SimpleNamespace(
        header=types.SimpleNamespace(frame_id="base_link", stamp=None),
        status=list(statuses),
    )


def _target_status(level=LEVEL_OK):
    return _status(STATUS_NAME_TARGET, level, "ok", [_kv("valid", "true")])


def _state_status(state="PRECISION_SERVO", level=LEVEL_OK, command_hint="descend"):
    return _status(
        STATUS_NAME_STATE, level, f"state={state}",
        [_kv("state", state),
         _kv("command_hint", command_hint),
         _kv("command_is_advisory", "true")],
    )


def _link_status(level=LEVEL_OK, event="connected"):
    return _status(STATUS_NAME_LINK, level, event, [_kv("event", event)])


def _nominal_array(state="PRECISION_SERVO", state_level=LEVEL_OK,
                   target_level=LEVEL_OK, command_hint="descend"):
    """정상 프레임 한 묶음 — target + setpoint + state 를 같은 stamp로."""
    return _status_array(
        _target_status(target_level),
        _status(STATUS_NAME_SETPOINT, LEVEL_OK, "ok"),
        _state_status(state, state_level, command_hint),
    )


# ══════════════════════════════════════════════════════════════════
# 1. ENU → [N, E, h_up] 변환 (축이 실제로 바뀌는가)
# ══════════════════════════════════════════════════════════════════
# 🔴 x != y 인 값으로만 테스트한다. x == y 면 항등변환과 구분이 안 돼
# 스왑을 제거해도 green으로 통과한다.

def test_enu_to_pos_ned_swaps_north_and_east():
    pos = enu_to_pos_ned_n_e_hup(3.0, -7.0, 12.5)   # x_enu=3(동), y_enu=-7(북)
    assert pos[0] == pytest.approx(-7.0)            # N ← y_enu
    assert pos[1] == pytest.approx(3.0)             # E ← x_enu
    assert pos[2] == pytest.approx(12.5)            # h_up ← z_enu


def test_setpoint_message_conversion_swaps_axes():
    bridge = VisionTargetBridge()
    bridge.update_from_setpoint(_setpoint_msg(3.0, -7.0, 12.5), now_s=100.0)
    vt = bridge.snapshot(100.0)
    assert vt.valid is True
    assert vt.pos_ned == pytest.approx(np.array([-7.0, 3.0, 12.5]))


def test_third_component_is_h_up_not_down():
    """🔴 `pos_ned[2]`는 h_up(위 양수)이지 D(아래 양수)가 아니다.

    같은 저장소의 `vel_ned`는 3번째가 아래 양수라 부호를 뒤집는 실수가 실제로 나온다.
    """
    bridge = VisionTargetBridge()
    bridge.update_from_setpoint(_setpoint_msg(0.0, 0.0, 8.0), now_s=0.0)
    assert bridge.snapshot(0.0).pos_ned[2] == pytest.approx(+8.0)


def test_snapshot_returns_copy_not_internal_array():
    bridge = VisionTargetBridge()
    bridge.update_from_setpoint(_setpoint_msg(1.0, 2.0, 3.0), now_s=0.0)
    vt = bridge.snapshot(0.0)
    vt.pos_ned[0] = 999.0
    assert bridge.snapshot(0.0).pos_ned[0] == pytest.approx(2.0)


# ══════════════════════════════════════════════════════════════════
# 2. yaw 디코드 (알려진 방향쌍 — 한 쌍만 쓰면 부호 뒤집힘을 못 잡는다)
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("yaw_enu, yaw_ned", [
    (0.0,             math.pi / 2),    # ENU 0 = 동쪽 → NED +90° = 동쪽
    (math.pi / 2,     0.0),            # ENU 90° = 북쪽 → NED 0 = 북쪽
    (math.pi,        -math.pi / 2),    # ENU 180° = 서쪽 → NED -90° = 서쪽
    (-math.pi / 2,    math.pi),        # ENU -90° = 남쪽 → NED ±180° = 남쪽
])
def test_enu_yaw_to_ned_yaw_known_directions(yaw_enu, yaw_ned):
    got = enu_yaw_to_ned_yaw(yaw_enu)
    assert math.cos(got - yaw_ned) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("yaw_enu, yaw_ned", [
    (0.0,         math.pi / 2),
    (math.pi / 2, 0.0),
    (math.pi,    -math.pi / 2),
])
def test_setpoint_yaw_decode_known_directions(yaw_enu, yaw_ned):
    """쿼터니언 → NED yaw 경로 전체(단위 쿼터니언이 '동쪽'이라는 사실 포함)."""
    bridge = VisionTargetBridge()
    bridge.update_from_setpoint(_setpoint_msg(0.0, 0.0, 5.0, yaw_enu), now_s=0.0)
    got = bridge.snapshot(0.0).yaw_ned
    assert math.cos(got - yaw_ned) == pytest.approx(1.0, abs=1e-9)


def test_identity_quaternion_means_east_not_unknown():
    """🔴 ENU 단위 쿼터니언은 '모름'이 아니라 '동쪽을 보라'다(flight04 yaw 스핀)."""
    bridge = VisionTargetBridge()
    bridge.update_from_setpoint(_setpoint_msg(0.0, 0.0, 5.0, yaw_enu=0.0), now_s=0.0)
    assert bridge.snapshot(0.0).yaw_ned == pytest.approx(math.pi / 2, abs=1e-9)


# ══════════════════════════════════════════════════════════════════
# 3. stale 만료 — valid=False **그리고** pos_ned is None
# ══════════════════════════════════════════════════════════════════

def test_stale_setpoint_invalidates_and_drops_coordinates():
    bridge = VisionTargetBridge(stale_timeout_s=1.0)
    bridge.update_from_setpoint(_setpoint_msg(3.0, -7.0, 12.5), now_s=10.0)

    fresh = bridge.snapshot(10.5)
    assert fresh.valid is True and fresh.pos_ned is not None

    stale = bridge.snapshot(11.6)          # age 1.6s > 1.0s
    assert stale.valid is False
    # 🔴 옛 좌표를 붙들고 있으면 소비자가 실수로 쓴다.
    assert stale.pos_ned is None
    assert stale.yaw_ned is None
    assert stale.age_s == pytest.approx(1.6)


def test_stale_boundary_is_inclusive():
    bridge = VisionTargetBridge(stale_timeout_s=1.0)
    bridge.update_from_setpoint(_setpoint_msg(1.0, 2.0, 3.0), now_s=0.0)
    assert bridge.snapshot(1.0).valid is True
    assert bridge.snapshot(1.0001).valid is False


def test_default_stale_timeout_matches_measured_4_4hz():
    """🔴 실측 4.4Hz(p95 0.310s) 근거. 0.5s로 조이면 헛경보가 난다(핸드오프 §7)."""
    assert DEFAULT_STALE_TIMEOUT_S == 1.0
    assert VisionTargetBridge().stale_timeout_s == 1.0
    # p95 0.310s 대비 3배 이상의 여유가 있어야 정상 지터로는 안 뜬다.
    assert DEFAULT_STALE_TIMEOUT_S >= 3.0 * 0.310


def test_never_received_reports_no_coordinates_and_infinite_age():
    """🔴 값이 없을 때 0을 채우지 않는다 — setpoint 자리의 0이 가장 위험한 거짓말."""
    vt = VisionTargetBridge().snapshot(123.0)
    assert vt.valid is False
    assert vt.pos_ned is None
    assert vt.yaw_ned is None
    assert vt.age_s == math.inf
    assert vt.status_age_s == math.inf


def test_non_finite_setpoint_is_rejected_not_stored():
    bridge = VisionTargetBridge()
    bridge.update_from_setpoint(_setpoint_msg(float("nan"), 1.0, 2.0), now_s=0.0)
    vt = bridge.snapshot(0.0)
    assert vt.valid is False and vt.pos_ned is None
    assert bridge.rejected_setpoints == 1
    assert bridge.setpoints_rx == 0


# ══════════════════════════════════════════════════════════════════
# 4. 거부권 — `vision/state` WARN → veto
# ══════════════════════════════════════════════════════════════════

def test_state_warn_is_veto():
    bridge = VisionTargetBridge()
    bridge.update_from_status(
        _nominal_array(state="HOLD", state_level=LEVEL_WARN, command_hint="hold"),
        now_s=0.0)
    vt = bridge.snapshot(0.0)
    assert vt.veto is True
    assert vt.state == "HOLD"


def test_state_ok_is_not_veto():
    bridge = VisionTargetBridge()
    bridge.update_from_status(_nominal_array(state="PRECISION_SERVO"), now_s=0.0)
    vt = bridge.snapshot(0.0)
    assert vt.veto is False
    assert vt.state == "PRECISION_SERVO"


def test_abort_ascend_is_veto():
    bridge = VisionTargetBridge()
    bridge.update_from_status(
        _nominal_array(state="ABORT_ASCEND", state_level=LEVEL_WARN,
                       command_hint="ascend"),
        now_s=0.0)
    assert bridge.snapshot(0.0).veto is True


def test_link_error_is_not_mistaken_for_state_veto():
    """`vision/link` ERROR(=2)는 거부권 채널이 아니다 — level 비교가 name별로 갈리는지."""
    bridge = VisionTargetBridge()
    bridge.update_from_status(
        _status_array(_state_status("LOCK", LEVEL_OK), _link_status(LEVEL_ERROR)),
        now_s=0.0)
    vt = bridge.snapshot(0.0)
    assert vt.veto is False
    assert vt.link_dead is True


def test_expired_state_reports_unknown_not_stale_string():
    bridge = VisionTargetBridge(stale_timeout_s=1.0)
    bridge.update_from_status(_nominal_array(state="LOCK"), now_s=0.0)
    vt = bridge.snapshot(2.0)
    assert vt.state is None
    assert vt.veto is False           # "거부 신호 없음"이지 "허가"가 아니다
    assert vt.status_age_s == pytest.approx(2.0)


# ══════════════════════════════════════════════════════════════════
# 5. command_hint는 advisory — 명령이 아니다
# ══════════════════════════════════════════════════════════════════

def test_land_hint_is_exposed_but_not_veto():
    bridge = VisionTargetBridge()
    bridge.update_from_status(
        _nominal_array(state="TERMINAL", state_level=LEVEL_OK,
                       command_hint=COMMAND_HINT_LAND),
        now_s=0.0)
    vt = bridge.snapshot(0.0)
    assert vt.land_handoff_hint is True
    assert vt.veto is False           # 🔴 힌트는 거부권을 만들지 않는다
    assert vt.state == "TERMINAL"


def test_non_land_hint_is_false():
    bridge = VisionTargetBridge()
    bridge.update_from_status(_nominal_array(command_hint="descend"), now_s=0.0)
    assert bridge.snapshot(0.0).land_handoff_hint is False


def test_null_keyvalue_decodes_to_none():
    """shim은 문자열이 아닌 값을 json으로 눌러 싣는다 — `None`은 `"null"`로 온다."""
    bridge = VisionTargetBridge()
    bridge.update_from_status(
        _status_array(_status(STATUS_NAME_STATE, LEVEL_WARN, "state_hint_missing",
                              [_kv("state", "null"), _kv("frame_id", "7")])),
        now_s=0.0)
    vt = bridge.snapshot(0.0)
    assert vt.state is None
    assert vt.veto is True            # level은 여전히 WARN


# ══════════════════════════════════════════════════════════════════
# 6. `vision/link` ERROR → link_dead
# ══════════════════════════════════════════════════════════════════

def test_link_error_marks_producer_dead():
    bridge = VisionTargetBridge()
    bridge.update_from_status(
        _status_array(_link_status(LEVEL_ERROR, "producer_eof")), now_s=0.0)
    assert bridge.snapshot(0.0).link_dead is True


def test_link_ok_clears_producer_dead():
    bridge = VisionTargetBridge()
    bridge.update_from_status(
        _status_array(_link_status(LEVEL_ERROR, "producer_eof")), now_s=0.0)
    bridge.update_from_status(
        _status_array(_link_status(LEVEL_OK, "connected")), now_s=0.5)
    assert bridge.snapshot(0.5).link_dead is False


def test_one_off_link_error_does_not_latch_forever():
    """깨진 줄 한 건(`ShimRouter._reject`)이 유도를 영구히 막으면 안 된다.

    진짜 사망은 1Hz 하트비트로 계속 갱신되므로 창 안에 항상 들어온다.
    """
    bridge = VisionTargetBridge(link_timeout_s=3.0)
    bridge.update_from_status(
        _status_array(_link_status(LEVEL_ERROR, "malformed_line")), now_s=0.0)
    assert bridge.snapshot(2.9).link_dead is True
    assert bridge.snapshot(3.1).link_dead is False


def test_repeated_error_heartbeat_keeps_link_dead():
    bridge = VisionTargetBridge(link_timeout_s=3.0)
    for t in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        bridge.update_from_status(
            _status_array(_link_status(LEVEL_ERROR, "disconnected")), now_s=t)
        assert bridge.snapshot(t).link_dead is True
    assert bridge.snapshot(5.5).link_dead is True


# ══════════════════════════════════════════════════════════════════
# 7. 🔴 페일세이프 3분법 — "안 보임" / "생산자 사망" / "shim 사망"이 구분되는가
# ══════════════════════════════════════════════════════════════════

def test_target_not_seen_is_distinguishable_from_death():
    """안 보임: setpoint 침묵 + status는 계속 신선(`vision/target` WARN)."""
    bridge = VisionTargetBridge(stale_timeout_s=1.0)
    # 처음엔 정상적으로 보였다.
    bridge.update_from_setpoint(_setpoint_msg(1.0, 2.0, 3.0), now_s=0.0)
    bridge.update_from_status(_nominal_array(), now_s=0.0)
    assert bridge.snapshot(0.0).target_seen is True

    # 이후 타겟만 놓쳤다 — status는 계속 온다.
    for t in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        bridge.update_from_status(
            _status_array(
                _target_status(LEVEL_WARN),
                _status(STATUS_NAME_SETPOINT, LEVEL_WARN, "no_target"),
                _state_status("CENTER_DESCEND", LEVEL_OK, "center"),
            ),
            now_s=t)

    vt = bridge.snapshot(1.5)
    assert vt.valid is False           # setpoint는 침묵
    assert vt.target_seen is False     # 사유: 안 보임
    assert vt.link_dead is False       # 생산자는 살아 있다
    assert vt.status_age_s == pytest.approx(0.0)   # shim도 살아 있다
    assert vt.state == "CENTER_DESCEND"


def test_producer_death_is_distinguishable_from_shim_death():
    """생산자 사망: setpoint 침묵 + `vision/link` ERROR 하트비트만 1Hz로 계속."""
    bridge = VisionTargetBridge(stale_timeout_s=1.0, link_timeout_s=3.0)
    bridge.update_from_setpoint(_setpoint_msg(1.0, 2.0, 3.0), now_s=0.0)
    bridge.update_from_status(_nominal_array(), now_s=0.0)

    for t in (1.0, 2.0, 3.0):
        bridge.update_from_status(
            _status_array(_link_status(LEVEL_ERROR, "disconnected")), now_s=t)

    dead_producer = bridge.snapshot(3.0)
    assert dead_producer.valid is False
    assert dead_producer.link_dead is True             # ← 여기서 사망이 드러난다
    assert dead_producer.status_age_s == pytest.approx(0.0)   # shim은 살아 있다
    # 🔴 하트비트에는 `vision/target`이 실리지 않으므로 옛 OK가 남아 있으면 안 된다.
    assert dead_producer.target_seen is False

    # shim 사망: status 자체가 멈춘다 → status_age_s가 계속 자란다.
    shim_dead = bridge.snapshot(10.0)
    assert shim_dead.valid is False
    assert shim_dead.status_age_s == pytest.approx(7.0)
    assert shim_dead.link_dead is False   # 더 이상 아무 신호도 없다(= 모른다)


def test_total_silence_from_the_start_is_not_confused_with_target_lost():
    """둘 다 안 옴(= shim 사망/미기동): status_age_s가 무한."""
    silent = VisionTargetBridge().snapshot(5.0)
    assert silent.valid is False
    assert silent.status_age_s == math.inf
    assert silent.target_seen is False
    assert silent.link_dead is False

    lost = VisionTargetBridge()
    lost.update_from_status(_status_array(_target_status(LEVEL_WARN)), now_s=5.0)
    lost_vt = lost.snapshot(5.0)
    assert lost_vt.valid is False
    assert lost_vt.status_age_s == pytest.approx(0.0)   # ← 여기서 갈린다
    assert lost_vt.target_seen is False


# ══════════════════════════════════════════════════════════════════
# 8. 부분 DiagnosticArray / rclpy byte level / 진단
# ══════════════════════════════════════════════════════════════════

def test_partial_array_does_not_wipe_other_entries():
    """하트비트(link 단독)가 방금 받은 state를 지우면 안 된다."""
    bridge = VisionTargetBridge(stale_timeout_s=1.0)
    bridge.update_from_status(_nominal_array(state="LOCK"), now_s=0.0)
    bridge.update_from_status(
        _status_array(_link_status(LEVEL_ERROR, "disconnected")), now_s=0.5)
    vt = bridge.snapshot(0.6)
    assert vt.state == "LOCK"
    assert vt.link_dead is True


def test_level_accepts_rclpy_byte_field():
    """🔴 `DiagnosticStatus.level`은 msg상 `byte`라 rclpy가 1바이트 bytes를 준다.

    `level == 1` 로 직접 비교하면 항상 False라 거부권이 조용히 사라진다.
    """
    assert level_to_int(bytes([LEVEL_WARN])) == LEVEL_WARN
    assert level_to_int(LEVEL_WARN) == LEVEL_WARN

    bridge = VisionTargetBridge()
    bridge.update_from_status(
        _status_array(_status(STATUS_NAME_STATE, bytes([LEVEL_WARN]), "state=HOLD",
                              [_kv("state", "HOLD")])),
        now_s=0.0)
    assert bridge.snapshot(0.0).veto is True


def test_setpoint_reason_is_exposed_for_diagnosis_only():
    bridge = VisionTargetBridge()
    bridge.update_from_status(
        _status_array(_status(STATUS_NAME_SETPOINT, LEVEL_WARN, "attitude_stale")),
        now_s=0.0)
    assert bridge.setpoint_reason == "attitude_stale"
    # 진단 문자열이 판정을 바꾸지 않는다.
    assert bridge.snapshot(0.0).valid is False


def test_reset_clears_state_but_keeps_counters():
    bridge = VisionTargetBridge()
    bridge.update_from_setpoint(_setpoint_msg(1.0, 2.0, 3.0), now_s=0.0)
    bridge.update_from_status(_nominal_array(), now_s=0.0)
    bridge.reset()
    vt = bridge.snapshot(0.0)
    assert vt.pos_ned is None and vt.state is None
    assert vt.age_s == math.inf and vt.status_age_s == math.inf
    assert bridge.setpoints_rx == 1 and bridge.statuses_rx == 1


def test_counters_track_receptions():
    bridge = VisionTargetBridge()
    for t in (0.0, 0.2, 0.4):
        bridge.update_from_setpoint(_setpoint_msg(1.0, 2.0, 3.0), now_s=t)
        bridge.update_from_status(_nominal_array(), now_s=t)
    assert bridge.setpoints_rx == 3
    assert bridge.statuses_rx == 3
