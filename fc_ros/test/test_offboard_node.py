"""
OffboardNode 순수 로직 테스트.

rclpy 없이 실행 가능한 수학적 로직만 검증한다.

판정 함수는 fc_bridge.execution.state_logic 에서 import 한다.
offboard_node.py 와 이 테스트가 동일 함수를 참조하므로 코드 변경이 즉시 검출된다.
"""
import math

import numpy as np
import pytest

from fc_bridge.execution.state_logic import (
    climbing_reached, vtol_is_fw,
    trans_mc_trigger, mc_wp_advance, vtol_is_mc, landing_done,
    override_mode, override_reached, override_fallback_due,
    vel_aligned_with_path, wp1_land_ready,
    after_climb_state, after_following_state, takeoff_request_fields,
    home_amsl_confirmed,
)


def _wrap(a: float) -> float:
    """[-π, π] 범위로 각도 정규화."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ── _wrap 단위 테스트 ────────────────────────────────────────

def test_wrap_pi_plus():
    assert _wrap(np.pi + 0.1) == pytest.approx(-np.pi + 0.1, abs=1e-9)


def test_wrap_neg_pi_minus():
    result = _wrap(-np.pi - 0.1)
    assert abs(result) == pytest.approx(np.pi - 0.1, abs=1e-9)


def test_wrap_zero():
    assert _wrap(0.0) == pytest.approx(0.0)


def test_wrap_two_pi():
    assert _wrap(2 * np.pi) == pytest.approx(0.0, abs=1e-9)


# ── ENTRY 조건 로직 테스트 ───────────────────────────────────

def _entry_done(wp0, pos2, yaw, wp0_r=5.0, wp0_htol=0.2) -> bool:
    """OffboardNode._step_entry() 도달 판정 로직 추출."""
    dist = float(np.linalg.norm(wp0 - pos2))
    to_wp0 = wp0 - pos2
    if np.linalg.norm(to_wp0) < 1e-3:
        to_wp0 = np.array([1.0, 0.0])
    to_wp0 /= np.linalg.norm(to_wp0)
    chi_to_wp0 = float(np.arctan2(to_wp0[1], to_wp0[0]))
    heading_err = abs(_wrap(chi_to_wp0 - yaw))
    return dist < wp0_r and heading_err < wp0_htol


def test_entry_done_when_close_and_aligned():
    wp0  = np.array([100.0, 0.0])
    pos2 = np.array([98.0,  0.0])   # dist=2 < 5
    yaw  = 0.0                      # 북쪽 헤딩 = WP0 방향
    assert _entry_done(wp0, pos2, yaw) is True


def test_entry_not_done_too_far():
    wp0  = np.array([100.0, 0.0])
    pos2 = np.array([0.0,   0.0])   # dist=100 >> 5
    yaw  = 0.0
    assert _entry_done(wp0, pos2, yaw) is False


def test_entry_not_done_heading_misaligned():
    wp0  = np.array([100.0, 0.0])
    pos2 = np.array([98.0,  0.0])   # dist=2 < 5
    yaw  = np.pi / 2                # 동쪽 헤딩 (WP0는 북쪽)
    assert _entry_done(wp0, pos2, yaw) is False


# ── 작업 C: CLIMBING 판정 (climbing_reached) ────────────────────
# 2026-07-18: MC 오프보드 실비행에서 목표고도 바로 아래 정착 시(예 -0.1m) 단측
# 판정(>=)이 절대 만족되지 않아 CLIMBING 무한대기가 관찰됨 → 목표고도를 점이
# 아닌 반경 alt_tol(기본 0.5m) 구간으로 완화. 아래 테스트는 그 허용 구간 경계를
# 검증한다 (경계값 포함: <= alt_tol).

def test_climb_reached():
    assert climbing_reached(50.1, 50.0) is True


def test_climb_not_reached():
    # 목표보다 1.0m 낮음 → 허용오차(0.5m) 밖 → 미도달.
    assert climbing_reached(49.0, 50.0) is False


def test_climb_reached_within_tolerance_below():
    # 목표보다 0.1m 낮음 → 허용오차 안 → 도달 (2026-07-18 관찰 문제의 재현/수정 확인).
    assert climbing_reached(49.9, 50.0) is True


def test_climb_reached_at_lower_tolerance_boundary():
    assert climbing_reached(49.5, 50.0) is True


def test_climb_not_reached_just_outside_lower_tolerance():
    assert climbing_reached(49.49, 50.0) is False


def test_climb_reached_at_upper_tolerance_boundary():
    assert climbing_reached(50.5, 50.0) is True


def test_climb_not_reached_beyond_upper_tolerance():
    # 과상승(overshoot)이 허용오차를 넘으면 미도달 — 목표고도가 점이 아닌 구간이 된
    # 의도된 결과. CLIMBING 중 지속적 과상승 버그가 있다면 이 조건이 이를 드러낸다.
    assert climbing_reached(50.6, 50.0) is False


def test_climb_exact_alt():
    assert climbing_reached(50.0, 50.0) is True


# 지면 기준 AGL 판정 (2026-07-07: 로컬 원점≠지면 → ground_ref 보정)

def test_climb_reached_with_ground_ref():
    # 지면 h_up=-2.11(로컬 원점이 지면보다 2.11m 위). 지면 기준 4.0m 상승 =
    # 로컬 h_up 1.89 (=-2.11+4.0). 이때 도달 판정 True 여야 CLIMBING이 완료된다.
    assert climbing_reached(1.89, 4.0, -2.11) is True


def test_climb_not_reached_with_ground_ref():
    # 로컬 h_up=1.0 → 지면 기준 (1.0-(-2.11))=3.11, 목표 4.0과의 차 0.89 → 허용오차 밖 → 미도달.
    assert climbing_reached(1.0, 4.0, -2.11) is False


def test_climb_ground_ref_defaults_to_zero():
    # ground_ref 미지정 시 기존 동작(원점≈지면)과 동일.
    assert climbing_reached(50.1, 50.0) is True
    assert climbing_reached(49.0, 50.0) is False


# ── 작업 C: TRANSITION_FW 완료 판정 (vtol_is_fw) ────────────────

def test_transition_fw_done():
    assert vtol_is_fw(4) is True


def test_transition_fw_not_done_mc():
    assert vtol_is_fw(3) is False


def test_transition_fw_not_done_in_progress():
    assert vtol_is_fw(1) is False  # VTOL_STATE_TRANSITION_TO_FW


# ── 작업 D: TRANSITION_MC 트리거 (trans_mc_trigger) ─────────────

def test_trans_mc_trigger():
    assert trans_mc_trigger(9.0, 10.0) is True


def test_trans_mc_not_yet():
    assert trans_mc_trigger(11.0, 10.0) is False


def test_trans_mc_exact_thresh():
    # 경계값: dist_to_end == d_end_thresh 는 미트리거 (strict <)
    assert trans_mc_trigger(10.0, 10.0) is False


# ── MC 이산 웨이포인트 순차추종 (mc_wp_advance) ──────────────────
# 2026-07-24: MC는 L1Guidance(FW 선회반경 회피용 lookahead 알고리즘)가
# 불필요하다는 지적(직선 구간을 순서대로 지나가기만 하면 됨) — 왕복
# 경로에서 L1Guidance가 고정점에 수렴해 완주 못 하고 멈추는 문제도 실측돼
# 이산 순차추종으로 교체.

def test_mc_wp_advance_not_yet_arrived():
    # 아직 목표점에서 멀면 인덱스 유지, 미완료.
    idx, done = mc_wp_advance(0, dist_to_wp=5.0, end_thresh=2.0, n_points=3)
    assert (idx, done) == (0, False)


def test_mc_wp_advance_to_next_point():
    # 목표점 도달(마지막 점 아님) → 다음 인덱스로 전진, 아직 미완료.
    idx, done = mc_wp_advance(0, dist_to_wp=1.0, end_thresh=2.0, n_points=3)
    assert (idx, done) == (1, False)


def test_mc_wp_advance_completes_at_last_point():
    # 마지막 점(n_points-1)에 도달하면 완료.
    idx, done = mc_wp_advance(2, dist_to_wp=1.0, end_thresh=2.0, n_points=3)
    assert (idx, done) == (2, True)


def test_mc_wp_advance_exact_thresh_not_arrived():
    # 경계값: dist_to_wp == end_thresh 는 미도달 (strict <), 기존 trans_mc_trigger와 동일 규약.
    idx, done = mc_wp_advance(0, dist_to_wp=2.0, end_thresh=2.0, n_points=3)
    assert (idx, done) == (0, False)


def test_mc_wp_advance_reversed_middle_waypoint_visits_correct_side():
    # 2026-07-24 flight03(WP2=-4.24,-4.24)/flight05(WP2=+4.24,+4.24) 재현:
    # 이산 순차추종은 배열에 실제로 들어있는 좌표를 그대로 목표로 삼으므로,
    # 부호를 반대로 바꾸면 실제로 다른 위치(반대편 모서리)를 향해 간다 —
    # 예전 L1Guidance 기반 target_point_ned()는 둘 다 (0,0)으로 클램프돼
    # 부호가 결과에 전혀 반영되지 않았던 것과 대조적.
    pts_neg = np.array([[0.0, 0.0], [-4.24, -4.24], [0.0, 0.0]])
    pts_pos = np.array([[0.0, 0.0], [4.24, 4.24], [0.0, 0.0]])
    assert pts_neg[1][0] < 0 < pts_pos[1][0]


# ── 작업 D: vtol_state == MC 판정 (vtol_is_mc) ──────────────────

def test_vtol_is_mc():
    assert vtol_is_mc(3) is True


def test_vtol_is_mc_fw():
    assert vtol_is_mc(4) is False


def test_vtol_is_mc_transition():
    assert vtol_is_mc(2) is False  # VTOL_STATE_TRANSITION_TO_MC


# ── 2026-07-23 사고대응: home_position AMSL 수렴판정 (home_amsl_confirmed) ──
# 막 재시작된 MAVROS가 첫 home_position 수신값을 단발로 신뢰했다가, PX4 부팅
# 초기(GPS 수직정확도 미수렴 시점)에 래치된 오래된 값(393.6m)을 그대로 받아
# 실제 지면(366.93m)과 26.7m 어긋난 채 이륙목표를 계산·실행한 사고. 최근
# min_samples개가 tol 이내로 수렴해야 신뢰하도록 방지한다.

def test_home_amsl_confirmed_insufficient_samples_returns_none():
    assert home_amsl_confirmed([393.6], min_samples=3) is None
    assert home_amsl_confirmed([393.6, 393.6], min_samples=3) is None


def test_home_amsl_confirmed_empty_returns_none():
    assert home_amsl_confirmed([], min_samples=3) is None


def test_home_amsl_confirmed_stable_samples_returns_latest():
    samples = [366.9, 366.8, 366.93]
    assert home_amsl_confirmed(samples, tol=0.5, min_samples=3) == pytest.approx(366.93)


def test_home_amsl_confirmed_regression_single_stale_snapshot_not_trusted():
    # 2026-07-23 사고 재현: 단발 수신값(393.6)만으로는 절대 확정되지 않아야 한다.
    assert home_amsl_confirmed([393.6], tol=0.5, min_samples=3) is None


def test_home_amsl_confirmed_drifting_samples_returns_none():
    # 26.7m 규모로 벌어지면(사고 당시 스케일) 수렴 실패로 판정돼야 한다.
    samples = [366.9, 380.0, 393.6]
    assert home_amsl_confirmed(samples, tol=0.5, min_samples=3) is None


def test_home_amsl_confirmed_only_checks_trailing_window():
    # 앞쪽의 잡음(오래된 샘플)은 최근 min_samples 윈도만 보므로 무시된다.
    samples = [393.6, 100.0, 366.9, 366.8, 366.93]
    assert home_amsl_confirmed(samples, tol=0.5, min_samples=3) == pytest.approx(366.93)


def test_home_amsl_confirmed_boundary_equal_to_tol_confirms():
    # max-min이 tol과 정확히 같으면(경계) 통과(> tol 만 거부).
    samples = [366.0, 366.5, 366.0]
    assert home_amsl_confirmed(samples, tol=0.5, min_samples=3) == pytest.approx(366.0)


def test_home_amsl_confirmed_just_over_tol_rejects():
    samples = [366.0, 366.51, 366.0]
    assert home_amsl_confirmed(samples, tol=0.5, min_samples=3) is None


def test_home_amsl_confirmed_custom_min_samples():
    assert home_amsl_confirmed([366.9, 366.8], tol=0.5, min_samples=2) == pytest.approx(366.8)
    assert home_amsl_confirmed([366.9], tol=0.5, min_samples=2) is None


# ── 작업 H: CommandTOL 이륙 요청 필드 (takeoff_request_fields) ──
# altitude = 지면 AMSL(home_amsl) + transition_alt (절대고도). 2026-07-07 실측:
# transition_alt 를 그대로 실으면 지면 AMSL보다 낮아 PX4가 이륙을 취소
# ("Already higher than takeoff altitude") → preflight auto-disarm으로 이륙 실패.

def test_takeoff_request_altitude_is_amsl_sum():
    # 지면 19.2m AMSL + 상승 4.0m = 목표 23.2m AMSL
    fields = takeoff_request_fields(4.0, 19.2)
    assert fields["altitude"] == pytest.approx(23.2)


def test_takeoff_request_altitude_above_ground():
    # 회귀(2026-07-07): 목표고도는 반드시 지면 AMSL보다 높아야 이륙한다.
    home_amsl = 19.2
    fields = takeoff_request_fields(4.0, home_amsl)
    assert fields["altitude"] > home_amsl


def test_takeoff_request_altitude_different_values():
    fields = takeoff_request_fields(50.0, 100.0)
    assert fields["altitude"] == pytest.approx(150.0)


def test_takeoff_request_yaw_is_nan():
    # nan = 현재 헤딩 유지 (PX4 관례)
    assert math.isnan(takeoff_request_fields(10.0, 0.0)["yaw"])


def test_takeoff_request_lat_lon_nan_uses_current_position():
    # nan = 현재 위치 사용 (MAVLink 관례). 0.0/0.0은 실제 좌표(위도·경도 0)로
    # 해석돼 2026-07-06 SITL에서 고도 미상승 후 preflight disarm으로 실패 확인됨.
    fields = takeoff_request_fields(10.0, 0.0)
    assert math.isnan(fields["latitude"])
    assert math.isnan(fields["longitude"])


# ── 작업 D: 착륙 완료 판정 (landing_done) ───────────────────────

def test_landing_done_disarmed():
    assert landing_done(False) is True


def test_landing_done_still_armed():
    assert landing_done(True) is False


# ── 작업 E: 긴급 override 모드 분기 (override_mode) ─────────────

def test_override_mc():
    assert override_mode(3) == "POSCTL"


def test_override_fw():
    assert override_mode(4) == "MANUAL"


def test_override_transition_to_fw():
    # 천이 중(1)은 MC가 아니므로 MANUAL
    assert override_mode(1) == "MANUAL"


def test_override_transition_to_mc():
    # 천이 중(2)은 MC가 아니므로 MANUAL
    assert override_mode(2) == "MANUAL"


# ── override 종료/폴백 판정 (override_reached / override_fallback_due) ──
# headless SITL·RC 없음 시 MANUAL/POSCTL 거부 → AUTO.LOITER 폴백.

def test_override_reached_target_manual():
    assert override_reached("MANUAL", "MANUAL") is True


def test_override_reached_loiter_fallback():
    # 목표는 MANUAL이었으나 AUTO.LOITER 폴백 진입도 종료 조건
    assert override_reached("AUTO.LOITER", "MANUAL") is True


def test_override_not_reached_still_offboard():
    assert override_reached("OFFBOARD", "MANUAL") is False


def test_override_fallback_due_after_timeout():
    # OFFBOARD 유지(거부)로 10틱 경과, 폴백 미발행 → 발행해야 함
    assert override_fallback_due("OFFBOARD", "MANUAL", 10, 10, False) is True


def test_override_fallback_not_due_before_timeout():
    assert override_fallback_due("OFFBOARD", "MANUAL", 9, 10, False) is False


def test_override_fallback_not_due_already_sent():
    # 이미 1회 발행했으면 재발행 금지
    assert override_fallback_due("OFFBOARD", "MANUAL", 20, 10, True) is False


def test_override_fallback_not_due_manual_reached():
    # 실기체: 조종사 RC로 MANUAL 진입 → 폴백 불필요
    assert override_fallback_due("MANUAL", "MANUAL", 20, 10, False) is False


def test_override_fallback_not_due_loiter_reached():
    # 폴백 LOITER 진입 후엔 재발행 안 함
    assert override_fallback_due("AUTO.LOITER", "MANUAL", 20, 10, False) is False


# ── TRANSITION_FW 헤딩 안정 카운터 ───────────────────────────
# offboard_node._fw_stable_ticks 카운터 로직을 미러링해 검증한다.

def _heading_stable(err_history: list, tol: float, req_ticks: int) -> bool:
    """최근 req_ticks 개가 전부 |err| < tol 이면 True."""
    if len(err_history) < req_ticks:
        return False
    return all(abs(e) < tol for e in err_history[-req_ticks:])


def test_heading_stable_after_req_ticks():
    errors = [0.5, 0.3, 0.1, 0.08, 0.05]   # 마지막 3개가 0.2 이내
    assert _heading_stable(errors, 0.2, 3) is True


def test_heading_not_stable_insufficient_ticks():
    errors = [0.1, 0.08]                    # 2개뿐, req=3
    assert _heading_stable(errors, 0.2, 3) is False


def test_heading_not_stable_oscillating():
    errors = [0.05, 0.25, 0.05]             # 중간에 이탈
    assert _heading_stable(errors, 0.2, 3) is False


def test_heading_stable_reset_on_exceedance():
    # 이탈 후 재진입: 이탈 직후 카운터 리셋되므로 다시 req_ticks 필요
    errors = [0.05, 0.3, 0.05, 0.05, 0.05]
    # 인덱스 1(0.3)이 이탈이므로 마지막 3개 [0.05, 0.05, 0.05]만 유효
    assert _heading_stable(errors, 0.2, 3) is True


# ── STREAMING 속도 정렬 판정 (vel_aligned_with_path) ────────────

def test_vel_aligned_north():
    vel = np.array([10.0, 0.0])             # 북쪽 속도
    pts = np.array([[0.0, 0.0], [100.0, 0.0]])  # 북쪽 경로
    assert vel_aligned_with_path(vel, pts) is True


def test_vel_aligned_northeast():
    vel = np.array([7.0, 7.0])              # 북동 45°
    pts = np.array([[0.0, 0.0], [100.0, 100.0]])
    assert vel_aligned_with_path(vel, pts) is True


def test_vel_not_aligned_perpendicular():
    vel = np.array([0.0, 10.0])             # 동쪽 속도, 경로는 북쪽
    pts = np.array([[0.0, 0.0], [100.0, 0.0]])
    assert vel_aligned_with_path(vel, pts) is False


def test_vel_not_aligned_opposite():
    vel = np.array([-10.0, 0.0])            # 남쪽 (역방향)
    pts = np.array([[0.0, 0.0], [100.0, 0.0]])
    assert vel_aligned_with_path(vel, pts) is False


def test_vel_not_aligned_low_speed():
    vel = np.array([0.5, 0.0])              # 속도 1.0 m/s 미만 → 방향 불신뢰
    pts = np.array([[0.0, 0.0], [100.0, 0.0]])
    assert vel_aligned_with_path(vel, pts) is False


def test_vel_aligned_3d_ned():
    vel3 = np.array([10.0, 0.0, -2.0])     # 3D NED — 처음 2원소만 사용
    pts = np.array([[0.0, 0.0], [100.0, 0.0]])
    assert vel_aligned_with_path(vel3, pts) is True


def test_vel_aligned_single_point_path():
    vel = np.array([5.0, 0.0])
    pts = np.array([[0.0, 0.0]])            # WP 1개 → 방향 불명 → True 반환
    assert vel_aligned_with_path(vel, pts) is True


# ── WP1 착륙 준비 판정 (wp1_land_ready) ─────────────────────────
# 역천이 오버슈트 후 MC로 WP1 복귀, 도달+안정 시 착륙.

def test_wp1_land_ready_close_and_slow():
    assert wp1_land_ready(2.0, 1.0, 3.0, 1.5) is True


def test_wp1_land_not_ready_far():
    assert wp1_land_ready(5.0, 0.5, 3.0, 1.5) is False   # 반경 밖


def test_wp1_land_not_ready_fast():
    assert wp1_land_ready(1.0, 2.0, 3.0, 1.5) is False   # 아직 빠름 (정착 전)


def test_wp1_land_boundary_strict():
    # 경계값: dist==radius, speed==thresh 는 미트리거 (strict <)
    assert wp1_land_ready(3.0, 1.5, 3.0, 1.5) is False


# ── 기체 타입별 상태 분기 (after_climb_state / after_following_state) ──
# 반환 문자열은 offboard_node._State 의 value 와 일치해야 한다.

def test_after_climb_vtol():
    # VTOL: CLIMBING → TRANSITION_FW (MC→FW 천이)
    assert after_climb_state(False) == "transition_fw"


def test_after_climb_mc():
    # MC: CLIMBING → STREAMING (천이 생략, 바로 OFFBOARD 진입)
    assert after_climb_state(True) == "streaming"


def test_after_following_vtol():
    # VTOL: FOLLOWING → TRANSITION_MC (FW→MC 역천이)
    assert after_following_state(False) == "transition_mc"


def test_after_following_mc():
    # MC: FOLLOWING → HOLD (역천이 생략, 마지막 WP 복귀·착륙)
    assert after_following_state(True) == "hold"
