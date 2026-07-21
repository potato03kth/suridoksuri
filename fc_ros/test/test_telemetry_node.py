"""
TelemetryNode 순수 로직 테스트.

rclpy/mavros 없이 실행 가능한 어댑터 변환 로직만 검증한다.
conftest.py가 fc_bridge 경로를 sys.path에 추가한다.
"""
import math
import types

import numpy as np
import pytest

from fc_bridge.comm.vehicle_state import VehicleState
from fc_bridge.utils.rotation import quat_to_euler_xyz
from fc_ros.adapters.vehicle_state_bridge import (
    update_from_pose,
    update_from_twist,
    update_from_mavros_state,
    update_from_extended_state,
    update_from_battery,
)


# ── 헬퍼: ROS2 메시지 모의 객체 ─────────────────────────────────

def _pose_msg(x_enu, y_enu, z_enu, q_w=1.0, q_x=0.0, q_y=0.0, q_z=0.0):
    msg = types.SimpleNamespace()
    msg.pose = types.SimpleNamespace(
        position=types.SimpleNamespace(x=x_enu, y=y_enu, z=z_enu),
        orientation=types.SimpleNamespace(w=q_w, x=q_x, y=q_y, z=q_z),
    )
    return msg


def _twist_msg(vx_enu, vy_enu, vz_enu):
    msg = types.SimpleNamespace()
    msg.twist = types.SimpleNamespace(
        linear=types.SimpleNamespace(x=vx_enu, y=vy_enu, z=vz_enu)
    )
    return msg


def _yaw_quat(yaw_rad):
    """ENU z축 기준 yaw only 쿼터니언 (w, x, y, z)."""
    return math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)


# ── quat_to_euler_xyz ────────────────────────────────────────────

def test_quat_identity():
    roll, pitch, yaw = quat_to_euler_xyz(1.0, 0.0, 0.0, 0.0)
    assert roll  == pytest.approx(0.0, abs=1e-9)
    assert pitch == pytest.approx(0.0, abs=1e-9)
    assert yaw   == pytest.approx(0.0, abs=1e-9)


def test_quat_yaw_90deg():
    w, x, y, z = _yaw_quat(np.pi / 2)
    _, _, yaw = quat_to_euler_xyz(w, x, y, z)
    assert yaw == pytest.approx(np.pi / 2, abs=1e-9)


def test_quat_yaw_180deg():
    w, x, y, z = _yaw_quat(np.pi)
    _, _, yaw = quat_to_euler_xyz(w, x, y, z)
    assert abs(yaw) == pytest.approx(np.pi, abs=1e-9)


def test_quat_yaw_neg90deg():
    w, x, y, z = _yaw_quat(-np.pi / 2)
    _, _, yaw = quat_to_euler_xyz(w, x, y, z)
    assert yaw == pytest.approx(-np.pi / 2, abs=1e-9)


# ── update_from_pose — 위치 (ENU → NED) ─────────────────────────

def test_pose_north_only():
    """ENU (0, 5, 0) → NED [5, 0, 0]."""
    s = VehicleState()
    update_from_pose(s, _pose_msg(0, 5, 0))
    assert s.pos_ned == pytest.approx([5.0, 0.0, 0.0])


def test_pose_east_only():
    """ENU (3, 0, 0) → NED [0, 3, 0]."""
    s = VehicleState()
    update_from_pose(s, _pose_msg(3, 0, 0))
    assert s.pos_ned == pytest.approx([0.0, 3.0, 0.0])


def test_pose_altitude():
    """ENU (0, 0, 10) → NED h_up = 10."""
    s = VehicleState()
    update_from_pose(s, _pose_msg(0, 0, 10))
    assert s.pos_ned[2] == pytest.approx(10.0)


def test_pose_combined():
    """ENU (2, 3, 4) → NED [3, 2, 4]."""
    s = VehicleState()
    update_from_pose(s, _pose_msg(2, 3, 4))
    assert s.pos_ned == pytest.approx([3.0, 2.0, 4.0])


# ── update_from_pose — yaw 변환 (ENU → NED) ─────────────────────

def test_pose_yaw_north_facing():
    """ENU yaw=π/2 (북쪽) → NED yaw ≈ 0."""
    s = VehicleState()
    w, x, y, z = _yaw_quat(np.pi / 2)
    update_from_pose(s, _pose_msg(0, 0, 0, w, x, y, z))
    assert s.yaw == pytest.approx(0.0, abs=1e-6)


def test_pose_yaw_east_facing():
    """ENU yaw=0 (동쪽) → NED yaw ≈ π/2."""
    s = VehicleState()
    w, x, y, z = _yaw_quat(0.0)
    update_from_pose(s, _pose_msg(0, 0, 0, w, x, y, z))
    assert s.yaw == pytest.approx(np.pi / 2, abs=1e-6)


def test_pose_yaw_west_facing():
    """ENU yaw=π (서쪽) → NED yaw ≈ -π/2."""
    s = VehicleState()
    w, x, y, z = _yaw_quat(np.pi)
    update_from_pose(s, _pose_msg(0, 0, 0, w, x, y, z))
    assert s.yaw == pytest.approx(-np.pi / 2, abs=1e-6)


def test_pose_yaw_south_facing():
    """ENU yaw=-π/2 (남쪽) → NED yaw ≈ π."""
    s = VehicleState()
    w, x, y, z = _yaw_quat(-np.pi / 2)
    update_from_pose(s, _pose_msg(0, 0, 0, w, x, y, z))
    assert abs(s.yaw) == pytest.approx(np.pi, abs=1e-6)


def test_pose_yaw_always_in_range():
    """ENU yaw 전 범위 → NED yaw 항상 [-π, π]."""
    for yaw_enu in np.linspace(-np.pi, np.pi, 37):
        s = VehicleState()
        w, x, y, z = _yaw_quat(float(yaw_enu))
        update_from_pose(s, _pose_msg(0, 0, 0, w, x, y, z))
        assert -np.pi <= s.yaw <= np.pi, f"yaw_enu={yaw_enu:.3f} → yaw={s.yaw:.3f}"


def test_pose_timestamp_updated():
    s = VehicleState()
    update_from_pose(s, _pose_msg(0, 0, 0))
    assert s.timestamp > 0.0


# ── update_from_twist — 속도 (ENU → NED) ────────────────────────

def test_twist_north_velocity():
    """ENU vy=5 (북쪽) → NED vN=5."""
    s = VehicleState()
    update_from_twist(s, _twist_msg(0, 5, 0))
    assert s.vel_ned == pytest.approx([5.0, 0.0, 0.0])


def test_twist_east_velocity():
    """ENU vx=3 (동쪽) → NED vE=3."""
    s = VehicleState()
    update_from_twist(s, _twist_msg(3, 0, 0))
    assert s.vel_ned == pytest.approx([0.0, 3.0, 0.0])


def test_twist_up_velocity():
    """ENU vz=2 (상승) → NED vD=-2 (부호 반전)."""
    s = VehicleState()
    update_from_twist(s, _twist_msg(0, 0, 2))
    assert s.vel_ned[2] == pytest.approx(-2.0)


def test_twist_combined():
    """ENU (1, 2, 3) → NED [2, 1, -3]."""
    s = VehicleState()
    update_from_twist(s, _twist_msg(1, 2, 3))
    assert s.vel_ned == pytest.approx([2.0, 1.0, -3.0])


# ── update_from_mavros_state ─────────────────────────────────────

def test_armed_true():
    s = VehicleState()
    update_from_mavros_state(s, types.SimpleNamespace(armed=True))
    assert s.armed is True


def test_armed_false():
    s = VehicleState()
    s.armed = True
    update_from_mavros_state(s, types.SimpleNamespace(armed=False))
    assert s.armed is False


# ── update_from_extended_state ───────────────────────────────────

def test_vtol_state_mc():
    """vtol_state=3 (멀티콥터 모드)."""
    s = VehicleState()
    update_from_extended_state(s, types.SimpleNamespace(vtol_state=3))
    assert s.vtol_state == 3


def test_vtol_state_fw():
    """vtol_state=4 (고정익 모드)."""
    s = VehicleState()
    update_from_extended_state(s, types.SimpleNamespace(vtol_state=4))
    assert s.vtol_state == 4


# ── update_from_battery ──────────────────────────────────────────

def _battery_msg(voltage, current, percentage):
    return types.SimpleNamespace(voltage=voltage, current=current, percentage=percentage)


def test_battery_basic_fields():
    s = VehicleState()
    update_from_battery(s, _battery_msg(11.13, -5.2, 0.62))
    assert s.battery_voltage == pytest.approx(11.13)
    assert s.battery_current == pytest.approx(-5.2)
    assert s.battery_remaining == pytest.approx(0.62)


def test_battery_nan_percentage_keeps_previous():
    """PX4가 remaining을 미보고(NaN)하면 이전값을 유지한다."""
    s = VehicleState()
    s.battery_remaining = 0.8
    update_from_battery(s, _battery_msg(11.0, -3.0, float("nan")))
    assert s.battery_remaining == pytest.approx(0.8)


def test_battery_timestamp_updated():
    s = VehicleState()
    update_from_battery(s, _battery_msg(11.0, -3.0, 0.5))
    assert s.timestamp > 0.0


# ── VehicleState.copy() 격리 검증 ────────────────────────────────

def test_copy_pos_independence():
    """copy() 스냅샷은 원본 pos_ned 변경에 영향받지 않는다."""
    s = VehicleState()
    s.pos_ned = np.array([1.0, 2.0, 3.0])
    snap = s.copy()
    s.pos_ned[0] = 99.0
    assert snap.pos_ned[0] == pytest.approx(1.0)


def test_copy_vel_independence():
    """copy() 스냅샷은 원본 vel_ned 변경에 영향받지 않는다."""
    s = VehicleState()
    s.vel_ned = np.array([4.0, 5.0, 6.0])
    snap = s.copy()
    s.vel_ned[2] = 99.0
    assert snap.vel_ned[2] == pytest.approx(6.0)


def test_copy_scalar_fields():
    """copy()는 yaw, armed, vtol_state 등 스칼라 필드도 복사한다."""
    s = VehicleState()
    s.yaw = 1.23
    s.armed = True
    s.vtol_state = 4
    s.battery_voltage = 11.1
    s.battery_current = -4.5
    s.battery_remaining = 0.42
    snap = s.copy()
    assert snap.yaw == pytest.approx(1.23)
    assert snap.armed is True
    assert snap.vtol_state == 4
    assert snap.battery_voltage == pytest.approx(11.1)
    assert snap.battery_current == pytest.approx(-4.5)
    assert snap.battery_remaining == pytest.approx(0.42)
