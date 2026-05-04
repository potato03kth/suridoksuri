"""
Telemetry 단위 테스트.
FakeMavfile의 recv_queue에 메시지를 주입해 상태 캐시 갱신을 검증.
"""
import time
import numpy as np
import pytest

from fc_bridge.tests.conftest import FakeMavfile, FakeMavMsg
from fc_bridge.comm.mavlink_conn import MavlinkConn
from fc_bridge.comm.telemetry import Telemetry


def _make_telemetry():
    fake = FakeMavfile()
    conn = MavlinkConn(mock_transport=fake)
    conn.connect()
    tel = Telemetry(conn, poll_timeout=0.01)
    return tel, fake


def test_heartbeat_armed():
    """HEARTBEAT 수신 시 armed 플래그가 갱신된다."""
    tel, fake = _make_telemetry()
    # base_mode bit7 = 128 → armed
    fake.recv_queue.append(
        FakeMavMsg("HEARTBEAT", base_mode=128, custom_mode=0,
                   type=2, autopilot=3, system_status=4, mavlink_version=3)
    )
    tel.start()
    time.sleep(0.05)
    tel.stop()
    assert tel.is_armed is True


def test_heartbeat_disarmed():
    """base_mode bit7 = 0 → armed=False."""
    tel, fake = _make_telemetry()
    fake.recv_queue.append(
        FakeMavMsg("HEARTBEAT", base_mode=0, custom_mode=0,
                   type=2, autopilot=3, system_status=4, mavlink_version=3)
    )
    tel.start()
    time.sleep(0.05)
    tel.stop()
    assert tel.is_armed is False


def test_local_position_ned():
    """LOCAL_POSITION_NED 수신 시 pos_ned, vel_ned 갱신."""
    tel, fake = _make_telemetry()
    fake.recv_queue.append(
        FakeMavMsg("LOCAL_POSITION_NED",
                   x=10.0, y=20.0, z=-30.0,
                   vx=1.0, vy=2.0, vz=0.5)
    )
    tel.start()
    time.sleep(0.05)
    tel.stop()

    pos = tel.pos_ned
    assert pos[0] == pytest.approx(10.0)
    assert pos[1] == pytest.approx(20.0)
    assert pos[2] == pytest.approx(30.0)   # h_up = -z = 30

    vel = tel.vel_ned
    assert vel[0] == pytest.approx(1.0)
    assert vel[1] == pytest.approx(2.0)
    assert vel[2] == pytest.approx(0.5)


def test_attitude():
    """ATTITUDE 수신 시 roll/pitch/yaw 갱신."""
    tel, fake = _make_telemetry()
    fake.recv_queue.append(
        FakeMavMsg("ATTITUDE",
                   roll=0.1, pitch=0.2, yaw=1.5,
                   rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0)
    )
    tel.start()
    time.sleep(0.05)
    tel.stop()

    assert tel.yaw == pytest.approx(1.5)
    s = tel.get_state()
    assert s.roll  == pytest.approx(0.1)
    assert s.pitch == pytest.approx(0.2)


def test_extended_sys_state_vtol():
    """EXTENDED_SYS_STATE 수신 시 vtol_state 갱신."""
    tel, fake = _make_telemetry()
    fake.recv_queue.append(
        FakeMavMsg("EXTENDED_SYS_STATE",
                   vtol_state=4,   # MAV_VTOL_STATE_FW
                   landed_state=2)
    )
    tel.start()
    time.sleep(0.05)
    tel.stop()

    assert tel.get_state().vtol_state == 4


def test_get_state_returns_copy():
    """get_state()는 독립 복사본을 반환해 외부 수정이 캐시에 영향 없다."""
    tel, fake = _make_telemetry()
    fake.recv_queue.append(
        FakeMavMsg("LOCAL_POSITION_NED",
                   x=1.0, y=2.0, z=-5.0,
                   vx=0.0, vy=0.0, vz=0.0)
    )
    tel.start()
    time.sleep(0.05)
    tel.stop()

    s1 = tel.get_state()
    s1.pos_ned[0] = 999.0
    s2 = tel.get_state()
    assert s2.pos_ned[0] == pytest.approx(1.0)
