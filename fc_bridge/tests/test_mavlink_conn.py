"""
MavlinkConn 단위 테스트.
실제 pymavlink / 하드웨어 없이 FakeMavfile로 검증.
"""
import pytest
from fc_bridge.tests.conftest import FakeMavfile, FakeMavMsg
from fc_bridge.comm.mavlink_conn import MavlinkConn


def test_connect_with_mock(conn):
    """mock transport 주입 시 예외 없이 connect() 완료."""
    assert conn.mav is not None


def test_send_message_stored(conn, fake_mav):
    """send_message()가 FakeMavfile.sent_messages에 쌓인다."""

    class _DummyMsg:
        pass

    msg = _DummyMsg()
    conn.send_message(msg)
    assert fake_mav.sent_messages[-1] is msg


def test_recv_match_returns_queued_message(conn, fake_mav):
    """recv_queue에 넣어둔 메시지가 recv_match()로 반환된다."""
    fake_msg = FakeMavMsg("LOCAL_POSITION_NED", x=1.0, y=2.0, z=-10.0)
    fake_mav.recv_queue.append(fake_msg)

    received = conn.recv_match("LOCAL_POSITION_NED", blocking=False)
    assert received is not None
    assert received.get_type() == "LOCAL_POSITION_NED"
    assert received.x == pytest.approx(1.0)


def test_recv_match_returns_none_on_empty(conn):
    """큐가 비어있으면 None 반환."""
    result = conn.recv_match("ATTITUDE", blocking=False, timeout=0.0)
    assert result is None


def test_close_clears_transport(conn):
    """close() 후 mav 속성이 None이 된다."""
    conn.close()
    assert conn.mav is None


def test_target_system_fallback():
    """target_system/component — mock에서 기본값 반환."""
    fake = FakeMavfile()
    c = MavlinkConn(mock_transport=fake)
    c.connect()
    assert c.target_system == 1
    assert c.target_component == 1
