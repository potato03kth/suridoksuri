"""
MissionUploader 단위 테스트.
FakeMavfile로 핸드셰이크 시퀀스를 검증.
"""
import math
import numpy as np
import pytest

from fc_bridge.tests.conftest import FakeMavfile, FakeMavMsg
from fc_bridge.comm.mavlink_conn import MavlinkConn
from fc_bridge.execution.mission_uploader import MissionUploader


class _FakeMavDialect:
    """
    pymavlink의 dialect 객체 (mavfile.mav) 흉내.
    MavlinkConn.send_message()는 mavfile.mav.send()를,
    MissionUploader는 mavfile.mav.mission_count_send() 등을 호출한다.
    """
    def __init__(self):
        self.sent_raw: list = []
        self.mission_count_calls: list = []
        self.mission_item_int_calls: list = []

    def send(self, msg):
        self.sent_raw.append(msg)

    def mission_count_send(self, sys_id, comp_id, count, mission_type):
        self.mission_count_calls.append((sys_id, comp_id, count, mission_type))

    def mission_item_int_send(self, sys_id, comp_id, seq, frame, command,
                               current, autocontinue, p1, p2, p3, p4,
                               x, y, z, mission_type):
        self.mission_item_int_calls.append({
            "seq": seq, "frame": frame, "command": command,
            "x": x, "y": y, "z": z,
        })


def _make_uploader(wps_ned, fake_responses):
    """
    fake_responses: recv_match가 순서대로 반환할 메시지 목록.
    """
    fake = FakeMavfile()
    dialect = _FakeMavDialect()
    fake.mav = dialect          # conn.mav.mav = dialect
    for msg in fake_responses:
        fake.recv_queue.append(msg)

    conn = MavlinkConn(mock_transport=fake)
    conn.connect()
    uploader = MissionUploader(conn, timeout=0.1)
    return uploader, dialect


def _mission_request_int(seq):
    return FakeMavMsg("MISSION_REQUEST_INT", seq=seq)


def _mission_ack(ok=True):
    return FakeMavMsg("MISSION_ACK", type=0 if ok else 1)


def test_upload_3_waypoints_success():
    """3개 WP 정상 업로드: REQUEST_INT 0,1,2 → ACK OK."""
    wps = np.array([
        [0.0, 0.0, 50.0],
        [100.0, 0.0, 50.0],
        [200.0, 100.0, 50.0],
    ])
    responses = [
        _mission_request_int(0),
        _mission_request_int(1),
        _mission_request_int(2),
        _mission_ack(ok=True),
    ]
    uploader, inner = _make_uploader(wps, responses)
    result = uploader.upload(wps)
    assert result is True
    assert inner.mission_count_calls[0][2] == 3   # N=3
    assert len(inner.mission_item_int_calls) == 3


def test_upload_item_z_is_negated():
    """MISSION_ITEM_INT의 z = -h_up (LOCAL_NED 컨벤션)."""
    wps = np.array([[0.0, 0.0, 100.0], [50.0, 0.0, 100.0]])
    responses = [_mission_request_int(0), _mission_request_int(1), _mission_ack()]
    uploader, inner = _make_uploader(wps, responses)
    uploader.upload(wps)
    assert inner.mission_item_int_calls[0]["z"] == pytest.approx(-100.0)


def test_upload_fails_on_nack():
    """MISSION_ACK type!=0 → False 반환."""
    wps = np.array([[0.0, 0.0, 50.0], [50.0, 0.0, 50.0]])
    responses = [_mission_request_int(0), _mission_request_int(1), _mission_ack(ok=False)]
    uploader, inner = _make_uploader(wps, responses)
    result = uploader.upload(wps)
    assert result is False


def test_upload_fails_on_timeout():
    """응답 없음 (빈 큐) → False 반환."""
    wps = np.array([[0.0, 0.0, 50.0], [50.0, 0.0, 50.0]])
    uploader, _ = _make_uploader(wps, [])  # 응답 없음
    result = uploader.upload(wps)
    assert result is False


def test_invalid_waypoints_shape():
    """shape 오류 시 ValueError."""
    uploader, _ = _make_uploader(np.zeros((3, 3)), [])
    with pytest.raises(ValueError):
        uploader.upload(np.zeros((3, 2)))   # 2열: 오류
