"""
pytest fixtures 공통.
FakeMav: pymavlink mavfile 인터페이스를 흉내내는 mock transport.
"""
from __future__ import annotations
import threading
from collections import deque
from typing import Any
import numpy as np
import pytest

from fc_bridge.comm.mavlink_conn import MavlinkConn


class FakeMavMsg:
    """단순 MAVLink 메시지 흉내."""
    def __init__(self, msgtype: str, **fields):
        self._type = msgtype
        for k, v in fields.items():
            setattr(self, k, v)

    def get_type(self) -> str:
        return self._type

    def __repr__(self):
        return f"FakeMavMsg({self._type})"


class FakeMavfile:
    """
    pymavlink mavfile을 흉내내는 mock.
    send()로 전송된 메시지를 sent_messages 큐에 저장.
    recv_queue에 미리 메시지를 넣어두면 recv_match()가 반환.
    """

    def __init__(self):
        self.sent_messages: list = []
        self.recv_queue: deque = deque()
        self.target_system = 1
        self.target_component = 1
        self._hb = FakeMavMsg("HEARTBEAT", type=2, autopilot=3,
                               base_mode=0, custom_mode=0,
                               system_status=4, mavlink_version=3)

        # mav.send() 호출 경로 지원
        self.mav = self

    def send(self, msg):
        self.sent_messages.append(msg)

    def recv_match(self, type=None, blocking=True, timeout=1.0):
        if self.recv_queue:
            msg = self.recv_queue.popleft()
            if type is None:
                return msg
            if isinstance(type, str):
                type = [type]
            if msg.get_type() in type:
                return msg
        return None

    def wait_heartbeat(self, timeout=10.0):
        return self._hb

    def close(self):
        pass


@pytest.fixture
def fake_mav():
    return FakeMavfile()


@pytest.fixture
def conn(fake_mav):
    c = MavlinkConn(mock_transport=fake_mav)
    c.connect()
    return c
