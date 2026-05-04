"""
MAVLink 연결 래퍼.

실제 연결(pymavlink)과 테스트용 mock transport를 동일 인터페이스로 사용한다.
mock_transport 인자에 FakeMavfile 등을 주입하면 하드웨어 없이 단위 테스트 가능.
"""
from __future__ import annotations
import threading
import time
from typing import Any


class MavlinkConn:
    """
    pymavlink 연결을 감싸는 thin wrapper.

    Parameters
    ----------
    connection_str : str
        pymavlink 연결 문자열. 예: "udp:127.0.0.1:14550", "/dev/ttyACM0"
    baud : int
        시리얼 연결 시 baud rate (UDP에서는 무시됨).
    source_system : int
        GCS sysid (기본 255).
    mock_transport : Any, optional
        테스트용 fake mavfile 주입. None이면 pymavlink를 사용.
    """

    def __init__(self,
                 connection_str: str = "udp:127.0.0.1:14550",
                 baud: int = 57600,
                 source_system: int = 255,
                 mock_transport: Any = None):
        self._conn_str = connection_str
        self._baud = baud
        self._sysid = source_system
        self._mav = mock_transport
        self._lock = threading.Lock()

    # ── 연결 관리 ────────────────────────────────────────────

    def connect(self, timeout: float = 10.0) -> None:
        """
        연결을 수립한다. mock_transport가 주입된 경우 아무것도 하지 않는다.
        """
        if self._mav is not None:
            return

        from pymavlink import mavutil
        self._mav = mavutil.mavlink_connection(
            self._conn_str,
            baud=self._baud,
            source_system=self._sysid,
        )
        # HEARTBEAT 대기로 기체 존재 확인
        hb = self._mav.wait_heartbeat(timeout=timeout)
        if hb is None:
            raise ConnectionError(
                f"HEARTBEAT 수신 실패 (timeout={timeout}s): {self._conn_str}"
            )

    def close(self) -> None:
        if self._mav is not None:
            try:
                self._mav.close()
            except Exception:
                pass
            self._mav = None

    # ── 메시지 송수신 ─────────────────────────────────────────

    def send_message(self, msg) -> None:
        """인코딩된 MAVLink 메시지를 전송한다."""
        with self._lock:
            self._mav.mav.send(msg)

    def recv_match(self, type: str | list[str],
                   blocking: bool = True,
                   timeout: float = 1.0):
        """
        지정 타입의 메시지를 수신한다.

        Parameters
        ----------
        type : str | list[str]
            메시지 타입명. 예: "LOCAL_POSITION_NED" 또는 ["ATTITUDE", "HEARTBEAT"]
        blocking : bool
        timeout : float

        Returns
        -------
        MAVLink 메시지 객체 또는 None (timeout)
        """
        if isinstance(type, str):
            type = [type]
        return self._mav.recv_match(type=type, blocking=blocking, timeout=timeout)

    # ── 편의 프로퍼티 ─────────────────────────────────────────

    @property
    def mav(self):
        """pymavlink mavfile 객체 (SET_MODE 등 직접 호출용)."""
        return self._mav

    @property
    def target_system(self) -> int:
        try:
            return self._mav.target_system
        except AttributeError:
            return 1

    @property
    def target_component(self) -> int:
        try:
            return self._mav.target_component
        except AttributeError:
            return 1

    def __repr__(self) -> str:
        return f"MavlinkConn({self._conn_str!r}, connected={self._mav is not None})"
