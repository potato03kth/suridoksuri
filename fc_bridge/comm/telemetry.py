"""
PX4 텔레메트리 수신·캐시.

백그라운드 스레드에서 MAVLink 메시지를 계속 읽어 최신 기체 상태를 캐시한다.
외부에서는 read-only 프로퍼티로 접근한다.

수신 메시지:
  #0   HEARTBEAT          — armed 여부, 비행 모드
  #30  ATTITUDE            — roll, pitch, yaw (rad)
  #32  LOCAL_POSITION_NED  — x(N), y(E), z(D), vx, vy, vz
  #245 EXTENDED_SYS_STATE  — vtol_state
"""
from __future__ import annotations
import threading
import time
import numpy as np
from typing import Optional

from fc_bridge.comm.mavlink_conn import MavlinkConn


# PX4 custom_mode 비트 상수 (MAV_MODE_FLAG 기반)
_MAV_MODE_FLAG_SAFETY_ARMED = 128


class VehicleState:
    """기체 상태 스냅샷 (불변 데이터클래스 대신 단순 class)."""
    __slots__ = (
        "pos_ned", "vel_ned",
        "roll", "pitch", "yaw",
        "vtol_state",
        "armed", "base_mode", "custom_mode",
        "timestamp",
    )

    def __init__(self):
        self.pos_ned   = np.zeros(3)   # [N, E, -D] → [N, E, h_up]  (h = -z)
        self.vel_ned   = np.zeros(3)   # [vN, vE, vD]
        self.roll      = 0.0
        self.pitch     = 0.0
        self.yaw       = 0.0
        self.vtol_state = 0             # 0=undefined,1=transition_to_fw,2=transition_to_mc,3=mc,4=fw
        self.armed      = False
        self.base_mode  = 0
        self.custom_mode = 0
        self.timestamp  = 0.0          # 마지막 갱신 시각 (time.monotonic)

    @property
    def heading_rad(self) -> float:
        return self.yaw

    @property
    def pos_ned_2d(self) -> np.ndarray:
        return self.pos_ned[:2]

    def copy(self) -> "VehicleState":
        s = VehicleState()
        s.pos_ned    = self.pos_ned.copy()
        s.vel_ned    = self.vel_ned.copy()
        s.roll       = self.roll
        s.pitch      = self.pitch
        s.yaw        = self.yaw
        s.vtol_state = self.vtol_state
        s.armed      = self.armed
        s.base_mode  = self.base_mode
        s.custom_mode = self.custom_mode
        s.timestamp  = self.timestamp
        return s


class Telemetry:
    """
    백그라운드 스레드로 PX4 상태를 수신·캐시한다.

    Parameters
    ----------
    conn : MavlinkConn
        연결된 MAVLink 연결 객체.
    poll_timeout : float
        recv_match() 한 번의 timeout (s). 너무 크면 stop() 지연.
    """

    def __init__(self, conn: MavlinkConn, poll_timeout: float = 0.05):
        self._conn = conn
        self._poll_timeout = poll_timeout
        self._state = VehicleState()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ── 외부 인터페이스 ──────────────────────────────────────

    def start(self) -> None:
        """수신 스레드를 시작한다."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="telemetry")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """수신 스레드를 정지한다."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def get_state(self) -> VehicleState:
        """현재 기체 상태의 스냅샷을 반환한다 (thread-safe)."""
        with self._lock:
            return self._state.copy()

    @property
    def is_armed(self) -> bool:
        with self._lock:
            return self._state.armed

    @property
    def pos_ned(self) -> np.ndarray:
        with self._lock:
            return self._state.pos_ned.copy()

    @property
    def vel_ned(self) -> np.ndarray:
        with self._lock:
            return self._state.vel_ned.copy()

    @property
    def yaw(self) -> float:
        with self._lock:
            return self._state.yaw

    # ── 수신 루프 ────────────────────────────────────────────

    _WATCH_TYPES = [
        "HEARTBEAT",
        "ATTITUDE",
        "LOCAL_POSITION_NED",
        "EXTENDED_SYS_STATE",
    ]

    def _loop(self) -> None:
        while self._running:
            msg = self._conn.recv_match(
                type=self._WATCH_TYPES,
                blocking=True,
                timeout=self._poll_timeout,
            )
            if msg is None:
                continue
            self._dispatch(msg)

    def _dispatch(self, msg) -> None:
        t = msg.get_type()
        with self._lock:
            if t == "HEARTBEAT":
                self._state.armed = bool(
                    msg.base_mode & _MAV_MODE_FLAG_SAFETY_ARMED
                )
                self._state.base_mode   = msg.base_mode
                self._state.custom_mode = msg.custom_mode
                self._state.timestamp   = time.monotonic()

            elif t == "ATTITUDE":
                self._state.roll  = float(msg.roll)
                self._state.pitch = float(msg.pitch)
                self._state.yaw   = float(msg.yaw)
                self._state.timestamp = time.monotonic()

            elif t == "LOCAL_POSITION_NED":
                # NED: x=N, y=E, z=D(아래가 양수)
                # pos_ned 저장: [N, E, h_up] where h_up = -z
                self._state.pos_ned = np.array([
                    float(msg.x),
                    float(msg.y),
                    -float(msg.z),   # 고도를 위쪽 양수로 변환
                ])
                self._state.vel_ned = np.array([
                    float(msg.vx),
                    float(msg.vy),
                    float(msg.vz),
                ])
                self._state.timestamp = time.monotonic()

            elif t == "EXTENDED_SYS_STATE":
                self._state.vtol_state = int(msg.vtol_state)
                self._state.timestamp  = time.monotonic()
