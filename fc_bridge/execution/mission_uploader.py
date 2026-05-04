"""
Phase 1: PX4 Mission 업로드.

MISSION_ITEM_INT 핸드셰이크 시퀀스:
  GCS → FC  MISSION_COUNT(N)
  FC  → GCS MISSION_REQUEST_INT(0)
  GCS → FC  MISSION_ITEM_INT(0)
  FC  → GCS MISSION_REQUEST_INT(1)
  ...
  GCS → FC  MISSION_ITEM_INT(N-1)
  FC  → GCS MISSION_ACK(type=0 OK)

waypoints_ned는 [N, E, h_up] 형식이며, 내부에서 MAV_FRAME_LOCAL_NED로 변환한다.
h_up > 0 = 고도 → LOCAL_NED에서 z = -h_up (아래가 양수).
"""
from __future__ import annotations
import numpy as np

from fc_bridge.comm.mavlink_conn import MavlinkConn


# MAVLink 상수 (pymavlink mavutil 없이 정수 직접 사용)
MAV_FRAME_LOCAL_NED = 1
MAV_CMD_NAV_WAYPOINT = 16
MAV_MISSION_ACCEPTED = 0
MAV_MISSION_TYPE_MISSION = 0


class MissionUploader:
    """
    경로의 waypoints_ned를 PX4 Mission으로 업로드한다.

    Parameters
    ----------
    conn : MavlinkConn
    timeout : float
        각 핸드셰이크 단계 응답 대기 시간 (s).
    """

    def __init__(self, conn: MavlinkConn, timeout: float = 5.0):
        self._conn = conn
        self._timeout = timeout

    # ── 공개 인터페이스 ──────────────────────────────────────

    def upload(self, waypoints_ned: np.ndarray,
               accept_radius: float = 2.0,
               speed: float | None = None,
               ) -> bool:
        """
        웨이포인트 배열을 PX4 Mission으로 업로드한다.

        Parameters
        ----------
        waypoints_ned : np.ndarray, shape (N, 3)
            NED 좌표 [N, E, h_up].
        accept_radius : float
            WP 도달 판정 반경 (m).
        speed : float | None
            각 WP에서 목표 속도 (m/s). None이면 0 (PX4 기본값 사용).

        Returns
        -------
        bool : 업로드 성공 여부.
        """
        wps = np.asarray(waypoints_ned, dtype=float)
        if wps.ndim != 2 or wps.shape[1] < 3:
            raise ValueError("waypoints_ned shape must be (N, 3)")
        items = self._build_items(wps, accept_radius, speed or 0.0)
        return self._handshake(items)

    # ── MISSION_ITEM_INT 생성 ────────────────────────────────

    def _build_items(self,
                     wps: np.ndarray,
                     accept_radius: float,
                     speed: float,
                     ) -> list[dict]:
        """waypoints → MISSION_ITEM_INT 파라미터 딕셔너리 리스트."""
        items = []
        for seq, wp in enumerate(wps):
            n, e, h = float(wp[0]), float(wp[1]), float(wp[2])
            items.append({
                "seq": seq,
                "frame": MAV_FRAME_LOCAL_NED,
                "command": MAV_CMD_NAV_WAYPOINT,
                "current": 1 if seq == 0 else 0,
                "autocontinue": 1,
                "param1": 0.0,             # hold time
                "param2": accept_radius,
                "param3": 0.0,             # pass radius
                "param4": float("nan"),    # yaw (NaN = 자동)
                "x": int(n * 1e4),         # MISSION_ITEM_INT: 1e7 for lat/lon, 1e4 for local
                "y": int(e * 1e4),
                "z": float(-h),            # LOCAL_NED z = -h_up
                "mission_type": MAV_MISSION_TYPE_MISSION,
            })
        return items

    # ── MAVLink 핸드셰이크 ───────────────────────────────────

    def _handshake(self, items: list[dict]) -> bool:
        N = len(items)
        sys_id = self._conn.target_system
        comp_id = self._conn.target_component
        mav = self._conn.mav

        # MISSION_COUNT 전송
        mav.mav.mission_count_send(
            sys_id, comp_id, N,
            MAV_MISSION_TYPE_MISSION,
        )

        # 아이템 요청 루프
        for _ in range(N + 5):   # 약간의 여유
            msg = self._conn.recv_match(
                type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"],
                blocking=True,
                timeout=self._timeout,
            )
            if msg is None:
                return False

            t = msg.get_type()

            if t in ("MISSION_REQUEST_INT", "MISSION_REQUEST"):
                seq = msg.seq
                if seq >= N:
                    return False
                item = items[seq]
                mav.mav.mission_item_int_send(
                    sys_id, comp_id,
                    item["seq"],
                    item["frame"],
                    item["command"],
                    item["current"],
                    item["autocontinue"],
                    item["param1"],
                    item["param2"],
                    item["param3"],
                    item["param4"],
                    item["x"],
                    item["y"],
                    item["z"],
                    item["mission_type"],
                )

            elif t == "MISSION_ACK":
                return int(msg.type) == MAV_MISSION_ACCEPTED

        return False
