"""
MissionNode: Phase 1 전용. 경로 생성 후 MAVROS 서비스로 미션 업로드.

서비스 호출: /mavros/mission/push (mavros_msgs/srv/WaypointPush)
fallback: LOCAL_NED 프레임 미지원 시 MissionUploader.upload() 직접 호출

필요 파라미터 (params YAML 또는 --ros-args -p):
  waypoints : [[N, E, h], ...] 2D 리스트  (NED, m)
"""
from __future__ import annotations
import numpy as np

import rclpy
from rclpy.node import Node
from mavros_msgs.msg import Waypoint
from mavros_msgs.srv import WaypointPush


# MAVROS Waypoint 상수
_MAV_FRAME_LOCAL_NED = 1
_MAV_CMD_NAV_WAYPOINT = 16


class MissionNode(Node):

    def __init__(self):
        super().__init__("mission_node")

        self.declare_parameter("waypoints",
                               [[0.0, 0.0, 150.0], [500.0, 0.0, 150.0]])

        raw = self.get_parameter("waypoints").value
        self._waypoints = np.array(raw, dtype=float)

        self._push_cli = self.create_client(WaypointPush, "/mavros/mission/push")
        self._uploaded = False

        # 노드 기동 후 1회 업로드
        self.create_timer(1.0, self._upload_once)

    # ── 업로드 ───────────────────────────────────────────────

    def _upload_once(self) -> None:
        if self._uploaded:
            return
        self._uploaded = True
        self._upload(self._waypoints)

    def _upload(self, waypoints: np.ndarray) -> None:
        """waypoints: (N, 3) [N, E, h_up] NED → /mavros/mission/push"""
        if not self._push_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                "/mavros/mission/push 서비스 없음 — fallback 시도")
            self._fallback_upload(waypoints)
            return

        wp_list = []
        for i, wp in enumerate(waypoints):
            m = Waypoint()
            m.frame       = _MAV_FRAME_LOCAL_NED
            m.command     = _MAV_CMD_NAV_WAYPOINT
            m.is_current  = (i == 0)
            m.autocontinue = True
            m.x_lat  = float(wp[0])     # N
            m.y_long = float(wp[1])     # E
            m.z_alt  = -float(wp[2])    # LOCAL_NED: z = -h_up
            wp_list.append(m)

        req = WaypointPush.Request()
        req.start_index = 0
        req.waypoints   = wp_list

        future = self._push_cli.call_async(req)
        future.add_done_callback(self._on_push_done)

    def _on_push_done(self, future) -> None:
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().info(
                    f"미션 업로드 성공: {resp.wp_transferred}개")
            else:
                self.get_logger().warn("미션 업로드 실패 — fallback 시도")
                self._fallback_upload(self._waypoints)
        except Exception as e:
            self.get_logger().error(f"미션 업로드 예외: {e}")

    def _fallback_upload(self, waypoints: np.ndarray) -> None:
        """LOCAL_NED 미지원 시 MissionUploader(pymavlink)로 직접 업로드.

        MavlinkConn 인스턴스가 필요하므로 여기서는 경고 로그만 출력한다.
        필요 시 run_phase1.py를 직접 사용하거나 MavlinkConn을 주입하도록 확장한다.
        """
        self.get_logger().error(
            "fallback: pymavlink MissionUploader가 필요합니다. "
            "MAVROS 없이 직접 업로드하려면 fc_bridge/run_phase1.py를 사용하세요.")


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
