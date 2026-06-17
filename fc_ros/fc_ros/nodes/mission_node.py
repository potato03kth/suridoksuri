"""
MissionNode: Phase 1 전용. 경로 생성 후 MAVROS 서비스로 미션 업로드.

서비스 호출: /mavros/mission/push (mavros_msgs/srv/WaypointPush)
프레임: MAV_FRAME_GLOBAL_RELATIVE_ALT (PX4 미션은 글로벌 프레임만 지원)

NED 웨이포인트를 home_lat/home_lon 기준으로 GPS 좌표로 변환하여 업로드한다.

필요 파라미터 (params YAML 또는 --ros-args -p):
  waypoints : [N, E, h, N, E, h, ...] 1D 리스트  (NED, m, 3개씩 묶음)
              — ROS2 파라미터는 중첩 리스트를 지원하지 않으므로 반드시 1D로 입력한다.
  home_lat  : 홈 위도 (degrees, 기본값: PX4 SITL gz_x500 기본 홈)
  home_lon  : 홈 경도 (degrees, 기본값: PX4 SITL gz_x500 기본 홈)
"""
from __future__ import annotations
import numpy as np

import rclpy
from rclpy.node import Node
from mavros_msgs.msg import Waypoint
from mavros_msgs.srv import WaypointPush


# MAVROS Waypoint 상수
_MAV_FRAME_GLOBAL_RELATIVE_ALT = 3
_MAV_CMD_NAV_WAYPOINT = 16

_R_EARTH = 6_371_000.0  # m


class MissionNode(Node):

    def __init__(self):
        super().__init__("mission_node")

        self.declare_parameter("waypoints",
                               [0.0, 0.0, 50.0, 100.0, 0.0, 50.0])
        self.declare_parameter("home_lat", 47.397742)   # PX4 SITL gz_x500 기본 홈
        self.declare_parameter("home_lon", 8.545594)

        raw = self.get_parameter("waypoints").value
        # ROS2 파라미터는 중첩 리스트 불가 → 1D로 받아 (N, 3)으로 변환
        self._waypoints = np.array(raw, dtype=float).reshape(-1, 3)
        self._home_lat = float(self.get_parameter("home_lat").value)
        self._home_lon = float(self.get_parameter("home_lon").value)

        self._push_cli = self.create_client(
            WaypointPush, "/mavros/mission/push")
        self._uploaded = False

        # 노드 기동 후 1회 업로드
        self.create_timer(1.0, self._upload_once)

    # ── 업로드 ───────────────────────────────────────────────

    def _upload_once(self) -> None:
        if self._uploaded:
            return
        self._uploaded = True
        self._upload(self._waypoints)

    @staticmethod
    def _ned_to_gps(home_lat: float, home_lon: float,
                    ned: np.ndarray) -> tuple[float, float, float]:
        """NED [N, E, h_up] → (lat_deg, lon_deg, alt_rel_m)."""
        lat = home_lat + ned[0] / _R_EARTH * (180.0 / np.pi)
        lon = home_lon + \
            ned[1] / (_R_EARTH * np.cos(np.radians(home_lat))) * \
            (180.0 / np.pi)
        return float(lat), float(lon), float(ned[2])

    def _upload(self, waypoints: np.ndarray) -> None:
        """waypoints: (N, 3) [N, E, h_up] NED → GPS 변환 후 /mavros/mission/push"""
        if not self._push_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/mavros/mission/push 서비스 없음")
            return

        wp_list = []
        for i, wp in enumerate(waypoints):
            lat, lon, alt = self._ned_to_gps(
                self._home_lat, self._home_lon, wp)
            m = Waypoint()
            m.frame = _MAV_FRAME_GLOBAL_RELATIVE_ALT
            m.command = _MAV_CMD_NAV_WAYPOINT
            m.is_current = (i == 0)
            m.autocontinue = True
            m.x_lat = lat
            m.y_long = lon
            m.z_alt = alt
            wp_list.append(m)

        req = WaypointPush.Request()
        req.start_index = 0
        req.waypoints = wp_list

        future = self._push_cli.call_async(req)
        future.add_done_callback(self._on_push_done)

    def _on_push_done(self, future) -> None:
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().info(
                    f"미션 업로드 성공: {resp.wp_transfered}개")
            else:
                self.get_logger().warn("미션 업로드 실패")
        except Exception as e:
            self.get_logger().error(f"미션 업로드 예외: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
