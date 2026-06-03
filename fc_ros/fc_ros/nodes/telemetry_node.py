"""
TelemetryNode: MAVROS 토픽 구독 → VehicleState 유지.

기존 fc_bridge.comm.Telemetry의 백그라운드 스레드를 ROS2 콜백으로 대체.
타이밍 제약 없음 — 수신 시점에만 콜백 실행.

구독 토픽:
  /mavros/local_position/pose           (geometry_msgs/PoseStamped)  — 위치 + quaternion
  /mavros/local_position/velocity_local (geometry_msgs/TwistStamped) — 속도
  /mavros/state                         (mavros_msgs/State)           — armed, connected, mode
  /mavros/extended_state                (mavros_msgs/ExtendedState)   — vtol_state
"""
from __future__ import annotations
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, ExtendedState

_MAVROS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

from fc_bridge.comm.vehicle_state import VehicleState
from fc_ros.adapters.vehicle_state_bridge import (
    update_from_pose,
    update_from_twist,
    update_from_mavros_state,
    update_from_extended_state,
)


class TelemetryNode(Node):

    def __init__(self):
        super().__init__("telemetry_node")

        self._state = VehicleState()
        self._lock = threading.Lock()

        self.create_subscription(
            PoseStamped,
            "/mavros/local_position/pose",
            self._cb_pose, _MAVROS_QOS)
        self.create_subscription(
            TwistStamped,
            "/mavros/local_position/velocity_local",
            self._cb_twist, _MAVROS_QOS)
        self.create_subscription(
            State,
            "/mavros/state",
            self._cb_state, _MAVROS_QOS)
        self.create_subscription(
            ExtendedState,
            "/mavros/extended_state",
            self._cb_extended, _MAVROS_QOS)

    def get_state(self) -> VehicleState:
        """현재 기체 상태의 thread-safe 스냅샷 반환."""
        with self._lock:
            return self._state.copy()

    def _cb_pose(self, msg: PoseStamped) -> None:
        with self._lock:
            update_from_pose(self._state, msg)

    def _cb_twist(self, msg: TwistStamped) -> None:
        with self._lock:
            update_from_twist(self._state, msg)

    def _cb_state(self, msg: State) -> None:
        with self._lock:
            update_from_mavros_state(self._state, msg)

    def _cb_extended(self, msg: ExtendedState) -> None:
        with self._lock:
            update_from_extended_state(self._state, msg)


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
