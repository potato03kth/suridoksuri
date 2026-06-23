"""
vel_ned ndarray → geometry_msgs/TwistStamped 발행 어댑터.

기존 OffboardFollower._send_velocity() 대체.
"""
from __future__ import annotations
import numpy as np
from geometry_msgs.msg import TwistStamped


class SetpointPublisher:
    """
    Parameters
    ----------
    publisher : rclpy Publisher
        /mavros/setpoint_velocity/cmd_vel 퍼블리셔
    frame_id : str
        TwistStamped header frame_id.
        MAVROS 버전에 따라 "base_link" 또는 "local_origin" — CC 환경에서 확인 후 설정.
    """

    def __init__(self, publisher, frame_id: str = "base_link"):
        self._pub = publisher
        self._frame_id = frame_id

    def publish(self, vel_ned: np.ndarray, yaw_rate: float = 0.0) -> None:
        """vel_ned shape (3,) [vN, vE, vD] → TwistStamped 발행.

        MAVROS setpoint_velocity/cmd_vel 은 ENU 프레임을 기대한다.
        NED [vN, vE, vD] → ENU [vE, vN, vU]

        yaw_rate: ENU angular.z (rad/s). CCW 양수, CW 음수.
                  NED heading_err 양수(CW 회전 필요) → 음수 전달.
        """
        msg = TwistStamped()
        msg.header.frame_id = self._frame_id
        msg.twist.linear.x = float(vel_ned[1])   # vE  → x_enu
        msg.twist.linear.y = float(vel_ned[0])   # vN  → y_enu
        msg.twist.linear.z = float(-vel_ned[2])  # -vD → z_enu (Up)
        msg.twist.angular.z = float(yaw_rate)
        self._pub.publish(msg)
