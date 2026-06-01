"""
ROS2 메시지 → VehicleState 변환 어댑터.

TelemetryNode와 OffboardNode 양쪽에서 공용으로 사용한다.
노드 로직과 메시지 타입 변환을 분리하기 위해 독립 모듈로 존재한다.
"""
from __future__ import annotations
import time

import numpy as np
from tf_transformations import euler_from_quaternion

from fc_bridge.comm.vehicle_state import VehicleState


def update_from_pose(state: VehicleState, msg) -> None:
    """geometry_msgs/PoseStamped → VehicleState 위치·자세 갱신.

    MAVROS는 ENU 프레임으로 발행한다 (ROS 표준).
    fc_bridge는 NED 기반이므로 변환이 필요하다.

    ENU → NED 위치:
      N = y_enu,  E = x_enu,  h_up = z_enu

    ENU → NED 헤딩:
      yaw_ned = pi/2 - yaw_enu
      (ENU yaw=0 은 동쪽, NED yaw=0 은 북쪽)
    """
    p = msg.pose.position
    # ENU(x=E, y=N, z=U) → NED 저장 형식 [N, E, h_up]
    state.pos_ned = np.array([p.y, p.x, p.z])

    q = msg.pose.orientation
    roll_enu, pitch_enu, yaw_enu = euler_from_quaternion([q.x, q.y, q.z, q.w])
    # ENU roll/pitch → NED roll/pitch 부호 변환 (우선회 양수 유지)
    state.roll  = float(roll_enu)
    state.pitch = float(-pitch_enu)
    # ENU yaw → NED yaw (북쪽 기준으로 변환 후 [-π, π] 정규화)
    yaw_ned = np.pi / 2.0 - yaw_enu
    state.yaw = float((yaw_ned + np.pi) % (2 * np.pi) - np.pi)
    state.timestamp = time.monotonic()


def update_from_twist(state: VehicleState, msg) -> None:
    """geometry_msgs/TwistStamped → VehicleState 속도 갱신.

    MAVROS velocity_local 역시 ENU 프레임.
    ENU(vx=vE, vy=vN, vz=vU) → NED [vN, vE, vD]
    """
    v = msg.twist.linear
    state.vel_ned = np.array([v.y, v.x, -v.z])
    state.timestamp = time.monotonic()


def update_from_mavros_state(state: VehicleState, msg) -> None:
    """mavros_msgs/State → VehicleState armed 갱신."""
    state.armed = bool(msg.armed)
    state.timestamp = time.monotonic()


def update_from_extended_state(state: VehicleState, msg) -> None:
    """mavros_msgs/ExtendedState → VehicleState vtol_state 갱신."""
    state.vtol_state = int(msg.vtol_state)
    state.timestamp = time.monotonic()
