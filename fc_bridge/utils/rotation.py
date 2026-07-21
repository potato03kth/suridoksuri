"""
쿼터니언 ↔ 오일러 변환 유틸리티.

외부 라이브러리(tf_transformations, scipy, transforms3d) 없이 numpy만으로 구현.
NumPy 버전에 무관하게 동작한다.
"""
from __future__ import annotations
import numpy as np


def quat_to_euler_xyz(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """쿼터니언 [w, x, y, z] → (roll, pitch, yaw) 라디안.

    tf_transformations.euler_from_quaternion(axes='sxyz') 와 동일한 ZYX 컨벤션.
    gimbal lock 근방에서 pitch를 ±π/2로 클램핑한다.
    """
    roll  = float(np.arctan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y)))
    pitch = float(np.arcsin(np.clip(2.0 * (w*y - z*x), -1.0, 1.0)))
    yaw   = float(np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z)))
    return roll, pitch, yaw


def yaw_ned_to_quat_enu(yaw_ned: float) -> tuple[float, float, float, float]:
    """NED 헤딩(rad, 0=북·양수=동) → ENU PoseStamped용 쿼터니언 [w, x, y, z].

    roll=pitch=0(수평 가정) 순수 yaw 쿼터니언. `vehicle_state_bridge.py`의
    디코드(`yaw_ned = pi/2 - yaw_enu`)와 정확히 역변환 관계 — 이 왕복이
    깨지면 setpoint 헤딩이 실제 기체 헤딩과 어긋난다(2026-07-21 flight04
    yaw 스핀 사고: orientation 미설정 시 ENU 기본값 yaw=0=NED yaw=90°가
    실제 헤딩과 무관하게 그대로 발행됨).
    """
    yaw_enu = np.pi / 2.0 - yaw_ned
    w = float(np.cos(yaw_enu / 2.0))
    z = float(np.sin(yaw_enu / 2.0))
    return w, 0.0, 0.0, z
