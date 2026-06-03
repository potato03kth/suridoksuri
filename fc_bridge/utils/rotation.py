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
