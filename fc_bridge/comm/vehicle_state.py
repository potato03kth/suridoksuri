"""
기체 상태 스냅샷.

telemetry.py에서 분리해 fc_ros에서도 import 가능하도록 독립 모듈로 분리.
"""
from __future__ import annotations
import numpy as np


class VehicleState:
    """기체 상태 스냅샷 (불변 데이터클래스 대신 단순 class)."""
    __slots__ = (
        "pos_ned", "vel_ned",
        "roll", "pitch", "yaw",
        "vtol_state",
        "armed", "base_mode", "custom_mode",
        "battery_voltage", "battery_current", "battery_remaining",
        "timestamp",
    )

    def __init__(self):
        self.pos_ned    = np.zeros(3)  # [N, E, h_up]  (h = -z_ned)
        self.vel_ned    = np.zeros(3)  # [vN, vE, vD]
        self.roll       = 0.0
        self.pitch      = 0.0
        self.yaw        = 0.0
        self.vtol_state = 0            # 0=undefined,1=to_fw,2=to_mc,3=mc,4=fw
        self.armed      = False
        self.base_mode  = 0
        self.custom_mode = 0
        self.battery_voltage   = 0.0  # V
        self.battery_current   = 0.0  # A (음수 = 방전 아님, MAVROS 부호 그대로)
        self.battery_remaining = 1.0  # 0.0~1.0 (미수신 시 1.0 — 저전압 오탐 방지)
        self.timestamp  = 0.0         # time.monotonic() 기준

    @property
    def heading_rad(self) -> float:
        return self.yaw

    @property
    def pos_ned_2d(self) -> np.ndarray:
        return self.pos_ned[:2]

    def copy(self) -> "VehicleState":
        s = VehicleState()
        s.pos_ned     = self.pos_ned.copy()
        s.vel_ned     = self.vel_ned.copy()
        s.roll        = self.roll
        s.pitch       = self.pitch
        s.yaw         = self.yaw
        s.vtol_state  = self.vtol_state
        s.armed       = self.armed
        s.base_mode   = self.base_mode
        s.custom_mode = self.custom_mode
        s.battery_voltage   = self.battery_voltage
        s.battery_current   = self.battery_current
        s.battery_remaining = self.battery_remaining
        s.timestamp   = self.timestamp
        return s
