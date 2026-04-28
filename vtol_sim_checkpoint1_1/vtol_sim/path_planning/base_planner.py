"""
경로 생성 추상 인터페이스
============================
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PathPoint:
    """경로 위의 한 점 — 호 길이 매개변수화."""
    pos: np.ndarray = field(default_factory=lambda: np.zeros(
        3))  # [x_N, x_E, h(표준 NED와 부호 반대)]
    v_ref: float = 0.0           # 계획된 속도 (m/s)
    chi_ref: float = 0.0         # 계획된 방위각 (rad)
    gamma_ref: float = 0.0       # 계획된 상승각 (rad)
    curvature: float = 0.0       # 곡률 (1/m), 좌선회 음 / 우선회 양 부호
    s: float = 0.0               # 호 길이 (m), 경로 시작에서부터의 누적 거리

    # WP 마커 — 이 점이 어느 WP인지 (또는 사이 점이면 None)
    wp_index: int | None = None


@dataclass
class Path:
    """완전한 경로 — PathPoint 시퀀스 + 메타데이터."""
    points: list[PathPoint] = field(default_factory=list)
    waypoints_ned: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # waypoints_ned[i]는 원본 WP, points[k]에서 wp_index==i인 점들이 그 WP를 통과
    total_length: float = 0.0
    planning_time: float = 0.0   # s, 경로 생성 wall-clock 시간

    def positions_array(self) -> np.ndarray:
        """모든 점의 위치를 (N, 3) numpy 배열로."""
        return np.array([p.pos for p in self.points])

    def waypoint_indices_in_path(self) -> list[int]:
        """각 WP가 경로 위 몇 번째 점에 위치하는지 — wp_index로 마킹된 점 인덱스."""
        out = []
        for k, p in enumerate(self.points):
            if p.wp_index is not None:
                out.append(k)
        return out


class BasePlanner(ABC):
    """모든 경로 생성 알고리즘의 추상 베이스."""

    @abstractmethod
    def plan(self, waypoints_ned: np.ndarray, aircraft_params: dict,
             initial_state: dict | None = None) -> Path:
        """
        WP들을 입력받아 비행 경로 생성.

        Parameters
        ----------
        waypoints_ned : (N, 3) array
            [x_N, x_E, h] 좌표의 WP 시퀀스
        aircraft_params : dict
            기체 파라미터 (v_cruise, phi_max_deg, a_max_g 등)
        initial_state : dict, optional
            초기 자세 등 추가 정보 (예: 'initial_heading')

        Returns
        -------
        Path
        """
