"""
Waypoint 자동 생성기
======================

운영 영역 내에서 N개의 waypoint를 무작위로 생성. 다음 제약을 만족:
- 모든 WP가 area_size × area_size 정사각형 영역 내
- 인접한 WP 간 거리 >= min_separation
- 모든 WP 쌍 간 거리 >= min_separation (혹시 모를 회로 형태에서도)
- 고도는 [altitude_range[0], altitude_range[1]]
- 시작점(첫 WP)은 영역 중심에서 시작하여 다양한 패턴 생성
- 시드로 재현성 확보

생성 알고리즘: rejection sampling (단순하지만 6개 정도는 충분히 빠름)
"""
from __future__ import annotations
import numpy as np


def generate_waypoints(
    n_waypoints: int = 6,
    area_size: float = 1500.0,
    min_separation: float = 330.0,
    altitude_range: tuple[float, float] = (120.0, 200.0),
    seed: int | None = 42,
    max_attempts: int = 10000,
) -> np.ndarray:
    """
    N개의 WP를 자동 생성.

    Parameters
    ----------
    n_waypoints : 생성할 WP 개수
    area_size : 사각형 영역의 한 변 (m). 영역 중심 = (0, 0).
    min_separation : 모든 WP 쌍 간 최소 거리 (m).
    altitude_range : (h_min, h_max) m
    seed : 재현성을 위한 시드
    max_attempts : 거부 샘플링 최대 시도 횟수 (실패 시 예외)

    Returns
    -------
    waypoints : (N, 3) array of [x_N, x_E, h]
    """
    rng = np.random.default_rng(seed)
    half = area_size / 2.0
    h_min, h_max = altitude_range

    # 첫 WP는 영역 중심 근처에서 시작 (이륙 지점 가정)
    waypoints = [np.array([0.0, 0.0, (h_min + h_max) / 2.0])]

    attempts = 0
    while len(waypoints) < n_waypoints and attempts < max_attempts:
        # 후보 생성
        x_N = rng.uniform(-half, half)
        x_E = rng.uniform(-half, half)
        h = rng.uniform(h_min, h_max)
        candidate = np.array([x_N, x_E, h])

        # 모든 기존 WP와의 거리 체크 (수평 거리만)
        ok = True
        for wp in waypoints:
            d_horiz = np.linalg.norm(candidate[:2] - wp[:2])
            if d_horiz < min_separation:
                ok = False
                break

        if ok:
            waypoints.append(candidate)

        attempts += 1

    if len(waypoints) < n_waypoints:
        raise RuntimeError(
            f"Failed to generate {n_waypoints} waypoints with "
            f"min_separation={min_separation}m in area {area_size}m. "
            f"Try smaller min_separation or larger area."
        )

    return np.array(waypoints)


def waypoints_from_config(scenario_cfg: dict) -> np.ndarray:
    """
    시나리오 설정에서 waypoints 추출. source가 'auto'면 자동 생성, 'manual'이면 직접 사용.
    """
    wp_cfg = scenario_cfg["waypoint"]
    source = wp_cfg.get("source", "auto")

    if source == "auto":
        a = wp_cfg["auto"]
        return generate_waypoints(
            n_waypoints=a["n_waypoints"],
            area_size=a["area_size"],
            min_separation=a["min_separation"],
            altitude_range=tuple(a["altitude_range"]),
            seed=a.get("seed"),
        )
    elif source == "manual":
        manual = wp_cfg["manual"]
        if not manual:
            raise ValueError("waypoint.source=manual인데 waypoint.manual 리스트가 비어있음")
        return np.array(manual, dtype=float)
    else:
        raise ValueError(f"알 수 없는 waypoint source: {source}")
