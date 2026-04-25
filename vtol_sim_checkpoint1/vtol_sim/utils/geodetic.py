"""
지오데틱 변환 유틸 — flat-earth 근사
=====================================

작은 운항 영역(<수 km)에 대해 위/경도와 지역 NED 좌표 사이를 변환합니다.
WGS84 적도반경을 사용한 단순 변환.
"""
from __future__ import annotations
import numpy as np


R_EARTH = 6_378_137.0  # m, WGS84 적도반경


def geodetic_to_ned(lat_deg: float, lon_deg: float, alt: float,
                    ref_lat_deg: float, ref_lon_deg: float, ref_alt: float
                    ) -> np.ndarray:
    """
    (위도, 경도, 고도) → 지역 NED [x_N, x_E, h]

    Parameters
    ----------
    lat_deg, lon_deg : float
        대상 점의 위도, 경도 (도 단위)
    alt : float
        대상 점의 고도 (m)
    ref_lat_deg, ref_lon_deg, ref_alt : float
        기준점

    Returns
    -------
    np.ndarray shape (3,)
        [x_N, x_E, h] (m). h는 양수 = 위쪽. (NED의 z는 down이지만 h = -z로 출력)
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0 = np.deg2rad(ref_lat_deg)
    lon0 = np.deg2rad(ref_lon_deg)

    x_N = R_EARTH * (lat - lat0)
    x_E = R_EARTH * np.cos(lat0) * (lon - lon0)
    h = alt - ref_alt
    return np.array([x_N, x_E, h], dtype=float)


def ned_to_geodetic(x_N: float, x_E: float, h: float,
                    ref_lat_deg: float, ref_lon_deg: float, ref_alt: float
                    ) -> tuple[float, float, float]:
    """
    지역 NED → (위도, 경도, 고도)
    """
    lat0 = np.deg2rad(ref_lat_deg)
    lat = lat0 + x_N / R_EARTH
    lon = np.deg2rad(ref_lon_deg) + x_E / (R_EARTH * np.cos(lat0))
    alt = h + ref_alt
    return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(alt)


def waypoints_to_ned(wps_geodetic: list[tuple[float, float, float]],
                     ref_lat_deg: float, ref_lon_deg: float, ref_alt: float
                     ) -> np.ndarray:
    """리스트 변환. shape (N, 3)."""
    out = np.zeros((len(wps_geodetic), 3))
    for i, (lat, lon, alt) in enumerate(wps_geodetic):
        out[i] = geodetic_to_ned(lat, lon, alt, ref_lat_deg, ref_lon_deg, ref_alt)
    return out
