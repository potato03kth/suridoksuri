import math
import numpy as np


def pixel_to_gps(
    px: int, py: int,
    image_width: int, image_height: int,
    drone_lat: float, drone_lon: float, drone_alt_m: float,
    camera_fov_deg: float = 60.0,
) -> tuple[float, float]:
    """
    픽셀 좌표 → GPS 좌표 (간이 평면 투영).
    카메라가 정면 아래를 향하고 롤/피치가 0인 이상적인 경우.
    드론 FC 연동 시 실제 자세(roll/pitch)를 추가 인자로 받도록 확장 필요.
    """
    fov_rad = math.radians(camera_fov_deg)
    ground_width_m = 2 * drone_alt_m * math.tan(fov_rad / 2)
    ground_height_m = ground_width_m * (image_height / image_width)

    dx_m = ((px / image_width) - 0.5) * ground_width_m
    dy_m = ((py / image_height) - 0.5) * ground_height_m

    # 위도 1도 ≈ 111_320 m
    dlat = dy_m / 111_320.0
    dlon = dx_m / (111_320.0 * math.cos(math.radians(drone_lat)))

    return drone_lat + dlat, drone_lon + dlon
