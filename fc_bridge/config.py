"""
fc_bridge 전역 설정.
실행 전 이 파일의 값을 환경에 맞게 수정한다.
"""

# ── 연결 ────────────────────────────────────────────────────
# SITL : "udp:127.0.0.1:14550"
# 실기체: "/dev/ttyACM0" 또는 "COM3" 등
CONNECTION_STR = "udp:127.0.0.1:14550"
BAUD = 57600

# ── 기체 파라미터 ────────────────────────────────────────────
VEHICLE_PARAMS = {
    "v_cruise": 15.0,   # m/s
    "a_max_g": 0.3,     # g 단위 횡방향 가속도 상한
    "gravity": 9.81,
}

# ── Offboard 제어 ────────────────────────────────────────────
CONTROL_HZ = 10          # 세트포인트 송신 주파수 (PX4 watchdog: ≥2Hz)
L1_DISTANCE = 20.0       # L1 look-ahead 거리 (m)

# Offboard 진입 시점 선택 (실행 전 결정)
#   "pre_takeoff" : arm 직후 Offboard 전환 → 이륙부터 알고리즘 제어
#   "mid_flight"  : 비행 중 Offboard 전환 → WP0 진입 기동 후 경로 추종
OFFBOARD_ENTRY_MODE = "pre_takeoff"

# mid_flight 전용 WP0 진입 판정 기준
WP0_ENTRY_RADIUS = 5.0    # WP0 도달 판정 반경 (m)
WP0_HEADING_TOL  = 0.2    # 진입 헤딩 허용 오차 (rad)

# ── 위치 오차 기반 가속도 제한 ────────────────────────────────
ERROR_STALL_STEPS = 20    # 오차 개선 없음 판정 스텝 수
ACCEL_REDUCTION_FACTOR = 0.9  # 오차 정체 시 a_max 감소 비율 (매 판정마다)
ACCEL_MIN_FRACTION = 0.3  # a_max 최소 비율 (원래 값 대비)
