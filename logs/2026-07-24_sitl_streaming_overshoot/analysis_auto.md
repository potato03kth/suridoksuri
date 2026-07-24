# 자동분석 — logs/2026-07-24_sitl_streaming_overshoot

(analyze_flight.py 자동생성 — 해석 없이 사실만. notes.md에 결론 작성 시 이 파일 인용)

## 주 로그: `05_07_03.ulg` (29.2s)

### 모터 배치 (CA_ROTOR 파라미터 기반)
- motor0 = 전우 (PX=0.12999999523162842, PY=0.2199999988079071, KM=0.05000000074505806)
- motor1 = 후좌 (PX=-0.12999999523162842, PY=-0.20000000298023224, KM=0.05000000074505806)
- motor2 = 전좌 (PX=0.12999999523162842, PY=-0.2199999988079071, KM=-0.05000000074505806)
- motor3 = 후우 (PX=-0.12999999523162842, PY=0.20000000298023224, KM=-0.05000000074505806)

### 주요 파라미터
- `BAT_LOW_THR` = 0.15000000596046448
- `BAT_CRIT_THR` = 0.07000000029802322
- `BAT_EMERGEN_THR` = 0.05000000074505806
- `FD_IMB_PROP_THR` = 30
- `MPC_TKO_SPEED` = 1.5
- `COM_ARM_WO_GPS` = 1
- `EKF2_HGT_REF` = 1

### 고도
- home_position AMSL=0.25m, EKF ref_alt AMSL=0.25m
- AGL 최대 5.54m (t=13.52s), 최소 -0.1m (t=25.08s), 종료시 -0.02m
- notes.md 목표 transition_alt=3.0m → **달성**

### 모드권한 타임라인
| t(s) | nav_state | user_intention | failsafe | arming | 분류 |
|---|---|---|---|---|---|
| 0.0 | 14:OFFBOARD | 14:OFFBOARD | 0 | ARMED | AUTO_RECOVERY_OR_UNCLASSIFIED |
| 5.11 | 17:AUTO_TAKEOFF | 17:AUTO_TAKEOFF | 0 | ARMED | INTENTIONAL_CHANGE |
| 13.21 | 14:OFFBOARD | 14:OFFBOARD | 0 | ARMED | INTENTIONAL_CHANGE |
| 15.3 | 18:AUTO_LAND | 18:AUTO_LAND | 0 | ARMED | INTENTIONAL_CHANGE |
| 28.2 | 14:OFFBOARD | 14:OFFBOARD | 0 | STANDBY | INTENTIONAL_CHANGE |

> INTENTIONAL_CHANGE는 조종기/지상국/컴패니언 커맨드를 다 포함 — 출처를 더 좁히려면 manual_control_switches/vehicle_command를 별도로 대조할 것. FAILSAFE_FORCED는 PX4 자체 강제전환이 확실함(user_intention 불변).

### 자세
- roll 최대|값|=-0.55° (t=10.5s)
- pitch 최대|값|=-0.61° (t=9.88s)
- roll 목표(setpoint) 최대|값|=-0.68° (t=10.2s) — 이 값이 실측 roll_deg_max_abs 근처까지 크면 세트포인트 자체 결함(순간점프 등) 의심 — 0 근처로 작으면 컨트롤러는 정상 수평을 명령했는데 실제가 못 따라간 것.
> 휴리스틱 후보일 뿐 — 잡음(±5deg/s 이내)에서 15deg/s 이상으로 튄 첫 시점. 이 근처를 §5 얼로케이터 포화 시점·§9 이함 시점과 대조해 진짜 트리거인지 육안 확인할 것.

### 얼로케이터 포화 (축별)
- roll(x): 비행 내내 지속적 포화 없음(arm 직후 단발 블립은 제외)
- pitch(y): 비행 내내 지속적 포화 없음(arm 직후 단발 블립은 제외)
- yaw(z): 비행 내내 지속적 포화 없음(arm 직후 단발 블립은 제외)
- 통합 achieved 플래그 True 비율: 1.0
> 이 플래그는 roll/pitch/yaw/thrust 통합 — 축별 판단은 위 unallocated_torque[]를 볼 것. 포화 시작 시점을 그 순간의 attitude.roll_deg_max_abs_at_s 등과 비교해 '오차가 이미 컸을 때부터 포화가 시작됐는지'를 확인하라 — 그렇다면 포화는 원인이 아니라 결과일 가능성이 크다.

### 모터별 커맨드 (0~1)
- 0:전우: mean=0.462 max=0.888 min=0.0
- 1:후좌: mean=0.462 max=0.889 min=0.001
- 2:전좌: mean=0.462 max=0.889 min=0.0
- 3:후우: mean=0.462 max=0.889 min=0.001

### 배터리
- 휴지전압 16.2V, 부하중 최저 15.35V, 전류 최대 -1.0A
- remaining: 1.0 → 0.8

### 실패감지기
- imbalanced_prop_metric_max_abs: -1.073
- fd_imbalanced_prop_ever_nonzero: False
- fd_motor_ever_nonzero: False
- motor_failure_mask_ever_nonzero: False

### 이함(liftoff) 순간
- ground_contact 1→0 시점: t=6.04s
| t(s) | roll(deg) | stick_roll |
|---|---|---|
| 6.04 | -0.0 | - |
| 6.54 | -0.0 | - |
| 7.04 | -0.02 | - |
| 7.54 | -0.02 | - |
| 8.04 | -0.02 | - |
| 8.54 | 0.09 | - |
| 9.04 | 0.02 | - |

### 이 스크립트가 다루지 않은 토픽 (필요시 직접 확인)
`actuator_armed`, `actuator_outputs`, `config_overrides`, `cpuload`, `ekf2_timestamps`, `esc_status`, `estimator_aid_src_baro_hgt`, `estimator_aid_src_gnss_hgt`, `estimator_aid_src_gnss_pos`, `estimator_aid_src_gnss_vel`, `estimator_aid_src_gravity`, `estimator_aid_src_mag`, `estimator_baro_bias`, `estimator_event_flags`, `estimator_fusion_control`, `estimator_gps_status`, `estimator_innovation_test_ratios`, `estimator_innovation_variances`, `estimator_innovations`, `estimator_sensor_bias`, `estimator_states`, `estimator_status`, `estimator_status_flags`, `event`, `failsafe_flags`, `hover_thrust_estimate`, `landing_gear`, `logger_status`, `mission_result`, `navigator_status`, `offboard_control_mode`, `position_setpoint_triplet`, `rate_ctrl_status`, `rtl_status`, `rtl_time_estimate`, `sensor_accel`, `sensor_baro`, `sensor_combined`, `sensor_gps`, `sensor_gyro`, `sensor_mag`, `sensors_status_imu`, `system_power`, `takeoff_status`, `telemetry_status`, `telemetry_status(#1)`, `telemetry_status(#2)`, `telemetry_status(#3)`, `timesync_status`, `trajectory_setpoint`, `vehicle_acceleration`, `vehicle_air_data`, `vehicle_angular_velocity_groundtruth`, `vehicle_attitude_groundtruth`, `vehicle_command`, `vehicle_command_ack`, `vehicle_constraints`, `vehicle_control_mode`, `vehicle_global_position`, `vehicle_global_position_groundtruth`, `vehicle_gps_position`, `vehicle_imu`, `vehicle_imu_status`, `vehicle_local_position_groundtruth`, `vehicle_local_position_setpoint`, `vehicle_magnetometer`, `vehicle_rates_setpoint`, `vehicle_thrust_setpoint`, `vehicle_torque_setpoint`, `yaw_estimator_status`
