# 자동분석 — logs/2026-07-25_flight19

(analyze_flight.py 자동생성 — 해석 없이 사실만. notes.md에 결론 작성 시 이 파일 인용)

## 주 로그: `log_203_2026-07-25-11-27-10.ulg` (11.75s)

### 모터 배치 (CA_ROTOR 파라미터 기반)
- motor0 = 전우 (PX=1.0, PY=1.0, KM=0.05000000074505806)
- motor1 = 후좌 (PX=-1.0, PY=-1.0, KM=0.05000000074505806)
- motor2 = 전좌 (PX=1.0, PY=-1.0, KM=-0.05000000074505806)
- motor3 = 후우 (PX=-1.0, PY=1.0, KM=-0.05000000074505806)
- motor4 = 중중 (PX=0.0, PY=0.0, KM=0.05000000074505806)

### 주요 파라미터
- `BAT_LOW_THR` = 0.15000000596046448
- `BAT_CRIT_THR` = 0.07000000029802322
- `BAT_EMERGEN_THR` = 0.05000000074505806
- `FD_IMB_PROP_THR` = 30
- `MPC_TKO_SPEED` = 1.5
- `COM_ARM_WO_GPS` = 1
- `EKF2_HGT_REF` = 1

### 고도
- home_position AMSL=21.13m, EKF ref_alt AMSL=22.0m
- AGL 최대 -0.0m (t=0.06s), 최소 -0.37m (t=11.67s), 종료시 -0.37m
- notes.md 목표 transition_alt=20.0m → **미달**

### 모드권한 타임라인
| t(s) | nav_state | user_intention | failsafe | arming | 분류 |
|---|---|---|---|---|---|
| 0.5 | 2:POSCTL | 2:POSCTL | 0 | ARMED | AUTO_RECOVERY_OR_UNCLASSIFIED |
| 0.91 | 17:AUTO_TAKEOFF | 17:AUTO_TAKEOFF | 0 | ARMED | INTENTIONAL_CHANGE |
| 5.43 | 4:AUTO_LOITER | 17:AUTO_TAKEOFF | 1 | ARMED | FAILSAFE_FORCED |
| 10.43 | 5:AUTO_RTL | 17:AUTO_TAKEOFF | 1 | ARMED | FAILSAFE_FORCED |
| 10.75 | 2:POSCTL | 2:POSCTL | 0 | STANDBY | INTENTIONAL_CHANGE |

> INTENTIONAL_CHANGE는 조종기/지상국/컴패니언 커맨드를 다 포함 — 출처를 더 좁히려면 manual_control_switches/vehicle_command를 별도로 대조할 것. FAILSAFE_FORCED는 PX4 자체 강제전환이 확실함(user_intention 불변).

### 자세
- roll 최대|값|=5.14° (t=5.82s)
- pitch 최대|값|=-5.67° (t=4.42s)
- roll 목표(setpoint) 최대|값|=5.49° (t=4.31s) — 이 값이 실측 roll_deg_max_abs 근처까지 크면 세트포인트 자체 결함(순간점프 등) 의심 — 0 근처로 작으면 컨트롤러는 정상 수평을 명령했는데 실제가 못 따라간 것.
- roll rate 이상조짐 후보: t=4.09s (-10.0deg/s)
- pitch rate 이상조짐 후보: t=4.35s (-24.8deg/s)
- yaw rate 이상조짐 후보: t=4.35s (-13.0deg/s)
> 휴리스틱 후보일 뿐 — 잡음(±5deg/s 이내)에서 15deg/s 이상으로 튄 첫 시점. 이 근처를 §5 얼로케이터 포화 시점·§9 이함 시점과 대조해 진짜 트리거인지 육안 확인할 것.

### 얼로케이터 포화 (축별)
- roll(x): 비행 내내 지속적 포화 없음(arm 직후 단발 블립은 제외)
- pitch(y): 비행 내내 지속적 포화 없음(arm 직후 단발 블립은 제외)
- yaw(z): 비행 내내 지속적 포화 없음(arm 직후 단발 블립은 제외)
- 통합 achieved 플래그 True 비율: 0.0
> 이 플래그는 roll/pitch/yaw/thrust 통합 — 축별 판단은 위 unallocated_torque[]를 볼 것. 포화 시작 시점을 그 순간의 attitude.roll_deg_max_abs_at_s 등과 비교해 '오차가 이미 컸을 때부터 포화가 시작됐는지'를 확인하라 — 그렇다면 포화는 원인이 아니라 결과일 가능성이 크다.

### 모터별 커맨드 (0~1)
- 0:전우: mean=0.66 max=1.0 min=0.002
- 1:후좌: mean=0.153 max=0.552 min=0.0
- 2:전좌: mean=0.405 max=1.0 min=0.002
- 3:후우: mean=0.395 max=0.713 min=0.0
- 4:중중: mean=nan max=nan min=nan

### 배터리
- 휴지전압 15.57V, 부하중 최저 13.96V, 전류 최대 17.3A
- remaining: 0.48 → 0.44

### 실패감지기
- imbalanced_prop_metric_max_abs: -2.141
- fd_imbalanced_prop_ever_nonzero: False
- fd_motor_ever_nonzero: False
- motor_failure_mask_ever_nonzero: False

### 이함(liftoff) 순간
- ground_contact 1→0 시점: t=1.91s
| t(s) | roll(deg) | stick_roll |
|---|---|---|
| 1.91 | 3.16 | 0.0 |
| 2.41 | 3.17 | 0.0 |
| 2.91 | 3.23 | 0.0 |
| 3.41 | 3.18 | 0.0 |
| 3.91 | 2.15 | 0.0 |
| 4.41 | 1.07 | 0.0 |
| 4.91 | 1.63 | 0.0 |

### 이 스크립트가 다루지 않은 토픽 (필요시 직접 확인)
`action_request`, `actuator_armed`, `actuator_outputs`, `actuator_outputs(#1)`, `actuator_servos`, `airspeed`, `airspeed_validated`, `airspeed_wind`, `airspeed_wind(#1)`, `can_interface_status`, `can_interface_status(#1)`, `config_overrides`, `control_allocator_status(#1)`, `cpuload`, `differential_pressure`, `distance_sensor_mode_change_request`, `estimator_aid_src_airspeed`, `estimator_aid_src_airspeed(#1)`, `estimator_aid_src_baro_hgt`, `estimator_aid_src_baro_hgt(#1)`, `estimator_aid_src_gnss_hgt`, `estimator_aid_src_gnss_hgt(#1)`, `estimator_aid_src_gnss_pos`, `estimator_aid_src_gnss_pos(#1)`, `estimator_aid_src_gnss_vel`, `estimator_aid_src_gnss_vel(#1)`, `estimator_aid_src_gravity`, `estimator_aid_src_gravity(#1)`, `estimator_aid_src_mag`, `estimator_aid_src_mag(#1)`, `estimator_attitude`, `estimator_attitude(#1)`, `estimator_baro_bias`, `estimator_baro_bias(#1)`, `estimator_event_flags`, `estimator_event_flags(#1)`, `estimator_fusion_control`, `estimator_global_position`, `estimator_global_position(#1)`, `estimator_gps_status`, `estimator_gps_status(#1)`, `estimator_innovation_test_ratios`, `estimator_innovation_test_ratios(#1)`, `estimator_innovation_variances`, `estimator_innovation_variances(#1)`, `estimator_innovations`, `estimator_innovations(#1)`, `estimator_local_position`, `estimator_local_position(#1)`, `estimator_odometry`, `estimator_odometry(#1)`, `estimator_selector_status`, `estimator_sensor_bias`, `estimator_sensor_bias(#1)`, `estimator_states`, `estimator_states(#1)`, `estimator_status`, `estimator_status(#1)`, `estimator_status_flags`, `estimator_status_flags(#1)`, `event`, `failsafe_flags`, `fixed_wing_lateral_guidance_status`, `fixed_wing_lateral_setpoint`, `fixed_wing_lateral_status`, `fixed_wing_longitudinal_setpoint`, `flaps_setpoint`, `flight_phase_estimation`, `gain_compression`, `input_rc`, `landing_gear`, `landing_gear_wheel`, `lateral_control_configuration`, `logger_status`, `longitudinal_control_configuration`, `magnetometer_bias_estimate`, `manual_control_switches`, `mission_result`, `navigator_mission_item`, `navigator_status`, `position_setpoint_triplet`, `px4io_status`, `rate_ctrl_status`, `rate_ctrl_status(#1)`, `rtl_status`, `rtl_time_estimate`, `sensor_accel`, `sensor_accel(#1)`, `sensor_baro`, `sensor_combined`, `sensor_gps`, `sensor_gyro`, `sensor_gyro(#1)`, `sensor_mag`, `sensor_mag(#1)`, `sensors_status_imu`, `spoilers_setpoint`, `system_power`, `takeoff_status`, `tecs_status`, `telemetry_status`, `timesync_status`, `trajectory_setpoint`, `vehicle_acceleration`, `vehicle_air_data`, `vehicle_command`, `vehicle_command_ack`, `vehicle_constraints`, `vehicle_control_mode`, `vehicle_global_position`, `vehicle_gps_position`, `vehicle_imu`, `vehicle_imu(#1)`, `vehicle_imu_status`, `vehicle_imu_status(#1)`, `vehicle_local_position_setpoint`, `vehicle_magnetometer`, `vehicle_rates_setpoint`, `vehicle_thrust_setpoint`, `vehicle_thrust_setpoint(#1)`, `vehicle_torque_setpoint`, `vehicle_torque_setpoint(#1)`, `vtol_vehicle_status`, `yaw_estimator_status`, `yaw_estimator_status(#1)`
