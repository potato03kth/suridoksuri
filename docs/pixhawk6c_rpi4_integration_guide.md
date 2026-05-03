---
doc_type: integration_guide
target_audience: ai_agent
project: pixhawk6c_rpi4_algorithm_validation
fc_hardware: Holybro Pixhawk 6C (FMUv6C, STM32H743)
fc_firmware: PX4 v1.14+
cc_hardware: Raspberry Pi 4
cc_language: Python 3
mavlink_version: v2
created: 2026-05-03
status: working_document
---

# Pixhawk 6C + Raspberry Pi 4 알고리즘 실증 통합 가이드

> AI 에이전트용 구조화 문서. 사람이 아니라 코딩 LLM이 빠르게 파싱/인용하도록 설계됨.

---

## 0. META: 이 문서를 읽는 AI에게

```yaml
how_to_use:
  - "사용자가 알고리즘 X와 FC 통신을 물으면 §3.X 표를 직접 인용하라."
  - "사용자가 PX4 파라미터를 물으면 §5 표를 인용하라."
  - "코드를 작성하기 전 §6 함정 체크리스트를 반드시 검토하라."
  - "Offboard 코드를 생성할 때는 §3.2의 진입 시퀀스를 그대로 따르라."
  - "메시지 ID는 임의로 만들지 말고 본 문서에 등재된 ID만 사용하라."

scope_in:
  - MAVLink v2 over UART (TELEM2) 기반 통신
  - pymavlink / MAVSDK-Python 사용 가정
  - 4개 알고리즘: 경로생성(eta3 clothoid + 자체) / 경로추종 / 비전 객체인식 / VTOL 천이

scope_out:
  - ROS 2 / uXRCE-DDS 풀스택 (Appendix A에서만 간단 언급)
  - 딥러닝 기반 비전 (사용자 명시적 제외)
  - Ardupilot (PX4 전용)
```

---

## 1. SYSTEM: 시스템 구성 요약

### 1.1 하드웨어 토폴로지

```text
┌──────────────────────────────┐    UART 921600bps    ┌──────────────────────────────┐
│ Companion Computer (CC)      │ ──────TELEM2────────>│ Flight Controller (FC)       │
│ Raspberry Pi 4 (4~8GB)       │ <─────MAVLink v2─────│ Pixhawk 6C / PX4 v1.14+      │
│                              │                       │                              │
│ Python 3 modules:            │                       │ Subsystems:                  │
│  - path_gen_eta3.py          │                       │  - EKF2 (state estimator)    │
│  - path_gen_custom.py        │                       │  - Position Controller       │
│  - nlgl_controller.py        │                       │  - Attitude Controller       │
│  - mpc_controller.py         │                       │  - Mission Manager           │
│  - pixel_to_gps.py (vision)   │                       │  - VTOL Transition Manager   │
│  - vtol_transition.py (TBD)   │                       │  - Commander (mode/arm)      │
└──────────────────────────────┘                       └──────────────────────────────┘
```

### 1.2 통신 미들웨어 결정 매트릭스

| 옵션 | 미들웨어 | RPi4 적합도 | 채택 |
|------|---------|-----------|------|
| A | MAVLink v2 (pymavlink) | 높음 (경량, Python 친화) | ✅ 1차 |
| B | MAVSDK-Python | 중상 (async/await) | 보조 가능 |
| C | uXRCE-DDS + ROS 2 Humble | 중하 (RPi4 부담) | 보류 |

### 1.3 좌표계 정의

| 좌표계 | X | Y | Z | 사용처 |
|--------|---|---|---|--------|
| NED | 북 | 동 | 하 | PX4 내부, MAVLink LOCAL_NED |
| ENU | 동 | 북 | 상 | ROS 표준 |
| Body (FRD) | 전 | 우 | 하 | IMU, body rate |
| Camera (OpenCV) | 우 | 하 | 전 | OpenCV 픽셀→3D |

변환 규칙: Camera → ENU → NED 순서.

---

## 2. PROTOCOL: 저수준 프로토콜

### 2.1 MAVLink 채널 설정

```yaml
physical_link:
  port: TELEM2
  type: UART
  baudrate: 921600
  flow_control: enabled

px4_parameters:
  MAV_1_CONFIG: 102
  SER_TEL2_BAUD: 921600
  MAV_PROTO_VER: 2
  MAV_1_RATE: 1200000
```

### 2.2 메시지 권장 스트림 레이트

| 메시지 | ID | 방향 | 권장 Hz | 사용 |
|--------|-----|------|---------|------|
| HEARTBEAT | 0 | 양방향 | 1 | 모든 모듈 |
| ATTITUDE | 30 | FC→CC | 50 | 추종/비전/VTOL |
| ATTITUDE_QUATERNION | 31 | FC→CC | 50 | 추종/VTOL |
| LOCAL_POSITION_NED | 32 | FC→CC | 50 | 경로생성/추종 |
| GLOBAL_POSITION_INT | 33 | FC→CC | 10 | 경로생성/비전 |
| HIGHRES_IMU | 105 | FC→CC | 100 | 추종/VTOL |
| EKF_STATUS_REPORT | 193 | FC→CC | 1 | 경로생성 |
| EXTENDED_SYS_STATE | 245 | FC→CC | 1~5 | VTOL |
| GIMBAL_DEVICE_ATTITUDE_STATUS | 285 | FC→CC | 10 | 비전 |

```python
# 스트림 레이트 설정
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
    0, 33, 100000, 0, 0, 0, 0, 0
)
```

---

## 3. ALGORITHMS: 알고리즘별 인터페이스

### 3.1 공통 규칙

```yaml
watchdog:
  rule: "Offboard setpoint < 2Hz 이면 PX4가 즉시 Position 모드로 복귀"
  countermeasure: "별도 스레드에서 최소 10Hz 유지"

mode_entry_order:
  - "1. setpoint 사전 스트리밍 ≥ 1초 (≥ 2Hz)"
  - "2. MAV_CMD_DO_SET_MODE → custom_mode=6 (OFFBOARD)"
  - "3. MAV_CMD_COMPONENT_ARM_DISARM param1=1"
  - "4. setpoint 지속 전송 유지"

ekf_precondition:
  - "EKF_STATUS_REPORT.flags bit4 = 1 확인 후 임무 시작"
```

### 3.2 Offboard 진입 정규 코드

```python
import time
from pymavlink import mavutil

PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6
BASE_MODE_OFFBOARD = 209

def enter_offboard_and_arm(master, send_setpoint_callable, hz=20.0):
    # 1. Pre-stream setpoints for >= 1s
    t0 = time.time()
    while time.time() - t0 < 1.5:
        send_setpoint_callable()
        time.sleep(1.0 / hz)

    # 2. Set OFFBOARD
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        BASE_MODE_OFFBOARD, PX4_CUSTOM_MAIN_MODE_OFFBOARD,
        0, 0, 0, 0, 0
    )

    # 3. ARM
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1, 0, 0, 0, 0, 0, 0
    )
```

### 3.3 알고리즘 ① 경로생성 (eta3 clothoid + 자체)

```yaml
role: "Waypoint 또는 (NED, v_ref, chi_ref, curvature) 경로점 생성"
output_format: "PathPoint(pos[N,E,D], v_ref, chi_ref, curvature)"
direct_fc_communication: false
```

| 방향 | 데이터 | 메시지 | ID | 주기 |
|------|--------|--------|-----|------|
| FC→CC | NED 위치 | LOCAL_POSITION_NED | 32 | 10Hz |
| FC→CC | GPS | GLOBAL_POSITION_INT | 33 | 10Hz |
| FC→CC | EKF 상태 | EKF_STATUS_REPORT | 193 | 1Hz |
| FC→CC | 홈 위치 | HOME_POSITION | 242 | event |
| CC→FC (A) | 사전계획 | MISSION_ITEM_INT | 73 | 1회 |
| CC→FC (B) | 동적 목표 | SET_POSITION_TARGET_GLOBAL_INT | 86 | 2Hz+ |

설계 결정:
- eta3 clothoid의 곡률 정보는 MISSION_ITEM_INT로 보존 불가
- → clothoid 객체는 CC 메모리에 유지, 추종 단계에서 setpoint로 샘플링
- 사전계획 가능 구간만 MISSION_ITEM_INT 업로드

```python
# Mission upload
master.mav.mission_count_send(
    master.target_system, master.target_component,
    N, mavutil.mavlink.MAV_MISSION_TYPE_MISSION
)
# FC → MISSION_REQUEST_INT (seq=0..N-1)
master.mav.mission_item_int_send(
    master.target_system, master.target_component,
    seq,
    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
    0, 1, 0, 0, 0, float("nan"),
    int(lat * 1e7), int(lon * 1e7), float(alt_m),
    mavutil.mavlink.MAV_MISSION_TYPE_MISSION
)
```

### 3.4 알고리즘 ② 경로추종 (NLGL / MPC)

```yaml
role: "PathPoint + 현재상태 → velocity setpoint 매 제어주기 송신"
control_frequency: "20~50 Hz"
mode: "PX4 Offboard"
preferred_setpoint: "velocity (vx, vy, vz)"
preferred_message: "SET_POSITION_TARGET_LOCAL_NED #84"
```

| 방향 | 데이터 | 메시지 | ID | 주기 |
|------|--------|--------|-----|------|
| FC→CC | NED 위치/속도 | LOCAL_POSITION_NED | 32 | 50Hz |
| FC→CC | 자세 | ATTITUDE | 30 | 50Hz |
| FC→CC | 자세 q | ATTITUDE_QUATERNION | 31 | 50Hz |
| FC→CC | IMU | HIGHRES_IMU | 105 | 100Hz |
| CC→FC | 속도 setpoint | SET_POSITION_TARGET_LOCAL_NED | 84 | 20~50Hz |

SET_POSITION_TARGET_LOCAL_NED (#84) 필드:

| 필드 | 타입 | 단위 | 비고 |
|------|------|------|------|
| coordinate_frame | uint8 | — | 1 = MAV_FRAME_LOCAL_NED |
| type_mask | uint16 | — | bit=1 → 무시 |
| x, y, z | float | m | NaN 무시 가능 |
| vx, vy, vz | float | m/s | NaN 무시 가능 |
| afx, afy, afz | float | m/s² | feedforward |
| yaw | float | rad | NaN 무시 |
| yaw_rate | float | rad/s | NaN 무시 |

type_mask 권장 패턴:

| 모드 | type_mask | 활성화 | 사용처 |
|------|-----------|--------|--------|
| 위치만 | 0x0FF8 | x,y,z | 호버 |
| 속도만 | 0x0FC7 | vx,vy,vz | NLGL/Pure Pursuit |
| 위치+FF | 0x0F00 | x,y,z,vx,vy,vz | MPC |
| 위치+yaw | 0x07F8 | x,y,z,yaw | 헤딩 동반 |

> AI 코드생성 권고: type_mask 비트 실수 잦음. 미사용 필드에 float("nan") 권장.

```python
def send_velocity_setpoint(master, vx, vy, vz, yaw_rate=float("nan")):
    type_mask = 0b0000_1101_1100_0111
    master.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        float("nan"), float("nan"), float("nan"),
        vx, vy, vz,
        0.0, 0.0, 0.0,
        float("nan"), yaw_rate
    )
```

### 3.5 알고리즘 ③ 비전 객체인식 + 위치추출

```yaml
role: "픽셀 → 실세계 NED/GPS 변환 후 FC로 전달"
deep_learning: false
two_data_flows:
  flow_A: "객체 위치를 FC로 송신 (착륙/추적)"
  flow_B: "객체 위치를 경로생성으로 피드백 (재계획)"
```

| 방향 | 데이터 | 메시지 | ID | 비고 |
|------|--------|--------|-----|------|
| FC→CC | 드론 GPS | GLOBAL_POSITION_INT | 33 | relative_alt × 1e-3 사용 |
| FC→CC | 드론 자세 | ATTITUDE | 30 | roll/pitch/yaw 보정 필수 |
| FC→CC | 짐벌 자세 | GIMBAL_DEVICE_ATTITUDE_STATUS | 285 | 짐벌 사용 시 |
| CC→FC (A) | 착륙 목표 | LANDING_TARGET | 149 | frame=1 필수 |
| CC→FC (A 대안) | 접근 GPS | SET_POSITION_TARGET_GLOBAL_INT | 86 | 접근 단계 |
| CC→FC (gimbal) | 짐벌 명령 | GIMBAL_MANAGER_SET_ATTITUDE | 282 | NaN 축 무시 |

pixel_to_gps() 확장 시그니처 (필수):

```python
def pixel_to_gps_with_attitude(
    px: float, py: float,
    image_width: int, image_height: int,
    drone_lat: float,    # GLOBAL_POSITION_INT.lat * 1e-7
    drone_lon: float,    # GLOBAL_POSITION_INT.lon * 1e-7
    drone_alt_m: float,  # GLOBAL_POSITION_INT.relative_alt * 1e-3 (MSL 아님!)
    roll: float,         # ATTITUDE.roll  (rad)
    pitch: float,        # ATTITUDE.pitch (rad)
    yaw: float,          # ATTITUDE.yaw   (rad)
    camera_fov_deg: float = 60.0,
    gimbal_pitch: float = 0.0,
) -> tuple[float, float]:
    """
    pitch=5°, alt=30m 미보정 → 약 2.6m 오차.
    """
    ...
```

LANDING_TARGET (#149) 필수 필드:

| 필드 | 값 | 설명 |
|------|-----|------|
| frame | 1 (LOCAL_NED) | Global 보내면 PX4가 무시 |
| position_valid | 1 | x,y,z 사용 |
| x, y, z | float | 목표 NED (m) |
| type | 3 | LANDING_TARGET_TYPE_VISION_OTHER |
| 전송률 | — | 10~50 Hz |

파이프라인 지연 분석:

| 단계 | 지연 |
|------|------|
| 카메라 노출+전송 | 10~30 ms |
| OpenCV 처리 (30fps) | 33~100 ms |
| MAVLink 전송 | 1~5 ms |
| 합산 | 50~150 ms |
| 10m/s 비행 시 영향 | 최대 1.5m 오차 |

### 3.6 알고리즘 ④ VTOL 천이 자세제어 (예정)

```yaml
role: "MC↔FW 천이 구간에서 PX4 내장 로직을 보조 또는 대체"
status: planned
critical_warning: "PX4 내부 천이 컨트롤러와 Offboard setpoint가 블렌딩됨"
```

| 방향 | 데이터 | 메시지 | ID | 비고 |
|------|--------|--------|-----|------|
| FC→CC | VTOL 상태 | EXTENDED_SYS_STATE | 245 | vtol_state |
| FC→CC | 자세 | ATTITUDE_QUATERNION | 31 | 100Hz |
| FC→CC | IMU | HIGHRES_IMU | 105 | 100Hz |
| CC→FC | 천이 트리거 | MAV_CMD_DO_VTOL_TRANSITION | cmd 3000 | param1=4(FW), 3(MC) |
| CC→FC | 자세/각속도 setpoint | SET_ATTITUDE_TARGET | 82 | 50~100Hz |

vtol_state 값:

| 값 | 의미 |
|----|------|
| 0 | UNDEFINED |
| 1 | TRANSITION_TO_FW |
| 2 | TRANSITION_TO_MC |
| 3 | MC (회전익 완료) |
| 4 | FW (고정익 완료) |

SET_ATTITUDE_TARGET type_mask:

| 패턴 | type_mask | 의미 | 사용 구간 |
|------|-----------|------|----------|
| 자세각 | 0x07 | q + thrust | 안정 |
| 각속도 | 0x40 | body_rate + thrust | 천이 중 권장 |
| 전체 | 0x00 | 모두 | 디버그 |

천이 권장 시퀀스:

```text
1. 천이 시작 전: Offboard 자세 안정화 (type_mask=0x07)
2. MAV_CMD_DO_VTOL_TRANSITION param1=4 전송
3. vtol_state == 1 동안: SET_ATTITUDE_TARGET type_mask=0x40
4. vtol_state == 4 즉시: attitude setpoint 중단
5. SET_POSITION_TARGET_LOCAL_NED (velocity)로 전환
```

> PX4 v1.14+ 필수. 1.13 이전 버그 존재.

---

## 4. SUMMARY: 알고리즘별 종합 매핑

```text
경로생성  ← LOCAL_POS_NED #32, EKF_STATUS #193
          → MISSION_ITEM_INT #73 (사전) | SET_POS_TGT_GLOBAL #86 (동적)

경로추종  ← LOCAL_POS_NED #32, ATTITUDE #30, HIGHRES_IMU #105
          → SET_POS_TGT_LOCAL_NED #84  (velocity, type_mask=0x0FC7)

비전+GPS  ← GLOBAL_POS_INT #33 (relative_alt!), ATTITUDE #30
          → LANDING_TARGET #149 (frame=1, position_valid=1)

VTOL천이  ← EXTENDED_SYS_STATE #245 (vtol_state), HIGHRES_IMU #105
          → CMD_DO_VTOL_TRANSITION 3000
          → SET_ATTITUDE_TARGET #82 (type_mask=0x40)
```

| 알고리즘 | 입력 (FC→CC) | 출력 (CC→FC) | 모드 | 최소 Hz |
|---------|-------------|-------------|------|--------|
| ① 경로생성 | #32, #33, #193, #242 | #73 또는 #86 | Mission/Offboard | 2 |
| ② 경로추종 | #30, #31, #32, #105 | #84 (velocity) | Offboard | 20 |
| ③ 비전 | #30, #33, #285 | #149 | Auto.Land+보정 | 10 |
| ④ VTOL | #31, #105, #245 | #82 + cmd 3000 | Offboard/VTOL | 50 |

---

## 5. PARAMETERS: PX4 파라미터 카탈로그

### 5.1 통신

| 파라미터 | 권장값 | 의미 |
|---------|--------|------|
| MAV_1_CONFIG | 102 | TELEM2 매핑 |
| SER_TEL2_BAUD | 921600 | TELEM2 보율 |
| MAV_PROTO_VER | 2 | MAVLink v2 강제 |
| MAV_1_MODE | 2 | Onboard 모드 |
| MAV_1_RATE | 1200000 | 데이터율 (bytes/s) |

### 5.2 Offboard 안전

| 파라미터 | 권장값 | 의미 |
|---------|--------|------|
| COM_OF_LOSS_T | 1.0 | Offboard 끊김 허용 (s) |
| COM_OBL_RC_ACT | 0 | 실패 시 동작 (0=Pos, 2=Land, 3=RTL) |
| COM_RC_OVERRIDE | 1 | RC 인계 허용 |
| COM_RCL_EXCEPT | 4 | Offboard에서 RC loss 예외 |

### 5.3 경로/미션

| 파라미터 | 권장값 | 의미 |
|---------|--------|------|
| COM_OBS_AVOID | 1 | 외부 경로계획 활성화 |
| NAV_ACC_RAD | 1.0~3.0 | Waypoint 도달 반경 (m) |
| NAV_LOITER_RAD | 50 | 루이터 반경 (m, FW) |
| MIS_TAKEOFF_ALT | 5.0 | 자동 이륙 고도 (m) |

### 5.4 EKF / 비전 융합

| 파라미터 | 권장값 | 의미 |
|---------|--------|------|
| EKF2_AID_MASK | bit3=1 | 비전 위치 융합 |
| EKF2_EV_DELAY | 실측 ms | 비전-IMU 지연 |
| EKF2_EV_NOISE_MD | 0 | 노이즈 모델 |
| EKF2_HGT_MODE | 3 | 고도원으로 비전 |
| EKF2_GPS_CHECK | 245 | GPS 품질 비트마스크 |

### 5.5 정밀 착륙

| 파라미터 | 권장값 | 의미 |
|---------|--------|------|
| PLD_BTOUT | 5.0 | 비콘 타임아웃 (s) |
| PLD_FAPPR_ALT | 0.5~1.0 | 최종 접근 고도 (m) |
| PLD_HACC_RAD | 0.2 | 수평 정확도 (m) |
| PLD_MAX_SRCH | 3 | 최대 재탐색 횟수 |

### 5.6 VTOL

| 파라미터 | 권장값 | 의미 |
|---------|--------|------|
| VT_TYPE | 0/1/2 | 0=Tailsitter, 1=Tiltrotor, 2=Standard |
| VT_TRANS_MIN_TM | 2.0 | 최소 천이 시간 (s) |
| VT_TRANS_TIMEOUT | 15 | 천이 타임아웃 (s) |
| VT_F_TRANS_DUR | 5 | 전진 천이 시간 |
| VT_B_TRANS_DUR | 4 | 후진 천이 시간 |
| VT_ARSP_TRANS | 10~15 | 천이 완료 대기속도 (m/s) |
| FW_AIRSPD_MIN | 기체별 | 고정익 최소 속도 |
| FW_AIRSPD_TRIM | 기체별 | 순항 속도 |

---

## 6. PITFALLS: 함정 모음

```yaml
critical_pitfalls:
  - id: P1
    bad: "MISSION_ITEM (#39, float32) 사용"
    good: "MISSION_ITEM_INT (#73, int32 degE7) 사용"
    impact: "위경도 정밀도 ~1m 손실"

  - id: P2
    bad: "GLOBAL_POSITION_INT.alt (MSL) 사용"
    good: "relative_alt × 1e-3 사용"
    impact: "지면까지 거리 수십 m 오차"

  - id: P3
    bad: "pixel_to_gps()에서 자세 보정 생략"
    good: "ATTITUDE.roll/pitch/yaw 인자 추가"
    impact: "pitch=5°, alt=30m → 약 2.6m 오차"

  - id: P4
    bad: "LANDING_TARGET frame=GLOBAL"
    good: "frame=1 (MAV_FRAME_LOCAL_NED)"
    impact: "PX4가 메시지 무시"

  - id: P5
    bad: "Offboard setpoint 1Hz 미만"
    good: "별도 스레드에서 10Hz+ 유지"
    impact: "Position 모드로 강제 복귀"

  - id: P6
    bad: "쿼터니언 정규화 누락"
    good: "norm(q)=1.0 강제"
    impact: "PX4 setpoint 거부, 자세 발산"

  - id: P7
    bad: "VTOL FW 진입 후 attitude setpoint 지속"
    good: "vtol_state==4 즉시 #82 중단, #84로 전환"
    impact: "FW 컨트롤러 충돌, 비행 불안정"

  - id: P8
    bad: "외부 알고리즘에서 RAW_IMU (#27) 사용"
    good: "HIGHRES_IMU (#105) 사용"
    impact: "ADC 원시값 단위 변환 누락"

  - id: P9
    bad: "type_mask 비트 직접 작성"
    good: "미사용 필드에 float(\"nan\") 입력"
    impact: "비트 실수로 오작동"

  - id: P10
    bad: "카메라 프레임 → NED 직접 사용"
    good: "Camera → ENU → NED 단계 변환"
    impact: "축 반전/90° 오류"
```

---

## 7. ROADMAP: 실증 단계별 절차

```yaml
phase_1_communication:
  duration: "1~2 주차"
  goals:
    - TELEM2-RPi4 UART 연결 및 보율 동기화
    - pymavlink HEARTBEAT 검증
    - 파라미터 RW 검증
    - HEARTBEAT 손실 감지

phase_2_state_subscription:
  duration: "3~4 주차"
  goals:
    - 6종 메시지 콜백 (#30, #32, #33, #105, #193, #245)
    - SET_MESSAGE_INTERVAL 조정
    - 공통 입력 레이어 완성

phase_3_vision_first:
  duration: "5 주차"
  goals:
    - pixel_to_gps_with_attitude() 확장 (P2, P3)
    - 지상 정적 테스트
    - 비행 없이 검증 가능한 가장 안전한 단계

phase_4_offboard_sitl:
  duration: "6~8 주차"
  goals:
    - Gazebo SITL 구축
    - NLGL/MPC → SET_POSITION_TARGET_LOCAL_NED 검증
    - Offboard 진입 + 워치독 안정성
    - RC override 인계 테스트

phase_5_path_integration:
  duration: "9~10 주차"
  goals:
    - eta3 + 자체 → 추종기 → SITL 폐루프
    - 곡률 기반 v_ref 적응 검증
    - Mission upload 하이브리드

phase_6_real_flight:
  duration: "11~12 주차"
  goals:
    - 실기체 Position 모드 정상 비행
    - Offboard 호버 → 저고도 추종
    - Geofence 활성화

phase_7_vtol:
  duration: "13+ 주차"
  goals:
    - 충분한 고도(20m+) MAV_CMD_DO_VTOL_TRANSITION 단독
    - Offboard 보조 추가 (각속도 모드)
    - vtol_state 천이 그래프 로깅
```

---

## 8. SAFETY: 안전 체크리스트

```yaml
pre_flight_checks:
  - "[ ] RC 송수신기 연결 및 바인딩 확인"
  - "[ ] COM_OBL_RC_ACT, COM_OF_LOSS_T 설정 확인"
  - "[ ] Offboard 진입 전 Position 모드 정상 동작"
  - "[ ] EKF_STATUS_REPORT.flags bit4 확인"
  - "[ ] 비상 모드 전환 스위치 할당 및 테스트"
  - "[ ] Geofence 설정 (개발 초기)"
  - "[ ] 배터리 전압 알림 설정"
  - "[ ] VTOL 천이 전 최소 고도 20m+ 확보"

during_flight_monitoring:
  - "HEARTBEAT 손실 1s+ → 즉시 RC 인계"
  - "vtol_state 0 진입 → 천이 중단, MC 복귀"
  - "LANDING_TARGET 5초 미수신 → 일반 착륙 fallback"
```

---

## 9. APPENDIX

### A. ROS 2 / uXRCE-DDS 옵션 (참조)

| 토픽 | 타입 | 비고 |
|------|------|------|
| /fmu/in/trajectory_setpoint | TrajectorySetpoint | 추종 setpoint |
| /fmu/in/offboard_control_mode | OffboardControlMode | 모드 활성화 |
| /fmu/in/vehicle_visual_odometry | VehicleOdometry | 비전 EKF 융합 |
| /fmu/out/vehicle_odometry | VehicleOdometry | 100Hz 상태 |
| /fmu/out/vehicle_attitude | VehicleAttitude | 100Hz 자세 |

설정:

```text
UXRCE_DDS_CFG = 102 (TELEM2)
SER_TEL2_BAUD = 921600
MAV_1_CONFIG  = 0
```

### B. 외부 참조

| 자료 | URL |
|------|-----|
| Pixhawk 6C 공식 | https://docs.px4.io/main/en/flight_controller/pixhawk6c |
| MAVLink 메시지 | https://mavlink.io/en/messages/common.html |
| PX4 Offboard | https://docs.px4.io/main/en/flight_modes/offboard |
| PX4 Path Planning | https://docs.px4.io/main/en/computer_vision/path_planning_interface |
| pymavlink | https://github.com/ArduPilot/pymavlink |
| MAVSDK-Python | https://github.com/mavlink/MAVSDK-Python |

### C. 갱신 트리거

```yaml
update_when:
  - "PX4 메이저 버전 변경 (v1.15+ 출시)"
  - "MAVLink 메시지 deprecate"
  - "코드베이스 큰 구조 변경 (예: ROS 2 도입)"
  - "VTOL 천이 알고리즘 구현 완료 (§3.6 확정)"
```

---

*End of document.*
