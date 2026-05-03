# Pixhawk 6C / PX4 FC 연동 프로토콜 정리

> VTOL 자율비행 대회용 알고리즘 실증을 위한 MAVLink 인터페이스 레퍼런스  
> 기준: PX4 v1.14+, MAVLink v2, Pixhawk 6C

---

## 현재 코드베이스 상태

| 항목 | 상태 |
|---|---|
| FC 통신 코드 | **전무** (MAVLink/MAVSDK/pymavlink 미사용) |
| `pixel_to_gps()` | roll=pitch=0 이상 가정, FC 자세 미보정 |
| `PathPoint` 출력 | NED 좌표 (`pos[N,E,D]`, `v_ref`, `chi_ref`, `curvature`) |
| 경로추종 | `nlgl_controller.py`, `mpc_controller.py` (시뮬레이터 내부) |

---

## 공통 기반 프로토콜

### MAVLink v2 개요

- PX4와 외부 컴퓨터(companion computer) 간 **유일한 표준 외부 인터페이스**
- 물리 연결: Pixhawk 6C의 **TELEM2 포트** (UART, 기본 57600bps) 또는 USB-C
- v1 대비 v2: 메시지 ID 범위 확장, 서명 지원, 확장 필드 지원

### Offboard 모드 진입 안전 시퀀스

```
1. setpoint 스트림 선행 전송 (>2Hz, 최소 1초)
2. MAV_CMD_DO_SET_MODE
     base_mode   = 209
     custom_mode = 6  (OFFBOARD)
3. MAV_CMD_COMPONENT_ARM_DISARM  param1=1
4. setpoint 지속 전송 (최소 2Hz, 권장 20~50Hz)
```

> **2Hz 워치독**: 전송이 2Hz 이하로 떨어지면 PX4가 즉시 Position 모드로 복귀.  
> 별도 스레드에서 최소 10Hz로 유지 필수.

---

## 1. 경로생성 알고리즘 (Path Planning)

### 입력: FC → 알고리즘

| 메시지 | ID | 핵심 필드 | 단위 |
|---|---|---|---|
| `GLOBAL_POSITION_INT` | #33 | `lat`, `lon` | degE7 (×1e-7°) |
| | | `relative_alt` | mm (홈 기준 상대 고도) |
| | | `vx`, `vy`, `vz` | cm/s |
| `LOCAL_POSITION_NED` | #32 | `x`, `y`, `z` | m (NED, 홈 기준) |
| | | `vx`, `vy`, `vz` | m/s |
| `EKF_STATUS_REPORT` | #193 | `flags` bit4 | GPS 위치 유효 여부 |

> EKF `flags` bit4(수평 위치 절대 유효) 확인 후 경로 실행 시작.  
> GPS lock 없이 시작하면 즉시 failsafe.

### 출력 A: Mission Upload (사전 계획 경로)

**적합한 상황**: 대회처럼 이륙~임무~착륙 전체를 자동화, 외부 컴퓨터 연결 끊겨도 FC 단독 실행

**핸드셰이크 순서:**
```
외부 컴퓨터                              Pixhawk 6C
  |-- MISSION_COUNT (count=N, type=0) --->|
  |<-- MISSION_REQUEST_INT (seq=0) -------|
  |-- MISSION_ITEM_INT (seq=0) ---------->|
  |<-- MISSION_REQUEST_INT (seq=1) -------|
  |-- MISSION_ITEM_INT (seq=1) ---------->|
  ... (N회 반복)
  |<-- MISSION_ACK (type=0: ACCEPTED) ----|
```

**`MISSION_ITEM_INT` (#73) 핵심 필드:**

| 필드 | 값 / 설명 |
|---|---|
| `frame` | `6` = MAV_FRAME_GLOBAL_RELATIVE_ALT_INT (홈 기준 상대 고도 권장) |
| `command` | `16`=NAV_WAYPOINT, `21`=NAV_LAND, `22`=NAV_TAKEOFF |
| `x`, `y` | 위도, 경도 (degE7, int32) |
| `z` | 고도 (m, float) |
| `autocontinue` | `1` = 다음 아이템 자동 진행 |

> **반드시 `MISSION_ITEM_INT` (#73) 사용.**  
> `MISSION_ITEM` (#39, float32)은 위경도 정밀도 손실 (~1m 오차) 발생.

**Dynamic Re-planning 절차:**
```
1. MISSION_CLEAR_ALL (#45) 전송
2. 새 MISSION_COUNT → MISSION_ITEM_INT 재업로드
3. MAV_CMD_DO_SET_CURRENT (param1=seq번호) 로 시작점 지정
4. MAV_CMD_MISSION_START 전송
```

### 출력 B: Offboard Position Setpoint (실시간 동적 재계획)

**적합한 상황**: 루프마다 다음 목표점이 바뀌는 동적 경로, 비전 피드백 기반 재계획

메시지: `SET_POSITION_TARGET_GLOBAL_INT` (#86) — 2번 경로추종 섹션 참조

### Mission vs Offboard 선택 기준

| 기준 | Mission Upload | Offboard Setpoint |
|---|---|---|
| 경로 변경 빈도 | 저빈도 (이륙 전 1회) | 고빈도 (실시간) |
| 외부 컴퓨터 끊김 시 | FC 단독 미션 계속 | 즉시 failsafe |
| Dynamic re-plan | 전체 재업로드 필요 | 즉시 반영 |
| 대회 안전성 | 높음 | 외부 컴퓨터 의존 |

> **권장 패턴**: Mission Upload로 기본 경로 → 비전 착륙 단계에서만 Offboard 전환 (하이브리드)

---

## 2. 경로추종 알고리즘 (Path Following)

### 입력: FC → 알고리즘

| 메시지 | ID | 핵심 필드 | 단위 |
|---|---|---|---|
| `LOCAL_POSITION_NED` | #32 | `x`, `y`, `z`, `vx`, `vy`, `vz` | m, m/s |
| `ATTITUDE` | #30 | `roll`, `pitch`, `yaw` | rad |
| | | `rollspeed`, `pitchspeed`, `yawspeed` | rad/s |
| `ATTITUDE_QUATERNION` | #31 | `q1`~`q4` | 단위 쿼터니언 |
| `HIGHRES_IMU` | #105 | `xacc`, `yacc`, `zacc` | m/s² |
| | | `xgyro`, `ygyro`, `zgyro` | rad/s |

### 출력: `SET_POSITION_TARGET_LOCAL_NED` (#84)

| 필드 | 타입 | 단위 |
|---|---|---|
| `coordinate_frame` | uint8 | `1` = MAV_FRAME_LOCAL_NED |
| `type_mask` | uint16 | 비트필드 (1=해당 필드 무시) |
| `x`, `y`, `z` | float | m (NaN = 무시) |
| `vx`, `vy`, `vz` | float | m/s (NaN = 무시) |
| `afx`, `afy`, `afz` | float | m/s² feedforward (NaN = 무시) |
| `yaw` | float | rad |
| `yaw_rate` | float | rad/s |

**type_mask 사용 패턴:**

| 제어 모드 | type_mask | 설명 |
|---|---|---|
| 위치만 | `0xFF8` | x,y,z 활성화 |
| **속도만 (Pure Pursuit/NLGL 권장)** | `0xFC7` | vx,vy,vz 활성화 |
| 위치 + 속도 feedforward | `0xF00` | 위치+속도 동시 |
| 위치 + yaw | `0x7F8` | 위치+yaw 활성화 |

> **NaN 방식 권장**: type_mask 대신 사용 안 할 필드에 `float('nan')` 입력.  
> type_mask 비트 조합 실수로 인한 버그 방지에 효과적.

### Velocity vs Position Setpoint 선택

Pure Pursuit / NLGL은 "lookahead point 방향으로의 속도 벡터"를 계산하므로  
**velocity setpoint**가 직접 매핑됨.

| 항목 | Velocity Setpoint | Position Setpoint |
|---|---|---|
| Pure Pursuit 출력 매핑 | 직접 | 간접 |
| 오버슈트 | 낮음 | 높음 (PX4 내부 PID 지연) |
| 곡선 추종 정확도 | 높음 | 낮음 |
| 연결 끊김 시 | 즉시 정지 위험 | 마지막 위치 유지 |

### SET_ATTITUDE_TARGET (#82) — 자세 직접 제어 시

| 필드 | 타입 | 설명 |
|---|---|---|
| `type_mask` | uint8 | 비트필드 (아래 참조) |
| `q` | float[4] | 목표 자세 [w,x,y,z] 단위 쿼터니언 |
| `body_roll_rate` | float | rad/s |
| `body_pitch_rate` | float | rad/s |
| `body_yaw_rate` | float | rad/s |
| `thrust` | float | 0.0~1.0 정규화 추력 |

**type_mask 패턴:**

| 모드 | type_mask | 의미 |
|---|---|---|
| 자세각 + thrust | `0x07` | 각속도 무시, 쿼터니언+추력 제어 |
| 각속도 + thrust | `0x40` | 자세각 무시, 각속도+추력 제어 |
| 전체 | `0x00` | 모두 활성화 |

### 전송 주파수 요구사항

| 용도 | 최소 | 권장 |
|---|---|---|
| Offboard 유지 (워치독) | 2 Hz | 10 Hz |
| 위치 제어 | — | 10~50 Hz |
| 자세/각속도 제어 | — | 50~100 Hz |
| Pure Pursuit / NLGL 루프 | — | 20~50 Hz |

---

## 3. 비전 객체인식 + GPS 위치추출

### 입력: FC → 알고리즘 (`pixel_to_gps()` 확장용)

| 메시지 | ID | 필드 | 용도 |
|---|---|---|---|
| `GLOBAL_POSITION_INT` | #33 | `lat * 1e-7` | 드론 위도 (°) |
| | | `lon * 1e-7` | 드론 경도 (°) |
| | | **`relative_alt * 1e-3`** | 드론 고도 (m, 홈 기준) |
| `ATTITUDE` | #30 | `roll`, `pitch`, `yaw` | 자세 보정용 (rad) |
| `GIMBAL_DEVICE_ATTITUDE_STATUS` | #285 | `q[4]` | 짐벌 자세 쿼터니언 |

**`alt` vs `relative_alt` 차이 (치명적 함정):**

| 필드 | 기준 | 주의사항 |
|---|---|---|
| `alt` | 해수면(MSL) | 지형 고도를 따로 빼야 지면까지의 실제 거리 계산 가능 |
| **`relative_alt`** | 홈(이륙) 포인트 | 평탄 지형에서 지면까지의 거리 ≈ 이 값. **사용 권장** |

> `pixel_to_gps()`의 `drone_alt_m`에 `alt`(MSL) 사용 시 수십 미터 오차 발생.  
> 반드시 `relative_alt * 1e-3` 사용.

**자세 미보정 오차 규모:**
- pitch=5°, 고도=30m → **약 2.6m GPS 오차**
- FC 연동 시 `ATTITUDE.roll/pitch/yaw` 로 반드시 보정 필요

**현재 `pixel_to_gps()` 확장 필요 시그니처:**
```python
def pixel_to_gps_with_attitude(
    px, py, image_width, image_height,
    drone_lat,    # GLOBAL_POSITION_INT.lat * 1e-7
    drone_lon,    # GLOBAL_POSITION_INT.lon * 1e-7
    drone_alt_m,  # GLOBAL_POSITION_INT.relative_alt * 1e-3
    roll,         # ATTITUDE.roll  (rad)
    pitch,        # ATTITUDE.pitch (rad)
    yaw,          # ATTITUDE.yaw   (rad)
    camera_fov_deg=60.0,
    gimbal_pitch=0.0,
) -> tuple[float, float]:  # (lat, lon)
```

### 출력: FC에 착륙 목표 전달

**방식 비교:**

| 방식 | 메시지 | ID | 언제 |
|---|---|---|---|
| **정밀착륙** | `LANDING_TARGET` | #149 | 착륙 마지막 단계 (정확도 최고) |
| 접근 단계 | `SET_POSITION_TARGET_GLOBAL_INT` | #86 | 착륙 접근 시 실시간 GPS 목표 |
| 미션 기반 | `MAV_CMD_NAV_LAND` | COMMAND | 미션 아이템으로 착륙 |

> **대회 권장 패턴**: Mission NAV_LAND로 착륙 시작 + LANDING_TARGET으로 마지막 보정

**`LANDING_TARGET` (#149) 핵심 필드:**

| 필드 | 값 | 설명 |
|---|---|---|
| `frame` | `1` | **MAV_FRAME_LOCAL_NED** (필수. Global 프레임 보내면 PX4가 무시) |
| `position_valid` | `1` | 위치 기반 모드 (x,y,z 사용) |
| `x`, `y`, `z` | float | 목표 NED 좌표 (m) |
| `type` | `3` | LANDING_TARGET_TYPE_VISION_OTHER |
| 전송 주파수 | — | **10~50 Hz** |

> PX4 `precision_land` 파라미터 활성화 필요 (`NAV_ACC_RAD`, `PLD_BTOUT`, `PLD_FAPPR_ALT`)

### 짐벌 제어: Gimbal Protocol v2

```
외부 컴퓨터
  → GIMBAL_MANAGER_SET_ATTITUDE (#282)
      [Gimbal Manager가 수신]
  → GIMBAL_DEVICE_SET_ATTITUDE (#284)
      [Gimbal 하드웨어]
```

외부에서는 `GIMBAL_MANAGER_SET_ATTITUDE` (#282)만 전송하면 됨.  
`q[4]`에 NaN 입력 시 해당 축 무시.

### 파이프라인 지연 보정

| 지연 원인 | 시간 |
|---|---|
| 카메라 노출 + 전송 | 10~30 ms |
| vision 파이프라인 처리 | 33~100 ms (30fps 기준) |
| MAVLink 전송 | 1~5 ms |
| **합산** | **50~150 ms** |

10m/s 비행 시 → **최대 1.5m GPS 오차** 발생.  
저고도 착륙 단계(저속)에서는 허용 가능. 고속 접근 구간에서는 FC 상태 히스토리 버퍼 보간 필수.

---

## 4. VTOL 비행모드 천이 자세제어 (예정)

### 천이 트리거: `MAV_CMD_DO_VTOL_TRANSITION`

```
COMMAND_LONG (#76)
  command = 3000   (MAV_CMD_DO_VTOL_TRANSITION)
  param1  = 4      (VTOL_STATE_FW: 고정익으로 전환)
           = 3      (VTOL_STATE_MC: 회전익으로 전환)
```

### 천이 상태 감지: `EXTENDED_SYS_STATE` (#245)

| `vtol_state` 값 | 의미 |
|---|---|
| 0 | UNDEFINED |
| 1 | TRANSITION_TO_FW (MC→FW 천이 중) |
| 2 | TRANSITION_TO_MC (FW→MC 천이 중) |
| 3 | **MC** (회전익 모드 완료) |
| 4 | **FW** (고정익 모드 완료) |

```python
def on_extended_sys_state(msg):
    if msg.vtol_state == 4:         # 고정익 완료
        switch_to_fw_control()
    elif msg.vtol_state in (1, 2):  # 천이 진행 중
        apply_transition_attitude_assist()
```

### 천이 중 자세 보조: `SET_ATTITUDE_TARGET` (#82)

**각속도 제어 모드 (천이 중 권장 — 응답 빠름):**
```
type_mask = 0x40
  bit6=1: 자세각(쿼터니언) 무시
  bit0~2=0: 각속도 활성화

body_roll_rate  = [rad/s]
body_pitch_rate = [rad/s]
body_yaw_rate   = [rad/s]
thrust          = [0.0~1.0]
```

**자세각 제어 모드 (안정 구간):**
```
type_mask = 0x07
  bit0~2=1: 각속도 무시
  bit6=0: 자세각(쿼터니언) 활성화

q      = [w, x, y, z]  (반드시 단위 쿼터니언, norm=1.0)
thrust = [0.0~1.0]
```

### PX4 내부 천이 로직과의 충돌

```
PX4 천이 중 내부 구조:
  MC 자세 컨트롤러 출력
       ↕ (천이 진행도 0→1에 따라 블렌딩)
  FW 자세 컨트롤러 출력

Offboard SET_ATTITUDE_TARGET → MC 경로로 입력됨
천이 진행도가 높아질수록 FW 비중 증가 → Offboard 효과 감소
```

**권장 패턴:**
```
1. 천이 시작 전: Offboard로 자세 안정화
2. MAV_CMD_DO_VTOL_TRANSITION 전송
3. vtol_state == TRANSITION_TO_FW(1) 동안: 각속도 보조 setpoint 전송
4. vtol_state == FW(4) 확인 즉시: attitude setpoint 중단
5. FW 모드: velocity/position setpoint로 전환
```

> **PX4 v1.14+ 사용 권장**: 구버전(1.13 이전)에서 VTOL Offboard attitude setpoint 처리 버그 존재.

### IMU: `RAW_IMU` vs `HIGHRES_IMU`

| 항목 | `RAW_IMU` (#27) | `HIGHRES_IMU` (#105) |
|---|---|---|
| 데이터 타입 | int16 원시 ADC | float, SI 단위 변환 완료 |
| 가속도 단위 | 제조사별 스케일 (직접 변환 필요) | m/s² |
| 자이로 단위 | 제조사별 스케일 | rad/s |
| 추가 필드 | — | 기압, 온도, `fields_updated` |
| 권장 용도 | 하드웨어 디버깅 | **외부 알고리즘 (항상 이것 사용)** |

**전송 주파수 설정:**
```bash
mavlink stream -u 14556 -s HIGHRES_IMU -r 100
```

---

## 전체 인터페이스 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                      Pixhawk 6C (PX4)                           │
│                                                                  │
│  FC → 알고리즘 (상태 수신)          알고리즘 → FC (제어 출력)   │
│  ─────────────────────              ─────────────────────────── │
│  GLOBAL_POSITION_INT  #33           MISSION_ITEM_INT       #73  │
│  LOCAL_POSITION_NED   #32           SET_POS_TARGET_LCL_NED #84  │
│  ATTITUDE             #30           SET_POS_TARGET_GLB_INT #86  │
│  ATTITUDE_QUATERNION  #31           SET_ATTITUDE_TARGET    #82  │
│  HIGHRES_IMU          #105          LANDING_TARGET         #149 │
│  EKF_STATUS_REPORT    #193          MAV_CMD_DO_VTOL_TRANS  3000 │
│  EXTENDED_SYS_STATE   #245          MAV_CMD_DO_SET_MODE    #11  │
│  GIMBAL_DEV_ATT_STAT  #285          GIMBAL_MGR_SET_ATT     #282 │
└─────────────────────────────────────────────────────────────────┘

알고리즘별 핵심 연결:

경로생성  ←[LOCAL_POS_NED #32, EKF_STATUS #193]
          →[MISSION_ITEM_INT #73 / SET_POS_TARGET_GLOBAL #86]

경로추종  ←[LOCAL_POS_NED #32, ATTITUDE #30, HIGHRES_IMU #105]
          →[SET_POS_TARGET_LOCAL_NED #84 (velocity setpoint)]

비전+GPS  ←[GLOBAL_POS_INT #33 (relative_alt!), ATTITUDE #30]
          →[LANDING_TARGET #149 (frame=LOCAL_NED, position_valid=1)]

VTOL천이  ←[EXTENDED_SYS_STATE #245 (vtol_state), HIGHRES_IMU #105]
          →[CMD_DO_VTOL_TRANSITION, SET_ATTITUDE_TARGET #82 (type_mask=0x40)]
```

---

## 주요 함정 모음

| # | 함정 | 올바른 방법 |
|---|---|---|
| 1 | `MISSION_ITEM` (float) 사용 | `MISSION_ITEM_INT` (#73) 사용 |
| 2 | `GLOBAL_POS_INT.alt` (MSL) 사용 | `relative_alt * 1e-3` 사용 |
| 3 | `pixel_to_gps()` 자세 미보정 | ATTITUDE.roll/pitch/yaw 보정 필수 |
| 4 | `LANDING_TARGET` frame=GLOBAL | `frame=1` (LOCAL_NED) 필수 |
| 5 | Offboard setpoint 2Hz 미만 | 별도 스레드에서 10Hz+ 유지 |
| 6 | SET_ATTITUDE_TARGET 쿼터니언 norm≠1 | 반드시 단위 쿼터니언 정규화 |
| 7 | VTOL 완료 후 attitude setpoint 지속 | vtol_state=FW 확인 후 즉시 중단 |
| 8 | RAW_IMU 외부 알고리즘 사용 | HIGHRES_IMU (#105) 사용 |

---

*기준: PX4 v1.14+, MAVLink v2*  
*참조: PX4 공식 문서, MAVLink 메시지 정의 (mavlink.io/en/messages/common.html)*
