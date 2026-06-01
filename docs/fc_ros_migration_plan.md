---
doc_type: migration_plan
target_audience: ai_agent
project: suridoksuri-1
task: fc_bridge → ROS2/MAVROS 노드 변환
status: 미착수
---

# fc_bridge → fc_ros 마이그레이션 계획

> 기존 `fc_bridge` 순수 Python 라이브러리를 ROS2/MAVROS 기반 노드로 전환하는 설계 계획.  
> 미들웨어: MAVROS. ROS2 배포판: 환경에 맞게 선택 (Humble 권장).

---

## 핵심 결정 사항

### 폴더 배치: 새 `fc_ros/` 패키지 생성 (fc_bridge 내부 X)

`fc_bridge/`에 ROS2 코드를 넣지 않는 이유:
- `ament_python` 패키지는 `package.xml` + colcon 빌드 구조를 요구한다
- `fc_bridge/`에 붙이면 기존 `pytest` 단독 테스트, pymavlink SITL 직접 실행이 ROS2 환경 의존성을 갖게 된다
- `fc_bridge`는 순수 Python 라이브러리로 유지하고, `fc_ros`가 이를 import해서 사용하는 구조가 맞다

---

## 최종 디렉터리 구조

```
suridoksuri-1/
├── fc_bridge/                          (기존 — 순수 Python 라이브러리)
│   ├── setup.py                        ← 신규: plain setuptools (package.xml 없음)
│   ├── config.py
│   ├── comm/
│   │   ├── mavlink_conn.py             (유지 — SITL 직접 테스트용)
│   │   ├── telemetry.py                (유지 — VehicleState 분리 후)
│   │   └── vehicle_state.py            ← 신규: VehicleState 클래스를 여기로 이동
│   ├── guidance/
│   │   └── l1_guidance.py              (무변경)
│   ├── planning/
│   │   ├── speed_profile.py            (무변경)
│   │   └── planner_runner.py           (무변경)
│   ├── execution/
│   │   ├── mission_uploader.py         (유지 — MissionNode fallback용)
│   │   └── offboard_follower.py        (유지 — SITL 직접 실행용)
│   └── tests/                          (무변경)
│
└── fc_ros/                             ← 신규: ROS2 ament_python 패키지
    ├── package.xml
    ├── setup.py
    ├── setup.cfg
    ├── resource/
    │   └── fc_ros                      (빈 마커 파일)
    ├── fc_ros/
    │   ├── __init__.py
    │   ├── adapters/
    │   │   ├── __init__.py
    │   │   ├── vehicle_state_bridge.py ← ROS2 msg → VehicleState 변환
    │   │   └── setpoint_publisher.py   ← vel_ned ndarray → TwistStamped 발행
    │   ├── nodes/
    │   │   ├── __init__.py
    │   │   ├── telemetry_node.py       ← MAVROS 구독 → VehicleState 유지
    │   │   ├── mission_node.py         ← 경로 생성 → /mavros/mission/push
    │   │   └── offboard_node.py        ← 상태머신 + create_timer 10 Hz
    │   └── params/
    │       └── fc_ros_params.yaml      ← config.py 파라미터를 ROS2 파라미터로
    ├── launch/
    │   ├── phase1.launch.py
    │   └── phase2.launch.py
    └── test/
        └── test_offboard_node.py
```

---

## 모듈 의존 관계

```
fc_ros (ROS2 패키지)
    imports ↓
fc_bridge (plain Python 라이브러리)
    — guidance.l1_guidance.L1Guidance
    — planning.speed_profile.compute_speed_profile
    — planning.planner_runner.run_planner
    — comm.vehicle_state.VehicleState   ← telemetry.py에서 분리 후

fc_ros가 fc_bridge에서 import하지 않는 것:
    — MavlinkConn     (MAVROS 연결 레이어로 대체)
    — Telemetry       (TelemetryNode subscriber로 대체)
    — OffboardFollower (OffboardNode timer로 대체)
    — MissionUploader  (mavros/mission/push 서비스로 대체, fallback 제외)
```

---

## 노드 구성 (3개)

### TelemetryNode (`nodes/telemetry_node.py`)

MAVROS 토픽 수신 전용. 타이밍 제약 없음, 수동 콜백만.

**구독 토픽:**
- `/mavros/local_position/pose` (geometry_msgs/PoseStamped) — 위치 + quaternion
- `/mavros/local_position/velocity_local` (geometry_msgs/TwistStamped) — 속도
- `/mavros/state` (mavros_msgs/State) — armed, connected, mode
- `/mavros/extended_state` (mavros_msgs/ExtendedState) — vtol_state

**역할:** 콜백에서 `vehicle_state_bridge.py`를 통해 `VehicleState` 인스턴스 갱신.  
quaternion → Euler 변환은 `tf_transformations.euler_from_quaternion()` 사용.  
NED 부호 주의: `h_up = -z_ned`

### OffboardNode (`nodes/offboard_node.py`)

`create_timer(1.0 / control_hz, self._control_callback)`으로 상태머신 구동.

**상태 머신:** `IDLE → STREAMING → ENTRY → FOLLOWING → DONE`  
(기존 `OffboardFollower`의 `_State` enum과 동일)

**핵심 변환:**
```
기존: while ... : time.sleep(dt)
변환: create_timer(dt, callback) + 상태머신 분기
```

**발행 토픽:** `/mavros/setpoint_velocity/cmd_vel` (geometry_msgs/TwistStamped)  
**서비스 호출:** `/mavros/set_mode` (mavros_msgs/srv/SetMode, mode="OFFBOARD")  
**내부 연산:** `L1Guidance`, `compute_speed_profile` — fc_bridge에서 그대로 import

### MissionNode (`nodes/mission_node.py`)

Phase 1 전용. 경로 생성 후 MAVROS 서비스로 업로드.

**서비스 호출:** `/mavros/mission/push` (mavros_msgs/srv/WaypointPush)  
**fallback:** LOCAL_NED 프레임 미션이 MAVROS에서 미지원 시 `MissionUploader.upload()` 직접 호출

---

## MAVROS 토픽 매핑

| 기존 `fc_bridge` | ROS2/MAVROS 대체 | 메시지 타입 |
|---|---|---|
| pymavlink `LOCAL_POSITION_NED` (pos) | `/mavros/local_position/pose` | geometry_msgs/PoseStamped |
| pymavlink `LOCAL_POSITION_NED` (vel) | `/mavros/local_position/velocity_local` | geometry_msgs/TwistStamped |
| pymavlink `ATTITUDE` (yaw) | `/mavros/local_position/pose` quaternion 필드 | geometry_msgs/PoseStamped |
| pymavlink `HEARTBEAT` (armed, mode) | `/mavros/state` | mavros_msgs/State |
| pymavlink `EXTENDED_SYS_STATE` | `/mavros/extended_state` | mavros_msgs/ExtendedState |
| `set_position_target_local_ned_send()` | `/mavros/setpoint_velocity/cmd_vel` | geometry_msgs/TwistStamped |
| `MAV_CMD_DO_SET_MODE` OFFBOARD | `/mavros/set_mode` 서비스 | mavros_msgs/srv/SetMode |
| `MissionUploader` MAVLink 핸드셰이크 | `/mavros/mission/push` 서비스 | mavros_msgs/srv/WaypointPush |

**`TwistStamped` 프레임 주의:** `cmd_vel`의 frame_id는 MAVROS 버전에 따라 다르다.  
`"base_link"` 또는 `"local_origin"` — CC 환경의 MAVROS 버전을 확인하고 설정.

---

## Adapter 레이어 역할

`fc_ros/adapters/`는 ROS2 메시지 타입과 fc_bridge 데이터 구조 사이의 변환 전담.  
노드 로직과 메시지 타입 변환을 분리하여 양쪽 모두 직접 변환 코드를 갖지 않도록 한다.

### `vehicle_state_bridge.py`

```python
# 입력: ROS2 메시지 객체들 (PoseStamped, TwistStamped, State, ExtendedState)
# 출력: fc_bridge.comm.vehicle_state.VehicleState 인스턴스
# 핵심 변환:
#   quaternion → euler (roll, pitch, yaw)
#   z_ned → h_up = -z_ned
```

### `setpoint_publisher.py`

```python
# 입력: vel_ned: np.ndarray shape (3,)
# 출력: geometry_msgs/TwistStamped 발행
# 기존 OffboardFollower._send_velocity() 대체
```

---

## fc_bridge에서 변경 없는 파일

| 파일 | 이유 |
|---|---|
| `guidance/l1_guidance.py` | 순수 연산, I/O 없음 |
| `planning/speed_profile.py` | 순수 함수, I/O 없음 |
| `planning/planner_runner.py` | vtol_sim 경로 계획기 호출, MAVLink 의존 없음 |
| `tests/` 전체 | 순수 연산 테스트, 변경 불필요 |

**유지되지만 fc_ros와 병존하는 파일 (SITL 직접 실행용):**
- `comm/mavlink_conn.py`
- `comm/telemetry.py` (VehicleState 분리 후)
- `execution/mission_uploader.py`
- `execution/offboard_follower.py`
- `run_phase1.py`, `run_phase2.py`

---

## 마이그레이션 순서

### Step 1 — `VehicleState` 분리 (fc_bridge 내부 리팩터)

**작업:** `fc_bridge/comm/telemetry.py`에서 `VehicleState` 클래스를 `fc_bridge/comm/vehicle_state.py`로 이동.  
`telemetry.py`와 `offboard_follower.py`의 import 경로 수정.  
**필요 환경:** 없음 (로컬 Python)  
**검증:** `pytest fc_bridge/tests/` 전체 통과

### Step 2 — `fc_ros/` 패키지 뼈대 생성 및 colcon 빌드 확인

**작업:** `package.xml`, `setup.py`, `setup.cfg`, `resource/fc_ros`, `fc_ros/__init__.py` 생성.  
`fc_bridge/setup.py` (plain setuptools) 추가.  
**필요 환경:** ROS2 설치 환경 (SITL 불필요, WSL2 가능)  
**검증:** `colcon build --packages-select fc_ros` 오류 없이 완료

### Step 3 — `TelemetryNode` 구현

**작업:** `vehicle_state_bridge.py` + `telemetry_node.py` 작성.  
**필요 환경:** ROS2 + MAVROS  
**검증:** `ros2 topic echo /mavros/local_position/pose`와 VehicleState 출력값 비교.  
quaternion → yaw 변환, NED 부호 확인 필수.

### Step 4 — `OffboardNode` 최소 구현 (STREAMING 상태만)

**작업:** `setpoint_publisher.py` + `OffboardNode` (STREAMING 상태: 영속도 TwistStamped 발행).  
`/mavros/set_mode` 서비스 호출로 OFFBOARD 모드 전환 확인.  
**필요 환경:** ROS2 + MAVROS + PX4 SITL  
**검증:** QGroundControl에서 OFFBOARD 모드 전환 확인, `cmd_vel` 토픽 수신 확인

### Step 5 — `OffboardNode` 상태머신 완성 (ENTRY + FOLLOWING)

**작업:** `OffboardFollower._step_entry()`, `_step_following()` 로직을  
`OffboardNode._control_callback()` 내 상태머신 분기로 이식.  
`L1Guidance`, `compute_speed_profile`은 fc_bridge에서 그대로 import.  
a_max 적응 감소 로직도 그대로 이식.  
**필요 환경:** ROS2 + MAVROS + PX4 SITL  
**검증:** 동일 경로점으로 기존 `run_phase2.py`와 궤적 비교

### Step 6 — `MissionNode` 구현 (Step 5와 병렬 가능)

**작업:** `mission_node.py` 작성. `/mavros/mission/push` 서비스 호출.  
LOCAL_NED 미지원 시 fallback 경로 확인.  
**필요 환경:** ROS2 + MAVROS + PX4 SITL  
**검증:** QGroundControl Plan 뷰에서 업로드된 미션 확인

### Step 7 — launch 파일 + 파라미터 YAML

**작업:** `phase1.launch.py`, `phase2.launch.py` 작성.  
`config.py`의 모든 파라미터를 `fc_ros_params.yaml`로 이관.  
**검증:** `ros2 launch fc_ros phase2.launch.py`로 전체 파이프라인 실행

### Step 8 — 통합 검증

**작업:** SITL에서 `phase2.launch.py` 실행, 기존 `run_phase2.py` 궤적과 비교.  
`L1Guidance`, `compute_speed_profile`이 동일하므로 궤적은 동일해야 함.  
**필요 환경:** CC 환경 전체

---

## 환경별 필요 조건 요약

| 단계 | 최소 요구 환경 |
|---|---|
| Step 1 | 로컬 Python + pytest |
| Step 2 | ROS2 설치 환경 (WSL2 가능, SITL 불필요) |
| Step 3 | ROS2 + MAVROS |
| Step 4–5 | ROS2 + MAVROS + PX4 SITL |
| Step 6–8 | CC 환경 전체 |

---

## 참고: 기존 코드 위치

| 대체 대상 | 현재 위치 |
|---|---|
| TelemetryNode가 대체하는 로직 | `fc_bridge/comm/telemetry.py` |
| OffboardNode가 대체하는 루프 | `fc_bridge/execution/offboard_follower.py:130–173` |
| `_send_velocity()` | `fc_bridge/execution/offboard_follower.py:277–290` |
| `_request_offboard()` | `fc_bridge/execution/offboard_follower.py:182–195` |
| Phase 2 진입점 | `fc_bridge/run_phase2.py` |
