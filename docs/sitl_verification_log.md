---
doc_type: verification_log
project: suridoksuri-1
scope: WSL SITL 환경 구축 및 hover 검증 (리포지토리 외부 진행분 기록)
status: Phase 3 진행 중
last_updated: 2026-06-18
---

# WSL SITL 환경 구축 및 검증 로그

> 이 문서는 리포지토리 외부(WSL `~/drone_ws`)에서 수행한 SITL 구축·검증 작업을 기록한다.
> 본 codebase의 `fc_ros/` 노드 개발 시 이 환경이 기준이 된다.

---

## 환경 스택

| 항목          | 내용                                            |
| ------------- | ----------------------------------------------- |
| 개발 OS       | Windows + WSL Ubuntu 22.04 (ARM64 아님, x86_64) |
| ROS2          | Humble (apt 바이너리)                           |
| MAVROS        | apt 바이너리 (`ros-humble-mavros`)              |
| PX4           | PX4-Autopilot (소스, WSL 로컬 빌드)             |
| 시뮬레이터    | Gazebo (HEADLESS=1 모드)                        |
| 기체 모델     | gz_x500                                         |
| GeographicLib | 데이터셋 설치 완료                              |
| GCS           | QGroundControl (Windows, UDP 14551로 수동 연결) |

---

## 왜 RPi4 → WSL로 전환했나

1. RPi4에서 작동 검증 진행 중 **저장공간 부족** 문제 발생
2. RPi5 + SSD 구성으로 원천 해결 가능했으나, 굳이 RPi5에 올릴 이유 없다고 판단
3. **WSL Ubuntu 22.04에서 계속 진행**으로 결정

---

## SITL 실행 명령어

```bash
# 터미널 1 — PX4 SITL
cd ~/PX4-Autopilot
make px4_sitl gz_x500

# 터미널 2 — MAVROS
ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557

# 터미널 3 — 노드 실행
cd ~/drone_ws && source install/setup.bash
ros2 run offboard_hover hover_node
```

---

## 외부 워크스페이스 구조

```
~/drone_ws/          (WSL, 리포지토리 외부)
└── src/
    └── offboard_hover/
        └── offboard_hover/
            └── hover_node.py   ← 검증 완료
```

---

## hover_node.py 동작 검증 결과

**결과: PASS**

| 검증 항목                        | 결과 |
| -------------------------------- | ---- |
| MAVROS 연결 확인 후 진행         | ✅   |
| 2초간 setpoint 선발행 후 ARM     | ✅   |
| Offboard 모드 전환               | ✅   |
| z=2.0m 호버 유지                 | ✅   |
| `Armed by external command` 로그 | ✅   |
| `Takeoff detected` 로그          | ✅   |

**토픽/설정 상세:**

- 토픽: `/mavros/setpoint_raw/local` (PositionTarget, FRAME_LOCAL_NED)
- type_mask: VX/VY/VZ/AFX/AFY/AFZ/YAW_RATE IGNORE → position + yaw 제어
- 발행 주기: 20Hz

> 주의: `hover_node.py`는 `setpoint_raw/local`(PositionTarget)을 사용했으나,
> `fc_ros`의 `OffboardNode`는 `setpoint_velocity/cmd_vel`(TwistStamped)을 사용한다.
> 마이그레이션 계획(`fc_ros_migration_plan.md`) 기준이 우선.

---

## 원격 접속 환경

| 접속 경로         | 방법                                                      |
| ----------------- | --------------------------------------------------------- |
| Android → WSL     | Tailscale VPN → Windows Tailscale IP + RemoteCommand jump |
| RPi → WSL         | Tailscale VPN → 직접 SSH + Mosh                           |
| 세션 관리         | tmux, `ta()` 함수로 수동 attach (전 Linux 머신 동일)      |
| Windows 재부팅 시 | Task Scheduler가 WSL tmux 세션 자동 생성                  |

---

## 원격 시각화 환경 (Foxglove Studio)

**구성 완료 (2026-06-01)**

### 스택

| 항목          | 내용                                                                       |
| ------------- | -------------------------------------------------------------------------- |
| bridge        | `ros-humble-foxglove-bridge` (apt)                                         |
| 포트          | TCP 8765 (WebSocket)                                                       |
| 접속 방법     | 모바일/원격 브라우저 → `app.foxglove.dev` → WebSocket                      |
| 네트워크 경로 | 원격 단말 → Tailscale → Windows(8765) → WSL2 포트 포워딩 → foxglove_bridge |

### 실행 명령어

```bash
# WSL2 터미널 (SITL 스택 실행 후)
source /opt/ros/humble/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765
```

### Windows 포트 포워딩 (WSL2 재시작 시 갱신 필요)

```cmd
netsh interface portproxy delete v4tov4 listenport=8765 listenaddress=0.0.0.0
for /f "tokens=1" %i in ('wsl -d Ubuntu-22.04 hostname -I') do netsh interface portproxy add v4tov4 listenport=8765 listenaddress=0.0.0.0 connectport=8765 connectaddress=%i
```

> WSL2 IP는 재시작마다 변경되므로 매번 갱신 필요.  
> `wsl -d Ubuntu-22.04 hostname -I` 로 특정 distro IP 지정.  
> SSH 세션에서 접속 시 cmd.exe 기본 셸이므로 PowerShell `$변수` 문법 아닌 위 cmd 문법 사용.

### 방화벽 규칙 (최초 1회)

```powershell
New-NetFirewallRule -DisplayName "Foxglove Bridge 8765" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow
```

### 검증 결과

- [x] foxglove_bridge MAVROS 토픽 광고 확인 (채널 155~162+)
- [x] 원격 모바일 웹 브라우저에서 WebSocket 연결 성공
- [x] `/mavros/local_position/pose` 등 토픽 수신 확인

---

## Windows QGC ↔ WSL SITL 연결 (2026-06-15 재검증)

WSL2는 NAT 구조라 PX4 기본 브로드캐스트가 Windows에 도달하지 않는다.
`-t <windows_ip>` 플래그로 직접 전송하는 커스텀 인스턴스가 필요하다.

### 네트워크 구조

| 변수       | 확인 명령 (WSL)                                               | 예시 값          | 용도                 |
| ---------- | ------------------------------------------------------------- | ---------------- | -------------------- |
| Windows IP | `cat /etc/resolv.conf \| grep nameserver \| awk '{print $2}'` | `172.29.160.1`   | PX4 `-t` 플래그 대상 |
| WSL IP     | `hostname -I \| awk '{print $1}'`                             | `172.29.168.225` | QGC Server Address   |

WSL2 재시작마다 두 IP 모두 변경될 수 있으므로 매번 확인한다.

### 연결 절차 (PX4 재시작 후마다 수행)

**Step 1 — IP 확인 (WSL 터미널)**

```bash
WIN_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
WSL_IP=$(hostname -I | awk '{print $1}')
echo "Windows: $WIN_IP  /  WSL: $WSL_IP"
```

**Step 2 — PX4 커스텀 MAVLink 인스턴스 추가 (PX4 콘솔)**

```
pxh> mavlink start -x -u 14551 -r 4000000 -t <WIN_IP>
```

14551 포트가 이미 사용 중이면 14552 등 다른 포트로 대체.  
기존 MAVROS 인스턴스(14540)는 건드리지 않는다.

**Step 3 — QGC Comm Link 설정 (Windows, 최초 1회)**

`Application Settings` → `Comm Links` → `Add`

| 항목           | 값               |
| -------------- | ---------------- |
| 타입           | UDP              |
| Listening Port | `14550`          |
| Server Address | `<WSL_IP>:14551` |

**Step 4 — QGC Connect**

Comm Links 목록에서 해당 링크 선택 → **Connect**.  
PX4 콘솔에 `INFO [commander] GCS connection regained` 가 뜨면 성공.

QGC에 vehicle이 뜨지 않으면 **QGC를 껐다 켜고** 다시 Connect한다.  
(포트 14550이 이전 세션에서 해제되지 않은 경우 발생)

### 방화벽 설정 (최초 1회, Windows PowerShell 관리자)

WSL vEthernet 어댑터는 Windows가 Public 프로필로 처리하므로 `-Profile Any` 필수.

```powershell
New-NetFirewallRule -DisplayName "PX4-QGC-UDP-14550" -Direction Inbound -Protocol UDP -LocalPort 14550 -Action Allow -Profile Any
```

### 연결 원리

```
QGC (port 14550) ──[heartbeat]──▶ WSL:14551
PX4              ──[MAVLink]───▶ Windows:14550  (heartbeat source port 학습)
```

QGC는 14550에서 heartbeat를 보내고 같은 포트에서 수신한다.  
PX4는 heartbeat 수신 시 source를 `172.29.160.1:14550`으로 학습해 그곳으로 전송한다.

### 검증된 사실 (2026-06-15)

- WSL→Windows UDP 경로 정상 동작 (`nc` 테스트로 확인)
- `portproxy` 규칙은 TCP 전용으로 UDP MAVLink에 무관하며 방해하지 않음
- `partner IP` 필드는 Normal 모드 인스턴스에서 표시되지 않을 수 있으나 정상 동작

### 아밍 불가 원인 분석 기록

`commander arm` 직접 실행 시 "Resolve system health failures first" 오류 원인 진단 과정:

| 확인 명령                 | 결과                                                         |
| ------------------------- | ------------------------------------------------------------ |
| `sensors status`          | gyro/accel/mag/baro 모두 OK                                  |
| `ekf2 status`             | attitude/local/global position 모두 1 (정상)                 |
| `listener vehicle_status` | `gcs_connection_lost: True`, `pre_flight_checks_pass: False` |

→ **GCS 미연결 상태에서 preflight 실패**가 직접 원인. QGC 연결 후 해소.

---

## fc_ros offboard_node 기동 검증 (2026-06-04)

### 수정 사항

`fc_ros` 노드를 SITL 환경에서 처음 실행하면서 발견된 버그 목록과 수정 내용.

#### 1. NumPy 2.0 호환성 — `fc_bridge/utils/rotation.py` 신규 생성

`tf_transformations`, `scipy`, `transforms3d` 모두 Ubuntu 22.04 apt 버전이
`np.maximum_sctype` (NumPy 2.0에서 제거)를 사용해 동일하게 실패.

**근본 원인:** pip NumPy 2.2.6 + apt Python 패키지(NumPy 1.x 기준 빌드) 혼용.
어떤 apt 패키지를 써도 같은 문제가 반복된다.

**해결:** `fc_bridge/utils/rotation.py`에 `quat_to_euler_xyz(w, x, y, z)` 구현.
외부 라이브러리 의존 없이 numpy 수식만 사용. NumPy 버전 무관하게 동작.
`vehicle_state_bridge.py`에서 import해 사용.

```python
# fc_bridge/utils/rotation.py 사용법
from fc_bridge.utils.rotation import quat_to_euler_xyz
roll, pitch, yaw = quat_to_euler_xyz(q.w, q.x, q.y, q.z)
```

#### 2. ROS2 파라미터 2D 리스트 미지원

`declare_parameter("waypoints", [[0,0,150], [500,0,150]])` → `TypeError`.
ROS2 파라미터는 1D flat 배열만 지원한다.

**해결:** flat 배열로 선언 후 reshape.

```python
tmp.declare_parameter("waypoints", [0.0, 0.0, 150.0, 500.0, 0.0, 150.0])
raw = np.array(tmp.get_parameter("waypoints").value, dtype=float).reshape(-1, 3)
```

#### 3. QoS RELIABILITY 불일치

MAVROS는 `BEST_EFFORT`로 발행하는데, 기본 subscriber QoS가 `RELIABLE`이어서
메시지를 아예 수신하지 못했다.

**해결:** `offboard_node.py`, `telemetry_node.py` 양쪽 MAVROS 구독에 `_MAVROS_QOS` 적용.

```python
_MAVROS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
```

#### 4. STREAMING 상태 로직 — ARM 추가 및 타이밍 수정

기존 STREAMING 상태가 1 tick만에 FOLLOWING으로 전환해
PX4가 요구하는 2초 선발행 조건을 충족하지 못했고, ARM 요청도 없었다.

**해결:** 20 tick(2초) 선발행 → OFFBOARD 전환 요청 → OFFBOARD 확인 → ARM 요청 →
armed + OFFBOARD 모두 확인 시 FOLLOWING 진입.

```
STREAMING 상태 흐름:
  tick 0~19  : setpoint(0,0,0) 발행
  tick 20    : /mavros/set_mode OFFBOARD 요청
  mode==OFFBOARD 확인 : /mavros/cmd/arming True 요청
  armed==True AND mode==OFFBOARD : → FOLLOWING 전환
```

### 검증 결과

| 항목                                          | 결과 |
| --------------------------------------------- | ---- |
| colcon 빌드                                   | ✅   |
| offboard_node import 오류 없음                | ✅   |
| MAVROS QoS 토픽 수신                          | ✅   |
| OFFBOARD 모드 전환                            | ✅   |
| ARM 시퀀스 (20tick → OFFBOARD → ARM 순서)     | ✅   |
| `/mavros/setpoint_velocity/cmd_vel` 10Hz 발행 | ✅   |
| **경로 추종 (설계 목적)**                     | ✅   |

> QGC 수동 이륙 후 offboard_node 실행. 기본 waypoint `[0,0,150] → [500,0,150]` (단일 직선 세그먼트)
> 기준으로 +N 방향 직선 이동 → WP1 3m 이내 도달 시 DONE 상태 전환 (속도=0, 제자리 호버) 확인 (2026-06-06).

### 미완료

- 자율 이륙 미구현: 현재 QGC 수동 이륙 후 노드 진입. 경로 시작 고도까지
  자율 상승하는 TAKEOFF 상태는 추후 필요 시 추가.

---

## TelemetryNode 검증 (2026-06-06)

### 단위 테스트 (로컬, rclpy 불필요)

`fc_ros/test/test_telemetry_node.py` 25개 케이스 전부 PASS (Windows Python 3.10, pytest).

| 테스트 그룹                        | 케이스 수 | 결과 |
| ---------------------------------- | --------- | ---- |
| `quat_to_euler_xyz`                | 4         | ✅   |
| `update_from_pose` — 위치 ENU→NED  | 4         | ✅   |
| `update_from_pose` — yaw ENU→NED   | 5         | ✅   |
| `update_from_twist` — 속도 ENU→NED | 4         | ✅   |
| `update_from_mavros_state`         | 2         | ✅   |
| `update_from_extended_state`       | 2         | ✅   |
| `VehicleState.copy()` 격리         | 3         | ✅   |

검증된 변환 규칙:

| 입력 (ENU)                     | 출력 (NED/fc_bridge)                      |
| ------------------------------ | ----------------------------------------- |
| position (x=E, y=N, z=U)       | `pos_ned = [y, x, z]`                     |
| velocity (vx=vE, vy=vN, vz=vU) | `vel_ned = [vy, vx, -vz]`                 |
| yaw_enu                        | `yaw_ned = π/2 - yaw_enu`, [-π, π] 정규화 |

### TelemetryNode 코드 변경 (2026-06-17)

SITL 통합 검증을 위해 2초 주기 디버그 로거 추가.
노드가 데이터를 수신하고 ENU→NED 변환을 수행하는지 콘솔에서 직접 관찰하기 위함.

```python
# 콘솔 출력 예시
[telemetry_node]: pos_ned=[ 0.02  0.01  0.00]  yaw=1.571  armed=False  vtol=0
```

확인 포인트:

- `pos_ned` 값이 0이 아닌 값으로 바뀌면 pose 콜백 정상
- 드론 이동 시 `pos_ned` 값이 변하면 ENU→NED 변환 정상
- ARM 후 `armed=True` 가 찍히면 state 콜백 정상

### SITL 통합 검증 (WSL에서 수행)

**준비 — SITL 스택 기동**

```bash
# T1
cd ~/PX4-Autopilot && make px4_sitl gz_x500
# T2
ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557
# T3
cd ~/drone_ws && source install/setup.bash
ros2 run fc_ros telemetry_node
```

**체크리스트**

| 항목               | 확인 방법                                          | 결과 |
| ------------------ | -------------------------------------------------- | ---- |
| 빌드 오류 없음     | `colcon build --packages-select fc_ros`            | ✅   |
| 노드 기동          | `ros2 node list` → `/telemetry_node`               | ✅   |
| 구독 등록 확인     | `ros2 node info /telemetry_node` → Subscribers 4개 | ✅   |
| 2초 주기 로그 출력 | 콘솔에 `pos_ned / yaw / armed / vtol` 로그 확인    | ✅   |
| pos_ned 변화 확인  | 드론 이동 시 `pos_ned` 값 변경 (ENU→NED 변환 정상) | ✅   |
| armed 반영 확인    | ARM 후 로그에 `armed=True` 반영                    | —    |

실제 로그 (2026-06-17):

```
[telemetry_node]: pos_ned=[-0.01829144 -0.00543481  0.02675617]  yaw=1.660  armed=False  vtol=0
[telemetry_node]: pos_ned=[ 0.01962453 -0.01534116  0.00549911]  yaw=1.660  armed=False  vtol=0
```

> `armed` 항목은 이번 검증에서 ARM 미수행으로 확인 생략. pose/twist/state 콜백 모두 정상 동작 확인.

---

## MissionNode 검증 (2026-06-18)

### 코드 수정 사항

SITL 실행 중 발견된 버그 목록과 수정 내용.

#### 1. waypoints 파라미터 1D 변환 + reshape 추가

ROS2 `declare_parameter`는 중첩 리스트(2D 배열)를 지원하지 않는다.  
→ flat 1D 리스트로 선언하고, 코드에서 `(N, 3)`으로 reshape.

고도 150m → 50m, 이동 거리 500m → 100m로 변경 (SITL 검증용 축소).

```python
# 변경 전 (오류)
self.declare_parameter("waypoints", [[0.0, 0.0, 150.0], [500.0, 0.0, 150.0]])
raw = self.get_parameter("waypoints").value
self._waypoints = np.array(raw, dtype=float)  # 1D array → _upload에서 wp[0]접근 실패

# 변경 후
self.declare_parameter("waypoints", [0.0, 0.0, 50.0, 100.0, 0.0, 50.0])
raw = self.get_parameter("waypoints").value
self._waypoints = np.array(raw, dtype=float).reshape(-1, 3)  # (N, 3)으로 복원
```

**파라미터 외부 주입 시 형식:**

```bash
ros2 run fc_ros mission_node --ros-args -p "waypoints:=[0.0, 0.0, 50.0, 100.0, 0.0, 50.0]"
```

#### 2. MAV_FRAME_LOCAL_NED 미지원 → GLOBAL_RELATIVE_ALT + NED→GPS 변환

PX4 미션은 글로벌 프레임(`MAV_FRAME_GLOBAL_RELATIVE_ALT = 3`)만 지원한다.  
`MAV_FRAME_LOCAL_NED (1)`로 업로드하면 `resp.success == False` 반환.

**해결 (C안):** NED 입력 인터페이스 유지, `home_lat`/`home_lon` 파라미터 추가,  
`_ned_to_gps()` 변환 후 GLOBAL_RELATIVE_ALT로 업로드.

```python
# home_lat/home_lon 파라미터 (기본값: SITL gz_x500 기본 홈)
self.declare_parameter("home_lat", 47.397742)
self.declare_parameter("home_lon", 8.545594)

@staticmethod
def _ned_to_gps(home_lat, home_lon, ned):
    lat = home_lat + ned[0] / _R_EARTH * (180.0 / np.pi)
    lon = home_lon + ned[1] / (_R_EARTH * np.cos(np.radians(home_lat))) * (180.0 / np.pi)
    return float(lat), float(lon), float(ned[2])
```

#### 3. WaypointPush 응답 필드명 오타

MAVROS `WaypointPush.srv` 응답에 `wp_transfered` (r 1개) 필드가 있다.  
`wp_transferred` (r 2개)로 접근하면 `AttributeError` 발생.

```python
# 오류
f"미션 업로드 성공: {resp.wp_transferred}개"
# 수정
f"미션 업로드 성공: {resp.wp_transfered}개"  # MAVROS 오타 (r 1개)
```

### SITL 통합 검증 (WSL에서 수행, 2026-06-18)

**준비 — SITL 스택 기동**

```bash
# T1
cd ~/PX4-Autopilot && make px4_sitl gz_x500
# T2
ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557
# T3
cd ~/drone_ws && source install/setup.bash
ros2 run fc_ros mission_node
```

**체크리스트**

| 항목             | 확인 방법                               | 결과 |
| ---------------- | --------------------------------------- | ---- |
| 빌드 오류 없음   | `colcon build --packages-select fc_ros` | ✅   |
| 노드 기동        | 터미널 오류 없음                        | ✅   |
| 서비스 연결      | `/mavros/mission/push` 서비스 대기 성공 | ✅   |
| 미션 업로드 성공 | `미션 업로드 성공: 2개` 로그 출력       | ✅   |
| 좌표 변환 정확도 | MAVROS 로그에서 GPS 좌표 검증           | ✅   |

**실제 MAVROS 로그 (2026-06-18):**

```
[mavros.mission]: WP: item #0* F:6 C: 16 p: 0 0 0 0 x: 47.39774 y: 8.545594 z: 50
[mavros.mission]: WP: item #1  F:6 C: 16 p: 0 0 0 0 x: 47.39864 y: 8.545594 z: 50
[mavros.mission]: WP: mission received
```

변환 검증:

- `#0` x=47.39774: 홈 위도 (N=0) ✅
- `#1` x=47.39864: 홈 위도 + 100m북 (Δlat≈0.000898°) ✅
- z=50: 입력 고도 50m ✅
- F:6: MAVROS 내부 GLOBAL_RELATIVE_ALT_INT 변환 (정상)

---

## SITL-1 — VTOL 환경 전환 + 상수 확인 (2026-06-19)

**결과: 조건부 PASS (COM_RC_OVERRIDE → POSCTL 전환은 실기체 SITL-5로 이월)**

### vtol_state 상수 실측

| 상태            | 예상값 | 실측값 |
| --------------- | ------ | ------ |
| MC (지상/호버)  | 3      | 3 ✅   |
| 천이 중 (MC→FW) | 1      | 1 ✅   |
| FW 비행 중      | 4      | 4 ✅   |
| 천이 중 (FW→MC) | 2      | 2 ✅   |

### QGC 수동 이륙 시퀀스

비행준비완료 → 시동됨 → takeoff → 비행중 → hold → land → 비행준비완료 흐름 정상 동작 ✅

### VTOL 천이 서비스 직접 호출

**핵심 발견: param1은 목표 상태를 의미함 (계획 문서가 반대로 기록되어 있었음)**

| 명령                                                                | 결과 |
| ------------------------------------------------------------------- | ---- |
| `CommandLong(3000, param1=4.0)` MC 상태에서 → **FW 천이 성공** ✅   |
| `CommandLong(3000, param1=3.0)` FW 상태에서 → **MC 역천이 성공** ✅ |
| result=0 항상 반환 → MAV_RESULT_ACCEPTED (정상)                     |
| param1=1.0, 2.0은 변화 없음 (지상 정지 중엔 무효)                   |

→ `flight_plan.md` 기술 참조 및 작업 C/D의 param1 값 수정 완료.

### AUTO.TAKEOFF 동작

확정 시퀀스:

```
ros2 service call /mavros/cmd/arming  CommandBool "{value: true}"   # ARM 먼저
ros2 service call /mavros/set_mode    SetMode "{custom_mode: 'AUTO.TAKEOFF'}"
```

- PX4 콘솔: `takeoff detected` 출력 ✅
- QGC: 이륙 확인 ✅
- **완료 후 전환 모드: HOLD** ✅ (작업 C 설계 입력값 확정)
- 방식 B (`cmd/takeoff` 서비스)도 동작하나, set_mode 방식과 결과 동일
- 주의: ARM 없이 AUTO.TAKEOFF 발행하면 모드만 바뀌고 실제 이륙 안 됨

### AUTO.TAKEOFF 완료 후 PX4 모드

ARM → AUTO.TAKEOFF → 목표 고도 도달 → **HOLD 모드로 전환** ✅
(비행준비완료 → 시동됨(HOLD) → TAKEOFF → 목표고도 도달 → HOLD)
→ 작업 C: HOLD 모드에서 CommandLong(3000, param1=4.0)으로 FW 천이 가능 (실측 확인).

### COM_RC_OVERRIDE

- PX4 파라미터에서 직접 3으로 변경, 재시작 후에도 유지 ✅
- rc/override 토픽 발행 → QGC에서 RC 채널 연결 UI 표시, 기체 반응(pitch → 상승) ✅
- OFFBOARD → POSCTL 자동 전환: SITL 재현 불가
  - 원인: `/mavros/rc/override`는 `RC_CHANNELS_OVERRIDE` 메시지로 GCS 긴급 명령 경로.
    COM_RC_OVERRIDE는 실제 RC 송신기(`RC_CHANNELS`)의 입력만 트리거함.
  - **실기체(SITL-5)에서 물리적 RC로 재확인 예정으로 이월.**

---

## SITL-4 — 전체 사이클 통합 (2026-06-30)

**결과: PASS** (직선 300 m + L자 경로, 전체 자율 시퀀스)

> SITL-2/3 상세는 `docs/flight_plan.md`·`docs/sitl3_fix_plan.md`·`docs/sitl3_tuning_notes.md` 참조.

### 전체 사이클 (직선 300 m)

상태 전이가 설계대로 끝까지 진행, disarmed 도달:

```
ARM_TAKEOFF → CLIMBING(50m) → 헤딩정렬(err -2.3°) → MC→FW 천이(vtol 3→1→4)
 → STREAMING → FOLLOWING(cte 최대 0.6m) → 경로끝(dist<10m) → TRANSITION_MC(vtol 4→2)
 → HOLD(WP1 복귀) → LANDING → 착륙 완료(disarmed) → DONE
```

| 항목 | 결과 |
|---|---|
| 전체 전이 로그 순서 + disarmed | ✅ HOLD 포함 |
| cross_track_error | ✅ 전 구간 ≤ 0.6 m (직선) |
| 역천이 중 가속도 ≤ 0.3g(2.94 m/s²) | ✅ ~1.5 m/s² (telemetry pos 미분, `VT_B_DEC_MSS` 1.0 설계값 부합) |
| WP1 착륙 정밀도 | ✅ 최종 [300.3 N, 0.05 E], WP1(300,0) 대비 ~0.3 m |

> 역천이 오버슈트 ~43 m(FW 관성, 정상 거동) → HOLD가 WP1으로 복귀시켜 그 자리 착륙. 줄이려면 `d_end_thresh` ↑.

### 긴급 override (Layer 2, `/fc_ros/override`)

**1차 실패 → 코드 수정 → 재검증 PASS.**

- **1차 실패 원인:** `긴급 수동 전환 실행`은 찍혔으나 MANUAL 미진입(OFFBOARD 유지) → 기체 직진 폭주(435 m). headless SITL은 RC·조이스틱 같은 수동제어 소스가 없어 PX4가 MANUAL/POSCTL을 거부한다(**SITL-1 `COM_RC_OVERRIDE`→POSCTL 재현불가와 동일 한계**). 게다가 거부 시 노드가 DONE으로 들어가 cmd_vel velocity-0을 계속 발행 → OFFBOARD 유지 + FW가 velocity 무시 → 폭주.
- **수정:** `_State.OVERRIDE` 신설. override 시 OFFBOARD setpoint 발행을 중단하고, manual 모드 1초 내 미진입이면 **AUTO.LOITER 안전 폴백**을 강제 발행. 실기체에선 조종사 RC로 manual이 즉시 잡혀 폴백 전 종료. (state_logic `override_reached`/`override_fallback_due` + 단위 테스트)
- **재검증:** QGC 모드 = Hold/Loiter 전환, 기체 선회(폭주 없음), setpoint 중단 확인. ✅
  ```
  긴급 수동 전환 실행 → MANUAL 요청
  수동 모드(MANUAL) 미진입 (mode=OFFBOARD) -> AUTO.LOITER 안전 폴백 요청
  수동/안전 모드 진입 확인 (mode=AUTO.LOITER) -> DONE
  ```
- **이월:** 실기체 RC로 MANUAL/POSCTL 직접 인계는 SITL-5에서 확인.

### L자 경로

`waypoints` yaml 교체(직선→L자) 후 FOLLOWING·역천이·착륙 전체 사이클 완료. ✅
(FW는 90° 코너를 타이트하게 못 돌아 코너 오버슈트 — 정상, WP 직선 레그 기준.)

---

## 작업 H — CommandTOL 이륙 SITL 검증 (2026-07-06)

**결과: PASS** (1차 실패 → 코드 수정 → 재검증 PASS)

**배경:** 실기체 비행에서 `SetMode("AUTO.TAKEOFF")`가 목표고도를 전달하지 않아 `MIS_TAKEOFF_ALT`(2.5m)에만 의존 → `transition_alt` 게이트 불충족으로 OFFBOARD 미진입. `CommandTOL(/mavros/cmd/takeoff, altitude=transition_alt)`로 교체.

- **1차 실패 원인:** `takeoff_request_fields()`가 `latitude=0.0, longitude=0.0`을 "현재 위치 사용"으로 잘못 가정(문서화된 "PX4 관례"가 틀림). MAVLink `MAV_CMD_NAV_TAKEOFF` 관례상 "현재 위치 사용"은 **NaN**이며, `0.0/0.0`은 실제 좌표(위도·경도 0, null island)로 해석된다. `transition_alt:=50.0`로 SITL 실행 시 QGC 모드는 AUTO.TAKEOFF로 정상 전환됐으나 고도가 전혀 상승하지 않고(`pos_ned[2]` ~0 근방에서 요동) PX4 자체 preflight 안전 로직으로 자동 disarm됨(`Armed by external command` → `Disarmed by auto preflight disarming`, ~13초).
- **수정:** `fc_bridge/execution/state_logic.py::takeoff_request_fields()`의 `latitude`/`longitude`를 `0.0` → `float('nan')`으로 교체 (커밋 `000f478`).
- **재검증 (WSL, gz_standard_vtol, `transition_alt:=50.0`):** ARM → `CommandTOL 이륙 요청 altitude=50.0m -> CLIMBING` → 실제 고도 상승 확인 → CLIMBING 통과, disarm 없이 정상 진행. ✅
- **잔존 경고 (무해, 별개 이슈):** `mavros.guided_target: PositionTargetGlobal failed because no origin`가 계속 출력됨 — MAVROS humble의 알려진 QoS 불일치 코스메틱 경고(guided_target 플러그인, `/mavros/global_position/gp_origin` 구독 QoS 불일치)로 우리 코드와 무관하고 비행 동작에 영향 없음(PX4 포럼에서도 "정상 동작함" 확인). 조치 불필요 — 제거하려면 `px4_pluginlists.yaml`에서 `guided_target` 블랙리스트 처리 가능(선택 사항, 미실행).

**합격 기준 충족:** `pytest fc_ros` 통과(위 참조) + SITL 재검증 PASS. 실기체 검증은 🚁 mc-실기체 트랙에서 다음 비행 시 확인.

---

## 작업 H-2 — 이륙 목표 AMSL 보정 + CLIMBING 지면기준 AGL (2026-07-11, ✅ 재검증 완료 — 2026-07-24 근본원인 재확정+추가수정)

> **2026-07-24 최종 결과:** 아래 체크리스트를 `PX4_HOME_LAT=35.1753 PX4_HOME_LON=126.8414
> PX4_HOME_ALT=19.2`(실측 지면 AMSL)로 통제 재현. **②geoid 정합 = 최초 FAIL 확정:**
> `/mavros/home_position/home.geo.altitude`가 19.2가 아니라 **43.98**로 수신됨(차이 24.78m,
> 한국 geoid separation과 거의 정확히 일치) — 실제 원인은 `docs/mc_hw_next_session_brief.md`가
> 애드혹으로 의심했던 "PX4 자체 버그"가 아니라 **MAVROS가 ROS REP-103 관례(NavSatFix.altitude=
> WGS84 타원체고) 때문에 HomePosition.geo.altitude에도 EGM96 보정을 적용**하는 것이었음(원인
> 위치가 PX4가 아니라 MAVROS 쪽으로 정정됨). `/mavros/altitude.amsl`(GLOBAL_POSITION_INT.alt를
> 보정 없이 relay)은 19.2와 정확히 일치함을 실측 확인 — 이걸로 소스 교체(`offboard_node.py::
> _cb_home`, 커밋 `568fbe5`). **재검증(같은 통제 조건)으로 PASS 확인:** home_amsl≈19.4 →
> CommandTOL 4m 정상 요청 → CLIMBING→STREAMING→OFFBOARD→FOLLOWING→HOLD→LANDING→disarm
> 전체 미션 정상 완주. 이 경로는 `_is_mc` 분기가 없어 MC/VTOL 동일 적용. 아래 체크리스트·시나리오
> 표는 최초 발견 당시(2026-07-11) 기록을 보존하고, 이번 재검증 결과는 이 배너와 표의 ✅/결과
> 갱신으로 반영한다.

**배경:** 작업 H(CommandTOL)의 실기체 첫 검증(2026-07-07 광주, ulog `02_17_49`)에서 이륙 실패. 근본원인 = `CommandTOL.altitude`(→ `MAV_CMD_NAV_TAKEOFF` param7)는 **AMSL 절대고도**인데 `transition_alt`(지면 기준 상승분, 예 4.0)를 그대로 실어, 지면 AMSL(19.2m)보다 낮은 목표라 PX4 navigator가 "Already higher than takeoff altitude"로 이륙을 취소 → 모터 미가동 → `COM_DISARM_PRFLT`(10s) preflight auto-disarm. 상세 분석·ulog: `logs/2026-07-07_0217_last/notes.md`.

**수정 (커밋됨):**
- `state_logic.py::takeoff_request_fields(transition_alt, home_amsl)` → `altitude = home_amsl + transition_alt`. `offboard_node`가 `/mavros/home_position/home`의 `geo.altitude`를 지면 AMSL로 사용, **home 미수신 시 이륙 보류**(잘못된 고도 이륙 차단).
- CLIMBING 게이트 `climbing_reached(pos_ned_up, transition_alt, ground_ref_up=0.0)` → 이륙 순간 지면높이(`pos_ned[2]`) 캡처 후 AGL로 판정(로컬 원점≠지면, 이 로그 2.11m 차 보정). 기본 0.0이라 원점≈지면인 SITL에선 기존 동작 불변.
- 단위테스트: `fc_ros/test 60 passed`(신규 7: AMSL 합산·지면초과 회귀·CLIMBING 지면기준), `fc_bridge 44 passed`.

**추가 수정 (2026-07-24, 커밋 `568fbe5`, 위 배너 참조):** `home_amsl` 샘플 소스를
`/mavros/home_position/home`(`HomePosition`, `geo.altitude`)에서 `/mavros/altitude`
(`Altitude`, `amsl` 필드)로 교체. 나머지(수렴판정 `home_amsl_confirmed`, 신선도 검사
`home_amsl_sample_fresh`, `takeoff_request_fields`)는 미변경 — 순수 데이터 소스만 교체.

### ⚠ 재현 조건 (필수 — 이 세팅 아니면 버그가 안 잡힌다)

작업 H 최초 SITL이 통과했던 이유: `transition_alt:=50` > SITL 지면 AMSL(≈0)이라 "50 AMSL"이 우연히 지면 위여서 climb됐다. **버그를 재현하려면 지면 AMSL을 `transition_alt`보다 크게** 만들어야 한다 → PX4 SITL home 고도를 실제 필드값으로 지정:

```bash
# T1 — PX4 SITL (MC: gz_x500). 실제 대회장 좌표+고도로 home 설정 (핵심)
PX4_HOME_LAT=35.1753 PX4_HOME_LON=126.8414 PX4_HOME_ALT=100.0 \
  make px4_sitl gz_x500
```
- 지면 AMSL = 100m, `transition_alt:=4.0`(작은 값)로 실행 시:
  - **수정 전 코드**면: `altitude=4.0`(<100) → "Already higher…" → 이륙 실패(=버그 재현).
  - **수정 후 코드**면: `altitude=104.0 AMSL`(100+4) → 4m AGL 상승 → 정상.

### 실행

```bash
# T2 — MAVROS
ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557
# T3 — fc_ros (MC, 낮은 transition_alt, 비퇴화 짧은 레그 → straight 플래너)
cd ~/drone_ws && source install/setup.bash
ros2 launch fc_ros phase2.launch.py vehicle_type:=mc \
  transition_alt:=4.0 \
  waypoints:="[0.0,0.0,4.0, 30.0,0.0,4.0]"
# T4 — 모니터
ros2 topic echo /mavros/statustext/recv     # "Already higher…" 안 떠야 정상
```

### 합격 기준 체크리스트

| 항목 | 확인 방법 | 결과 (최초 시도, `home_position.geo.altitude` 소스) | 결과 (재검증, `/mavros/altitude.amsl` 소스) |
|---|---|---|---|
| ① home_amsl 수신 | 노드 로그 `CommandTOL 이륙 요청 alt=…` 출력 | ✅ (44.0 수신) | ✅ (19.4 수신) |
| ② **geoid 정합** | 로그의 "지면" 값이 `PX4_HOME_ALT`(19.2)와 일치해야 정상 | ❌ **FAIL** — 43.98 수신(24.78m 오차, ellipsoid) | ✅ 19.4 (실측 19.2와 일치) |
| ③ 이륙 목표 = 지면+transition_alt | 로그 `alt=` 값 | 48.0m AMSL (오염된 44.0+4.0) | 23.4m AMSL (정상 19.4+4.0) |
| ④ "Already higher than takeoff altitude" **미출력** | `/mavros/statustext/recv` | ✅ 미출력(목표가 지면보다는 높아서 이 실패모드는 안 걸림) | ✅ 미출력 |
| ⑤ 실제 상승 (preflight disarm 없음) | 고도가 지면 기준 ~4m AGL 도달 | ❌ **~28.8m까지 과상승**(체크리스트 예견한 "+25~40m" 시나리오 그대로 재현) | ✅ 4.0m 정상 도달 |
| ⑥ CLIMBING 통과 | 로그 `운용 고도 4.0m 도달 → streaming` | (과상승 후 결국 통과는 함 — 근본원인은 ②) | ✅ |
| ⑦ 회귀: 이후 시퀀스 정상 | STREAMING/OFFBOARD 진입 (MC) | (미검증, ②에서 이미 FAIL 확정돼 중단) | ✅ FOLLOWING→HOLD→LANDING→disarm까지 전체 완주 |

### 실패 시 시나리오 분기 (증상 → 후보 원인 → 확인)

> 한 가설에 고정하지 말고, PASS 안 나오면 아래 표로 어느 가지인지 먼저 좁힌다.

| 증상 (SITL) | 후보 원인 | 확인 |
|---|---|---|
| 여전히 `Already higher than takeoff altitude` | home_amsl 미반영(0으로 전송) / `geo.altitude`가 0 | 노드 로그 `(지면 X.X+...)`의 X가 0이면 home 콜백이 값을 못 실음 |
| 노드가 `home_position 미수신`에서 멈춤(이륙 안 함) | GPS 락 없어 home 미설정 / QoS 불일치 / 토픽명 오류 | `ros2 topic echo /mavros/home_position/home`, GPS fix_type≥3 |
| 이륙은 하는데 목표보다 **과상승**(예 +25~40m) | `geo.altitude`가 AMSL 아닌 ellipsoid → 목표=타원체고+상승분 | 로그 "지면" 값이 `PX4_HOME_ALT`와 geoid만큼 차이나면 확정 |
| 정상 이륙했으나 **CLIMBING 무한대기** | `ground_ref` 캡처 오류 / 부호 / 로컬 원점 프레임 | 로그의 `pos_ned[2]` − `takeoff_ground_h` vs `transition_alt` 비교 |
| 이륙·CLIMBING OK인데 STREAMING/OFFBOARD 미진입 | 별개 이슈(OFFBOARD 요청·`MIS_TAKEOFF_ALT` interplay) | ulog `vehicle_command`에 `DO_SET_MODE(OFFBOARD)` 유무 |
| ARM 직후 곧바로 disarm | preflight(센서/EKF/배터리) — 이번 수정과 무관 | `/mavros/statustext/recv` 거부 사유 |

> CLIMBING 지면기준 AGL 보정(로컬 원점≠지면)은 SITL에선 원점≈지면이라 재현 불가 — 단위테스트 + 실비행으로 검증하고, SITL은 `ground_ref≈0` 회귀(기존 동작 유지)만 확인한다.
> **PASS 시:** 🚁/✈ 트랙의 "transition_alt를 `MIS_TAKEOFF_ALT` 이하로" 임시조치를 완전 제거하고, 실기체 재검증(VTOL 결함 해소 후)으로 이월.
> **✅ 2026-07-24: 위 배너대로 PASS 확정, 커밋 `568fbe5`.** 실기체(RPi5) 재검증은 아직 미실행 —
> 다음 실비행(MC 또는 VTOL) 전 SITL 관례대로 필수. RPi5 fc_ros 배포본이 이 커밋을 반영했는지
> 먼저 확인할 것(`git pull` + `colcon build --packages-select fc_ros` 재빌드 필요).

---

## MC 웨이포인트 정착 — fly-by 통과 폐기 (2026-07-27)

**결과: PASS** (커밋 `4e8e378`, 환경: 이 노트북 `Ubuntu-22.04` + `gz_x500`)

**배경:** `mc_wp_advance()`가 거리 조건 하나만 봐서 WP 반경 경계를 스치는 순간 목표가
다음 WP로 바뀌었다 — 기체가 WP 위에 실제로 가보지 않고 코너를 자르며 지나감
(2026-07-25 flight04 실측: 1.8~1.9m 지점에서 통과 판정). MC는 호버가 되므로 각 WP에
정착(도달→정지→출발)하는 게 자연스럽고, "찍은 경로대로 가는지"를 보는 이 테스트기체의
목적에도 맞다는 사용자 지적으로 정착 판정 도입.

**실행:**

```bash
ros2 launch fc_ros phase2.launch.py vehicle_type:=mc transition_alt:=3.0 \
  waypoints:='[0.0,0.0,3.0, 5.0,0.0,3.0, 5.0,5.0,3.0, 0.0,5.0,3.0, 0.0,0.0,3.0]'
```

**결과 — WP 5점 전부 정착 후 전진, 미션 완주:**

| WP | 정착 시점 WP 오차 | 정착 시점 수평속도 |
|---|---|---|
| WP0 | 0.02 m | 0.01 m/s |
| WP1 | 0.17 m | 0.17 m/s |
| WP2 | 0.05 m | 0.17 m/s |
| WP3 | 0.14 m | 0.20 m/s |
| WP4 | 0.10 m | 0.21 m/s |

종전 fly-by의 1.8~1.9m 통과와 대조된다. ARM→CLIMBING→STREAMING→OFFBOARD→FOLLOWING
→HOLD→LANDING→disarm 전 구간 완주, `mc_wp_timeout` 발동·경고 없음. 마지막 WP 정착 후
HOLD가 곧바로 안정 판정(`dist=0.1m speed=0.1m/s`)되어 착륙까지 이어졌다.
노드 로그: 배포판 로컬 `/root/fc_launch_settle_20260727.log`.

**동일 커밋 RPi5 배포·검증 완료** — 소스↔install md5 일치, yaml 신규 파라미터 3종 반영,
`import fc_ros.nodes.offboard_node` 통과(`docs/rpi_deploy.md` §5 절차).

---

## 현재 진행 상태 (2026-06-30 기준)

- [x] WSL SITL 환경 구축
- [x] MAVROS ↔ SITL 연결 확인 (`/mavros/state: connected=true`)
- [x] hover_node SITL 검증 (z=2.0m, position 제어)
- [x] `fc_ros/` 패키지 뼈대 생성 (본 리포지토리)
- [x] Foxglove Studio 원격 시각화 환경 구축 (모바일 웹 접속 검증 완료)
- [x] Windows QGC ↔ WSL SITL 연결 (MAVLink 14551 포트, 수동 연결)
- [x] fc_ros offboard_node 기동 검증 (NumPy/QoS/STREAMING 픽스, OFFBOARD+ARM 시퀀스 확인)
- [x] OffboardNode 설계 목적 검증 (L1 경로 추종, DONE 상태 전환, 2026-06-06)
- [x] TelemetryNode 단위 테스트 25/25 PASS (2026-06-06)
- [x] TelemetryNode SITL 통합 검증 (2026-06-17)
- [x] MissionNode SITL 통합 검증 (2026-06-18, NED→GPS 변환 확인)
- [x] **작업 A: params/YAML 정비** (2026-06-19, flat waypoints + 신규 파라미터)
- [x] **SITL-1: VTOL 환경 전환 + 상수 확인** (2026-06-19, 조건부 PASS)
- [x] **작업 B: 종단 감속 헬퍼 + 배선** (2026-06-20, `apply_terminal_decel()` + `offboard_node.py` main() 배선, pytest 5/5 PASS)
- [x] **작업 C: 상태머신 ① 이륙·상승·천이** (2026-06-20, ARM_TAKEOFF/CLIMBING/TRANSITION_FW 구현, pytest 13/13 PASS)
- [x] **작업 D: 상태머신 ② 역천이·착륙** (2026-06-20, TRANSITION_MC/LANDING 구현, pytest 21/21 PASS → 전체 31/31)
- [x] SITL-2: launch 통합 기동 (2026-06-20)
- [x] 작업 E: 긴급 수동 override (2026-06-20 구현 → 2026-06-30 SITL-4서 AUTO.LOITER 폴백 추가)
- [x] SITL-3: 경로 추종 검증 (2026-06-30)
- [x] SITL-4: 전체 사이클 통합 (2026-06-30, 직선+L자, override 재검증 포함)
- [ ] SITL-5: RPi4 배포, 작업 F·SITL-6 (후속)

---

## 다음 작업 (Phase 3)

`fc_ros/` 노드들을 SITL 환경에서 순차 검증한다.
마이그레이션 순서는 `docs/fc_ros_migration_plan.md`의 Step 1~8을 따른다.

**입력:** 목표 waypoint 리스트  
**출력:** `/mavros/setpoint_raw/local` 또는 `setpoint_velocity/cmd_vel`로 setpoint 스트림  
**기존 알고리즘:** `fc_bridge/` 순수 Python 코드베이스 → `fc_ros/` 노드로 래핑
