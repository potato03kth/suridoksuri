---
doc_type: verification_log
project: suridoksuri-1
scope: WSL SITL 환경 구축 및 hover 검증 (리포지토리 외부 진행분 기록)
status: Phase 3 진행 중
last_updated: 2026-06-04
---

# WSL SITL 환경 구축 및 검증 로그

> 이 문서는 리포지토리 외부(WSL `~/drone_ws`)에서 수행한 SITL 구축·검증 작업을 기록한다.
> 본 codebase의 `fc_ros/` 노드 개발 시 이 환경이 기준이 된다.

---

## 환경 스택

| 항목 | 내용 |
|------|------|
| 개발 OS | Windows + WSL Ubuntu 22.04 (ARM64 아님, x86_64) |
| ROS2 | Humble (apt 바이너리) |
| MAVROS | apt 바이너리 (`ros-humble-mavros`) |
| PX4 | PX4-Autopilot (소스, WSL 로컬 빌드) |
| 시뮬레이터 | Gazebo (HEADLESS=1 모드) |
| 기체 모델 | gz_x500 |
| GeographicLib | 데이터셋 설치 완료 |
| GCS | QGroundControl (Windows, UDP 14551로 수동 연결) |

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

| 검증 항목 | 결과 |
|-----------|------|
| MAVROS 연결 확인 후 진행 | ✅ |
| 2초간 setpoint 선발행 후 ARM | ✅ |
| Offboard 모드 전환 | ✅ |
| z=2.0m 호버 유지 | ✅ |
| `Armed by external command` 로그 | ✅ |
| `Takeoff detected` 로그 | ✅ |

**토픽/설정 상세:**

- 토픽: `/mavros/setpoint_raw/local` (PositionTarget, FRAME_LOCAL_NED)
- type_mask: VX/VY/VZ/AFX/AFY/AFZ/YAW_RATE IGNORE → position + yaw 제어
- 발행 주기: 20Hz

> 주의: `hover_node.py`는 `setpoint_raw/local`(PositionTarget)을 사용했으나,
> `fc_ros`의 `OffboardNode`는 `setpoint_velocity/cmd_vel`(TwistStamped)을 사용한다.
> 마이그레이션 계획(`fc_ros_migration_plan.md`) 기준이 우선.

---

## 원격 접속 환경

| 접속 경로 | 방법 |
|-----------|------|
| Android → WSL | Tailscale VPN → Windows Tailscale IP + RemoteCommand jump |
| RPi → WSL | Tailscale VPN → 직접 SSH + Mosh |
| 세션 관리 | tmux, `ta()` 함수로 수동 attach (전 Linux 머신 동일) |
| Windows 재부팅 시 | Task Scheduler가 WSL tmux 세션 자동 생성 |

---

## 원격 시각화 환경 (Foxglove Studio)

**구성 완료 (2026-06-01)**

### 스택

| 항목 | 내용 |
|------|------|
| bridge | `ros-humble-foxglove-bridge` (apt) |
| 포트 | TCP 8765 (WebSocket) |
| 접속 방법 | 모바일/원격 브라우저 → `app.foxglove.dev` → WebSocket |
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

## Windows QGC ↔ WSL SITL 연결 (2026-06-04)

QGC가 Windows에, PX4 SITL이 WSL에 있는 경우 브로드캐스트가 도달하지 않으므로 수동 연결이 필요하다.

### 연결 방법

**Step 1 — Windows 호스트 IP 확인 (WSL 터미널)**

```bash
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
# 보통 172.x.x.1 형태
```

**Step 2 — PX4에 새 MAVLink 인스턴스 추가 (pxh, 기존 14540/14550 유지)**

```
pxh> mavlink start -x -u 14551 -r 4000000 -t <windows_ip>
```

**Step 3 — QGC 수동 연결 (Windows)**

`Application Settings` → `Comm Links` → `Add` → Type: UDP, Port: `14551` → Connect

### 주의사항

- `mavlink start`는 기존 인스턴스를 건드리지 않고 추가만 한다 (14540 MAVROS, 14550 기본 GCS 유지)
- WSL2 IP는 재시작마다 변경되므로 매번 갱신 필요

### 아밍 불가 원인 분석 기록

`commander arm` 직접 실행 시 "Resolve system health failures first" 오류 원인 진단 과정:

| 확인 명령 | 결과 |
|---|---|
| `sensors status` | gyro/accel/mag/baro 모두 OK |
| `ekf2 status` | attitude/local/global position 모두 1 (정상) |
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

| 항목 | 결과 |
|------|------|
| colcon 빌드 | ✅ |
| offboard_node import 오류 없음 | ✅ |
| MAVROS QoS 토픽 수신 | ✅ |
| OFFBOARD 모드 전환 | ✅ |
| ARM 시퀀스 (20tick → OFFBOARD → ARM 순서) | ✅ |
| `/mavros/setpoint_velocity/cmd_vel` 10Hz 발행 | ✅ |
| **경로 추종 (설계 목적)** | ⚠️ 미검증 |

> `commander takeoff` 후 hold 상태에서 offboard_node 실행 → OFFBOARD/ARM 시퀀스 로그 확인.
> 실제 경로 추종 동작은 별도 검증 필요.

### 미완료

- OffboardNode 설계 목적 검증: 실제 경로 추종 중 L1 guidance + speed profile 동작 확인 필요.
- 자율 이륙 미구현: 현재 `commander takeoff`로 수동 이륙 후 노드 진입. 경로 시작 고도까지
  자율 상승하는 TAKEOFF 상태는 추후 필요 시 추가.

---

## 현재 진행 상태 (2026-06-04 기준)

- [x] WSL SITL 환경 구축
- [x] MAVROS ↔ SITL 연결 확인 (`/mavros/state: connected=true`)
- [x] hover_node SITL 검증 (z=2.0m, position 제어)
- [x] `fc_ros/` 패키지 뼈대 생성 (본 리포지토리)
- [x] `docs/fc_ros_migration_plan.md` 작성
- [x] Foxglove Studio 원격 시각화 환경 구축 (모바일 웹 접속 검증 완료)
- [x] Windows QGC ↔ WSL SITL 연결 (MAVLink 14551 포트, 수동 연결)
- [x] `commander takeoff` 이륙 검증 완료 (2026-06-04)
- [x] fc_ros offboard_node 기동 검증 (NumPy/QoS/STREAMING 픽스, OFFBOARD+ARM 시퀀스 확인)
- [ ] **OffboardNode 설계 목적 검증 (경로 추종)** ← 현재 위치
- [ ] TelemetryNode 검증
- [ ] MissionNode 검증
- [ ] launch 파일 통합 검증
- [ ] RPi4 배포

---

## 다음 작업 (Phase 3)

`fc_ros/` 노드들을 SITL 환경에서 순차 검증한다.
마이그레이션 순서는 `docs/fc_ros_migration_plan.md`의 Step 1~8을 따른다.

**입력:** 목표 waypoint 리스트  
**출력:** `/mavros/setpoint_raw/local` 또는 `setpoint_velocity/cmd_vel`로 setpoint 스트림  
**기존 알고리즘:** `fc_bridge/` 순수 Python 코드베이스 → `fc_ros/` 노드로 래핑
