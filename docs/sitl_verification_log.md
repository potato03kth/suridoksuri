---
doc_type: verification_log
project: suridoksuri-1
scope: WSL SITL 환경 구축 및 hover 검증 (리포지토리 외부 진행분 기록)
status: Phase 3 진행 중
last_updated: 2026-06-01
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

## 현재 진행 상태 (2026-06-01 기준)

- [x] WSL SITL 환경 구축
- [x] MAVROS ↔ SITL 연결 확인 (`/mavros/state: connected=true`)
- [x] hover_node SITL 검증 (z=2.0m, position 제어)
- [x] `fc_ros/` 패키지 뼈대 생성 (본 리포지토리)
- [x] `docs/fc_ros_migration_plan.md` 작성
- [x] Foxglove Studio 원격 시각화 환경 구축 (모바일 웹 접속 검증 완료)
- [ ] **경로 생성 알고리즘 ROS2 노드 통합** ← 현재 위치 (Phase 3)
- [ ] TelemetryNode 검증
- [ ] OffboardNode velocity 제어 검증
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
