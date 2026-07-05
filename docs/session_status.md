---
doc_type: session_status
project: suridoksuri-1
scope: FC 세션 유일 진입점 — 트랙 보드(병행 작업 상태) + 환경 절차
last_updated: 2026-07-06
---

# FC 세션 진입 상태 문서

> **새 세션 진입:** 아래 트랙 보드에서 **재개할 트랙 블록 하나만** 읽고, 그 블록의 참조 문서만 필요 섹션 위주로 연다.
> 사용자가 "○○ 트랙 재개"라고 하면 해당 트랙, 지시가 없으면 ▶ 활성 트랙이 기본이다.
> `/session-log`는 세션이 건드린 트랙 블록**만** 갱신한다 — 다른 트랙의 상태는 보존된다.

---

## 공통 상태 (2026-07-06 갱신)

- **브랜치:** `dev--vision-computing-module` (전 트랙 공용 단일 브랜치. main 병합은 SITL-5 안정화 후 결정)
- **커밋 규율:** 트랙 전환 전 반드시 커밋(WIP 허용, 메시지에 `[main]`/`[mc-hw]`/`[sitl]`/`[vtol-hw]` 태그)
- **파라미터 규율:** 테스트 임시값은 yaml 수정 금지 — `phase2.launch.py v_cruise:=18.0 waypoints:="[...]"` launch 인자로만

---

## 트랙 보드

### 🚁 mc-실기체 — ▶ 활성

- **내용:** RPi5(Ubuntu 24.04) + Pixhawk 6C 순수 MC 테스트기체 브링업 (SITL-5 변형, `vehicle_type:=mc`)
- **마지막:** 2026-07-03 — Docker 배포환경 구축, 6C ArduCopter→PX4 플래시, 수동비행 검증 (커밋 `cccaa52`)
- **다음:** ① MAVROS 링크 안정화 — RTT 2~5 s·heartbeat 플래핑 → 태블릿 QGC 끊고 **USB 직결**부터 ② AUTO.TAKEOFF 미실행 진단 — MAVROS 서비스 미준비 vs GPS 락 없음, `statustext`로 판별
- **주의:** AUTO.TAKEOFF는 GPS 락 필수(실내/벤치 불가) · 실기체 FC는 PX4인지 확인부터
- **참조:** `flight_plan.md` SITL-5 섹션 · `pixhawk6c_rpi4_integration_guide.md`

### 🔧 main-code — ⏸ 대기

- **내용:** fc_ros/fc_bridge 기능 개발. 다음 작업단위는 **작업 F** (임의 WP 견고성 하니스 — SITL-5와 병행 가능)
- **마지막:** 2026-07-06 — 문서 재구성(트랙 보드 도입) + launch 파라미터 오버라이드 추가, 120/120 PASS
- **다음:** 작업 F 진입 — `flight_plan.md` "작업 F" 섹션대로 실행
- **주의:** `waypoints`는 테스트값(직선 300 m) 유지 결정됨(2026-06-30) — 실미션 좌표 확정 시 yaml **두 곳**(offboard_node·mission_node) 동시 교체. `v_cruise: 20.0`도 유지 결정(FW는 TECS가 속도 관장)
- **참조:** `flight_plan.md` · `fc_bridge/CLAUDE.md` · `sitl3_tuning_notes.md`(튜닝 노브)

### 🛩 sitl-vtol — ✅ 완료 (회귀검증 시에만 재개)

- **내용:** WSL SITL VTOL 검증. SITL-1~4 전부 PASS (2026-06-30)
- **재개 조건:** 비행 로직 코드 변경 후 회귀검증 필요 시 — `gz_standard_vtol`로 SITL-4 절차 재실행
- **참조:** `sitl_verification_log.md` · `sitl3_tuning_notes.md` · `archive/flight_plan_completed.md`(절차)

### ✈ vtol-실기체 — ⬜ 미착수 (선행: 🚁 mc-실기체)

- **내용:** VTOL 실기체 전체 사이클 + RC override→POSCTL 실측(SITL-1 이월 항목)
- **진입 전 필수:** `flight_plan.md` "첫 비행 전 지상 안전 테스트" + "필수 조정 파라미터 체크리스트" 전 항목
- **참조:** `flight_plan.md` SITL-5·튜닝 가이드·안전 섹션

---

## 환경 참조 (절차 — 자주 바뀌지 않음)

### 실기체 (RPi5) — 🚁 트랙

| 항목 | 내용 |
|---|---|
| 하드웨어 | RPi5 (Ubuntu 24.04) + Pixhawk 6C (PX4 플래시됨), 순수 MC 테스트기체 |
| ROS2 | Docker `ros:humble` 컨테이너 (이름 `fc`, 항상 `sudo`). 네이티브 Jazzy 미채택 |
| 설치물 | MAVROS·numpy 설치됨. fc_ros는 colcon 빌드, fc_bridge+vtol_sim은 `PYTHONPATH=/drone_ws/src/suridoksuri` |
| 기동 | `phase2.launch.py vehicle_type:=mc` |

> **개발컴은 22.04/Humble 유지** — 업그레이드하지 않는다 (검증된 환경 재현 우선).

### SITL (WSL, 개발컴) — 🛩 트랙

```bash
# T1 — PX4 SITL (VTOL. MC 검증은 gz_x500)
cd ~/PX4-Autopilot && make px4_sitl gz_standard_vtol

# T2 — MAVROS
ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557

# T3 — fc_ros
cd ~/drone_ws && source install/setup.bash
ros2 launch fc_ros phase2.launch.py
```

**코드 동기화 (Windows 수정·커밋 후 WSL에서):**

```bash
cd ~/drone_ws
git pull
colcon build --packages-select fc_ros
source install/setup.bash   # 빌드 후 매번
```

> `fc_bridge`는 colcon 패키지가 아니라 순수 Python 라이브러리 — `pip install -e .`로 설치 (1회).

### QGC ↔ WSL 연결 (PX4 재기동마다)

```bash
# Step 1 — IP 확인 (WSL)
WIN_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'); echo "Windows IP: $WIN_IP"

# Step 2 — PX4 콘솔
pxh> mavlink start -x -u 14551 -r 4000000 -t <WIN_IP>

# Step 3 — QGC (Windows): Comm Links → Add → UDP 14551 → Connect
```

상세: `docs/sitl_verification_log.md` "Windows QGC ↔ WSL SITL 연결" 섹션.
