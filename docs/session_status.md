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
- **미커밋:** `docs/session_log.md`·`docs/session_status.md` (세션 종료 기록 — 다음 커밋에 포함하면 됨)

---

## 트랙 보드

### 🚁 mc-실기체 — ▶ 활성

- **내용:** RPi5(Ubuntu 24.04) + Pixhawk 6C 순수 MC 테스트기체 브링업 (SITL-5 변형, `vehicle_type:=mc`)
- **마지막:** 2026-07-06 — 07-03 비행 3건(`10_11_00.ulg`, `10_42_29.ulg`, `10_50_55.ulg`) ulog+launch.log 교차분석 완료. 결론:
  - **OFFBOARD 미진입 원인 확정 (`/fc_ros/override` 아님):** `fc_ros_params.yaml`의 `transition_alt: 50.0`(m)이 그대로 적용됐는데 실측 최고고도는 3.3m뿐 → `offboard_node`가 `CLIMBING` 상태에서 영원히 대기, OFFBOARD 요청 자체가 발행된 적이 없음(launch.log상 `AUTO.TAKEOFF 요청 -> CLIMBING` 이후 침묵으로 확인). 이후 보인 `AUTO_LOITER`는 fc_ros 명령이 아니라 **PX4가 AUTO.TAKEOFF 완료 후 자체적으로 전환하는 기본 동작**.
  - 배터리: 무장 후 6~8초 시점에 12V대→10.2~10.6V 새그로 LOW/CRITICAL/EMERGENCY 페일세이프 3연속 발생이 **3개 비행 모두 재현** — 배터리 파라미터(셀수/전압임계) 오탐 의심, 점검 필요.
  - `10_42_29.ulg`·`10_50_55.ulg` 공통: AUTO_LOITER 중 나침반 결함(`Compass 0 fault`)+가속도계 클리핑(`Accel 1 clipping`) 겹치는 순간 GPS로 확인되는 실제 수 미터 위치 이탈(예: 정남향 ~6m) 후 자연 복귀 — 하드웨어(나침반 캘리브레이션·진동/마운트) 점검 필요.
  - 착륙은 전부 조종자 RC 스틱 개입(`Pilot took over using sticks`) 후 PX4 자체 착륙감지+자동해제(`COM_DISARM_LAND`) — 소프트웨어의 명시적 착륙 명령 아님, 정상 동작.
- **다음 비행 직전 필수 — MC 테스트용 파라미터 변경:**
  - **왜:** 위 OFFBOARD 미진입 원인이 `transition_alt` 기본값(50m, VTOL 장거리 기준)이 MC 저고도 벤치테스트에 안 맞아서였음. 그대로 두면 다음 비행도 OFFBOARD 진입 자체가 또 안 됨.
  - **방법 (yaml 직접 수정 금지 — launch 인자로만):**
    ```bash
    ros2 launch fc_ros phase2.launch.py vehicle_type:=mc \
      transition_alt:=4.0 \
      waypoints:="[0.0,0.0,4.0, 8.0,0.0,4.0]"
    ```
  - **추천값:** `transition_alt`(이륙고도) **4.0m** 내외로 낮게 — 실측 도달 가능 고도보다 여유 있게. `waypoints`는 각 WP 간격 **10m 미만**으로 짧게 (예: `[0,0,4, 8,0,4]`) — 부지 내 저고도 왕복 테스트 기준.
  - 이 변경 없이 실비행 진행 금지. RPi 코드 갱신(`git pull`+빌드) 직후, arm 전에 반드시 이 launch 인자로 실행할 것.
- **다음:** ① 위 launch 인자 반영 후 OFFBOARD 진입 재검증(첫 실질적 OFFBOARD 테스트) ② 배터리 파라미터 점검 ③ 나침반 재캘리브레이션·진동원 점검 ④ 작업 G 로그 체계로 비행 기록 지속
- **주의:** AUTO.TAKEOFF는 GPS 락 필수(실내/벤치 불가) · 실기체 FC는 PX4인지 확인부터
- **참조:** `flight_plan.md` SITL-5 섹션 · `pixhawk6c_rpi4_integration_guide.md` · `fc_ros/fc_ros/params/fc_ros_params.yaml`(transition_alt/waypoints 기본값) · `fc_ros/fc_ros/nodes/offboard_node.py`(CLIMBING/OFFBOARD 상태머신)

### 🔧 main-code — ⏸ 대기

- **내용:** fc_ros/fc_bridge 기능 개발 및 공용 인프라. **작업 G(비행 로그 수집·분석) [코드] 완료·검증(V2)·커밋**
- **마지막:** 2026-07-06 — ① **planner 2종 본선 이식**(다른 계정 repo `suridouksuri`의 Fable 작업 회수): eta3 **v3.3**(2D 퇴화 WP NaN 근본수정)+**StraightLinePlanner**(신규)+`resolve_planner_name` 기체타입 자동선택(mc→straight/vtol→eta3, `planner` 명시 우선), sim 검증 vtol_sim 6·fc_bridge 44·fc_ros 82 — `584cff3` ② **transition_alt launch 오버라이드** `356ae5a` ③ 그 전: V2 검증·pull_ulog livelock 수정 `b580953`
- **다음:** ① **RPi 배포 검증** — pull_ulog 실측 속도·byte 동일(작업 G 속도 판정 최종, 15 MB>5분이면 작업 G-2 등록) ② planner·transition_alt **실기체 검증은 🚁 mc-실기체 트랙에서** ③ 남은 V-unit: V1(download_log 재작성으로 갱신 필요)·V3·V4·V5 ④ 작업 F(임의 WP 견고성)
- **주의:** 최신 코드(`356ae5a`)가 WSL `~/suridoksuri-1`·RPi에 **미전파** — 각 환경 `git pull` 필요(RPi 정본=호스트 `~/drone_ws/src/suridoksuri`, potato03kth). `waypoints` 300 m·`v_cruise 20.0` 유지 결정(2026-06-30). V2/V5는 MAVROS 중지 필요(단독 링크)
- **참조:** `fc_bridge/planning/planner_runner.py`(resolve_planner_name) · `vtol_sim/…/straight_line_planner.py`·`eta3clothoid_v3_1_planner.py`(v3.3) · `tools/flight_logs/VERIFY.md`(V1~V5) · `flight_plan.md`

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
