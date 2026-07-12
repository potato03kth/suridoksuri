---
doc_type: session_status
project: suridoksuri-1
scope: FC 세션 유일 진입점 — 트랙 보드(병행 작업 상태) + 환경 절차
last_updated: 2026-07-11
---

# FC 세션 진입 상태 문서

> **새 세션 진입:** 아래 트랙 보드에서 **재개할 트랙 블록 하나만** 읽고, 그 블록의 참조 문서만 필요 섹션 위주로 연다.
> 사용자가 "○○ 트랙 재개"라고 하면 해당 트랙, 지시가 없으면 ▶ 활성 트랙이 기본이다.
> `/session-log`는 세션이 건드린 트랙 블록**만** 갱신한다 — 다른 트랙의 상태는 보존된다.

---

## 공통 상태 (2026-07-11 갱신)

- **브랜치:** `dev--vision-computing-module` (전 트랙 공용 단일 브랜치. main 병합은 SITL-5 안정화 후 결정)
- **커밋 규율:** 트랙 전환 전 반드시 커밋(WIP 허용, 메시지에 `[main]`/`[mc-hw]`/`[sitl]`/`[vtol-hw]` 태그)
- **파라미터 규율:** 테스트 임시값은 yaml 수정 금지 — `phase2.launch.py v_cruise:=18.0 waypoints:="[...]"` launch 인자로만
- **미커밋:** `docs/session_log.md`·`docs/session_status.md`·`docs/archive/session_log_2026-06.md` (이번 세션-로그 갱신 — 다음 커밋에 포함하면 됨). 코드 수정(작업 H-2)은 `9451861`로 커밋 완료
- **vision 도메인은 별도 진입점:** `docs/vision_status.md` (트랙 보드) + `docs/vision_plan.md` (설계), 커밋 태그 `[vision]`. **FC 세션은 vision 문서를 읽지 않는다**(컨텍스트 격리).
- **하드웨어 (2026-07-09 갱신):** MC 테스트기체 **해체됨** → VTOL 테스트기체 **조립됨**. 코드에서 VTOL 천이를 진행하지 않으면 여전히 MC처럼 사용 가능하나, **PX6C의 PX4 파라미터로는 현재 조립이 MC인지 VTOL인지 구분할 방법이 없다** — PX4 설정값이 물리적 기체 형상을 반영하지 않으므로 어느 기체가 실제로 붙어 있는지는 이 문서(트랙 보드)로만 추적한다. 새 세션 진입 시 실기체 관련 가정("PX4 보고 mc/vtol 확인")을 하지 말 것. VTOL 기체에 **미진단 결함**이 있어 현재 비행 자체가 불가능 — 아래 ✈ vtol-실기체 트랙 참조.

---

## 트랙 보드

### 🚁 mc-실기체 — ⏸ 보류 (하드웨어 해체됨, 2026-07-09)

- **내용:** RPi5(Ubuntu 24.04) + Pixhawk 6C 순수 MC 테스트기체 브링업 (SITL-5 변형, `vehicle_type:=mc`). **2026-07-09 MC 테스트기체 물리적 해체 — 이 형상으로는 재개 불가.** RPi5/Pixhawk6C 자체는 VTOL 기체로 옮겨갔으므로 전자장치·코드 이력은 ✈ vtol-실기체 트랙에서 계승. 아래는 해체 전 마지막 상태(참고용).
- **마지막:** 2026-07-07 — **오늘 실비행 시도 실패**: SD카드를 컴퓨터에 꽂아둔 채 까먹고 옮겨두지 않아 PX4 prearm check(`Logging is enabled, but no SD card is detected`)로 arming 자체 거부 — 비행 데이터 없음. 그 전(07-06) ulog(`b9fc748d-...`) 재분석으로 "20m 지정했는데 3m만" 문제의 **진짜 원인 확정**: `offboard_node`의 이륙 요청이 목표고도를 안 보내 PX4 자체 `MIS_TAKEOFF_ALT`(2.5m)에만 의존 → OFFBOARD가 아예 요청된 적 없음(vehicle_command 로그에 전무). → **작업 H(`CommandTOL` 목표고도 명시)로 수정, SITL PASS·커밋까지 완료**(main-code 트랙 참조). **(2026-07-11 갱신)** 그 07-07 세션에 SD카드 넣고 재시도한 arming-성공 비행 ulog(`02_17_49`, `logs/2026-07-07_0217_last/`)를 이번에 확보·분석 → CommandTOL 이륙이 실기체서 실행됐으나 **altitude를 AMSL 절대고도가 아닌 `transition_alt`(상대)로 보내** 지면 AMSL(19.2m)보다 낮은 목표 → PX4 `Already higher than takeoff altitude`로 이륙취소·preflight disarm. **작업 H-2로 수정 완료(`9451861`, main-code) — SITL 재검증 대기.**
- **다음:** ① **작업 H 이륙실패 = AMSL 버그로 규명·수정 완료**(작업 H-2 `9451861`; `altitude` AMSL/relative 질문 종결 — 상대값을 보낸 게 원인). **남은 것: SITL 재검증**(main-code, `sitl_verification_log.md` 작업 H-2), 실기체 재검증은 ✈ vtol 결함 해소 후 ② PASS 시 이 문서의 과거 "transition_alt를 낮게" 임시조치는 참고용으로만 남기고 실질 의존 제거 ③ 배터리 파라미터 점검(07-03 3개 비행 모두 6~8초 시점 전압 새그 페일세이프 재현, 미해결) ④ 나침반 재캘리브레이션·진동원 점검(미해결) ⑤ 작업 G 로그 체계(`record_flight.sh`)로 비행 기록 지속 — 아직 실기체로 첫 실사용 전
- **주의:** AUTO.TAKEOFF는 GPS 락 필수(실내/벤치 불가) · 실기체 FC는 PX4인지 확인부터 · **비행 전 SD카드 삽입 확인 (2026-07-07 이걸로 비행 실패)**
- **참조:** **`docs/mc_flight_procedure.md`(비행 절차 전체 — 로깅 사용/미사용 둘 다, "절차는?" 질문엔 이 문서 그대로 출력)** · `flight_plan.md` 작업 H·SITL-5 섹션 · `pixhawk6c_rpi4_integration_guide.md` · `fc_ros/fc_ros/nodes/offboard_node.py`(`_step_arm_takeoff`/CLIMBING·OFFBOARD 상태머신)

### 🔧 main-code — ⏸ 대기

- **내용:** fc_ros/fc_bridge 기능 개발 및 공용 인프라. **작업 G(비행 로그 수집·분석) [코드] 완료·검증(V2)·커밋**
- **마지막:** 2026-07-11 — **작업 H-2: 이륙 실패 실기체 ulog 진단 + 수정**(`9451861`). 2026-07-07 광주 비행 ulog(`02_17_49`) pyulog 분석 → `CommandTOL.altitude`(NAV_TAKEOFF param7)는 **AMSL 절대고도**인데 `transition_alt`(상대)를 그대로 실어 지면 AMSL(19.2m)보다 낮은 목표 → PX4 `Already higher than takeoff altitude`로 이륙취소·preflight disarm(배터리·GPS·SD 전부 정상, 무관). 수정: `takeoff_request_fields(transition_alt, home_amsl)`→`altitude=home_amsl+transition_alt`(+`/mavros/home_position/home` 구독·home 미수신 시 이륙 보류), CLIMBING 게이트를 지면기준 AGL로(`climbing_reached(…, ground_ref_up)`, 로컬 원점≠지면 2.11m 보정). pytest fc_ros 60/fc_bridge 44(신규 7). **SITL 재검증 대기** — `sitl_verification_log.md` 작업 H-2(재현엔 `PX4_HOME_ALT`로 지면 AMSL>transition_alt 필요 + geoid 확인). 그 전 2026-07-06 — ① **작업 H 완료·SITL PASS·커밋** — `offboard_node.py` `_step_arm_takeoff`를 `SetMode(AUTO.TAKEOFF)`→`CommandTOL(/mavros/cmd/takeoff, altitude=transition_alt)`로 교체(`7414c1d`). 요청 필드 조립은 순수함수 `fc_bridge/execution/state_logic.py::takeoff_request_fields()`로 분리(rclpy 없는 Windows에서도 pytest 가능). **1차 SITL 실패 → 원인 수정 → 재검증 PASS:** `latitude=0.0, longitude=0.0`을 "현재 위치"로 잘못 가정(MAVLink 관례는 **NaN**, `0.0/0.0`은 실좌표라 PX4가 고도 미상승 후 preflight disarm) → NaN으로 수정(`000f478`). WSL gz_standard_vtol `transition_alt:=50.0` 재검증 시 정상 상승·CLIMBING 통과 확인. 잔존 `guided_target`/"no origin" 경고는 MAVROS humble 알려진 QoS 코스메틱 이슈로 무해. pytest 130(fc_ros+fc_bridge) 전부 통과. 상세: `flight_plan.md`·`sitl_verification_log.md` "작업 H" ② 그 전: **planner 2종 본선 이식**(다른 계정 repo `suridouksuri`의 Fable 작업 회수): eta3 **v3.3**(2D 퇴화 WP NaN 근본수정)+**StraightLinePlanner**(신규)+`resolve_planner_name` 기체타입 자동선택 — `584cff3` ③ **transition_alt launch 오버라이드** `356ae5a` ④ 그 전: V2 검증·pull_ulog livelock 수정 `b580953`
- **다음(우선순위순, 2026-07-11 갱신 — ✈ vtol-실기체 결함으로 비행 보류 중):** ① **지금 가능(WSL SITL) — 작업 H-2 SITL 재검증** (`sitl_verification_log.md` 작업 H-2 체크리스트대로: `PX4_HOME_ALT`로 지면 AMSL>transition_alt 재현 후 AMSL 수정 확인 + geoid 정합) ② ~~실기체 검증~~ **결함 해소까지 보류**(🚁 하드웨어 해체, 이설된 ✈ vtol-실기체서 재개) ③ **지금 가능 — 작업 F**(임의 WP 견고성, [코드] Claude 단독) ④ **지금 가능 — V1·V3·V4**(하드웨어 불필요, Claude 단독) ⑤ **WSL SITL만 있으면 가능 — V2·V5** ⑥ RPi 배포 검증(pull_ulog 실측 속도)은 결함 해소 후
- **주의:** 최신 코드(작업 H 포함, `000f478`까지 커밋·푸시 완료)가 RPi에 **미전파** — RPi에서 `git pull` 필요(RPi 정본=호스트 `~/drone_ws/src/suridoksuri`, potato03kth). WSL(`~/suridoksuri-1`)은 이미 pull·재빌드 완료. `waypoints` 300 m·`v_cruise 20.0` 유지 결정(2026-06-30). V2/V5는 MAVROS 중지 필요(단독 링크). **작업 H가 실기체로 검증되기 전까지** 🚁 트랙의 "transition_alt를 낮게" 임시조치를 유지할 것 — SITL은 PASS했으나 실기체 미확인. **작업 H-2(AMSL 이륙고도 수정, `9451861`)는 단위테스트만 통과 — SITL 재검증 전이라 실비행 반영 금지.** geoid 리스크(MAVROS `geo.altitude`가 ellipsoid면 이륙 과상승) SITL 로그로 판별.
- **참조:** `fc_ros/fc_ros/nodes/offboard_node.py`(`_step_arm_takeoff`) · `fc_bridge/execution/state_logic.py`(`takeoff_request_fields`) · `fc_bridge/planning/planner_runner.py`(resolve_planner_name) · `vtol_sim/…/straight_line_planner.py`·`eta3clothoid_v3_1_planner.py`(v3.3) · `tools/flight_logs/VERIFY.md`(V1~V5) · `flight_plan.md`·`sitl_verification_log.md`(작업 H) · `docs/flight_plan.md`(작업 G)

### 🛩 sitl-vtol — ✅ 완료 (회귀검증 시에만 재개)

- **내용:** WSL SITL VTOL 검증. SITL-1~4 전부 PASS (2026-06-30)
- **재개 조건:** 비행 로직 코드 변경 후 회귀검증 필요 시 — `gz_standard_vtol`로 SITL-4 절차 재실행
- **참조:** `sitl_verification_log.md` · `sitl3_tuning_notes.md` · `archive/flight_plan_completed.md`(절차)

### ✈ vtol-실기체 — ▶ 활성 (기체 결함으로 비행 보류)

- **내용:** VTOL 실기체 전체 사이클 + RC override→POSCTL 실측(SITL-1 이월 항목). RPi5+Pixhawk6C 전자장치를 🚁 mc-실기체에서 이설.
- **마지막:** 2026-07-09 — VTOL 테스트기체 조립 완료. 그러나 **기체 결함으로 비행 시도 불가**(원인 미상 — 다음 세션 진입 시 사용자에게 결함 내용 확인). 코드 천이 로직 미실행 상태에서 MC처럼 사용은 여전히 가능하나 지금은 시도하지 않음. **PX4 파라미터로 mc/vtol 물리 형상 구분 불가**(공통 상태 참조) — 실기체 세션 시작 전 어느 기체가 붙어 있는지 이 문서로 먼저 확인할 것.
- **결함 해결 전에도 진행 가능한 작업 (비행 불필요):**
  - **main-code 트랙**: 작업 F(임의 WP 경로 견고성 하니스, `fc_bridge/tests/test_arbitrary_wp.py`, [코드] Claude 단독) · `VERIFY.md` V1(pull_ulog 재조립 단위테스트)·V3(record_flight.sh 하니스)·V4(fetch_logs.ps1 증분복사) — 전부 하드웨어·비행 불필요, Claude 단독 완결
  - **sitl-vtol 트랙**: WSL SITL(`gz_standard_vtol`)로 `VERIFY.md` V2·V5, SITL-6(임의 WP 생성·추종) — 시뮬레이터만 있으면 되고 실기체 결함과 무관
  - **vision 도메인**: 별도 트랙으로 분리됨 → `docs/vision_status.md` (FC와 독립, 병행 가능)
- **진입 전 필수 (결함 해결 후 실비행 재개 시):** `flight_plan.md` "첫 비행 전 지상 안전 테스트" + "필수 조정 파라미터 체크리스트" 전 항목
- **참조:** `flight_plan.md` SITL-5·튜닝 가이드·안전 섹션 · `tools/flight_logs/VERIFY.md`(V1~V5) · `flight_plan.md` 작업 F 섹션

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
