---
doc_type: session_status
project: suridoksuri-1
scope: FC 세션 유일 진입점 — 트랙 보드(병행 작업 상태) + 환경 절차
last_updated: 2026-07-18
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

### 🚁 mc-실기체 — ▶ 활성 (부활, 2026-07-18 확인 — 세부 미확인)

- **내용:** RPi5(Ubuntu 24.04) + Pixhawk 6C 순수 MC 테스트기체 브링업 (SITL-5 변형, `vehicle_type:=mc`). 2026-07-09 물리적 해체됐던 것과 별개로(또는 그것을 재조립한 것인지 미확인) **2026-07-18 "부활한 MC 테스트기체"로 실비행 진행 중임을 로그로 확인.**
- **마지막:** **(2026-07-18)** 사용자 요청으로 Claude가 RPi5(Tailscale `100.67.27.83`, hostname `doksuri`, 계정 `suri`)에 SSH로 직접 접속해 비행 로그를 조사 → **문서에 기록 없이 07-17(6회)·07-18(8회) 총 14회 실비행이 이미 진행돼 있었음** 발견(`vehicle_type:=mc`, `transition_alt` 5.0~6.0m, 삼각 왕복 웨이포인트). `record_flight.sh` 사용으로 rosbag+launch.log는 14회 전부 존재. FC(Pixhawk)에서 직접 회수한 ulog는 **11개 전부 오늘(07-18) 새벽~오전 것뿐**(UTC 03:06~04:17) — 이 FC에 07-17 이전 로그가 전혀 없어 **SD카드가 새것이거나 다른 Pixhawk 유닛일 가능성**(미확정). 그중 8개(id3~10)는 오늘 `flight01~08`과 시각이 정확히 1:1 매칭돼 각 폴더에 편입 완료, 앞선 3개(id0~2, KST 12:06/12:09/12:59)는 `record_flight.sh` 쓰기 전 로그라 대응 rosbag/launch.log 없음 → `logs/2026-07-18_unlogged/`에 "비행기록 부족함"으로 별도 보관. `flight08` launch.log 확인상 ARM→CommandTOL 이륙(6.0m)→CLIMBING 정상 진입, 텔레메트리 정상 수신. **notes.md(비행조건 외 관찰/결론)는 14회 전부 비어있음** — 조종사가 아직 안 채움.
- **미확인 (다음 세션 진입 시 사용자에게 확인):** ① 이 "부활한 MC 기체"가 ✈ vtol-실기체 트랙의 그 결함 기체(2026-07-09 조립, 원인미상 결함으로 비행 보류)와 같은 물리 개체인지, 완전히 별도로 새로 조립한 것인지 ② 그렇다면 ✈ vtol-실기체의 결함이 해소된 것인지, 아니면 이 MC 비행은 그 결함과 무관한 별개 기체인지 — **✈ vtol-실기체 트랙 상태는 이번에 갱신하지 않았음** (아래 참조) ③ 07-17·07-18 14회 비행의 실제 결과(관찰/결론) — notes.md 채우기 필요
- **로그 인프라 버그 2건 발견 (2026-07-18, main-code 트랙에서 상세):** ulog 자동회수가 지금까지 한 번도 성공한 적 없었음 — (a) RPi 호스트에 pymavlink 자체가 미설치 (b) `record_flight.sh`를 컨테이너 `fc` 안 root로 실행해 `logs/` 하위 폴더가 root 소유가 되어 suri 계정 쓰기 불가. 상세·수정 상태는 🔧 main-code 트랙 참조.
- **주의:** AUTO.TAKEOFF는 GPS 락 필수(실내/벤치 불가) · 실기체 FC는 PX4인지 확인부터 · **비행 전 SD카드 삽입 확인 (2026-07-07 이걸로 비행 실패 이력)**
- **참조:** **`docs/mc_flight_procedure.md`(비행 절차 전체 — 로깅 사용/미사용 둘 다, "절차는?" 질문엔 이 문서 그대로 출력)** · `flight_plan.md` 작업 H·SITL-5 섹션 · `pixhawk6c_rpi4_integration_guide.md` · `fc_ros/fc_ros/nodes/offboard_node.py`(`_step_arm_takeoff`/CLIMBING·OFFBOARD 상태머신) · `logs/2026-07-18_flight01~08`·`logs/2026-07-18_unlogged/`(이번에 회수한 원본)

### 🔧 main-code — ⏸ 대기

- **내용:** fc_ros/fc_bridge 기능 개발 및 공용 인프라. **작업 G(비행 로그 수집·분석) [코드] 완료·검증(V2)·커밋**
- **마지막:** **(2026-07-18)** 🚁 mc-실기체 로그 조사 중 작업 G 인프라의 실사용 버그 2건 발견 — ① **ulog 자동회수가 지금까지 한 번도 성공한 적 없었음**: RPi 호스트에 pymavlink가 아예 미설치라 `pull_ulog.py`가 매번 조용히 실패(실패 메시지가 어디에도 저장 안 돼 발견이 늦어짐) — Claude가 임시로 `pip install --user --break-system-packages pymavlink`로 우회 설치·확인함(영구화 필요: 컨테이너 이미지 또는 RPi 셋업 스크립트/문서에 pymavlink 설치 단계 반영할 것). ② **`record_flight.sh`를 컨테이너 `fc` 안 root로 실행**해 `logs/<날짜>_flightNN/` 폴더가 root 소유가 됨 → `suri` 계정이 그 안에 쓰기 불가(ulog를 못 넣음, 향후 `fetch_logs.ps1`/scp도 root 소유 파일 자체는 읽기는 되지만 정리는 어려움) — **수정 완료**: `record_flight.sh` 종료 시 `$FLIGHT_DIR`을 `$LOG_ROOT` 소유자로 chown하도록 추가(best-effort, `2>/dev/null || true`로 실패해도 스크립트 안 죽음). `bash -n` 통과, 이 WSL에 `pytest`/`pymavlink` 미설치라 `test_flight_logs.py` 로컬 실행은 못함(그 테스트는 `pull_ulog.py` 순수함수 대상이라 이 변경과 무관 — 회귀 위험 낮음). **커밋됨 — RPi에 `git pull` 반영 전까지는 다음 비행에도 이 문제 재발함.** 오늘 14회 비행분(`logs/2026-07-18_*`)은 수동으로 회수·정리 완료(🚁 트랙 참조).
- 그 전 2026-07-11 — **작업 H-2: 이륙 실패 실기체 ulog 진단 + 수정**(`9451861`). 2026-07-07 광주 비행 ulog(`02_17_49`) pyulog 분석 → `CommandTOL.altitude`(NAV_TAKEOFF param7)는 **AMSL 절대고도**인데 `transition_alt`(상대)를 그대로 실어 지면 AMSL(19.2m)보다 낮은 목표 → PX4 `Already higher than takeoff altitude`로 이륙취소·preflight disarm(배터리·GPS·SD 전부 정상, 무관). 수정: `takeoff_request_fields(transition_alt, home_amsl)`→`altitude=home_amsl+transition_alt`(+`/mavros/home_position/home` 구독·home 미수신 시 이륙 보류), CLIMBING 게이트를 지면기준 AGL로(`climbing_reached(…, ground_ref_up)`, 로컬 원점≠지면 2.11m 보정). pytest fc_ros 60/fc_bridge 44(신규 7). **SITL 재검증 대기** — `sitl_verification_log.md` 작업 H-2(재현엔 `PX4_HOME_ALT`로 지면 AMSL>transition_alt 필요 + geoid 확인). 그 전 2026-07-06 — ① **작업 H 완료·SITL PASS·커밋** — `offboard_node.py` `_step_arm_takeoff`를 `SetMode(AUTO.TAKEOFF)`→`CommandTOL(/mavros/cmd/takeoff, altitude=transition_alt)`로 교체(`7414c1d`). 요청 필드 조립은 순수함수 `fc_bridge/execution/state_logic.py::takeoff_request_fields()`로 분리(rclpy 없는 Windows에서도 pytest 가능). **1차 SITL 실패 → 원인 수정 → 재검증 PASS:** `latitude=0.0, longitude=0.0`을 "현재 위치"로 잘못 가정(MAVLink 관례는 **NaN**, `0.0/0.0`은 실좌표라 PX4가 고도 미상승 후 preflight disarm) → NaN으로 수정(`000f478`). WSL gz_standard_vtol `transition_alt:=50.0` 재검증 시 정상 상승·CLIMBING 통과 확인. 잔존 `guided_target`/"no origin" 경고는 MAVROS humble 알려진 QoS 코스메틱 이슈로 무해. pytest 130(fc_ros+fc_bridge) 전부 통과. 상세: `flight_plan.md`·`sitl_verification_log.md` "작업 H" ② 그 전: **planner 2종 본선 이식**(다른 계정 repo `suridouksuri`의 Fable 작업 회수): eta3 **v3.3**(2D 퇴화 WP NaN 근본수정)+**StraightLinePlanner**(신규)+`resolve_planner_name` 기체타입 자동선택 — `584cff3` ③ **transition_alt launch 오버라이드** `356ae5a` ④ 그 전: V2 검증·pull_ulog livelock 수정 `b580953`
- **다음(우선순위순, 2026-07-11 갱신 — ✈ vtol-실기체 결함으로 비행 보류 중):** ① **지금 가능(WSL SITL) — 작업 H-2 SITL 재검증** (`sitl_verification_log.md` 작업 H-2 체크리스트대로: `PX4_HOME_ALT`로 지면 AMSL>transition_alt 재현 후 AMSL 수정 확인 + geoid 정합) ② ~~실기체 검증~~ **결함 해소까지 보류**(🚁 하드웨어 해체, 이설된 ✈ vtol-실기체서 재개) ③ **지금 가능 — 작업 F**(임의 WP 견고성, [코드] Claude 단독) ④ **지금 가능 — V1·V3·V4**(하드웨어 불필요, Claude 단독) ⑤ **WSL SITL만 있으면 가능 — V2·V5** ⑥ RPi 배포 검증(pull_ulog 실측 속도)은 결함 해소 후
- **주의:** 최신 코드(작업 H 포함, `000f478`까지 커밋·푸시 완료)가 RPi에 **미전파** — RPi에서 `git pull` 필요(RPi 정본=호스트 `~/drone_ws/src/suridoksuri`, potato03kth). WSL(`~/suridoksuri-1`)은 이미 pull·재빌드 완료. `waypoints` 300 m·`v_cruise 20.0` 유지 결정(2026-06-30). V2/V5는 MAVROS 중지 필요(단독 링크). **작업 H가 실기체로 검증되기 전까지** 🚁 트랙의 "transition_alt를 낮게" 임시조치를 유지할 것 — SITL은 PASS했으나 실기체 미확인. **작업 H-2(AMSL 이륙고도 수정, `9451861`)는 단위테스트만 통과 — SITL 재검증 전이라 실비행 반영 금지.** geoid 리스크(MAVROS `geo.altitude`가 ellipsoid면 이륙 과상승) SITL 로그로 판별.
- **참조:** `fc_ros/fc_ros/nodes/offboard_node.py`(`_step_arm_takeoff`) · `fc_bridge/execution/state_logic.py`(`takeoff_request_fields`) · `fc_bridge/planning/planner_runner.py`(resolve_planner_name) · `vtol_sim/…/straight_line_planner.py`·`eta3clothoid_v3_1_planner.py`(v3.3) · `tools/flight_logs/VERIFY.md`(V1~V5) · `flight_plan.md`·`sitl_verification_log.md`(작업 H) · `docs/flight_plan.md`(작업 G)

### 🛩 sitl-vtol — ✅ 완료 (회귀검증 시에만 재개)

- **내용:** WSL SITL VTOL 검증. SITL-1~4 전부 PASS (2026-06-30)
- **재개 조건:** 비행 로직 코드 변경 후 회귀검증 필요 시 — `gz_standard_vtol`로 SITL-4 절차 재실행
- **참조:** `sitl_verification_log.md` · `sitl3_tuning_notes.md` · `archive/flight_plan_completed.md`(절차)

### ✈ vtol-실기체 — ▶ 활성 (기체 결함으로 비행 보류 — ⚠ 2026-07-18: 아래 상태와 모순되는 정황 발견, 미확인)

- **⚠ 확인 필요 (2026-07-18):** 🚁 mc-실기체 트랙에서 "부활한 MC 테스트기체"로 07-17·07-18 총 14회 실비행이 진행된 것을 발견했다. 이 기체가 아래 "결함으로 비행 보류" 상태인 VTOL 기체와 같은 물리 개체인지(→ 결함 해소됨?), 아니면 별도로 새로 조립한 기체인지 불명 — **이 블록 자체는 아직 갱신하지 않았으니 다음 세션에서 사용자에게 확인 후 갱신할 것.** 상세는 🚁 mc-실기체 트랙 "마지막"·"미확인" 참조.
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
| 원격접속 | Tailscale `100.67.27.83` (hostname `doksuri`, 계정 `suri`). **Claude용 SSH 키 등록됨**(2026-07-18, `claude-code-wsl-suridoksuri`, 이 WSL 개발컴 `~/.ssh/id_ed25519`) — 새 세션에서도 바로 `ssh suri@100.67.27.83` 가능, 비밀번호 불필요. `sudo`/`docker` 명령은 여전히 비밀번호 필요(그룹 미가입) — 안 되면 사용자에게 요청 |
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
