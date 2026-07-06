---
doc_type: session_log
project: suridoksuri-1
---

# 세션 로그

> 최신 세션이 위에 온다. `/session-log` 커맨드로 세션 종료 전 자동 작성.
> **최근 8개 세션만 유지** — 초과분은 `/session-log`가 `docs/archive/session_log_YYYY-MM.md`로 이동한다.
> 과거 기록: `docs/archive/session_log_2026-06.md` (2026-06-18 ~ 06-20)

---

## 2026-07-06 — [main][sitl] 작업 H: CommandTOL 이륙 + SITL PASS

**브랜치:** `dev--vision-computing-module`
**목적:** 작업 H(`AUTO.TAKEOFF`→`CommandTOL` 목표고도 명시 전달) 구현 → SITL 검증 → 커밋

### 완료

- **작업 H 구현** — `offboard_node.py` `_step_arm_takeoff`: `SetMode("AUTO.TAKEOFF")` → `CommandTOL(/mavros/cmd/takeoff, altitude=transition_alt)` 교체. 요청 필드 조립은 순수함수 `fc_bridge/execution/state_logic.py::takeoff_request_fields()`로 분리(Windows에 rclpy 없어 mock client 대신 이 방식 채택, 계획 대비 구현 방식만 변경). 커밋 `7414c1d`
- **1차 SITL 실패 → 원인 진단 → 수정 → 재검증 PASS** — `latitude=0.0, longitude=0.0`을 "현재 위치 사용"으로 가정했으나 틀림: MAVLink `MAV_CMD_NAV_TAKEOFF` 관례상 "현재 위치"는 **NaN**, `0.0/0.0`은 실좌표(null island)로 해석됨. 1차 SITL에서 QGC 모드는 AUTO.TAKEOFF로 전환됐으나 고도 미상승 → PX4 preflight 안전 disarm으로 실패 재현. `takeoff_request_fields()`의 lat/lon을 `NaN`으로 수정(`000f478`) 후 WSL gz_standard_vtol(`transition_alt:=50.0`) 재검증 PASS(정상 상승·CLIMBING 통과)
- **잔존 경고 원인 규명** — `mavros.guided_target: PositionTargetGlobal failed because no origin`는 우리 코드와 무관한 MAVROS humble의 알려진 QoS 코스메틱 이슈(웹 검색으로 확인, PX4 포럼에서도 "정상 동작"으로 보고됨) — 조치 불필요
- **pytest** — vtol_sim 6·fc_bridge 44·fc_ros 82 = 130~151 전부 통과(단계별)
- **커밋 3건 + 푸시 + WSL git pull·재빌드** — `7414c1d`(CommandTOL 교체) → `000f478`(NaN 수정) → `458d626`(SITL PASS 기록, sitl_verification_log.md·flight_plan.md·session_status.md 갱신)
- **문서 갱신** — `flight_plan.md`·`session_status.md`·`sitl_verification_log.md`에 작업 H 완료·PASS 반영

### 결정

- **CommandTOL 요청 필드는 lat/lon=NaN 고정** — 0.0/0.0 재도입 금지(실좌표로 해석되는 실측 확인된 버그)
- **작업 H 실기체 검증 전까지 🚁 트랙의 "transition_alt 낮게" 임시조치 유지** — SITL PASS는 실기체 보증 아님

### 다음 세션

1. **🚁 mc-실기체 트랙에서 작업 H 실기체 검증** — 다음 비행에서 CommandTOL 이륙 정상 동작 확인. PASS 시 "transition_alt 낮게" 임시조치 제거
2. **RPi 배포 검증** — pull_ulog 실측 속도·byte 동일(작업 G 속도 판정 최종)
3. 남은 V-unit(V1·V3·V4·V5) + 작업 F(임의 WP 견고성)

### 주의

> **RPi에 최신 코드(`458d626`까지) 미전파** — RPi에서 `git pull` 필요(정본 `~/drone_ws/src/suridoksuri`, `potato03kth`). WSL(`~/suridoksuri-1`)은 pull·재빌드 완료됨.
> `CommandTOL.altitude`가 AMSL/relative 중 어느 쪽으로 해석되는지 SITL에서 명확히 확인 안 됨 — 실기체 검증 시 함께 확인.

---

## 2026-07-06 — [main] planner 이식(다른 repo 회수) + transition_alt 오버라이드

**브랜치:** `dev--vision-computing-module`
**목적:** 실기체 OFFBOARD 실패 관련 — 다른 계정 repo(Fable 작업)의 planner 수정 회수 + MC 저고도 테스트 배선

### 완료

- **RPi 저장소 구조 규명** — 정본은 `~/drone_ws/src/suridoksuri`(`potato03kth`, `-v ~/drone_ws:/drone_ws` 마운트, 컨테이너엔 git 없음 → 호스트에서 `git pull`). nested `~/drone_ws/suridoksuri/suridoksuri`는 **다른 계정(`suridouksuri`) repo**라 무관 — 그동안 여기서 git해 어긋났던 원인. 정본에서 pull하면 됨(정리 불필요)
- **planner 2종 본선 이식** (stray repo Fable 5 작업, `584cff3`): ① eta3 **v3.3** — 2D 퇴화 WP(이륙점 수직 상방 천이고도 WP) 병합 + s strictly-increasing → `np.gradient` NaN 제거(offboard 경로추종 미시작 근본결함). 우리 SITL-3 코드에 수동 병합 ② **StraightLinePlanner(신규)** — 3D 직선·NR 없음·수직 NaN 불가 ③ **자동선택** `resolve_planner_name`: `planner:"auto"` 기본 → mc=straight/vtol=eta3, 명시 우선. sim 검증 vtol_sim 6·fc_bridge 44(신규 resolve 6)·fc_ros 82 pass + e2e 스모크
- **transition_alt launch 오버라이드** (`356ae5a`) — 07-03 OFFBOARD 미진입 원인(`transition_alt:50` 미도달→CLIMBING 무한대기) 대응. 계획했던 `transition_alt:=4.0` 인자가 **미선언이라 무시되던 것**을 실제 먹히게 함(v_cruise/waypoints와 동일 패턴)

### 결정

- planner **기체 타입 자동선택**(mc→straight, vtol→eta3), `planner` 명시 시 우선 — 요구사항 반영
- stray repo `suridouksuri`는 **폐기** — 필요한 planner 코드 다 회수함
- RPi 정본 = `potato03kth`, pull 위치 = `~/drone_ws/src/suridoksuri`

### 다음 세션

1. **mc-실기체 첫 실질 OFFBOARD 테스트** — RPi `git pull`(356ae5a까지)+`colcon build --packages-select fc_ros` 후 `vehicle_type:=mc transition_alt:=4.0 waypoints:="[0,0,4, 8,0,4]"`. transition_alt(CLIMBING 통과)+straight(짧은 레그 NaN 없음)가 맞물려야 첫 OFFBOARD
2. main-code: RPi pull_ulog 실측(작업 G 속도 판정) + 남은 V-unit(V1은 재작성으로 갱신 필요)
3. planner·transition_alt 벤치·실비행 검증

### 주의

> planner·transition_alt는 **sim만 검증, 실기체 미검증.** RPi는 `git pull`(356ae5a까지)+rebuild 후 비행 — 584cff3까지만 받았으면 launch 변경 누락으로 transition_alt 또 무시됨.
> 07-03 배터리 새그(12V→10.2V 페일세이프)·나침반 결함·가속도계 클리핑은 **하드웨어 점검** 항목(코드 무관).

---

## 2026-07-06 — [main] V2 검증·pull_ulog 다운로드 livelock 수정

**브랜치:** `dev--vision-computing-module`
**목적:** `tools/flight_logs/VERIFY.md` V2 실행 — pull_ulog.py를 실제 PX4 SITL 상대로 검증

### 완료

- **V2 실링크 검증(WSL SITL, MAVROS 중지)** — `--list` 84/84 rootfs 일치(PASS). 다운로드에서 **livelock(FAIL)** 발견: PX4가 `log_request_data(ofs,0xFFFFFFFF)`에 로그 전체를 UDP 버스트로 전송 → ~78% 손실 + "gap마다 남은 전체 재요청" → 무진행 무한 hang (faulthandler 스택 덤프로 recv 루프 확정)
- **`download_log` 재작성** — 윈도우드 요청 + 누락 구간만 재요청 + offset seek 기록 + 진행기반 stall 가드/하드 타임아웃(hang 제거) + 불완전 시 부분파일 삭제 후 raise(조용한 exit0 제거) + UDP SO_RCVBUF 확대(serial 무영향). **serial 경로 동작 불변**
- **순수함수**(`merge_intervals`/`missing_ranges`/`coverage`) 추출 + **fake-link 테스트**(serial 무손실=바이트 동일, lossy/reorder/dead) — **pytest 37 pass**(신규 24)
- **V2 재실행 PASS** — 다운로드 sha256 원본과 바이트 동일, `ulog_info` 정상, ~60 KB/s
- **커밋·푸시** — Task G 도구 최초 커밋 + 수정(`b580953` [main], origin 반영). 스크립트 LF·+x 고정(RPi CRLF/실행권한 대비)

### 결정

- pull_ulog 다운로드는 **윈도우드 손실복구 유지** — whole-log 재요청으로 되돌리지 말 것(livelock 재발)
- SITL UDP 속도(~60 KB/s)는 PX4/MAVLink 페이싱 지배 → serial과 유사할 것. V2 속도 판정 "충분"(15MB≈4.2분)이나 **최종은 RPi USB 실측 우선**

### 다음 세션

1. **RPi 배포 검증** — 실기체 USB 직결 pull_ulog 실측 속도·byte 동일 → 속도 판정 최종(15MB가 5분 초과면 작업 G-2 등록)
2. 남은 V-unit: V1(재작성으로 갱신 필요)·V3(record_flight.sh)·V4(fetch_logs.ps1)·V5(dry-run 통합)
3. 이후 작업 F(임의 WP 견고성)

### 주의

> **신 pull_ulog 미전파** — 개발컴 `~/suridoksuri-1`·RPi는 `git pull` 해야 반영(WSL엔 `fc_ros_params.yaml` 미커밋 변경도 있음). V2/V5는 MAVROS 중지 필요(단독 링크). SITL을 Windows에서 몰 때 wsl.exe 경유 `*`glob·`$()` 뭉개짐 주의 — `/mnt/d` 실행 + `MSYS_NO_PATHCONV=1`

---

## 2026-07-06 — [main] 문서 재구성·트랙 보드·작업 G 계획

**브랜치:** `dev--vision-computing-module`
**목적:** 세션 추적 토큰 낭비·컨텍스트 오염 해결 + 병행 트랙(메인코드/드론테스트/SITL/실기체) 전환 시 상태 유실 해결

### 완료

- **문서 3층 재구성** — flight_plan.md 다이어트(41→21KB, 완료된 작업 A~E·SITL-1~4 상세는 `docs/archive/flight_plan_completed.md`로), session_log.md 최근 8개 롤링(과거분 `docs/archive/session_log_2026-06.md`), session_status.md 현행화
- **트랙 보드 도입** — session_status.md를 트랙 4개(🔧main-code/🚁mc-실기체/🛩sitl-vtol/✈vtol-실기체) 블록 구조로 전환. `/session-log`는 건드린 트랙 블록만 갱신 → 병행 작업 간 상태 덮어쓰기 차단
- **CLAUDE.md FC 절차 정형화** — 진입(활성 트랙 블록만 읽기)·자가 복구(기록 없이 끝난 세션 감지 시 git log/diff로 보드 복원)·트랙 전환 규칙·종료(/session-log)
- **/session-log 커맨드 확장** — 로그 + 트랙 블록 갱신 + 8개 초과 아카이브 3단계, `축약` 모드 추가
- **phase2.launch.py 오버라이드 추가** — `v_cruise:=`/`waypoints:=` launch 인자 (빈 값이면 YAML). pytest **120/120 PASS** (이상 커밋 `89ab44f`)
- **작업 G 계획 확정·등록** — 비행 로그 자동수집·분석 체계(record_flight.sh 래퍼 + MAVFTP ulog 회수 + rosbag 토픽 11개 + 개발컴 fetch)를 flight_plan.md에 상세 등록 (커밋 `4ac7df7`)
- **트랙 보드에 🚁 첫 offboard 비행 부분 성공 반영** — 사용자 보고, 상세(된 것/안 된 것) 미기록 상태로 표시

### 결정

- 상태 관리를 세션 단위 → **트랙 단위**로 (진입 = "○○ 트랙 재개" 한마디)
- **테스트 임시 파라미터는 yaml 수정 금지** — launch 인자로만
- **트랙 전환 전 WIP 커밋** — 메시지에 `[main]`/`[mc-hw]`/`[sitl]`/`[vtol-hw]` 태그
- llm wiki 형식 미도입 (유지비 대비 이점 없음 — 포인터 기반 lazy loading으로 충분)
- 세션 로그 롤링 기준 8개
- 로그 수집 체계는 **main-code 트랙 작업 G**로 (전 트랙 공용 인프라는 만드는 곳 한 곳 — 새 트랙 아님)
- **GitHub 로그 업로드 안 함** — 대용량 바이너리로 git 팽창·LFS 쿼터·`results/` 규칙 충돌. RPi→개발컴 직접 fetch(분석하는 곳이 목적지) + 공유 필요 ulog만 선택적 Flight Review

### 다음 세션

1. **main-code 트랙 재개 — 작업 G 실행** ([코드] 부분은 Claude 완결, WSL dry-run·RPi 검증만 사람)
2. 🚁 mc-실기체 재개 시: 첫 offboard 부분 성공의 상세(된 것/안 된 것)부터 확인·기록
3. 작업 G 완료 후 🔧 작업 F (임의 WP 견고성)

### 주의

> `v_cruise: 20.0`·`waypoints: 300 m`는 **유지 결정**(2026-06-30, `sitl3_tuning_notes.md`) — 복구 대상 아님. 실미션 좌표 확정 시 waypoints만 yaml 두 곳 교체.
> 🚁 첫 offboard 부분 성공의 상세 미기록 — mc-hw 세션 진입 시 사용자에게 확인해 트랙 블록부터 갱신.

---

## 2026-07-03 — 실기체 MC 브링업 (RPi5/24.04 + PX6C)

**브랜치:** `dev--vision-computing-module`
**목적:** RPi5(Ubuntu 24.04) + Pixhawk 6C **순수 MC** 테스트기체에 fc_ros 배포·검증 (SITL-5 변형)

### 완료

- **`vehicle_type` 런타임 파라미터 추가** — `"vtol"`(기본)|`"mc"`. MC는 FW 천이 2단계(TRANSITION_FW/TRANSITION_MC) 생략하고 CLIMBING→STREAMING, FOLLOWING→HOLD 직행. 코드 분기만, **VTOL 동작 불변**. 순수함수 `after_climb_state`/`after_following_state` + 테스트 4개 추가(90 passed)
- **launch 런타임 오버라이드** — `phase2.launch.py vehicle_type:=mc` + yaml 기본값. 코드 교체 없이 파라미터로 MC 전환
- **RPi5 배포 환경 구축** — Docker `ros:humble` 컨테이너(이름 `fc`, 항상 `sudo`), MAVROS·numpy 설치, fc_ros colcon 빌드, fc_bridge+vtol_sim은 PYTHONPATH(`/drone_ws/src/suridoksuri`)로 로드
- **Pixhawk 6C 펌웨어 ArduCopter→PX4 교체** — PC 데스크톱 QGC로 플래시, 에어프레임/캘리브레이션 재설정, **수동비행 검증 성공**

### 결정

- **RPi5(24.04)는 Docker Humble로 운용** — Humble이 22.04 전용이라. **개발컴은 22.04/Humble 유지**(업그레이드 안 함). 네이티브 Jazzy 미채택("오류 나면 안 됨" 우선 → 검증된 Humble 환경 재현)
- **MC 추종은 위치 setpoint 재사용** — 속도+L1 복원 안 함. 속도는 PX4 MPC가 관장, `v_terminal`/`decel_dist`는 MC에서 무의미
- **MC 검증은 코드포크가 아니라 파라미터 스위치로** — SITL은 gz_x500(=MC)로 선검증

### 다음 세션

1. **MAVROS 링크 문제 해결** — RTT 2~5초·heartbeat 플래핑·935 params 정체. 태블릿 QGC 끊고 **USB 직결**로 링크 안정화부터
2. **AUTO.TAKEOFF 미실행 진단** — offboard가 이륙명령 발행 안 함. (a) MAVROS 서비스 미준비인지 (b) PX4 GPS 락 없어 AUTO.TAKEOFF 거부인지 `statustext`로 판별
3. **커밋** — `vehicle_type` 변경 등 이번 세션 전체 미커밋

### 주의

> **근본 교훈: 6C는 ArduCopter였다.** 우리 코드·SITL 검증은 전부 **PX4 전용**(모드명·AUTO.TAKEOFF·OFFBOARD·vtol_state). 실기체는 PX4 확인부터.
> **AUTO.TAKEOFF는 GPS 락 필수** — 수동비행 성공 ≠ GPS 락. 실내/벤치 불가.
> **웨이포인트 비퇴화 필수** — 시작=끝 동일하거나 초단거리 레그면 플래너 divide-by-zero(NaN).
> **이번 세션 전체 미커밋.**

---

## 2026-06-30 — SITL-3 해결 (FW 위치 setpoint 전환)

**브랜치:** `dev--vision-computing-module`
**목적:** SITL-3 두 핵심 버그(천이 원호, 경로가 초기 heading에 종속) 진단·수정

### 완료

- **근본 원인 규명** — PX4 FW 오프보드는 velocity setpoint를 무시(flower-pattern 선회), 위치 setpoint 필수. 천이 원호·heading 종속·FOLLOWING 미진입이 **단일 원인**이었음 (frame_id 가설 기각)
- **FW 활성 구간 전부 위치 setpoint 전환** — STREAMING/FOLLOWING/천이/역천이, lookahead 70 m. 천이·역천이까지 직선화
- **신규 HOLD 상태** — 역천이 오버슈트 후 MC로 WP1 복귀·홀드 → WP1 지점 착륙. TRANSITION_MC keepalive로 151 m RTL 해결, 역천이 동향 45° 꺾임 해결
- **단위 테스트 108 passed** (순수 함수 `target_point_ned`·`wp1_land_ready` 신규) + SITL 직선 300 m 전체 시퀀스 정상 확인
- **문서/메모리 갱신** — `sitl3_fix_plan.md` 해결기록 재작성, `sitl3_tuning_notes.md` 신규, 메모리의 틀린 frame_id 진단 정정 + FW 위치 setpoint 메모리 신규
- **검증단 점검** — 노드 테스트가 로직 복사본을 검증하는 "거울 테스트" 구조 확인 → 신규 로직은 순수 함수로 추출해 실제 검증

### 결정

- FW 경로추종은 **OFFBOARD + 위치 setpoint** 채택 (AUTO.MISSION 미채택 — 향후 vision 동적목표 대비)
- 사전가속(Phase 2.5) **폐기 확정** — 불필요
- WP1 착륙은 **HOLD 상태로 복귀·홀드 후 착륙** 확정

### 다음 세션

1. **임시값 복구** — `v_cruise: 20→15`, `waypoints: 300→실제 미션 좌표` (yaml 2곳)
2. **SITL-4 전체 사이클** — L자/사각형 경로 FW 추종 + 천이 가속도 측정
3. **커밋** — 이번 세션 변경 미커밋 상태

### 주의

> **이번 세션 변경 미커밋** — 사용자 확인 후 커밋 예정
> **임시값 유지 중** — `v_cruise: 20`, `waypoints: 300 m`. 상세 `docs/sitl3_tuning_notes.md`

---

## 2026-06-24 — SITL-3 Bug 2 재수정 (MC yaw rate 헤딩 정렬)

**브랜치:** `dev--vision-computing-module`
**목적:** 이전 세션 Bug 2 수정이 동작하지 않아 재수정 — MC OFFBOARD에서 yaw가 바뀌지 않는 근본 원인 해결

### 완료

- **Bug 2 초기 수정 실패 원인 확인** — velocity 세트포인트만으로는 PX4 MC가 yaw를 변경하지 않음 (MC 위치 제어기에서 yaw는 독립 축)
- **`SetpointPublisher.publish(yaw_rate=0.0)` 파라미터 추가** — `fc_ros/fc_ros/adapters/setpoint_publisher.py`: `twist.angular.z` 활성화
- **`_step_transition_fw` Phase 2 yaw rate P제어 추가**
  - Phase 1: hover 세트포인트 20틱으로 OFFBOARD 프라이밍 (HOLD에서는 무시됨)
  - Phase 2: MC OFFBOARD hover(`np.zeros(3)`) + yaw rate P제어(`-heading_err * 1.0`, 포화 ±1 rad/s)로 헤딩 정렬
  - Phase 3: 헤딩 정렬 완료 → WP 방향 전진 + MC→FW 천이 명령
  - Phase 4: vtol_state==FW 대기 → STREAMING
- **`v_cruise` 임시 20.0 m/s** — terminal 속도 반영 확인용 (검증 후 복구 필요)
- **메모리 저장** — `project_sitl_state.md` 업데이트, `feedback_px4_mc_offboard_yaw.md` 신규 작성
- **pytest 25/25 PASS**

### 결정

- PX4 MC OFFBOARD에서 헤딩 정렬은 반드시 `twist.angular.z` yaw rate를 함께 보내야 한다 (velocity만으로는 불가)
- yaw rate 부호 규칙: `heading_err` 양수(NED CW 필요) → ENU `angular.z` 음수 (`-heading_err`) — SITL에서 반전 가능성 있음

### 다음 세션

1. **WSL 재빌드** — `colcon build --packages-select fc_ros && source install/setup.bash`
2. **SITL-3 재실행** — Bug 2 yaw 정렬 동작 확인 (드론이 WP 방향으로 회전 후 직선 천이하는지 관측)
3. yaw_rate 부호 검증 — 반대로 돌면 `setpoint_publisher.py` 37번째 줄 `-heading_err` → `+heading_err` 수정
4. terminal 속도 확인 완료 후 `v_cruise: 20.0` → `15.0` 복구

### 주의

> **`v_cruise: 20.0` 임시 변경 중** — `fc_ros/fc_ros/params/fc_ros_params.yaml` 복구 필요
> **yaw_rate 부호 SITL 미검증** — 드론이 WP 반대 방향으로 돌면 `setpoint_publisher.py:37`에서 부호 반전

---

## 2026-06-20 — SITL-3 버그 3종 수정 (FW 천이·방향·추종)

**브랜치:** `dev--vision-computing-module`
**목적:** SITL-3 경로 추종 중 발견된 버그 3종 수정 및 dry-run 검증

### 완료

- **dry-run 검증 PASS** — 경로 A(직선 100m)/B(L자 200m)/C(사각형 400m) 끝점 속도 전부 v_terminal=15.2 m/s 일치
- **cp949 인코딩 버그 수정** — `eta3clothoid_v3_1_planner.py:373` 한국어+특수문자(`⚠`) → ASCII 영문으로 교체
- **버그 1 수정 (FW 천이 명령 간헐 실패)** — `_step_transition_fw`: `wait_for_service` 블로킹 제거 → `service_is_ready()` 비차단 확인 + 30틱마다 재시도, vtol_state==1(천이 중) 이면 대기
- **버그 2 수정 (천이 후 방향 이상)** — STREAMING 핸들러: 10틱 고정 대기 → `_heading_aligned_with_path()` (현재 속도 방향 vs 첫 세그먼트 ≤60°) 통과 시에만 OFFBOARD 요청
- **버그 3 수정 (FOLLOWING 종료 판정 안 남)** — `_step_following`: FW 모드(`vtol_state==4`) 시 velocity 세트포인트 → PoseStamped lookahead 위치 세트포인트(`/mavros/setpoint_position/local`) 전환; MC 모드는 기존 속도 세트포인트 유지; 10틱마다 `cte`·`dist_end`·`seg` 로그 추가
- **`cmd_vel_frame_id` 수정** — YAML `"base_link"` → `"local_origin"` (velocity가 body frame으로 회전되는 오류 방지)
- **`_heading_aligned_with_path()` 헬퍼 추가** — 순수 numpy, rclpy 없이 테스트 가능
- **pytest 25/25 PASS** — 회귀 없음

### 결정

- FW OFFBOARD에서 velocity 세트포인트는 불안정 (PX4 FW yaw 오실레이션) → position setpoint 사용 확정
- FW loiter 후 OFFBOARD 진입 조건: 속도-경로 각도 60° 이내 (cos > 0.5) — 하드코딩 틱 대기 폐기

### 다음 세션

1. **WSL 재빌드** — `colcon build --packages-select fc_ros && source install/setup.bash`
2. **SITL-3 재실행** — 경로 A(직선 100m)로 3종 버그 수정 확인
3. 버그 수정 확인 후 경로 B·C 추종 테스트 진행

### 주의

> FW FOLLOWING 중 lookahead 위치 세트포인트 고도는 `self._transition_alt`로 고정. 실제 비행에서 지형 고도 변화 있을 경우 향후 수정 필요.
> `_heading_aligned_with_path`는 `self._pts`가 `__init__`에서 설정된 후에만 유효 — STREAMING 진입 시점에는 항상 설정돼 있음.

