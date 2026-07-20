---
doc_type: session_log_archive
project: suridoksuri-1
period: 2026-07-03 ~
---

# 세션 로그 아카이브 — 2026-07

> `docs/session_log.md`에서 이동된 과거 세션 기록 (최신이 위).
> 현행 로그는 최근 8개 세션만 유지하며, 초과분은 `/session-log` 실행 시 이 디렉터리로 이동된다.

---

## 2026-07-07 — [main][mc-hw] ulog 재진단·작업 H 확정 + 실비행 SD카드 실패

**브랜치:** `dev--vision-computing-module`
**목적:** "20m 지정했는데 3m만 남" 사용자 재보고 → ulog 직접 재분석으로 원인 재확정, 실기체 배포 절차 정리, 다음 비행 시도

### 완료

- **ulog(`b9fc748d-...`) pyulog 직접 파싱으로 원인 재진단** — 이전 세션의 "transition_alt 기본값 미반영" 가설은 이 비행에는 **틀렸음**을 확인(사용자가 매번 수동 override했다고 반박, 근거 있음). 재분석 결과: `nav_state`가 AUTO_TAKEOFF→AUTO_LOITER만 거치고 **OFFBOARD 요청 자체가 `vehicle_command` 로그에 전무**(`flag_control_offboard_enabled` 비행 내내 0). 실측 고도(~19.7m)는 `home_alt(17.2)+MIS_TAKEOFF_ALT(2.5)`의 우연 일치였을 뿐, waypoint 20m와 무관
- **근본원인 확정** — `offboard_node.py` `_step_arm_takeoff`가 `SetMode("AUTO.TAKEOFF")`만 보내고 목표고도를 전혀 안 실어보냄 → PX4가 자체 `MIS_TAKEOFF_ALT`까지만 상승 후 자동 AUTO_LOITER. `transition_alt` 게이트가 그 실제 도달고도보다 높으면 OFFBOARD 요청이 영원히 안 나감
- **작업 H 계획 → main-code 트랙에 등록 → (세션 중 반영 확인)** `CommandTOL(/mavros/cmd/takeoff, altitude=transition_alt)`로 교체, SITL PASS·pytest 130 통과까지 완료된 상태를 문서(`flight_plan.md`/`session_status.md`)에서 확인 — lat/lon은 NaN(0.0/0.0은 실좌표로 오인식되는 실측 버그) 사용
- **RPi5 실비행 배포 절차 정리** — 최초 절차 설명에서 MAVROS 기동 단계가 누락됐던 것을 사용자가 지적해 보완(호스트 git pull → 컨테이너 `fc` 빌드 → MAVROS 별도 기동 → phase2 launch 순서 확정)
- **작업 G(record_flight.sh) 로깅 도구를 절차에 통합** — phase2.launch.py 직접 호출 대신 `record_flight.sh`로 감싸는 방식 + MAVROS와 pull_ulog 간 시리얼 포트 경쟁(종료 순서) 반영
- **PX4 SD카드 prearm check 확인** — SD카드 미삽입 시 arming 자체가 거부됨(웹 검색으로 확인). 오늘 실비행 시도가 **SD카드를 컴퓨터에 꽂아둔 채 까먹어 실패** — 작업 H 실기체 검증 아직 미완료
- **`docs/mc_flight_procedure.md` 신규 작성** — 로깅 사용(A)/미사용(B) 절차 전부 + 0단계 비행 전 체크리스트(SD카드 포함)를 고정 문서화, 다음 세션 "절차는?" 질문에 그대로 인용하도록 트랙 참조에 등록
- **메모리 갱신** — `project_rpi5_mc_bringup.md`에 이번 재진단으로 이전 오진단 정정, `feedback_flight_procedure_output.md` 신규(향후 "절차는?" 질문엔 두 버전 다 출력)

### 결정

- **"절차는?" 질문엔 로깅 사용/미사용 절차를 항상 둘 다 출력** — `docs/mc_flight_procedure.md`를 그대로 인용
- 비행 전 체크리스트에 **SD카드 삽입 확인**을 0순위 항목으로 고정

### 다음 세션

1. **🚁 mc-실기체 — 작업 H 실기체 검증 (최우선, 아직 미시도)** — `docs/mc_flight_procedure.md` 절차대로, SD카드 확인부터. `CommandTOL` 이륙이 실기체에서 정상 동작하는지, `altitude` AMSL/relative 해석 확인
2. PASS 시 "transition_alt를 MIS_TAKEOFF_ALT 이하로 낮춰라"는 임시조치 문서에서 제거
3. RPi 배포 검증(pull_ulog 실측 속도) 및 남은 V-unit

### 주의

> **오늘 실비행 시도 실패 — SD카드 미삽입으로 arming 거부, 비행 데이터 없음.** 작업 H는 여전히 SITL PASS만 확보된 상태, 실기체 미검증.
> `docs/mc_flight_procedure.md`가 이제 절차의 단일 진입점 — 이후 절차 변경 시 이 문서부터 갱신할 것.

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
