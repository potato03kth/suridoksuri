---
doc_type: session_log_archive
project: suridoksuri-1
period: 2026-07-03 ~
---

# 세션 로그 아카이브 — 2026-07

> `docs/session_log.md`에서 이동된 과거 세션 기록 (최신이 위).
> 현행 로그는 최근 8개 세션만 유지하며, 초과분은 `/session-log` 실행 시 이 디렉터리로 이동된다.

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
