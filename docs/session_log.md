---
doc_type: session_log
project: suridoksuri-1
---

# 세션 로그

> 최신 세션이 위에 온다. `/session-log` 커맨드로 세션 종료 전 자동 작성.
> **최근 8개 세션만 유지** — 초과분은 `/session-log`가 `docs/archive/session_log_YYYY-MM.md`로 이동한다.
> 과거 기록: `docs/archive/session_log_2026-06.md` (2026-06-18 ~ 06-30) · `docs/archive/session_log_2026-07.md` (2026-07-03 ~)

---

## 2026-07-18 — [mc-hw] 현장 원격 리빌드 + flight09 로그수집 + 고도미달 진단

**브랜치:** `dev--vision-computing-module`
**목적:** 사용자가 비행장에서 RPi5 터미널에 접근 못하는 상태로 "리빌드해달라" 요청 → 이후 비행(flight09) 로그 수집·고도미달("천이 명령 미하달") 원인 진단까지 이어서 처리

### 완료

- **원격 리빌드** — SSH로 RPi5 접속, `docker exec fc colcon build --packages-select fc_ros` 실행. 처음엔 `sudo docker exec`가 비밀번호 요구로 막혔으나 사용자가 `suri`를 `docker` 그룹에 추가해줘서 sudo 없이 해결. 빌드 성공 확인 + `install/`이 최신 소스와 `diff` 일치 확인. 이 리빌드가 별도 백그라운드 세션이 동시에 진단 중이던 "flight01~08 stale colcon build" 근본원인을 실제로 해소한 조치였음(교차참조, 아래 결정 참조)
- **flight09 로그 수집** — 리빌드 직후 사용자가 실비행(`transition_alt:=4.0`, 삼각 왕복 웨이포인트) 진행. `record_flight.sh`로 rosbag+launch.log는 정상 수집(`logs/2026-07-18_flight09/`). ulog는 FC(Pixhawk)가 비행 후 전원이 내려간 것으로 판단돼(USB enumeration은 유지, `/dev/ttyACM0` 3초간 0바이트 수신) 회수 실패 — 사용자가 바빠서 재시도 보류, 미회수 상태로 종료
- **고도미달 진단** — 사용자 보고("목표 4.0m인데 3.6m가 최대, 천이 명령 미하달")를 rosbag 직접 디코드(`rosbag2_py`+`deserialize_message`로 `/mavros/state`·`/mavros/local_position/pose` 추출)로 검증. AMSL 계산(`home_amsl+transition_alt`)은 정확했음(리빌드 덕분) — 실제 원인은 PX4가 `AUTO.TAKEOFF` 진입 4.66초 만에 자체적으로 `AUTO.LOITER`로 복귀(`pose.z` 최댓값 3.63m)하고, 이후 조종사 재이륙 시도도 동일 패턴 반복 후 42초간 LOITER 고착 — **OFFBOARD 모드 진입이 이 비행 내내 한 번도 없었음**. `offboard_node._step_climbing()`이 PX4 모드 이탈을 감지하지 않는 순수 폴링 설계라 소프트웨어가 이를 알아채지 못했던 것으로 결론
- **RPi5 네트워크 단절 대응** — 진단 도중 RPi5가 Tailscale에서 완전히 끊겨(ping 100% 손실) 약 한 턴 동안 재접속 대기. 사용자가 "지금 연결했다"고 알려온 뒤에도 실제 재접속까지 재시도 필요했음(즉시 복구 아님)
- **다른 백그라운드 세션과의 협업** — flight01~08을 전수분석 중이던 별도 세션이 `worktree-agent-*` 브랜치에 남긴 커밋(근본원인: stale colcon build)을 발견 → `dev--vision-computing-module`로 fast-forward 병합 후 이 세션의 flight09 발견과 교차참조해 `session_status.md`에 함께 기록

### 결정

- **ulog 재시도 중단** — 사용자가 "바쁘니까 받지 마라"고 명시적으로 중단 지시. rosbag 기반 분석만으로 원인 결론은 이미 확정적이라고 판단해 그대로 수용, ulog는 다음 기회로 미룸
- **코드 수정은 보류** — `_step_climbing()`의 PX4 모드 이탈 감지 부재를 발견했지만 이번 세션에선 진단까지만 하고 실제 코드 변경은 하지 않음(사용자 판단 대기)

### 다음 세션

1. FC 전원 재연결 시 ulog id=13 회수 + `MIS_TAKEOFF_ALT` 파라미터 확인으로 "PX4가 왜 3.6m대에서 자체 이륙완료 처리했는지" 근본원인 확정
2. flight09 rosbag엔 배터리 토픽이 없어 전압붕괴 가설(위 flight01~08 분석 ④) 재현 여부 확인 불가 — ulog 확보 후 대조
3. `_step_climbing()` AUTO.TAKEOFF 이탈 감지·재시도 로직 추가 여부 사용자와 논의
4. `logs/2026-07-18_flight09/`가 아직 git 미커밋 — 다음 로그 커밋 배치에 포함

---

## 2026-07-18 — [main][mc-hw] 라파5 원격 로그 조사 → 문서 뒤처짐·인프라 버그 2건 발견

**브랜치:** `dev--vision-computing-module`
**목적:** 사용자가 완료한 실비행의 로그 확인 요청 → SSH 원격 접속 체계 구축 → 실제 접속해 조사 → 발견 사항 정리

### 완료

- **Tailscale SSH 키 등록** — RPi5(`100.67.27.83`, hostname `doksuri`)에 이 WSL 개발컴용 ed25519 키(`claude-code-wsl-suridoksuri`) 등록. 이후 세션에서 비밀번호 없이 바로 SSH 가능(`sudo`/`docker`는 여전히 비밀번호 필요, 그룹 미가입)
- **원격 조사로 문서-현실 괴리 발견** — `docs/session_status.md`엔 "✈ vtol-실기체: 07-09 이후 기체 결함으로 비행 보류"로 남아있었으나, 실제 `logs/` 디렉터리엔 07-07·07-11·07-17(6회)·07-18(8회, 오늘) 비행 폴더가 존재. 07-17·07-18 14회는 문서에 전혀 기록되지 않은 채 진행됨(`vehicle_type:=mc`)
- **작업 G(로그 인프라) 실사용 버그 2건 확정** — ① RPi 호스트에 pymavlink 미설치로 `pull_ulog.py` 자동회수가 지금까지 한 번도 성공한 적 없었음(실패가 어디에도 기록 안 돼 발견이 늦어짐) ② `record_flight.sh`를 컨테이너 `fc` 안 root로 실행해 `logs/<날짜>_flightNN/`이 root 소유가 되어 `suri` 계정 쓰기 불가
- **오늘(07-18) 비행 11개 ulog 전량 회수** — RPi 호스트에 pip 부트스트랩(`--user --break-system-packages`)으로 pymavlink 설치 → FC에서 직접 `.ulg` 11개 다운로드. 8개(id3~10)는 기존 `flight01~08`(rosbag+launch.log) 폴더와 시각 매칭해 완전한 폴더로 합침, 3개(id0~2, `record_flight.sh` 쓰기 전 로그)는 대응 rosbag/launch.log 없이 `logs/2026-07-18_unlogged/`에 "비행기록 부족함"으로 보관. root 소유 폴더 문제로 RPi 쪽 직접 write는 실패해 staging 폴더 경유 → 이 개발컴으로 scp 후 로컬에서 재조립
- **`record_flight.sh` 수정** — 종료 시 `$FLIGHT_DIR`을 `$LOG_ROOT` 소유자로 chown(best-effort, 실패해도 스크립트 안 죽음)해 향후 비행부터 root 소유 문제 방지. `bash -n` 통과. (`test_flight_logs.py`는 이 WSL에 pytest/pymavlink 미설치라 로컬 실행 못함 — 대상이 `pull_ulog.py` 순수함수라 이 변경과 무관해 회귀 위험은 낮음)
- **`docs/flight_plan.md` 작업 G 표 최신화** — "계획 확정, 미착수" → "✅ 완료"로 수정 (실제로는 이미 완료·검증됨)
- **RPi `git pull` (서브에이전트 위임)** — 처음엔 `origin`에 `07681d3`가 미푸시 상태라 반영 안 됨을 서브에이전트가 정확히 진단·보고 → 사용자 승인으로 push 후 재실행, RPi에 chown 수정 반영 확인(`grep chown record_flight.sh`)
- **RPi 소유권 정리 확인 + 07-18 로그 RPi 원본도 완결** — 사용자가 RPi에서 `sudo chown -R suri:suri logs/` 직접 실행 → SSH로 확인. 로컬 스테이징에 있던 07-18 ulog 8개도 (이제 쓰기 가능해진) RPi 원본 `flight01~08` 폴더로 이동해 RPi 쪽 사본도 완전해짐
- **비행로그 git 커밋 방침 전환** — "GitHub 업로드 안 함"(2026-07-06 결정) 재검토를 사용자에게 요청 → **일반 git 커밋으로 전환**(LFS 아님, 트레이드오프 인지하고 승인) 결정. `.gitignore` 루트의 `logs/` 제외 규칙 제거(`*.log`는 유지하되 `!logs/**/*.log`로 예외 처리해 `launch.log`/`rosbag_record.log`도 추적되게), `tools/flight_logs/README.md`·`flight_plan.md` "업로드 방침" 갱신. 오늘 07-18 로그 53개 파일(rosbag+ulog+launch.log 등) 커밋

### 결정

- 서브에이전트에 `record_flight.sh` 수정을 위임 시도했으나 `isolation: worktree`가 오래된 브랜치에서 갈라진 고아 워크트리를 만들어 `tools/flight_logs/`가 아예 없는 상태로 실패 — **이 프로젝트는 세션이 in-place로 작업하도록 설정돼 있어 worktree 격리를 쓰면 안 됨**(에이전트는 이를 정확히 감지하고 파일을 지어내지 않은 채 보고했음 → 직접 적용). 향후 서브에이전트 위임 시 isolation 옵션 쓰지 말 것. 반대로 순수 SSH/git 원격 작업(RPi git pull)은 isolation 없이 위임해 문제없이 완결됨
- **비행 로그를 git에 그대로 커밋하기로 번복** — 2026-07-06 "GitHub 업로드 안 함"(대용량 바이너리 이력 팽창 우려) 결정을 사용자가 다기기 공유 편의를 우선해 뒤집음. git 이력이 로그만큼 영구히 커지고 clone이 느려지는 트레이드오프는 알고 승인한 것 — 되돌리려면 히스토리 재작성(rebase/filter-repo) 같은 파괴적 작업이 필요해짐을 유의

### 다음 세션

1. **RPi pymavlink 설치를 임시 우회(`~/.local`, `--break-system-packages`)에서 영구화** — 컨테이너 이미지 또는 셋업 스크립트/문서에 반영
2. 07-17·07-18 14회 비행 notes.md(관찰/결론) 전부 비어있음 — 조종사가 채워야 실제 비행 평가 가능
3. 앞으로 `record_flight.sh`로 생기는 새 플라이트 폴더는 평소 커밋 워크플로에 포함(잊지 말 것 — 더는 `.gitignore` 자동 제외가 아님)

> ✈ vtol-실기체 vs 🚁 mc-실기체 정체 확인은 **해결됨(2026-07-18, 사용자 확인)** — 별도 물리 기체(Pixhawk·ESC 모두 다름, 외형만 동일)로 두 트랙 블록 정리 완료.

### 주의

> `logs/` 방침이 바뀌어 이제 **비행 로그는 커밋 대상**이다 — 새 플라이트 폴더 생성 후 커밋을 잊지 말 것. 저장소 용량이 계속 늘어나는 것은 의도된 트레이드오프.
> RPi `sudo`/`docker` 권한이 이 세션엔 없음(비밀번호 필요) — 컨테이너 안쪽 작업이 필요하면 사용자에게 요청할 것.

---

## 2026-07-15 — [vision] 계획 갭 반영·headless main.py·테스트환경

**브랜치:** `dev--vision-computing-module`
**목적:** vision 트랙 재개 — 목적·구현범위 점검 → 계획서 갭 반영 → 개발단계 디버깅 착수(headless main.py) → 실테스트 환경 구축

### 완료

- **vision 트랙 이해·구현범위 점검 → 계획서(`vision_plan.md`) 갭 8건 반영** (커밋 `af32ccf`): ④단순착륙 전략공백/내부불일치(§2 표+§5.6 신설), TERMINAL 데드레코닝·blob 타겟 스케일 융합규칙(§5.1), 빨강 ①원↔③십자 혼동 방어(§5.4), `TargetEstimate` 좌표 프레임 계약 미확정(§7.1+§10), CC 명령 수신 시임 `CommandSource`(§7.2), **개발단계 디버깅 워크플로 §7.9 신설**, 성능/지연 예산 등 §10/§11
- **main.py headless-safe** — `--display {none|window|file|stream}`, 모든 GUI(imshow/waitKey)를 window 뒤로 격리, 기본 `none`=GUI 미호출(드론 헤드리스 크래시 원천 제거). `tests/test_main.py` 회귀 4종(none=imshow 0회 불변식)
- **테스트 규칙 정비** — `vision/CLAUDE.md`에 테스트 방법 + 단위별 필수 테스트 표(15단위, ✅4/TODO 다수/폐기 1) + 공통 규칙 4. `vision/requirements.txt`(ASCII) 신설, `.gitignore`에 `.venv/` 등
- **개발컴 실테스트 환경 구축·통과** — `.venv`(Python 3.10.11, opencv-python 5.0.0.93, numpy 2.2.6, PyYAML 6.0.3, pytest 9.1.1). `pytest vision/tests/` → **16 passed**

### 결정

- **실테스트 환경 4구분 기록·검증**(사용자 지시) — 개발컴(항상 필수, 이번에 설치완료)·개발노트북(실비행 휴대)·개발노트북의 wsl·rpi(headless=`opencv-python-headless`). 단계별 추가, **최종 단계엔 4환경 전부 검증**. 매트릭스는 메모리 `project_vision_dev_env.md`에
- `vision/requirements.txt`는 **ASCII 유지** — 개발컴 pip(cp949)가 한글 주석에 `UnicodeDecodeError`. 한글 안내는 `vision/CLAUDE.md`에
- `geo_project.pixel_to_gps`는 폐기 예정 → 신규 테스트 금지

### 다음 세션

1. **미커버 단위 테스트 채우기** — `color` HSV 초록/빨강 모드 우선(정밀착륙 직결) + edge/morphology/fusion 등. 대상·규칙은 `vision/CLAUDE.md` 단위테스트 표
2. **또는 관측성 골격 §7.9 다음 항목** — 이중싱크 로거 + provenance 헤더(config+git해시+캘리브id)
3. (선행 대기) 카메라 인트린식/왜곡 캘리브레이션 + 실기체 3타겟 데이터 — 골든셋·색 캘리브 착수 조건

### 주의

> 개발컴만 `.venv` 준비됨 — **개발노트북·그 wsl·rpi는 미설치**(필요 단계에서 `vision/requirements.txt`, rpi는 headless 변형).
> 대회 상세규정 여전히 대기(`vision_plan.md` §10: ArUco ID·③빨간십자·초록 스펙·성공판정·CC 인터페이스).

---

## 2026-07-11 — [main][mc-hw] 이륙실패 ulog 진단 + AMSL 이륙고도 수정

**브랜치:** `dev--vision-computing-module`
**목적:** MC 테스트기체 마지막 이륙 실패(사용자 제공 ulog) 원인 분석 → 수정

### 완료

- **마지막 MC 이륙 실패 근본원인 확정** — 2026-07-07 광주 실비행 ulog(`02_17_49`, `logs/2026-07-07_0217_last/`에 저장·notes.md 분석)를 pyulog 직접 파싱. ARM·CommandTOL(NAV_TAKEOFF param7=4.0, lat/lon=NaN) 모두 ACCEPTED됐으나 navigator `Already higher than takeoff altitude` → 모터 미가동(출력 0.002) → 10초 후 `Disarmed by auto preflight disarming`. 배터리(12.15V, 새그 없음)·GPS(3D 22위성)·SD·OFFBOARD 전부 정상이라 무관 — 이전 실패(07-03 전압새그, 07-07 SD)와 다른 새 원인
- **원인 = AMSL/relative 프레임 버그(07-06 열린 질문의 실측 종결)** — `CommandTOL.altitude`(→ NAV_TAKEOFF param7)는 AMSL 절대고도인데 `transition_alt`(4.0, 지면 상대)를 그대로 실어 지면 AMSL(19.2m)보다 낮은 목표가 됨. SITL은 `transition_alt:=50`>지면(≈0)이라 그동안 가려졌음
- **수정 커밋 `9451861`** — ① `takeoff_request_fields(transition_alt, home_amsl)` → `altitude=home_amsl+transition_alt`, `/mavros/home_position/home` 구독, home 미수신 시 이륙 보류 ② CLIMBING 게이트 `climbing_reached(…, ground_ref_up)` 지면기준 AGL 보정(로컬 원점≠지면 2.11m — 안 고치면 이륙해도 CLIMBING 무한대기) ③ pytest fc_ros 60/fc_bridge 44 pass(신규 7)
- **SITL 재검증 체크리스트 작성** — `sitl_verification_log.md` "작업 H-2". 재현엔 `PX4_HOME_ALT`로 지면 AMSL>transition_alt 세팅 필수 + geoid(geo.altitude가 AMSL인지) 확인

### 결정

- 이륙 목표고도는 항상 **AMSL 절대고도(home_amsl + transition_alt)** 로 전송 — transition_alt 직접 전달 금지
- CLIMBING 판정은 이륙 순간 캡처한 지면 높이 기준 AGL로

### 다음 세션

1. **SITL 재검증(작업 H-2)** — `sitl_verification_log.md` 체크리스트대로. `PX4_HOME_ALT=100` 등으로 버그 재현 조건을 만든 뒤 수정 확인, **geoid 정합 반드시 확인**
2. PASS 시 "transition_alt를 MIS_TAKEOFF_ALT 이하로" 임시조치 완전 제거
3. 실기체 검증은 ✈ vtol-실기체 결함 해소 후

### 주의

> 수정은 **단위테스트만 통과, SITL 재검증 전**이다 — 실비행 반영 금지. geoid 미확인 리스크(MAVROS `geo.altitude`가 ellipsoid면 과상승) 있음, SITL 로그로 판별
> ulog·분석 notes는 `logs/`(git 제외)에 있음 — GitHub엔 없다

---

## 2026-07-09 — [vision] 정밀착륙 계획 확정 + 트랙 분리

**브랜치:** `dev--vision-computing-module`
**목적:** 비전 객체인식 본격 개발 전 고려사항 컨설팅(블라인드스팟·unknown-unknowns 중심) → 계획 확정·문서화

### 완료

- **착수 전 컨설팅 완료** — 타겟 3종 확정(①버티포트 원형 3m+중앙 50cm ArUco ②초록 매트+흰 박스, 박스 옆 착륙 ③빨간 십자, 규정 대기), 물리 제약 정량화(GSD/고도·tilt 오차·GPS 한계), 검출 전략(고전 CV, 타겟별 coarse→fine 2단)
- **하드웨어 갈림길 해소** — 카메라 mono OV9285→컬러 필요 판명→**RPi Cam Module 3 Wide 표준(IR-cut)** 확정(롤링셔터 수용+완화·초점 무한대·수동노출), 짐벌 없음(나디르+고무댐핑)→**자세 de-rotation 필수**, 라이다 1D 40m급
- **변경내성/관측성 설계** — ports&adapters, 레이어드 config+현장 색 캘리브레이터, 구조적 로깅(터미널+파일 JSONL, 비차단), 기록/재생, 세 화면(전송/연산/연출)+격리 규칙, FPV 인코더 어댑터(Pi5 무-HW인코더→USB2 raw 대역폭 벽)
- **문서화** — `docs/vision_plan.md` 신규(계획 정본), 루트 `CLAUDE.md` 의존관계 `vision→fc_ros`(상대 pose)로 교체(`pixel_to_gps` 폐기), `vision/CLAUDE.md` 재설계 배너
- **vision 트랙 분리** — `docs/vision_status.md` 신규(vision 전용 진입점, FC와 컨텍스트 격리), `/session-log` 도메인 라우팅 + `[vision]` 태그 추가, 메모리 `project_vision_plan.md` 신규

### 결정

- **검출은 고전 CV(ML 없음)** — 타겟이 전부 피듀셜/고대비 색·형상, 결정론=신뢰도(대회 오인식 절대불가)
- **측위: GPS 접근 + 30cm는 비전 폐루프** — 일반 GPS 절대좌표 한계, `geo_project.pixel_to_gps` 폐기
- **통합: 독립 ROS2 노드 + offboard 정밀착륙 서브상태**, 출력=상대 pose(LANDING_TARGET 피벗 호환)
- **vision은 FC와 별도 트랙·진입점** — 콜드스타트 컨텍스트 격리 우선(상호 안 읽음), 서술 로그만 공용

### 다음 세션

1. **카메라 인트린식+왜곡 캘리브레이션** (102° 광각, 없으면 pose 거짓)
2. **관측성 골격 먼저** — `vision/main.py` headless-safe 수정 + 구조적 로깅/JSONL 스캐폴딩
3. 실기체 데이터 수집(고도별 3타겟) 착수

### 주의

> **대회 상세규정 미공개** — ArUco 딕셔너리/ID·③빨간십자 규정·초록 색·치수 스펙 대기(`vision_plan.md` §10). 하드웨어(카메라/Pi4 인코더/라이다)도 변경 가능 → 전부 어댑터로 흡수 설계.
> **이번 세션 변경 미커밋**(문서만, 코드 착수 전). `.claude/commands/session-log.md` 라우팅 편집은 git 무시라 로컬만 반영.

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
