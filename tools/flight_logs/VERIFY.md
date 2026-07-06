# 작업 G 검증 계획 — 실행 단위 V1~V5

작업 G 산출물(2026-07-06 구현)의 버그·문제 확인용. 각 유닛은 독립 실행 가능하며,
새 세션에서 아래 **진입 프롬프트 한 문장**으로 시작한다.

기존 커버리지: `test_flight_logs.py` 13개(순수 함수만). **아래 유닛들은 그 밖의 미검증 영역이다.**

---

## 공통 규칙 — 거짓 테스트 금지

모든 유닛에 적용. 위반하면 그 유닛은 불합격이다.

1. **산출물 수정 금지.** 테스트를 통과시키기 위해 대상 코드(record_flight.sh / pull_ulog.py / fetch_logs.ps1 / topics.txt)를 고치지 않는다. 버그를 발견하면 → 테스트는 실패 상태로 두고, 버그 내용·수정안을 보고한 뒤 사용자 승인 후 수정하고 재실행한다.
2. **변이 검증 의무 (자동화 유닛 V1·V3·V4).** 테스트 전부 통과 후, 대상 코드에 의도적 결함 1개를 임시 주입해 테스트가 **실패하는 것을 확인**하고 원복한다 (주입 위치는 각 유닛에 명시). 실패하지 않으면 그 테스트는 판별력이 없는 거짓 테스트다.
3. **단언은 실제 부산물에만.** 바이트 동일성(checksum), 파일 존재·내용, stub이 기록한 인자, 종료코드. 성공 메시지 문자열 매칭만으로 합격 처리 금지.
4. **stub은 기록하는 실행체.** 무조건 성공을 반환하는 no-op stub 금지 — 받은 인자·신호를 파일에 기록해야 하고, 단언은 그 기록에 대해 한다.
5. **결과 보고는 PASS/FAIL 목록 + 실패 시 원인 분석.** 애매하면 FAIL로 보고한다.

---

## V1 — pull_ulog.py 다운로드 재조립 로직 단위테스트 [자동화 · 어디서든]

> 진입 프롬프트: `main-code 트랙 재개 — tools/flight_logs/VERIFY.md V1을 실행하라`

- **대상:** `download_log()` `request_log_list()` — 이번 작업에서 가장 복잡하고 전혀 실행되지 않은 코드.
- **환경:** Windows/WSL 무관. pymavlink 불필요 — 가짜 mav 객체를 주입한다.
- **방법:** `test_pull_ulog_link.py` (신규)에 시뮬레이션 링크 클래스를 만든다. `recv_match`가 시나리오대로 LOG_DATA/LOG_ENTRY를 돌려주고, `mav.log_request_*_send` 호출을 기록한다. 원본 바이트열(예: 1000·1024·**900(=90×10, 짧은 마지막 청크 없음)**·89바이트)을 90바이트 청크로 쪼개 공급하고, 다운로드 결과 파일이 원본과 **바이트 동일**한지 단언한다.
- **필수 시나리오:**
  - 정상 순차 수신 → 바이트 동일
  - 청크 1개 유실(offset 건너뜀) → 현재 offset부터 재요청하는지 (기록된 `log_request_data_send` 인자로 단언) + 최종 바이트 동일
  - 중복/과거 offset 청크 도착 → 무시하고 파일 오염 없음
  - 타 log_id 청크 혼입 → 무시
  - 크기가 90의 정확한 배수 → 마지막 청크 count==90이어도 정상 종료 (무한 대기/조기 break 없음)
  - `recv_match`가 계속 None(스톨) → 5회 재요청 후 TimeoutError
  - `request_log_list`: num_logs==0 → `[]` / 목록 일부 유실 → 경고 후 수신분 반환
- **변이 검증:** `download_log`의 `if msg.ofs != ofs:` 분기를 임시로 무력화(`if False:`) → 유실 시나리오 테스트가 실패해야 함. 확인 후 원복.
- **합격:** 신규 테스트 전부 통과 + 기존 `pytest test_flight_logs.py` 여전히 13 통과 + 변이 검증 성립.

## V2 — pull_ulog.py 실링크 검증 (SITL) [사람 협업 · WSL]

> 진입 프롬프트: `main-code 트랙 재개 — tools/flight_logs/VERIFY.md V2를 실행하라 (WSL SITL 준비됨)`

- **대상:** 실제 PX4를 상대로 한 로그 목록·다운로드 — V1이 못 잡는 실프로토콜 불일치(메시지 필드, PX4 응답 습성)를 잡는다.
- **환경:** WSL, `make px4_sitl gz_standard_vtol` (MAVROS는 **띄우지 않는다** — 링크 단독 점유 확인 겸).
- **방법:** SITL의 rootfs에 .ulg가 최소 1개 있어야 한다(없으면 QGC/pxh로 arm→disarm 1회). 이후:
  1. `python3 pull_ulog.py --url udp:127.0.0.1:14550 --list` → 목록이 rootfs `log/` 실제 파일 수·크기와 부합하는지
  2. `--out /tmp/v2` 다운로드 → **rootfs의 해당 원본 .ulg와 `sha256sum` 비교** (이것이 유일한 합격 판정 — 출력 메시지 아님)
  3. `ulog_info`로 다운로드본이 정상 ulog로 열리는지
- **부수 확인:** git pull 후 WSL에서 실행하므로 줄바꿈(CRLF 오염)·실행권한·pymavlink 의존이 실환경에서 겸사 검증된다.
- **합격:** sha256 일치 + ulog_info 정상.
- **속도 판정 (이 유닛 안에서 완결할 것):** 실측 속도로 `대표 비행 ulog 예상 크기 ÷ 실측 속도` = 다운로드 소요시간을 계산한다. 대표 크기는 실비행 로그가 없으면 15 MB(MC 기본 SDLOG_PROFILE, 10분 비행 가정)로 잡는다.
  - **소요시간 ≤ 5분(통상 재배터리 간격):** 판정 "충분" — 현행 launch 종료 후 다운로드 유지, 틈새 전송(작업 G-2) 기각. 결과를 flight_plan.md 미결 항목에 기록.
  - **소요시간 > 5분:** 판정 "부족" — 같은 세션에서 **작업 G-2(gcs_url 브리지 + disarmed 게이팅 틈새 다운로드, 2026-07-08 대화 참조)를 flight_plan.md에 등록**하고 트랙 보드 "다음"에 반영한다.
  - 단, USB 직결 실측은 RPi 배포 검증에서 한 번 더 갱신한다 (SITL UDP 속도는 참고치 — 최종 판정은 RPi 실측이 우선).

## V3 — record_flight.sh 동작 하니스 (ros2 stub) [자동화 · WSL 권장]

> 진입 프롬프트: `main-code 트랙 재개 — tools/flight_logs/VERIFY.md V3을 실행하라`

- **대상:** 래퍼의 제어 흐름 전체 — ROS2 없이 검증한다. `bash -n`은 문법만 봤고 실행 흐름은 미검증.
- **환경:** WSL(권장) 또는 Git Bash. ROS2 불필요.
- **방법:** 임시 디렉터리에 stub `ros2` 스크립트를 만들어 PATH 앞에 둔다. stub은 (a) 받은 전체 인자를 `args_<pid>.txt`에 기록, (b) `bag record` 모드면 SIGINT 받을 때까지 대기 후 "got INT" 기록, (c) `launch` 모드면 stdout 몇 줄 출력 후 대기. 하니스가 스크립트를 백그라운드 실행 → 3초 후 **스크립트 프로세스 그룹에 SIGINT** (실사용 Ctrl-C 재현) → 종료 후 단언.
- **필수 단언 (stub 기록·실파일에 대해):**
  - 연속 2회 실행 → `_flight01`, `_flight02` 폴더 (넘버링이 셸 경유로도 동작)
  - rosbag stub 인자에 `-o <플라이트폴더>/rosbag` + topics.txt의 **11개 토픽 전부** 포함
  - launch stub 인자에 `vehicle_type:=mc v_cruise:=18.0`이 그대로, `--sitl`/`--no-ulog`는 **미포함**
  - SIGINT 후 스크립트가 죽지 않고 수집 단계 완료 (종료코드 0 + "수집 완료" 이후 로직 실행 흔적 = ulog 복사/생략 동작)
  - rosbag stub이 INT를 받고 종료됨 ("got INT" 기록 존재)
  - `launch.log`에 launch stub의 stdout이 있음 / `notes.md`에 launch 인자가 박혀 있음
  - `--sitl` + 가짜 `PX4_SITL_LOG_DIR`(mtime 다른 .ulg 2개) → **최신 것만** 플라이트 폴더에 복사됨
- **변이 검증:** `trap ':' INT` 줄을 임시 제거 → SIGINT 생존 단언이 실패해야 함. 확인 후 원복.
- **합격:** 위 단언 전부 + 변이 검증 성립. (테스트 하니스는 `test_record_flight.sh`로 저장해 재사용 가능하게.)

## V4 — fetch_logs.ps1 증분 복사 로직 (ssh/scp stub) [자동화 · Windows]

> 진입 프롬프트: `main-code 트랙 재개 — tools/flight_logs/VERIFY.md V4를 실행하라`

- **대상:** 증분 필터·패턴 필터·실패 종료코드. 파서 검증만 했고 실행은 미검증.
- **환경:** Windows PowerShell 5.1. SSH 서버 불필요 — 임시 폴더에 stub `ssh.cmd`/`scp.cmd`를 만들어 `$env:Path` 앞에 붙인다. stub은 받은 인자를 파일에 기록하고, ssh.cmd는 준비된 목록을 stdout으로 출력한다.
- **필수 단언:**
  - 원격 목록 = 플라이트 폴더 3개 + 잡파일(`README.md`, `foo`) / 로컬에 그중 1개 존재 → scp가 **정확히 나머지 2개 폴더에 대해서만** 호출됨 (stub 기록으로 단언), 잡파일은 후보에도 없음
  - 신규 없음 → scp 호출 0회, 종료코드 0
  - ssh.cmd가 exit 1 → 스크립트 종료코드 1, scp 미호출
  - scp.cmd 1건 exit 1 → 스크립트 종료코드 1 (재실행 시 실패분 재시도 가능 상태)
  - `-Remote` 미지정 + `$env:RPI_REMOTE` 설정 → ssh stub이 그 값을 받음
- **변이 검증:** `$localDirs -notcontains $_` 를 `-contains`로 임시 변경 → 증분 단언 실패 확인 후 원복.
- **합격:** 위 단언 전부 + 변이 검증 성립.

## V5 — WSL SITL dry-run 통합 (계획서 합격 기준 원문) [사람 협업 · WSL]

> 진입 프롬프트: `main-code 트랙 재개 — tools/flight_logs/VERIFY.md V5를 실행하라 (WSL SITL 준비됨)`

- **대상:** flight_plan.md 작업 G [코드] 합격 기준의 "WSL SITL dry-run 1회" — 스텁 없는 실물 통합.
- **환경:** WSL. T1 SITL + (T2 MAVROS) 기동 후 `./record_flight.sh --sitl` 로 T3를 대체 실행. **V2와 같은 SITL 세션에서 이어서 하면 준비 비용이 준다** (V5 비행이 만든 ulog를 V2가 받아도 됨 — 단, pull_ulog는 MAVROS 종료 후).
- **합격 (실파일 존재·내용으로 판정):**
  - `logs/<오늘>_flightNN/` 생성, `rosbag/`에 실데이터(`ros2 bag info` 메시지 수 > 0), `launch.log`에 노드 출력, `notes.md` 템플릿
  - Ctrl-C 종료 후 SITL rootfs의 최신 .ulg가 폴더에 복사됨 (arm 안 했으면 "없음(정상)" 경고 경로 확인)
- **완료 시:** flight_plan.md 작업 G 테스트 항목의 dry-run 체크를 갱신하고 `/session-log`.

---

## 유닛 ↔ 리스크 매핑 (왜 이 다섯인가)

| 미검증 리스크 | 유닛 |
|---|---|
| 다운로드 재조립(유실·중복·경계) — 최고 복잡도, 실행 0회 | V1 |
| 실프로토콜 불일치·바이트 무결성·실측 속도 | V2 |
| 래퍼 제어 흐름(Ctrl-C 생존·rosbag 정지·인자 통과·최신 ulog 선택) | V3 |
| 증분 복사·실패 종료코드 (파싱만 검증됨) | V4 |
| 계획서 자체 합격 기준(dry-run) + 실환경 이식성 | V5, V2 |

V1·V3·V4는 Claude 단독 완결. V2·V5는 SITL 기동이 필요해 사람 협업(한 WSL 세션에 묶어 실행 권장).
