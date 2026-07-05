# suridoksuri-1 — Claude 작업 가이드

VTOL 자율비행 대회용 통합 소프트웨어 저장소다.  
도메인별로 독립 패키지로 분리되어 있으며, 현재 **시뮬레이터**, **FC(비행 제어)**, **객체인식** 세 도메인이 구현 중이다.

---

## 도메인 지도

| 디렉터리 | 도메인 | 상태 |
|---|---|---|
| `vtol_sim_checkpoint1_1/vtol_sim/` | 비행 시뮬레이터 (역학, 경로 계획, 제어) | 구현됨 |
| `fc_bridge/` | FC 라이브러리 (경로 계획, L1 유도, VehicleState) | 구현됨 |
| `fc_ros/` | FC ROS2 노드 (TelemetryNode, OffboardNode, MissionNode) | SITL 검증 완료, 실기체 배포 중 |
| `vision/` | 객체인식 (착륙지점 탐지) | 구현됨 |
| CC 도메인 | 명령 제어 | 미구현 |

---

## 도메인 간 의존 관계

현재는 **도메인 간 교차 import가 없다.** 각 도메인은 독립 실행된다.

향후 연동 예정:
- `vision` → FC: `vision/utils/geo_project.py`의 `pixel_to_gps()`로 GPS 좌표를 FC에 전달

새 도메인 간 의존을 추가하기 전에 반드시 이 파일에 의존 관계를 먼저 기록할 것.

---

## 각 도메인의 CLAUDE.md

작업 전에 해당 도메인의 CLAUDE.md를 먼저 읽는다.

- 시뮬레이터 경로 계획: `vtol_sim_checkpoint1_1/vtol_sim/path_planning/CLAUDE.md`
- FC 라이브러리: `fc_bridge/CLAUDE.md`
- 객체인식: `vision/CLAUDE.md`

FC 작업 세션 절차 (정형):

1. **진입:** `docs/session_status.md`의 트랙 보드에서 재개할 트랙 블록 하나만 읽는다 (사용자가 "○○ 트랙 재개"라 지정, 없으면 ▶ 활성 트랙). 그 블록의 참조 문서만, 필요 섹션만 추가로 읽는다. 다른 트랙 블록·`flight_plan.md` 전체 정독 금지 — 완료 작업 상세는 `docs/archive/`에 있다.
2. **자가 복구:** `session_status.md`의 last_updated 이후 커밋이 있거나, 트랙 보드로 설명되지 않는 미커밋 변경이 있으면 — 직전 세션이 기록 없이 끝난 것이다. `git log`/`git diff`로 무슨 일이 있었는지 파악해 해당 트랙 블록을 먼저 갱신한 뒤 작업을 시작한다.
3. **트랙 전환 규칙:** ① 다른 트랙 작업을 시작하기 전에 현재 변경을 커밋한다 (WIP 허용, 메시지에 `[main]`/`[mc-hw]`/`[sitl]`/`[vtol-hw]` 태그). ② 테스트용 임시 파라미터는 yaml을 고치지 않는다 — `phase2.launch.py v_cruise:=18.0 waypoints:="[...]"` launch 인자로만 준다.
4. **종료:** 세션 종료 전 `/session-log` — 이번 세션이 건드린 트랙 블록 갱신 + 로그 기록 + 로그 아카이브가 자동 수행된다. 급하면 `/session-log 축약`.

---

## 공통 규칙

- 도메인 디렉터리 밖으로 import 하기 전에 의존 관계를 이 파일에 기록한다.
- 각 도메인은 자체 `tests/` 폴더를 가진다. 작업 후 해당 도메인 테스트를 실행한다.
- `results/` 디렉터리의 출력물은 git에 포함시키지 않는다.
