# suridoksuri-1 — Claude 작업 가이드

VTOL 자율비행 대회용 통합 소프트웨어 저장소다.  
도메인별로 독립 패키지로 분리되어 있으며, 현재 **시뮬레이터**와 **객체인식** 두 도메인이 구현되어 있다.

---

## 도메인 지도

| 디렉터리 | 도메인 | 상태 |
|---|---|---|
| `vtol_sim_checkpoint1_1/vtol_sim/` | 비행 시뮬레이터 (역학, 경로 계획, 제어) | 구현됨 |
| `vision/` | 객체인식 (착륙지점 탐지) | 구현됨 |
| FC 도메인 | 비행 제어기 | 미구현 |
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
- 객체인식: `vision/CLAUDE.md`

---

## 공통 규칙

- 도메인 디렉터리 밖으로 import 하기 전에 의존 관계를 이 파일에 기록한다.
- 각 도메인은 자체 `tests/` 폴더를 가진다. 작업 후 해당 도메인 테스트를 실행한다.
- `results/` 디렉터리의 출력물은 git에 포함시키지 않는다.
