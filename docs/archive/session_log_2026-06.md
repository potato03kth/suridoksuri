---
doc_type: session_log_archive
project: suridoksuri-1
period: 2026-06-18 ~ 2026-06-20 (파라미터 버그 수정 세션 포함)
---

# 세션 로그 아카이브 — 2026-06

> `docs/session_log.md`에서 이동된 과거 세션 기록 (최신이 위).
> 현행 로그는 최근 8개 세션만 유지하며, 초과분은 `/session-log` 실행 시 이 디렉터리로 이동된다.

---

## 2026-06-2e 파라미터 버그 수정

**브랜치:** `dev--vision-computing-module`
**목적:** SITL-2(phase2 launch 통합 기동) 수행 및 판정

### 완료

- **SITL-2 PASS** — `ros2 launch fc_ros phase2.launch.py` 정상 기동
  - TelemetryNode + OffboardNode 두 노드 기동 확인 (TypeError 없음)
  - 신규 파라미터 5개 값 실측: `transition_alt=50.0`, `d_end_thresh=10.0`, `v_terminal=15.2`, `decel_dist=80.0`, `landing_timeout=60.0`
- **`offboard_node.py` 파라미터 버그 수정** — YAML `waypoints` 미적용 문제
  - 원인: `main()`의 `_offboard_param_reader` 임시 노드가 YAML 파라미터를 받지 못해 `waypoints`, `planner`, `v_cruise` 등이 항상 하드코딩 기본값으로 동작
  - 수정: 경로 계획 파라미터 선언 및 계획 로직을 `OffboardNode.__init__`으로 이동, `main()` 임시 노드 제거
  - 결과: `waypoints`, `planner`, `v_cruise`, `a_max_g`, `gravity`가 이제 YAML 값으로 로드됨, 30/30 pytest PASS 유지
- **QGC `MIS_TAKEOFF_ALT = 50.0` 설정** — 기본값 10m라 CLIMBING이 50m까지 진행 불가했던 문제 해결

### 결정

- `fc_bridge`는 colcon 패키지가 아니라 순수 Python 라이브러리 → `pip install -e .`로 WSL에 설치, `colcon build --packages-select fc_ros`만 사용

### 다음 세션

1. **SITL-3** — 경로 추종 검증 (선행: B·C·D·SITL-2 전부 ✅)
   - dry-run 속도 프로파일 끝점 = v_terminal 확인
   - 3종 경로(직선/L자/사각형) SITL 추종 및 cross_track_error 확인

### 주의

> SITL-3 진입 전 WSL에서 `colcon build --packages-select fc_ros && source install/setup.bash` 재빌드 필수 (offboard_node.py 수정됨).
> `MIS_TAKEOFF_ALT = 50.0` QGC 설정 완료 — SITL 재시작 시 유지 여부 확인 권장.

---

## 2026-06-20 — 전체 테스트 재실행 + 31/31 PASS 확인

**브랜치:** `dev--vision-computing-module`
**목적:** 사용자가 수정한 테스트를 전부 재실행해 flight_plan.md 기준 합격 여부 확인

### 완료

- 사용자가 테스트 파일을 대폭 수정한 후 전체 재실행 요청
- flight_plan.md에 명시된 3개 테스트 파일 전부 실행 → **31/31 PASS**

| 테스트 파일                              | 케이스 수 | 결과       |
| ---------------------------------------- | --------- | ---------- |
| `fc_ros/test/test_params.py`             | 5         | 5/5 PASS   |
| `fc_bridge/tests/test_terminal_decel.py` | 5         | 5/5 PASS   |
| `fc_ros/test/test_offboard_node.py`      | 21        | 21/21 PASS |

- 커버 범위: 작업 A(파라미터 정비) · 작업 B(종단 감속) · 작업 C(이륙·천이) · 작업 D(역천이·착륙) 전부 통과

### 다음 세션

1. **작업 E** — 긴급 수동 override (`/fc_ros/override` Bool 토픽, vtol_state 분기, `test_offboard_node.py`에 `test_override_mc`/`test_override_fw` 추가)
2. 작업 E 완료 → 코드 단위 A~E 전부 완료 → **SITL-3 진입 조건 충족**
3. SITL-2(phase2 launch 통합 기동)는 사람이 언제든 WSL에서 수행 가능 (선행: 작업 A ✅)

---

## 2026-06-20 — 작업 D 완료 (상태머신 ② 역천이·착륙)

**브랜치:** `dev--vision-computing-module`
**목적:** flight_plan.md 작업 D(상태머신 ② 역천이·착륙) 자율 실행 + 테스트 품질 점검

### 완료

- **테스트 품질 구조 개선** — 기존 작업 C 테스트의 "로직 복사" 문제 수정
  - 문제: `_climb_reached`, `_vtol_is_fw` 등 판정 로직이 테스트 파일 안에 복사되어 있었음. `offboard_node.py`가 바뀌어도 테스트는 통과할 수 있는 구조 (코드와 단절된 스펙 테스트).
  - 원인: `offboard_node.py`가 최상단에 `rclpy` import → 테스트에서 직접 import 불가.
  - **해결**: `fc_bridge/execution/state_logic.py` 신규 생성 (rclpy 의존 없는 순수 판정 함수 5개). `offboard_node.py`와 테스트 양쪽이 동일 함수를 참조 → 코드 변경이 즉시 테스트 실패로 이어짐.
  - `test_offboard_node.py`의 기존 작업 C 테스트도 import 방식으로 전환.

- **작업 D** — `offboard_node.py` 상태머신 확장
  - `_State` enum에 `TRANSITION_MC`, `LANDING` 추가
  - `__init__`에 `_d_end_thresh`, `_landing_timeout` 파라미터 읽기 추가 (이미 선언됨, 읽기만 추가)
  - `_step_following()` 종료조건: `dist_to_end < 3.0` (하드코딩) → `trans_mc_trigger(dist_to_end, self._d_end_thresh)`. 전환 대상: `DONE` → `TRANSITION_MC`
  - `_step_transition_mc()`: `CommandLong(3000, param1=3.0)` 역천이 명령 → `vtol_is_mc(state.vtol_state)` 확인 → LANDING
  - `_step_landing()`: `set_mode "AUTO.LAND"` → `landing_done(state.armed)` 확인 → DONE. `landing_timeout` 초과 시 경고 1회.
  - 기존 `_step_climbing`, `_step_transition_fw`도 `state_logic` 함수 사용으로 통일
  - `test_offboard_node.py`에 케이스 8개 추가: `trans_mc_trigger`×3, `vtol_is_mc`×3, `landing_done`×2

### 결과

| 테스트 파일                              | PASS      |
| ---------------------------------------- | --------- |
| `fc_ros/test/test_offboard_node.py`      | 21/21     |
| `fc_bridge/tests/test_terminal_decel.py` | 5/5       |
| `fc_ros/test/test_params.py`             | 5/5       |
| **합계**                                 | **31/31** |

### 결정

- 판정 순수 함수는 `fc_bridge/execution/state_logic.py`에 집중 관리. 향후 추가 판정 로직도 이곳에.
- `_landing_timeout_warned` 플래그로 타임아웃 경고 1회만 출력 (스팸 방지).
- `_entry_done` (ENTRY 판정) 은 numpy 로직이 복잡하고 작업 D 범위 밖이므로 현행 유지.

### 다음 세션

1. **작업 E** — 긴급 수동 override (`/fc_ros/override` Bool 토픽, vtol_state 분기, 선행: C ✅ D ✅)
2. 작업 E 완료 후 → **SITL-3** 진입 가능 (선행: B·C·D·SITL-2 전부 충족)
3. SITL-2(launch 통합 기동)는 A 완료 후 진입 가능이었으나 아직 사람 미수행

### 주의

> 작업 E 완료 시 코드 단위 작업(A~E) 전부 완료 → SITL-3(경로 추종 검증) 진입 조건 충족.
> SITL-2(phase2 launch 기동)는 작업 A 선행이 완료된 상태이므로 언제든 사람이 WSL에서 수행 가능.

---

## 2026-06-20 — 작업 B·C 완료 (종단 감속 헬퍼 + 상태머신 이륙·천이)

**브랜치:** `dev--vision-computing-module`
**목적:** flight_plan.md 작업 B(종단 감속 헬퍼) + 작업 C(상태머신 ① 이륙·상승·천이) 자율 실행

### 완료

- **작업 B** — `fc_bridge/planning/terminal_decel.py` 신규 생성: `apply_terminal_decel(v_profile, s_arc, v_terminal, decel_dist)` 구현
  - `offboard_node.py` `main()` 배선: `run_planner` 직후 `v_terminal`/`decel_dist`를 tmp 노드에서 읽어 적용
  - `fc_bridge/tests/test_terminal_decel.py` 신규 작성 — 5/5 PASS
- **작업 C** — `offboard_node.py` 상태머신 확장
  - `_State` enum에 `ARM_TAKEOFF`, `CLIMBING`, `TRANSITION_FW` 추가, 초기 상태 → `ARM_TAKEOFF`
  - `CommandLong` 서비스 클라이언트 추가 (`/mavros/cmd/command`)
  - `_step_arm_takeoff()`: ARM 요청 → `state.armed` 확인 → `AUTO.TAKEOFF` 요청 → CLIMBING 전환
  - `_step_climbing()`: `pos_ned[2] >= transition_alt` → TRANSITION_FW 전환
  - `_step_transition_fw()`: `CommandLong(3000, param1=4.0)` → `vtol_state==FW` 확인 → STREAMING 전환
  - STREAMING 리팩터: ARM 제거, 속도 0 → 첫 WP 방향 전진속도 발행, `mode==OFFBOARD` 조건만으로 전환
  - `fc_ros/test/test_offboard_node.py`에 케이스 6개 추가 — 13/13 PASS (기존 7 + 신규 6)

### 결정

- `main()`의 `v_terminal`/`decel_dist`는 tmp 노드에서 읽어 처리 (OffboardNode 생성 전 경로 계획 필요 구조 유지)
- STREAMING에서 ARM 제거: ARM은 `ARM_TAKEOFF`에서 1회 완료되어 중복 불필요
- `VTOL_STATE_MC = 3`, `VTOL_STATE_FW = 4` 모듈 상수로 추출 (작업 D·E도 사용)

### 다음 세션

1. **작업 D** — 상태머신 ② 역천이·착륙 (`TRANSITION_MC`, `LANDING` 구현, 선행: C ✅)
2. **작업 E** — 긴급 수동 override (`/fc_ros/override` 토픽, 선행: C ✅)
3. 작업 D·E 완료 후 → SITL-2(launch 통합 기동, 사람 수행) → SITL-3

### 주의

> 작업 D에서 `dist_to_end < self._d_end_thresh` 조건 변경 시 `_step_following()`의 하드코딩 `3.0` 제거 필요.
> `_d_end_thresh`는 `__init__`에서 `declare_parameter("d_end_thresh", 10.0)`로 이미 선언됨 — 읽기만 하면 됨.

---

## 2026-06-19 — SITL-1 완료 + 확정값 반영

**브랜치:** `dev--vision-computing-module`
**목적:** SITL-1(VTOL 환경 전환 + 상수 확인) 수행, 결과 판정 및 문서 반영

### 완료

- SITL-1 절차 수행 (사람) + Claude 판정
  - vtol_state 상수 실측: MC=3, FW=4, 천이→FW=1, 천이→MC=2 (예상값과 동일)
  - QGC 수동이륙 시퀀스 확인 (HOLD → TAKEOFF → HOLD 흐름)
  - VTOL 천이 서비스 직접 호출 확인 (MC→FW param1=4, FW→MC param1=3)
  - AUTO.TAKEOFF 확인: ARM 선행 필수, 완료 후 HOLD 모드 전환
  - COM_RC_OVERRIDE = 3 설정 및 PX4 재시작 후 유지 확인
- `flight_plan.md` 오류 수정: param1 매핑이 반대로 기록되어 있었음 → 전면 수정
  - 기술 참조, 작업 C `_step_transition_fw()`, 작업 D `_step_transition_mc()` 모두 수정
- `flight_plan.md` 확정값 및 SITL-1 실측 근거 주석 추가
- `flight_plan.md` 실기체 첫 비행 전 지상 안전 테스트 프로토콜 추가
- `sitl_verification_log.md` SITL-1 결과 기록 및 진행 상태 갱신

### 결정

- **VTOL 천이 param1 확정**: MC→FW = `param1=4.0`, FW→MC = `param1=3.0` (목표 상태 값임)
- **AUTO.TAKEOFF 완료 후 모드 = HOLD** (작업 C 설계 입력값)
- **COM_RC_OVERRIDE → POSCTL 전환**: SITL에서 물리 RC 없이 재현 불가 → 실기체 지상 테스트로 이월 (비행 전 필수 체크리스트에 추가)
- **SITL-1: 조건부 PASS** — RC override → POSCTL 전환만 이월, 나머지 전 항목 PASS

### 다음 세션

1. **작업 B** — `apply_terminal_decel()` 헬퍼 구현 (`fc_bridge/planning/terminal_decel.py` 신규, pytest)
2. **작업 C** — 상태머신 ① 이륙·상승·천이 구현 (선행: A ✅, SITL-1 ✅ → 진입 가능)
3. 작업 B/C 완료 후 SITL-2 진행 (사람 수행)

### 주의

> 작업 C에서 사용할 vtol_state 상수와 param1 값이 이번 세션에서 확정됨. flight_plan.md 기술 참조 섹션 참고. 이전 세션 기록에 나오는 "param1=3 (MC→FW)" 는 오류였으며 수정 완료.

---

## 2026-06-19 — 작업 A 완료 (params/YAML 정비)

**브랜치:** `dev--vision-computing-module`
**목적:** flight_plan.md의 작업 A(params/YAML 정비) 자율 실행 및 pytest 합격 기준 통과

### 완료

- `fc_ros_params.yaml` waypoints 2D → flat 1D 변환 (offboard_node·mission_node 양쪽), 고도 150m → 50m 통일
- `offboard_node.py` 신규 파라미터 5개 declare_parameter 추가: `transition_alt`(50.0), `d_end_thresh`(10.0), `landing_timeout`(60.0), `v_terminal`(15.2), `decel_dist`(80.0)
- `fc_ros/test/test_params.py` 신규 작성 — 5개 테스트 케이스 전부 PASS
- `flight_plan.md` 미완료 목록에서 작업 A 완료 표시

### 결정

- 없음 (작업 A는 이전 세션에서 설계 확정됨, 이번 세션은 실행만)

### 다음 세션

1. **작업 B — 종단 감속 헬퍼 `apply_terminal_decel()` 구현** (`fc_bridge/planning/terminal_decel.py` 신규, OffboardNode 배선, `pytest fc_bridge/tests/test_terminal_decel.py`)
2. **SITL-2** — `ros2 launch fc_ros phase2.launch.py` 통합 기동 (작업 A 선행 완료됨, 사람 수행)
3. 작업 C(상태머신 이륙·상승·천이)는 작업 A + SITL-1(vtol_state 상수) 선행 후 진입

### 주의

> 작업 B와 작업 C는 선행 의존이 없어 병행 시작 가능. 단 작업 C는 SITL-1의 vtol_state 상수 실측값이 필요하므로 SITL-1 미완료 상태라면 상수를 플랜 문서 예상값(MC=3, FW=4)으로 임시 사용 후 확인 필요.

---

## 2026-06-19 — flight_plan 정합성 검토 & 작업단위 재구성

**브랜치:** `dev--vision-computing-module`
**목적:** "전체 비행 사이클 검증" 계획(flight_plan.md)의 정합성 검토 + 새 컨텍스트에서 "실행하라" 한마디로 진입 가능한 작업단위로 재분할

### 완료

- 실제 코드와 대조해 정합성 오류 5건 식별·수정 (flight_plan.md 전면 재작성)
  1. `v_terminal` 경로감속 미동작 확정 (eta3/diterpin이 `v_ref=v_cruise` 고정, v_terminal 미사용 — no-op)
  2. 튜닝 가이드 자기모순 (폐기된 `d_pre_trans`/`v_transition_max` 참조) 제거
  3. 기존/신규 상태 미구분 + STREAMING의 ARM 중복 + `_step_following` 하드코딩 3.0 → 명시
  4. YAML 2D 수정 위치 중복/모호 → 작업 A 단독 소유로 정리
  5. 운용 고도 50/150m 혼재 → 50m 일원화
- flight_plan.md를 **[코드] 작업 A~E**(Claude 자율, pytest) + **[SITL] SITL-1~5**(사람 수행, 체크리스트)로 재분할 (구 세션 B를 C/D/E로 분할)
- 후속 단계 **작업 F + SITL-6**(임의 WP 생성·추종 검증) 추가 — SITL-4 이후, 핵심 범위 밖
- `session_status.md` 새 구조로 동기화 (구 "세션 A~F" 명칭 전면 폐기)
- memory `project_sitl_state.md` + MEMORY.md 인덱스 갱신

### 결정

- **역천이 전 감속은 후처리 헬퍼로 구현하기로 함**: 플래너 수정 대신 `apply_terminal_decel()`(작업 B)로 v_profile 끝 구간을 v_terminal로 ramp-down
- **작업단위를 [코드](Claude 자율)와 [SITL](사람 수행)로 분리하기로 함**: "실행하라" 진입은 코드단위, SITL은 사람이 트리거하고 Claude가 체크리스트·로그판정 보조
- 임의 WP 경로 생성·추종 검증은 전체 사이클 검증(SITL-4) **이후**의 후속 작업으로 하기로 함

### 다음 세션

1. **작업 A — params/YAML 정비** (Claude 자율, WSL 불필요). 새 컨텍스트 트리거: `docs/flight_plan.md 의 "작업 A — params/YAML 정비"를 실행하라.`
2. (병행) **SITL-1** — VTOL 환경 전환 + vtol_state 상수 확인 (WSL, 사람). 작업 C가 상수 참조
3. 작업 A 후 → SITL-2(launch 기동) → main 병합 → `dev--fc-vtol-sitl` 분기

### 주의

> 작업 A는 현재 브랜치에서 수행(SITL-2·병합 선행). 작업 B~E는 main 병합 후 `dev--fc-vtol-sitl`에서.
> 작업 C는 vtol_state 상수가 필요하므로 SITL-1을 먼저 돌려두면 매끄럽다.

---

## 2026-06-18 — 시스템 갭 분석 & 비행 계획 최종 확인

**브랜치:** `dev--vision-computing-module`
**목적:** MissionNode 검증 완료 후 launch 파일 통합 검증 진입 전 전체 시스템 점검

### 완료

- launch 파일 통합 검증 작업 범위 분석 (`phase1.launch.py`, `phase2.launch.py`, `fc_ros_params.yaml`)
- `fc_ros_params.yaml` 2D 리스트 버그 재확인 (mission_node + offboard_node 양쪽)
- 시스템 전체 갭 7가지 식별 및 정리
- `docs/flight_plan.md` 검토 + memory `project_sitl_state.md` 업데이트 (세션 A~F 반영)

### 결정

- 이륙/천이/착륙 전부 fc_ros 자동 (사람 개입 없음) — 이 세션에서 명시적으로 확인
- Phase1(MissionNode)은 디버그/백업 전용, 실제 비행은 Phase2 단독 — 확인
- 착륙은 역천이 + AUTO.LAND 자동 — 확인
- **세션 A 이전 선행 작업 확정**: `fc_ros_params.yaml` 2D → flat 변환 + `ros2 launch fc_ros phase2.launch.py` 통합 launch 테스트 → main 병합 → `dev--fc-vtol-sitl` 분기

### 식별된 갭 (우선순위 순)

1. 이륙 시퀀스 없음 (ARM_TAKEOFF, CLIMBING 상태 미구현)
2. VTOL 천이 없음 (TRANSITION_FW/MC 상태 미구현, MAV_CMD_DO_VTOL_TRANSITION 미사용)
3. 착륙 없음 (LANDING 상태 미구현)
4. 경로 생성 SITL 검증 없음 (eta3 파이프라인 코드 존재, 검증 안 됨)
5. Phase1↔Phase2 연결 미정의
6. 고정익 OFFBOARD velocity 제어 동작 미검증

### 다음 세션

1. `fc_ros_params.yaml` — `waypoints` 2D → flat 변환
2. `ros2 launch fc_ros phase2.launch.py` 통합 launch 테스트
3. `dev--vision-computing-module` → `main` 병합
4. `dev--fc-vtol-sitl` 분기 후 세션 A 진입

### 주의

> `docs/flight_plan.md`가 이미 세션 B~F 상세 계획(v_terminal, override 구현 등)을 포함하고 있다.
> 다음 세션 진입 전 반드시 이 파일을 먼저 읽을 것 — CLAUDE.md에 링크됨.

---

## 2026-06-18 — 새 세션 진입용 문서 보완

**브랜치:** `dev--vision-computing-module`
**목적:** 새 세션에서 "이대로 실행하라"가 통하도록 누락 문서 파악 및 작성

### 완료

- `docs/flight_plan.md` + 관련 문서 전체 분석 → 새 세션 진입 시 막히는 지점 식별
- `docs/session_status.md` 신규 작성 (WSL 상태, 브랜치 전략, 코드 동기화, 세션별 선행조건 표)
- `fc_bridge/CLAUDE.md` 신규 작성 (run_planner 시그니처, Path/PathPoint 속성, dry-run 사용법)
- `docs/flight_plan.md` 패치 — 세션 B 앞에 YAML 2D 버그 선행 수정 경고 추가
- `CLAUDE.md` 패치 — 도메인 맵에 `fc_bridge/`, `fc_ros/` 추가, FC 작업 진입 문서 링크 추가
- `docs/session_status.md` 코드 동기화 절차 수정: rsync → `git pull + colcon build`

### 결정

- drone_ws 동기화: `git pull + colcon build + source install/setup.bash` (rsync 불필요)
- 브랜치 순서 확정: 현재 브랜치에서 통합 launch 테스트 → `main` 병합 → `dev--fc-vtol-sitl` 분기 → 세션 A

### 다음 세션

1. `fc_ros_params.yaml` — `waypoints` 2D → flat 변환 (세션 C 선행 작업)
2. `ros2 launch fc_ros phase2.launch.py` 통합 launch 테스트
3. `dev--vision-computing-module` → `main` 병합
4. `dev--fc-vtol-sitl` 분기 후 세션 A 진입

### 주의

> `session_status.md`의 rsync 명령이 이번 세션에서 `git pull`로 수정됨. 혹시 이전 버전을 참조한 곳이 있다면 확인 필요.

---

## 2026-06-18 — flight_plan.md 설계 검토 및 안전장치 추가

**브랜치:** `dev--vision-computing-module`
**목적:** `docs/flight_plan.md` 전면 검토 후 누락된 설계 결정 보완

### 완료

- flight_plan.md / sitl_verification_log.md 분석 (RTK 필요성, 속도 제어 방식, 루프 크기 등 6개 항목)
- **치명적 안전 오류 수정**: `v_transition_max = 5 m/s` → FW 스톨 추락 위험, 삭제
- `v_terminal = 13.8 × 1.1 = 15.2 m/s` 전 문서 반영
- 역천이 감속 방식 결정 및 문서 반영 (경로 생성 수준 / OffboardNode 거리 조건만)
- MVP 긴급 수동 전환 설계 및 문서 반영 (두 레이어: PX4 COM_RC_OVERRIDE + `/fc_ros/override` 토픽)
- 파라미터 튜닝 가이드 섹션 신규 추가
- 안전 및 긴급 수동 전환 섹션 신규 추가

### 결정

- **역천이 전 감속**: 제어기 수준 클램프 폐기 → 경로 생성 시 `v_terminal` 적용 (eta3 planner `vehicle_params`에 추가). OffboardNode는 `dist_to_end < d_end_thresh` 거리 조건만 사용
- **RTK 불필요**: WP 위치오차 평가가 GPS 상대값 기준 → 동일 GPS 편향 상쇄됨
- **긴급 수동 전환 두 레이어**: Layer 1 = PX4 `COM_RC_OVERRIDE=3` (조이스틱 항시 유효), Layer 2 = ROS2 `/fc_ros/override` (키보드 명령)
- **MC → POSCTL / FW → MANUAL** 전환 행동 확정
- 세션 A~F 계획 비대하지 않음 — 단방향 사이클 검증 범위 그대로 유지

### 다음 세션

1. Session A: `gz_standard_vtol`로 SITL 전환, `COM_RC_OVERRIDE=3` QGC 설정
2. `vtol_state` 상수값 실 SITL 확인 (문서 예상값 검증)
3. `MAV_CMD_DO_VTOL_TRANSITION` 서비스 호출 직접 테스트
4. RC 오버라이드 → POSCTL 모드 전환 동작 확인

### 주의

> `v_terminal = 15.2 m/s` 는 `v_cruise = 15.0 m/s` 보다 겨우 0.2 m/s 높다. 실기체 파라미터(`v_cruise ≥ 17 m/s` 예상) 확정 후 감속 효과 재검증 필요.
> eta3 planner가 `v_terminal` 파라미터를 실제로 지원하는지 Session D에서 확인 필요.

---

## 2026-06-18 — 비행 시퀀스 확정 & 세션 로그 체계 구축

**브랜치:** `dev--vision-computing-module`
**목적:** MissionNode SITL 검증 완료 후 전체 비행 계획 문서화, 이후 세션 연속성 체계 수립

### 완료

- MissionNode SITL 검증 (버그 3개 수정: waypoints 2D→flat, 기타)
- `docs/flight_plan.md` 신규 작성 — 세션 A~F 전체 비행 시퀀스 및 작업 계획 확정
- `docs/session_status.md` 신규 작성 — WSL 환경 상태, 브랜치 전략, SITL 기동 명령
- `fc_bridge/CLAUDE.md` 신규 작성
- `/session-log` 커맨드 + `docs/session_log.md` 롤링 로그 체계 구축

### 결정

- 이륙/천이/착륙 전부 fc_ros 자동 (ARM → AUTO.TAKEOFF → VTOL_TRANSITION → OFFBOARD → AUTO.LAND)
- Phase1은 디버그/백업 전용, 실제 비행은 Phase2 단독
- 역천이 전 감속: **경로 생성 수준** (v_terminal = 스톨 × 1.1 = 15.2 m/s). OffboardNode는 거리 조건만으로 트리거
- 세션 A~F 순서로 VTOL SITL 검증 진행 후 실기체 배포

### 수정 파일

- `fc_ros/fc_ros/nodes/mission_node.py` — 버그 3개 수정
- `docs/flight_plan.md` — 신규 (세션 A~F 전체 계획)
- `docs/session_status.md` — 신규 (WSL 상태 및 브랜치 전략)
- `fc_bridge/CLAUDE.md` — 신규
- `CLAUDE.md` — 업데이트
- `docs/sitl_verification_log.md` — 업데이트

### 다음 세션

1. `fc_ros_params.yaml` — `waypoints` 2D → flat 변환 (미수정, 런치 시 TypeError 발생)
2. `ros2 launch fc_ros phase2.launch.py` 통합 launch 테스트 (TelemetryNode + OffboardNode)
3. `dev--vision-computing-module` → `main` 병합
4. `dev--fc-vtol-sitl` 브랜치 생성 후 세션 A 진입

### 주의

> `fc_ros_params.yaml`의 `waypoints`가 현재 2D 리스트 상태 → `ros2 launch fc_ros phase2.launch.py` 실행 시 TypeError. 런치 전 반드시 flat 변환 먼저.
