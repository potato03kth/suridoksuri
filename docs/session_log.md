---
doc_type: session_log
project: suridoksuri-1
---

# 세션 로그

> 최신 세션이 위에 온다.  
> `/session-log` 커맨드로 세션 종료 전 자동 작성.

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
