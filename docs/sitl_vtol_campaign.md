---
doc_type: campaign_brief
project: suridoksuri-1
track: 🛩 sitl-vtol
scope: VTOL 오프보드 전 구간 SITL 전면 회귀 캠페인 (SITL-7) — 계획·시나리오·판정기준
created: 2026-07-27
---

# SITL-7 — VTOL 오프보드 전면 회귀 캠페인

## 0. 왜 지금 하는가

`sitl-vtol` 트랙의 마지막 VTOL 전체 사이클 검증은 **SITL-4(2026-06-30)** 이다.
그 이후 `offboard_node.py` / `state_logic.py` 에 다음 변경이 쌓였고 **어느 것도 VTOL로
회귀검증되지 않았다** (전부 MC(`gz_x500`)로만 검증됨):

| 커밋/시점 | 변경 | VTOL 경로 영향 |
|---|---|---|
| 2026-07-06 `000f478` | `takeoff_request_fields` lat/lon NaN | ARM_TAKEOFF (VTOL 재검증됨) |
| 2026-07-21 | `_publish_pos_setpoint(pos, yaw)` yaw 필수 인자화 + `yaw_ned_to_quat_enu` | **TRANSITION_FW p3 / TRANSITION_MC / HOLD / FOLLOWING(FW) 전부** |
| 2026-07-24 `568fbe5` | home_amsl 소스를 `/mavros/altitude` 로 교체 | ARM_TAKEOFF·CLIMBING |
| 2026-07-24 | `home_amsl_sample_fresh` + `climbing_reached(vz_down)` | CLIMBING → TRANSITION_FW 전이 타이밍 |
| 2026-07-24 `0785777` | eta3 `np.trapz` → `_trapz` | **VTOL 기본 플래너(eta3) NR 경로 전체** |
| 2026-07-25 `3f389b6` | `waypoint_frame` 신설(기본 `takeoff`) + `_State.PILOT_TAKEOVER` | **경로 기준계 전체 / 모든 상태의 조종사 인계** |
| 2026-07-25 `cd5fda9` | 컨벤션·토픽 의미론 감사 수정 6건 | 전반 |
| 2026-07-27 `4e8e378` | MC WP 정착 (`mc_wp_advance` 3튜플) | MC 전용 (VTOL FOLLOWING은 미변경 — 회귀만 확인) |

즉 **VTOL 천이 경로는 3주치 변경을 안고 한 번도 안 돌아본 상태**다.
사용자 요구는 세 가지 질문에 답하는 것:

1. **좋은 시퀀스를 가졌는가** — 상태머신 설계 자체가 타당한가
2. **각 단계의 흐름은 부드러운가** — setpoint 점프·급가감속·헤딩 요동·고도 손실이 없는가
3. **예상대로 작동하는가** — 회귀·경계조건·장애주입에서 설계대로 동작하는가

---

## 1. 환경

| 항목 | 값 |
|---|---|
| 실행 호스트 | 이 노트북 WSL 배포판 `Ubuntu-22.04` (E드라이브, root 사용자) |
| PX4 | `/root/PX4-Autopilot`, `v1.18.0-beta1-155-g9bb0d365c4` |
| 기체 모델 | `gz_standard_vtol` (airframe `4004_gz_standard_vtol` 빌드 존재 확인, gz 모델 `~/.simulation-gazebo/models/standard_vtol` 존재) |
| MAVROS | `udp://:14540@localhost:14580` (이 배포판은 14580 — 개발컴 문서의 14557 아님) |
| 워크스페이스 | `/root/drone_ws` (`src/suridoksuri` = 이 저장소 클론, 브랜치 `dev--vision-computing-module`) |
| MC 대조군 | `gz_x500` (필요 시) |

### 반드시 지킬 함정 (기존 실측)

- **PX4 콘솔(`pxh>`)을 파일로 리다이렉트 금지** — 비-TTY 재출력 루프로 20초에 195MB, 방치 시 GB급 폭주.
  상태확인은 `ros2 topic/service`·`ss -uln`으로.
- **SITL 프리플라이트 우회**(벤치 arm 전용, 실기체 금지): `CBRK_SUPPLY_CHK=894281`, `NAV_DLL_ACT=0`.
- **`wsl.exe -d Ubuntu-22.04 -- bash -lc '...'` 는 복잡한 셸 구문이 깨진다**(`$(seq)`+`for`+`if` 조합에서
  `syntax error near unexpected token`). → **모든 로직은 저장소에 스크립트 파일로 두고 한 줄로 호출**한다.
- **프로세스 정리**: `pkill -f` 로 `gz sim`·`make px4_sitl` 이 남으면 다음 런이 이전 gz 서버에 얹혀 중복
  인스턴스가 된다. 씨름하지 말고 `wsl.exe --terminate Ubuntu-22.04` 로 배포판을 통째 재기동하는 게 확실.
- `fc_bridge` 는 `pip install -e .` 금지 — `.pth` 방식(이미 구성됨).
- `mavros.guided_target: PositionTargetGlobal failed because no origin` 경고는 **알려진 코스메틱**(작업 H 참조).

---

## 2. 산출물 구조

```
tools/sitl/
  run_scenario.py      # 시나리오 1건 실행: SITL기동→MAVROS→파라미터→launch→감시→수집→정리
  analyze_run.py       # ulog+노드로그 → 지표 JSON/MD
  scenarios.yaml       # 시나리오 정의(아래 3장)
  README.md
logs/2026-07-27_sitl_vtol_campaign/
  <scenario_id>/
    node.log           # offboard_node/telemetry_node stdout
    *.ulg              # PX4 ulog
    metrics.json       # analyze_run.py 산출
    verdict.md         # PASS/FAIL + 근거
  campaign_report.md   # 전체 종합 (Phase 4)
```

---

## 3. 시나리오 목록

`WP` 좌표는 `waypoint_frame` 기본값 `takeoff`(이륙지점 상대 NED, z는 h_up) 기준.
**주의:** 코드상 `_cruise_alt = waypoints[-1].z` 스칼라 하나 — 중간 WP의 z는 setpoint에 반영되지 않는다
(이 사실 자체를 A4에서 검증·기록한다).

### Phase 1 — 기준선 회귀 (SITL-4 재현)

| ID | 목적 | launch 인자 |
|---|---|---|
| **A1** | SITL-4 직선 300m 재현 (3주치 변경 후 회귀 여부) | `transition_alt:=50.0 waypoints:="[0,0,50, 300,0,50]" waypoint_frame:=local` |
| **A2** | 동일 경로, **`waypoint_frame:=takeoff`(현 기본값)** — VTOL 첫 검증 | `transition_alt:=50.0 waypoints:="[0,0,50, 300,0,50]"` |
| **A3** | SITL-4 L자 경로 재현 | `waypoints:="[0,0,50, 200,0,50, 200,200,50]"` |
| **A4** | 중간 WP 고도가 다른 경로 → `_cruise_alt` 스칼라화 실증 | `waypoints:="[0,0,50, 150,0,80, 300,0,50]"` |

### Phase 2 — 경로 다양성

| ID | 목적 | 경로 |
|---|---|---|
| **B1** | 장거리 직선 (500m) — 순항 안정성 | `[0,0,50, 500,0,50]` |
| **B2** | 완만 곡선 4WP (30°급 꺾임) — eta3 NR 경로(`_trapz` 수정) 실행 검증 | `[0,0,50, 150,0,50, 300,80,50, 450,200,50]` |
| **B3** | 직각 코너 (90°) — 코너 오버슈트 정량화 | `[0,0,50, 250,0,50, 250,250,50]` |
| **B4** | 예각/U턴 (135°) — 선회반경 초과 시 거동 | `[0,0,50, 250,0,50, 100,150,50]` |
| **B5** | 사각 폐곡선 (시점≈종점) — 종점 근접 오판 여부 | `[0,0,50, 200,0,50, 200,200,50, 0,200,50, 0,20,50]` |
| **B6** | 2-WP 최소 경로 (플래너 N≤2 특수케이스, NR 우회) | `[0,0,50, 200,0,50]` |
| **B7** | 단거리 경로 (`d_end_thresh`=10m 대비 짧음) — FOLLOWING 즉시완료 오판 | `[0,0,50, 40,0,50]` |
| **B8** | 후방 경로 (초기 헤딩과 180° 반대) — 헤딩 정렬 P제어 최악조건 | `[0,0,50, -300,0,50]` |

### Phase 3 — 천이 집중 + 장애주입

| ID | 목적 | 방법 |
|---|---|---|
| **C1** | 천이고도 민감도 — 저(20m) / 고(120m) | `transition_alt:=20.0` / `:=120.0`, 경로는 A1 |
| **C2** | 헤딩 정렬 90° 조건 | `waypoints:="[0,0,50, 0,300,50]"` (동쪽 경로) |
| **C3** | **천이 중 OFFBOARD 강제 이탈** — 외부에서 `AUTO.LOITER` 주입 후 재요청 복구 확인 | A1 실행 중 `vtol_state`가 1(천이중)일 때 `ros2 service call /mavros/set_mode` |
| **C4** | **바람 주입** — `gz topic` 또는 `PX4_SIM_SPEED`/wind 플러그인으로 8m/s | A1 + 바람 |
| **C5** | 역천이 오버슈트 정량화 (`d_end_thresh` 10/30/60 스윕) | A1 × 3회, `d_end_thresh` 는 launch 인자 신설 필요 |
| **C6** | 긴급 OVERRIDE (FW 순항 중 / MC HOLD 중 각 1회) | `/fc_ros/override` 에 `true` 발행 |
| **C7** | **조종사 인계(PILOT_TAKEOVER)** — FOLLOWING 중 POSCTL 강제 주입 → 노드가 손을 떼는가 | 2026-07-25 사고 재발방지 검증 |
| **C8** | home_amsl/geoid 회귀 (작업 H-2 체크리스트를 VTOL로) | `PX4_HOME_LAT/LON/ALT` 통제조건 |
| **C9** | STREAMING 오버슈트 재현 여부 (미해결 이슈, VTOL에서도 나오는가) | A1 로그의 CLIMBING→STREAMING 구간 정밀분석 |
| **C10** | `entry_mode:=mid_flight` (ENTRY 상태) — 거의 미검증 경로 | launch 인자 신설 필요 |

---

## 4. 판정 지표 (analyze_run.py 가 산출)

### 공통 (모든 시나리오)
- **상태 전이 타임라인**: `ARM_TAKEOFF→CLIMBING→TRANSITION_FW→STREAMING→FOLLOWING→TRANSITION_MC→HOLD→LANDING→DONE`
  순서·각 상태 체류시간·경고/타임아웃 발생 여부
- **완주 여부**: disarmed 도달, 총 소요시간
- **`vtol_state` 시퀀스**: 3→1→4 (정천이), 4→2→3 (역천이) 및 각 소요시간

### 부드러움(질문 2) — 정량 지표
| 지표 | 산출 | 합격 기준(초안) |
|---|---|---|
| setpoint 점프 | 연속 두 틱 `setpoint_position` 거리 / dt | 상태 전이 경계에서 급점프(> `v_approach`×dt×3) 없음 |
| 수직 가속 | `vehicle_local_position.az` 피크 | 천이 구간 제외 \|az\| ≤ 0.5g |
| 수평 감속 | 역천이 구간 `d(|v|)/dt` | ≤ 0.3g (2.94 m/s², SITL-4 실측 ~1.5) |
| 헤딩 요동 | TRANSITION_FW 정렬구간 yaw 오차 시계열 | 오버슈트 없이 단조수렴, 정렬완료 err ≤ `wp0_htol`(0.2rad) |
| 고도 유지 | 천이·순항 구간 고도 편차 | 정천이 중 고도 손실 ≤ 5m, 순항 ±3m |
| cte | FW FOLLOWING cross-track error | 직선 ≤ 2m, 코너 오버슈트는 별도 기록 |
| CLIMBING 오버슈트 | `transition_alt` 대비 최대 AGL | ≤ +10% (C9 핵심 지표) |

### 시퀀스 설계(질문 1) — 정성 항목
각 시나리오에서 아래를 관찰·기록:
- 상태 전이 조건이 **관측가능량**으로만 판정되는가 (타이머 의존/열린 대기 없는가)
- 각 상태에 **타임아웃/폴백**이 있는가 (없는 상태: TRANSITION_FW·TRANSITION_MC·CLIMBING ← 검증 표적)
- 실패 시 **안전측 폴백**으로 떨어지는가 (OFFBOARD 이탈·조종사 인계·override)

---

## 5. 세션 분할 (오케스트레이션)

| 세션 | 범위 | 게이트 (오케스트레이터가 직접 재현 확인) |
|---|---|---|
| **S1** | 환경 브링업 + `tools/sitl/` 하니스 제작 + launch 인자 확장 + **A1 1건 완주** | 오케스트레이터가 하니스를 직접 재실행해 산출물(ulog·metrics.json) 재현 |
| **S2** | Phase 1 (A1~A4) | 각 verdict의 근거 수치를 로그에서 직접 교차확인 |
| **S3** | Phase 2 (B1~B8) | 〃 |
| **S4** | Phase 3 (C1~C10) | 〃 |
| **S5** | 종합 분석 + `campaign_report.md` + `sitl_verification_log.md` SITL-7 기록 | 보고서 수치 ↔ metrics.json 대조 |

**동시 실행 금지:** SITL은 gz 서버·UDP 14540/14580 을 점유하므로 **시나리오 실행 세션은 항상 1개만**.
분석 세션은 실행 세션과 병행 가능.

**세션 규율 (전 세션 공통):**
- 커밋 태그 `[sitl]`, 푸시 대상 `dev--vision-computing-module` (PR 아님 — 프로젝트 관례)
- push 전 `git fetch` + 충돌 확인 (다른 도메인 세션이 같은 브랜치에 병행 커밋할 수 있음)
- **코드는 고치지 않는다** — 결함을 발견하면 `campaign_report.md` 에 기록만. 수정 여부는 오케스트레이터가 판단.
  (예외: 하니스 자체 / launch 인자 확장 / 명백한 테스트 인프라 버그)
- **"측정 창이 짧아 기능 없다고 오판"** 금지 — null 결과는 기능부재를 결론내기 전에 측정방법부터 의심할 것.
- **경고를 "not a blocker"로 스스로 판단해 통과시키지 말 것** — 경고는 전부 보고, 무해성 판단은 오케스트레이터가 한다.
