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
| `vision/` | 객체인식 (착륙지점 탐지) | 임시 틀 구현됨 — 정밀착륙 재설계 계획: `docs/vision_plan.md` |
| CC 도메인 | 명령 제어 | 미구현 |

---

## 도메인 간 의존 관계

현재는 **도메인 간 교차 import가 없다.** 각 도메인은 독립 실행된다.

향후 연동 예정:
- `vision` → `fc_ros`: 비전이 인식한 착륙지점의 **기체 기준 상대 pose**(`TargetEstimate`)를
  `offboard_node`의 정밀착륙 서브상태에 전달해 폐루프 유도한다. `LANDING_TARGET`(MAVLink) 네이티브
  precision-land 피벗과 호환되도록 상대 pose 형식으로 출력한다. 설계 상세는 `docs/vision_plan.md` §8.
  (기존 `geo_project.pixel_to_gps` GPS 절대좌표 방식은 GPS 정확도 한계(~1~2m)로 30cm 요구 미달 →
  **2026-07-28 삭제 완료**, 상대 pose 폐루프로 대체.)

**실제 이음매(2026-07-28 구현 완료, 빌드 의존 0):** 두 도메인은 **localhost TCP 소켓 + JSON
Lines**로만 만난다 — 코드 수준 교차 import도, `colcon`/`pip` 빌드 의존도 **여전히 없다.**

```
[호스트 picam-venv Py3.12, ROS 없음]           [fc 컨테이너 ros:humble Py3.10]
 vision/main.py --target-sink                   vision/ros/shim_node.py
   └ utils/target_sink.py (TCP 서버) ──────────▶  (TCP 클라이언트, 재접속 담당)
       127.0.0.1:8091  JSON Lines                    │  ▲
                                                     │  └ /mavros/local_position/pose 구독
                                                     ├─▶ /vision/landing_setpoint (PoseStamped, ENU)
                                                     ├─▶ /vision/target_pose   (PoseWithCovarianceStamped)
                                                     ├─▶ /vision/target_status (DiagnosticArray)
                                                     └─▶ /mavros/landing_target/raw  [기본 꺼짐]
                                                              │
                                                    fc_ros OffboardNode 정밀착륙 서브상태 [🔴 미구현 = F2]
```

> 🎯 **FC 세션이 정밀착륙(F2)을 착수한다면 → `docs/fc_precision_land_handoff.md` 하나만 읽으면 된다.**
> 계약(토픽·프레임·QoS·거부권)·붙는 자리·함정·검증·잠정값이 전부 거기 있고, vision 도메인 문서를
> 읽지 않아도 완주하도록 썼다.

**좌표 변환은 shim이 한다(2026-07-28 사용자 결정).** vision은 body FLU 상대 pose까지만 내고,
shim이 컨테이너 안에서 기체 자세·위치를 곱해 **절대 목표점**(`/vision/landing_setpoint`, ENU)을
만든다. 핵심 계약은 **"절대 setpoint를 만들되 절대 좌표를 기억하지 않는다"** — 매 레코드마다
그 순간의 최신 pose로 다시 계산해야 EKF 드리프트가 현재위치와 목표점에 똑같이 실려 상쇄된다.
vision이 소켓 앞에서 변환하는 원안은 **attitude 지연**(4.4Hz + 소켓 왕복 → 10m AGL에서 44cm)과
**`LandingTarget` 네이티브 피벗 경로 폐쇄** 때문에 기각됐다.

- shim은 **`vision/` 안에 있다**(`fc_ros/`가 아니다) — `docs/vision_fc_interface.md` §9 F1의
  배치 권고와 갈리는 지점이고, 그 대가로 `docs/rpi_deploy.md`의 `colcon build` 절차가 무변경이다
  (근거·트레이드오프는 `vision/CLAUDE.md` "컨테이너 ROS2 shim 노드" 절).
- shim은 vision 코드를 import하지 않는 stdlib 소비 규약을 쓴다(컨테이너에 cv2가 없다).
- FC 쪽 소비자(`OffboardNode` 정밀착륙 서브상태, §9 F2)는 **아직 없다** — 지금은 토픽만 나온다.

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
3. **트랙 전환 규칙:** ① 다른 트랙 작업을 시작하기 전에 현재 변경을 커밋한다 (WIP 허용, 메시지에 `[main]`/`[mc-hw]`/`[sitl]`/`[vtol-hw]`/`[vision]` 태그). ② 테스트용 임시 파라미터는 yaml을 고치지 않는다 — `phase2.launch.py v_cruise:=18.0 waypoints:="[...]"` launch 인자로만 준다.
4. **종료:** 세션 종료 전 `/session-log` — 이번 세션이 건드린 트랙 블록 갱신 + 로그 기록 + 로그 아카이브가 자동 수행된다. 급하면 `/session-log 축약`.

**vision 세션 진입 (FC와 분리):** vision 작업은 `docs/vision_status.md`(라이브 트랙)로 진입하고 설계는 `docs/vision_plan.md`의 필요 섹션만 연다. FC 트랙 보드(`docs/session_status.md`)와 **상호 읽지 않아 컨텍스트가 격리**된다. 커밋 태그 `[vision]`, 서술 로그는 공용 `docs/session_log.md`.

---

## 공통 규칙

- **FC 코드를 고쳤으면 즉시 실기체까지 반영한다 — 묻지 말고 바로.**
  `git push` → RPi `git pull` → 컨테이너 안에서 `colcon build --packages-select fc_ros` → 검증.
  **유일한 예외: 사용자가 "현재 비행중"이라고 말한 경우.** 그 외엔 "SITL 검증 먼저 할까요?",
  "배포할까요?" 같은 확인을 받지 말고 그냥 끝까지 한다. 절차·함정·검증 명령은
  `docs/rpi_deploy.md`. 커밋만 하고 멈추면 기체는 옛날 코드로 난다 —
  실제로 stale colcon build가 실비행 8건의 근본원인이었다(`4dc30f9`).
- 도메인 디렉터리 밖으로 import 하기 전에 의존 관계를 이 파일에 기록한다.
- 각 도메인은 자체 `tests/` 폴더를 가진다. 작업 후 해당 도메인 테스트를 실행한다.
- **로컬 전용 파일이 협업·재현을 여러 번 망친 이력이 있다 → 앵간하면 다 커밋·push한다.** `results/`
  플롯 등 재현 산출물도 포함해 올린다. 유일한 예외는 **대용량 raw 바이너리**(실측 캡처 등)이며
  그것만 `.gitignore`로 뺀다(현재: `vision/data/calibration_raw/`). "일단 로컬에만 두자"는 금지.
