---
doc_type: flight_plan
project: suridoksuri-1
scope: fc_ros 전체 비행 사이클 검증 — 활성 작업단위 계획 및 기술 참조
status: SITL-1~4 PASS · SITL-5(실기체 배포) 진행 중
last_updated: 2026-07-05
---

# fc_ros 전체 비행 사이클 검증 — 작업 계획

> **완료된 작업단위(작업 A~E, SITL-1~4)의 상세 계획·체크리스트는 [archive/flight_plan_completed.md](archive/flight_plan_completed.md)로 이동했다.**
> 이 문서는 **활성 작업**(SITL-5, 작업 F, SITL-6)과 계속 참조되는 **기술 자료**(설계 결정·상수·튜닝·안전)만 유지한다.
> 세션 진입은 `docs/session_status.md`에서 시작한다 — 이 문서는 전체를 정독하지 말고 필요한 섹션만 읽는다.

> **범위:** 본 계획은 **단방향 비행 사이클** (이륙→천이→경로 추종→역천이→착륙) 검증에 집중한다.
> 대회 전체 미션 (왕복 + 복수 착륙-이륙 사이클)은 본 계획 완료 후 별도 계획으로 진행한다.

---

## 진행 현황

| 작업단위 | 결과 | 상세 기록 |
|---|---|---|
| 작업 A~E [코드] | ✅ 전부 완료 (2026-06-19~20) | [archive/flight_plan_completed.md](archive/flight_plan_completed.md) |
| SITL-1 — VTOL 환경 + 상수 | ✅ 조건부 PASS (06-19) — RC override→POSCTL만 실기체 이월 | 〃 |
| SITL-2 — launch 통합 기동 | ✅ PASS (06-20) | 〃 |
| SITL-3 — 경로 추종 | ✅ PASS (06-30) — 핵심: FW는 위치 setpoint 필수 | `sitl3_fix_plan.md` · `sitl3_tuning_notes.md` |
| SITL-4 — 전체 사이클 통합 | ✅ PASS (06-30) — override는 AUTO.LOITER 폴백 | `sitl_verification_log.md` |
| **SITL-5 — 실기체 배포** | **진행 중** (07-03~, RPi5 MC 브링업. 07-06 첫 offboard 부분 성공) | [아래](#sitl-5--실기체-배포-진행-중) |
| **작업 H — AUTO.TAKEOFF 목표고도 명시 전달** | ✅ **완료** (07-06, `pytest` + SITL PASS) — 실기체 검증은 🚁 트랙 다음 비행 | [아래](#작업-h--autotakeoff-목표고도-명시-전달-commandtol) |
| **작업 G — 로그 자동수집 체계** | **계획 확정** (07-06), 미착수 | [아래](#작업-g--비행-로그-자동수집분석-체계-인프라) |
| 작업 F — 임의 WP 견고성 | 미착수 (후속) | [아래](#작업-f--임의-wp-경로-생성-견고성-하니스) |
| SITL-6 — 임의 WP SITL | 미착수 (후속) | [아래](#sitl-6--임의-wp-생성추종-sitl) |

---

## 작업단위 실행 규약

이 계획의 작업단위는 두 종류다. 새 컨텍스트에서 **"실행하라"** 한마디로 진입한다.

| 유형       | 실행 주체                | "실행하라"의 의미                                                                     | 합격 판정(테스트)       |
| ---------- | ------------------------ | ------------------------------------------------------------------------------------- | ----------------------- |
| **[코드]** | Claude 자율              | 코드 수정 → `pytest` 실행까지 Claude가 완료 (Windows, SITL 불필요)                    | pytest 통과             |
| **[SITL]** | 사람 (WSL) + Claude 보조 | Claude가 절차·체크리스트 준비 → 사람이 WSL에서 수행 → 로그를 붙여넣으면 Claude가 판정 | 체크리스트 전 항목 충족 |

규칙:

- **[코드] 단위는 SITL 없이 완결**된다. 순수 로직을 함수로 추출해 `rclpy` 없이 pytest로 검증한다 (`fc_bridge/execution/state_logic.py` 패턴 — 테스트와 노드가 동일 함수를 import).
- **[SITL] 게이트는 사람이 손으로 수행**한다. Claude는 기동 명령·관찰 포인트·합격 기준을 제시하고, 결과 로그로 PASS/FAIL을 판정한 뒤 `sitl_verification_log.md`에 기록한다.
- 각 단위는 **선행 조건**을 명시한다. 선행이 끝나지 않았으면 진입하지 않는다.

---

## 확정된 설계 결정

| 항목           | 결정                                                                                                                                                                                                |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 이륙/천이      | fc_ros 전부 자동 (ARM + AUTO.TAKEOFF + VTOL_TRANSITION)                                                                                                                                             |
| Phase1 역할    | 디버그/백업용 (실제 비행은 Phase2 단독)                                                                                                                                                             |
| 착륙           | fc_ros 자동 (역천이 → HOLD로 WP1 복귀·정착 → AUTO.LAND)                                                                                                                                             |
| 경로 생성기    | eta3 (기본), diterpin (대안)                                                                                                                                                                        |
| **FW 경로 추종** | **OFFBOARD + 위치 setpoint** (lookahead 70 m). PX4 FW 오프보드는 velocity setpoint를 무시하므로 STREAMING/FOLLOWING/천이/역천이 전 구간 위치 setpoint (SITL-3 확정). AUTO.MISSION 미채택 — 향후 vision 동적목표 대비 |
| MC 헤딩 정렬   | velocity만으로 yaw 불변 — `twist.angular.z` yaw rate P제어 병행 필수 (2026-06-24 확정)                                                                                                              |
| 역천이 전 감속 | **경로 생성 후처리 수준** — `apply_terminal_decel()`(작업 B)가 v_profile 마지막 `decel_dist` 구간을 v_terminal(≥스톨×1.1)로 ramp-down. OffboardNode는 거리 조건(`d_end_thresh`)만으로 역천이 트리거. 플래너는 `v_ref=v_cruise` 고정이라 `v_terminal`을 직접 읽지 않는다 |
| 긴급 수동 전환 | RC 모드스위치(COM_RC_OVERRIDE) + ROS2 `/fc_ros/override` 토픽. `_State.OVERRIDE`: setpoint 중단 → manual 시도(MC→POSCTL, FW→MANUAL) → 1초 내 미진입 시 **AUTO.LOITER 안전 폴백** (SITL-4 정정)        |
| **vehicle_type** | 런타임 파라미터 `"vtol"`(기본) \| `"mc"` — MC는 FW 천이 2단계 생략(CLIMBING→STREAMING, FOLLOWING→HOLD 직행). MC 추종도 위치 setpoint 재사용, `v_terminal`/`decel_dist`는 MC에서 무의미 (2026-07-03) |

---

## 목표 비행 시퀀스

```
[ros2 launch fc_ros phase2.launch.py]        (vehicle_type:=mc 로 MC 전용 기체 전환 가능)
         │
         ├─ TelemetryNode  ← MAVROS 구독 상시 실행
         │
         └─ OffboardNode 상태머신
                │
                ▼ ARM + AUTO.TAKEOFF 명령
          [ARM_TAKEOFF]
                │
                ▼ pos_ned[2] >= transition_alt 확인
          [CLIMBING]
                │
                ▼ 헤딩 정렬(yaw rate P제어) → MAV_CMD_DO_VTOL_TRANSITION(param1=4: MC→FW)
          [TRANSITION_FW]                     ← vehicle_type=mc 는 생략(CLIMBING→STREAMING 직행)
                │
                ▼ vtol_state == FW 확인 후 OFFBOARD 모드 전환
          [STREAMING]  ← 위치 setpoint 발행 (PX4 watchdog)
                │
                ▼ (entry_mode == "mid_flight" 일 때만)
          [ENTRY]
                │
                ▼ lookahead 70 m 위치 setpoint 추종, eta3 경로
          [FOLLOWING]
                │
                ▼ dist_to_end < d_end_thresh (v_profile은 v_terminal까지 감속됨)
          [TRANSITION_MC]  ← MAV_CMD_DO_VTOL_TRANSITION(param1=3: FW→MC), keepalive 유지
                │                             ← vehicle_type=mc 는 생략(FOLLOWING→HOLD 직행)
                ▼ vtol_state == MC 확인
          [HOLD]  ← 역천이 오버슈트 보정: MC로 WP1 복귀·정착 (SITL-3에서 추가)
                │
                ▼ wp1_land_ready 확인
          [LANDING]  ← AUTO.LAND 명령
                │
                ▼ disarmed 확인
          [DONE]

          [OVERRIDE]  ← /fc_ros/override 수신 시 어느 상태에서든 진입 (setpoint 중단 → manual → LOITER 폴백)
```

---

# 활성 작업단위

---

## SITL-5 — 실기체 배포 (진행 중)

**유형:** [배포] (사람 + Claude 보조)
**목적:** 실기체에 fc_ros 배포·검증.
**선행:** SITL-4 ✅
**현 단계:** **RPi5(Ubuntu 24.04) + Pixhawk 6C 순수 MC 테스트기체** 브링업 (`vehicle_type:=mc`, SITL-5 변형). VTOL 실기체는 그 후속.
**환경·기동 절차:** `docs/session_status.md` "환경 참조" (Docker Humble 컨테이너 등) · `docs/pixhawk6c_rpi4_integration_guide.md`

**완료 (2026-07-03):** Docker `ros:humble` 배포 환경, MAVROS·fc_ros 빌드, 6C ArduCopter→PX4 플래시 + 수동비행 검증.

**남은 작업:**

1. **MAVROS 링크 안정화** — RTT 2~5 s·heartbeat 플래핑·935 params 정체. 태블릿 QGC 끊고 **USB 직결**부터.
2. **AUTO.TAKEOFF 미실행 진단** — (a) MAVROS 서비스 미준비 vs (b) GPS 락 없어 PX4가 거부 → `statustext`로 판별. (AUTO.TAKEOFF는 GPS 락 필수 — 실내/벤치 불가.)
3. **MC 전체 사이클 실비행** — 이륙 → 위치 setpoint 추종 → HOLD → AUTO.LAND.
4. **VTOL 실기체 반복** — FW 천이 포함 전체 사이클 + RC override→POSCTL 실측 (SITL-1 이월 항목).

**비행 전 필수:** [첫 비행 전 지상 안전 테스트](#첫-비행-전-지상-안전-테스트-sitl-5-비행-직전) + [필수 조정 파라미터 체크리스트](#실기체-배포-시-필수-조정-파라미터-체크리스트-sitl-5) 전 항목 통과. **통과 전 이륙 금지.**

---

## 작업 H — AUTO.TAKEOFF 목표고도 명시 전달 (CommandTOL)

**유형:** [코드] (Claude 자율) — **완료 (2026-07-06), SITL PASS**
**목적:** 07-03/07-06 실기체 비행(`b9fc748d-...ulg` 등)에서 확인된 근본 원인 수정.
**선행:** 없음. 실기체 검증은 🚁 mc-실기체 트랙 다음 비행에서.

### 근본 원인 (ulog `b9fc748d-...` pyulog 직접 파싱으로 확정, 2026-07-06)

`_step_arm_takeoff`가 `SetMode(custom_mode="AUTO.TAKEOFF")`만 보내고 **목표고도를 전혀 전달하지 않는다.** PX4는 그래서 자기 자신의 파라미터 `MIS_TAKEOFF_ALT`(기본 2.5m)까지만 오르고 자체적으로 AUTO_LOITER로 전환한다. `CLIMBING` 상태는 `pos_ned[2] >= transition_alt`를 기다리는데, 실제 도달 가능한 고도(2.5m)가 `transition_alt`보다 낮으면 이 조건이 **영원히 만족되지 않아 OFFBOARD 요청 자체가 나가지 않는다.**

실측 증거: 해당 비행 `vehicle_command` 로그 전체에 OFFBOARD 모드 요청(DO_SET_MODE, custom_main_mode=6)이 단 한 건도 없고, `vehicle_control_mode.flag_control_offboard_enabled`가 비행 내내 0. 실측 고도는 home(~17.2m MSL) + `MIS_TAKEOFF_ALT`(2.5m) ≈ 19.7m — waypoint의 20m와는 무관한 우연의 일치였다.

### 작업 목록

1. `fc_ros/fc_ros/nodes/offboard_node.py` `_step_arm_takeoff`: `SetMode("AUTO.TAKEOFF")` 대신 `mavros_msgs/srv/CommandTOL`(`/mavros/cmd/takeoff`)로 교체. `altitude=self._transition_alt`를 실어 보내 **이륙 목표고도와 CLIMBING이 기다리는 고도를 동일 값으로 강제 일치**시킨다 — `MIS_TAKEOFF_ALT`(PX4 저장 파라미터, 코드와 별개로 수동 동기화 필요했던 값)에 대한 의존을 구조적으로 제거.
2. 서비스 클라이언트 추가: `self._takeoff_cli = self.create_client(CommandTOL, "/mavros/cmd/takeoff")` (import `CommandTOL` from `mavros_msgs.srv`).
3. 요청 필드: `min_pitch=0.0`, `yaw=float('nan')`(현재 헤딩 유지), `latitude=float('nan')`, `longitude=float('nan')`(현재 위치 사용 — MAVLink 관례. **주의: `0.0/0.0`은 실제 좌표로 해석되어 실패함, 아래 SITL 참조**), `altitude=self._transition_alt`.
4. 기존 논블로킹 패턴 유지 — `service_is_ready()`만 검사, `wait_for_service()` 금지 (SITL-3 Bug1 준-데드락 재발 방지).
5. `fc_ros/test/test_offboard_node.py`: `_step_arm_takeoff`가 `CommandTOL` 요청을 보내고 `req.altitude == transition_alt`인지 mock client로 검증. 기존 `SetMode` 기반 ARM_TAKEOFF 테스트를 대체.
6. 회귀: `pytest fc_ros` 전체 통과.

### 실제 구현 메모 (2026-07-06)

- Windows 개발컴엔 `rclpy`가 없어(계획상 mock client 검증 불가) 요청 필드 조립 로직을 순수함수 `fc_bridge/execution/state_logic.py::takeoff_request_fields(transition_alt)`로 추출 — `offboard_node._step_arm_takeoff`와 `fc_ros/test/test_offboard_node.py`가 동일 함수를 참조(기존 판정함수와 동일 패턴).
- `offboard_node.py`: `self._takeoff_cli = self.create_client(CommandTOL, "/mavros/cmd/takeoff")` 추가, `_step_arm_takeoff`의 이륙 단계가 `SetMode("AUTO.TAKEOFF")` 대신 `CommandTOL.Request()`에 `takeoff_request_fields()` 필드를 실어 발행.
- `pytest`: vtol_sim 6 · fc_bridge 44(+4 신규) · fc_ros 82(+4 신규) = 151 전부 통과.
- **SITL 1차 검증에서 2차 버그 발견·수정:** `latitude=0.0, longitude=0.0`을 "현재 위치 사용"으로 가정했으나 틀림 — MAVLink `MAV_CMD_NAV_TAKEOFF` 관례상 "현재 위치 사용"은 **NaN**이고 `0.0/0.0`은 실제 좌표(null island)로 해석됨. QGC상 AUTO.TAKEOFF 모드 전환은 확인됐으나 고도 미상승 → PX4 preflight 안전 disarm으로 실패 재현. `takeoff_request_fields()`의 lat/lon을 `NaN`으로 수정(커밋 `000f478`) 후 재검증 PASS. 상세: `sitl_verification_log.md` "작업 H".

### 검증 (별도 트랙)

- **SITL** (🛩): ✅ **PASS (2026-07-06)** — gz_standard_vtol, `transition_alt:=50.0`, 실제 climb altitude가 `MIS_TAKEOFF_ALT`와 무관하게 상승·CLIMBING 통과 확인. 잔존 `guided_target` "no origin" 경고는 MAVROS humble의 알려진 QoS 코스메틱 이슈로 무해(상세: `sitl_verification_log.md`).
- **실기체** (🚁): 다음 비행에서 검증 대기. 이 수정이 들어갔으니 "`transition_alt`를 `MIS_TAKEOFF_ALT` 이하로 낮춰라"는 임시조치(session_status.md에 기록된)가 이론상 불필요해지나, **실기체 PASS 확인 전까지는 임시조치를 유지**할 것.

**주의:** `CommandTOL.altitude`가 AMSL/relative 중 어느 쪽으로 PX4에 해석되는지는 SITL에서 명확히 재확인되지 않음 — 실기체 검증 시 함께 확인.

**합격 기준:** `pytest fc_ros` 통과 + SITL PASS. ✅ 충족 (2026-07-06).

---

## 작업 G — 비행 로그 자동수집·분석 체계 (인프라)

**유형:** [코드] (Claude 자율 — 스크립트·규약·문서) + [배포] (사람 — RPi 1회 검증)
**목적:** 비행 1회 = 폴더 1개 규약으로 PX4 ulog·rosbag·터미널 로그를 자동 수집하고, 개발컴으로 회수해 즉시 분석 가능하게 한다. 전 테스트 트랙(🚁 mc-실기체 / 🛩 sitl-vtol / ✈ vtol-실기체) 공용 인프라.
**선행:** 없음 (SITL-5와 병행 가능)
**계획 확정:** 2026-07-06. 진입 트리거: `main-code 트랙 재개 — flight_plan.md 작업 G를 실행하라`

### 배경 (설계 입력)

| 로그 소스 | 실체 | 수집 방식 |
|---|---|---|
| PX4 ulog (6C 실기체) | SD카드 `/fs/microsd/log/`에 arming~disarm 자동 기록. 정보량 최대(자세·모드전이·setpoint 수락·failsafe 사유) | 비행 종료 후 **MAVLink FTP**로 다운로드 (pymavlink). 실패 시 SD 수동 회수 폴백 |
| PX4 ulog (SITL) | `~/PX4-Autopilot/build/px4_sitl_default/rootfs/log/` 자동 생성 | 최신 파일 복사 |
| 파이썬 노드 터미널 | `ros2 launch` stdout — 현재 세션 종료 시 소실 | `tee launch.log` |
| MAVROS / ROS2 | `~/.ros/log/` + `/rosout` · `/mavros/statustext/recv`(PX4 거부 사유) | rosbag2 녹화에 포함 |

**핵심 제약:** ulog 다운로드는 반드시 **launch(MAVROS) 종료 후** — 시리얼 포트를 MAVROS가 단독 점유하고, 비행 중 같은 링크로 대용량 전송 시 제어 링크 오염(현재도 링크 여유 없음).

**업로드 방침 (2026-07-06 결정):** GitHub 업로드 안 함 — 대용량 바이너리로 git 이력 팽창 + LFS 쿼터 + `results/` 금지 규칙과 충돌. 로그의 목적지는 분석하는 곳 = **개발컴** (RPi → 개발컴 직접 fetch). 공유·웹 분석이 필요한 ulog만 선택적으로 PX4 Flight Review(logs.px4.io)에 업로드.

### 산출물 및 작업 목록

1. **`tools/flight_logs/record_flight.sh`** — RPi/WSL 겸용 비행 래퍼
   - `logs/YYYY-MM-DD_flightNN/` 자동 생성 (NN = 당일 순번 자동 증가)
   - rosbag2 record 백그라운드 시작 (토픽 목록: `topics.txt`)
   - `ros2 launch fc_ros phase2.launch.py "$@" 2>&1 | tee launch.log` — launch 인자 그대로 통과 (`vehicle_type:=`/`v_cruise:=`/`waypoints:=` 오버라이드 호환)
   - launch 종료(Ctrl-C) 시: rosbag 정지 → `--sitl`이면 SITL log 디렉터리에서 최신 ulog 복사, 실기체면 `pull_ulog.py` 실행
   - `notes.md` 템플릿 생성 (비행 조건 / 관찰 / 결론 3줄 양식)
2. **`tools/flight_logs/pull_ulog.py`** — pymavlink MAVLink FTP로 최신 `.ulg` 다운로드
   - 인자: `--url`(기본: RPi USB serial), `--out`, `--list`(로그 목록만)
   - 최신 로그 선택·플라이트 폴더 넘버링 로직은 **순수 함수로 분리** (pytest 대상, 규약대로)
   - 실패 시 명확한 폴백 안내 출력: "SD 카드 수동 회수 → 폴더에 넣어라"
3. **`tools/flight_logs/topics.txt`** — rosbag 녹화 토픽:
   `/mavros/state` `/mavros/extended_state` `/mavros/local_position/pose` `/mavros/local_position/velocity_local` `/mavros/setpoint_position/local` `/mavros/setpoint_velocity/cmd_vel` `/mavros/statustext/recv` `/mavros/global_position/global` `/mavros/imu/data` `/fc_ros/override` `/rosout`
4. **`tools/flight_logs/fetch_logs.ps1`** — 개발컴(Windows)에서 scp/rsync over SSH로 RPi `logs/` → 로컬 `logs/` (신규 폴더만 증분 복사)
5. **`.gitignore`에 `logs/` 추가** (`results/` 규칙과 동일 — 로그는 git 밖, 분석 결론만 docs에)
6. **`tools/flight_logs/README.md`** — 사용법 + 분석 절차: pyulog(`ulog_info`/`ulog2csv`), Flight Review 업로드 방법, rosbag 대조(`ros2 bag info/play`), "분석은 개발컴에서 Claude에게 로그 폴더 경로를 주면 됨"

### 테스트 — 합격 기준 [코드]

- `tools/flight_logs/test_flight_logs.py`: 폴더 넘버링·최신 ulog 선택 순수 함수 pytest 통과
- `bash -n record_flight.sh` 문법 검증 + WSL SITL dry-run 1회 (rosbag·launch.log 생성 확인 — 사람과 협업)

### 배포 검증 — 합격 기준 [배포] (사람, RPi)

```
[ ] record_flight.sh 벤치 기동 1회 (arm 불필요) → 폴더에 rosbag + launch.log 생성
[ ] arm~disarm 1회 후 pull_ulog.py로 .ulg 자동 회수 (실패 시 SD 폴백 안내 동작 확인)
[ ] 개발컴 fetch_logs.ps1 → logs/ 회수 → pyulog로 열람 확인
[ ] Docker 컨테이너 `fc` 안의 logs/ 경로가 호스트에서 접근 가능한지 확인 (볼륨 마운트 여부)
```

### 미결 (구현 세션에서 확인할 것)

- 컨테이너 `fc`의 `/drone_ws/src/suridoksuri`가 호스트 마운트인지 — 아니면 logs 위치를 호스트 경로로 조정
- RPi에 pymavlink 설치 여부 · MAVLink FTP 다운로드 실측 속도 (USB 직결 기준. 너무 느리면 SD 회수를 기본, FTP를 옵션으로 뒤집음)
- PX4 `SDLOG_PROFILE` 기본값으로 충분한지 (고빈도 디버깅 필요 시 조정)

---

# 후속 계획 — 임의 WP 경로 생성·추종 검증

> **시점:** SITL-4 완료됨 → 진입 가능하나 SITL-5(실기체)가 우선.
> 대회 전체 미션(왕복 + 복수 사이클)과 vision→FC 연동(`pixel_to_gps`로 임의 GPS WP 주입)으로 가는 다리.

**배경:** `OffboardNode`는 시작 시 `waypoints` 파라미터를 1회 읽어 `run_planner`로 경로를 생성하고 추종한다.
따라서 **임의 WP를 launch 시점에 주입해 생성→추종**하는 것은 이미 가능하다(`ros2 launch ... -p waypoints:=[...]` 또는 YAML 교체).
미검증 영역은 (a) 다양한 WP 조합에 대한 **경로 생성 견고성**, (b) **런타임 WP 주입(재계획)** 이다.

> ⚠ **웨이포인트 비퇴화 필수** — 시작=끝 동일하거나 초단거리 레그면 플래너 divide-by-zero(NaN) (2026-07-03 실측).

---

## 작업 F — 임의 WP 경로 생성 견고성 하니스

**유형:** [코드] (Claude 자율)
**목적:** 임의/무작위 WP 세트가 항상 유효한 경로로 생성되는지 자동 검증한다.
**선행:** 작업 B (`apply_terminal_decel`) ✅. SITL-5 진행과 병행 가능.

**작업 목록:**

1. `fc_bridge/tests/test_arbitrary_wp.py` (신규) — 무작위 유효 WP 세트(레그 수 2~5, 레그 길이 50~300 m, 회전각 가변)를 생성해 `run_planner` + `apply_terminal_decel`을 통과시키고 경로 불변식을 검증.
2. (선택) **런타임 WP 주입 인터페이스** — `/fc_ros/waypoints` 토픽/서비스로 WP를 받아 재계획. vision→FC 연동의 기반. 본 작업에선 인터페이스 골격 + 단위 테스트만, SITL 통합은 SITL-6.

**테스트:** `fc_bridge/tests/test_arbitrary_wp.py`

```python
def test_arbitrary_wp_path_invariants():
    for _ in range(20):
        wps = random_valid_waypoints()                  # 2~5 WP, 50~300 m 레그
        path = run_planner("eta3", wps, vehicle_params)
        s = np.array([p.s for p in path.points])
        v = apply_terminal_decel(
            np.array([p.v_ref for p in path.points]), s, v_terminal, decel_dist)
        assert np.all(np.diff(s) >= 0)                   # 호길이 단조 증가
        assert v[-1] == pytest.approx(v_terminal)        # 끝점 감속
        assert np.all(v <= vehicle_params["v_cruise"] + 1e-6)  # 속도 상한
        kappa = np.abs([p.curvature for p in path.points])
        assert np.all(kappa <= a_max / vehicle_params["v_cruise"]**2 + 1e-3)  # 곡률 상한
```

**합격 기준:** `pytest fc_bridge/tests/test_arbitrary_wp.py` 통과.

---

## SITL-6 — 임의 WP 생성·추종 SITL

**유형:** [SITL] (사람 + Claude 보조)
**목적:** launch 시점에 임의 WP 세트를 주입해 생성→전체 사이클 추종을 SITL 검증한다.
**선행:** 작업 F · SITL-4 ✅

**절차 (사람):**

1. 임의/무작위 WP 세트 3~5종을 `-p waypoints:=[...]`로 주입.
2. 각 세트에 대해 `ros2 launch fc_ros phase2.launch.py` → 이륙~착륙 전체 사이클 추종.
3. WP 통과 정확도(기체 GPS) + 끝점 감속 로그 확인.

**합격 기준 (체크리스트):**

- [ ] 임의 WP 세트 전부 경로 생성 성공 (런타임 오류 없음)
- [ ] 전부 FOLLOWING 정상 진입 및 전체 사이클 완료
- [ ] WP 통과 오차 기준 이내, 끝점 v_terminal 도달 확인

---

## 주요 기술 참조

### VTOL 천이 MAVLink 명령

```
MAV_CMD_DO_VTOL_TRANSITION = 3000
param1 = 4  (MC → FW)   ← 목표 상태 FW(4)로 전환. ✅ SITL-1 실측 확인 (2026-06-19)
param1 = 3  (FW → MC)   ← 목표 상태 MC(3)로 전환. ✅ SITL-1 실측 확인 (2026-06-19)
```

MAVROS 호출: `/mavros/cmd/command` (mavros_msgs/srv/CommandLong)
응답: `result=0` = MAV_RESULT_ACCEPTED (성공)

### vtol_state 상수 (mavros_msgs/ExtendedState)

```python
VTOL_STATE_UNDEFINED        = 0   # 미사용
VTOL_STATE_TRANSITION_TO_FW = 1   # ✅ SITL-1 실측 확인
VTOL_STATE_TRANSITION_TO_MC = 2   # ✅ SITL-1 실측 확인
VTOL_STATE_MC               = 3   # ✅ SITL-1 실측 확인
VTOL_STATE_FW               = 4   # ✅ SITL-1 실측 확인
```

> **확정값 (2026-06-19 SITL-1 실측).**

### AUTO.TAKEOFF 동작 (SITL-1 실측)

```
ARM (CommandBool true)
  → set_mode "AUTO.TAKEOFF"
  → PX4: takeoff detected → 목표 고도까지 자율 상승
  → 완료 후 HOLD 모드로 전환   ← ✅ 확정 (2026-06-19)
```

> **실기체 주의:** AUTO.TAKEOFF는 **GPS 락 필수** (2026-07-03 실측 교훈). 수동비행 성공 ≠ GPS 락.

### 고도 판정 주의

```python
# VehicleState.pos_ned[2] = h_up (양수 = 고도 증가, NED D축과 반대 부호)
if state.pos_ned[2] >= self._transition_alt:   # CLIMBING 판정
    # 천이 고도 도달
```

---

## 파라미터 튜닝 가이드

> **시점:** SITL-5(실기체 배포)와 병행 수행한다.

### 천이 최대 가속도 ≤ 0.3g (≈ 2.94 m/s²)

천이 중 가속도는 두 원인에서 발생한다:

1. MC→FW 천이 시 PX4 내부 자세/추력 전환
2. FW→MC 역천이 시 고속 상태에서 MC가 제동하는 충격

역천이(2번)가 더 위험하다. 이를 제어하는 핵심 수단은 **작업 B의 종단 감속**(`apply_terminal_decel` → 끝점 속도 v_terminal)이다. OffboardNode에는 `d_pre_trans`/`v_transition_max` 같은 제어기 감속 파라미터가 **없다** (설계상 감속은 경로 후처리가 전담).

> SITL-4 실측: 역천이 가속도 ~1.5 m/s² (`VT_B_DEC_MSS` 1.0 설계값 부합) — 기준 충족. 실기체에서 재측정.

**조정 순서:**

| 단계 | 작업                                                   | 확인 방법                                                             |
| ---- | ------------------------------------------------------ | --------------------------------------------------------------------- |
| 1    | 역천이 직전 속도 로그 확인                             | `vel_ned` 크기 출력                                                   |
| 2    | `v_terminal` 조정 (= 스톨 × 1.1 = 15.2 m/s)            | v_profile 끝점 속도. 낮출수록 천이 충격 감소, 단 스톨(13.8) 이하 금지 |
| 3    | `decel_dist` 조정 (기본 80 m → 순항속도 빠를수록 길게) | dry-run 속도 프로파일에서 감속 시작 시점 확인                         |
| 4    | IMU `/mavros/imu/data` 로 천이 중 가속도 측정          | `linear_acceleration` 크기 ≤ 2.94                                     |
| 5    | 실기체 동일 측정 후 PX4 파라미터 추가 조정             | QGC 파라미터 편집기                                                   |

**관련 PX4 파라미터 (QGC):**

| PX4 파라미터       | 기본값   | 역할                       |
| ------------------ | -------- | -------------------------- |
| `VT_ARSP_TRANS`    | 10 m/s   | MC→FW 천이 시작 에어스피드 |
| `VT_TRANS_TIMEOUT` | 15 s     | 천이 타임아웃              |
| `VT_B_TRANS_DUR`   | 4 s      | 역천이 최대 지속 시간      |
| `VT_B_DEC_MSS`     | 1.0 m/s² | 역천이 중 목표 감속도      |

> `VT_B_DEC_MSS`를 줄이면 역천이가 부드러워지지만 고도 손실이 커진다. 실기체에서 교환관계 확인.

### WP 통과 위치 오차 최소화 (평가: 기체 GPS 값)

평가가 기체 GPS 값 기준이라 GPS 절대 편향은 상쇄되어 RTK 불필요.
실질 오차 원인: 위치 setpoint lookahead 거리와 코너 통과 속도.

| 단계 | 작업                                   | 비고                  |
| ---- | -------------------------------------- | --------------------- |
| 1    | cross_track_error 로그 수집            | FOLLOWING 중 `cte`    |
| 2    | lookahead 거리 조정 (현행 70 m)        | 너무 낮으면 진동      |
| 3    | `v_cruise` 감소 테스트                 | 느릴수록 오차↓ 시간↑  |
| 4    | eta3 WP 통과 반경 확인                 | `fc_bridge/planning/` |
| 5    | 코너 WP 감속 프로파일 확인             | v_profile 코너 속도   |

### 첫 비행 전 지상 안전 테스트 (SITL-5, 비행 직전)

> 프로펠러 제거 또는 기체를 고정한 상태에서 수행. Layer 1/2 동시 검증.

```
[ ] COM_RC_OVERRIDE = 3 QGC 파라미터 확인
[ ] ARM → setpoint 20Hz 발행 → OFFBOARD 진입 확인 (QGC 모드 표시)
[ ] RC 스틱 중립 이탈 → QGC에서 POSCTL 즉시 전환 확인 (Layer 1)
[ ] OFFBOARD 재진입 → /fc_ros/override 발행 → POSCTL/MANUAL 전환 + setpoint 중단 확인 (Layer 2)
[ ] 두 레이어 동시 트리거 → 충돌 없음 확인
```

> **이 테스트 통과 전 이륙 금지.**

### 실기체 배포 시 필수 조정 파라미터 체크리스트 (SITL-5)

```
[ ] home_lat / home_lon — 실제 이륙 지점 GPS (현재 기본: 스위스 취리히)
[ ] transition_alt — 실제 운용 고도 (법규·대회 규정)
[ ] v_cruise — 실기체 최적 순항 속도 (풍속 고려). 현재 20.0 유지 결정(2026-06-30, FW는 TECS가 속도 관장)
[ ] waypoints — 실제 미션 좌표로 교체 (yaml 두 곳: offboard_node·mission_node). 현재 테스트값 직선 300 m
[ ] 테스트 임시값은 yaml 수정 금지 — phase2.launch.py v_cruise:=… waypoints:=… launch 인자로만
[ ] l1_dist / lookahead — 실기체 비행 특성
[ ] v_terminal — 실기체 스톨 × 1.1 이상 (기준: 13.8 × 1.1 = 15.2 m/s)
[ ] decel_dist — 종단 감속 구간 길이 (v_cruise 클수록 길게)
[ ] VT_ARSP_TRANS — 실속 속도 + 여유값
[ ] VT_B_DEC_MSS — 역천이 감속도 (0.3g 조건 준수)
[ ] COM_RC_OVERRIDE = 3 PX4 적용 확인
[ ] RC 모드 스위치 → POSCTL 채널 연결 확인
[ ] /fc_ros/override 토픽 → 모드 전환 동작 테스트
[ ] RC 안전 스위치 채널 확인
[ ] 배터리 전압 임계값 확인
```

---

## 안전 및 긴급 수동 전환

> **현재 목표 (MVP):** 언제든 자동비행을 즉시 중단하고 수동 모드로 전환할 수 있어야 한다.
> **최종 목표:** 다층 failsafe (하드웨어·PX4·ROS2 레이어 분리, 비전 기반 이상 탐지) — 별도 계획.

### 레이어 구조

```
Layer 1 (PX4 하드웨어): RC 송신기 모드 스위치
        ↓ COM_RC_OVERRIDE=3 적용 시
        PX4가 OFFBOARD 무시 → POSCTL (MC) 또는 RC 모드 (FW)
        → 가장 빠르고 신뢰성 높음 (ROS2 스택과 무관)

Layer 2 (ROS2 소프트웨어): /fc_ros/override 토픽
        ↓ ros2 topic pub --once /fc_ros/override std_msgs/Bool "data: true"
        OffboardNode(_State.OVERRIDE): OFFBOARD setpoint 발행 중단 + manual 모드 요청
        (MC→POSCTL, FW→MANUAL) → 미진입(RC 없음) 시 AUTO.LOITER 안전 폴백
        → 키보드 터미널 또는 스크립트에서 트리거
```

### 동작 보장 조건

- OFFBOARD 진입 전/후 어느 상태에서도 두 레이어 모두 독립 동작.
- Layer 1(RC)은 Layer 2(ROS2) 의존 없이 단독 동작.
- Layer 2 실패(ROS2 크래시 등) 시 Layer 1이 최후 수단.

### 동작 정의

| 상황        | 트리거                              | 결과                                                |
| ----------- | ----------------------------------- | --------------------------------------------------- |
| MC 모드 중  | RC 모드스위치 또는 /fc_ros/override | POSCTL — 제자리 hover hold                          |
| FW 모드 중  | RC 모드스위치 또는 /fc_ros/override | MANUAL — 조종사 직접 조작                           |
| 어느 상태든 | RC 스틱 입력 (COM_RC_OVERRIDE)      | PX4가 즉시 POSCTL/RC 모드 전환                      |
| RC 부재 시  | /fc_ros/override                    | manual 거부 → 1초 후 AUTO.LOITER 폴백 (SITL-4 실측) |

> 판정 순수 함수: `override_reached`/`override_fallback_due` (`fc_bridge/execution/state_logic.py`).
> 실기체 검증 항목(RC 스틱 → POSCTL 전환)은 SITL-5 지상 안전 테스트에서 수행.
