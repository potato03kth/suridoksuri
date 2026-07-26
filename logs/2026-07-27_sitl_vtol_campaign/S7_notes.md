# SITL-7 S7 실행기록 — Phase 3 후반부 / 장애주입·안전 경로 (2026-07-27)

실행 세션 S7(캠페인 마지막 실행 세션). **C3 → C6a → C6b → C7 → C4** 를 지정 순서대로 실행.
각 런의 지표·판정은 `<run>/verdict.md`·`metrics.json`. 여기는 **verdict.md 가 산출하지 않는
장애주입 포렌식과 교차비교**만 남긴다.

- 저장소 HEAD: `5c131d5`(WSL 클론은 `3f6c517` + S6 하니스 uncommitted, 내용 동일)
- PX4: **`c890d9db0a` = 실기체 빌드**(`/root/PX4-vehicle`) — 전건. 기존 SITL 빌드(`9bb0d365c4`)는 미사용
- 실행: WSL `Ubuntu-22.04`, `gz_standard_vtol`, 시나리오 사이 `wsl.exe --terminate` 전건 적용
- 새 ulog 포렌식 도구: `tools/sitl/inject_probe.py`(nav_state 구간 / setpoint 스트림 단절 / DO_SET_MODE / EKF 풍속)

---

## 0. 이 세션의 핵심 답 — 안전 경로는 실제로 작동한다

| 안전 경로 | 결과 | 결정적 근거 (ulog) |
|---|---|---|
| **OVERRIDE (FW)** `C6a` | **작동** | MANUAL 요청(`176 p2=1`)@86.47 → PX4 거부(nav_state 불변) → 10틱 후 AUTO.LOITER(`176 p2=4 p3=3`)@87.45 → nav_state=AUTO_LOITER@87.47 → DONE. **`offboard_control_mode` 마지막 발행 86.47s = OVERRIDE 진입 시각** |
| **OVERRIDE (MC)** `C6b` | **작동 — 최초 검증** | POSCTL 요청(`176 p2=3`)@102.79 → 거부 → AUTO.LOITER@103.73 → nav_state@103.75 → DONE. `offboard_control_mode` 마지막 102.73s |
| **PILOT_TAKEOVER** `C7_pxvehicle_rcstream` | **작동 — 최초 검증** | 주입 POSCTL 이 **실제로 먹혀** nav_state=POSCTL@81.41(101.5초 유지). 노드가 `조종사 인계 감지` 1회 → **`offboard_control_mode` 82.14s 에서 영구 정지**, 이후 `vehicle_command` **0건**(재요청 0) |
| **OFFBOARD 재요청 복구** `C3` | **작동** | 천이 구간에 AUTO.LOITER 주입 → 0.97s 만에 OFFBOARD 복귀, 미션 완주. setpoint 스트림은 **한 번도 끊기지 않음**(gap 0) |

**setpoint 발행 중단은 세 경로 모두에서 ulog 로 확인됐다.** `offboard_control_mode` 는 MAVROS 가
오프보드 setpoint 를 중계할 때만 생기는 토픽이라, 이 토픽이 끊긴 시각 = 노드가 손을 뗀 시각이다.

---

## 1. 결과 요약

| 런 | 시나리오 | 종료 | 판정 | 주입 발화 | 주입 효과 |
|---|---|---|---|---|---|
| `C3_pxvehicle` | 천이 중 AUTO.LOITER | done(0) | FAIL 2 / PASS 10 / WARN 1 | ✅ | ✅ nav_state 변경 |
| `C3_pxvehicle_try1_noinject` | (무주입 대조군) | done(0) | FAIL 2 / PASS 10 / WARN 1 | ❌ 트리거 미성립 | — |
| `C3_pxvehicle_try2_cli_late` | 천이 노렸으나 FOLLOWING 착탄 | done(0) | FAIL 3 / PASS 9 / WARN 1 | ✅ (4.04s 늦음) | ✅ |
| `C3_pxvehicle_try3_px4lost` | 천이 명중, 착륙 중 PX4 유실 | done(0)* | FAIL 3 / PASS 9 / WARN 1 | ✅ | ✅ |
| `C4_pxvehicle_wind8` | **바람 8 m/s** | done(0) | FAIL 3 / PASS 8 / WARN 2 | ✅ probe | ✅ 풍속 실측 7.89 m/s |
| `C6a_pxvehicle` | OVERRIDE (FW 순항) | done(0) | FAIL 3 / NULL 1 / PASS 8 / WARN 1 | ✅ | ✅ |
| `C6b_pxvehicle` | OVERRIDE (MC HOLD) | done(0) | FAIL 3 / PASS 9 / WARN 1 | ✅ | ✅ |
| `C7_pxvehicle` | 조종사 인계(RC 없음) | done(0) | FAIL 2 / PASS 10 / WARN 1 | ✅ 발화·FCU 도달 | ❌ **PX4 가 거부** |
| `C7_pxvehicle_rcstream` | 조종사 인계(MAVLink 조이스틱) | **range_exceeded(6)** | FAIL 5 / NULL 1 / PASS 6 / WARN 1 | ✅ | ✅ POSCTL 진입 |

\* `try3` 는 착륙 중 PX4 가 사라졌다 — §6.

**C6a/C6b/C7_rcstream 의 `disarm 확인`·`vtol_state 시퀀스`·`완주` FAIL 은 결함이 아니다.**
이 시나리오들은 **미션을 의도적으로 중단**시키는 것이 목적이라 역천이·착륙·disarm 이 원래 없다.
`setpoint 점프`·`수직 가속` FAIL 은 전 캠페인 공통(지표 정의 문제, S2 §1과 동일 원인).

---

## 2. 장애주입 훅 — 액션별 실사용 검증 결과

S6 까지 **`probe` 만** 실사용 검증됨이었다. 이번 세션에서 나머지 전부를 처음 사용했다.

| 액션 | 상태 | 근거 |
|---|---|---|
| `probe` | 기검증 | C4 에서 재확인 (`/mavros/state`, rc=0) |
| `set_mode` | **✅ 검증** | C3(AUTO.LOITER)·C7(POSCTL). ulog `vehicle_command 176` 으로 FCU 도달 확인 |
| `override` | **✅ 검증** | C6a/C6b. in-process 퍼블리셔가 **구독자 1개 확인 후** 발행 |
| `param_set` | **미사용 — 여전히 미검증** | 이번 세션 시나리오에 없음 |

### 2-1. 트리거 지연이 C3 를 두 번 망쳤다 (하니스 수정의 직접 원인)

노려야 하는 구간(`천이 명령 요청` → `FW 전환 완료`)은 실측 **2.50~2.60초**뿐이다.

| 시도 | 트리거 | 발화 지연 | 결과 |
|---|---|---|---|
| 1 | `on_vtol_state: 1` (`ros2 topic echo --once` 폴링) | — | **트리거 자체가 안 걸림.** 폴링 1바퀴가 1~2초라 2.6초 창을 통째로 놓침 |
| 2 | `on_log` + `ros2 service call` CLI | **4.04s** | 천이 종료 **1.66초 뒤**(FOLLOWING) 착탄 |
| 3·4 | `on_log` + **in-process rclpy** | **0.20s** | 창 시작 0.13s / 0.20s 지점 명중 |

- 2차 실측 분해: 트리거 로그 21:27:31.961 → Injector 발화 21:27:32.168(**+0.207s**) →
  PX4 모드 변경 ulog 80.64s ≈ 21:27:36.21. **CLI `ros2 service call` 이 3.8초를 먹는다**
  (매 호출마다 rclpy 노드 생성 + 디스커버리).
- in-process 클라이언트의 서비스 왕복은 **0.002~0.005s**(meta.json `inject_results[].output`).

---

## 3. C3 — 천이 중 OFFBOARD 강제 이탈

### 3-1. 재요청 빈도 — **로그는 10Hz, 실제 요청은 1Hz** (2026-07-25 회귀 아님)

| 런 | 로그 문장 | 건수 | 로그 주기 | ulog `176 p2=6`(실제 요청) |
|---|---|---|---|---|
| `C3_pxvehicle` | `천이 중 OFFBOARD 이탈 → 재요청` | 10 | 0.90s → **10.0 Hz** | **1건**(76.89s) |
| `C3_pxvehicle_try3_px4lost` | 〃 | 11 | 1.000s → **10.00 Hz** | **2건**(77.00, 77.99 — 간격 **0.99s**) |
| `C3_pxvehicle_try2_cli_late` | `FOLLOWING 중 OFFBOARD 이탈 → 재요청` | 11 | 1.016s → 9.85 Hz | **2건**(81.02, 81.98 — 간격 **0.96s**) |

**결론:** `_request_offboard()` 의 1Hz throttle(`_offboard_req_min_interval=1.0`)은 정상 동작한다.
**다만 WARN 로그는 throttle 밖에 있어 제어루프 10Hz 그대로 찍힌다** —
`offboard_node.py:722-725`(천이) / `:1153-1156`(FOLLOWING) 둘 다 `warn()` 이 `_request_offboard()` 와
같은 블록에 조건 없이 놓여 있다. **로그만 보면 "0.9초에 10회 재요청"으로 보인다** — 2026-07-25
사고와 육안상 구별이 안 된다. 실기체 로그로 사고를 판정할 때 이 착시를 반드시 감안해야 한다.
(사고 판정의 실제 근거는 ulog `vehicle_command` 여야 한다.)

### 3-2. 복구·완주

| 항목 | `C3_pxvehicle` | `try3` | `try2`(FOLLOWING 착탄) |
|---|---|---|---|
| 주입 도달(ulog `176 p2=4 p3=3`) | 75.92s | 76.63s | 80.62s |
| nav_state=AUTO_LOITER | 75.94s | 76.65s | 80.64s |
| OFFBOARD 복귀 | 76.91s | 77.02s | 81.04s |
| **OFFBOARD 상실 시간** | **0.97s** | **0.37s** | **0.40s** |
| 천이 중단 여부 | **없음** (TRANS_TO_FW 2.52s 정상 완료) | 없음(2.50s) | 해당 없음 |
| 미션 | 완주 (DONE, disarm 확인) | 완주 | 완주 |

### 3-3. 1Hz mode flapping — **관측되지 않았다**

세 런 모두 AUTO_LOITER 구간이 **단 1회**(0.37~0.97s)였고 PX4 가 되받아치지 않았다.
정적 감사 §D 가 "잔여 위험"으로 남긴 flapping 은 이 조건에선 발생하지 않는다.

### 3-4. setpoint 스트림은 끊기지 않았다

`offboard_control_mode` 518샘플 / 10.0Hz / **gap 0건**. 설계 의도(천이 중에는 위치 명령을
끊지 않고 재요청만)와 일치.

---

## 4. C4 — 바람 8 m/s: **방법 확정, 실행 성공**

### 4-1. 확정된 주입 방법 (PX4 소스·원본 world 무수정)

1. `Tools/simulation/gz/worlds/default.sdf` 를 읽어 world 이름과 `<wind><linear_velocity>` 만
   바꾼 **새 파일** `windy8.sdf` 를 만든다 (`tools/sitl/` 밖의 실행 보조 스크립트 `/root/s7/mkwind.sh`).
2. `PX4_GZ_WORLD=windy8` 로 실행 (`/root/s7/run_wind.sh` = `run_vehicle.sh` + 이 환경변수).
3. 런 후 `windy8.sdf` 삭제. 원본 무결성 확인 — `git -C /root/PX4-vehicle status` 깨끗, HEAD `c890d9db0a` 불변.

**시도했다가 기각한 것:**
- `PX4_GZ_WORLDS` 로 PX4 트리 밖 디렉터리 지정 → **불가.** `build/px4_sitl_default/rootfs/gz_env.sh:15`
  가 `export PX4_GZ_WORLDS=/root/PX4-vehicle/Tools/simulation/gz/worlds` 를 **무조건 덮어쓴다**.
  월드 파일은 PX4 트리 안에 있어야 한다.
- PX4 파라미터(`SIM_GZ_*`)에 바람 항목 없음 — scenarios.yaml 주석대로 확인.
- `gz topic -t /world/<w>/wind` 발행 → 불필요했다. 월드 SDF `<wind>` 만으로 물리에 반영된다.

**왜 되는가:** `standard_vtol` 모델은 `gz::sim::systems::LiftDrag` 를 쓰고, 이 플러그인은
`components::Wind`/`WindTag` 를 직접 조회한다(설치본 `.so` 심볼 확인). 즉 world 의 `<wind>` 만
있으면 되고 **`WindEffects` 플러그인도 링크의 `enable_wind` 도 필요 없다**
(PX4 동봉 `windy.sdf` 에는 지면 링크에만 `enable_wind` 가 있고 차량 모델엔 어디에도 없다).

### 4-2. 바람이 실제로 먹혔다는 증거

- gz 서버 명령줄 `gz sim ... -s .../windy8.sdf`, 토픽 `/world/windy8/clock` 확인
- **EKF 풍속 추정 최대 7.89 m/s**(주입값 8.0), FW 순항 종반 `east 3.72 / north 1.48`
  (`inject_probe.py` `wind_estimate`, ulog `wind` 토픽 3988샘플)
- MC 호버 중 `yaw` 가 1.627 rad 까지 돌아감 — 무풍 런에서는 0.02~0.07 rad

### 4-3. 무풍 대조군(`C3_pxvehicle_try1_noinject`, 동일 경로·동일 인자) 대비

| 지표 | 무풍 | **바람 8 m/s** | 배수 |
|---|---|---|---|
| FW cte 최대 (node.log) | 1.2 m | **4.0 m** | 3.3× |
| FW cte 평균 | 0.32 m | **1.34 m** | 4.2× |
| 순항 고도편차 최대 | 2.24 m | **3.99 m**(±3m 기준 **초과**) | 1.8× |
| 헤딩 정렬 소요 | 13.40 s | 13.77 s | 1.03× |
| **정렬 잔류오차(창 종료 시점)** | 1.996° | **15.34°** | **7.7×** |
| 정천이 / 역천이 | 2.528 / 4.976 s | 2.520 / 4.940 s | 동일 |
| 접지 제외 수직가속 피크 | 0.660 g | 0.660 g | 동일 |
| node.log 경고 | 1건 | **0건** | — |

- cte 부호 이력: `-1.6 → -4.0 → -1.8 → +0.7 → +1.4 → +1.7 → -0.5 → -0.2` — 초반 좌측(서쪽)으로
  4.0m 밀렸다가 L1 이 되잡고 반대편 1.7m 로 오버슈트한 뒤 수렴. **발산하지 않는다.**
- **정적 감사 §D 예측("P1/P2 위치 피드백 없음 → 16~64m 표류")은 이 조건에서 빗나갔다.**
  실측 표류는 없었다(호버 위치 ±0.05m). CLIMBING 은 PX4 `AUTO_TAKEOFF`(위치제어)이고,
  천이 전 정렬구간도 MC 위치제어가 바람을 상쇄했다. 대신 **헤딩 쪽에 나타났다**
  (잔류오차 1.996°→15.34°). 예측의 방향(피드백 부재 → 바람 취약)은 맞았으나 **나타나는 축이 위치가 아니라 헤딩**이다.
- `climbing_reached` 의 vz 게이트가 안 닫혀 CLIMBING 이 지연될 것이라는 예측도 빗나갔다
  (CLIMBING 체류가 무풍 런과 같은 수준).
- **미션은 완주했고 node.log 경고가 0건이다.** 8 m/s 는 이 기체·이 경로에 치명적이지 않다.

---

## 5. C7 — 조종사 인계: **1차 주입 실패 → 조건 보정 후 검증 성공**

### 5-1. `C7_pxvehicle` — 주입은 FCU 까지 갔지만 PX4 가 거부

- `inject_results`: rc=0, `mode_sent=True`, in-process 왕복 0.003s
- ulog `vehicle_command` **`176 p1=129 p2=3`(POSCTL) @88.45s** — MAVROS 는 정상 중계했다
- **`nav_state` 는 OFFBOARD 그대로**(63.11→113.04s 연속). 모드가 안 바뀌었다
- 노드는 `조종사 인계 감지` 를 찍지 않았고 미션을 그대로 완주했다 — **정상 동작이다**
  (모드가 안 바뀌었으니 인계도 없다)
- **SITL-1 의 "headless SITL 에서 POSCTL 진입 재현 불가"가 그대로 재현됐다.**
  원인은 PX4 `modeCheck.cpp:144` — `manual_control_signal_lost` 이면 MANUAL 계열 모드의
  canRun 비트가 꺼진다. headless SITL 엔 RC 도 조이스틱도 없다.

### 5-2. `C7_pxvehicle_rcstream` — MAVLink 조이스틱을 붙여 조건 성립

`COM_RC_IN_MODE` 기본값이 **3 = "RC or MAVLink keep first"**(`commander_params.yaml:62`) 이므로
RC 가 한 번도 유효하지 않았던 SITL 에서는 MAVLink `MANUAL_CONTROL` 이 그대로 첫 유효 소스가 된다.
`/mavros/manual_control/send` 로 **중립 스틱**을 20Hz 로 흘린 것 외에 **비행코드·하니스·PX4·파라미터 무수정**.

| 관측 | 값 |
|---|---|
| 주입 도달 | ulog `176 p1=129 p2=3` @81.39s |
| **nav_state → POSCTL** | **81.41s** (이후 101.49초 유지) |
| 노드 반응 | `조종사 인계 감지 (mode=POSCTL) — 세트포인트 발행 중단, OFFBOARD 재요청 안 함` **1회** |
| **setpoint 발행 중단** | `offboard_control_mode` 마지막 **82.14s** → 이후 영구 정지 (모드 변경 인지에 0.73s = `/mavros/state` 1Hz 지연) |
| **이후 재요청** | `vehicle_command` **0건**. 런 전체 WARN **1건**(그 인계 로그뿐) |
| 기체 거동 | POSCTL 중립스틱 → 직진 유지. **1564m 지점에서 거리 감시 발동, 종료** |

**"되찾아오지 않는다"가 실증됐다.**

---

## 6. 이상 런 2건 (보존, 삭제 금지)

- **`C3_pxvehicle_try1_noinject`** — 주입 트리거 미성립. 무주입 완주라서 **C4 의 무풍 대조군으로 사용**했다.
- **`C3_pxvehicle_try3_px4lost`** — 주입·복구는 정상이었으나 **착륙 중 PX4 가 사라졌다.**
  근거: `mavros.log` `CON: Lost connection, HEARTBEAT timed out.` @1785101732.71 (이 캠페인 통틀어 1건),
  같은 순간 node.log 가 `armed=False` 를 보고 `착륙 완료 (disarmed) -> DONE` 오판,
  LANDING→DONE **10.2초**(정상 런 43.3~45.2초), ulog 가 `AUTO_LAND` 시작 0.66초 뒤 끊김,
  `ulog_duration_s/elapsed_s = 0.829`(정상대 0.92~0.99). **주입 결과는 유효하나 착륙 구간은 무효.**
  동일 조건 재실행(`C3_pxvehicle`)이 완주(ratio 0.917, disarm 확인)했으므로 일회성 환경 실패로 판단.

---

## 7. 정적 감사 §D 예측 대조

| ID | 예측 | 실측 | 판정 |
|---|---|---|---|
| C3 | PASS. 위치 setpoint 안 끊고 1Hz 재요청. 잔여 위험 = 1Hz mode flapping | 완주. 스트림 gap 0. 실제 요청 1Hz. **flapping 없음** | **적중** (+ 로그는 10Hz 라는 미기재 사실 발견) |
| C4 | 부분 FAIL. P1/P2 위치 피드백 없어 **16~64m 표류**, vz 게이트 미폐쇄로 CLIMBING 지연 | 완주. **표류 없음**(호버 ±0.05m), CLIMBING 지연 없음. 대신 **헤딩 잔류오차 1.996°→15.34°**, cte 3.3배, 고도편차 3.99m(기준 초과) | **부분 빗나감** — 취약성은 맞고 **나타나는 축이 틀림** |
| C6(FW) | PASS. MANUAL 거부 → 10틱 후 AUTO.LOITER → DONE (SITL-4 와 동일 3줄) | **정확히 3줄 일치** | **적중** |
| C6(MC) | PASS. POSCTL 분기 미검증 | POSCTL 분기 발화 확인, 폴백·DONE 동일 | **적중** |
| C7 | PASS 예상, 단 **측정 불가 위험** (SITL-1: headless POSCTL 재현 불가) | 1차 **그대로 재현 불가**. MAVLink 조이스틱 부여 후 **완전 검증** | **적중** (경고까지 적중) |

---

## 8. 경고 전량 (9런 합산, 종류별) — 무해성 판단은 사람이 한다

`verdict.md` 의 표를 인스턴스 식별자(pid/param#/ms/EVENT args)만 정규화해 합산했다.

| 출처 | 레벨 | 문구 | 합계 | 비고 |
|---|---|---|---|---|
| mavros | ERROR | `FCU: EVENT 7791755 with args ...` | 99 | 전 캠페인 상시 |
| mavros | WARN | `FCU: UNK(8): EVENT 11047904 with args ...` | 80 | 〃 |
| mavros | WARN | `VER: unicast request timeout, retries left 2` | 46 | 브링업 |
| **node** | **WARN** | **`/mavros/cmd/arming 서비스 없음`** | **44** | **C7 21건 / C6a 13 / C6b 9 — 런마다 0~21건으로 요동.** ARM 직전 서비스 디스커버리 경합. 결국 ARM 은 전건 성공 |
| mavros | WARN | `VER: broadcast request timeout, retries left 4` | 36 | 브링업 |
| mavros | ERROR | `VER: command plugin service call failed!` | 36 | 〃 |
| mavros | WARN | `CMD: Unexpected command 520, result 3` | 31 | 〃 |
| **node** | **WARN** | **`천이 중 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER)`** | **21** | **C3 주입의 의도된 결과**(10 + 11) |
| mavros | WARN | `TM: RTT too high for timesync: N ms.` | 21 | 878~2097 ms |
| mavros | ERROR | `TM: Time jump detected. Resetting time synchroniser.` | 20 | lockstep 시뮬 클록 |
| node | ERROR | `[offboard_node-2] Traceback` / `KeyboardInterrupt` | 18 / 18 | 하니스 SIGINT 이후 |
| node | ERROR | `[telemetry_node-1] Traceback` / `KeyboardInterrupt` | 18 / 18 | 〃 |
| mavros | WARN | `PR: Failed to get parameter type: CBRK_SUPPLY_CHK` | 11 | 프리플라이트 우회, 되읽기 검증은 전건 통과 |
| **node** | **WARN** | **`FOLLOWING 중 OFFBOARD 이탈 → 재요청`** | **11** | **try2 늦은 착탄의 결과** |
| node | ERROR | `process has died [pid N, exit code -2 ...]` ×2종 | 9 + 9 | 하니스 SIGINT |
| node | WARN | `[WARNING] [launch]: user interrupted with ctrl-c` | 9 | 〃 |
| mavros | WARN | `VER: your FCU don't support AUTOPILOT_VERSION` | 9 | 알려진 코스메틱 |
| mavros | WARN | `UAS Executor terminated` | 9 | 종료 |
| mavros | WARN | `PR: request param #N timeout, ... N params still missing` | 8 | 파라미터 동기 |
| mavros | WARN | `PR: Failed to get parameter type: NAV_DLL_ACT` | 3 | 〃 |
| **node** | **WARN** | `수동/안전 모드 진입 확인 (mode=AUTO.LOITER) -> DONE` | **2** | C6a·C6b 의도된 결과 |
| **mavros** | **WARN** | **`CON: Lost connection, HEARTBEAT timed out.`** | **1** | **`try3` 의 PX4 유실 — §6** |
| node | WARN | `긴급 수동 전환 실행 → MANUAL 요청` | 1 | C6a 의도 |
| node | WARN | `수동 모드(MANUAL) 미진입 → AUTO.LOITER 안전 폴백 요청` | 1 | 〃 |
| node | WARN | `긴급 수동 전환 실행 → POSCTL 요청` | 1 | C6b 의도 |
| node | WARN | `수동 모드(POSCTL) 미진입 → AUTO.LOITER 안전 폴백 요청` | 1 | 〃 |
| node | WARN | `조종사 인계 감지 (mode=POSCTL) — 세트포인트 발행 중단...` | 1 | C7_rcstream 의도 |

**새로 눈에 띄는 것은 두 가지뿐이다:** ① `/mavros/cmd/arming 서비스 없음` 이 런마다 0~21건으로
크게 요동하는 것(경합, 결과적으로는 무해했으나 실기체에서 ARM 지연으로 나타날 수 있음),
② `CON: Lost connection` 1건(=`try3` 환경 실패).

---

## 9. 하니스 수정 내역 (전부 이번 세션)

| 파일 | 변경 | 이유 |
|---|---|---|
| `run_scenario.py` | **`RangeGuard`** — `/mavros/local_position/pose` 4초 폴링, 이륙지점 수평거리 > `--range-limit-m`(기본 1500) 이면 즉시 종료, **exit 6 / `reason="range_exceeded"`**. 매 런 `meta.json.range_guard` 에 관측 최대거리 기록 | 지시 작업 (0). C10 의 5.85km 이탈 |
| `run_scenario.py` | **`on_log:` 트리거** — node.log 정규식 최초 매치 시 발화(Monitor 가 이미 읽는 스트림, 루프 0.1s) | `on_vtol_state` 폴링(1~2s)이 2.6초 천이창을 놓쳐 C3 1차 실패 |
| `run_scenario.py` | **`RosInjectClient`** — set_mode/override 를 in-process rclpy 로. 실패 시 CLI 자동 폴백, `inject_results[].via` 에 기록 | CLI 4.04s → 0.20s |
| `run_scenario.py` | `meta.json` 에 `px4_gz_world` / `px4_gz_worlds_dir`, `inject_results[].done_mono_s` 추가 | C4 provenance, 주입 소요시간 |
| `scenarios.yaml` | C3 트리거 `on_vtol_state:1` → `on_log:"MC→FW 천이 명령 요청"`, C4 주석에 확정된 바람 주입법 | 위와 같음 |
| `inject_probe.py` | **신규.** ulog → nav_state 구간 / `offboard_control_mode`·`trajectory_setpoint` 단절 / `vehicle_command` / EKF 풍속 | "주입이 먹혔는가"·"setpoint 가 끊겼는가"를 `analyze_run.py` 와 무관하게 답하기 위해 |
| `README.md` | 종료코드 6, 거리감시·`on_log`·in-process 주입 문단 | 문서화 |

**판정 로직(`analyze_run.py`)과 임계값은 손대지 않았다.** 비행코드(`offboard_node.py`,
`state_logic.py`, `fc_bridge/`)·PX4 소스도 무수정. `setpoint_raw/local`(A안) 미구현.

보조 실행 스크립트(저장소 밖, WSL `/root/s7/`): `mkwind.sh`(바람 월드 생성/삭제),
`run_wind.sh`(C4 실행), `run_c7_rc.sh`(C7 조이스틱 스트림), `analyze_many.sh`, `xfer_out.sh`.

---

## 10. 남은 것

- **`param_set` 주입 경로는 여전히 미검증** — 이번 세션 시나리오에 없었다.
- **C9 미실행** — Phase 3 중 유일한 미실행 시나리오(캠페인 26런 중).
- **`천이/FOLLOWING 중 OFFBOARD 이탈` WARN 이 10Hz 로 찍히는 문제** — 코드 수정 없이 기록만 함
  (비행코드 수정 금지 지시). 실기체 사고 판정 시 착시 주의.
- **헤딩 정렬(Phase 2) 중 OFFBOARD 이탈은 복구 경로가 없다(코드 정독 결과, 미실측).**
  `_step_transition_fw` 의 Phase 2 는 `_current_mode != "OFFBOARD"` 면 속도-0 만 발행하고
  `_fw_offboard_requested` 가 이미 True 라 **재요청하지 않는다** → 무한 대기 가능.
  이번 세션은 ACTIVE TRANSITION 구간을 노렸으므로 이 구간은 주입해보지 못했다.
