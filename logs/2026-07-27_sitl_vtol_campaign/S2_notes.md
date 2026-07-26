# SITL-7 S2 실행기록 — Phase 1 + 직선 회귀 (2026-07-27)

실행 세션 S2. A1(S1 기준선)에 이어 **A2 → B6 → B1 → A3 → A4** 5건을 정적 감사 §6 권고 순서대로 실행.
각 시나리오 산출물은 같은 디렉터리의 `<ID>/verdict.md`·`metrics.json`. 여기는 **verdict.md 가 산출하지
않는 교차비교·ulog 포렌식**만 남긴다.

- 저장소 HEAD: `3b52ac1` (5건 전부 동일)
- 실행 호스트: WSL `Ubuntu-22.04`, PX4 `v1.18.0-beta1-155-g9bb0d365c4`, `gz_standard_vtol`
- 시나리오 사이 `wsl.exe --terminate Ubuntu-22.04` 로 gz 잔류 제거 (전건 적용)

## 1. 결과 요약

| ID | 경로 | 종료 | 판정 요약 | 비고 |
|---|---|---|---|---|
| A1 | 직선 300m, `frame=local` | done(0) | FAIL 2 / PASS 10 / WARN 1 | S1 기준선 |
| A2 | 직선 300m, `frame=takeoff` | done(0) | FAIL 2 / PASS 10 / WARN 1 | A1과 동등 |
| B6 | 직선 200m | done(0) | FAIL 2 / PASS 10 / WARN 1 | A1과 동등 |
| B1 | 직선 500m | done(0) | FAIL 2 / PASS 10 / WARN 1 | A1과 동등 |
| **A3** | **L자 200/200 (90°)** | **timeout(2)** | **FAIL 5 / NULL 1 / PASS 5 / WARN 2** | **6.5km 직진 폭주 — 아래 §5** |
| A4 | 중간 WP z=80m | done(0) | FAIL 2 / PASS 10 / WARN 1 | 중간 z 무시 실증 |

**FAIL 2 의 내용은 5건 전부 동일**(`setpoint 점프`·`수직 가속`) — 둘 다 A1 기준선과 같은 원인이며
지표 정의의 문제다(계획서 4장 임계가 FW lookahead 전진량보다 작음 / 접지 충격이 판정 지배).
즉 **A1·A2·B6·B1·A4 는 기준선 동등 = 직선 회귀 확보**, A3만 실질 실패다.

## 2. 상태 전이 타임라인 (체류 초)

| 상태 | A1 | A2 | B6 | B1 | A3 | A4 |
|---|---|---|---|---|---|---|
| ARM_TAKEOFF | 0.31 | 1.87 | 0.83 | 1.11 | 0.55 | 1.00 |
| CLIMBING | 33.69 | 29.10 | 32.50 | 29.99 | 26.90 | 28.49 |
| TRANSITION_FW | 18.50 | 21.40 | 21.70 | 21.70 | 21.30 | 22.00 |
| STREAMING | 0.107 | 0.104 | 0.108 | 0.108 | 0.108 | 0.106 |
| FOLLOWING | 21.71 | 19.00 | 11.30 | 35.71 | **472.32** | 21.71 |
| TRANSITION_MC | 5.39 | 8.29 | 5.79 | 5.78 | — | 5.39 |
| HOLD | 8.60 | 8.80 | 12.30 | 9.30 | — | 8.70 |
| LANDING | 49.30 | 48.20 | 45.40 | 47.90 | — | 46.20 |
| 미션 총 | 137.6 | 136.8 | 129.9 | 151.6 | 573(강제종료) | 133.6 |

STREAMING 은 6건 전부 **1틱(≈0.1s) 통과** — 정적 감사 C9 예상("VTOL의 STREAMING은 구조적으로 1틱")과 일치.

## 3. 헤딩 정렬 (E-5 마진 추적)

`wp0_heading_tol = 0.05 rad = 2.9°`. `node.log '헤딩 정렬 완료'` 의 err 이 판정에 쓰인 값이다.

| ID | node.log err | 마진 | ulog 최종오차 | 정렬 소요 | 단조수렴 | tol 진입 후 재증가 |
|---|---|---|---|---|---|---|
| A1 | **−2.4°** | 0.5° | 1.21° | 13.24s | True | 0 |
| A2 | **−2.3°** | 0.6° | 1.42° | 13.83s | True | 0 |
| B6 | **−2.4°** | 0.5° | 1.56° | 14.12s | True | 0 |
| B1 | **−2.4°** | 0.5° | 2.02° | 14.48s | True | 0 |
| A3 | **−2.4°** | 0.5° | 0.38° | 15.41s | True | 0 |
| A4 | **−2.3°** | 0.6° | 0.38° | 16.09s | True | 0 |

- **6/6 이 −2.3~−2.4° 로 재현.** 분산이 0.1°밖에 안 되지만 **허용치까지 마진이 0.5~0.6°뿐**인 것도
  그대로 재현됐다(E-5). 목표 헤딩은 6건 전부 0°(북)라 **다른 방위에서의 잔류 오차는 미검증**
  (C2 동쪽 90°, B8 남쪽 180° 에서 확인 필요).
- `_fw_stable_ticks` 안정구간에서 유지되는 미세 P제어(gain 0.1) 덕에 ulog 상 실제 최종오차는
  0.38~2.02° 로 node.log 값보다 작다 — 즉 **로그의 −2.4°는 "정렬 판정 순간"의 값이고
  천이 명령 시점의 실제 오차는 더 작다.**

## 4. setpoint 계단 — 두 개가 있고 둘 다 구조적

### (a) TRANSITION_FW → FOLLOWING 인수인계 계단
`_step_transition_fw` 는 위치 setpoint 로 **`_pts[-1]`(경로 종점)** 을 발행하고,
FOLLOWING 은 **`pos + 70m` lookahead**(`_FW_LOOKAHEAD`)를 발행한다. 그 차이가 그대로 1틱 계단이 된다.

| ID | 실측 계단 | `경로길이 − 인수인계 pos − 70` | 인수인계 pos |
|---|---|---|---|
| B6(200m) | 114.83 m | 200 − 15.1 − 70 = 114.9 | 15.1 |
| A1(300m) | 216.10 m | 300 − 13.9 − 70 = 216.1 | 13.9 |
| A2(300m) | 217.80 m | 300 − 13.9 − 70 ≈ 216 | 13.9 |
| A4(300m) | 214.40 m | 300 − 15.6 − 70 = 214.4 | 15.6 |
| B1(500m) | 418.00 m | 500 − 12.0 − 70 = 418.0 | 12.0 |
| A3(L자) | 233.05 m | \|[200,200] − [80.3,0]\| = 233.1 | 11.8 |

**경로가 길수록 계단이 선형으로 커진다.** 계단 크기 = 경로길이에 비례. 500m에서 418m,
순간 환산 속도 2009.6 m/s. 실기체에서 이 계단이 실제 가속 프로파일에 어떻게 먹히는지는
정적 감사 F-3 미해결 항목 그대로다.

### (b) TRANSITION_MC → HOLD 의 1틱 왕복 스파이크
`_step_transition_mc` 의 `far_tgt = pos + end_dir*70` 이 **역천이가 끝나는 딱 그 1틱** 발행된 뒤
곧바로 HOLD 의 WP1 목표로 되돌아온다. ulog 원시 확인(A1, 200ms 간격):

```
 92.556  [300.000, 0.000, -50.000]   ← TRANSITION_FW 종점 목표
 92.808 ~ 97.784  NaN                ← 역천이 중 PX4가 위치 setpoint 를 NaN 으로 덮음
 97.940  [413.155, -1.389, -50.000]  ← far_tgt (= pos + 70m), 1틱
 98.140  [300.000, 0.000, -50.000]   ← HOLD WP1 목표, 복귀
```

| ID | 스파이크 | = WP1 초과거리 + 70m | 포착 여부 |
|---|---|---|---|
| A1 | 113.16 m | 43.4 + 70 | 포착 |
| B6 | 116.58 m | 46 + 70 | 포착 |
| B1 | 117.09 m | 47 + 70 | 포착 |
| A4 | 112.83 m | 43 + 70 | 포착 |
| **A2** | — | — | **미포착(§7 참조)** |

A2 에서만 안 잡힌 이유는 거동 차이가 아니라 **ulog 의 `trajectory_setpoint` 기록률이 5Hz**여서
1틱(100ms) 스파이크를 절반 확률로 놓치기 때문이다(§7).

## 5. 역천이 오버슈트 (ulog 실측)

| ID | TRANSITION_MC 진입 시 WP1까지 | HOLD 진입 시 WP1 거리 | 구간 최대 초과거리 |
|---|---|---|---|
| A1 | 2.70 m | 43.42 m | 47.01 m |
| A2 | 19.27 m* | 46.95 m | 47.92 m |
| B6 | 18.23 m* | 39.69 m | 50.40 m |
| B1 | 12.84 m* | 42.23 m | 50.93 m |
| A4 | 3.32 m | 43.00 m | 46.86 m |

\* TRANSITION_MC 진입 시각은 node.log↔ulog 시각정렬 오차(±0.6~1.3s)에 14 m/s 를 곱한 만큼
(최대 ±18m) 흔들린다 — `d_end_thresh=10` 트리거 자체는 node.log 가 근거. **HOLD 진입 거리와
구간 최대값은 알고리즘 특성(≈40~51m)으로 재현성이 높고 SITL-4 기록 "~43m"과 일치.**

## 6. A3 — 최우선 결함: L자 경로에서 6.5km 직진 폭주 (FOLLOWING 무한)

`waypoints=[0,0,50, 200,0,50, 200,200,50]`, `waypoint_frame=takeoff`. exit=2(timeout 480s).

### 관측
- 상태는 FOLLOWING 까지만 진행. **FOLLOWING 체류 472.3s, `_follow_ticks` 4360+**.
- 기체는 코너([200,0])를 **전혀 돌지 않고** 북쪽 직진, 로그 마지막 `pos=[6492.5, 22.7]`.
  **cte 최대 6292.4 m**(평균 3045.3 m).
- 고도는 정상 유지(순항 편차 max 2.09m), 속도 정상(≈14 m/s), `mode=OFFBOARD` 유지.
- **경보 0건** — OFFBOARD 이탈 없음, `offboard_reacquire=0`, PX4 페일세이프 미발동,
  지오펜스 미발동. 즉 **조종사 개입 없이는 멈추지 않는다.**

### ulog 포렌식 — FW OFFBOARD 가 위치 setpoint 의 lateral 성분을 쓰지 않는다
`vehicle_local_position` × `vehicle_attitude` × `trajectory_setpoint` 대조 (ulog 초):

```
t=147.60 pos=[159.7,-1.2] yaw= 0.35°  sp=[200.0, 27.7] sp_yaw=  3.28°   ← 목표가 전방우측
t=150.00 pos=[194.3,-1.1] yaw=-0.75°  sp=[199.9, 65.3] sp_yaw= 73.60°   ← 목표 우측 65m
t=152.40 pos=[228.5,-0.9] yaw=-0.33°  sp=[199.9, 70.0] sp_yaw=144.06°   ← 목표가 후방
t=198.01 pos=[885.5,-0.3] yaw= 2.18°  sp=[200.0, 77.7] sp_yaw=177.65°
```

- **목표가 전방 우측일 때(t=147~150)조차 롤/요 응답이 전혀 없다.** yaw 는 −0.75~+2.4° 범위에서
  6.5km 내내 북쪽 고정. 6300m 이동 동안 lateral 변위는 +23m(≈0.2°).
- 반면 **고도(sp z=−49.8)는 정확히 추종** — 즉 FW 는 `trajectory_setpoint` 의 z 는 쓰고 xy 는 안 쓴다.
- `position_setpoint_triplet.current.valid = 0`, `type = 5` 로 전 구간 고정 — 위치 triplet 경로가
  아예 활성화되지 않았다.
- **정적 감사 F-2("FW OFFBOARD 가 위치 setpoint 의 yaw 필드를 쓰는지")의 답: yaw 도 xy 도 안 쓴다.**

### 직선 4건이 통과한 것은 반증이 아니다
A1/A2/B6/B1/A4 의 경로는 전부 **천이 시점의 기수방위와 같은 직선**이다. FW 가 위치 setpoint 를
무시하고 그냥 직진해도 cte 는 0 근처가 나온다. **즉 직선 시나리오는 FW lateral 추종을 검증하지 못한다.**
`cte ≤ 1.5m` 는 "추종이 잘 된다"가 아니라 "기수 방향이 우연히 경로와 같다"의 결과일 수 있다.

### FOLLOWING 에 타임아웃·이탈 폴백이 없다 (정적 감사 A표 확정)
종료조건은 `dist_to_end = |pos − _pts[-1]| < d_end_thresh(10m)` 하나뿐이다. 기체가 멀어지면
이 값은 **단조 증가**하므로 영원히 성립하지 않는다. cte 6.3km 에도 아무 조치가 없다.

### 회귀 여부
`docs/sitl_verification_log.md` SITL-4(2026-06-30) 기록: **"L자 경로 … FOLLOWING·역천이·착륙 전체
사이클 완료 ✅ (FW는 90° 코너를 타이트하게 못 돌아 코너 오버슈트 — 정상)"**.
→ **당시 통과했던 시나리오가 지금 폭주한다 = 회귀다.** 다만 확정에는 아래가 필요하다(S2 범위 밖):
1. SITL-4 당시의 L자 WP 좌표가 기록돼 있지 않다(현재 A3 좌표와 동일한지 불명).
2. 그 사이 setpoint 발행 경로가 `cmd_vel`(속도) → `PoseStamped`(위치)로 바뀌었다.
   SITL-4 override 실패 기술에 "FW가 velocity 무시"가 나오므로 당시엔 속도 발행이었다.
3. PX4 버전이 바뀌었을 가능성(현재 `v1.18.0-beta1-155`).

**수정 판단은 오케스트레이터 몫 — S2는 코드를 고치지 않았다.**

## 7. 하니스·측정 한계 (S3 이후가 반드시 알아야 할 것)

1. **ulog `trajectory_setpoint` 은 5Hz 로만 기록된다** (발행은 10Hz). 같은 창의
   `offboard_control_mode` 는 10Hz(A1 FOLLOWING 200샘플/20s vs `trajectory_setpoint` 99샘플).
   → `analyze_run.py` 의 setpoint 점프 분석은 **실제 스트림의 절반만 본다.** 1틱 스파이크는
   50% 확률로 누락된다(A2 에서 실제 누락). **"점프 없음"을 근거로 쓰지 말 것.**
2. **`near_state_boundary`(±1s) 판정은 시각정렬 잔차에 취약하다.** A2 의 잔차가 1.298s 로
   ±1s 창보다 커서, A1 과 물리적으로 같은 사건이 A2 에선 경계위반 10건, A1 에선 2건으로 집계됐다.
   **경계위반 "건수" 자체는 시나리오 간 비교에 쓸 수 없다.** 크기(m)와 발생 지점으로 봐야 한다.
   매 런 `metrics.json.time_alignment.max_abs_residual_s` 확인 필수.
3. **node.log 타임스탬프는 벽시계, 제어 틱은 시뮬 클록이다.** ROS2 로그 스탬프는 시스템 클록이라
   sim 이 느려진 구간에서 "20틱이 4.9초 걸린 것처럼" 보인다(A1 tick40→60, B1 tick180→200).
   PX4 수신 측(`offboard_control_mode`)에는 그런 공백이 없다 — **제어루프 스톨이 아니다.**
4. `analyze_run.py` 의 `az` 판정은 접지 충격이 지배한다(전건 FAIL). 비행 중 값은
   `vertical_accel.excl_touchdown` 을 봐야 한다(전건 0.44~0.94g).

## 8. B1 — `_find_segment` O(N) 비용 (정적 감사 E-12) 실측

`offboard_control_mode` 수신 간격(= 노드 10Hz 발행의 프록시), 시뮬 클록 기준, FOLLOWING 구간:

| ID | 보간점 N | n | mean | p95 | p99 | max |
|---|---|---|---|---|---|---|
| A1 | 301 | 200 | 100.0 ms | 103.9 ms | 107.9 ms | 208.0 ms |
| **B1** | **501** | **322** | **100.0 ms** | **104.0 ms** | **108.0 ms** | **208.0 ms** |

예산 100 ms 대비 **p95 104 ms(+4%)**, 최대값은 두 런 모두 208 ms 이고 **둘 다 FOLLOWING 진입
경계 1건**(A1 t=73.7, B1 t=68.9)이다. **N 이 301→501 로 늘어도 지터가 커지지 않는다 —
E-12 는 이 규모에서 실측상 무의미.** (RPi5 는 별개 — 여기 결과로 실기체를 면제하지 말 것.)

node.log 의 20틱 블록 평균 주기도 A2/B6 는 99.98~99.99 ms 로 정확하다(A1·B1 의 이상치는 §7-3).

## 9. A2 — `waypoint_frame=takeoff` 첫 검증

```
경로 원점 = 이륙지점 [N=-0.08, E=-0.02, h_up=-0.11] 적용 (frame=takeoff)
  → WP0=[-0.08,-0.02] 순항고도 h_up=49.89
```

A1(`frame=local`)은 이 로그 자체가 없고 `_cruise_alt = waypoints[-1].z = 50.0`.

| ID | frame | 원점 오프셋 N/E/h_up | 적용된 `_cruise_alt` |
|---|---|---|---|
| A1 | local | (미적용) | 50.00 |
| A2 | takeoff | −0.08 / −0.02 / −0.11 | 49.89 |
| B6 | takeoff | −0.06 / −0.01 / +0.07 | 50.07 |
| B1 | takeoff | 0.00 / −0.02 / −0.04 | 49.96 |
| A3 | takeoff | −0.03 / −0.00 / −0.20 | 49.80 |
| A4 | takeoff | −0.02 / +0.00 / +0.04 | 50.04 |

**SITL 에서 EKF 로컬원점 ↔ 이륙지점 오프셋은 8cm/2cm/20cm 급**이다(PX4 SITL 이 스폰 지점에서
EKF 원점을 잡기 때문). 따라서 **A2 는 `waypoint_frame` 코드경로가 도는 것은 검증하지만
두 프레임의 차이는 검증하지 못한다** — 실기체(GPS 수렴 전 이륙)에서는 오프셋이 훨씬 크다.
프레임 차이의 실효성 검증은 실기체 몫으로 남는다.

## 10. A4 — `_cruise_alt` 스칼라화(E-10) 실증

`waypoints=[0,0,50, 150,0,80, 300,0,50]`. 중간 WP z=80m.

- 로그: `순항고도 h_up=50.04` (= `waypoints[-1].z` + 원점보정). **80 은 어디에도 안 나온다.**
- **전 비행 최대 AGL = 52.16 m** (ulog `vehicle_local_position`, t=78.49s, x≈51m — 천이 직후 블립).
  중간 WP 부근(x=150, t≈83s)의 AGL 은 49.4~49.6 m.
- 순항 고도편차: 평균 −0.04 m, 최대 |편차| 2.14 m (기준 50.03 m).

→ **중간 WP 의 z 는 완전히 무시된다.** 플래너 XY 출력이 A1 과 동일 301점인 것(정적 감사 E-2)과
합쳐 **E-10 확정.** 나머지 지표는 A1 과 동등(계단 214.4m, 역천이 오버슈트 43.0m, cte 1.2m).

## 11. 플래너 블로킹 실측 (E-11)

`meta.json.planner_blocking_s` = launch 기동 → offboard_node 첫 로그.

| ID | 경로 | 실측 | 정적 감사 E-2 (오케스트레이터 노트북) |
|---|---|---|---|
| A1/A2/B6/B1 | 2WP 직선 | 1.0 ~ 1.6 s | 0.00 ~ 0.01 s |
| A4 | 3WP 동일직선 | 2.2 s | 0.01 s |
| **A3** | **L자 3WP** | **68.5 s** | 50.72 s |

이 워크스테이션 WSL(SITL+MAVROS 동시 구동)에서 A3 는 **68.5초**. `--boot-timeout-s` 기본 300s 는
Phase 2 의 B5(감사 실측 129s → 이 환경에선 3분 이상 예상)까지는 버티지만 **여유가 크지 않다.**

## 12. 경고 전량 대조 (무해성 판단 안 함)

| 출처 | 레벨 | 패턴 | A1 | A2 | B6 | B1 | A3 | A4 |
|---|---|---|---|---|---|---|---|---|
| mavros | ERROR | `FCU: EVENT # with args …` | 13 | 13 | 13 | 13 | 8 | 13 |
| mavros | ERROR | `TM: Time jump detected. Resetting time synchroniser.` | 3 | 3 | 2 | 3 | **12** | 3 |
| mavros | WARN | `FCU: UNK(#): EVENT # with args …` | 10 | 10 | 10 | 10 | 6 | 10 |
| mavros | WARN | `VER: unicast request timeout, retries left #` | 5 | 5 | 5 | 4 | 5 | 5 |
| mavros | WARN | `VER: broadcast request timeout, retries left #` | 4 | 4 | 4 | 4 | 4 | 4 |
| mavros | ERROR | `VER: command plugin service call failed!` | 4 | 4 | 4 | 4 | 4 | 4 |
| mavros | WARN | `CMD: Unexpected command #, result #` | 3 | 3 | 3 | 3 | 2 | 3 |
| mavros | WARN | `TM: RTT too high for timesync: # ms.` | 3 | 1 | 3 | 3 | 3 | 2 |
| mavros | WARN | `VER: your FCU don't support AUTOPILOT_VERSION …` | 1 | 1 | 1 | 1 | 1 | 1 |
| mavros | WARN | `PR: Failed to get parameter type: CBRK_SUPPLY_CHK` | 1 | 1 | 1 | 1 | – | 2 |
| mavros | WARN | `PR: Failed to get parameter type: NAV_DLL_ACT` | – | – | – | – | 1 | – |
| mavros | WARN | `PR: request param ## timeout, … params still missing` | 1 | – | – | – | 1 | 1 |
| mavros | WARN | `UAS Executor terminated` | 1 | 1 | 1 | 1 | 1 | 1 |
| node | WARN | `/mavros/cmd/arming 서비스 없음` | 6 | – | – | 3 | – | – |
| node | WARN | `altitude 메시지 지연 age=#s > #s — 래치/캐시 의심` | – | – | – | – | 1 | – |
| **합계** | | | 49/12종 | 46/11종 | 47/11종 | 47/11종 | 48/12종 | 49/12종 |

- **경고 프로파일은 6건이 사실상 동일.** 새 종류는 A3 의 `altitude 메시지 지연 age=3.2s` 1건뿐.
- A3 의 `TM: Time jump` 12건은 다른 런의 4배 — 런 길이가 4배(573s)이고 플래너 68s 블로킹 중
  시뮬 클록이 크게 느려진 것과 상관 가능(미확정).
- 전건 `timeouts=[]`, `offboard_reacquire=0` — **OFFBOARD 재요청이 한 번도 발생하지 않았다**
  (C3 의 1Hz 재요청 경로는 이 6건에서 미발화 = 미검증).
- `mavros.guided_target: PositionTargetGlobal failed because no origin`(알려진 코스메틱)은
  이 6건 로그에 **나타나지 않았다**(`guided_target` 문자열은 플러그인 생성/초기화 INFO 2줄뿐).
- 교차확인: `grep -cE '\[(WARN|ERROR)\]' <id>/mavros.log` = 49/46/47/47/48/49 로 위 합계와 일치 —
  하니스가 누락 없이 집계했다.

## 13. 정적 감사 §D 예측 대조

| ID | 예측 | 실측 | 판정 |
|---|---|---|---|
| A2 | PASS. `_apply_path_origin` 로그 + `_cruise_alt=50+ground_h` 확인이 실검증점 | PASS. 로그·`_cruise_alt=49.89` 확인 | **적중** (단 SITL 오프셋이 10cm급이라 프레임 차이는 미검증 — §9) |
| B6 | PASS (A1 축소판) | PASS, 전 지표 A1 동등 | **적중** |
| B1 | PASS. 변수는 N=501 시 `_find_segment` 비용(E-12) | PASS. 지터 A1과 동일(p95 104ms) | **적중 + E-12는 기각** |
| A3 | 부분 FAIL. 천이가 [200,200](45° 이탈)을 목표 → STREAMING 경계 lateral 스냅백, FOLLOWING 시작 cte 큼 | **전면 FAIL(6.5km 폭주).** 그런데 **예측한 메커니즘은 틀렸다** — 천이 구간 첫 레그 이탈 최대 1.93m, FOLLOWING 시작 cte −0.9m, 스냅백 없음 | **틀림(심각도 과소평가 + 원인 오진)** |
| A4 | 완주 PASS + 중간 WP z=80 무시, 전 구간 h_up=50 고정(E-10) | 완주 PASS, 최대 AGL 52.16m, `h_up=50.04` | **적중** |

**A3 예측이 틀린 방식이 중요하다.** 감사는 "천이 목표가 45° 이탈이라 궤적이 휜다"고 봤지만,
실제로는 **FW 가 lateral 위치 setpoint 자체를 무시**하므로 45° 이탈도 스냅백도 일어나지 않았고,
대신 **코너를 못 도는 근본 결함**이 드러났다. 정적 분석이 "setpoint 를 주면 기체가 따라간다"를
전제했기 때문에 생긴 오진이다.

이 결과는 **B2·B3·B4·B5·B7·B8 예측 전부를 재검토하게 만든다** — 이들 예측도 같은 전제 위에 있다.
