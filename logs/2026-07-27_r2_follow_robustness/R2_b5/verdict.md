# R2_b5 — 판정

- 목적: R2 핵심: B5 사각 폐곡선(시점≈종점, 20m 마진) — _find_segment 회귀 검증의 핵심
- 실행: 2026-07-27T14:20:55.136275+00:00 ~ 2026-07-27T14:27:56.986849+00:00 (경과 384.7s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0, 0.0,200.0,50.0, 0.0,20.0,50.0] range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 14_21_06.ulg (meta.json 기록: 14_21_06.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.11794 x ulog + 1785162057.740` (앵커 4개, 최대 잔차 1.491s). 시뮬 클록이 벽시계보다 +11.8% 빠름/느림 — 상수 오프셋만 쓰면 10.57s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 171.898s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 215.624s → disarmed 369.632s (비행 154.008s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.6s / 역천이 5.096s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 9건 / 전체 위반 209건 / 샘플 1699개, 최대 70.5442m (352.7211 m/s). 경계 최대: 3.8693m@262.592s(TRANSITION_FW), 3.866m@262.392s(TRANSITION_FW), 3.7447m@262.792s(TRANSITION_FW). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=36.6121 m/s² (3.7334g) @366.464s state=LANDING; 접지(disarm−5s) 제외 시 15.372 m/s² (1.5675g) @313.388s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3406 m/s² (0.2387g), 13.9555→5.0298 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -96.1249° → 정렬 14.344s 소요, 최대 96.1907°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.3922m vs transition_alt 50.0m → -1.2156% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 52.7926m → 최저 52.7926m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.8981m, 평균편차 0.0064m, 최대 \|편차\| 5.9602m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 8.2m 평균 2.4875m (부호 -1.2~8.2m, n=24) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 16건/10종, mavros.log 53건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.2035 |
| CLIMBING | +0.2s | 30.5944 |
| TRANSITION_FW | +30.8s | 22.4001 |
| STREAMING | +53.2s | 0.1029 |
| FOLLOWING | +53.3s | 49.498 |
| TRANSITION_MC | +102.8s | 5.4 |
| HOLD | +108.2s | 14.0996 |
| LANDING | +122.3s | 49.5995 |
| DONE | +171.9s | 6.43 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.324 | 253.788 |
| 1 | TRANS_TO_FW | 259.112 | 2.6 |
| 4 | FW | 261.712 | 46.232 |
| 2 | TRANS_TO_MC | 307.944 | 5.096 |
| 3 | MC | 313.04 | 57.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785162476.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785162476.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785162476.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785162476.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785162476.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1147, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785162476.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1149, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 3 | 1785162363.1 |  | 세그먼트 인덱스 급변 192→222 (Δ+30, 전체 810) pos=[185.9,15.8] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | 1785162331.3 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785162296.7 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 4.983m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785162476.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785162067.1 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785162068.2 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | ERROR | 7 | 1785162120.2 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 5 | 1785162066.3 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785162064.5 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785162071.3 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785162067.2 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785162080.0 |  | TM: RTT too high for timesync: 1994.51 ms. |
| mavros.log | WARN | 1 | 1785162074.0 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785162074.3 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785162080.9 |  | PR: request param #504 timeout, retries left 2, and 343 params still missing |
| mavros.log | WARN | 1 | 1785162477.2 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
