# B2 — 판정

- 목적: 완만 곡선 4WP(30°급 꺾임) — eta3 NR 경로(_trapz 수정) 실행 검증
- 실행: 2026-07-26T19:46:29.232009+00:00 ~ 2026-07-26T19:49:47.459750+00:00 (경과 181.3s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 150.0,0.0,50.0, 300.0,80.0,50.0, 450.0,200.0,50.0]`
- 저장소 HEAD: `3b52ac1`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 19_46_40.ulg (meta.json 기록: 19_46_40.ulg)
- 요약: FAIL 2, PASS 9, WARN 2

- 시각 정렬: `wall = 1.08479 x ulog + 1785095196.232` (앵커 4개, 최대 잔차 0.106s). 시뮬 클록이 벽시계보다 +8.5% 빠름/느림 — 상수 오프셋만 쓰면 6.90s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 152.4592s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 29.732s → disarmed 167.728s (비행 137.996s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.504s / 역천이 5.472s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 3건 / 전체 위반 145건 / 샘플 690개, 최대 116.2829m (581.4146 m/s). 경계 최대: 116.2829m@116.816s(HOLD), 3.588m@77.596s(FOLLOWING), 3.377m@77.396s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=82.86 m/s² (8.4494g) @164.632s state=LANDING; 접지(disarm−5s) 제외 시 4.548 m/s² (0.4638g) @77.172s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3514 m/s² (0.2398g), 14.4548→5.0576 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -99.5036° → 정렬 13.024s 소요, 최대 99.5235°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.7655m vs transition_alt 50.0m → -0.469% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.6569m → 최저 50.6569m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.0218m, 평균편차 -0.4151m, 최대 \|편차\| 1.953m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 4.2m 평균 1.3529m (부호 -0.5~4.2m, n=17) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 12건/8종, mavros.log 50건/13종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.0554 |
| CLIMBING | +1.1s | 30.9868 |
| TRANSITION_FW | +32.0s | 18.7995 |
| STREAMING | +50.8s | 0.1082 |
| FOLLOWING | +51.0s | 37.0115 |
| TRANSITION_MC | +88.0s | 5.7807 |
| HOLD | +93.7s | 12.6999 |
| LANDING | +106.4s | 46.0171 |
| DONE | +152.5s | 4.9749 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.296 | 69.088 |
| 1 | TRANS_TO_FW | 74.384 | 2.504 |
| 4 | FW | 76.888 | 34.032 |
| 2 | TRANS_TO_MC | 110.92 | 5.472 |
| 3 | MC | 116.392 | 52.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785095386.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785095386.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785095386.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785095386.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785095386.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1123, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785095386.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1125, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | ≈1785095228.1 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.451m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785095386.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785095201.2 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785095202.3 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785095200.3 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785095198.5 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785095205.4 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785095201.2 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785095214.0 |  | TM: RTT too high for timesync: 1866.00 ms. |
| mavros.log | ERROR | 3 | 1785095253.9 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785095208.0 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785095208.1 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785095214.9 |  | PR: Failed to get parameter type: NAV_DLL_ACT |
| mavros.log | WARN | 1 | 1785095214.9 |  | PR: request param #335 timeout, retries left 2, and 486 params still missing |
| mavros.log | WARN | 1 | 1785095387.7 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
