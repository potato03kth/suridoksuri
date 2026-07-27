# R1_a1 — 판정

- 목적: R1 ⑤: A1 정상 완주 회귀 — 타임아웃 4종 오발동 0건 확인
- 실행: 2026-07-27T12:59:24.583542+00:00 ~ 2026-07-27T13:02:20.217929+00:00 (경과 159.0s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 12_59_36.ulg (meta.json 기록: 12_59_36.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.13090 x ulog + 1785157169.128` (앵커 4개, 최대 잔차 2.011s). 시뮬 클록이 벽시계보다 +13.1% 빠름/느림 — 상수 오프셋만 쓰면 10.88s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 137.3219s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 22.956s → disarmed 145.796s (비행 122.84s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.48s / 역천이 4.864s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 3건 / 전체 위반 76건 / 샘플 584개, 최대 216.5334m (644.4447 m/s). 경계 최대: 216.5334m@70.968s(FOLLOWING), 108.3796m@95.392s(HOLD), 3.4997m@71.16s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=41.8201 m/s² (4.2645g) @142.74s state=DONE; 접지(disarm−5s) 제외 시 7.1529 m/s² (0.7294g) @71.256s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2107 m/s² (0.2254g), 13.5571→5.0386 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -93.2185° → 정렬 15.672s 소요, 최대 93.2185°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.0413m vs transition_alt 50.0m → -1.9173% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.2216m → 최저 49.0733m (손실 0.1484m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.0234m, 평균편차 -0.8903m, 최대 \|편차\| 2.0377m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.2m 평균 0.1m (부호 -0.2~0.1m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 11건/8종, mavros.log 47건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 4.9256 |
| CLIMBING | +4.9s | 27.9958 |
| TRANSITION_FW | +32.9s | 22.3005 |
| STREAMING | +55.2s | 0.1089 |
| FOLLOWING | +55.3s | 22.8146 |
| TRANSITION_MC | +78.1s | 5.0776 |
| HOLD | +83.2s | 8.1995 |
| LANDING | +91.4s | 45.8996 |
| DONE | +137.3s | 9.4861 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.228 | 62.808 |
| 1 | TRANS_TO_FW | 68.036 | 2.48 |
| 4 | FW | 70.516 | 19.78 |
| 2 | TRANS_TO_MC | 90.296 | 4.864 |
| 3 | MC | 95.16 | 51.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785157340.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785157340.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785157340.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785157340.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785157340.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1134, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785157340.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1136, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785157228.2 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785157340.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785157177.0 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785157178.1 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785157175.9 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785157174.0 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785157180.4 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785157176.8 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 3 | 1785157228.7 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 2 | 1785157188.8 |  | TM: RTT too high for timesync: 1126.29 ms. |
| mavros.log | WARN | 1 | 1785157183.1 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785157183.8 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785157340.6 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
