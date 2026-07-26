# B1 — 판정

- 목적: 장거리 직선 500m — 순항 안정성
- 실행: 2026-07-26T17:46:47.276373+00:00 ~ 2026-07-26T17:49:53.643160+00:00 (경과 172.1s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 500.0,0.0,50.0]`
- 저장소 HEAD: `3b52ac1`
- ulog: 17_46_59.ulg (meta.json 기록: 17_46_59.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.11048 x ulog + 1785088011.495` (앵커 4개, 최대 잔차 0.689s). 시뮬 클록이 벽시계보다 +11.0% 빠름/느림 — 상수 오프셋만 쓰면 8.57s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 151.5994s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 21.876s → disarmed 158.816s (비행 136.94s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.536s / 역천이 5.516s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 7건 / 전체 위반 145건 / 샘플 646개, 최대 418.0049m (2009.6389 m/s). 경계 최대: 418.0049m@69.076s(TRANSITION_FW), 3.7992m@69.684s(FOLLOWING), 3.7283m@69.88s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=48.8838 m/s² (4.9848g) @155.768s state=LANDING; 접지(disarm−5s) 제외 시 5.5952 m/s² (0.5705g) @69.468s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.4106 m/s² (0.2458g), 14.3632→5.0644 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -96.139° → 정렬 14.48s 소요, 최대 96.2178°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.3776m vs transition_alt 50.0m → -1.2448% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.1266m → 최저 50.1016m (손실 0.025m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9457m, 평균편차 -0.5042m, 최대 \|편차\| 2.1799m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.2m 평균 0.6882m (부호 -1.2~-0.4m, n=17) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 14건/8종, mavros.log 47건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.1068 |
| CLIMBING | +1.1s | 29.9923 |
| TRANSITION_FW | +31.1s | 21.7 |
| STREAMING | +52.8s | 0.1076 |
| FOLLOWING | +52.9s | 35.7079 |
| TRANSITION_MC | +88.6s | 5.7849 |
| HOLD | +94.4s | 9.3001 |
| LANDING | +103.7s | 47.8998 |
| DONE | +151.6s | 5.9153 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 6.224 | 60.052 |
| 1 | TRANS_TO_FW | 66.276 | 2.536 |
| 4 | FW | 68.812 | 33.188 |
| 2 | TRANS_TO_MC | 102.0 | 5.516 |
| 3 | MC | 107.516 | 52.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785088193.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785088193.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785088193.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785088193.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785088193.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1092, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785088193.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1094, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 3 | 1785088035.4 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | ≈1785088193.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785088019.5 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785088020.7 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 4 | 1785088017.9 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | WARN | 4 | 1785088019.7 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | ERROR | 4 | 1785088024.1 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785088019.7 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785088032.0 |  | TM: RTT too high for timesync: 1488.97 ms. |
| mavros.log | ERROR | 3 | 1785088072.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785088026.8 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785088027.8 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785088193.8 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
