# C1b — 판정

- 목적: 천이고도 민감도 — 고(120m). 경로는 A1
- 실행: 2026-07-26T20:39:22.914914+00:00 ~ 2026-07-26T20:42:59.093682+00:00 (경과 216.1s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=120.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 20_39_34.ulg (meta.json 기록: 20_39_34.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.00161 x ulog + 1785098369.323` (앵커 4개, 최대 잔차 0.035s). 시뮬 클록이 벽시계보다 +0.2% 빠름/느림 — 상수 오프셋만 쓰면 0.20s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 175.0985s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 28.12s → disarmed 202.876s (비행 174.756s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.528s / 역천이 6.092s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 5건 / 전체 위반 72건 / 샘플 862개, 최대 123.3584m (616.7919 m/s). 경계 최대: 123.3584m@149.804s(HOLD), 3.7287m@100.54s(FOLLOWING), 3.6884m@100.344s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=84.2134 m/s² (8.5874g) @199.804s state=LANDING; 접지(disarm−5s) 제외 시 20.7011 m/s² (2.1109g) @136.036s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.6072 m/s² (0.2659g), 16.7918→5.7146 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -95.0° → 정렬 14.096s 소요, 최대 95.104°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 119.4861m vs transition_alt 120.0m → -0.4282% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 117.2342m → 최저 116.785m (손실 0.4492m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0112m, 평균편차 27.9557m, 최대 \|편차\| 66.6707m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 69.0m 평균 15.4682m (부호 -69.0~40.4m, n=22) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 20건/8종, mavros.log 57건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.8052 |
| CLIMBING | +0.8s | 52.1929 |
| TRANSITION_FW | +53.0s | 18.6999 |
| STREAMING | +71.7s | 0.107 |
| FOLLOWING | +71.8s | 43.6064 |
| TRANSITION_MC | +115.4s | 6.3869 |
| HOLD | +121.8s | 10.7998 |
| LANDING | +132.6s | 42.5003 |
| DONE | +175.1s | 5.5316 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.38 | 91.76 |
| 1 | TRANS_TO_FW | 97.14 | 2.528 |
| 4 | FW | 99.668 | 43.744 |
| 2 | TRANS_TO_MC | 143.412 | 6.092 |
| 3 | MC | 149.504 | 54.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785098578.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785098578.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785098578.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 1 | ≈1785098578.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1137, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785098578.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785098578.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1135, exit code 1, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node |
| node.log | WARN | 10 | 1785098396.6 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | ≈1785098578.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 17 | 1785098375.1 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 14 | 1785098376.3 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785098374.6 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785098372.6 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785098379.8 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785098375.6 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 3 | 1785098429.8 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 2 | 1785098381.9 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 2 | 1785098389.1 |  | TM: RTT too high for timesync: 1717.23 ms. |
| mavros.log | WARN | 1 | 1785098382.8 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785098390.1 |  | PR: request param #258 timeout, retries left 2, and 579 params still missing |
| mavros.log | WARN | 1 | 1785098579.7 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
