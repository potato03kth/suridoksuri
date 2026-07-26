# C1a — 판정

- 목적: 천이고도 민감도 — 저(20m). 경로는 A1
- 실행: 2026-07-26T20:34:55.774923+00:00 ~ 2026-07-26T20:37:28.840181+00:00 (경과 153.0s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=20.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 20_35_07.ulg (meta.json 기록: 20_35_07.ulg)
- 요약: FAIL 3, PASS 9, WARN 1

- 시각 정렬: `wall = 1.00637 x ulog + 1785098101.951` (앵커 4개, 최대 잔차 0.220s). 시뮬 클록이 벽시계보다 +0.6% 빠름/느림 — 상수 오프셋만 쓰면 0.56s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 114.5972s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 25.892s → disarmed 139.556s (비행 113.664s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.516s / 역천이 5.132s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 6건 / 전체 위반 74건 / 샘플 550개, 최대 214.0892m (660.7692 m/s). 경계 최대: 214.0892m@63.336s(FOLLOWING), 112.3252m@87.908s(HOLD), 3.8341m@63.728s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=49.1955 m/s² (5.0165g) @136.404s state=LANDING; 접지(disarm−5s) 제외 시 8.4813 m/s² (0.8649g) @63.636s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.4544 m/s² (0.2503g), 13.992→5.0621 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -95.305° → 정렬 14.236s 소요, 최대 95.5058°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 19.6223m vs transition_alt 20.0m → -1.8883% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 21.7431m → 최저 21.7431m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.9675m, 평균편차 -10.1892m, 최대 \|편차\| 27.7153m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.8m 평균 0.48m (부호 -0.8~-0.1m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 17건/8종, mavros.log 46건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.4041 |
| CLIMBING | +1.4s | 17.093 |
| TRANSITION_FW | +18.5s | 19.2006 |
| STREAMING | +37.7s | 0.1133 |
| FOLLOWING | +37.8s | 19.1044 |
| TRANSITION_MC | +56.9s | 5.4822 |
| HOLD | +62.4s | 8.7026 |
| LANDING | +71.1s | 43.4971 |
| DONE | +114.6s | 4.9024 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.344 | 54.972 |
| 1 | TRANS_TO_FW | 60.316 | 2.516 |
| 4 | FW | 62.832 | 19.492 |
| 2 | TRANS_TO_MC | 82.324 | 5.132 |
| 3 | MC | 87.456 | 53.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 5 | ≈1785098247.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 3 | ≈1785098247.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785098247.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785098247.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785098247.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1107, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785098247.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1105, exit code 1, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node |
| node.log | WARN | 2 | 1785098127.6 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | ≈1785098247.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785098108.0 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785098109.1 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785098107.4 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785098105.6 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785098112.7 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785098108.5 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 2 | 1785098121.9 |  | TM: RTT too high for timesync: 1882.28 ms. |
| mavros.log | WARN | 1 | 1785098115.6 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785098116.9 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785098122.9 |  | PR: request param #293 timeout, retries left 2, and 574 params still missing |
| mavros.log | ERROR | 1 | 1785098163.1 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785098249.2 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
