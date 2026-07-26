# C5b — 판정

- 목적: 역천이 오버슈트 — d_end_thresh=30
- 실행: 2026-07-26T20:47:36.891593+00:00 ~ 2026-07-26T20:50:20.464521+00:00 (경과 163.5s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local d_end_thresh:=30.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 20_47_48.ulg (meta.json 기록: 20_47_48.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.01041 x ulog + 1785098862.809` (앵커 4개, 최대 잔차 0.308s). 시뮬 클록이 벽시계보다 +1.0% 빠름/느림 — 상수 오프셋만 쓰면 0.97s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 121.629s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 29.728s → disarmed 149.784s (비행 120.056s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.512s / 역천이 5.092s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 6건 / 전체 위반 74건 / 샘플 609개, 최대 214.345m (687.0033 m/s). 경계 최대: 214.345m@78.136s(FOLLOWING), 92.033m@101.176s(HOLD), 3.8216m@78.544s(FOLLOWING). 스트림 재개 갭 7건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=69.5642 m/s² (7.0936g) @146.604s state=LANDING; 접지(disarm−5s) 제외 시 9.1955 m/s² (0.9377g) @78.364s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.0764 m/s² (0.2117g), 13.7637→5.0547 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -98.1035° → 정렬 13.544s 소요, 최대 98.115°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5658m vs transition_alt 50.0m → -0.8685% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 47.9566m → 최저 47.9545m (손실 0.0021m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9999m, 평균편차 -0.9898m, 최대 \|편차\| 2.5678m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.1m 평균 0.3222m (부호 -1.1~0.1m, n=9) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 11건/7종, mavros.log 48건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.8321 |
| CLIMBING | +0.8s | 29.6965 |
| TRANSITION_FW | +30.5s | 18.5001 |
| STREAMING | +49.0s | 0.1123 |
| FOLLOWING | +49.1s | 17.6033 |
| TRANSITION_MC | +66.7s | 5.3845 |
| HOLD | +72.1s | 6.3998 |
| LANDING | +78.5s | 43.1004 |
| DONE | +121.6s | 5.8423 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.352 | 69.776 |
| 1 | TRANS_TO_FW | 75.128 | 2.512 |
| 4 | FW | 77.64 | 17.94 |
| 2 | TRANS_TO_MC | 95.58 | 5.092 |
| 3 | MC | 100.672 | 50.004 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785099020.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785099020.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785099020.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785099020.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785099020.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1133, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785099020.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1135, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | ≈1785099020.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785098868.9 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785098870.3 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785098868.6 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785098866.6 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785098873.8 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785098869.6 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 2 | 1785098877.2 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 2 | 1785098883.1 |  | TM: RTT too high for timesync: 1808.64 ms. |
| mavros.log | ERROR | 2 | 1785098926.3 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785098876.8 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785098884.0 |  | PR: request param #296 timeout, retries left 2, and 583 params still missing |
| mavros.log | WARN | 1 | 1785099020.9 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
