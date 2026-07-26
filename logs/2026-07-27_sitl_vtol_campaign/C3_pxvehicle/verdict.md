# C3 — 판정

- 목적: 천이 중 OFFBOARD 강제 이탈 — AUTO.LOITER 주입 후 재요청 복구 확인
- 실행: 2026-07-26T21:59:42.039777+00:00 ~ 2026-07-26T22:02:36.120216+00:00 (경과 170.1s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 21_59_54.ulg (meta.json 기록: 21_59_54.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.03474 x ulog + 1785103187.948` (앵커 4개, 최대 잔차 0.361s). 시뮬 클록이 벽시계보다 +3.5% 빠름/느림 — 상수 오프셋만 쓰면 2.75s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 130.3063s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 30.736s → disarmed 154.952s (비행 124.216s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.524s / 역천이 5.096s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 5건 / 전체 위반 74건 / 샘플 628개, 최대 214.8793m (679.9976 m/s). 경계 최대: 214.8793m@78.704s(FOLLOWING), 110.3889m@103.1s(HOLD), 3.8054m@79.104s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=83.7129 m/s² (8.5363g) @151.848s state=LANDING; 접지(disarm−5s) 제외 시 6.3501 m/s² (0.6475g) @78.864s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.5337 m/s² (0.2584g), 13.6931→5.0148 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -98.6426° → 정렬 13.912s 소요, 최대 99.0497°, tol 진입 후 재증가 0.0188 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.6058m vs transition_alt 50.0m → -0.7884% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.8706m → 최저 49.8706m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.991m, 평균편차 -0.6496m, 최대 \|편차\| 1.9792m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.8m 평균 0.25m (부호 -0.8~0.0m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 21건/8종, mavros.log 50건/13종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.8089 |
| CLIMBING | +1.8s | 28.599 |
| TRANSITION_FW | +30.4s | 18.9982 |
| STREAMING | +49.4s | 0.1118 |
| FOLLOWING | +49.5s | 20.1058 |
| TRANSITION_MC | +69.6s | 5.484 |
| HOLD | +75.1s | 8.6992 |
| LANDING | +83.8s | 46.4994 |
| DONE | +130.3s | 5.29 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.252 | 70.464 |
| 1 | TRANS_TO_FW | 75.716 | 2.524 |
| 4 | FW | 78.24 | 19.452 |
| 2 | TRANS_TO_MC | 97.692 | 5.096 |
| 3 | MC | 102.788 | 53.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785103355.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785103355.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785103355.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785103355.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785103355.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1161, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785103355.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1165, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 10 | 1785103267.3 |  | 천이 중 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785103355.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785103194.3 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785103195.5 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785103193.7 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785103191.7 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | WARN | 4 | 1785103194.7 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 4 | 1785103198.9 |  | VER: command plugin service call failed! |
| mavros.log | ERROR | 3 | 1785103249.0 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 2 | 1785103207.9 |  | TM: RTT too high for timesync: 1886.75 ms. |
| mavros.log | WARN | 1 | 1785103201.7 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785103203.7 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785103208.8 |  | PR: request param #324 timeout, retries left 2, and 548 params still missing |
| mavros.log | WARN | 1 | 1785103210.7 |  | PR: Failed to get parameter type: NAV_DLL_ACT |
| mavros.log | WARN | 1 | 1785103356.5 |  | UAS Executor terminated |

## 장애주입 결과

- `set_mode` spec={"on_log": "MC→FW 천이 명령 요청", "delay_s": 0.0, "action": "set_mode", "mode": "AUTO.LOITER"} → 발화 +48.576s rc=0

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
