# C7 — 판정

- 목적: 조종사 인계(PILOT_TAKEOVER) — FOLLOWING 중 POSCTL 강제 주입, 노드가 손을 떼는가
- 실행: 2026-07-26T21:41:52.265544+00:00 ~ 2026-07-26T21:44:46.764518+00:00 (경과 171.2s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 21_42_04.ulg (meta.json 기록: 21_42_04.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.03457 x ulog + 1785102118.200` (앵커 4개, 최대 잔차 0.270s). 시뮬 클록이 벽시계보다 +3.5% 빠름/느림 — 상수 오프셋만 쓰면 2.53s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 129.6969s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 31.128s → disarmed 155.156s (비행 124.028s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.536s / 역천이 5.048s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 4건 / 전체 위반 77건 / 샘플 632개, 최대 218.5243m (1050.5977 m/s). 경계 최대: 218.5243m@80.464s(FOLLOWING), 3.8482m@81.056s(FOLLOWING), 3.4194m@80.664s(FOLLOWING). 스트림 재개 갭 4건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=62.3149 m/s² (6.3544g) @152.02s state=LANDING; 접지(disarm−5s) 제외 시 7.9156 m/s² (0.8072g) @80.9s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.4499 m/s² (0.2498g), 13.6358→5.0405 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -95.1515° → 정렬 14.508s 소요, 최대 95.1967°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.7589m vs transition_alt 50.0m → -0.4823% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.16m → 최저 49.158m (손실 0.0019m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9846m, 평균편차 -0.7727m, 최대 \|편차\| 2.0124m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.9m 평균 0.41m (부호 -0.9~0.4m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 32건/8종, mavros.log 50건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.0977 |
| CLIMBING | +1.1s | 29.9987 |
| TRANSITION_FW | +31.1s | 19.5003 |
| STREAMING | +50.6s | 0.1104 |
| FOLLOWING | +50.7s | 20.2113 |
| TRANSITION_MC | +70.9s | 5.2788 |
| HOLD | +76.2s | 8.3007 |
| LANDING | +84.5s | 45.1989 |
| DONE | +129.7s | 6.1242 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.552 | 72.148 |
| 1 | TRANS_TO_FW | 77.7 | 2.536 |
| 4 | FW | 80.236 | 19.352 |
| 2 | TRANS_TO_MC | 99.588 | 5.048 |
| 3 | MC | 104.636 | 51.004 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785102286.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785102286.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785102286.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785102286.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785102286.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1170, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785102286.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1172, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 21 | 1785102148.4 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | ≈1785102286.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785102124.6 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785102125.7 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785102123.5 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785102121.6 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | WARN | 4 | 1785102124.5 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 4 | 1785102129.3 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785102138.4 |  | TM: RTT too high for timesync: 1938.95 ms. |
| mavros.log | ERROR | 3 | 1785102179.9 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785102132.2 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785102136.3 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785102139.3 |  | PR: request param #280 timeout, retries left 2, and 504 params still missing |
| mavros.log | WARN | 1 | 1785102287.0 |  | UAS Executor terminated |

## 장애주입 결과

- `set_mode` spec={"on_state": "FOLLOWING", "delay_s": 8.0, "action": "set_mode", "mode": "POSCTL"} → 발화 +62.506s rc=0

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
