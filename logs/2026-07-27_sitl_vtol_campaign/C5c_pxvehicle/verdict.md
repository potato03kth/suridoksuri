# C5c — 판정

- 목적: 역천이 오버슈트 — d_end_thresh=60
- 실행: 2026-07-26T20:50:46.246265+00:00 ~ 2026-07-26T20:53:23.294474+00:00 (경과 157.0s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local d_end_thresh:=60.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 20_50_58.ulg (meta.json 기록: 20_50_58.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.00276 x ulog + 1785099052.820` (앵커 4개, 최대 잔차 0.043s). 시뮬 클록이 벽시계보다 +0.3% 빠름/느림 — 상수 오프셋만 쓰면 0.20s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 113.2977s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 30.772s → disarmed 143.176s (비행 112.404s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.528s / 역천이 4.924s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 6건 / 전체 위반 71건 / 샘플 571개, 최대 60.2215m (301.1075 m/s). 경계 최대: 60.2215m@98.04s(HOLD), 4.0383m@91.884s(FOLLOWING), 3.7199m@78.004s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=64.663 m/s² (6.5938g) @139.976s state=LANDING; 접지(disarm−5s) 제외 시 4.824 m/s² (0.4919g) @77.652s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2539 m/s² (0.2298g), 13.7768→5.0589 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -97.9938° → 정렬 13.728s 소요, 최대 98.4622°, tol 진입 후 재증가 0.0171 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.4471m vs transition_alt 50.0m → -1.1058% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.5036m → 최저 50.5036m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.0171m, 평균편차 -0.3473m, 최대 \|편차\| 2.2585m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.2m 평균 0.5375m (부호 -1.2~-0.2m, n=8) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 16건/8종, mavros.log 47건/13종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.0037 |
| CLIMBING | +1.0s | 26.9939 |
| TRANSITION_FW | +28.0s | 18.3001 |
| STREAMING | +46.3s | 0.1113 |
| FOLLOWING | +46.4s | 15.7126 |
| TRANSITION_MC | +62.1s | 5.1766 |
| HOLD | +67.3s | 2.6997 |
| LANDING | +70.0s | 43.2998 |
| DONE | +113.3s | 5.7885 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.24 | 69.156 |
| 1 | TRANS_TO_FW | 74.396 | 2.528 |
| 4 | FW | 76.924 | 15.844 |
| 2 | TRANS_TO_MC | 92.768 | 4.924 |
| 3 | MC | 97.692 | 46.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785099202.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785099202.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785099202.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785099202.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785099202.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1137, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785099202.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1135, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | WARN | 6 | 1785099083.1 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | ≈1785099202.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785099058.7 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785099059.6 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785099058.0 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785099056.0 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785099063.2 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785099059.0 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 2 | 1785099072.5 |  | TM: RTT too high for timesync: 1604.50 ms. |
| mavros.log | WARN | 1 | 1785099066.2 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785099067.3 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785099073.4 |  | PR: request param #314 timeout, retries left 2, and 552 params still missing |
| mavros.log | WARN | 1 | 1785099073.9 |  | PR: Failed to get parameter type: NAV_DLL_ACT |
| mavros.log | ERROR | 1 | 1785099125.6 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785099203.5 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
