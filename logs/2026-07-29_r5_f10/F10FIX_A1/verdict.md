# A1 — 판정

- 목적: SITL-4 직선 300m 재현 (3주치 변경 후 회귀 여부). waypoint_frame=local(SITL-4 당시 동작)
- 실행: 2026-07-29T02:28:21.857090+00:00 ~ 2026-07-29T02:30:37.749212+00:00 (경과 131.4s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local range_limit_m:=1200.0`
- 저장소 HEAD: `afce94d`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 02_28_34.ulg (meta.json 기록: 02_28_34.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.04344 x ulog + 1785292107.839` (앵커 4개, 최대 잔차 0.488s). 시뮬 클록이 벽시계보다 +4.3% 빠름/느림 — 상수 오프셋만 쓰면 2.43s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 96.9989s | DONE 상태 도달 |
| disarm 확인 | **FAIL** | 로그 끝까지 disarm 되지 않음 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.484s / 역천이 5.156s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 11건 / 전체 위반 84건 / 샘플 448개, 최대 219.5183m (1119.9914 m/s). 경계 최대: 219.5183m@67.316s(TRANSITION_FW), 69.6938m@92.108s(HOLD), 63.1676m@86.78s(TRANSITION_MC). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=6.5338 m/s² (0.6663g) @67.828s state=FOLLOWING; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2541 m/s² (0.2299g), 13.689→5.0373 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -98.1395° → 정렬 8.004s 소요, 최대 98.1395°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.6192m vs transition_alt 50.0m → -0.7616% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.5118m → 최저 49.4875m (손실 0.0243m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9686m, 평균편차 -0.8808m, 최대 \|편차\| 1.9946m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 2.1m 평균 0.65m (부호 0.0~2.1m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 24건/9종, mavros.log 43건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.7006 |
| CLIMBING | +0.7s | 29.9992 |
| TRANSITION_FW | +30.7s | 13.5001 |
| STREAMING | +44.2s | 0.1029 |
| FOLLOWING | +44.3s | 19.3974 |
| TRANSITION_MC | +63.7s | 5.4996 |
| HOLD | +69.2s | 11.6998 |
| LANDING | +80.9s | 16.0994 |
| DONE | +97.0s | 5.451 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.344 | 59.296 |
| 1 | TRANS_TO_FW | 64.64 | 2.484 |
| 4 | FW | 67.124 | 19.56 |
| 2 | TRANS_TO_MC | 86.684 | 5.156 |
| 3 | MC | 91.84 | 18.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785292236.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785292236.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785292236.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 1 | ≈1785292236.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785292236.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 8732, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | ERROR | 1 | ≈1785292236.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 8730, exit code 1, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node:= |
| node.log | WARN | 13 | 1785292132.9 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785292167.0 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785292236.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 10 | 1785292115.1 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 8 | 1785292116.1 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785292114.0 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785292112.1 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785292119.7 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785292115.0 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 2 | 1785292121.5 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 2 | 1785292128.4 |  | TM: RTT too high for timesync: 1744.32 ms. |
| mavros.log | ERROR | 2 | 1785292168.2 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785292122.6 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785292231.1 |  | CON: Lost connection, HEARTBEAT timed out. |
| mavros.log | WARN | 1 | 1785292238.2 |  | UAS Executor terminated |

## 장애주입 결과

- `probe` spec={"on_state": "FOLLOWING", "delay_s": 3.0, "action": "probe", "topic": "/mavros/state"} → 발화 +48.234s rc=0

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
