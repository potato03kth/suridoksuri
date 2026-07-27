# R2_a1 — 판정

- 목적: R2 회귀: A1 직선 300m — 창 탐색이 단순 직선을 깨지 않는가
- 실행: 2026-07-27T15:24:28.967809+00:00 ~ 2026-07-27T15:27:14.851455+00:00 (경과 163.7s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 15_24_40.ulg (meta.json 기록: 15_24_40.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.01982 x ulog + 1785165874.396` (앵커 4개, 최대 잔차 0.005s). 시뮬 클록이 벽시계보다 +2.0% 빠름/느림 — 상수 오프셋만 쓰면 1.36s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 131.0008s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 22.54s → disarmed 149.88s (비행 127.34s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.468s / 역천이 5.176s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 5건 / 전체 위반 76건 / 샘플 605개, 최대 70.468m (359.5307 m/s). 경계 최대: 70.468m@96.764s(HOLD), 3.6751m@72.308s(FOLLOWING), 3.6676m@72.508s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=43.0873 m/s² (4.3937g) @146.76s state=LANDING; 접지(disarm−5s) 제외 시 8.2677 m/s² (0.8431g) @72.272s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3811 m/s² (0.2428g), 13.7614→5.0466 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -88.7876° → 정렬 13.748s 소요, 최대 88.8269°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5917m vs transition_alt 50.0m → -0.8165% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 48.3384m → 최저 48.2263m (손실 0.1121m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.0174m, 평균편차 -1.0835m, 최대 \|편차\| 2.8795m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.5m 평균 0.3m (부호 -0.1~0.5m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 39건/9종, mavros.log 45건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.2013 |
| CLIMBING | +0.2s | 31.1985 |
| TRANSITION_FW | +31.4s | 18.4997 |
| STREAMING | +49.9s | 0.1031 |
| FOLLOWING | +50.0s | 19.8983 |
| TRANSITION_MC | +69.9s | 5.5005 |
| HOLD | +75.4s | 11.0989 |
| LANDING | +86.5s | 44.5005 |
| DONE | +131.0s | 4.5561 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.288 | 63.724 |
| 1 | TRANS_TO_FW | 69.012 | 2.468 |
| 4 | FW | 71.48 | 19.7 |
| 2 | TRANS_TO_MC | 91.18 | 5.176 |
| 3 | MC | 96.356 | 54.012 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 5 | ≈1785166032.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 3 | ≈1785166032.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785166032.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785166032.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785166032.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1113, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785166032.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1111, exit code 1, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node |
| node.log | WARN | 23 | 1785165895.1 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785165930.9 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785166032.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785165881.0 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785165882.1 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785165880.4 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785165878.4 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785165885.5 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785165881.4 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 2 | 1785165933.1 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785165887.6 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785165888.4 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785165893.3 |  | TM: RTT too high for timesync: 556.60 ms. |
| mavros.log | WARN | 1 | 1785166035.3 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
