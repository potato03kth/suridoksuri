# R2_b7 — 판정

- 목적: R2 회귀: B7 단거리 40m — 경로 전장 < lookahead(70m) < 창 폭(150m). 창이 짧은 경로를 깨지 않는가
- 실행: 2026-07-27T14:04:43.572149+00:00 ~ 2026-07-27T14:07:27.405830+00:00 (경과 150.4s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 40.0,0.0,50.0] range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 14_04_55.ulg (meta.json 기록: 14_04_55.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.08899 x ulog + 1785161090.493` (앵커 4개, 최대 잔차 0.369s). 시뮬 클록이 벽시계보다 +8.9% 빠름/느림 — 상수 오프셋만 쓰면 4.44s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 124.3983s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 24.328s → disarmed 136.832s (비행 112.504s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.5s / 역천이 6.524s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 1건 / 전체 위반 1건 / 샘플 531개, 최대 70.5271m (352.6353 m/s). 경계 최대: 70.5271m@78.844s(HOLD). 스트림 재개 갭 3건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=88.6541 m/s² (9.0402g) @133.732s state=LANDING; 접지(disarm−5s) 제외 시 16.5717 m/s² (1.6898g) @78.824s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.6466 m/s² (0.2699g), 17.1194→5.0462 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -93.3309° → 정렬 16.0s 소요, 최대 93.3309°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 48.4268m vs transition_alt 50.0m → -3.1464% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 51.3964m → 최저 51.3878m (손실 0.0086m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.8961m, 평균편차 0.6801m, 최대 \|편차\| 1.3865m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.5m 평균 0.5m (부호 -0.5~-0.5m, n=1) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 12건/8종, mavros.log 48건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.3027 |
| CLIMBING | +1.3s | 26.9949 |
| TRANSITION_FW | +28.3s | 22.4003 |
| STREAMING | +50.7s | 0.11 |
| FOLLOWING | +50.8s | 1.0909 |
| TRANSITION_MC | +51.9s | 6.8991 |
| HOLD | +58.8s | 18.1008 |
| LANDING | +76.9s | 47.4996 |
| DONE | +124.4s | 5.3059 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.312 | 62.592 |
| 1 | TRANS_TO_FW | 67.904 | 2.5 |
| 4 | FW | 70.404 | 1.476 |
| 2 | TRANS_TO_MC | 71.88 | 6.524 |
| 3 | MC | 78.404 | 59.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785161246.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785161246.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785161246.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785161246.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785161246.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1147, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785161246.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1149, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785161150.5 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785161246.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785161095.6 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785161096.7 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785161094.6 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785161092.8 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785161099.8 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785161095.5 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785161108.4 |  | TM: RTT too high for timesync: 2092.86 ms. |
| mavros.log | ERROR | 2 | 1785161151.8 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785161102.5 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785161103.3 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785161109.3 |  | PR: request param #572 timeout, retries left 2, and 331 params still missing |
| mavros.log | WARN | 1 | 1785161247.8 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
