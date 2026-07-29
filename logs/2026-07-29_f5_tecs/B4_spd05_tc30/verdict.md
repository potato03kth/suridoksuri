# B4 — 판정

- 목적: 예각/U턴(135°) — 선회반경 초과 시 거동
- 실행: 2026-07-29T03:22:26.213836+00:00 ~ 2026-07-29T03:25:23.251064+00:00 (경과 170.3s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 100.0,150.0,50.0]`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 03_22_37.ulg (meta.json 기록: 03_22_37.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.04412 x ulog + 1785295351.409` (앵커 4개, 최대 잔차 0.237s). 시뮬 클록이 벽시계보다 +4.4% 빠름/느림 — 상수 오프셋만 쓰면 3.05s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 133.2994s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 30.592s → disarmed 157.324s (비행 126.732s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.452s / 역천이 5.416s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 4건 / 전체 위반 102건 / 샘플 667개, 최대 150.9024m (503.0079 m/s). 경계 최대: 150.9024m@72.26s(FOLLOWING), 70.4651m@103.628s(HOLD), 3.4311m@72.66s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=60.9968 m/s² (6.2199g) @154.192s state=LANDING; 접지(disarm−5s) 제외 시 11.4329 m/s² (1.1658g) @103.576s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.167 m/s² (0.221g), 14.3789→5.1555 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -94.1739° → 정렬 8.244s 소요, 최대 94.174°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5771m vs transition_alt 50.0m → -0.8458% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.5122m → 최저 50.5122m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.9973m, 평균편차 -0.7339m, 최대 \|편차\| 5.621m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 13.7m 평균 4.0538m (부호 -3.9~13.7m, n=13) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 18건/11종, mavros.log 47건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.3024 |
| CLIMBING | +0.3s | 29.4968 |
| TRANSITION_FW | +29.8s | 13.1007 |
| STREAMING | +42.9s | 0.1023 |
| FOLLOWING | +43.0s | 27.1983 |
| TRANSITION_MC | +70.2s | 5.6992 |
| HOLD | +75.9s | 12.7004 |
| LANDING | +88.6s | 44.6993 |
| DONE | +133.3s | 5.4181 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.336 | 64.04 |
| 1 | TRANS_TO_FW | 69.376 | 2.452 |
| 4 | FW | 71.828 | 26.016 |
| 2 | TRANS_TO_MC | 97.844 | 5.416 |
| 3 | MC | 103.26 | 55.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785295522.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785295522.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785295522.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 1 | ≈1785295522.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 5104, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=of |
| node.log | ERROR | 1 | ≈1785295522.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785295522.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 5102, exit code 1, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node:=t |
| node.log | WARN | 5 | 1785295382.9 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785295415.3 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785295442.0 |  | 세그먼트 인덱스 급변 218→291 (Δ+73, 전체 476) pos=[213.8,16.5] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785295382.2 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.593m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785295522.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785295358.2 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785295359.3 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 4 | 1785295355.9 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | WARN | 4 | 1785295357.9 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | ERROR | 4 | 1785295362.7 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785295357.9 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785295371.9 |  | TM: RTT too high for timesync: 1689.99 ms. |
| mavros.log | ERROR | 3 | 1785295412.2 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785295365.4 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785295366.3 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785295523.7 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
