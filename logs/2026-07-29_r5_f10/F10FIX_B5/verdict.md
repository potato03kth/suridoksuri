# B5 — 판정

- 목적: 사각 폐곡선(시점≈종점) — 종점 근접 오판 여부
- 실행: 2026-07-29T02:46:36.539828+00:00 ~ 2026-07-29T02:49:50.802584+00:00 (경과 186.5s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0, 0.0,200.0,50.0, 0.0,20.0,50.0]`
- 저장소 HEAD: `afce94d`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 02_46_48.ulg (meta.json 기록: 02_46_48.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.04382 x ulog + 1785293202.427` (앵커 4개, 최대 잔차 0.278s). 시뮬 클록이 벽시계보다 +4.4% 빠름/느림 — 상수 오프셋만 쓰면 3.92s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 154.9993s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 25.736s → disarmed 173.516s (비행 147.78s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.432s / 역천이 5.164s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 3건 / 전체 위반 211건 / 샘플 748개, 최대 70.4857m (352.4286 m/s). 경계 최대: 70.4857m@119.708s(HOLD), 3.3313m@67.684s(FOLLOWING), 3.0792m@67.488s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=58.1453 m/s² (5.9292g) @170.296s state=LANDING; 접지(disarm−5s) 제외 시 14.382 m/s² (1.4666g) @119.68s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2549 m/s² (0.2299g), 14.144→5.0431 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -96.3926° → 정렬 8.456s 소요, 최대 96.6628°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.6081m vs transition_alt 50.0m → -0.7837% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.3649m → 최저 50.3649m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0491m, 평균편차 -0.9527m, 최대 \|편차\| 6.1215m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 13.3m 평균 3.1083m (부호 0.2~13.3m, n=24) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 32건/11종, mavros.log 56건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.2022 |
| CLIMBING | +0.2s | 29.3972 |
| TRANSITION_FW | +29.6s | 13.1001 |
| STREAMING | +42.7s | 0.1033 |
| FOLLOWING | +42.8s | 49.4975 |
| TRANSITION_MC | +92.3s | 5.4 |
| HOLD | +97.7s | 11.1 |
| LANDING | +108.8s | 46.199 |
| DONE | +155.0s | 4.4639 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.332 | 59.26 |
| 1 | TRANS_TO_FW | 64.592 | 2.432 |
| 4 | FW | 67.024 | 47.192 |
| 2 | TRANS_TO_MC | 114.216 | 5.164 |
| 3 | MC | 119.38 | 55.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785293388.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785293388.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785293388.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785293388.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785293388.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 20966, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node |
| node.log | ERROR | 1 | ≈1785293388.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 20968, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:= |
| node.log | WARN | 16 | 1785293227.8 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 3 | 1785293285.6 |  | 세그먼트 인덱스 급변 191→221 (Δ+30, 전체 810) pos=[184.7,15.4] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | 1785293261.1 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785293226.9 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 4.976m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785293388.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 18 | 1785293208.5 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 14 | 1785293209.6 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785293207.6 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785293205.7 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785293214.4 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785293208.6 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 3 | 1785293263.6 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 2 | 1785293222.6 |  | TM: RTT too high for timesync: 1419.58 ms. |
| mavros.log | WARN | 1 | 1785293215.9 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785293217.3 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785293391.0 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
