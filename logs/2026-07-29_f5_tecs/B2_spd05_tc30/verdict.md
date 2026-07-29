# B2 — 판정

- 목적: 완만 곡선 4WP(30°급 꺾임) — eta3 NR 경로(_trapz 수정) 실행 검증
- 실행: 2026-07-29T04:01:32.213933+00:00 ~ 2026-07-29T04:04:39.472164+00:00 (경과 179.0s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 150.0,0.0,50.0, 300.0,80.0,50.0, 450.0,200.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 04_01_43.ulg (meta.json 기록: 04_01_43.ulg)
- 요약: FAIL 2, PASS 9, WARN 2

- 시각 정렬: `wall = 1.04862 x ulog + 1785297697.726` (앵커 4개, 최대 잔차 0.406s). 시뮬 클록이 벽시계보다 +4.9% 빠름/느림 — 상수 오프셋만 쓰면 4.06s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 142.7994s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 31.36s → disarmed 166.32s (비행 134.96s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.568s / 역천이 4.992s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 6건 / 전체 위반 151건 / 샘플 715개, 최대 419.2598m (1379.1439 m/s). 경계 최대: 419.2598m@73.432s(FOLLOWING), 70.4967m@112.992s(HOLD), 3.7572m@73.84s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=49.3815 m/s² (5.0355g) @163.268s state=LANDING; 접지(disarm−5s) 제외 시 10.9738 m/s² (1.119g) @112.956s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3499 m/s² (0.2396g), 13.7482→5.0503 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -94.2615° → 정렬 7.668s 소요, 최대 94.5831°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.7017m vs transition_alt 50.0m → -0.5967% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.0459m → 최저 50.0459m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.0267m, 평균편차 -0.1702m, 최대 \|편차\| 1.5252m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 7.2m 평균 3.0278m (부호 0.5~7.2m, n=18) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 12건/9종, mavros.log 49건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.2011 |
| CLIMBING | +1.2s | 30.1982 |
| TRANSITION_FW | +31.4s | 12.8997 |
| STREAMING | +44.3s | 0.1018 |
| FOLLOWING | +44.4s | 35.8991 |
| TRANSITION_MC | +80.3s | 5.2 |
| HOLD | +85.5s | 11.9992 |
| LANDING | +97.5s | 45.3001 |
| DONE | +142.8s | 4.9747 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.316 | 65.22 |
| 1 | TRANS_TO_FW | 70.536 | 2.568 |
| 4 | FW | 73.104 | 34.592 |
| 2 | TRANS_TO_MC | 107.696 | 4.992 |
| 3 | MC | 112.688 | 54.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785297878.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785297878.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785297878.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785297878.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785297878.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 24411, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | ERROR | 1 | ≈1785297878.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 24409, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785297763.7 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785297730.0 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.450m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785297878.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785297704.2 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785297705.3 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785297703.2 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785297701.4 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785297708.6 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785297704.1 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785297717.5 |  | TM: RTT too high for timesync: 2106.25 ms. |
| mavros.log | ERROR | 3 | 1785297759.1 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785297710.5 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785297711.5 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785297718.6 |  | PR: request param #1287 timeout, retries left 2, and 12 params still missing |
| mavros.log | WARN | 1 | 1785297879.7 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
