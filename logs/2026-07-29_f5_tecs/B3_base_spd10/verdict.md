# B3 — 판정

- 목적: 직각 코너(90°) — 코너 오버슈트 정량화
- 실행: 2026-07-29T04:24:42.955013+00:00 ~ 2026-07-29T04:27:56.802117+00:00 (경과 184.4s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 250.0,250.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 04_24_56.ulg (meta.json 기록: 04_24_56.ulg)
- 요약: FAIL 2, PASS 9, WARN 2

- 시각 정렬: `wall = 1.05197 x ulog + 1785299089.757` (앵커 4개, 최대 잔차 0.379s). 시뮬 클록이 벽시계보다 +5.2% 빠름/느림 — 상수 오프셋만 쓰면 4.35s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 143.0989s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 36.096s → disarmed 170.908s (비행 134.812s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.46s / 역천이 5.276s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 5건 / 전체 위반 144건 / 샘플 734개, 최대 310.1478m (1033.8261 m/s). 경계 최대: 310.1478m@79.664s(FOLLOWING), 70.4555m@118.04s(HOLD), 3.5594m@80.268s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=61.0016 m/s² (6.2204g) @167.716s state=LANDING; 접지(disarm−5s) 제외 시 6.3315 m/s² (0.6456g) @80.096s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2917 m/s² (0.2337g), 13.8876→5.045 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -103.8691° → 정렬 8.3s 소요, 최대 103.8691°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.7168m vs transition_alt 50.0m → -0.5664% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.3205m → 최저 49.3098m (손실 0.0108m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.1133m, 평균편차 -0.9086m, 최대 \|편차\| 2.4874m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 12.6m 평균 3.9353m (부호 1.1~12.6m, n=17) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 11건/9종, mavros.log 47건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.5002 |
| CLIMBING | +1.5s | 30.8988 |
| TRANSITION_FW | +32.4s | 13.3 |
| STREAMING | +45.7s | 0.1025 |
| FOLLOWING | +45.8s | 34.7987 |
| TRANSITION_MC | +80.6s | 5.4988 |
| HOLD | +86.1s | 12.3004 |
| LANDING | +98.4s | 44.6995 |
| DONE | +143.1s | 4.691 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.336 | 71.532 |
| 1 | TRANS_TO_FW | 76.868 | 2.46 |
| 4 | FW | 79.328 | 33.116 |
| 2 | TRANS_TO_MC | 112.444 | 5.276 |
| 3 | MC | 117.72 | 54.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785299275.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785299275.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 1 | ≈1785299275.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785299275.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785299275.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 37168, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | ERROR | 1 | ≈1785299275.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 37166, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785299161.9 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785299127.2 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 37.445m is large. affine correction guarantees WP passage but curve may be deformed |
| node.log | WARN | 1 | ≈1785299275.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785299096.7 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785299097.8 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785299095.8 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785299093.9 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785299101.0 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785299096.7 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 3 | 1785299149.3 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 2 | 1785299109.7 |  | TM: RTT too high for timesync: 1931.87 ms. |
| mavros.log | WARN | 1 | 1785299103.6 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785299103.8 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785299277.0 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
