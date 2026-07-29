# B3 — 판정

- 목적: 직각 코너(90°) — 코너 오버슈트 정량화
- 실행: 2026-07-29T04:34:55.479506+00:00 ~ 2026-07-29T04:38:07.290336+00:00 (경과 183.9s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 250.0,250.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 04_35_08.ulg (meta.json 기록: 04_35_08.ulg)
- 요약: FAIL 2, PASS 9, WARN 2

- 시각 정렬: `wall = 1.04218 x ulog + 1785299702.012` (앵커 4개, 최대 잔차 0.058s). 시뮬 클록이 벽시계보다 +4.2% 빠름/느림 — 상수 오프셋만 쓰면 3.26s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 144.8988s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 32.56s → disarmed 170.432s (비행 137.872s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.48s / 역천이 4.96s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 5건 / 전체 위반 144건 / 샘플 733개, 최대 310.565m (1021.5952 m/s). 경계 최대: 310.565m@75.992s(FOLLOWING), 70.4827m@114.184s(HOLD), 3.5512m@76.592s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=58.1343 m/s² (5.928g) @167.304s state=LANDING; 접지(disarm−5s) 제외 시 12.4153 m/s² (1.266g) @114.176s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2575 m/s² (0.2302g), 13.858→5.0494 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -101.583° → 정렬 8.132s 소요, 최대 101.583°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.7339m vs transition_alt 50.0m → -0.5323% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.9071m → 최저 49.9071m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.1254m, 평균편차 -0.1986m, 최대 \|편차\| 1.5692m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 15.3m 평균 4.0118m (부호 1.1~15.3m, n=17) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 18건/10종, mavros.log 50건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.7028 |
| CLIMBING | +0.7s | 31.0958 |
| TRANSITION_FW | +31.8s | 13.0999 |
| STREAMING | +44.9s | 0.1014 |
| FOLLOWING | +45.0s | 34.3994 |
| TRANSITION_MC | +79.4s | 5.2999 |
| HOLD | +84.7s | 12.2003 |
| LANDING | +96.9s | 47.9994 |
| DONE | +144.9s | 6.3085 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.488 | 67.704 |
| 1 | TRANS_TO_FW | 73.192 | 2.48 |
| 4 | FW | 75.672 | 33.132 |
| 2 | TRANS_TO_MC | 108.804 | 4.96 |
| 3 | MC | 113.764 | 57.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785299887.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785299887.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785299887.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785299887.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785299887.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 42463, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785299887.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 42465, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 5 | 1785299735.4 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785299769.8 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785299735.3 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 37.445m is large. affine correction guarantees WP passage but curve may be deformed |
| node.log | WARN | 1 | ≈1785299887.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785299708.2 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785299709.2 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785299706.9 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785299704.9 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785299712.7 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785299708.8 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785299721.4 |  | TM: RTT too high for timesync: 1937.46 ms. |
| mavros.log | ERROR | 3 | 1785299763.0 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785299715.3 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785299715.5 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785299722.6 |  | PR: request param #778 timeout, retries left 2, and 88 params still missing |
| mavros.log | WARN | 1 | 1785299887.5 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
