# B4 — 판정

- 목적: 예각/U턴(135°) — 선회반경 초과 시 거동
- 실행: 2026-07-29T04:42:36.761161+00:00 ~ 2026-07-29T04:46:00.699256+00:00 (경과 189.7s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 100.0,150.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 04_42_51.ulg (meta.json 기록: 04_42_51.ulg)
- 요약: FAIL 4, PASS 7, WARN 2

- 시각 정렬: `wall = 1.07522 x ulog + 1785300164.059` (앵커 4개, 최대 잔차 0.383s). 시뮬 클록이 벽시계보다 +7.5% 빠름/느림 — 상수 오프셋만 쓰면 4.88s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 140.8992s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 45.844s → disarmed 176.476s (비행 130.632s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.432s / 역천이 1.568s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 4건 / 전체 위반 84건 / 샘플 783개, 최대 151.3048m (741.6903 m/s). 경계 최대: 151.3048m@87.38s(FOLLOWING), 69.6461m@109.152s(HOLD), 3.4755m@87.776s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=39.8374 m/s² (4.0623g) @173.344s state=LANDING; 접지(disarm−5s) 제외 시 28.2472 m/s² (2.8804g) @106.02s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **FAIL** | 최대 84.4905 m/s² (8.6156g), 65.3941→36.2577 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -92.1114° → 정렬 7.96s 소요, 최대 92.1274°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.6571m vs transition_alt 50.0m → -0.6858% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.7159m → 최저 50.7159m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0377m, 평균편차 2.9126m, 최대 \|편차\| 15.0784m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 58.1m 평균 16.04m (부호 -58.1~4.5m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 14건/10종, mavros.log 62건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.5008 |
| CLIMBING | +0.5s | 30.5983 |
| TRANSITION_FW | +31.1s | 12.8997 |
| STREAMING | +44.0s | 0.102 |
| FOLLOWING | +44.1s | 22.0995 |
| TRANSITION_MC | +66.2s | 1.799 |
| HOLD | +68.0s | 27.8002 |
| LANDING | +95.8s | 45.0996 |
| DONE | +140.9s | 6.2867 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.424 | 79.256 |
| 1 | TRANS_TO_FW | 84.68 | 2.432 |
| 4 | FW | 87.112 | 20.148 |
| 2 | TRANS_TO_MC | 107.26 | 1.568 |
| 3 | MC | 108.828 | 68.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785300360.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785300360.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785300360.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785300360.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785300360.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 46632, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785300360.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 46634, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 1 | 1785300246.5 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785300268.1 |  | 세그먼트 인덱스 급변 231→276 (Δ+45, 전체 476) pos=[228.3,10.8] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785300212.6 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.593m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785300360.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 21 | 1785300171.7 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 18 | 1785300172.8 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785300170.8 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785300166.5 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785300176.5 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785300171.8 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 3 | 1785300223.4 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785300178.9 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785300179.4 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785300183.9 |  | TM: RTT too high for timesync: 598.26 ms. |
| mavros.log | WARN | 1 | 1785300360.9 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
