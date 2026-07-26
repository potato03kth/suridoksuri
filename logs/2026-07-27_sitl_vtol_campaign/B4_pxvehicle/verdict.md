# B4 — 판정

- 목적: 예각/U턴(135°) — 선회반경 초과 시 거동
- 실행: 2026-07-26T20:03:50.303579+00:00 ~ 2026-07-26T20:08:15.938375+00:00 (경과 243.7s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 100.0,150.0,50.0]`
- 저장소 HEAD: `3b52ac1`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 20_04_01.ulg (meta.json 기록: 20_04_01.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.07879 x ulog + 1785096238.687` (앵커 4개, 최대 잔차 0.551s). 시뮬 클록이 벽시계보다 +7.9% 빠름/느림 — 상수 오프셋만 쓰면 6.08s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 143.7471s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 99.152s → disarmed 230.252s (비행 131.1s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.5s / 역천이 5.416s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 2건 / 전체 위반 103건 / 샘플 1006개, 최대 151.0472m (564.1981 m/s). 경계 최대: 151.0472m@147.908s(FOLLOWING), 112.8396m@179.064s(HOLD). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=57.9587 m/s² (5.9101g) @227.14s state=LANDING; 접지(disarm−5s) 제외 시 5.625 m/s² (0.5736g) @163.528s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2122 m/s² (0.2256g), 14.1444→5.0386 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -98.5246° → 정렬 13.02s 소요, 최대 98.7587°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 50.0489m vs transition_alt 50.0m → 0.0978% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.1667m → 최저 50.1593m (손실 0.0073m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.3984m, 평균편차 -1.1529m, 최대 \|편차\| 7.4536m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 13.9m 평균 2.0615m (부호 -4.7~13.9m, n=13) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 12건/8종, mavros.log 59건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.658 |
| CLIMBING | +0.7s | 32.5886 |
| TRANSITION_FW | +33.2s | 18.2007 |
| STREAMING | +51.4s | 0.1177 |
| FOLLOWING | +51.6s | 28.6019 |
| TRANSITION_MC | +80.2s | 5.5803 |
| HOLD | +85.7s | 11.6002 |
| LANDING | +97.3s | 46.3997 |
| DONE | +143.7s | 5.0819 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.404 | 139.688 |
| 1 | TRANS_TO_FW | 145.092 | 2.5 |
| 4 | FW | 147.592 | 25.78 |
| 2 | TRANS_TO_MC | 173.372 | 5.416 |
| 3 | MC | 178.788 | 52.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785096494.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785096494.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785096494.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785096494.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785096494.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1144, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785096494.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1146, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | ≈1785096344.6 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.776m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785096494.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 18 | 1785096242.1 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 14 | 1785096243.3 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785096241.1 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785096239.3 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785096246.3 |  | VER: command plugin service call failed! |
| mavros.log | ERROR | 4 | 1785096298.3 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 3 | 1785096242.0 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 2 | 1785096248.6 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 2 | 1785096255.0 |  | TM: RTT too high for timesync: 1685.82 ms. |
| mavros.log | WARN | 1 | 1785096249.1 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785096255.9 |  | PR: request param #297 timeout, retries left 2, and 503 params still missing |
| mavros.log | WARN | 1 | 1785096496.1 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
