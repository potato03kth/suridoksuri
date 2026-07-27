# R1_a3 — 판정

- 목적: R1 ⑤: A3 L자 정상 완주 회귀 — 타임아웃 4종 오발동 0건 확인
- 실행: 2026-07-27T13:02:30.351097+00:00 ~ 2026-07-27T13:06:19.962749+00:00 (경과 211.7s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0] range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 13_02_42.ulg (meta.json 기록: 13_02_42.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.09800 x ulog + 1785157355.756` (앵커 4개, 최대 잔차 1.171s). 시뮬 클록이 벽시계보다 +9.8% 빠름/느림 — 상수 오프셋만 쓰면 7.55s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 142.4425s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 68.592s → disarmed 197.064s (비행 128.472s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.504s / 역천이 5.04s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 2건 / 전체 위반 102건 / 샘플 841개, 최대 232.3617m (1117.1238 m/s). 경계 최대: 110.224m@145.716s(TRANSITION_MC), 61.6346m@140.336s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=60.8575 m/s² (6.2057g) @193.96s state=LANDING; 접지(disarm−5s) 제외 시 5.2051 m/s² (0.5308g) @116.576s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2546 m/s² (0.2299g), 13.7946→5.0393 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -97.4562° → 정렬 13.64s 소요, 최대 97.4562°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.7017m vs transition_alt 50.0m → -0.5966% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.1353m → 최저 50.1353m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.1216m, 평균편차 -0.7376m, 최대 \|편차\| 5.2134m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 15.0m 평균 2.0462m (부호 -0.5~15.0m, n=13) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 13건/9종, mavros.log 49건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.3522 |
| CLIMBING | +0.4s | 31.2903 |
| TRANSITION_FW | +31.6s | 18.7001 |
| STREAMING | +50.3s | 0.1068 |
| FOLLOWING | +50.4s | 28.7048 |
| TRANSITION_MC | +79.2s | 5.3887 |
| HOLD | +84.5s | 11.0004 |
| LANDING | +95.5s | 46.8993 |
| DONE | +142.4s | 5.7301 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.212 | 108.08 |
| 1 | TRANS_TO_FW | 113.292 | 2.504 |
| 4 | FW | 115.796 | 24.54 |
| 2 | TRANS_TO_MC | 140.336 | 5.04 |
| 3 | MC | 145.376 | 52.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785157579.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785157579.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785157579.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785157579.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785157579.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1199, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785157579.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1201, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785157465.1 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785157429.5 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 5.057m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785157579.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785157362.6 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785157363.7 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785157361.4 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785157359.6 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785157367.1 |  | VER: command plugin service call failed! |
| mavros.log | ERROR | 4 | 1785157418.1 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 3 | 1785157363.3 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 2 | 1785157375.2 |  | TM: RTT too high for timesync: 1554.47 ms. |
| mavros.log | WARN | 1 | 1785157369.5 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785157369.9 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785157580.2 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
