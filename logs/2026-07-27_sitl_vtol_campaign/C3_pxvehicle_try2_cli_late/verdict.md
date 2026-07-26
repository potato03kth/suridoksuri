# C3 — 판정

- 목적: 천이 중 OFFBOARD 강제 이탈 — AUTO.LOITER 주입 후 재요청 복구 확인
- 실행: 2026-07-26T21:26:04.823859+00:00 ~ 2026-07-26T21:29:00.492250+00:00 (경과 175.6s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 21_26_16.ulg (meta.json 기록: 21_26_16.ulg)
- 요약: FAIL 3, PASS 9, WARN 1

- 시각 정렬: `wall = 1.07195 x ulog + 1785101170.080` (앵커 4개, 최대 잔차 0.183s). 시뮬 클록이 벽시계보다 +7.2% 빠름/느림 — 상수 오프셋만 쓰면 4.88s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 129.286s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 32.82s → disarmed 153.972s (비행 121.152s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.596s / 역천이 5.06s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 9건 / 전체 위반 74건 / 샘플 627개, 최대 215.8475m (1037.7282 m/s). 경계 최대: 215.8475m@79.324s(TRANSITION_FW), 110.7344m@103.644s(HOLD), 61.8644m@98.12s(TRANSITION_MC). 스트림 재개 갭 4건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=44.2325 m/s² (4.5105g) @150.86s state=LANDING; 접지(disarm−5s) 제외 시 9.2297 m/s² (0.9412g) @79.552s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.4825 m/s² (0.2531g), 13.6921→5.0503 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -96.0745° → 정렬 13.616s 소요, 최대 96.0745°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.4363m vs transition_alt 50.0m → -1.1275% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.4278m → 최저 49.427m (손실 0.0008m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0159m, 평균편차 -0.3752m, 최대 \|편차\| 3.122m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.5m 평균 0.68m (부호 -0.7~1.5m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 22건/8종, mavros.log 50건/13종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.5884 |
| CLIMBING | +0.6s | 29.5979 |
| TRANSITION_FW | +30.2s | 19.9008 |
| STREAMING | +50.1s | 0.122 |
| FOLLOWING | +50.2s | 19.9364 |
| TRANSITION_MC | +70.1s | 5.6407 |
| HOLD | +75.8s | 9.002 |
| LANDING | +84.8s | 44.4978 |
| DONE | +129.3s | 6.105 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.4 | 70.988 |
| 1 | TRANS_TO_FW | 76.388 | 2.596 |
| 4 | FW | 78.984 | 19.14 |
| 2 | TRANS_TO_MC | 98.124 | 5.06 |
| 3 | MC | 103.184 | 51.016 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785101340.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785101340.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785101340.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785101340.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785101340.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1114, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785101340.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1116, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 11 | 1785101257.1 |  | FOLLOWING 중 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785101340.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785101177.1 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785101178.2 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785101176.4 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785101174.4 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | WARN | 4 | 1785101177.4 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 4 | 1785101181.8 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785101191.1 |  | TM: RTT too high for timesync: 2048.41 ms. |
| mavros.log | ERROR | 2 | 1785101235.8 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785101184.8 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785101187.4 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785101192.1 |  | PR: request param #200 timeout, retries left 2, and 710 params still missing |
| mavros.log | WARN | 1 | 1785101195.8 |  | PR: Failed to get parameter type: NAV_DLL_ACT |
| mavros.log | WARN | 1 | 1785101341.1 |  | UAS Executor terminated |

## 장애주입 결과

- `set_mode` spec={"on_log": "MC→FW 천이 명령 요청", "delay_s": 0.0, "action": "set_mode", "mode": "AUTO.LOITER"} → 발화 +50.394s rc=0

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
