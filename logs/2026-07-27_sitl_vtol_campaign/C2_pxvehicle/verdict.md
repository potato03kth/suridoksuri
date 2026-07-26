# C2 — 판정

- 목적: 헤딩 정렬 90° 조건 (동쪽 경로)
- 실행: 2026-07-26T20:31:52.911158+00:00 ~ 2026-07-26T20:34:23.950335+00:00 (경과 144.4s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 0.0,300.0,50.0]`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 20_32_04.ulg (meta.json 기록: 20_32_04.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.11634 x ulog + 1785097915.819` (앵커 4개, 최대 잔차 0.378s). 시뮬 클록이 벽시계보다 +11.6% 빠름/느림 — 상수 오프셋만 쓰면 6.74s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 119.7766s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 19.9s → disarmed 131.436s (비행 111.536s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.528s / 역천이 5.104s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 4건 / 전체 위반 70건 / 샘플 566개, 최대 216.9777m (713.7424 m/s). 경계 최대: 216.9777m@56.572s(FOLLOWING), 3.068m@56.94s(FOLLOWING), 2.9657m@57.14s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=56.2569 m/s² (5.7366g) @128.3s state=DONE; 접지(disarm−5s) 제외 시 6.7137 m/s² (0.6846g) @59.956s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2947 m/s² (0.234g), 14.273→5.3251 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -0.6299° → 정렬 0.0s 소요, 최대 18.673°, tol 진입 후 재증가 0.3149 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.6485m vs transition_alt 50.0m → -0.703% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.9704m → 최저 49.9704m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0161m, 평균편차 -0.9978m, 최대 \|편차\| 6.7607m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 19.6m 평균 6.64m (부호 -19.6~-0.1m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 11건/7종, mavros.log 45건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.784 |
| CLIMBING | +1.8s | 31.3922 |
| TRANSITION_FW | +33.2s | 7.5997 |
| STREAMING | +40.8s | 0.1057 |
| FOLLOWING | +40.9s | 20.6066 |
| TRANSITION_MC | +61.5s | 5.2904 |
| HOLD | +66.8s | 8.6982 |
| LANDING | +75.5s | 44.2998 |
| DONE | +119.8s | 5.9992 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.312 | 48.36 |
| 1 | TRANS_TO_FW | 53.672 | 2.528 |
| 4 | FW | 56.2 | 18.52 |
| 2 | TRANS_TO_MC | 74.72 | 5.104 |
| 3 | MC | 79.824 | 52.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785098063.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785098063.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785098063.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785098063.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785098063.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1109, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785098063.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1111, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | ≈1785098063.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785097924.8 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785097925.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785097923.9 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785097922.1 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785097928.9 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785097924.9 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 2 | 1785097976.2 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785097931.0 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785097931.6 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785097936.4 |  | TM: RTT too high for timesync: 803.43 ms. |
| mavros.log | WARN | 1 | 1785098064.2 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
