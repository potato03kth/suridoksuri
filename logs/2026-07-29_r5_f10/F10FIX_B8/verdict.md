# B8 — 판정

- 목적: 후방 경로(초기 헤딩과 180° 반대) — 헤딩 정렬 P제어 최악조건
- 실행: 2026-07-29T02:50:15.959811+00:00 ~ 2026-07-29T02:52:58.105636+00:00 (경과 157.5s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, -300.0,0.0,50.0] range_limit_m:=1200.0`
- 저장소 HEAD: `afce94d`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 02_50_27.ulg (meta.json 기록: 02_50_27.ulg)
- 요약: FAIL 2, PASS 9, WARN 2

- 시각 정렬: `wall = 1.03166 x ulog + 1785293422.262` (앵커 4개, 최대 잔차 0.392s). 시뮬 클록이 벽시계보다 +3.2% 빠름/느림 — 상수 오프셋만 쓰면 2.04s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 124.199s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 24.932s → disarmed 144.632s (비행 119.7s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.564s / 역천이 4.908s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 8건 / 전체 위반 74건 / 샘플 611개, 최대 216.6247m (712.5814 m/s). 경계 최대: 216.6247m@66.128s(TRANSITION_FW), 70.4976m@90.32s(TRANSITION_MC), 3.8316m@66.728s(FOLLOWING). 스트림 재개 갭 3건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=74.9756 m/s² (7.6454g) @141.46s state=LANDING; 접지(disarm−5s) 제외 시 7.296 m/s² (0.744g) @66.424s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3691 m/s² (0.2416g), 13.6544→5.0601 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 87.2534° → 정렬 7.864s 소요, 최대 87.3725°, tol 진입 후 재증가 0.0159 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.3958m vs transition_alt 50.0m → -1.2084% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.4201m → 최저 49.39m (손실 0.0301m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9819m, 평균편차 -0.6248m, 최대 \|편차\| 1.931m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 4.7m 평균 1.34m (부호 -4.7~0.1m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 10건/7종, mavros.log 48건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.6039 |
| CLIMBING | +1.6s | 28.1956 |
| TRANSITION_FW | +29.8s | 13.2 |
| STREAMING | +43.0s | 0.1022 |
| FOLLOWING | +43.1s | 18.9982 |
| TRANSITION_MC | +62.1s | 6.1996 |
| HOLD | +68.3s | 10.2002 |
| LANDING | +78.5s | 45.6993 |
| DONE | +124.2s | 5.4707 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.324 | 57.92 |
| 1 | TRANS_TO_FW | 63.244 | 2.564 |
| 4 | FW | 65.808 | 19.288 |
| 2 | TRANS_TO_MC | 85.096 | 4.908 |
| 3 | MC | 90.004 | 55.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785293577.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785293577.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785293577.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 1 | ≈1785293577.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 22562, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node |
| node.log | ERROR | 1 | ≈1785293577.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785293577.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 22564, exit code 1, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 1 | ≈1785293577.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785293428.1 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785293429.1 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785293427.1 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785293425.1 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785293432.6 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785293428.1 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785293441.4 |  | TM: RTT too high for timesync: 1989.73 ms. |
| mavros.log | ERROR | 2 | 1785293483.0 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785293435.5 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785293437.8 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785293442.5 |  | PR: request param #494 timeout, retries left 2, and 154 params still missing |
| mavros.log | WARN | 1 | 1785293578.5 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
