# R1_c10_entry — 판정

- 목적: R1 ①: C10 재현 — ENTRY 무한대기(실측 432.67s / 5.85km)가 entry_timeout 에 걸려 안전 폴백되는가. 거리 상한은 꺼서(5000) ENTRY 타임아웃만 격리
- 실행: 2026-07-27T12:42:33.067298+00:00 ~ 2026-07-27T12:45:14.959415+00:00 (경과 148.2s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local entry_mode:=mid_flight range_limit_m:=5000.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 12_42_48.ulg (meta.json 기록: 12_42_48.ulg)
- 요약: FAIL 4, NULL 2, PASS 6, WARN 1

- 시각 정렬: `wall = 1.08752 x ulog + 1785156160.127` (앵커 3개, 최대 잔차 0.355s). 시뮬 클록이 벽시계보다 +8.8% 빠름/느림 — 상수 오프셋만 쓰면 4.28s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → OVERRIDE → DONE, 소요 120.9321s | DONE 상태 도달 |
| disarm 확인 | **FAIL** | 로그 끝까지 disarm 되지 않음 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → OVERRIDE → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **FAIL** | seq=[3, 1, 4], 정천이 2.52s / 역천이 Nones | 3→1→4, 4→2→3 |
| setpoint 점프 | **PASS** | 임계 1.5m, 경계±1s 위반 0건 / 전체 위반 0건 / 샘플 232개, 최대 0.6596m (3.0 m/s). 경계 최대: -. 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=7.944 m/s² (0.8101g) @74.104s state=FOLLOWING; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **NULL** | 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다 | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -95.1953° → 정렬 12.708s 소요, 최대 95.3312°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.6894m vs transition_alt 50.0m → -0.6212% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 48.1513m → 최저 48.1326m (손실 0.0188m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0205m, 평균편차 -10.9652m, 최대 \|편차\| 19.285m | ±3m |
| FW cte | **NULL** | node.log 에 'FOLLOWING tick= ... cte=' 샘플이 없음 (FOLLOWING 미진입이거나 20틱 미만 체류) | 직선 ≤ 2m |
| 경고/타임아웃 | **WARN** | node.log 16건/12종, mavros.log 46건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.2333 |
| CLIMBING | +1.2s | 31.9983 |
| TRANSITION_FW | +33.2s | 18.6998 |
| STREAMING | +51.9s | 0.108 |
| FOLLOWING | +52.0s | 66.9954 |
| OVERRIDE | +119.0s | 1.8974 |
| DONE | +120.9s | 4.4003 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 9.448 | 61.456 |
| 1 | TRANS_TO_FW | 70.904 | 2.52 |
| 4 | FW | 73.424 | 67.012 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 1 | 1785156307.0 |  | ENTRY 타임아웃 60s 초과 (체류 60.0s) → 안전 폴백(OVERRIDE) 실행 |
| node.log | ERROR | 2 | ≈1785156313.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785156313.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785156313.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785156313.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785156313.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1112, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785156313.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1115, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785156223.3 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785156307.0 |  | 긴급 수동 전환 실행 → MANUAL 요청 |
| node.log | WARN | 1 | 1785156308.0 |  | 수동 모드(MANUAL) 미진입 (mode=OFFBOARD) -> AUTO.LOITER 안전 폴백 요청 |
| node.log | WARN | 1 | 1785156308.9 |  | 수동/안전 모드 진입 확인 (mode=AUTO.LOITER) -> DONE |
| node.log | WARN | 1 | ≈1785156313.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 11 | 1785156168.6 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 8 | 1785156169.5 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785156165.0 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785156164.3 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785156173.2 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785156170.4 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785156181.4 |  | TM: RTT too high for timesync: 1844.58 ms. |
| mavros.log | WARN | 2 | 1785156167.6 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | ERROR | 2 | 1785156225.0 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785156175.5 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785156186.1 |  | PR: request param #1165 timeout, retries left 2, and 10 params still missing |
| mavros.log | WARN | 1 | 1785156315.2 |  | UAS Executor terminated |

## 미산출 지표 (null)

- **역천이 감속률**: 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다
- **FW cte**: node.log 에 'FOLLOWING tick= ... cte=' 샘플이 없음 (FOLLOWING 미진입이거나 20틱 미만 체류)

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
