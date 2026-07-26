# C10 — 판정

- 목적: entry_mode=mid_flight (ENTRY 상태) — 거의 미검증 경로
- 실행: 2026-07-26T20:53:40.934974+00:00 ~ 2026-07-26T21:02:16.224154+00:00 (경과 515.3s)
- 종료: `timeout` (exit=2)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local entry_mode:=mid_flight`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 20_53_53.ulg (meta.json 기록: 20_53_53.ulg)
- 요약: FAIL 6, NULL 2, PASS 4, WARN 1

- 시각 정렬: `wall = 1.01342 x ulog + 1785099227.744` (앵커 3개, 최대 잔차 0.090s). 시뮬 클록이 벽시계보다 +1.3% 빠름/느림 — 상수 오프셋만 쓰면 0.67s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **FAIL** | 관측 상태: ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING; 종료사유=timeout | DONE 상태 도달 |
| disarm 확인 | **FAIL** | 로그 끝까지 disarm 되지 않음 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **FAIL** | seq=[3, 1, 4], 정천이 2.416s / 역천이 Nones | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 1건 / 전체 위반 1건 / 샘플 252개, 최대 219.2044m (1096.0222 m/s). 경계 최대: 219.2044m@74.124s(STREAMING). 스트림 재개 갭 1건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=5.7246 m/s² (0.5837g) @74.16s state=FOLLOWING; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **NULL** | 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다 | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -95.8538° → 정렬 13.736s 소요, 최대 96.1745°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.556m vs transition_alt 50.0m → -0.888% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 51.7767m → 최저 51.7767m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0167m, 평균편차 6.7289m, 최대 \|편차\| 21.1355m | ±3m |
| FW cte | **NULL** | node.log 에 'FOLLOWING tick= ... cte=' 샘플이 없음 (FOLLOWING 미진입이거나 20틱 미만 체류) | 직선 ≤ 2m |
| 경고/타임아웃 | **WARN** | node.log 12건/8종, mavros.log 44건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.3943 |
| CLIMBING | +0.4s | 27.8921 |
| TRANSITION_FW | +28.3s | 18.6006 |
| STREAMING | +46.9s | 0.1104 |
| FOLLOWING | +47.0s | 432.6743 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.408 | 65.968 |
| 1 | TRANS_TO_FW | 71.376 | 2.416 |
| 4 | FW | 73.792 | 434.02 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785099735.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785099735.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785099735.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785099735.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785099735.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1134, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785099735.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1136, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785099255.9 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | ≈1785099735.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 8 | 1785099233.4 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | ERROR | 8 | 1785099289.8 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 6 | 1785099234.5 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785099232.7 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785099230.7 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785099238.1 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785099247.4 |  | TM: RTT too high for timesync: 1949.30 ms. |
| mavros.log | WARN | 2 | 1785099233.7 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 1 | 1785099241.1 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785099243.6 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785099248.4 |  | PR: request param #282 timeout, retries left 2, and 613 params still missing |
| mavros.log | WARN | 1 | 1785099736.6 |  | UAS Executor terminated |

## 미산출 지표 (null)

- **역천이 감속률**: 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다
- **FW cte**: node.log 에 'FOLLOWING tick= ... cte=' 샘플이 없음 (FOLLOWING 미진입이거나 20틱 미만 체류)

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
