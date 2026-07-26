# C7 — 판정

- 목적: 조종사 인계(PILOT_TAKEOVER) — FOLLOWING 중 POSCTL 강제 주입, 노드가 손을 떼는가
- 실행: 2026-07-26T21:49:42.650570+00:00 ~ 2026-07-26T21:52:56.858622+00:00 (경과 190.7s)
- 종료: `range_exceeded` (exit=6)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 21_49_55.ulg (meta.json 기록: 21_49_55.ulg)
- 요약: FAIL 5, NULL 1, PASS 6, WARN 1

- 시각 정렬: `wall = 1.02977 x ulog + 1785102589.294` (앵커 3개, 최대 잔차 0.308s). 시뮬 클록이 벽시계보다 +3.0% 빠름/느림 — 상수 오프셋만 쓰면 1.61s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **FAIL** | 관측 상태: ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → PILOT_TAKEOVER; 종료사유=range_exceeded | DONE 상태 도달 |
| disarm 확인 | **FAIL** | 로그 끝까지 disarm 되지 않음 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → PILOT_TAKEOVER | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **FAIL** | seq=[3, 1, 4], 정천이 2.456s / 역천이 Nones | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 7건 / 전체 위반 39건 / 샘플 284개, 최대 218.0877m (1048.4984 m/s). 경계 최대: 218.0877m@73.3s(FOLLOWING), 3.4966m@73.92s(FOLLOWING), 3.4622m@73.712s(FOLLOWING). 스트림 재개 갭 1건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=5.2219 m/s² (0.5325g) @73.288s state=FOLLOWING; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **NULL** | 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다 | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -96.6313° → 정렬 14.524s 소요, 최대 97.0317°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.4445m vs transition_alt 50.0m → -1.1111% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 51.3378m → 최저 51.3378m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9863m, 평균편차 -0.2396m, 최대 \|편차\| 1.3032m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.0m 평균 0.42m (부호 -1.0~0.1m, n=5) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 12건/8종, mavros.log 40건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.5276 |
| CLIMBING | +1.5s | 27.2968 |
| TRANSITION_FW | +28.8s | 19.7005 |
| STREAMING | +48.5s | 0.1081 |
| FOLLOWING | +48.6s | 8.9916 |
| PILOT_TAKEOVER | +57.6s | 102.018 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.496 | 65.024 |
| 1 | TRANS_TO_FW | 70.52 | 2.456 |
| 4 | FW | 72.976 | 110.004 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785102775.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785102775.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785102775.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785102775.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785102775.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1286, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785102775.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1288, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785102673.7 |  | 조종사 인계 감지 (mode=POSCTL) — 세트포인트 발행 중단, OFFBOARD 재요청 안 함. 기체는 조종사 것. |
| node.log | WARN | 1 | ≈1785102775.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | WARN | 8 | 1785102596.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-0-0-0-0-3-191-1-128-3-191-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785102594.0 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | ERROR | 6 | 1785102595.9 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 4 | 1785102592.0 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785102600.7 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785102596.0 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 3 | 1785102651.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 2 | 1785102609.9 |  | TM: RTT too high for timesync: 1694.06 ms. |
| mavros.log | WARN | 1 | 1785102603.8 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785102605.0 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785102610.9 |  | PR: request param #324 timeout, retries left 2, and 521 params still missing |
| mavros.log | WARN | 1 | 1785102777.3 |  | UAS Executor terminated |

## 장애주입 결과

- `set_mode` spec={"on_state": "FOLLOWING", "delay_s": 8.0, "action": "set_mode", "mode": "POSCTL"} → 발화 +58.787s rc=0

## 미산출 지표 (null)

- **역천이 감속률**: 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
