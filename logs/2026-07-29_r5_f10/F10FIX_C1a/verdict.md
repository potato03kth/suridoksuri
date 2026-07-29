# C1a — 판정

- 목적: 천이고도 민감도 — 저(20m). 경로는 A1
- 실행: 2026-07-29T02:35:44.530104+00:00 ~ 2026-07-29T02:38:19.023121+00:00 (경과 149.5s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=20.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local range_limit_m:=1200.0`
- 저장소 HEAD: `afce94d`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 02_36_02.ulg (meta.json 기록: 02_36_02.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.02931 x ulog + 1785292553.402` (앵커 4개, 최대 잔차 0.325s). 시뮬 클록이 벽시계보다 +2.9% 빠름/느림 — 상수 오프셋만 쓰면 1.36s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 112.4993s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 26.18s → disarmed 134.0s (비행 107.82s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.584s / 역천이 4.92s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 6건 / 전체 위반 74건 / 샘플 544개, 최대 70.6709m (360.566 m/s). 경계 최대: 70.6709m@81.852s(HOLD), 3.9959m@57.836s(FOLLOWING), 3.9236m@57.636s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=43.0545 m/s² (4.3903g) @130.868s state=LANDING; 접지(disarm−5s) 제외 시 9.3921 m/s² (0.9577g) @57.616s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3769 m/s² (0.2424g), 13.5401→5.0032 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -95.9941° → 정렬 8.544s 소요, 최대 96.0029°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 19.3282m vs transition_alt 20.0m → -3.359% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 19.9852m → 최저 19.973m (손실 0.0122m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.9715m, 평균편차 -11.9284m, 최대 \|편차\| 30.2826m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 2.7m 평균 0.91m (부호 0.2~2.7m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 19건/10종, mavros.log 42건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.7015 |
| CLIMBING | +0.7s | 17.6978 |
| TRANSITION_FW | +18.4s | 13.7009 |
| STREAMING | +32.1s | 0.1021 |
| FOLLOWING | +32.2s | 19.4998 |
| TRANSITION_MC | +51.7s | 5.1979 |
| HOLD | +56.9s | 11.2025 |
| LANDING | +68.1s | 44.3969 |
| DONE | +112.5s | 4.6038 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 8.72 | 45.64 |
| 1 | TRANS_TO_FW | 54.36 | 2.584 |
| 4 | FW | 56.944 | 19.664 |
| 2 | TRANS_TO_MC | 76.608 | 4.92 |
| 3 | MC | 81.528 | 53.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785292697.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785292697.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785292697.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785292697.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785292697.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 14459, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node |
| node.log | ERROR | 1 | ≈1785292697.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 14461, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:= |
| node.log | WARN | 6 | 1785292579.7 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785292600.8 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785292609.7 |  | ⚠️ 천이고도와 경로고도가 30.0m 어긋난다 (transition_alt=20.0m vs 경로 순항고도, 현재 20.0m → 50.0m). 램프가 계단은 막지만 기체는 이 고도차를 순항 중에 메워야 한다 — 의도한 값인지 확인할 것 |
| node.log | WARN | 1 | ≈1785292697.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | WARN | 8 | 1785292563.1 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-16-0-128-0-0-0-176-58-127-128-176-58-127-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | ERROR | 8 | 1785292563.1 |  | FCU: EVENT 11464574 with args -0-64-0-0-20-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785292557.5 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785292555.6 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785292566.3 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785292563.4 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 2 | 1785292560.4 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 2 | 1785292576.5 |  | TM: RTT too high for timesync: 3021.23 ms. |
| mavros.log | ERROR | 2 | 1785292616.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785292568.7 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785292577.5 |  | PR: request param #486 timeout, retries left 2, and 286 params still missing |
| mavros.log | WARN | 1 | 1785292699.2 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
