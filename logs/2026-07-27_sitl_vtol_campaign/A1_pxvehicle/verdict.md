# A1 — 판정

- 목적: SITL-4 직선 300m 재현 (3주치 변경 후 회귀 여부). waypoint_frame=local(SITL-4 당시 동작)
- 실행: 2026-07-26T19:21:29.514051+00:00 ~ 2026-07-26T19:24:27.586813+00:00 (경과 162.3s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local`
- 저장소 HEAD: `3b52ac1`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 19_21_43.ulg (meta.json 기록: 19_21_43.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.11746 x ulog + 1785093694.627` (앵커 4개, 최대 잔차 1.400s). 시뮬 클록이 벽시계보다 +11.7% 빠름/느림 — 상수 오프셋만 쓰면 7.03s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 138.3977s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 25.408s → disarmed 149.052s (비행 123.644s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.568s / 역천이 5.232s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 10건 / 전체 위반 72건 / 샘플 610개, 최대 214.1971m (677.8389 m/s). 경계 최대: 214.1971m@73.244s(TRANSITION_FW), 4.9386m@74.44s(FOLLOWING), 3.8901m@73.64s(TRANSITION_FW). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=58.9285 m/s² (6.009g) @145.86s state=LANDING; 접지(disarm−5s) 제외 시 5.9818 m/s² (0.61g) @73.408s state=TRANSITION_FW | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2517 m/s² (0.2296g), 14.1354→5.0502 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -94.004° → 정렬 14.188s 소요, 최대 94.0817°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5484m vs transition_alt 50.0m → -0.9033% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.8708m → 최저 49.8708m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.0104m, 평균편차 -0.4847m, 최대 \|편차\| 2.4178m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.1m 평균 0.29m (부호 -1.1~0.1m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 23건/8종, mavros.log 47건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.4007 |
| CLIMBING | +0.4s | 31.7971 |
| TRANSITION_FW | +32.2s | 22.2999 |
| STREAMING | +54.5s | 0.1071 |
| FOLLOWING | +54.6s | 18.5026 |
| TRANSITION_MC | +73.1s | 5.4911 |
| HOLD | +78.6s | 12.1997 |
| LANDING | +90.8s | 47.5996 |
| DONE | +138.4s | 5.762 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.304 | 64.924 |
| 1 | TRANS_TO_FW | 70.228 | 2.568 |
| 4 | FW | 72.796 | 18.772 |
| 2 | TRANS_TO_MC | 91.568 | 5.232 |
| 3 | MC | 96.8 | 53.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785093867.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785093867.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 1 | ≈1785093867.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785093867.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785093867.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1093, exit code 1, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node |
| node.log | ERROR | 1 | ≈1785093867.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1095, exit code 1, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:= |
| node.log | WARN | 14 | 1785093721.5 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | ≈1785093867.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785093703.4 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785093704.5 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785093702.6 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785093700.7 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785093707.6 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785093703.5 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 3 | 1785093754.9 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 2 | 1785093715.0 |  | TM: RTT too high for timesync: 890.70 ms. |
| mavros.log | WARN | 1 | 1785093710.3 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785093710.8 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785093868.0 |  | UAS Executor terminated |

## 장애주입 결과

- `probe` spec={"on_state": "FOLLOWING", "delay_s": 3.0, "action": "probe", "topic": "/mavros/state"} → 발화 +54.637s rc=0

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
