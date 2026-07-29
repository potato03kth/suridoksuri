# C1b — 판정

- 목적: 천이고도 민감도 — 고(120m). 경로는 A1
- 실행: 2026-07-29T02:40:05.158961+00:00 ~ 2026-07-29T02:43:31.562062+00:00 (경과 197.2s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=120.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local range_limit_m:=1200.0`
- 저장소 HEAD: `afce94d`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 02_40_18.ulg (meta.json 기록: 02_40_18.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.04672 x ulog + 1785292811.036` (앵커 4개, 최대 잔차 0.158s). 시뮬 클록이 벽시계보다 +4.7% 빠름/느림 — 상수 오프셋만 쓰면 4.09s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 165.8964s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 25.78s → disarmed 184.104s (비행 158.324s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.6s / 역천이 5.96s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 6건 / 전체 위반 72건 / 샘플 801개, 최대 216.2263m (720.7545 m/s). 경계 최대: 216.2263m@91.42s(FOLLOWING), 78.295m@117.648s(HOLD), 3.9573m@92.02s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=52.9638 m/s² (5.4008g) @180.932s state=LANDING; 접지(disarm−5s) 제외 시 6.2786 m/s² (0.6402g) @110.832s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.7807 m/s² (0.2835g), 15.9678→5.1601 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -98.2285° → 정렬 7.5s 소요, 최대 98.294°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 119.6363m vs transition_alt 120.0m → -0.3031% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 119.9488m → 최저 119.9488m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.9805m, 평균편차 54.0999m, 최대 \|편차\| 70.0995m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 33.2m 평균 3.9818m (부호 -33.2~2.2m, n=11) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 13건/9종, mavros.log 49건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.1986 |
| CLIMBING | +0.2s | 55.4979 |
| TRANSITION_FW | +55.7s | 12.5003 |
| STREAMING | +68.2s | 0.1016 |
| FOLLOWING | +68.3s | 21.2995 |
| TRANSITION_MC | +89.6s | 6.1986 |
| HOLD | +95.8s | 22.8002 |
| LANDING | +118.6s | 47.2996 |
| DONE | +165.9s | 7.2355 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.34 | 83.188 |
| 1 | TRANS_TO_FW | 88.528 | 2.6 |
| 4 | FW | 91.128 | 20.252 |
| 2 | TRANS_TO_MC | 111.38 | 5.96 |
| 3 | MC | 117.34 | 67.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785293011.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785293011.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785293011.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785293011.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785293011.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 17088, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node |
| node.log | ERROR | 1 | ≈1785293011.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 17090, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:= |
| node.log | WARN | 1 | 1785292838.1 |  | home_position AMSL 미수렴(최근 2개: ['0.1', '0.1'], tol=0.5) — 이륙 보류, 수렴 대기 |
| node.log | WARN | 1 | 1785292903.5 |  | ⚠️ 천이고도와 경로고도가 70.0m 어긋난다 (transition_alt=120.0m vs 경로 순항고도, 현재 120.0m → 50.0m). 램프가 계단은 막지만 기체는 이 고도차를 순항 중에 메워야 한다 — 의도한 값인지 확인할 것 |
| node.log | WARN | 1 | ≈1785293011.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785292818.1 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785292819.4 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785292817.6 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785292815.6 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785292822.5 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785292818.5 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785292831.1 |  | TM: RTT too high for timesync: 1857.58 ms. |
| mavros.log | ERROR | 3 | 1785292871.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785292825.2 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785292825.3 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785292832.4 |  | PR: request param #854 timeout, retries left 2, and 64 params still missing |
| mavros.log | WARN | 1 | 1785293011.8 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
