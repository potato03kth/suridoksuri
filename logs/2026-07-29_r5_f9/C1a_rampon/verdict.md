# C1a — 판정

- 목적: 천이고도 민감도 — 저(20m). 경로는 A1
- 실행: 2026-07-28T17:49:36.795246+00:00 ~ 2026-07-28T17:52:18.959862+00:00 (경과 154.0s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=20.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local range_limit_m:=1200.0`
- 저장소 HEAD: `94989b6`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 17_49_50.ulg (meta.json 기록: 17_49_50.ulg)
- 요약: FAIL 3, PASS 9, WARN 1

- 시각 정렬: `wall = 1.02824 x ulog + 1785260984.656` (앵커 4개, 최대 잔차 0.365s). 시뮬 클록이 벽시계보다 +2.8% 빠름/느림 — 상수 오프셋만 쓰면 1.45s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → DONE, 소요 120.5992s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 25.26s → disarmed 141.096s (비행 115.836s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.516s / 역천이 5.224s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 8건 / 전체 위반 77건 / 샘플 560개, 최대 216.0674m (710.748 m/s). 경계 최대: 216.0674m@62.544s(TRANSITION_FW), 70.4787m@87.36s(TRANSITION_MC), 63.7019m@81.996s(TRANSITION_MC). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=45.1602 m/s² (4.6051g) @137.976s state=HOLD; 접지(disarm−5s) 제외 시 9.1907 m/s² (0.9372g) @62.904s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.545 m/s² (0.2595g), 13.6733→5.0112 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -93.219° → 정렬 14.152s 소요, 최대 93.2469°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 19.3679m vs transition_alt 20.0m → -3.1606% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 19.3788m → 최저 19.372m (손실 0.0068m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0258m, 평균편차 -12.4884m, 최대 \|편차\| 31.0516m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.3m 평균 0.12m (부호 -0.3~0.2m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 35건/11종, mavros.log 49건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.208 |
| CLIMBING | +0.2s | 18.5913 |
| TRANSITION_FW | +18.8s | 19.6998 |
| STREAMING | +38.5s | 0.1102 |
| FOLLOWING | +38.6s | 19.3932 |
| TRANSITION_MC | +58.0s | 6.7973 |
| HOLD | +64.8s | 55.7995 |
| DONE | +120.6s | 6.8997 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 6.392 | 53.256 |
| 1 | TRANS_TO_FW | 59.648 | 2.516 |
| 4 | FW | 62.164 | 19.736 |
| 2 | TRANS_TO_MC | 81.9 | 5.224 |
| 3 | MC | 87.124 | 54.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785261138.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785261138.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785261138.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785261138.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785261138.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1499, exit code -2, cmd '/root/ws_r5b/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785261138.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1502, exit code -2, cmd '/root/ws_r5b/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 21 | 1785261008.4 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785261010.6 |  | home_position AMSL 미수렴(최근 2개: ['0.1', '0.1'], tol=0.5) — 이륙 보류, 수렴 대기 |
| node.log | WARN | 1 | 1785261031.4 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785261046.3 |  | ⚠️ 천이고도와 경로고도가 30.6m 어긋난다 (transition_alt=20.0m vs 경로 순항고도, 현재 19.4m → 50.0m). 램프가 계단은 막지만 기체는 이 고도차를 순항 중에 메워야 한다 — 의도한 값인지 확인할 것 |
| node.log | WARN | 1 | ≈1785261138.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 14 | 1785260990.3 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785260991.1 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785260989.2 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785260987.3 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785260995.0 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785260990.2 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785261005.0 |  | TM: RTT too high for timesync: 1852.11 ms. |
| mavros.log | ERROR | 2 | 1785261045.3 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785260997.9 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785260999.1 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785261006.3 |  | PR: request param #870 timeout, retries left 2, and 52 params still missing |
| mavros.log | WARN | 1 | 1785261139.2 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
