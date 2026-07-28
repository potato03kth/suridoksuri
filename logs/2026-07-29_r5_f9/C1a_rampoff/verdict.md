# C1a — 판정

- 목적: 천이고도 민감도 — 저(20m). 경로는 A1
- 실행: 2026-07-28T17:37:08.329570+00:00 ~ 2026-07-28T17:39:41.433652+00:00 (경과 147.5s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=20.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local alt_slew_rate:=0.0 range_limit_m:=1200.0`
- 저장소 HEAD: `94989b6`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 17_37_21.ulg (meta.json 기록: 17_37_21.ulg)
- 요약: FAIL 3, PASS 9, WARN 1

- 시각 정렬: `wall = 1.04008 x ulog + 1785260234.559` (앵커 4개, 최대 잔차 0.159s). 시뮬 클록이 벽시계보다 +4.0% 빠름/느림 — 상수 오프셋만 쓰면 2.34s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → DONE, 소요 120.2001s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 19.388s → disarmed 134.224s (비행 114.836s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.508s / 역천이 5.1s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 4건 / 전체 위반 78건 / 샘플 524개, 최대 70.4607m (352.3036 m/s). 경계 최대: 70.4607m@81.736s(HOLD), 3.7487m@57.148s(FOLLOWING), 3.2303m@56.792s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=41.7843 m/s² (4.2608g) @131.132s state=HOLD; 접지(disarm−5s) 제외 시 9.0642 m/s² (0.9243g) @57.14s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3952 m/s² (0.2442g), 13.6452→5.0238 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -96.3336° → 정렬 13.616s 소요, 최대 96.5323°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 19.6538m vs transition_alt 20.0m → -1.731% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 20.3947m → 최저 20.3947m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.9996m, 평균편차 -12.0246m, 최대 \|편차\| 30.134m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.6m 평균 0.27m (부호 -0.6~-0.1m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 16건/10종, mavros.log 45건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.3011 |
| CLIMBING | +0.3s | 19.2983 |
| TRANSITION_FW | +19.6s | 18.7 |
| STREAMING | +38.3s | 0.104 |
| FOLLOWING | +38.4s | 20.7974 |
| TRANSITION_MC | +59.2s | 5.299 |
| HOLD | +64.5s | 55.7003 |
| DONE | +120.2s | 5.1932 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.212 | 48.684 |
| 1 | TRANS_TO_FW | 53.896 | 2.508 |
| 4 | FW | 56.404 | 19.936 |
| 2 | TRANS_TO_MC | 76.34 | 5.1 |
| 3 | MC | 81.44 | 53.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785260380.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785260380.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785260380.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 1 | ≈1785260380.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1333, exit code -2, cmd '/root/ws_r5b/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | ERROR | 1 | ≈1785260380.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785260380.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1331, exit code 1, cmd '/root/ws_r5b/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node:= |
| node.log | WARN | 4 | 1785260254.4 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785260276.5 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785260290.5 |  | ⚠️ 천이고도와 경로고도가 29.6m 어긋난다 (transition_alt=20.0m vs 경로 순항고도, 현재 20.4m → 50.0m). 램프가 계단은 막지만 기체는 이 고도차를 순항 중에 메워야 한다 — 의도한 값인지 확인할 것 |
| node.log | WARN | 1 | ≈1785260380.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785260241.5 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785260242.6 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785260240.7 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785260238.8 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785260245.9 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785260241.7 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 2 | 1785260293.3 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785260247.6 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785260248.8 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785260253.7 |  | TM: RTT too high for timesync: 753.09 ms. |
| mavros.log | WARN | 1 | 1785260381.8 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
