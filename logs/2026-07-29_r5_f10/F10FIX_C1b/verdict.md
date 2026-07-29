# C1b — 판정

- 목적: 천이고도 민감도 — 고(120m). 경로는 A1
- 실행: 2026-07-29T02:22:24.551553+00:00 ~ 2026-07-29T02:25:39.395748+00:00 (경과 188.5s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=120.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local range_limit_m:=1200.0`
- 저장소 HEAD: `afce94d`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 02_22_36.ulg (meta.json 기록: 02_22_36.ulg)
- 요약: FAIL 4, PASS 7, WARN 2

- 시각 정렬: `wall = 1.04645 x ulog + 1785291750.132` (앵커 4개, 최대 잔차 0.399s). 시뮬 클록이 벽시계보다 +4.6% 빠름/느림 — 상수 오프셋만 쓰면 4.36s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 160.5003s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 21.684s → disarmed 175.592s (비행 153.908s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.476s / 역천이 5.644s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 6건 / 전체 위반 95건 / 샘플 786개, 최대 216.8346m (713.2716 m/s). 경계 최대: 216.8346m@86.264s(FOLLOWING), 76.3176m@112.504s(HOLD), 3.5473m@86.66s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=40.8713 m/s² (4.1677g) @172.468s state=LANDING; 접지(disarm−5s) 제외 시 7.6565 m/s² (0.7807g) @106.4s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **FAIL** | 최대 3.7052 m/s² (0.3778g), 17.3317→5.713 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -92.8349° → 정렬 7.352s 소요, 최대 92.9846°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 119.5317m vs transition_alt 120.0m → -0.3902% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 119.8184m → 최저 119.8184m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.9726m, 평균편차 52.8412m, 최대 \|편차\| 69.9411m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 19.9m 평균 2.9909m (부호 -19.9~3.3m, n=11) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 14건/10종, mavros.log 47건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.1005 |
| CLIMBING | +1.1s | 54.0996 |
| TRANSITION_FW | +55.2s | 12.3996 |
| STREAMING | +67.6s | 0.1033 |
| FOLLOWING | +67.7s | 21.4983 |
| TRANSITION_MC | +89.2s | 5.8988 |
| HOLD | +95.1s | 20.0005 |
| LANDING | +115.1s | 45.3997 |
| DONE | +160.5s | 5.2762 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.348 | 78.012 |
| 1 | TRANS_TO_FW | 83.36 | 2.476 |
| 4 | FW | 85.836 | 20.788 |
| 2 | TRANS_TO_MC | 106.624 | 5.644 |
| 3 | MC | 112.268 | 64.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785291938.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785291938.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785291938.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785291938.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785291938.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 5800, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785291938.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 5802, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 1 | 1785291773.4 |  | home_position AMSL 미수렴(최근 2개: ['-0.0', '-0.0'], tol=0.5) — 이륙 보류, 수렴 대기 |
| node.log | WARN | 1 | 1785291829.7 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785291837.3 |  | ⚠️ 천이고도와 경로고도가 69.8m 어긋난다 (transition_alt=120.0m vs 경로 순항고도, 현재 119.8m → 50.0m). 램프가 계단은 막지만 기체는 이 고도차를 순항 중에 메워야 한다 — 의도한 값인지 확인할 것 |
| node.log | WARN | 1 | ≈1785291938.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785291756.7 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785291757.7 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785291755.7 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785291753.8 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785291762.3 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785291756.7 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 3 | 1785291810.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 2 | 1785291770.3 |  | TM: RTT too high for timesync: 1140.46 ms. |
| mavros.log | WARN | 1 | 1785291765.0 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785291765.3 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785291939.6 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
