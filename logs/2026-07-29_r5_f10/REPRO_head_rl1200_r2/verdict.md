# C1b — 판정

- 목적: 천이고도 민감도 — 고(120m). 경로는 A1
- 실행: 2026-07-29T01:54:45.463520+00:00 ~ 2026-07-29T01:58:22.724456+00:00 (경과 212.4s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=120.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local range_limit_m:=1200.0`
- 저장소 HEAD: `9c5d17f`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 01_54_57.ulg (meta.json 기록: 01_54_57.ulg)
- 요약: FAIL 4, PASS 7, WARN 2

- 시각 정렬: `wall = 1.02491 x ulog + 1785290091.349` (앵커 4개, 최대 잔차 0.451s). 시뮬 클록이 벽시계보다 +2.5% 빠름/느림 — 상수 오프셋만 쓰면 3.26s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 182.6998s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 22.348s → disarmed 199.62s (비행 177.272s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.448s / 역천이 6.628s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 8건 / 전체 위반 95건 / 샘플 897개, 최대 217.9892m (726.6306 m/s). 경계 최대: 217.9892m@87.588s(TRANSITION_FW), 69.8759m@141.256s(HOLD), 3.4968m@88.188s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=84.838 m/s² (8.6511g) @196.528s state=LANDING; 접지(disarm−5s) 제외 시 21.316 m/s² (2.1736g) @125.612s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **FAIL** | 최대 3.3059 m/s² (0.3371g), 13.9945→5.2036 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -96.1419° → 정렬 8.74s 소요, 최대 96.237°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 119.5655m vs transition_alt 120.0m → -0.3621% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 119.7085m → 최저 119.5956m (손실 0.1129m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0005m, 평균편차 26.9113m, 최대 \|편차\| 68.5228m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 62.9m 평균 17.05m (부호 -62.9~28.8m, n=24) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 24건/10종, mavros.log 60건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.6009 |
| CLIMBING | +1.6s | 51.7981 |
| TRANSITION_FW | +53.4s | 14.1004 |
| STREAMING | +67.5s | 0.1017 |
| FOLLOWING | +67.6s | 47.5985 |
| TRANSITION_MC | +115.2s | 6.9 |
| HOLD | +122.1s | 16.5005 |
| LANDING | +138.6s | 44.0997 |
| DONE | +182.7s | 5.9911 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.464 | 79.332 |
| 1 | TRANS_TO_FW | 84.796 | 2.448 |
| 4 | FW | 87.244 | 47.136 |
| 2 | TRANS_TO_MC | 134.38 | 6.628 |
| 3 | MC | 141.008 | 59.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785290302.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785290302.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785290302.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785290302.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785290302.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 3029, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785290302.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 3031, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 11 | 1785290112.8 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785290169.4 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785290178.0 |  | ⚠️ 천이고도와 경로고도가 69.7m 어긋난다 (transition_alt=120.0m vs 경로 순항고도, 현재 119.7m → 50.0m). 램프가 계단은 막지만 기체는 이 고도차를 순항 중에 메워야 한다 — 의도한 값인지 확인할 것 |
| node.log | WARN | 1 | ≈1785290302.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 20 | 1785290097.8 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 16 | 1785290098.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785290097.6 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785290095.6 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785290102.5 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785290098.5 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 3 | 1785290152.0 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 2 | 1785290111.1 |  | TM: RTT too high for timesync: 1716.65 ms. |
| mavros.log | WARN | 1 | 1785290105.4 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785290105.6 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785290303.1 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
