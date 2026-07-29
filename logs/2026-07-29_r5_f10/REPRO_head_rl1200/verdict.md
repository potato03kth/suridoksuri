# C1b — 판정

- 목적: 천이고도 민감도 — 고(120m). 경로는 A1
- 실행: 2026-07-29T01:51:01.325219+00:00 ~ 2026-07-29T01:54:40.386527+00:00 (경과 213.3s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=120.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local range_limit_m:=1200.0`
- 저장소 HEAD: `9c5d17f`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 01_51_14.ulg (meta.json 기록: 01_51_14.ulg)
- 요약: FAIL 4, PASS 7, WARN 2

- 시각 정렬: `wall = 1.02382 x ulog + 1785289868.545` (앵커 4개, 최대 잔차 0.206s). 시뮬 클록이 벽시계보다 +2.4% 빠름/느림 — 상수 오프셋만 쓰면 2.60s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 179.5986s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 25.296s → disarmed 200.428s (비행 175.132s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.516s / 역천이 6.0s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 7건 / 전체 위반 79건 / 샘플 887개, 최대 216.7621m (722.5403 m/s). 경계 최대: 216.7621m@91.132s(STREAMING), 3.6822m@91.732s(FOLLOWING), 3.6681m@91.532s(FOLLOWING). 스트림 재개 갭 3건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=40.6162 m/s² (4.1417g) @197.336s state=LANDING; 접지(disarm−5s) 제외 시 21.5871 m/s² (2.2013g) @128.62s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **FAIL** | 최대 3.7056 m/s² (0.3779g), 14.2996→5.6738 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -93.375° → 정렬 8.384s 소요, 최대 93.3768°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 119.4036m vs transition_alt 120.0m → -0.497% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 119.2579m → 최저 119.0677m (손실 0.1902m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.9962m, 평균편차 27.4237m, 최대 \|편차\| 68.6433m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 65.9m 평균 16.9542m (부호 -65.9~33.8m, n=24) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 26건/10종, mavros.log 61건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.4084 |
| CLIMBING | +0.4s | 53.4907 |
| TRANSITION_FW | +53.9s | 13.4998 |
| STREAMING | +67.4s | 0.1103 |
| FOLLOWING | +67.5s | 47.0905 |
| TRANSITION_MC | +114.6s | 6.2002 |
| HOLD | +120.8s | 15.3995 |
| LANDING | +136.2s | 43.3994 |
| DONE | +179.6s | 4.8911 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.18 | 83.072 |
| 1 | TRANS_TO_FW | 88.252 | 2.516 |
| 4 | FW | 90.768 | 46.632 |
| 2 | TRANS_TO_MC | 137.4 | 6.0 |
| 3 | MC | 143.4 | 58.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 5 | ≈1785290078.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 3 | ≈1785290078.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785290078.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785290078.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785290078.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1342, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | ERROR | 1 | ≈1785290078.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1340, exit code 1, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node:= |
| node.log | WARN | 9 | 1785289893.5 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785289951.2 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785289959.1 |  | ⚠️ 천이고도와 경로고도가 69.3m 어긋난다 (transition_alt=120.0m vs 경로 순항고도, 현재 119.3m → 50.0m). 램프가 계단은 막지만 기체는 이 고도차를 순항 중에 메워야 한다 — 의도한 값인지 확인할 것 |
| node.log | WARN | 1 | ≈1785290078.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 20 | 1785289874.3 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 16 | 1785289875.4 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785289873.6 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785289871.7 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785289878.8 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785289874.6 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785289888.1 |  | TM: RTT too high for timesync: 1592.52 ms. |
| mavros.log | ERROR | 3 | 1785289928.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785289881.7 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785289882.0 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785290080.8 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
