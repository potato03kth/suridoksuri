# B8 — 판정

- 목적: 후방 경로(초기 헤딩과 180° 반대) — 헤딩 정렬 P제어 최악조건
- 실행: 2026-07-27T17:05:33.679900+00:00 ~ 2026-07-27T17:08:18.380005+00:00 (경과 160.2s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, -300.0,0.0,50.0] range_limit_m:=1200.0`
- 저장소 HEAD: `893a5eb`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 17_05_45.ulg (meta.json 기록: 17_05_45.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.03842 x ulog + 1785171938.870` (앵커 4개, 최대 잔차 0.508s). 시뮬 클록이 벽시계보다 +3.8% 빠름/느림 — 상수 오프셋만 쓰면 2.29s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 130.0996s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 22.332s → disarmed 146.896s (비행 124.564s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.452s / 역천이 5.064s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 8건 / 전체 위반 86건 / 샘플 601개, 최대 217.8518m (716.6178 m/s). 경계 최대: 217.8518m@69.244s(TRANSITION_FW), 69.4859m@94.072s(TRANSITION_MC), 3.6126m@69.84s(FOLLOWING). 스트림 재개 갭 4건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=80.8667 m/s² (8.2461g) @143.784s state=LANDING; 접지(disarm−5s) 제외 시 6.2256 m/s² (0.6348g) @94.14s state=TRANSITION_MC | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.4475 m/s² (0.2496g), 13.5607→5.0213 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 80.2077° → 정렬 13.692s 소요, 최대 80.2077°, tol 진입 후 재증가 0.0494 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.4677m vs transition_alt 50.0m → -1.0646% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.1615m → 최저 49.1562m (손실 0.0054m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9549m, 평균편차 -0.862m, 최대 \|편차\| 2.2342m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.4m 평균 0.75m (부호 -1.4~-0.6m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 34건/9종, mavros.log 45건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.5116 |
| CLIMBING | +0.5s | 29.0878 |
| TRANSITION_FW | +29.6s | 19.4001 |
| STREAMING | +49.0s | 0.1016 |
| FOLLOWING | +49.1s | 19.4993 |
| TRANSITION_MC | +68.6s | 6.4993 |
| HOLD | +75.1s | 10.5 |
| LANDING | +85.6s | 44.4999 |
| DONE | +130.1s | 5.4961 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.308 | 61.136 |
| 1 | TRANS_TO_FW | 66.444 | 2.452 |
| 4 | FW | 68.896 | 19.832 |
| 2 | TRANS_TO_MC | 88.728 | 5.064 |
| 3 | MC | 93.792 | 54.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785172097.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785172097.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785172097.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785172097.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785172097.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1169, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785172097.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1171, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 22 | 1785171959.8 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785171993.7 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785172097.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785171945.8 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785171946.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785171945.0 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785171943.0 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785171950.2 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785171946.0 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 2 | 1785171997.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785171952.5 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785171953.1 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785171958.0 |  | TM: RTT too high for timesync: 640.44 ms. |
| mavros.log | WARN | 1 | 1785172098.6 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
