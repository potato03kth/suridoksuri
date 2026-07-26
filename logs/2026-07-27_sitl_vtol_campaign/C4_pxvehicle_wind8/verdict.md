# C4 — 판정

- 목적: 바람 주입 8m/s — A1 + 바람. gz wind 파라미터를 param_set/gz topic 으로 주입
- 실행: 2026-07-26T21:54:01.801211+00:00 ~ 2026-07-26T21:56:47.867107+00:00 (경과 162.9s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 21_54_13.ulg (meta.json 기록: 21_54_13.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.03380 x ulog + 1785102847.455` (앵커 4개, 최대 잔차 0.161s). 시뮬 클록이 벽시계보다 +3.4% 빠름/느림 — 상수 오프셋만 쓰면 2.38s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 124.8172s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 28.436s → disarmed 148.512s (비행 120.076s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.52s / 역천이 4.94s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 6건 / 전체 위반 71건 / 샘플 600개, 최대 216.6831m (694.497 m/s). 경계 최대: 216.6831m@75.116s(FOLLOWING), 113.5548m@97.564s(HOLD), 62.3089m@92.372s(FOLLOWING). 스트림 재개 갭 3건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=54.107 m/s² (5.5174g) @145.464s state=LANDING; 접지(disarm−5s) 제외 시 6.4745 m/s² (0.6602g) @79.136s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2136 m/s² (0.2257g), 14.5939→5.7297 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -93.2105° → 정렬 13.772s 소요, 최대 93.2136°, tol 진입 후 재증가 0.2178 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.6134m vs transition_alt 50.0m → -0.7731% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.4038m → 최저 50.3876m (손실 0.0162m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0117m, 평균편차 -0.3117m, 최대 \|편차\| 3.9939m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 4.0m 평균 1.3444m (부호 -4.0~1.7m, n=9) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 11건/7종, mavros.log 49건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.7188 |
| CLIMBING | +0.7s | 28.3981 |
| TRANSITION_FW | +29.1s | 18.7998 |
| STREAMING | +47.9s | 0.107 |
| FOLLOWING | +48.0s | 18.2186 |
| TRANSITION_MC | +66.2s | 5.2755 |
| HOLD | +71.5s | 9.11 |
| LANDING | +80.6s | 44.1893 |
| DONE | +124.8s | 4.7487 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.344 | 66.864 |
| 1 | TRANS_TO_FW | 72.208 | 2.52 |
| 4 | FW | 74.728 | 17.644 |
| 2 | TRANS_TO_MC | 92.372 | 4.94 |
| 3 | MC | 97.312 | 52.008 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785103006.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785103006.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785103006.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785103006.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785103006.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1198, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785103006.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1200, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | ≈1785103006.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785102853.9 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785102855.0 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785102853.0 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785102851.0 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785102858.5 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785102854.0 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785102867.5 |  | TM: RTT too high for timesync: 2097.31 ms. |
| mavros.log | WARN | 2 | 1785102860.4 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | ERROR | 2 | 1785102908.6 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785102861.4 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785102869.0 |  | PR: request param #1249 timeout, retries left 2, and 57 params still missing |
| mavros.log | WARN | 1 | 1785103008.3 |  | UAS Executor terminated |

## 장애주입 결과

- `probe` spec={"at_s": 5.0, "action": "probe", "topic": "/mavros/state"} → 발화 +5.242s rc=0

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
