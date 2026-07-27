# R2_b7 — 판정

- 목적: R2 회귀: B7 단거리 40m — 경로 전장 < lookahead(70m) < 창 폭(150m). 창이 짧은 경로를 깨지 않는가
- 실행: 2026-07-27T15:27:28.583230+00:00 ~ 2026-07-27T15:29:58.549818+00:00 (경과 149.0s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 40.0,0.0,50.0] range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 15_27_40.ulg (meta.json 기록: 15_27_40.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.00574 x ulog + 1785166054.655` (앵커 4개, 최대 잔차 0.088s). 시뮬 클록이 벽시계보다 +0.6% 빠름/느림 — 상수 오프셋만 쓰면 0.35s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 113.7984s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 23.564s → disarmed 135.088s (비행 111.524s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.516s / 역천이 6.616s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 1건 / 전체 위반 1건 / 샘플 523개, 최대 70.5339m (352.6695 m/s). 경계 최대: 70.5339m@78.2s(HOLD). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=80.1913 m/s² (8.1772g) @131.984s state=LANDING; 접지(disarm−5s) 제외 시 15.9493 m/s² (1.6264g) @78.16s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.8479 m/s² (0.2904g), 17.0287→5.0253 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -92.8088° → 정렬 13.72s 소요, 최대 92.8088°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5445m vs transition_alt 50.0m → -0.9109% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.4122m → 최저 50.4122m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9918m, 평균편차 -0.147m, 최대 \|편차\| 0.4293m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.2m 평균 0.2m (부호 -0.2~-0.2m, n=1) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 17건/9종, mavros.log 48건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.6051 |
| CLIMBING | +0.6s | 27.4941 |
| TRANSITION_FW | +28.1s | 18.4993 |
| STREAMING | +46.6s | 0.102 |
| FOLLOWING | +46.7s | 1.0992 |
| TRANSITION_MC | +47.8s | 6.9992 |
| HOLD | +54.8s | 15.4 |
| LANDING | +70.2s | 43.5995 |
| DONE | +113.8s | 5.2358 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.348 | 61.772 |
| 1 | TRANS_TO_FW | 67.12 | 2.516 |
| 4 | FW | 69.636 | 1.464 |
| 2 | TRANS_TO_MC | 71.1 | 6.616 |
| 3 | MC | 77.716 | 58.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785166197.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785166197.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785166197.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785166197.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785166197.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1131, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785166197.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1129, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | WARN | 5 | 1785166077.8 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785166108.5 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785166197.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785166060.8 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785166061.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785166060.0 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785166058.0 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785166065.5 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785166061.0 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785166074.8 |  | TM: RTT too high for timesync: 2025.41 ms. |
| mavros.log | ERROR | 2 | 1785166115.3 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785166068.5 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785166069.2 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785166075.8 |  | PR: request param #562 timeout, retries left 2, and 351 params still missing |
| mavros.log | WARN | 1 | 1785166198.8 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
