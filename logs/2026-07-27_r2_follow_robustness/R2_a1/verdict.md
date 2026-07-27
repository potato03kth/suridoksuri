# R2_a1 — 판정

- 목적: R2 회귀: A1 직선 300m — 창 탐색이 단순 직선을 깨지 않는가
- 실행: 2026-07-27T14:01:23.730405+00:00 ~ 2026-07-27T14:04:29.060289+00:00 (경과 167.4s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 14_01_35.ulg (meta.json 기록: 14_01_35.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.10927 x ulog + 1785160889.761` (앵커 4개, 최대 잔차 0.730s). 시뮬 클록이 벽시계보다 +10.9% 빠름/느림 — 상수 오프셋만 쓰면 7.92s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 142.4987s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 27.404s → disarmed 154.46s (비행 127.056s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.54s / 역천이 5.164s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 1건 / 전체 위반 73건 / 샘플 626개, 최대 214.9846m (707.1861 m/s). 경계 최대: 70.463m@99.26s(HOLD). 스트림 재개 갭 3건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=42.1844 m/s² (4.3016g) @151.276s state=LANDING; 접지(disarm−5s) 제외 시 6.1303 m/s² (0.6251g) @75.06s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2365 m/s² (0.2281g), 13.7438→5.0304 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -86.729° → 정렬 13.068s 소요, 최대 86.9282°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5951m vs transition_alt 50.0m → -0.8099% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.9491m → 최저 49.9491m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.008m, 평균편차 -0.4075m, 최대 \|편차\| 2.2692m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.3m 평균 0.18m (부호 -0.3~0.2m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 12건/8종, mavros.log 50건/13종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.1016 |
| CLIMBING | +1.1s | 31.6968 |
| TRANSITION_FW | +32.8s | 18.7997 |
| STREAMING | +51.6s | 0.1017 |
| FOLLOWING | +51.7s | 22.5005 |
| TRANSITION_MC | +74.2s | 5.3988 |
| HOLD | +79.6s | 10.7995 |
| LANDING | +90.4s | 52.1002 |
| DONE | +142.5s | 5.2309 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.332 | 66.548 |
| 1 | TRANS_TO_FW | 71.88 | 2.54 |
| 4 | FW | 74.42 | 19.352 |
| 2 | TRANS_TO_MC | 93.772 | 5.164 |
| 3 | MC | 98.936 | 56.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785161067.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785161067.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785161067.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785161067.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785161067.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1219, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785161067.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1221, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785160954.9 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785161067.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785160895.7 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785160896.8 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785160894.8 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785160893.0 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785160899.9 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785160895.7 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785160912.1 |  | TM: RTT too high for timesync: 2118.41 ms. |
| mavros.log | ERROR | 3 | 1785160951.8 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785160902.6 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785160903.1 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785160913.0 |  | PR: Failed to get parameter type: NAV_DLL_ACT |
| mavros.log | WARN | 1 | 1785160913.0 |  | PR: request param #429 timeout, retries left 2, and 375 params still missing |
| mavros.log | WARN | 1 | 1785161069.3 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
