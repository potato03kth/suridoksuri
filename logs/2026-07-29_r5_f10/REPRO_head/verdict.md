# C1b — 판정

- 목적: 천이고도 민감도 — 고(120m). 경로는 A1
- 실행: 2026-07-29T01:48:21.460046+00:00 ~ 2026-07-29T01:50:24.787146+00:00 (경과 119.1s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=120.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local`
- 저장소 HEAD: `9c5d17f`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 01_48_34.ulg (meta.json 기록: 01_48_34.ulg)
- 요약: FAIL 5, NULL 1, PASS 5, WARN 2

- 시각 정렬: `wall = 1.03567 x ulog + 1785289707.934` (앵커 3개, 최대 잔차 0.538s). 시뮬 클록이 벽시계보다 +3.6% 빠름/느림 — 상수 오프셋만 쓰면 2.74s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → OVERRIDE → DONE, 소요 89.8999s | DONE 상태 도달 |
| disarm 확인 | **FAIL** | 로그 끝까지 disarm 되지 않음 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → OVERRIDE → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **FAIL** | seq=[3, 1, 4], 정천이 2.52s / 역천이 Nones | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 5건 / 전체 위반 72건 / 샘플 437개, 최대 214.5495m (724.8293 m/s). 경계 최대: 214.5495m@85.328s(FOLLOWING), 3.7325m@85.72s(FOLLOWING), 3.6744m@85.92s(FOLLOWING). 스트림 재개 갭 1건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=7.2129 m/s² (0.7355g) @105.116s state=OVERRIDE; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **NULL** | 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다 | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -92.7869° → 정렬 8.472s 소요, 최대 92.7869°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 119.372m vs transition_alt 120.0m → -0.5233% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 120.0478m → 최저 120.0478m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0084m, 평균편차 54.5611m, 최대 \|편차\| 69.9335m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 3.2m 평균 1.43m (부호 -1.1~3.2m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 18건/14종, mavros.log 41건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.4081 |
| CLIMBING | +1.4s | 52.4915 |
| TRANSITION_FW | +53.9s | 13.5999 |
| STREAMING | +67.5s | 0.1021 |
| FOLLOWING | +67.6s | 19.8994 |
| OVERRIDE | +87.5s | 2.3989 |
| DONE | +89.9s | 5.8886 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.168 | 77.168 |
| 1 | TRANS_TO_FW | 82.336 | 2.52 |
| 4 | FW | 84.856 | 29.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 1 | 1785289816.0 |  | 거리 상한 초과 — 이륙지점에서 300m (상한 300m, 상태=following) → 안전 폴백(OVERRIDE) 실행 |
| node.log | ERROR | 2 | ≈1785289824.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785289824.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785289824.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785289824.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785289824.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1359, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785289824.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1361, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 1 | 1785289729.9 |  | ⚠️ 계획 경로가 거리 상한 밖이다 — 경로 최원점 300m > 상한 300m. 그대로 날면 종점 부근에서 안전 폴백(OVERRIDE)이 걸린다. range_limit_m 을 키우거나 경로를 줄일 것 |
| node.log | WARN | 1 | 1785289785.2 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785289793.2 |  | ⚠️ 천이고도와 경로고도가 70.0m 어긋난다 (transition_alt=120.0m vs 경로 순항고도, 현재 120.0m → 50.0m). 램프가 계단은 막지만 기체는 이 고도차를 순항 중에 메워야 한다 — 의도한 값인지 확인할 것 |
| node.log | WARN | 1 | 1785289816.0 |  | 긴급 수동 전환 실행 → MANUAL 요청 |
| node.log | WARN | 1 | 1785289817.7 |  | 수동 모드(MANUAL) 미진입 (mode=OFFBOARD) -> AUTO.LOITER 안전 폴백 요청 |
| node.log | WARN | 1 | 1785289818.4 |  | 수동/안전 모드 진입 확인 (mode=AUTO.LOITER) -> DONE |
| node.log | WARN | 1 | ≈1785289824.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 11 | 1785289714.1 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 8 | 1785289715.5 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785289713.8 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785289711.9 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785289719.8 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785289714.8 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 2 | 1785289767.2 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785289722.0 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785289722.8 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785289727.7 |  | TM: RTT too high for timesync: 724.43 ms. |
| mavros.log | WARN | 1 | 1785289825.0 |  | UAS Executor terminated |

## 미산출 지표 (null)

- **역천이 감속률**: 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
