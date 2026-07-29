# B5 — 판정

- 목적: 사각 폐곡선(시점≈종점) — 종점 근접 오판 여부
- 실행: 2026-07-29T03:39:15.938410+00:00 ~ 2026-07-29T03:42:38.875168+00:00 (경과 192.1s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0, 0.0,200.0,50.0, 0.0,20.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 03_39_27.ulg (meta.json 기록: 03_39_27.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.06255 x ulog + 1785296361.573` (앵커 4개, 최대 잔차 0.450s). 시뮬 클록이 벽시계보다 +6.3% 빠름/느림 — 상수 오프셋만 쓰면 5.55s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 154.199s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 33.024s → disarmed 179.512s (비행 146.488s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.496s / 역천이 5.14s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 3건 / 전체 위반 212건 / 샘플 786개, 최대 85.2562m (352.2428 m/s). 경계 최대: 85.2562m@74.424s(FOLLOWING), 70.4486m@126.66s(HOLD), 4.1381m@74.62s(FOLLOWING). 스트림 재개 갭 3건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=72.6565 m/s² (7.4089g) @176.332s state=LANDING; 접지(disarm−5s) 제외 시 15.0475 m/s² (1.5344g) @126.616s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2578 m/s² (0.2302g), 14.066→5.0426 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -93.2404° → 정렬 8.04s 소요, 최대 93.2478°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.3993m vs transition_alt 50.0m → -1.2014% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.64m → 최저 49.6373m (손실 0.0027m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.8516m, 평균편차 -0.6824m, 최대 \|편차\| 5.6429m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 13.1m 평균 4.4625m (부호 0.1~13.1m, n=24) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 16건/10종, mavros.log 49건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.8017 |
| CLIMBING | +0.8s | 29.2982 |
| TRANSITION_FW | +30.1s | 13.001 |
| STREAMING | +43.1s | 0.1037 |
| FOLLOWING | +43.2s | 50.2958 |
| TRANSITION_MC | +93.5s | 5.4996 |
| HOLD | +99.0s | 11.0995 |
| LANDING | +110.1s | 44.0995 |
| DONE | +154.2s | 6.4701 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.324 | 66.192 |
| 1 | TRANS_TO_FW | 71.516 | 2.496 |
| 4 | FW | 74.012 | 47.06 |
| 2 | TRANS_TO_MC | 121.072 | 5.14 |
| 3 | MC | 126.212 | 54.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785296557.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785296557.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785296557.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785296557.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785296557.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 14795, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785296557.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 14797, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 3 | 1785296451.5 |  | 세그먼트 인덱스 급변 190→223 (Δ+33, 전체 810) pos=[183.8,17.3] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | 1785296429.0 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785296395.6 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 4.976m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785296557.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785296368.0 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785296369.1 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785296367.3 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785296365.4 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785296372.4 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785296368.3 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785296380.7 |  | TM: RTT too high for timesync: 1706.03 ms. |
| mavros.log | ERROR | 3 | 1785296423.0 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785296375.2 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785296375.6 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785296382.3 |  | PR: request param #1153 timeout, retries left 2, and 52 params still missing |
| mavros.log | WARN | 1 | 1785296559.1 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
