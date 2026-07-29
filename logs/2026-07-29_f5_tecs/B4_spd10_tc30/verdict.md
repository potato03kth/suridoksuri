# B4 — 판정

- 목적: 예각/U턴(135°) — 선회반경 초과 시 거동
- 실행: 2026-07-29T03:25:39.697722+00:00 ~ 2026-07-29T03:28:35.709595+00:00 (경과 168.1s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 100.0,150.0,50.0]`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 03_25_51.ulg (meta.json 기록: 03_25_51.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.06398 x ulog + 1785295546.585` (앵커 4개, 최대 잔차 1.125s). 시뮬 클록이 벽시계보다 +6.4% 빠름/느림 — 상수 오프셋만 쓰면 3.89s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 132.3993s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 28.736s → disarmed 154.924s (비행 126.188s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.452s / 역천이 5.288s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 9건 / 전체 위반 103건 / 샘플 655개, 최대 70.4668m (352.3339 m/s). 경계 최대: 3.5654m@70.892s(TRANSITION_FW), 3.527m@71.088s(FOLLOWING), 3.4696m@70.688s(TRANSITION_FW). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=78.9255 m/s² (8.0482g) @151.808s state=LANDING; 접지(disarm−5s) 제외 시 12.4921 m/s² (1.2738g) @101.584s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.1858 m/s² (0.2229g), 14.1807→5.0916 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -94.0221° → 정렬 6.672s 소요, 최대 94.0526°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.7506m vs transition_alt 50.0m → -0.4988% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.9454m → 최저 49.9454m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.9319m, 평균편차 -0.881m, 최대 \|편차\| 6.1594m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 12.5m 평균 2.4923m (부호 -4.3~12.5m, n=13) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 15건/11종, mavros.log 48건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.2011 |
| CLIMBING | +0.2s | 32.0982 |
| TRANSITION_FW | +32.3s | 12.8 |
| STREAMING | +45.1s | 0.1018 |
| FOLLOWING | +45.2s | 25.8994 |
| TRANSITION_MC | +71.1s | 5.4989 |
| HOLD | +76.6s | 11.2017 |
| LANDING | +87.8s | 44.5982 |
| DONE | +132.4s | 4.6202 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.352 | 62.14 |
| 1 | TRANS_TO_FW | 67.492 | 2.452 |
| 4 | FW | 69.944 | 26.092 |
| 2 | TRANS_TO_MC | 96.036 | 5.288 |
| 3 | MC | 101.324 | 54.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785295713.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785295713.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785295713.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785295713.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785295713.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 6780, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node:= |
| node.log | ERROR | 1 | ≈1785295713.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 6782, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=of |
| node.log | WARN | 1 | 1785295577.0 |  | home_position AMSL 미수렴(최근 2개: ['0.0', '0.0'], tol=0.5) — 이륙 보류, 수렴 대기 |
| node.log | WARN | 1 | 1785295611.3 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785295636.3 |  | 세그먼트 인덱스 급변 218→290 (Δ+72, 전체 476) pos=[213.8,15.3] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785295576.1 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.593m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785295713.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785295551.8 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785295552.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785295550.5 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785295548.7 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785295555.9 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785295551.4 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 3 | 1785295606.9 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 2 | 1785295564.2 |  | TM: RTT too high for timesync: 1705.66 ms. |
| mavros.log | WARN | 1 | 1785295558.3 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785295558.6 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785295715.9 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
