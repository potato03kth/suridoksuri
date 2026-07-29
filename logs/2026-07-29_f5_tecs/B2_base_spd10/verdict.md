# B2 — 판정

- 목적: 완만 곡선 4WP(30°급 꺾임) — eta3 NR 경로(_trapz 수정) 실행 검증
- 실행: 2026-07-29T03:46:29.605564+00:00 ~ 2026-07-29T03:49:46.478344+00:00 (경과 188.2s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 150.0,0.0,50.0, 300.0,80.0,50.0, 450.0,200.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 03_46_41.ulg (meta.json 기록: 03_46_41.ulg)
- 요약: FAIL 2, PASS 9, WARN 2

- 시각 정렬: `wall = 1.06569 x ulog + 1785296794.072` (앵커 4개, 최대 잔차 0.662s). 시뮬 클록이 벽시계보다 +6.6% 빠름/느림 — 상수 오프셋만 쓰면 5.10s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 144.1995s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 39.82s → disarmed 175.508s (비행 135.688s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.548s / 역천이 5.22s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 8건 / 전체 위반 150건 / 샘플 759개, 최대 417.5762m (1391.9208 m/s). 경계 최대: 417.5762m@80.668s(TRANSITION_FW), 70.4378m@120.7s(HOLD), 3.6155m@81.068s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=39.7147 m/s² (4.0498g) @172.404s state=LANDING; 접지(disarm−5s) 제외 시 16.5716 m/s² (1.6898g) @120.672s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2104 m/s² (0.2254g), 13.9724→5.0435 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -98.1898° → 정렬 8.808s 소요, 최대 98.405°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.3196m vs transition_alt 50.0m → -1.3608% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 51.5437m → 최저 51.5437m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.8335m, 평균편차 0.177m, 최대 \|편차\| 1.8772m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 6.6m 평균 2.7056m (부호 0.4~6.6m, n=18) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 13건/9종, mavros.log 49건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.8004 |
| CLIMBING | +1.8s | 27.4987 |
| TRANSITION_FW | +29.3s | 14.9999 |
| STREAMING | +44.3s | 0.1026 |
| FOLLOWING | +44.4s | 35.7997 |
| TRANSITION_MC | +80.2s | 5.5982 |
| HOLD | +85.8s | 11.4004 |
| LANDING | +97.2s | 46.9996 |
| DONE | +144.2s | 5.4444 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.328 | 72.344 |
| 1 | TRANS_TO_FW | 77.672 | 2.548 |
| 4 | FW | 80.22 | 34.796 |
| 2 | TRANS_TO_MC | 115.016 | 5.22 |
| 3 | MC | 120.236 | 56.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785296985.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785296985.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785296985.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785296985.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785296985.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 18605, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | ERROR | 1 | ≈1785296985.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 18602, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785296867.4 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785296835.8 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.450m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785296985.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785296801.5 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785296802.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785296800.8 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785296798.8 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785296806.4 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785296801.8 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785296816.8 |  | TM: RTT too high for timesync: 2026.38 ms. |
| mavros.log | ERROR | 3 | 1785296856.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785296810.6 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785296810.6 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785296817.7 |  | PR: request param #488 timeout, retries left 2, and 207 params still missing |
| mavros.log | WARN | 1 | 1785296986.7 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
