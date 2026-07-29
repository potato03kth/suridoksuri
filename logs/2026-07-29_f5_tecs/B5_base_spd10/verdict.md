# B5 — 판정

- 목적: 사각 폐곡선(시점≈종점) — 종점 근접 오판 여부
- 실행: 2026-07-29T03:35:34.303299+00:00 ~ 2026-07-29T03:39:00.361549+00:00 (경과 195.5s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0, 0.0,200.0,50.0, 0.0,20.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 03_35_46.ulg (meta.json 기록: 03_35_46.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.06276 x ulog + 1785296140.051` (앵커 4개, 최대 잔차 0.873s). 시뮬 클록이 벽시계보다 +6.3% 빠름/느림 — 상수 오프셋만 쓰면 5.39s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 155.9994s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 35.808s → disarmed 182.48s (비행 146.672s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.484s / 역천이 5.244s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 10건 / 전체 위반 213건 / 샘플 797개, 최대 83.8436m (410.9978 m/s). 경계 최대: 83.8436m@77.34s(TRANSITION_FW), 69.5021m@129.5s(HOLD), 3.4911m@77.94s(STREAMING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=85.4297 m/s² (8.7114g) @179.38s state=LANDING; 접지(disarm−5s) 제외 시 14.3855 m/s² (1.4669g) @129.572s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2559 m/s² (0.23g), 13.9754→5.039 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -94.0286° → 정렬 8.156s 소요, 최대 94.1018°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5926m vs transition_alt 50.0m → -0.8147% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.6355m → 최저 50.6355m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.1478m, 평균편차 -0.9074m, 최대 \|편차\| 6.3803m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 13.8m 평균 3.9458m (부호 0.3~13.8m, n=24) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 15건/10종, mavros.log 49건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.2015 |
| CLIMBING | +0.2s | 30.498 |
| TRANSITION_FW | +30.7s | 14.3 |
| STREAMING | +45.0s | 0.1021 |
| FOLLOWING | +45.1s | 48.3993 |
| TRANSITION_MC | +93.5s | 5.4988 |
| HOLD | +99.0s | 12.7998 |
| LANDING | +111.8s | 44.1999 |
| DONE | +156.0s | 4.9174 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.352 | 69.204 |
| 1 | TRANS_TO_FW | 74.556 | 2.484 |
| 4 | FW | 77.04 | 46.964 |
| 2 | TRANS_TO_MC | 124.004 | 5.244 |
| 3 | MC | 129.248 | 54.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785296338.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785296338.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785296338.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 1 | ≈1785296338.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 12954, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | ERROR | 1 | ≈1785296338.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785296338.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 12952, exit code 1, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node:= |
| node.log | WARN | 3 | 1785296234.8 |  | 세그먼트 인덱스 급변 190→223 (Δ+33, 전체 810) pos=[184.3,17.0] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | 1785296210.7 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785296176.8 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 4.976m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785296338.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785296146.4 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785296149.2 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785296145.6 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785296143.6 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785296152.7 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785296146.5 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785296161.3 |  | TM: RTT too high for timesync: 1856.44 ms. |
| mavros.log | ERROR | 3 | 1785296201.3 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785296155.5 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785296156.6 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785296162.5 |  | PR: request param #590 timeout, retries left 2, and 129 params still missing |
| mavros.log | WARN | 1 | 1785296340.8 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
