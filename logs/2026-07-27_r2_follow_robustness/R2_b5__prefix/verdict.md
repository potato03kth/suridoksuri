# R2_b5 — 판정

- 목적: R2 핵심: B5 사각 폐곡선(시점≈종점, 20m 마진) — _find_segment 회귀 검증의 핵심
- 실행: 2026-07-27T13:41:32.430824+00:00 ~ 2026-07-27T13:48:57.996260+00:00 (경과 404.3s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0, 0.0,200.0,50.0, 0.0,20.0,50.0] range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 13_41_55.ulg (meta.json 기록: 13_41_55.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.11506 x ulog + 1785159707.860` (앵커 4개, 최대 잔차 1.487s). 시뮬 클록이 벽시계보다 +11.5% 빠름/느림 — 상수 오프셋만 쓰면 10.44s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 168.1991s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 229.676s → disarmed 379.488s (비행 149.812s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.476s / 역천이 5.144s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 10건 / 전체 위반 206건 / 샘플 1747개, 최대 111.9648m (583.1501 m/s). 경계 최대: 84.1652m@276.396s(TRANSITION_FW), 3.5472m@276.98s(TRANSITION_FW), 3.5125m@276.784s(TRANSITION_FW). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=45.2843 m/s² (4.6177g) @376.404s state=LANDING; 접지(disarm−5s) 제외 시 5.2751 m/s² (0.5379g) @301.336s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3014 m/s² (0.2347g), 13.9973→5.0387 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -99.4429° → 정렬 13.924s 소요, 최대 99.4731°, tol 진입 후 재증가 0.0116 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5361m vs transition_alt 50.0m → -0.9278% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 51.9625m → 최저 51.9625m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0336m, 평균편차 -0.6574m, 최대 \|편차\| 6.5531m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 14.4m 평균 2.9m (부호 -0.3~14.4m, n=24) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 16건/10종, mavros.log 58건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.5029 |
| CLIMBING | +0.5s | 31.0958 |
| TRANSITION_FW | +31.6s | 21.8004 |
| STREAMING | +53.4s | 0.1264 |
| FOLLOWING | +53.5s | 50.4061 |
| TRANSITION_MC | +103.9s | 5.4683 |
| HOLD | +109.4s | 12.1992 |
| LANDING | +121.6s | 46.6 |
| DONE | +168.2s | 5.1874 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 8.084 | 265.484 |
| 1 | TRANS_TO_FW | 273.568 | 2.476 |
| 4 | FW | 276.044 | 47.192 |
| 2 | TRANS_TO_MC | 323.236 | 5.144 |
| 3 | MC | 328.38 | 52.004 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785160137.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785160137.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785160137.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785160137.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785160137.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1163, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785160137.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1165, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 3 | 1785160029.1 |  | 세그먼트 인덱스 급변 192→221 (Δ+29, 전체 810) pos=[185.7,15.5] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | 1785159997.3 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785159963.0 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 4.983m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785160137.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785159716.8 |  | FCU: EVENT 11464574 with args -0-64-0-0-20-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 12 | 1785159716.8 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-16-0-128-0-0-0-176-58-127-128-176-58-127-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | ERROR | 8 | 1785159772.0 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 6 | 1785159710.5 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785159708.7 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785159722.9 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785159716.8 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785159731.6 |  | TM: RTT too high for timesync: 2094.74 ms. |
| mavros.log | WARN | 2 | 1785159713.2 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785159728.4 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785159732.7 |  | PR: request param #361 timeout, retries left 2, and 486 params still missing |
| mavros.log | WARN | 1 | 1785160138.2 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
