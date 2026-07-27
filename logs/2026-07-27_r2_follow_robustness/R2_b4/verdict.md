# R2_b4 — 판정

- 목적: R2 핵심: B4 U턴(135°) — 두 레그가 서로 가까워지는 구간. 전역탐색이 반대 레그를 잡을 수 있던 조건
- 실행: 2026-07-27T14:16:32.795545+00:00 ~ 2026-07-27T14:20:37.102969+00:00 (경과 223.4s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 100.0,150.0,50.0] range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 14_16_44.ulg (meta.json 기록: 14_16_44.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.09327 x ulog + 1785161799.010` (앵커 4개, 최대 잔차 0.389s). 시뮬 클록이 벽시계보다 +9.3% 빠름/느림 — 상수 오프셋만 쓰면 6.90s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 149.0985s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 75.472s → disarmed 210.292s (비행 134.82s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.488s / 역천이 5.024s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 2건 / 전체 위반 101건 / 샘플 904개, 최대 70.5334m (352.667 m/s). 경계 최대: 70.5334m@153.364s(HOLD), 3.2963m@122.144s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=40.9649 m/s² (4.1773g) @207.108s state=LANDING; 접지(disarm−5s) 제외 시 15.2527 m/s² (1.5553g) @153.324s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2629 m/s² (0.2308g), 13.8034→5.0235 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -92.0829° → 정렬 13.232s 소요, 최대 92.0829°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5242m vs transition_alt 50.0m → -0.9515% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 51.1876m → 최저 51.1295m (손실 0.0581m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.7493m, 평균편차 -0.9485m, 최대 \|편차\| 7.4926m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 11.6m 평균 1.6143m (부호 -4.5~11.6m, n=14) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 14건/10종, mavros.log 50건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.3085 |
| CLIMBING | +0.3s | 30.7901 |
| TRANSITION_FW | +31.1s | 18.8001 |
| STREAMING | +49.9s | 0.1029 |
| FOLLOWING | +50.0s | 29.298 |
| TRANSITION_MC | +79.3s | 5.3 |
| HOLD | +84.6s | 14.1992 |
| LANDING | +98.8s | 50.2997 |
| DONE | +149.1s | 4.8911 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.276 | 113.9 |
| 1 | TRANS_TO_FW | 119.176 | 2.488 |
| 4 | FW | 121.664 | 26.304 |
| 2 | TRANS_TO_MC | 147.968 | 5.024 |
| 3 | MC | 152.992 | 58.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785162035.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785162035.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785162035.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785162035.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785162035.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1123, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785162035.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1125, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785161914.8 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785161949.1 |  | 세그먼트 인덱스 급변 219→286 (Δ+67, 전체 472) pos=[215.1,15.9] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785161879.6 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.776m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785162035.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785161804.8 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785161805.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785161803.8 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785161802.0 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785161808.9 |  | VER: command plugin service call failed! |
| mavros.log | ERROR | 4 | 1785161857.3 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 3 | 1785161804.7 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785161817.6 |  | TM: RTT too high for timesync: 2024.82 ms. |
| mavros.log | WARN | 1 | 1785161811.6 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785161812.5 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785161818.5 |  | PR: request param #689 timeout, retries left 2, and 230 params still missing |
| mavros.log | WARN | 1 | 1785162037.3 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
