# R2_closed — 판정

- 목적: R2 신설: **완전 폐회로**(종점 = 시점 = 이륙지점). 대회 경로 형상이 폐회로로 확정됐는데(2026-07-27 사용자) 캠페인에는 완전 폐회로가 없었다 — B5 는 종점을 20m 남겨 일부러 벌려 놓은 것이다. 마지막 레그가 첫 레그의 시작점으로 돌아오므로 전역 최근접 탐색이 인덱스를 0 근처로 되감을 수 있는 유일한 시나리오다
- 실행: 2026-07-27T15:41:26.828220+00:00 ~ 2026-07-27T15:48:30.707447+00:00 (경과 417.2s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0, 0.0,200.0,50.0, 0.0,0.0,50.0] range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 15_41_38.ulg (meta.json 기록: 15_41_38.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.02015 x ulog + 1785166896.483` (앵커 4개, 최대 잔차 0.242s). 시뮬 클록이 벽시계보다 +2.0% 빠름/느림 — 상수 오프셋만 쓰면 1.88s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 156.3983s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 246.656s → disarmed 399.676s (비행 153.02s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.512s / 역천이 4.452s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 6건 / 전체 위반 217건 / 샘플 1852개, 최대 69.3772m (346.8861 m/s). 경계 최대: 69.3772m@347.108s(TRANSITION_MC), 3.6299m@294.836s(FOLLOWING), 3.6017m@295.036s(FOLLOWING). 스트림 재개 갭 3건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=82.1174 m/s² (8.3736g) @396.548s state=LANDING; 접지(disarm−5s) 제외 시 13.8692 m/s² (1.4143g) @347.24s state=TRANSITION_MC | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3291 m/s² (0.2375g), 13.5824→5.6425 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -97.4606° → 정렬 14.636s 소요, 최대 97.4606°, tol 진입 후 재증가 0.0486 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.6417m vs transition_alt 50.0m → -0.7165% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 51.3692m → 최저 51.3692m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.1058m, 평균편차 -0.6371m, 최대 \|편차\| 6.4419m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 12.8m 평균 2.896m (부호 -1.8~12.8m, n=25) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 16건/10종, mavros.log 54건/13종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.31 |
| CLIMBING | +0.3s | 28.5888 |
| TRANSITION_FW | +28.9s | 19.7996 |
| STREAMING | +48.7s | 0.1099 |
| FOLLOWING | +48.8s | 48.5915 |
| TRANSITION_MC | +97.4s | 5.3988 |
| HOLD | +102.8s | 10.0002 |
| LANDING | +112.8s | 43.5995 |
| DONE | +156.4s | 6.2196 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.368 | 286.172 |
| 1 | TRANS_TO_FW | 291.54 | 2.512 |
| 4 | FW | 294.052 | 48.22 |
| 2 | TRANS_TO_MC | 342.272 | 4.452 |
| 3 | MC | 346.724 | 53.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785167310.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785167310.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785167310.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785167310.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785167310.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1154, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785167310.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1156, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 3 | 1785167208.6 |  | 세그먼트 인덱스 급변 192→222 (Δ+30, 전체 826) pos=[185.7,16.2] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | 1785167179.0 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785167146.7 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 5.057m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785167310.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785166899.1 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785166900.1 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | ERROR | 7 | 1785166954.0 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 5 | 1785166898.2 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785166896.3 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785166903.7 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785166899.2 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785166912.8 |  | TM: RTT too high for timesync: 2065.94 ms. |
| mavros.log | WARN | 1 | 1785166906.6 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785166907.4 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785166913.8 |  | PR: request param #549 timeout, retries left 2, and 344 params still missing |
| mavros.log | WARN | 1 | 1785166914.0 |  | PR: Failed to get parameter type: NAV_DLL_ACT |
| mavros.log | WARN | 1 | 1785167310.9 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
