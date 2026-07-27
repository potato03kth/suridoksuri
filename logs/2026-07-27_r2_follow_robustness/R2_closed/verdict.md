# R2_closed — 판정

- 목적: R2 신설: **완전 폐회로**(종점 = 시점 = 이륙지점). 대회 경로 형상이 폐회로로 확정됐는데(2026-07-27 사용자) 캠페인에는 완전 폐회로가 없었다 — B5 는 종점을 20m 남겨 일부러 벌려 놓은 것이다. 마지막 레그가 첫 레그의 시작점으로 돌아오므로 전역 최근접 탐색이 인덱스를 0 근처로 되감을 수 있는 유일한 시나리오다
- 실행: 2026-07-27T14:28:14.984809+00:00 ~ 2026-07-27T14:35:11.135807+00:00 (경과 377.0s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0, 0.0,200.0,50.0, 0.0,0.0,50.0] range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 14_28_26.ulg (meta.json 기록: 14_28_26.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.11958 x ulog + 1785162498.990` (앵커 4개, 최대 잔차 1.612s). 시뮬 클록이 벽시계보다 +12.0% 빠름/느림 — 상수 오프셋만 쓰면 10.95s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 169.7992s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 211.08s → disarmed 362.292s (비행 151.212s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.58s / 역천이 4.488s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 11건 / 전체 위반 217건 / 샘플 1667개, 최대 84.0588m (352.4412 m/s). 경계 최대: 84.0588m@257.908s(TRANSITION_FW), 4.4446m@258.508s(TRANSITION_FW), 3.7052m@258.108s(TRANSITION_FW). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=59.4736 m/s² (6.0646g) @359.18s state=LANDING; 접지(disarm−5s) 제외 시 10.5202 m/s² (1.0728g) @310.348s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3467 m/s² (0.2393g), 13.5116→5.661 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -90.1278° → 정렬 14.328s 소요, 최대 90.1525°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.4253m vs transition_alt 50.0m → -1.1493% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 52.3464m → 최저 52.3464m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.9395m, 평균편차 -0.4295m, 최대 \|편차\| 6.7574m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 9.2m 평균 2.5375m (부호 -0.1~9.2m, n=24) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 15건/10종, mavros.log 53건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.6024 |
| CLIMBING | +0.6s | 30.797 |
| TRANSITION_FW | +31.4s | 22.4999 |
| STREAMING | +53.9s | 0.1026 |
| FOLLOWING | +54.0s | 51.2989 |
| TRANSITION_MC | +105.3s | 4.699 |
| HOLD | +110.0s | 13.3997 |
| LANDING | +123.4s | 46.3998 |
| DONE | +169.8s | 6.2451 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.204 | 249.808 |
| 1 | TRANS_TO_FW | 255.012 | 2.58 |
| 4 | FW | 257.592 | 47.936 |
| 2 | TRANS_TO_MC | 305.528 | 4.488 |
| 3 | MC | 310.016 | 53.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785162910.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785162910.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785162910.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 1 | ≈1785162910.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1128, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785162910.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785162910.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1130, exit code 1, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:= |
| node.log | WARN | 3 | 1785162800.4 |  | 세그먼트 인덱스 급변 192→222 (Δ+30, 전체 826) pos=[185.9,16.1] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | 1785162768.4 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785162732.9 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 5.051m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785162910.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785162506.9 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785162508.0 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | ERROR | 7 | 1785162559.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 5 | 1785162506.0 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785162504.2 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785162511.1 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785162507.0 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785162519.9 |  | TM: RTT too high for timesync: 2076.27 ms. |
| mavros.log | WARN | 1 | 1785162513.9 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785162515.2 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785162520.8 |  | PR: request param #888 timeout, retries left 2, and 158 params still missing |
| mavros.log | WARN | 1 | 1785162911.5 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
