# C1b — 판정

- 목적: 천이고도 민감도 — 고(120m). 경로는 A1
- 실행: 2026-07-28T17:54:58.841459+00:00 ~ 2026-07-28T18:04:54.711912+00:00 (경과 570.2s)
- 종료: `timeout` (exit=2)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=120.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local alt_slew_rate:=0.0 range_limit_m:=1200.0`
- 저장소 HEAD: `94989b6`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 17_55_11.ulg (meta.json 기록: 17_55_11.ulg)
- 요약: FAIL 6, NULL 1, PASS 4, WARN 2

- 시각 정렬: `wall = 1.05966 x ulog + 1785261304.723` (앵커 3개, 최대 잔차 1.110s). 시뮬 클록이 벽시계보다 +6.0% 빠름/느림 — 상수 오프셋만 쓰면 5.15s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **FAIL** | 관측 상태: ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING; 종료사유=timeout | DONE 상태 도달 |
| disarm 확인 | **FAIL** | 로그 끝까지 disarm 되지 않음 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **FAIL** | seq=[3, 1, 4], 정천이 2.516s / 역천이 Nones | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 4건 / 전체 위반 96건 / 샘플 2696개, 최대 216.6758m (712.7493 m/s). 경계 최대: 216.6758m@95.3s(FOLLOWING), 3.6569m@95.7s(FOLLOWING), 3.6237m@95.9s(FOLLOWING). 스트림 재개 갭 1건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=2936.6353 m/s² (299.4535g) @141.984s state=FOLLOWING; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **NULL** | 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다 | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -97.08° → 정렬 14.304s 소요, 최대 97.1179°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 119.3155m vs transition_alt 120.0m → -0.5704% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 120.7502m → 최저 120.6659m (손실 0.0843m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0463m, 평균편차 -42.3563m, 최대 \|편차\| 70.5869m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 141.5m 평균 128.3631m (부호 -141.5~53.9m, n=236) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 13건/9종, mavros.log 93건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 2.5076 |
| CLIMBING | +2.5s | 53.1913 |
| TRANSITION_FW | +55.7s | 20.0001 |
| STREAMING | +75.7s | 0.117 |
| FOLLOWING | +75.8s | 488.985 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.336 | 87.072 |
| 1 | TRANS_TO_FW | 92.408 | 2.516 |
| 4 | FW | 94.924 | 469.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785261894.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785261894.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785261894.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785261894.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785261894.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1348, exit code -2, cmd '/root/ws_r5b/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785261894.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1350, exit code -2, cmd '/root/ws_r5b/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 1 | 1785261387.4 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785261402.6 |  | ⚠️ 천이고도와 경로고도가 70.7m 어긋난다 (transition_alt=120.0m vs 경로 순항고도, 현재 120.7m → 50.0m). 램프가 계단은 막지만 기체는 이 고도차를 순항 중에 메워야 한다 — 의도한 값인지 확인할 것 |
| node.log | WARN | 1 | ≈1785261894.4 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 34 | 1785261312.0 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 28 | 1785261313.2 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | ERROR | 11 | 1785261363.6 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 5 | 1785261311.3 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785261309.4 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785261316.5 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 2 | 1785261312.3 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 2 | 1785261318.2 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785261319.4 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785261324.1 |  | TM: RTT too high for timesync: 505.09 ms. |
| mavros.log | WARN | 1 | 1785261894.9 |  | UAS Executor terminated |

## 미산출 지표 (null)

- **역천이 감속률**: 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
