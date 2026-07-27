# A3 — 판정

- 목적: SITL-4 L자 경로 재현 (90° 1회)
- 실행: 2026-07-27T17:11:36.729458+00:00 ~ 2026-07-27T17:15:23.301658+00:00 (경과 217.5s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0] range_limit_m:=1200.0`
- 저장소 HEAD: `893a5eb`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 17_11_49.ulg (meta.json 기록: 17_11_49.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.03873 x ulog + 1785172303.148` (앵커 4개, 최대 잔차 0.334s). 시뮬 클록이 벽시계보다 +3.9% 빠름/느림 — 상수 오프셋만 쓰면 2.87s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 135.3999s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 75.32s → disarmed 204.616s (비행 129.296s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.484s / 역천이 5.104s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 5건 / 전체 위반 102건 / 샘플 879개, 최대 232.8999m (576.4849 m/s). 경계 최대: 232.8999m@121.696s(FOLLOWING), 70.4783m@151.428s(HOLD), 3.6934m@122.296s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=75.2581 m/s² (7.6742g) @201.448s state=LANDING; 접지(disarm−5s) 제외 시 7.7606 m/s² (0.7914g) @122.236s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2684 m/s² (0.2313g), 13.6537→5.0407 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -93.3717° → 정렬 13.76s 소요, 최대 93.435°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.4146m vs transition_alt 50.0m → -1.1708% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 48.5796m → 최저 48.5758m (손실 0.0038m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.8411m, 평균편차 -1.4409m, 최대 \|편차\| 5.2613m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 14.6m 평균 1.6769m (부호 -0.4~14.6m, n=13) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 18건/10종, mavros.log 50건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.5016 |
| CLIMBING | +0.5s | 28.498 |
| TRANSITION_FW | +29.0s | 18.6004 |
| STREAMING | +47.6s | 0.1032 |
| FOLLOWING | +47.7s | 25.7977 |
| TRANSITION_MC | +73.5s | 5.2993 |
| HOLD | +78.8s | 12.1001 |
| LANDING | +90.9s | 44.4996 |
| DONE | +135.4s | 5.2471 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.328 | 113.704 |
| 1 | TRANS_TO_FW | 119.032 | 2.484 |
| 4 | FW | 121.516 | 24.532 |
| 2 | TRANS_TO_MC | 146.048 | 5.104 |
| 3 | MC | 151.152 | 54.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 5 | ≈1785172522.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 3 | ≈1785172522.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785172522.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785172522.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785172522.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 4115, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785172522.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 4113, exit code 1, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node |
| node.log | WARN | 1 | 1785172412.6 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785172442.5 |  | 세그먼트 인덱스 급변 191→221 (Δ+30, 전체 414) pos=[185.1,15.5] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785172380.1 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 5.019m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785172522.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785172309.7 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785172310.8 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785172308.9 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785172307.0 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785172314.1 |  | VER: command plugin service call failed! |
| mavros.log | ERROR | 4 | 1785172363.1 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 3 | 1785172309.9 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785172322.6 |  | TM: RTT too high for timesync: 1770.99 ms. |
| mavros.log | WARN | 1 | 1785172317.2 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785172318.7 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785172324.1 |  | PR: request param #1309 timeout, retries left 2, and 1 params still missing |
| mavros.log | WARN | 1 | 1785172523.7 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
