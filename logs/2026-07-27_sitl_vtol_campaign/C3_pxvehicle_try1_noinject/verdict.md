# C3 — 판정

- 목적: 천이 중 OFFBOARD 강제 이탈 — AUTO.LOITER 주입 후 재요청 복구 확인
- 실행: 2026-07-26T21:20:10.724914+00:00 ~ 2026-07-26T21:22:51.405490+00:00 (경과 160.7s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 21_20_22.ulg (meta.json 기록: 21_20_22.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.01292 x ulog + 1785100815.962` (앵커 4개, 최대 잔차 0.387s). 시뮬 클록이 벽시계보다 +1.3% 빠름/느림 — 상수 오프셋만 쓰면 1.20s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 123.7995s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 23.38s → disarmed 145.4s (비행 122.02s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.528s / 역천이 4.976s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 7건 / 전체 위반 75건 / 샘플 584개, 최대 110.1992m (626.1317 m/s). 경계 최대: 110.1992m@94.424s(HOLD), 62.2208m@89.072s(TRANSITION_MC), 5.7375m@70.472s(FOLLOWING). 스트림 재개 갭 4건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=78.7882 m/s² (8.0342g) @142.284s state=LANDING; 접지(disarm−5s) 제외 시 6.47 m/s² (0.6598g) @70.328s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3221 m/s² (0.2368g), 13.7145→5.0578 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -96.0307° → 정렬 13.396s 소요, 최대 96.0307°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5929m vs transition_alt 50.0m → -0.8142% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.2244m → 최저 50.2099m (손실 0.0145m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.0092m, 평균편차 -0.586m, 최대 \|편차\| 2.2442m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.2m 평균 0.32m (부호 -1.2~0.1m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 12건/8종, mavros.log 45건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.209 |
| CLIMBING | +1.2s | 27.9906 |
| TRANSITION_FW | +29.2s | 18.1999 |
| STREAMING | +47.4s | 0.1139 |
| FOLLOWING | +47.5s | 19.3063 |
| TRANSITION_MC | +66.8s | 5.2812 |
| HOLD | +72.1s | 8.3992 |
| LANDING | +80.5s | 43.2995 |
| DONE | +123.8s | 7.9588 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.312 | 61.864 |
| 1 | TRANS_TO_FW | 67.176 | 2.528 |
| 4 | FW | 69.704 | 19.372 |
| 2 | TRANS_TO_MC | 89.076 | 4.976 |
| 3 | MC | 94.052 | 52.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785100971.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785100971.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785100971.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785100971.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785100971.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1116, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785100971.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1118, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785100839.2 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | ≈1785100971.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785100822.8 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785100823.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785100822.3 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785100820.3 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785100827.4 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785100823.3 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 2 | 1785100876.3 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785100830.3 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785100830.3 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785100835.4 |  | TM: RTT too high for timesync: 878.00 ms. |
| mavros.log | WARN | 1 | 1785100971.6 |  | UAS Executor terminated |

## 장애주입 결과

- `set_mode` spec={"on_vtol_state": 1, "delay_s": 0.0, "action": "set_mode", "mode": "AUTO.LOITER"} → 발화 +Nones rc=None

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
