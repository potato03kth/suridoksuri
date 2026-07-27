# R1_f15 — 판정

- 목적: R1 ②: F-15 — 헤딩 정렬(Phase 2) 구간에 AUTO.LOITER 주입. 종전엔 재요청 경로가 없어 무한 대기했다(C3 는 ACTIVE TRANSITION 구간만 주입해 미실측)
- 실행: 2026-07-27T12:48:22.121985+00:00 ~ 2026-07-27T12:51:21.501253+00:00 (경과 162.1s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 12_48_34.ulg (meta.json 기록: 12_48_34.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.16247 x ulog + 1785156504.089` (앵커 4개, 최대 잔차 0.373s). 시뮬 클록이 벽시계보다 +16.2% 빠름/느림 — 상수 오프셋만 쓰면 11.44s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 141.1992s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 23.116s → disarmed 148.432s (비행 125.316s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.528s / 역천이 5.228s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 5건 / 전체 위반 73건 / 샘플 591개, 최대 218.4059m (1070.6172 m/s). 경계 최대: 218.4059m@72.432s(FOLLOWING), 112.0519m@97.068s(HOLD), 3.9839m@73.036s(FOLLOWING). 스트림 재개 갭 3건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=43.1229 m/s² (4.3973g) @145.296s state=DONE; 접지(disarm−5s) 제외 시 6.4931 m/s² (0.6621g) @72.828s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.274 m/s² (0.2319g), 13.6916→5.0273 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -95.6529° → 정렬 15.616s 소요, 최대 95.9098°, tol 진입 후 재증가 0.0039 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.4022m vs transition_alt 50.0m → -1.1955% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.2694m → 최저 49.2673m (손실 0.0021m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9629m, 평균편차 -0.764m, 최대 \|편차\| 1.7741m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.7m 평균 0.23m (부호 -0.7~-0.0m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 22건/9종, mavros.log 49건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.9029 |
| CLIMBING | +1.9s | 32.0964 |
| TRANSITION_FW | +34.0s | 23.1997 |
| STREAMING | +57.2s | 0.1056 |
| FOLLOWING | +57.3s | 22.4127 |
| TRANSITION_MC | +79.7s | 5.5824 |
| HOLD | +85.3s | 8.7998 |
| LANDING | +94.1s | 47.0997 |
| DONE | +141.2s | 7.7754 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.376 | 64.256 |
| 1 | TRANS_TO_FW | 69.632 | 2.528 |
| 4 | FW | 72.16 | 19.316 |
| 2 | TRANS_TO_MC | 91.476 | 5.228 |
| 3 | MC | 96.704 | 52.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785156679.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785156679.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785156679.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785156679.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785156679.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1126, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785156679.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1128, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 9 | 1785156529.8 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 2 | 1785156566.7 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785156679.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785156514.4 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785156515.4 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785156513.2 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785156511.4 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785156518.8 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785156515.0 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785156527.0 |  | TM: RTT too high for timesync: 1706.11 ms. |
| mavros.log | ERROR | 3 | 1785156567.7 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785156521.6 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785156522.0 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785156681.7 |  | UAS Executor terminated |

## 장애주입 결과

- `set_mode` spec={"on_log": "천이 전 MC OFFBOARD 요청", "delay_s": 5.0, "action": "set_mode", "mode": "AUTO.LOITER"} → 발화 +40.261s rc=0

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
