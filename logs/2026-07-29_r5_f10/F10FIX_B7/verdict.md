# B7 — 판정

- 목적: 단거리 경로(40m, d_end_thresh=10m 대비 짧음) — FOLLOWING 즉시완료 오판
- 실행: 2026-07-29T02:33:06.324212+00:00 ~ 2026-07-29T02:35:33.626454+00:00 (경과 142.4s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 40.0,0.0,50.0]`
- 저장소 HEAD: `afce94d`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 02_33_18.ulg (meta.json 기록: 02_33_18.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.02954 x ulog + 1785292392.914` (앵커 4개, 최대 잔차 0.057s). 시뮬 클록이 벽시계보다 +3.0% 빠름/느림 — 상수 오프셋만 쓰면 1.27s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 111.3007s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 22.432s → disarmed 128.864s (비행 106.432s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.496s / 역천이 6.312s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 2건 / 전체 위반 2건 / 샘플 535개, 최대 70.5358m (352.6792 m/s). 경계 최대: 70.5358m@72.336s(HOLD), 62.7549m@65.556s(TRANSITION_MC). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=70.7052 m/s² (7.2099g) @125.676s state=LANDING; 접지(disarm−5s) 제외 시 13.3542 m/s² (1.3617g) @72.34s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.8356 m/s² (0.2892g), 17.059→5.2192 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -96.9696° → 정렬 7.792s 소요, 최대 97.2922°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5712m vs transition_alt 50.0m → -0.8576% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.2749m → 최저 49.2604m (손실 0.0145m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9913m, 평균편차 -1.2177m, 최대 \|편차\| 1.5301m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.5m 평균 1.5m (부호 1.5~1.5m, n=1) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 15건/9종, mavros.log 45건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.6006 |
| CLIMBING | +1.6s | 28.699 |
| TRANSITION_FW | +30.3s | 12.8001 |
| STREAMING | +43.1s | 0.1006 |
| FOLLOWING | +43.2s | 1.1008 |
| TRANSITION_MC | +44.3s | 6.6988 |
| HOLD | +51.0s | 15.6 |
| LANDING | +66.6s | 44.7009 |
| DONE | +111.3s | 6.2084 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.38 | 56.192 |
| 1 | TRANS_TO_FW | 61.572 | 2.496 |
| 4 | FW | 64.068 | 1.492 |
| 2 | TRANS_TO_MC | 65.56 | 6.312 |
| 3 | MC | 71.872 | 57.004 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785292533.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785292533.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785292533.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785292533.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785292533.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 12896, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node |
| node.log | ERROR | 1 | ≈1785292533.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 12898, exit code -2, cmd '/root/ws_c1b/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:= |
| node.log | WARN | 3 | 1785292415.7 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785292448.4 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785292533.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785292398.5 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785292399.6 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785292397.5 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785292395.5 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785292402.9 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785292398.4 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 2 | 1785292451.9 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785292405.9 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785292406.0 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785292412.2 |  | TM: RTT too high for timesync: 735.48 ms. |
| mavros.log | WARN | 1 | 1785292534.0 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
