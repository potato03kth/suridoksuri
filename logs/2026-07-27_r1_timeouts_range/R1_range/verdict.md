# R1_range — 판정

- 목적: R1 ③: 거리 상한 발동 — A1(편도 300m) 을 기본 상한 300.0m 그대로 날려 종점 부근에서 OVERRIDE 로 떨어지는가
- 실행: 2026-07-27T12:46:18.128051+00:00 ~ 2026-07-27T12:48:11.566901+00:00 (경과 103.1s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 12_46_29.ulg (meta.json 기록: 12_46_29.ulg)
- 요약: FAIL 3, PASS 9, WARN 1

- 시각 정렬: `wall = 1.12901 x ulog + 1785156381.449` (앵커 4개, 최대 잔차 1.573s). 시뮬 클록이 벽시계보다 +12.9% 빠름/느림 — 상수 오프셋만 쓰면 7.98s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → OVERRIDE → DONE, 소요 80.1005s | DONE 상태 도달 |
| disarm 확인 | **FAIL** | 로그 끝까지 disarm 되지 않음 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → OVERRIDE → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.404s / 역천이 4.036s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 13건 / 전체 위반 78건 / 샘플 348개, 최대 217.3396m (696.6013 m/s). 경계 최대: 217.3396m@67.968s(TRANSITION_FW), 62.4299m@87.416s(OVERRIDE), 3.5208m@68.568s(TRANSITION_FW). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=5.7663 m/s² (0.588g) @68.332s state=TRANSITION_FW; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.8923 m/s² (0.2949g), 13.6145→5.1771 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -97.2998° → 정렬 13.992s 소요, 최대 97.3853°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.4271m vs transition_alt 50.0m → -1.1457% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.2323m → 최저 49.23m (손실 0.0023m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9249m, 평균편차 -0.9617m, 최대 \|편차\| 2.2859m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.4m 평균 0.47m (부호 -1.4~-0.1m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 21건/14종, mavros.log 40건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.0095 |
| CLIMBING | +1.0s | 30.9904 |
| TRANSITION_FW | +32.0s | 22.3008 |
| STREAMING | +54.3s | 0.1046 |
| FOLLOWING | +54.4s | 19.7033 |
| TRANSITION_MC | +74.1s | 0.6921 |
| OVERRIDE | +74.8s | 5.2997 |
| DONE | +80.1s | 5.3359 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.216 | 59.876 |
| 1 | TRANS_TO_FW | 65.092 | 2.404 |
| 4 | FW | 67.496 | 19.928 |
| 2 | TRANS_TO_MC | 87.424 | 4.036 |
| 3 | MC | 91.46 | 5.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 1 | 1785156479.7 |  | 거리 상한 초과 — 이륙지점에서 300m (상한 300m, 상태=transition_mc) → 안전 폴백(OVERRIDE) 실행 |
| node.log | ERROR | 2 | ≈1785156490.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785156490.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785156490.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785156490.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785156490.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1118, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785156490.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1120, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 4 | 1785156404.5 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785156405.9 |  | ⚠️ 계획 경로가 거리 상한 밖이다 — 경로 최원점 300m > 상한 300m. 그대로 날면 종점 부근에서 안전 폴백(OVERRIDE)이 걸린다. range_limit_m 을 키우거나 경로를 줄일 것 |
| node.log | WARN | 1 | 1785156439.0 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785156479.7 |  | 긴급 수동 전환 실행 → MANUAL 요청 |
| node.log | WARN | 1 | 1785156480.7 |  | 수동 모드(MANUAL) 미진입 (mode=OFFBOARD) -> AUTO.LOITER 안전 폴백 요청 |
| node.log | WARN | 1 | 1785156485.0 |  | 수동/안전 모드 진입 확인 (mode=AUTO.LOITER) -> DONE |
| node.log | WARN | 1 | ≈1785156490.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 11 | 1785156389.9 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 8 | 1785156391.1 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785156389.1 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785156387.3 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785156394.3 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785156390.0 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 1 | 1785156397.0 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785156397.9 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785156401.6 |  | TM: RTT too high for timesync: 815.69 ms. |
| mavros.log | ERROR | 1 | 1785156441.1 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785156491.8 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
