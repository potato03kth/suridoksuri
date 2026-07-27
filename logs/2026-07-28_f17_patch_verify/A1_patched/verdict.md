# A1 — 판정

- 목적: SITL-4 직선 300m 재현 (3주치 변경 후 회귀 여부). waypoint_frame=local(SITL-4 당시 동작)
- 실행: 2026-07-27T17:08:34.348923+00:00 ~ 2026-07-27T17:11:23.039975+00:00 (경과 162.9s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local range_limit_m:=1200.0`
- 저장소 HEAD: `893a5eb`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 17_08_46.ulg (meta.json 기록: 17_08_46.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.03998 x ulog + 1785172120.302` (앵커 4개, 최대 잔차 0.329s). 시뮬 클록이 벽시계보다 +4.0% 빠름/느림 — 상수 오프셋만 쓰면 3.03s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 130.2981s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 25.276s → disarmed 149.844s (비행 124.568s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.52s / 역천이 5.04s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 4건 / 전체 위반 75건 / 샘플 604개, 최대 70.4806m (167.8109 m/s). 경계 최대: 70.4806m@96.856s(HOLD), 3.7418m@72.472s(FOLLOWING), 3.5779m@72.272s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=43.7524 m/s² (4.4615g) @146.744s state=LANDING; 접지(disarm−5s) 제외 시 10.336 m/s² (1.054g) @96.928s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.0591 m/s² (0.21g), 13.6881→5.0499 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -91.5494° → 정렬 13.416s 소요, 최대 91.6216°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5584m vs transition_alt 50.0m → -0.8831% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.7379m → 최저 49.7379m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9944m, 평균편차 -0.5144m, 최대 \|편차\| 1.8625m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.2m 평균 0.07m (부호 -0.2~0.1m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 12건/8종, mavros.log 48건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.4011 |
| CLIMBING | +1.4s | 28.4971 |
| TRANSITION_FW | +29.9s | 18.4 |
| STREAMING | +48.3s | 0.1019 |
| FOLLOWING | +48.4s | 20.3995 |
| TRANSITION_MC | +68.8s | 5.3988 |
| HOLD | +74.2s | 11.1 |
| LANDING | +85.3s | 44.9999 |
| DONE | +130.3s | 5.4252 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.408 | 63.764 |
| 1 | TRANS_TO_FW | 69.172 | 2.52 |
| 4 | FW | 71.692 | 19.448 |
| 2 | TRANS_TO_MC | 91.14 | 5.04 |
| 3 | MC | 96.18 | 54.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785172282.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785172282.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785172282.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785172282.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785172282.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 2679, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785172282.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 2682, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785172178.3 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785172282.0 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785172126.3 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785172127.4 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785172125.4 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785172123.4 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785172130.9 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785172126.3 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785172141.2 |  | TM: RTT too high for timesync: 2112.76 ms. |
| mavros.log | ERROR | 2 | 1785172181.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785172134.9 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785172135.3 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785172142.1 |  | PR: request param #612 timeout, retries left 2, and 252 params still missing |
| mavros.log | WARN | 1 | 1785172283.2 |  | UAS Executor terminated |

## 장애주입 결과

- `probe` spec={"on_state": "FOLLOWING", "delay_s": 3.0, "action": "probe", "topic": "/mavros/state"} → 발화 +53.095s rc=0

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
