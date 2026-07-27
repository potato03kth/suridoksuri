# C2 — 판정

- 목적: 헤딩 정렬 90° 조건 (동쪽 경로)
- 실행: 2026-07-27T17:01:22.936323+00:00 ~ 2026-07-27T17:03:59.283821+00:00 (경과 150.9s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 0.0,300.0,50.0] range_limit_m:=1200.0`
- 저장소 HEAD: `893a5eb`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 17_01_34.ulg (meta.json 기록: 17_01_34.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.04406 x ulog + 1785171688.499` (앵커 4개, 최대 잔차 0.438s). 시뮬 클록이 벽시계보다 +4.4% 빠름/느림 — 상수 오프셋만 쓰면 2.97s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 119.7988s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 23.204s → disarmed 138.416s (비행 115.212s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.508s / 역천이 4.972s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 5건 / 전체 위반 76건 / 샘플 595개, 최대 216.7105m (722.3683 m/s). 경계 최대: 216.7105m@60.932s(FOLLOWING), 70.4682m@85.724s(HOLD), 3.6616m@61.332s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=39.9915 m/s² (4.078g) @135.292s state=LANDING; 접지(disarm−5s) 제외 시 11.7272 m/s² (1.1958g) @85.696s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.4374 m/s² (0.2485g), 13.5892→5.0328 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -4.5228° → 정렬 4.1s 소요, 최대 4.535°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.6982m vs transition_alt 50.0m → -0.6035% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.1377m → 최저 50.1377m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.123m, 평균편차 -0.3625m, 최대 \|편차\| 2.0561m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.4m 평균 0.2m (부호 -0.4~-0.1m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 13건/9종, mavros.log 45건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.407 |
| CLIMBING | +1.4s | 29.0913 |
| TRANSITION_FW | +30.5s | 8.9001 |
| STREAMING | +39.4s | 0.1031 |
| FOLLOWING | +39.5s | 20.4978 |
| TRANSITION_MC | +60.0s | 5.2993 |
| HOLD | +65.3s | 10.3001 |
| LANDING | +75.6s | 44.2001 |
| DONE | +119.8s | 5.9931 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.176 | 52.88 |
| 1 | TRANS_TO_FW | 58.056 | 2.508 |
| 4 | FW | 60.564 | 19.764 |
| 2 | TRANS_TO_MC | 80.328 | 4.972 |
| 3 | MC | 85.3 | 54.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785171838.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785171838.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785171838.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785171838.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785171838.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1367, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785171838.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1369, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785171744.9 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785171801.4 |  | altitude 메시지 지연 age=1.1s > 1.0s — 래치/캐시 의심, 수렴 표본에서 제외 |
| node.log | WARN | 1 | ≈1785171838.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785171694.8 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785171695.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785171694.2 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785171692.3 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785171699.3 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785171695.2 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 2 | 1785171747.8 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785171702.9 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785171703.4 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785171708.3 |  | TM: RTT too high for timesync: 671.13 ms. |
| mavros.log | WARN | 1 | 1785171839.5 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
