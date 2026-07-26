# C6b — 판정

- 목적: 긴급 OVERRIDE — MC HOLD 중 /fc_ros/override true
- 실행: 2026-07-26T21:39:32.216293+00:00 ~ 2026-07-26T21:41:31.497085+00:00 (경과 117.4s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 21_39_44.ulg (meta.json 기록: 21_39_44.ulg)
- 요약: FAIL 3, PASS 9, WARN 1

- 시각 정렬: `wall = 1.02727 x ulog + 1785101978.419` (앵커 4개, 최대 잔차 0.157s). 시뮬 클록이 벽시계보다 +2.7% 빠름/느림 — 상수 오프셋만 쓰면 1.89s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → OVERRIDE → DONE, 소요 78.1989s | DONE 상태 도달 |
| disarm 확인 | **FAIL** | 로그 끝까지 disarm 되지 않음 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → OVERRIDE → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.544s / 역천이 4.936s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 7건 / 전체 위반 77건 / 샘플 409개, 최대 216.5155m (693.96 m/s). 경계 최대: 216.5155m@75.34s(FOLLOWING), 110.0028m@99.744s(HOLD), 3.7869m@75.94s(FOLLOWING). 스트림 재개 갭 3건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=6.0228 m/s² (0.6142g) @75.636s state=FOLLOWING; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2839 m/s² (0.2329g), 13.7409→5.0415 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -95.3789° → 정렬 14.12s 소요, 최대 95.429°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.4439m vs transition_alt 50.0m → -1.1122% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.2448m → 최저 50.2444m (손실 0.0004m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.0179m, 평균편차 -0.3473m, 최대 \|편차\| 2.3438m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.0m 평균 0.31m (부호 -1.0~0.2m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 23건/11종, mavros.log 42건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.3002 |
| CLIMBING | +0.3s | 28.8985 |
| TRANSITION_FW | +29.2s | 19.1 |
| STREAMING | +48.3s | 0.1122 |
| FOLLOWING | +48.4s | 19.9041 |
| TRANSITION_MC | +68.3s | 5.2844 |
| HOLD | +73.6s | 3.2568 |
| OVERRIDE | +76.9s | 1.3425 |
| DONE | +78.2s | 4.8156 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.4 | 67.044 |
| 1 | TRANS_TO_FW | 72.444 | 2.544 |
| 4 | FW | 74.988 | 19.396 |
| 2 | TRANS_TO_MC | 94.384 | 4.936 |
| 3 | MC | 99.32 | 13.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785102090.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785102090.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785102090.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785102090.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785102090.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1168, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785102090.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1170, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 9 | 1785102006.2 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785102083.9 |  | 긴급 수동 전환 실행 → POSCTL 요청 |
| node.log | WARN | 1 | 1785102084.9 |  | 수동 모드(POSCTL) 미진입 (mode=OFFBOARD) -> AUTO.LOITER 안전 폴백 요청 |
| node.log | WARN | 1 | 1785102085.3 |  | 수동/안전 모드 진입 확인 (mode=AUTO.LOITER) -> DONE |
| node.log | WARN | 1 | ≈1785102090.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 10 | 1785101984.5 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 8 | 1785101985.6 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785101983.7 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785101981.7 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785101989.2 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785101984.7 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 2 | 1785101998.4 |  | TM: RTT too high for timesync: 1795.95 ms. |
| mavros.log | ERROR | 2 | 1785102039.7 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785101992.1 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785101994.7 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785101999.9 |  | PR: request param #331 timeout, retries left 2, and 529 params still missing |
| mavros.log | WARN | 1 | 1785102091.9 |  | UAS Executor terminated |

## 장애주입 결과

- `override` spec={"on_state": "HOLD", "delay_s": 3.0, "action": "override"} → 발화 +79.02s rc=0

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
