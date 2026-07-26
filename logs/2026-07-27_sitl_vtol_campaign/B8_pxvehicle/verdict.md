# B8 — 판정

- 목적: 후방 경로(초기 헤딩과 180° 반대) — 헤딩 정렬 P제어 최악조건
- 실행: 2026-07-26T19:40:09.215553+00:00 ~ 2026-07-26T19:43:10.809063+00:00 (경과 164.0s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, -300.0,0.0,50.0]`
- 저장소 HEAD: `3b52ac1`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 19_40_24.ulg (meta.json 기록: 19_40_24.ulg)
- 요약: FAIL 2, PASS 9, WARN 2

- 시각 정렬: `wall = 1.12626 x ulog + 1785094815.240` (앵커 4개, 최대 잔차 1.621s). 시뮬 클록이 벽시계보다 +12.6% 빠름/느림 — 상수 오프셋만 쓰면 7.78s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 141.7018s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 24.824s → disarmed 150.936s (비행 126.112s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.556s / 역천이 5.364s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 15건 / 전체 위반 89건 / 샘플 653개, 최대 219.2544m (1074.7763 m/s). 경계 최대: 5.3732m@73.248s(FOLLOWING), 3.8328m@72.748s(TRANSITION_FW), 3.7526m@72.548s(TRANSITION_FW). 스트림 재개 갭 1건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=57.9994 m/s² (5.9143g) @147.8s state=LANDING; 접지(disarm−5s) 제외 시 6.5764 m/s² (0.6706g) @72.524s state=TRANSITION_FW | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.4207 m/s² (0.2468g), 14.0435→5.0361 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 86.967° → 정렬 13.332s 소요, 최대 87.212°, tol 진입 후 재증가 0.0779 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.4822m vs transition_alt 50.0m → -1.0355% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.4371m → 최저 49.4371m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9759m, 평균편차 -0.4027m, 최대 \|편차\| 2.1559m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 2.6m 평균 0.77m (부호 -2.6~0.4m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 11건/7종, mavros.log 49건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.7126 |
| CLIMBING | +1.7s | 31.8897 |
| TRANSITION_FW | +33.6s | 21.2997 |
| STREAMING | +54.9s | 0.1074 |
| FOLLOWING | +55.0s | 18.707 |
| TRANSITION_MC | +73.7s | 5.5857 |
| HOLD | +79.3s | 12.3001 |
| LANDING | +91.6s | 50.0996 |
| DONE | +141.7s | 5.8585 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.304 | 64.036 |
| 1 | TRANS_TO_FW | 69.34 | 2.556 |
| 4 | FW | 71.896 | 18.944 |
| 2 | TRANS_TO_MC | 90.84 | 5.364 |
| 3 | MC | 96.204 | 55.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785094990.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785094990.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785094990.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785094990.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785094990.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1228, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785094990.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1230, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | ≈1785094990.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785094824.6 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785094825.7 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785094823.7 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785094821.8 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785094828.8 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785094824.6 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785094837.4 |  | TM: RTT too high for timesync: 1964.77 ms. |
| mavros.log | ERROR | 3 | 1785094877.1 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785094831.4 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785094831.5 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785094838.3 |  | PR: request param #398 timeout, retries left 2, and 438 params still missing |
| mavros.log | WARN | 1 | 1785094991.0 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
