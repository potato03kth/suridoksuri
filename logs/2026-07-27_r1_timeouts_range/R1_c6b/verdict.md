# R1_c6b — 판정

- 목적: R1 ④: 안전경로 회귀 — OVERRIDE(MC HOLD)
- 실행: 2026-07-27T12:53:33.735136+00:00 ~ 2026-07-27T12:55:34.878774+00:00 (경과 110.3s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 12_53_46.ulg (meta.json 기록: 12_53_46.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.13050 x ulog + 1785156817.181` (앵커 4개, 최대 잔차 1.706s). 시뮬 클록이 벽시계보다 +13.0% 빠름/느림 — 상수 오프셋만 쓰면 8.01s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → OVERRIDE → DONE, 소요 86.8213s | DONE 상태 도달 |
| disarm 확인 | **FAIL** | 로그 끝까지 disarm 되지 않음 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → OVERRIDE → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.54s / 역천이 5.04s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 13건 / 전체 위반 78건 / 샘플 374개, 최대 110.5596m (552.7982 m/s). 경계 최대: 3.6887m@69.012s(TRANSITION_FW), 3.6606m@69.212s(TRANSITION_FW), 3.5514m@68.812s(TRANSITION_FW). 스트림 재개 갭 3건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **PASS** | 피크 \|az\|=4.7453 m/s² (0.4839g) @68.492s state=TRANSITION_FW; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3984 m/s² (0.2446g), 13.7426→5.0576 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -99.2273° → 정렬 13.384s 소요, 최대 99.2846°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.553m vs transition_alt 50.0m → -0.8941% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 51.7192m → 최저 51.7192m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9941m, 평균편차 -0.1441m, 최대 \|편차\| 1.9658m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.1m 평균 0.28m (부호 -1.1~0.1m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 15건/11종, mavros.log 42건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.9304 |
| CLIMBING | +0.9s | 31.491 |
| TRANSITION_FW | +32.4s | 21.6 |
| STREAMING | +54.0s | 0.1046 |
| FOLLOWING | +54.1s | 19.2021 |
| TRANSITION_MC | +73.3s | 8.8936 |
| HOLD | +82.2s | 3.086 |
| OVERRIDE | +85.3s | 1.5137 |
| DONE | +86.8s | 4.8922 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.668 | 60.036 |
| 1 | TRANS_TO_FW | 65.704 | 2.54 |
| 4 | FW | 68.244 | 19.436 |
| 2 | TRANS_TO_MC | 87.68 | 5.04 |
| 3 | MC | 92.72 | 11.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785156933.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785156933.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785156933.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785156933.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785156933.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1153, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785156933.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1155, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785156876.3 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785156927.1 |  | 긴급 수동 전환 실행 → POSCTL 요청 |
| node.log | WARN | 1 | 1785156928.0 |  | 수동 모드(POSCTL) 미진입 (mode=OFFBOARD) -> AUTO.LOITER 안전 폴백 요청 |
| node.log | WARN | 1 | 1785156928.6 |  | 수동/안전 모드 진입 확인 (mode=AUTO.LOITER) -> DONE |
| node.log | WARN | 1 | ≈1785156933.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 10 | 1785156826.6 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 8 | 1785156827.6 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785156825.9 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785156824.1 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785156831.1 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785156826.8 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785156838.5 |  | TM: RTT too high for timesync: 1147.25 ms. |
| mavros.log | ERROR | 2 | 1785156879.1 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785156833.7 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785156833.8 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785156935.1 |  | UAS Executor terminated |

## 장애주입 결과

- `override` spec={"on_state": "HOLD", "delay_s": 3.0, "action": "override"} → 발화 +76.123s rc=0

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
