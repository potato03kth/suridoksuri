# C6a — 판정

- 목적: 긴급 OVERRIDE — FW 순항(FOLLOWING) 중 /fc_ros/override true
- 실행: 2026-07-26T21:37:32.211829+00:00 ~ 2026-07-26T21:39:13.456654+00:00 (경과 101.2s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 21_37_44.ulg (meta.json 기록: 21_37_44.ulg)
- 요약: FAIL 3, NULL 1, PASS 8, WARN 1

- 시각 정렬: `wall = 1.00986 x ulog + 1785101858.396` (앵커 3개, 최대 잔차 0.002s). 시뮬 클록이 벽시계보다 +1.0% 빠름/느림 — 상수 오프셋만 쓰면 0.44s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → OVERRIDE → DONE, 소요 57.5984s | DONE 상태 도달 |
| disarm 확인 | **FAIL** | 로그 끝까지 disarm 되지 않음 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → OVERRIDE → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **FAIL** | seq=[3, 1, 4], 정천이 2.544s / 역천이 Nones | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 10건 / 전체 위반 41건 / 샘플 312개, 최대 215.0606m (716.8686 m/s). 경계 최대: 215.0606m@78.46s(FOLLOWING), 3.7349m@78.868s(FOLLOWING), 3.6738m@78.664s(FOLLOWING). 스트림 재개 갭 1건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **PASS** | 피크 \|az\|=4.4134 m/s² (0.45g) @78.276s state=STREAMING; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **NULL** | 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다 | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -94.0153° → 정렬 14.072s 소요, 최대 94.0413°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.6096m vs transition_alt 50.0m → -0.7807% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 52.2231m → 최저 52.2231m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9876m, 평균편차 0.573m, 최대 \|편차\| 2.2709m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.8m 평균 0.46m (부호 -0.8~0.1m, n=5) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 27건/11종, mavros.log 42건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.1021 |
| CLIMBING | +1.1s | 27.5966 |
| TRANSITION_FW | +28.7s | 19.0001 |
| STREAMING | +47.7s | 0.1216 |
| FOLLOWING | +47.8s | 8.197 |
| OVERRIDE | +56.0s | 1.581 |
| DONE | +57.6s | 4.9454 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.432 | 70.04 |
| 1 | TRANS_TO_FW | 75.472 | 2.544 |
| 4 | FW | 78.016 | 17.004 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785101952.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785101952.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785101952.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785101952.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785101952.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1152, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785101952.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1154, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 13 | 1785101888.5 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785101945.7 |  | 긴급 수동 전환 실행 → MANUAL 요청 |
| node.log | WARN | 1 | 1785101946.7 |  | 수동 모드(MANUAL) 미진입 (mode=OFFBOARD) -> AUTO.LOITER 안전 폴백 요청 |
| node.log | WARN | 1 | 1785101947.3 |  | 수동/안전 모드 진입 확인 (mode=AUTO.LOITER) -> DONE |
| node.log | WARN | 1 | ≈1785101952.3 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 10 | 1785101864.6 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 8 | 1785101865.8 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785101863.8 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785101861.8 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785101869.3 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785101864.8 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 2 | 1785101873.9 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 2 | 1785101878.7 |  | TM: RTT too high for timesync: 1519.89 ms. |
| mavros.log | WARN | 1 | 1785101872.4 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785101879.6 |  | PR: request param #308 timeout, retries left 2, and 550 params still missing |
| mavros.log | ERROR | 1 | 1785101919.1 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785101953.9 |  | UAS Executor terminated |

## 장애주입 결과

- `override` spec={"on_state": "FOLLOWING", "delay_s": 8.0, "action": "override"} → 발화 +59.673s rc=0

## 미산출 지표 (null)

- **역천이 감속률**: 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
