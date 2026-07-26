# B5 — 판정

- 목적: 사각 폐곡선(시점≈종점) — 종점 근접 오판 여부
- 실행: 2026-07-26T20:08:56.530281+00:00 ~ 2026-07-26T20:17:17.913034+00:00 (경과 456.5s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0, 0.0,200.0,50.0, 0.0,20.0,50.0]`
- 저장소 HEAD: `3b52ac1`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 20_09_08.ulg (meta.json 기록: 20_09_08.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.11011 x ulog + 1785096542.691` (앵커 4개, 최대 잔차 1.126s). 시뮬 클록이 벽시계보다 +11.0% 빠름/느림 — 상수 오프셋만 쓰면 10.56s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 165.3273s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 287.932s → disarmed 439.432s (비행 151.5s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.584s / 역천이 5.288s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 2건 / 전체 위반 212건 / 샘플 2061개, 최대 113.824m (580.7346 m/s). 경계 최대: 113.824m@387.144s(HOLD), 2.5065m@386.352s(TRANSITION_MC). 스트림 재개 갭 3건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=48.1628 m/s² (4.9112g) @436.3s state=LANDING; 접지(disarm−5s) 제외 시 6.0836 m/s² (0.6204g) @336.532s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2146 m/s² (0.2258g), 14.3652→5.0326 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -99.5393° → 정렬 13.24s 소요, 최대 99.5405°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.953m vs transition_alt 50.0m → -0.0939% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.2189m → 최저 50.2178m (손실 0.0012m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.3251m, 평균편차 -0.7295m, 최대 \|편차\| 6.8157m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 8.6m 평균 2.4087m (부호 -0.4~8.6m, n=23) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 12건/8종, mavros.log 56건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.8381 |
| CLIMBING | +0.8s | 32.2898 |
| TRANSITION_FW | +33.1s | 18.499 |
| STREAMING | +51.6s | 0.1154 |
| FOLLOWING | +51.7s | 52.4329 |
| TRANSITION_MC | +104.2s | 5.4523 |
| HOLD | +109.6s | 8.7999 |
| LANDING | +118.4s | 46.9001 |
| DONE | +165.3s | 9.6171 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.352 | 328.004 |
| 1 | TRANS_TO_FW | 333.356 | 2.584 |
| 4 | FW | 335.94 | 45.728 |
| 2 | TRANS_TO_MC | 381.668 | 5.288 |
| 3 | MC | 386.956 | 53.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785097037.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785097037.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785097037.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785097037.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785097037.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1130, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785097037.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1132, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | ≈1785096861.6 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 4.983m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785097037.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785096548.4 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785096549.7 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | ERROR | 9 | 1785096601.3 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 5 | 1785096547.7 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785096545.9 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785096552.9 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785096548.7 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785096561.5 |  | TM: RTT too high for timesync: 2028.20 ms. |
| mavros.log | WARN | 2 | 1785096555.8 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785096555.6 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785096562.4 |  | PR: request param #313 timeout, retries left 2, and 486 params still missing |
| mavros.log | WARN | 1 | 1785097038.1 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
