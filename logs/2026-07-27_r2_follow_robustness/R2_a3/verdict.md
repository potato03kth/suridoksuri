# R2_a3 — 판정

- 목적: R2 회귀: A3 L자(90° 1회)
- 실행: 2026-07-27T14:07:43.745641+00:00 ~ 2026-07-27T14:11:47.145597+00:00 (경과 222.8s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0] range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 14_07_55.ulg (meta.json 기록: 14_07_55.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.11145 x ulog + 1785161268.667` (앵커 4개, 최대 잔차 1.832s). 시뮬 클록이 벽시계보다 +11.1% 빠름/느림 — 상수 오프셋만 쓰면 7.20s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 144.8988s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 78.692s → disarmed 208.78s (비행 130.088s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.508s / 역천이 4.94s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 11건 / 전체 위반 97건 / 샘플 908개, 최대 69.4889m (347.4444 m/s). 경계 최대: 3.7613m@127.32s(TRANSITION_FW), 3.7m@127.12s(TRANSITION_FW), 3.6703m@127.52s(TRANSITION_FW). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=78.6561 m/s² (8.0207g) @205.66s state=LANDING; 접지(disarm−5s) 제외 시 8.151 m/s² (0.8312g) @156.044s state=TRANSITION_MC | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2679 m/s² (0.2313g), 13.5917→5.0328 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -90.9951° → 정렬 13.74s 소요, 최대 90.9977°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.7332m vs transition_alt 50.0m → -0.5336% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.8697m → 최저 49.8617m (손실 0.008m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.2115m, 평균편차 -1.2231m, 최대 \|편차\| 5.1448m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 11.7m 평균 1.5231m (부호 0.1~11.7m, n=13) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 14건/10종, mavros.log 50건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.0031 |
| CLIMBING | +1.0s | 32.0961 |
| TRANSITION_FW | +33.1s | 21.9002 |
| STREAMING | +55.0s | 0.1095 |
| FOLLOWING | +55.1s | 24.1911 |
| TRANSITION_MC | +79.3s | 8.2994 |
| HOLD | +87.6s | 10.3999 |
| LANDING | +98.0s | 46.8997 |
| DONE | +144.9s | 5.7455 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.24 | 118.68 |
| 1 | TRANS_TO_FW | 123.92 | 2.508 |
| 4 | FW | 126.428 | 24.456 |
| 2 | TRANS_TO_MC | 150.884 | 4.94 |
| 3 | MC | 155.824 | 53.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785161506.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785161506.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785161506.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785161506.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785161506.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1146, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785161506.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1148, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785161391.0 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785161422.6 |  | 세그먼트 인덱스 급변 191→221 (Δ+30, 전체 414) pos=[185.2,15.6] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785161354.5 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 5.057m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785161506.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785161275.3 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785161276.7 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785161274.8 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785161273.0 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785161279.8 |  | VER: command plugin service call failed! |
| mavros.log | ERROR | 4 | 1785161328.2 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 3 | 1785161275.7 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785161288.4 |  | TM: RTT too high for timesync: 2087.77 ms. |
| mavros.log | WARN | 1 | 1785161282.5 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785161284.8 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785161289.4 |  | PR: request param #489 timeout, retries left 2, and 320 params still missing |
| mavros.log | WARN | 1 | 1785161507.3 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
