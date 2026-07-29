# B3 — 판정

- 목적: 직각 코너(90°) — 코너 오버슈트 정량화
- 실행: 2026-07-29T04:31:28.291953+00:00 ~ 2026-07-29T04:34:40.619134+00:00 (경과 182.9s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 250.0,250.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 04_31_42.ulg (meta.json 기록: 04_31_42.ulg)
- 요약: FAIL 2, PASS 9, WARN 2

- 시각 정렬: `wall = 1.04917 x ulog + 1785299495.771` (앵커 4개, 최대 잔차 0.396s). 시뮬 클록이 벽시계보다 +4.9% 빠름/느림 — 상수 오프셋만 쓰면 3.99s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 143.1988s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 34.072s → disarmed 169.496s (비행 135.424s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.544s / 역천이 4.916s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 6건 / 전체 위반 142건 / 샘플 729개, 최대 70.5153m (352.5763 m/s). 경계 최대: 70.5153m@113.968s(HOLD), 3.7551m@76.616s(FOLLOWING), 3.754m@76.412s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=37.9012 m/s² (3.8648g) @166.36s state=LANDING; 접지(disarm−5s) 제외 시 13.2442 m/s² (1.3505g) @113.932s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3062 m/s² (0.2352g), 13.7239→5.0527 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -101.8365° → 정렬 7.668s 소요, 최대 102.0085°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.6511m vs transition_alt 50.0m → -0.6978% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.8672m → 최저 49.8672m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9998m, 평균편차 -0.1436m, 최대 \|편차\| 1.8573m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 12.8m 평균 3.6706m (부호 0.9~12.8m, n=17) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 13건/9종, mavros.log 50건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.9006 |
| CLIMBING | +1.9s | 29.2978 |
| TRANSITION_FW | +31.2s | 12.9001 |
| STREAMING | +44.1s | 0.1033 |
| FOLLOWING | +44.2s | 34.3004 |
| TRANSITION_MC | +78.5s | 5.1971 |
| HOLD | +83.7s | 12.0993 |
| LANDING | +95.8s | 47.4002 |
| DONE | +143.2s | 4.6164 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.528 | 67.588 |
| 1 | TRANS_TO_FW | 73.116 | 2.544 |
| 4 | FW | 75.66 | 33.012 |
| 2 | TRANS_TO_MC | 108.672 | 4.916 |
| 3 | MC | 113.588 | 56.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785299678.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785299678.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785299678.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785299678.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785299678.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 40635, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785299678.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 40637, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 1 | 1785299564.4 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785299531.0 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 37.445m is large. affine correction guarantees WP passage but curve may be deformed |
| node.log | WARN | 1 | ≈1785299678.9 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785299502.6 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785299503.6 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785299501.2 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785299499.2 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785299507.1 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785299503.1 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785299515.7 |  | TM: RTT too high for timesync: 1924.83 ms. |
| mavros.log | ERROR | 3 | 1785299556.2 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785299509.4 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785299509.9 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785299517.0 |  | PR: request param #547 timeout, retries left 2, and 193 params still missing |
| mavros.log | WARN | 1 | 1785299680.8 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
