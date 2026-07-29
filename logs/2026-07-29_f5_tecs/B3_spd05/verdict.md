# B3 — 판정

- 목적: 직각 코너(90°) — 코너 오버슈트 정량화
- 실행: 2026-07-29T04:28:11.735255+00:00 ~ 2026-07-29T04:31:14.830155+00:00 (경과 174.8s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 250.0,250.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 04_28_24.ulg (meta.json 기록: 04_28_24.ulg)
- 요약: FAIL 2, PASS 9, WARN 2

- 시각 정렬: `wall = 1.05087 x ulog + 1785299297.876` (앵커 4개, 최대 잔차 0.363s). 시뮬 클록이 벽시계보다 +5.1% 빠름/느림 — 상수 오프셋만 쓰면 4.06s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 139.9991s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 29.408s → disarmed 161.428s (비행 132.02s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.516s / 역천이 5.016s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 4건 / 전체 위반 139건 / 샘플 687개, 최대 11.7688m (58.8438 m/s). 경계 최대: 3.7141m@71.532s(FOLLOWING), 3.6734m@71.332s(FOLLOWING), 3.4969m@71.132s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=57.361 m/s² (5.8492g) @158.304s state=LANDING; 접지(disarm−5s) 제외 시 8.3489 m/s² (0.8514g) @108.796s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2235 m/s² (0.2267g), 13.766→5.046 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -101.9894° → 정렬 7.98s 소요, 최대 101.9894°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.602m vs transition_alt 50.0m → -0.796% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.2736m → 최저 50.2736m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9305m, 평균편차 -0.458m, 최대 \|편차\| 1.8555m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 13.4m 평균 3.6882m (부호 1.2~13.4m, n=17) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 13건/9종, mavros.log 47건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.4019 |
| CLIMBING | +1.4s | 29.0971 |
| TRANSITION_FW | +30.5s | 13.2001 |
| STREAMING | +43.7s | 0.1022 |
| FOLLOWING | +43.8s | 34.2992 |
| TRANSITION_MC | +78.1s | 5.1993 |
| HOLD | +83.3s | 10.5998 |
| LANDING | +93.9s | 46.0995 |
| DONE | +140.0s | 5.7706 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.372 | 62.672 |
| 1 | TRANS_TO_FW | 68.044 | 2.516 |
| 4 | FW | 70.56 | 32.976 |
| 2 | TRANS_TO_MC | 103.536 | 5.016 |
| 3 | MC | 108.552 | 53.004 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785299474.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785299474.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785299474.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785299474.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785299474.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 38912, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785299474.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 38915, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 1 | 1785299361.0 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | ≈1785299328.2 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 37.445m is large. affine correction guarantees WP passage but curve may be deformed |
| node.log | WARN | 1 | ≈1785299474.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785299304.3 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785299305.4 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785299303.3 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785299301.4 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785299308.8 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785299304.2 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 3 | 1785299358.1 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 2 | 1785299316.8 |  | TM: RTT too high for timesync: 1005.23 ms. |
| mavros.log | WARN | 1 | 1785299311.6 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785299311.8 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785299475.0 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
