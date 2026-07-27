# R2_b3 — 판정

- 목적: R2 회귀: B3 직각 코너(90°) — 코너 오버슈트 구간의 세그먼트 선택
- 실행: 2026-07-27T14:12:04.878154+00:00 ~ 2026-07-27T14:16:17.831635+00:00 (경과 228.3s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 250.0,250.0,50.0] range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 14_12_16.ulg (meta.json 기록: 14_12_16.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.08809 x ulog + 1785161531.922` (앵커 4개, 최대 잔차 0.308s). 시뮬 클록이 벽시계보다 +8.8% 빠름/느림 — 상수 오프셋만 쓰면 7.00s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 154.2995s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 74.716s → disarmed 214.328s (비행 139.612s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.572s / 역천이 5.048s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 3건 / 전체 위반 138건 / 샘플 945개, 최대 300.1013m (1415.572 m/s). 경계 최대: 300.1013m@121.908s(FOLLOWING), 3.6808m@122.1s(FOLLOWING), 1.8533m@122.2s(FOLLOWING). 스트림 재개 갭 3건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=60.5132 m/s² (6.1706g) @211.22s state=LANDING; 접지(disarm−5s) 제외 시 10.719 m/s² (1.093g) @158.38s state=TRANSITION_MC | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3038 m/s² (0.2349g), 14.0242→5.2535 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -93.7394° → 정렬 12.48s 소요, 최대 93.8611°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.8147m vs transition_alt 50.0m → -0.3706% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.2968m → 최저 49.2957m (손실 0.0011m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0931m, 평균편차 -0.6748m, 최대 \|편차\| 5.2394m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 6.4m 평균 1.875m (부호 -0.5~6.4m, n=16) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 15건/11종, mavros.log 49건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.2035 |
| CLIMBING | +0.2s | 32.1956 |
| TRANSITION_FW | +32.4s | 18.2003 |
| STREAMING | +50.6s | 0.1066 |
| FOLLOWING | +50.7s | 34.5942 |
| TRANSITION_MC | +85.3s | 8.7991 |
| HOLD | +94.1s | 10.7 |
| LANDING | +104.8s | 49.5001 |
| DONE | +154.3s | 9.2176 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.284 | 113.716 |
| 1 | TRANS_TO_FW | 119.0 | 2.572 |
| 4 | FW | 121.572 | 31.54 |
| 2 | TRANS_TO_MC | 153.112 | 5.048 |
| 3 | MC | 158.16 | 57.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785161776.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785161776.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785161776.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785161776.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785161776.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1136, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785161776.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1138, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785161613.4 |  | home_position AMSL 미수렴(최근 2개: ['0.2', '0.2'], tol=0.5) — 이륙 보류, 수렴 대기 |
| node.log | WARN | 1 | 1785161647.8 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785161682.5 |  | 세그먼트 인덱스 급변 239→269 (Δ+30, 전체 510) pos=[235.6,16.0] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785161612.8 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 6.283m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785161776.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785161537.0 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785161538.1 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785161535.9 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785161534.1 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785161541.2 |  | VER: command plugin service call failed! |
| mavros.log | ERROR | 4 | 1785161590.7 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 3 | 1785161536.8 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785161549.9 |  | TM: RTT too high for timesync: 2079.25 ms. |
| mavros.log | WARN | 1 | 1785161543.9 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785161550.8 |  | PR: request param #469 timeout, retries left 2, and 377 params still missing |
| mavros.log | WARN | 1 | 1785161778.0 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
