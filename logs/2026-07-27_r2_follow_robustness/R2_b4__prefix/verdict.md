# R2_b4 — 판정

- 목적: R2 핵심: B4 U턴(135°) — 두 레그가 서로 가까워지는 구간. 전역탐색이 반대 레그를 잡을 수 있던 조건
- 실행: 2026-07-27T13:49:14.887343+00:00 ~ 2026-07-27T13:53:19.966321+00:00 (경과 223.9s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 100.0,150.0,50.0] range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 13_49_26.ulg (meta.json 기록: 13_49_26.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.14896 x ulog + 1785160156.643` (앵커 4개, 최대 잔차 0.516s). 시뮬 클록이 벽시계보다 +14.9% 빠름/느림 — 상수 오프셋만 쓰면 10.72s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 144.9995s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 80.16s → disarmed 210.692s (비행 130.532s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.452s / 역천이 5.232s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 13건 / 전체 위반 120건 / 샘플 930개, 최대 151.0799m (555.3793 m/s). 경계 최대: 151.0799m@127.812s(FOLLOWING), 62.3693m@153.832s(TRANSITION_MC), 3.5563m@128.42s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=83.7924 m/s² (8.5445g) @207.596s state=DONE; 접지(disarm−5s) 제외 시 5.9356 m/s² (0.6053g) @143.772s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.1967 m/s² (0.224g), 13.8039→5.0241 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -93.4529° → 정렬 14.86s 소요, 최대 93.5043°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.4193m vs transition_alt 50.0m → -1.1614% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.239m → 최저 50.239m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.1306m, 평균편차 -1.1302m, 최대 \|편차\| 7.3887m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 11.4m 평균 1.65m (부호 -4.4~11.4m, n=14) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 14건/10종, mavros.log 51건/13종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.6085 |
| CLIMBING | +0.6s | 31.5903 |
| TRANSITION_FW | +32.2s | 22.4999 |
| STREAMING | +54.7s | 0.1083 |
| FOLLOWING | +54.8s | 29.5061 |
| TRANSITION_MC | +84.3s | 5.3857 |
| HOLD | +89.7s | 8.9007 |
| LANDING | +98.6s | 46.4 |
| DONE | +145.0s | 5.9701 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.264 | 119.74 |
| 1 | TRANS_TO_FW | 125.004 | 2.452 |
| 4 | FW | 127.456 | 26.38 |
| 2 | TRANS_TO_MC | 153.836 | 5.232 |
| 3 | MC | 159.068 | 52.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785160399.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785160399.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785160399.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785160399.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785160399.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1168, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785160399.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1170, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785160283.0 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785160317.8 |  | 세그먼트 인덱스 급변 219→285 (Δ+66, 전체 472) pos=[215.5,15.9] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785160247.6 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.776m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785160399.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785160167.0 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785160168.0 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785160166.0 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785160164.2 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785160171.2 |  | VER: command plugin service call failed! |
| mavros.log | ERROR | 4 | 1785160219.4 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 3 | 1785160166.9 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785160179.8 |  | TM: RTT too high for timesync: 1801.10 ms. |
| mavros.log | WARN | 1 | 1785160173.8 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785160174.9 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785160180.7 |  | PR: request param #396 timeout, retries left 2, and 443 params still missing |
| mavros.log | WARN | 1 | 1785160182.1 |  | PR: Failed to get parameter type: NAV_DLL_ACT |
| mavros.log | WARN | 1 | 1785160403.5 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
