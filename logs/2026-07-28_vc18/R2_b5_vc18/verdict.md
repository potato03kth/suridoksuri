# R2_b5 — 판정

- 목적: R2 핵심: B5 사각 폐곡선(시점≈종점, 20m 마진) — _find_segment 회귀 검증의 핵심
- 실행: 2026-07-27T15:34:15.786863+00:00 ~ 2026-07-27T15:41:12.359570+00:00 (경과 413.8s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0, 0.0,200.0,50.0, 0.0,20.0,50.0] range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 15_34_28.ulg (meta.json 기록: 15_34_28.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.01824 x ulog + 1785166461.158` (앵커 4개, 최대 잔차 0.206s). 시뮬 클록이 벽시계보다 +1.8% 빠름/느림 — 상수 오프셋만 쓰면 1.75s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 156.6996s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 243.0s → disarmed 396.376s (비행 153.376s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.504s / 역천이 5.1s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 5건 / 전체 위반 211건 / 샘플 1837개, 최대 85.9357m (421.2533 m/s). 경계 최대: 85.9357m@290.432s(FOLLOWING), 69.5249m@342.224s(HOLD), 3.7841m@290.828s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=78.3983 m/s² (7.9944g) @393.24s state=LANDING; 접지(disarm−5s) 제외 시 14.4407 m/s² (1.4725g) @342.296s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2433 m/s² (0.2288g), 13.9085→5.0308 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -96.4024° → 정렬 13.32s 소요, 최대 96.4024°, tol 진입 후 재증가 0.0545 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5659m vs transition_alt 50.0m → -0.8683% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 48.7133m → 최저 48.6998m (손실 0.0135m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0362m, 평균편차 -1.0092m, 최대 \|편차\| 7.0594m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 12.1m 평균 3.1583m (부호 -1.9~12.1m, n=24) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 15건/9종, mavros.log 64건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.3116 |
| CLIMBING | +0.3s | 29.388 |
| TRANSITION_FW | +29.7s | 18.1997 |
| STREAMING | +47.9s | 0.1039 |
| FOLLOWING | +48.0s | 47.4977 |
| TRANSITION_MC | +95.5s | 5.3996 |
| HOLD | +100.9s | 10.8994 |
| LANDING | +111.8s | 44.8997 |
| DONE | +156.7s | 5.3542 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 6.388 | 281.164 |
| 1 | TRANS_TO_FW | 287.552 | 2.504 |
| 4 | FW | 290.056 | 46.792 |
| 2 | TRANS_TO_MC | 336.848 | 5.1 |
| 3 | MC | 341.948 | 55.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785166870.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785166870.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785166870.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785166870.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785166870.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1102, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785166870.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1104, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 3 | 1785166768.7 |  | 세그먼트 인덱스 급변 191→222 (Δ+31, 전체 810) pos=[184.7,16.3] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785166706.7 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 4.976m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785166870.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 19 | 1785166468.4 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 14 | 1785166469.3 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | ERROR | 7 | 1785166523.3 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 6 | 1785166467.3 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785166465.3 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785166473.3 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785166469.3 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785166482.6 |  | TM: RTT too high for timesync: 1996.51 ms. |
| mavros.log | WARN | 1 | 1785166476.3 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785166478.4 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785166483.5 |  | PR: request param #678 timeout, retries left 2, and 255 params still missing |
| mavros.log | WARN | 1 | 1785166872.6 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
