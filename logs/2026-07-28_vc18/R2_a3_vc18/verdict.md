# R2_a3 — 판정

- 목적: R2 회귀: A3 L자(90° 1회)
- 실행: 2026-07-27T15:30:12.613414+00:00 ~ 2026-07-27T15:34:01.697337+00:00 (경과 226.0s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0] range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 15_30_24.ulg (meta.json 기록: 15_30_24.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.01571 x ulog + 1785166218.619` (앵커 4개, 최대 잔차 0.235s). 시뮬 클록이 벽시계보다 +1.6% 빠름/느림 — 상수 오프셋만 쓰면 1.01s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 132.9991s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 82.06s → disarmed 212.484s (비행 130.424s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.448s / 역천이 4.796s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 6건 / 전체 위반 104건 / 샘플 918개, 최대 232.2388m (774.1294 m/s). 경계 최대: 232.2388m@128.984s(STREAMING), 70.4416m@158.584s(TRANSITION_MC), 5.1652m@129.88s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=43.9993 m/s² (4.4867g) @209.348s state=LANDING; 접지(disarm−5s) 제외 시 15.6802 m/s² (1.5989g) @158.556s state=TRANSITION_MC | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2577 m/s² (0.2302g), 13.5886→5.0574 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -91.2305° → 정렬 13.788s 소요, 최대 91.3768°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.4008m vs transition_alt 50.0m → -1.1984% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 48.9033m → 최저 48.9011m (손실 0.0023m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.8771m, 평균편차 -1.1011m, 최대 \|편차\| 4.7921m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 14.3m 평균 1.8846m (부호 -0.8~14.3m, n=13) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 13건/10종, mavros.log 49건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.9022 |
| CLIMBING | +0.9s | 27.8969 |
| TRANSITION_FW | +28.8s | 18.9 |
| STREAMING | +47.7s | 0.1031 |
| FOLLOWING | +47.8s | 24.3984 |
| TRANSITION_MC | +72.2s | 5.5994 |
| HOLD | +77.8s | 10.0998 |
| LANDING | +87.9s | 45.0992 |
| DONE | +133.0s | 6.7173 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.356 | 120.832 |
| 1 | TRANS_TO_FW | 126.188 | 2.448 |
| 4 | FW | 128.636 | 24.704 |
| 2 | TRANS_TO_MC | 153.34 | 4.796 |
| 3 | MC | 158.136 | 55.004 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785166441.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785166441.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785166441.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 1 | ≈1785166441.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785166441.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1134, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785166441.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1132, exit code 1, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node |
| node.log | WARN | 1 | 1785166332.8 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785166361.7 |  | 세그먼트 인덱스 급변 192→222 (Δ+30, 전체 414) pos=[185.9,16.4] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785166301.6 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 5.019m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785166441.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785166224.9 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785166225.8 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785166224.1 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785166222.1 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785166229.4 |  | VER: command plugin service call failed! |
| mavros.log | ERROR | 4 | 1785166279.4 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 3 | 1785166225.1 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 2 | 1785166238.6 |  | TM: RTT too high for timesync: 1745.66 ms. |
| mavros.log | WARN | 1 | 1785166232.4 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785166235.0 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785166239.5 |  | PR: request param #409 timeout, retries left 2, and 398 params still missing |
| mavros.log | WARN | 1 | 1785166442.3 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
