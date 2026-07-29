# B4 — 판정

- 목적: 예각/U턴(135°) — 선회반경 초과 시 거동
- 실행: 2026-07-29T04:21:15.091441+00:00 ~ 2026-07-29T04:24:22.038603+00:00 (경과 177.9s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 100.0,150.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 04_21_26.ulg (meta.json 기록: 04_21_26.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.04880 x ulog + 1785298880.793` (앵커 4개, 최대 잔차 0.152s). 시뮬 클록이 벽시계보다 +4.9% 빠름/느림 — 상수 오프셋만 쓰면 3.34s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 135.7994s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 37.372s → disarmed 165.392s (비행 128.02s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.484s / 역천이 5.308s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 5건 / 전체 위반 101건 / 샘플 719개, 최대 151.1484m (503.8281 m/s). 경계 최대: 151.1484m@79.248s(FOLLOWING), 69.4882m@110.492s(HOLD), 3.5991m@79.848s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=36.7332 m/s² (3.7457g) @162.244s state=LANDING; 접지(disarm−5s) 제외 시 13.2565 m/s² (1.3518g) @110.552s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.1462 m/s² (0.2189g), 14.0005→5.0798 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -92.3664° → 정렬 7.384s 소요, 최대 92.4961°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5821m vs transition_alt 50.0m → -0.8359% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 48.9023m → 최저 48.8649m (손실 0.0374m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0135m, 평균편차 -0.7814m, 최대 \|편차\| 3.9415m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 13.1m 평균 3.0231m (부호 -2.1~13.1m, n=13) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 32건/10종, mavros.log 47건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.502 |
| CLIMBING | +0.5s | 30.5978 |
| TRANSITION_FW | +31.1s | 12.2997 |
| STREAMING | +43.4s | 0.1017 |
| FOLLOWING | +43.5s | 27.3007 |
| TRANSITION_MC | +70.8s | 5.5981 |
| HOLD | +76.4s | 12.8 |
| LANDING | +89.2s | 46.5995 |
| DONE | +135.8s | 5.4221 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.344 | 71.108 |
| 1 | TRANS_TO_FW | 76.452 | 2.484 |
| 4 | FW | 78.936 | 25.984 |
| 2 | TRANS_TO_MC | 104.92 | 5.308 |
| 3 | MC | 110.228 | 56.004 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785299061.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785299061.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785299061.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785299061.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785299061.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 35272, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785299061.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 35274, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 19 | 1785298918.1 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785298978.9 |  | 세그먼트 인덱스 급변 217→291 (Δ+74, 전체 476) pos=[213.3,16.2] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785298917.4 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.593m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785299061.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785298887.2 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785298888.2 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785298886.2 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785298884.3 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785298891.6 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785298887.1 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 3 | 1785298941.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 2 | 1785298901.6 |  | TM: RTT too high for timesync: 3166.92 ms. |
| mavros.log | WARN | 1 | 1785298893.8 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785298894.5 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785299062.2 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
