# B4 — 판정

- 목적: 예각/U턴(135°) — 선회반경 초과 시 거동
- 실행: 2026-07-29T05:15:08.621925+00:00 ~ 2026-07-29T05:18:17.492479+00:00 (경과 175.2s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 100.0,150.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 05_15_20.ulg (meta.json 기록: 05_15_20.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.08107 x ulog + 1785302114.228` (앵커 4개, 최대 잔차 0.274s). 시뮬 클록이 벽시계보다 +8.1% 빠름/느림 — 상수 오프셋만 쓰면 5.44s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 136.5992s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 37.248s → disarmed 162.624s (비행 125.376s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.616s / 역천이 5.06s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 4건 / 전체 위반 101건 / 샘플 705개, 최대 69.4914m (347.4571 m/s). 경계 최대: 69.4914m@108.788s(HOLD), 3.867m@78.788s(FOLLOWING), 3.533m@78.588s(FOLLOWING). 스트림 재개 갭 6건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=44.7126 m/s² (4.5594g) @159.532s state=LANDING; 접지(disarm−5s) 제외 시 14.8046 m/s² (1.5096g) @108.86s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2146 m/s² (0.2258g), 13.9151→5.0308 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -96.7127° → 정렬 7.288s 소요, 최대 96.9243°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5918m vs transition_alt 50.0m → -0.8164% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.6407m → 최저 49.6406m (손실 0.0001m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.9676m, 평균편차 -0.5232m, 최대 \|편차\| 3.6447m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 4.3m 평균 1.7923m (부호 -2.3~4.3m, n=13) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 19건/10종, mavros.log 51건/13종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.7012 |
| CLIMBING | +0.7s | 30.5982 |
| TRANSITION_FW | +31.3s | 12.4998 |
| STREAMING | +43.8s | 0.1024 |
| FOLLOWING | +43.9s | 27.6986 |
| TRANSITION_MC | +71.6s | 5.2994 |
| HOLD | +76.9s | 14.4004 |
| LANDING | +91.3s | 45.2994 |
| DONE | +136.6s | 6.051 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.32 | 70.272 |
| 1 | TRANS_TO_FW | 75.592 | 2.616 |
| 4 | FW | 78.208 | 25.284 |
| 2 | TRANS_TO_MC | 103.492 | 5.06 |
| 3 | MC | 108.552 | 55.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785302297.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785302297.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785302297.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785302297.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785302297.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 59417, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | ERROR | 1 | ≈1785302297.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 59415, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | WARN | 6 | 1785302154.0 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785302213.9 |  | 세그먼트 인덱스 급변 218→290 (Δ+72, 전체 476) pos=[214.1,15.4] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785302153.3 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.593m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785302297.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785302120.8 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785302121.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785302119.5 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785302117.7 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785302125.1 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785302121.3 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785302133.7 |  | TM: RTT too high for timesync: 2089.44 ms. |
| mavros.log | ERROR | 3 | 1785302176.6 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785302127.5 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785302127.8 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785302133.5 |  | PR: Failed to get parameter type: NAV_DLL_ACT |
| mavros.log | WARN | 1 | 1785302134.7 |  | PR: request param #567 timeout, retries left 2, and 321 params still missing |
| mavros.log | WARN | 1 | 1785302297.7 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
