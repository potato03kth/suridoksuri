# B4 — 판정

- 목적: 예각/U턴(135°) — 선회반경 초과 시 거동
- 실행: 2026-07-29T04:38:24.416692+00:00 ~ 2026-07-29T04:42:23.474121+00:00 (경과 226.7s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 100.0,150.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 04_38_36.ulg (meta.json 기록: 04_38_36.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.03061 x ulog + 1785299913.427` (앵커 4개, 최대 잔차 0.467s). 시뮬 클록이 벽시계보다 +3.1% 빠름/느림 — 상수 오프셋만 쓰면 1.71s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 136.1998s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 84.972s → disarmed 213.056s (비행 128.084s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.404s / 역천이 5.856s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 10건 / 전체 위반 91건 / 샘플 945개, 최대 151.1307m (497.1403 m/s). 경계 최대: 151.1307m@124.5s(TRANSITION_FW), 70.7232m@153.3s(TRANSITION_MC), 64.6952m@147.164s(TRANSITION_MC). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=44.7519 m/s² (4.5634g) @209.928s state=LANDING; 접지(disarm−5s) 제외 시 11.131 m/s² (1.135g) @153.352s state=TRANSITION_MC | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.5337 m/s² (0.2584g), 16.7334→6.1736 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -92.6413° → 정렬 9.164s 소요, 최대 92.7416°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 48.7837m vs transition_alt 50.0m → -2.4326% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 51.37m → 최저 51.37m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.6496m, 평균편차 2.9755m, 최대 \|편차\| 6.211m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 10.6m 평균 3.6667m (부호 -10.6~6.3m, n=12) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 33건/11종, mavros.log 57건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.1015 |
| CLIMBING | +1.1s | 25.2981 |
| TRANSITION_FW | +26.4s | 14.6 |
| STREAMING | +41.0s | 0.1022 |
| FOLLOWING | +41.1s | 22.5998 |
| TRANSITION_MC | +63.7s | 7.7986 |
| HOLD | +71.5s | 14.8 |
| LANDING | +86.3s | 49.8996 |
| DONE | +136.2s | 5.656 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.424 | 116.28 |
| 1 | TRANS_TO_FW | 121.704 | 2.404 |
| 4 | FW | 124.108 | 22.96 |
| 2 | TRANS_TO_MC | 147.068 | 5.856 |
| 3 | MC | 152.924 | 61.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785300142.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785300142.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785300142.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785300142.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785300142.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 44778, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785300142.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 44780, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 19 | 1785299999.1 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785300029.4 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785300054.4 |  | 세그먼트 인덱스 급변 222→285 (Δ+63, 전체 476) pos=[218.2,13.3] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785299996.9 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.593m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785300142.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 17 | 1785299917.0 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 14 | 1785299918.1 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785299915.7 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785299913.9 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785299921.6 |  | VER: command plugin service call failed! |
| mavros.log | ERROR | 4 | 1785299969.7 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 3 | 1785299917.7 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 2 | 1785299923.1 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785299924.6 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785299929.0 |  | TM: RTT too high for timesync: 417.70 ms. |
| mavros.log | WARN | 1 | 1785300143.7 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
