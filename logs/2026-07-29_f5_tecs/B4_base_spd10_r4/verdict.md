# B4 — 판정

- 목적: 예각/U턴(135°) — 선회반경 초과 시 거동
- 실행: 2026-07-29T05:11:47.778955+00:00 ~ 2026-07-29T05:14:52.791253+00:00 (경과 172.5s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 100.0,150.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 05_11_59.ulg (meta.json 기록: 05_11_59.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.07409 x ulog + 1785301913.522` (앵커 4개, 최대 잔차 0.360s). 시뮬 클록이 벽시계보다 +7.4% 빠름/느림 — 상수 오프셋만 쓰면 5.15s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 136.798s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 33.644s → disarmed 159.696s (비행 126.052s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.588s / 역천이 5.476s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 7건 / 전체 위반 100건 / 샘플 682개, 최대 150.499m (501.6632 m/s). 경계 최대: 150.499m@75.72s(FOLLOWING), 69.4128m@106.36s(HOLD), 61.9261m@100.576s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=83.7516 m/s² (8.5403g) @156.616s state=LANDING; 접지(disarm−5s) 제외 시 11.108 m/s² (1.1327g) @106.412s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.1535 m/s² (0.2196g), 14.4288→5.1005 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -95.584° → 정렬 8.052s 소요, 최대 95.584°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5161m vs transition_alt 50.0m → -0.9678% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.3108m → 최저 49.3094m (손실 0.0013m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.9232m, 평균편차 -1.069m, 최대 \|편차\| 7.388m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 12.9m 평균 4.7538m (부호 -5.9~12.9m, n=13) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 14건/10종, mavros.log 49건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.4019 |
| CLIMBING | +0.4s | 30.6992 |
| TRANSITION_FW | +31.1s | 13.2973 |
| STREAMING | +44.4s | 0.1018 |
| FOLLOWING | +44.5s | 27.4992 |
| TRANSITION_MC | +72.0s | 5.7991 |
| HOLD | +77.8s | 14.1008 |
| LANDING | +91.9s | 44.8987 |
| DONE | +136.8s | 6.129 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.328 | 67.3 |
| 1 | TRANS_TO_FW | 72.628 | 2.588 |
| 4 | FW | 75.216 | 25.364 |
| 2 | TRANS_TO_MC | 100.58 | 5.476 |
| 3 | MC | 106.056 | 54.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785302092.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785302092.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785302092.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785302092.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785302092.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 57780, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785302092.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 57782, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 1 | 1785301982.9 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785302010.1 |  | 세그먼트 인덱스 급변 219→289 (Δ+70, 전체 476) pos=[214.8,15.4] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785301948.7 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.593m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785302092.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785301919.9 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785301920.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785301919.3 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785301917.4 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785301924.2 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785301920.2 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785301935.3 |  | TM: RTT too high for timesync: 1900.77 ms. |
| mavros.log | ERROR | 3 | 1785301975.2 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785301926.6 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785301927.0 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785301936.2 |  | PR: request param #321 timeout, retries left 2, and 457 params still missing |
| mavros.log | WARN | 1 | 1785302093.0 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
