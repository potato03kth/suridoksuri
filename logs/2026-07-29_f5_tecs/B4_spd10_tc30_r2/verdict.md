# B4 — 판정

- 목적: 예각/U턴(135°) — 선회반경 초과 시 거동
- 실행: 2026-07-29T05:08:24.094594+00:00 ~ 2026-07-29T05:11:32.342945+00:00 (경과 176.6s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 100.0,150.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 05_08_35.ulg (meta.json 기록: 05_08_35.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.08757 x ulog + 1785301709.581` (앵커 4개, 최대 잔차 1.481s). 시뮬 클록이 벽시계보다 +8.8% 빠름/느림 — 상수 오프셋만 쓰면 5.36s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 138.4956s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 35.304s → disarmed 162.98s (비행 127.676s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.492s / 역천이 5.5s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 12건 / 전체 위반 102건 / 샘플 706개, 최대 150.8107m (502.7022 m/s). 경계 최대: 70.4344m@108.508s(TRANSITION_MC), 61.5451m@102.532s(TRANSITION_MC), 3.6514m@77.876s(TRANSITION_FW). 스트림 재개 갭 3건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=60.6099 m/s² (6.1805g) @159.796s state=LANDING; 접지(disarm−5s) 제외 시 11.3678 m/s² (1.1592g) @108.504s state=TRANSITION_MC | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.1102 m/s² (0.2152g), 14.5287→5.1258 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -97.8126° → 정렬 7.94s 소요, 최대 97.9217°, tol 진입 후 재증가 0.0272 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5803m vs transition_alt 50.0m → -0.8394% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.5143m → 최저 49.5071m (손실 0.0072m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0861m, 평균편차 -0.8548m, 최대 \|편차\| 5.9482m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 8.1m 평균 3.5615m (부호 -4.8~8.1m, n=13) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 13건/9종, mavros.log 50건/13종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.6979 |
| CLIMBING | +0.7s | 31.5983 |
| TRANSITION_FW | +32.3s | 14.9995 |
| STREAMING | +47.3s | 0.1019 |
| FOLLOWING | +47.4s | 25.0993 |
| TRANSITION_MC | +72.5s | 8.099 |
| HOLD | +80.6s | 11.7999 |
| LANDING | +92.4s | 46.0999 |
| DONE | +138.5s | 4.5429 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.348 | 69.196 |
| 1 | TRANS_TO_FW | 74.544 | 2.492 |
| 4 | FW | 77.036 | 25.5 |
| 2 | TRANS_TO_MC | 102.536 | 5.5 |
| 3 | MC | 108.036 | 55.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785301890.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785301890.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785301890.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785301890.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785301890.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 56189, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785301890.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 56191, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 1 | 1785301808.6 |  | 세그먼트 인덱스 급변 218→289 (Δ+71, 전체 476) pos=[214.6,15.2] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785301746.8 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.593m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785301890.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785301716.3 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785301717.4 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785301715.2 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785301713.3 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785301722.8 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785301716.1 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785301731.6 |  | TM: RTT too high for timesync: 2091.04 ms. |
| mavros.log | ERROR | 3 | 1785301771.6 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785301725.8 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785301726.2 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785301732.4 |  | PR: Failed to get parameter type: NAV_DLL_ACT |
| mavros.log | WARN | 1 | 1785301732.6 |  | PR: request param #499 timeout, retries left 2, and 367 params still missing |
| mavros.log | WARN | 1 | 1785301892.6 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
