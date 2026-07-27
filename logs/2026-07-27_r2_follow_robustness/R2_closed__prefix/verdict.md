# R2_closed — 판정

- 목적: R2 신설: **완전 폐회로**(종점 = 시점 = 이륙지점). 대회 경로 형상이 폐회로로 확정됐는데(2026-07-27 사용자) 캠페인에는 완전 폐회로가 없었다 — B5 는 종점을 20m 남겨 일부러 벌려 놓은 것이다. 마지막 레그가 첫 레그의 시작점으로 돌아오므로 전역 최근접 탐색이 인덱스를 0 근처로 되감을 수 있는 유일한 시나리오다
- 실행: 2026-07-27T13:53:36.407585+00:00 ~ 2026-07-27T14:00:52.731817+00:00 (경과 395.5s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0, 0.0,200.0,50.0, 0.0,0.0,50.0] range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 13_53_48.ulg (meta.json 기록: 13_53_48.ulg)
- 요약: FAIL 2, PASS 9, WARN 2

- 시각 정렬: `wall = 1.11115 x ulog + 1785160422.846` (앵커 4개, 최대 잔차 1.038s). 시뮬 클록이 벽시계보다 +11.1% 빠름/느림 — 상수 오프셋만 쓰면 10.74s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 168.899s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 228.764s → disarmed 379.032s (비행 150.268s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.484s / 역천이 4.504s | 3→1→4, 4→2→3 |
| setpoint 점프 | **PASS** | 임계 1.5m, 경계±1s 위반 0건 / 전체 위반 217건 / 샘플 1753개, 최대 83.3315m (236.7372 m/s). 경계 최대: -. 스트림 재개 갭 3건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=49.8333 m/s² (5.0816g) @375.964s state=LANDING; 접지(disarm−5s) 제외 시 5.231 m/s² (0.5334g) @300.324s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2908 m/s² (0.2336g), 13.4836→5.5438 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -95.7766° → 정렬 13.54s 소요, 최대 95.9412°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.6988m vs transition_alt 50.0m → -0.6024% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 51.3668m → 최저 51.3179m (손실 0.0489m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.9979m, 평균편차 -0.4508m, 최대 \|편차\| 6.4711m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 13.7m 평균 3.0m (부호 -2.9~13.7m, n=25) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 17건/11종, mavros.log 54건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.209 |
| CLIMBING | +0.2s | 30.7901 |
| TRANSITION_FW | +31.0s | 18.9003 |
| STREAMING | +49.9s | 0.1202 |
| FOLLOWING | +50.0s | 55.6109 |
| TRANSITION_MC | +105.6s | 4.6687 |
| HOLD | +110.3s | 7.9 |
| LANDING | +118.2s | 50.6996 |
| DONE | +168.9s | 6.4557 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.332 | 267.104 |
| 1 | TRANS_TO_FW | 272.436 | 2.484 |
| 4 | FW | 274.92 | 48.8 |
| 2 | TRANS_TO_MC | 323.72 | 4.504 |
| 3 | MC | 328.224 | 51.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785160852.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785160852.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785160852.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785160852.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785160852.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1157, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785160852.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1159, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 3 | 1785160742.7 |  | 세그먼트 인덱스 급변 191→220 (Δ+29, 전체 826) pos=[185.3,14.8] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | 1785160710.4 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785160736.1 |  | altitude 메시지 지연 age=3.5s > 1.0s — 래치/캐시 의심, 수렴 표본에서 제외 |
| node.log | WARN | 1 | ≈1785160676.7 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 5.051m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785160852.7 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785160428.4 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785160429.6 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | ERROR | 8 | 1785160481.4 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 5 | 1785160427.6 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785160425.8 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785160432.8 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785160428.5 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785160441.5 |  | TM: RTT too high for timesync: 1982.99 ms. |
| mavros.log | WARN | 1 | 1785160435.6 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785160441.9 |  | PR: Failed to get parameter type: NAV_DLL_ACT |
| mavros.log | WARN | 1 | 1785160442.4 |  | PR: request param #399 timeout, retries left 2, and 370 params still missing |
| mavros.log | WARN | 1 | 1785160852.9 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
