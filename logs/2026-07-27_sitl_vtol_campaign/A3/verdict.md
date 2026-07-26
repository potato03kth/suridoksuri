# A3 — 판정

- 목적: SITL-4 L자 경로 재현 (90° 1회)
- 실행: 2026-07-26T17:52:55.644162+00:00 ~ 2026-07-26T18:03:25.575339+00:00 (경과 579.3s)
- 종료: `timeout` (exit=2)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0]`
- 저장소 HEAD: `3b52ac1`
- ulog: 17_53_07.ulg (meta.json 기록: 17_53_07.ulg)
- 요약: FAIL 5, NULL 1, PASS 5, WARN 2

- 시각 정렬: `wall = 1.07588 x ulog + 1785088385.070` (앵커 3개, 최대 잔차 0.047s). 시뮬 클록이 벽시계보다 +7.6% 빠름/느림 — 상수 오프셋만 쓰면 3.26s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **FAIL** | 관측 상태: ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING; 종료사유=timeout | DONE 상태 도달 |
| disarm 확인 | **FAIL** | 로그 끝까지 disarm 되지 않음 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **FAIL** | seq=[3, 1, 4], 정천이 2.504s / 역천이 Nones | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 5건 / 전체 위반 61건 / 샘플 2751개, 최대 233.048m (1120.4229 m/s). 경계 최대: 233.048m@137.908s(FOLLOWING), 3.7404m@138.62s(FOLLOWING), 3.6538m@138.416s(FOLLOWING). 스트림 재개 갭 1건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=5.2038 m/s² (0.5306g) @138.428s state=FOLLOWING; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **NULL** | 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다 | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -91.4128° → 정렬 15.412s 소요, 최대 91.7138°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 48.4934m vs transition_alt 50.0m → -3.0131% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.1879m → 최저 50.1879m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.842m, 평균편차 0.0377m, 최대 \|편차\| 2.0866m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 6292.4m 평균 3045.2922m (부호 -6292.4~-0.9m, n=219) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 17건/9종, mavros.log 48건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.5514 |
| CLIMBING | +0.6s | 26.897 |
| TRANSITION_FW | +27.4s | 21.3001 |
| STREAMING | +48.7s | 0.1076 |
| FOLLOWING | +48.9s | 472.3151 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.32 | 129.888 |
| 1 | TRANS_TO_FW | 135.208 | 2.504 |
| 4 | FW | 137.712 | 435.024 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 5 | ≈1785089005.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 3 | ≈1785089005.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785089005.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785089005.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785089005.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1089, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785089005.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1091, exit code 1, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:= |
| node.log | WARN | 1 | 1785088730.3 |  | altitude 메시지 지연 age=3.2s > 1.0s — 래치/캐시 의심, 수렴 표본에서 제외 |
| node.log | WARN | 1 | ≈1785088483.0 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 5.057m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785089005.6 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 12 | 1785088440.7 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | ERROR | 8 | 1785088387.6 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785088388.7 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785088386.7 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785088384.9 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785088391.8 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785088400.1 |  | TM: RTT too high for timesync: 1909.38 ms. |
| mavros.log | WARN | 2 | 1785088387.6 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 1 | 1785088394.5 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785088401.1 |  | PR: Failed to get parameter type: NAV_DLL_ACT |
| mavros.log | WARN | 1 | 1785088401.3 |  | PR: request param #366 timeout, retries left 2, and 493 params still missing |
| mavros.log | WARN | 1 | 1785089006.2 |  | UAS Executor terminated |

## 미산출 지표 (null)

- **역천이 감속률**: 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
