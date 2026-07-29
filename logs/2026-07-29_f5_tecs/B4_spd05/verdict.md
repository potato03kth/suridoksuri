# B4 — 판정

- 목적: 예각/U턴(135°) — 선회반경 초과 시 거동
- 실행: 2026-07-29T03:19:01.278966+00:00 ~ 2026-07-29T03:22:13.035727+00:00 (경과 180.9s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 100.0,150.0,50.0]`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 03_19_17.ulg (meta.json 기록: 03_19_17.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.05487 x ulog + 1785295150.211` (앵커 4개, 최대 잔차 0.066s). 시뮬 클록이 벽시계보다 +5.5% 빠름/느림 — 상수 오프셋만 쓰면 3.63s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 130.0972s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 42.0s → disarmed 166.528s (비행 124.528s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.532s / 역천이 5.292s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 5건 / 전체 위반 101건 / 샘플 713개, 최대 150.892m (496.3551 m/s). 경계 최대: 150.892m@82.344s(FOLLOWING), 70.5178m@113.572s(HOLD), 3.6513m@82.744s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=43.9471 m/s² (4.4814g) @163.424s state=LANDING; 접지(disarm−5s) 제외 시 14.5216 m/s² (1.4808g) @113.536s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.1122 m/s² (0.2154g), 13.9032→5.0324 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -95.4289° → 정렬 7.444s 소요, 최대 95.438°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5374m vs transition_alt 50.0m → -0.9252% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.1922m → 최저 50.1894m (손실 0.0028m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.9068m, 평균편차 -1.0856m, 최대 \|편차\| 6.6959m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 13.4m 평균 3.1923m (부호 -4.0~13.4m, n=13) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 17건/12종, mavros.log 51건/13종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.8003 |
| CLIMBING | +0.8s | 28.6974 |
| TRANSITION_FW | +29.5s | 12.5999 |
| STREAMING | +42.1s | 0.1045 |
| FOLLOWING | +42.2s | 27.2972 |
| TRANSITION_MC | +69.5s | 5.4982 |
| HOLD | +75.0s | 12.7544 |
| LANDING | +87.8s | 42.3453 |
| DONE | +130.1s | 7.1191 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 6.708 | 72.74 |
| 1 | TRANS_TO_FW | 79.448 | 2.532 |
| 4 | FW | 81.98 | 25.996 |
| 2 | TRANS_TO_MC | 107.976 | 5.292 |
| 3 | MC | 113.268 | 54.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785295331.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785295331.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785295331.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785295331.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785295331.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 3480, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=of |
| node.log | ERROR | 1 | ≈1785295331.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 3478, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node:= |
| node.log | WARN | 2 | 1785295194.3 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785295226.1 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785295248.4 |  | altitude 메시지 지연 age=1.7s > 1.0s — 래치/캐시 의심, 수렴 표본에서 제외 |
| node.log | WARN | 1 | 1785295252.2 |  | 세그먼트 인덱스 급변 219→290 (Δ+71, 전체 476) pos=[214.9,16.1] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785295193.3 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.593m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785295331.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785295158.1 |  | FCU: EVENT 13426421 with args -252-0-126-0-17-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785295158.1 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785295155.5 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785295153.6 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785295162.0 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785295157.4 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785295170.9 |  | TM: RTT too high for timesync: 2109.11 ms. |
| mavros.log | ERROR | 3 | 1785295210.8 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785295164.8 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785295165.2 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785295171.9 |  | PR: request param #413 timeout, retries left 2, and 402 params still missing |
| mavros.log | WARN | 1 | 1785295171.9 |  | PR: Failed to get parameter type: NAV_DLL_ACT |
| mavros.log | WARN | 1 | 1785295333.2 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
