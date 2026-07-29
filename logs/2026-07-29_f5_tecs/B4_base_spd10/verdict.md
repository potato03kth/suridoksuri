# B4 — 판정

- 목적: 예각/U턴(135°) — 선회반경 초과 시 거동
- 실행: 2026-07-29T03:15:27.472566+00:00 ~ 2026-07-29T03:18:36.917989+00:00 (경과 177.4s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 100.0,150.0,50.0]`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 03_15_44.ulg (meta.json 기록: 03_15_44.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.09777 x ulog + 1785294934.176` (앵커 4개, 최대 잔차 0.688s). 시뮬 클록이 벽시계보다 +9.8% 빠름/느림 — 상수 오프셋만 쓰면 6.38s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 138.6991s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 32.332s → disarmed 161.364s (비행 129.032s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.504s / 역천이 5.4s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 9건 / 전체 위반 103건 / 샘플 697개, 최대 151.0624m (377.656 m/s). 경계 최대: 151.0624m@74.94s(TRANSITION_FW), 69.5239m@106.272s(HOLD), 63.5054m@100.748s(TRANSITION_MC). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=69.4904 m/s² (7.086g) @158.192s state=LANDING; 접지(disarm−5s) 제외 시 13.4629 m/s² (1.3728g) @106.352s state=HOLD | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.0803 m/s² (0.2121g), 13.9621→5.0673 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -96.3843° → 정렬 9.572s 소요, 최대 96.6625°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.1574m vs transition_alt 50.0m → -1.6852% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.8502m → 최저 49.8502m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0714m, 평균편차 -1.2952m, 최대 \|편차\| 7.0702m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 13.5m 평균 2.5m (부호 -4.6~13.5m, n=13) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 15건/11종, mavros.log 50건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.0123 |
| CLIMBING | +1.0s | 30.1868 |
| TRANSITION_FW | +31.2s | 15.8997 |
| STREAMING | +47.1s | 0.1034 |
| FOLLOWING | +47.2s | 27.3999 |
| TRANSITION_MC | +74.6s | 5.6989 |
| HOLD | +80.3s | 11.3993 |
| LANDING | +91.7s | 46.9989 |
| DONE | +138.7s | 7.89 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 7.1 | 65.052 |
| 1 | TRANS_TO_FW | 72.152 | 2.504 |
| 4 | FW | 74.656 | 25.996 |
| 2 | TRANS_TO_MC | 100.652 | 5.4 |
| 3 | MC | 106.052 | 56.004 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785295116.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785295116.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785295116.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785295116.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785295116.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1826, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=of |
| node.log | ERROR | 1 | ≈1785295116.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1824, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node:= |
| node.log | WARN | 1 | 1785295002.9 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785295009.9 |  | altitude 메시지 지연 age=3.0s > 1.0s — 래치/캐시 의심, 수렴 표본에서 제외 |
| node.log | WARN | 1 | 1785295030.7 |  | 세그먼트 인덱스 급변 218→290 (Δ+72, 전체 476) pos=[213.7,15.0] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785294968.4 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.593m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785295116.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785294944.9 |  | FCU: EVENT 13426421 with args -252-0-126-0-17-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785294944.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785294942.9 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785294941.0 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785294949.1 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785294944.8 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785294957.2 |  | TM: RTT too high for timesync: 1646.48 ms. |
| mavros.log | ERROR | 3 | 1785294998.1 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785294951.9 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785294952.1 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785294958.8 |  | PR: request param #512 timeout, retries left 2, and 202 params still missing |
| mavros.log | WARN | 1 | 1785295117.3 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
