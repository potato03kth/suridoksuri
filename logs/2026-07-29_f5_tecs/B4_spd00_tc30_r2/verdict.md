# B4 — 판정

- 목적: 예각/U턴(135°) — 선회반경 초과 시 거동
- 실행: 2026-07-29T05:04:56.541326+00:00 ~ 2026-07-29T05:08:09.868050+00:00 (경과 180.1s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 100.0,150.0,50.0] range_limit_m:=800.0`
- 저장소 HEAD: `bc3229e`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 05_05_08.ulg (meta.json 기록: 05_05_08.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.08567 x ulog + 1785301501.644` (앵커 4개, 최대 잔차 1.446s). 시뮬 클록이 벽시계보다 +8.6% 빠름/느림 — 상수 오프셋만 쓰면 5.19s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 136.7997s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 39.928s → disarmed 166.26s (비행 126.332s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.548s / 역천이 5.464s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 12건 / 전체 위반 106건 / 샘플 728개, 최대 150.7024m (502.3413 m/s). 경계 최대: 150.7024m@82.244s(TRANSITION_FW), 69.4652m@112.9s(TRANSITION_MC), 3.7527m@82.444s(TRANSITION_FW). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=83.3927 m/s² (8.5037g) @163.16s state=LANDING; 접지(disarm−5s) 제외 시 12.6807 m/s² (1.2931g) @112.968s state=TRANSITION_MC | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2466 m/s² (0.2291g), 14.2374→5.0988 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -93.9493° → 정렬 8.316s 소요, 최대 93.9493°, tol 진입 후 재증가 0.0013 rad, 단조수렴=True | 단조수렴 + err ≤ 15.0° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5913m vs transition_alt 50.0m → -0.8174% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.0703m → 최저 49.0695m (손실 0.0008m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.0571m, 평균편차 -0.5434m, 최대 \|편차\| 3.9164m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 8.5m 평균 3.0615m (부호 -2.6~8.5m, n=13) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 14건/10종, mavros.log 48건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.9011 |
| CLIMBING | +0.9s | 30.8997 |
| TRANSITION_FW | +31.8s | 15.3983 |
| STREAMING | +47.2s | 0.1026 |
| FOLLOWING | +47.3s | 25.0998 |
| TRANSITION_MC | +72.4s | 7.8977 |
| HOLD | +80.3s | 11.6006 |
| LANDING | +91.9s | 44.8999 |
| DONE | +136.8s | 8.264 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.308 | 73.94 |
| 1 | TRANS_TO_FW | 79.248 | 2.548 |
| 4 | FW | 81.796 | 25.44 |
| 2 | TRANS_TO_MC | 107.236 | 5.464 |
| 3 | MC | 112.7 | 54.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785301689.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785301689.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785301689.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785301689.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785301689.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 54537, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785301689.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 54539, exit code -2, cmd '/root/ws_f5/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 1 | 1785301578.6 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785301605.2 |  | 세그먼트 인덱스 급변 219→289 (Δ+70, 전체 476) pos=[215.0,15.0] — 경로상 전진이 아니라 다른 레그 선택일 수 있다 |
| node.log | WARN | 1 | ≈1785301543.9 | stdout 중계(비-ROS 포맷) | [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.593m is large. affine correction guarantees WP passage but curve may be deformed. |
| node.log | WARN | 1 | ≈1785301689.8 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 13 | 1785301508.7 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785301509.7 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785301507.9 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785301506.0 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785301513.0 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785301508.8 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 3 | 1785301563.9 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 2 | 1785301524.1 |  | TM: RTT too high for timesync: 1773.61 ms. |
| mavros.log | WARN | 1 | 1785301518.0 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785301524.5 |  | PR: Failed to get parameter type: NAV_DLL_ACT |
| mavros.log | WARN | 1 | 1785301525.0 |  | PR: request param #524 timeout, retries left 2, and 300 params still missing |
| mavros.log | WARN | 1 | 1785301690.3 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
