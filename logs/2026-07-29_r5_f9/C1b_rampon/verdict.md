# C1b — 판정

- 목적: 천이고도 민감도 — 고(120m). 경로는 A1
- 실행: 2026-07-28T18:05:40.696382+00:00 ~ 2026-07-28T18:15:43.754438+00:00 (경과 570.5s)
- 종료: `timeout` (exit=2)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=120.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local range_limit_m:=1200.0`
- 저장소 HEAD: `94989b6`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 18_05_54.ulg (meta.json 기록: 18_05_54.ulg)
- 요약: FAIL 6, NULL 1, PASS 4, WARN 2

- 시각 정렬: `wall = 1.05931 x ulog + 1785261948.395` (앵커 3개, 최대 잔차 0.022s). 시뮬 클록이 벽시계보다 +5.9% 빠름/느림 — 상수 오프셋만 쓰면 3.95s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **FAIL** | 관측 상태: ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING; 종료사유=timeout | DONE 상태 도달 |
| disarm 확인 | **FAIL** | 로그 끝까지 disarm 되지 않음 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **FAIL** | seq=[3, 1, 4], 정천이 2.504s / 역천이 Nones | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 4건 / 전체 위반 97건 / 샘플 2695개, 최대 5.8617m (29.3083 m/s). 경계 최대: 3.6733m@95.488s(FOLLOWING), 3.6509m@95.688s(FOLLOWING), 3.548m@95.288s(FOLLOWING). 스트림 재개 갭 1건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=5661.4712 m/s² (577.3094g) @141.456s state=FOLLOWING; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **NULL** | 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다 | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -93.0308° → 정렬 14.932s 소요, 최대 93.1414°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 119.0109m vs transition_alt 120.0m → -0.8243% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 121.5796m → 최저 121.5796m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 49.9639m, 평균편차 -42.1802m, 최대 \|편차\| 71.8838m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 136.3m 평균 122.1517m (부호 -136.3~55.6m, n=236) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 23건/10종, mavros.log 84건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.0023 |
| CLIMBING | +1.0s | 51.9969 |
| TRANSITION_FW | +53.0s | 20.4996 |
| STREAMING | +73.5s | 0.11 |
| FOLLOWING | +73.6s | 494.2923 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 6.328 | 85.864 |
| 1 | TRANS_TO_FW | 92.192 | 2.504 |
| 4 | FW | 94.696 | 469.008 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785262543.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785262543.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785262543.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785262543.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785262543.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1321, exit code -2, cmd '/root/ws_r5b/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __node: |
| node.log | ERROR | 1 | ≈1785262543.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1323, exit code -2, cmd '/root/ws_r5b/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node:=o |
| node.log | WARN | 10 | 1785261974.1 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785262030.2 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785262046.0 |  | ⚠️ 천이고도와 경로고도가 71.6m 어긋난다 (transition_alt=120.0m vs 경로 순항고도, 현재 121.6m → 50.0m). 램프가 계단은 막지만 기체는 이 고도차를 순항 중에 메워야 한다 — 의도한 값인지 확인할 것 |
| node.log | WARN | 1 | ≈1785262543.1 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 29 | 1785261954.2 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 22 | 1785261955.2 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | ERROR | 11 | 1785262009.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 6 | 1785261953.1 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785261951.3 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785261958.9 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785261967.7 |  | TM: RTT too high for timesync: 2040.80 ms. |
| mavros.log | WARN | 2 | 1785261954.0 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 1 | 1785261961.7 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785261962.0 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785262544.0 |  | UAS Executor terminated |

## 미산출 지표 (null)

- **역천이 감속률**: 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
