# R1_c10_entry — 판정

- 목적: R1 ①: C10 재현 — ENTRY 무한대기(실측 432.67s / 5.85km)가 entry_timeout 에 걸려 안전 폴백되는가. 거리 상한은 꺼서(5000) ENTRY 타임아웃만 격리
- 실행: 2026-07-27T14:48:01.130977+00:00 ~ 2026-07-27T14:50:31.025684+00:00 (경과 144.1s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local entry_mode:=mid_flight range_limit_m:=5000.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 14_48_12.ulg (meta.json 기록: 14_48_12.ulg)
- 요약: FAIL 3, NULL 3, PASS 6, WARN 1

- 시각 정렬: `wall = 1.04156 x ulog + 1785163687.098` (앵커 3개, 최대 잔차 0.007s). 시뮬 클록이 벽시계보다 +4.2% 빠름/느림 — 상수 오프셋만 쓰면 1.85s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → ENTRY → OVERRIDE → DONE, 소요 113.8996s | DONE 상태 도달 |
| disarm 확인 | **FAIL** | 로그 끝까지 disarm 되지 않음 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → ENTRY → OVERRIDE → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **FAIL** | seq=[3, 1, 4], 정천이 2.556s / 역천이 Nones | 3→1→4, 4→2→3 |
| setpoint 점프 | **PASS** | 임계 1.5m, 경계±1s 위반 0건 / 전체 위반 0건 / 샘플 229개, 최대 0.648m (3.0 m/s). 경계 최대: -. 스트림 재개 갭 1건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=7.5044 m/s² (0.7652g) @70.412s state=ENTRY; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **NULL** | 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다 | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -92.8296° → 정렬 13.552s 소요, 최대 93.0158°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5547m vs transition_alt 50.0m → -0.8906% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 48.5711m → 최저 48.544m (손실 0.0271m) | ≤ 5m |
| 순항 고도편차 | **NULL** | FOLLOWING 상태창 또는 순항고도 기준을 알 수 없음 | ±3m |
| FW cte | **NULL** | node.log 에 'FOLLOWING tick= ... cte=' 샘플이 없음 (FOLLOWING 미진입이거나 20틱 미만 체류) | 직선 ≤ 2m |
| 경고/타임아웃 | **WARN** | node.log 26건/14종, mavros.log 40건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.3072 |
| CLIMBING | +0.3s | 29.9925 |
| TRANSITION_FW | +30.3s | 18.8999 |
| STREAMING | +49.2s | 0.1098 |
| ENTRY | +49.3s | 62.6913 |
| OVERRIDE | +112.0s | 1.8989 |
| DONE | +113.9s | 5.0388 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.28 | 61.932 |
| 1 | TRANS_TO_FW | 67.212 | 2.556 |
| 4 | FW | 69.768 | 69.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 1 | 1785163822.6 |  | ENTRY 타임아웃 60s 초과 (체류 60.0s) → 안전 폴백(OVERRIDE) 실행 |
| node.log | ERROR | 2 | ≈1785163829.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785163829.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785163829.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785163829.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785163829.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1125, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785163829.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1127, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 9 | 1785163709.7 |  | /mavros/cmd/arming 서비스 없음 |
| node.log | WARN | 1 | 1785163710.8 |  | home_position AMSL 미수렴(최근 2개: ['0.0', '0.0'], tol=0.5) — 이륙 보류, 수렴 대기 |
| node.log | WARN | 1 | 1785163743.0 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785163822.6 |  | 긴급 수동 전환 실행 → MANUAL 요청 |
| node.log | WARN | 1 | 1785163823.6 |  | 수동 모드(MANUAL) 미진입 (mode=OFFBOARD) -> AUTO.LOITER 안전 폴백 요청 |
| node.log | WARN | 1 | 1785163824.5 |  | 수동/안전 모드 진입 확인 (mode=AUTO.LOITER) -> DONE |
| node.log | WARN | 1 | ≈1785163829.5 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 10 | 1785163693.0 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 8 | 1785163694.1 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785163692.3 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785163690.4 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785163697.4 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785163693.3 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 2 | 1785163746.0 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785163701.3 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785163701.5 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785163706.1 |  | TM: RTT too high for timesync: 591.92 ms. |
| mavros.log | WARN | 1 | 1785163831.2 |  | UAS Executor terminated |

## 미산출 지표 (null)

- **역천이 감속률**: 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다
- **순항 고도편차**: FOLLOWING 상태창 또는 순항고도 기준을 알 수 없음
- **FW cte**: node.log 에 'FOLLOWING tick= ... cte=' 샘플이 없음 (FOLLOWING 미진입이거나 20틱 미만 체류)

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
