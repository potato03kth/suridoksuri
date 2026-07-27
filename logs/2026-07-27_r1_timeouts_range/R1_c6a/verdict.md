# R1_c6a — 판정

- 목적: R1 ④: 안전경로 회귀 — OVERRIDE(FW 순항)
- 실행: 2026-07-27T12:51:33.211309+00:00 ~ 2026-07-27T12:53:22.074693+00:00 (경과 98.2s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local range_limit_m:=1500.0`
- 저장소 HEAD: `3f6c517`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 12_51_46.ulg (meta.json 기록: 12_51_46.ulg)
- 요약: FAIL 3, NULL 1, PASS 8, WARN 1

- 시각 정렬: `wall = 1.09027 x ulog + 1785156703.249` (앵커 3개, 최대 잔차 0.172s). 시뮬 클록이 벽시계보다 +9.0% 빠름/느림 — 상수 오프셋만 쓰면 3.94s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → OVERRIDE → DONE, 소요 58.543s | DONE 상태 도달 |
| disarm 확인 | **FAIL** | 로그 끝까지 disarm 되지 않음 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → OVERRIDE → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **FAIL** | seq=[3, 1, 4], 정천이 2.524s / 역천이 Nones | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 13건 / 전체 위반 39건 / 샘플 284개, 최대 216.2771m (1039.7938 m/s). 경계 최대: 216.2771m@72.508s(FOLLOWING), 6.5457m@79.344s(FOLLOWING), 3.6982m@72.908s(FOLLOWING). 스트림 재개 갭 1건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **PASS** | 피크 \|az\|=4.3087 m/s² (0.4394g) @72.4s state=FOLLOWING; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **NULL** | 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다 | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -99.0739° → 정렬 15.672s 소요, 최대 99.4432°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 47.9781m vs transition_alt 50.0m → -4.0438% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 52.0091m → 최저 52.0091m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.0533m, 평균편차 0.4976m, 최대 \|편차\| 2.0884m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.9m 평균 0.44m (부호 -0.9~0.4m, n=5) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 15건/11종, mavros.log 48건/13종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.9484 |
| CLIMBING | +0.9s | 25.4945 |
| TRANSITION_FW | +26.4s | 22.1002 |
| STREAMING | +48.5s | 0.108 |
| FOLLOWING | +48.7s | 8.3141 |
| OVERRIDE | +57.0s | 1.5777 |
| DONE | +58.5s | 9.2479 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 6.484 | 63.148 |
| 1 | TRANS_TO_FW | 69.632 | 2.524 |
| 4 | FW | 72.156 | 17.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| node.log | ERROR | 2 | ≈1785156801.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785156801.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] Traceback (most recent call last): |
| node.log | ERROR | 2 | ≈1785156801.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [telemetry_node-1] KeyboardInterrupt |
| node.log | ERROR | 2 | ≈1785156801.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [offboard_node-2] KeyboardInterrupt |
| node.log | ERROR | 1 | ≈1785156801.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [telemetry_node-1]: process has died [pid 1188, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/telemetry_node --ros-args -r __nod |
| node.log | ERROR | 1 | ≈1785156801.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [ERROR] [offboard_node-2]: process has died [pid 1190, exit code -2, cmd '/root/drone_ws/install/fc_ros/lib/fc_ros/offboard_node --ros-args -r __node: |
| node.log | WARN | 1 | 1785156765.4 |  | 정렬 구간 OFFBOARD 이탈 → 재요청 (mode=AUTO.LOITER) |
| node.log | WARN | 1 | 1785156790.4 |  | 긴급 수동 전환 실행 → MANUAL 요청 |
| node.log | WARN | 1 | 1785156791.4 |  | 수동 모드(MANUAL) 미진입 (mode=OFFBOARD) -> AUTO.LOITER 안전 폴백 요청 |
| node.log | WARN | 1 | 1785156792.0 |  | 수동/안전 모드 진입 확인 (mode=AUTO.LOITER) -> DONE |
| node.log | WARN | 1 | ≈1785156801.2 | stdout 중계(비-ROS 포맷), 하니스 SIGINT 이후 | [WARNING] [launch]: user interrupted with ctrl-c (SIGINT) |
| mavros.log | ERROR | 11 | 1785156706.0 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 8 | 1785156707.0 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785156705.0 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 5 | 1785156718.8 |  | TM: RTT too high for timesync: 1738.47 ms. |
| mavros.log | WARN | 4 | 1785156703.2 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785156710.5 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785156706.8 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 2 | 1785156714.9 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785156713.2 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785156721.3 |  | WP: timeout, retries left 2 |
| mavros.log | WARN | 1 | 1785156722.3 |  | PR: request param #245 timeout, retries left 2, and 593 params still missing |
| mavros.log | ERROR | 1 | 1785156765.0 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785156802.3 |  | UAS Executor terminated |

## 장애주입 결과

- `override` spec={"on_state": "FOLLOWING", "delay_s": 8.0, "action": "override"} → 발화 +55.46s rc=0

## 미산출 지표 (null)

- **역천이 감속률**: 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
