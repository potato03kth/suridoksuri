# A1 — 판정

- 목적: SITL-4 직선 300m 재현 (3주치 변경 후 회귀 여부). waypoint_frame=local(SITL-4 당시 동작)
- 실행: 2026-07-26T17:26:36.192355+00:00 ~ 2026-07-26T17:29:34.101401+00:00 (경과 162.6s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0] waypoint_frame:=local`
- 저장소 HEAD: `4e8e378`
- ulog: 17_26_47.ulg (meta.json 기록: 17_26_47.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.08601 x ulog + 1785086803.334` (앵커 4개, 최대 잔차 0.684s). 시뮬 클록이 벽시계보다 +8.6% 빠름/느림 — 상수 오프셋만 쓰면 6.21s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 137.5973s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 24.472s → disarmed 148.756s (비행 124.284s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.556s / 역천이 5.14s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 2건 / 전체 위반 73건 / 샘플 598개, 최대 216.1008m (683.8634 m/s). 경계 최대: 216.1008m@73.968s(FOLLOWING), 113.1633m@98.14s(HOLD). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=81.8044 m/s² (8.3417g) @145.648s state=LANDING; 접지(disarm−5s) 제외 시 9.1878 m/s² (0.9369g) @74.26s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2371 m/s² (0.2281g), 14.0326→5.0858 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -97.35° → 정렬 13.236s 소요, 최대 97.6442°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.6981m vs transition_alt 50.0m → -0.6038% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 48.4747m → 최저 48.4675m (손실 0.0072m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.9868m, 평균편차 -0.6094m, 최대 \|편차\| 2.4454m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.1m 평균 0.8m (부호 -1.1~-0.6m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 6건/1종, mavros.log 49건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.3084 |
| CLIMBING | +0.3s | 33.689 |
| TRANSITION_FW | +34.0s | 18.4998 |
| STREAMING | +52.5s | 0.1065 |
| FOLLOWING | +52.6s | 21.7055 |
| TRANSITION_MC | +74.3s | 5.3884 |
| HOLD | +79.7s | 8.5999 |
| LANDING | +88.3s | 49.2997 |
| DONE | +137.6s | 5.4938 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.22 | 65.836 |
| 1 | TRANS_TO_FW | 71.056 | 2.556 |
| 4 | FW | 73.612 | 19.036 |
| 2 | TRANS_TO_MC | 92.648 | 5.14 |
| 3 | MC | 97.788 | 51.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 알려진 코스메틱 | 예시 |
|---|---|---|---|---|---|
| node.log | WARN | 6 | 1785086829.5 |  | /mavros/cmd/arming 서비스 없음 |
| mavros.log | ERROR | 13 | 1785086808.2 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785086809.3 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785086807.1 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785086805.3 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785086812.5 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785086808.0 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785086824.7 |  | TM: RTT too high for timesync: 2119.69 ms. |
| mavros.log | ERROR | 3 | 1785086864.7 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785086815.3 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785086815.5 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785086825.7 |  | PR: request param #387 timeout, retries left 2, and 356 params still missing |
| mavros.log | WARN | 1 | 1785086974.3 |  | UAS Executor terminated |

## 장애주입 결과

- `probe` spec={"on_state": "FOLLOWING", "delay_s": 3.0, "action": "probe", "topic": "/mavros/state"} → 발화 +54.93s rc=0

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
