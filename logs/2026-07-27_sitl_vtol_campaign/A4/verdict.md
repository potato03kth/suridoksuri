# A4 — 판정

- 목적: 중간 WP 고도가 다른 경로 → _cruise_alt 스칼라화(중간 WP z 무시) 실증
- 실행: 2026-07-26T18:07:19.213930+00:00 ~ 2026-07-26T18:10:18.939707+00:00 (경과 164.7s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 150.0,0.0,80.0, 300.0,0.0,50.0]`
- 저장소 HEAD: `3b52ac1`
- ulog: 18_07_31.ulg (meta.json 기록: 18_07_31.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.09567 x ulog + 1785089244.877` (앵커 4개, 최대 잔차 0.583s). 시뮬 클록이 벽시계보다 +9.6% 빠름/느림 — 상수 오프셋만 쓰면 6.98s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 133.5876s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 28.548s → disarmed 151.284s (비행 122.736s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.54s / 역천이 5.244s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 7건 / 전체 위반 80건 / 샘플 618개, 최대 214.4003m (678.4819 m/s). 경계 최대: 214.4003m@76.304s(FOLLOWING), 112.8306m@100.344s(HOLD), 62.7293m@94.948s(FOLLOWING). 스트림 재개 갭 4건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=78.2384 m/s² (7.9781g) @148.164s state=LANDING; 접지(disarm−5s) 제외 시 4.426 m/s² (0.4513g) @76.128s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.2392 m/s² (0.2283g), 14.1361→5.0617 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -97.5908° → 정렬 16.088s 소요, 최대 98.0934°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 48.5245m vs transition_alt 50.0m → -2.9509% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 51.4982m → 최저 51.4982m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.0278m, 평균편차 -0.044m, 최대 \|편차\| 2.1358m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.2m 평균 0.51m (부호 -1.2~0.2m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 0건/0종, mavros.log 49건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.9976 |
| CLIMBING | +1.0s | 28.4905 |
| TRANSITION_FW | +29.5s | 21.9992 |
| STREAMING | +51.5s | 0.1061 |
| FOLLOWING | +51.6s | 21.7081 |
| TRANSITION_MC | +73.3s | 5.3864 |
| HOLD | +78.7s | 8.6998 |
| LANDING | +87.4s | 46.1997 |
| DONE | +133.6s | 7.8879 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.34 | 67.952 |
| 1 | TRANS_TO_FW | 73.292 | 2.54 |
| 4 | FW | 75.832 | 19.12 |
| 2 | TRANS_TO_MC | 94.952 | 5.244 |
| 3 | MC | 100.196 | 52.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 알려진 코스메틱 | 예시 |
|---|---|---|---|---|---|
| mavros.log | ERROR | 13 | 1785089251.0 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785089252.4 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785089250.4 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785089248.5 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785089255.6 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785089251.3 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 3 | 1785089304.0 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 2 | 1785089258.8 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 2 | 1785089264.7 |  | TM: RTT too high for timesync: 1721.97 ms. |
| mavros.log | WARN | 1 | 1785089258.5 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785089265.6 |  | PR: request param #268 timeout, retries left 2, and 568 params still missing |
| mavros.log | WARN | 1 | 1785089419.1 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
