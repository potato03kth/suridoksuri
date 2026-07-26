# B7 — 판정

- 목적: 단거리 경로(40m, d_end_thresh=10m 대비 짧음) — FOLLOWING 즉시완료 오판
- 실행: 2026-07-26T19:57:15.843038+00:00 ~ 2026-07-26T20:00:05.532270+00:00 (경과 155.4s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 40.0,0.0,50.0]`
- 저장소 HEAD: `3b52ac1`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 19_57_27.ulg (meta.json 기록: 19_57_27.ulg)
- 요약: FAIL 3, PASS 9, WARN 1

- 시각 정렬: `wall = 1.07780 x ulog + 1785095843.040` (앵커 4개, 최대 잔차 0.156s). 시뮬 클록이 벽시계보다 +7.8% 빠름/느림 — 상수 오프셋만 쓰면 3.64s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 122.2981s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 31.3s → disarmed 142.056s (비행 110.756s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.588s / 역천이 7.248s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 1건 / 전체 위반 1건 / 샘플 554개, 최대 143.613m (718.0649 m/s). 경계 최대: 143.613m@87.648s(HOLD). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=77.2768 m/s² (7.88g) @138.92s state=LANDING; 접지(disarm−5s) 제외 시 8.8527 m/s² (0.9027g) @79.324s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **FAIL** | 최대 3.0886 m/s² (0.315g), 18.3411→5.0189 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -95.2726° → 정렬 12.472s 소요, 최대 95.2777°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.9521m vs transition_alt 50.0m → -0.0958% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.6484m → 최저 49.6476m (손실 0.0007m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.1979m, 평균편차 -0.9273m, 최대 \|편차\| 1.2401m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 0.6m 평균 0.6m (부호 -0.6~-0.6m, n=1) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 23건/1종, mavros.log 50건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.6074 |
| CLIMBING | +0.6s | 32.1907 |
| TRANSITION_FW | +32.8s | 18.3998 |
| STREAMING | +51.2s | 0.1016 |
| FOLLOWING | +51.3s | 1.0005 |
| TRANSITION_MC | +52.3s | 7.4982 |
| HOLD | +59.8s | 16.3001 |
| LANDING | +76.1s | 46.1998 |
| DONE | +122.3s | 5.1483 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.352 | 70.8 |
| 1 | TRANS_TO_FW | 76.152 | 2.588 |
| 4 | FW | 78.74 | 1.312 |
| 2 | TRANS_TO_MC | 80.052 | 7.248 |
| 3 | MC | 87.3 | 55.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 알려진 코스메틱 | 예시 |
|---|---|---|---|---|---|
| node.log | WARN | 23 | 1785095874.6 |  | /mavros/cmd/arming 서비스 없음 |
| mavros.log | ERROR | 13 | 1785095847.9 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785095848.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785095847.0 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785095845.2 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785095852.0 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785095847.9 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785095860.7 |  | TM: RTT too high for timesync: 1830.27 ms. |
| mavros.log | ERROR | 3 | 1785095900.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 2 | 1785095855.4 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785095854.7 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785095861.5 |  | PR: request param #306 timeout, retries left 2, and 562 params still missing |
| mavros.log | WARN | 1 | 1785096005.9 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
