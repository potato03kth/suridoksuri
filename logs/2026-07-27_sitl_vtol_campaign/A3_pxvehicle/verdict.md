# A3 — 판정

- 목적: SITL-4 L자 경로 재현 (90° 1회)
- 실행: 2026-07-26T19:25:32.197459+00:00 ~ 2026-07-26T19:29:52.782841+00:00 (경과 237.4s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0]`
- 저장소 HEAD: `3b52ac1`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 19_25_44.ulg (meta.json 기록: 19_25_44.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.08677 x ulog + 1785093941.239` (앵커 4개, 최대 잔차 0.674s). 시뮬 클록이 벽시계보다 +8.7% 빠름/느림 — 상수 오프셋만 쓰면 6.74s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 145.8504s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 91.404s → disarmed 223.016s (비행 131.612s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.6s / 역천이 5.364s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 9건 / 전체 위반 114건 / 샘플 995개, 최대 229.1298m (725.0942 m/s). 경계 최대: 229.1298m@142.352s(FOLLOWING), 113.2603m@170.952s(HOLD), 61.7405m@165.364s(FOLLOWING). 스트림 재개 갭 1건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=77.7204 m/s² (7.9253g) @219.9s state=LANDING; 접지(disarm−5s) 제외 시 9.0084 m/s² (0.9186g) @142.428s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.252 m/s² (0.2296g), 14.2096→5.048 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -95.5189° → 정렬 13.38s 소요, 최대 95.726°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.9705m vs transition_alt 50.0m → -0.059% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 47.041m → 최저 47.0281m (손실 0.013m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.3185m, 평균편차 -1.4133m, 최대 \|편차\| 5.5904m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 7.2m 평균 2.1583m (부호 -1.0~7.2m, n=12) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 0건/0종, mavros.log 50건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.4674 |
| CLIMBING | +0.5s | 34.6822 |
| TRANSITION_FW | +35.1s | 18.9007 |
| STREAMING | +54.1s | 0.1139 |
| FOLLOWING | +54.2s | 26.411 |
| TRANSITION_MC | +80.6s | 5.5754 |
| HOLD | +86.2s | 8.8998 |
| LANDING | +95.1s | 50.8002 |
| DONE | +145.9s | 5.0898 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.36 | 133.888 |
| 1 | TRANS_TO_FW | 139.248 | 2.6 |
| 4 | FW | 141.848 | 23.52 |
| 2 | TRANS_TO_MC | 165.368 | 5.364 |
| 3 | MC | 170.732 | 53.004 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 알려진 코스메틱 | 예시 |
|---|---|---|---|---|---|
| mavros.log | ERROR | 13 | 1785093944.2 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785093945.4 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785093943.3 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785093941.5 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785093948.5 |  | VER: command plugin service call failed! |
| mavros.log | ERROR | 4 | 1785093997.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 3 | 1785093944.2 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785093957.2 |  | TM: RTT too high for timesync: 1940.51 ms. |
| mavros.log | WARN | 1 | 1785093951.1 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785093951.2 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785093958.1 |  | PR: request param #331 timeout, retries left 2, and 521 params still missing |
| mavros.log | WARN | 1 | 1785094193.0 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
