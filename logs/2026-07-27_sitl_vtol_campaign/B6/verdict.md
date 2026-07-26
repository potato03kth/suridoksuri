# B6 — 판정

- 목적: 2-WP 최소 경로 (플래너 N≤2 특수케이스, NR 우회)
- 실행: 2026-07-26T17:42:14.427184+00:00 ~ 2026-07-26T17:45:00.591262+00:00 (경과 155.1s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0]`
- 저장소 HEAD: `3b52ac1`
- ulog: 17_42_26.ulg (meta.json 기록: 17_42_26.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.11648 x ulog + 1785087737.355` (앵커 4개, 최대 잔차 0.946s). 시뮬 클록이 벽시계보다 +11.6% 빠름/느림 — 상수 오프셋만 쓰면 6.71s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 129.9271s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 24.572s → disarmed 141.684s (비행 117.112s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.6s / 역천이 5.536s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 8건 / 전체 위반 38건 / 샘플 559개, 최대 116.5841m (477.8036 m/s). 경계 최대: 114.8327m@73.096s(TRANSITION_FW), 3.9551m@73.492s(STREAMING), 3.918m@73.692s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=58.9323 m/s² (6.0094g) @138.556s state=LANDING; 접지(disarm−5s) 제외 시 4.2948 m/s² (0.438g) @73.56s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3342 m/s² (0.238g), 14.4793→5.0472 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -95.5916° → 정렬 14.12s 소요, 최대 95.6327°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.5833m vs transition_alt 50.0m → -0.8335% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 50.707m → 최저 50.707m (손실 0.0m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 50.0482m, 평균편차 0.2312m, 최대 \|편차\| 2.7931m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.4m 평균 1.1m (부호 -1.4~-0.8m, n=6) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 0건/0종, mavros.log 47건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.8292 |
| CLIMBING | +0.8s | 32.4979 |
| TRANSITION_FW | +33.3s | 21.7002 |
| STREAMING | +55.0s | 0.108 |
| FOLLOWING | +55.1s | 11.2981 |
| TRANSITION_MC | +66.4s | 5.7947 |
| HOLD | +72.2s | 12.2995 |
| LANDING | +84.5s | 45.3996 |
| DONE | +129.9s | 5.6721 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.2 | 64.888 |
| 1 | TRANS_TO_FW | 70.088 | 2.6 |
| 4 | FW | 72.688 | 11.7 |
| 2 | TRANS_TO_MC | 84.388 | 5.536 |
| 3 | MC | 89.924 | 52.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 알려진 코스메틱 | 예시 |
|---|---|---|---|---|---|
| mavros.log | ERROR | 13 | 1785087746.4 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785087747.5 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785087745.4 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785087743.6 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785087750.6 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785087746.3 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785087758.4 |  | TM: RTT too high for timesync: 1431.76 ms. |
| mavros.log | ERROR | 2 | 1785087799.5 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785087753.3 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785087754.2 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785087900.8 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
