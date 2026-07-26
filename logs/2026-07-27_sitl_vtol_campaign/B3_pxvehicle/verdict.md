# B3 — 판정

- 목적: 직각 코너(90°) — 코너 오버슈트 정량화
- 실행: 2026-07-26T19:51:59.781499+00:00 ~ 2026-07-26T19:56:22.044960+00:00 (경과 237.8s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 250.0,250.0,50.0]`
- 저장소 HEAD: `3b52ac1`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 19_52_11.ulg (meta.json 기록: 19_52_11.ulg)
- 요약: FAIL 3, PASS 8, WARN 2

- 시각 정렬: `wall = 1.13797 x ulog + 1785095521.072` (앵커 4개, 최대 잔차 0.628s). 시뮬 클록이 벽시계보다 +13.8% 빠름/느림 — 상수 오프셋만 쓰면 10.63s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 153.2446s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 89.108s → disarmed 224.088s (비행 134.98s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.504s / 역천이 5.084s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 6건 / 전체 위반 132건 / 샘플 973개, 최대 300.562m (951.1457 m/s). 경계 최대: 300.562m@137.144s(TRANSITION_FW), 3.6858m@137.752s(FOLLOWING), 3.6857m@137.544s(FOLLOWING). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=63.315 m/s² (6.4563g) @220.904s state=LANDING; 접지(disarm−5s) 제외 시 5.3158 m/s² (0.5421g) @137.46s state=FOLLOWING | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.359 m/s² (0.2405g), 14.3777→5.3022 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -92.3805° → 정렬 14.936s 소요, 최대 92.3805°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.4881m vs transition_alt 50.0m → -1.0239% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 49.706m → 최저 49.7052m (손실 0.0008m) | ≤ 5m |
| 순항 고도편차 | **FAIL** | 기준 AGL 50.1473m, 평균편차 -0.7849m, 최대 \|편차\| 4.7328m | ±3m |
| FW cte | **WARN** | 최대 \|cte\| 7.0m 평균 1.0875m (부호 -0.8~7.0m, n=16) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 0건/0종, mavros.log 50건/12종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.5543 |
| CLIMBING | +0.6s | 31.5902 |
| TRANSITION_FW | +32.1s | 22.5999 |
| STREAMING | +54.7s | 0.1097 |
| FOLLOWING | +54.9s | 34.4138 |
| TRANSITION_MC | +89.3s | 5.2767 |
| HOLD | +94.5s | 8.7999 |
| LANDING | +103.3s | 49.9001 |
| DONE | +153.2s | 6.1122 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.248 | 128.984 |
| 1 | TRANS_TO_FW | 134.232 | 2.504 |
| 4 | FW | 136.736 | 31.136 |
| 2 | TRANS_TO_MC | 167.872 | 5.084 |
| 3 | MC | 172.956 | 52.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 알려진 코스메틱 | 예시 |
|---|---|---|---|---|---|
| mavros.log | ERROR | 13 | 1785095531.6 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785095532.7 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785095530.8 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785095529.0 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785095535.8 |  | VER: command plugin service call failed! |
| mavros.log | ERROR | 4 | 1785095585.3 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 3 | 1785095531.7 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 3 | 1785095544.4 |  | TM: RTT too high for timesync: 1988.47 ms. |
| mavros.log | WARN | 1 | 1785095538.5 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785095539.0 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785095545.3 |  | PR: request param #420 timeout, retries left 2, and 475 params still missing |
| mavros.log | WARN | 1 | 1785095782.2 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
