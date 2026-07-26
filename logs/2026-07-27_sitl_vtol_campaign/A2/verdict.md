# A2 — 판정

- 목적: 동일 경로, waypoint_frame=takeoff(현 기본값) — VTOL 첫 검증
- 실행: 2026-07-26T17:38:13.920516+00:00 ~ 2026-07-26T17:41:02.793304+00:00 (경과 158.0s)
- 종료: `done` (exit=0)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 300.0,0.0,50.0]`
- 저장소 HEAD: `3b52ac1`
- ulog: 17_38_25.ulg (meta.json 기록: 17_38_25.ulg)
- 요약: FAIL 2, PASS 10, WARN 1

- 시각 정렬: `wall = 1.09895 x ulog + 1785087497.532` (앵커 4개, 최대 잔차 1.298s). 시뮬 클록이 벽시계보다 +9.9% 빠름/느림 — 상수 오프셋만 쓰면 6.20s 벌어진다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **PASS** | 상태 ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE, 소요 136.7655s | DONE 상태 도달 |
| disarm 확인 | **PASS** | armed 20.764s → disarmed 145.312s (비행 124.548s) | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → HOLD → LANDING → DONE | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **PASS** | seq=[3, 1, 4, 2, 3], 정천이 2.5s / 역천이 5.32s | 3→1→4, 4→2→3 |
| setpoint 점프 | **FAIL** | 임계 1.5m, 경계±1s 위반 10건 / 전체 위반 74건 / 샘플 580개, 최대 217.8034m (1067.6639 m/s). 경계 최대: 217.8034m@67.032s(TRANSITION_FW), 3.5968m@67.636s(TRANSITION_FW), 3.5559m@67.436s(TRANSITION_FW). 스트림 재개 갭 2건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **FAIL** | 피크 \|az\|=73.9705 m/s² (7.5429g) @142.164s state=LANDING; 접지(disarm−5s) 제외 시 4.7132 m/s² (0.4806g) @67.04s state=TRANSITION_FW | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **PASS** | 최대 2.3325 m/s² (0.2378g), 14.1537→5.0484 m/s | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **PASS** | 시작오차 -97.7361° → 정렬 13.828s 소요, 최대 97.7364°, tol 진입 후 재증가 0.0 rad, 단조수렴=True | 단조수렴 + err ≤ 2.9° |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 49.4529m vs transition_alt 50.0m → -1.0941% | ≤ +10% |
| 정천이 고도손실 | **PASS** | 51.719m → 최저 51.7062m (손실 0.0128m) | ≤ 5m |
| 순항 고도편차 | **PASS** | 기준 AGL 49.8999m, 평균편차 -0.1442m, 최대 \|편차\| 1.838m | ±3m |
| FW cte | **PASS** | 최대 \|cte\| 1.5m 평균 1.22m (부호 -1.5~-0.5m, n=10) — 코너 오버슈트 포함값 | 직선 ≤ 2m (코너는 별도 기록) |
| 경고/타임아웃 | **WARN** | node.log 0건/0종, mavros.log 46건/11종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 1.8663 |
| CLIMBING | +1.9s | 29.0991 |
| TRANSITION_FW | +31.0s | 21.3997 |
| STREAMING | +52.4s | 0.104 |
| FOLLOWING | +52.5s | 19.0036 |
| TRANSITION_MC | +71.5s | 8.2928 |
| HOLD | +79.8s | 8.7999 |
| LANDING | +88.6s | 48.2001 |
| DONE | +136.8s | 4.9999 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.24 | 58.992 |
| 1 | TRANS_TO_FW | 64.232 | 2.5 |
| 4 | FW | 66.732 | 19.392 |
| 2 | TRANS_TO_MC | 86.124 | 5.32 |
| 3 | MC | 91.444 | 54.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 알려진 코스메틱 | 예시 |
|---|---|---|---|---|---|
| mavros.log | ERROR | 13 | 1785087505.9 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 10 | 1785087507.0 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 5 | 1785087505.4 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785087503.5 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785087510.2 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785087506.3 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | ERROR | 3 | 1785087556.8 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785087512.9 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785087513.5 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785087517.6 |  | TM: RTT too high for timesync: 621.75 ms. |
| mavros.log | WARN | 1 | 1785087663.0 |  | UAS Executor terminated |

## 미산출 지표 (null)

없음 — 계획서 4장 지표 전부 산출됨.

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
