# A3 — 판정

- 목적: SITL-4 L자 경로 재현 (90° 1회)
- 실행: 2026-07-26T19:06:10.044533+00:00 ~ 2026-07-26T19:17:43.034150+00:00 (경과 629.1s)
- 종료: `timeout` (exit=2)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 200.0,0.0,50.0, 200.0,200.0,50.0]`
- 저장소 HEAD: `3b52ac1`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 19_06_22.ulg (meta.json 기록: 19_06_22.ulg)
- 요약: FAIL 3, NULL 5, PASS 4, WARN 1

- 시각 정렬: `wall = 1.00000 x ulog + 1785092805.021` (앵커 2개, 최대 잔차 0.036s). 시뮬 클록이 벽시계보다 +0.0% 빠름/느림 — 상수 오프셋만 쓰면 0.07s 벌어진다
  - ⚠️ 앵커가 1개뿐(또는 구간이 5초 미만)이라 scale=1 상수 오프셋으로 폴백 — 시뮬/벽시계 클록 드리프트가 보정되지 않았다

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **FAIL** | 관측 상태: ARM_TAKEOFF → CLIMBING; 종료사유=timeout | DONE 상태 도달 |
| disarm 확인 | **FAIL** | 로그 끝까지 disarm 되지 않음 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** | ARM_TAKEOFF → CLIMBING | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **FAIL** | seq=[3], 정천이 Nones / 역천이 Nones | 3→1→4, 4→2→3 |
| setpoint 점프 | **PASS** | 임계 1.5m, 경계±1s 위반 0건 / 전체 위반 0건 / 샘플 737개, 최대 0.5744m (2.8966 m/s). 경계 최대: -. 스트림 재개 갭 0건은 별도 집계 | 상태 경계 ±1s 에서 1.5m 초과 점프 없음 |
| 수직 가속 | **PASS** | 피크 \|az\|=3.0737 m/s² (0.3134g) @141.32s state=CLIMBING; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **NULL** | 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다 | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **NULL** | TRANSITION_FW 상태창이 없음 (상태 미도달이거나 시각정렬 실패) | 오버슈트 없이 단조수렴, 정렬완료 err ≤ wp0_heading_tol |
| CLIMBING 오버슈트 | **PASS** | 최대 AGL 43.911m vs transition_alt 50.0m → -12.178% | ≤ +10% |
| 정천이 고도손실 | **NULL** | 정천이 구간(vtol_state==1)을 특정할 수 없음 — 천이 미발생 가능 | ≤ 5m |
| 순항 고도편차 | **NULL** | FOLLOWING 상태창 또는 순항고도 기준을 알 수 없음 | ±3m |
| FW cte | **NULL** | node.log 에 'FOLLOWING tick= ... cte=' 샘플이 없음 (FOLLOWING 미진입이거나 20틱 미만 체류) | 직선 ≤ 2m |
| 경고/타임아웃 | **WARN** | node.log 0건/0종, mavros.log 57건/13종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|
| ARM_TAKEOFF | +0.0s | 0.7757 |
| CLIMBING | +0.8s | 527.7064 |

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.528 | 149.04 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 알려진 코스메틱 | 예시 |
|---|---|---|---|---|---|
| mavros.log | ERROR | 14 | 1785092782.9 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | ERROR | 12 | 1785092839.4 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 8 | 1785092783.9 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-16-18-1-128-0-0-0-0-0-0-0-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 6 | 1785092781.4 |  | VER: unicast request timeout, retries left 2 |
| mavros.log | WARN | 4 | 1785092779.6 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 4 | 1785092787.4 |  | VER: command plugin service call failed! |
| mavros.log | WARN | 3 | 1785092796.0 |  | TM: RTT too high for timesync: 1739.68 ms. |
| mavros.log | WARN | 1 | 1785092783.2 |  | CMD: Unexpected command 520, result 3 |
| mavros.log | WARN | 1 | 1785092790.1 |  | VER: your FCU don't support AUTOPILOT_VERSION, switched to default capabilities |
| mavros.log | WARN | 1 | 1785092791.4 |  | PR: Failed to get parameter type: CBRK_SUPPLY_CHK |
| mavros.log | WARN | 1 | 1785092800.1 |  | PR: request param #324 timeout, retries left 2, and 543 params still missing |
| mavros.log | WARN | 1 | 1785092992.0 |  | CON: Lost connection, HEARTBEAT timed out. |
| mavros.log | WARN | 1 | 1785093463.2 |  | UAS Executor terminated |

## 미산출 지표 (null)

- **역천이 감속률**: 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다
- **TRANSITION_FW 헤딩**: TRANSITION_FW 상태창이 없음 (상태 미도달이거나 시각정렬 실패)
- **정천이 고도손실**: 정천이 구간(vtol_state==1)을 특정할 수 없음 — 천이 미발생 가능
- **순항 고도편차**: FOLLOWING 상태창 또는 순항고도 기준을 알 수 없음
- **FW cte**: node.log 에 'FOLLOWING tick= ... cte=' 샘플이 없음 (FOLLOWING 미진입이거나 20틱 미만 체류)

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
