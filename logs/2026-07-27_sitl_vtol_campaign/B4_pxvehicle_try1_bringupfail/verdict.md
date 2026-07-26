# B4 — 판정

- 목적: 예각/U턴(135°) — 선회반경 초과 시 거동
- 실행: 2026-07-26T20:00:56.616025+00:00 ~ 2026-07-26T20:03:23.528838+00:00 (경과 132.7s)
- 종료: `mavros_not_connected` (exit=3)
- launch: `ros2 launch fc_ros phase2.launch.py transition_alt:=50.0 waypoints:=[0.0,0.0,50.0, 250.0,0.0,50.0, 100.0,150.0,50.0]`
- 저장소 HEAD: `3b52ac1`
- PX4 빌드: `c890d9db0a` (`/root/PX4-vehicle`)
- ulog: 20_01_08.ulg (meta.json 기록: 20_01_08.ulg)
- 요약: FAIL 2, NULL 8, PASS 2, WARN 1

> ⚠️ node.log ↔ ulog 시각 정렬 실패 (node.log 와 ulog 에 공통 앵커(ARM/이륙/천이 명령)가 없음) — 상태창 기반 지표는 전부 NULL 이다.

## 판정표

| 항목 | 판정 | 근거 수치 | 기준 |
|---|---|---|---|
| 완주 (DONE 도달) | **FAIL** | 관측 상태: (없음); 종료사유=mavros_not_connected | DONE 상태 도달 |
| disarm 확인 | **NULL** | ulog 상 armed(=2) 구간이 없음 — ARM 실패 | ulog arming_state 2→1 |
| 상태 순서 | **PASS** |  | 정상 순서의 부분수열 |
| vtol_state 시퀀스 | **FAIL** | seq=[3], 정천이 Nones / 역천이 Nones | 3→1→4, 4→2→3 |
| setpoint 점프 | **NULL** | 최대 0.0142m — 그러나 상태 경계 시각을 알 수 없어 '경계에서' 판정 불가 | 상태 경계에서 급점프 없음 |
| 수직 가속 | **PASS** | 피크 \|az\|=0.0738 m/s² (0.0075g) @114.204s state=None; 접지 제외값 없음(disarm 시각을 몰라 접지 구간을 제외할 수 없음) | 천이 제외 |az| ≤ 0.5g (4.90 m/s²) |
| 역천이 감속률 | **NULL** | 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다 | ≤ 0.3g (2.94 m/s²) |
| TRANSITION_FW 헤딩 | **NULL** | TRANSITION_FW 상태창이 없음 (상태 미도달이거나 시각정렬 실패) | 오버슈트 없이 단조수렴, 정렬완료 err ≤ wp0_heading_tol |
| CLIMBING 오버슈트 | **NULL** | CLIMBING 상태창 없음 (시각정렬 실패 또는 상태 미도달) | transition_alt 대비 최대 AGL ≤ +10% |
| 정천이 고도손실 | **NULL** | 정천이 구간(vtol_state==1)을 특정할 수 없음 — 천이 미발생 가능 | ≤ 5m |
| 순항 고도편차 | **NULL** | FOLLOWING 상태창 또는 순항고도 기준을 알 수 없음 | ±3m |
| FW cte | **NULL** | node.log 에 'FOLLOWING tick= ... cte=' 샘플이 없음 (FOLLOWING 미진입이거나 20틱 미만 체류) | 직선 ≤ 2m |
| 경고/타임아웃 | **WARN** | node.log 0건/0종, mavros.log 19건/6종 — verdict 하단 목록 참조 | 무해성 판단은 사람이 한다(계획서 5장) |

## 상태 전이 타임라인

| 상태 | 진입(벽시계) | 체류(s) |
|---|---|---|

## vtol_state 시퀀스

| vtol_state | 이름 | 시작(ulog s) | 지속(s) |
|---|---|---|---|
| 3 | MC | 5.312 | 129.0 |

## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)

| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |
|---|---|---|---|---|---|
| mavros.log | ERROR | 7 | 1785096068.7 |  | FCU: EVENT 7791755 with args -255-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 4 | 1785096069.8 |  | FCU: UNK(8): EVENT 11047904 with args -0-0-0-18-1-128-0-0-0-0-58-1-128-0-58-1-128-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0- |
| mavros.log | WARN | 3 | 1785096081.5 |  | TM: RTT too high for timesync: 1878.10 ms. |
| mavros.log | WARN | 2 | 1785096065.9 |  | VER: broadcast request timeout, retries left 4 |
| mavros.log | ERROR | 2 | 1785096121.2 |  | TM: Time jump detected. Resetting time synchroniser. |
| mavros.log | WARN | 1 | 1785096082.4 |  | PR: request param #393 timeout, retries left 2, and 444 params still missing |

## 미산출 지표 (null)

- **disarm 확인**: ulog 상 armed(=2) 구간이 없음 — ARM 실패
- **setpoint 점프**: 최대 0.0142m — 그러나 상태 경계 시각을 알 수 없어 '경계에서' 판정 불가
- **역천이 감속률**: 역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 특정할 수 없음 — 역천이가 일어나지 않았을 수 있다
- **TRANSITION_FW 헤딩**: TRANSITION_FW 상태창이 없음 (상태 미도달이거나 시각정렬 실패)
- **CLIMBING 오버슈트**: CLIMBING 상태창 없음 (시각정렬 실패 또는 상태 미도달)
- **정천이 고도손실**: 정천이 구간(vtol_state==1)을 특정할 수 없음 — 천이 미발생 가능
- **순항 고도편차**: FOLLOWING 상태창 또는 순항고도 기준을 알 수 없음
- **FW cte**: node.log 에 'FOLLOWING tick= ... cte=' 샘플이 없음 (FOLLOWING 미진입이거나 20틱 미만 체류)

---

판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. 경고의 무해성 판단은 이 스크립트가 하지 않는다.
