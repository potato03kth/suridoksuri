# 2026-07-25_flight04 — offboard 경로 완주 + AUTO.LAND 착륙 (오늘 유일한 성공)

- **비행 조건:** vehicle_type:=mc transition_alt:=3.0 waypoints:=[0.0,0.0,3.0, -4.24,4.24,3.0, 0.0,0.0,3.0]
- **시각:** 2026-07-25 06:54:23~06:56:05 UTC (**15:54 KST**), armed 06:54:26~06:55:03 (37.6초)
- **FC ulog:** `../2026-07-25_manual/log_152_2026-07-25-06-55-04.ulg`
- **결과:** ARM → 이륙 → OFFBOARD → WP0→WP1→WP2 순차통과 → AUTO.LAND → disarm.
  **조종사 개입 0회, OFFBOARD 이탈 0회.** 상태기계는 설계대로 끝까지 돌았다.

---

## 1. 시퀀스 (ulog `vehicle_status` + launch.log)

| UTC | 사건 |
|---|---|
| 06:54:25 | ARM 요청 → armed |
| 06:54:26 | CommandTOL 이륙 요청 alt=56.8 m AMSL (지면 53.8 + 3.0) → CLIMBING |
| 06:54:34 | AUTO_LOITER (이륙 완료) |
| 06:54:35 | 노드: "운용 고도 3.0 m 도달 → streaming" |
| 06:54:38 | **OFFBOARD 진입** → FOLLOWING 시작 pos=[-3.5,0.1] tgt=[0,0] cte=3.5 m |
| 06:54:40 | WP0 통과 (dist=1.8 m) → WP1 이동 |
| 06:54:42 | WP1 통과 (dist=1.9 m) → WP2 이동 |
| 06:54:46 | 경로 추종 완료 → hold |
| 06:54:47 | WP1 도달·안정 → LANDING (dist=0.5 m speed=0.2 m/s) |
| 06:54:48 | AUTO.LAND |
| 06:55:03 | 착륙 완료 (disarmed) → DONE |

## 2. 그런데 "제대로" 간 건 아니다 — flight01과 같은 좌표계 오류, 크기만 작았을 뿐

시동 시점 로컬 위치가 `pos_ned=[-3.36, 0.087, -2.09]` — 즉 **EKF 로컬원점이 이륙지점에서
수평 3.37 m, 수직 2.09 m 어긋나 있었다.** flight01의 10.9 m / 10.6 m보다 작았기 때문에
사고가 안 났을 뿐, 구조는 같다.

ulog `vehicle_local_position_setpoint` vs `vehicle_local_position` (NED, D는 아래가 +):

```
UTC        sp_N   sp_E   sp_D  |  ac_N   ac_E   ac_D   수평오차
06:54:39  -3.09   0.09  -0.96 | -3.54   0.10  -0.68     0.46   ← 지상 D=+2.11이 기준
06:54:40   0.00   0.00  -3.00 | -2.49   0.11  -1.10     2.50   ← WP0 = 3.4 m 떨어진 점
06:54:42  -4.24   4.24  -3.00 | -1.95   1.99  -2.72     3.21
06:54:44   0.00   0.00  -3.00 | -4.00   3.94  -2.30     5.62
06:54:47   0.00   0.00  -3.00 |  0.18  -0.28  -3.13     0.33
```

- **수평:** WP0 `[0,0]`은 이륙지점이 아니라 3.37 m 떨어진 곳. 조종사 눈에는 이륙하자마자
  3.4 m 옆으로 슬쩍 이동한 것으로 보인다.
- **고도:** 지상 D=+2.11인데 `cruise_alt`는 D=−3.00으로 발행됐다 →
  **실제 비행고도 5.11 m AGL** (의도한 3 m가 아님). 실측 최고 D=−3.15 = 5.26 m AGL.
- 착륙 후 D=+3.55로 이륙 때(+2.11)보다 1.44 m 낮게 찍힌다 — 37초 사이 EKF 고도가 그만큼 흘렀다.

**즉 "잘 날았다"가 아니라 "어긋남이 작아서 티가 덜 났다"**. 오차 원인·조치는
`../2026-07-25_flight01/notes.md` §3, §6과 동일하다.

## 3. WP 통과 판정이 헐겁다

`mc_end_thresh = 2.0 m`이라 WP0을 dist=1.8 m, WP1을 dist=1.9 m에서 "통과"로 처리했다.
경로 자체가 6 m짜리(원점↔[-4.24,4.24])라 **허용반경이 경로길이의 1/3**이다. 실제로
가보지 않고 다음 점으로 넘어가는 것과 구분이 안 된다. 짧은 경로에선 `mc_end_thresh`를
0.5~1.0 m로 줄여야 한다.

## 4. 추력·배터리 (기존 가설 트래커 연결)

PX4 로그 메시지:
```
1387.30 [INFO]    commander: Takeoff detected
1390.63 [WARNING] health_and_arming_checks: Low battery
1395.04 [ERROR]   health_and_arming_checks: Emergency battery level
1408.37 [ERROR]   health_and_arming_checks: Emergency battery level
1421.03 [INFO]    commander: Landing detected
```

- 비행중 평균 모터출력 median **0.789**, 최대 0.940, **개별 모터가 1.000에 붙은 샘플 74/319 (23%)**
- 4S 팩: 휴지 16.5 V → 부하중 최저 **12.71 V** (3.18 V/cell), 최대 35.7 A
- `control_allocator_status.actuator_saturation`: m0 = +2(상한), m1·m3 = −2(하한)
- 모터별 평균: m0 **0.905** / m1 0.701 / m2 0.729 / m3 0.683 — **motor0 단독으로 튄다**
  (2026-07-21의 "CCW쌍이 1.4~1.5배" 패턴과 방향이 반대. `docs/mc_hw_open_hypotheses.md` H13)
- `COM_LOW_BAT_ACT=0`이라 페일세이프 동작은 없었지만, `BAT1_CAPACITY=-1`(전압만으로 잔량추정)
  + `BAT1_V_EMPTY=3.6 V`라 부하중 sag이 곧바로 "Emergency"로 읽힌다 — 임계 재설정 필요.

## 5. 요약

| 항목 | 판정 |
|---|---|
| 상태기계 (ARM→이륙→OFFBOARD→WP추종→AUTO.LAND→DONE) | ✅ 설계대로 완주 |
| 위치 기준계 | ❌ 이륙지점이 아니라 EKF 원점 기준 (3.4 m 오프셋) |
| 고도 | ❌ 3 m 의도 → 실제 5.1 m AGL |
| WP 통과 판정 | ⚠️ 반경 2.0 m, 6 m 경로에 비해 헐거움 |
| 추력 여유 | ❌ 개별 모터 23% 구간 포화, Emergency battery |
