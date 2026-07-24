# 2026-07-24_flight03

- **비행 조건:** vehicle_type:=mc transition_alt:=4.0 waypoints:=[0.0,0.0,4.0, **-4.24,-4.24**,4.0, 0.0,0.0,4.0] (2번째 WP가 음의 방향). ARM 09:25:36 UTC(18:25:36 KST) — 성공, 전체 미션 완주(착륙까지).
- **원격 notes.md(비행 당시 기록):** 관찰="direction", 결론="direction" — 사용자가 이 비행에서 방향 이상을 직접 인지.

## 질의 1 — 오프보드 발행 여부

`launch.log`에 `OFFBOARD 전환 요청 (폴백)` → `OFFBOARD 확인 → FOLLOWING`이 명시적으로 기록됨
(t=1784885149.32~149.42, ARM 후 약 12.8초). **오프보드 발행·진입 확인.** 단 `AUTO.LAND 요청`이
1784885151.92(OFFBOARD 진입 후 **2.5초** 만)에 바로 이어져 OFFBOARD 유지시간이 매우 짧음.

## 질의 2 — WP 방향을 반대로 명령했는데 같은 방향으로 비행 (근본원인 확정)

`launch.log` 그대로 인용:
```
FOLLOWING 시작 pos=[-2.1,-1.3] tgt=[0.0,0.0] cte=-0.6m mode=OFFBOARD
경로 추종 완료 -> hold (MC, 역천이 생략)     ← 바로 다음 줄, 1ms 뒤
```
**FOLLOWING 진입 직후(사실상 즉시) "경로 추종 완료"로 판정되어 두 번째 waypoint(-4.24,-4.24)
방향으로 전혀 이동하지 않고 곧장 hold→LANDING으로 넘어감.**

코드 확인(`fc_ros/fc_ros/nodes/offboard_node.py:919-921`):
```python
last_pt = self._pts[-1]
dist_to_end = float(np.linalg.norm(pos[:2] - last_pt))
return trans_mc_trigger(dist_to_end, self._d_end_thresh)
```
`_pts[-1]` = waypoint 리스트의 **마지막 점**(이 리스트는 palindrome이라 항상 원점 (0,0)).
`d_end_thresh` 기본값 **10.0m**(`fc_ros_params.yaml:39`, 주석 "역천이 진입 거리 기준") —
이함 직후 위치(원점에서 2.5m 남짓 떨어진 곳)가 이미 이 10m 반경 안에 들어오므로,
FOLLOWING이 실제로 두 번째 WP를 향해 가보기도 전에 "경로 끝에 도달했다"고 즉시 판정.

**근본원인: `d_end_thresh`가 FW(대형 VTOL, 300m급 항로·큰 선회반경) 기준으로 잡힌 10m 상수인데,
이 MC 테스트기체의 왕복 삼각경로(~4~6m 스케일)에 그대로 적용되면서, 경로 전체가 그 판정 반경
안에 들어가 버림.** 그래서 두 번째 waypoint의 부호를 양(+)으로 바꾸든 음(-)으로 바꾸든
**애초에 그 방향으로 날아가 보지도 않고 즉시 "완료"로 착지** — "반대로 명령했는데 같은
방향으로 날았다"가 아니라 **"어느 쪽으로도 실제로 날아가지 않았다"**가 정확한 설명.
flight05(양의 방향, 아래 참조)도 동일 메커니즘으로 재현.

## 그 외
- 착륙 완료(disarm) t=1784885164.42(ARM 후 약 27.8초).
