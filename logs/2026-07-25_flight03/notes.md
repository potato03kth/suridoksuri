# 2026-07-25_flight03 — 무산 (MAVROS 미기동, 즉시 종료)

- **비행 조건:** vehicle_type:=mc transition_alt:=3.0 waypoints:=[0.0,0.0,3.0, -4.24,4.24,3.0, 0.0,0.0,3.0]
- **시각:** 2026-07-25 06:32:43~06:32:44 UTC (**15:32 KST**), 1초
- **결과:** 비행 없음. flight02와 동일하게 `/mavros/cmd/arming 서비스 없음`만 반복,
  1초 만에 Ctrl-C. launch.log 13줄.
- flight02와 같은 원인 — 상세는 `../2026-07-25_flight02/notes.md`.
