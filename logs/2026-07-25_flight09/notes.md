# 2026-07-25_flight09 — launch 인자 오타로 무산

- **비행 조건:** (기체/모드/launch 인자: vehicle_type:=mc transition_alt:=5.0 waypoints:=[0.0,0.0,5.0, 0,0,-5.0,5.0, -5.0, -5.0,5.0,  0.0,0.0,5.0])
- **관찰:** `waypoints must be a flat [x,y,z,...] list (len % 3 == 0), got len=13` — 원소 13개. 1초 만에 종료.
- **결론:** launch 인자 사전검증 필요.

> 세션 전체 리뷰(장소 분류·판정표·원인분석): `../2026-07-25_review.md`
