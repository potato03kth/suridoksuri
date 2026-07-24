# 2026-07-24_flight01

- **비행 조건:** (기체/모드/launch 인자: vehicle_type:=mc transition_alt:=4.0 waypoints:=[0.0,0.0,4.0, -4.24,-4.24,4.0, 0.0,0.0,4.0]). ARM 요청 09:24:37 UTC(18:24:37 KST).
- **관찰:** `ARM 요청` 이후 텔레메트리가 계속 `armed=False`로 유지 — **ARM 자체가 실패**(원인 미상, 이 launch.log만으론 거부사유 확인 불가). OFFBOARD/FOLLOWING 로그 없음.
- **결론:** 오프보드·WP 질의와 무관(비행 자체가 시작 안 됨).
