# 2026-07-24_flight04

- **비행 조건:** (기체/모드/launch 인자: vehicle_type:=mc transition_alt:=4.0 waypoints:=[0.0,0.0,4.0, 4.24,4.24,4.0, 0.0,0.0,4.0] — 부호를 반대로 바꿔 테스트). ARM 요청 09:27:14 UTC(18:27:14 KST).
- **관찰:** flight01·02와 동일 — `ARM 요청` 후 `armed=False` 유지, ARM 실패. OFFBOARD/FOLLOWING 로그 없음.
- **결론:** 오프보드·WP 질의와 무관(비행 자체가 시작 안 됨) — 방향을 바꾼 뒤 첫 시도는 ARM부터 실패해서 방향전환 자체를 테스트 못 함. 실제 방향전환 비교는 flight03(음의 방향) vs flight05(양의 방향, ARM 성공)로 이뤄짐.
