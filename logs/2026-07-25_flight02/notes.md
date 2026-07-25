# 2026-07-25_flight02 — 무산 (MAVROS 미기동)

- **비행 조건:** vehicle_type:=mc transition_alt:=3.0 waypoints:=[0.0,0.0,3.0, -4.24,4.24,3.0, 0.0,0.0,3.0]
- **시각:** 2026-07-25 06:32:05~06:32:18 UTC (**15:32 KST**), 13초
- **결과:** 비행 없음. `offboard_node`가 시작하자마자 `/mavros/cmd/arming 서비스 없음`을
  10 Hz로 계속 경고했고 `telemetry_node`는 `pos_ned=[0. 0. 0.] armed=False vtol=0`
  (기본값 그대로 = MAVROS 토픽 미수신)만 찍었다. **launch에 MAVROS가 안 떠 있었다.**
- **원인 정황:** flight01 직후 06:26~06:31 사이 FC 재부팅이 있었고
  (`../2026-07-25_manual/notes.md` 참조), 그 뒤 MAVROS를 다시 올리기 전에 launch를 실행한 것.
- **주의:** 같은 시각의 FC 로그 `log_145`(06:32:09, armed 0.60s)는 이 launch가 아니라
  **조종기로 한 아밍체크**다. flight02 자체는 FC에 아무 명령도 못 보냈다.
- **조치:** `record_flight.sh` 실행 전 `/mavros/state`의 `connected=True` 확인 절차를
  넣을 것. 지금은 서비스가 없어도 launch가 그냥 진행돼 13초를 버린다.
