# 2026-07-23_manual

- **내용:** ulog 1건(`10_42_23.ulg`, PX4 부팅 19:40:31 KST, 로그 19:42:23~19:43:34 KST, 71.9초). `2026-07-23_flight01`(20:32:15 KST 시작, 오프보드 사고 발생 비행)보다 앞선, 별개의 짧은 비행 — RPi(컴패니언 100.67.27.83)는 이 시각(18:23~20:03 KST, journalctl 확인) 완전히 다운돼 있었으므로 **컴패니언/오프보드 개입 없이 진행된 순수 수동 비행**이다. 대응하는 rosbag·launch.log·notes.md는 존재하지 않음(record_flight.sh 미사용).
- **분석 (analyze_flight.py):** `nav_state`가 전 구간 POSCTL(t=0.5s)이다가 종료 직전 t=69.16s에 MANUAL로 전환 — `vehicle_command` 토픽 자체가 없어 어떤 MAVLink 커맨드도 수신하지 않았음(오프보드/미션 관련 토픽 전무, `position_setpoint_triplet`도 없음). AGL 최대 18.57m(t=25.25s, 조종사 수동 상승). 컴패스 결함("Compass needs calibration")·"Emergency battery level" 경고가 반복 기록됨. t=66.96s Kill engaged → t=70.93s Disarmed by landing.
- **결론:** `2026-07-23_flight01`의 30m 상승/오프보드 미이행 사고와 무관한 별개 비행 — 아마 이함 전 점검성 수동 호버 테스트로 추정(컴패스/배터리 경고 확인 목적일 가능성, 미확인). 상세는 `analysis_auto.md` 참조.
