# 2026-07-24_flight01

- **비행 조건:** 08:42:22 KST, ulog만 수동회수(RPi tailscale 단절로 rosbag/launch.log 없음, F:\log 경유 E:\Downloads → 이 저장소로 수동 이관)
- **관찰:** `nav_state`가 시작(POSCTL, ARMED)부터 종료(t=31.57s, MANUAL, ARMED)까지 AUTO_TAKEOFF/OFFBOARD 전환이 전혀 없음 — 순수 수동(POSCTL/MANUAL) 비행. roll 최대|값|=174.65°(t=32.82s), pitch 66.12°(급격한 자세, 착륙 직전 추정) — `analysis_auto.md` 참조.
- **결론:** 오프보드 미관련 비행(수동 조종 테스트 또는 착지 시도). 이 비행 자체는 오늘의 오프보드/WP 질의와 무관.
