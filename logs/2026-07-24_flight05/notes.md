# 2026-07-24_flight05

- **비행 조건:** 09:27:27(단발 블립, 200KB) + 09:28:05(본비행, 1.57MB) KST. ulog만 수동회수(rosbag/launch.log 없음).

## 관찰 — 오프보드 발행 여부 (사용자 질의 1)

flight04와 동일 패턴: `vehicle_command` `DO_SET_MODE`(176, param2=6=OFFBOARD) **t=11.61s**에 명시적 기록 → `nav_state` **t=11.61s에 14:OFFBOARD 진입 확인**. **오프보드 발행됨, 실제 오프보드 모드 맞음.** t=15.51s `DO_SET_MODE`(param2=4=AUTO)로 18:AUTO_LAND 전환(OFFBOARD 유지시간 3.9초).

## 관찰 — WP 반대방향 명령인데 같은 방향으로 비행 (사용자 질의 2)

`trajectory_setpoint` 디코드 결과 flight04와 사실상 동일한 패턴:
- STREAMING 단계: 현재 드리프트 위치(N=-4.6~-4.8m, E=-2.4~-2.7m) 미러링.
- **OFFBOARD 진입 직후(t≈11.2s) 목표가 (N=0.0, E=0.0)으로 스냅** — flight04와 동일한 원점 목표.

**해석:** flight04·flight05 두 비행 모두 FOLLOWING이 추종한 첫(유일) 목표가 동일하게 원점(0,0) — 두 비행 사이에 waypoint 순서를 반대로 명령했더라도 **관측된 결과(원점으로 향함)는 두 비행에서 구분되지 않음**. flight04 notes.md의 팰린드롬 리스트 가설과 정합적. 확정하려면 launch.log(`waypoints:=` 문자열) 필요 — 이번엔 미기록.

## 그 외
- roll 최대 25.24°(t=12.92s), 배터리 부하중 최저 11.99V/34.5A. 얼로케이터 포화 없음.
