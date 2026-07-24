# 2026-07-24_flight04

- **비행 조건:** 09:25:21(단발 블립, 200KB) + 09:25:37(본비행, 1.48MB) KST. ulog만 수동회수(RPi tailscale 단절로 rosbag/launch.log 없음 — `waypoints:=` 실제 launch 인자 문자열은 확인 불가, trajectory_setpoint로 역추정).

## 관찰 — 오프보드 발행 여부 (사용자 질의 1)

`vehicle_command`에 `DO_SET_MODE`(176, param2=6=OFFBOARD)가 **t=11.90s**에 명시적으로 기록됨(컴패니언 컴퓨터→FC 요청 — PX4가 스스로 OFFBOARD로 안 감, 외부 요청만 가능) → `nav_state`가 **t=11.9s에 실제로 14:OFFBOARD 진입 확인**. 즉 **오프보드 발행됨, 실제 오프보드 모드 맞음.**
단 지속시간이 매우 짧음 — t=14.49s에 다시 `DO_SET_MODE`(param2=4=AUTO)가 기록되며 `nav_state`가 18:AUTO_LAND로 전환(OFFBOARD는 2.6초만 유지). 이후 t=26.17s STANDBY(disarm)까지 OFFBOARD가 다시 표시되나 이는 disarmed 상태의 잔여 표시로 실질적 재진입 아님.

## 관찰 — WP 반대방향 명령인데 같은 방향으로 비행 (사용자 질의 2)

`trajectory_setpoint`(offboard_node가 발행하는 위치 커맨드) 직접 디코드 결과:
- OFFBOARD 진입 전(STREAMING 단계): 목표가 계속 기체의 현재 위치(약 N=-2.1~-2.8m, E=-1.1~-1.5m, 이함 후 자연 드리프트)를 그대로 미러링 — 설계대로 정상.
- **OFFBOARD 진입 직후(t≈11.17s, `nav_state` OFFBOARD 진입과 거의 동시) 목표가 (N=0.0, E=0.0)으로 스냅** — 이것이 FOLLOWING이 추종한 **유일한 실제 waypoint 목표점**. 이후 AUTO_LAND 전환까지 이 값 부근에 머묾(2차 목표점에는 도달 전 착륙 전환됨).

**해석:** 이 세션 내 실사용 waypoint 리스트는 과거 세션들과 동일하게 `[0,0,h, -Δx,-Δy,h, 0,0,h]` 형태의 **왕복 삼각경로(원점으로 회귀)**로 추정됨 — 이 리스트는 **원점(wp0)과 종점(wp2)이 같아 통째로 뒤집어도(reverse) 배열이 동일**하다. 따라서 "방향을 반대로 명령"했더라도 FOLLOWING이 추종하는 목표(원점 (0,0))는 바뀌지 않아 **같은 방향으로 나는 게 코드상 정상 동작**이다(버그 아닐 가능성 높음). `fc_bridge/guidance/l1_guidance.py::_find_segment()`(183~208행)도 리스트 인덱스가 아니라 **현재 위치에서 가장 가까운 구간을 매 순간 전체탐색으로 선택**하는 방식이라, 설령 비대칭 리스트를 뒤집었더라도 이함 위치가 같다면 유도기가 같은 첫 구간을 고를 수 있음 — 이 케이스 역시 배제 못함.
**미확정 부분:** 실제로 두 번의 비행에 각각 어떤 `waypoints:="[...]"` 문자열을 넣었는지는 launch.log가 없어 직접 확인 못함(rosbag 미기록) — 위 결론은 trajectory_setpoint 역추정 근거. 다음 비행부터는 `record_flight.sh`로 launch.log를 남겨야 이 부분이 확정됨.

## 그 외
- roll 최대 22.27°(t=12.62s), 배터리 부하중 최저 12.17V/33.6A. 얼로케이터 포화 없음.
