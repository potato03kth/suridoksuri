# 2026-07-28_flight02

- **비행 조건:** (기체/모드/launch 인자: vehicle_type:=vtol transition_alt:=20.0 waypoints:=[0.0,0.0,20.0, -60.0,-20.0,20.0])
  - ✈ vtol-실기체 트랙. **F-17/F-4 PX4 패치 플래시 후 첫 실비행**, 첫 MC→FW 천이 시도.
  - ⚠ **실제로 적용된 인자는 위와 다르다** — 아래 ① 참조. 실효 `transition_alt = 50.0`(YAML 기본값).
- **관찰:**
  - 이륙~50m 상승 34s → 헤딩 정렬 17s(−35°→−162°, err 0.6°) → 천이 명령 → **3.4초 후 조종사 인계(POSCTL)**.
  - `vtol_state`: 3(MC) →[t=58.2]→ 1(TRANSITION_TO_FW) →[t=69.0, 10.8초 후]→ 3(MC). **FW(4) 도달 못 함.**
  - 천이 명령 후 기체는 목표(남서 −161.6°)가 아니라 **정동으로 표류** — 코스가 90.7→96.7→92.4°의 좁은 밴드에 갇힌 채 속도만 0→2.4 m/s로 단조 가속.
  - 같은 구간 고도: 지령 21.2m인데 실측 50.5m 유지(하강 시도 없음).
  - 배터리 4S: 만충 16.67V → **상승 중 최저 11.63V(셀당 2.91V)** → 호버 복귀 13.1~13.5V.
- **결론:** 천이 실패. 원인은 아래 ①~④가 겹친 것이고, ③은 실기체에서 처음 드러난 신규 결함이다.

## ① launch 인자가 non-breaking space로 붙어 `transition_alt`가 통째로 유실 (확정)

launch가 노드에 넘긴 오버라이드 파일(`/tmp/launch_params_qd2l7x1s`, 컨테이너 내부):

```yaml
/**:
  ros__parameters:
    vehicle_type: "vtol\_transition_alt:=20.0"   # \_ = U+00A0
    waypoints: !!python/tuple [0.0, 0.0, 20.0, -60.0, -20.0, 20.0]
```

인자 구분 공백이 U+00A0(non-breaking space)여서 쉘이 단어 분리를 하지 않았다. `transition_alt:=20.0`이
`vehicle_type` 값에 흡수돼 **천이 고도는 YAML 기본값 50.0m가 쓰였다.** 같은 원인으로 flight01은
`malformed launch argument ' '`로 즉사했다. `vehicle_type`이 `"vtol transition_alt:=20.0"`라는
쓰레기 문자열이 됐는데도 **아무 검증에 걸리지 않았다** — `_is_mc` 판정이 `== "mc"`라 우연히 VTOL로
동작했을 뿐이다.

## ② 그 결과 F-9(천이 고도 계단)가 최대폭으로 발현

`_step_transition_fw` Phase 3은 위치 setpoint 고도로 `self._cruise_alt`(= `wp[-1].z` = 21.20m)를 쓴다
(`offboard_node.py:1154`). 기체는 `transition_alt`(50m)까지 올라가 있으므로 천이 시작 순간
**−29.6m의 고도 계단**을 지령했다. `transition_alt`가 의도대로 20.0이었다면 계단은 1.2m였다.
캠페인에서 이미 F-9(`transition_alt≠wp[-1].z` 시 고도계단 ±30·−70m)로 기록된 결함의 실기체 재현이다.

## ③ 🔴 천이 중 PX4가 우리 위치 setpoint를 따르지 않았다 (신규, 실기체 최초 관측)

코드는 "위치 setpoint는 MC·FW 양쪽에서 작동한다"는 전제로 짜여 있다(`:1150-1153` 주석).
실측은 그 전제를 반증한다 — 천이 명령 후 3.4초간 10Hz로 `(N=-65.14, E=-21.64, U=21.20)`을
계속 발행했는데:

| 축 | 지령 | 실측 거동 |
|---|---|---|
| 수평 | 남서 −161.6°, 65m | **정동 90~96°**로 가속, N방향 변위 사실상 0 |
| 수직 | 21.2m (−29.6m) | 50.5m 유지, 하강 시도 없음 |

세 축 모두 지령과 무관하게 움직였다. 코스가 정동 좁은 밴드에 고정된 채 속도만 단조 증가하는
패턴은 바람 표류의 모양이 아니라 **유도 지령의 모양**이다(직전 정렬 20초간 위치는 1.2m 이내로
잘 유지됐으므로 그만한 바람이 불고 있었다는 근거도 없다).

F-17(천이 중 PX4가 `course=0`=정북을 지령)과 같은 계열로 의심되나, **원래 F-17은 정북이고 이번은
정동이라 동일 결함이라고 단정할 수 없다.** 확정에는 ulog의 `position_setpoint_triplet.current.course`가
필요하다 — 아래 "미확보 데이터" 참조.

## ④ 천이 자체가 완료되지 못함 + 전압 붕괴

`TRANSITION_TO_FW`에 10.8초 머물렀으나 수평속도 최대 2.4 m/s로, 천이 완료 대기속도에 한참 못 미쳤다
(SITL 정천이 실측은 2.42~2.60초). 전진 가속이 사실상 없었다. 배경으로 **상승 중 전압이 11.63V
(셀당 2.91V)까지 무너진 상태**였다 — 50m 상승(①의 결과)이 배터리를 먼저 소진시킨 뒤 고출력이
필요한 천이에 들어간 셈이다. pusher 추력 부족인지 지령 부재인지는 ulog `actuator_outputs` 필요.

## 조종사 개입 평가

천이 명령 3.4초 후 POSCTL 인계. 기체가 목표와 무관한 방향으로 가속하던 중이므로 **판단·시점 모두
타당**하다. 노드는 `PILOT_TAKEOVER`로 정상 진입해 세트포인트 발행을 멈췄고 재요청하지 않았다
(2026-07-25 사고 대책이 실기체에서 의도대로 동작한 것 — 이 부분은 합격).

## 미확보 데이터 (다음 세션 필수)

**이 비행의 ulog가 아직 없다.** 비행 후 FC USB가 분리돼(`/dev/ttyACM*` 없음) `record_flight.sh`의
자동 회수와 `collect_new_logs.py`의 스윕이 둘 다 건너뛰어졌다. FC에 전원을 넣고
`pull_ulog.py --out logs/2026-07-28_flight02/`로 받아야 아래가 확정된다:

- `position_setpoint_triplet.current.course` / `.type` → ③이 F-17 계열인지, 패치가 실제로 먹었는지
- `actuator_outputs` (pusher PWM) → ④가 지령 부재인지 추력 부족인지
- `wind_estimate` / `airspeed` → 정동 표류의 바람 기여분
- `vtol_vehicle_status`, `battery_status` (부하 전류)

## 재현 방법

위 수치는 전부 rosbag만으로 재현된다(ROS 설치 불필요 — CDR 직접 파싱). 이번에 쓴 도구를
`tools/flight_logs/`에 정식 편입해 뒀다:

```bash
# 천이 구간 타임라인 (③의 표·코스 밴드가 그대로 나온다)
python3 tools/flight_logs/transition_timeline.py \
    logs/2026-07-28_flight02/rosbag/rosbag_0.db3 57.8 62.0 --step 0.4 --t0 1785234421.4

# 개별 토픽 CSV 덤프
python3 tools/flight_logs/bagdump.py <db3> {pose|vel|ext|state|sp_pos|sp_vel}
```

`--t0 1785234421.4`는 이 비행의 launch 시작 시각이다(생략하면 첫 pose 샘플이 0).
