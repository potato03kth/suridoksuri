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
- **결론:** 천이 실패. 원인은 아래 ①~④가 겹친 것이고, **③이 이번 비행의 최대 수확**이다 —
  `_step_transition_fw`가 기대던 "천이 중 위치 setpoint로 수평 유도" 전제가 PX4 소스 수준에서
  성립하지 않는다는 것이 확인됐다. 펌웨어 교체 자체는 성공했다(파라미터 1437→1436,
  `RC_CRSF_*` 소멸·`UXRCE_DDS_CFG` 신설 — `post-flash.json`은 비행 후 11:14Z 채취본).

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

## ③ 🔴 천이 중에는 위치 setpoint로 수평 유도가 되지 않는다 — 코드의 설계 전제가 틀렸다

`_step_transition_fw` Phase 3 주석(`offboard_node.py:1150-1153`)은 이렇게 적혀 있다:

> 위치 setpoint는 MC·FW 양쪽에서 작동: MC가 WP1 방향으로 가속하며 전이 → FW가 동일 위치
> setpoint로 직선 추종한다. (사전가속 불필요)

**실기체 PX4(`c890d9db0a`) 소스가 이 전제를 반증한다.** `standard.cpp`의 `TRANSITION_TO_FW`
분기(`:201-245`)는 자세 지령을 이렇게 덮어쓴다:

```cpp
if (_v_control_mode->flag_control_climb_rate_enabled) {
    roll_body = Eulerf(Quatf(_fw_virtual_att_sp->q_d)).phi();   // FW 제어기 값
}
...
pitch_body = math::radians(_param_fw_psp_off.get()) * (1.0f - mc_weight);  // FW_PSP_OFF=0.0
_v_att_sp->thrust_body[0] = _pusher_throttle;
```

**천이 중 roll·pitch는 MC 위치제어가 아니라 FW 제어기/`FW_PSP_OFF`가 지배한다.** 이 기체는
`FW_PSP_OFF = 0.0`이라 pitch는 0°로 고정된다 — 멀티콥터가 수평 이동하려면 기울어야 하는데,
천이 중에는 기울 수 없다. 즉 **우리가 10Hz로 발행한 위치 setpoint의 수평 성분은 천이 구간에서
기체를 움직일 수단 자체가 없다.** SITL에서 통했던 것은 SITL이 무풍이라 표류가 안 보였기 때문이다.

실측이 이와 정확히 일치한다 — 천이 명령 후 3.4초간 `(N=-65.14, E=-21.64, U=21.20)`을 계속
발행했지만:

| 축 | 지령 | 실측 거동 |
|---|---|---|
| 수평 | 남서 −161.6°, 65m | **정동 90~96°**로 0→2.4 m/s, N방향 변위 사실상 0 |
| 수직 | 21.2m (−29.6m 계단) | 50.5m 유지, 하강 시도 없음 |

직전 정렬 20초간 위치가 1.2m 이내로 잘 유지된 것과 대조된다 — **위치제어가 살아 있을 땐 바람을
이겼고, 천이에 들어가 자세 권한을 잃자 그대로 밀렸다.** 정동 표류는 바람 방향으로 보는 것이
가장 단순한 설명이다(확정에는 ulog `wind_estimate` 필요).

> 앞서 이 정동 고정을 F-17(천이 중 PX4가 `course=0`=정북 지령) 계열로 의심했으나, 위 소스 근거가
> 나온 뒤로는 **그 가설을 밀 이유가 약해졌다.** F-17은 정북이고 이번은 정동인 데다, 자세 권한
> 박탈만으로 관측이 다 설명된다. 그래도 배제는 ulog `position_setpoint_triplet.current.course`로
> 해야 한다(같은 ulog가 F-17 패치 적용 여부 검증도 겸한다).

## ④ 천이 완료 조건은 대기속도 12 m/s — 시간만으로는 절대 안 넘어간다

`vtol_type.cpp:174-193`:

```cpp
if (airspeed_triggers_transition) {           // 대기속도계가 유효하면
    transition_to_fw = minimum_trans_time_elapsed          // VT_TRANS_MIN_TM = 8.0 s
                       && _attc->get_calibrated_airspeed() >= getTransitionAirspeed();  // VT_ARSP_TRANS = 12.0 m/s
} else {
    transition_to_fw = openloop_trans_time_elapsed;        // VT_F_TR_OL_TM = 10.0 s
}
```

이 기체는 `SENS_EN_MS4525DO=1`·`ASPD_PRIMARY=1`로 **대기속도계가 활성**이므로 위쪽 가지를 탄다.
**12 m/s에 도달하지 못하면 시간이 아무리 지나도 FW로 넘어가지 않고**, `VT_TRANS_TIMEOUT=20.0 s`에
걸려 천이 실패로 끝난다. 관측된 10.8초 지속은 그 20초 창 안에 있었다 — 조종사가 개입하지 않았어도
**12 m/s를 못 냈다면 결과는 같았다.**

그런데 가속이 거의 없었다: `VT_PSHER_SLEW=1.0`/s · `VT_F_TRANS_THR=1.0`이라 **pusher는 1초 만에
100% 지령**에 도달한다. 그럼에도 3.4초간 지면속도 2.4 m/s에 정착(가속이 멎음)했다. pusher가 100%로
돌았다면 나올 수 없는 값이므로 **전진 추력이 실제로는 발생하지 않았다는 정황이 강하다.**

### ✅ ulog 로 확정 (2026-07-28 저녁 회수, `log_215_2026-07-28-10-29-12.ulg`)

로그 동일성 확인: `vtol_state` 3→1 at t=56.28s → 3 at t=67.12s = **10.84초**(관측 10.8초와 일치),
`nav_state` 14(OFFBOARD)→2(POSCTL) at t=59.35s = 천이 명령 **3.07초** 후 인계. 같은 비행이 맞다.

**🔴 결함 A — pusher 지령·PWM 은 완벽했다. 모터가 안 돌았다.**

| 단계 | 실측 | 판정 |
|---|---|---|
| 제어할당 (`actuator_motors`) | m4 = 0.22 → **1.00 포화** (56.47→57.67s, 1.4초) | `VT_PSHER_SLEW=1.0/s` 대로 정상 |
| PWM 출력 (`actuator_outputs` **multi_id=1**) | out4 = 1000(정지) → 평균 **1868**, 최대 **1900** | `PWM_AUX_MAX5`(1900) 포화 — 정상 |
| 배터리 전류 | 호버 22.40 A → pusher 100% 구간 23.22 A = **+0.81 A** | 🔴 **사실상 무부하** |

`CA_ROTOR4_AX=1.0`(유일한 X축 추력) → m4 가 pusher, `PWM_AUX_FUNC5=105`(Motor5) → AUX5 출력.
4S 프로펠러 모터가 100%로 돌면 수십 A 가 늘어야 하는데 **+0.81 A** 다.
⚠ `multi_id=0`(PWM MAIN)은 제어면 서보 4개(`PWM_MAIN_FUNC1~4 = 201~204`)라 out4~7 이 0 이다 —
이것만 보고 "PWM 미출력"으로 오판하기 쉽다. 모터는 **`multi_id=1`(AUX)** 을 봐야 한다.

> **결론: 소프트웨어·설정은 무죄다.** PX4 는 지령을 냈고 PWM 도 최대로 나갔다.
> 원인은 pusher **ESC / 모터 / 배선 / 프로펠러** — 순수 하드웨어다.
> (전류계가 pusher ESC 급전 경로를 지나는지는 배선 확인 필요. 다만 지면속도 2.4 m/s 정착이
> 무추력을 독립적으로 뒷받침한다.)

**🔴 결함 B — 대기속도계가 음수를 낸다. pusher 를 고쳐도 천이는 완료되지 않는다.**

```
differential_pressure   평균 -26.95 Pa   범위 [-45.77, -13.15]     ← 전진 중인데 계속 음수
calibrated_airspeed     평균  -2.48 m/s  최대 +3.98 m/s
SENS_DPRES_OFF = -21.78 (오프셋 보정을 이미 먹인 뒤에도 음수)
```

전진 비행에서 차압은 **양수**여야 한다. 계속 음수라면 핏토관 **역결선**(total↔static 뒤바뀜) ·
막힘 · 미장착 중 하나다. 이게 왜 치명적이냐면 — ④ 위 인용대로 `SENS_EN_MS4525DO=1`·
`ASPD_PRIMARY=1` 이라 천이 완료 판정이 **대기속도 가지**를 타고, 그 값이 음수면 기체가 실제로
12 m/s 를 내도 **`CAS >= VT_ARSP_TRANS(12.0)` 이 영원히 성립하지 않는다.**

> **따라서 A·B 를 모두 고쳐야 천이가 된다. 하나만 고치면 여전히 실패한다.**
> ⚠ `SENS_EN_MS4525DO=0` / `ASPD_PRIMARY=-1` 로 대기속도계를 꺼서 개루프 가지
> (`VT_F_TR_OL_TM=10s`, 시간만으로 천이)로 우회하는 길이 있으나 **권장하지 않는다** —
> 속도 미달 상태로 FW 로 넘어가면 실속·낙하다. A 를 먼저 고칠 것.

배경으로 **상승 중 전압이 11.63V(셀당 2.91V)까지 무너진 상태**였다 — 50m 상승(①의 결과)이
배터리를 먼저 소진시킨 뒤 고출력이 필요한 천이에 들어간 셈이다.

## 조종사 개입 평가

천이 명령 3.4초 후 POSCTL 인계. 기체가 목표와 무관한 방향으로 가속하던 중이므로 **판단·시점 모두
타당**하다. 노드는 `PILOT_TAKEOVER`로 정상 진입해 세트포인트 발행을 멈췄고 재요청하지 않았다
(2026-07-25 사고 대책이 실기체에서 의도대로 동작한 것 — 이 부분은 합격).

## ✅ ulog 회수 완료 (2026-07-28 저녁)

`log_215_2026-07-28-10-29-12.ulg` (6,518,323 B,
sha256 `1fc386b30c97861e08d3d1b1a69d39be4058f660c218499c9bccdb0a41c35ce1`).
비행 후 FC USB 가 분리돼 자동회수·스윕이 모두 건너뛰어졌던 것을 `pull_ulog.py --log-id 215` 로
직접 받았다. 펌웨어 `ver_sw = c890d9db0a…`(실기체 커밋과 일치), 로그 길이 129.6초.

| 확인 대상 | 결과 |
|---|---|
| `actuator_motors` / `actuator_outputs` (pusher) | ✅ **④ 확정 — 지령·PWM 정상, 전류 +0.81 A = 무추력.** 하드웨어 결함 A |
| `airspeed` / `differential_pressure` | ✅ **신규 결함 B 발견 — 차압 -26.95 Pa, CAS 음수.** 천이 완료 조건을 영구 차단 |
| `wind` / `wind_estimate` | ⚠ **토픽 자체가 로그에 없다**(104개 토픽 중 부재) — 정동 표류의 바람 기여분은 이 로그로 정량화 불가 |
| `vtol_vehicle_status`, `battery_status` | ✅ 위 표에 반영 |

**남은 미확인 1건:** `position_setpoint_triplet.current.course` / `.type` → ③이 F-17 계열인지,
F-17 패치가 실제로 먹었는지. 토픽은 로그에 있으므로 **추가 회수 없이 이 파일로 분석 가능하다.**
다만 결함 A·B 가 천이 실패를 이미 완전히 설명하므로 우선순위는 낮다.

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
