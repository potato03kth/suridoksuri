# 배터리 게이트 조사 보고서 — flight02 "Emergency battery level" 규명

작성 2026-07-29 / 조사 세션(코드 수정 없음) / 대상 커밋 `5d55b3f`

조사 지시: `offboard_node` 에 배터리 게이트가 없어 Emergency 수준 배터리로 임무를
계속했다(`f8e951f` 결론). A) PX4 페일세이프가 왜 안 걸렸나 B) 배터리 실측 추이
C) 우리 노드 어디에 무엇을 넣나.

---

## 0. 요약 — 전제가 절반 뒤집혔다

**확정 사실 1.** PX4 가 임무를 멈추지 않은 직접 원인은 `COM_LOW_BAT_ACT = 0`
("Warning" — 어떤 배터리 등급에서도 **아무 동작도 하지 않음") 이다. 코드 게이트
이전에 **파라미터 한 개** 문제다.

**확정 사실 2 (전제 반전).** t=8.64s 의 `Emergency battery level` 은 **오탐이다.**
그 순간 팩은 만충 부근이었다. 근거: 이륙 직전 무부하 16.59V(셀당 4.147V), 착륙 후
무부하 15.2V(셀당 3.80V), 총 소비 802mAh. 방전이 아니라 **전압 새그(sag)를 PX4 가
방전으로 읽은 것**이다. 원인은 `BAT1_R_INTERNAL=-1` + `BAT1_CAPACITY=-1` 조합에서
PX4 내부저항 추정기가 실측의 **1/35** 값(0.5mΩ/셀 vs 실측 17.8mΩ/셀)을 내놓은 것.

**확정 사실 3 (가장 중요).** 이 오탐은 flight02 한정이 아니다. 2026-07-20 ~ 07-28
비행 ulog 11건 중 **의미 있게 비행한 8건 전부**가 이륙 후 **4.5 ~ 7.7초 만에**
`remaining = 0.000` / `warning = EMERGENCY` 에 도달했다(3S·4S 무관). 즉,

> 🔴 **PX4 `remaining` / MAVROS `percentage` 에 거는 게이트는 이 기체에서 모든
> 임무를 이륙 10초 안에 중단시킨다.** 지시서가 암시한 "Emergency 면 중단" 형태의
> 게이트를 그대로 구현하면 **만충 팩으로도 100% 임무 실패**한다.

따라서 이 문서의 권고는 "게이트를 넣자"가 아니라 **"① PX4 파라미터 3개를 먼저
고치고 ② 그 위에 `remaining` 을 쓰지 않는 게이트를 얹는다"** 이다.

**재현성 표기 규약** — 이 문서의 수치는 전부 본 세션이 직접 재현했다. 남의 보고를
인용한 곳은 `[인용]` 으로 표시하고 재현 결과를 나란히 적었다.

---

## A. PX4 쪽 — 왜 페일세이프가 임무를 멈추지 않았나

### A-1. 실측 파라미터

출처: `logs/2026-07-28_px4_flash/px4_params_2026-07-28_final-crsf.json`
(`captured_utc 2026-07-28T13:27:13Z`, 1438개). 실기체 SSH 접속 없음.

| 파라미터 | 실측값 | 의미 |
|---|---|---|
| **`COM_LOW_BAT_ACT`** | **0** | **"Warning" — LOW/CRITICAL/EMERGENCY 전부 경고만** 🔴 |
| `BAT_LOW_THR` | 0.15 | SoC 15% 미만 → LOW |
| `BAT_CRIT_THR` | 0.07 | SoC 7% 미만 → CRITICAL |
| `BAT_EMERGEN_THR` | 0.05 | SoC 5% 미만 → EMERGENCY |
| `BAT1_N_CELLS` | 4 | 4S |
| `BAT1_V_EMPTY` | 3.60 | SoC 0% 기준 셀전압 |
| `BAT1_V_CHARGED` | 4.20 | SoC 100% 기준 셀전압 |
| **`BAT1_CAPACITY`** | **-1.0** | **용량 미설정 → 쿨롱 카운팅 비활성** 🔴 |
| **`BAT1_R_INTERNAL`** | **-1.0** | **내부저항 미설정 → 추정기 사용(아래 A-4 에서 실패 확인)** 🔴 |
| `COM_ARM_BAT_MIN` | -1.0 | 별도 시동 임계 없음 → `BAT_CRIT_THR` 이 기준 |
| `COM_DISARM_LAND` | 2.0 | 착륙 감지 2초 후 자동 시동해제 |
| `NAV_RCL_ACT` | 2 | RC 상실 시 RTL |
| `COM_RCL_EXCEPT` | 0 | RC 상실 예외 없음 (**OFFBOARD 도 예외 아님**) |
| `COM_OBL_RC_ACT` | 0 | Offboard 상실 시 Position 모드 |
| `COM_FAIL_ACT_T` | 5.0 | 페일세이프 발동 전 Hold 5초 |
| `NAV_FORCE_VT` | 1 | RTL 시 VTOL 강제 역천이 (→ RTL 이 안전한 복귀 수단) |
| `RTL_RETURN_ALT` / `RTL_TYPE` | 60.0 / 1 | |
| `COM_QC_ACT` | 0 | 쿼드슈트 후 동작 |

`COM_OBC_LOSS_T`, `COM_POS_FS_DELAY`, `COM_TAKEOFF_ACT` 는 **이 펌웨어에 존재하지
않는다**(1438개 덤프에 부재).

### A-2. `COM_LOW_BAT_ACT = 0` 의 정확한 의미 (소스 확인)

WSL `Ubuntu-22.04` 의 `/root/PX4-vehicle` (`git log -1` = `c890d9db0a`, 실기체
`ver` 와 동일 커밋. 읽기만 수행, 빌드·git 상태 변경 없음).

`src/modules/commander/commander_params.yaml:156-167` — enum 정의:

```
COM_LOW_BAT_ACT:  0: Warning / 2: Land mode / 3: Return at critical level, land at emergency level
```

`src/modules/commander/failsafe/failsafe.cpp:168-224` `fromBatteryWarningActParam()`:
`WARNING_EMERGENCY` 분기에서 `LowBatteryAction::Warning`(=0) 이면
`options.action = Action::Warn` 로 끝난다. `Action::Warn` 은 메시지만 낸다 —
모드 전환도, 착륙도, RTL 도 없다.

`failsafe.cpp:649-654` 의 게이트도 통과한다:

```cpp
const bool warning_worse_than_at_arming = (status_flags.battery_warning > _battery_warning_at_arming);
const int32_t low_battery_action = warning_worse_than_at_arming ?
                                   _param_com_low_bat_act.get() : (int32_t)LowBatteryAction::Warning;
```

flight02 는 시동 시점 `warning = 0`(무부하 16.59V) 이었고 이후 3 으로 올랐으므로
`warning_worse_than_at_arming == true` → `_param_com_low_bat_act.get()` 이 그대로
쓰였다. 그 값이 0 이라 결과는 `Warn`.

> ✅ **확정:** `COM_LOW_BAT_ACT` 가 2 또는 3 이었다면 PX4 는 **OFFBOARD 를 무시하고**
> Land/RTL 로 전환했을 것이다. 배터리 페일세이프에 OFFBOARD 예외는 없다
> (`COM_RCL_EXCEPT` 는 RC 상실 전용이고 값도 0).

### A-3. 비행 중 "Preflight Fail" 이 반복된 이유 (부수 규명)

`src/modules/commander/HealthAndArmingChecks/checks/batteryCheck.cpp:200-206`:

```cpp
const bool configured_arm_threshold_in_use = !context.isArmed() && (_param_com_arm_bat_min.get() >= -FLT_EPSILON);
...
NavModes affected_modes = (!configured_arm_threshold_in_use && critical_or_higher) ? NavModes::All : NavModes::None;
```

`COM_ARM_BAT_MIN = -1.0` 이므로 `configured_arm_threshold_in_use == false` →
EMERGENCY 이면 `affected_modes = NavModes::All` → `armingCheckFailure()` 발생.
**이미 시동된 상태에서는 시동해제 효과가 없고 보고만 된다.** 로그의
`Preflight Fail: ...` 반복이 이것이다. 이 메시지들은 모드 전환 요청 시점마다
재평가되어 출력된다(t=33.26 AUTO.LOITER, 39.28 OFFBOARD 전환 시점과 일치).

> ⚠️ **부작용:** 지상에서 EMERGENCY 상태면 **시동이 거부된다.** flight02 는 시동
> 시점 팩이 쉬고 있어(무부하) `warning=0` 이라 통과했다. 방금 착륙한 팩으로 즉시
> 재시동하려 하면 거부될 수 있다.

### A-4. 왜 SoC 가 0% 로 붕괴했나 — 소스 + 실측 결합

`src/lib/battery/battery.cpp:244-257` `calculateStateOfChargeVoltageBased()`:

```cpp
float cell_voltage = voltage_v / _params.n_cells;
if (current_a > FLT_EPSILON) {
    updateInternalResistanceEstimation(voltage_v, current_a);
    if (_params.r_internal >= 0.f) { cell_voltage += _params.r_internal * current_a; }   // 미설정
    else                           { cell_voltage += _internal_resistance_estimate * current_a; }  // ← 이 경로
}
_cell_voltage_filter_v.update(cell_voltage);
return math::interpolate(_cell_voltage_filter_v.getState(), _params.v_empty, _params.v_charged, 0.f, 1.f);
```

`battery.cpp:310-326` `estimateStateOfCharge()` — `_capacity_mah > 0` 이 아니면
`_state_of_charge = _state_of_charge_volt_based` 로 **전압만** 쓴다. `BAT1_CAPACITY=-1`
이므로 쿨롱 카운팅 융합이 통째로 비활성이다.

핵심은 새그 보정항 `_internal_resistance_estimate * current_a` 다. **ulog 실측:**

- `battery_status.internal_resistance_estimate` 의 전 구간 유일값 = **{0.0, 0.0005}** Ω/셀
- 본 세션 실측 팩 저항(4초 슬라이딩 창 최소자승, 80창 중앙값) = **71.3 mΩ/팩 = 17.8 mΩ/셀**

즉 PX4 는 실제 새그의 **약 3%** 만 보정했다(0.0005×23A = 0.012 V/셀 보정 vs 실제
0.0178×23A = 0.41 V/셀). `battery.h:219` 의 초기값 `R_DEFAULT = 0.005` 조차 실측의
1/3.6 인데, RLS 추정기가 여기서 **더 낮은 쪽으로** 수렴했다.

추정기가 실패한 구조적 이유(소스 근거, `battery.cpp:277-283`): 공분산 노름이
**감소할 때만** 추정치를 갱신한다(`if (estimation_covariance_temp_norm < _estimation_covariance_norm)`).
공분산 노름은 단조 감소하므로 초기 몇 초의 나쁜 추정이 잠기면 이후 갱신이
차단된다. → *[추론]* 이 잠김이 0.0005Ω 고착의 원인으로 보이나, 리플레이로
확정하지는 않았다(`src/lib/battery/int_res_est_replay.py` 로 검증 가능).

결과적으로 SoC 는 사실상 **무보정 부하전압** 을 `V_EMPTY=3.6` ~ `V_CHARGED=4.2`
에 선형 사상한 값이 된다. 이 기체의 호버 전류 21~24A 에서 팩은 13.8~14.5V
(3.45~3.62 V/셀)에 앉는다. `V_EMPTY×4 = 14.4V` 이므로 **호버에 진입하는 순간
SoC 정의상 0% 근처**다. 이것이 A-3/B 에서 관측된 전부다.

---

## B. ulog 실측 — 배터리가 실제로 어땠나

원본 `logs/2026-07-28_flight02/log_215_2026-07-28-10-29-12.ulg`, `pyulog` 로 직접 추출.
`battery_status` 649 샘플 / 5.02 Hz / 129.5초. `cell_count = 4` 전 구간 일정.

### B-1. warning 전이 (전량)

| t (s) | warning | 팩 전압 | 셀당 | 전류 | remaining | 고도 |
|---|---|---|---|---|---|---|
| 0.06 | 0 NONE | 16.59V | 4.147 | 0.0A | 0.947 | 1.2m |
| 5.66 | **1 LOW** | 14.47V | 3.618 | 21.3A | 0.146 | 2.0m |
| 6.46 | **3 EMERGENCY** | 13.80V | 3.451 | 23.1A | 0.050 | 2.8m |
| 124.66 | 2 CRITICAL | 15.05V | 3.761 | 0.0A | 0.056 | 착륙 후 |
| 124.86 | 1 LOW | 15.07V | 3.768 | 0.0A | 0.097 | 〃 |
| 125.26 | **0 NONE** | 15.09V | 3.772 | 0.0A | 0.159 | 〃 |

`remaining` 은 **t=6.86s 에 0.000 에 도달해 t=124.46s 까지 118초간 0.000 고정**이다.

> ⚠️ **CRITICAL(2) 을 건너뛰고 LOW → EMERGENCY 로 직행**했다. LOW 15% → EMERGENCY 5%
> 사이를 0.8초 만에 통과했기 때문이다(5Hz 로그에서 4샘플). 전압 사상이 선형이라
> 새그 하나로 10%p 를 순식간에 훑는다.

### B-2. `logged_messages` 와의 정합 — 2.2초 지연의 정체

`battery_status.warning` 은 t=6.46s 에 3 이 됐는데 `[health_and_arming_checks]
Emergency battery level` 메시지는 **t=8.64s** 다. 이 메시지는 warning 전이가 아니라
**arming check 재평가 시점**에 나온다(A-3). 이후 t=34.19 / 37.16 / 39.28 / 60.67 /
67.13 / 123.59 에 반복 출력되며, 이는 모드 전환 요청 시점과 일치한다
(t=33.26 AUTO.LOITER, t=39.28 **OFFBOARD 진입**).

> ✅ **확정:** 우리 노드의 OFFBOARD 진입 요청(t=39.28)은 `warning = EMERGENCY`
> 상태에서 **PX4 에 수락됐다.** 시동된 기체의 모드 전환은 배터리 등급으로 막히지
> 않았다.

### B-3. 최저 전압 — 앞선 세션 보고의 재현 검증

**[인용]** `logs/2026-07-28_flight02/notes.md:11` — "만충 16.67V → **상승 중 최저
11.63V(셀당 2.91V)** → 호버 복귀 13.1~13.5V".

**본 세션 재현 결과 — 인용값은 맞다. 다만 출처가 ulog 가 아니다.**

- **ulog `battery_status` 전 구간 최저 = 11.726V (셀당 2.932V), t=26.26s, 고도 36.9m,
  전류 35.6A.** 11.63V 는 ulog 어느 샘플과도 ±0.02V 내에 일치하지 않는다.
- 11.63V 의 실제 출처는 **`logs/2026-07-28_flight02/launch.log:41`** 이다:
  `[telemetry_node] ... alt 36.57m ... batt=11.63V/0%`. 즉 `/mavros/battery` 경로다.
  같은 줄의 16.67V(launch.log:25)가 notes.md 의 "만충 16.67V" 와도 정확히 일치한다.
- **두 값은 모순이 아니다.** ulog 의 `battery_status` 는 **5Hz 로 데시메이션**되어
  기록되고, MAVLink `SYS_STATUS` 는 uORB 를 직접 샘플링한다
  (`src/modules/mavlink/streams/SYS_STATUS.hpp:172` — `voltage_battery = lowest_battery.voltage_v * 1000`).
  MAVROS 가 5Hz 기록 사이의 골을 잡은 것이다. 시각도 일치한다(launch.log:41 의
  고도 36.57m ↔ ulog t=26.26s 의 36.94m, 0.3초 이내).

> 📌 **결론:** 인용값 11.63V/2.91V 는 **유효하다.** 출처만 `launch.log`(MAVROS)로
> 정정하면 된다. ulog 기준 최저는 11.726V/2.932V.

### B-4. 방전인가 새그인가 — 판단 근거

**새그다.** 근거 5종, 전부 본 세션 직접 산출:

1. **무부하 양 끝점.** 이륙 전(I=0.0A) 16.59V = **셀당 4.147V**. 착륙 후 모터
   정지 6초 뒤(I=0.0A) 15.21V = **셀당 3.80V** (여전히 상승 중 — 3.749→3.807).
   `V_EMPTY=3.6`/`V_CHARGED=4.2` 로 사상하면 91% → 33%. **빈 팩이 아니다.**
2. **회복 거동.** t=123.06 에 전류가 20A→7A 로 떨어지자 전압이 0.2초 만에
   13.02V→14.17V 로 **1.15V 튀어올랐다.** 방전된 팩은 이렇게 회복하지 않는다.
3. **소비 전하.** 본 세션 자체 적분 802mAh / PX4 `discharged_mah` 891.5−107.3(초기값)
   = 784.2mAh. 팩 정격을 모르나(=`BAT1_CAPACITY` 미설정) 무부하 양 끝점의 91%→33%
   와 결합하면 정격 ≈ **2000~2500mAh** 로 추정된다 *[추정 — LiPo OCV 곡선의
   비선형성 때문에 단일 비행으로는 확정 불가]*.
4. **새그 크기.** 실측 팩저항 71.3mΩ 에서 23A 시 새그 = 1.64V(셀당 0.41V).
   EMERGENCY 선언 시점(t=6.46) 부하전압 3.451 V/셀 + 새그 보정 = **OCV 3.84 V/셀.**
5. **부하-전압 상관.** 최저전압 시점(t=26.26)의 전류가 35.6A 로 **전 구간 최대치
   부근**이다(최대 37.9A). 최저 전압과 최대 전류가 같은 순간에 온다.

### B-5. 구간별 추이 (10초 이동중앙값)

| t | 고도 | 부하 셀전압 | med10 | 새그보정 OCV med10 | 전류 |
|---|---|---|---|---|---|
| 5 | 1.4m | 3.624 | 4.020 | 4.100 | 22.9A |
| 15 | 12.8m | 3.364 | 3.362 | 3.843 | 26.7A |
| 25 | 34.0m | 3.028 | 3.105 | 3.708 | 35.6A |
| 30 | 44.9m | 2.968 | 3.029 | **3.620** | 33.9A |
| 40 | 51.5m | 3.233 | 3.319 | 3.693 | 19.4A |
| 55 | 50.6m | 3.277 | 3.302 | 3.685 | 20.9A |
| 70 | 46.3m | 3.269 | 3.336 | 3.691 | 20.3A |
| 100 | 18.9m | 3.279 | 3.321 | 3.658 | 20.1A |
| 120 | 4.1m | 3.252 | 3.214 | 3.602 | 20.9A |

**상승 구간(t=10~33)이 전 비행 최고 부하다** — 평균 32.4A, 순항/로이터는 22.0A.
50m 까지 올라간 것(notes.md ① `transition_alt` 유실) 자체가 배터리 최악 조건을
만들었다. 천이 지령 시점(t=56.29)의 상태: 부하 3.197 V/셀, OCV 약 3.56 V/셀.

### B-6. 🔴 이 오탐은 전 비행 공통이다 (신규 발견)

`logs/2026-07-2*_flight*/*.ulg` 11건 일괄 추출. `Vrest` 는 시동 전 `I<0.5A` 샘플 중앙값.

| 비행 | 셀 | 지속 | Vrest/셀 | 최저부하/셀 | 평균I | 소비 | **EMERGENCY 도달** |
|---|---|---|---|---|---|---|---|
| 2026-07-20_flight03 | 3 | 20.4s | 3.794 | 3.252 | 9.6A | 50mAh | **5.0s** |
| 2026-07-21_flight03 | 3 | 8.4s | 3.980 | 3.638 | 4.2A | 9mAh | – |
| 2026-07-23_flight01 | 4 | 88.1s | 3.912 | **2.731** | 23.5A | 559mAh | **5.7s** |
| 2026-07-25_flight05 | 4 | 12.9s | 4.192 | 3.452 | 6.2A | 18mAh | – |
| 2026-07-25_flight08 | 4 | 25.7s | 4.070 | 3.073 | 17.7A | 112mAh | **7.7s** |
| 2026-07-25_flight10 | 4 | 35.5s | 4.009 | 2.999 | 21.8A | 199mAh | **4.9s** |
| 2026-07-25_flight14 | 4 | 36.9s | 3.823 | 2.880 | 19.7A | 186mAh | **5.1s** |
| 2026-07-25_flight16 | 4 | 40.1s | 3.729 | **2.461** | 20.3A | 211mAh | **4.5s** |
| 2026-07-25_flight17 | 4 | 72.3s | 3.996 | 3.132 | 20.1A | 389mAh | – |
| 2026-07-25_flight19 | 4 | 11.7s | 3.868 | 3.490 | 3.8A | 11mAh | – |
| **2026-07-28_flight02** | 4 | 129.5s | 4.150 | 2.932 | 22.8A | 802mAh | **6.5s** |

EMERGENCY 미도달 4건은 모두 **평균전류 6.2A 이하이거나(호버 안 함) 72초 이상
저부하 비행**이다. **실제로 상승·기동한 비행은 예외 없이 이륙 7.7초 안에
EMERGENCY 에 들어갔다.**

팩 저항도 전 기체 공통이다(같은 방법, 13건):

```
3S: 11.9 / 21.7 / 14.9 / 18.4 / 15.1 / 19.7  mΩ/셀
4S: 16.8 / 23.4 / 21.9 / 19.7 / 21.0 / 14.4 / 17.8  mΩ/셀
=> 전체 중앙값 18.4 mΩ/셀  (PX4 추정치 0.5 mΩ/셀, R_DEFAULT 5.0 mΩ/셀)
```

> 🔴 **파생 결론:** 최저 부하전압이 **2.46 ~ 3.49 V/셀** 로 흩어진다. 이 기체는
> 정상 비행에서 일상적으로 3.0 V/셀 아래로 새그한다. 따라서 **"부하전압이 X V/셀
> 아래면 중단" 형태의 게이트는 이 하드웨어에서 정상과 위험을 구분하지 못한다.**
> 2.80 V/셀 백스톱을 걸면 무사히 끝난 flight16(2.461)·flight01(2.731)이 오탐된다.

---

## C. 우리 노드 — 현재 상태와 붙일 자리

### C-1. 현재 배터리 관련 구독·판정: **전무 확인**

`fc_ros/fc_ros/nodes/offboard_node.py` (1994줄) 에 `battery` 문자열 **0건**
(`grep -ci battery` = 0). 구독은 `offboard_node.py:591-613` 의 6개뿐:

```
/mavros/local_position/pose, /mavros/local_position/velocity_local,
/mavros/state, /mavros/extended_state, /mavros/altitude, /fc_ros/override
```

**`/mavros/battery` 구독 없음. 배터리 판정 없음. 확정.**

### C-2. 다만 배선은 이미 절반 깔려 있다 (신규 발견 — 구현량이 작다)

| 위치 | 내용 |
|---|---|
| `fc_bridge/comm/vehicle_state.py:17,31-33` | `VehicleState` 에 `battery_voltage` / `battery_current` / `battery_remaining` **필드가 이미 있다.** 기본값 `remaining = 1.0` 이며 주석에 "미수신 시 1.0 — 저전압 오탐 방지" 로 안전측 규약까지 적혀 있다 |
| `fc_ros/fc_ros/adapters/vehicle_state_bridge.py:67-77` | `update_from_battery()` **구현 완료** (NaN 가드 포함) |
| `fc_ros/fc_ros/nodes/telemetry_node.py:65-67, 92-94` | `TelemetryNode` 는 **이미 `/mavros/battery` 를 구독**한다 |
| `fc_ros/test/test_telemetry_node.py` | 해당 경로 테스트 존재 |

즉 `OffboardNode` 에 필요한 것은 **구독 4줄 + 콜백 3줄** 이고, 새 어댑터는 필요 없다.
`TelemetryNode` 와 완전히 같은 형태로 붙는다.

> ⚠️ **`state.battery_remaining` 은 절대 판정에 쓰면 안 된다.** MAVROS `percentage`
> 는 PX4 `remaining` 그대로다(`SYS_STATUS.hpp:174`). B-6 의 실기체 실측:
> `launch.log:32` 이후 **`0%` 가 비행 내내 고정**. 반면 `state.battery_voltage`
> (= `SYS_STATUS.voltage_battery`, 원 부하전압)와 `state.battery_current` 는 쓸 수 있다.

### C-3. 기존 방어 장치의 관례 (여기에 맞춰야 한다)

| 장치 | 위치 | 형태 |
|---|---|---|
| 조종사 인계 감시 | `offboard_node.py:735-741` | `_control_callback` 최상단, 어떤 상태든 최우선, `return` |
| 거리 상한 감시 | `_range_guard_breached()` `:1310-1329` | 최상단 2순위, **bool 반환**, 초과 시 `_safety_fallback()` |
| 타임아웃 4종 | `_timeout_fallback()` `:1298-1308` | 각 `_step_*` 안에서 호출, **bool 반환** |
| 유일한 폴백 출구 | `_safety_fallback()` `:1285-1296` | `_request_override()` 재사용. docstring 이 **"새 폴백 경로를 만들지 않는다"** 를 명문화 |
| 비활성 규약 | `state_logic.py:362-386` | 임계값 **≤ 0 이면 감시 비활성** |
| 순수 판정 함수 분리 | `fc_bridge/execution/state_logic.py` | 노드는 얇은 래퍼, 테스트는 `fc_bridge/tests/test_state_logic.py` |
| 이륙 거부는 하지 않는 선례 | `_check_path_within_range()` `:954-984` | 경로가 상한 밖이어도 **경고만.** 이유를 docstring 에 명시: *"여기서 이륙을 거부하면 '왜 안 뜨지'가 되어 현장에서 더 위험한 우회(감시 자체를 끄기)를 유도한다"* |

### C-4. 세 층위 검토

#### ① 이륙 전 게이트 (ARM/이륙 거부) — `_step_arm_takeoff()` `:859` 의 `_request_arm` 직전

- **신호:** 무부하 팩 전압. `abs(state.battery_current) < 2.0A` 일 때만 유효 표본으로
  인정 → 시동 전이라 항상 성립.
- **장점:** 유일하게 **깨끗한 측정점**이다(새그 없음, 부하 없음). B-6 표의
  `Vrest/셀` 열이 그 값이고 3.729~4.192 로 잘 분산돼 있어 판별력이 있다.
- **트레이드오프:** `_check_path_within_range()` 는 같은 상황에서 **경고만** 하기로
  이미 결정한 선례가 있다. 다만 그 근거는 "거리 상한은 현장에서 launch 인자로
  조정하는 값"이라는 것이고, **배터리는 팩 교체라는 즉시·지상 해결책이 있다**는
  점에서 성질이 다르다. → 차단하되 `battery_arm_min_v <= 0` 으로 끌 수 있게 한다
  (타임아웃·거리상한과 동일한 비활성 규약).
- **한계 (정직하게):** Vrest 와 최저 부하전압의 상관은 **r = 0.364 (n=11)** 로 약하다.
  비행시간·평균전류가 교란변수다. 저부하 4건을 빼도 표본이 7건이라 유의하다고
  말할 수 없다. **이 게이트는 "새그를 예측한다"가 아니라 "반쯤 쓴 팩으로 뜨지
  않는다"는 위생 규칙으로만 정당화된다.**

#### ② 천이 전 게이트 (FW 천이 차단) — `_step_transition_fw()` `:1010`, `_fw_transition_sent` 발행 직전

- **가치:** MC→FW 천이는 전 비행 최대 출력 이벤트이고 **되돌리기가 가장 비싸다.**
- 🔴 **함정 (이게 핵심):** **천이를 막고 그 자리에 있으면 배터리를 더 쓴다.**
  실측으로 MC 호버·상승이 32.4A, FW 순항 후보 구간이 20A 대다. 50m 에서 MC 로
  홀드하는 것이 천이하는 것보다 소모가 크다. → **천이 차단은 반드시 즉시 하강
  경로와 묶여야 한다.** "막고 대기"는 상황을 악화시킨다.
- 따라서 이 층은 독립 게이트가 아니라 **③의 강등이 천이 직전에 한 번 더 평가되는
  것**으로 설계하는 편이 옳다.

#### ③ 비행 중 강등 — `_control_callback()` `:743` 거리 상한 감시 바로 다음

- **형태:** `_range_guard_breached()` 와 동일하게 bool 반환 + 최상단 배치 + 래치.
  기존 관례와 완전히 일치하며 새 패턴이 아니다.
- **강등 목적지 3안:**
  - (a) `_safety_fallback()` → OVERRIDE(수동 인계). **기존 유일 출구**이지만
    배터리 사건에는 어색하다 — 준비 안 된 조종사에게 넘긴다.
  - (b) `_State.LANDING` → `AUTO.LAND`. **새 경로가 아니다** — 정상 임무 말미에서
    매번 쓰는 검증된 상태다(`_step_landing()` `:1780`). 단, **MC 일 때만 안전**하다.
    FW 상태에서 `AUTO.LAND` 면 PX4 가 고정익 착륙 패턴을 시도한다.
  - (c) `AUTO.RTL`. `NAV_FORCE_VT = 1` 이 실측 확인되므로 PX4 가 역천이를 스스로
    수행한다. `set_mode` 한 번으로 끝나고 VTOL 상태와 무관하다. **다만 우리 노드가
    한 번도 발행해 본 적 없는 모드다** — SITL 실증이 선행돼야 한다.
- **오실레이션:** 전압은 전류가 떨어지면 즉시 회복하므로(B-4 근거 2) 전압 기반
  판정은 반드시 **래치**해야 한다. 기존 `_safety_fallback()` 이 이미 단방향이므로
  관례와 일치한다.

---

## D. 설계안 3개와 트레이드오프

### 설계안 1 — "PX4 등급 그대로 쓰기" (지시서가 암시한 형태)

`state.battery_remaining` 또는 MAVROS `percentage` 를 보고 임계 이하면 강등.

- 구현량 최소.
- 🔴 **채택 불가.** B-6: 실제로 기동한 비행 7건 전부가 이륙 4.5~7.7초에 EMERGENCY.
  `launch.log:32` 이후 `0%` 고정이 실기체 MAVROS 인터페이스에서 직접 확인된다.
  **만충 팩으로 100% 임무 실패한다.**

### 설계안 2 — 부하전압 하한 백스톱

`state.battery_voltage / n_cells` 가 X V/셀 아래로 N초 지속되면 강등.

- PX4 SoC 에 의존하지 않음. 구현 단순.
- 🔴 **판별력 없음.** B-6: 무사히 끝난 비행들의 최저 부하전압이 2.461 / 2.731 /
  2.880 V/셀이다. 오탐 없이 잡으려면 임계를 2.4 V/셀 아래로 내려야 하는데, 그건
  LiPo 손상 영역이라 **경보로서 이미 늦다.** 정상과 위험이 겹쳐 있다.
- 부수 문제: `/mavros/battery` 발행 주기가 **미측정**이라 N(디바운스)을 정할
  근거가 없다(rosbag 에 이 토픽이 없다 — D-4 참조).

### ✅ 설계안 3 (권고) — **2단: PX4 캘리브레이션 선행 + 노드는 전하 예산 게이트**

#### 3-A. PX4 파라미터 3개 (이것이 주 조치다)

| 파라미터 | 현재 | → | 근거 |
|---|---|---|---|
| `BAT1_R_INTERNAL` | -1.0 | **0.020** | 13개 비행 슬라이딩창 실측 중앙값 18.4 mΩ/셀, flight02 단독 17.8 mΩ/셀 (D-3 에 검증 있음) |
| `BAT1_CAPACITY` | -1.0 | **팩 정격 mAh** | 설정 시 `battery.cpp:310-322` 가 쿨롱 카운팅으로 SoC 상한을 잡아 **전압 새그에 둔감해진다.** 현재는 이 경로가 통째로 죽어 있다 |
| `COM_LOW_BAT_ACT` | 0 | **3** (critical→RTL, emergency→Land) | A-2. `NAV_FORCE_VT=1` 이라 RTL 이 역천이를 알아서 한다 |

**이것이 최고가치 조치인 이유:** PX4 자체 페일세이프는 **모드 무관**하게 동작한다
(A-2). 우리 노드가 OFFBOARD 로 무슨 짓을 하든 PX4 가 뺏어온다. 노드 게이트는
`offboard_node` 가 살아 있을 때만 작동하지만, PX4 페일세이프는 노드가 죽어도,
링크가 끊겨도 작동한다.

⚠️ **반드시 위 3개를 함께 바꿔야 한다.** `COM_LOW_BAT_ACT` 만 3 으로 올리고
캘리브레이션을 안 하면 **모든 비행이 이륙 7초에 자동 착륙한다.**

#### 3-B. 노드 게이트 — 소비 전하 예산

`fc_bridge/execution/state_logic.py` 에 순수 함수 추가, `offboard_node` 는 얇은 래퍼.

```
누적 소비 = Σ |I| · dt        (state.battery_current 사다리꼴 적분, 노드 자체 계산)
```

**왜 전하인가 — 세 가지가 전부 실측으로 뒷받침된다:**

1. **정확도 검증됨.** 본 세션이 `/mavros/battery` 와 같은 전류 신호를 자체 적분한
   결과가 PX4 쿨롱 카운터와 **120초에 0.6mAh 차이**로 일치했다
   (자체 784.8 mAh vs PX4 891.5−107.3 = 784.2 mAh). 우리가 직접 계산해도 된다는 증거.
2. **단조증가라 오실레이션이 구조적으로 없다.** 히스테리시스·디바운스 불필요 —
   설계안 2 의 주 약점이 사라진다.
3. **부하 무관.** 상승 중이든 순항 중이든 같은 척도다. 설계안 1·2 는 둘 다
   "상승 구간에서 무조건 비관적"이라는 결함이 있다(B-5).

**층위 배치:**

- **① 이륙 전:** 무부하 팩 전압 하한 (`battery_arm_min_v`, ≤0 이면 비활성).
  차단하되 끌 수 있게. C-4① 의 한계를 docstring 에 명시할 것.
- **② 천이 전:** **독립 게이트를 두지 않는다.** C-4② 의 이유 — 막고 대기하면
  더 나빠진다. 대신 ③의 예산 임계를 천이 직전에 평가해 남은 예산이 부족하면
  천이 없이 곧장 강등한다.
- **③ 비행 중:** `_battery_guard_breached(state) -> bool` 을 `_control_callback`
  거리상한 바로 다음에 배치. 누적 소비가 `battery_budget_mah` 초과 시 강등, 래치.
  강등 목적지는 **`vtol_state` 분기**: MC → `_State.LANDING`(검증된 기존 경로),
  FW → `AUTO.RTL`(SITL 실증 선행 필요).

---

## E. 권고안 임계값의 근거

**전부 실측 전압/전류 곡선에서 나온 값이다. "보통 이렇게 한다"로 정한 값 없음.**

### E-1. `BAT1_R_INTERNAL = 0.020` Ω/셀

- **측정:** 4초 슬라이딩 창 최소자승 `V = OCV − I·R`(느린 OCV 변화와 빠른 R 응답을
  분리). 13개 비행: 11.9 ~ 23.4 mΩ/셀, **중앙값 18.4**, flight02 단독 17.8 (84창).
- **독립 검증 (이게 결정적이다):** 이 값으로 PX4 SoC 를 재계산한 반사실:

  | `BAT1_R_INTERNAL` | LOW 도달 | CRIT | EMERG | 천이시점(t=56.3) SoC | **착륙 후 무부하 SoC** |
  |---|---|---|---|---|---|
  | 0.0005 (실제 동작) | 6.7s | 7.3s | 7.5s | 0.00 | 0.29 |
  | 0.017 | 22.1s | 26.1s | 26.5s | 0.09 | 0.31 |
  | **0.020** | **86.3s** | 없음 | 없음 | **0.20** | **0.31** |
  | 0.025 | 없음 | 없음 | 없음 | 0.39 | 0.32 |

  `R=0.020` 일 때 계산된 착륙 후 SoC 0.31 이 **물리적으로 측정된 착륙 후 무부하
  전압 3.80 V/셀 → (3.80−3.6)/0.6 = 0.33 과 일치**한다. 이 정합이 값의 근거다.
- ⚠️ **민감도 경고:** 0.017 → 0.020 사이에서 EMERGENCY 가 "26.5초"에서 "없음"으로
  바뀐다. 이 팩은 `BAT1_V_EMPTY = 3.6` 경계에 걸터앉아 있다. 그래서 `BAT1_CAPACITY`
  설정이 **선택이 아니라 필수**다 — 쿨롱 카운팅이 들어와야 이 민감도가 사라진다.

### E-2. `battery_arm_min_v` (이륙 전 무부하 하한)

- **측정된 시동 전 무부하 전압:** 3.729 ~ 4.192 V/셀 (B-6, 4S 7건).
- **권고 3.95 V/셀 (4S = 15.8V).** 근거: 이 값은 실측 분포를 **최저 부하전압
  하위 2건을 자르는 지점**이다 — flight16(Vrest 3.729 → 최저 2.461)과
  flight14(3.823 → 2.880)가 걸러지고, 나머지 5건은 통과한다. flight02 는
  4.150 이므로 **통과**한다(정상 — 팩은 만충이었다).
- **한계 명시:** C-4① 대로 상관계수 r=0.364 로 약하다. 이 값은 "새그 예측"이
  아니라 "쓰던 팩으로 뜨지 않는다"는 위생선이다.

### E-3. `battery_budget_mah` (비행 중 예산)

- **실측 소모율:** flight02 = 802mAh / 129.5초 armed, 평균 22.8A → **6.20 mAh/s.**
  구간별로는 상승 32.4A(8.9 mAh/s), 로이터 22.0A(6.1 mAh/s).
- **실측 팩 여력:** 802mAh 소비 시 무부하 셀전압 4.150 → 3.80. `V_EMPTY/V_CHARGED`
  선언 구간 기준 91% → 33%, 즉 **선언 구간의 58%p 를 802mAh 로 썼다.**
- 🔴 **여기서 정직하게 멈춰야 한다.** LiPo OCV 곡선이 비선형(위쪽이 가파름)이라
  단일 비행의 상단 구간만으로 정격 용량을 외삽할 수 없다. **`battery_budget_mah`
  는 계산으로 뽑지 말고 팩 라벨 정격 × 0.6 으로 운용자가 선언해야 한다.**
  정격 2200mAh 가정 시 예산 1320mAh → 실측 소모율로 **약 213초(3분33초)**의
  MC 위주 비행. flight02 는 129.5초에 802mAh 를 썼으므로 이 예산의 61% 를 소진한
  셈이고, **게이트는 발동하지 않았을 것이다.**
- 팩 정격을 확인하면 이 값을 확정할 수 있다 — **현재 미확인 항목**(D-4).

---

## F. 반사실 — 이 게이트가 있었다면 flight02 는

### F-1. 설계안 3(권고안)이 적용됐다면

| 층위 | 판정 | 결과 |
|---|---|---|
| ① 이륙 전 무부하 ≥ 3.95 V/셀 | 실측 **4.147** | ✅ **통과** — 이륙한다 |
| ③ 전하 예산 1320mAh | 최대 소비 802mAh (61%) | ✅ **미발동** |
| ② 천이 전 잔여 예산 | 천이 지령 t=56.29 시점 395mAh (30%) | ✅ **천이 허용** |

> 🔴 **결론: 권고안은 flight02 를 하나도 바꾸지 않는다. 그리고 그것이 정답이다.**
> flight02 의 배터리는 문제가 아니었다. 진짜 결함은 notes.md ①(`transition_alt`
> 유실로 50m 상승)·③·⑤(천이 중 무추력)이며 배터리는 그 결함들의 **증상**이었다 —
> 50m 상승이 상승 구간 평균 32.4A 를 만들었고, 그 새그를 잘못 보정된 PX4 가
> Emergency 로 읽은 것이다.

### F-2. 설계안 1(PX4 등급 그대로)이 적용됐다면 — 반드시 기록해야 할 반사실

| 게이트 | 발동 시각 | 고도 | 그 순간 실제 팩 상태 |
|---|---|---|---|
| `remaining < 0.15` (LOW) | **t=5.66s** | **2.0m** | 무부하 환산 **91% 만충** |
| `remaining < 0.05` (EMERGENCY) | **t=6.46s** | **2.8m** | 〃 |

> 이륙 5.7초·고도 2m 에서 만충 팩으로 임무가 중단된다. 그리고 B-6 에 따라
> **모든 비행에서 그렇게 된다.** 지시서가 암시한 형태의 게이트를 그대로 넣었다면
> 2차예선 당일 기체가 뜨지 못했을 것이다.

### F-3. PX4 파라미터만 고쳤다면 (3-A, 노드 무변경)

- `COM_LOW_BAT_ACT=3` **만** 바꿨다면 → t=6.46s(고도 2.8m)에 **자동 Land 발동.**
  만충 팩으로 즉시 착륙. 최악의 조합이다.
- `COM_LOW_BAT_ACT=3` + `BAT1_R_INTERNAL=0.020` → E-1 표: LOW 가 t=86.3s
  (하강 중, 고도 약 33m), CRITICAL 미도달 → **동작 없음.** 임무 정상 진행.
- 여기에 `BAT1_CAPACITY` 까지 설정하면 SoC 가 쿨롱 기반이 되어 R 민감도가 사라진다.

---

## G. 미확인 항목 / 후속 조치

1. 🔴 **팩 정격 용량 미확인.** `battery_budget_mah` 와 `BAT1_CAPACITY` 를 확정할 수
   없다. 라벨 확인 필요. 본 문서의 추정 2000~2500mAh 는 *[추정]* 이다.
2. 🔴 **`/mavros/battery` 발행 주기 미측정.** flight02 rosbag
   (`logs/2026-07-28_flight02/rosbag/metadata.yaml`)에 이 토픽이 **없다.** 디바운스
   창을 정하려면 필요하다. → **rosbag 기록 토픽 목록에 `/mavros/battery` 추가 권고.**
   (`launch.log` 의 `telemetry_node` 2초 로그로는 상한만 알 수 있다.)
3. *[추론, 미확정]* PX4 RLS 내부저항 추정기가 0.0005Ω 에 고착한 원인이
   `battery.cpp:277` 의 공분산 단조감소 게이트인지. `src/lib/battery/int_res_est_replay.py`
   로 리플레이 검증 가능.
4. `AUTO.RTL` 은 우리 노드가 한 번도 발행한 적 없는 모드다. C-4③(c) 채택 시
   SITL 실증이 선행돼야 한다.
5. 본 조사는 **코드 미수정**이다. 구현은 오케스트레이터 승인 후 별도 진행.

---

## H. 재현 방법

```bash
# A. 파라미터 (실기체 접속 불필요)
python3 -c "
import json; d=json.load(open('logs/2026-07-28_px4_flash/px4_params_2026-07-28_final-crsf.json'))['params']
for k in ['COM_LOW_BAT_ACT','BAT_LOW_THR','BAT_CRIT_THR','BAT_EMERGEN_THR','BAT1_N_CELLS',
          'BAT1_V_EMPTY','BAT1_V_CHARGED','BAT1_CAPACITY','BAT1_R_INTERNAL','COM_ARM_BAT_MIN']:
    print(k, d[k]['value'])"

# B. 배터리 추이 + warning 전이
python3 -c "
from pyulog import ULog; import numpy as np
u=ULog('logs/2026-07-28_flight02/log_215_2026-07-28-10-29-12.ulg',['battery_status'])
b=[x for x in u.data_list if x.name=='battery_status'][0].data
t=(b['timestamp']-u.start_timestamp)/1e6; V=b['voltage_v']; I=b['current_a']; W=b['warning']
p=None
for i in range(len(t)):
    if W[i]!=p: print(f't={t[i]:7.2f} w={W[i]} V={V[i]:.2f} ({V[i]/4:.3f}/c) I={I[i]:.1f} rem={b[\"remaining\"][i]:.3f}'); p=W[i]
j=int(np.argmin(V)); print('min', t[j], V[j], V[j]/4, I[j])"

# B-3. 11.63V 의 출처
grep -n "batt=11.63V" logs/2026-07-28_flight02/launch.log     # → :41

# 전 비행 EMERGENCY 도달 일괄 확인 (B-6 표)
#   logs/2026-07-2*_flight*/*.ulg 를 순회하며 battery_status.warning==3 최초 시각 출력
```

PX4 소스 인용 위치(WSL `Ubuntu-22.04:/root/PX4-vehicle`, `c890d9db0a`, **읽기 전용**):

- `src/modules/commander/commander_params.yaml:156` — `COM_LOW_BAT_ACT` enum
- `src/modules/commander/failsafe/failsafe.cpp:168-224, 649-654` — 배터리 페일세이프 분기
- `src/modules/commander/HealthAndArmingChecks/checks/batteryCheck.cpp:200-206` — 시동 게이트
- `src/lib/battery/battery.cpp:144-154, 244-257, 277-283, 310-341` — SoC/내부저항/경고 판정
- `src/lib/battery/battery.h:219-222` — `R_DEFAULT = 0.005`
- `src/modules/mavlink/streams/SYS_STATUS.hpp:172-174` — MAVROS 로 나가는 값의 정체
