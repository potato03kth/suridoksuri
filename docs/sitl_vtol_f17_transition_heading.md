---
doc_type: investigation
project: suridoksuri-1
track: 🛩 sitl-vtol
scope: F-17 — 전방천이 중 PX4 가상 WP가 항상 「정북 3000m」인 결함의 조사
created: 2026-07-27
status: 조사 완료 (코드 수정 없음)
px4_build: /root/PX4-vehicle @ c890d9db0a (실기체 탑재본, PX4_FMU_V6C, v1.18.0-alpha)
repo_head_at_investigation: 27cb888
---

# F-17 조사 — 전방천이 헤딩이 항상 정북이 되는 결함

> **표지 한 줄**
> **우리 코드로는 수정 불가능하다** — yaw는 `trajectory_setpoint.yaw`까지 정확히 도달하지만
> PX4의 FW 오프보드 변환 블록이 그 필드를 **한 번도 읽지 않는다**.
> **권고: 마감 전에는 「천이 레그를 정북 ±10°로 고정」하는 운영 회피(코드 변경 0), 마감 후 PX4 한 줄 패치(`yaw = NAN`)로 근본 해결.**

**조사 범위 제한:** 소스 정독 + 기존 캠페인 로그 분석만. SITL 실행·비행코드 수정·PX4 수정 없음.

---

## 0. 요약 — 새로 확정된 것

| # | 확정 사실 | 근거 |
|---|---|---|
| 1 | 우리가 발행한 yaw는 `trajectory_setpoint.yaw`까지 **정확히 도달한다** (C2 1.5707964 = 정확히 90°) | ulog 실측, `mavlink_receiver.cpp:1258` |
| 2 | `FixedWingModeManager.cpp` 2866줄 전체에서 **`trajectory_setpoint.yaw`를 읽는 코드가 0곳** | grep 전수 (§Q1-3) |
| 3 | 오프보드에서 `_pos_sp_triplet.current.yaw`는 **구조적으로 항상 0.0f** — 쓰는 곳이 PX4 전체에 Navigator뿐인데 오프보드는 Navigator를 안 쓴다 | `:2128`, `:2193`, navigator grep |
| 4 | `HDG_HOLD_DIST_NEXT = 3000.0f`는 **`static constexpr`** — 파라미터 우회 불가 | `FixedWingModeManager.hpp:106` |
| 5 | 천이 중 우리 위치 setpoint의 lat/lon은 **통째로 덮어써진다** (`:558-559`) — 위치 명령조차 무시된다 | 소스 |
| 6 | **실측 발현:** C2(정동 90° 레그)에서 천이 중 PX4 지령 course = **-0.00°~-0.13°(정북)**, 기체 헤딩 90.4°→70.7°→최소 43.6°, 북향 이탈 **21.78m**, 고도 **43.23m(-6.77m)** | ulog `fixed_wing_lateral_setpoint` (§Q3) |
| 7 | **C2는 직선 미션인데 코너가 있는 전 런보다 cte가 크다** (21.76m vs A3 15.38 / B4 14.34 / B5 14.89). 같은 직선인 A1(정북) 1.31m, B8(정남) 3.03m | `metrics.json` |
| 8 | 결함은 **2022-09-13(`f9b6edab07`)부터 존재** — `course` 결함(2025-05 가드→2025-07 revert)과 달리 **한 번도 가드된 적이 없다** | `git log -L` |
| 9 | 이 결함은 **OFFBOARD 전용**이다. AUTO에서는 Navigator가 `current.yaw`를 NAN 또는 실제 천이헤딩으로 채워 `:549`의 `_yaw` 폴백이 정상 동작한다 | `navigator_main.cpp:682, :1067`, `takeoff.cpp:238` |

---

## Q1. 우리 쪽 코드만으로 고칠 수 있는가 → **불가능 (확정)**

### Q1-1. 발행 → PX4 경로 추적 (한 단계씩)

**① 우리 노드 (`fc_ros/fc_ros/nodes/offboard_node.py` @ 27cb888)**

- `:814 _publish_pos_setpoint(pos_ned, yaw_ned)` — `PoseStamped`를 `/mavros/setpoint_position/local`로 발행,
  `:832-836`에서 `yaw_ned_to_quat_enu(yaw_ned)`로 orientation을 **명시 설정**한다 (2026-07-21 수정).
- `TRANSITION_FW` Phase 3(`:798-799`)와 ACTIVE TRANSITION(`:719-721`) 모두
  `_publish_pos_setpoint(pts[-1], chi_wp)` — `chi_wp`(`:714`)는 WP0→WP1 방위다.
  즉 **천이 구간 내내 올바른 헤딩을 명시적으로 보내고 있다.**

**② MAVROS → MAVLink → PX4 `mavlink_receiver.cpp`**

- `:1177 handle_message_set_position_target_local_ned()`
- `:1258 setpoint.yaw = (type_mask & POSITION_TARGET_TYPEMASK_YAW_IGNORE) ? (float)NAN : target_local_ned.yaw;`
- `:1282 _trajectory_setpoint_pub.publish(setpoint);`

**③ 실측 — 이 구간은 정상 동작한다 (ulog 확인)**

| 런 | 미션 첫 레그 | 천이 중 `trajectory_setpoint.yaw` (rad) | 해석 |
|---|---|---|---|
| `C2_pxvehicle` | 정동 (90°) | **1.5707964** | 정확히 π/2 |
| `B8_pxvehicle` | 정남 (180°) | **3.1415927** | 정확히 π |
| `A1_pxvehicle` | 정북 (0°) | **7.4988e-33** | ≈0 |

원본: `/root/drone_ws/src/suridoksuri/logs/2026-07-27_sitl_vtol_campaign/{C2,B8,A1}_pxvehicle/*.ulg`
(호스트 사본은 `.ulg` 미포함 — `logs/…/{런}/`에 `node.log`/`metrics.json`만 있다)

⇒ **MAVROS·MAVLink·`mavlink_receiver` 구간은 결백하다. yaw는 온전히 전달된다.**

### Q1-2. 변환 블록이 `trajectory_setpoint.yaw`를 읽는가 → **읽지 않는다 (전수 확인)**

`src/modules/fw_mode_manager/FixedWingModeManager.cpp` (2866줄) 전체에서
`trajectory_setpoint` 를 참조하는 줄은 **다음이 전부**다:

```
2124  trajectory_setpoint_s trajectory_setpoint;
2126  if (_trajectory_setpoint_sub.update(&trajectory_setpoint)) {
2129  _pos_sp_triplet.timestamp = trajectory_setpoint.timestamp;
2130  _pos_sp_triplet.current.timestamp = trajectory_setpoint.timestamp;
2141  ...ISFINITE(trajectory_setpoint.position[0]) && ...position[1])
2145  _global_local_proj_ref.reproject(trajectory_setpoint.position[0], ...position[1], lat, lon);
2154  ...ISFINITE(trajectory_setpoint.velocity[0]) && ...velocity[1])
2157  _pos_sp_triplet.current.vx = trajectory_setpoint.velocity[0];
2158  _pos_sp_triplet.current.vy = trajectory_setpoint.velocity[1];
2160  Vector3f(trajectory_setpoint.acceleration).isAllFinite()
2161  velocity_sp_2d(trajectory_setpoint.velocity[0], trajectory_setpoint.velocity[1]);
2163  acceleration_sp_2d(trajectory_setpoint.acceleration[0], trajectory_setpoint.acceleration[1]);
2175  ...ISFINITE(trajectory_setpoint.position[2])
2177  _pos_sp_triplet.current.alt = _reference_altitude - trajectory_setpoint.position[2];
2181  ...ISFINITE(trajectory_setpoint.velocity[2])
2182  _pos_sp_triplet.current.vz = trajectory_setpoint.velocity[2];
2185  !ISFINITE(trajectory_setpoint.position[2]) && !ISFINITE(trajectory_setpoint.velocity[2])
```

**`trajectory_setpoint.yaw` / `.yawspeed` 는 단 한 번도 나오지 않는다.**
읽는 것은 `position[0..2]` · `velocity[0..2]` · `acceleration[0..2]` · `timestamp` 뿐이다.

### Q1-3. `trajectory_setpoint` 메시지에 yaw 필드가 있는가 → **있다**

`msg/versioned/TrajectorySetpoint.msg` (경로 주의 — `msg/` 직하가 아니다):

```
11  float32[3] position
12  float32[3] velocity
13  float32[3] acceleration
14  float32[3] jerk
16  float32 yaw       # euler angle of desired attitude in radians -PI..+PI
17  float32 yawspeed
```

⇒ 필드는 존재하고, 값도 채워져 도착하고, **소비자만 없다.**

### Q1-4. `_pos_sp_triplet.current.yaw` 를 쓰는 곳이 PX4 어디에도 없는가

- `FixedWingModeManager.cpp` 안의 `.yaw` 참조 전수: `:549`(읽기) · `:2602`·`:2603`(`vehicle_local_position_setpoint`의 별개 필드) · `:2628`(orbit). **`_pos_sp_triplet.current.yaw`에 대입하는 줄은 0개.**
- `_pos_sp_triplet` 자체는 `:2128`에서 `= {}` 로 제로초기화되고, 오프보드 분기에서는
  `:2131-2139`가 `cruising_speed/cruising_throttle/vx/vy/vz/lat/lon/alt/gliding_enabled` 에만 NaN을 넣는다. **`yaw`와 `course`가 빠져 있다** → 둘 다 `0.0f`(finite).
- 유일한 대체 경로인 `:2193 _pos_sp_triplet_sub.update(&_pos_sp_triplet)` 은 **`else` 분기**
  (= `flag_control_offboard_enabled == false`)에만 있다. 오프보드에서는 실행되지 않는다.
- PX4 전체에서 `pos_sp_triplet->current.yaw` 에 값을 쓰는 곳은 **Navigator 뿐**이다
  (`navigator_main.cpp:336/339/497/682/1067`, `takeoff.cpp:238`, `precland.cpp:703/707…`).
  오프보드에서는 Navigator가 이 triplet의 생산자가 아니다.

### Q1-5. 그래서 어떻게 되는가

```
549  const float transition_heading = PX4_ISFINITE(current_sp.yaw) ? current_sp.yaw : _yaw;
550  waypoint_from_heading_and_distance(_current_latitude, _current_longitude,
                                        transition_heading, HDG_HOLD_DIST_NEXT, &lat_transition, &lon_transition);
```

`current_sp.yaw`는 항상 `0.0f`(finite) → **항상 참 분기** → `transition_heading = 0.0f = 정북`.
`FixedWingModeManager.hpp:106  static constexpr float HDG_HOLD_DIST_NEXT = 3000.0f;`

그리고 `:558-559`:

```
558  current_sp.lat = _transition_waypoint(0);
559  current_sp.lon = _transition_waypoint(1);
```

⇒ **전방천이 중 우리가 보낸 위치 setpoint의 lat/lon조차 「정북 3000m」 가상 WP로 통째로 대체된다.**
우리가 보낼 수 있는 값 중 이 결과를 바꾸는 것은 **하나도 없다.**

`:543` 때문에 가상 WP는 천이 진입 첫 틱에 **한 번만 latch**되고 천이 종료(`:561-565`)에 리셋된다.

### Q1 결론

> **우리 코드만으로는 불가능하다.**
> 어떤 토픽에 어떤 값을 어떤 형식으로 보내도, FW 오프보드 경로에서 yaw는 `_pos_sp_triplet.current.yaw`에
> 도달할 수 없고, 천이 중 위치 지령도 무시된다. 남은 선택지는 **운영 회피** 또는 **PX4 수정**뿐이다.

---

## Q2. 다른 발행 채널로 우회 가능한가 → **모두 불가**

> ⚠ 아래는 「yaw 전달 경로가 존재하는가」만 본 것이다. **A안(위치+속도) 재검토가 아니며,
> A안은 §4-2에서 기각 확정 상태 그대로다.**

| 채널 | yaw가 `trajectory_setpoint.yaw`에 실리는가 | `_pos_sp_triplet.current.yaw`에 도달하는가 | 판정 |
|---|---|---|---|
| `/mavros/setpoint_position/local` (`PoseStamped`) — **현행** | **실린다** (ulog 실측, §Q1-1) | ✗ (`:2123-2190`이 안 읽음) | 불가 |
| `/mavros/setpoint_raw/local` (`PositionTarget`) | **실린다** — 동일 MAVLink 메시지(`SET_POSITION_TARGET_LOCAL_NED`)이므로 `mavlink_receiver.cpp:1258`이 그대로 처리. `YAW_IGNORE`만 안 걸면 된다 | ✗ (동일 블록) | 불가 |
| `SET_ATTITUDE_TARGET` | 해당 없음 — `mavlink_receiver.cpp:1793`은 `vehicle_attitude_setpoint`/`vehicle_rates_setpoint`만 발행(`:1847/:1850/:1853/:1880`), `_pos_sp_triplet`을 건드리지 않는다 | ✗ | 불가. 게다가 자세 오프보드는 `flag_control_position_enabled`를 끄므로 `:387` 조건이 깨져 FW 유도 자체가 `FW_POSCTRL_MODE_OTHER`로 빠진다 |
| `position_setpoint_triplet` uORB 직접 주입 | — | ✗ — 오프보드에서는 `:2193`(else 분기)이 실행되지 않아 구독조차 안 한다 | 불가 |
| PX4 파라미터 | — | ✗ — `HDG_HOLD_DIST_NEXT`는 `hpp:106`의 `static constexpr`, 파라미터가 아니다. 천이 WP 생성을 끄는 파라미터도 없다 | 불가 |

**부수 확인(참고용, A안과 무관):** `control_auto_path()`(`:1085-1123`)는
`move_position_setpoint_for_vtol_transition()`을 호출하지 않는다 — 그 함수는 `control_auto()`(`:574`)에서만 불린다.
**사실로만 기록한다. A안은 §4-2 기각 확정이며 이 사실은 재검토 사유가 아니다**(A안은 R-a·R-c 봉쇄 불가로 기각됐고 그 근거는 이 항목과 무관하다).

---

## Q3. 실제 발현 범위 — 캠페인 로그 실측 증거

### Q3-1. 캠페인 24런 중 **비-정북 천이는 단 2건뿐이었다**

천이는 항상 미션 **첫 레그** 방향으로 정렬한 뒤 일어난다(`offboard_node.py:714 chi_wp`). 전 런의 첫 레그:

| 첫 레그 방향 | 런 |
|---|---|
| **정북 (0°)** | A1, A1_pxvehicle, A2, A3, A3_pxvehicle, A4, B1, B2, B3, B4, B5, B6, B7, C1a, C1b, C3, C4, C5b, C5c, C6a, C6b, C7, C8, C10 — **22런** |
| **정동 (90°)** | **C2_pxvehicle** |
| **정남 (180°)** | **B8_pxvehicle** |

> A3(L자)·B4(U턴)·B5(폐곡선)의 코너는 전부 **FW 순항 중**에 있고 천이 구간이 아니다.
> 즉 「비-정북 레그가 있는 런」이라도 **천이 헤딩은 정북**이었다.
> ⇒ 캠페인이 F-17을 못 잡은 것이 아니라, **잡을 수 있었던 런이 C2 하나뿐이었다.**
> (§7-1의 경고 "정북 직선만으로는…"가 그대로 재현됐다.)

### Q3-2. C2 — 결정적 증거 (PX4 자신의 지령값)

원본: `/root/drone_ws/src/suridoksuri/logs/2026-07-27_sitl_vtol_campaign/C2_pxvehicle/20_32_04.ulg`
호스트: `logs/2026-07-27_sitl_vtol_campaign/C2_pxvehicle/{node.log,metrics.json,meta.json}`
미션: `waypoints:=[0.0,0.0,50.0, 0.0,300.0,50.0]` = **정동 300m 직선** (코너 0개)
천이 구간(ulog): **53.684 s ~ 56.204 s (2.52 s)**, `vehicle_status.in_transition_to_fw`

`node.log:28` — 우리 노드는 완벽하게 정렬했다:
```
헤딩 정렬 완료 target=90.0° current=90.4° err=-0.4° (20틱 안정) → 전진 + 천이 명령
```

그런데 PX4가 그 순간 만든 지령(`fixed_wing_lateral_setpoint.course`, ulog):

| ulog t (s) | in_transition | **course 지령 (deg)** | 기체 yaw (deg) | N (m) | E (m) |
|---|---|---|---|---|---|
| 54.276 | 1 | **-0.00** | 90.2 | 0.0 | -0.0 |
| 54.876 | 1 | **-0.01** | 88.7 | -0.0 | 0.4 |
| 55.076 | 1 | **-0.02** | 86.9 | 0.0 | 0.8 |
| 55.476 | 1 | **-0.04** | 82.2 | 0.3 | 2.3 |
| 55.876 | 1 | **-0.09** | 76.2 | 1.0 | 4.9 |
| 56.076 | 1 | **-0.13** | 71.4 | 1.7 | 6.8 |
| **56.276** | **0** | **+90.53** | 66.5 | 2.7 | 9.2 |

**미션은 정동(90°)인데 천이 내내 지령 course는 0°(정북)다.**

**수치 교차검증 — 가상 WP가 정확히 「정북 3000m」임의 증명:**
기체가 동쪽으로 E만큼 밀리면, latch 지점에서 정북 3000m인 점의 방위는 `atan2(-E, 3000)`이 된다.

| t | 실측 E | 예측 `atan2(-E,3000)` | **실측 course** |
|---|---|---|---|
| 55.476 | 2.3 m | -0.044° | **-0.04°** |
| 55.876 | 4.9 m | -0.094° | **-0.09°** |
| 56.076 | 6.8 m | -0.130° | **-0.13°** |

소수 둘째 자리까지 일치 → `HDG_HOLD_DIST_NEXT = 3000.0f`, `transition_heading = 0.0f` 확정.

**계단:** `in_transition_to_fw`가 0이 되는 바로 그 틱에 course 지령이 **-0.13° → +90.53°** 로
**90.7° 순간 계단**을 밟는다(56.076 → 56.276).

### Q3-3. 정량 피해 — 천이는 2.5초지만 피해는 그 뒤에 온다

C2 (ulog 기준, 천이 종료 = 56.204):

| 지표 | 값 | 시각 |
|---|---|---|
| 천이 시작 yaw | 90.4° | 53.68 |
| 천이 **종료** yaw | **70.7°** (−19.7°) | 56.20 |
| yaw **최저** | **43.6°** (목표 대비 **−46.8°**) | 57.30 (천이 후 1.1s) |
| 헤딩 손실률 (유도 활성 구간) | ≈ **11.8 °/s** | 54.28~56.08 |
| **북향 최대 이탈** | **N = 21.78 m** | 59.42 (천이 후 3.2s) |
| **고도 최저** | **43.23 m** (순항 50.0m 대비 **−6.77 m**) | 59.95 |
| 회복 시 반대편 헤딩 오버슈트 | **129.2°** (목표 90° 대비 +39.2°) | 63.14 |
| cte 5m 이하로 회복까지 | ≈ **10.6 s / ≈250 m 비행** | ~66.8 |

`metrics.json` 집계:
- `cte.geometric_cte.max_m = 21.76` (2308샘플), `cte.node_log_cte.max_m = 19.6`
- `altitude.cruise_alt_dev.max_abs_dev_m = 6.76`, `min_agl_m = 43.26`
- `altitude.transition_alt_loss.value = 0.0` (천이 구간 53.672~56.2) — **천이 중 고도 손실은 0**

⇒ **답: 천이 2.5초 자체의 직접 피해는 작다(헤딩 −19.7°, 횡 3m, 고도 0m).
피해의 대부분은 천이가 넘겨준 초기조건 — 46.8° 헤딩오차 + 북향 각속도 — 때문에
천이 *직후* 발생하는 대선회다: 횡 21.8m, 고도 −6.8m, 헤딩 39.2° 반대 오버슈트, 회복 250m.**

### Q3-4. 대조군 — 같은 직선 미션인데 정북/정남은 멀쩡하다

| 런 | 첫 레그 | 코너 | **geometric cte max** | node cte max | **고도편차 max** | 천이 시작→종료 yaw |
|---|---|---|---|---|---|---|
| **A1_pxvehicle** | 정북 0° | 0 | **1.31 m** | 1.1 m | 2.42 m | 2.4° → −1.2° |
| **B8_pxvehicle** | 정남 180° | 0 | **3.03 m** | 2.6 m | 2.16 m | 177.6° → 173.2° |
| **C2_pxvehicle** | **정동 90°** | **0** | **21.76 m** | **19.6 m** | **6.76 m** | **90.4° → 70.7°** |
| A3_pxvehicle | 정북 | 90°×1 | 15.38 m | 7.2 m | 5.59 m | — |
| B3_pxvehicle | 정북 | 90°×1 | 14.88 m | 7.0 m | 4.73 m | — |
| B4_pxvehicle | 정북 | U턴 | 14.34 m | 13.9 m | 7.45 m | — |
| B5_pxvehicle | 정북 | 폐곡선 | 14.89 m | 8.6 m | 6.82 m | — |

**C2는 코너가 하나도 없는 직선인데 코너가 있는 전 런보다 cte가 크다.**
같은 「코너 0개 직선」인 A1·B8 대비 **7~17배**다.

**부수 성과 — F-5의 미해결 이상치가 설명된다.** §1-4 기준선표의
「선회 중 고도 침하 … **직선인데 C2 6.76m**」는 그동안 설명되지 않은 항목이었다.
고도 최저(43.23 m, t=59.95)가 북향 최대이탈(21.78 m, t=59.42)과 **0.5초 차이로 겹친다.**
⇒ **C2의 고도 침하는 독립적인 종방향 결함이 아니라 F-17이 강제한 회복 선회의 결과다.**
(F-5의 나머지 항목 — 실제 코너가 있는 A3/B4/B5 — 은 여전히 별개 사안이다.)

### Q3-5. ⚠ 정직하게 — 발현이 **간헐적**이었고 그 조건을 확정하지 못했다

`fixed_wing_lateral_setpoint`(= `control_auto_position()`의 산출물) **천이 구간 내 샘플 수**:

| 런 | 천이 중 샘플 | 천이 중 발행률 | 천이 후 발행률 |
|---|---|---|---|
| C2 | **7** | ≈2.8 Hz | ≈40 Hz |
| A1 | **0** | — | 40 Hz (천이 종료 +0.65s부터) |
| B8 | **0** | — | 40 Hz (천이 종료 +0.59s부터) |

세 런 모두 `fixed_wing_lateral_guidance_status`/`fixed_wing_lateral_status`는 천이 구간에 26~27샘플 있다
→ FW 모듈의 `Run()` 루프는 세 런 모두 정상 실행됐다(`:377`이 천이 중 실행을 허용한다).
차이는 `_control_mode_current`가 C2에서만 `FW_POSCTRL_MODE_AUTO`였고 A1/B8은 `FW_POSCTRL_MODE_OTHER`였다는 것이다
(= `:387`의 `_position_setpoint_current_valid`가 갈렸다).

**어떤 조건이 이를 갈랐는지는 확정하지 못했다.**
관측된 것: C2는 천이 구간 동안 `trajectory_setpoint`가 `p=(nan,nan,nan) v=(nan,nan,-0.00)` 형태였고,
A1/B8은 `p=(300,0,-50)/(−300,0,−50) v=(nan,nan,nan)` 였다. 소스만 보면 A1/B8 쪽이
`valid_setpoint = true`가 되어 **오히려 더 잘 발현해야 한다** — 관측과 반대다.
ulog의 `trajectory_setpoint` 로깅이 서브샘플링(630샘플/131s ≈ 4.8 Hz, 실제 발행 10 Hz+)이라
두 채널이 교대 도착하는 구간을 온전히 못 봤을 가능성이 크지만 **확인하지 못했다.**

**이 미확정이 결론에 주는 영향:** 완화가 아니라 **가중**이다.
- 발현 조건을 모른다 = **실비행에서 걸릴지 안 걸릴지 예측할 수 없다.**
- 그리고 **B8은 「180°가 안전하다」의 증거가 되지 못한다.** B8은 천이 중 FW 횡유도가
  아예 발행되지 않았으므로, 180°에서 유도가 활성이었을 때 어떻게 되는지는 **미측정**이다.
- 따라서 "PX4를 천이 중 AUTO에 안 들어가게 만든다" 류의 **우리 코드 우회는 근거가 없다** —
  가르는 조건을 모르는 채로 그 조건을 노리는 것이기 때문이다. §Q4에서 후보로 올리지 않는다.

### Q3-6. 위험 임계 (θ = 천이 헤딩의 정북 대비 이탈각)

측정점은 θ=0°(A1), 90°(C2), 180°(B8, 단 유도 비활성) 세 개뿐이다. 아래는 **추정**임을 명시한다.

지령 course 오차는 곧 θ이고, C2 실측 헤딩 손실률은 **≈11.8 °/s**, 유도 활성 시간은 천이 2.5s + 관성 ~1s ≈ 3.5s
→ 누적 헤딩오차는 대략 `min(θ, ~40°)` 로 포화한다.

| θ | 예상 거동 | 근거 |
|---|---|---|
| **≤ 10°** | 실질 무해 (cte 수 m 이내) | A1 실측 (θ=2.4° → cte 1.31m) |
| 10~30° | θ에 비례해 pull. cte 대략 3~10m | 외삽 |
| **≥ 30°** | 헤딩오차 포화(~40°), cte 15~22m, 고도 −5~7m, 회복 250m | C2 실측(θ=90°) |
| **180° 부근** | **미측정·불확정.** 「정북 3000m」가 기체 정후방이라 선회 방향이 수치적으로 불안정한 평형점 — 최악의 경우 완전 반전 지령 | B8은 유도 비활성이라 증거 아님 |

**권고 임계: |θ| ≤ 10° 를 안전대로 잡는다.** 15° 이상은 실기체 첫 FW+OFFBOARD 비행에 쓰지 않는다.

---

## Q4. 대응책 후보와 평가

### 후보 ① 운영 회피 — 천이를 정북 부근에서만

두 형태가 있고 비용이 크게 다르다.

**①-a 「미션 설계로 첫 레그를 정북 ±10°에 둔다」 — 코드 변경 0**
- 대회 경로가 **폐회로로 확정**됐다(§4 결정). 폐회로는 **어느 꼭짓점에서 어느 방향으로 출발할지 우리가 고른다.**
  출발 꼭짓점·주회 방향을 골라 첫 레그를 정북 ±10°에 맞추면 된다.
- 비용: 코드 0, 신규 검증 0. 경로 자유도 상실은 **출발점 선택 하나**뿐이며 폐회로 전체 형상은 그대로다.
- 리스크: 현장에서 이륙지점·풍향 때문에 출발 꼭짓점을 못 고르면 무력화된다. 그때는 ①-b로 내려간다.
- 마감 실행가능성: **즉시. 오늘 확정 가능.**

**①-b 「노드가 천이 정렬 목표를 정북으로 강제」 — 소규모 코드 변경**
- 수정량 (F-7 정렬 로직): `offboard_node.py`
  `:714 chi_wp` 옆에 `chi_trans`(기본 0.0) 도입 → `:758 heading_err` 기준, `:721`·`:799`의
  `_publish_pos_setpoint` yaw 인자, Phase 3의 위치 목표(현재 `_pts[-1]`)를 「현재 위치 정북 200m」로 교체.
  **약 5~10줄 + 파라미터 1개**(`transition_heading_mode: north|leg`, 기본 `north`).
  `:759 self._wp0_htol`(yaml `wp0_heading_tol: 0.05`) 등 정렬 판정 로직 자체는 **손대지 않는다.**
- 비용: 천이 후 기체가 정북·순항속도로 나가 있고 경로 첫 레그와 어긋난다 →
  **FW 진입 직후 1회 선회가 강제된다.** 캠페인 실측 90° 코너 성능(오버슈트 0.085~0.915m,
  WP 최근접 19~22m)이 적용되므로 C2의 21.8m/6.8m보다 **명백히 낫다** — 정상 FW 속도·정상 L1에서 도는 선회이기 때문이다.
- 리스크: 천이 종료 시점 기체가 WP0에서 북쪽으로 ~60~70m 벗어나 있어 FOLLOWING 진입 cte가 크다.
  `_FW_LOOKAHEAD` 70m와의 상호작용(F-11/B7 교훈)을 **SITL로 반드시 확인해야 한다.**
- 마감 실행가능성: **세션 1회 + SITL 회귀(A1·A3·C2·B5) 필요.** 마감 전 가능하나 여유는 없다.

### 후보 ② PX4 한 줄 패치 — `_pos_sp_triplet.current.yaw = NAN;`

`:2131-2139` 목록에 한 줄 추가. `course = NAN`과 **완전히 동일 계열**(같은 블록, 같은 누락 유형, 세 번째 사례).
효과: `:549`가 처음으로 `_yaw` 폴백으로 떨어짐 → `transition_heading = 기체 실제 헤딩`
= 우리가 13~15초 들여 맞춘 그 헤딩. **PX4 주석(`:547-548`)이 의도한 동작 그대로다.**

**「업그레이드 금지」와 이 패치는 다른 사안이다 — 구분을 분명히 한다:**

| | PX4 업그레이드 | 이 한 줄 패치 |
|---|---|---|
| 변경 범위 | `c890d9db0a` → 상류 HEAD = 178+ 커밋 (§1-2) | **커밋 그대로 + 1줄** |
| §1-2 금지 사유 | `course` 가드가 `1499238f1c`로 revert돼 **폭주 버그가 들어온다** | 해당 없음 — revert를 안 가져온다 |
| 회귀 표면 | 제어·EKF·VTOL 전 영역 | `_pos_sp_triplet.current.yaw` 하나. 오프보드 아닐 땐 이 블록이 실행조차 안 됨 |
| 되돌리기 | 어려움 | 한 줄 삭제 |

⇒ **§1-2의 「실기체 PX4 업그레이드 금지」는 이 패치를 금지하지 않는다.** 오히려 §4-2 기각 조치 3항이
같은 계열의 상류 한 줄 패치(`course = NAN`)를 **정식 대안으로 이미 명시**하고 있다.

**진짜 비용은 코드가 아니라 빌드·플래시다:**
- 실기체 펌웨어 = `c890d9db0a` / `PX4_FMU_V6C` / **v1.18.0-alpha** (`sitl_vtol_fw_offboard_rootcause.md:291-293`).
  안정 릴리스가 아니라 main 스냅샷이므로 **소스에서 재빌드하는 것이 원래 유일한 조달 경로**다.
- 그러나 `/root/PX4-vehicle/build/` 에는 **`px4_sitl_default` 하나뿐** — `px4_fmu-v6c` 크로스빌드를
  이 환경에서 **한 번도 해본 적이 없다.** `gcc-arm-none-eabi` 툴체인 설치부터 시작해야 한다.
  (보드 설정 `boards/px4/fmu-v6c/` 는 존재한다.)
- 플래시 = USB 물리 접근 + 파라미터 보존 확인 + 전 기능 재검증(MC 비행 포함).

**리스크 총평:** 코드 리스크는 **낮다**(1줄, 영향 범위 명확). **작업 리스크가 높다** —
마감 직전에 실기체 펌웨어를 처음으로 자체 빌드해서 갈아끼우는 것, 그리고 그 빌드가
「실기체 실적 78 ulog」의 검증 이력과 **더 이상 동일 바이너리가 아니게 되는 것**이다.

마감 실행가능성: **마감 전 비권장. 마감 후 정식 절차로 수행.**

### 후보 ③ 천이 구간만 OFFBOARD 이탈 (PX4 AUTO에 맡김) — **비권장**

원리는 맞다. §Q1-4·`navigator_main.cpp:682/:1067`·`takeoff.cpp:238` 대로 AUTO에서는 Navigator가
`current.yaw`를 NAN 또는 실제 천이헤딩으로 채우므로 `:549`의 `_yaw` 폴백이 정상 동작한다 —
**F-17은 OFFBOARD 전용 결함이다.**

그러나 실행이 성립하지 않는다:
- **헤딩 정렬 자체가 OFFBOARD를 요구한다.** 우리 정렬은 MC OFFBOARD + yaw-rate P제어다
  (`offboard_node.py:757-790`). AUTO로 넘기는 순간 그 제어권이 사라진다.
- **정렬 후 AUTO.LOITER로 넘기면 Navigator가 즉시 선회를 건다**(loiter radius 기본 ~80m) →
  13~15초 들여 맞춘 헤딩이 **다른 이유로 똑같이 파괴된다.** 이득이 상쇄된다.
- OFFBOARD 복귀 자체는 실증돼 있으나(§1-5, 상실 0.97s 후 복구), **가장 동적인 순간에
  모드 전이를 2회 추가**하는 것이며, 그 경로는 검증 이력이 없다.
- 마감 실행가능성: **없음.** 새 안전경로를 만들게 되어 §4 결정 2의 「새 폴백 경로 신설 금지」 정신에도 어긋난다.

### 후보 ④ (기존 방어선, 단독으로는 해결책 아님) — R1 거리 상한 300m + 조종사 즉시 인계

C2 실측 최대 이탈이 21.8m이므로 300m 상한은 이 결함 단독으로는 **발동하지 않는다**.
사고 억제 최종 방어선이지 해결이 아니다. **①과 반드시 병행**한다.

### 비교표

| 후보 | 근본해결 | 코드 변경 | 신규 검증 부담 | 실기체 리스크 | **마감(2차예선) 실행가능성** |
|---|---|---|---|---|---|
| **①-a 미션 설계로 첫 레그 정북 ±10°** | ✗ (회피) | **없음** | **없음** | **없음** | ✅ **즉시** |
| ①-b 노드가 천이 정렬을 정북 강제 | ✗ (회피) | 5~10줄 + 파라미터 1 | SITL A1·A3·C2·B5 회귀 | 낮음 (제어법칙 불변) | 🟡 세션 1회 필요 |
| **② PX4 한 줄 패치 `yaw = NAN`** | ✅ **완전** | PX4 1줄 | 크로스툴체인 신규 구축 + 재빌드 + 재플래시 + 전기능 재검증 | **중~높음** (첫 자체빌드 펌웨어) | ❌ **마감 후** |
| ③ 천이 구간 OFFBOARD 이탈 | 부분 | 중 | 신규 안전경로 검증 | 높음 | ❌ 없음 |
| ④ R1 거리상한 300m | ✗ (억제) | 완료됨 | — | — | ✅ (①과 병행 전제) |

### 권고

1. **마감 전 — ①-a 를 확정한다.** 폐회로 출발 꼭짓점·주회 방향을 골라 **천이 레그를 정북 ±10°** 로 고정.
   코드 변경 0, 검증 0, 오늘 결정 가능. **실기체 첫 FW+OFFBOARD 비행(실적 0건)도 반드시 정북 레그로 한다**
   — §3 R7의 실기체 검증 계획이 명시한 "짧은 **비-정북** 직선 레그"는
   **이 결함 때문에 지금은 정확히 최악의 선택이다. 정북 레그로 바꿔야 한다.**
2. **현장에서 ①-a가 불가능해질 대비로 ①-b를 준비한다** (파라미터 기본 `north`).
   여유가 있을 때만. 없으면 ①-a + ④로 마감을 넘긴다.
3. **마감 후 ②를 정식 절차로 수행한다.** 이것만이 근본 해결이며, `course = NAN`과 묶어
   **두 줄을 한 번에** 넣는 것이 합리적이다(같은 블록·같은 누락 유형).
4. **③은 채택하지 않는다.**
5. **§7 「회귀 시나리오 필수 요건」 1번을 보강한다** — "비-정북 레그를 포함할 것"에
   **"비-정북 *천이*(= 첫 레그가 비-정북)를 반드시 포함할 것"** 을 추가한다.
   이번 캠페인 24런 중 그 조건을 만족한 것은 C2·B8 두 건뿐이었다.

---

## 5. 재현 방법 (오케스트레이터용)

```bash
# 소스 — 실기체 빌드
wsl.exe -d Ubuntu-22.04 --cd /root/PX4-vehicle -- git rev-parse HEAD          # c890d9db0a…
wsl.exe -d Ubuntu-22.04 --cd /root/PX4-vehicle -- cat -n src/modules/fw_mode_manager/FixedWingModeManager.cpp > /tmp/f.txt
sed -n '535,566p;2123,2192p' /tmp/f.txt              # :549 / :558-559 / :2128-2139
grep -n "trajectory_setpoint" /tmp/f.txt             # yaw 참조 0곳 확인
grep -n "\.yaw" /tmp/f.txt                           # :549 / :2602 / :2603 / :2628 뿐
wsl.exe -d Ubuntu-22.04 --cd /root/PX4-vehicle -- grep -n "HDG_HOLD_DIST_NEXT" src/modules/fw_mode_manager/FixedWingModeManager.hpp
wsl.exe -d Ubuntu-22.04 --cd /root/PX4-vehicle -- cat -n msg/versioned/TrajectorySetpoint.msg   # :16 yaw
wsl.exe -d Ubuntu-22.04 --cd /root/PX4-vehicle -- grep -n "target_local_ned" src/modules/mavlink/mavlink_receiver.cpp   # :1258
```

ulog 분석 스크립트(임시, `/mnt/c/sitl7_xfer/`에 남겨둠 — WSL 양쪽에서 접근 가능):
`f17_probe3.py`(천이 창 course/yaw/위치 전수), `f17_probe4.py`(플래그·유한성 타임라인),
`f17_probe5.py`(nav_state·발행갭), `f17_probe6.py`(토픽별 천이구간 샘플수), `f17_probe8.py`(피크 요약).

```bash
wsl.exe -d Ubuntu-22.04 -- python3 /mnt/c/sitl7_xfer/f17_probe3.py \
  /root/drone_ws/src/suridoksuri/logs/2026-07-27_sitl_vtol_campaign/C2_pxvehicle/20_32_04.ulg 3 6
```

**⚠ 인터롭 함정:** `wsl.exe … -- awk '…$0…'` 는 `$0`이 치환돼 깨진다. 범위 출력은
`cat -n` 으로 호스트에 받아 호스트측 `sed`로 자른다. `grep`에 `\|` 다중패턴도 깨진다.

---

## 6. 미확인 사항 (지어내지 않았음을 명시)

1. **§Q3-5** — 천이 중 FW 횡유도가 C2에서만 활성이고 A1/B8에서 비활성이었던 **판별 조건을 확정하지 못했다.**
   소스 논리상으로는 A1/B8이 더 잘 발현해야 하는데 관측이 반대다. ulog `trajectory_setpoint` 서브샘플링이
   원인일 가능성이 크나 확인 못 했다.
2. **θ = 180° 부근의 거동은 미측정.** B8은 유도 비활성이라 증거가 아니다.
3. **θ 임계표(§Q3-6)의 10~30° 구간은 외삽**이다. 실측점은 θ=2.4°와 θ=90° 두 개뿐이다.
4. 실기체 펌웨어가 **누가 어떻게 빌드/플래시했는지**의 기록은 저장소에서 찾지 못했다
   (커밋·보드·버전만 `sitl_vtol_fw_offboard_rootcause.md:291-293`에 있다). 후보 ②의 실제 작업량은
   그 조달 경로가 확인돼야 확정된다.
5. AUTO 전 서브모드에서 `current.yaw`가 항상 NAN/유효 헤딩임을 **전수 확인하지는 않았다**
   (reposition `:682`, geofence loiter `:1067`, takeoff `takeoff.cpp:238`, precland는 확인).
