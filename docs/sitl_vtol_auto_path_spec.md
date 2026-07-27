---
doc_type: adversarial_review
project: suridoksuri-1
track: 🛩 sitl-vtol
scope: R3-(a) — A안(`setpoint_raw/local` 위치+속도 → `FW_POSCTRL_MODE_AUTO_PATH`) 적대적 소스 심문
created: 2026-07-27
status: 판정 완료
심사방침: `docs/sitl_vtol_remediation_plan.md` §4-1 (사용자 지시 — 기본 입장 「기각」, 입증책임은 A안 쪽)
---

# 🔴 판정: **A안 — 기각 권고**

> **R3-(b) 실측을 실행할 필요 없음.** §4-1 1항(리스크 6건 전부 봉쇄 입증)에서 **R-a·R-b·R-c·R-e
> 4건이 실재로 확정**됐고, §4-1 2항(이득 실측 입증)은 **근거가 재현 불가능**한 데다 C4·B7 두
> 인과가 **소스와 실측으로 반증**됐다. 두 관문 모두 통과하지 못한다.
>
> **다만 R-f는 반증됐다** — 업그레이드 취약성 가설은 git 이력이 지지하지 않는다. 사실대로 적는다.

**대상 빌드:** `/root/PX4-vehicle` = `c890d9db0a` (실기체 탑재본, course 가드 **있음**).
비교 대조: `/root/PX4-Autopilot` = `9bb0d365c4` (가드 **없음**).
줄번호는 **명시가 없으면 `c890d9db0a`(실기체 빌드)** 기준이다. 두 빌드가 다른 곳은 양쪽을 병기했다.

> **저장소 쪽 줄번호는 `5574251`(이 심문 시작 시점의 워크트리 HEAD) 기준이다.**
> 심문 중 다른 세션이 `fc_ros/`·`fc_bridge/`를 동시에 수정하고 있었으므로,
> 작업트리가 아니라 `git show 5574251:<파일>`로 대조해야 재현된다.
> (`l1_guidance.py`·`state_logic.py`는 미변경이라 작업트리와 동일.)

---

## 1. 리스크 판정표 (R-a ~ R-f)

| # | 리스크 | 판정 | 소스 근거 | 봉쇄 가능성 |
|---|---|---|---|---|
| **R-a** | 모드 토글 채터링 | **🔴 실재 — 봉쇄 불가** | `FixedWingModeManager.cpp:387-398` (모드선택, 히스테리시스 없음) · `:2126` `_trajectory_setpoint_sub.update()` · `:2154-2158` · `:2240` (Run 내 호출 순서) · `offboard_node.py:506,510,720,798,953,987,1127,1132` | **불가.** "NaN을 안 보내기"는 봉쇄되지만 **발행 채널 전환 그 자체가 모드 이탈**이다. FW 구간 진입/이탈 시 `PoseStamped`↔`PositionTarget` 전환이 최소 2회 필요하고, 그 전환 틱은 정의상 `AUTO_PATH`↔`AUTO` 전이다 |
| **R-b** | 접선 반전 + 방향지령 민감성 | **🔴 실재 — 부분 봉쇄만 가능** | `l1_guidance.py:183-208` (`:190` 캐시 초기값을 `:192-206` 전역루프가 무조건 덮어씀) · `FixedWingModeManager.cpp:1098`, `:2770-2785` | **부분.** `_find_segment` 수정은 **레그 점프**만 없앤다. A안 고유의 **잔여 위험**이 남는다 — 접선은 세그먼트 인덱스의 불연속 함수라 **코너마다 지령 코스가 계단**으로 뛴다(B4 135° 코너 = 135° 계단). 점 추적에는 없는 신규 위험이며 별도 코너 블렌딩(신규 미검증 코드)이 필요하다 |
| **R-c** | 종점 무한 이탈 | **🔴 실재 — 봉쇄 불가(성격 악화만)** | `FixedWingModeManager.cpp:1085-1123` (**종점 로직 전무**) vs `:748-772` (`POSITION→LOITER` 승격) + `:2669-2688` (`navigateWaypoint` 점 선회) | **불가.** 현행의 종점 포착 실패는 **유계**(F-10 / C1b flower-pattern, `campaign_report.md:248`)지만 A안은 **무계 직선 이탈**이다. R1 거리상한(300 m)은 이탈을 **막는 게 아니라 페일세이프로 승격**시킬 뿐이다 |
| **R-d** | NaN 마스크 함정 (accel 기본값 0.0) | **🟡 봉쇄 가능 — 단 계획서 서술은 정정 필요** | `mavlink_receiver.cpp:1200-1202` · `FixedWingModeManager.cpp:2160-2172`(SITL `:2139-2151`) · `matrix/Vector.hpp:105-109,127-146` · `:1095-1097` | **가능**(마스크 + 값 명시 NaN 이중방어). **단** 실제 기전은 계획서 서술과 다르다 — §3-4 참조. 그리고 **A안이 "velocity 크기는 무시"라며 단위벡터를 쓰기로 한 것이 이 함정의 피해를 약 400배 증폭**시킨다 |
| **R-e** | yaw 무시 × 천이 상호작용 | **🔴 실재 — 봉쇄 불가(양자택일 딜레마)** | `FixedWingModeManager.cpp:536-566` + `:574` (`move_position_setpoint_for_vtol_transition`, `control_auto`에서만 호출 / `control_auto_path`는 **호출 안 함**) · `:377-381` · `PositionControl.cpp:120` · `offboard_node.py:814-838` | **불가.** ①천이까지 A안으로 덮으면 PX4의 전방천이 전용 보호가 통째로 우회되고 yaw는 무지령(강성 0)이 된다. ②천이를 덮지 않으면 `TRANSITION_FW`·`TRANSITION_MC` 구간이 `AUTO`로 남아 **취약빌드에서 course 버그가 그대로 산다 → A안의 유일한 확정 이득(F-4 제거)이 부분달성에 그친다** |
| **R-f** | 업그레이드 취약성 | **🟢 반증됨 — 기각사유 아님** | git `-L` 이력(§4) — `control_auto_path()`/`navigatePathTangent()` 최근 12개월 **변경 0건**, `control_auto()`+`control_auto_position()` **4건**(우리를 깨뜨린 회귀 3건 포함) | 해당 없음. **가설이 데이터와 반대다.** 다만 두 경로가 **공유**하는 오프보드 변환 블록이 12개월간 2건 변경됐고 그중 `04af7db228`은 AUTO_PATH 전용 곡률식의 **부호 버그 수정**이었다 |

**§4-1 1항 판정: 6건 중 4건(R-a·R-b·R-c·R-e)이 "실재"로 남는다 → 그 시점에서 기각.**

---

## 2. 7문항 답 (전부 줄번호 인용)

### Q1. `velocity` 방향의 의미 — 접선인가 베어링인가

**답: 경로 접선이다. 확정.**

```
FixedWingModeManager.cpp:1094   const matrix::Vector2f velocity_2d(pos_sp_curr.vx, pos_sp_curr.vy);
FixedWingModeManager.cpp:1095-1097
        const float curvature = PX4_ISFINITE(_pos_sp_triplet.current.loiter_radius) ? 1 /
                                _pos_sp_triplet.current.loiter_radius : 0.0f;
FixedWingModeManager.cpp:1098-1099
        const DirectionalGuidanceOutput sp = navigatePathTangent(curr_pos_local, curr_wp_local,
                                velocity_2d.normalized(), ground_speed, _wind_vel, curvature);
```

`navigatePathTangent()` 정의:

```
FixedWingModeManager.cpp:2780-2784
        const Vector2f unit_path_tangent{tangent_setpoint.normalized()};
        _closest_point_on_path = position_setpoint;
        return _directional_guidance.guideToPath(vehicle_pos, ground_vel, wind_vel,
                        tangent_setpoint.normalized(), position_setpoint, curvature);
```

`guideToPath()`에서 그 (점, 접선)이 곧 경로 정의다:

```
DirectionalGuidance.cpp:58-59
        const Vector2f path_pos_to_vehicle{curr_pos_local - position_on_path};
        signed_track_error_ = unit_path_tangent.cross(path_pos_to_vehicle);
```

→ 경로 = **`position` setpoint를 지나고 `velocity` 방향을 접선으로 갖는 (곡률 `curvature`의) 선**.
따라서 넣어야 할 값은 **경로 접선**이지 기체→목표 베어링(`chi_cmd`)이 **아니다**. 확정.

**부수 함정 1 — 축퇴 가드가 이 호출경로에서는 죽은 코드다.**
```
FixedWingModeManager.cpp:2775-2778
        if (tangent_setpoint.norm() <= FLT_EPSILON) {
                // degenerate case: no direction. maintain the last npfg command.
                return DirectionalGuidanceOutput{};
        }
```
그런데 `:1098`이 **이미 `.normalized()` 한 벡터**를 넘긴다. `matrix::Vector::normalized()`는
`unit()` = `(*this)/norm()` (`matrix/Vector.hpp:127-130,143-146`)이므로 영벡터를 넣으면 `0/0 = NaN`이다.
즉 이 인자의 norm은 **항상 1.0 아니면 NaN**이고, `NaN <= FLT_EPSILON`은 거짓 → **가드가 절대 발동하지 않는다.**
`velocity=(0,0)`은 `PX4_ISFINITE` 검사(`:389-390`, `:2154`)를 통과해 `AUTO_PATH`로 들어간 뒤
NaN 접선을 그대로 `guideToPath`에 밀어넣고, `DirectionalGuidance.cpp:99` `course_sp_ = atan2f(NaN, NaN)` = NaN이
된다. **"속도 0을 보내면 안 된다"가 A안의 추가 불문율이 된다** — 코드로 강제해야 하는 계약이 하나 늘어난다.

---

### Q2. 모드 이탈 (R-a) — 한 틱 NaN이면 강등되는가  🔴 **A안 최대 리스크, 실재**

**답: 강등된다. 히스테리시스는 존재하지 않는다. 그러나 기전은 계획서 서술보다 정확히 규정해야 한다.**

```
FixedWingModeManager.cpp:387-398   (실기체 c890d9db0a — 원문 그대로)
387:  if (_control_mode.flag_control_offboard_enabled && _position_setpoint_current_valid
388:      && _control_mode.flag_control_position_enabled) {
389:          if (PX4_ISFINITE(_pos_sp_triplet.current.lat) && PX4_ISFINITE(_pos_sp_triplet.current.lon)
              && PX4_ISFINITE(_pos_sp_triplet.current.vx)
390:              && PX4_ISFINITE(_pos_sp_triplet.current.vy)) {
391:                  // Offboard position with velocity setpoints
392:                  _control_mode_current = FW_POSCTRL_MODE_AUTO_PATH;
393:                  return;
395:          } else {
396:                  // Offboard position setpoint only
397:                  _control_mode_current = FW_POSCTRL_MODE_AUTO;
398:                  return;
```

- **매 사이클 무조건 재평가된다.** `Run()`(`:2029`)이 오프보드 변환(`:2123-2190`) → `set_control_mode_current()`(`:2240`)
  순으로 돌고, 모드 전이 시 수행되는 것은 착륙/이륙 상태 리셋(`:2256-2264`)과
  `_commanded_manual_airspeed_setpoint = NAN`(`:521-523`)뿐이다. **래치·유예·히스테리시스 없음.**
- **정정 1 — "발행 누락 한 틱"은 강등시키지 않는다.** `:2126`이
  `if (_trajectory_setpoint_sub.update(&trajectory_setpoint))`이므로 **새 메시지가 없으면 `_pos_sp_triplet`이
  그대로 유지**되고 모드도 유지된다. 위험한 것은 *누락*이 아니라 **velocity가 NaN인 메시지가 도착하는 것**이다
  (`:2154` 게이트를 못 넘으면 `vx/vy`는 `:2133-2134`의 NaN인 채 남는다 → `:389` 불성립 → `:397`).
  → 계획서 §3 R4-4의 "velocity가 유효하지 않은 틱에는 아예 발행하지 않는다"는 **소스적으로 옳은 대책**이다.
- **그러나 봉쇄되지 않는다 — 채널 전환이 곧 모드 이탈이다.**
  MAVROS `setpoint_position/local`(PoseStamped)도 `setpoint_raw/local`(PositionTarget)도 결국
  `SET_POSITION_TARGET_LOCAL_NED` → 같은 `_trajectory_setpoint_pub`(`mavlink_receiver.cpp:1282`)로 합류한다.
  PoseStamped 경로는 velocity ignore 비트를 세우므로 `mavlink_receiver.cpp:1196-1198`이 velocity를 **NaN**으로 만든다.
  현행 노드는 `_publish_pos_setpoint()`(PoseStamped)를 `offboard_node.py:506,510,720,798,953,987,1127,1132`
  **8개 지점**에서 쓴다(STREAMING/ENTRY/TRANSITION_FW/TRANSITION_MC/HOLD/FOLLOWING).
  A안은 이 중 FW 추종 구간만 `PositionTarget`으로 바꾸므로 **FW 구간 진입 1회 + 이탈 1회, 최소 2회의
  채널 전환**이 발생하고, 그 전환은 **정의상 `AUTO_PATH`↔`AUTO` 모드 전이**다.
  §2-1 채택기준 3("모드 이탈 0건")은 **A안 설계상 달성 불가능한 기준**이다.
- **강등의 대가는 빌드에 따라 다르다.** 가드 있는 실기체 빌드(`:577-578`, `:782-783`에
  `&& nav_state == NAVIGATION_STATE_GUIDED_COURSE` 존재)에서는 강등 = 현행 점 추적(무해).
  **가드 없는 빌드(`9bb0d365c4`)에서는 강등 = `course=0.0f` 분기 발화 = 정북 폭주.**
  그런데 §3 R4의 합격 기준이 바로 "취약 빌드에서도 통과"다 → **A안은 자기 합격 기준을 구조적으로 못 넘는다.**

---

### Q3. 종점 처리 (R-c) — 지나쳐 직진인가, loiter인가  🔴 **무한 직진. 실재**

**답: 무한 직진이다. `control_auto_path()`에는 종점 개념이 아예 없다.**

`control_auto_path()` 전문(`FixedWingModeManager.cpp:1085-1123`)에는
**수용반경·`handle_setpoint_type()`·LOITER 승격·`switchDistance` 어느 것도 없다.**
`:1098`이 무한 직선(또는 곡률 원)을 만들고 `:1101-1105`가 그대로 발행할 뿐이다.

대조 — 현행 경로(`FW_POSCTRL_MODE_AUTO` → `control_auto()`):
```
FixedWingModeManager.cpp:748-771   // POSITION → LOITER 승격 (수용반경 안 + 고도 미달 시)
FixedWingModeManager.cpp:2652-2658 // "we are beyond the end waypoint, fly back to it"
FixedWingModeManager.cpp:2669-2688 // navigateWaypoint = 점을 향한 pure pursuit → 지나치면 선회
```
→ **현행의 종점 포착 실패는 유계**다. 실측으로도 확인됨: F-10 / C1b가 flower-pattern으로 돌았다
(`campaign_report.md:248` "고도오차가 종점 포착 실패 → flower-pattern + 2.111g").
**A안에서 같은 실패는 20 m/s 무한 직선 이탈**이 된다. R1 거리상한 300 m는 이탈을
막는 게 아니라 **AUTO.LOITER 페일세이프 이벤트로 승격**시킨다 — 마감 직전에 새로 만드는 위험이다.

---

### Q4. NaN 마스크 함정 (R-d) — accel 0.0이 곡률 지령이 되는가

**답: 계획서 서술은 정정이 필요하다. 그러나 함정 자체는 실재하고, A안 설계가 그 피해를 증폭시킨다.**

경로: `mavlink_receiver.cpp:1200-1202`
```
1200:  setpoint.acceleration[0] = (type_mask & POSITION_TARGET_TYPEMASK_AX_IGNORE) ? (float)NAN : target_local_ned.afx;
```
→ `type_mask`에 AFX/AFY/AFZ ignore 비트가 없으면 `mavros_msgs/PositionTarget`의 **ROS 기본값 0.0**이
그대로 유효값으로 들어간다. 그 다음:
```
FixedWingModeManager.cpp:2160-2168   (SITL 9bb0d365c4 에서는 :2139-2147)
2160:  if (Vector3f(trajectory_setpoint.acceleration).isAllFinite()) {
2162:          Vector2f normalized_velocity_sp_2d = velocity_sp_2d.normalized();
2164:          Vector2f acceleration_normal = acceleration_sp_2d - acceleration_sp_2d.dot(...) * ...;
2166:          float direction = normalized_velocity_sp_2d.cross(acceleration_normal.normalized());
2167:          _pos_sp_triplet.current.loiter_radius = direction * velocity_sp_2d.norm() * velocity_sp_2d.norm()
2168:                                                  / acceleration_normal.norm();
```

**정정 (계획서 §2-1 R-d 서술이 정확하지 않다):**
가속도가 **정확히 (0,0,0)** 이면 `acceleration_normal = (0,0)` → `.normalized()`는
`matrix/Vector.hpp:105-109,127-130,143-146`에 의해 `0/0 = NaN` → `direction = NaN` →
`loiter_radius = NaN` → `:1095-1097`의 `PX4_ISFINITE` 검사에 걸려 **`curvature = 0.0f`(직선)**.
즉 **"accel 0.0 → 선회를 건다"는 그대로는 성립하지 않는다.** 우연히 무해한 쪽으로 축퇴한다.

**그러나 진짜 함정은 더 나쁜 곳에 있다.**
`|a_n|`이 0이 아닌 아주 작은 값이면 `loiter_radius = ±|v|²/|a_n|`이 **유한**해지고 곧바로 곡률 지령이 된다.
그리고 §1-3 규약 4는 **"`velocity`는 크기가 무시되므로 단위벡터를 쓴다"** 고 정하고 있다.
- `|v| = 20`(실속도) + `|a_n| = 0.1` → 반경 4000 m (사실상 직선, 무해)
- **`|v| = 1`(단위벡터) + `|a_n| = 0.1` → 반경 10 m** → 곡률 0.1 /m, 기체가 물리적으로 못 도는 선회 지령

즉 **"무시된다"고 문서화된 필드(velocity 크기)가 실제로는 선회반경을 400배 스케일**한다.
이건 `course = 0.0f`가 "정북 유지"로 읽힌 이번 사고와 **정확히 같은 계열의 함정을 우리 손으로 들여오는 것**이고,
게다가 **완화 방향이 아니라 악화 방향**으로 설계된 것이다.
심지어 이 표현식은 **2026-06-08에 상류에서 부호 버그가 고쳐진**(`04af7db228`, `direction` 앞의 `-` 제거) 코드다.

**봉쇄 방법(있기는 하다):** ①`type_mask`에 `IGN_AFX|IGN_AFY|IGN_AFZ`를 반드시 포함(=3520),
②동시에 `acceleration_or_force.{x,y,z}`에 **명시적으로 `float('nan')` 대입**(이중방어),
③단위테스트로 발행 메시지의 `type_mask == 3520` 및 accel 3성분 NaN을 매번 검증,
④**velocity를 단위벡터가 아니라 실제 순항속도 크기로 발행**(위 400배 증폭 제거).
→ **R-d 하나만은 "봉쇄 가능"으로 판정한다.** 다만 방어선이 상수 하나에 걸려 있다는 성질은 남는다.

---

### Q5. 고도(`position.z`) 추종 — F-5가 A안으로 나아지는가

**답: 나아질 여지가 없다. 계획서 §0의 추정(종방향 TECS 문제)이 소스로 지지된다.**

```
FixedWingModeManager.cpp:1107-1116   (control_auto_path — 종방향)
        .altitude = pos_sp_curr.alt,     ← 1109. 그대로 발행. 그게 전부다.
        .height_rate = NAN, .equivalent_airspeed = target_airspeed,
        .pitch_direct = NAN, .throttle_direct = NAN
```
```
FixedWingModeManager.cpp:846-855     (control_auto_position — 종방향)
        .altitude = position_sp_alt,     ← 812-844의 FOH(1차 홀드) 램프를 거친 값
        .height_rate = NAN, ...          ← 나머지 필드 동일
```

- **횡방향 유도(AUTO_PATH vs AUTO)와 무관하게 종방향 인터페이스는 동일**하다. 둘 다
  `fixed_wing_longitudinal_setpoint`에 목표 고도만 던지고 실제 제어는 `fw_lateral_longitudinal_control`(TECS)이 한다.
  선회 중 고도 침하는 **선회 중 유효 양력 감소 ↔ TECS 응답**의 문제이며 A안은 그 루프를 건드리지 않는다.
- 오히려 **A안이 잃는 것이 있다**: `control_auto_position`의 고도 FOH 램프(`:815-844`)가 `control_auto_path`에는
  없다. 다만 오프보드에서는 `_position_setpoint_previous_valid`가 **오프보드 분기에서 한 번도 대입되지 않으므로**
  (`:2189`는 `current`만, `:2195`는 비-오프보드 분기 전용) 현행에서도 이 FOH는 사실상 발동하지 않는다.
  → **A안의 z 처리 = 현행과 실질 동일. F-5 개선 근거 0.**
- F-5는 WP-E(선회 품질) 항목이고, F-8/F-9/F-10(`_cruise_alt` 스칼라화·천이 고도 계단·종점 포착)은
  전부 **우리 코드(`offboard_node.py`)** 문제다. A안은 어느 것도 건드리지 않는다.

---

### Q6. yaw 무시 (R-e) — `TRANSITION_FW` Phase 3에서 안전한가  🔴 **실재(양자택일 딜레마)**

**답: 안전하지 않다. 그리고 "천이는 A안에서 빼면 된다"는 회피로도 해결되지 않는다.**

**(가) A안이 천이 구간까지 덮는 경우 — PX4의 전방천이 전용 보호가 통째로 우회된다.**
```
FixedWingModeManager.cpp:536-566   move_position_setpoint_for_vtol_transition()
541:  if (_vehicle_status.in_transition_to_fw) {
546:  // Create a virtual waypoint HDG_HOLD_DIST_NEXT meters in front of the vehicle ...
549:  const float transition_heading = PX4_ISFINITE(current_sp.yaw) ? current_sp.yaw : _yaw;
550:  waypoint_from_heading_and_distance(_current_latitude, _current_longitude, transition_heading,
                                        HDG_HOLD_DIST_NEXT, &lat_transition, &lon_transition);
558-559:  current_sp.lat = _transition_waypoint(0);  current_sp.lon = _transition_waypoint(1);
```
이 함수는 **`control_auto()`의 `:574`에서만 호출된다. `control_auto_path()`(`:1085-1123`)는 호출하지 않는다.**
즉 A안으로 천이를 덮으면, `:546-548` 주석이 밝히는 목적("천이 중에 경로항법기가 추종할 수 있도록
전방 `HDG_HOLD_DIST_NEXT`(3000 m) 가상 WP를 만든다") 자체가 **사라지고**, 저속·불안정한 천이 구간에서
곧바로 지령 트랙을 추종하려 든다 — 목표점이 기체 뒤/옆에 있으면 그대로 뱅킹 지령이 된다.

동시에 yaw 지령이 사라진다:
```
mc_pos_control/PositionControl/PositionControl.cpp:119-120
        _yawspeed_sp = PX4_ISFINITE(_yawspeed_sp) ? _yawspeed_sp : 0.f;
        _yaw_sp      = PX4_ISFINITE(_yaw_sp) ? _yaw_sp : _yaw;   // TODO: better way to disable yaw control
```
`_yaw_sp = _yaw`는 **매 틱 현재 헤딩으로 갱신**된다 → "정렬된 헤딩을 잡아둔다"가 아니라
**yaw 강성 0(자유 표류)**이다. 13~15초 들여 맞춘 헤딩이 천이 진입 방향을 정하는데 그 구간에서
헤딩이 자유표류하게 된다. 이는 `offboard_node.py:814-838` 독스트링에 명시된
**2026-07-21 flight04 yaw 스핀 사고의 대응책(“yaw_ned는 필수 인자다”)을 무력화**하는 방향이다.

**(나) A안이 천이 구간을 덮지 않는 경우 — A안의 유일한 확정 이득이 부분달성에 그친다.**
```
FixedWingModeManager.cpp:377-380
        if (_vehicle_status.vehicle_type == ROTARY_WING && !_vehicle_status.in_transition_mode) {
                _control_mode_current = FW_POSCTRL_MODE_OTHER;   return;   // ← 천이 중에는 return 하지 않는다
        }
```
**천이 중(`in_transition_mode`)에는 FW 모드매니저가 계속 돈다.** 위치-only(PoseStamped)로 남기면
`:389` 불성립 → `:397` `AUTO` → `control_auto()` → **취약빌드에서 `course = 0.0f` 분기 발화**.
정천이 2.42~2.60 s + 역천이 4.92~6.09 s + `TRANSITION_MC` 초반 FW 구간이 전부 그 상태로 남는다.
→ **"PX4 버전 의존 제거"라는 A안의 유일한 확정 이득이 FOLLOWING 구간에만 적용된다.**

**(가)도 (나)도 받아들일 수 없다 → R-e는 봉쇄 불가.**

---

### Q7. 가드 있는 빌드(`c890d9db0a`)에서도 `:389-392`가 동일한가

**답: 동일하다. 확인 완료.** (§2 Q2에 인용한 `:387-398` 원문이 `/root/PX4-vehicle` = `c890d9db0a`에서
직접 읽은 것이다. `grep -n` 결과도 `387` / `392` / `402`로 SITL 빌드와 같은 위치.)

추가로 **`control_auto_path()` 본문도 두 빌드가 완전히 동일**하다:
- 실기체 `c890d9db0a`: `FixedWingModeManager.cpp:1085-1123`
- SITL `9bb0d365c4`: `FixedWingModeManager.cpp:1060-1098`
- 오프보드 변환 블록: 실기체 `:2123-2190` / SITL `:2102-2169` — **내용 동일**

두 빌드의 차이는 §1-2가 말한 대로 course 분기 가드 2곳(`:577-578`, `:782-783`)뿐이다.
즉 **A안의 동작 자체는 두 빌드에서 같을 것으로 예상된다.** (단 §3-2 참조 — 실기체 빌드에서 A안을
실제로 돌린 실측은 **0건**이다.)

---

## 3. 이득 검증 — 「0.2 m」의 정체와 C4·B7 인과

### 3-1. 0.2 m는 우리 cte와 **비교 불가**이며, 애초에 **거의 항등식**이다  🔴

`0.2 m`의 출처는 ulog `fixed_wing_lateral_guidance_status.signed_track_error`이고, 그 정의는:
```
DirectionalGuidance.cpp:58-59
        const Vector2f path_pos_to_vehicle{curr_pos_local - position_on_path};
        signed_track_error_ = unit_path_tangent.cross(path_pos_to_vehicle);
```
- **AUTO_PATH에서** `position_on_path`는 **우리가 보낸 점**, `unit_path_tangent`는 **우리가 보낸 속도 방향**
  (`:1098` → `:2781-2783`). 즉 **"PX4가 자기가 받은 지령선을 얼마나 잘 따랐는가"** 이지
  **우리 계획경로와의 오차가 아니다.**
- **현행(AUTO)에서 같은 필드는 구조적으로 항상 0이다.** `navigateWaypoint()`(`:2680-2685`)가
  `unit_path_tangent = (waypoint − vehicle).normalized()`, `_closest_point_on_path = waypoint_pos`로 넘기므로
  `path_pos_to_vehicle`은 접선과 **정확히 반평행** → 외적 ≡ 0.
  실제로 rootcause §1.4의 A3 폭주 ulog에서 `signed_track_error` **min = max = 0.000000 m**로 관측됐다.
- 즉 **"현행 0.000 m vs A안 0.2 m"** 라는 대조가 성립하지 않는다. 두 값 모두 **자기 지령 대비 자기 오차**이며
  경로 품질 정보가 0이다. **0.2 m를 캠페인 cte(1.1~19.6 m)와 나란히 놓는 것은 범주 오류다.**

**우리 캠페인 cte의 정의(전혀 다른 물리량):**
```
analyze_run.py:889-916   node.log 의 'FOLLOWING tick=… cte=…' — L1Guidance 가 **우리 계획경로**에 대해 계산
analyze_run.py:918-948   ulog 위치를 **원본 waypoints 폴리라인**에 투영한 기하편차(보조)
```
→ **비교 가능 여부: 불가.** 물리량이 다르고, 구간(직선 1레그 55 s vs 천이 진입 포함 전 미션)이 다르고,
기준(자기 지령 vs 계획경로)이 다르다.

### 3-2. 그 0.2 m는 **재현 불가능**하고, **실기체 빌드에서 측정된 적이 없다**  🔴

- 보존된 산출물 `logs/2026-07-27_s3_fw_offboard_probe/probe_trace.csv`의 헤더는
  `t,phase,sp_mode,east,north,up,mode,armed,vtol_state` — **`track_err` 열이 없다**(1011행 전부).
- 원본 ulog `18_31_30.ulg`는 `find /root -name 18_31_30.ulg` 결과 **어디에도 없다**(분석 후 삭제).
- 삭제 전 위치는 `/root/PX4-Autopilot/.../log/2026-07-26/`(A3 폭주 `17_53_07.ulg`와 같은 디렉터리, 마지막 잔존 파일 `18_07_31.ulg`)
  → **프로브는 취약 빌드 `9bb0d365c4`에서 돌았다. A안은 실기체 빌드에서 단 한 번도 실행된 적이 없다.**
- 프로브 조건(`tools/sitl/fw_offboard_probe.py:118-156,247-266`): **무풍**, **직선 1레그**,
  목표 = 현재위치 + 정동 800 m, 접선 = 정동 → **기체가 지령선 위에서 출발**, 코너 없음, 천이 미포함,
  `--speed 18`(velocity 크기 18, 단위벡터 아님).
  즉 "이미 올라타 있는 직선을 무풍에서 55초 유지" — **개선 여지가 존재하지 않는 실험**이다.

**§4-1 3항("재현 안 되는 개선 … 전부 기각 사유") 적용 → 이득 입증 실패.**

### 3-3. C4(바람 cte 1.2 → 4.0 m) — 개선 인과 **불성립**  🔴

§2-1은 "점 추적이라 밀린 만큼이 그대로 오프셋으로 남는다 / 정상상태 오프셋을 아무도 규제하지 않는다"를
전제로 삼는다. **캠페인 자기 데이터가 그 전제를 반증한다.**

`logs/2026-07-27_sitl_vtol_campaign/C4_pxvehicle_wind8/node.log` cte 시계열:
```
node.log:37  FOLLOWING 시작 pos=[13.3,-0.2] tgt=[83.3,0.0] cte=-0.2m   ← 경로 위에서 출발
tick= 20  cte=-1.6m
tick= 40  cte=-4.0m     ← 최대. 진입 과도구간
tick= 60  cte=-1.8m
tick= 80  cte=+0.7m     ← 부호 반전
tick=100  cte=+1.4m
tick=120  cte=+1.7m
tick=140  cte=-0.5m     ← 다시 반전
tick=160  cte=-0.2m     ← 출발값으로 복귀
```
(`metrics.json`: `n_samples 9 / first_m -0.2 / last_m -0.2 / max 4.0`.
무풍 대조 `A1_pxvehicle`은 −0.8 → −1.1 → −0.6 → −0.1 → 0.0 → 0.1로 단조 수렴, max 1.1 m.)

- **부호가 두 번 바뀌는 감쇠 진동**이다. "밀린 만큼 남는 오프셋"이면 부호가 바뀔 수 없다.
  현행 루프는 **8 m/s 바람에서도 cte를 ±0.2 m로 되돌린다.** 규제가 없는 게 아니라 **감쇠가 부족한 것**이다.
- 그리고 결정적으로 — **두 모드는 완전히 같은 함수를 호출한다.**
  `control_auto_path` → `navigatePathTangent`(`:1098`→`:2782`) 와
  `control_auto_position` → `navigateWaypoint`(`:880`→`:2684`) 는 **둘 다
  `DirectionalGuidance::guideToPath()`** 로 들어간다. 바람 삼각형·bearing feasibility·
  적응 주기·트랙오차 경계 처리(`DirectionalGuidance.cpp:53-99`)가 **동일 코드**다.
  달라지는 것은 넘기는 (점, 접선) 쌍뿐이다.
  → **"바람 대응이 좋아진다"는 소스상 인과가 없다.** 진동을 줄이려면 `NPFG_PERIOD`/`NPFG_DAMPING`
  또는 우리 lookahead 기하(70 m 고정)를 손대야 하며, 그건 A안과 무관하고 훨씬 싸다.

### 3-4. B7(추종 구간 0.9초 소멸) — 개선 인과 **불성립**  🔴

`campaign_report.md:160`: "`_FW_LOOKAHEAD`(70 m) > 경로 전장 → 목표가 항상 종점에 클램프".
그러나 **FOLLOWING 창의 길이를 정하는 것은 lookahead가 아니라 우리 상태기계**다:
```
offboard_node.py:78       _FW_LOOKAHEAD = 70.0
offboard_node.py:144     self.declare_parameter("d_end_thresh", 10.0)
offboard_node.py:1161    return trans_mc_trigger(dist_to_end, self._d_end_thresh)
fc_bridge/execution/state_logic.py:53-55
        def trans_mc_trigger(dist_to_end, d_end_thresh):  return dist_to_end < d_end_thresh
```
40 m 경로에서 FOLLOWING 창의 **상한**은 `(40 − 10) / 18 ≈ 1.7 s`이며, 이는 **A안 유무와 완전히 무관**하다.
A안은 "어떤 setpoint를 보내는가"만 바꾸고 "언제 FOLLOWING에 있는가"는 바꾸지 않는다.
→ B7의 해법은 F-11(lookahead 경로장 적응)과 결정 5(`d_end_thresh`)이며 **둘 다 우리 코드, WP-D 항목**이다.

---

## 4. R-f 검증 — git 이력 실측 (가설 반증)

`git log -L <범위>:src/modules/fw_mode_manager/FixedWingModeManager.cpp`, `/root/PX4-Autopilot`(전체 이력 보유),
최근 12개월 = **2025-07-27 이후**.

| 코드 영역 | 최근 12개월 변경 | 최신 커밋 |
|---|---|---|
| **`control_auto_path()` 본문** (A안 전용) | **0건** | `8c1f7ec7c0` 2025-04-05 (모듈 rename) |
| **`navigatePathTangent()`** (A안 전용) | **0건** | `8c1f7ec7c0` 2025-04-05 (모듈 rename) |
| `control_auto()` 머리 + course 분기 (**현행 전용**) | **3건** | `1499238f1c` 2026-07-03 revert / `2e59c98b7c` 2026-05-27 guard / `8b3ef1cf9e` 2026-05-22 feat |
| `control_auto_position()` (**현행 전용**) | **4건** | 위 3건 + `8424463102` 2026-07-23 FOH 램프 |
| `navigateWaypoints()`/`navigateWaypoint()` (**현행 전용**) | **1건** | `d0286481c5` 2026-06-23 |
| 오프보드→triplet 변환 블록 (**공유**) | **2건** | `171f0f38cf` 2026-06-25 gliding 회귀수정 / `04af7db228` 2026-06-08 loiter 방향 부호수정 |
| 파일 전체 | 24건 | — |

**판정: R-f 가설(“A안이 업그레이드에 더 취약하다”)은 데이터가 지지하지 않는다.**
오히려 우리를 깨뜨린 회귀 3건은 전부 **현행 경로**(`control_auto`/`control_auto_position`)에서 났고,
A안 전용 코드는 1년 넘게 손대지 않은 안정 영역이다. **이 항목은 기각 사유가 아니다.**

단 두 가지 단서를 남긴다:
1. **공유 영역이 실제 위험 지점이다.** 최근 1년 회귀 2건이 전부 오프보드 변환 블록에서 났고,
   그중 `04af7db228`은 **AUTO_PATH만 소비하는 곡률식의 부호 버그**(`direction` 앞 `-` 제거)였다.
   실기체 빌드에는 이 수정이 포함돼 있다(`git merge-base --is-ancestor 04af7db228 c890d9db0a` → YES).
   즉 **A안이 활성화하는 코드경로에서 최근 1년 안에 부호 버그가 실제로 출하됐다.**
2. `8424463102`(2026-07-23, FOH 램프 go-to 확장)는 실기체 빌드에 **미포함**(→ NO).

---

## 5. 부수 발견 — `course`와 **같은 계열의 제로초기화 함정이 하나 더 있다** (A안과 무관, 별건)

오프보드 변환 블록(`FixedWingModeManager.cpp:2128-2189`)은 `_pos_sp_triplet = {}` 후
`cruising_speed / cruising_throttle / vx / vy / vz / lat / lon / alt / gliding_enabled`에만 NaN을 넣는다.
**`yaw`가 그 목록에 없다.** 그런데 규약은:
```
msg/PositionSetpoint.msg:25   float32 yaw   # yaw (only in hover), in rad [-PI..PI), NaN = leave to flight task
msg/PositionSetpoint.msg:36   float32 course # [rad] desired course (bearing) over ground, NaN = unused
```
→ `_pos_sp_triplet.current.yaw == 0.0f`가 **유효한 "yaw 0 rad = 정북" 지령**으로 남는다.
그 결과 `move_position_setpoint_for_vtol_transition()`의
```
FixedWingModeManager.cpp:549  const float transition_heading = PX4_ISFINITE(current_sp.yaw) ? current_sp.yaw : _yaw;
FixedWingModeManager.hpp:106  static constexpr float HDG_HOLD_DIST_NEXT = 3000.0f;
```
가 **항상 참 분기**를 타서, 오프보드 전방천이의 가상 WP가 실제 헤딩과 무관하게 **항상 정북 3000 m**에 놓인다.
`course`가 **두 번째**로 같은 함정에 빠진 필드라는 rootcause §2의 지적에 이어, **`yaw`가 세 번째**다.

- **A안의 기각/채택과는 무관하다** (오히려 A안은 이 함수를 아예 호출하지 않아 이 경로를 우회한다).
- 정천이가 2.42~2.60 s로 짧아 실측 영향이 작았을 가능성이 크나, **확인된 바 없다.**
- **후속 조사 항목으로 등록 권고** — 신규 결함 후보. B8(정남 300 m) 등 비-정북 레그의
  천이 구간 ulog에서 `fixed_wing_lateral_guidance_status.course_setpoint`를 확인하면 판정된다.

---

## 6. 최종 권고

### 🔴 **A안: 기각.**

**§4-1 1항 불충족 (결정적):**
- **R-a 실재** — `:387-398`에 히스테리시스가 없고, **발행 채널 전환 자체가 모드 이탈**이라
  "모드 이탈 0건" 기준은 A안 설계상 달성 불가능하다.
- **R-c 실재** — `control_auto_path()`(`:1085-1123`)에 종점 로직이 **전무**하다. 현행의 유계 실패(flower-pattern)를
  무계 직선 이탈로 바꾼다.
- **R-e 실재** — 천이를 덮으면 `move_position_setpoint_for_vtol_transition()`(`:536-566`, `control_auto`에서만 호출)이
  우회되고 yaw 강성이 0이 되며(2026-07-21 사고 대응 무력화), 덮지 않으면 천이 구간이 `AUTO`로 남아
  **A안의 유일한 확정 이득(F-4 제거)이 부분달성**에 그친다.
- **R-b 부분 봉쇄만 가능** — `_find_segment`(`l1_guidance.py:183-208`) 수정은 레그 점프만 없앤다.
  코너에서의 **접선 계단(B4는 135°)** 은 A안 고유의 신규 위험이며 별도 블렌딩 코드가 필요하다.
- (R-d는 이중방어로 봉쇄 가능, R-f는 반증됨 — 그러나 §4-1 1항은 **6건 전부**를 요구한다.)

**§4-1 2항 불충족:**
- 유일한 근거 **0.2 m는 재현 불가능**(ulog 삭제, CSV에 해당 열 없음)하며, 정의상 **자기 지령 대비 자기 오차**로
  현행 값(구조적 0.000 m)과 비교조차 성립하지 않는다.
- **C4·B7 두 기대이득 모두 인과 불성립** — C4의 cte는 부호가 두 번 바뀌는 **감쇠 진동**(±0.2 m로 복귀)이라
  "규제 부재"라는 전제가 반증됐고, 두 모드는 **같은 `guideToPath()`** 를 쓴다. B7의 0.9초는
  우리 `_FW_LOOKAHEAD`/`d_end_thresh`가 정하는 값으로 A안이 건드리지 않는다.
- A안은 **실기체 빌드에서 한 번도 실행된 적이 없다**(프로브는 취약 빌드에서 실행).

**§4-1 4항 부합:** 실기체 FW+OFFBOARD 실적 **0건** 상태에서 유도법칙까지 바꾸면 첫 실비행의
미검증 변수가 2개가 된다. 마감은 가산점이 아니라 감점이다.

### 기각 시 조치 (§4-1 5항 — 이미 정해져 있음)

1. **현행 유지** — `setpoint_position/local`(PoseStamped) 위치+yaw 방식.
2. **PX4 핀 문서화** — 실기체는 `c890d9db0a`에 고정. `docs/rpi_deploy.md`에 "PX4 업그레이드 금지,
   사유: `FixedWingModeManager.cpp:577-578` 가드"를 못박는다.
3. **업그레이드가 필요해지면 상류 한 줄 패치** — 오프보드 변환 블록(`:2128-2139`)에
   `_pos_sp_triplet.current.course = NAN;` 추가. **본 심문 결과 `yaw`도 같이 넣어야 한다** (§5):
   `_pos_sp_triplet.current.yaw = NAN;`
4. **R4 세션은 통째로 건너뛴다.** 결정 1(MC 구간 통일 여부)은 자동 소멸.
5. **R3-(b) head-to-head 프로브 실험은 실행하지 않는다** — §4-1 1항에서 이미 탈락했으므로
   실측으로 뒤집힐 수 있는 상태가 아니다. SITL 시간은 R1·R2 회귀검증에 쓴다.

### 이 판정이 뒤집히려면 (반증 가능 형태로 못박음)

아래 **5개 전부**가 충족되어야만 재심 대상이 된다. 하나라도 빠지면 재심 없음.
1. **R-a**: 발행 채널 전환 없이 **미션 전 구간을 단일 `PositionTarget` 스트림으로** 처리하는 설계가
   제시되고, 그 설계가 MC 구간의 yaw 제어(`PositionControl.cpp:120`)와
   MC 속도 피드포워드 오염(단위벡터가 MC에서는 1 m/s 실지령이 된다)을 **동시에** 해결할 것.
2. **R-c**: `control_auto_path()`에 종점 개념이 없다는 사실을 전제로, 종점 포착 실패 시
   **유계 거동**을 보장하는 우리 쪽 메커니즘이 SITL로 실증될 것(R1 거리상한의 페일세이프 발동은 불인정).
3. **R-e**: 전방/역천이 구간에서 A안이 `move_position_setpoint_for_vtol_transition()` 부재를 대체할
   보호를 갖췄음을 실증하고, **동시에** 취약 빌드에서 천이 구간 `course_setpoint`가 정북으로 래치되지 않음을
   ulog로 보일 것.
4. **R-b**: `_find_segment` 수정 후 B5(폐곡선)·B4(U턴)에서 **지령 접선의 프레임간 각도 변화**를 로깅해
   90° 초과 계단이 0건임을 보일 것.
5. **이득**: 실기체 빌드(`c890d9db0a`)에서 C4와 **동일한 바람 8 m/s A1 경로**를 A안/현행 각 3회씩 돌려,
   **`node.log`의 L1Guidance cte(같은 물리량)** 최대값이 현행 4.0 m 대비 유의하게 감소하고
   **그 감소가 3회 모두 재현**될 것. (PX4 내부 `track_err`는 근거로 인정하지 않는다 — §3-1.)

---

## 부록 A. 인용한 소스 위치 일람 (재현 검증용)

| 파일 | 줄 | 내용 |
|---|---|---|
| `/root/PX4-vehicle` `src/modules/fw_mode_manager/FixedWingModeManager.cpp` | 377-380 | 회전익·비천이 시 `MODE_OTHER` 조기 return (천이 중에는 return 안 함) |
| 〃 | **387-398** | **오프보드 모드선택. 389-392 = AUTO_PATH 진입, 397 = AUTO 강등. 히스테리시스 없음** |
| 〃 | 521-523 | 모드 전이 시 수행되는 유일한 처리 |
| 〃 | 536-566, 574 | `move_position_setpoint_for_vtol_transition()` — `control_auto`에서만 호출 |
| 〃 | 577-578, 782-783 | **course 가드**(실기체 빌드에 존재. `9bb0d365c4`에는 없음) |
| 〃 | 748-772 | `handle_setpoint_type()` POSITION→LOITER 승격 (AUTO 전용) |
| 〃 | 812-855, 875-881 | `control_auto_position` 고도 FOH + `navigateWaypoint(s)` 분기 |
| 〃 | **1085-1123** | **`control_auto_path()` 전문 — 종점 로직 전무, 고도는 `alt` 그대로** |
| 〃 | 1094-1099 | velocity → `.normalized()` → 접선, `loiter_radius` → 곡률 |
| 〃 | 2123-2190 | 오프보드 `trajectory_setpoint` → `_pos_sp_triplet` 변환 |
| 〃 | **2160-2172** | **accel finite 시 `loiter_radius = direction·\|v\|²/\|a_n\|`** (SITL 빌드 `:2139-2151`) |
| 〃 | 2240 | `Run()` 내 `set_control_mode_current()` 호출 (변환 이후) |
| 〃 | 2297-2300, 2322-2324 | AUTO_PATH 디스패치 / OTHER = 무발행 |
| 〃 | 2669-2688, 2770-2785 | `navigateWaypoint()` / `navigatePathTangent()` |
| `FixedWingModeManager.hpp` | 106 | `HDG_HOLD_DIST_NEXT = 3000.0f` |
| `src/lib/npfg/DirectionalGuidance.cpp` | 53-59, 88-99 | 공통 `guideToPath()` — 바람 삼각형·`signed_track_error` 정의 |
| `src/lib/npfg/DirectionalGuidance.hpp` | 64-67 | `DirectionalGuidanceOutput{NAN, NAN}` |
| `src/lib/matrix/matrix/Vector.hpp` | 105-109, 127-130, 143-146 | `norm()` / `unit()` / `normalized()` — 영벡터 → NaN |
| `src/modules/mavlink/mavlink_receiver.cpp` | 1164-1174, 1192-1202, 1258-1282 | type_mask → NaN 매핑, `fill_offboard_control_mode`, 단일 `_trajectory_setpoint_pub` |
| `src/modules/commander/ModeUtil/control_mode.cpp` | 111-119 | OFFBOARD position 우선 |
| `src/modules/mc_pos_control/PositionControl/PositionControl.cpp` | 119-120 | NaN yaw → `_yaw_sp = _yaw` (매 틱 갱신 = 강성 0) |
| `msg/PositionSetpoint.msg` | 25, 36 | `yaw` / `course` — `NaN = unused` 규약 |
| `fc_bridge/guidance/l1_guidance.py` | 183-208 | `_find_segment` 전역 최근접 탐색(190행 캐시 초기값을 192-206 루프가 덮어씀) |
| `fc_bridge/execution/state_logic.py` | 53-55 | `trans_mc_trigger` |
| `fc_ros/fc_ros/nodes/offboard_node.py` | 78, 144, 814-838, 1161 | `_FW_LOOKAHEAD`, `d_end_thresh`, `_publish_pos_setpoint`(yaw 필수 사유), 종료 트리거 |
| `tools/sitl/analyze_run.py` | 889-948 | 캠페인 cte 정의(우리 계획경로 기준) |
| `tools/sitl/fw_offboard_probe.py` | 45-56, 118-156, 247-266 | 프로브 마스크·페이즈 조건(무풍·직선·지령선 위 출발) |
