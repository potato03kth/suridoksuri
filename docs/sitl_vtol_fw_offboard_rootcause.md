# FW 오프보드 경로추종 폭주 — 근본원인 규명 (SITL-7 S3, 2026-07-27)

> **한 줄 결론:** 우리 코드는 정상이다. PX4 상류(upstream)의 회귀 버그다.
> `FixedWingModeManager` 가 오프보드 위치 setpoint 를 **"코스 0 rad(정북) 유지"** 로
> 오해석한다. **우리 SITL 빌드는 이 버그에 걸리고, 실기체 펌웨어는 안 걸린다.**
> 그러나 실기체를 최신 PX4 로 올리는 순간 똑같이 걸린다.
> **우회로는 확보했다 — SITL 에서 실측 검증 완료(횡오차 0.2 m).**

- 조사 대상 SITL: PX4 `v1.18.0-beta1-155-g9bb0d365c4` (WSL `Ubuntu-22.04`, `/root/PX4-Autopilot`)
- 실기체 펌웨어: PX4 `c890d9db0a300795594fd5ba6c045be9ebd71c09` (`ver_hw=PX4_FMU_V6C`)
- 증거 ulog: A3 폭주 `…/rootfs/log/2026-07-26/17_53_07.ulg`, 프로브 `18_31_30.ulg`(183 MB, 분석 후 삭제)
- 재현 도구: `tools/sitl/fw_offboard_probe.py`, 궤적 `logs/2026-07-27_s3_fw_offboard_probe/`

---

## 1. 근본원인 — `position_setpoint.course` 제로초기화

### 1.1 오프보드 setpoint 는 FW 에서 어떤 경로를 타는가

`v1.18` 에는 `flight_mode_manager/tasks/Offboard` 같은 전용 태스크가 **없다**.
FW 오프보드는 `fw_mode_manager` 모듈이 직접 처리한다. 경로는 다음과 같다.

1. `mavlink_receiver.cpp:1401` — `SET_POSITION_TARGET_LOCAL_NED`(및 `…_GLOBAL_INT`)
   수신 → `offboard_control_mode` 발행 + **`_trajectory_setpoint_pub.publish(setpoint)`**
   (두 메시지가 **동일한** `trajectory_setpoint` 토픽으로 합류한다 — 이게 §3 의 근거)
2. `commander/ModeUtil/control_mode.cpp:111-115` — `NAVIGATION_STATE_OFFBOARD` 이고
   `offboard_control_mode.position` 이면 `SetpointType::Trajectory` →
   `setpoint_types.cpp:75-83` 이 `flag_control_position_enabled = true` 로 설정
3. `fw_mode_manager/FixedWingModeManager.cpp:2102` — `flag_control_offboard_enabled` 분기에서
   `trajectory_setpoint` 를 **내부 멤버** `_pos_sp_triplet` 으로 변환
4. `FixedWingModeManager.cpp:387` — 모드 선택
5. `FixedWingModeManager.cpp:2260` — 선택된 모드로 제어함수 디스패치

> **S2 관측의 정정:** ulog 의 `position_setpoint_triplet.current.valid=0 / type=5(IDLE)` 는
> 오작동 증거가 **아니다**. 그 uORB 토픽은 `navigator` 가 발행하는데 OFFBOARD 에서는
> navigator 가 유휴다. FW 오프보드가 쓰는 건 발행되지 않는 **내부 멤버** `_pos_sp_triplet`
> 이다(위 3번). 이 토픽을 근거로 "PX4 가 setpoint 를 안 받았다"고 판단하면 안 된다.

### 1.2 결함 지점

`FixedWingModeManager.cpp:2106-2118` — 변환 블록이 구조체를 통째로 제로초기화한 뒤,
**0 이 유효값으로 오인될 수 있는 필드만 골라 명시적으로 NaN 을 넣는다**:

```cpp
2107:  _pos_sp_triplet = {}; // clear any existing
2110:  _pos_sp_triplet.current.cruising_speed   = NAN; // ignored
2111:  _pos_sp_triplet.current.cruising_throttle = NAN; // ignored
2112:  _pos_sp_triplet.current.vx  = NAN;
2113:  _pos_sp_triplet.current.vy  = NAN;
2114:  _pos_sp_triplet.current.vz  = NAN;
2115:  _pos_sp_triplet.current.lat = static_cast<double>(NAN);
2116:  _pos_sp_triplet.current.lon = static_cast<double>(NAN);
2117:  _pos_sp_triplet.current.alt = NAN;
2118:  _pos_sp_triplet.current.gliding_enabled = false;
```

**`course` 가 이 목록에 없다.** 그래서 `_pos_sp_triplet.current.course == 0.0f` 로 남는다.

그런데 `msg/PositionSetpoint.msg:36` 의 규약은:

```
float32 course   # [rad] desired course (bearing) over ground, NaN = unused
```

**NaN 이라야 "미사용"이다. 0.0f 는 `PX4_ISFINITE` 를 통과하는 유효한 "코스 0 rad = 정북" 명령이다.**

`navigator` 쪽은 이 규약을 지킨다 —
`navigator/mission_block.cpp:618` `sp->course = NAN; // mission items never command a course, only Course mode sets it`,
`navigator/navigator_main.cpp:1341` `sp.course = NAN;`.
**오프보드 경로만 navigator 를 거치지 않아 규약에서 누락됐다.**

### 1.3 결함이 발현되는 지점

`FixedWingModeManager.cpp:576-580` (`control_auto`):

```cpp
576:  // Course setpoints are handled directly to avoid entering hold mode
577:  if (PX4_ISFINITE(current_sp.course)) {
578:      control_auto_position(control_interval, curr_pos, ground_speed, pos_sp_prev, current_sp);
579:      return;
580:  }
```

`course = 0.0f` → 조건 성립 → `handle_setpoint_type()` 이하 **정상 웨이포인트 항법 전체를 건너뛴다**.

`FixedWingModeManager.cpp:780-785` (`control_auto_position`):

```cpp
780:  // Course Hold: if a course is explicitly set, navigate along that bearing (ground track)
781:  if (PX4_ISFINITE(pos_sp_curr.course)) {
785:      const DirectionalGuidanceOutput sp = navigateBearing(curr_pos_local, pos_sp_curr.course, ground_speed, _wind_vel);
```

`navigateBearing` 의 정의 (`FixedWingModeManager.cpp:2778-2784`):

```cpp
2781:  const Vector2f unit_path_tangent = Vector2f{cosf(bearing), sinf(bearing)};
2782:  _closest_point_on_path = vehicle_pos;
2783:  return _directional_guidance.guideToPath(vehicle_pos, ground_vel, wind_vel, unit_path_tangent, vehicle_pos, 0.0f);
```

경로 위의 기준점으로 **기체의 현재 위치 자신**(`vehicle_pos`)을 넘긴다.
→ 횡방향 오차가 **구조적으로 항상 0** 인 무한직선.
→ `lat`/`lon` 은 단 한 번도 읽히지 않는다.

**결과: 목표점이 어디에 있든 기체는 정북으로 영원히 직진하며, PX4 는 스스로를 "완벽히 온트랙"으로 판단한다.** 경보가 0건이고 페일세이프가 안 걸린 이유가 이것이다.

### 1.4 A3 폭주 ulog 의 실측 확인 (`17_53_07.ulg`)

OFFBOARD FW 구간 **t=135~566 s, 4310 샘플 전부**:

| 항목 | 실측값 |
|---|---|
| `fixed_wing_lateral_guidance_status.course_setpoint` | min = max = **4.371138829e-08 rad** (= 0.0000025°, 정북) |
| `fixed_wing_lateral_guidance_status.signed_track_error` | min = max = **0.000000 m** |
| `fixed_wing_lateral_setpoint.lateral_acceleration` | min = max = **0.000000** |
| `fixed_wing_lateral_setpoint` 발행 공백 (>1 s) | **0 건** (즉 제어루프는 정상 동작 중이었다) |
| `trajectory_setpoint.position` finite 비율 | 0.973 (우리 노드는 setpoint 를 정상 발행 중이었다) |
| 최종 setpoint (t=560.4) | `[199.97, 103.62, -49.80]` |
| 기체 위치 | x −0.3 → **6523.1 m**, y −1.9 → **23.3 m** |

OFFBOARD 이탈 **직후** `t=567.3` 에 `signed_track_error = −6442.7 m` 가 곧바로 나타난다 —
정상 항법으로 돌아오자마자 진짜 오차가 계산된 것. 버그 구간에서만 0 이었다는 결정적 대조다.

고도가 정확히 추종된 것도 이 메커니즘으로 설명된다: `control_auto_position` 의 코스분기는
횡방향만 가짜로 만들고, 종방향은 `FixedWingModeManager.cpp:793-802` 에서
`altitude = pos_sp_curr.alt` 로 **정상 발행**한다.

### 1.5 "직선 5건 성공" 은 전부 우연이었다 — 확정

`course = 0 rad` = **정북** = 로컬 NED `+x`. 캠페인 시나리오의 waypoint 를 대조하면:

| 시나리오 | waypoints | 방향 | 결과 |
|---|---|---|---|
| A1 | `[0,0,50, 300,0,50]` | 정북 | "성공" |
| A2 | `[0,0,50, 300,0,50]` | 정북 | "성공" |
| A4 | `[0,0,50, 150,0,80, 300,0,50]` | 정북 | "성공" |
| B1 | `[0,0,50, 500,0,50]` | 정북 | "성공" |
| B6 | `[0,0,50, 200,0,50]` | 정북 | "성공" |
| **A3** | `[0,0,50, 200,0,50, 200,200,50]` | 정북 → **정동** | **정북 레그만 통과 후 폭주** |

**성공한 5건 전부가 정북 직선이다.** A3 의 노드 로그도 정확히 이 경계에서 갈린다 —
`tick=140 pos=[220.9,-0.9]` 까지(정북 레그) 정상, 목표가 동쪽 레그
`tgt=[199.9, 70.0]` 로 바뀐 `tick=160` 부터 `cte=-48.7m` 로 발산 시작.

> **따라서 이번 캠페인의 FW 경로추종 "성공" 기록 5건은 전부 무효다.**
> 검증된 것은 "정북으로 나는 기능"뿐이며, 경로추종은 한 번도 검증된 적이 없다.

---

## 2. 상류 회귀 이력 — 언제 들어왔고, 왜 실기체는 무사한가

`git log`/`git blame`/`git merge-base` 로 확정한 타임라인이다.

| 커밋 | author / 병합일 | 내용 |
|---|---|---|
| `8b3ef1cf9e` | 2026-05-22 / **2026-05-27** | `feat(navigator): add Guided Course mode for fixed-wing` — `course` 필드 신설 + `control_auto`/`control_auto_position` 코스 분기 추가. navigator 는 `course=NAN` 으로 갱신했으나 **오프보드 경로는 누락**. ⇒ **버그 삽입** |
| `2e59c98b7c` | 2026-05-27 / **2026-05-28** | `fix(fw_mode_manager): guard course guidance on GUIDED_COURSE nav state` — 코스 분기에 `nav_state == NAVIGATION_STATE_GUIDED_COURSE` 조건 추가. 커밋 메시지 원문: *"This prevents **zero-initialized course setpoints** from overriding GCS commanded auto-modes."* ⇒ **버그 차단** |
| **`c890d9db0a`** | **2026-07-06** | **← 실기체 펌웨어.** feature + guard 둘 다 포함 |
| `fd13202851` | 2026-07-08 | tag `v1.18.0-beta1` |
| `1499238f1c` | 2026-07-03 / **2026-07-17** | `revert(fw_mode_manager): drop GUIDED_COURSE guard on course setpoints` — 가드 제거. 사유: *"The navigator now guarantees the course field is NaN unless Course mode explicitly commands it"* ⇒ **전제가 틀렸다. 오프보드는 navigator 를 거치지 않는다.** ⇒ **버그 재삽입** |
| **`9bb0d365c4`** | **2026-07-23** | **← 우리 SITL.** 실기체보다 178 커밋 앞섬 |

검증 명령과 결과(`/root/PX4-Autopilot`):

```
git merge-base --is-ancestor 8b3ef1cf9e  c890d9db0a  → YES   (feature 포함)
git merge-base --is-ancestor 2e59c98b7c  c890d9db0a  → YES   (guard 포함 → 안전)
git merge-base --is-ancestor 1499238f1c2 c890d9db0a  → NO    (revert 미포함)
git merge-base --is-ancestor 1499238f1c2 9bb0d365c4  → YES   (revert 포함 → 취약)
git rev-list --count c890d9db0a..9bb0d365c4          → 178
```

실기체 펌웨어의 소스 원문 (`git show c890d9db0a:…/FixedWingModeManager.cpp`, 577-578행):

```cpp
577:  if (PX4_ISFINITE(current_sp.course)
578:      && _vehicle_status.nav_state == vehicle_status_s::NAVIGATION_STATE_GUIDED_COURSE) {
```

OFFBOARD 의 `nav_state` 는 14(`NAVIGATION_STATE_OFFBOARD`)이지 `GUIDED_COURSE` 가 아니다.
→ **실기체에서는 코스 분기가 발화하지 않고 정상 웨이포인트 항법이 돈다.**

> **전례:** 같은 블록에서 **같은 유형의 버그가 한 달 전에도 났다** —
> `171f0f38cf fix(fw_mode_manager): Fix regression with offboard gliding setpoints (#26538)`
> (2026-06-25) 가 `gliding_enabled` 제로초기화 누수를 고쳐 2118행을 추가했다.
> `course` 는 같은 함정에 두 번째로 빠진 필드다. 구조적 결함이며 또 재발할 수 있다.

---

## 3. SITL 실험 — 대안 setpoint 3종 실측

`tools/sitl/fw_offboard_probe.py` 로 **최소 비행 1회**만 만들어 페이즈를 이어 붙였다
(하니스 미사용, ARM → `CommandTOL` → MC OFFBOARD → `MAV_CMD_DO_VTOL_TRANSITION(param1=4)` →
FW 확인 후 페이즈별 60초). 모든 페이즈에서 목표는 **현재 위치에서 정동(EAST) 800 m**.

판정 기준: `/mavros/local_position/pose` 의 60초 변위로 계산한 평균 지상코스.
**정동 = 90°, 정북 = 0°.**

| 페이즈 | 발행 채널 / `type_mask` | dE | dN | 평균 지상코스 | 판정 |
|---|---|---|---|---|---|
| **pos_vel (A안)** | `setpoint_raw/local`, 위치+속도 | **+794.6 m** | **+0.0 m** | **90.0°** | ✅ **선회함 — 목표 정확 추종** |
| vel_only (B안) | `setpoint_raw/local`, 속도만 | +41.6 m | +775.4 m | 3.1° | ❌ 선회 안 함 (정북 이탈) |
| pos_only (현행) | `setpoint_raw/local`, 위치만 | −1.7 m | +814.7 m | 359.9° | ❌ 선회 안 함 (정북) |

프로브 ulog `18_31_30.ulg` 내부 상태로 각 페이즈의 **메커니즘까지** 확인했다.

**pos_vel (t=388.5~445.1)** — `offboard_control_mode position=1 velocity=1` →
`vehicle_control_mode position=1` → `FixedWingModeManager.cpp:389-392` 의
`ISFINITE(lat)&&ISFINITE(lon)&&ISFINITE(vx)&&ISFINITE(vy)` 성립 →
**`FW_POSCTRL_MODE_AUTO_PATH`**:

```
t= 402.9  course_sp= +90.25°  track_err= -0.16 m   pos N=  29.3 E= 244.2
t= 422.9  course_sp= +90.03°  track_err= -0.02 m   pos N=  29.1 E= 519.4
t= 442.9  course_sp= +89.95°  track_err= +0.03 m   pos N=  29.1 E= 806.2
```

**진짜 폐루프 경로추종이다 — 횡오차 0.2 m 이내, 북좌표 ±0.2 m 유지.**
`FW_POSCTRL_MODE_AUTO_PATH` 는 스위치(`:2288-2291`)에서 `control_auto_path()` 를
**직접** 호출하므로 `control_auto()` 안의 코스 분기를 **원천적으로 거치지 않는다**.
이것이 A안이 통하는 이유다.

**pos_only (t=498.5~553.0)** — `position=1 velocity=0` → `:397` `FW_POSCTRL_MODE_AUTO`
→ `control_auto()` → 코스 분기 발화:

```
t= 502.9  course_sp=  +0.00°  track_err= +0.00 m   pos N= 872.4 E= 880.5
t= 522.9  course_sp=  +0.00°  track_err= +0.00 m   pos N=1174.8 E= 879.5
t= 542.9  course_sp=  +0.00°  track_err= +0.00 m   pos N=1481.9 E= 878.9
```

정동 800 m 를 명령하는 동안 코스 0.00°, 오차 0.00 — **§1 버그의 in-vivo 직접 확인.**

**vel_only (t=445.1~498.5)** — `position=0` → `control_mode.cpp:119` 가
`flag_control_position_enabled = false` 로 강제 → `:387` 조건 불성립.
오프보드에서는 `flag_control_auto_enabled` 도 `flag_control_manual_enabled` 도 0 이라
이후 분기가 전부 불성립 → **`:517-518` `FW_POSCTRL_MODE_OTHER`** → `:2313-2315` `break` →
**아무 setpoint 도 발행하지 않는다.** ulog 실측으로 확인됨:

```
fixed_wing_lateral_setpoint 발행 공백: 전 비행 통틀어 1건
   t=445.5 → 498.4  (52.96 s)   ← vel_only 구간과 0.1초 단위로 일치
```

즉 **FW 오프보드 속도제어는 "무시"가 아니라 "제어기 자체가 정지"** 한다.
`offboard_node.py:817` 주석의 "FW 는 속도/가속도를 무시한다" 는 결론은 맞지만,
실제 기전은 더 나쁘다 — 기체는 마지막 횡방향 명령으로 무기한 활공한다.

> **미확정:** vel_only 진입 직후 기체가 정동에서 **정북으로 돌아선** 정확한 원인.
> `offboard_control_mode`(t=445.1)와 `vehicle_control_mode`(t=445.5) 사이 **0.4초 창**에서
> `position_enabled` 는 아직 1, `lat/lon` 은 이미 NaN → `:389` 불성립 → `:397`
> `FW_POSCTRL_MODE_AUTO` → 코스 분기(course=0) 가 1회 발화해 정북이 래치됐다는 게
> 가장 그럴듯한 설명이지만, 0.4초 구간을 직접 샘플링해 확정하지는 못했다.
> 결론(대안 B 기각)에는 영향 없다.

**C안 (`setpoint_raw/global`, GLOBAL_INT) — 실험 불필요, 소스로 기각.**
`mavlink_receiver.cpp:1401` 에서 `SET_POSITION_TARGET_GLOBAL_INT` 도
`SET_POSITION_TARGET_LOCAL_NED` 와 **동일한 `_trajectory_setpoint_pub`** 로 합류한다.
하류 경로가 완전히 같으므로 프레임만 바꾸는 것으로는 아무것도 달라지지 않는다.
(단, C안도 `velocity` 를 함께 실으면 A안과 동일하게 `AUTO_PATH` 로 들어간다.)

---

## 4. SITL-4(2026-06-30) "L자 경로 완료 ✅" 기록의 진위 — 진짜였을 것

`docs/sitl_verification_log.md:609` (`### L자 경로`, SITL-4 섹션은 `:570`) 의 L자 완주
기록은 **환경 차이로 설명되며, 조작이나 착오로 볼 근거가 없다.**

- SITL-4 실행일 **2026-06-30** 은 가드가 살아 있던 구간
  (`2e59c98b7c` 병합 2026-05-28 ~ `1499238f1c` 병합 2026-07-17) **한복판**이다.
  그 시점의 어떤 main 빌드를 쓰더라도 코스 분기는 발화하지 않는다.
- 그보다 이전 소스였다면 `course` 필드 자체가 없다(`8b3ef1cf9e` 병합 2026-05-27).
  어느 쪽이든 L자 추종은 **정상 동작한다**.
- 당시 코드가 발행한 것도 지금과 같은 위치 setpoint 다. `git log` 상 `offboard_node.py`
  의 FOLLOWING 분기는 그때도 `/mavros/setpoint_position/local`(PoseStamped) 였고,
  가드 하에서는 그것으로 충분했다.
- **미확정:** 당시 개발컴의 PX4 커밋 해시. 저장소 문서에 버전이 기록돼 있지 않고
  (`docs/sitl_verification_log.md:23` 은 "PX4-Autopilot (소스, WSL 로컬 빌드)" 라고만 적음),
  SITL-4 의 ulog 도 저장소에 없어 `ver_sw` 를 뽑을 수 없다. 날짜 구간으로만 판정했다.
  → **교훈: 앞으로 SITL 기록에는 `ver_sw` 를 반드시 남긴다.**

---

## 5. 실기체는 어떤가

### 5.1 펌웨어 버전 (ulog 헤더 실측)

| 로그 | `ver_sw` | `ver_sw_release` | `ver_hw` |
|---|---|---|---|
| `logs/2026-07-23_flight01/11_32_15.ulg` | `c890d9db0a30…` | 17956928 (`0x01120040`, v1.18.0-alpha) | `PX4_FMU_V6C` |
| `logs/2026-07-25_flight14/log_190_….ulg` | `c890d9db0a30…` | 17956928 | `PX4_FMU_V6C` |
| `logs/2026-07-25_flight17/log_201_….ulg` | `c890d9db0a30…` | 17956928 | `PX4_FMU_V6C` |
| (참고) SITL A3 `17_53_07.ulg` | `9bb0d365c4ff…` | 17956992 (`0x01120080`, v1.18.0-beta) | `PX4_SITL` |

실기체는 전 비행 동일 펌웨어 `c890d9db0a` 이며, **§2 대로 가드가 살아 있어 이 버그에 걸리지 않는다.**

### 5.2 실기체 FW 오프보드 실적 — **전무하다**

저장소의 실비행 ulog 전수(78건 이상, `logs/2026-07-*/`)를 `vehicle_status.vehicle_type`
기준으로 스캔한 결과:

- **`FW(vehicle_type=2)` + `OFFBOARD(nav_state=14)` 동시 구간이 있는 비행: 0 건.**
- FW 구간 자체가 존재하는 로그도 `logs/2026-07-24_manual_recovered_ulog/08_42_22.ulg` 의
  **2.4 초**(POSCTL 수동 천이 테스트)가 유일하다.
- 2026-07-25 flight14/16/17/19 는 전부 `vehicle_type=1`(멀티콥터) 전용이다.
  (flight17: `AUTO.TAKEOFF → AUTO.LOITER → OFFBOARD(21.4 s) → POSCTL`, FW 진입 없음)

> **따라서 "실기체에선 된다"는 직접 증거는 없다.** 실기체가 안전하다는 판단은
> §2 의 소스 계보(가드 존재)에 **전적으로** 근거한 것이며, 비행 실측으로 확인된 바 없다.

---

## 6. 결론 — "우리 FW 경로추종 설계가 실기체에서 동작하는가?"

### **(b) PX4 v1.18 한계인데 실기체 버전에선 동작 — 단, 조건부다.**

근거:
1. 우리 발행 방식(위치 setpoint)은 **PX4 규약상 올바르다**. 실기체 펌웨어
   `c890d9db0a` 에는 가드가 있어 정상 웨이포인트 항법으로 처리된다(§2).
2. 폭주는 우리 SITL 빌드 `9bb0d365c4` 가 `1499238f1c`(2026-07-17 병합) 이후라서 생긴
   **상류 회귀**다. 우리 코드 변경과 무관하다.

**단, (b) 를 안심의 근거로 쓰면 안 된다:**
- 실기체 FW+OFFBOARD 비행 실적이 **0 건**이다(§5.2). "된다"는 소스 추론일 뿐 미검증이다.
- 실기체 펌웨어를 2026-07-17 이후 PX4 로 올리는 순간 **똑같이 폭주한다.**
  → **대회 전 PX4 업그레이드 금지.** 올려야 한다면 §7 조치를 **먼저** 넣는다.
- 현행 위치-only 방식으로는 이 회귀가 상류에 남아 있는 한 SITL 회귀검증을 **영원히 할 수 없다.**

### 우리가 취할 수 있는 자세 — §7 의 A안으로 (a) 로 전환 가능

발행 방식을 바꾸면 **가드 유무와 무관하게 두 버전 모두에서 동작**하는 코드가 된다.
그렇게 하면 이 문제는 "고치면 되는 우리 쪽 문제 (a)" 로 강등된다. 그게 권고안이다.

---

## 7. 권고 조치 (실행 방법)

> 비행코드는 이번 세션에서 **수정하지 않았다.** 아래는 제안이며 채택 여부는 오케스트레이터 판단이다.

### A안 — `setpoint_raw/local` 로 위치 + 속도 동시 발행 (권고)

**효과:** `FW_POSCTRL_MODE_AUTO_PATH` 로 진입해 코스 분기를 **원천 회피**한다.
SITL 실측 횡오차 **0.2 m**(§3). 가드가 있는 실기체 펌웨어에서도 `:389-392` 조건은
동일하게 성립하므로 **양쪽 버전 모두에서 동작**한다.

**구현 요지** (`offboard_node.py` 의 FW FOLLOWING 분기, `_publish_setpoint` 계열):

1. 발행 토픽을 `/mavros/setpoint_position/local`(`PoseStamped`) →
   `/mavros/setpoint_raw/local`(`mavros_msgs/msg/PositionTarget`) 로 교체.
2. `coordinate_frame = PositionTarget.FRAME_LOCAL_NED`(=1).
   **값은 ENU 로 채운다** — MAVROS 가 NED 로 변환한다(현행 PoseStamped 와 동일 규약).
3. `type_mask` = 가속도·yaw·yaw_rate 무시, **위치와 속도는 활성**:
   `IGNORE_AFX|IGNORE_AFY|IGNORE_AFZ|IGNORE_YAW|IGNORE_YAW_RATE` (= 64+128+256+1024+2048 = **3520**)
4. `position` = 현행 L1 목표점(ENU), `velocity` = **레그 접선 단위벡터 × 속력**(ENU).
   `control_auto_path` 는 `velocity_2d.normalized()` 만 쓰므로(`:1073`) **방향만 의미가 있고
   크기는 무시된다.** 속력은 아무 양수나 넣어도 되지만 접선 방향은 정확해야 한다.
5. **가속도는 반드시 NaN(마스크로 무시)로 둘 것.** finite 가속도를 넣으면
   `:2139-2151` 이 곡률(`loiter_radius`)을 계산해 선회를 걸어버린다.
6. `cruising_speed` 는 `:2110` 에서 무조건 NaN 으로 덮이므로 **오프보드에서 순항속도는
   지정할 수 없다.** 대기속도는 `FW_AIRSPD_TRIM` 파라미터로만 정해진다 — `v_cruise` 를
   setpoint 로 전달하려는 시도는 무의미하다(현행 코드가 그러고 있다면 정리 대상).

**MC 구간 영향:** `mc_pos_control` 은 `trajectory_setpoint` 를 그대로 쓰고 `course` 와
무관하므로 MC 거동은 바뀌지 않는다. 다만 MC/FW 가 채널을 공유하면 `type_mask` 를
구간별로 다르게 줘야 한다 — MC 는 현행대로 위치-only 가 안전하다.

### B안 — 속도 setpoint: **기각**

§3 실측 + `:517-518` 소스로 확정. `flag_control_position_enabled=false` 가 되면 FW
위치제어기가 `FW_POSCTRL_MODE_OTHER` 로 빠져 **횡·종방향 setpoint 발행을 전부 중단**한다
(실측 52.96초 공백). 기존 코드 주석의 "FW 가 무시한다" 는 가정은 **결론은 맞았고 기전은 틀렸다.**

### C안 — `setpoint_raw/global`: **단독으로는 기각**

`mavlink_receiver.cpp:1401` 에서 LOCAL_NED 와 같은 토픽으로 합류. 프레임만 바꿔서는 무의미.

### 부수 조치

- **PX4 업그레이드 동결.** 실기체 `c890d9db0a` 유지. 올릴 경우 A안 선행 필수.
- **상류 이슈 제기 검토.** `1499238f1c` 의 revert 전제가 오프보드 경로에서 성립하지 않는다.
  최소 수정은 `FixedWingModeManager.cpp:2118` 뒤에 `_pos_sp_triplet.current.course = NAN;`
  한 줄이다. (`gliding_enabled` 가 `171f0f38cf` 에서 받은 것과 동일한 처방)
- **SITL 로컬 패치 검토.** 회귀검증을 계속하려면 위 한 줄을 우리 SITL 트리에 넣고
  재빌드하는 방법도 있다. 단 A안을 넣으면 패치 없이도 검증 가능해진다.
- **캠페인 회귀 재실행.** A안 채택 시 A1/A2/A4/B1/B6 는 **전부 무효이므로 재실행해야 한다**
  (§1.5). 그리고 앞으로 회귀 시나리오에는 **반드시 비-정북 레그를 포함**시킨다 —
  정북 직선만으로는 이 급의 버그가 통과한다는 게 이번에 실증됐다.
- **SITL 기록에 `ver_sw` 필수화.** SITL-4 진위 판정이 어려웠던 이유가 이것이다(§4).

---

## 8. 재현 방법

```bash
# 1) PX4 SITL + MAVROS (WSL Ubuntu-22.04)
export PX4_SIM_MODEL=gz_standard_vtol GZ_IP=127.0.0.1 HEADLESS=1
cd /root/PX4-Autopilot/build/px4_sitl_default/rootfs
/root/PX4-Autopilot/build/px4_sitl_default/bin/px4 >/dev/null 2>&1 &   # ⚠️ 콘솔 파일 리다이렉트 금지
ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14580 &

# 2) 프로브 (비행코드 무수정, ros2 토픽만 사용)
python3 tools/sitl/fw_offboard_probe.py --out /tmp/fw_probe --dur 60 \
        --phases pos_vel,vel_only,pos_only
```

각 페이즈 종료 시 `dE / dN / 평균지상코스` 를 로그로 출력한다. `pos_vel` 만 90°(정동)가
나오면 재현 성공이다. 궤적 CSV 는 `--out` 경로에 남는다.

**함정 (실측):**
- `CommandTOL` 의 `latitude/longitude` 는 **NaN** 이어야 한다. `0.0` 은 널 아일랜드로
  해석돼 이륙이 취소된다(저장소 작업 H, 커밋 `000f478` 과 동일 함정에 프로브도 한 번 걸렸다).
- `MIS_TAKEOFF_ALT` 는 float 파라미터라 `ros2 param set … 50`(정수) 되읽기 검증이
  실패해 무한 재시도에 빠진다. `CommandTOL` 로 고도를 직접 주므로 설정 자체가 불필요하다.
- `wsl.exe -d Ubuntu-22.04 -- bash -c '…'` 는 `$변수`·루프가 호스트 셸에서 먼저 전개돼
  깨진다. 로직은 스크립트 파일로 두고 stdin 파이프로 넣을 것.
- 프로브 1회 비행의 ulog 는 **183 MB** 다. 분석 후 삭제했다.
