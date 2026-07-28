---
doc_type: fc_handoff
project: suridoksuri-1
track: 🎯 fc-정밀착륙 (F2)
scope: F2 정본 — 계약·구현 상태·재개 순서. (원래는 vision → FC 인수인계였고 그 내용도 그대로 남아 있다)
status: ▶ 재개 가능 — 플래시 선행조건 해소(2026-07-29). 단 실비행은 자기계·배터리게이트 2건 미해결로 불가
created: 2026-07-28
last_updated: 2026-07-29
---

# FC 정밀착륙(F2) — 구현 상태 및 재개 인수인계

> **이 문서 하나로 F2를 완주할 수 있게 썼다.** FC 세션은 `docs/vision_plan.md`·
> `docs/vision_orchestration_handoff.md`를 읽지 않아도 된다(도메인 컨텍스트 격리).
> 더 깊은 배경이 필요하면 `docs/vision_fc_interface.md`(852줄, 정찰 사실확정)의 **필요한 절만** 열어라.
>
> **§0~§0-2가 현재 상태이고, §1 이후는 원래의 계약·함정 정본이다** — 계약 내용은 구현 후에도
> 바뀌지 않았으므로 그대로 유효하다. 재개할 때는 **§0-2(재개 순서)부터** 읽으면 된다.

---

## 0. 🔴 현재 상태 — 한 줄 요약

**F2는 구현·배포까지 끝났고 검증이 미완이다. 보류는 2026-07-29 해제됐다**(선행이던 PX4 패치
실기체 플래시 완료 — §0-1). **다만 실비행 검증은 아직 못 한다** — 자기계·배터리게이트 2건이
미해결이다(§0-1b). `vision_landing:=false`(기본)라 **지금 기체에 얹혀 있어도 무해**하다.

```
[호스트 picam-venv]              [fc 컨테이너]                        [offboard_node]
 main.py --target-sink ─JSONL─▶ shim_node ─┬─▶ /vision/landing_setpoint  ✅ 받는 코드 있음
                                           ├─▶ /vision/target_status     ✅ 받는 코드 있음
                                           ├─▶ /vision/target_pose       ← (선택) 안 쓴다
                                           └─▶ /mavros/landing_target/raw  ⏸ 기본 꺼짐(D-c 유지)
                                           ▲
                                 /mavros/local_position/pose ✅ 이미 구독 중
```

### 0-1. ✅ 보류는 해제됐다 (2026-07-29) — 그러나 실비행 선행조건이 **바뀌었다**

> **✅ 플래시 완료.** 아래 "왜 보류인가"의 선행조건이었던 **F-17/F-4 패치 실기체 플래시가
> 2026-07-28~29 끝났다.** 1차 플래시에서 CRSF RC 드라이버 누락으로 조종기가 완전히 두절되는
> 사고가 있었고, `CONFIG_DRIVERS_RC_CRSF_RC=y` 를 넣어 재빌드한 본으로 해소했다
> (`crsf_rc status` 수신 실측, 원본 대비 파라미터 MISSING 0건). 패치 자체도 실기체 ulog 로
> 정상작동이 확인됐다(`psp_triplet` 4건 전부 `course=NaN`, 패치 전이라면 `0.0f`).
> 전문: `docs/px4_v6c_patch_build.md` **§11**(특히 §11-6).
>
> **🔴 대신 실비행을 막는 새 선행조건 2건이 생겼다 — 아래 §0-1b.**

**(이력 — 원래의 보류 사유) 천이 문제 해결(PX4 패치 실기체 플래시)이 먼저다.** 두 가지 이유이고 둘 다 강하다:

1. **F2는 `HOLD` 이후에 발동한다.** 거기 도달하려면 FW 순항 → 역천이(`TRANSITION_MC`) → `HOLD`를
   거쳐야 한다. **천이가 안 고쳐지면 F2 코드가 실행되는 지점까지 기체가 가지 못한다** — 지금
   F2를 아무리 다듬어도 실비행으로 확인할 방법이 없다.
2. **`sitl_vtol_remediation_plan.md` §4-1 4번이 "첫 실비행에 미검증 변수를 둘로 만들지 말라"를
   명문화**하고 있다. 천이 패치와 정밀착륙을 같은 비행에서 처음 켜는 것 자체가 금지다.

선행 작업은 **🛩 sitl-vtol 트랙**이다(`docs/session_status.md`, 절차는 `docs/px4_v6c_patch_build.md`):
①실기체 플래시(USB·QGC 필요, 사용자 승인 후 별도 세션) → ②비-정북 레그 실비행(R7).

> 🚨 **플래시 시 주의(그 트랙이 기록한 것):** 패치 펌웨어가 **자기신고를 하지 않는다** —
> `git_identity`가 순정과 완전히 동일(`v1.18.0-alpha1-592-gc890d9db0a`)해서 **구별 수단은
> sha256(`f1c16e2b…`)뿐**이다. 플래시 직전 반드시 재확인할 것.

플래시 전후 자료는 `logs/2026-07-28_px4_flash/`에 전부 커밋돼 있다 — 플래시 **전** 백업 4종
(`px4_params_2026-07-28_pre-flash.{csv,json,params,txt}`) · `ver_all_*_pre-flash.txt` ·
**플래시 후** 덤프 2종(`*_post-flash.json` = 사고본 / `*_final-crsf.json` = **현재 탑재본**) ·
`POST_FLASH_CHECKLIST.md`(대조 절차, 이번 사고를 실제로 잡아낸 절차다) · `CRITICAL_PARAMS.md`.

### 0-1b. 🔴 새 선행조건 — 실비행 검증을 막는 미해결 2건 (2026-07-29)

둘 다 **F2 코드 문제가 아니라 기체·상태기계 결함**이고, 해소 전엔 `vision_landing:=true` 실비행을
해도 결과를 신뢰할 수 없다.

| # | 결함 | 근거 | 통과 기준 |
|---|---|---|---|
| 1 | **자기계 헤딩 의존 오차** — 재캘리브레이션을 했는데 지표가 오히려 악화(`test_ratio` 평균 1.97→2.62, `cs_mag_fault` ON 0%→92.7%). 원인은 전류 간섭이 아니라 **캘리 방향 커버리지**(−45~0°에서 0.06, −180~−135°에서 4.18) | `5d55b3f` | **기수를 남쪽에 두고 `test_ratio<1`** 확인. 이게 흔들리면 `_step_transition_fw` Phase 2 "헤딩 정렬 완료" 판정 근거 자체가 흔들린다 |
| 2 | **배터리 게이트 부재** — flight02 는 `Emergency battery level` 이 t=8.64s(고도 약 4m)부터 떠 있는 채로 50m 까지 올라가 천이를 시도했다. `offboard_node` 상태기계에 게이트가 없다 | `f8e951f` | 게이트 도입. **F2 는 임무 끝단**(탐색 최악 171s)이라 이 결함의 직격 대상이다 |

### 0-2. ▶ 재개 순서 (여기서 시작하라)

| # | 할 일 | 왜 이 순서인가 |
|---|---|---|
| 1 | ~~🛩 sitl-vtol 플래시~~ **✅ 완료(2026-07-29)** + R7 실비행 · **자기계/배터리게이트 2건(§0-1b)** | 실비행 확인의 전제. **이게 끝나기 전엔 4번을 해도 확인이 안 된다** |
| 2 | **SITL 장애주입** (§6-1) | 하드웨어 없이 지금도 가능한 유일한 검증. 실비행 전에 끝내둘 것 |
| 3 | `vision_landing:=true` SITL 완주 | 탐색→래치→정렬→인계 전 구간 |
| 4 | 실기체 `vision_landing:=true` 첫 비행 | ①이 끝난 뒤에만 |

**2번은 1번을 기다릴 필요가 없다** — 천이와 무관하게 노트북 SITL에서 돌릴 수 있고, 실비행
기회가 왔을 때 바로 쓰려면 미리 해두는 편이 낫다(메모리 `project_fc_sitl_laptop_env`).

### 0-3. ✅ 무엇이 구현됐나 (커밋 `8cb0861`)

```
HOLD ─┬─(vision_landing=false, 기본)─▶ LANDING              ← 종전과 100% 동일
      └─(true)─▶ VISION_SEARCH ─(래치)─▶ PRECISION_LAND ─▶ LANDING
```

- **`VISION_SEARCH`** — 탐색고도 정렬 → 제자리 확인(3s) → 아르키메데스 나선 확대.
  실패 시 15m/18m로 **재탐색 1회**, 그래도 실패면 GPS 착륙 폴백(무한 재탐색 금지 —
  성공 판정이 *"재시도 없이"* 를 포함한다).
  싼 것부터 시도하는 구조다: 25m 풋프린트가 33.4×18.8m라 **WP 오차 ±9.4m 이내면 나선을 한
  바퀴도 돌지 않고 제자리에서 잡힌다.**
- **`PRECISION_LAND`** — 수평은 vision, **수직은 FC가 스케줄**한다. 정렬 오차가
  `vision_align_tol` 안에 든 틱에만 목표고도를 내린다.
- 신규 모듈 3종: `fc_bridge/execution/search_pattern.py`(나선 기하·풋프린트·검출고도 상한) ·
  `fc_bridge/execution/precision_land.py`(래치·하강게이트·인계판정, 순수 함수) ·
  `fc_ros/adapters/vision_target_bridge.py`(두 토픽 구독·계약 파싱).
- **테스트:** `fc_bridge` 221 + `fc_ros` 182 = **403 passed**. 파괴검증 23종.
- **배포 검증 완료:** 소스↔install md5 일치 · import 통과 · **QoS BEST_EFFORT 실기체 확인**
  (유닛테스트로는 원리적으로 못 잡는 항목이라 실기체에서 직접 봐야 했다).

### 0-4. 🔴 무엇이 안 됐나 (정직하게)

| 미검증 항목 | 상태 |
|---|---|
| **SITL 장애주입** (§6-1) | **전무.** vision SIGKILL / veto / setpoint 침묵 어느 것도 실증 안 함 |
| **`vision_landing:=true` 비행** | **0건** (SITL·실기체 둘 다) |
| 실촬영 검출 | **전무** — vision 골든셋이 전부 합성이다 |
| 라이다 AGL | 여전히 **미배선**(D-d). 지금은 이륙지점 지면 기준 근사 = **평탄지 가정** |

### 0-5. 구현 중 실측으로 잡은 결함 3건 (재조사 불필요)

1. **나선 선회속도를 최대반경 기준으로 전 구간에 쓰면 안 된다.** r=5m에서 tilt 27° →
   25m 고도에서 시선이 12.75m 밀리는데 이건 **링 간격(12.2m)과 같은 자릿수**라 커버리지가
   통째로 틀어진다(r=2m면 51.9°로 비행 자체가 위험). **매 틱 그 순간의 반경으로 속도를 다시
   구하고, 나선을 링 간격의 절반에서 시작**하도록 고쳤다 → 최대 tilt 6.0°, 오프셋 2.62m.
2. **`vision_search_timeout` 90s는 1회차를 완주 직전에 자른다** — 나선 89s + 정렬·dwell 8s
   = 97s다. 120s로 역산 재설정했다. **반경·간격·속도를 바꾸면 이 값도 같이 재야 한다.**
3. 🔴 **`DiagnosticStatus.level`이 1바이트 `bytes`로 온다**(`vision/ros/shim_node.py:221`이
   `st.level = bytes([...])`로 넣는다 — msg상 타입이 `byte`라 rclpy가 요구하는 형식이다).
   **아래 §2-2가 "level 값: OK=0, WARN=1, ERROR=2"라고만 적어둔 탓에** 그대로 믿고
   `status.level == 1`로 짜면 `b'\x01' == 1`이 항상 False라 **테스트는 전건 green인 채
   실기체에서 거부권(veto)만 조용히 사라진다.** `level_to_int()`로 흡수하고 회귀로 고정했다.

### 0-6. 탐색 시간 예산 (실측)

| 구간 | 시간 |
|---|---|
| 1회차 (25m / 반경 30m) | 정렬 ~5s + dwell 3s + 나선 **89s** = **97s** |
| 재탐색 (15m / 반경 18m) | 정렬 + dwell + 나선 66s = **74s** |
| **최악 합계** | **171s (2.9분)** |

임무 끝단이라 배터리 여유가 관건이다. 줄이려면 `vision_search_radius`를 먼저 깎아라 —
실효 커버반경이 `반경 + 풋프린트단폭/2`라 30m 설정이면 실제로는 **39.4m까지 훑는다**(과잉일 수 있다).

---

## 1. 붙는 자리와 재사용할 패턴

> ✅ **이 절은 이미 이행됐다**(커밋 `8cb0861`). 아래 내용은 **왜 그렇게 붙였는지**의 근거로
> 남겨 둔다 — 구조를 바꾸려 할 때 여기 적힌 사고 이력을 먼저 읽어라.

### 1-1. 위치

`fc_ros/fc_ros/nodes/offboard_node.py`의 **`_step_hold`와 `_step_landing` 사이**에 새 서브상태를
넣는다. `_State` enum에 항목 추가.

**실제 구현:** `_State.VISION_SEARCH` / `_State.PRECISION_LAND` 2종을 추가했고, 진입점은
`_step_hold` 말미가 아니라 **`_exit_hold()`라는 단일 분기 함수**다 — HOLD 종료 경로가 두 개
(안정 도달 / 타임아웃)라 각각에 분기를 복제하면 한쪽만 고치는 사고가 나기 때문이다.
`vision_landing=false`면 `_exit_hold`가 곧장 `LANDING`으로 보내 **종전과 완전히 동일**하다.

현재 `HOLD → LANDING` 전이는 두 경로다(`_step_hold` 말미):
- `wp1_land_ready(dist, speed, ...)`가 `_HOLD_STABLE_REQ` 틱 연속 참 → `LANDING`
- `_hold_elapsed > _hold_timeout` → 강제 `LANDING`

정밀착륙은 **이 사이에 끼어들어** "WP1 상공 도달·안정" 다음, "AUTO.LAND" 전에 비전 유도로 정렬한다.

### 1-2. 🔴 재사용해야 할 것 — 새 폴백 경로를 만들지 마라

`_step_hold`가 이미 확립한 규약을 그대로 따른다:

```python
# 램프 시작점은 직전 setpoint 가 아니라 **기체 현재 위치**다.
#   근거: "OFFBOARD 가 이어받는 setpoint 는 항상 실제 위치와 일치시킨다"
#         (2026-07-20 제어상실 사고 대응). _step_following / STREAMING MC 분기와 같은 규약.
if self._pl_ramp is None:
    self._pl_ramp = np.array(state.pos_ned, dtype=float)
self._pl_ramp = slew_setpoint(self._pl_ramp, raw_target, self._v_approach * self._dt)
self._publish_pos_setpoint(self._pl_ramp, state.yaw)
```

- `slew_setpoint(current, target, max_step)` — `fc_bridge/execution/state_logic.py:458`.
  `current`/`target` 둘 다 **NED `[N, E, h_up]`**. 남은 거리가 `max_step` 이하면 target을 그대로
  반환해 정확히 안착한다.
- `_publish_pos_setpoint(pos_ned, yaw_ned)`(`:1007`) — `yaw_ned`는 **필수 인자**다. 생략하면
  ROS2 기본 단위쿼터니언(ENU yaw=0=동쪽)이 나가 **2026-07-21 flight04 yaw 스핀 사고**가 재발한다.
  호출부는 항상 현재 실제 헤딩(`state.yaw`)을 넘긴다.
- 안정 카운터(`_hold_stable_ticks` 패턴)와 타임아웃(`_hold_timeout` 패턴)을 그대로 베낀다.

### 1-3. `_RANGE_GUARDED_STATES`(`:132`) 결정 필요

거리 상한 감시를 거는 상태 집합이다. 현재 `LANDING`은 **일부러 빠져 있다** —
*"이미 내려오는 중 — 여기서 OVERRIDE를 걸면 착륙을 방해한다"*. 정밀착륙 상태를 넣을지는
**FC 판단**이다. 넣으면 유도 중 이탈에 OVERRIDE가 걸리고, 안 넣으면 안 걸린다.

---

## 2. 소비할 계약 (전부 실기체 검증됨)

### 2-1. `/vision/landing_setpoint` — 목표 좌표 (핵심)

| 항목 | 값 |
|---|---|
| 타입 | `geometry_msgs/PoseStamped` |
| `header.frame_id` | `"map"` |
| **좌표계** | 🔴 **ENU** (`x=동, y=북, z=상`) — mavros 전역 관례 |
| `pose.position` | **목표 착륙점의 절대 위치** (기체 상대량이 아니다) |
| `pose.orientation` | 🔴 **현재 기수방위**(단위 쿼터니언 아님). "모름"이 아니라 "이 헤딩 유지"라는 뜻 |
| `header.stamp` | wall clock(`CLOCK_REALTIME`). ROS 기본 클록과 같은 기준 |
| QoS | **BEST_EFFORT / KEEP_LAST / depth=1** (`telemetry_node::_MAVROS_QOS`와 같은 계열) |

**`_publish_pos_setpoint`가 요구하는 `pos_ned = [N, E, h_up]`로의 변환은 한 줄이다:**

```python
pos_ned = np.array([msg.pose.position.y,    # N ← y_enu
                    msg.pose.position.x,    # E ← x_enu
                    msg.pose.position.z])   # h_up ← z_enu (위 양수 그대로)
```
`fc_bridge`의 `vehicle_state_bridge.update_from_pose`가 쓰는 `[p.y, p.x, p.z]`와 **같은 관용구**다.

> 🔴 **`pos_ned`의 3번째는 `h_up`(위 양수)이지 `D`가 아니다.** 같은 저장소의 `vel_ned`는
> `[vN, vE, vD]`로 **3번째가 아래 양수**다 — **같은 `_ned` 접미사가 위치와 속도에서 반대 부호
> 규약**을 쓴다. 이걸 혼동한 사고 이력이 있다.

**편의:** `/vision/target_status`의 `vision/setpoint` 항목에 `setpoint_position_ned_n_e_hup`
KeyValue가 이미 변환된 삼중항으로 실려 나온다(진단용). 다만 **권위 있는 값은 `PoseStamped`**이고
그 KeyValue는 에코다 — 둘 중 하나만 골라 쓰고 섞지 마라.

#### 발행되지 않는 조건 (침묵의 의미)

| 상황 | landing_setpoint | 뜻 |
|---|---|---|
| attitude 미수신 | **침묵** | shim이 `/mavros/local_position/pose`를 아직 못 받음 |
| attitude stale(>0.25s) | **침묵** | 자세가 오래됨 — 변환을 신뢰할 수 없음 |
| 타겟 안 보임(`valid=false`) | **침묵** | 실을 좌표가 없음 |
| 생산자(vision) 사망 | **침묵** | — |

🔴 **0이나 추측값을 채우지 않는다.** setpoint 자리의 0은 "기체 바로 아래로 가라"는 뜻이라
가장 위험한 거짓말이다. **침묵을 반드시 처리하라** — 마지막 값을 무한정 붙들면 안 된다.

### 2-2. `/vision/target_status` — 상태·거부권·진단

타입 `diagnostic_msgs/DiagnosticArray`. **`DiagnosticStatus`에는 header가 없어서** 배열로 싣는다
(같은 `header.stamp`로 묶임). `status[].name`:

| name | 의미 | level |
|---|---|---|
| `vision/target` | 검출 유효성 | OK / WARN(안 보임) |
| `vision/state` | 🔴 **상태머신 상태 = 거부권** | **WARN이면 veto** |
| `vision/setpoint` | setpoint 발행 여부·사유 | OK / WARN |
| `vision/link` | 생산자 생존 | **ERROR = 생산자 사망**(1Hz 하트비트) |

**level 값:** `OK=0`, `WARN=1`, `ERROR=2`.

> 🔴 **하지만 그 값은 `int`가 아니라 1바이트 `bytes`로 온다.** msg상 타입이 `byte`라
> rclpy가 `bytes`를 요구하고, 발행 측도 `st.level = bytes([int(...)])`로 넣는다
> (`vision/ros/shim_node.py:221`). **`status.level == 1`로 비교하면 `b'\x01' == 1`이 항상
> False라 거부권이 조용히 사라진다** — 테스트는 전건 green이고 실기체에서만 틀린다.
> `fc_ros/adapters/vision_target_bridge.py::level_to_int()`가 양쪽을 흡수하므로 **그걸 써라.**
> (이 문단은 2026-07-28 F2 구현 중 발견돼 추가됐다. 원래 이 절은 위 한 줄뿐이었고, 그게
> 정확히 함정의 원인이었다.)

#### 🔴 거부권 계약

```
state ∈ {HOLD, ABORT_ASCEND}  →  veto (vision/state 가 WARN)
```
`state` 전체 집합: `ACQUIRE` / `CENTER_DESCEND` / `LOCK` / `PRECISION_SERVO` / `TERMINAL` /
`HOLD` / `ABORT_ASCEND`.

**`command_hint`는 advisory다 — 명령으로 소비하지 마라.** 와이어가 `command_hint` +
`command_is_advisory:true` + 타입명 `state_hint` **세 겹으로** "명령 아님"을 형식에 박아 뒀다.
거부권은 오직 `state`다. (`command` 문자열 집합은 참고용:
`scan`/`center`/`descend`/`hold`/`ascend`/`land`.)

#### 🔴 `command_hint == "land"` 의 뜻

TERMINAL에서 블라인드가 2초를 넘으면 vision이 `land`를 낸다. **"횡 유도를 놓고 수직 강하하라"**
= AUTO.LAND 인계 신호다. 근거: 2초째 못 보는 추정으로 횡방향을 계속 물고 늘어지는 것이 오차를
**키우는** 모드이고(접지 순간 횡속도 → 초록구역 라이즈드 가장자리 전복), `closed_loop_floor_agl_m`
= 3.0m가 vision의 `terminal_agl_m`과 **같은 숫자**라 계약상 이미 "3m부터 AUTO.LAND 인계"다.

advisory이므로 강제는 아니지만, **이걸 무시하고 계속 횡 유도하면 설계 의도와 반대로 간다.**

### 2-3. 페일세이프 3분법 (실기체 실증됨)

| 상황 | `landing_setpoint` | `target_status` |
|---|---|---|
| 정상 | 발행 | 전부 OK |
| 안 보임 | 침묵 | `vision/target` **WARN** + 사유 |
| **생산자 사망** | 침묵 | `vision/link` **ERROR** 계속(1Hz) |
| **shim 사망** | 침묵 | **침묵** |

**→ setpoint 침묵만으로는 "안 보임"과 "죽음"을 못 가른다. 반드시 status를 같이 봐라.**

### 2-4. `/vision/target_pose` (선택)

`geometry_msgs/PoseWithCovarianceStamped`, `frame_id="base_link"`, **body FLU 상대 pose**.
자체 서보 컨트롤러를 짜고 싶으면 이쪽을 쓴다. `landing_setpoint`를 쓰면 안 봐도 된다.
`orientation`은 **단위 쿼터니언**이다(원본 카메라 광학 자세는 KeyValue로만 나간다 — 프레임
거짓말을 피하려고 일부러 뺐다).

---

## 3. ✅ FC 결정사항 — 전부 처리됨 (2026-07-28, 커밋 `8cb0861`)

| # | 항목 | **내린 결정** |
|---|---|---|
| **D-a** | `HOLD` 종결 시한 | ✅ **`vision_veto_timeout`(기본 10s) 신설.** vision이 `HOLD`/`ABORT_ASCEND`(=veto)로 빠진 채 이 시한을 넘기면 GPS 착륙으로 폴백한다. vision이 *일부러* 안 만든 값이고(`FailsafeContract`가 "FC가 자기 제어틱에서 재는 값"으로 못박음), **없으면 무한 호버링**이었다 |
| **D-b** | `_RANGE_GUARDED_STATES` 포함 여부 | ✅ **`VISION_SEARCH`는 포함, `PRECISION_LAND`는 제외.** 탐색은 의도적으로 WP1 바깥으로 나가지만 상한 300m 대비 탐색 반경(30m)은 자릿수가 다르고, "점점 커지는 원"은 스스로 멈출 조건이 없어(`_step_transition_mc`의 "도망가는 캐럿"과 같은 실패 모드) 거리 상한이 **유일한 기하학적 제동장치**다. `PRECISION_LAND`는 `LANDING`과 같은 논리로 제외(내려오는 중 OVERRIDE는 착륙 방해) |
| **D-c** | `listen_lt: true` 네이티브 피벗 | ✅ **열지 않는다.** `/vision/landing_setpoint` 자체 서보 경로로 충분하고, 열면 §4 함정 1(`frame` off-by-one)을 떠안는다. `px4_config.yaml:214`는 `false` 그대로 |
| **D-d** | AGL 소스 | ⚠️ **미해결 — 이륙지점 지면 기준 근사로 진행 중.** `offboard_node._agl()`이 `pos_ned[2] − _takeoff_ground_h`를 쓴다(CLIMBING의 AGL 판정과 같은 기준). **평탄지 가정**이라 착륙지 지면 높이가 이륙지점과 다르면 그만큼 틀린다. 라이다 배선은 **여전히 없다**(§5 그대로 유효) |

---

## 4. 🔴 함정 (전부 실측 확정 — 재조사 금지)

1. **`mavros_msgs/LandingTarget`의 `frame` 상수가 실제 `MAV_FRAME`과 off-by-one.**
   msg는 `LOCAL_NED=2`인데 실제 enum(`common.hpp:154`)은 `LOCAL_NED=1`, `MISSION=2`.
   **상수 이름을 믿고 짜면 MAVLink가 `MISSION`으로 읽어 조용히 `position_valid=false`로 떨어진다.**
   정수 리터럴 `1`(LOCAL_NED) / `12`(BODY_FRD)를 써라. (D-c를 열 때만 해당.)
   ⚠️ 하필 `LandingTarget.msg`에 `GLOBAL_TERRAIN_ALT_INT = 12`가 있어 "상수를 썼는데 값은 맞는"
   경로가 존재한다 — 값이 맞아도 이름이 거짓말이다.
2. **`frame=12`에 넣는 좌표는 FRD가 아니라 FLU다.** mavros `landtarget_cb`가
   `transform_frame_baselink_aircraft`로 **플러그인이 FLU→FRD를 직접 한다.** FRD를 넣으면
   변환이 두 번 걸려 y·z 부호가 뒤집힌다.
3. **shim 발행 QoS는 BEST_EFFORT다.** RELIABLE로 구독하면 **조용히 아무것도 안 받는다.**
   `ros2 topic echo`의 기본값도 RELIABLE이라 디버깅 때 "발행이 안 된다"고 오판하기 쉽다 →
   `--qos-reliability best_effort --qos-durability volatile`.
4. **ros2cli 데몬이 죽으면 `topic echo`/`node list`가 `xmlrpc Fault: !rclpy.ok()`로 전멸한다.**
   토픽은 멀쩡한데 CLI만 못 본다. `ros2 daemon stop && ros2 daemon start`, 또는 rclpy로 직접 구독.
5. **컨테이너 `/tmp`는 호스트 `/tmp`와 별개다.** 마운트는 `/home/suri/drone_ws → /drone_ws`
   **하나뿐**이다.
6. **`fc` 컨테이너는 `/dev/ttyACM0`을 필수 device로 요구한다** — 픽스호크가 안 올라오면
   `docker start fc` 자체가 실패한다(`error gathering device information`).
7. **`CLOCK_MONOTONIC`은 호스트↔컨테이너 완전 동일**(time namespace inode 동일, 실측).
   `clock_offset_ns` 환산이 필요 없다. 종단간 실측 age 0.67ms.

### shim 실행 (F2 개발 중 띄워둘 것)

```bash
docker exec fc bash -lc '
  source /opt/ros/humble/setup.bash
  export PYTHONPATH=/drone_ws/src/suridoksuri:$PYTHONPATH
  python3 -m vision.ros.shim_node
'
```
🔴 **`PYTHONPATH`는 반드시 이어붙인다(`:$PYTHONPATH`).** 덮어쓰면 `setup.bash`가 넣은 ROS
경로가 날아가 `import rclpy`가 즉사한다(실기체 재현 완료).

⚠️ shim은 **ROS 패키지가 아니라 `vision/ros/`에 있다** — `colcon build` 대상이 아니고
`ros2 run`/launch로 못 띄운다. `phase2.launch.py`에 넣고 싶으면 `fc_ros/`로 옮기거나 얇은
래퍼가 필요하다(**그때 옮겨도 `shim_core.py`는 그대로 재사용된다**). 자동 기동 방법은 미정.

---

## 5. 🔴 AGL 소스가 배선돼 있지 않다 (D-d)

vision 설계는 여러 곳에서 **"라이다 AGL"**을 가정하는데, **`fc_ros`에 그 배선이 없다.**
확인 결과 `offboard_node`가 구독하는 고도 소스는 `/mavros/altitude`(`:450`)뿐이고 `.amsl`만
쓴다. `telemetry_node`의 구독 목록에도 거리계 토픽이 없다.

- `/mavros/altitude`의 `bottom_clearance` 필드가 거리계 기반인데, **라이다가 실제로 붙어 있고
  PX4가 그 필드를 채우는지 아무도 확인한 적이 없다.**
- 이건 F2 자체의 블로커는 아니다(정밀착륙은 vision이 준 절대 좌표로 간다). 다만
  **vision의 사전정보 스코어링이 AGL을 요구**하므로, 그걸 살리려면 이 소스가 확정돼야 한다(§7).

---

## 6. 검증 방법

### 6-1. SITL 장애주입 (필수) — 🔴 **미실시. 재개 시 여기가 2번 항목이다**

vision 프로세스 **SIGKILL** / 소켓 끊김 / attitude stale / `valid=false` 각각에 대해 FC가
의도한 상태로 전이하는지 실증하라. **천이 플래시를 기다릴 필요가 없다** — 노트북 SITL에서
지금도 돌릴 수 있다(메모리 `project_fc_sitl_laptop_env`).

| # | 주입 | 기대 동작 | 구현상 어디서 걸리나 |
|---|---|---|---|
| 1 | 생산자 SIGKILL | `vision/link` ERROR → 탐색 중단·GPS 착륙 | `_step_vision_search` ②번 분기 |
| 2 | `state=HOLD`(veto) 지속 | 하강 정지 → 10s 후 GPS 착륙 | `descend_allowed(veto=True)` → `vision_veto_timeout` |
| 3 | setpoint 침묵 | **마지막 값을 붙들지 않는다** — `pos_ned`가 `None`으로 떨어지고 고도가 고정된다 | `VisionTargetBridge.snapshot()` stale 만료 + `descend_allowed(guided=False)` |
| 4 | shim 자체 사망 | status까지 끊김 → `status_age_s` 증가로 1번과 **구분되어야** 한다 | 페일세이프 3분법(§2-3) |
| 5 | 단발 오탐 1프레임 | 탐색이 **중단되지 않는다** | `latch_candidate` 연속 3틱 + 산포 3m |
| 6 | 나선 완주 | 15m 재탐색 1회 후 GPS 착륙 (무한 반복 없음) | `search_pass_next` |

> ⚠️ **`ros2 topic info -v`로 QoS를 반드시 눈으로 확인하라.** BEST_EFFORT↔RELIABLE 불일치는
> **유닛테스트로 원리적으로 못 잡는다**(단일 프로세스로 재현 불가). 배포 시 컨테이너에서
> `_vision_qos()`가 BEST_EFFORT/depth=1/KEEP_LAST임은 확인했지만, **구독자-발행자 짝이 실제로
> 붙는지는 토픽이 실제로 흐를 때만 보인다.**

### 6-2. 🔴 실기체 배포 규율 (이 저장소의 반복 사고)

**FC 코드를 고쳤으면 즉시 실기체까지 반영한다.** `git push` → RPi `git pull` → 컨테이너 안에서
`colcon build --packages-select fc_ros` → 검증. 유일한 예외는 사용자가 "현재 비행중"이라고
말한 경우다. 절차는 `docs/rpi_deploy.md`.

⚠️ **stale colcon build가 실비행 8건의 근본원인이었다**(`4dc30f9`). 커밋만 하고 멈추면 기체는
옛날 코드로 난다.

### 6-3. 🔴 기본 off 파라미터로 배포할 것

실기체 **FW+OFFBOARD 실적이 0건**이라 `sitl_vtol_remediation_plan.md` §4-1 4번이
*"첫 실비행에 미검증 변수를 둘로 만들지 말라"*를 명문화하고 있다. 정밀착륙은 파라미터로
**기본 비활성**으로 넣고, 켜는 것을 별도 결정으로 둬라.

---

## 7. 잠정값·미검증 (억지로 신뢰하지 마라)

| 항목 | 상태 |
|---|---|
| **기체 최외곽 반경 `R`** | 🔴 **미측정**(사용자 확인은 "0.5m 초과"뿐, 2026-07-29 실측 예정). vision 착륙점이 이 값에서 도출되므로 **착륙점 좌표 자체가 잠정값**이다 |
| **EKF 드리프트 상쇄 성능** | 🔴 미측정. shim 변환은 매 레코드 최신 pose로 재계산해 구조적으로 상쇄되게 돼 있고 그건 실기체에서 확인했지만, mavros 미가동이라 **pose를 합성 주입**해 검증했다 |
| 카메라 캘리브레이션 | `nominal.yaml`(HFOV 75° 가정 근사). **`not_for_closed_loop_30cm: true`가 100% 붙어 나간다** — 해석은 "폐루프 금지"가 아니라 **"최종 커밋 금지"**(3.0m까지 정렬 후 AUTO.LAND 인계) |
| 카메라 마운트 요각 ψ_m | 🔴 미측정(기본 0). **틀리면 착륙 오프셋의 *방향 자체*가 틀린다** — 수정할수록 멀어지는 증상이면 이걸 의심하라. **첫 폐루프 시험의 1번 체크항목** |
| 발행 주파수 | **실측 4.4Hz**(median 0.2207s / p95 0.310s). 🔴 **10Hz를 가정하지 마라.** `stale_timeout`을 0.5s 근처로 잡으면 정상 지터와 여유가 1.6배뿐이라 헛경보가 난다 |
| 실촬영 검증 | 🔴 **전무**. 검출 골든셋이 전부 합성이다 |

---

## 8. 파라미터 레퍼런스 (2026-07-28 신설분)

전부 `offboard_node`의 ROS2 파라미터다. **테스트 임시값은 yaml을 고치지 말고 launch 인자로만
준다**(이 저장소 규율): `phase2.launch.py vision_landing:=true vision_search_alt:=20.0`.

| 파라미터 | 기본 | 의미 / 근거 |
|---|---|---|
| `vision_landing` | **`false`** | 🔴 **마스터 스위치.** false면 구독조차 만들지 않는다 = 종전과 100% 동일 |
| `vision_search_alt` | 25.0 | 탐색고도(AGL). 검출 상한 33.57m 대비 8.6m 여유. **검출률이 나쁘면 20m로 내리는 것이 첫 카드**(25m 명목 픽셀면적이 `min_area`의 1.80배뿐, 20m는 2.82배) |
| `vision_search_radius` | 30.0 | 최대 탐색반경. 실효 커버는 `반경 + 풋프린트단폭/2` = **39.4m** |
| `vision_search_spacing` | 0.0 | 링 간격. **0이면 풋프린트에서 자동 산출**(겹침 35% → 25m에서 12.2m) |
| `vision_search_speed` | 0.0 | 선회속도 **상한**. 0이면 `v_approach`. 실제 속도는 매 틱 반경별로 재산출된다 |
| `vision_search_dwell` | 3.0 | 탐색고도 도달 후 제자리 확인(s) |
| `vision_search_timeout` | **120.0** | 1회 탐색 상한. 🔴 실측 역산값 — 1회차가 97s다. **반경·간격·속도를 바꾸면 같이 재라** |
| `vision_retry_alt` / `_radius` | 15.0 / 18.0 | 재탐색 회차 |
| `vision_latch_frames` / `_spread_m` | 3 / 3.0 | 래치 조건(연속 **vision 프레임** + 수평 산포). 단발 오탐 차단. 🔴 **단위는 제어틱이 아니다** — 2026-07-29 개명, 아래 §8-1 |
| `vision_stale_timeout` | 1.0 | 🔴 실측 발행 4.4Hz 기준. **0.5s로 잡으면 헛경보** |
| `vision_link_timeout` | 3.0 | `link_dead` 유효기간(하트비트 1Hz × 3). 일회성 ERROR가 영구 래치되는 것을 막는다 |
| `vision_veto_timeout` | 10.0 | **D-a.** veto 지속 시 GPS 착륙 폴백까지의 시한 |
| `vision_align_tol` | 1.0 | 이 안에 들어와야 하강한다 — **수평/수직 분리의 핵심** |
| `vision_descend_speed` | 0.8 | 정렬 후 하강률(m/s) |
| `vision_land_handoff_agl` | 3.0 | AUTO.LAND 인계 고도. 🔴 vision의 `terminal_agl_m`과 **같은 숫자여야 한다** |
| `precision_land_timeout` | 60.0 | 정렬이 영영 안 설 때의 상한 |

> ✅ **2026-07-29 노출 완료.** 위 18개 전부가 `fc_ros_params.yaml` + `phase2.launch.py`
> (`DeclareLaunchArgument` + `_make_nodes` overrides)에 들어갔다. 3중 일치는
> `fc_ros/test/test_params.py`의 `test_f2_*`가 고정한다.
> **기본값은 `vision_landing: false`** — 종전 경로가 기본이라는 계약 그대로다.
>
> 🔴 **그 전까지는 실기체에서 F2를 켤 방법이 아예 없었다.** launch 인자 위생검사가
> 미선언 인자에 `RuntimeError`를 던지므로 `phase2.launch.py vision_landing:=true`는
> **launch 단계에서 실패**했다(값이 무시되는 게 아니라 기동 자체가 안 됐다).

### 8-1. 🔴 2026-07-29 수정 3건 — 재개 전에 반드시 읽어라

`8cb0861`(구현)과 `6e3ea0c`(검증) 사이에서 드러난 것들이다.

| # | 무엇이 틀렸나 | 지금 |
|---|---|---|
| 1 | **`_enter_vision_search()`가 `self._sm`을 세팅하지 않았다.** `_exit_hold`가 불러도 상태가 `HOLD`로 남아 다음 틱에 `_step_hold`가 다시 안정조건을 만족 → `_exit_hold` 재호출의 **10Hz 무한 루프**. SITL 실측 진입 로그 1687회, `stable=2205/10`, 450.8s 타임아웃. **실기체였다면 WP1 상공에서 영원히 호버**했다. `8cb0861` 원본부터 없었다(후속 회귀 아님) | 첫 줄에 `self._sm = _State.VISION_SEARCH`. 회귀 그물 `fc_ros/test/test_offboard_f2_state.py` |
| 2 | **래치가 vision 프레임이 아니라 제어틱을 셌다.** `vt.valid`는 `vision_stale_timeout`(1.0s) 동안 True라 **같은 메시지 하나가 최대 10번 계수**됐다 — setpoint 1건만 오고 침묵해도 0.2s 만에 래치가 서고, 버퍼가 같은 좌표의 사본이라 산포 필터까지 통과했다. §6-1 #5·코드 자신의 docstring과 정면 배치 | `VisionTargetBridge.setpoints_rx`의 **증가분**으로만 버퍼에 넣는다. 파라미터도 `vision_latch_ticks` → **`vision_latch_frames`**로 개명(이름이 단위를 거짓말하면 같은 버그가 다시 난다) |
| 3 | **하니스가 LANDING을 못 봤다.** `_exit_hold` 단일 분기 도입으로 HOLD 종료 문구가 바뀌었는데(`WP1 도달·안정 → LANDING (…)` → `WP1 도달·안정 (…) → LANDING`, `→ 강제 LANDING` 소멸) `run_scenario.py`·`analyze_run.py`의 정규식이 그대로라 `state_timeline`에서 **LANDING이 통째로 누락**됐다(R1_base: HOLD 72.50 → DONE 127.98) | **정규식을 고쳤다**(문구는 안 되돌린다 — 단일 분기는 "두 경로 중 한쪽만 고치는 사고"를 막는 구조다). 구·신 문구를 **합집합**으로 받는다(아카이브 런 재분석이 깨지면 안 된다). `VISION_SEARCH`/`PRECISION_LAND` 2종과 GPS 착륙 폴백 3종도 같이 추가 |

> ⚠️ **왜 440건 테스트가 전부 통과하면서 1·2번이 살아남았나:** `fc_ros/test/` 어디에도
> `_exit_hold`·`_enter_vision_search`·`_step_vision_search`·`VISION_SEARCH` 참조가 **0건**이었다.
> 순수 함수 테스트는 "`_sm`을 세팅했는가" 같은 **전이 자체**를 원리적으로 볼 수 없다.
> `test_offboard_f2_state.py`가 그 층을 새로 만든다 — rclpy 스텁 + `__new__` 껍데기로
> **노드 메서드를 실제로 실행**한다.

**SITL 합성 vision 발행기는 이제 저장소에 있다: `tools/sitl/fake_vision.py`**
(§6-1 장애주입 6종을 인자로 낼 수 있다). 종전엔 폐기 클론에만 있었다.

---

## 9. 참조

- **`docs/px4_v6c_patch_build.md`** — 🔴 **선행 작업(천이 플래시) 절차 전문.** §0-1 참조
- `docs/session_status.md` 🛩 sitl-vtol 트랙 — 선행 작업의 현재 상태
- `docs/vision_fc_interface.md` — 정찰 사실확정(852줄). **필요한 절만.** §9에 F1~F6 원래 작업 목록
  (F1=shim은 vision이 대신 완료했다)
- `docs/rpi_deploy.md` — 실기체 배포 절차·함정. §6이 새 파라미터를 yaml/launch에 넣는 법
- `docs/sitl_vtol_remediation_plan.md` §4-1 — "첫 실비행에 미검증 변수 둘 금지"
- `fc_bridge/execution/state_logic.py:458` — `slew_setpoint`
- **`fc_bridge/execution/search_pattern.py`** — 나선 기하·풋프린트·검출고도 상한 (F2 신설)
- **`fc_bridge/execution/precision_land.py`** — 래치·하강게이트·인계판정 순수 함수 (F2 신설)
- **`fc_ros/fc_ros/adapters/vision_target_bridge.py`** — 두 토픽 구독·계약 파싱 (F2 신설)
- `vision/ros/shim_core.py` — 계약 상수·변환 순수 로직(읽기만; vision 도메인 소유)
