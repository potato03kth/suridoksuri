---
doc_type: fc_handoff
project: suridoksuri-1
track: 🎯 fc-정밀착륙 (F2)
scope: vision → FC 인수인계 — `OffboardNode` 정밀착륙 서브상태 구현을 위한 계약·함정·검증 정본
status: ▶ 착수 대기 (vision 쪽 절반 완료·실기체 검증됨)
created: 2026-07-28
last_updated: 2026-07-28
---

# FC 정밀착륙(F2) 인수인계

> **이 문서 하나로 F2를 완주할 수 있게 썼다.** FC 세션은 `docs/vision_plan.md`·
> `docs/vision_orchestration_handoff.md`를 읽지 않아도 된다(도메인 컨텍스트 격리).
> 더 깊은 배경이 필요하면 `docs/vision_fc_interface.md`(852줄, 정찰 사실확정)의 **필요한 절만** 열어라.
>
> **작성자:** vision 오케스트레이터 세션(2026-07-28). 아래 "실측/검증됨" 표시가 붙은 것은
> **전부 실기체 또는 실행으로 직접 확인**한 것이고 재조사할 필요가 없다.

---

## 0. 한 줄 요약

**vision은 낼 것을 다 내고 있고, 받는 코드가 FC에 한 줄도 없다.** F2는 폐루프를 닫는 마지막 칸이다.

```
[호스트 picam-venv]              [fc 컨테이너]                        [offboard_node]
 main.py --target-sink ─JSONL─▶ shim_node ─┬─▶ /vision/landing_setpoint  ← ❗받는 코드 없음
                                           ├─▶ /vision/target_status     ← ❗받는 코드 없음
                                           ├─▶ /vision/target_pose       ← (선택) 안 써도 된다
                                           └─▶ /mavros/landing_target/raw  ⏸ 기본 꺼짐
                                           ▲
                                 /mavros/local_position/pose ✅ 이미 구독 중
```

**`fc_ros`/`fc_bridge`는 지금까지 한 줄도 수정되지 않았다.** 인터페이스 파일 3종 md5가
기준선(`893a5eb`) 이후 계속 동일함을 vision 세션이 반복 확인했다 — 즉 **기존 계약이 그대로 유효**하다.

---

## 1. 붙는 자리와 재사용할 패턴

### 1-1. 위치

`fc_ros/fc_ros/nodes/offboard_node.py`의 **`_step_hold`(`:1230`)와 `_step_landing`(`:1300`) 사이**에
새 서브상태를 넣는다. `_State` enum(`:111`)에 항목 추가.

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

## 3. 🔴 FC가 결정해야 할 것

| # | 항목 | 왜 FC 결정인가 |
|---|---|---|
| **D-a** | **`HOLD` 종결 시한** (`hold_before_reascend_s`) | vision이 **일부러 안 만들었다** — `FailsafeContract`가 이 값을 *"FC가 자기 제어틱에서 재는 값, vision이 정해 보낼 값이 아니다"*로 못박았다. 🔴 **없으면 vision이 `HOLD`로 빠졌을 때 무한 호버링**이다(실측 확인: 재획득 없으면 `HOLD`가 영원히 유지됨). **F2에서 반드시 같이 넣어라.** |
| **D-b** | `_RANGE_GUARDED_STATES` 포함 여부 | §1-3 |
| **D-c** | `listen_lt: true`로 네이티브 precision-land 피벗을 열 것인가 | `px4_config.yaml:214`가 현재 `false`라 `/mavros/landing_target/raw` 구독자가 **아예 없다**. 열면 `frame`에 **정수 리터럴 12**를 써야 한다(§4 함정 1) |
| **D-d** | AGL 소스 확정 | §5 참조 — vision이 가정하는 "라이다 AGL"이 `fc_ros`에 배선돼 있지 않다 |

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

### 6-1. SITL 장애주입 (필수)

vision 프로세스 **SIGKILL** / 소켓 끊김 / attitude stale / `valid=false` 각각에 대해 FC가
의도한 상태로 전이하는지 실증하라. 특히:
- 생산자 SIGKILL → `vision/link` **ERROR** → FC가 홀드로 빠지는가
- `state=HOLD`(veto) → FC가 하강을 멈추는가
- setpoint 침묵이 길어질 때 **마지막 값을 붙들고 있지 않은가**

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

## 8. 참조

- `docs/vision_fc_interface.md` — 정찰 사실확정(852줄). **필요한 절만.** §9에 F1~F6 원래 작업 목록
  (F1=shim은 vision이 대신 완료했다)
- `docs/rpi_deploy.md` — 실기체 배포 절차·함정
- `docs/sitl_vtol_remediation_plan.md` §4-1 — "첫 실비행에 미검증 변수 둘 금지"
- `fc_bridge/execution/state_logic.py:458` — `slew_setpoint`
- `vision/ros/shim_core.py` — 계약 상수·변환 순수 로직(읽기만; vision 도메인 소유)
