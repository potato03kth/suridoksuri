---
doc_type: flight_plan_archive
project: suridoksuri-1
scope: 완료된 작업단위(작업 A~E, SITL-1~4)의 원본 상세 계획·체크리스트
archived: 2026-07-05
---

# flight_plan 완료 작업단위 아카이브

> `docs/flight_plan.md`에서 이동된 **완료** 작업단위의 상세 계획이다. 결과 기록은 `docs/session_log.md`(아카이브 포함)와 `docs/sitl_verification_log.md`에 있다.
> 활성 계획은 `docs/flight_plan.md` 참조.

| 작업단위 | 완료일 | 비고 |
|---|---|---|
| 작업 A — params/YAML 정비 | 2026-06-19 | |
| 작업 B — 종단 감속 헬퍼 | 2026-06-20 | |
| 작업 C — 상태머신 ① 이륙·상승·천이 | 2026-06-20 | |
| 작업 D — 상태머신 ② 역천이·착륙 | 2026-06-20 | |
| 작업 E — 긴급 수동 override | 2026-06-20 | SITL-4에서 OVERRIDE 상태 + AUTO.LOITER 폴백으로 정정 |
| SITL-1 — VTOL 환경 전환 + 상수 확인 | 2026-06-19 | 조건부 PASS (RC override→POSCTL은 실기체 이월) |
| SITL-2 — launch 통합 기동 | 2026-06-20 | |
| SITL-3 — 경로 추종 검증 | 2026-06-30 | FW 위치 setpoint 전환이 핵심. `sitl3_fix_plan.md` |
| SITL-4 — 전체 사이클 통합 | 2026-06-30 | `sitl_verification_log.md` |

---

# 코드 작업단위

---

## 작업 A — params/YAML 정비

**유형:** [코드] (Claude 자율)
**목적:** 모든 노드 파라미터를 정상화하고 신규 파라미터를 추가한다. 운용 고도를 일원화한다.
**선행:** 없음

**파일:**

- `fc_ros/fc_ros/params/fc_ros_params.yaml`
- `fc_ros/fc_ros/nodes/offboard_node.py` (`declare_parameter`)

**작업 목록:**

1. **`waypoints` 2D → 1D flat (기존 버그).** `offboard_node`, `mission_node` 양쪽.
   ROS2 파라미터는 중첩 리스트를 지원하지 않아 당시 YAML(`- [0,0,150]` 형식)은 launch 시 TypeError를 유발했다.
   ```yaml
   # 수정 전 (오류)
   waypoints:
     - [0.0, 0.0, 50.0]
     - [100.0, 0.0, 50.0]
   # 수정 후 (flat, 코드에서 reshape(-1,3))
   waypoints: [0.0, 0.0, 50.0, 100.0, 0.0, 50.0]
   ```
2. **운용 고도 일원화 = 50 m (SITL).** `transition_alt`, waypoint 고도, MissionNode 고도를 모두 50 m로 맞춘다.
   (이전 문서 간 50/150 m 혼재 → SITL 검증은 50 m로 통일.)
3. **신규 파라미터 추가** (`offboard_node` `declare_parameter` + YAML):
   ```python
   self.declare_parameter("transition_alt",  50.0)  # 천이 고도 (m, h_up 양수)
   self.declare_parameter("d_end_thresh",    10.0)  # 역천이 진입 거리 기준 (m)
   self.declare_parameter("landing_timeout", 60.0)  # AUTO.LAND 타임아웃 (s)
   self.declare_parameter("v_terminal",      15.2)  # 경로 끝점 도달 속도 (작업 B가 소비)
   self.declare_parameter("decel_dist",      80.0)  # 종단 감속 시작 거리 (작업 B가 소비)
   ```

   - `v_terminal = 15.2` 근거: 스톨 13.8 m/s × 1.1 = 15.18 ≈ 15.2 m/s.

**테스트:** `fc_ros/test/test_params.py` (신규, rclpy 불필요)

```python
def test_flat_waypoints_reshape():
    raw = [0.0, 0.0, 50.0, 100.0, 0.0, 50.0]
    wps = np.array(raw, dtype=float).reshape(-1, 3)
    assert wps.shape == (2, 3)
    assert wps[1, 0] == 100.0   # 북 100 m
    assert wps[0, 2] == 50.0    # 고도 50 m
```

**합격 기준:** `pytest fc_ros/test/test_params.py` 통과. (launch 실측은 SITL-2.)

---

## 작업 B — 종단 감속 헬퍼 + 배선

**유형:** [코드] (Claude 자율)
**목적:** 경로 끝에서 v_terminal까지 자연 감속하도록 v_profile을 후처리한다.
**선행:** 없음 (작업 A와 독립)

**배경:** `run_planner()`→eta3/diterpin은 `v_ref=v_cruise`로 고정한다(`eta3clothoid_v3_1_planner.py:469`, `D_iterpin_planner.py:346`). `v_terminal`을 읽지 않으므로, **플래너 수정이 아니라 결과 v_profile에 후처리**를 적용한다.

**파일:**

- `fc_bridge/planning/terminal_decel.py` (신규)
- `fc_ros/fc_ros/nodes/offboard_node.py` (`main()` 배선)

**작업 목록:**

1. **헬퍼 구현** — 경로 끝 `decel_dist` 구간을 v_terminal로 선형(또는 √) ramp-down:
   ```python
   def apply_terminal_decel(v_profile, s_arc, v_terminal, decel_dist):
       """경로 끝 decel_dist 구간을 v_terminal로 수렴시킨다.
       s_arc: 각 점의 누적 호길이 (path.points[i].s).
       끝점 속도 = v_terminal, decel_dist 밖 구간은 불변."""
       s_end = s_arc[-1]
       d_remain = s_end - s_arc                      # 끝점까지 남은 거리
       in_zone = d_remain < decel_dist
       frac = np.clip(d_remain / decel_dist, 0.0, 1.0)  # 1(시작)→0(끝)
       v_ramp = v_terminal + (v_profile - v_terminal) * frac
       return np.where(in_zone, np.minimum(v_profile, v_ramp), v_profile)
   ```
2. **`OffboardNode.main()` 배선** — `run_planner` 직후 적용:
   ```python
   v_profile = np.array([pt.v_ref for pt in path.points])
   s_arc     = np.array([pt.s     for pt in path.points])
   v_profile = apply_terminal_decel(v_profile, s_arc, v_terminal, decel_dist)
   ```

**테스트:** `fc_bridge/tests/test_terminal_decel.py` (신규)

```python
def test_endpoint_reaches_v_terminal():
    s = np.linspace(0, 200, 201)
    v = np.full_like(s, 15.0)            # v_cruise 고정 (직선)
    out = apply_terminal_decel(v, s, v_terminal=12.0, decel_dist=80.0)
    assert out[-1] == pytest.approx(12.0)         # 끝점 = v_terminal
    assert out[0]  == pytest.approx(15.0)         # decel_dist 밖 불변
    assert np.all(np.diff(out[-80:]) <= 1e-9)     # 마지막 구간 단조 비증가
```

> **SITL 가시성 주의:** SITL에서 `v_cruise=15.0`, `v_terminal=15.2`는 거의 같아 감속이 미미하다. SITL-3에서 감속을 **눈으로 확인**하려면 일시적으로 `v_cruise`를 18~20 m/s로 올려 `v_terminal=15.2`와 차이를 만든다. 실기체는 `v_cruise ≥ 17 m/s` 예상이라 실측 시 자연히 드러난다.

**합격 기준:** `pytest fc_bridge/tests/test_terminal_decel.py` 통과.

---

## 작업 C — 상태머신 ① 이륙·상승·천이(FW)

**유형:** [코드] (Claude 자율)
**목적:** 자율 이륙 → 상승 → MC→FW 천이 → OFFBOARD 진입을 OffboardNode에 구현한다.
**선행:** 작업 A. vtol_state 상수는 기술 참조의 값을 사용하고 **SITL-1에서 확정**한다.

**파일:** `fc_ros/fc_ros/nodes/offboard_node.py`

**작업 목록:**

1. **`_State` enum에 신규 상태 추가** (기존 IDLE/STREAMING/ENTRY/FOLLOWING/DONE 유지):
   ```python
   ARM_TAKEOFF   = "arm_takeoff"
   CLIMBING      = "climbing"
   TRANSITION_FW = "transition_fw"
   ```
   초기 상태를 `STREAMING` → `ARM_TAKEOFF`로 변경.
2. **`CommandLong` 서비스 클라이언트 추가** (VTOL 천이용):
   ```python
   from mavros_msgs.srv import CommandLong
   self._cmd_cli = self.create_client(CommandLong, "/mavros/cmd/command")
   ```
3. **상태 핸들러 구현:**
   - `_step_arm_takeoff()`: ARM 요청 + AUTO.TAKEOFF (set_mode `"AUTO.TAKEOFF"` 또는 `/mavros/cmd/takeoff`. 정확한 방식은 SITL-1에서 확정).
   - `_step_climbing()`: `state.pos_ned[2] >= transition_alt` (h_up, 양수=위) 확인 → TRANSITION_FW.
   - `_step_transition_fw()`: `CommandLong(command=3000, param1=4.0)` 호출 후 `vtol_state == VTOL_STATE_FW` 확인 → STREAMING.
4. **STREAMING 리팩터:**
   - **ARM 요청 제거** (ARM은 ARM_TAKEOFF에서 1회 완료). STREAMING은 OFFBOARD 전환만 담당.
   - **더미 세트포인트를 전진속도로 변경**: FW 상태에서 속도 0은 스톨 위험. 첫 WP 방향 `v_cruise` 전진 세트포인트를 2초(20 tick) 발행 후 OFFBOARD 전환 → ENTRY/FOLLOWING.

**설계 주의:**

- AUTO.TAKEOFF 후 PX4 모드가 **HOLD**로 전환됨 (SITL-1 실측). OFFBOARD 전환은 **TRANSITION_FW 완료 후 STREAMING**에서만.
- VTOL 천이는 HOLD 모드에서 동작 확인됨 (SITL-1). TRANSITION_FW 시점에 OFFBOARD가 아니어야 함.

**테스트:** `fc_ros/test/test_offboard_node.py`에 순수 로직 케이스 추가 (기존 `_entry_done` 패턴):

```python
def _climb_reached(pos_ned_up, transition_alt):
    return pos_ned_up >= transition_alt

def test_climb_reached(): assert _climb_reached(50.1, 50.0) is True
def test_climb_not_reached(): assert _climb_reached(49.9, 50.0) is False

def _vtol_is_fw(vtol_state, FW=4): return vtol_state == FW
def test_transition_fw_done(): assert _vtol_is_fw(4) is True
```

**합격 기준:** `pytest fc_ros/test/test_offboard_node.py` 통과.

> **사후 정정 (2026-06-24/30):** MC OFFBOARD에서 헤딩 정렬은 velocity만으로 불가 — `twist.angular.z` yaw rate 필수. 이후 SITL-3에서 FW 활성 구간 전부 **위치 setpoint**로 전환됨 (velocity는 FW가 무시).

---

## 작업 D — 상태머신 ② 역천이·착륙

**유형:** [코드] (Claude 자율)
**목적:** 경로 끝 도달 → FW→MC 역천이 → AUTO.LAND를 구현한다.
**선행:** 작업 C

**파일:** `fc_ros/fc_ros/nodes/offboard_node.py`

**작업 목록:**

1. **`_State` enum에 추가:** `TRANSITION_MC`, `LANDING`.
2. **`_step_following()` 종료조건 변경:**
   - 당시: `dist_to_end < 3.0` (하드코딩) → `DONE`.
   - 변경: `dist_to_end < self._d_end_thresh` → `TRANSITION_MC` 반환.
   - **감속은 작업 B의 v_profile이 담당**한다. L1 Guidance는 v_profile을 따라 끝점에서 v_terminal로 자연 감속한다. controller-level 속도 clamp를 mid-flight에 적용하지 않는 이유: L1은 속도·기하를 동시에 사용하므로 중간 clamp 시 cross-track error가 증가한다.
3. **상태 핸들러 구현:**
   - `_step_transition_mc()`: `CommandLong(command=3000, param1=3.0)` 호출 후 `vtol_state == VTOL_STATE_MC` 확인 → LANDING.
   - `_step_landing()`: set_mode `"AUTO.LAND"` → `not state.armed`(disarmed) 확인 → DONE. `landing_timeout` 초과 시 경고 로그.

**테스트:** `fc_ros/test/test_offboard_node.py`에 추가:

```python
def _trans_mc_trigger(dist_to_end, d_end_thresh): return dist_to_end < d_end_thresh
def test_trans_mc_trigger(): assert _trans_mc_trigger(9.0, 10.0) is True
def test_trans_mc_not_yet(): assert _trans_mc_trigger(11.0, 10.0) is False

def _landing_done(armed): return not armed
def test_landing_done(): assert _landing_done(False) is True
```

**합격 기준:** `pytest fc_ros/test/test_offboard_node.py` 통과.

> **사후 추가 (2026-06-30, SITL-3):** 역천이 오버슈트 보정을 위해 **HOLD 상태**가 추가됨 — TRANSITION_MC 후 MC로 WP1 복귀·정착 → WP1 지점 착륙. 판정 순수 함수는 `fc_bridge/execution/state_logic.py`에 집중 관리.

---

## 작업 E — 긴급 수동 override

**유형:** [코드] (Claude 자율)
**목적:** `/fc_ros/override` 토픽으로 자동비행을 즉시 중단하고 수동 모드로 전환한다.
**선행:** 작업 C (vtol_state 분기 사용)

**파일:** `fc_ros/fc_ros/nodes/offboard_node.py`

**작업 목록:**

1. **`/fc_ros/override` (std_msgs/Bool) 구독:**

   ```python
   from std_msgs.msg import Bool
   self.create_subscription(Bool, "/fc_ros/override", self._cb_override, 10)

   def _cb_override(self, msg):
       if msg.data:
           self._request_override()
   ```

2. **vtol_state 기반 분기 모드 전환:**
   ```python
   def _request_override(self):
       req = SetMode.Request()
       req.custom_mode = ("POSCTL" if self._vtol_state == VTOL_STATE_MC
                          else "MANUAL")      # MC→POSCTL, FW→MANUAL
       self._set_mode_cli.call_async(req)
       self._sm = _State.DONE                  # setpoint 발행 즉시 중단
       self.get_logger().warn("긴급 수동 전환 실행")
   ```
3. **트리거 (사람):** `ros2 topic pub --once /fc_ros/override std_msgs/Bool "data: true"`

> **RC 레이어와 독립:** PX4 `COM_RC_OVERRIDE`(SITL-1에서 설정)는 ROS2 override와 별개로 동작하는 하드웨어 레이어다. 두 레이어 모두 OFFBOARD 진입 전/후 어느 상태에서도 독립 동작해야 한다.

> ⚠ **SITL-4 정정 (2026-06-30):** 위 코드처럼 `_request_override`가 곧장 `_sm=DONE`으로 가면 모드 전환 거부 시 cmd_vel velocity-0 발행이 OFFBOARD를 살려둬 **FW가 직진 폭주**한다(SITL-4 실측). 또 MANUAL/POSCTL은 RC·조이스틱 같은 수동제어 소스가 필요해 **headless SITL에선 거부**된다(SITL-1 `COM_RC_OVERRIDE`→POSCTL 재현불가와 동일).
> → 현재 구현은 **`_State.OVERRIDE`** 를 두고: ⓐ override 시 OFFBOARD setpoint 발행 중단, ⓑ manual 모드 시도, ⓒ 1초 내 미진입이면 **AUTO.LOITER 안전 폴백** 강제. 실기체에선 조종사 RC로 manual이 즉시 잡혀 폴백 전 종료. 판정함수 `override_reached`/`override_fallback_due`(state_logic).

**테스트:** `fc_ros/test/test_offboard_node.py`에 추가:

```python
def _override_mode(vtol_state, MC=3): return "POSCTL" if vtol_state == MC else "MANUAL"
def test_override_mc(): assert _override_mode(3) == "POSCTL"
def test_override_fw(): assert _override_mode(4) == "MANUAL"
```

**합격 기준:** `pytest fc_ros/test/test_offboard_node.py` 통과.

---

# SITL 검증 게이트

> 모든 SITL 게이트는 **사람이 WSL에서 수행**했다. Claude는 절차·관찰 포인트를 제시하고 로그로 판정했다.

---

## SITL-1 — VTOL 환경 전환 + 상수 확인

**목적:** x500 → standard_vtol 전환. 이후 모든 VTOL 검증의 기반. vtol_state 상수 실측.
**선행:** WSL SITL 환경 정상 (기존 x500 환경과 동일 WSL)

**절차 (사람):**

1. VTOL SITL 기동: `cd ~/PX4-Autopilot && make px4_sitl gz_standard_vtol`
2. MAVROS + ROS2 실행 (기존과 동일), QGC UDP 연결 확인
3. `ros2 topic echo /mavros/extended_state` → vtol_state 상수값 기록 (MC / FW / 천이 중)
4. QGC 수동 이륙 → 천이 → 착륙 (fc_ros 없이)
5. VTOL 천이 서비스 직접 호출:
   ```bash
   ros2 service call /mavros/cmd/command mavros_msgs/srv/CommandLong "{command: 3000, param1: 3.0}"
   ```
6. AUTO.TAKEOFF 동작 방식 확인 (set_mode "AUTO.TAKEOFF" vs `/mavros/cmd/takeoff`) — **작업 C 입력**
7. RC 오버라이드 설정 (QGC): `COM_RC_OVERRIDE = 3`, RC 모드 스위치 → POSCTL/LOITER 채널 매핑

**합격 기준 (체크리스트):**

- [x] vtol_state 상수값 실측 기록 (MC=3/FW=4/천이=1·2) — 작업 C/D/E가 참조
- [x] AUTO.TAKEOFF 후 PX4 전환 모드 = **HOLD** (2026-06-19 실측)
- [x] CommandLong(3000, param1=4) 으로 MC→FW 천이 성공 (param1 수정됨)
- [x] vtol_state == FW 인 상태에서 velocity 세트포인트 수락 여부
- [x] COM_RC_OVERRIDE = 3 설정 + 재시작 유지 확인
- [ ] RC 스틱 입력으로 OFFBOARD→POSCTL 전환 — SITL 재현 불가, SITL-5(실기체)로 이월

---

## SITL-2 — launch 통합 기동

**목적:** `ros2 launch fc_ros phase2.launch.py` 정상 기동 (구 계획 "최종 launch 검증" 대체).
**선행:** 작업 A

**절차 (사람):**

1. WSL 동기화 + `colcon build --packages-select fc_ros fc_bridge` + `source install/setup.bash`
2. `ros2 launch fc_ros phase2.launch.py`
3. `ros2 node list` → TelemetryNode + OffboardNode 확인
4. `ros2 param list` → YAML 파라미터 정상 로드 확인
5. (디버그) `phase1.launch.py` 기동 — MissionNode + 파라미터 로드만 확인

**합격 기준 (체크리스트):**

- [x] TypeError 없이 두 노드 기동 (2026-06-20 실측)
- [x] 신규 파라미터(`transition_alt`, `d_end_thresh`, `v_terminal` 등) 로드 확인 (2026-06-20 실측)

---

## SITL-3 — 경로 추종 검증

> ✅ **PASS (2026-06-30)** — 직선 300 m 전체 시퀀스 검증. 핵심 버그(원호천이·heading종속·FOLLOWING미진입·151m RTL·역천이꺾임) 해결.
> 근본 원인: **PX4 FW 오프보드는 위치 setpoint 필수**(velocity 무시). 상세 `docs/sitl3_fix_plan.md` · 튜닝 `docs/sitl3_tuning_notes.md`.

**목적:** 경로 생성 → OffboardNode 주입 → L1 추종 + 종단 감속을 SITL 검증.
**선행:** 작업 B · C · D · SITL-2

**절차 (사람 + Claude dry-run 보조):**

1. **dry-run 경로 + 감속 확인** (SITL 없이, Windows): `cd fc_bridge && python run_phase1.py --dry-run --plot --planner eta3`
   - 속도 프로파일 끝점이 v_terminal로 떨어지는지 확인 (작업 B 적용 확인. 필요 시 `v_cruise` 일시 상향)
2. **테스트 경로 3종** (`fc_ros_params.yaml` 교체 또는 별도 주입):
   - 경로 A: 직선 2 WP (100 m)
   - 경로 B: L자 3 WP (N 100 m → E 100 m)
   - 경로 C: 사각형 귀환 5 WP (100×100 m)
3. **SITL 추종** (각 경로): `ros2 launch fc_ros phase2.launch.py` → FOLLOWING 로그 + cross_track_error 확인

**합격 기준 (체크리스트):**

- [x] dry-run 속도 프로파일 끝점 = v_terminal (작업 B 동작 확인, 2026-06-20)
- [x] 직선 경로 FOLLOWING 진입·끝점 도달·전체 시퀀스 (2026-06-30. L자·사각형은 SITL-4에서 커버)

---

## SITL-4 — 전체 사이클 통합

> ✅ **PASS (2026-06-30)** — 직선 300 m + L자 전체 자율 사이클 검증. 역천이 가속도 ~1.5 m/s²(<0.3g), WP1 착륙오차 ~0.3 m.
> override는 1차 실패(headless SITL은 RC 없어 MANUAL 거부) → **AUTO.LOITER 안전 폴백** 추가 후 재검증 PASS. 상세 `docs/sitl_verification_log.md`.

**목적:** 이륙→천이→경로추종→역천이→착륙 전체 자율 시퀀스 SITL 검증.
**선행:** 작업 E · SITL-3

**절차 (사람):**

1. VTOL SITL 기동 → `ros2 launch fc_ros phase2.launch.py`
2. 전체 상태 전이 로그 수집:
   ```
   [ARM_TAKEOFF] ARM + AUTO.TAKEOFF
   [CLIMBING]    고도 50 m 도달
   [TRANSITION_FW] 헤딩 정렬 → vtol_state == FW
   [STREAMING→FOLLOWING] OFFBOARD 경로 추종 시작
   [TRANSITION_MC] vtol_state == MC (역천이, 직선 감속)
   [HOLD]        MC로 WP1 복귀·정착 (역천이 오버슈트 보정)
   [LANDING]     AUTO.LAND → disarmed
   [DONE]
   ```
   > `entry_mode="pre_takeoff"` 이므로 ENTRY는 건너뛴다. HOLD는 SITL-3에서 추가된 상태(역천이 오버슈트 후 WP1 복귀·착륙).
3. QGC Plan 뷰에서 실제 궤적 확인
4. 천이 가속도 측정: `/mavros/imu/data` `linear_acceleration` 크기 ≤ 0.3g(≈2.94 m/s²) 확인
5. 긴급 override 1회 트리거 테스트 (FOLLOWING 중 `/fc_ros/override`)
6. 경로 A/B/C 각각 반복

**합격 기준 (체크리스트):**

- [x] 전체 상태 전이 로그 순서대로 출력 + disarmed 도달 (2026-06-30, HOLD 포함)
- [x] 역천이 중 가속도 ≤ 0.3g (~1.5 m/s², `VT_B_DEC_MSS` 1.0 설계값 부합)
- [x] override 트리거 시 안전 전환 + setpoint 중단 (SITL: AUTO.LOITER 폴백 — manual은 RC 필요, 실기체 SITL-5 이월)
- [x] 직선·L자 경로 FOLLOWING·착륙 완료, WP1 착륙오차 ~0.3 m
- [x] `sitl_verification_log.md` 업데이트

---

## 초기 검증 완료 항목 (계획 수립 전)

- [x] WSL SITL 구축 (PX4 + MAVROS + ROS2 Humble, 기체: gz_x500)
- [x] QGC ↔ WSL SITL 연결 (UDP 14551)
- [x] hover_node SITL 검증 (2026-06-01)
- [x] OffboardNode 기본 검증 (x500, STREAMING→FOLLOWING→DONE, 2026-06-06)
- [x] TelemetryNode 단위 테스트 25/25 PASS (2026-06-06)
- [x] TelemetryNode SITL 검증 (2026-06-17)
- [x] MissionNode SITL 검증 (버그 3개 수정, 2026-06-18)
