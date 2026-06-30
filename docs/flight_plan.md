---
doc_type: flight_plan
project: suridoksuri-1
scope: fc_ros 전체 비행 사이클 검증 — 작업단위 분할 및 실행 계획
status: 진행 중
last_updated: 2026-06-18
---

# fc_ros 전체 비행 사이클 검증 — 작업 계획

> 2026-06-18 세션에서 확정된 설계 결정 + 정합성 검토 결과 반영.
> 각 작업단위는 이 문서를 기준으로 독립 진입 가능하다.

> **범위:** 본 계획은 **단방향 비행 사이클** (이륙→천이→경로 추종→역천이→착륙) SITL 검증에 집중한다.
> 대회 전체 미션 (왕복 + 복수 착륙-이륙 사이클)은 본 계획 완료 후 별도 계획으로 진행한다.

---

## 작업단위 실행 규약

이 계획의 작업단위는 두 종류다. 새 컨텍스트에서 **"실행하라"** 한마디로 진입한다.

| 유형       | 실행 주체                | "실행하라"의 의미                                                                     | 합격 판정(테스트)       |
| ---------- | ------------------------ | ------------------------------------------------------------------------------------- | ----------------------- |
| **[코드]** | Claude 자율              | 코드 수정 → `pytest` 실행까지 Claude가 완료 (Windows, SITL 불필요)                    | pytest 통과             |
| **[SITL]** | 사람 (WSL) + Claude 보조 | Claude가 절차·체크리스트 준비 → 사람이 WSL에서 수행 → 로그를 붙여넣으면 Claude가 판정 | 체크리스트 전 항목 충족 |

규칙:

- **[코드] 단위는 SITL 없이 완결**된다. 순수 로직을 함수로 추출해 `rclpy` 없이 pytest로 검증한다 (기존 `test_offboard_node.py` 패턴).
- **[SITL] 게이트는 사람이 손으로 수행**한다. Claude는 기동 명령·관찰 포인트·합격 기준을 제시하고, 결과 로그로 PASS/FAIL을 판정한 뒤 `sitl_verification_log.md`에 기록한다.
- 각 단위는 **선행 조건**을 명시한다. 선행이 끝나지 않았으면 진입하지 않는다.

---

## 작업단위 목록 및 의존 관계

| 작업단위                                | 유형   | 선행               | 테스트                                                     |
| --------------------------------------- | ------ | ------------------ | ---------------------------------------------------------- |
| **작업 A** — params/YAML 정비           | [코드] | —                  | `test_params.py`: flat→(N,3) reshape, 신규 파라미터 기본값 |
| **작업 B** — 종단 감속 헬퍼 + 배선      | [코드] | —                  | `test_terminal_decel.py`: 끝점=v_terminal, 단조감소        |
| **작업 C** — 상태머신 ① 이륙·상승·천이  | [코드] | A · (SITL-1 상수)  | `test_offboard_node.py`: 상승/천이 트리거 순수로직         |
| **작업 D** — 상태머신 ② 역천이·착륙     | [코드] | C                  | `test_offboard_node.py`: d_end_thresh/착륙 판정            |
| **작업 E** — 긴급 수동 override         | [코드] | C                  | `test_offboard_node.py`: vtol_state 분기 로직              |
| **SITL-1** — VTOL 환경 전환 + 상수 확인 | [SITL] | 환경               | 체크리스트 (vtol_state 상수 기록, 수동 천이, RC override)  |
| **SITL-2** — launch 통합 기동           | [SITL] | A                  | 체크리스트 (phase2 기동, 파라미터 로드)                    |
| **SITL-3** — 경로 추종 검증             | [SITL] | B · C · D · SITL-2 | 체크리스트 (cross_track_error, 끝점 감속)                  |
| **SITL-4** — 전체 사이클 통합           | [SITL] | E · SITL-3         | 체크리스트 (전체 상태 전이 로그, disarmed, 천이 가속도)    |
| **SITL-5** — RPi4 실기체 배포           | [배포] | SITL-4             | 체크리스트 (배포·튜닝·안전)                                |

### 권장 실행 순서

```
SITL-1 (사람, 환경+vtol_state 확정)  ─┐   ← 작업 C가 상수를 참조하므로 먼저 권장
작업 A (Claude)  ──────────────────┐ │
작업 B (Claude)  ──────────────────┤ │   ← A·B는 vtol_state 무관, SITL-1과 병행 가능
작업 C (Claude)  ←─ A, SITL-1 ─────┘ │
작업 D (Claude)  ←─ C ───────────────┤
작업 E (Claude)  ←─ C ───────────────┘
SITL-2 (사람)  ←─ A
SITL-3 (사람)  ←─ B, C, D, SITL-2
SITL-4 (사람)  ←─ E, SITL-3
SITL-5 (사람)  ←─ SITL-4
```

> 본 계획의 **SITL-2(launch 통합 기동)** 가 구 계획의 마지막 잔여분("최종 launch 검증")을 대체한다.
> 브랜치 전략(main 병합, `dev--fc-vtol-sitl` 분기)은 `docs/session_status.md` 참조.
> 전체 사이클(SITL-4) **이후**의 임의 WP 생성·추종 검증은 [후속 계획](#후속-계획--임의-wp-경로-생성추종-검증)(작업 F · SITL-6) 참조 — 본 계획의 핵심 범위 밖이다.

---

## 확정된 설계 결정

| 항목           | 결정                                                                                                                                                                                                |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 이륙/천이      | fc_ros 전부 자동 (ARM + AUTO.TAKEOFF + VTOL_TRANSITION)                                                                                                                                             |
| Phase1 역할    | 디버그/백업용 (실제 비행은 Phase2 단독)                                                                                                                                                             |
| 착륙           | fc_ros 자동 (역천이 + AUTO.LAND)                                                                                                                                                                    |
| 경로 생성기    | eta3 (기본), diterpin (대안)                                                                                                                                                                        |
| 경로 추종      | L1 Guidance (OffboardNode, OFFBOARD 모드)                                                                                                                                                           |
| 역천이 전 감속 | **경로 생성 후처리 수준** — `apply_terminal_decel()`(작업 B)가 v_profile 마지막 `decel_dist` 구간을 v_terminal(≥스톨×1.1)로 ramp-down. OffboardNode는 거리 조건(`d_end_thresh`)만으로 역천이 트리거 |
| 긴급 수동 전환 | RC 모드스위치(COM_RC_OVERRIDE) + ROS2 `/fc_ros/override` 토픽. MC→POSCTL, FW→MANUAL                                                                                                                 |

> ⚠ **감속 전략 정정 (2026-06-18 검토):** 당초 "`vehicle_params`에 `v_terminal`을 넣으면 경로 생성기가 끝점을 v_terminal로 수렴"한다는 전제는 **현재 코드에서 동작하지 않는다.** `run_planner()`→eta3/diterpin 플래너는 `v_ref=v_cruise`로 **고정**하며 `v_terminal`을 읽지 않는다(검증 완료). 따라서 감속은 **작업 B의 후처리 헬퍼**가 담당한다. 상세는 [작업 B](#작업-b--종단-감속-헬퍼--배선) 참조.

---

## 목표 비행 시퀀스

```
[ros2 launch fc_ros phase2.launch.py]
         │
         ├─ TelemetryNode  ← MAVROS 구독 상시 실행
         │
         └─ OffboardNode 상태머신
                │
                ▼ ARM + AUTO.TAKEOFF 명령        ← ARM은 여기서 1회 (STREAMING 아님)
          [ARM_TAKEOFF]
                │
                ▼ pos_ned[2] >= transition_alt 확인
          [CLIMBING]
                │
                ▼ MAV_CMD_DO_VTOL_TRANSITION(param1=3: MC→FW)
          [TRANSITION_FW]
                │
                ▼ vtol_state == FW 확인 후 OFFBOARD 모드 전환
          [STREAMING]  ← 더미 세트포인트 발행 (PX4 watchdog). FW 상태이므로 전진속도(0 금지)
                │
                ▼ (entry_mode == "mid_flight" 일 때만)
          [ENTRY]
                │
                ▼ L1 Guidance, eta3 경로
          [FOLLOWING]
                │
                ▼ dist_to_end < d_end_thresh (v_profile이 작업 B로 v_terminal까지 감속됨)
          [TRANSITION_MC]  ← MAV_CMD_DO_VTOL_TRANSITION(param1=4: FW→MC)
                │
                ▼ vtol_state == MC 확인
          [LANDING]  ← AUTO.LAND 명령
                │
                ▼ disarmed 확인
          [DONE]
```

> **상태 변경 요약 (기존 코드 대비):**
>
> - 신규 상태: `ARM_TAKEOFF`, `CLIMBING`, `TRANSITION_FW`, `TRANSITION_MC`, `LANDING`
> - 기존 유지: `IDLE`, `STREAMING`, `ENTRY`, `FOLLOWING`, `DONE` (+ `_step_entry`/`_step_following` 핸들러)
> - **STREAMING 리팩터**: 현재 STREAMING은 OFFBOARD 전환 **+ ARM**을 함께 수행한다. ARM은 `ARM_TAKEOFF`로 이동하므로 **STREAMING에서 ARM 요청을 제거**한다(이중 ARM 방지).

---

# 코드 작업단위

---

## 작업 A — params/YAML 정비

**유형:** [코드] (Claude 자율)
**목적:** 모든 노드 파라미터를 정상화하고 신규 파라미터를 추가한다. 운용 고도를 일원화한다.
**선행:** 없음

**파일:**

- [fc_ros/fc_ros/params/fc_ros_params.yaml](../fc_ros/fc_ros/params/fc_ros_params.yaml)
- [fc_ros/fc_ros/nodes/offboard_node.py](../fc_ros/fc_ros/nodes/offboard_node.py) (`declare_parameter`)

**작업 목록:**

1. **`waypoints` 2D → 1D flat (기존 버그).** `offboard_node`, `mission_node` 양쪽.
   ROS2 파라미터는 중첩 리스트를 지원하지 않아 현재 YAML(`- [0,0,150]` 형식)은 launch 시 TypeError를 유발한다.
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

**배경:** `run_planner()`→eta3/diterpin은 `v_ref=v_cruise`로 고정한다([eta3clothoid_v3_1_planner.py:469](../vtol_sim_checkpoint1_1/vtol_sim/path_planning/eta3clothoid_v3_1_planner.py#L469), [D_iterpin_planner.py:346](../vtol_sim_checkpoint1_1/vtol_sim/path_planning/D_iterpin_planner.py#L346)). `v_terminal`을 읽지 않으므로, **플래너 수정이 아니라 결과 v_profile에 후처리**를 적용한다.

**파일:**

- `fc_bridge/planning/terminal_decel.py` (신규)
- [fc_ros/fc_ros/nodes/offboard_node.py](../fc_ros/fc_ros/nodes/offboard_node.py) (`main()` 배선)

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
**선행:** 작업 A. vtol_state 상수는 [기술 참조](#vtol_state-상수-mavros_msgsextendedstate)의 값을 사용하고 **SITL-1에서 확정**한다.

**파일:** [fc_ros/fc_ros/nodes/offboard_node.py](../fc_ros/fc_ros/nodes/offboard_node.py)

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

**테스트:** [fc_ros/test/test_offboard_node.py](../fc_ros/test/test_offboard_node.py)에 순수 로직 케이스 추가 (기존 `_entry_done` 패턴):

```python
def _climb_reached(pos_ned_up, transition_alt):
    return pos_ned_up >= transition_alt

def test_climb_reached(): assert _climb_reached(50.1, 50.0) is True
def test_climb_not_reached(): assert _climb_reached(49.9, 50.0) is False

def _vtol_is_fw(vtol_state, FW=4): return vtol_state == FW
def test_transition_fw_done(): assert _vtol_is_fw(4) is True
```

**합격 기준:** `pytest fc_ros/test/test_offboard_node.py` 통과.

---

## 작업 D — 상태머신 ② 역천이·착륙

**유형:** [코드] (Claude 자율)
**목적:** 경로 끝 도달 → FW→MC 역천이 → AUTO.LAND를 구현한다.
**선행:** 작업 C

**파일:** [fc_ros/fc_ros/nodes/offboard_node.py](../fc_ros/fc_ros/nodes/offboard_node.py)

**작업 목록:**

1. **`_State` enum에 추가:** `TRANSITION_MC`, `LANDING`.
2. **`_step_following()` 종료조건 변경:**
   - 현재: `dist_to_end < 3.0` (하드코딩) → `DONE`.
   - 변경: `dist_to_end < self._d_end_thresh` → `TRANSITION_MC` 반환.
   - **감속은 작업 B의 v_profile이 담당**한다. L1 Guidance는 v_profile을 따라 끝점에서 v_terminal로 자연 감속한다. controller-level 속도 clamp를 mid-flight에 적용하지 않는 이유: L1은 속도·기하를 동시에 사용하므로 중간 clamp 시 cross-track error가 증가한다.
3. **상태 핸들러 구현:**
   - `_step_transition_mc()`: `CommandLong(command=3000, param1=3.0)` 호출 후 `vtol_state == VTOL_STATE_MC` 확인 → LANDING.
   - `_step_landing()`: set_mode `"AUTO.LAND"` → `not state.armed`(disarmed) 확인 → DONE. `landing_timeout` 초과 시 경고 로그.

**테스트:** [fc_ros/test/test_offboard_node.py](../fc_ros/test/test_offboard_node.py)에 추가:

```python
def _trans_mc_trigger(dist_to_end, d_end_thresh): return dist_to_end < d_end_thresh
def test_trans_mc_trigger(): assert _trans_mc_trigger(9.0, 10.0) is True
def test_trans_mc_not_yet(): assert _trans_mc_trigger(11.0, 10.0) is False

def _landing_done(armed): return not armed
def test_landing_done(): assert _landing_done(False) is True
```

**합격 기준:** `pytest fc_ros/test/test_offboard_node.py` 통과.

---

## 작업 E — 긴급 수동 override

**유형:** [코드] (Claude 자율)
**목적:** `/fc_ros/override` 토픽으로 자동비행을 즉시 중단하고 수동 모드로 전환한다.
**선행:** 작업 C (vtol_state 분기 사용)

**파일:** [fc_ros/fc_ros/nodes/offboard_node.py](../fc_ros/fc_ros/nodes/offboard_node.py)

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

> **RC 레이어와 독립:** PX4 `COM_RC_OVERRIDE`(SITL-1에서 설정)는 ROS2 override와 별개로 동작하는 하드웨어 레이어다. 두 레이어 모두 OFFBOARD 진입 전/후 어느 상태에서도 독립 동작해야 한다. 상세는 [안전 및 긴급 수동 전환](#안전-및-긴급-수동-전환) 참조.

> ⚠ **SITL-4 정정 (2026-06-30):** 위 코드처럼 `_request_override`가 곧장 `_sm=DONE`으로 가면 모드 전환 거부 시 cmd_vel velocity-0 발행이 OFFBOARD를 살려둬 **FW가 직진 폭주**한다(SITL-4 실측). 또 MANUAL/POSCTL은 RC·조이스틱 같은 수동제어 소스가 필요해 **headless SITL에선 거부**된다(SITL-1 `COM_RC_OVERRIDE`→POSCTL 재현불가와 동일).
> → 현재 구현은 **`_State.OVERRIDE`** 를 두고: ⓐ override 시 OFFBOARD setpoint 발행 중단, ⓑ manual 모드 시도, ⓒ 1초 내 미진입이면 **AUTO.LOITER 안전 폴백** 강제. 실기체에선 조종사 RC로 manual이 즉시 잡혀 폴백 전 종료. 판정함수 `override_reached`/`override_fallback_due`(state_logic).

**테스트:** [fc_ros/test/test_offboard_node.py](../fc_ros/test/test_offboard_node.py)에 추가:

```python
def _override_mode(vtol_state, MC=3): return "POSCTL" if vtol_state == MC else "MANUAL"
def test_override_mc(): assert _override_mode(3) == "POSCTL"
def test_override_fw(): assert _override_mode(4) == "MANUAL"
```

**합격 기준:** `pytest fc_ros/test/test_offboard_node.py` 통과.

---

# SITL 검증 게이트

> 모든 SITL 게이트는 **사람이 WSL에서 수행**한다. Claude는 절차·관찰 포인트를 제시하고 로그로 판정한다.
> 공통 기동 절차·QGC 연결은 `docs/session_status.md`, `docs/sitl_verification_log.md` 참조.

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

- [ ] vtol_state 상수값 실측 기록 (MC/FW/천이) — 작업 C/D/E가 참조
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

- [ ] dry-run 속도 프로파일 끝점 = v_terminal (작업 B 동작 확인)
- [ ] 3종 경로 모두 FOLLOWING 진입 및 끝점 도달
- [ ] 각 경로 cross_track_error 기준 이내 (코너 진입 전 감속 확인 — L자)

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

## SITL-5 — RPi4 실기체 배포

**목적:** 실기체 배포.
**선행:** SITL-4

**절차 (사람):**

1. RPi4에 ROS2 Humble + MAVROS apt 설치
2. `fc_bridge`, `fc_ros` 배포 (colcon build)
3. 실기체 PX4 연결 테스트 (텔레메트리)
4. 이륙 전 체크리스트는 [실기체 배포 필수 조정 파라미터](#실기체-배포-시-필수-조정-파라미터-체크리스트-sitl-5) 참조.

---

# 후속 계획 — 임의 WP 경로 생성·추종 검증

> **시점:** SITL-4(전체 사이클 통합) **완료 이후**. 본 계획의 핵심 범위(단방향 전체 비행 사이클) **밖**이다.
> 대회 전체 미션(왕복 + 복수 사이클)과 vision→FC 연동(`pixel_to_gps`로 임의 GPS WP 주입)으로 가는 다리.

**배경:** `OffboardNode.main()`은 시작 시 `waypoints` 파라미터를 1회 읽어 `run_planner`로 경로를 생성하고 추종한다.
따라서 **임의 WP를 launch 시점에 주입해 생성→추종**하는 것은 이미 가능하다(`ros2 launch ... -p waypoints:=[...]` 또는 YAML 교체).
미검증 영역은 (a) 다양한 WP 조합에 대한 **경로 생성 견고성**, (b) **런타임 WP 주입(재계획)** 이다.

---

## 작업 F — 임의 WP 경로 생성 견고성 하니스

**유형:** [코드] (Claude 자율)
**목적:** 임의/무작위 WP 세트가 항상 유효한 경로로 생성되는지 자동 검증한다.
**선행:** 작업 B (`apply_terminal_decel`). SITL-4 완료 후 진입 권장.

**작업 목록:**

1. `fc_bridge/tests/test_arbitrary_wp.py` (신규) — 무작위 유효 WP 세트(레그 수 2~5, 레그 길이 50~300 m, 회전각 가변)를 생성해 `run_planner` + `apply_terminal_decel`을 통과시키고 경로 불변식을 검증.
2. (선택) **런타임 WP 주입 인터페이스** — `/fc_ros/waypoints` 토픽/서비스로 WP를 받아 재계획. vision→FC 연동의 기반. 본 작업에선 인터페이스 골격 + 단위 테스트만, SITL 통합은 SITL-6.

**테스트:** `fc_bridge/tests/test_arbitrary_wp.py`

```python
def test_arbitrary_wp_path_invariants():
    for _ in range(20):
        wps = random_valid_waypoints()                  # 2~5 WP, 50~300 m 레그
        path = run_planner("eta3", wps, vehicle_params)
        s = np.array([p.s for p in path.points])
        v = apply_terminal_decel(
            np.array([p.v_ref for p in path.points]), s, v_terminal, decel_dist)
        assert np.all(np.diff(s) >= 0)                   # 호길이 단조 증가
        assert v[-1] == pytest.approx(v_terminal)        # 끝점 감속
        assert np.all(v <= vehicle_params["v_cruise"] + 1e-6)  # 속도 상한
        kappa = np.abs([p.curvature for p in path.points])
        assert np.all(kappa <= a_max / vehicle_params["v_cruise"]**2 + 1e-3)  # 곡률 상한
```

**합격 기준:** `pytest fc_bridge/tests/test_arbitrary_wp.py` 통과.

---

## SITL-6 — 임의 WP 생성·추종 SITL

**유형:** [SITL] (사람 + Claude 보조)
**목적:** launch 시점에 임의 WP 세트를 주입해 생성→전체 사이클 추종을 SITL 검증한다.
**선행:** 작업 F · SITL-4

**절차 (사람):**

1. 임의/무작위 WP 세트 3~5종을 `-p waypoints:=[...]`로 주입.
2. 각 세트에 대해 `ros2 launch fc_ros phase2.launch.py` → 이륙~착륙 전체 사이클 추종.
3. WP 통과 정확도(기체 GPS) + 끝점 감속 로그 확인.

**합격 기준 (체크리스트):**

- [ ] 임의 WP 세트 전부 경로 생성 성공 (런타임 오류 없음)
- [ ] 전부 FOLLOWING 정상 진입 및 전체 사이클 완료
- [ ] WP 통과 오차 기준 이내, 끝점 v_terminal 도달 확인

---

## 완료된 항목 (참조)

- [x] WSL SITL 구축 (PX4 + MAVROS + ROS2 Humble, 기체: gz_x500)
- [x] QGC ↔ WSL SITL 연결 (UDP 14551)
- [x] hover_node SITL 검증 (2026-06-01)
- [x] OffboardNode 기본 검증 (x500, STREAMING→FOLLOWING→DONE, 2026-06-06)
- [x] TelemetryNode 단위 테스트 25/25 PASS (2026-06-06)
- [x] TelemetryNode SITL 검증 (2026-06-17)
- [x] MissionNode SITL 검증 (버그 3개 수정, 2026-06-18)

## 미완료 항목

- [x] 작업 A: params/YAML 정비 [코드] — 2026-06-19 완료
- [x] 작업 B: 종단 감속 헬퍼 + 배선 [코드] — 2026-06-20 완료
- [x] 작업 C: 상태머신 ① 이륙·상승·천이 [코드] — 2026-06-20 완료
- [x] 작업 D: 상태머신 ② 역천이·착륙 [코드] — 2026-06-20 완료
- [x] 작업 E: 긴급 수동 override [코드] — 2026-06-20 완료
- [x] SITL-1: VTOL 환경 전환 + 상수 확인 [SITL] — 2026-06-19 조건부 PASS
- [x] SITL-2: launch 통합 기동 [SITL] — 2026-06-20 완료
- [x] SITL-3: 경로 추종 검증 [SITL] — PASS 2026-06-30 (FW 위치 setpoint 전환)
- [x] SITL-4: 전체 사이클 통합 [SITL] — PASS 2026-06-30 (직선+L자; override는 AUTO.LOITER 폴백 추가, manual 인계는 SITL-5)
- [ ] SITL-5: RPi4 배포 [배포]
- [ ] 작업 F: 임의 WP 경로 생성 견고성 하니스 [코드] (후속, SITL-4 이후)
- [ ] SITL-6: 임의 WP 생성·추종 SITL [SITL] (후속, SITL-4 이후)

---

## 주요 기술 참조

### VTOL 천이 MAVLink 명령

```
MAV_CMD_DO_VTOL_TRANSITION = 3000
param1 = 4  (MC → FW)   ← 목표 상태 FW(4)로 전환. ✅ SITL-1 실측 확인 (2026-06-19)
param1 = 3  (FW → MC)   ← 목표 상태 MC(3)로 전환. ✅ SITL-1 실측 확인 (2026-06-19)
```

MAVROS 호출: `/mavros/cmd/command` (mavros_msgs/srv/CommandLong)
응답: `result=0` = MAV_RESULT_ACCEPTED (성공)

### vtol_state 상수 (mavros_msgs/ExtendedState)

```python
VTOL_STATE_UNDEFINED        = 0   # 미사용
VTOL_STATE_TRANSITION_TO_FW = 1   # ✅ SITL-1 실측 확인
VTOL_STATE_TRANSITION_TO_MC = 2   # ✅ SITL-1 실측 확인
VTOL_STATE_MC               = 3   # ✅ SITL-1 실측 확인 — 작업 C/D/E 사용
VTOL_STATE_FW               = 4   # ✅ SITL-1 실측 확인 — 작업 C/D/E 사용
```

> **확정값 (2026-06-19 SITL-1 실측). 작업 C/D/E는 이 값을 그대로 사용한다.**

### AUTO.TAKEOFF 동작 (SITL-1 실측)

```
ARM (CommandBool true)
  → set_mode "AUTO.TAKEOFF"
  → PX4: takeoff detected → 목표 고도까지 자율 상승
  → 완료 후 HOLD 모드로 전환   ← ✅ 확정 (2026-06-19)
```

작업 C `_step_arm_takeoff()` → `_step_climbing()` 이후 HOLD 상태에서 VTOL 천이 명령 발행.

### 고도 판정 주의

```python
# VehicleState.pos_ned[2] = h_up (양수 = 고도 증가, NED D축과 반대 부호)
if state.pos_ned[2] >= self._transition_alt:   # CLIMBING 판정
    # 천이 고도 도달
```

---

## 파라미터 튜닝 가이드

> **시점:** SITL-4(전체 통합 검증) 완료 후, SITL-5(실기체 배포) 직전에 수행한다.
> 로직 구현([작업 A~E]) 전에 이 파라미터를 건드리지 않는다.

---

### 천이 최대 가속도 ≤ 0.3g (≈ 2.94 m/s²)

천이 중 가속도는 두 원인에서 발생한다:

1. MC→FW 천이 시 PX4 내부 자세/추력 전환
2. FW→MC 역천이 시 고속 상태에서 MC가 제동하는 충격

역천이(2번)가 더 위험하다. 이를 제어하는 핵심 수단은 **작업 B의 종단 감속**(`apply_terminal_decel` → 끝점 속도 v_terminal)이다. OffboardNode에는 `d_pre_trans`/`v_transition_max` 같은 제어기 감속 파라미터가 **없다** (설계상 감속은 경로 후처리가 전담).

**조정 순서:**

| 단계 | 작업                                                   | 확인 방법                                                             |
| ---- | ------------------------------------------------------ | --------------------------------------------------------------------- |
| 1    | SITL에서 역천이 직전 속도 로그 확인                    | `vel_ned` 크기 출력                                                   |
| 2    | `v_terminal` 조정 (= 스톨 × 1.1 = 15.2 m/s)            | v_profile 끝점 속도. 낮출수록 천이 충격 감소, 단 스톨(13.8) 이하 금지 |
| 3    | `decel_dist` 조정 (기본 80 m → 순항속도 빠를수록 길게) | dry-run 속도 프로파일에서 감속 시작 시점 확인                         |
| 4    | IMU `/mavros/imu/data` 로 천이 중 가속도 측정          | `linear_acceleration` 크기 ≤ 2.94                                     |
| 5    | 실기체 동일 측정 후 PX4 파라미터 추가 조정             | QGC 파라미터 편집기                                                   |

**관련 PX4 파라미터 (QGC):**

| PX4 파라미터       | 기본값   | 역할                       |
| ------------------ | -------- | -------------------------- |
| `VT_ARSP_TRANS`    | 10 m/s   | MC→FW 천이 시작 에어스피드 |
| `VT_TRANS_TIMEOUT` | 15 s     | 천이 타임아웃              |
| `VT_B_TRANS_DUR`   | 4 s      | 역천이 최대 지속 시간      |
| `VT_B_DEC_MSS`     | 1.0 m/s² | 역천이 중 목표 감속도      |

> `VT_B_DEC_MSS`를 줄이면 역천이가 부드러워지지만 고도 손실이 커진다. 실기체에서 교환관계 확인.

---

### WP 통과 위치 오차 최소화 (평가: 기체 GPS 값)

평가가 기체 GPS 값 기준이라 GPS 절대 편향은 상쇄되어 RTK 불필요.
실질 오차 원인: L1 Guidance lookahead(`l1_dist`)와 코너 통과 속도.

| 단계 | 작업                                   | 비고                  |
| ---- | -------------------------------------- | --------------------- |
| 1    | SITL-3에서 cross_track_error 로그 수집 | FOLLOWING 중 `cte`    |
| 2    | `l1_dist` 감소 (기본 20 m → 10~15 m)   | 너무 낮으면 진동      |
| 3    | `v_cruise` 감소 테스트                 | 느릴수록 오차↓ 시간↑  |
| 4    | eta3 WP 통과 반경 확인                 | `fc_bridge/planning/` |
| 5    | 코너 WP 감속 프로파일 확인             | v_profile 코너 속도   |

**L1 동적 lookahead (향후 개선):** WP 직전 N m 이내에서 `l1_dist`를 속도 비례로 줄이면 정밀도 향상. 위치: `fc_bridge/guidance/l1_guidance.py` 또는 `_step_following()`. 현재 계획 미포함 — SITL-3 결과 보고 결정.

---

### 첫 비행 전 지상 안전 테스트 (SITL-5, 비행 직전)

> 프로펠러 제거 또는 기체를 고정한 상태에서 수행. Layer 1/2 동시 검증.

```
[ ] COM_RC_OVERRIDE = 3 QGC 파라미터 확인
[ ] ARM → setpoint 20Hz 발행 → OFFBOARD 진입 확인 (QGC 모드 표시)
[ ] RC 스틱 중립 이탈 → QGC에서 POSCTL 즉시 전환 확인 (Layer 1)
[ ] OFFBOARD 재진입 → /fc_ros/override 발행 → POSCTL/MANUAL 전환 + setpoint 중단 확인 (Layer 2)
[ ] 두 레이어 동시 트리거 → 충돌 없음 확인
```

> **이 테스트 통과 전 이륙 금지.**

---

### 실기체 배포 시 필수 조정 파라미터 체크리스트 (SITL-5)

```
[ ] home_lat / home_lon — 실제 이륙 지점 GPS (현재 기본: 스위스 취리히)
[ ] transition_alt — 실제 운용 고도 (법규·대회 규정)
[ ] v_cruise — 실기체 최적 순항 속도 (풍속 고려)
[ ] l1_dist — 실기체 비행 특성
[ ] v_terminal — 실기체 스톨 × 1.1 이상 (기준: 13.8 × 1.1 = 15.2 m/s)
[ ] decel_dist — 종단 감속 구간 길이 (v_cruise 클수록 길게)
[ ] VT_ARSP_TRANS — 실속 속도 + 여유값
[ ] VT_B_DEC_MSS — 역천이 감속도 (0.3g 조건 준수)
[ ] COM_RC_OVERRIDE = 3 PX4 적용 확인
[ ] RC 모드 스위치 → POSCTL 채널 연결 확인
[ ] /fc_ros/override 토픽 → 모드 전환 동작 테스트
[ ] RC 안전 스위치 채널 확인
[ ] 배터리 전압 임계값 확인
```

---

## 안전 및 긴급 수동 전환

> **현재 목표 (MVP):** 언제든 자동비행을 즉시 중단하고 수동 모드로 전환할 수 있어야 한다.
> **최종 목표:** 다층 failsafe (하드웨어·PX4·ROS2 레이어 분리, 비전 기반 이상 탐지) — 별도 계획.

### 레이어 구조

```
Layer 1 (PX4 하드웨어): RC 송신기 모드 스위치
        ↓ COM_RC_OVERRIDE=3 적용 시
        PX4가 OFFBOARD 무시 → POSCTL (MC) 또는 RC 모드 (FW)
        → 가장 빠르고 신뢰성 높음 (ROS2 스택과 무관)

Layer 2 (ROS2 소프트웨어): /fc_ros/override 토픽
        ↓ ros2 topic pub --once /fc_ros/override std_msgs/Bool "data: true"
        OffboardNode(_State.OVERRIDE): OFFBOARD setpoint 발행 중단 + manual 모드 요청
        (MC→POSCTL, FW→MANUAL) → 미진입(RC 없음) 시 AUTO.LOITER 안전 폴백
        → 키보드 터미널 또는 스크립트에서 트리거
```

### 각 레이어 설정

**Layer 1 — PX4 (SITL-1, QGC에서 1회):** `COM_RC_OVERRIDE = 3`, RC 채널 → POSCTL/HOLD 매핑.

**Layer 2 — ROS2 (작업 E, OffboardNode 통합):** `_cb_override()` 구현, vtol_state 분기, `_sm = OVERRIDE`로 setpoint 즉시 중단 + manual 모드 요청 → 미진입 시 AUTO.LOITER 폴백 (SITL-4 정정).

### 동작 보장 조건

- OFFBOARD 진입 전/후 어느 상태에서도 두 레이어 모두 독립 동작.
- Layer 1(RC)은 Layer 2(ROS2) 의존 없이 단독 동작.
- Layer 2 실패(ROS2 크래시 등) 시 Layer 1이 최후 수단.

### 동작 정의

| 상황        | 트리거                              | 결과                           |
| ----------- | ----------------------------------- | ------------------------------ |
| MC 모드 중  | RC 모드스위치 또는 /fc_ros/override | POSCTL — 제자리 hover hold     |
| FW 모드 중  | RC 모드스위치 또는 /fc_ros/override | MANUAL — 조종사 직접 조작      |
| 어느 상태든 | RC 스틱 입력 (COM_RC_OVERRIDE)      | PX4가 즉시 POSCTL/RC 모드 전환 |

### MC 모드 POSCTL hold 확인 항목 (SITL-1에서 검증)

```
[ ] QGC 수동 이륙 후 OFFBOARD 진입 상태에서 RC 모드 스위치 → POSCTL 즉시 전환
[ ] /fc_ros/override 발행 후 set_mode POSCTL 적용 및 setpoint 중단 확인
[ ] 두 레이어 동시 트리거 시 충돌 없음 확인
```
