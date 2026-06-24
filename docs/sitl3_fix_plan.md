# SITL-3 수정 계획: FW 천이 직선화 + 위치 기반 경로 추종

> 작성 목적: 세션 간 컨텍스트 손실 없이 두 핵심 버그를 완전히 수정하기 위한 설명 문서.  
> 이 두 버그는 SITL-3 통과의 필수 조건이다.

---

## 1. 시스템 구조 (변경 전 기준)

### 1-1. 핵심 파일 맵

| 파일 | 역할 |
|---|---|
| `fc_ros/fc_ros/nodes/offboard_node.py` | **주 수정 대상**. ROS2 상태머신 노드 |
| `fc_ros/fc_ros/adapters/setpoint_publisher.py` | NED 속도 → ENU TwistStamped 변환 후 발행 |
| `fc_ros/fc_ros/adapters/vehicle_state_bridge.py` | MAVROS 메시지 → VehicleState(NED) 변환 |
| `fc_bridge/guidance/l1_guidance.py` | **보조 수정 대상**. L1 Guidance 계산기 |
| `fc_ros/fc_ros/params/fc_ros_params.yaml` | ROS2 파라미터 (wp0_heading_tol: 0.05 rad) |

### 1-2. 좌표 프레임 규칙

**내부 표현은 전부 NED다.**

| 값 | 표현 | 단위 |
|---|---|---|
| `state.pos_ned` | `[N, E, h_up]` — h_up 양수=위 | m |
| `state.vel_ned` | `[vN, vE, vD]` — vD 양수=아래 | m/s |
| `state.yaw` | NED 헤딩. 0=북, π/2=동, CW 양수 | rad |
| `chi_cmd` (L1 출력) | NED 코스각. `arctan2(vE, vN)` 동일 규칙 | rad |

**MAVROS ↔ PX4 경계에서 ENU 변환이 일어난다.**

`vehicle_state_bridge.py`:
```python
# 구독 (PX4 → 코드): ENU → NED
state.pos_ned = [p.y, p.x, p.z]          # y_enu=N, x_enu=E, z_enu=h_up
state.vel_ned = [v.y, v.x, -v.z]         # y_enu=vN, x_enu=vE, z_enu=vU
yaw_ned = pi/2 - yaw_enu                 # ENU 0°=동 → NED 0°=북
```

`setpoint_publisher.py`:
```python
# 발행 (코드 → PX4): NED → ENU
twist.linear.x = vel_ned[1]   # vE → x_enu
twist.linear.y = vel_ned[0]   # vN → y_enu
twist.linear.z = -vel_ned[2]  # -vD → z_enu(Up)
twist.angular.z = yaw_rate    # ENU: CCW 양수
```

**yaw_rate 부호 규칙** (검증됨):
- NED heading_err > 0 → CW 회전 필요 → ENU angular.z 음수
- `yaw_rate = -heading_err * gain` ← 이 부호가 올바르다

### 1-3. 현재 상태머신 흐름

```
ARM_TAKEOFF → CLIMBING → TRANSITION_FW → STREAMING → FOLLOWING → TRANSITION_MC → LANDING → DONE
```

`TRANSITION_FW` 내부 단계 (현재 코드):
```
Phase 1: 20틱 hover 세트포인트 → MC OFFBOARD 요청
Phase 2: MC OFFBOARD hover + yaw rate P제어 (gain 0.3, 포화 ±0.5)
         abs(heading_err) < wp0_htol(0.05rad) 로 20틱 안정 시 → aligned=True
Phase 3 (1틱): aligned=True → fwd_vel 발행 + MAV_CMD_DO_VTOL_TRANSITION 발행
Phase "ACTIVE TRANSITION" (이후 매틱): fwd_vel + yaw_rate_hold(gain 0.5) 발행
Phase 4: vtol_state==4(FW) → STREAMING 전환
```

`STREAMING` (현재 코드):
```python
vel_cmd = guidance.ned_velocity_cmd(pos_ned, vel_ned)  # L1 속도 벡터
setpoint.publish(vel_cmd)
if OFFBOARD and vel_aligned_with_path(vel, pts, cos_thresh=0.966):
    → FOLLOWING
```

`FOLLOWING` (현재 코드):
```python
chi_cmd, v_cmd, cte = guidance.compute(pos, vel)
vel_cmd = [v_cmd*cos(chi_cmd), v_cmd*sin(chi_cmd), -v_cmd*sin(gamma)]
setpoint.publish(vel_cmd)
```

---

## 2. 문제 1: MC→FW 천이 중 원호운동 (최대 160° 헤딩 이탈)

### 2-1. 증상

- `MAV_CMD_DO_VTOL_TRANSITION` 발행 직후 기체가 약 4초간 원호를 그리며 비행
- 천이 완료 시 헤딩이 최대 160° 틀어진 상태로 FW 진입
- **단 한 번도 직선 천이가 된 적 없음**

### 2-2. 근본 원인

PX4 VTOL 천이(`vtol_state==1`)는 내부 transition controller가 구동된다. 이 controller는:

1. MC 모터를 점진적으로 감소
2. FW(pusher) 모터를 점진적으로 증가
3. 에어스피드가 `VT_ARSP_TRANS`에 도달하면 완전 FW 전환

**이 과정에서 OFFBOARD velocity setpoint의 방향 제어 권한이 현저히 약화된다.** 정지 상태(hover)에서 천이를 시작하면:
- FW 추진력이 드론을 밀지만 방향 제어가 불안정
- 제어 권한이 MC↔FW 사이에서 분산되는 동안 yaw 교란에 취약
- 결과: 원호 운동

**현재 ACTIVE TRANSITION 블록으로는 이 문제를 해결할 수 없다.** PX4 transition controller가 velocity setpoint 방향을 무시하기 때문이다. yaw_rate 명령도 MC 권한이 줄어드는 동안 효과가 약해진다.

### 2-3. 해결 방법: 사전 가속 후 천이 (Phase 2.5)

**핵심 원리**: 드론이 이미 경로 방향으로 충분한 속도를 가진 상태에서 천이를 시작하면, 공기역학적 힘이 기체를 직선으로 유지해준다. 정지 → 가속 중 천이가 아닌, 이미 순항 속도로 비행 중 천이.

**수정 순서**:
```
Phase 2: 헤딩 정렬 (yaw rate P제어, 기존과 동일)
Phase 2.5 (신규): MC OFFBOARD에서 fwd_vel 발행 → 실제 속도가
                  경로 방향 && v >= _PREACCEL_SPEED(10 m/s)가 될 때까지 대기
Phase 3: 사전 가속 완료 → MAV_CMD_DO_VTOL_TRANSITION 발행
Phase 4: vtol_state==4 → STREAMING
```

**Phase 2.5 중 주의**: 이 단계에서 yaw rate는 발행하지 않는다. MC OFFBOARD에서 fwd_vel(경로 방향 전진 속도)을 발행하면 PX4가 자동으로 속도 방향으로 기수를 맞춘다.

### 2-4. 구현: `offboard_node.py`

#### 추가 상수 (파일 상단 모듈 레벨)
```python
_FW_STABLE_REQ = 20       # 기존 유지
_PREACCEL_SPEED = 10.0    # m/s, 천이 전 최소 전진 속도
_PREACCEL_ALIGN = 0.174   # rad ≈ 10°, 속도 벡터 방향 허용 오차
```

#### 추가 인스턴스 변수 (`__init__` 내 기존 플래그 다음에)
```python
self._fw_preaccel_done = False  # Phase 2.5 완료 여부
```

#### `_step_transition_fw` 내부 수정

현재 Phase 2 fall-through → Phase 3 사이에 Phase 2.5를 삽입한다.

**변경 전** (현재 코드):
```python
        # Phase 3: 헤딩 정렬 완료 — WP 방향 전진 + 천이 명령
        v_fwd = float(self._v[0]) if len(self._v) > 0 else 15.0
        fwd_vel = np.array([seg[0] * v_fwd, seg[1] * v_fwd, 0.0])
        self._setpoint.publish(fwd_vel)  # OFFBOARD keepalive + 전진

        if not self._fw_transition_sent:
            ...
            self._fw_transition_sent = True
```

**변경 후**:
```python
        # Phase 2.5: 헤딩 정렬 완료 → 사전 가속 (천이 전 충분한 전진 속도 확보)
        # 정지 상태에서 천이하면 원호를 그리므로, 이미 경로 방향으로 비행 중일 때 천이한다.
        if not self._fw_preaccel_done:
            v_fwd = float(self._v[0]) if len(self._v) > 0 else 15.0
            fwd_vel = np.array([seg[0] * v_fwd, seg[1] * v_fwd, 0.0])
            self._setpoint.publish(fwd_vel)  # yaw_rate 없음: PX4가 속도 방향으로 기수 정렬

            speed_ne = float(np.linalg.norm(state.vel_ned[:2]))
            if speed_ne >= 1.0:
                chi_vel = float(np.arctan2(state.vel_ned[1], state.vel_ned[0]))
                vel_err = abs(_wrap(chi_wp - chi_vel))
            else:
                vel_err = float('inf')

            if speed_ne >= _PREACCEL_SPEED and vel_err < _PREACCEL_ALIGN:
                self._fw_preaccel_done = True
                self.get_logger().info(
                    f"사전 가속 완료 speed={speed_ne:.1f}m/s "
                    f"vel_err={np.degrees(vel_err):.1f}° → 천이 명령 준비")
            else:
                self.get_logger().debug(
                    f"사전 가속 중 speed={speed_ne:.1f}/{_PREACCEL_SPEED:.0f} "
                    f"vel_err={np.degrees(vel_err):.1f}°")
            return

        # Phase 3: 사전 가속 완료 → 천이 명령 (이미 전진 중이므로 직선 천이 기대)
        v_fwd = float(self._v[0]) if len(self._v) > 0 else 15.0
        fwd_vel = np.array([seg[0] * v_fwd, seg[1] * v_fwd, 0.0])
        self._setpoint.publish(fwd_vel)

        if not self._fw_transition_sent:
            if not self._cmd_cli.service_is_ready():
                self.get_logger().warn("/mavros/cmd/command 서비스 없음")
                return
            req = CommandLong.Request()
            req.command = 3000
            req.param1 = 4.0
            self._cmd_cli.call_async(req)
            self._fw_transition_sent = True
            self.get_logger().info("MC→FW 천이 명령 요청 (사전 가속 완료 후 직선 천이)")
```

#### ACTIVE TRANSITION 블록은 그대로 유지

현재 삽입된 ACTIVE TRANSITION 블록(Phase 3 앞, Phase 1 뒤)은 그대로 유지한다. `_fw_preaccel_done=True && _fw_transition_sent=True` 조합으로만 동작하므로 Phase 2.5와 충돌하지 않는다. (ACTIVE TRANSITION 조건: `_fw_heading_aligned and _fw_transition_sent`)

#### 체크리스트
- [ ] `_fw_preaccel_done = False` → `__init__` 에 추가
- [ ] `_PREACCEL_SPEED = 10.0`, `_PREACCEL_ALIGN = 0.174` → 모듈 상단에 추가
- [ ] Phase 2.5 블록 삽입 (Phase 2 fall-through 이후, Phase 3 이전)
- [ ] Phase 3 로그 메시지 업데이트

---

## 3. 문제 2: FW OFFBOARD에서 경로 추종 미작동

### 3-1. 증상

- FOLLOWING 진입 시 `chi_cmd`가 계산되어도 드론은 방향을 바꾸지 않음
- 처음 FW 진입 시의 헤딩 그대로 직선 비행 → 경로 전체가 그 헤딩만큼 틀어짐
- L1이 올바른 보정 방향을 계산해도 아무런 효과 없음

### 3-2. 근본 원인

**PX4 FW OFFBOARD 모드에서 `/mavros/setpoint_velocity/cmd_vel`의 속도 방향이 코스 명령으로 변환되지 않는다.**

`setpoint_velocity/cmd_vel`은 MAVROS를 통해 PX4에 `SET_POSITION_TARGET_LOCAL_NED`(velocity mask)로 전달된다. PX4 **FW** controller는 이 velocity setpoint를 받았을 때:
- **속도 크기** → 스로틀 제어 (반영됨)
- **속도 방향** → FW 롤/러더 제어 (제대로 반영 안 됨)

결과: 드론은 처음 자세 그대로 날면서 속도 크기만 맞추고, L1이 계산한 코스 보정은 완전히 무시된다.

검증: L1 `compute()` 반환 chi_cmd가 매 틱 다른 값을 반환해도, 드론이 방향을 바꾸지 않으면 velocity setpoint 방향이 무시되는 것이다.

### 3-3. 해결 방법: Moving Lookahead Position Setpoint

**핵심 원리**: velocity setpoint 대신 `/mavros/setpoint_position/local`(PoseStamped)로 **L1 lookahead 위치**를 발행한다. PX4 FW는 이 위치를 자체 autopilot으로 추종한다. 위치 setpoint는 FW 모드에서 확실하게 코스 제어를 유발한다.

**동작 원리**:
1. 현재 드론 위치에서 경로를 따라 L1_dist(20m) 앞의 점을 계산
2. 그 점을 position setpoint로 발행
3. PX4 FW가 그 위치를 향해 비행 (롤 제어로 코스 조정)
4. 드론이 이동함에 따라 lookahead 위치가 항상 20m 앞으로 이동 → L1 guidance와 동일 효과
5. 드론이 경로를 벗어나면 lookahead가 경로 방향으로 당겨지므로 자동 보정

### 3-4. 구현

#### A. `l1_guidance.py` — lookahead 위치 공개 메서드 추가

파일: `fc_bridge/guidance/l1_guidance.py`

`ned_velocity_cmd()` 메서드 뒤에 추가:

```python
    def lookahead_pos_ned(self, pos_ned: np.ndarray) -> np.ndarray:
        """현재 위치에서 L1 거리만큼 경로를 따라간 위치 [N, E] 반환.

        Parameters
        ----------
        pos_ned : np.ndarray, shape (3,) or (2,)
            현재 위치 [N, E, (h)].

        Returns
        -------
        np.ndarray, shape (2,)  [N, E]
        """
        p2 = np.asarray(pos_ned[:2], dtype=float)
        seg = self._find_segment(p2)
        self._seg_idx = seg
        lh_pt, _ = self._lookahead_point(p2, seg)
        return lh_pt.copy()
```

#### B. `offboard_node.py` — PoseStamped publisher 추가

**import 섹션** (파일 상단 `from geometry_msgs.msg import PoseStamped` 추가):
```python
from geometry_msgs.msg import PoseStamped, TwistStamped
```

**`__init__`** (기존 `pub = self.create_publisher(...)` 다음에):
```python
        self._pos_pub = self.create_publisher(
            PoseStamped, "/mavros/setpoint_position/local", 10)
```

**새 메서드** (`_request_offboard` 위에 추가):
```python
    def _publish_pos_setpoint(self, pos_ned: np.ndarray) -> None:
        """NED 위치 [N, E, h_up] → ENU PoseStamped 발행.

        /mavros/setpoint_position/local 은 ENU 프레임을 기대한다.
        NED [N, E, h_up] → ENU [x=E, y=N, z=h_up]
        """
        msg = PoseStamped()
        msg.header.frame_id = "local_origin"
        msg.pose.position.x = float(pos_ned[1])   # E → x_enu
        msg.pose.position.y = float(pos_ned[0])   # N → y_enu
        msg.pose.position.z = float(pos_ned[2])   # h_up = z_enu
        self._pos_pub.publish(msg)
```

#### C. `offboard_node.py` — STREAMING 상태 교체

**변경 전** (현재 코드):
```python
        elif self._sm == _State.STREAMING:
            seg_i = min(self._guidance.current_segment, len(self._gamma) - 1)
            gamma = float(self._gamma[seg_i])
            vel_cmd = self._guidance.ned_velocity_cmd(
                state.pos_ned, state.vel_ned, gamma_ref=gamma)
            self._setpoint.publish(vel_cmd)

            if self._current_mode == "OFFBOARD":
                if vel_aligned_with_path(state.vel_ned, self._pts, cos_thresh=0.966):
                    self.get_logger().info("OFFBOARD + 속도 정렬(15°) → FOLLOWING")
                    self._sm = (_State.ENTRY if self._entry_mode == "mid_flight"
                                else _State.FOLLOWING)
                else:
                    chi_v = float(np.degrees(
                        np.arctan2(state.vel_ned[1], state.vel_ned[0])))
                    self.get_logger().debug(
                        f"속도 정렬 대기 vel_heading={chi_v:.1f}°")
                return

            self._stream_ticks += 1
            if self._stream_ticks == 20 and not self._offboard_requested:
                self._request_offboard()
                self._offboard_requested = True
                self.get_logger().info("OFFBOARD 전환 요청 (폴백)")
```

**변경 후**:
```python
        elif self._sm == _State.STREAMING:
            # 위치 세트포인트로 lookahead 위치 발행.
            # FW OFFBOARD에서 velocity setpoint 방향은 코스 제어에 반영되지 않으므로
            # position setpoint를 사용한다. PX4 FW autopilot이 위치를 직접 추종.
            lh_ne = self._guidance.lookahead_pos_ned(state.pos_ned)
            lh_ned = np.array([lh_ne[0], lh_ne[1], state.pos_ned[2]])
            self._publish_pos_setpoint(lh_ned)

            if self._current_mode == "OFFBOARD":
                self.get_logger().info("OFFBOARD 확인 → FOLLOWING")
                self._sm = (_State.ENTRY if self._entry_mode == "mid_flight"
                            else _State.FOLLOWING)
                self._follow_ticks = 0
                return

            self._stream_ticks += 1
            if self._stream_ticks == 20 and not self._offboard_requested:
                self._request_offboard()
                self._offboard_requested = True
                self.get_logger().info("OFFBOARD 전환 요청 (폴백)")
            elif self._stream_ticks % 10 == 0:
                self.get_logger().debug(
                    f"OFFBOARD 대기 tick={self._stream_ticks} "
                    f"mode={self._current_mode} "
                    f"lh=[{lh_ned[0]:.1f},{lh_ned[1]:.1f}]")
```

#### D. `offboard_node.py` — FOLLOWING 상태 교체

**변경 전** (`_step_following` 메서드 내 핵심 부분):
```python
        chi_cmd, v_cmd, cte = self._guidance.compute(pos, vel)
        vel_cmd = np.array([
            v_cmd * np.cos(chi_cmd),
            v_cmd * np.sin(chi_cmd),
            -v_cmd * np.sin(gamma),
        ])
        ...
        self._setpoint.publish(vel_cmd)
```

**변경 후**:
```python
        # 위치 세트포인트: lookahead 위치를 목표로 발행
        # (velocity setpoint는 FW OFFBOARD에서 코스 방향을 제어하지 못함)
        lh_ne = self._guidance.lookahead_pos_ned(pos)
        _, v_cmd, cte = self._guidance.compute(pos, vel)
        lh_ned = np.array([lh_ne[0], lh_ne[1], pos[2]])  # 현재 고도 유지
        self._publish_pos_setpoint(lh_ned)
        ...
        # self._setpoint.publish(vel_cmd) 제거
```

> **주의**: `v_cmd`(속도)는 위치 setpoint에서 직접 제어하지 않는다.
> PX4 FW의 TECS(Total Energy Control System)가 속도를 조절한다.
> 속도 제어가 필요하면 별도로 `/mavros/setpoint_velocity/cmd_vel`의 z=0 크기=v_cmd로 보낼 수 있으나, 우선순위는 경로 추종이다.

#### E. `offboard_node.py` — `vel_aligned_with_path` import 정리

STREAMING에서 `vel_aligned_with_path`를 더 이상 사용하지 않는다. 그러나 `test_offboard_node.py`는 `vel_aligned_with_path`를 `fc_bridge.execution.state_logic`에서 직접 import하므로 `state_logic.py`는 수정하지 않는다.

`offboard_node.py` import에서 `vel_aligned_with_path` 제거:
```python
from fc_bridge.execution.state_logic import (
    climbing_reached, vtol_is_fw,
    trans_mc_trigger, vtol_is_mc, landing_done,
    override_mode,
    # vel_aligned_with_path,  ← 제거
)
```

---

## 4. OFFBOARD keepalive 중요 사항

PX4는 OFFBOARD 모드 유지를 위해 2Hz 이상의 setpoint 스트림이 필요하다. 속도에서 위치 setpoint로 전환 시:

- `/mavros/setpoint_position/local` (PoseStamped)은 PX4에서 OFFBOARD keepalive로 인정됨
- 두 토픽(`/cmd_vel`과 `/setpoint_position/local`)을 동시에 발행하면 PX4가 마지막으로 받은 것을 사용
- 전환 이후에는 `/mavros/setpoint_position/local` 만 발행하면 됨

---

## 5. 변경 요약 체크리스트

### `fc_bridge/guidance/l1_guidance.py`
- [ ] `lookahead_pos_ned(pos_ned)` 메서드 추가 (`ned_velocity_cmd` 뒤)

### `fc_ros/fc_ros/nodes/offboard_node.py`
- [ ] `from geometry_msgs.msg import PoseStamped, TwistStamped` (import 수정)
- [ ] `_PREACCEL_SPEED = 10.0`, `_PREACCEL_ALIGN = 0.174` 상수 추가 (모듈 상단)
- [ ] `self._fw_preaccel_done = False` 추가 (`__init__`)
- [ ] `self._pos_pub` publisher 추가 (`__init__`)
- [ ] `_publish_pos_setpoint()` 메서드 추가
- [ ] Phase 2.5 (사전 가속) 블록 삽입 (`_step_transition_fw`)
- [ ] Phase 3 로그 업데이트 (`_step_transition_fw`)
- [ ] STREAMING 상태 로직 교체 (`_control_callback`)
- [ ] `_step_following()`: velocity setpoint → position setpoint
- [ ] `vel_aligned_with_path` import 제거

### 수정하지 않는 파일
- `fc_ros/fc_ros/adapters/setpoint_publisher.py` — 그대로
- `fc_ros/fc_ros/adapters/vehicle_state_bridge.py` — 그대로
- `fc_ros/fc_ros/params/fc_ros_params.yaml` — `wp0_heading_tol: 0.05` 이미 적용
- `fc_bridge/execution/state_logic.py` — 그대로 (테스트가 직접 import)

---

## 6. 테스트

### 단위 테스트

수정 후 반드시 실행:
```bash
# 리포지토리 루트에서
python -m pytest fc_ros/test/ fc_bridge/tests/ -q
```
기대값: `100 passed`

### WSL 빌드 및 SITL 실행

```bash
# WSL에서 (반드시 실행)
cd ~/ros2_ws
colcon build --packages-select fc_ros fc_bridge && source install/setup.bash

# SITL 실행 (별도 터미널)
# 1. PX4 SITL 시작
# 2. MAVROS 시작
# 3. fc_ros 노드 시작
```

### SITL 검증 포인트

**문제 1 (직선 천이) 확인 로그**:
```
[INFO] 헤딩 정렬 완료 target=0.0° current=X.X° err=Y.Y°     ← err < 3°
[DEBUG] 사전 가속 중 speed=5.0/10.0 vel_err=3.5°
[INFO] 사전 가속 완료 speed=10.2m/s vel_err=2.1° → 천이 명령 준비
[INFO] MC→FW 천이 명령 요청 (사전 가속 완료 후 직선 천이)
[INFO] FW 전환 완료 -> STREAMING
```
STREAMING 진입 시 `state.yaw`가 목표 헤딩과 ±10° 이내이면 성공.

**문제 2 (위치 추종) 확인 로그**:
```
[INFO] OFFBOARD 확인 → FOLLOWING
[INFO] FOLLOWING 시작 pos=[X,Y] chi=Z.Z° cte=W.Wm mode=OFFBOARD
[INFO] FOLLOWING tick=20 mode=OFFBOARD chi=Z.Z° cte=W.Wm pos=[X,Y]
```
`cte`(횡방향 오차)가 시간이 지남에 따라 감소하면 경로 추종 성공.

---

## 7. 추가 고려사항

### 문제 1 — 사전 가속이 실패하는 경우

MC OFFBOARD에서 10 m/s까지 가속하는 데 시간이 걸린다. 기본 `v_cruise=20 m/s`로 명령해도 MC hover에서 수평 가속이 느릴 수 있다. 만약 사전 가속이 30초 이상 걸리면:
- `_PREACCEL_SPEED`를 5.0 m/s로 낮추거나
- `_PREACCEL_ALIGN`을 0.35 rad(20°)로 넓힌다

### 문제 2 — 속도 제어 추가

Position setpoint만으로는 FW 속도를 직접 지정할 수 없다. PX4 FW의 TECS가 속도를 조절하는데, 이 속도는 `MPC_XY_VEL_MAX` 등 PX4 파라미터에 의존한다. 만약 속도도 제어해야 한다면:
- `/mavros/setpoint_raw/local`에 `POSITION | VELOCITY` 마스크로 함께 발행
- 이 경우 `PositionTarget` 메시지 타입 사용 (`mavros_msgs.msg.PositionTarget`)

### 문제 1, 2 모두 실패 시 대안

PX4 VTOL을 OFFBOARD로 제어하는 대신:
- MAVLink Mission Item으로 VTOL 천이 경로를 업로드
- `MissionNode`를 사용해 AUTO.MISSION으로 비행
- 이 경우 PX4의 내장 VTOL 미션 로직이 천이와 경로 추종을 모두 담당
