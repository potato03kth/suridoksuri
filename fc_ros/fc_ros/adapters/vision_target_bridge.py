"""
vision 정밀착륙 토픽 → `VisionTarget` 변환·보관 어댑터.

`OffboardNode`의 정밀착륙 서브상태(F2)가 소비할 vision 산출물 두 개를 받아
**최신 상태를 보관하고 나이를 재는 것까지만** 한다. 기체 제어 판단(하강/홀드/인계)은
한 줄도 하지 않는다 — `vehicle_state_bridge.py`와 같은 역할 분담이다.

계약 정본: `docs/fc_precision_land_handoff.md` §2(소비할 계약)·§4(함정).
아래 수치·부호는 전부 그 문서에서 "실기체 검증됨"으로 확정된 것이다.

    /vision/landing_setpoint  geometry_msgs/PoseStamped         (ENU, frame_id="map")
    /vision/target_status     diagnostic_msgs/DiagnosticArray   (거부권·생존신호)

## 🔴 QoS는 BEST_EFFORT / KEEP_LAST / depth=1 이다

발행자(vision shim)가 BEST_EFFORT로 낸다. **RELIABLE로 구독하면 QoS 비호환으로 연결
자체가 성립하지 않아 조용히 아무것도 받지 못한다** — 예외도 경고도 없다
(핸드오프 §4 함정 3). `ros2 topic echo`의 기본값도 RELIABLE이라 디버깅 중에
"발행이 안 된다"고 오판하기 쉽다:

    ros2 topic echo /vision/landing_setpoint --qos-reliability best_effort --qos-durability volatile

⚠️ **이 위험은 단위테스트로 잡히지 않는다.** rclpy가 없는 개발 환경에서는 QoS 객체를
만들 수조차 없고, 있더라도 "구독은 성공했는데 데이터가 안 온다"는 런타임 미들웨어
동작이라 프로세스 하나로는 재현되지 않는다. `_vision_qos()`의 reliability를 바꾸면
**전 테스트가 green인 채로 기체에서만 유도가 죽는다.** 실기체/SITL에서
`ros2 topic info -v`로 구독자 QoS를 눈으로 확인하는 것이 유일한 그물이다.

## 🔴 좌표 규약 — 틀리면 기체가 반대로 간다

`landing_setpoint`는 **ENU**(`x=동, y=북, z=상`)이고 `_publish_pos_setpoint`가 먹는
`pos_ned`는 **`[N, E, h_up]`**이다. 축을 바꿔 담는다:

    pos_ned = [msg.pose.position.y,    # N    ← y_enu
               msg.pose.position.x,    # E    ← x_enu
               msg.pose.position.z]    # h_up ← z_enu (위 양수 그대로)

🔴 **`pos_ned`의 3번째는 `h_up`(위 양수)이지 `D`가 아니다.** 같은 저장소의 `vel_ned`는
`[vN, vE, vD]`로 **3번째가 아래 양수**다 — **같은 `_ned` 접미사가 위치와 속도에서 반대
부호 규약**을 쓰고, 이걸 혼동한 사고 이력이 있다. 여기서 부호를 뒤집으면 setpoint가
지면 아래를 가리킨다.

`vehicle_state_bridge.update_from_pose`가 쓰는 `[p.y, p.x, p.z]`와 **문자 그대로 같은
관용구**이고, yaw 디코드(`quat_to_euler_xyz` → `yaw_ned = π/2 − yaw_enu`)도 그쪽 경로를
그대로 재사용한다(새로 짜지 않는다).

## 🔴 setpoint의 orientation은 "모름"이 아니라 "이 헤딩 유지"다

vision shim은 그 순간 기체의 실제 yaw만 뽑은 쿼터니언을 싣는다. ENU 단위 쿼터니언은
"모름"이 아니라 **"동쪽을 보라"는 명령**이라 그대로 흘리면 2026-07-21 flight04 yaw
스핀 사고가 재발한다. 그래서 여기서 디코드한 `yaw_ned`는 "vision이 관측한 현재 기수
방위"이고, 소비자는 `_publish_pos_setpoint(pos_ned, yaw_ned)`의 yaw 인자로 쓰거나
자기 `state.yaw`를 쓰거나 하면 된다 — **어느 쪽이든 실제 헤딩과 일치해야 한다.**

## 🔴 stale 판정 기준 1.0초 — 4.4Hz 실측에서 나온 값이다

vision의 **실측 발행 주파수는 4.4Hz**(프레임 간격 median 0.2207s / p95 0.310s,
컨테이너 `ros2 topic hz` 4.35Hz). 🔴 **10Hz를 가정하지 마라** — 이 저장소 문서 여러
곳이 오래 그렇게 적어 왔지만 실측과 2배 이상 다르다. `stale_timeout_s`를 0.5s 근처로
잡으면 정상 지터(p95 0.310s)와의 여유가 1.6배뿐이라 **헛경보**가 난다. 1.0s는 p95의
약 3.2배 = 연속 4프레임 이상 누락에 해당한다(핸드오프 §7).

나이는 **`header.stamp`가 아니라 이 노드가 메시지를 받은 시각**으로 잰다. `header.stamp`는
wall clock(`CLOCK_REALTIME`)이라 NTP 점프에 흔들리고, `vehicle_state_bridge`도 같은
이유로 `time.monotonic()`을 쓴다. 판정 로직은 클록을 모르게 순수 함수로 두고
(`snapshot(now_s)`), 시각은 콜백이 주입한다.

## 🔴 침묵/무효/사망 3분법 — setpoint 침묵만으로는 못 가른다

| 상황 | `landing_setpoint` | `target_status` | 이 어댑터의 표현 |
|---|---|---|---|
| 정상 | 발행 | 전부 OK | `valid=True, target_seen=True` |
| 안 보임 | **침묵** | `vision/target` WARN | `valid=False, target_seen=False`, status는 신선 |
| **생산자 사망** | **침묵** | `vision/link` ERROR 계속(1Hz) | `link_dead=True` |
| **shim 사망** | **침묵** | **침묵** | `status_age_s`가 계속 자란다 |

**→ 소비자는 반드시 `valid`와 status 계열(`link_dead`/`status_age_s`)을 같이 봐야 한다.**

## 🔴 하지 않는 것

- **마지막 setpoint를 무한정 붙들지 않는다.** `age_s > stale_timeout_s`면 `valid=False`
  이면서 `pos_ned`·`yaw_ned`를 **`None`으로 내린다** — 옛 좌표를 계속 노출하면 소비자가
  실수로 쓴다.
- **값이 없을 때 0을 채우지 않는다.** setpoint 자리의 0은 "기체 바로 아래로 가라"는
  뜻이라 이 인터페이스에서 가장 위험한 거짓말이다(vision `core/wire.py`가 붙인 이름
  그대로다).
- **`command_hint`를 명령으로 소비하지 않는다.** 와이어가 `command_hint` +
  `command_is_advisory:true` + 타입명 `state_hint` 세 겹으로 "명령 아님"을 박아 뒀다.
  거부권은 오직 `state`(=`vision/state`의 level)다. 그래서 필드 이름도
  `land_handoff_hint`이지 `land_handoff`가 아니다.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from fc_bridge.utils.rotation import quat_to_euler_xyz

# ── 토픽 ─────────────────────────────────────────────────────────
LANDING_SETPOINT_TOPIC = "/vision/landing_setpoint"
TARGET_STATUS_TOPIC = "/vision/target_status"

# ── DiagnosticStatus level (diagnostic_msgs 상수 복제) ───────────
# vision `ros/shim_core.py`가 같은 값을 복제해 쓴다. 여기서 다시 복제하는 이유는
# 도메인 간 빌드 의존을 만들지 않기 위해서이고(CLAUDE.md), 이 네 값은 ROS 표준이라
# 어긋날 여지가 없다.
LEVEL_OK = 0
LEVEL_WARN = 1
LEVEL_ERROR = 2
LEVEL_STALE = 3

# ── DiagnosticStatus.name (shim 계약) ────────────────────────────
STATUS_NAME_TARGET = "vision/target"     # 검출 유효성      OK / WARN(안 보임)
STATUS_NAME_STATE = "vision/state"       # 🔴 상태머신 = 거부권. WARN이면 veto
STATUS_NAME_SETPOINT = "vision/setpoint"  # setpoint 발행 사유(진단용)
STATUS_NAME_LINK = "vision/link"         # 생산자 생존. ERROR = 사망(1Hz 하트비트)

# ── KeyValue 키 (vision/ros/shim_core.py::build_state_status 원문) ──
#: 상태머신 상태 문자열. `DiagnosticStatus.message`("state=LOCK")를 파싱하지 않고
#: 이 KeyValue를 읽는다 — message는 사람용 서술이라 형식이 언제든 바뀐다.
KV_STATE = "state"
#: 🔴 advisory. 명령이 아니다(파일 docstring 참조).
KV_COMMAND_HINT = "command_hint"

#: 🔴 거부권 상태 집합. 실제 판정은 **level(WARN)**로 하고 이 집합은 교차검증·문서용이다
#: — shim이 level을 정하는 단일 출처이므로 문자열 집합을 여기서 다시 판정하면
#: 두 곳이 조용히 갈라질 수 있다.
VETO_STATES = frozenset({"HOLD", "ABORT_ASCEND"})
#: `TERMINAL`에서 블라인드 2초 초과 시 vision이 내는 힌트 = "횡 유도를 놓고 AUTO.LAND로
#: 인계하라". advisory이지만 무시하면 설계 의도와 반대로 간다(핸드오프 §2-2).
COMMAND_HINT_LAND = "land"

#: 🔴 실측 4.4Hz 근거(파일 docstring "stale 판정 기준" 절). 10Hz 가정 금지.
DEFAULT_STALE_TIMEOUT_S = 1.0
#: `vision/link` 판정 유효기간. shim의 하트비트 주기가 1.0s(`shim_node._DEFAULT_HEARTBEAT_PERIOD_S`)
#: 이므로 3.0s = 하트비트 3회 누락. 이 창이 필요한 이유는 두 가지다:
#:   ① 진짜 사망은 1Hz로 ERROR가 계속 갱신되므로 창 안에 항상 들어온다.
#:   ② 깨진 줄 한 건(`ShimRouter._reject`)도 일회성 ERROR를 내는데, 창이 없으면 그
#:      한 건이 `link_dead=True`를 **영원히 래치**해 정상 유도를 막는다.
DEFAULT_LINK_TIMEOUT_S = 3.0

#: QoS depth. 제어용 스트림이라 최신성 > 완전성(shim 발행 측과 같은 값).
VISION_QOS_DEPTH = 1


# ──────────────────────────────────────────────────────────────────
# 순수 변환
# ──────────────────────────────────────────────────────────────────


def enu_to_pos_ned_n_e_hup(x_enu: float, y_enu: float, z_enu: float) -> np.ndarray:
    """ENU(x=동, y=북, z=상) → `_publish_pos_setpoint`의 `pos_ned` = **[N, E, h_up]**.

    🔴 이름이 길고 못생긴 것은 의도다 — `pos_ned`(3번째가 **위** 양수)와
    `vel_ned`(3번째가 **아래** 양수)가 같은 `_ned` 접미사로 반대 부호 규약을 쓴다.
    그냥 `to_ned()`라고 부르면 다음 사람이 D를 기대한다.
    """
    return np.array([float(y_enu), float(x_enu), float(z_enu)])


def enu_yaw_to_ned_yaw(yaw_enu: float) -> float:
    """ENU yaw(0=동, 반시계 양수) → NED yaw(0=북, 시계 양수), [-π, π] 정규화.

    `vehicle_state_bridge.update_from_pose`의 그 줄과 **같은 식**이다(재구현 아님).
    """
    yaw_ned = math.pi / 2.0 - float(yaw_enu)
    return float((yaw_ned + math.pi) % (2.0 * math.pi) - math.pi)


def decode_setpoint_yaw_ned(orientation) -> float:
    """`PoseStamped.pose.orientation`(ENU) → NED yaw.

    🔴 이 쿼터니언은 **단위 쿼터니언이 아니라 "현재 기수방위 유지"** 를 뜻한다
    (파일 docstring 참조). 기존 디코드 경로(`quat_to_euler_xyz` → `π/2 − yaw_enu`)를
    그대로 쓴다.
    """
    _, _, yaw_enu = quat_to_euler_xyz(
        float(orientation.w), float(orientation.x),
        float(orientation.y), float(orientation.z),
    )
    return enu_yaw_to_ned_yaw(yaw_enu)


def level_to_int(level) -> int:
    """`DiagnosticStatus.level` → int.

    🔴 msg 상 타입이 `byte`라 rclpy는 **1바이트 `bytes`**를 준다(발행 측도
    `st.level = bytes([...])`로 넣는다 — 실기체 확인 사항). `status.level == 1`로
    비교하면 `b'\\x01' == 1`이 **항상 False**라 거부권이 조용히 사라진다.
    int로 오는 구현(가짜 메시지·다른 바인딩)도 그대로 받는다.
    """
    if isinstance(level, (bytes, bytearray)):
        return int(level[0]) if len(level) else -1
    return int(level)


def kv_lookup(status, key: str) -> Optional[str]:
    """`DiagnosticStatus.values`(KeyValue 배열)에서 값 문자열을 찾는다.

    shim은 문자열이 아닌 값을 `json.dumps`로 눌러 싣는다 — `None`은 `"null"`이 되므로
    여기서 다시 `None`으로 되돌린다(문자열 `"null"`을 상태 이름으로 오인하지 않게).
    """
    for kv in getattr(status, "values", ()) or ():
        if kv.key == key:
            value = kv.value
            if value is None or value == "null":
                return None
            return str(value)
    return None


# ──────────────────────────────────────────────────────────────────
# 소비자에게 노출하는 스냅샷
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VisionTarget:
    """정밀착륙 서브상태가 한 틱에 보는 vision 상태 전량.

    `frozen=True`인 이유는 스레드다 — 콜백 스레드가 만들고 제어틱이 읽으므로
    **완성된 객체를 통째로 넘겨야** 부분갱신 상태가 보이지 않는다
    (vision `shim_core.VehiclePose`와 같은 판단).
    """

    #: 신선한 setpoint가 있는가. `pos_ned is not None`과 항상 같이 움직인다.
    valid: bool
    #: 🔴 **[N, E, h_up]** — 3번째가 `h_up`(위 양수)이지 D가 아니다. 무효면 **None**
    #: (옛 좌표도 0도 채우지 않는다).
    pos_ned: Optional[np.ndarray]
    #: setpoint orientation에서 디코드한 NED yaw(rad). 무효면 None.
    yaw_ned: Optional[float]
    #: 마지막 setpoint 수신 후 경과(s). 한 번도 못 받았으면 `inf`.
    age_s: float
    #: vision 상태머신 상태 문자열(`ACQUIRE`/`CENTER_DESCEND`/`LOCK`/`PRECISION_SERVO`/
    #: `TERMINAL`/`HOLD`/`ABORT_ASCEND`). 모르면 None.
    state: Optional[str]
    #: 🔴 거부권 — `vision/state`의 level이 WARN(=상태가 `HOLD`/`ABORT_ASCEND`).
    #: ⚠️ `veto=False`는 "vision이 허가했다"가 아니라 "거부 신호가 없다"이다.
    #: 하강 허가는 `valid`·`link_dead`·`status_age_s`를 함께 봐야 한다.
    veto: bool
    #: `vision/link` level == ERROR — **생산자(vision) 사망**.
    link_dead: bool
    #: 마지막 status 수신 후 경과(s). 이것이 계속 자라면 **shim 자신이 죽은 것**이다
    #: (생산자 사망은 1Hz ERROR가 계속 오므로 자라지 않는다).
    status_age_s: float
    #: 🔴 advisory. `command_hint == "land"` = "횡 유도를 놓고 AUTO.LAND로 인계하라".
    #: **명령이 아니다** — 거부권은 `veto`뿐이다.
    land_handoff_hint: bool
    #: `vision/target` level == OK(타겟을 실제로 보고 있다).
    target_seen: bool


#: setpoint를 한 번도 못 받은 초기 스냅샷(0으로 채우지 않는다).
_NEVER = float("inf")


# ──────────────────────────────────────────────────────────────────
# 어댑터
# ──────────────────────────────────────────────────────────────────


class VisionTargetBridge:
    """두 토픽의 최신 수신값을 보관하고 나이를 잰다. **제어 판단은 하지 않는다.**

    사용법(노드 쪽):

        bridge = VisionTargetBridge()
        subscribe(self, bridge)                 # QoS·메시지 타입은 이 함수가 책임진다
        ...
        vt = bridge.snapshot(time.monotonic())  # 제어틱에서
        if vt.valid and not vt.veto:
            ...

    `update_from_*`는 수신 시각을 인자로 받고 `snapshot`은 판정 시각을 인자로 받는다 —
    rclpy 클록에 의존하지 않아 랩탑에서 그대로 단위테스트된다.
    """

    def __init__(
        self,
        *,
        stale_timeout_s: float = DEFAULT_STALE_TIMEOUT_S,
        link_timeout_s: float = DEFAULT_LINK_TIMEOUT_S,
    ):
        self.stale_timeout_s = float(stale_timeout_s)
        self.link_timeout_s = float(link_timeout_s)
        self._lock = threading.Lock()

        # setpoint
        self._pos_ned: Optional[np.ndarray] = None
        self._yaw_ned: Optional[float] = None
        self._setpoint_t: Optional[float] = None

        # status — 🔴 이름별로 따로 보관한다. 한 DiagnosticArray가 네 항목을 전부
        # 싣는다는 보장이 없다(끊긴 동안의 하트비트는 `vision/link` 하나뿐이고,
        # 짝 없는 state_hint는 `vision/state` 하나뿐이다). 통째로 덮어쓰면 그때
        # 다른 항목이 조용히 초기화된다.
        self._status_t: Optional[float] = None          # 배열 자체의 마지막 수신 시각
        self._target_level: Optional[int] = None
        self._target_t: Optional[float] = None
        self._state_level: Optional[int] = None
        self._state_str: Optional[str] = None
        self._command_hint: Optional[str] = None
        self._state_t: Optional[float] = None
        self._link_level: Optional[int] = None
        self._link_t: Optional[float] = None
        self._setpoint_reason: Optional[str] = None

        # 관측성 카운터(§7.4 — 침묵 금지)
        self.setpoints_rx = 0
        self.statuses_rx = 0
        self.rejected_setpoints = 0

    # ---------- 입력 ----------

    def update_from_setpoint(self, msg, now_s: float) -> None:
        """`/vision/landing_setpoint`(geometry_msgs/PoseStamped, **ENU**) 수신.

        `now_s`는 **수신 시각**(단조 클록)이다. `header.stamp`를 쓰지 않는 이유는
        파일 docstring "stale 판정" 절 참조.
        """
        p = msg.pose.position
        pos_ned = enu_to_pos_ned_n_e_hup(p.x, p.y, p.z)
        yaw_ned = decode_setpoint_yaw_ned(msg.pose.orientation)

        # NaN/inf setpoint는 **받지 않는다**. 좌표 자리의 NaN은 하류 산술을 조용히
        # 오염시키고(vision `shim_core`가 pose에 NaN을 안 채운 것과 같은 판단),
        # 무시하면 그냥 stale로 떨어져 안전한 쪽으로 degrade한다.
        if not (np.all(np.isfinite(pos_ned)) and math.isfinite(yaw_ned)):
            with self._lock:
                self.rejected_setpoints += 1
            return

        with self._lock:
            self._pos_ned = pos_ned
            self._yaw_ned = float(yaw_ned)
            self._setpoint_t = float(now_s)
            self.setpoints_rx += 1

    def update_from_status(self, msg, now_s: float) -> None:
        """`/vision/target_status`(diagnostic_msgs/DiagnosticArray) 수신."""
        now = float(now_s)
        with self._lock:
            self._status_t = now
            self.statuses_rx += 1
            for status in msg.status:
                name = status.name
                level = level_to_int(status.level)
                if name == STATUS_NAME_TARGET:
                    self._target_level = level
                    self._target_t = now
                elif name == STATUS_NAME_STATE:
                    # 🔴 거부권은 **level**로 읽는다. 상태 문자열은 진단·표시용이다
                    # (shim이 level을 정하는 단일 출처이고, 문자열 집합을 여기서 다시
                    # 판정하면 두 곳이 조용히 갈라진다).
                    self._state_level = level
                    self._state_str = kv_lookup(status, KV_STATE)
                    self._command_hint = kv_lookup(status, KV_COMMAND_HINT)
                    self._state_t = now
                elif name == STATUS_NAME_LINK:
                    self._link_level = level
                    self._link_t = now
                elif name == STATUS_NAME_SETPOINT:
                    # 발행 사유(`ok`/`no_target`/`attitude_missing`/`attitude_stale`/
                    # `attitude_invalid_quaternion`/`disabled`) — 진단 전용이다.
                    # setpoint 자체의 유무가 권위이므로 여기서 판정에 쓰지 않는다.
                    self._setpoint_reason = status.message

    # ---------- 출력 ----------

    def snapshot(self, now_s: float) -> VisionTarget:
        """판정 시각 `now_s` 기준 스냅샷. **순수 함수** — 클록을 만지지 않는다."""
        now = float(now_s)
        with self._lock:
            age_s = _NEVER if self._setpoint_t is None else now - self._setpoint_t
            fresh = age_s <= self.stale_timeout_s

            # 🔴 stale이면 좌표를 **내려버린다**. 옛 좌표를 계속 노출하면 소비자가
            # 실수로 쓰고, 0으로 채우면 "기체 바로 아래로 가라"가 된다.
            pos_ned = (
                np.array(self._pos_ned, dtype=float)
                if (fresh and self._pos_ned is not None)
                else None
            )
            yaw_ned = self._yaw_ned if (fresh and self._yaw_ned is not None) else None

            status_age_s = _NEVER if self._status_t is None else now - self._status_t

            # `vision/target`·`vision/state`는 vision 프레임(4.4Hz)과 함께 오므로
            # setpoint와 같은 창으로 만료시킨다. 생산자가 죽으면 1Hz 하트비트에는
            # 이 둘이 실리지 않아 자연히 만료된다.
            target_fresh = (
                self._target_t is not None
                and (now - self._target_t) <= self.stale_timeout_s
            )
            state_fresh = (
                self._state_t is not None
                and (now - self._state_t) <= self.stale_timeout_s
            )
            link_fresh = (
                self._link_t is not None
                and (now - self._link_t) <= self.link_timeout_s
            )

            target_seen = bool(target_fresh and self._target_level == LEVEL_OK)
            state = self._state_str if state_fresh else None
            veto = bool(state_fresh and self._state_level == LEVEL_WARN)
            land_hint = bool(
                state_fresh and self._command_hint == COMMAND_HINT_LAND
            )
            link_dead = bool(
                link_fresh
                and self._link_level is not None
                and self._link_level >= LEVEL_ERROR
            )

        return VisionTarget(
            valid=bool(pos_ned is not None),
            pos_ned=pos_ned,
            yaw_ned=yaw_ned,
            age_s=float(age_s),
            state=state,
            veto=veto,
            link_dead=link_dead,
            status_age_s=float(status_age_s),
            land_handoff_hint=land_hint,
            target_seen=target_seen,
        )

    @property
    def setpoint_reason(self) -> Optional[str]:
        """마지막 `vision/setpoint` status의 message(진단 전용, 판정에 쓰지 않는다)."""
        with self._lock:
            return self._setpoint_reason

    def reset(self) -> None:
        """상태 전량 초기화(정밀착륙 서브상태 재진입 시). 카운터는 유지한다."""
        with self._lock:
            self._pos_ned = None
            self._yaw_ned = None
            self._setpoint_t = None
            self._status_t = None
            self._target_level = None
            self._target_t = None
            self._state_level = None
            self._state_str = None
            self._command_hint = None
            self._state_t = None
            self._link_level = None
            self._link_t = None
            self._setpoint_reason = None


# ──────────────────────────────────────────────────────────────────
# rclpy 배선 (지연 import — 이 모듈은 rclpy 없이도 import된다)
# ──────────────────────────────────────────────────────────────────


def _vision_qos():
    """🔴 **BEST_EFFORT / KEEP_LAST / depth=1.**

    `telemetry_node::_MAVROS_QOS`와 같은 계열이되 depth만 1이다(제어용 스트림이라
    최신성 > 완전성. 호환성을 정하는 것은 reliability/durability이지 depth가 아니다).

    🔴 **여기를 RELIABLE로 바꾸면 조용히 아무것도 안 받는다** — 예외도 로그도 없고
    단위테스트도 잡지 못한다(파일 docstring 참조).

    rclpy는 함수 안에서 import한다 — `vehicle_state_bridge.py`와 마찬가지로 이
    어댑터는 rclpy 없는 환경에서도 import·테스트되어야 한다.
    """
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=VISION_QOS_DEPTH,
    )


def subscribe(
    node,
    bridge: VisionTargetBridge,
    *,
    setpoint_topic: str = LANDING_SETPOINT_TOPIC,
    status_topic: str = TARGET_STATUS_TOPIC,
    now_fn=None,
):
    """`node`에 두 구독을 만들어 `bridge`에 물린다. 생성된 구독 2개를 돌려준다.

    `now_fn`은 수신 시각 소스(기본 `time.monotonic`) — 나이를 재는 기준이므로
    **wall clock을 쓰지 마라**(NTP 점프에 흔들린다, `vehicle_state_bridge`와 같은 판단).
    """
    import time as _time

    from geometry_msgs.msg import PoseStamped
    from diagnostic_msgs.msg import DiagnosticArray

    clock = now_fn or _time.monotonic
    qos = _vision_qos()

    sub_sp = node.create_subscription(
        PoseStamped, setpoint_topic,
        lambda msg: bridge.update_from_setpoint(msg, clock()), qos)
    sub_st = node.create_subscription(
        DiagnosticArray, status_topic,
        lambda msg: bridge.update_from_status(msg, clock()), qos)
    return sub_sp, sub_st
