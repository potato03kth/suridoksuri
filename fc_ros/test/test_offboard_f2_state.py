"""F2(비전 정밀착륙) **상태전이 회귀 테스트** — 노드 메서드를 실제로 실행한다.

## 왜 이 파일이 따로 있나

2026-07-29 검증에서 `_enter_vision_search()` 가 `self._sm` 을 세팅하지 않아
`vision_landing:=true` 경로가 **통째로 도달 불가**였다는 것이 드러났다. 증상은
조용하지 않다 — `_exit_hold` → `_enter_vision_search` 를 불러도 상태가 `HOLD` 로
남아 다음 틱에 `_step_hold` 가 다시 안정조건을 만족하고 `_exit_hold` 를 또 부르는
**10Hz 무한 루프**가 된다(SITL 실측 진입 로그 1687회, `stable=2205/10`).
실기체였다면 WP1 상공에서 영원히 호버했다.

그런데 **440건 테스트가 전부 통과했다.** `fc_ros/test/` 어디에도
`_exit_hold`·`_enter_vision_search`·`_step_vision_search`·`VISION_SEARCH` 참조가
0건이었기 때문이다 — F2 노드측 상태기계 테스트가 아예 없었다.
`test_offboard_node.py` 는 순수 함수 + AST 배선검사만 하고, 순수 함수는
"`_sm` 을 세팅했는가" 같은 **상태 전이 자체**를 볼 수 없다.

그래서 이 파일은 다른 층을 본다: **노드 메서드를 진짜로 호출해 `_sm` 의 변화와
연속 틱의 거동을 단언한다.**

## rclpy 없이 노드 메서드를 실행하는 법

이 저장소 `.venv` 에는 rclpy 도 mavros_msgs 도 없다(랩탑/Windows 에서 그냥 돌아야
한다는 규율). 그래서:

  ① 없는 ROS 모듈만 최소 스텁으로 `sys.modules` 에 꽂는다 — **실물이 있으면
     건드리지 않는다**(컨테이너/실기체에서는 진짜 rclpy 로 같은 테스트가 돈다).
  ② `OffboardNode.__new__` 로 `Node.__init__` 을 건너뛴 껍데기를 만들고,
     상태머신이 읽는 속성만 손으로 채운다. ROS 입출력(`_publish_pos_setpoint`,
     `_request_offboard`, `get_logger`)은 인스턴스 속성으로 덮어 기록만 한다.

`__init__` 을 돌리지 않으므로 **파라미터 읽기 경로는 이 파일의 검증 대상이
아니다**(그쪽은 `test_params.py` 가 YAML↔declare↔launch 3중으로 고정한다).
여기서 보는 것은 오직 전이 로직이다.
"""
import sys
import time
import types

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────
# ① 없는 ROS 모듈만 스텁으로 채운다
# ──────────────────────────────────────────────────────────────────

def _permissive(name):
    """무엇을 꺼내도 받아주는 더미 클래스 (`ReliabilityPolicy.BEST_EFFORT` 등)."""
    meta = type("_Meta", (type,), {"__getattr__": lambda cls, attr: 0})
    return meta(name, (), {
        "__init__": lambda self, *a, **k: None,
        "__getattr__": lambda self, attr: 0,
    })


def _stub(name):
    mod = types.ModuleType(name)
    mod.__path__ = []                    # 패키지 취급(하위 모듈 스텁을 달 수 있게)

    def _getattr(attr):
        # 🔴 던더는 반드시 AttributeError 로 넘긴다 — import 기계장치가
        # `__path__`/`__all__` 을 물어볼 때 더미를 돌려주면 그 자리에서 깨진다.
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _permissive(attr)

    mod.__getattr__ = _getattr           # PEP 562 — `from X import Y` 가 통한다
    return mod


def _ensure(name):
    """`name` 이 진짜로 import 되면 그대로 두고, 안 되면 스텁을 꽂는다."""
    try:
        __import__(name)
        return
    except ImportError:
        pass
    parts = name.split(".")
    for i in range(len(parts)):
        sub = ".".join(parts[:i + 1])
        if sub not in sys.modules:
            sys.modules[sub] = _stub(sub)
    for i in range(1, len(parts)):
        setattr(sys.modules[".".join(parts[:i])], parts[i],
                sys.modules[".".join(parts[:i + 1])])


for _m in ("rclpy", "rclpy.node", "rclpy.qos", "rclpy.time",
           "geometry_msgs.msg", "mavros_msgs.msg", "mavros_msgs.srv",
           "std_msgs.msg"):
    _ensure(_m)

from fc_ros.nodes import offboard_node as ON          # noqa: E402
from fc_ros.adapters.vision_target_bridge import (    # noqa: E402
    VisionTargetBridge,
)

_State = ON._State


# ──────────────────────────────────────────────────────────────────
# ② 노드 껍데기
# ──────────────────────────────────────────────────────────────────

class _Logger:
    def __init__(self):
        self.lines = []

    def _rec(self, msg, **kw):
        self.lines.append(str(msg))

    info = warn = error = debug = fatal = _rec


def _vehicle_state(pos_ned, vel_ned=(0.0, 0.0, 0.0), yaw=0.0):
    return types.SimpleNamespace(
        pos_ned=np.array(pos_ned, dtype=float),
        vel_ned=np.array(vel_ned, dtype=float),
        yaw=float(yaw), armed=True)


#: 기본 WP1 (NED [N, E]) 과 순항고도. 값 자체에 의미는 없다.
WP1 = (300.0, 0.0)
CRUISE_ALT = 50.0
GROUND_H = 0.0


def make_node(*, vision_landing=True, latch_frames=3, latch_spread=3.0,
              stale_timeout=1.0):
    """상태머신이 읽는 속성만 채운 `OffboardNode` 껍데기."""
    n = ON.OffboardNode.__new__(ON.OffboardNode)

    n.log = _Logger()
    n.get_logger = lambda: n.log
    n.published = []
    n._publish_pos_setpoint = lambda pos, yaw: n.published.append(
        (np.array(pos, dtype=float), yaw))
    n.offboard_requests = 0

    def _req():
        n.offboard_requests += 1
    n._request_offboard = _req

    # 제어 루프 / 경로
    n._dt = 0.1
    n._pts = np.array([[0.0, 0.0], [WP1[0], WP1[1]]], dtype=float)
    n._cruise_alt = CRUISE_ALT
    n._v_approach = 5.0
    n._current_mode = "OFFBOARD"
    n._takeoff_ground_h = GROUND_H

    # HOLD
    n._mc_pos_ramp = None
    n._hold_ticks = 0
    n._hold_stable_ticks = 0
    n._hold_elapsed = 0.0
    n._hold_timeout = 30.0
    n._wp1_land_radius = 5.0
    n._wp1_land_speed = 1.0

    # VISION_SEARCH
    n._vision_landing = bool(vision_landing)
    n._vs_alt = 25.0
    n._vs_radius = 30.0
    n._vs_retry_alt = 15.0
    n._vs_retry_radius = 18.0
    n._vs_spacing_param = 0.0
    n._vs_speed_param = 0.0
    n._vs_dwell = 3.0
    n._vs_timeout = 120.0
    n._vs_pass = 0
    n._vs_phase = "align"
    n._vs_elapsed = 0.0
    n._vs_dwell_elapsed = 0.0
    n._vs_spiral = ON.SpiralState(0.0, 0.0, 0.0)
    n._vs_r_start = 0.0
    n._vs_ramp = None
    n._vs_center = None
    n._vs_spacing = 0.0
    n._vs_speed = 0.0
    n._vs_latch_buf = []
    n._vs_latch_rx = None
    n._vs_latched = None
    n._vs_latch_frames = int(latch_frames)
    n._vs_latch_spread = float(latch_spread)
    n._vs_stale_timeout = float(stale_timeout)
    n._vs_link_timeout = 3.0
    n._vs_veto_timeout = 10.0
    n._vs_align_tol = 1.0
    n._vs_descend_speed = 0.8
    n._vs_handoff_agl = 3.0

    # PRECISION_LAND
    n._pl_ramp = None
    n._pl_elapsed = 0.0
    n._pl_veto_elapsed = 0.0
    n._pl_lost_elapsed = 0.0
    n._pl_target_ne = None
    n._pl_alt = None
    n._pl_timeout = 60.0

    n._vision = VisionTargetBridge(stale_timeout_s=float(stale_timeout),
                                   link_timeout_s=3.0)
    n._sm = _State.HOLD
    return n


def _setpoint_msg(n_ned, e_ned, h_up):
    """`/vision/landing_setpoint` (PoseStamped, **ENU**) 모의 메시지.

    노드가 소비하는 것은 NED `[N, E, h_up]` 이므로 여기서 ENU 로 되돌려 넣는다
    (x_enu=E, y_enu=N, z_enu=h_up) — 어댑터의 축 교환까지 같이 통과시킨다.
    """
    return types.SimpleNamespace(
        header=types.SimpleNamespace(frame_id="map", stamp=None),
        pose=types.SimpleNamespace(
            position=types.SimpleNamespace(
                x=float(e_ned), y=float(n_ned), z=float(h_up)),
            orientation=types.SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0)))


def feed_frame(node, n_ned, e_ned, h_up=0.0):
    """vision 프레임 1건 도착 — 실제 콜백과 같은 경로를 탄다."""
    node._vision.update_from_setpoint(
        _setpoint_msg(n_ned, e_ned, h_up), time.monotonic())


def at_wp1(alt=CRUISE_ALT):
    return _vehicle_state((WP1[0], WP1[1], alt))


# ══════════════════════════════════════════════════════════════════
# BLOCKER-1 — `_enter_vision_search` 가 `_sm` 을 세팅하지 않았다
# ══════════════════════════════════════════════════════════════════

def test_exit_hold_enters_vision_search_state():
    """🔴 핵심 회귀. `_enter_vision_search` 첫 줄의 `self._sm = ...` 이 없으면
    여기서 상태가 `HOLD` 로 남는다 — 2026-07-28 `8cb0861` 원본의 결함."""
    n = make_node(vision_landing=True)
    n._exit_hold(at_wp1(), "테스트")
    assert n._sm == _State.VISION_SEARCH, (
        f"_exit_hold 후 상태가 {n._sm} 입니다 — _enter_vision_search 가 "
        f"self._sm 을 세팅하지 않으면 _step_hold 무한 루프가 됩니다")


def test_exit_hold_default_off_still_goes_straight_to_landing():
    """`vision_landing=false`(기본)는 종전과 100% 동일해야 한다 — 계약."""
    n = make_node(vision_landing=False)
    n._exit_hold(at_wp1(), "테스트")
    assert n._sm == _State.LANDING


def test_enter_vision_search_sets_state_on_retry_pass_too():
    """재탐색 진입도 같은 함수를 쓴다 — 회차만 바뀐다."""
    n = make_node(vision_landing=True)
    n._vs_center = np.array(WP1, dtype=float)
    n._enter_vision_search(at_wp1(), pass_idx=1)
    assert n._sm == _State.VISION_SEARCH
    assert n._vs_pass == 1


def test_enter_precision_land_sets_state():
    """대조군 — 이쪽은 처음부터 옳았다. 두 진입점이 갈라지지 않게 같이 고정한다."""
    n = make_node(vision_landing=True)
    n._vs_latched = np.array([WP1[0], WP1[1], 0.0])
    n._enter_precision_land(at_wp1())
    assert n._sm == _State.PRECISION_LAND


def test_hold_exit_is_not_re_entered_on_following_ticks():
    """🔴 **무한 루프 회귀.** 상태 전이가 빠지면 `_step_hold` 가 매 틱 안정조건을
    다시 만족해 `_exit_hold` 를 10Hz 로 재호출한다(SITL 실측 1687회).

    `_control_callback` 의 디스패치를 그대로 흉내내 연속 틱을 돌린다 —
    `_exit_hold` 호출은 **정확히 1회**여야 한다."""
    n = make_node(vision_landing=True)
    calls = []
    real_exit = n._exit_hold
    n._exit_hold = lambda state, why: (calls.append(why), real_exit(state, why))[1]

    state = at_wp1()
    for _ in range(60):                     # 6초치 (10Hz)
        if n._sm == _State.HOLD:
            n._step_hold(state)
        elif n._sm == _State.VISION_SEARCH:
            n._step_vision_search(state)
        else:
            break

    assert len(calls) == 1, (
        f"_exit_hold 가 {len(calls)}회 호출됐습니다 — HOLD 무한 루프 회귀")
    assert n._sm != _State.HOLD
    assert n._hold_stable_ticks <= ON._HOLD_STABLE_REQ, (
        f"stable 카운터가 {n._hold_stable_ticks} 까지 폭주했습니다 "
        f"(SITL 실측 회귀 신호는 stable=2205/10 이었다)")


def test_hold_timeout_exit_also_leaves_hold():
    """HOLD 종료 경로는 둘이다 — 안정 도달과 타임아웃. 둘 다 봐야 한다."""
    n = make_node(vision_landing=True)
    n._hold_timeout = 1.0
    far = _vehicle_state((0.0, 0.0, CRUISE_ALT), vel_ned=(5.0, 0.0, 0.0))
    for _ in range(30):
        if n._sm != _State.HOLD:
            break
        n._step_hold(far)
    assert n._sm == _State.VISION_SEARCH


def test_every_enter_helper_assigns_state():
    """일반화된 그물 — `_enter_*` 는 정의상 상태 진입 함수다.

    `self._sm = ...` 이 없는 `_enter_*` 가 하나라도 생기면 그 상태는 **도달
    불가**이고, 호출한 쪽은 자기가 성공했다고 믿는다. 이번 결함이 정확히 그
    형태였으므로 이름 규약 전체를 고정한다."""
    import ast
    from pathlib import Path
    src = (Path(ON.__file__)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    missing = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef)
                and node.name.startswith("_enter_")):
            continue
        assigns_sm = any(
            isinstance(t, ast.Attribute) and t.attr == "_sm"
            for a in ast.walk(node) if isinstance(a, ast.Assign)
            for t in a.targets)
        if not assigns_sm:
            missing.append(node.name)
    assert not missing, f"상태를 세팅하지 않는 진입 함수: {missing}"


# ══════════════════════════════════════════════════════════════════
# BLOCKER-2 — 래치가 제어틱이 아니라 vision 프레임을 세야 한다
# ══════════════════════════════════════════════════════════════════

def _latch_ticks(node, ticks):
    """제어틱을 `ticks` 번 돌리며 래치 성립 여부를 돌려준다(프레임 주입 없음)."""
    for _ in range(ticks):
        if node._vision_latch_update(node._vision.snapshot(time.monotonic())):
            return True
    return False


def test_single_frame_then_silence_does_not_latch():
    """🔴 **핵심 회귀 (인수인계 §6-1 #5).** setpoint 가 1건만 오고 침묵해도
    `vt.valid` 는 `vision_stale_timeout`(1.0s) 동안 True 다. 제어틱을 세면
    같은 메시지 하나가 최대 10번 계수돼 **0.2s 만에 래치**되고, 버퍼가 같은
    좌표의 사본이라 산포 필터까지 통과한다(검증 세션 실측 `[[100,50,2]]×3`).

    단발 오탐 1프레임으로 나선을 버리면 회복할 방법이 없다."""
    n = make_node(latch_frames=3)
    feed_frame(n, 100.0, 50.0, 2.0)
    assert _latch_ticks(n, 10) is False, (
        "vision 프레임 1건으로 래치가 섰습니다 — 래치가 제어틱을 세고 있습니다")
    assert len(n._vs_latch_buf) == 1


def test_repeated_ticks_on_same_frame_do_not_grow_the_buffer():
    """같은 프레임을 다시 보는 틱은 **아무 일도 하지 않는다** — 버퍼를 키우지도,
    (유도가 살아 있으므로) 비우지도 않는다."""
    n = make_node(latch_frames=3)
    feed_frame(n, 100.0, 50.0, 2.0)
    _latch_ticks(n, 8)
    assert len(n._vs_latch_buf) == 1
    feed_frame(n, 100.2, 50.1, 2.0)
    _latch_ticks(n, 8)
    assert len(n._vs_latch_buf) == 2


def test_three_distinct_frames_latch():
    """정상 경로 — 서로 다른 프레임 3건이 좁은 산포로 오면 래치가 선다."""
    n = make_node(latch_frames=3)
    got = False
    for i, (nn, ee) in enumerate([(100.0, 50.0), (100.3, 50.2), (99.8, 49.9)]):
        feed_frame(n, nn, ee, 2.0)
        got = n._vision_latch_update(n._vision.snapshot(time.monotonic()))
    assert got is True
    assert n._vs_latched is not None
    assert n._vs_latched[0] == pytest.approx(100.03, abs=0.05)


def test_spread_filter_still_rejects_scattered_frames():
    """프레임 수를 채워도 산포가 넓으면 "같은 물체"가 아니다."""
    n = make_node(latch_frames=3, latch_spread=3.0)
    got = False
    for nn, ee in [(100.0, 50.0), (120.0, 50.0), (80.0, 50.0)]:
        feed_frame(n, nn, ee, 2.0)
        got = n._vision_latch_update(n._vision.snapshot(time.monotonic()))
    assert got is False


def test_invalid_snapshot_clears_the_buffer():
    """연속성이 깨지면 버퍼를 비운다 — 띄엄띄엄 본 것을 평균 내면 실재하지 않는
    좌표가 나온다."""
    n = make_node(latch_frames=3)
    feed_frame(n, 100.0, 50.0, 2.0)
    n._vision_latch_update(n._vision.snapshot(time.monotonic()))
    assert len(n._vs_latch_buf) == 1
    # setpoint 가 stale 로 떨어진 상황 = `valid=False`
    n._vision_latch_update(n._vision.snapshot(time.monotonic() + 5.0))
    assert n._vs_latch_buf == []


def _with_veto(vt):
    """`VisionTarget` 은 frozen dataclass — veto 만 바꾼 사본을 만든다."""
    import dataclasses
    return dataclasses.replace(vt, veto=True)


def test_veto_frames_never_enter_the_buffer():
    """vision 이 거부권을 든 프레임은 증거로 세지 않는다."""
    n = make_node(latch_frames=3)
    for _ in range(5):
        feed_frame(n, 100.0, 50.0, 2.0)
        vt = _with_veto(n._vision.snapshot(time.monotonic()))
        assert n._vision_latch_update(vt) is False
    assert n._vs_latch_buf == []


def test_latch_counter_resets_on_search_entry():
    """재탐색으로 다시 들어오면 이전 회차의 프레임 계수를 물려받지 않는다."""
    n = make_node(latch_frames=3)
    feed_frame(n, 100.0, 50.0, 2.0)
    n._vision_latch_update(n._vision.snapshot(time.monotonic()))
    assert n._vs_latch_rx is not None
    n._vs_center = np.array(WP1, dtype=float)
    n._enter_vision_search(at_wp1(), pass_idx=1)
    assert n._vs_latch_rx is None
    assert n._vs_latch_buf == []


def test_search_does_not_abandon_spiral_on_one_frame():
    """상태 층에서 본 같은 계약 — 프레임 1건으로 `PRECISION_LAND` 로 넘어가지
    않는다. `_step_vision_search` 를 실제로 돌린다."""
    n = make_node(vision_landing=True, latch_frames=3)
    n._exit_hold(at_wp1(), "테스트")
    assert n._sm == _State.VISION_SEARCH
    feed_frame(n, WP1[0], WP1[1], 0.0)
    for _ in range(8):
        n._step_vision_search(at_wp1(alt=GROUND_H + n._vs_alt))
    assert n._sm == _State.VISION_SEARCH, \
        "vision 프레임 1건으로 탐색이 중단됐습니다"


def test_search_latches_and_hands_over_on_steady_frames():
    """반대 방향 — 프레임이 계속 오면 정상적으로 `PRECISION_LAND` 로 넘어간다."""
    n = make_node(vision_landing=True, latch_frames=3)
    n._exit_hold(at_wp1(), "테스트")
    st = at_wp1(alt=GROUND_H + n._vs_alt)
    for _ in range(6):
        feed_frame(n, WP1[0], WP1[1], 0.0)
        n._step_vision_search(st)
        if n._sm == _State.PRECISION_LAND:
            break
    assert n._sm == _State.PRECISION_LAND


# ══════════════════════════════════════════════════════════════════
# 파라미터 이름 — "틱" 이라는 거짓말이 되돌아오지 않게
# ══════════════════════════════════════════════════════════════════

def _declared_param_names():
    """`declare_parameter` / `get_parameter` 의 첫 인자 문자열 전부."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path(ON.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("declare_parameter", "get_parameter")
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            names.add(node.args[0].value)
    return names


def test_latch_parameter_is_named_frames_not_ticks():
    """이름이 단위를 거짓말하면 다음 사람이 같은 버그를 다시 만든다.

    (주석 안의 옛 이름은 사고 이력이라 남겨 둔다 — 여기서는 **실제 파라미터
    이름**만 본다.)"""
    names = _declared_param_names()
    assert "vision_latch_frames" in names
    assert "vision_latch_ticks" not in names, \
        "구 파라미터명이 남아 있습니다 — 단위가 프레임인데 이름은 틱입니다"


# ══════════════════════════════════════════════════════════════════
# 관측 계층 — 하니스가 노드의 실제 로그 문구를 보는가
# ══════════════════════════════════════════════════════════════════
#
# 🔴 2026-07-29 실측 회귀: `_exit_hold()` 단일 분기 도입으로 HOLD 종료 문구가
#   "WP1 도달·안정 → LANDING (…)"  →  "WP1 도달·안정 (…) → LANDING"
#   "… → 강제 LANDING"             →  "홀드 타임아웃 → LANDING"
# 로 바뀌었는데 하니스 정규식이 그대로라 `state_timeline` 에서 **LANDING 이
# 통째로 누락**됐다(R1_base: HOLD 72.50 → DONE 127.98). 비행 거동은 무변경인데
# 판정만 눈을 감은 것이라 더 위험하다.
#
# 여기서는 **모의 로그 문자열을 적어 두고 비교하지 않는다** — 그러면 같은
# 거짓말을 두 곳에 적는 셈이다. 위 테스트들이 노드를 실제로 돌리며 남긴
# `n.log.lines` 를 하니스 정규식에 그대로 먹인다.

def _harness_modules():
    import importlib
    import os
    import sys as _sys
    from pathlib import Path
    tools = Path(__file__).resolve().parents[2] / "tools" / "sitl"
    if not tools.is_dir():                      # 설치본에는 tools/ 가 없다
        pytest.skip("tools/sitl 없음 (설치 트리)")
    if str(tools) not in _sys.path:
        _sys.path.insert(0, str(tools))
    return (importlib.import_module("run_scenario"),
            importlib.import_module("analyze_run"))


def _classify(patterns, msg):
    for state, pat in patterns:
        if pat.search(msg):
            return state
    return None


def _states_from(patterns, lines):
    out = []
    for line in lines:
        s = _classify(patterns, line)
        if s is not None and s not in out:
            out.append(s)
    return out


@pytest.mark.parametrize("which", [0, 1])
def test_harness_sees_hold_exit_to_landing(which):
    """`vision_landing=false` — 하니스가 LANDING 진입을 **반드시** 봐야 한다."""
    patterns = _harness_modules()[which].STATE_ENTRY_PATTERNS
    n = make_node(vision_landing=False)
    n._exit_hold(at_wp1(), "WP1 도달·안정 (dist=0.7m speed=0.1m/s)")
    assert "LANDING" in _states_from(patterns, n.log.lines), \
        f"하니스가 LANDING 을 못 봅니다: {n.log.lines}"


@pytest.mark.parametrize("which", [0, 1])
def test_harness_sees_hold_timeout_exit(which):
    """타임아웃 경로도 마찬가지다 — 종전 "강제 LANDING" 문구는 사라졌다."""
    patterns = _harness_modules()[which].STATE_ENTRY_PATTERNS
    n = make_node(vision_landing=False)
    n._exit_hold(at_wp1(), "홀드 타임아웃")
    assert "LANDING" in _states_from(patterns, n.log.lines)


@pytest.mark.parametrize("which", [0, 1])
def test_harness_sees_f2_states(which):
    """F2 구간(VISION_SEARCH → PRECISION_LAND → LANDING)이 timeline 에 잡히는가.

    안 잡히면 앞으로의 모든 F2 SITL 판정이 눈을 감고 돈다."""
    patterns = _harness_modules()[which].STATE_ENTRY_PATTERNS
    n = make_node(vision_landing=True, latch_frames=3)
    n._exit_hold(at_wp1(), "WP1 도달·안정 (dist=0.7m speed=0.1m/s)")
    st = at_wp1(alt=GROUND_H + n._vs_alt)
    for _ in range(6):
        feed_frame(n, WP1[0], WP1[1], 0.0)
        n._step_vision_search(st)
        if n._sm == _State.PRECISION_LAND:
            break
    assert n._sm == _State.PRECISION_LAND
    # 인계고도 도달 → LANDING
    n._step_precision_land(_vehicle_state((WP1[0], WP1[1], GROUND_H + 1.0)))
    assert n._sm == _State.LANDING

    seen = _states_from(patterns, n.log.lines)
    for want in ("VISION_SEARCH", "PRECISION_LAND", "LANDING"):
        assert want in seen, f"하니스가 {want} 를 못 봅니다: seen={seen}"
    assert seen.index("VISION_SEARCH") < seen.index("PRECISION_LAND") \
        < seen.index("LANDING"), f"순서가 뒤집혔습니다: {seen}"


@pytest.mark.parametrize("which", [0, 1])
def test_harness_still_reads_archived_log_wording(which):
    """하니스는 **아카이브 런도 다시 분석**한다 — 구 문구를 버리면 종전 캠페인
    24런의 재분석이 조용히 깨진다."""
    patterns = _harness_modules()[which].STATE_ENTRY_PATTERNS
    assert _classify(patterns,
                     "WP1 도달·안정 → LANDING (dist=0.7m speed=0.1m/s)") == "LANDING"
    assert _classify(patterns,
                     "WP1 홀드 타임아웃 30s 초과 → 강제 LANDING") == "LANDING"


@pytest.mark.parametrize("which", [0, 1])
def test_harness_gps_fallbacks_count_as_landing(which):
    """F2 의 GPS 착륙 폴백 3종도 LANDING 진입이다 — 놓치면 그 런의 결말이
    timeline 에서 사라진다."""
    patterns = _harness_modules()[which].STATE_ENTRY_PATTERNS
    for msg in ("vision 생산자 사망(link ERROR) → 탐색 중단, GPS 착륙",
                "탐색 2회차 실패 (나선 완주) → GPS 착륙 폴백",
                "vision 거부권 10s 지속 (state=HOLD) → GPS 착륙 폴백",
                "정밀착륙 타임아웃 60s 초과 (수평오차 4.20m) → GPS 착륙 폴백"):
        assert _classify(patterns, msg) == "LANDING", msg


def test_analyze_canonical_order_contains_f2_states():
    """순서 검사(`order_is_subsequence_of_canonical`)에 F2 상태가 없으면
    정상 F2 런이 "비정상 순서" 로 찍힌다."""
    _, analyze = _harness_modules()
    order = analyze.CANONICAL_ORDER
    assert order.index("HOLD") < order.index("VISION_SEARCH") \
        < order.index("PRECISION_LAND") < order.index("LANDING")


def test_two_harness_pattern_tables_agree():
    """`run_scenario` 와 `analyze_run` 이 갈라지면 실행 중 판정과 사후 분석이
    다른 답을 낸다 — ENTRY 한 줄(analyze 전용)만 차이여야 한다."""
    run, analyze = _harness_modules()
    a = {s for s, _ in run.STATE_ENTRY_PATTERNS}
    b = {s for s, _ in analyze.STATE_ENTRY_PATTERNS}
    assert b - a == {"ENTRY"}, f"두 표가 갈라졌습니다: run-only={a - b}, analyze-only={b - a}"
