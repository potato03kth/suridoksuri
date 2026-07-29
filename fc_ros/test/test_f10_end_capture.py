"""F-10 — FOLLOWING **종점 포착** 회귀 테스트 (2026-07-29).

## 이 파일이 고정하는 것

종전 FOLLOWING 완료 조건은 `dist_to_end < d_end_thresh`(기본 10m) **하나뿐**이었다.
FW 는 종점을 반드시 지나친 뒤 선회반경 110~140m 로 되돌아오므로, 그 10m 원을
스치느냐가 곧 완주/미완주였다 — 즉 완료가 제어가 아니라 운이었다.

`tools/sitl/f10_endcapture_probe.py` 로 ulog 에서 직접 잰 **종점 최근접 거리**:

| 런 | 저장소 | 결과 | 1바퀴 | 2바퀴 | 3바퀴 | 임계 10m 대비 |
|---|---|---|---|---|---|---|
| `C1b_pxvehicle` (캠페인) | `3f6c517` | 완주 | 12.78 | 13.77 | **8.52** | **+1.48m** |
| `C1b_rampon` (F-9)       | `94989b6` | timeout | 12.53 | 12.89 | (접지) | **−2.53m** |
| `C1b_rampoff` (F-9)      | `94989b6` | timeout | 12.71 | 12.88 | (접지) | **−2.71m** |
| `REPRO_head_rl1200`      | `9c5d17f` | 완주 | 11.82 | 13.75 | 10.37 | ≈0 |
| `REPRO_head_rl1200_r2`   | `9c5d17f` | 완주 | 11.71 | 13.01 | **7.19** | +2.81m |

**5런이 전부 7.2~13.8m 대역에 몰려 있다.** 임계 10m 가 그 대역 한복판에
놓여 있으므로 완주/미완주는 커밋이 아니라 표본 잡음이 가른다 — "캠페인은
완주했는데 회귀했다"는 관측의 정체가 이것이다(캠페인의 여유는 **1.48m** 였다).

그래서 두 가지를 고정한다:
  ① `path_end_passed()` — 거리 원 대신 **결승선**(종점의 수직 반평면) 통과.
     횡방향으로 얼마나 벌어져 있든 마지막 세그먼트를 다 타면 완료다.
  ② `following_timeout` — R1 타임아웃 4종에서 유일하게 빠져 있던 상태.
     실측 469초 무한 선회 + 접지를 아무것도 잡지 못했다.

## 층 구분

`fc_bridge/tests/test_state_logic.py` 가 순수 함수 `path_end_passed()` 자체를
보고, 이 파일은 **노드가 그 함수를 실제 FOLLOWING 틱에서 쓰는지**와
파라미터 3중 일치(YAML↔declare↔launch)를 본다. 순수 함수만으로는
"`_step_following` 이 그것을 부르는가"를 원리적으로 볼 수 없다(F2 사고의 교훈).
"""
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml


# ──────────────────────────────────────────────────────────────────
# ROS 스텁 — 실물이 있으면 건드리지 않는다 (test_offboard_f2_state.py 와 같은 규약)
# ──────────────────────────────────────────────────────────────────

def _permissive(name):
    meta = type("_Meta", (type,), {"__getattr__": lambda cls, attr: 0})
    return meta(name, (), {
        "__init__": lambda self, *a, **k: None,
        "__getattr__": lambda self, attr: 0,
    })


def _stub(name):
    mod = types.ModuleType(name)
    mod.__path__ = []

    def _getattr(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _permissive(attr)

    mod.__getattr__ = _getattr
    return mod


def _ensure(name):
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

from fc_ros.nodes import offboard_node as ON              # noqa: E402
from fc_bridge.guidance.l1_guidance import L1Guidance     # noqa: E402

_YAML_PATH = (Path(__file__).parent.parent / "fc_ros" / "params"
              / "fc_ros_params.yaml")
_NODE_SRC = (Path(__file__).parent.parent / "fc_ros" / "nodes"
             / "offboard_node.py").read_text(encoding="utf-8")
_LAUNCH_SRC = (Path(__file__).parent.parent / "launch"
               / "phase2.launch.py").read_text(encoding="utf-8")


def _params():
    with open(_YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["offboard_node"]["ros__parameters"]


# ── 실측 상수 (위 표의 출처: logs/2026-07-29_r5_f10/notes.md) ─────────────

#: C1b 경로(정북 300m)의 종점.
C1B_END = (300.0, 0.0)
#: F-9 `C1b_rampon` 이 종점 포착에 실패한 채 지나간 실측 궤적
#: (`logs/2026-07-29_r5_f9/C1b_rampon/node.log` tick=180/200/220).
#: 이 세 점을 이은 호는 종점 10m 원에 **한 번도 들어가지 않는다**(최근접 17.2m)
#: — 그러므로 이 궤적으로 완료가 나온다면 그것은 결승선 판정뿐이다.
C1B_MISS_ARC = [(276.7, -0.7), (297.1, -21.6), (313.9, -46.0)]
C1B_MISS_POS = C1B_MISS_ARC[-1]
#: 정상 완주 런의 FOLLOWING 최대 체류 (s). `REPRO_head_rl1200_r2` 46.44s.
FOLLOWING_MAX_DWELL_S = 46.44
#: 포착 실패 시 실측된 체류 (s) — 하니스 timeout 이 없었으면 무한이었다.
FOLLOWING_RUNAWAY_S = 469.29


# ──────────────────────────────────────────────────────────────────
# 노드 껍데기 — `_step_following` FW 분기가 읽는 속성만 채운다
# ──────────────────────────────────────────────────────────────────

class _Logger:
    def __init__(self):
        self.lines = []

    def _rec(self, msg, **kw):
        self.lines.append(str(msg))

    info = warn = error = debug = fatal = _rec


def _vehicle_state(pos_ned, vel_ned=(16.0, 0.0, 0.0), yaw=0.0):
    return types.SimpleNamespace(
        pos_ned=np.array(pos_ned, dtype=float),
        vel_ned=np.array(vel_ned, dtype=float),
        yaw=float(yaw), armed=True)


def make_fw_node(pts, d_end_thresh=10.0):
    """FW FOLLOWING 만 돌릴 수 있는 `OffboardNode` 껍데기."""
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

    n._dt = 0.1
    n._is_mc = False
    n._pts = np.asarray(pts, dtype=float)
    n._v = np.full(len(n._pts), 18.0)
    n._l1_dist = 20.0
    n._guidance = L1Guidance(n._l1_dist, n._pts, n._v)
    n._cruise_alt = 50.0
    n._transition_alt = 50.0
    n._alt_slew_rate = 0.0        # 램프 비활성 — 이 파일의 관심사가 아니다
    n._alt_ramp = None
    n._d_end_thresh = float(d_end_thresh)
    n._follow_ticks = 0
    n._last_seg = None
    n._current_mode = "OFFBOARD"
    n._offboard_req_min_interval = 1.0
    return n


def _straight_path(n_pts=301, length=300.0):
    x = np.linspace(0.0, length, n_pts)
    return np.column_stack([x, np.zeros_like(x)])


def _drive(n, positions):
    """주어진 위치들을 순서대로 한 틱씩 먹인다. 완료된 틱 인덱스를 돌려준다."""
    for i, p in enumerate(positions):
        if n._step_following(_vehicle_state((p[0], p[1], 50.0))):
            return i
    return None


# ══════════════════════════════════════════════════════════════════
# ① 노드가 결승선 판정을 실제로 쓰는가
# ══════════════════════════════════════════════════════════════════

def _miss_arc(step=20):
    """`C1B_MISS_ARC` 를 한 틱 간격으로 보간한 궤적."""
    out = []
    for a, b in zip(C1B_MISS_ARC[:-1], C1B_MISS_ARC[1:]):
        a, b = np.array(a, dtype=float), np.array(b, dtype=float)
        out += [a + (b - a) * t for t in np.linspace(0.0, 1.0, step,
                                                     endpoint=False)]
    out.append(np.array(C1B_MISS_ARC[-1], dtype=float))
    return out


def test_fw_following_completes_when_it_crosses_the_finish_line_wide():
    """**이 파일의 핵심.** 종점 10m 원에 한 번도 안 들어가는 실측 궤적으로
    완료가 나와야 한다.

    F-9 `C1b_rampon` 실측이다 — 종전 코드는 여기서 완료를 못 내고 469초를
    더 돌다가 접지했다(`min_agl` −2.61m).
    """
    n = make_fw_node(_straight_path())
    end = np.array(C1B_END)
    arc = _miss_arc()
    # 이 궤적이 정말 거리 원 밖인지부터 확인한다 — 아니면 이 테스트는 무의미하다.
    assert min(float(np.linalg.norm(p - end)) for p in arc) > n._d_end_thresh

    # 경로를 정상적으로 타고 와야 세그먼트 인덱스가 마지막까지 전진한다.
    assert _drive(n, [(x, 0.0) for x in range(0, 274, 2)]) is None, \
        "경로 중간에서 완료가 나면 안 된다"
    assert _drive(n, [(p[0], p[1]) for p in arc]) is not None, \
        "결승선을 넘었는데 완료가 나지 않았다"


def test_fw_following_still_completes_inside_the_distance_circle():
    """종전 조건(거리 원)은 그대로 살아 있어야 한다 — 정상 런의 거동 보존."""
    n = make_fw_node(_straight_path())
    done = _drive(n, [(x, 0.0) for x in range(0, 296, 2)])
    assert done is not None
    # 결승선(x>300)을 넘기 전에 완료됐어야 한다.
    assert n._pts[-1][0] - 296 < n._d_end_thresh


def test_fw_following_does_not_complete_before_the_end():
    n = make_fw_node(_straight_path())
    assert _drive(n, [(x, 0.0) for x in range(0, 250, 2)]) is None


def test_fw_following_does_not_complete_from_lateral_excursion_midpath():
    """경로 중간에서 크게 벌어져도(코너 오버슈트) 완료로 오판하면 안 된다."""
    n = make_fw_node(_straight_path())
    path = [(x, 0.0) for x in range(0, 150, 2)] + \
           [(150.0 + i, -60.0) for i in range(0, 40, 2)]
    assert _drive(n, path) is None


def test_closed_loop_start_is_not_mistaken_for_the_finish():
    """폐회로 — 출발점이 종점 바로 옆이라도 출발하자마자 완료되면 안 된다.

    `R2_closed` 는 FOLLOWING 진입 시점에 종점까지 **12.18m** 뿐이다.
    `d_end_thresh` 를 키우는 방식(§4 결정 5 의 ≈57)이 깨지는 지점이고,
    결승선 판정이 `seg_idx` 게이트로 그것을 피한다는 것을 여기서 고정한다.
    """
    # 한 변 100m 사각 폐회로, 1m 간격.
    corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0),
               (0.0, 100.0), (0.0, 0.0)]
    pts = []
    for a, b in zip(corners[:-1], corners[1:]):
        a, b = np.array(a), np.array(b)
        k = int(np.linalg.norm(b - a))
        pts.extend(a + (b - a) * (np.arange(k) / k)[:, None])
    pts.append(np.array(corners[-1]))
    n = make_fw_node(np.array(pts))

    # `R2_closed` 실측 진입점 — 종점(0,0)까지 **12.18m** 뿐인 자리에서 출발한다.
    # (거리 원 임계를 ≈57 로 키우는 대안이 바로 여기서 깨진다.)
    assert _drive(n, [(x, 0.0) for x in np.arange(12.18, 30.0, 0.5)]) is None
    # 한 바퀴를 실제로 돌면 완료된다.
    lap = ([(x, 0.0) for x in range(0, 100, 2)]
           + [(100.0, y) for y in range(0, 100, 2)]
           + [(x, 100.0) for x in range(100, 0, -2)]
           + [(0.0, y) for y in range(100, -10, -2)])
    assert _drive(n, lap) is not None


def test_finish_line_logs_why_it_completed():
    """거리 원 밖에서 끝났다는 사실이 로그에 남아야 한다 — 안 남으면
    현장에서 '왜 여기서 역천이했지'를 설명할 수 없다."""
    n = make_fw_node(_straight_path())
    _drive(n, [(x, 0.0) for x in range(0, 274, 2)])
    _drive(n, [(p[0], p[1]) for p in _miss_arc()])
    assert any("결승선" in ln for ln in n.log.lines), n.log.lines[-3:]


# ══════════════════════════════════════════════════════════════════
# ② following_timeout — YAML ↔ declare ↔ launch 3중 일치
# ══════════════════════════════════════════════════════════════════

_FOLLOWING_TIMEOUT_DEFAULT = 240.0


def test_following_timeout_present_in_yaml():
    params = _params()
    assert "following_timeout" in params, "YAML에 'following_timeout' 이 없습니다"
    assert params["following_timeout"] == pytest.approx(
        _FOLLOWING_TIMEOUT_DEFAULT)


def test_following_timeout_declare_default_matches_yaml():
    m = re.search(
        r'declare_parameter\(\s*"following_timeout"\s*,\s*([0-9.]+)\s*\)',
        _NODE_SRC)
    assert m, "offboard_node.py 에 'following_timeout' declare_parameter 가 없습니다"
    assert float(m.group(1)) == pytest.approx(_params()["following_timeout"])


def test_following_timeout_exposed_as_launch_arg():
    assert '"following_timeout"' in _LAUNCH_SRC, \
        "phase2.launch.py 에 'following_timeout' 오버라이드 통로가 없습니다"


def test_following_timeout_enabled_by_default():
    """0 이하면 비활성이다 — 끄고 배포하면 실기체에 아무 효과가 없다."""
    assert _params()["following_timeout"] > 0.0


def test_following_timeout_has_margin_over_measured_normal_dwell():
    """정상 완주 런을 죽이면 안 된다. 실측 최대 46.44s 의 3배 이상."""
    assert _params()["following_timeout"] >= FOLLOWING_MAX_DWELL_S * 3.0


def test_following_timeout_would_have_caught_the_c1b_runaway():
    """실측 469.29초 무한 선회를 잡았어야 한다."""
    assert FOLLOWING_RUNAWAY_S > _params()["following_timeout"]


def test_following_timeout_is_wired_into_the_control_loop():
    """파라미터만 있고 호출부가 없으면 아무 일도 안 일어난다 (F2 의 교훈)."""
    assert '"FOLLOWING", self._follow_ticks * self._dt' in _NODE_SRC, \
        "_control_callback 의 FOLLOWING 분기에 _timeout_fallback 배선이 없습니다"
    assert "self._following_timeout" in _NODE_SRC


def test_following_timeout_uses_the_shared_safety_fallback():
    """새 폴백 경로를 만들지 않는다(§4 결정 2) — `_timeout_fallback` 재사용."""
    seg = _NODE_SRC.split("elif self._sm == _State.FOLLOWING:")[1][:900]
    assert "_timeout_fallback(" in seg
    assert "_request_override" not in seg, \
        "FOLLOWING 분기가 안전경로를 직접 호출하고 있습니다"


# ══════════════════════════════════════════════════════════════════
# ③ _path_end_dir — 결승선의 법선
# ══════════════════════════════════════════════════════════════════

def test_path_end_dir_is_the_last_leg_direction():
    n = make_fw_node(_straight_path())
    d = n._path_end_dir()
    assert d[0] > 0 and abs(d[1]) < 1e-9


def test_path_end_dir_backs_off_duplicate_tail_points():
    """마지막 두 점이 겹쳐도 방향을 잡아야 한다 (플래너 보간 산출물)."""
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [20.0, 0.0]])
    n = make_fw_node(pts)
    d = n._path_end_dir()
    assert d[0] > 0 and abs(d[1]) < 1e-9


def test_path_end_dir_gives_up_on_a_degenerate_path():
    """전부 같은 점이면 영벡터 — 판정을 포기해 조기 완료를 막는다."""
    pts = np.array([[5.0, 5.0]] * 4)
    n = make_fw_node(pts)
    assert float(np.linalg.norm(n._path_end_dir())) == 0.0
