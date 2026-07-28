"""launch 인자·파라미터 위생 검사 테스트 — rclpy/launch 불필요, 순수 로직만 검증.

2026-07-28 실비행 2건(flight01 즉사 / flight02 조용한 오적용)을 회귀로 박제한다.
사고 경위는 `fc_ros/param_hygiene.py` docstring 과
`logs/2026-07-28_flight02/notes.md` ①.
"""
import pytest

from fc_ros.param_hygiene import (
    BAD_SPACE, VEHICLE_TYPES, ENTRY_MODES,
    describe, check_launch_argv, check_value, check_choice,
)

NBSP = " "

# flight02 에서 실제로 쉘을 통과한 토큰. 이 한 줄이 비행을 망쳤다.
FLIGHT02_TOKEN = f"vehicle_type:=vtol{NBSP}transition_alt:=20.0"

DECLARED = {"vehicle_type", "transition_alt", "waypoints", "v_cruise"}


# ── 문자 집합 ───────────────────────────────────────────────────────────────

def test_bad_space_covers_nbsp():
    """이번 사고의 범인 U+00A0 이 반드시 들어 있어야 한다."""
    assert NBSP in BAD_SPACE


def test_bad_space_excludes_ordinary_space():
    """보통 공백·탭·개행은 정상이다 — 잡으면 정상 사용까지 막는다."""
    for ch in (" ", "\t", "\n", "\r"):
        assert ch not in BAD_SPACE


def test_source_has_no_literal_bad_space():
    """검사 모듈 자신이 같은 함정이 되면 안 된다 — 실물이 아니라 이스케이프로 적을 것."""
    from pathlib import Path
    import fc_ros.param_hygiene as mod
    text = Path(mod.__file__).read_text(encoding="utf-8")
    found = [f"line {n}: U+{ord(c):04X}"
             for n, line in enumerate(text.split("\n"), 1)
             for c in line if c in BAD_SPACE]
    assert not found, f"소스에 실물 오염문자가 박혀 있다: {found}"


def test_describe_names_the_character():
    assert describe(NBSP) == "U+00A0 NO-BREAK SPACE"
    assert describe("　") == "U+3000 IDEOGRAPHIC SPACE"


# ── check_launch_argv — argv 원본 층 ────────────────────────────────────────

def test_argv_clean_passes():
    check_launch_argv(DECLARED, ["vehicle_type:=vtol", "transition_alt:=20.0"])


def test_argv_flight02_token_is_caught():
    """🔴 핵심 회귀: flight02 를 망친 그 토큰이 반드시 걸려야 한다."""
    with pytest.raises(RuntimeError) as e:
        check_launch_argv(DECLARED, [FLIGHT02_TOKEN])
    assert "U+00A0" in str(e.value)


def test_argv_undeclared_arg_is_caught():
    """오타난 인자는 ros2 launch 가 조용히 삼킨다 — 이 층만이 잡을 수 있다."""
    with pytest.raises(RuntimeError) as e:
        check_launch_argv(DECLARED, ["transtion_alt:=20.0"])
    assert "선언되지 않은 인자" in str(e.value)


def test_argv_non_ascii_key_is_caught():
    with pytest.raises(RuntimeError) as e:
        check_launch_argv(DECLARED, [f"vehicle{NBSP}type:=vtol"])
    assert "비-ASCII" in str(e.value)


def test_argv_reports_all_problems_at_once():
    """하나 고치고 다시 돌리는 왕복을 없애려면 전부 모아 보고해야 한다."""
    with pytest.raises(RuntimeError) as e:
        check_launch_argv(DECLARED, ["oops:=1", "alsobad:=2"])
    msg = str(e.value)
    assert "oops" in msg and "alsobad" in msg


def test_argv_ignores_non_arg_tokens():
    """`ros2 launch fc_ros phase2.launch.py` 같은 위치인자는 검사 대상이 아니다."""
    check_launch_argv(DECLARED, ["fc_ros", "phase2.launch.py"])


def test_argv_splits_on_first_assign_only():
    """launch 와 같은 규칙 — 값에 ':=' 가 더 있어도 키는 첫 토막이다."""
    with pytest.raises(RuntimeError):
        check_launch_argv(DECLARED, [FLIGHT02_TOKEN])
    # 키는 선언된 vehicle_type 으로 잡히고, 걸린 이유는 값 오염이어야 한다
    with pytest.raises(RuntimeError) as e:
        check_launch_argv(DECLARED, [FLIGHT02_TOKEN])
    assert "'vehicle_type'" in str(e.value)


# ── check_value — 해석된 값 층 ──────────────────────────────────────────────

def test_value_clean_passes():
    assert check_value("v_cruise", "18.0") == "18.0"
    assert check_value("waypoints", "[0.0,0.0,50.0, 300.0,0.0,50.0]").startswith("[")


def test_value_absorbed_arg_is_caught():
    """흡수된 값에는 ':=' 가 남는다 — 공백을 못 봐도 이걸로 잡힌다."""
    with pytest.raises(ValueError) as e:
        check_value("vehicle_type", "vtol transition_alt:=20.0")
    assert ":=" in str(e.value)


def test_value_nbsp_is_caught():
    with pytest.raises(ValueError):
        check_value("v_cruise", f"18.0{NBSP}")


def test_value_empty_passes():
    """빈 값 = YAML 기본값 사용. 정상 경로다."""
    assert check_value("v_cruise", "") == ""


# ── check_choice — 열거형 층 ───────────────────────────────────────────────

def test_choice_accepts_valid():
    assert check_choice("vehicle_type", "vtol", VEHICLE_TYPES) == "vtol"
    assert check_choice("vehicle_type", "MC", VEHICLE_TYPES) == "mc"
    assert check_choice("entry_mode", "mid_flight", ENTRY_MODES) == "mid_flight"


def test_choice_rejects_flight02_garbage():
    """🔴 핵심 회귀: flight02 의 오염된 vehicle_type 이 통과하면 안 된다.

    당시엔 `_is_mc = (value == "mc")` 뿐이라 이 쓰레기 문자열이 조용히 VTOL 로
    동작했다. 검증이 있었다면 기동 시점에 멈췄을 것이다.
    """
    garbage = f"vtol{NBSP}transition_alt:=20.0"
    with pytest.raises(ValueError) as e:
        check_choice("vehicle_type", garbage, VEHICLE_TYPES)
    assert "vtol | mc" in str(e.value)


def test_choice_rejects_typo():
    with pytest.raises(ValueError):
        check_choice("entry_mode", "midflight", ENTRY_MODES)


def test_choice_strips_ordinary_whitespace():
    """보통 공백은 관용한다 — YAML 에서 흔하고 위험하지 않다."""
    assert check_choice("vehicle_type", "  vtol  ", VEHICLE_TYPES) == "vtol"
