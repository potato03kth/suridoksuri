"""파라미터·launch 인자 위생 검사 — rclpy/launch 불필요한 순수 로직.

2026-07-28 실비행 2건이 launch 인자에 섞인 U+00A0(non-breaking space) **하나**로
날아간 뒤 신설했다. 인자를 나누는 공백이 비-ASCII 공백이면 쉘이 단어 분리를 하지
않아 **뒤 인자가 앞 인자의 값 안으로 통째로 흡수된다.**

  flight01  토큰이 U+00A0 단독이 돼 ros2 launch 가
            `malformed launch argument ' '` 로 즉사 — 비행 자체가 없었다
  flight02  `vehicle_type:=vtol<U+00A0>transition_alt:=20.0` 이 한 토큰이 되어
            **조용히** 기동했다. launch 가 첫 `:=` 에서만 쪼개므로 결과는
            vehicle_type="vtol transition_alt:=20.0" 이고, transition_alt 라는 키는
            launch 에 도달조차 못 했다 (물증: 노드에 넘어간 오버라이드 파일
            `/tmp/launch_params_qd2l7x1s` — `logs/2026-07-28_flight02/notes.md` ①).
            그래서 천이 고도가 YAML 기본값 50m 로 실행됐고, 지시값 21.2m 였다면
            1.2m 였을 고도계단(F-9)이 **-29.6m** 로 최대폭 발현했다.
            쓰레기 문자열이 된 vehicle_type 은 `_is_mc` 판정이 `== "mc"` 라 우연히
            VTOL 로 동작했을 뿐 **아무 검증에도 걸리지 않았다.**

조용한 오적용보다 기동 실패가 낫다 — 그래서 걸리면 예외를 던진다.

그물은 3겹이고 각각 다른 것을 본다:

  1) `check_launch_argv`  sys.argv 원본 — **선언 안 된 인자(오타)까지 잡는 유일한 층.**
                          ros2 launch 는 선언 안 된 `key:=value` 를 에러 없이 삼키므로
                          오타난 인자는 아무 신호 없이 YAML 기본값으로 날아간다.
                          해석된 값에는 그 인자가 아예 없어 2)로는 볼 수 없다.
  2) `phase2.launch.py:_arg`  해석된 값 — 값 안에 남은 ':=' 가 흡수의 결정적 증거.
                          인자가 argv 아닌 경로(include, 기본값)로 들어와도 덮는다.
  3) `check_choice`       열거형 유효성 — launch 를 안 거치는 경로(직접 `ros2 run`,
                          테스트, YAML 오타)까지 덮는 마지막 층.

이 모듈이 launch 파일과 노드 양쪽에서 import 되는 이유가 그것이다. 한 곳에 두어야
세 층이 같은 문자 집합을 보고, rclpy·launch 없이 테스트할 수 있다
(`fc_ros/test/test_param_hygiene.py`).
"""
import re
import sys
import unicodedata

# `key:=value` 토큰. launch 와 같은 규칙으로 **첫** ':=' 에서만 쪼갠다.
ARG_RE = re.compile(r"^([^:=]+):=(.*)$", re.DOTALL)

# 눈으로는 보통 공백과 구별되지 않는 공백·제로폭 문자들.
# **반드시 이스케이프로 적는다** — 소스에 실물을 박아두면 이 파일 자체가 같은 함정이 된다.
BAD_SPACE = frozenset(
    "\u00a0"                                       # NO-BREAK SPACE — 이번 사고의 범인
    "\u1680"                                       # OGHAM SPACE MARK
    "\u2000\u2001\u2002\u2003\u2004\u2005"        # EN QUAD ~ FOUR-PER-EM SPACE
    "\u2006\u2007\u2008\u2009\u200a"              # SIX-PER-EM ~ HAIR SPACE
    "\u200b"                                       # ZERO WIDTH SPACE
    "\u202f"                                       # NARROW NO-BREAK SPACE
    "\u205f"                                       # MEDIUM MATHEMATICAL SPACE
    "\u3000"                                       # IDEOGRAPHIC SPACE
    "\ufeff"                                       # ZERO WIDTH NO-BREAK SPACE (BOM)
)

# 열거형 파라미터의 허용값. 오타·오염을 **비행 중이 아니라 기동 시점에** 잡는다.
VEHICLE_TYPES = ("vtol", "mc")
ENTRY_MODES   = ("pre_takeoff", "mid_flight")


def describe(ch: str) -> str:
    """오염 문자를 사람이 읽을 수 있게 — "U+00A0 NO-BREAK SPACE"."""
    try:
        return f"U+{ord(ch):04X} {unicodedata.name(ch)}"
    except ValueError:              # 이름 없는 코드포인트
        return f"U+{ord(ch):04X}"


def check_launch_argv(declared, argv=None) -> None:
    """`key:=value` 토큰을 검사해 오염·오타를 launch 가 시작되기 전에 잡는다.

    Args:
        declared: 이 launch 파일이 선언한 인자 이름 집합.
        argv:     검사할 토큰 목록. 기본은 `sys.argv[1:]` (테스트에서 주입 가능).

    Raises:
        RuntimeError: 오염·오타가 하나라도 있으면. 발견한 문제를 **전부** 모아
            한 번에 보고한다 — 하나 고치고 다시 돌리는 왕복을 없애기 위해서다.
    """
    if argv is None:
        argv = sys.argv[1:]

    problems = []
    for tok in argv:
        m = ARG_RE.match(tok)
        if not m:
            continue
        key, val = m.group(1), m.group(2)

        bad_key = [describe(c) for c in key if not c.isascii()]
        if bad_key:
            problems.append(
                f"  인자 이름에 비-ASCII 문자: {key!r}  ({', '.join(bad_key)})")
            continue
        if key not in declared:
            problems.append(
                f"  선언되지 않은 인자: {key!r}  — 오타이거나 이름이 오염된 것이다. "
                f"그냥 두면 YAML 기본값으로 조용히 날아간다")
            continue
        bad_val = [describe(c) for c in val if c in BAD_SPACE]
        if bad_val:
            problems.append(
                f"  인자 {key!r} 의 값에 비정상 공백: {val!r}  ({', '.join(bad_val)})")

    if problems:
        raise RuntimeError(
            "launch 인자가 오염됐다 — 실행을 중단한다.\n"
            + "\n".join(problems)
            + "\n\n터미널에 직접 다시 입력할 것"
              "(문서·채팅에서 복사하면 U+00A0 가 섞인다).\n"
            f"선언된 인자: {', '.join(sorted(declared))}")


def check_value(name: str, val: str) -> str:
    """인자 값 하나를 검사한다. 통과하면 값을 그대로 돌려준다.

    Raises:
        ValueError: 비정상 공백이 있거나, 값 안에 ':=' 가 남아 있으면
            (뒤 인자가 이 인자의 값으로 흡수됐다는 결정적 증거).
    """
    bad = [describe(c) for c in val if c in BAD_SPACE]
    if bad:
        raise ValueError(
            f"launch 인자 '{name}' 값에 비정상 공백이 들어 있다: {val!r}  "
            f"({', '.join(bad)})\n"
            f"  → 인자 사이는 보통 공백(U+0020)으로 띄울 것. "
            f"복사·붙여넣기로 섞여 들어오는 경우가 대부분이다.")
    if ":=" in val:
        raise ValueError(
            f"launch 인자 '{name}' 값 안에 ':=' 가 들어 있다: {val!r}\n"
            f"  → 뒤 인자가 이 인자의 값으로 흡수됐다는 뜻이다"
            f"(2026-07-28 flight02와 동일한 실패). "
            f"인자 사이를 보통 공백으로 띄웠는지 확인할 것.")
    return val


def check_choice(name: str, val, allowed) -> str:
    """열거형 파라미터를 허용값과 대조한다. 통과하면 정규화된 값을 돌려준다.

    이 그물이 없어서 flight02 의 오염된 `vehicle_type` 이 통과했다 (모듈 docstring).

    Raises:
        ValueError: 허용값에 없으면.
    """
    v = str(val).strip().lower()
    if v not in allowed:
        raise ValueError(
            f"파라미터 '{name}' 값이 유효하지 않다: {val!r}\n"
            f"  허용값: {' | '.join(allowed)}\n"
            f"  → 값 안에 ':=' 나 눈에 안 보이는 공백이 섞여 있다면 launch 인자가 "
            f"흡수된 것이다(2026-07-28 flight02). 인자 사이를 보통 공백으로 띄울 것.")
    return v
