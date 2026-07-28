"""vision→fc shim의 **순수 변환 로직** — JSON Lines 레코드 → "발행 계획"(ROS 메시지 서술).

`docs/vision_fc_interface.md` §8 권고 R4의 컨테이너 쪽 절반(Phase 2)이다. 구조:

```
[호스트]  vision main.py --target-sink  (picam-venv, Py3.12, ROS 없음)   ← Phase 1
            │  127.0.0.1:8091  TCP / JSON Lines
            ▼
[컨테이너] shim_node.py (Humble, Py3.10)  ← rclpy를 import하는 유일한 vision 파일
            │      └ 이 파일(shim_core.py): rclpy도 vision도 import하지 않는 순수 로직
            ▼
          /vision/target_pose · /vision/target_status · [옵션] /mavros/landing_target/raw
```

## 🔴 이 파일이 stdlib만 쓰는 이유 — 취향이 아니라 강제다

`vision.core.wire`를 import하면 **컨테이너에서 즉시 죽는다.** 체인이
`wire.py → core/target.py → import cv2`인데 **fc 컨테이너에 cv2가 없다**(실측). numpy는
있지만(1.21.5) cv2는 없다. 그래서 계약 상수(`SCHEMA_VERSION`/필수키)를 **의도적으로 복제**한다
— R4의 핵심 가치가 "두 도메인 사이에 빌드 의존을 만들지 않는 것"이라 이 중복은 비용이 아니라
설계 목표 자체다.

⚠️ 복제의 진짜 위험은 "조용한 드리프트"(wire.py가 스키마를 바꿨는데 여기가 안 따라감)인데,
그건 **랩탑 테스트가 잡는다** — `tests/test_shim_core.py`가 `vision.core.wire`와 이 파일의
상수를 직접 대조한다(랩탑엔 cv2가 있으므로 양쪽 다 import된다). 즉 **런타임은 격리, 검증은
대조**다.

## 두 클록의 역할 분담 (§4.6, Phase 2에서 실측으로 확정)

**실측(2026-07-28, 실기체):** 호스트와 `fc` 컨테이너의 time namespace inode가
`time:[4026531834]`로 **동일**하고 `/proc/self/timens_offsets`가 양쪽 다
`monotonic 0 0 / boottime 0 0`이다. 컨테이너 `time.monotonic_ns()`가 호스트
`/proc/uptime`과 일치한다(145453.053 vs 145453.06). → **`CLOCK_MONOTONIC`은 완전히 같은
기준이다.** Phase 1이 "추정"으로 남겼던 최대 리스크가 닫혔다.

그래서 역할을 이렇게 나눈다:

| 용도 | 쓰는 클록 | 근거 |
|---|---|---|
| `header.stamp` | 레코드의 **`stamp_wall_ns`** | ROS 기본 클록(`RCL_SYSTEM_TIME`)이 `CLOCK_REALTIME`이다. 여기에 monotonic(≈145453초)을 넣으면 `node.get_clock().now()`(≈1.7e9초)와 비교하는 mavros/fc_ros 쪽에서 전부 무의미해진다 |
| **stale 판정** | **`stamp_monotonic_ns`** | 벽시계는 NTP 점프·시간 되돌림에 흔들린다. shim이 **자기** `time.monotonic_ns()`로 나이를 재는 것이 위 실측으로 정당해졌다 — 같은 기준이므로 뺄셈이 성립한다 |

⚠️ **shim은 stale을 이유로 발행을 막지 않는다.** 잰 나이(`age_s`)를 KeyValue로 싣고
`stale_warn_s`를 넘으면 `level`만 올린다. 최종 거부는 FC의 몫이다 — §5.4가 "소비자는 vision의
협조 없이 스스로 나이를 잰다"고 못박았고, `stale_timeout_s`를 vision 쪽에 두지 않은 이유도
그것이다(`vision/CLAUDE.md` "target_sink 소켓 기본값" 절).

## 🔴 침묵/무효/사망의 3분법이 토픽 배치를 결정한다 (§5.4)

Phase 1 계약: **`valid=false` = "안 보임", 침묵/EOF = "죽음".** 이 구분이 shim에서 살아남게
하려면 두 토픽의 역할이 달라야 한다.

| 상황 | `/vision/target_pose` | `/vision/target_status` |
|---|---|---|
| 정상 검출 | **발행** | `OK` |
| 안 보임(`valid=false`) | **침묵**(실을 좌표가 없다) | **`WARN` + 사유** ← 여기서 구분된다 |
| 생산자 사망(EOF) | 침묵 | **`ERROR` + `producer_eof`** |
| shim 자신도 사망 | 침묵 | **침묵** |

즉 **pose 토픽의 침묵만으로는 "안 보임"과 "죽음"을 못 가른다 — 소비자는 반드시 status를
같이 봐야 한다.** pose에 0이나 NaN을 채워 "계속 발행"하는 안은 기각했다: 0은
`core/wire.py`가 "가장 위험한 거짓말"이라 부른 그것(기체 바로 아래에 타겟이 있다)이고,
NaN도 좌표 자리에 들어가면 하류 산술을 조용히 오염시킨다.

## header.stamp 동기 — `DiagnosticStatus`가 아니라 `DiagnosticArray`인 이유

§3.2가 "두 토픽으로 나누면 반드시 `header.stamp`를 같은 값으로 채우고 소비자는 stamp가
일치하는 쌍만 유효로 본다"를 계약으로 걸었는데, **`diagnostic_msgs/DiagnosticStatus`에는
header가 없다**(실기체에서 msg 원문 확인: `level/name/message/hardware_id/values`뿐).
header를 가진 것은 `DiagnosticArray`(`std_msgs/Header header` + `DiagnosticStatus[] status`)다.
그래서 status 토픽 타입은 **`DiagnosticArray`** 이고, 한 배열 안에 `target`/`state` 두
`DiagnosticStatus`를 **같은 stamp로 함께** 싣는다.

## `target`과 `state_hint`를 frame_id로 짝짓는다 (`_Pairer`)

와이어는 매 프레임 `target` 1건 + `state_hint` 1건을 **따로** 보낸다. 그대로 흘려보내면 두
레코드가 서로 다른 stamp로 나가 위 계약이 깨진다. 그래서 `target`을 잠깐 들고 있다가 같은
`frame_id`의 `state_hint`가 오면 **한 쌍으로** 내보낸다. `main.py`가 항상 target→state_hint
순으로 같은 프레임에 발행하므로 대기는 실질적으로 0프레임이다.

- `state_hint`가 안 오고 다음 `target`이 오면 → 들고 있던 것을 **혼자 내보낸다**(`state`
  DiagnosticStatus는 `level=WARN`, `message="state_hint_missing"`). 유도를 볼모로 잡지 않는다.
- 짝 없는 `state_hint`(선행 target 없음)는 state 단독 status로 내보낸다.

## `LandingTarget` 피벗 경로 — 🔴 두 개의 함정

**① `frame` 상수를 절대 쓰지 마라.** `mavros_msgs/msg/LandingTarget.msg`의 상수와 실제
`MAV_FRAME` enum이 어긋나 있다(같은 기체에서 두 파일 직접 대조):

| 이름 | `LandingTarget.msg` | 실제 `MAV_FRAME`(`common.hpp:154~167`) |
|---|---|---|
| `LOCAL_NED` | 2 | **1** |
| `MISSION` | 3 | **2** |
| `BODY_FRD` | **상수 자체가 없음** | **12** |

`landtarget_cb`는 `static_cast<MAV_FRAME>(req->frame)`으로 **실제 enum 기준** 분기라,
`LandingTarget.LOCAL_NED`(=2)를 넣으면 `MAV_FRAME::MISSION`으로 읽혀 `default` 분기 →
`position_valid=false`로 조용히 죽는다. 그래서 이 파일은 **정수 리터럴 12**를 상수
`MAV_FRAME_BODY_FRD`로 박아둔다(회귀테스트 + `shim_node.py` 소스 원문 검사로 이중 고정).
⚠️ 하필 `LandingTarget.msg`에 `GLOBAL_TERRAIN_ALT_INT = 12`가 있어 "상수를 썼는데 값은 맞는"
경로가 존재한다 — 값이 맞아도 이름이 거짓말이므로 쓰지 않는다.

**② `frame=12`에 넣을 좌표는 FRD가 아니라 FLU다.** mavros 2.14.0 원문 확인:

```cpp
case MAV_FRAME::BODY_FRD:
    position = ftf::transform_frame_baselink_aircraft(Eigen::Vector3d(tr.translation()));
```

`baselink`(FLU) → `aircraft`(FRD) 변환을 **플러그인이 한다.** 즉 ROS 쪽 입력은 항상 FLU다
(mavros 전역 관례). 여기에 `position_frd`를 넣으면 변환이 **두 번** 걸려 y·z 부호가 뒤집힌다.
→ 이 파일은 pose 토픽이든 LandingTarget이든 **전부 `position_flu`만** 쓴다. (`position_frd`는
mavros를 거치지 않고 MAVLink를 직접 만들 때를 위한 값으로 와이어에 남아 있는 것이다.)

**③ 기본값은 꺼짐이다.** 실기체 `px4_config.yaml:214`가 `listen_lt: false`라
`/mavros/landing_target/raw`에 **구독자가 아예 없다**. 켜는 것은 FC 도메인 결정(D2)이므로
shim은 `publish_landing_target=False`를 기본값으로 두고 코드만 준비해 둔다.

## 손실 압축은 shim의 책임 (§3.1)

`LandingTarget`에 자리가 없는 `confidence`/`calib_accuracy`/`not_for_closed_loop_30cm`/
`target_type`/`command`는 **`DiagnosticStatus.values`(KeyValue)로 전부** 나간다. 와이어가
무손실로 실어 온 provenance를 shim에서 버리지 않는다.

## 🔴 초록구역 0.105m — shim은 z를 고치지 않는다

초록구역 레코드는 `meta.plane_reference="mat_top_surface"` / `platform_height_m=0.105`를
싣는다. 비전이 재는 것은 **매트 윗면까지의 거리**이고 기체는 그 윗면에 내려앉으므로 착륙
자체엔 보정이 필요 없다 — 그래서 **z를 건드리지 않고 그대로 발행한다.**

보정하지 않는 적극적 이유: 라이다 빔이 **매트 위에 떨어지면 라이다도 윗면을 보므로 둘이
일치**하고, 빔이 **매트 밖 지면에 떨어질 때만** 0.105m 어긋난다. 어느 쪽인지는 코드가 알 수
없다 — 모르는 기하를 가정해 조용히 더하는 것이 이 저장소가 금지하는 그 실수다(캘리브 해상도
불일치를 자동 재스케일하지 않은 것과 같은 판단, `vision/CLAUDE.md`). 대신 어긋남을 **명시적
숫자로 수출**한다: `plane_reference` / `platform_height_m` /
`ground_agl_minus_vision_z_m`(= 지면기준 AGL − 비전 z, 빔이 매트 밖일 때). FC가 라이다와
섞을 때 이 값을 보고 스스로 정한다.

## 🔴 `orientation`을 Pose에 싣지 않는다

와이어의 `orientation`은 **카메라 광학 프레임**(`orientation_frame:"cam_optical"`)이다.
그런데 `PoseWithCovarianceStamped`는 position과 orientation이 **하나의 `frame_id`를 공유**한다
— body FLU 좌표 옆에 cam_optical 쿼터니언을 놓으면 그 자체가 프레임 거짓말이고, 이 저장소가
이미 당한 사고 유형(`pos_ned` vs `vel_ned` 부호 규약 불일치, §4.2)과 정확히 같은 종류다.

그래서 **Pose의 orientation은 단위 쿼터니언(0,0,0,1)** 이고 회전 공분산 대각을
`unknown_covariance`로 채워 "모른다"를 ROS 관례대로 표시한다. 원본 쿼터니언은 버리지 않고
KeyValue `orientation_cam_optical_xyzw`로 나간다(§7.3 provenance). 정사각 타겟은 어차피 90°
자기대칭이라 자세가 물리적으로 무의미하기도 하다(`vision/CLAUDE.md` 초록구역 절).

## `level`에 `not_for_closed_loop_30cm`을 태우지 않는 이유

이 플래그는 **지금 100% True**다(실측 캘리브 보류). 이걸 WARN에 태우면 `level`이 **항상**
WARN이라 진짜 고장이 묻힌다. 이건 고장이 아니라 **바닥 고도 제약**이므로 KeyValue
(`not_for_closed_loop_30cm` + 이미 번역된 `closed_loop_floor_agl_m`)로만 나간다.
계약 해석은 "폐루프 금지"가 아니라 **"최종 커밋 금지"** 다 — 바닥까지 정렬 후 인계
(`core/wire.py::closed_loop_floor_agl_m`).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Optional

# ──────────────────────────────────────────────────────────────────────────────
# 와이어 계약 (vision/core/wire.py 복제 — 파일 docstring "stdlib만 쓰는 이유" 참조)
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA_VERSION = 1

RECORD_TYPE_TARGET = "target"
RECORD_TYPE_STATE_HINT = "state_hint"

ORIENTATION_FRAME_CAM = "cam_optical"

REQUIRED_TARGET_KEYS = (
    "schema_version", "type", "seq", "frame_id", "valid", "reason",
    "timestamp", "stamp_monotonic_ns", "stamp_wall_ns", "clock_offset_ns",
    "position_cam", "position_flu", "position_frd",
    "orientation", "orientation_frame", "uncertainty",
    "confidence", "target_type",
    "calib_accuracy", "not_for_closed_loop_30cm", "calib_id",
    "closed_loop_floor_agl_m", "mount_yaw_psi_m_rad", "mount_yaw_measured",
)

REQUIRED_STATE_HINT_KEYS = (
    "schema_version", "type", "seq", "frame_id",
    "stamp_monotonic_ns", "stamp_wall_ns", "clock_offset_ns",
    "state", "command_hint", "command_is_advisory", "reason",
    "blind_duration_s", "scale_source",
)

# ──────────────────────────────────────────────────────────────────────────────
# MAVLink / mavros 상수 — 🔴 msg 상수를 쓰지 않는다 (파일 docstring "두 개의 함정")
# ──────────────────────────────────────────────────────────────────────────────

#: 실제 `MAV_FRAME::BODY_FRD`. `common.hpp:167`에서 직접 읽은 값이고
#: `mavros_msgs/msg/LandingTarget.msg`에는 **이 이름의 상수가 아예 없다.**
MAV_FRAME_BODY_FRD = 12
#: 실제 `MAV_FRAME::LOCAL_NED`(`common.hpp:154`). msg 상수는 2로 **틀렸다.** 지금 안 쓰지만
#: 누가 "LOCAL_NED로 바꿔볼까" 할 때 msg 상수로 손이 가지 않게 정답을 여기 박아둔다.
MAV_FRAME_LOCAL_NED = 1

#: `LANDING_TARGET_TYPE` — 이쪽은 msg 상수와 MAVLink가 일치한다(직접 대조 확인).
LANDING_TARGET_TYPE_VISION_FIDUCIAL = 2
LANDING_TARGET_TYPE_VISION_OTHER = 3

# `diagnostic_msgs/DiagnosticStatus`의 level 상수. msg를 import하지 않으므로 여기 복제한다
# (프레임 상수와 달리 이쪽은 MAVLink enum과 대조할 것이 없어 어긋날 여지가 없다).
LEVEL_OK = 0
LEVEL_WARN = 1
LEVEL_ERROR = 2
LEVEL_STALE = 3

# ──────────────────────────────────────────────────────────────────────────────
# 기본값 (매직넘버 금지, §7.3 — 근거는 vision/CLAUDE.md "shim 노드 기본값" 절)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8091
DEFAULT_POSE_TOPIC = "/vision/target_pose"
DEFAULT_STATUS_TOPIC = "/vision/target_status"
DEFAULT_LANDING_TARGET_TOPIC = "/mavros/landing_target/raw"
#: body FLU의 ROS 표준 프레임 이름. mavros가 baselink=FLU로 쓰는 그 이름이다.
DEFAULT_POSE_FRAME_ID = "base_link"
DEFAULT_STALE_WARN_S = 0.5
#: `uncertainty`가 None(=지금 항상)일 때 공분산 대각에 넣는 "모름" 값. 0은 "완벽히 확실"이라는
#: 거짓말이라 쓸 수 없다.
DEFAULT_UNKNOWN_COVARIANCE = 1e6
DEFAULT_HARDWARE_ID = "vision"

STATUS_NAME_TARGET = "vision/target"
STATUS_NAME_STATE = "vision/state"
STATUS_NAME_LINK = "vision/link"


@dataclass
class ShimConfig:
    """shim 파라미터 전량. 전부 이름 있는 필드 — 매직넘버 금지(§7.3)."""

    pose_topic: str = DEFAULT_POSE_TOPIC
    status_topic: str = DEFAULT_STATUS_TOPIC
    landing_target_topic: str = DEFAULT_LANDING_TARGET_TOPIC
    pose_frame_id: str = DEFAULT_POSE_FRAME_ID
    #: 🔴 기본 꺼짐 — `px4_config.yaml:214`가 `listen_lt: false`라 구독자가 없다(D2, FC 결정).
    publish_landing_target: bool = False
    stale_warn_s: float = DEFAULT_STALE_WARN_S
    unknown_covariance: float = DEFAULT_UNKNOWN_COVARIANCE
    hardware_id: str = DEFAULT_HARDWARE_ID


# ──────────────────────────────────────────────────────────────────────────────
# 발행 계획 (ROS 메시지를 **서술**하기만 한다 — rclpy 타입이 여기 없다)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PosePlan:
    """`geometry_msgs/PoseWithCovarianceStamped` 한 건."""

    stamp_ns: int
    frame_id: str
    position: tuple  # (x, y, z) — **body FLU**
    orientation: tuple  # (x, y, z, w) — 항상 단위(파일 docstring 참조)
    covariance: tuple  # 36개 row-major 6x6


@dataclass
class StatusPlan:
    """`diagnostic_msgs/DiagnosticStatus` 한 건(배열 원소)."""

    level: int
    name: str
    message: str
    hardware_id: str
    values: tuple  # ((key, value_str), ...)


@dataclass
class LandingTargetPlan:
    """`mavros_msgs/LandingTarget` 한 건. **`frame`은 정수 리터럴 경로로만 채워진다.**"""

    stamp_ns: int
    frame_id: str
    target_num: int
    frame: int
    angle: tuple  # (angle_x, angle_y) rad
    distance: float
    size: tuple  # (size_x, size_y) rad
    position: tuple  # (x, y, z) — 🔴 **FLU**. 플러그인이 FRD로 바꾼다
    orientation: tuple
    type: int


@dataclass
class ShimOutput:
    """한 번의 발행 묶음. `statuses`는 **하나의 `DiagnosticArray`** 로 나간다(같은 stamp)."""

    stamp_ns: int
    statuses: tuple = ()
    pose: Optional[PosePlan] = None
    landing_target: Optional[LandingTargetPlan] = None


# ──────────────────────────────────────────────────────────────────────────────
# KeyValue 직렬화
# ──────────────────────────────────────────────────────────────────────────────


def kv_value(value: Any) -> str:
    """`diagnostic_msgs/KeyValue.value`는 string 하나뿐이라 전부 문자열로 눌러야 한다.

    문자열은 그대로, 나머지는 `json.dumps` — 그래야 소비자가 `json.loads`로 **정확한 타입을
    되찾는다**(`true`/`null`/`1.5`/`[1,2,3]`). `str(True)`가 `"True"`가 되어 JSON도 아니고
    파이썬 밖에선 파싱 불가한 문자열이 되는 것을 막는다.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _kvs(pairs) -> tuple:
    return tuple((str(k), kv_value(v)) for k, v in pairs)


# ──────────────────────────────────────────────────────────────────────────────
# 레코드 검증
# ──────────────────────────────────────────────────────────────────────────────


def parse_line(line: str) -> dict:
    """JSON Lines 한 줄 → dict. 빈 줄/비-dict/파싱실패는 전부 `ValueError`."""
    s = line.strip()
    if not s:
        raise ValueError("빈 줄")
    try:
        record = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 실패: {e}") from e
    if not isinstance(record, dict):
        raise ValueError(f"dict가 아님 — {type(record).__name__}")
    return record


def validate_record(record: dict) -> tuple:
    """`(ok, 사유)`. 🔴 **`schema_version` 불일치는 반드시 거절한다.**

    Phase 1 인수인계 계약 1번이다. 모르는 버전을 "대충 읽어보는" 것이 가장 위험하다 — 필드
    의미가 바뀌었을 수 있는데 키 이름만 같으면 조용히 통과해 **틀린 좌표로 유도**하게 된다.
    """
    rtype = record.get("type")
    if rtype == RECORD_TYPE_TARGET:
        required = REQUIRED_TARGET_KEYS
    elif rtype == RECORD_TYPE_STATE_HINT:
        required = REQUIRED_STATE_HINT_KEYS
    else:
        return False, f"unknown_record_type({rtype!r})"
    if record.get("schema_version") != SCHEMA_VERSION:
        return False, f"schema_version_mismatch({record.get('schema_version')!r})"
    missing = [k for k in required if k not in record]
    if missing:
        return False, f"missing_keys({','.join(missing)})"
    return True, "ok"


# ──────────────────────────────────────────────────────────────────────────────
# 변환
# ──────────────────────────────────────────────────────────────────────────────


def _covariance(uncertainty, unknown: float) -> tuple:
    """6x6 row-major 36개.

    `uncertainty`는 **이번 Phase 내내 None**이다(`core/target.py` 확정 전제). None이면
    **위치·회전 대각 6개 전부** `unknown`으로 채운다 — 0을 채우면 "분산 0 = 완벽히 정확"이라는
    뜻이 되어 `robot_localization` 같은 기성 필터가 이 추정치를 절대 신뢰로 받아들인다.

    회전 블록(대각 21/28/35)은 **값이 들어와도 항상 `unknown`** 이다: Pose의 orientation이
    단위 쿼터니언(=모름)이므로 그 공분산도 모름이어야 일관된다(파일 docstring 참조).
    """
    cov = [0.0] * 36
    diag_pos = (0, 7, 14)
    diag_rot = (21, 28, 35)
    if uncertainty is None:
        for i in diag_pos:
            cov[i] = float(unknown)
    else:
        flat = _flatten6x6(uncertainty)
        if flat is None:
            for i in diag_pos:
                cov[i] = float(unknown)
        else:
            cov = flat
    for i in diag_rot:
        cov[i] = float(unknown)
    return tuple(cov)


def _flatten6x6(uncertainty) -> Optional[list]:
    """`uncertainty`(JSON 리스트)가 6x6 또는 36개면 평탄화, 아니면 None(→ 모름 처리)."""
    try:
        if len(uncertainty) == 36 and not isinstance(uncertainty[0], (list, tuple)):
            return [float(v) for v in uncertainty]
        if len(uncertainty) == 6:
            flat = []
            for row in uncertainty:
                if len(row) != 6:
                    return None
                flat.extend(float(v) for v in row)
            return flat
    except (TypeError, ValueError, IndexError):
        return None
    return None


def _meta_kv_pairs(record: dict) -> list:
    """`meta`에서 소비자가 실제로 쓸 것만 뽑는다 + 🔴 0.105m 어긋남을 명시적 숫자로 수출.

    파일 docstring "초록구역 0.105m" 절 참조 — **z는 고치지 않는다.**
    """
    meta = record.get("meta") or {}
    pairs = []
    for key in (
        "plane_reference", "platform_height_m", "landing_point_source",
        "frame_size_px", "calib_image_size_px",
    ):
        if key in meta:
            pairs.append((key, meta[key]))
    height = meta.get("platform_height_m")
    if height is not None:
        # 지면기준 라이다 AGL − 비전 z. 라이다 빔이 **매트 밖 지면**에 떨어질 때만 이만큼
        # 어긋난다(빔이 매트 위면 둘이 일치). 어느 쪽인지 코드가 모르므로 보정하지 않고
        # 숫자만 넘긴다.
        pairs.append(("ground_agl_minus_vision_z_m", float(height)))
    return pairs


def _landing_target_type(target_type: Optional[str]) -> int:
    """`target_type` 문자열 → `LANDING_TARGET_TYPE`(손실 압축, §3.1).

    `"aruco_23"` 같은 피듀셜만 `VISION_FIDUCIAL`, 나머지(초록매트/흰박스)는 `VISION_OTHER`.
    ArUco ID를 `target_num`에 밀어넣는 안은 채택하지 않았다 — MAVLink가 "타겟 ID"로 쓰는
    필드를 사설 코드북으로 전용하는 것이라(§3.1) 원문은 KeyValue `target_type`으로 보낸다.
    """
    if isinstance(target_type, str) and target_type.startswith("aruco"):
        return LANDING_TARGET_TYPE_VISION_FIDUCIAL
    return LANDING_TARGET_TYPE_VISION_OTHER


def _angles_from_flu(position_flu) -> tuple:
    """FLU 상대벡터 → `LANDING_TARGET.angle_x/angle_y`(나디르 기준 각오프셋, rad).

    MAVLink 정의는 body **FRD** 기준이라 FLU에서 부호를 맞춘다:
    `x_frd = x_flu`, `y_frd = −y_flu`, `z_frd = −z_flu`(아래 양수).
    타겟이 아래에 있으면 `z_flu < 0` → `z_frd > 0`.

    `angle_x = atan2(x_frd, z_frd)`(전방 오프셋), `angle_y = atan2(y_frd, z_frd)`(우측 오프셋).
    **지어낸 값이 아니라 같은 좌표의 결정론적 투영**이다 — `frame=12`에선 `position_valid=true`라
    PX4가 position을 쓰므로 angle은 보조 정보다. `z_frd <= 0`(타겟이 기체보다 위)면 나디르
    각도가 정의되지 않으므로 0을 낸다.
    """
    x_flu, y_flu, z_flu = (float(v) for v in position_flu)
    x_frd, y_frd, z_frd = x_flu, -y_flu, -z_flu
    if z_frd <= 0.0:
        return (0.0, 0.0)
    return (math.atan2(x_frd, z_frd), math.atan2(y_frd, z_frd))


def build_target_plans(record: dict, cfg: ShimConfig, *, recv_monotonic_ns: int) -> ShimOutput:
    """`type="target"` 레코드 → pose(+status, +선택 LandingTarget) 계획.

    `valid=false`면 **pose를 만들지 않는다**(실을 좌표가 없다). 대신 status가 `WARN`+사유로
    나가 "안 보임"이 소비자에게 명시적으로 전달된다 — 파일 docstring의 3분법 표 참조.
    """
    stamp_ns = int(record["stamp_wall_ns"])
    age_s = (int(recv_monotonic_ns) - int(record["stamp_monotonic_ns"])) / 1e9
    valid = bool(record.get("valid"))
    stale = age_s > cfg.stale_warn_s

    pairs = [
        ("schema_version", record.get("schema_version")),
        ("seq", record.get("seq")),
        ("frame_id", record.get("frame_id")),
        ("valid", valid),
        ("reason", record.get("reason")),
        # 🔴 손실 압축 대상 — LandingTarget에 자리가 없는 것들(§3.1)
        ("confidence", record.get("confidence")),
        ("target_type", record.get("target_type")),
        ("calib_accuracy", record.get("calib_accuracy")),
        ("not_for_closed_loop_30cm", record.get("not_for_closed_loop_30cm")),
        ("calib_id", record.get("calib_id")),
        ("closed_loop_floor_agl_m", record.get("closed_loop_floor_agl_m")),
        ("mount_yaw_psi_m_rad", record.get("mount_yaw_psi_m_rad")),
        ("mount_yaw_measured", record.get("mount_yaw_measured")),
        # 프레임을 이름에 박은 3종 전부 보존 — 한 번 버린 provenance는 못 되살린다
        ("position_cam", record.get("position_cam")),
        ("position_flu", record.get("position_flu")),
        ("position_frd", record.get("position_frd")),
        # 🔴 Pose에 못 싣는 카메라 광학 프레임 자세(파일 docstring 참조)
        ("orientation_frame", record.get("orientation_frame")),
        ("orientation_cam_optical_xyzw", record.get("orientation")),
        ("uncertainty_source", "none" if record.get("uncertainty") is None else "wire"),
        # 시계 — stale은 monotonic으로 잰다
        ("timestamp", record.get("timestamp")),
        ("stamp_monotonic_ns", record.get("stamp_monotonic_ns")),
        ("stamp_wall_ns", record.get("stamp_wall_ns")),
        ("clock_offset_ns", record.get("clock_offset_ns")),
        ("recv_monotonic_ns", int(recv_monotonic_ns)),
        ("age_s", age_s),
        ("stale_warn_s", cfg.stale_warn_s),
    ]
    pairs.extend(_meta_kv_pairs(record))

    if not valid:
        level = LEVEL_WARN
        message = f"target_not_seen({record.get('reason')})"
    elif stale:
        level = LEVEL_WARN
        message = f"stale({age_s:.3f}s>{cfg.stale_warn_s:.3f}s)"
    else:
        level = LEVEL_OK
        message = "ok"

    status = StatusPlan(
        level=level,
        name=STATUS_NAME_TARGET,
        message=message,
        hardware_id=cfg.hardware_id,
        values=_kvs(pairs),
    )

    pose = None
    landing_target = None
    position_flu = record.get("position_flu")
    if valid and position_flu is not None:
        pose = PosePlan(
            stamp_ns=stamp_ns,
            frame_id=cfg.pose_frame_id,
            # 🔴 유도 입력은 position_flu다. position_frd를 쓰면 y/z 부호가 뒤집힌다.
            position=tuple(float(v) for v in position_flu),
            # 카메라 광학 프레임 자세를 body 프레임 Pose에 실지 않는다(파일 docstring).
            orientation=(0.0, 0.0, 0.0, 1.0),
            covariance=_covariance(record.get("uncertainty"), cfg.unknown_covariance),
        )
        if cfg.publish_landing_target:
            p = pose.position
            landing_target = LandingTargetPlan(
                stamp_ns=stamp_ns,
                frame_id=cfg.pose_frame_id,
                target_num=0,
                # 🔴 정수 리터럴. msg 상수(`LandingTarget.LOCAL_NED`=2)를 쓰면 MAVLink가
                # MISSION으로 읽어 position_valid=false가 된다 — 파일 docstring 참조.
                frame=MAV_FRAME_BODY_FRD,
                angle=_angles_from_flu(p),
                distance=math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2),
                # 타겟 각크기는 와이어에 실리지 않는다(물리 크기가 레코드에 없다).
                # 지어내지 않고 0으로 둔다 — frame=12는 position_valid=true라 PX4가
                # position을 쓰므로 size는 보조 정보다.
                size=(0.0, 0.0),
                position=p,  # 🔴 FLU. 플러그인이 baselink→aircraft로 바꾼다
                orientation=(0.0, 0.0, 0.0, 1.0),
                type=_landing_target_type(record.get("target_type")),
            )

    return ShimOutput(
        stamp_ns=stamp_ns, statuses=(status,), pose=pose, landing_target=landing_target
    )


def build_state_status(record: dict, cfg: ShimConfig) -> StatusPlan:
    """`type="state_hint"` 레코드 → state `DiagnosticStatus`.

    🔴 **`command_hint`는 advisory다** — 소비하면 안 된다. 거부권은 `state`뿐이다
    (`core/wire.py` "command는 명령이 아니다"). 와이어가 이름·플래그·타입명 세 겹으로 박아둔
    사실을 shim도 그대로 옮긴다: 키 이름은 `command_hint`이고 `command_is_advisory`가 함께
    나간다. 맨 이름 `command` 키는 **만들지 않는다**(회귀테스트로 고정).

    `level`은 `state`가 아니라 **`HOLD`/`ABORT_ASCEND` 여부**로 정한다 — §6.3의 거부권
    권고 그대로다("vision은 하강의 필요조건이지 충분조건이 아니다"). `PRECISION_SERVO`를
    `OK`로 내보내도 그것이 하강 허가가 되지는 않는다.
    """
    state = record.get("state")
    veto = state in ("HOLD", "ABORT_ASCEND")
    pairs = [
        ("schema_version", record.get("schema_version")),
        ("seq", record.get("seq")),
        ("frame_id", record.get("frame_id")),
        ("state", state),
        ("command_hint", record.get("command_hint")),
        ("command_is_advisory", record.get("command_is_advisory")),
        ("reason", record.get("reason")),
        ("blind_duration_s", record.get("blind_duration_s")),
        ("scale_source", record.get("scale_source")),
        ("stamp_monotonic_ns", record.get("stamp_monotonic_ns")),
        ("stamp_wall_ns", record.get("stamp_wall_ns")),
    ]
    return StatusPlan(
        level=LEVEL_WARN if veto else LEVEL_OK,
        name=STATUS_NAME_STATE,
        message=f"state={state}" + (" (veto)" if veto else ""),
        hardware_id=cfg.hardware_id,
        values=_kvs(pairs),
    )


def build_link_status(
    cfg: ShimConfig, *, level: int, event: str, message: str, extra=()
) -> StatusPlan:
    """링크 상태(EOF/재접속/거절 레코드) status. **pose는 절대 동반하지 않는다.**"""
    pairs = [("event", event)]
    pairs.extend(extra)
    return StatusPlan(
        level=level,
        name=STATUS_NAME_LINK,
        message=message,
        hardware_id=cfg.hardware_id,
        values=_kvs(pairs),
    )


def _missing_state_status(cfg: ShimConfig, frame_id) -> StatusPlan:
    return StatusPlan(
        level=LEVEL_WARN,
        name=STATUS_NAME_STATE,
        message="state_hint_missing",
        hardware_id=cfg.hardware_id,
        values=_kvs([("frame_id", frame_id), ("state", None)]),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 라우터 — target/state_hint 짝짓기 + 거절/사망 처리
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _Pending:
    output: ShimOutput
    frame_id: Any


class ShimRouter:
    """소켓에서 온 줄을 받아 `ShimOutput` 목록을 낸다. **상태는 pending target 하나뿐이다.**

    `push()`는 절대 예외를 던지지 않는다 — 깨진 줄 하나가 shim 전체를 죽이면 유도가 끊긴다.
    거절 사유는 삼키지 않고 `vision/link` status(`level=ERROR`)로 **반드시 내보낸다**(§5.4의
    "사유를 로그에 남긴다"를 토픽 레벨로 옮긴 것).
    """

    def __init__(self, cfg: Optional[ShimConfig] = None):
        self.cfg = cfg or ShimConfig()
        self._pending: Optional[_Pending] = None
        # 관측성 카운터(§7.4)
        self.targets = 0
        self.state_hints = 0
        self.rejected = 0
        self.unpaired_targets = 0
        self.orphan_state_hints = 0

    # ---------- 입력 ----------

    def push(self, line: str, *, recv_monotonic_ns: int, recv_wall_ns: int) -> list:
        outputs: list = []
        try:
            record = parse_line(line)
        except ValueError as e:
            self.rejected += 1
            outputs.append(self._reject(recv_wall_ns, "malformed_line", str(e)))
            return outputs

        ok, reason = validate_record(record)
        if not ok:
            self.rejected += 1
            outputs.append(self._reject(recv_wall_ns, "record_rejected", reason))
            return outputs

        rtype = record["type"]
        if rtype == RECORD_TYPE_TARGET:
            self.targets += 1
            # 앞의 target이 짝을 못 만났다 — 혼자 내보내고 새 것을 들고 간다.
            outputs.extend(self._flush_pending())
            out = build_target_plans(record, self.cfg, recv_monotonic_ns=recv_monotonic_ns)
            self._pending = _Pending(output=out, frame_id=record.get("frame_id"))
            return outputs

        # state_hint
        self.state_hints += 1
        status = build_state_status(record, self.cfg)
        pending = self._pending
        if pending is not None and pending.frame_id == record.get("frame_id"):
            self._pending = None
            outputs.append(
                ShimOutput(
                    stamp_ns=pending.output.stamp_ns,
                    statuses=pending.output.statuses + (status,),
                    pose=pending.output.pose,
                    landing_target=pending.output.landing_target,
                )
            )
            return outputs
        # 짝 없는 state_hint — 들고 있던 target(있으면)을 먼저 내보내고 state만 단독 발행.
        outputs.extend(self._flush_pending())
        self.orphan_state_hints += 1
        outputs.append(
            ShimOutput(stamp_ns=int(record["stamp_wall_ns"]), statuses=(status,))
        )
        return outputs

    # ---------- 연결 수명주기 ----------

    def on_producer_gone(self, *, recv_wall_ns: int, reason: str) -> list:
        """🔴 EOF/연결 끊김 = **생산자 사망**. pose 토픽은 침묵하고 status만 ERROR로 알린다.

        §5.4의 "침묵을 유일한 권위로 삼고, 명시적 무효는 지연 단축용 최적화로만 얹는다"를
        토픽 배치로 옮긴 것 — pose 침묵이 권위, ERROR status가 지연 단축이다. 들고 있던
        pending target은 **버린다**(생산자가 죽은 뒤에 그 좌표를 새로 발행하는 것은
        "죽은 데이터로 유도"다).
        """
        self._pending = None
        return [
            ShimOutput(
                stamp_ns=int(recv_wall_ns),
                statuses=(
                    build_link_status(
                        self.cfg,
                        level=LEVEL_ERROR,
                        event="producer_eof",
                        message=f"producer_gone({reason})",
                        extra=[("reason", reason)],
                    ),
                ),
            )
        ]

    def on_connected(self, *, recv_wall_ns: int, endpoint: str) -> list:
        return [
            ShimOutput(
                stamp_ns=int(recv_wall_ns),
                statuses=(
                    build_link_status(
                        self.cfg,
                        level=LEVEL_OK,
                        event="connected",
                        message=f"connected({endpoint})",
                        extra=[("endpoint", endpoint)],
                    ),
                ),
            )
        ]

    def on_disconnected_heartbeat(self, *, recv_wall_ns: int, endpoint: str) -> list:
        """재접속 대기 중 주기 하트비트.

        **"생산자 사망"과 "shim 사망"을 가르는 유일한 신호다** — 둘 다 pose 침묵이지만,
        전자는 status에 ERROR가 계속 나오고 후자는 status도 멈춘다.
        """
        return [
            ShimOutput(
                stamp_ns=int(recv_wall_ns),
                statuses=(
                    build_link_status(
                        self.cfg,
                        level=LEVEL_ERROR,
                        event="disconnected",
                        message=f"waiting_for_producer({endpoint})",
                        extra=[("endpoint", endpoint)],
                    ),
                ),
            )
        ]

    # ---------- 내부 ----------

    def _flush_pending(self) -> list:
        pending, self._pending = self._pending, None
        if pending is None:
            return []
        self.unpaired_targets += 1
        return [
            ShimOutput(
                stamp_ns=pending.output.stamp_ns,
                statuses=pending.output.statuses
                + (_missing_state_status(self.cfg, pending.frame_id),),
                pose=pending.output.pose,
                landing_target=pending.output.landing_target,
            )
        ]

    def _reject(self, recv_wall_ns: int, event: str, reason: str) -> ShimOutput:
        return ShimOutput(
            stamp_ns=int(recv_wall_ns),
            statuses=(
                build_link_status(
                    self.cfg,
                    level=LEVEL_ERROR,
                    event=event,
                    message=reason,
                    extra=[("reason", reason)],
                ),
            ),
        )
