"""
vision → fc 인터페이스 **와이어 포맷**(JSON Lines) + 페일세이프 계약.

`docs/vision_fc_interface.md` §8 권고 R4의 vision 쪽 절반(작업 V2 + §5 계약)이다. 구조는:

```
[호스트]  vision core (picam-venv, Py3.12)     ← 이 파일 + utils/target_sink.py  ← Phase 1(여기)
            │  localhost TCP 소켓 / JSON Lines
            ▼
[컨테이너] shim node (Humble, Py3.10)                                            ← Phase 2
            │  ROS2 topic
            ▼
          offboard_node 정밀착륙 서브상태                                        ← FC 도메인
```

**이 파일은 transport-agnostic이다** — 소켓도, 파일도, 클록도 모른다. `vision/CLAUDE.md` import
규칙("core/ ← numpy, opencv만 허용")과 `core/target.py`/`core/state_machine.py`가 세운 패턴을
그대로 따른다: 파일 I/O·로깅 없음, **wall-clock/난수 없음**(타임스탬프는 전부 호출자가 인자로
넣어준다 → 같은 입력이면 항상 같은 바이트가 나온다, §7.5 결정론. 실제 클록 샘플링은 어댑터인
`vision/utils/target_sink.py::sample_clocks()`가 한다).

## 왜 JSON Lines인가 (와이어 포맷 결정 근거)

1. **이 저장소가 이미 JSONL 관측성 스택을 쓴다** — `utils/blackbox.py`가 프레임별 JSONL을 쓰고
   `tools/jsonl_view.py`가 그걸 읽어 플롯한다. 새 직렬화 관례를 발명하지 않는다.
2. **Phase 2 shim이 vision 코드를 import할 수 없다.** 컨테이너는 Python 3.10 + Humble이고
   호스트는 Python 3.12다(§2.1). 두 도메인 사이에 **빌드 의존을 만들지 않는 것**이 R4의 핵심
   가치인데, protobuf/msgpack/FlatBuffers는 전부 스키마 컴파일 산출물이나 서드파티 패키지를
   양쪽에 깔아야 한다. JSONL은 **양쪽 표준 라이브러리만으로** 끝난다(`json` + `readline`).
3. **자기 프레이밍(self-framing)이 공짜다.** `json.dumps`는 문자열 안의 개행을 반드시 `\\n`으로
   이스케이프하므로 레코드 하나가 정확히 한 줄이다 → 길이 프리픽스 프로토콜이 필요 없고,
   TCP 스트림에서 `readline()`만으로 재동기화된다(`tests/test_wire.py`가 실제 개행/유니코드가
   섞인 문자열로 이 성질을 회귀로 박아 뒀다).
4. **눈으로 디버깅된다** — `nc 127.0.0.1 8091`로 실기체에서 바로 본다(§8 권고 5번).
5. 대역폭이 문제가 아니다 — 레코드 1개가 ~600바이트, 10Hz면 6KB/s다. 바이너리 포맷으로 얻을
   게 없다.

## 레코드 타입 2종

| `type` | 내용 | 소비자 취급 |
|---|---|---|
| `"target"` | `TargetEstimate` 전량 + 좌표변환 결과 + provenance + 페일세이프 필드 | 폐루프 유도 입력 |
| `"state_hint"` | `Decision`의 `state`/`command`/`reason` | `state`=거부권, `command_hint`=**참고용** |

## 🔴 `command`는 명령이 아니다 — 형식에 드러나게 실었다

`core/state_machine.py`가 두 번 못박았다: "`Decision.command`는 **문자열 힌트일 뿐이다**".
`docs/vision_fc_interface.md` §6.3의 권고도 "거부권(veto)으로만 싣는다"이다:

```
✅ vision의 상태는 FC 하강의 [필요조건]이다.  (withhold 가능)
❌ vision의 상태는 FC 하강의 [충분조건]이 아니다. (compel 불가)
```

그래서 이 포맷은 **세 겹으로** 그 사실을 형식에 박아 뒀다 — ① 레코드 타입 이름이 `"state_hint"`,
② 키 이름이 `"command"`가 아니라 **`"command_hint"`**, ③ 상수 필드 `"command_is_advisory": true`.
`nc`로 raw 덤프만 봐도 이게 명령이 아니라는 걸 놓칠 수 없게 하는 것이 목적이다.
(D3 — `state`를 실제로 거부권으로 쓸지는 **사용자 미결 사항**이다. 이 포맷은 양쪽 다 지원한다:
소비자가 `state`를 무시해도 `target` 레코드만으로 동작한다.)

## 🔴 필드 이름에 프레임을 박은 이유 — `position` 대신 `position_cam`/`_flu`/`_frd`

이 저장소는 **접미사가 프레임을 거짓으로 암시해 사고가 난 이력**이 있다: `fc_bridge`의
`pos_ned`는 `[N, E, h_up]`(위 양수)인데 `vel_ned`는 `[vN, vE, vD]`(아래 양수)로, **같은 `_ned`
접미사가 위치와 속도에서 반대 부호 규약**을 쓴다(§4.2). vision 쪽에서도 이미 같은 종류의
정정이 있었다(`center_error_px` → `center_error_norm`, `vision/CLAUDE.md`).

3개 프레임 값이 한 레코드에 같이 실리는데 그중 하나만 이름 없는 `position`이면 Phase 2 shim이
어느 것을 집어야 하는지 **읽어서는 알 수 없다.** 그래서 위치는 전부 프레임을 이름에 박는다.
`TargetEstimate.position`(카메라 광학 프레임 원본)은 `position_cam`으로 **무손실 보존**되고
왕복 복원의 출처도 이 필드다. `orientation`은 프레임이 하나뿐이라 이름을 유지하되
`orientation_frame`으로 명시한다.

## 시계(clock) 계약 — 두 클록을 둘 다 싣는다 (§4.6)

**측정된 사실:** vision은 `time.time()`(벽시계)을 쓰고(`utils/frame_source.py`) fc_ros는
`time.monotonic()`을 쓴다(`vehicle_state_bridge.py`). 서로 다른 클록이라 지금은 정렬 자체가
불가능하다.

**결론: 한쪽을 고르지 않고 둘 다 싣는다.** 근거와 각 필드의 용도:

| 필드 | 클록 | 용도 |
|---|---|---|
| `stamp_monotonic_ns` | `time.monotonic_ns()` | **stale 판정용(권장).** 리눅스에서 `time.monotonic`은 `CLOCK_MONOTONIC`이고 이건 **커널 전역**이다 — 컨테이너는 호스트 커널을 공유하므로(§2.1) fc 쪽 `time.monotonic()`과 **같은 기준의 값**이다. 벽시계와 달리 NTP 점프·시간 되돌림에 영향받지 않는다 |
| `stamp_wall_ns` | `time.time_ns()` | **로그 상관용.** JSONL 블랙박스가 `time.time()`을 쓰므로 사후분석에서 비행로그와 맞추려면 이 값이 필요하다 |
| `clock_offset_ns` | `stamp_wall_ns - stamp_monotonic_ns` (동일 시점 샘플) | 소비자가 두 클록 사이를 **환산**할 수 있게 하는 값(§4.6 계약 1번이 요구한 것) |
| `timestamp` | 벽시계 초 | `TargetEstimate.timestamp` **원본 그대로**(무손실 왕복용). 프레임 핸드오프 시각이지 노출 시각이 아니다 — 아래 미해결 참조 |

⚠️ **미검증 2건(추측으로 메우지 않았다):**
- `CLOCK_MONOTONIC`이 이 컨테이너에서 호스트와 정말 같은 기준인지 **실측하지 않았다**(이번
  Phase는 전부 로컬 완결, RPi 접속 금지). Docker는 기본적으로 time namespace를 쓰지 않으므로
  같을 것으로 보지만, **Phase 2에서 양쪽 `time.monotonic()`을 1회 찍어 대조하는 것이 첫 검증
  항목**이다. 그래서 `clock_offset_ns`를 같이 실어 두 클록 중 어느 쪽으로도 환산 가능하게 했다.
- `timestamp`는 **노출 시각이 아니라 프레임 핸드오프 시각**이다. §4.6 계약 1번은 picamera2
  `SensorTimestamp`(CLOCK_BOOTTIME ns) 기반으로 통일하라고 했으나 그건 별도 작업(V4,
  `frame_source.py` 수정)이고 **이번 Phase 범위 밖이다.** 지금 실리는 값이 노출 시각이 아님을
  여기 명시해 둔다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from . import frames
from .state_machine import Decision, LandingState
from .target import TargetEstimate

# 스키마가 바뀌면 올린다. 소비자(Phase 2 shim)는 모르는 버전을 받으면 거절해야 한다
# (§7.2가 `schema_version` 필드를 요구한 이유).
SCHEMA_VERSION = 1

RECORD_TYPE_TARGET = "target"
RECORD_TYPE_STATE_HINT = "state_hint"

# `position`의 좌표계를 이름에 박은 대신, 어느 필드가 원본인지 한 곳에 적어둔다.
ORIENTATION_FRAME_CAM = "cam_optical"

# Phase 2 shim이 "이 키들이 다 있는가"만 확인하면 되도록 계약을 상수로 노출한다
# (`tests/test_wire.py`가 실제 레코드와 이 목록을 대조하는 회귀테스트를 건다).
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
# 페일세이프 계약 (`docs/vision_fc_interface.md` §5)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class FailsafeContract:
    """§5.6 계약 파라미터. **전부 이름 있는 필드 — 매직넘버 금지**(§7.3).

    소비자 쪽 파라미터(`stale_timeout_s`, `hold_before_reascend_s`)는 **일부러 여기 없다** —
    그건 FC 도메인이 자기 제어틱에서 재는 값이고, vision이 정해 보낼 값이 아니다(§5.4 "침묵을
    유일한 권위로 삼는다": 소비자는 vision의 협조 없이 스스로 나이를 잰다).
    """

    # 🔴 지금은 **no-op다.** 아래 `gate_confidence()` docstring의 측정 사실 참조.
    # 기본값 0.0(비활성)은 §5.3 계약 3번 그대로 — 지금 0.9 같은 값을 넣으면 "동작하는 것처럼
    # 보이지만 아무것도 거르지 않는" 가짜 안전장치가 된다.
    min_confidence: float = 0.0

    # `not_for_closed_loop_30cm=True`일 때 비전만 믿고 내려가도 되는 **바닥 고도**(m).
    # 권고 초기값 3.0 = 상태머신 `LandingSMConfig.terminal_agl_m` 기본값과 정렬(§5.6).
    unverified_calib_floor_agl_m: float = 3.0
    # 실측 캘리브레이션이 끝나 플래그가 False가 되면 바닥이 사라진다(TERMINAL까지 폐루프 커밋).
    verified_calib_floor_agl_m: float = 0.0

    # 🔴 미측정(§7 U3) — `core/frames.py` 참조. 값을 지어내지 않고 공칭 0.0 + "미측정" 플래그.
    mount_yaw_psi_m_rad: float = frames.MOUNT_YAW_PSI_M_RAD_DEFAULT
    mount_yaw_measured: bool = frames.MOUNT_YAW_PSI_M_MEASURED


def gate_confidence(confidence: float, contract: FailsafeContract) -> tuple[bool, str]:
    """`min_confidence` 하한 게이트. `(통과여부, 사유문자열)`.

    🔴 **이 게이트는 지금 ArUco 경로에서 아무것도 거르지 않는다(no-op).** 측정된 사실
    (`docs/vision_fc_interface.md` §5.3):

    - `Detection.confidence` 기본값 = **1.0** (`core/state.py`)
    - `modules/aruco.py`와 `modules/distress_box.py`는 `confidence`를 **한 번도 설정하지 않는다**
    - `main.py`의 ArUco 경로는 `confidence=det.confidence`를 그대로 넘긴다
    - **→ ArUco `TargetEstimate.confidence`는 항상 정확히 1.0이다.**

    그런데도 게이트를 만들어 두는 이유: 값이 실제로 들어가는 다른 검출기가 붙었을 때 배선을
    새로 하지 않아도 되게 하고, 계약이 코드로 존재해야 나중에 켜는 게 파라미터 한 줄이 되기
    때문이다. 지금 **기본값이 0.0(비활성)** 인 것도 그래서다.

    ⚠️ 그리고 켤 때도 **단일 전역 임계값을 쓰면 안 된다**(§5.3 계약 1번) — `confidence`는
    검출기 간 비교 불가다: `detector.py`는 `solidity × min(1, area/min_area)`(형상 휴리스틱),
    `vertiport_field.py`는 `min(1, circularity)`로 **의미가 아예 다르다**. 둘 다 확률이 아니다.

    폐루프 게이팅의 진짜 담당은 `confidence`가 아니라 **상태머신의 커밋 게이트**
    (`lock_confirm_frames` 연속 `fine_locked` + `max_candidates_for_lock` 모호 거절)다 —
    이건 이미 구조적으로 강제돼 있고 테스트도 있다(§5.3 계약 2번).
    """
    if contract.min_confidence <= 0.0:
        return True, "confidence_gate_disabled"
    if float(confidence) < contract.min_confidence:
        return False, f"confidence_below_min({confidence:.3f}<{contract.min_confidence:.3f})"
    return True, "ok"


def closed_loop_floor_agl_m(
    not_for_closed_loop_30cm: bool, contract: FailsafeContract
) -> float:
    """`not_for_closed_loop_30cm` → **비전만 믿고 내려가도 되는 바닥 고도(m)**.

    🔴 **이 플래그를 "폐루프 금지"로 읽으면 통합이 착수 즉시 죽는다.** 실측 캘리브레이션이
    보류됐고(메모리 `project_vision_calibration_deferred`), `nominal.yaml`이
    `not_for_closed_loop_30cm: true`를 담고 있으며, `TargetEstimate` 기본값도 `True`다 —
    **지금 100% 확률로 True다**(§5.5). 그렇다고 무시하면 provenance echo를 만든 이유가 사라진다.

    그래서 계약은 **"서보 금지"가 아니라 "최종 커밋 금지"** 로 해석한다(§5.5 표):

    | 플래그 | 소비자가 해도 되는 것 | 하면 안 되는 것 |
    |---|---|---|
    | `True`(현재) | ACQUIRE/CENTER/SERVO로 **바닥 고도까지** 하강·정렬 | 그 아래로 **비전만 믿고** 하강. 바닥에 닿으면 정렬을 유지한 채 **AUTO.LAND/GPS+라이다에 인계** |
    | `False`(실측 캘리브 후) | TERMINAL까지 비전 폐루프로 커밋 | — |

    이 설계는 **지금 당장 가치가 있고**(3m까지 비전 정렬 > GPS만 쓰기), **실측 캘리브레이션이
    끝나면 플래그 하나로 자동 해금**된다 — 코드 변경이 필요 없다. 그래서 이 함수는 "금지/허용"
    불리언이 아니라 **고도(m)** 를 돌려준다: 소비자는 `agl <= floor`에서 인계하면 되고,
    플래그가 False가 되는 순간 floor가 0.0이 되어 같은 코드가 그대로 TERMINAL까지 간다.
    """
    return (
        contract.unverified_calib_floor_agl_m
        if not_for_closed_loop_30cm
        else contract.verified_calib_floor_agl_m
    )


# ──────────────────────────────────────────────────────────────────────────────
# 레코드 조립
# ──────────────────────────────────────────────────────────────────────────────


def _clock_fields(stamp_monotonic_ns: int, stamp_wall_ns: int) -> dict:
    return {
        "stamp_monotonic_ns": int(stamp_monotonic_ns),
        "stamp_wall_ns": int(stamp_wall_ns),
        # 같은 시점에 뽑은 두 클록의 차 — 소비자가 자기 클록으로 환산하는 데 쓴다(§4.6).
        "clock_offset_ns": int(stamp_wall_ns) - int(stamp_monotonic_ns),
    }


def _uncertainty_to_json(uncertainty) -> Optional[list]:
    """`TargetEstimate.uncertainty`는 `Optional[np.ndarray]`이고 **이번 Phase는 항상 None**
    (`core/target.py` 확정 전제 — 자리만 만들어둠). None이 아닌 값이 들어와도 JSON으로 나가게
    지금 처리해 둔다: 실측 캘리브레이션 이후 공분산이 채워질 때 이 파일을 다시 안 열어도 된다."""
    if uncertainty is None:
        return None
    return np.asarray(uncertainty, dtype=np.float64).tolist()


def build_target_record(
    estimate: TargetEstimate,
    *,
    seq: int,
    stamp_monotonic_ns: int,
    stamp_wall_ns: int,
    contract: Optional[FailsafeContract] = None,
) -> dict:
    """`TargetEstimate` → 와이어 레코드 dict(아직 JSON 문자열 아님).

    `TargetEstimate`의 **모든 필드를 무손실로** 싣는다. `mavros_msgs/LandingTarget`에는 이 중
    절반을 실을 자리가 없다는 게 §3.1에서 확인됐지만, **손실 압축은 Phase 2 shim의 책임**이고
    소켓 단계에서는 전부 보존한다 — 한 번 버린 provenance는 되살릴 수 없다.
    """
    cfg = contract or FailsafeContract()
    psi = cfg.mount_yaw_psi_m_rad

    p_cam = np.asarray(estimate.position, dtype=np.float64)
    p_flu = frames.cam_to_flu(p_cam, psi)
    p_frd = frames.cam_to_frd(p_cam, psi)

    ok, gate_reason = gate_confidence(estimate.confidence, cfg)

    record = {
        "schema_version": SCHEMA_VERSION,
        "type": RECORD_TYPE_TARGET,
        "seq": int(seq),
        "frame_id": int(estimate.frame_id),
        # §5.4 계약 2번 — 검출을 잃어도 발행을 멈추지 않고 valid=false를 보낸다. 그래야
        # 소비자가 "노드가 죽음"(=침묵/EOF)과 "안 보임"(=valid=false)을 구분할 수 있다.
        "valid": bool(ok),
        "reason": gate_reason,
        # 원본 그대로(벽시계 초). ⚠️ 노출 시각이 아니라 프레임 핸드오프 시각 — 파일 docstring 참조.
        "timestamp": float(estimate.timestamp),
        **_clock_fields(stamp_monotonic_ns, stamp_wall_ns),
        # 위치는 전부 프레임을 이름에 박는다 — 파일 docstring "필드 이름에 프레임을 박은 이유".
        "position_cam": [float(v) for v in p_cam],   # 원본(무손실 왕복의 출처)
        "position_flu": [float(v) for v in p_flu],   # §4.4 정식 권고 프레임
        "position_frd": [float(v) for v in p_frd],   # MAVLink LANDING_TARGET frame=12(BODY_FRD)
        # ⚠️ orientation은 **변환하지 않고 카메라 광학 프레임 그대로** 싣는다. 이번 Phase의
        # 좌표변환 범위는 "상대 벡터"까지이고(세션 지시), 타겟 자세의 body 프레임 표현은
        # 지금 소비자(4단 사다리 유도)가 쓰지 않는다. 프레임을 옆 필드에 명시해 Phase 2가
        # 이 값을 body 자세로 오인하지 못하게 한다.
        "orientation": [float(v) for v in estimate.orientation],  # (x, y, z, w)
        "orientation_frame": ORIENTATION_FRAME_CAM,
        "uncertainty": _uncertainty_to_json(estimate.uncertainty),
        "confidence": float(estimate.confidence),
        "target_type": str(estimate.target_type),
        # provenance echo(§7.3) — "어느 캘리브로 난 비행인가"를 사후에 특정할 수 있게 한다.
        "calib_accuracy": str(estimate.calib_accuracy),
        "not_for_closed_loop_30cm": bool(estimate.not_for_closed_loop_30cm),
        "calib_id": str(estimate.calib_id),
        # 위 플래그를 소비자가 바로 쓸 수 있는 형태로 번역한 것(§5.5).
        "closed_loop_floor_agl_m": float(
            closed_loop_floor_agl_m(estimate.not_for_closed_loop_30cm, cfg)
        ),
        # 🔴 미측정 사실이 소비자와 로그까지 전파된다(§7 U3).
        "mount_yaw_psi_m_rad": float(psi),
        "mount_yaw_measured": bool(cfg.mount_yaw_measured),
    }
    if estimate.meta:
        record["meta"] = dict(estimate.meta)
    return record


def build_invalid_target_record(
    *,
    seq: int,
    frame_id: int,
    reason: str,
    stamp_monotonic_ns: int,
    stamp_wall_ns: int,
    contract: Optional[FailsafeContract] = None,
) -> dict:
    """검출이 없는 프레임용 `valid=false` 레코드 (§5.4 계약 2번).

    **발행을 멈추지 않는 것이 핵심이다.** 침묵은 "노드가 죽었다"는 뜻으로 예약돼 있다(§5.4:
    "침묵을 유일한 권위로 삼고, 명시적 무효는 지연 단축용 최적화로만 얹는다"). 검출 상실 때도
    침묵해 버리면 소비자가 죽음과 안 보임을 구분할 수 없고, 사유도 로그에 안 남는다.

    포즈 필드는 전부 None이다 — 0.0을 채우면 소비자가 `valid`를 안 보고 썼을 때 "정확히
    기체 바로 아래에 타겟이 있다"는 **가장 위험한 거짓말**이 된다.
    """
    cfg = contract or FailsafeContract()
    return {
        "schema_version": SCHEMA_VERSION,
        "type": RECORD_TYPE_TARGET,
        "seq": int(seq),
        "frame_id": int(frame_id),
        "valid": False,
        "reason": str(reason),
        "timestamp": None,
        **_clock_fields(stamp_monotonic_ns, stamp_wall_ns),
        "position_cam": None,
        "position_flu": None,
        "position_frd": None,
        "orientation": None,
        "orientation_frame": ORIENTATION_FRAME_CAM,
        "uncertainty": None,
        "confidence": None,
        "target_type": None,
        "calib_accuracy": None,
        # 무효 레코드라도 보수적인 쪽(True)을 유지한다 — 소비자가 이 필드만 보고 해금하는 일이
        # 없게 한다.
        "not_for_closed_loop_30cm": True,
        "calib_id": None,
        "closed_loop_floor_agl_m": float(cfg.unverified_calib_floor_agl_m),
        "mount_yaw_psi_m_rad": float(cfg.mount_yaw_psi_m_rad),
        "mount_yaw_measured": bool(cfg.mount_yaw_measured),
    }


def build_state_hint_record(
    decision: Decision,
    *,
    seq: int,
    frame_id: int,
    stamp_monotonic_ns: int,
    stamp_wall_ns: int,
) -> dict:
    """`Decision` → `state_hint` 레코드.

    `state`는 §6.3의 **거부권(veto)** 후보다 — 소비자는 `HOLD`/`ABORT_ASCEND`를 보면 반드시
    멈추거나 올라가되, `PRECISION_SERVO`를 봤다고 해서 내려갈 의무는 없다(FC 자체 안전조건이
    항상 이긴다: "더 보수적인 쪽이 이긴다").

    `command`는 **`command_hint`로 이름을 바꿔** 싣고 `command_is_advisory: true`를 함께 박는다
    — 파일 docstring "🔴 `command`는 명령이 아니다" 참조.

    `reason`은 지금 JSONL 블랙박스에도 안 실리는 값인데(in-process에만 남음) 인터페이스에는
    싣는다 — 사후 분석에서 "왜 거절했나"가 가장 중요하다(§6.3).
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "type": RECORD_TYPE_STATE_HINT,
        "seq": int(seq),
        "frame_id": int(frame_id),
        **_clock_fields(stamp_monotonic_ns, stamp_wall_ns),
        "state": str(decision.state.value if isinstance(decision.state, LandingState) else decision.state),
        "command_hint": str(decision.command),
        "command_is_advisory": True,
        "reason": str(decision.reason),
        "blind_duration_s": float(decision.blind_duration_s),
        "scale_source": decision.scale_source,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 인코딩 / 디코딩
# ──────────────────────────────────────────────────────────────────────────────


def encode_line(record: dict) -> str:
    """레코드 dict → **개행으로 끝나는 한 줄** JSON.

    `ensure_ascii=False`는 `utils/blackbox.py`의 관례를 따른 것(한글 사유 문자열이 그대로
    읽힌다). `separators`로 공백을 없애 대역폭을 줄인다.

    **개행 안전성:** `json.dumps`는 문자열 값 안의 실제 개행을 반드시 `\\n` 2글자로 이스케이프
    하므로 반환값에는 **맨 끝 하나 말고 개행이 없다** — 그래서 소비자가 `readline()`만으로
    레코드 경계를 정확히 찾을 수 있다(`tests/test_wire.py`가 회귀로 박아 둠).
    """
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"


def decode_line(line: str) -> dict:
    """한 줄 JSON → 레코드 dict. 빈 줄/공백 줄은 `ValueError`."""
    s = line.strip()
    if not s:
        raise ValueError("decode_line: 빈 줄")
    record = json.loads(s)
    if not isinstance(record, dict):
        raise ValueError(f"decode_line: dict가 아님 — {type(record).__name__}")
    return record


def target_estimate_from_record(record: dict) -> TargetEstimate:
    """`type="target"` 레코드 → `TargetEstimate` **왕복 복원**.

    복원의 출처는 `position_cam`이다 — 원본이 카메라 광학 프레임이고 `_flu`/`_frd`는 거기서
    유도된 값이므로, 유도값에서 되돌리면 왕복이 회전 오차만큼 흐려진다.

    ⚠️ Phase 2 shim은 **이 함수를 쓸 수 없다**(컨테이너에 vision 코드가 없다). 이 함수의 존재
    이유는 ① 왕복 테스트로 스키마가 무손실임을 증명하는 것, ② 호스트 쪽 소비자/재생 도구가
    쓰는 것 두 가지다. shim은 같은 키를 stdlib `json`으로 직접 읽으면 된다.
    """
    if record.get("type") != RECORD_TYPE_TARGET:
        raise ValueError(f"target 레코드가 아님 — type={record.get('type')!r}")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"스키마 버전 불일치 — 기대 {SCHEMA_VERSION}, 받음 {record.get('schema_version')!r}"
        )
    if not record.get("valid", False) or record.get("position_cam") is None:
        raise ValueError(f"무효 레코드는 TargetEstimate로 복원할 수 없다 — reason={record.get('reason')!r}")

    unc = record.get("uncertainty")
    return TargetEstimate(
        position=tuple(float(v) for v in record["position_cam"]),
        orientation=tuple(float(v) for v in record["orientation"]),
        confidence=float(record["confidence"]),
        target_type=str(record["target_type"]),
        frame_id=int(record["frame_id"]),
        timestamp=float(record["timestamp"]),
        uncertainty=None if unc is None else np.asarray(unc, dtype=np.float64),
        calib_accuracy=str(record["calib_accuracy"]),
        not_for_closed_loop_30cm=bool(record["not_for_closed_loop_30cm"]),
        calib_id=str(record["calib_id"]),
        meta=dict(record.get("meta") or {}),
    )


def decision_from_record(record: dict) -> Decision:
    """`type="state_hint"` 레코드 → `Decision` 왕복 복원.

    `command_hint` → `Decision.command`로 되돌린다(이름을 바꾼 건 **와이어에서만** — 소비자가
    이걸 명령으로 쓰면 안 된다는 신호를 형식에 남기기 위한 것이고, 내부 dataclass 계약은 그대로).
    """
    if record.get("type") != RECORD_TYPE_STATE_HINT:
        raise ValueError(f"state_hint 레코드가 아님 — type={record.get('type')!r}")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"스키마 버전 불일치 — 기대 {SCHEMA_VERSION}, 받음 {record.get('schema_version')!r}"
        )
    return Decision(
        state=LandingState(record["state"]),
        command=str(record["command_hint"]),
        reason=str(record["reason"]),
        blind_duration_s=float(record["blind_duration_s"]),
        scale_source=record.get("scale_source"),
    )


def validate_record(record: dict) -> tuple[bool, str]:
    """필수 키가 다 있는가 — Phase 2 shim이 그대로 베껴 쓸 수 있는 최소 검증.

    `(ok, 사유)`. 값의 타당성이 아니라 **스키마 형태만** 본다(과설계 금지).
    """
    rtype: Any = record.get("type")
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
