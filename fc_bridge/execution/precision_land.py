"""
비전 정밀착륙 판정 로직 (순수 함수).

`fc_ros.nodes.offboard_node` 의 `VISION_SEARCH` / `PRECISION_LAND` 서브상태가
쓰는 판정만 모은다. **ROS·rclpy·numpy 외 의존 없음** — `state_logic.py` 와 같은
규율이고, 같은 이유다: 노드 메서드 안에 판정을 묻어 두면 rclpy 없는 랩탑에서
단위테스트가 불가능해 회귀 그물이 통째로 사라진다.

탐색 **궤적** 기하는 `search_pattern.py` 소관이다(나선·풋프린트·선회속도).
이 파일은 "언제 래치하는가 / 언제 내려가는가 / 언제 손을 떼는가"만 다룬다.

계약 배경은 `docs/fc_precision_land_handoff.md` §2. 이 파일이 소비하는 것은
그 문서가 정의한 vision 쪽 산출물(`landing_setpoint` + `target_status`)을
`fc_ros.adapters.vision_target_bridge` 가 정규화한 결과다.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def latch_candidate(buf: Sequence,
                    min_ticks: int,
                    max_spread_m: float) -> Optional[np.ndarray]:
    """연속 관측 버퍼가 "같은 것을 계속 보고 있다"를 만족하면 착륙점을 돌려준다.

    `buf` 는 최신순이 아니라 **도착순** [N, E, h_up] 배열들이고, 마지막
    `min_ticks` 개만 본다. 반환값은 그 창의 평균 [N, E, h_up], 아니면 None.

    🔴 **단발 검출로 래치하지 않는 이유:** 탐색 중에는 기체가 계속 움직여 타겟이
    화각을 스쳐 지나간다. 한 프레임짜리 오탐 하나로 나선을 버리고 날아가면
    돌아올 방법이 없다(나선 상태를 이미 잃었다). 연속성 + 공간 일관성 두 가지를
    같이 요구하는 것이 "같은 물체"의 최소 증거다.

    🔴 **산포는 수평(N, E)만 본다.** 고도 성분은 카메라-타겟 거리 추정에서 오고
    그건 `nominal.yaml`(미검증 캘리브레이션) 정확도에 직접 걸려 있어, 수평보다
    훨씬 큰 분산을 갖는다. 여기에 같은 임계를 걸면 실제로는 잘 보고 있는데도
    영원히 래치가 안 선다.
    """
    n = int(min_ticks)
    if n <= 0 or len(buf) < n:
        return None
    win = np.array([np.asarray(b, dtype=float) for b in buf[-n:]])
    mean_ne = win[:, :2].mean(axis=0)
    spread = float(np.max(np.linalg.norm(win[:, :2] - mean_ne, axis=1)))
    if spread > float(max_spread_m):
        return None
    return np.array([mean_ne[0], mean_ne[1], float(win[:, 2].mean())])


def descend_allowed(guided: bool,
                    horiz_err_m: float,
                    align_tol_m: float,
                    veto: bool) -> bool:
    """수직 하강을 허가할 틱인가.

    🔴 **수평/수직 분리의 핵심.** `/vision/landing_setpoint` 를 통째로 위치
    setpoint 로 흘리면 `slew_setpoint` 가 3D 벡터 하나로 램프하므로 수평 정렬과
    수직 하강이 같은 속도 예산을 나눠 쓴다 — 25m 상공에서 15m 옆 타겟이면 3D
    거리 29m 를 `v_approach`(5m/s)로 걷느라 **수직 성분이 4m/s 하강**이 되고,
    정렬이 끝나기 전에 지면에 닿는다.

    그래서 "정렬됐고, 그 틱에 신선한 유도가 있고, 거부권이 없을 때"만 내려간다.
    vision 상태머신의 `CENTER_DESCEND`("중심 맞추며 하강") 의도를 FC 쪽에서
    구현한 것이다.

    `guided=False`(유도 상실)면 **고도를 붙든다** — 마지막으로 본 오차를 믿고
    추측 하강하면 그 오차가 그대로 착지 오차가 된다.
    """
    return bool(guided) and not bool(veto) and \
        float(horiz_err_m) < float(align_tol_m)


def handoff_due(agl_m: float,
                handoff_agl_m: float,
                land_hint: bool) -> bool:
    """AUTO.LAND 로 인계할 시점인가.

    두 경로다:
      ① `agl <= handoff_agl_m` — 계약상 바닥(`closed_loop_floor_agl_m` =
         vision 의 `terminal_agl_m` = 3.0m). 그 밑은 비전 폐루프의 몫이 아니다.
      ② vision 의 `command_hint == "land"` — TERMINAL 에서 블라인드가 2초를
         넘었다는 뜻이다. advisory 지만 **무시하면 설계 의도와 반대로 간다**:
         2초째 못 보는 추정으로 횡방향을 계속 물고 늘어지는 것은 오차를
         **키우는** 모드이고, 접지 순간의 횡속도가 0.105m 라이즈드 매트
         가장자리 전복으로 직결된다.
    """
    return float(agl_m) <= float(handoff_agl_m) or bool(land_hint)


def search_pass_next(pass_idx: int, max_passes: int = 2) -> Optional[int]:
    """탐색 회차 실패 시 다음 회차 번호. 더 없으면 None(= GPS 착륙 폴백).

    회차를 늘리는 것이 공짜가 아니라는 것이 이 함수의 존재 이유다 — 대회 성공
    판정이 *"재시도 없이"* 를 포함한 정성 기준이라(`vision_plan.md` §10),
    탐색을 오래 끄는 것 자체가 감점이다. 기본 2회에서 멈춘다.
    """
    nxt = int(pass_idx) + 1
    return nxt if nxt < int(max_passes) else None
