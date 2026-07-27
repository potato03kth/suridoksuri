"""`modules/fusion.py::TemporalFusion` 회귀 테스트.

검증 대상은 이 모듈이 존재하는 이유 — **시간축 확정과 흔들림 억제**다.
(`vision/CLAUDE.md` 테스트 규칙표: "detections→confirmed 시간확정·흔들림 억제")

**[2026-07-28] 구조적 결함이 수정됐다 — 이 파일은 그에 맞춰 갱신됐다.**

예전 `_decay()`는 방금 매칭돼 `count += 1` 된 후보의 count까지 깎아, 프레임당 검출이
1개뿐이면 순증이 0이 되고 후보가 목록에서 제거됐다 → 몇 프레임을 연속으로 나타나든
절대 확정되지 않았다(클래스 docstring의 계약과 정면 충돌). 수정 갈래와 기각한 대안은
`modules/fusion.py::_decay()`의 설계 노트 참조.

수정으로 확정된 `count`의 의미: **연속으로 관측된 프레임 수.** 관측된 프레임 +1(그
프레임에 겹치는 검출이 몇 개든 +1), 미관측 프레임 -1. 이 파일은 그 계약을 고정한다 —
특히 "프레임당 최대 +1"은 흔들림 억제를 지탱하는 핵심이라 별도 테스트로 못 박았다
(`test_count_counts_frames_not_detections_per_frame`,
`test_flickering_dense_detection_is_never_confirmed`).
"""

import numpy as np
import pytest

from vision.core.state import VisionState, Detection
from vision.modules.fusion import TemporalFusion


# 같은 타깃이 만든 겹치는 검출 쌍 (IoU ≈ 0.82 — 기본 임계 0.4를 넉넉히 넘김)
TARGET_A = [(100, 100, 40, 40), (102, 102, 40, 40)]
# A와 전혀 겹치지 않는 다른 자리의 타깃 (IoU = 0.0)
TARGET_B = [(300, 300, 40, 40), (302, 302, 40, 40)]


def _blank_state() -> VisionState:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    return VisionState(original=img, current=img.copy())


def _feed(fusion: TemporalFusion, boxes) -> VisionState:
    """한 프레임 분량의 bbox 목록을 흘려 넣고 결과 state를 돌려준다."""
    state = _blank_state()
    state.detections = [Detection(bbox=b) for b in boxes]
    return fusion(state)


def _confirm_frame_index(fusion: TemporalFusion, boxes, n_frames: int):
    """confirmed가 처음 채워진 프레임 번호(1-based). 끝까지 없으면 None."""
    for i in range(1, n_frames + 1):
        if _feed(fusion, boxes).confirmed is not None:
            return i
    return None


# --------------------------------------------------------------------------
# 시간축 확정 — 증거가 min_frames만큼 쌓이기 전에는 확정하지 않는다
# --------------------------------------------------------------------------

def test_evidence_below_min_frames_is_not_confirmed():
    """증거가 임계에 못 미치는 동안은 confirmed도 meta도 남기지 않는다."""
    fusion = TemporalFusion(min_frames=5, iou_threshold=0.4)
    for _ in range(4):
        state = _feed(fusion, TARGET_A)
        assert state.confirmed is None
        assert "fusion" not in state.meta


def test_steady_target_is_confirmed_exactly_at_min_frames():
    """연속 프레임에 걸쳐 나타난 타깃은 정확히 min_frames번째 프레임에 승격된다."""
    fusion = TemporalFusion(min_frames=5, iou_threshold=0.4)
    assert _confirm_frame_index(fusion, TARGET_A, n_frames=10) == 5

    # 승격된 detection은 그 타깃 자리의 것이어야 한다 (엉뚱한 후보가 아니라)
    state = _feed(fusion, TARGET_A)
    assert state.confirmed is not None
    assert state.confirmed.bbox in TARGET_A
    assert state.meta["fusion"]["confirmed_count"] >= 5


@pytest.mark.parametrize("min_frames", [2, 3, 5, 7])
def test_min_frames_threshold_is_configurable(min_frames):
    """승격 임계는 설정값 그대로 동작한다 (하드코딩된 5가 아니다)."""
    fusion = TemporalFusion(min_frames=min_frames, iou_threshold=0.4)
    assert _confirm_frame_index(fusion, TARGET_A, n_frames=min_frames * 3) == min_frames


def test_confirmed_count_grows_while_target_persists():
    """타깃이 계속 보이면 확정 카운트가 단조 증가한다 (증거 누적의 직접 증거)."""
    fusion = TemporalFusion(min_frames=3, iou_threshold=0.4)
    counts = []
    for _ in range(8):
        state = _feed(fusion, TARGET_A)
        if state.confirmed is not None:
            counts.append(state.meta["fusion"]["confirmed_count"])
    assert len(counts) >= 5
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


# --------------------------------------------------------------------------
# 흔들림 억제 — 깜빡이는 검출은 확정되지 않는다
# --------------------------------------------------------------------------

def test_flickering_detection_is_never_confirmed():
    """한 프레임 걸러 나타나는(깜빡이는) 검출은 20프레임을 줘도 승격되지 않는다.

    같은 자리·같은 총 등장횟수라도 '연속'이 아니면 억제된다는 것이 이 모듈의 존재 이유다.
    """
    fusion = TemporalFusion(min_frames=5, iou_threshold=0.4)
    for i in range(20):
        state = _feed(fusion, TARGET_A if i % 2 == 0 else [])
        assert state.confirmed is None, f"깜빡이는 검출이 프레임 {i}에서 승격됐다"


def test_single_frame_spike_leaves_no_residue():
    """딱 한 프레임만 튄 검출은 다음 프레임에 흔적 없이 사라진다."""
    fusion = TemporalFusion(min_frames=2, iou_threshold=0.4)
    _feed(fusion, TARGET_A)
    _feed(fusion, [])
    assert fusion._candidates == []
    # 이후 다른 자리에 타깃이 떠도 사라진 후보가 되살아나 오염시키지 않는다
    state = _feed(fusion, TARGET_B)
    assert state.confirmed is None


def test_confirmation_lapses_when_target_disappears_and_must_be_re_earned():
    """확정된 뒤에도 타깃이 사라지면 증거가 깎이고, 재확정에 다시 프레임이 필요하다."""
    fusion = TemporalFusion(min_frames=5, iou_threshold=0.4)
    for _ in range(5):
        _feed(fusion, TARGET_A)
    assert _feed(fusion, TARGET_A).confirmed is not None  # 6프레임째 → count 6

    # 빈 프레임 3장 → 프레임당 1씩 감쇠 (6 → 3), 그 사이 확정도 안 실린다
    for _ in range(3):
        state = _feed(fusion, [])
        assert state.confirmed is None

    # 임계 아래로 떨어졌으므로 한 프레임 재등장으로는 되살아나지 않는다
    assert _feed(fusion, TARGET_A).confirmed is None
    # 한 프레임 더 쌓이면 다시 확정
    assert _feed(fusion, TARGET_A).confirmed is not None


# --------------------------------------------------------------------------
# IoU 연관 — 어떤 검출이 "같은 자리"로 묶이는가
# --------------------------------------------------------------------------

def test_iou_exact_values_for_known_boxes():
    """연관 판단의 근간인 IoU 계산 자체를 알려진 값으로 고정한다."""
    box = (0, 0, 10, 10)
    assert TemporalFusion._iou(box, box) == pytest.approx(1.0)
    assert TemporalFusion._iou(box, (100, 100, 10, 10)) == pytest.approx(0.0)
    assert TemporalFusion._iou(box, (10, 0, 10, 10)) == pytest.approx(0.0)  # 변만 맞닿음
    # 가로로 절반 겹침 → 교집합 50, 합집합 150
    assert TemporalFusion._iou(box, (5, 0, 10, 10)) == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    "iou_threshold, expect_shared",
    [
        (0.30, True),   # 임계 < 실제 IoU(1/3) → 한 후보로 병합
        (0.90, False),  # 임계 > 실제 IoU     → 별개 후보로 분리
    ],
)
def test_iou_threshold_decides_whether_boxes_share_a_candidate(iou_threshold, expect_shared):
    """iou_threshold 설정값이 실제로 연관/분리를 가른다.

    ⚠️ 예전 버전은 임계가 높으면 "끝내 확정 못 함"을 기대했는데, 그건 계약이 아니라
    결함(프레임당 순증 0)의 부산물이었다 — 분리된 두 박스도 매 프레임 꾸준히
    나타나는 이상 각자 정당한 지속 타깃이므로 각자 확정되는 것이 맞다. 그래서
    이 테스트가 실제로 겨냥하는 계약(= 몇 개의 후보로 묶이는가)을 직접 확인한다.
    """
    partial_pair = [(100, 100, 40, 40), (120, 100, 40, 40)]  # IoU = 1/3
    assert TemporalFusion._iou(*partial_pair) == pytest.approx(1.0 / 3.0)

    fusion = TemporalFusion(min_frames=3, iou_threshold=iou_threshold)
    for _ in range(3):
        state = _feed(fusion, partial_pair)

    assert len(fusion._candidates) == (1 if expect_shared else 2)
    # 병합되든 분리되든 3프레임 연속 등장했으므로 확정 자체는 된다
    assert state.confirmed is not None
    assert state.confirmed.bbox in partial_pair


def test_separate_targets_accumulate_independently_and_strongest_wins():
    """멀리 떨어진 두 타깃은 서로의 증거를 뺏지 않고, 증거가 많은 쪽이 확정된다.

    ⚠️ 비대칭은 반드시 **프레임 수**로 만든다 — count는 등장 프레임 수이지 프레임당
    검출 개수가 아니므로, 예전 버전처럼 "프레임당 검출을 더 많이 넣는" 방식으로는
    비대칭이 아예 생기지 않는다(둘 다 같은 count가 돼 `max()`의 목록 순서 우연으로
    통과하는 pseudo 테스트가 된다 — 수정 직후 실제로 4:4 동률임을 확인했다).
    그래서 **더 늦게 등록된 쪽을 더 강하게** 만들어, 순서가 아니라 count 비교가
    승자를 가르는지 확인한다.
    """
    fusion = TemporalFusion(min_frames=2, iou_threshold=0.4)

    for _ in range(3):                       # A가 먼저 후보로 등록된다
        _feed(fusion, TARGET_A + TARGET_B)
    for _ in range(2):                       # A만 사라져 감쇠 (3 → 1), B는 계속 누적
        _feed(fusion, TARGET_B)
    for _ in range(2):                       # A 복귀
        state = _feed(fusion, TARGET_A + TARGET_B)

    # 두 후보가 별개로 살아 있고 서로의 증거를 뺏지 않았다
    assert len(fusion._candidates) == 2
    assert sorted(c["count"] for c in fusion._candidates) == [3, 7]
    # 목록 순서상 앞은 A(3)지만, 확정되는 것은 증거가 많은 B(7)다
    assert state.confirmed is not None
    assert state.confirmed.bbox in TARGET_B
    assert state.confirmed.bbox not in TARGET_A


# --------------------------------------------------------------------------
# 공통 규칙 (vision/CLAUDE.md "공통 규칙")
# --------------------------------------------------------------------------

def test_empty_detection_stream_is_safe():
    """검출이 한 번도 없어도 크래시 없이 아무것도 확정하지 않는다."""
    fusion = TemporalFusion()
    for _ in range(5):
        state = _feed(fusion, [])
    assert state.confirmed is None
    assert state.meta == {}


def test_only_declared_fields_are_written():
    """선언 필드 계약: detections를 읽고 confirmed/meta만 쓴다."""
    fusion = TemporalFusion(min_frames=1, iou_threshold=0.4)
    state = _blank_state()
    original_ref, current_ref = state.original, state.current
    original_copy = state.original.copy()
    detections = [Detection(bbox=b) for b in TARGET_A]
    state.detections = list(detections)

    out = fusion(state)

    assert out is state
    assert out.original is original_ref
    assert np.array_equal(out.original, original_copy)
    assert out.current is current_ref
    assert out.mask is None
    assert out.detections == detections          # detections는 건드리지 않는다
    assert set(out.meta).issubset({"fusion"})    # meta 네임스페이스는 "fusion" 하나


def test_deterministic_for_identical_input_sequence():
    """같은 관측열 → 같은 확정열 (plan §7.5)."""
    sequence = [TARGET_A, TARGET_A, [], TARGET_A, TARGET_A, TARGET_A, TARGET_B, TARGET_A]

    def run():
        fusion = TemporalFusion(min_frames=3, iou_threshold=0.4)
        trace = []
        for boxes in sequence:
            state = _feed(fusion, boxes)
            bbox = state.confirmed.bbox if state.confirmed is not None else None
            trace.append((bbox, state.meta.get("fusion")))
        return trace

    assert run() == run()


# --------------------------------------------------------------------------
# 문서화된 계약 vs 현재 구현 — 위 파일 docstring의 결함
# --------------------------------------------------------------------------

def test_documented_contract_single_detection_per_frame_confirms_after_min_frames():
    """docstring이 약속한 계약: 한 프레임에 검출이 1개여도 min_frames 연속이면 확정.

    2026-07-28 이전에는 `xfail(strict=True)`로 박혀 있던 결함이다 — `_decay()`가
    방금 매칭된 후보까지 깎아 프레임당 검출 1개는 순증이 0이었다. 수정 후 정상
    통과 테스트로 승격.
    """
    fusion = TemporalFusion(min_frames=3, iou_threshold=0.4)
    assert _confirm_frame_index(fusion, [TARGET_A[0]], n_frames=20) == 3


def test_count_counts_frames_not_detections_per_frame():
    """count는 "등장한 프레임 수"다 — 한 프레임에 겹치는 검출이 몇 개든 +1.

    이 규칙이 없으면(감쇠 제외만 하고 프레임당 +N을 허용하면) 빈 프레임의 -1로는
    상쇄가 안 돼 순증이 생기고, 깜빡이는 검출까지 승격돼 흔들림 억제가 무너진다.
    """
    dense = [(100, 100, 40, 40), (101, 101, 40, 40),
             (102, 102, 40, 40), (103, 103, 40, 40)]  # 한 프레임에 겹치는 검출 4개
    fusion = TemporalFusion(min_frames=3, iou_threshold=0.4)

    _feed(fusion, dense)
    assert [c["count"] for c in fusion._candidates] == [1]  # 4가 아니라 1

    # 검출 밀도를 4배로 줘도 승격 시점은 min_frames "프레임" 그대로다
    assert _feed(fusion, dense).confirmed is None       # 2프레임째
    assert _feed(fusion, dense).confirmed is not None   # 3프레임째


def test_flickering_dense_detection_is_never_confirmed():
    """한 프레임에 검출이 여러 개여도 깜빡이면 승격되지 않는다.

    `test_flickering_detection_is_never_confirmed`의 다중검출판 —
    프레임당 +N을 허용하는 변형에서 실제로 승격되는 것을 확인했기에 별도로 박아둔다.
    """
    dense = [(100, 100, 40, 40), (101, 101, 40, 40), (102, 102, 40, 40)]
    fusion = TemporalFusion(min_frames=5, iou_threshold=0.4)
    for i in range(20):
        state = _feed(fusion, dense if i % 2 == 0 else [])
        assert state.confirmed is None, f"깜빡이는 다중검출이 프레임 {i}에서 승격됐다"
