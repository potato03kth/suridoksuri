"""`modules/tracker.py::KalmanTracker` 회귀 테스트.

검증 대상은 이 모듈이 존재하는 이유 — **관측으로의 수렴과 검출 공백에서의 연속성**이다.
(`vision/CLAUDE.md` 테스트 규칙표: "detections→meta 추적·연속성")

⚠️ **범위 주의 — 이 모듈은 다중 트랙 추적기가 아니다.**
docstring 그대로 "단일 타깃 Kalman 추적"이라 **트랙 ID도, 검출↔트랙 연관(association)도
없다.** 매 프레임 `max(confidence)` 검출 하나만 측정치로 쓰는 단일 필터다. 따라서
"프레임 간 같은 물체가 같은 트랙 ID를 유지" / "서로 다른 물체가 트랙을 뒤바꿔 잡지 않음"
같은 다중트랙 계약은 여기서 검증할 수 없다 — 대응되는 실제 계약은
`test_highest_confidence_detection_is_used_as_the_measurement`(측정치 선택 규칙)과
`test_track_state_survives_dropout_on_reacquisition`(연속성)이다.
"""

import numpy as np
import pytest

from vision.core.state import VisionState, Detection
from vision.modules.tracker import KalmanTracker


def _blank_state() -> VisionState:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    return VisionState(original=img, current=img.copy())


def _det(cx: int, cy: int, confidence: float = 1.0, size: int = 20) -> Detection:
    """중심이 (cx, cy)인 detection (Detection.center = x + w//2, y + h//2)."""
    half = size // 2
    return Detection(bbox=(cx - half, cy - half, size, size), confidence=confidence)


def _step(tracker: KalmanTracker, detections=()) -> VisionState:
    state = _blank_state()
    state.detections = list(detections)
    return tracker(state)


def _predicted(tracker: KalmanTracker, detections=()):
    return _step(tracker, detections).meta["kalman"]["predicted"]


# --------------------------------------------------------------------------
# 관측 추종 — 예측이 실제로 관측 방향으로 수렴하는가
# --------------------------------------------------------------------------

def test_first_detection_locks_prediction_onto_the_measurement():
    """최초 검출은 필터를 그 자리에서 초기화한다 (0에서 끌려오는 과도구간 없음)."""
    tracker = KalmanTracker()
    assert _predicted(tracker, [_det(50, 70)]) == (50, 70)


def test_static_target_prediction_stays_locked():
    """정지 타깃을 계속 보면 예측이 그 자리에 머문다 (드리프트 없음)."""
    tracker = KalmanTracker()
    for _ in range(10):
        assert _predicted(tracker, [_det(50, 50)]) == (50, 50)


def test_prediction_converges_onto_a_relocated_target():
    """타깃이 순간이동하면 예측이 새 관측 쪽으로 수렴한다 (correct 단계의 존재 증거)."""
    tracker = KalmanTracker()
    for _ in range(5):
        _step(tracker, [_det(50, 50)])
    assert _predicted(tracker, [_det(50, 50)]) == (50, 50)  # 이동 전에는 옛 자리

    for _ in range(15):
        pred = _predicted(tracker, [_det(150, 50)])

    assert pred[0] == pytest.approx(150, abs=2)
    assert pred[1] == pytest.approx(50, abs=2)


def test_prediction_error_shrinks_monotonically_once_settling():
    """수렴은 요행이 아니다 — 후반부 오차가 전반부보다 확실히 작다."""
    tracker = KalmanTracker()
    for _ in range(5):
        _step(tracker, [_det(50, 50)])

    errors = [abs(_predicted(tracker, [_det(150, 50)])[0] - 150) for _ in range(20)]
    assert max(errors[:5]) > 10          # 초반엔 아직 못 따라잡았고
    assert max(errors[-5:]) <= 2         # 끝에는 붙어 있다


@pytest.mark.parametrize(
    "kwargs_fast, kwargs_slow",
    [
        ({"measurement_noise": 0.001}, {"measurement_noise": 10.0}),
        ({"process_noise": 1.0}, {"process_noise": 0.0001}),
    ],
)
def test_noise_parameters_control_how_fast_the_filter_follows_measurements(kwargs_fast, kwargs_slow):
    """생성자 노이즈 파라미터가 실제로 추종 속도를 바꾼다 (하드코딩된 튜닝이 아니다)."""

    def error_after_jump(**kwargs):
        tracker = KalmanTracker(**kwargs)
        for _ in range(5):
            _step(tracker, [_det(50, 50)])
        for _ in range(6):
            pred = _predicted(tracker, [_det(150, 50)])
        return abs(pred[0] - 150)

    fast = error_after_jump(**kwargs_fast)
    slow = error_after_jump(**kwargs_slow)
    assert fast < slow, f"{kwargs_fast} 오차 {fast} 가 {kwargs_slow} 오차 {slow} 보다 작아야 한다"


def test_highest_confidence_detection_is_used_as_the_measurement():
    """같은 프레임에 여러 검출이 있으면 신뢰도가 가장 높은 것만 측정치가 된다."""
    tracker = KalmanTracker()
    low, high = _det(10, 10, confidence=0.2), _det(200, 200, confidence=0.9)
    for _ in range(10):
        pred = _predicted(tracker, [low, high])
    assert pred == (200, 200)

    # 검출 순서가 바뀌어도 결과는 신뢰도로만 정해진다
    tracker = KalmanTracker()
    for _ in range(10):
        pred = _predicted(tracker, [high, low])
    assert pred == (200, 200)


# --------------------------------------------------------------------------
# 연속성 — 검출이 빠진 프레임에서도 트랙이 살아남는가 (칼만을 쓰는 이유)
# --------------------------------------------------------------------------

def test_track_dead_reckons_through_detection_dropout():
    """등속 이동 중 검출이 끊겨도 예측이 진행 방향으로 계속 외삽된다."""
    tracker = KalmanTracker()
    for i in range(12):
        _step(tracker, [_det(50 + 10 * i, 50)])

    xs = []
    for _ in range(5):
        state = _step(tracker, [])
        assert state.meta["kalman"]["source"] == "predict"
        xs.append(state.meta["kalman"]["predicted"][0])

    assert xs == sorted(xs) and len(set(xs)) == len(xs), f"예측이 멈춰 있다: {xs}"
    steps = [b - a for a, b in zip(xs, xs[1:])]
    assert all(8 <= s <= 12 for s in steps), f"학습된 속도(10 px/frame)와 다르다: {steps}"


def test_track_state_survives_dropout_on_reacquisition():
    """공백 뒤 재포착 시 필터를 새로 초기화하지 않고 이전 상태를 이어간다."""
    tracker = KalmanTracker()
    for i in range(12):
        _step(tracker, [_det(50 + 10 * i, 50)])
    for _ in range(3):
        _step(tracker, [])

    state = _step(tracker, [_det(200, 50)])
    assert state.meta["kalman"]["source"] == "correct"
    reacquired_x = state.meta["kalman"]["predicted"][0]

    # 새 트래커는 같은 측정치를 그냥 그대로 물고 시작한다 (속도 0)
    fresh_x = _predicted(KalmanTracker(), [_det(200, 50)])[0]
    assert fresh_x == 200
    # 살아남은 트랙은 학습한 속도만큼 앞서 있어야 한다 = 상태가 초기화되지 않았다는 증거
    assert reacquired_x > fresh_x + 5, f"재포착에서 상태가 리셋됐다 ({reacquired_x} vs {fresh_x})"


def test_no_meta_before_the_first_detection():
    """검출을 한 번도 못 본 상태에서 빈 프레임이 와도 아무 예측도 지어내지 않는다."""
    tracker = KalmanTracker()
    for _ in range(5):
        state = _step(tracker, [])
        assert state.meta == {}


def test_source_label_distinguishes_measured_from_extrapolated():
    """meta의 source가 측정 반영(correct)과 순수 외삽(predict)을 구분해준다."""
    tracker = KalmanTracker()
    assert _step(tracker, [_det(50, 50)]).meta["kalman"]["source"] == "correct"
    assert _step(tracker, []).meta["kalman"]["source"] == "predict"
    assert _step(tracker, [_det(60, 50)]).meta["kalman"]["source"] == "correct"


# --------------------------------------------------------------------------
# 공통 규칙 (vision/CLAUDE.md "공통 규칙")
# --------------------------------------------------------------------------

def test_zero_size_bbox_is_safe():
    """면적 0짜리 경계 입력에서도 크래시 없이 그 자리를 추적한다."""
    tracker = KalmanTracker()
    state = _step(tracker, [Detection(bbox=(30, 40, 0, 0))])
    assert state.meta["kalman"]["predicted"] == (30, 40)


def test_only_declared_fields_are_written():
    """선언 필드 계약: detections를 읽고 meta만 쓴다."""
    tracker = KalmanTracker()
    state = _blank_state()
    original_ref, current_ref = state.original, state.current
    original_copy = state.original.copy()
    detections = [_det(50, 50)]
    state.detections = list(detections)

    out = tracker(state)

    assert out is state
    assert out.original is original_ref
    assert np.array_equal(out.original, original_copy)
    assert out.current is current_ref
    assert out.mask is None
    assert out.confirmed is None                  # confirmed는 TemporalFusion 담당
    assert out.detections == detections           # detections는 건드리지 않는다
    assert set(out.meta) == {"kalman"}            # meta 네임스페이스는 "kalman" 하나


def test_deterministic_for_identical_input_sequence():
    """같은 관측열 → 같은 예측열 (plan §7.5)."""
    sequence = [
        [_det(50, 50)], [_det(60, 52)], [], [_det(80, 56)],
        [_det(90, 58, 0.3), _det(300, 300, 0.9)], [], [], [_det(110, 62)],
    ]

    def run():
        tracker = KalmanTracker()
        return [_step(tracker, dets).meta.get("kalman") for dets in sequence]

    assert run() == run()
