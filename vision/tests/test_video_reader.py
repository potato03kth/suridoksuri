"""`utils/video_reader.py::VideoReader` 회귀 테스트.

검증 대상은 `vision/CLAUDE.md` 테스트 규칙표의 계약 —
**"프레임 이터레이트·fps·컨텍스트 종료"**.

몽키패치 없이 **실제 mp4를 써서 실제로 디코딩**한다(`test_frame_source.py`와 같은 원칙).
프레임을 세는 것으로 끝내지 않고 **순서가 보존되는지**를 프레임마다 다른 밝기를 심어
확인한다 — 순서가 뒤집히면 `TemporalFusion`의 시간축 확정이 통째로 무의미해지므로
이 모듈에서 가장 중요한 계약이다.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.utils.video_reader import VideoReader

N_FRAMES = 12
FPS = 20.0
W, H = 64, 48


def _brightness(i: int) -> int:
    """프레임 i의 고유 밝기 — 순서 검증용 지문."""
    return 20 + i * 15


@pytest.fixture
def video(tmp_path: Path) -> Path:
    """프레임마다 밝기가 다른 실제 mp4."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert writer.isOpened(), "mp4v 인코더를 열 수 없다 (환경 문제)"
    for i in range(N_FRAMES):
        writer.write(np.full((H, W, 3), _brightness(i), dtype=np.uint8))
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


# --------------------------------------------------------------------------
# 프레임 이터레이트
# --------------------------------------------------------------------------

def test_iterates_every_frame_as_bgr_ndarray(video):
    with VideoReader(str(video)) as reader:
        frames = list(reader)

    assert len(frames) == N_FRAMES
    for f in frames:
        assert isinstance(f, np.ndarray)
        assert f.dtype == np.uint8
        assert f.shape == (H, W, 3)      # (H, W, 3) BGR


def test_frame_order_is_preserved(video):
    """순서 계약 — 시간축 모듈(fusion/tracker)이 전적으로 여기에 기댄다."""
    with VideoReader(str(video)) as reader:
        means = [float(f.mean()) for f in reader]

    expected = [_brightness(i) for i in range(N_FRAMES)]
    # mp4 는 손실 압축이라 정확히 일치하진 않는다 — 순서(단조증가)와 근사값으로 본다
    assert means == sorted(means), f"프레임 순서가 뒤집혔다: {means}"
    for got, want in zip(means, expected):
        assert abs(got - want) < 8, f"프레임 밝기 {got} 가 기대 {want} 와 너무 다르다"


def test_iteration_is_exhausted_not_infinite(video):
    """`while True` 루프가 파일 끝에서 실제로 멈춘다 (무한루프 회귀)."""
    with VideoReader(str(video)) as reader:
        assert len(list(reader)) == N_FRAMES
        assert list(reader) == []        # 이미 소진된 뒤엔 빈 목록


def test_is_deterministic_across_reads(video):
    """같은 파일 → 같은 프레임열 (plan §7.5)."""
    with VideoReader(str(video)) as a:
        first = [f.mean() for f in a]
    with VideoReader(str(video)) as b:
        second = [f.mean() for f in b]
    assert first == second


# --------------------------------------------------------------------------
# fps / frame_count 메타데이터
# --------------------------------------------------------------------------

def test_fps_reports_the_encoded_rate(video):
    with VideoReader(str(video)) as reader:
        assert reader.fps == pytest.approx(FPS, abs=0.5)


def test_frame_count_matches_actual_iteration(video):
    """메타데이터가 실제 디코딩 결과와 어긋나지 않는다."""
    with VideoReader(str(video)) as reader:
        assert reader.frame_count == N_FRAMES
        assert isinstance(reader.frame_count, int)
        assert len(list(reader)) == reader.frame_count


# --------------------------------------------------------------------------
# 컨텍스트 종료
# --------------------------------------------------------------------------

def test_context_manager_returns_self_and_releases_capture(video):
    """`__enter__`는 자기 자신을, `__exit__`은 실제로 VideoCapture를 놓아준다."""
    with VideoReader(str(video)) as reader:
        assert isinstance(reader, VideoReader)
        assert reader._cap.isOpened()
    assert not reader._cap.isOpened(), "__exit__ 이후에도 capture가 열려 있다"


def test_capture_is_released_even_when_body_raises(video):
    """예외로 빠져나가도 자원을 놓는다 (누수 방지)."""
    reader = VideoReader(str(video))
    with pytest.raises(RuntimeError):
        with reader:
            raise RuntimeError("boom")
    assert not reader._cap.isOpened()


def test_iteration_after_close_yields_nothing_instead_of_crashing(video):
    """닫힌 뒤 읽어도 크래시가 아니라 조용히 끝난다 (공통 규칙 3: 경계 입력)."""
    with VideoReader(str(video)) as reader:
        pass
    assert list(reader) == []


# --------------------------------------------------------------------------
# 없는 파일 / 못 여는 파일 에러
# --------------------------------------------------------------------------

def test_missing_file_raises_file_not_found(tmp_path):
    missing = tmp_path / "nope.mp4"
    with pytest.raises(FileNotFoundError) as e:
        VideoReader(str(missing))
    assert str(missing) in str(e.value)


def test_unopenable_file_raises_io_error(tmp_path):
    """존재하지만 영상이 아닌 파일 — 조용히 0프레임을 흘리지 않고 즉시 실패한다."""
    bogus = tmp_path / "broken.mp4"
    bogus.write_bytes(b"not a video at all")
    with pytest.raises(IOError) as e:
        VideoReader(str(bogus))
    assert str(bogus) in str(e.value)
