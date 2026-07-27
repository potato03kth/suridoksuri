"""`utils/image_loader.py::load_image` 회귀 테스트.

검증 대상은 `vision/CLAUDE.md` 테스트 규칙표의 계약 —
**"경로→BGR ndarray·없는 파일 에러"**.

`load_image`는 파이프라인의 입구다(`main.py` 이미지 경로). 여기서 색공간이 뒤집히거나
(RGB↔BGR) 실패가 조용히 None으로 새어나가면 뒤의 모든 HSV 색 검출
(`ColorFilter`/`RedRingDetector`)이 통째로 어긋난다 — 그래서 "ndarray가 나온다"에서
멈추지 않고 **채널 순서가 실제로 BGR인지**를 알려진 색으로 왕복 확인한다.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.utils.image_loader import load_image


@pytest.fixture
def bgr_probe(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """채널을 서로 구별할 수 있는 이미지 — 3분할 순수 B / G / R 띠."""
    img = np.zeros((12, 30, 3), dtype=np.uint8)
    img[:, 0:10] = (255, 0, 0)     # BGR 관례에서 파랑
    img[:, 10:20] = (0, 255, 0)    # 초록
    img[:, 20:30] = (0, 0, 255)    # 빨강
    path = tmp_path / "probe.png"
    assert cv2.imwrite(str(path), img)   # PNG = 무손실이라 값이 그대로 돌아온다
    return path, img


# --------------------------------------------------------------------------
# 경로 → BGR ndarray
# --------------------------------------------------------------------------

def test_returns_bgr_ndarray_with_expected_shape_and_dtype(bgr_probe):
    path, expected = bgr_probe
    img = load_image(str(path))

    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.shape == (12, 30, 3)     # (H, W, 3) — 그레이스케일로 접히지 않는다


def test_channel_order_is_bgr_not_rgb(bgr_probe):
    """색공간 계약 — 채널이 뒤집히면 뒤의 HSV 검출이 전부 어긋난다."""
    path, _ = bgr_probe
    img = load_image(str(path))

    assert tuple(img[6, 5]) == (255, 0, 0), "파랑 띠가 BGR로 안 읽혔다"
    assert tuple(img[6, 15]) == (0, 255, 0)
    assert tuple(img[6, 25]) == (0, 0, 255), "빨강 띠가 BGR로 안 읽혔다"


def test_pixel_values_round_trip_losslessly(bgr_probe):
    """PNG 무손실 왕복 — 로더가 임의 스케일/정규화를 끼워넣지 않는다."""
    path, expected = bgr_probe
    assert np.array_equal(load_image(str(path)), expected)


def test_accepts_path_object_as_well_as_str(tmp_path):
    """호출자가 `Path`를 넘겨도 동작한다 (내부에서 `Path(path)`로 감싸므로)."""
    path = tmp_path / "p.png"
    cv2.imwrite(str(path), np.full((4, 4, 3), 7, dtype=np.uint8))
    assert load_image(path).shape == (4, 4, 3)


def test_grayscale_source_is_expanded_to_three_channels(tmp_path):
    """1채널 파일도 파이프라인 계약(BGR 3채널)에 맞춰 나온다."""
    path = tmp_path / "gray.png"
    cv2.imwrite(str(path), np.full((8, 8), 128, dtype=np.uint8))
    img = load_image(str(path))
    assert img.shape == (8, 8, 3)


def test_is_deterministic(bgr_probe):
    """같은 파일 → 같은 배열 (plan §7.5)."""
    path, _ = bgr_probe
    assert np.array_equal(load_image(str(path)), load_image(str(path)))


# --------------------------------------------------------------------------
# 없는 파일 / 못 읽는 파일 에러
# --------------------------------------------------------------------------

def test_missing_file_raises_file_not_found(tmp_path):
    """조용히 None을 흘리지 않고 즉시 실패한다."""
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(FileNotFoundError) as e:
        load_image(str(missing))
    assert str(missing) in str(e.value)   # 어느 경로가 문제인지 메시지에 남는다


def test_directory_path_raises_rather_than_returning_none(tmp_path):
    """존재하지만 이미지가 아닌 경로(디렉터리) — 디코드 실패로 ValueError."""
    with pytest.raises((ValueError, IsADirectoryError, cv2.error)):
        load_image(str(tmp_path))


def test_undecodable_file_raises_value_error(tmp_path):
    """확장자만 이미지인 쓰레기 파일 — cv2가 None을 주면 ValueError로 승격한다."""
    bogus = tmp_path / "broken.png"
    bogus.write_bytes(b"this is definitely not a png")
    with pytest.raises(ValueError) as e:
        load_image(str(bogus))
    assert "decode" in str(e.value).lower()


def test_empty_file_raises_value_error(tmp_path):
    """0바이트 파일도 크래시(AttributeError 등)가 아니라 ValueError로 떨어진다."""
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        load_image(str(empty))
