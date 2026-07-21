"""rpi_capture.py 순수 함수(unpack_raw10/debayer_to_bgr8) 단위 테스트.

rpi_capture.py 자체는 하드웨어(v4l2-ctl/media-ctl 서브프로세스) 전용 도구라 vision/CLAUDE.md
import 규칙상 tools/의 "CI/pytest 대상 아님" 예외가 적용되지만, 그 안의 디베이어/언패킹 로직은
numpy/cv2만 쓰는 순수 함수라 하드웨어 없이 검증 가능하다(CLAUDE.md "하드웨어 비의존 부분은
.venv 설치·pytest 대상" 원칙, jsonl_view.py와 동일한 예외 패턴).

unpack_raw10()의 왕복(pack→unpack) 정확성은 실기체 캡처(2026-07-22, RPi5+IMX708 실촬영,
bytes/line 5760@width4608로 실측 확인된 MIPI RAW10 레이아웃)와 동일한 패킹 규칙을 이 테스트
안에서 재구현해(`_pack_raw10`) 대조한다 — 실기체 없이도 언패킹 로직 자체의 정확성을 보장한다.
"""
import numpy as np
import pytest

from vision.tools.rpi_capture import debayer_to_bgr8, unpack_raw10


def _pack_raw10(pixels: np.ndarray) -> np.ndarray:
    """unpack_raw10()의 역함수 — 테스트 전용. (height, width) uint16(0~1023) ->
    MIPI RAW10 패킹된 1D uint8 바이트 배열. width는 4의 배수여야 한다."""
    height, width = pixels.shape
    assert width % 4 == 0
    p0 = pixels[:, 0::4].astype(np.uint16)
    p1 = pixels[:, 1::4].astype(np.uint16)
    p2 = pixels[:, 2::4].astype(np.uint16)
    p3 = pixels[:, 3::4].astype(np.uint16)
    b0 = (p0 >> 2).astype(np.uint8)
    b1 = (p1 >> 2).astype(np.uint8)
    b2 = (p2 >> 2).astype(np.uint8)
    b3 = (p3 >> 2).astype(np.uint8)
    b4 = ((p0 & 0x03) | ((p1 & 0x03) << 2) | ((p2 & 0x03) << 4) | ((p3 & 0x03) << 6)).astype(np.uint8)
    bytes_per_row = width // 4 * 5
    out = np.empty((height, bytes_per_row), dtype=np.uint8)
    out[:, 0::5] = b0
    out[:, 1::5] = b1
    out[:, 2::5] = b2
    out[:, 3::5] = b3
    out[:, 4::5] = b4
    return out.reshape(-1)


# ---------- unpack_raw10 ----------

def test_unpack_raw10_roundtrip_synthetic_random():
    rng = np.random.default_rng(42)
    height, width = 8, 16  # width % 4 == 0
    pixels = rng.integers(0, 1024, size=(height, width), dtype=np.uint16)  # 10비트 값 범위
    packed = _pack_raw10(pixels)
    unpacked = unpack_raw10(packed, width=width, height=height)
    assert unpacked.dtype == np.uint16
    assert unpacked.shape == (height, width)
    np.testing.assert_array_equal(unpacked, pixels)


def test_unpack_raw10_roundtrip_extremes():
    """0과 1023(10비트 최댓값) 극단값이 비트 손실 없이 왕복되는지 — 하위 2비트 패킹 버그가
    있다면 극단값에서 가장 먼저 드러난다."""
    height, width = 4, 8
    pixels = np.zeros((height, width), dtype=np.uint16)
    pixels[:, 0::2] = 1023
    pixels[:, 1::2] = 0
    packed = _pack_raw10(pixels)
    unpacked = unpack_raw10(packed, width=width, height=height)
    np.testing.assert_array_equal(unpacked, pixels)


def test_unpack_raw10_matches_real_hardware_bytes_per_line():
    """실기체(2026-07-22, RPi5+IMX708, 4608x2592 pRAA)에서 v4l2-ctl --get-fmt-video가
    보고한 Bytes per Line=5760을 그대로 재현하는지 — 언패킹 레이아웃이 실측 하드웨어와
    일치함을 고정하는 회귀 테스트."""
    width, height = 4608, 2592
    bytes_per_row = width // 4 * 5
    assert bytes_per_row == 5760
    packed = np.zeros(bytes_per_row * height, dtype=np.uint8)
    unpacked = unpack_raw10(packed, width=width, height=height)
    assert unpacked.shape == (height, width)


def test_unpack_raw10_rejects_width_not_multiple_of_4():
    with pytest.raises(ValueError, match="4의 배수"):
        unpack_raw10(np.zeros(100, dtype=np.uint8), width=10, height=4)


def test_unpack_raw10_rejects_size_mismatch():
    with pytest.raises(ValueError, match="크기 불일치"):
        unpack_raw10(np.zeros(3, dtype=np.uint8), width=8, height=4)


# ---------- debayer_to_bgr8 ----------

def test_debayer_to_bgr8_shape_dtype():
    bayer = np.full((16, 16), 512, dtype=np.uint16)
    bgr = debayer_to_bgr8(bayer, pattern="rggb")
    assert bgr.shape == (16, 16, 3)
    assert bgr.dtype == np.uint8


def test_debayer_to_bgr8_unknown_pattern_raises():
    bayer = np.zeros((8, 8), dtype=np.uint16)
    with pytest.raises(ValueError, match="알 수 없는 베이어 패턴"):
        debayer_to_bgr8(bayer, pattern="not_a_pattern")


def test_debayer_to_bgr8_rggb_channel_order_matches_synthetic_intensities():
    """RGGB 2x2 타일(R>>G>B 밝기)을 합성해, 디베이어 후 실제로 R채널>G채널>B채널 순서로
    나오는지 확인 — OpenCV Bayer 코드명 오프셋(§ 모듈 docstring "흔한 실수 지점")을
    거꾸로 매핑했다면 이 테스트가 채널 순서를 뒤집어 잡아낸다."""
    h, w = 32, 32
    bayer = np.zeros((h, w), dtype=np.uint16)
    # RGGB: (짝행,짝열)=R, (짝행,홀열)=G, (홀행,짝열)=G, (홀행,홀열)=B
    bayer[0::2, 0::2] = 900   # R
    bayer[0::2, 1::2] = 500   # G
    bayer[1::2, 0::2] = 500   # G
    bayer[1::2, 1::2] = 100   # B
    bgr = debayer_to_bgr8(bayer, pattern="rggb")
    # 가장자리 보간 아티팩트를 피해 중앙부만 본다
    center = bgr[8:-8, 8:-8, :]
    mean_b, mean_g, mean_r = (center[:, :, i].mean() for i in range(3))
    assert mean_r > mean_g > mean_b, f"R={mean_r} G={mean_g} B={mean_b} (기대: R>G>B)"
