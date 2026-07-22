"""rpi_capture.py 순수 함수(unpack_raw10/debayer_to_bgr8) 단위 테스트.

rpi_capture.py 자체는 하드웨어(v4l2-ctl/media-ctl 서브프로세스) 전용 도구라 vision/CLAUDE.md
import 규칙상 tools/의 "CI/pytest 대상 아님" 예외가 적용되지만, 그 안의 디베이어/언패킹 로직은
numpy/cv2만 쓰는 순수 함수라 하드웨어 없이 검증 가능하다(CLAUDE.md "하드웨어 비의존 부분은
.venv 설치·pytest 대상" 원칙, jsonl_view.py와 동일한 예외 패턴).

unpack_raw10()의 왕복(pack→unpack) 정확성은 실기체 캡처(2026-07-22, RPi5+IMX708 실촬영,
bytes/line 5760@width4608로 실측 확인된 MIPI RAW10 레이아웃)와 동일한 패킹 규칙을 이 테스트
안에서 재구현해(`_pack_raw10`) 대조한다 — 실기체 없이도 언패킹 로직 자체의 정확성을 보장한다.
"""
import subprocess

import numpy as np
import pytest

from vision.tools.rpi_capture import (
    _CFE_DRIVER_NAME,
    _SENSOR_ENTITY_NAME,
    _find_cfe_media_device,
    _media_ctl_driver_name,
    _media_ctl_topology_has_camera,
    apply_gray_world_white_balance,
    debayer_to_bgr8,
    unpack_raw10,
)
import vision.tools.rpi_capture as rpi_capture


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
    거꾸로 매핑했다면 이 테스트가 채널 순서를 뒤집어 잡아낸다.

    white_balance=False로 명시: 이 테스트는 순수 디베이어(채널 매핑) 검증이 목적이라, gray-world
    화이트밸런스(채널 평균을 서로 맞추는 보정, 기본 켜짐)가 끼어들면 의도적으로 만든 채널 간
    밝기 차이 자체가 지워져 버려 검증 대상이 사라진다."""
    h, w = 32, 32
    bayer = np.zeros((h, w), dtype=np.uint16)
    # RGGB: (짝행,짝열)=R, (짝행,홀열)=G, (홀행,짝열)=G, (홀행,홀열)=B
    bayer[0::2, 0::2] = 900   # R
    bayer[0::2, 1::2] = 500   # G
    bayer[1::2, 0::2] = 500   # G
    bayer[1::2, 1::2] = 100   # B
    bgr = debayer_to_bgr8(bayer, pattern="rggb", white_balance=False)
    # 가장자리 보간 아티팩트를 피해 중앙부만 본다
    center = bgr[8:-8, 8:-8, :]
    mean_b, mean_g, mean_r = (center[:, :, i].mean() for i in range(3))
    assert mean_r > mean_g > mean_b, f"R={mean_r} G={mean_g} B={mean_b} (기대: R>G>B)"


# ---------- apply_gray_world_white_balance ----------

def _channel_means(bgr: np.ndarray) -> tuple:
    return tuple(float(bgr[:, :, i].mean()) for i in range(3))


def test_apply_gray_world_white_balance_neutralizes_synthetic_green_bias():
    """의도적으로 강한 초록 편향(rpi_capture.py raw 경로에서 실제로 관측된 것과 같은 종류)을
    준 합성 이미지 -> 보정 후 채널 평균 간 격차가 보정 전보다 뚜렷이 줄어드는지 확인."""
    h, w = 32, 32
    biased = np.empty((h, w, 3), dtype=np.uint8)
    biased[:, :, 0] = 40   # B
    biased[:, :, 1] = 150  # G (강한 초록 편향)
    biased[:, :, 2] = 45   # R

    before_b, before_g, before_r = _channel_means(biased)
    before_spread = max(before_b, before_g, before_r) - min(before_b, before_g, before_r)

    corrected = apply_gray_world_white_balance(biased)
    after_b, after_g, after_r = _channel_means(corrected)
    after_spread = max(after_b, after_g, after_r) - min(after_b, after_g, after_r)

    assert after_spread < before_spread * 0.15, (
        f"보정 후 채널 격차가 충분히 줄지 않음: before={before_spread:.1f} after={after_spread:.1f}"
    )
    # gray-world 정의상 보정 후 세 채널 평균은 서로 거의 같아야 한다(±반올림 오차)
    assert max(after_b, after_g, after_r) - min(after_b, after_g, after_r) <= 2.0


def test_apply_gray_world_white_balance_already_neutral_is_near_noop():
    """이미 채널 평균이 같은(무채색 편향 없는) 이미지는 보정해도 크게 달라지지 않아야 한다."""
    h, w = 16, 16
    neutral = np.full((h, w, 3), 128, dtype=np.uint8)
    corrected = apply_gray_world_white_balance(neutral)
    np.testing.assert_allclose(corrected.astype(np.float64), neutral.astype(np.float64), atol=1.0)


def test_apply_gray_world_white_balance_shape_dtype_preserved():
    bgr = np.full((10, 12, 3), 90, dtype=np.uint8)
    corrected = apply_gray_world_white_balance(bgr)
    assert corrected.shape == bgr.shape
    assert corrected.dtype == np.uint8


def test_apply_gray_world_white_balance_black_image_returns_unchanged():
    """채널 평균이 0인 완전 검은 이미지는 0으로 나누는 걸 피해 원본 그대로 반환해야 한다."""
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    corrected = apply_gray_world_white_balance(black)
    np.testing.assert_array_equal(corrected, black)


def test_apply_gray_world_white_balance_rejects_wrong_shape():
    with pytest.raises(ValueError, match="BGR"):
        apply_gray_world_white_balance(np.zeros((8, 8), dtype=np.uint8))


def test_apply_gray_world_white_balance_rejects_wrong_dtype():
    with pytest.raises(ValueError, match="BGR"):
        apply_gray_world_white_balance(np.zeros((8, 8, 3), dtype=np.float32))


# ---------- debayer_to_bgr8 white_balance 옵션 ----------

def _biased_rggb_bayer(h: int, w: int) -> np.ndarray:
    """디베이어 결과가 강한 초록 편향을 갖도록 만든 합성 RGGB 베이어 평면(10비트 컨테이너)."""
    bayer = np.zeros((h, w), dtype=np.uint16)
    bayer[0::2, 0::2] = 160  # R
    bayer[0::2, 1::2] = 600  # G
    bayer[1::2, 0::2] = 600  # G
    bayer[1::2, 1::2] = 170  # B
    return bayer


def test_debayer_to_bgr8_white_balance_default_is_true():
    bayer = _biased_rggb_bayer(32, 32)
    default_out = debayer_to_bgr8(bayer, pattern="rggb")
    explicit_out = debayer_to_bgr8(bayer, pattern="rggb", white_balance=True)
    np.testing.assert_array_equal(default_out, explicit_out)


def test_debayer_to_bgr8_white_balance_reduces_channel_spread_vs_disabled():
    bayer = _biased_rggb_bayer(32, 32)
    without_wb = debayer_to_bgr8(bayer, pattern="rggb", white_balance=False)
    with_wb = debayer_to_bgr8(bayer, pattern="rggb", white_balance=True)

    center_without = without_wb[8:-8, 8:-8, :]
    center_with = with_wb[8:-8, 8:-8, :]

    b0, g0, r0 = _channel_means(center_without)
    spread_without = max(b0, g0, r0) - min(b0, g0, r0)
    b1, g1, r1 = _channel_means(center_with)
    spread_with = max(b1, g1, r1) - min(b1, g1, r1)

    assert spread_with < spread_without, (
        f"white_balance=True가 채널 격차를 줄이지 못함: "
        f"without={spread_without:.1f} with={spread_with:.1f}"
    )


# ---------- 미디어 디바이스 동적 탐색 (2026-07-22d) ----------
#
# 아래 3개 픽스처 텍스트는 실기체(RPi5, 2026-07-22d)에서 `media-ctl -d <dev> -p`로 실제로
# 받은 출력을 그대로(핵심 라인만 남기고) 옮긴 것 — 판별 로직이 진짜 media-ctl 출력 포맷과
# 어긋나지 않게 하기 위해 지어낸 텍스트가 아니라 실측 텍스트를 픽스처로 쓴다.

# 카메라가 실제로 연결된 CSI 포트(당시 /dev/media3) — imx708 센서 엔티티가 있다.
_MEDIA_CTL_OUTPUT_CONNECTED_RP1_CFE = """\
Media controller API version 6.8.12

Media device information
------------------------
driver          rp1-cfe
model           rp1-cfe
serial
bus info        platform:1f00128000.csi
hw revision     0x114666
driver version  6.8.12

Device topology
- entity 1: csi2 (8 pads, 9 links, 0 routes)
            type V4L2 subdev subtype Unknown flags 0
            device node name /dev/v4l-subdev0
	pad0: Sink
		[stream:0 fmt:SRGGB10_1X10/4608x2592 field:none]
		<- "imx708":0 [ENABLED,IMMUTABLE]

- entity 16: imx708 (2 pads, 2 links, 0 routes)
             type V4L2 subdev subtype Sensor flags 0
             device node name /dev/v4l-subdev2
	pad0: Source
		[stream:0 fmt:SRGGB10_1X10/4608x2592 field:none]
		-> "csi2":0 [ENABLED,IMMUTABLE]

- entity 21: rp1-cfe-csi2_ch0 (1 pad, 1 link)
             type Node subtype V4L flags 0
             device node name /dev/video0
	pad0: Sink
		<- "csi2":4 [ENABLED]
"""

# 카메라가 연결 안 된 다른 CSI 포트(당시 /dev/media2) — driver는 같은 rp1-cfe이지만
# 센서 엔티티가 아예 없고 링크도 0이다. driver명만으로 판별하면 이 디바이스를 오탐한다.
_MEDIA_CTL_OUTPUT_UNCONNECTED_RP1_CFE = """\
Media controller API version 6.8.12

Media device information
------------------------
driver          rp1-cfe
model           rp1-cfe
serial
bus info        platform:1f00110000.csi
hw revision     0x114666
driver version  6.8.12

Device topology
- entity 1: csi2 (8 pads, 0 link, 0 routes)
            type V4L2 subdev subtype Unknown flags 0
	pad0: Sink
	pad1: Sink
	pad2: Sink
	pad3: Sink
	pad4: Source
	pad5: Source
	pad6: Source
	pad7: Source

- entity 10: pisp-fe (5 pads, 0 link, 0 routes)
             type V4L2 subdev subtype Unknown flags 0
	pad0: Sink
	pad1: Sink
	pad2: Source
	pad3: Source
	pad4: Source
"""

# 카메라 캡처와 무관한 ISP 백엔드(당시 /dev/media1) — 완전히 다른 driver.
# 이전 세션(2026-07-22b)까지는 이 위치가 rp1-cfe였다가 이번 조사(2026-07-22d)에 pispbe로
# 바뀌어 있었던 것 자체가 이번 세션의 버그 재현 계기였다.
_MEDIA_CTL_OUTPUT_PISPBE = """\
Media controller API version 6.8.12

Media device information
------------------------
driver          pispbe
model           pispbe
serial
bus info        platform:1000880000.pisp_be
hw revision     0x2252701
driver version  6.8.12

Device topology
- entity 1: pispbe (9 pads, 9 links, 0 routes)
            type V4L2 subdev subtype Unknown flags 0
	pad0: Sink
		<- "pispbe-input":0 [ENABLED,IMMUTABLE]
"""


def test_media_ctl_topology_has_camera_true_for_connected_instance():
    assert _media_ctl_topology_has_camera(_MEDIA_CTL_OUTPUT_CONNECTED_RP1_CFE) is True


def test_media_ctl_topology_has_camera_false_for_unconnected_same_driver_instance():
    """driver는 일치(rp1-cfe)해도 센서 엔티티가 없으면 오탐하면 안 된다 — 이번 버그의 핵심
    함정(driver명만으로 판별했다면 media2/media3 둘 다 걸려 어느 쪽이 진짜인지 못 가린다)."""
    assert _media_ctl_topology_has_camera(_MEDIA_CTL_OUTPUT_UNCONNECTED_RP1_CFE) is False


def test_media_ctl_topology_has_camera_false_for_different_driver():
    assert _media_ctl_topology_has_camera(_MEDIA_CTL_OUTPUT_PISPBE) is False


def test_media_ctl_topology_has_camera_false_for_empty_output():
    assert _media_ctl_topology_has_camera("") is False


def test_media_ctl_topology_has_camera_respects_custom_driver_and_entity_args():
    """driver/sensor_entity를 다른 값으로 줬을 때도 그 기준으로 판별하는지(하드코딩 아님)."""
    assert _media_ctl_topology_has_camera(
        _MEDIA_CTL_OUTPUT_PISPBE, driver="pispbe", sensor_entity="pispbe-input"
    ) is False  # pispbe-input은 엔티티가 아니라 링크 대상일 뿐, "- entity N: pispbe-input" 없음
    assert _media_ctl_topology_has_camera(
        _MEDIA_CTL_OUTPUT_PISPBE, driver="pispbe", sensor_entity="pispbe"
    ) is True  # entity 1: pispbe 자체는 있음


def test_media_ctl_driver_name_extracts_rp1_cfe():
    assert _media_ctl_driver_name(_MEDIA_CTL_OUTPUT_CONNECTED_RP1_CFE) == "rp1-cfe"


def test_media_ctl_driver_name_extracts_pispbe():
    assert _media_ctl_driver_name(_MEDIA_CTL_OUTPUT_PISPBE) == "pispbe"


def test_media_ctl_driver_name_unknown_when_driver_line_missing():
    assert _media_ctl_driver_name("garbage output with no driver line") == "?"


def _make_fake_run(outputs_by_device: dict, errors_for: tuple = ()):
    """`_run()`을 몽키패치할 가짜 구현. `cmd`에서 "-d" 다음 인자를 device 경로로 보고
    `outputs_by_device`에서 찾아 CompletedProcess.stdout으로 돌려준다. `errors_for`에 든
    device는 실제 media-ctl 실패를 흉내내 CalledProcessError를 던진다."""

    def _fake_run(cmd, **kw):
        device = cmd[cmd.index("-d") + 1]
        if device in errors_for:
            raise subprocess.CalledProcessError(1, cmd, output="", stderr="no such device")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=outputs_by_device[device])

    return _fake_run


def test_find_cfe_media_device_picks_the_connected_one_among_several(monkeypatch):
    monkeypatch.setattr(
        rpi_capture, "_iter_media_device_paths",
        lambda: ["/dev/media1", "/dev/media2", "/dev/media3"],
    )
    monkeypatch.setattr(rpi_capture, "_run", _make_fake_run({
        "/dev/media1": _MEDIA_CTL_OUTPUT_PISPBE,
        "/dev/media2": _MEDIA_CTL_OUTPUT_UNCONNECTED_RP1_CFE,
        "/dev/media3": _MEDIA_CTL_OUTPUT_CONNECTED_RP1_CFE,
    }))

    assert _find_cfe_media_device() == "/dev/media3"


def test_find_cfe_media_device_finds_it_regardless_of_enumeration_order(monkeypatch):
    """실기체에서 실제로 관측된 시나리오: 카메라 파이프라인이 media1이었다가 media3으로
    옮겨가도(순번이 바뀌어도) 여전히 정확히 찾아야 한다 — 이번 버그가 하드코딩 때문에
    못 했던 바로 그 일."""
    monkeypatch.setattr(
        rpi_capture, "_iter_media_device_paths",
        lambda: ["/dev/media0", "/dev/media1"],
    )
    monkeypatch.setattr(rpi_capture, "_run", _make_fake_run({
        "/dev/media0": _MEDIA_CTL_OUTPUT_UNCONNECTED_RP1_CFE,
        "/dev/media1": _MEDIA_CTL_OUTPUT_CONNECTED_RP1_CFE,  # 예전 세션엔 여기가 카메라였음
    }))

    assert _find_cfe_media_device() == "/dev/media1"


def test_find_cfe_media_device_raises_with_device_list_when_none_match(monkeypatch):
    monkeypatch.setattr(
        rpi_capture, "_iter_media_device_paths",
        lambda: ["/dev/media1", "/dev/media2"],
    )
    monkeypatch.setattr(rpi_capture, "_run", _make_fake_run({
        "/dev/media1": _MEDIA_CTL_OUTPUT_PISPBE,
        "/dev/media2": _MEDIA_CTL_OUTPUT_UNCONNECTED_RP1_CFE,
    }))

    with pytest.raises(RuntimeError) as exc_info:
        _find_cfe_media_device()

    message = str(exc_info.value)
    assert _CFE_DRIVER_NAME in message
    assert _SENSOR_ENTITY_NAME in message
    # 진단을 위해 확인한 디바이스 목록이 에러 메시지에 다 들어있어야 한다
    assert "/dev/media1" in message
    assert "/dev/media2" in message


def test_find_cfe_media_device_raises_when_no_media_devices_at_all(monkeypatch):
    monkeypatch.setattr(rpi_capture, "_iter_media_device_paths", lambda: [])

    with pytest.raises(RuntimeError, match="media\\* 디바이스가 하나도 없다"):
        _find_cfe_media_device()


def test_find_cfe_media_device_skips_device_that_errors_and_still_finds_correct_one(monkeypatch):
    """일부 media 디바이스에 대한 media-ctl 조회 자체가 실패해도(권한 문제 등) 나머지를
    계속 살펴 정답을 찾아야 한다 — 첫 실패에서 바로 죽으면 안 된다."""
    monkeypatch.setattr(
        rpi_capture, "_iter_media_device_paths",
        lambda: ["/dev/media1", "/dev/media2", "/dev/media3"],
    )
    monkeypatch.setattr(rpi_capture, "_run", _make_fake_run(
        outputs_by_device={
            "/dev/media3": _MEDIA_CTL_OUTPUT_CONNECTED_RP1_CFE,
        },
        errors_for=("/dev/media1", "/dev/media2"),
    ))

    assert _find_cfe_media_device() == "/dev/media3"
