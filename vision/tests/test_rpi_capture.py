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

import cv2

from vision.tools.rpi_capture import (
    _CFE_DRIVER_NAME,
    _FOCUS_MAX,
    _FOCUS_MIN,
    _SENSOR_ENTITY_NAME,
    _find_cfe_media_device,
    _media_ctl_driver_name,
    _media_ctl_topology_has_camera,
    apply_gray_world_white_balance,
    crop_center_fraction,
    debayer_to_bgr8,
    laplacian_sharpness,
    parse_focus_sweep_spec,
    pick_best_focus,
    set_exposure_gain,
    set_focus_absolute,
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


# ---------- laplacian_sharpness (2026-07-22e) ----------
#
# 순수 함수 — 합성 이미지(선명한 원본 vs cv2.GaussianBlur로 흐린 버전)로 검증. 실기체 검증
# (vision/CLAUDE.md "rpi_capture.py 수동 초점/노출/게인 제어" 절)에서 이 지표가 노출 확보 후
# 실제로 focus_absolute 변화에 반응함을 확인했고, 여기서는 그 지표 계산 자체의 정확성만 본다.

def _synthetic_checkerboard(size: int = 64, square: int = 8) -> np.ndarray:
    """고대비 체커보드 패턴 그레이스케일 합성 이미지 — 라플라시안 분산이 반응할 뚜렷한 에지."""
    idx = np.indices((size, size)).sum(axis=0)
    pattern = ((idx // square) % 2).astype(np.uint8) * 255
    return pattern


def test_laplacian_sharpness_blurred_is_lower_than_sharp():
    sharp_gray = _synthetic_checkerboard()
    blurred_gray = cv2.GaussianBlur(sharp_gray, (15, 15), sigmaX=5.0)
    sharp_score = laplacian_sharpness(sharp_gray)
    blurred_score = laplacian_sharpness(blurred_gray)
    assert sharp_score > blurred_score * 3, (
        f"블러 처리된 이미지가 충분히 낮은 점수를 받지 못함: sharp={sharp_score:.2f} "
        f"blurred={blurred_score:.2f}"
    )


def test_laplacian_sharpness_accepts_bgr_color_image():
    sharp_gray = _synthetic_checkerboard()
    sharp_bgr = cv2.cvtColor(sharp_gray, cv2.COLOR_GRAY2BGR)
    blurred_bgr = cv2.GaussianBlur(sharp_bgr, (15, 15), sigmaX=5.0)
    assert laplacian_sharpness(sharp_bgr) > laplacian_sharpness(blurred_bgr)


def test_laplacian_sharpness_flat_image_is_near_zero():
    flat = np.full((32, 32), 128, dtype=np.uint8)
    assert laplacian_sharpness(flat) == pytest.approx(0.0, abs=1e-6)


def test_laplacian_sharpness_returns_python_float():
    score = laplacian_sharpness(_synthetic_checkerboard())
    assert isinstance(score, float)


# ---------- crop_center_fraction ----------

def test_crop_center_fraction_shape():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cropped = crop_center_fraction(img, 0.5)
    assert cropped.shape == (50, 100, 3)


def test_crop_center_fraction_is_centered():
    """중앙 크롭이 실제로 이미지 중심에 위치하는지 — 좌상단 모서리에만 마커를 찍고 크롭 결과에
    마커가 없어야 한다(중앙만 남았다는 뜻)."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[0:5, 0:5] = 255  # 좌상단 모서리 마커
    cropped = crop_center_fraction(img, 0.5)
    assert cropped.shape == (50, 50)
    assert cropped.max() == 0, "중앙 크롭에 모서리 마커가 섞여 들어옴 — 중앙 정렬 계산 오류"


def test_crop_center_fraction_full_frame_is_noop():
    img = np.arange(24, dtype=np.uint8).reshape(4, 6)
    cropped = crop_center_fraction(img, 1.0)
    np.testing.assert_array_equal(cropped, img)


def test_crop_center_fraction_rejects_invalid_fraction():
    img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="0~1"):
        crop_center_fraction(img, 0.0)
    with pytest.raises(ValueError, match="0~1"):
        crop_center_fraction(img, 1.5)


# ---------- pick_best_focus ----------

def test_pick_best_focus_picks_highest_sharpness():
    results = [(100, 5.0), (200, 18.2), (300, 9.4)]
    assert pick_best_focus(results) == 200


def test_pick_best_focus_single_result():
    assert pick_best_focus([(480, 3.3)]) == 480


def test_pick_best_focus_ties_pick_first_occurrence():
    results = [(100, 10.0), (200, 10.0), (300, 5.0)]
    assert pick_best_focus(results) == 100


def test_pick_best_focus_rejects_empty():
    with pytest.raises(ValueError, match="빈 결과"):
        pick_best_focus([])


# ---------- parse_focus_sweep_spec ----------

def test_parse_focus_sweep_spec_basic_range():
    assert parse_focus_sweep_spec("400:700:20") == list(range(400, 701, 20))


def test_parse_focus_sweep_spec_single_value_range():
    assert parse_focus_sweep_spec("500:500:10") == [500]


def test_parse_focus_sweep_spec_rejects_wrong_arity():
    with pytest.raises(ValueError, match="start:end:step"):
        parse_focus_sweep_spec("400:700")
    with pytest.raises(ValueError, match="start:end:step"):
        parse_focus_sweep_spec("400:700:20:1")


def test_parse_focus_sweep_spec_rejects_non_integer():
    with pytest.raises(ValueError, match="정수만"):
        parse_focus_sweep_spec("abc:700:20")


def test_parse_focus_sweep_spec_rejects_non_positive_step():
    with pytest.raises(ValueError, match="step은 양수"):
        parse_focus_sweep_spec("400:700:0")
    with pytest.raises(ValueError, match="step은 양수"):
        parse_focus_sweep_spec("700:400:-20")


def test_parse_focus_sweep_spec_rejects_empty_range():
    """start > end 인데 step이 양수라 range()가 빈 리스트가 되는 경우 — 조용히 빈 스윕을
    수행하지 않고 명시적으로 실패해야 한다."""
    with pytest.raises(ValueError, match="비어있다"):
        parse_focus_sweep_spec("700:400:20")


# ---------- set_focus_absolute (하드웨어 호출은 _run 몽키패치로 격리) ----------

def test_set_focus_absolute_rejects_out_of_range(monkeypatch):
    calls = []
    monkeypatch.setattr(rpi_capture, "_run", lambda cmd, **kw: calls.append(cmd))
    with pytest.raises(ValueError, match=f"{_FOCUS_MIN}~{_FOCUS_MAX}"):
        set_focus_absolute(_FOCUS_MAX + 1, settle_s=0)
    with pytest.raises(ValueError, match=f"{_FOCUS_MIN}~{_FOCUS_MAX}"):
        set_focus_absolute(_FOCUS_MIN - 1, settle_s=0)
    assert calls == [], "범위 밖 값은 v4l2-ctl 호출 전에 걸러져야 한다"


def test_set_focus_absolute_calls_v4l2_ctl_with_correct_control(monkeypatch):
    calls = []
    monkeypatch.setattr(rpi_capture, "_run", lambda cmd, **kw: calls.append(cmd))
    set_focus_absolute(560, settle_s=0)
    assert calls == [[rpi_capture._V4L2_CTL, "-d", rpi_capture._LENS_SUBDEV,
                       "-c", "focus_absolute=560"]]


def test_set_focus_absolute_settles_for_requested_duration(monkeypatch):
    monkeypatch.setattr(rpi_capture, "_run", lambda cmd, **kw: None)
    slept = []
    monkeypatch.setattr(rpi_capture.time, "sleep", lambda s: slept.append(s))
    set_focus_absolute(300, settle_s=0.2)
    assert slept == [0.2]


def test_set_focus_absolute_skips_sleep_when_settle_zero(monkeypatch):
    monkeypatch.setattr(rpi_capture, "_run", lambda cmd, **kw: None)
    slept = []
    monkeypatch.setattr(rpi_capture.time, "sleep", lambda s: slept.append(s))
    set_focus_absolute(300, settle_s=0)
    assert slept == []


# ---------- set_exposure_gain ----------

def test_set_exposure_gain_noop_when_both_none(monkeypatch):
    calls = []
    monkeypatch.setattr(rpi_capture, "_run", lambda cmd, **kw: calls.append(cmd))
    set_exposure_gain()
    assert calls == []


def test_set_exposure_gain_both_values_combined_in_one_call(monkeypatch):
    calls = []
    monkeypatch.setattr(rpi_capture, "_run", lambda cmd, **kw: calls.append(cmd))
    set_exposure_gain(exposure=2400, gain=800)
    assert calls == [[rpi_capture._V4L2_CTL, "-d", rpi_capture._SENSOR_SUBDEV,
                       "-c", "exposure=2400,analogue_gain=800"]]


def test_set_exposure_gain_exposure_only(monkeypatch):
    calls = []
    monkeypatch.setattr(rpi_capture, "_run", lambda cmd, **kw: calls.append(cmd))
    set_exposure_gain(exposure=2400)
    assert calls == [[rpi_capture._V4L2_CTL, "-d", rpi_capture._SENSOR_SUBDEV,
                       "-c", "exposure=2400"]]


def test_set_exposure_gain_gain_only(monkeypatch):
    calls = []
    monkeypatch.setattr(rpi_capture, "_run", lambda cmd, **kw: calls.append(cmd))
    set_exposure_gain(gain=800)
    assert calls == [[rpi_capture._V4L2_CTL, "-d", rpi_capture._SENSOR_SUBDEV,
                       "-c", "analogue_gain=800"]]


# ---------- _CaptureSession 초점/노출/게인 지연 적용 로직 ----------
#
# 실제 v4l2-ctl/캡처는 하지 않는다 — set_focus_absolute/set_exposure_gain/capture_frame_bgr을
# 모듈 레벨에서 몽키패치해 "값이 바뀐 경우에만 하드웨어에 적용" 로직 자체만 검증한다.

def _make_session(tmp_path, **kw):
    return rpi_capture._CaptureSession(
        out_dir=tmp_path, preview_size=(320, 240), main_size=(640, 480), **kw)


def test_capture_session_current_controls_defaults_when_unset(tmp_path):
    session = _make_session(tmp_path)
    assert session.current_controls() == (
        rpi_capture._FOCUS_DEFAULT, rpi_capture._EXPOSURE_DEFAULT, rpi_capture._GAIN_DEFAULT)


def test_capture_session_current_controls_reflects_constructor_values(tmp_path):
    session = _make_session(tmp_path, focus=560, exposure=2400, gain=800)
    assert session.current_controls() == (560, 2400, 800)


def test_capture_session_set_controls_updates_only_given_fields(tmp_path):
    session = _make_session(tmp_path, focus=100, exposure=500, gain=200)
    session.set_controls(focus=560)
    assert session.current_controls() == (560, 500, 200)


def test_capture_session_set_controls_rejects_out_of_range_focus(tmp_path):
    session = _make_session(tmp_path)
    with pytest.raises(ValueError, match=f"{_FOCUS_MIN}~{_FOCUS_MAX}"):
        session.set_controls(focus=_FOCUS_MAX + 1)


def test_capture_session_apply_pending_controls_calls_hardware_once_per_change(tmp_path, monkeypatch):
    focus_calls = []
    exposure_calls = []
    monkeypatch.setattr(rpi_capture, "set_focus_absolute",
                         lambda v, settle_s=0.0: focus_calls.append((v, settle_s)))
    monkeypatch.setattr(rpi_capture, "set_exposure_gain",
                         lambda exposure=None, gain=None: exposure_calls.append((exposure, gain)))

    session = _make_session(tmp_path, focus=560, exposure=2400, gain=800, focus_settle_s=0.2)
    with session._lock:
        session._apply_pending_controls_locked()
    assert focus_calls == [(560, 0.2)]
    assert exposure_calls == [(2400, 800)]

    # 값이 그대로면 두 번째 호출에서는 하드웨어를 다시 부르면 안 된다
    with session._lock:
        session._apply_pending_controls_locked()
    assert focus_calls == [(560, 0.2)], "값이 안 바뀌었는데 focus를 다시 적용함"
    assert exposure_calls == [(2400, 800)], "값이 안 바뀌었는데 exposure/gain을 다시 적용함"

    # focus만 바뀌면 focus만 다시 적용되고 exposure/gain은 재적용 안 됨
    session.set_controls(focus=300)
    with session._lock:
        session._apply_pending_controls_locked()
    assert focus_calls == [(560, 0.2), (300, 0.2)]
    assert exposure_calls == [(2400, 800)]


def test_capture_session_apply_pending_controls_noop_when_nothing_set(tmp_path, monkeypatch):
    focus_calls = []
    exposure_calls = []
    monkeypatch.setattr(rpi_capture, "set_focus_absolute", lambda v, settle_s=0.0: focus_calls.append(v))
    monkeypatch.setattr(rpi_capture, "set_exposure_gain",
                         lambda exposure=None, gain=None: exposure_calls.append((exposure, gain)))

    session = _make_session(tmp_path)
    with session._lock:
        session._apply_pending_controls_locked()
    assert focus_calls == []
    assert exposure_calls == []


# ---------- _render_page ----------

def test_render_page_embeds_current_values():
    html = rpi_capture._render_page(focus=560, exposure=2400, gain=800)
    assert 'value="560"' in html
    assert 'value="2400"' in html
    assert 'value="800"' in html


def test_render_page_focus_quick_buttons_clamped_to_range():
    html_low = rpi_capture._render_page(focus=_FOCUS_MIN, exposure=1, gain=1)
    assert f"/controls?focus={_FOCUS_MIN}" in html_low  # -10이 하한 밑으로 안 내려감
    html_high = rpi_capture._render_page(focus=_FOCUS_MAX, exposure=1, gain=1)
    assert f"/controls?focus={_FOCUS_MAX}" in html_high  # +10이 상한 위로 안 올라감
