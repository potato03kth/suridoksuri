"""
RPi5(Ubuntu, rp1-cfe V4L2) 헤드리스 캘리브레이션 촬영 도구 — V4L2 RAW 직접 캡처 + 수동 디베이어.

이 RPi는 libcamera가 PiSP IPA 모듈(ipa_rpi_pisp.so) 없이 빌드돼 있어 libcamera 상위 레벨
(picamera2/GStreamer libcamerasrc 포함)이 전부 막혀 있다(메모리 project_rpi5_ubuntu_camera_stack.md
참고). 커널 V4L2/media-controller 레벨은 정상이므로 libcamera를 완전히 우회해 V4L2 RAW를 직접
캡처하고 베이어를 수동으로 디베이어한다.

장착 카메라: 서드파티 클론 CAM109-IMX708AF-75 (IMX708 센서). 커널에 다음으로 잡힌다
(실기체 media-ctl -p 로 확인, 2026-07-22):
    /dev/video0        = rp1-cfe-csi2_ch0 (캡처 노드)
    /dev/v4l-subdev0   = csi2 (CSI-2 수신기)
    /dev/v4l-subdev2   = imx708 (센서)
픽셀포맷: 'pRAA' (V4L2_PIX_FMT_SRGGB10P, MIPI RAW10 패킹) — v4l2-ctl -d /dev/video0
--list-formats-ext 로 확인. imx708 소스 패드 mbus code는 SRGGB10_1X10 → 베이어 패턴 RGGB.

**미디어 디바이스 번호(`/dev/mediaN`)는 고정이 아니다 — 매 실행 시 동적으로 찾는다.**
2026-07-22 브링업 세션 때는 카메라 파이프라인(driver `rp1-cfe`)이 `/dev/media1`이었는데,
2026-07-22d에 같은 RPi에서 재확인하니 `/dev/media1`은 `pispbe`(ISP 백엔드, 이 카메라 캡처
경로에 안 쓰임)로 바뀌어 있었고 카메라 파이프라인은 `/dev/media3`으로 옮겨가 있었다. 이 RPi는
`rp1-cfe` 드라이버 인스턴스가 CSI 포트마다(cam0/cam1) 하나씩 총 2개 등록되는데, 실제로 카메라가
연결된 포트만 센서(`imx708`) 엔티티를 갖고 나머지 하나는 빈 상태(`csi2 (8 pads, 0 link)`, 센서
엔티티 자체가 없음)로 잡힌다 — 그리고 `pispbe`도 2개 등록된다(`/dev/media0`,`/dev/media1`). 리눅스
미디어 디바이스 번호는 드라이버가 프로브(등록)되는 순서대로 부여되는데, 이 순서는 커널 모듈
로드 순서·부팅마다 달라질 수 있어(실측으로 재부팅/재연결 사이 실제로 바뀐 것을 확인, 근본
메커니즘까지는 규명하지 않음) 특정 번호를 하드코딩할 수 없다. 그래서 `_find_cfe_media_device()`가
매 `configure_pipeline()` 호출마다 `/dev/media*`를 순회해 driver가 `rp1-cfe`이고 `imx708` 센서
엔티티를 실제로 가진 디바이스를 찾는다(상세는 해당 함수 docstring, 단위테스트는
`vision/tests/test_rpi_capture.py`). `/dev/video0`/`/dev/v4l-subdev0`/`/dev/v4l-subdev2`(캡처
노드/csi2/센서 자체의 번호)는 이번 조사에서는 안정적으로 재현됐고 이번 세션 범위 밖이라 그대로
하드코딩 유지 — 향후 이것도 흔들리는 게 관측되면 같은 방식으로 동적 탐색 확장을 검토한다.

## media-controller 파이프라인이 필요한 이유 (2026-07-22 실기체로 확정)

/dev/video0는 "MC-centric" 캡처 노드라 단순히 --set-fmt-video만으로는 VIDIOC_STREAMON이 안
된다. 아래 3가지를 media-ctl/v4l2-ctl로 명시적으로 맞춰야 media_pipeline_start()가 통과한다
(각각 실제로 막혔던 지점 — dmesg dynamic_debug로 확인):

  1. **링크 활성화** — `"csi2":4 -> "rp1-cfe-csi2_ch0":0` 링크가 기본 비활성 상태
     (커널 로그: "csi2_ch0 node link is not enabled."). media-ctl -l 로 활성화해야 한다.
  2. **필드(field) 일치** — imx708 소스 패드는 field=None인데 csi2 싱크 패드0 기본값은
     field=Any. `v4l2_subdev_link_validate_default`가 이 둘을 다르다고 보고 링크 검증을
     거부한다(커널 로그: "field does not match", VIDIOC_STREAMON이 -EPIPE/-32로 실패).
     media-ctl -V 로 csi2 패드0도 명시적으로 field:none 으로 맞춰야 한다.
  3. **임베디드 메타데이터 패드 폭 일치** — imx708 패드1(임베디드 데이터, IMMUTABLE 링크라
     항상 파이프라인 그래프에 포함됨)의 실제 폭은 28800인데 csi2 싱크 패드1 기본값은 16384
     (커널 로그: "width does not match", 마찬가지로 -EPIPE). imx708 패드1의 *실제* 폭을
     읽어(하드코딩 금지 — 모드에 따라 달라질 수 있음) csi2 싱크 패드1에 그대로 맞춰야 한다.
     media-ctl -V는 SENSOR_DATA 심볼릭 코드명을 못 받아들여 실패하므로 v4l2-ctl
     --set-subdev-fmt로 mbus code 0x7002(MEDIA_BUS_FMT_SENSOR_DATA)를 숫자로 직접 지정한다.

이 셋을 다 맞추기 전엔 VIDIOC_STREAMON이 "Broken pipe"(-EPIPE)로 실패한다 — 재현 명령·근본
원인 전체 경과는 `docs/vision_status.md` 2026-07-22 항목 참고.

## 센서 모드 스냅

imx708는 임의 해상도를 그대로 받지 않고 가장 가까운 지원 모드로 스냅한다(실기체 확인:
640x480 요청 시 1536x864로 스냅됨). 그래서 `configure_pipeline()`은 센서에 설정을 요청한
직후 실제 적용된 크기를 다시 읽어(readback) 그 값을 나머지 파이프라인(csi2 싱크/소스 패드,
video0 포맷)에 그대로 전파한다 — 요청값을 그대로 믿지 않는다.

RPi에서 실행 (HTTP 미리보기+촬영 서버):
    python3 vision/tools/rpi_capture.py --out vision/data/calibration_raw
노트북에서: http://<rpi-ip>:8000/ 열어서 위치 잡고 버튼으로 촬영, 또는 SSH 터미널에서 Enter.

RPi에서 논-인터랙티브 단발 캡처(스크립트/원격 검증용):
    python3 vision/tools/rpi_capture.py --single-shot --out vision/data/calibration_raw
"""
import argparse
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # RPi에는 항상 있음. 순수 unpack_raw10()만 쓰는 개발 호스트는 cv2 없이도 import 가능.
    cv2 = None

_V4L2_CTL = "v4l2-ctl"
_MEDIA_CTL = "media-ctl"

_VIDEO_DEVICE = "/dev/video0"
_CSI2_SUBDEV = "/dev/v4l-subdev0"
_SENSOR_SUBDEV = "/dev/v4l-subdev2"
_LENS_SUBDEV = "/dev/v4l-subdev3"  # dw9807 VCM(오토포커스 렌즈 드라이버) — 실기체 확인, media 항목 19

# 초점/노출/게인 수동 제어 (2026-07-22e) — libcamera 없이는 연속 AF/AE가 전혀 동작하지 않아
# (§ 모듈 docstring 하단 "수동 초점/노출/게인 제어" 절 참고) focus_absolute가 한 번도 안 움직인
# 채, exposure/analogue_gain도 최솟값 근처에 방치된 채로 모든 캡처가 이뤄지고 있었다(체커보드
# 캘리브레이션 촬영이 초점 안 맞고 과다 어둡게 나온 원인). 아래는 v4l2-ctl --list-ctrls-menus로
# 확인한 각 컨트롤의 실제 범위/기본값.
_FOCUS_MIN, _FOCUS_MAX = 0, 1023      # dw9807, 렌즈 드라이버 고유 범위 — 센서 모드와 무관, 고정
_FOCUS_DEFAULT = 480                  # imx708 센서모듈 기본값(실측)
_EXPOSURE_DEFAULT = 874               # /dev/v4l-subdev2 exposure 기본값(실측)
_GAIN_DEFAULT = 112                   # /dev/v4l-subdev2 analogue_gain 기본값(실측, 곧 최솟값)
# exposure/analogue_gain은 vertical_blanking(모드별 프레임 길이)에 따라 최댓값이 달라질 수 있어
# focus처럼 고정 범위로 하드코딩하지 않는다 — set_exposure_gain()은 값을 그대로 v4l2-ctl에 전달하고
# 범위 밖이면 v4l2-ctl 자체가 CalledProcessError로 거부하도록 둔다(근거는 해당 함수 docstring).

# VCM 물리 이동 정착시간(settle) 기본값 — 실기체 실측(2026-07-22e)으로 확정한 값.
# 실측 방법·수치: vision/CLAUDE.md "rpi_capture.py 수동 초점/노출/게인 제어" 절 참조.
FOCUS_SETTLE_S_DEFAULT = 0.2

# 카메라 파이프라인을 가진 media 디바이스를 판별하는 기준 — 실기체 조사(2026-07-22d)로 확정.
# rp1-cfe 드라이버 인스턴스가 CSI 포트마다(cam0/cam1) 하나씩 잡히는데, 실제로 카메라가 연결된
# 포트만 이 센서 엔티티를 갖는다(연결 안 된 포트는 driver는 같지만 엔티티 목록이 비어 있다).
_CFE_DRIVER_NAME = "rp1-cfe"
_SENSOR_ENTITY_NAME = "imx708"

_EMBEDDED_DATA_CODE = 0x7002  # MEDIA_BUS_FMT_SENSOR_DATA
_BAYER_MBUS_CODE = "SRGGB10_1X10"
_PIXELFORMAT = "pRAA"  # V4L2_PIX_FMT_SRGGB10P (10-bit Bayer RGRG/GBGB Packed)

# 센서가 보고하는 베이어 패턴 -> OpenCV cvtColor 코드.
# OpenCV의 Bayer 코드명은 픽셀(0,0)이 아니라 (1,1) 기준으로 패턴을 기술하는 관례라, 센서가
# "RGGB"로 보고해도 OpenCV 코드로는 한 칸 밀린 "BG"가 대응된다(IMX219/IMX708류 라즈베리파이
# raw 캡처에서 흔히 나오는 실수 지점 — 여기서 표로 명시적으로 고정해 둔다).
_BAYER_CV_CODE = {
    "rggb": "COLOR_BayerBG2BGR",
    "bggr": "COLOR_BayerRG2BGR",
    "grbg": "COLOR_BayerGB2BGR",
    "gbrg": "COLOR_BayerGR2BGR",
}


def unpack_raw10(packed: np.ndarray, width: int, height: int) -> np.ndarray:
    """MIPI RAW10 패킹된 바이트 배열(1D uint8) -> (height, width) uint16 배열(값 범위 0~1023).

    4픽셀이 5바이트로 패킹된다(V4L2_PIX_FMT_SRGGB10P/'pRAA'): 앞 4바이트는 각 픽셀의 상위
    8비트, 5번째 바이트는 4픽셀의 하위 2비트씩을 모아 담는다(픽셀0이 최하위 2비트).
    width는 4의 배수여야 한다 — 한 줄 바이트수 = width/4*5 (실기체 4608폭에서 5760바이트/줄로
    실측 확인, video0 --get-fmt-video의 Bytes per Line과 정확히 일치).
    """
    if width % 4 != 0:
        raise ValueError(f"unpack_raw10: width는 4의 배수여야 한다 (width={width})")
    bytes_per_row = width // 4 * 5
    expected = bytes_per_row * height
    if packed.size != expected:
        raise ValueError(
            f"unpack_raw10: 크기 불일치 — 입력 {packed.size}바이트, "
            f"기대값 {expected}바이트 (width={width}, height={height}, "
            f"bytes_per_row={bytes_per_row})"
        )
    rows = packed.reshape(height, bytes_per_row)
    b0 = rows[:, 0::5].astype(np.uint16)
    b1 = rows[:, 1::5].astype(np.uint16)
    b2 = rows[:, 2::5].astype(np.uint16)
    b3 = rows[:, 3::5].astype(np.uint16)
    b4 = rows[:, 4::5].astype(np.uint16)
    p0 = (b0 << 2) | (b4 & 0x03)
    p1 = (b1 << 2) | ((b4 >> 2) & 0x03)
    p2 = (b2 << 2) | ((b4 >> 4) & 0x03)
    p3 = (b3 << 2) | ((b4 >> 6) & 0x03)
    out = np.empty((height, width), dtype=np.uint16)
    out[:, 0::4] = p0
    out[:, 1::4] = p1
    out[:, 2::4] = p2
    out[:, 3::4] = p3
    return out


def apply_gray_world_white_balance(bgr8: np.ndarray) -> np.ndarray:
    """Gray-world 가정 기반 화이트밸런스 보정: BGR uint8 이미지 -> BGR uint8 이미지.

    "전체 이미지의 R/G/B 채널 평균이 같아야 한다"는 gray-world 가정에 따라, 채널별 평균의
    평균(회색 기준값)에 각 채널 평균이 맞춰지도록 채널별 게인(gray/mean_c)을 곱한다. 픽셀별
    보정이 아니라 프레임 전체 통계 1개(채널당 평균)만 쓰는 가장 단순한 변형 — 체커보드 등
    특정 기준 패치나 조명 스펙트럼 추정 없이 numpy만으로 계산 가능해 이 raw 캡처 후처리
    단계에 적합하다고 판단(근거는 vision/CLAUDE.md 참조). 채널 평균이 0인 완전 검은 이미지는
    0으로 나누는 것을 피하기 위해 원본을 그대로 반환한다.
    """
    if bgr8.dtype != np.uint8 or bgr8.ndim != 3 or bgr8.shape[2] != 3:
        raise ValueError(
            f"apply_gray_world_white_balance: (H,W,3) uint8 BGR 배열이 필요하다 "
            f"(입력 shape={bgr8.shape}, dtype={bgr8.dtype})"
        )
    bgr = bgr8.astype(np.float64)
    channel_means = bgr.reshape(-1, 3).mean(axis=0)  # [mean_b, mean_g, mean_r]
    if np.any(channel_means == 0):
        return bgr8.copy()
    gray = channel_means.mean()
    gains = gray / channel_means
    out = bgr * gains  # 브로드캐스트: 마지막 축(채널)별로 게인 적용
    return np.clip(out, 0, 255).astype(np.uint8)


def debayer_to_bgr8(bayer10: np.ndarray, pattern: str = "rggb", white_balance: bool = True) -> np.ndarray:
    """unpack_raw10() 출력(10비트 값을 16비트 컨테이너에 담은 베이어 평면) -> 8비트 BGR.

    cv2.cvtColor의 Bayer 디모자이킹은 8비트 입력을 기대하므로, 먼저 10비트(0~1023)를
    8비트(0~255)로 다운시프트(>>2)한 뒤 디베이어한다. 캘리브레이션 정밀도가 더 필요하면
    unpack_raw10()의 16비트(실질 10비트) 출력을 직접 쓰는 것도 가능 — 이 함수는 미리보기/
    시각 확인/체커보드 코너검출(8비트 입력 요구)용.

    white_balance=True(기본값)면 디베이어 직후 apply_gray_world_white_balance()로 화이트밸런스를
    보정한다 — raw 베이어 경로는 ISP(libcamera)를 완전히 우회하므로 화이트밸런스가 전혀 적용되지
    않은 채 강한 색편향(이 카메라는 초록 편향)이 나타난다(근거는 vision/CLAUDE.md 참조).
    """
    if cv2 is None:
        raise RuntimeError("debayer_to_bgr8: cv2(OpenCV)가 설치돼 있지 않다")
    pattern = pattern.lower()
    if pattern not in _BAYER_CV_CODE:
        raise ValueError(
            f"debayer_to_bgr8: 알 수 없는 베이어 패턴 {pattern!r} "
            f"(지원: {sorted(_BAYER_CV_CODE)})"
        )
    bayer8 = (bayer10 >> 2).astype(np.uint8)
    cv_code = getattr(cv2, _BAYER_CV_CODE[pattern])
    bgr = cv2.cvtColor(bayer8, cv_code)
    if white_balance:
        bgr = apply_gray_world_white_balance(bgr)
    return bgr


def laplacian_sharpness(bgr: np.ndarray) -> float:
    """선명도 지표: 그레이스케일 변환 후 라플라시안 분산(값이 클수록 선명). 순수 함수 — cv2만
    사용, 하드웨어 없이 합성 이미지(선명한 원본 vs cv2.GaussianBlur로 흐린 버전)로 단위테스트
    가능(`vision/tests/test_rpi_capture.py`).

    **신뢰도 전제조건(2026-07-22e 실기체 검증으로 확정):** 이 지표는 장면이 충분히 밝고 대비가
    있어야 의미 있게 움직인다 — 어둡고 대비 낮은 원본(노출 미보정 상태)에서는 focus_absolute를
    전체 범위(0~1023) 스윕해도 값이 거의 안 변해(오케스트레이터의 초기 관찰 2.49 vs 2.47과 일치)
    초점 차이를 전혀 못 잡아낸다. exposure/analogue_gain을 먼저 올려 노출을 확보한 뒤에야
    이 지표가 실제로 초점 변화에 반응한다(vision/CLAUDE.md 실측 근거 참조). 프레임 전체가 아니라
    관심 영역(ROI, 예: `crop_center_fraction`)에 대해 계산하는 것도 중요 — 근접 배경(카메라
    초점범위 밖의 항상-흐린 영역)이 프레임 대부분을 차지하면 전체 프레임 분산이 그 영역에
    압도돼 초점 변화가 묻힌다(실측으로 확인, 상세는 CLAUDE.md).

    bgr이 이미 그레이스케일(2차원)이면 변환을 건너뛴다."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_center_fraction(bgr: np.ndarray, fraction: float = 0.6) -> np.ndarray:
    """이미지 중앙을 너비/높이 각각 fraction 비율만큼 크롭한다. 순수 함수(형상 계산만, 하드웨어
    없이 테스트 가능).

    초점 스윕의 선명도 계산을 프레임 전체가 아니라 중앙 영역으로 제한하는 이유: 캘리브레이션
    촬영은 사용자가 체커보드 등 타겟을 화면 중앙에 두는 것을 전제로 하고, 실기체 검증에서
    프레임 전체 기준 라플라시안 분산은 근접 바닥 등 항상 흐린 배경에 압도돼 초점 변화에
    거의 반응하지 않았지만 중앙 영역만 보면 뚜렷한 피크가 나타났다(vision/CLAUDE.md 참조)."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"crop_center_fraction: fraction은 0~1 사이여야 한다 (입력: {fraction})")
    h, w = bgr.shape[:2]
    ch, cw = max(1, int(h * fraction)), max(1, int(w * fraction))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return bgr[y0:y0 + ch, x0:x0 + cw]


def pick_best_focus(results: list) -> int:
    """[(focus_value, sharpness), ...] 목록에서 sharpness가 가장 높은 focus_value를 고른다.
    순수 함수 — 스윕 결과 선택 로직만 담당, 실제 촬영/하드웨어와 분리해 단위테스트 가능.

    동점이면 목록에서 먼저 나온(작은 인덱스) 값을 고른다(`max()`의 기본 동작 그대로 사용 —
    특별한 동점 처리 정책이 필요하다는 근거가 없어 가장 단순한 것을 채택)."""
    if not results:
        raise ValueError("pick_best_focus: 빈 결과 목록으로는 최적값을 고를 수 없다")
    return max(results, key=lambda item: item[1])[0]


def parse_focus_sweep_spec(spec: str) -> list:
    """"start:end:step" 형식 문자열 -> focus_absolute 값 리스트(양끝 포함). 순수 파싱 함수.
    예: "400:700:20" -> [400, 420, ..., 700]."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"parse_focus_sweep_spec: 'start:end:step' 형식이어야 한다 (입력: {spec!r})")
    try:
        start, end, step = (int(p) for p in parts)
    except ValueError:
        raise ValueError(f"parse_focus_sweep_spec: 정수만 허용된다 (입력: {spec!r})")
    if step <= 0:
        raise ValueError(f"parse_focus_sweep_spec: step은 양수여야 한다 (입력: {step})")
    values = list(range(start, end + 1, step))
    if not values:
        raise ValueError(f"parse_focus_sweep_spec: 범위가 비어있다 (입력: {spec!r})")
    return values


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def _parse_subdev_size(output: str) -> tuple:
    """v4l2-ctl --get-subdev-fmt 출력에서 "Width/Height : W/H" 줄을 파싱."""
    for line in output.splitlines():
        if "Width/Height" in line:
            w, h = (int(v) for v in line.split(":", 1)[1].strip().split("/"))
            return w, h
    raise RuntimeError(f"_parse_subdev_size: Width/Height 줄을 못 찾음:\n{output}")


def _sensor_active_size(pad: int) -> tuple:
    """imx708(v4l-subdev2)의 pad 현재 활성 크기를 읽는다(모드 스냅 이후 실제값)."""
    out = _run([_V4L2_CTL, "-d", _SENSOR_SUBDEV, "--get-subdev-fmt", f"pad={pad}"]).stdout
    return _parse_subdev_size(out)


def _media_ctl_topology_has_camera(
    media_ctl_output: str,
    driver: str = _CFE_DRIVER_NAME,
    sensor_entity: str = _SENSOR_ENTITY_NAME,
) -> bool:
    """`media-ctl -d <dev> -p` 출력 텍스트를 파싱해 이 media 디바이스가 실제로 카메라
    캡처 파이프라인(driver == `driver`, 센서 엔티티 `sensor_entity` 보유)인지 판별한다.

    순수 문자열 파싱 — subprocess 호출 없이 하드웨어 없는 호스트에서도 단위테스트 가능
    (`vision/tests/test_rpi_capture.py`).

    실기체 조사(2026-07-22d)로 확정한 판별 기준: 이 RPi는 `rp1-cfe` 드라이버 인스턴스가
    CSI 포트(cam0/cam1)마다 하나씩 등록되지만, 카메라가 실제로 연결된 포트만 센서 엔티티
    (`imx708`)를 갖는다 — 연결 안 된 포트는 driver 줄은 똑같이 `rp1-cfe`이지만 엔티티 목록에
    센서가 아예 없다(예: "csi2 (8 pads, 0 link)"만 있고 imx708 항목 자체가 없음). 그래서
    driver 일치만으로는 부족하고 센서 엔티티 존재까지 같이 봐야 한다.
    """
    driver_match = re.search(r"^driver\s+(\S+)\s*$", media_ctl_output, re.MULTILINE)
    if driver_match is None or driver_match.group(1) != driver:
        return False
    entity_pattern = re.compile(rf"^- entity \d+:\s+{re.escape(sensor_entity)}\b", re.MULTILINE)
    return entity_pattern.search(media_ctl_output) is not None


def _media_ctl_driver_name(media_ctl_output: str) -> str:
    """`media-ctl -d <dev> -p` 출력에서 driver 이름만 뽑는다(에러 메시지용, 못 찾으면 '?')."""
    driver_match = re.search(r"^driver\s+(\S+)\s*$", media_ctl_output, re.MULTILINE)
    return driver_match.group(1) if driver_match else "?"


def _iter_media_device_paths() -> list:
    """`/dev/media*` 경로 목록(정렬됨) — 별도 함수로 분리해 테스트에서 monkeypatch하기 쉽게 함
    (하드웨어 없는 호스트에서 `_find_cfe_media_device()`의 순회·에러 처리 로직을 검증할 때
    실제 `/dev`를 건드리지 않고 이 함수만 가짜 목록으로 바꿔치기한다)."""
    return sorted(str(p) for p in Path("/dev").glob("media*"))


def _find_cfe_media_device() -> str:
    """`/dev/media*`를 순회해 실제 카메라(imx708) 캡처 파이프라인을 가진 media 디바이스
    경로를 찾는다. 매 `configure_pipeline()` 호출마다 새로 탐색한다 — media 디바이스
    번호는 부팅/모듈 로드 순서에 따라 바뀔 수 있어 하드코딩·캐시 둘 다 하지 않는다
    (근거: `_CFE_DRIVER_NAME`/`_SENSOR_ENTITY_NAME` 정의부 및 모듈 docstring 상단 참조).

    찾으면 그 경로(str)를 반환. 못 찾으면 확인한 디바이스 목록(경로+driver명)을 포함한
    명확한 에러 메시지와 함께 RuntimeError.
    """
    media_devices = _iter_media_device_paths()
    if not media_devices:
        raise RuntimeError(
            "_find_cfe_media_device: /dev/media* 디바이스가 하나도 없다 — "
            "카메라 모듈이 물리적으로 연결/인식됐는지 먼저 확인해라."
        )

    checked = []
    for dev in media_devices:
        try:
            output = _run([_MEDIA_CTL, "-d", dev, "-p"]).stdout
        except subprocess.CalledProcessError as e:
            checked.append(f"{dev}: media-ctl 조회 실패 ({e})")
            continue
        checked.append(f"{dev}: driver={_media_ctl_driver_name(output)}")
        if _media_ctl_topology_has_camera(output):
            return dev

    raise RuntimeError(
        f"_find_cfe_media_device: driver={_CFE_DRIVER_NAME!r} + 센서 엔티티 "
        f"{_SENSOR_ENTITY_NAME!r}를 가진 media 디바이스를 못 찾았다. "
        "카메라가 연결 안 됐거나 다른 센서일 수 있다. 확인한 디바이스:\n  "
        + "\n  ".join(checked)
    )


def configure_pipeline(width: int, height: int) -> tuple:
    """미디어 파이프라인 전체(링크+포맷)를 요청 해상도로 맞춘다.

    센서가 요청값을 가장 가까운 지원 모드로 스냅할 수 있으므로, 센서에 설정한 뒤 실제
    적용값을 다시 읽어 나머지 파이프라인에 그 실제값을 전파한다.
    반환값: (실제 폭, 실제 높이) — 요청값과 다를 수 있다.
    """
    # 0. media 디바이스 번호는 부팅마다 바뀔 수 있어(모듈 docstring·_find_cfe_media_device
    #    참조) 매 호출마다 새로 탐색한다.
    media_device = _find_cfe_media_device()

    # 1. 링크: csi2 소스 패드4 -> rp1-cfe-csi2_ch0 활성화. pisp-fe(ISP) 경로는 이 도구가
    #    쓰지 않으므로 명시적으로 비활성 유지(파이프라인 검증 시 불필요한 경로가 끼어들지
    #    않도록 — 실기체에서는 원래도 비활성 상태였지만 명시적으로 고정해 둔다).
    _run([_MEDIA_CTL, "-d", media_device, "-l", '"csi2":4 -> "rp1-cfe-csi2_ch0":0 [1]'])
    _run([_MEDIA_CTL, "-d", media_device, "-l", '"csi2":4 -> "pisp-fe":0 [0]'])

    # 2. 센서 이미지 소스 패드(pad0)에 요청 후 실제 스냅값 readback
    _run([_MEDIA_CTL, "-d", media_device, "-V",
          f'"imx708":0 [fmt:{_BAYER_MBUS_CODE}/{width}x{height} field:none]'])
    actual_w, actual_h = _sensor_active_size(pad=0)

    # 3. csi2 이미지 싱크(패드0)/소스(패드4)도 실제값+field:none으로 맞춤 — field 불일치가
    #    실기체에서 실제로 -EPIPE 원인이었다(csi2 싱크 기본값 field:any).
    _run([_MEDIA_CTL, "-d", media_device, "-V",
          f'"csi2":0 [fmt:{_BAYER_MBUS_CODE}/{actual_w}x{actual_h} field:none]'])
    _run([_MEDIA_CTL, "-d", media_device, "-V",
          f'"csi2":4 [fmt:{_BAYER_MBUS_CODE}/{actual_w}x{actual_h} field:none]'])

    # 4. 임베디드 메타데이터 패드(imx708 패드1, IMMUTABLE 링크라 파이프라인 그래프에 항상
    #    포함됨) — 실제 폭을 읽어 csi2 싱크 패드1에 그대로 맞춤.
    embed_w, embed_h = _sensor_active_size(pad=1)
    _run([_V4L2_CTL, "-d", _CSI2_SUBDEV, "--set-subdev-fmt",
          f"pad=1,width={embed_w},height={embed_h},code={_EMBEDDED_DATA_CODE:#x},field=none"])

    # 5. 비디오 캡처 노드 포맷 (실제 해상도로)
    _run([_V4L2_CTL, "-d", _VIDEO_DEVICE, "--set-fmt-video",
          f"width={actual_w},height={actual_h},pixelformat={_PIXELFORMAT}"])

    return actual_w, actual_h


def capture_raw_frame(width: int, height: int, out_path: Path) -> tuple:
    """파이프라인을 (width,height)로 구성하고 프레임 1장을 raw 파일로 캡처한다.
    반환값: 실제 적용된 (width, height) — 센서 모드 스냅으로 요청값과 다를 수 있다."""
    actual_w, actual_h = configure_pipeline(width, height)
    _run([_V4L2_CTL, "-d", _VIDEO_DEVICE, "--stream-mmap", "--stream-count=1",
          f"--stream-to={out_path}"])
    return actual_w, actual_h


def capture_frame_bgr(width: int, height: int, raw_tmp_path: Path,
                       bayer_pattern: str = "rggb", white_balance: bool = True) -> np.ndarray:
    """capture_raw_frame + unpack_raw10 + debayer_to_bgr8을 묶은 헬퍼.
    raw_tmp_path에 중간 raw 파일을 남긴다(디버깅용, 매 호출 덮어씀)."""
    actual_w, actual_h = capture_raw_frame(width, height, raw_tmp_path)
    packed = np.fromfile(raw_tmp_path, dtype=np.uint8)
    bayer16 = unpack_raw10(packed, actual_w, actual_h)
    return debayer_to_bgr8(bayer16, pattern=bayer_pattern, white_balance=white_balance)


def set_focus_absolute(value: int, settle_s: float = FOCUS_SETTLE_S_DEFAULT) -> None:
    """dw9807 VCM에 focus_absolute를 설정하고 물리 이동 정착시간만큼 대기한다.

    **정착시간이 왜 필요한가:** VCM은 전기 신호를 렌즈의 물리적 이동으로 바꾸는 액추에이터라
    레지스터 쓰기가 끝났다고 렌즈가 이미 목표 위치에 도달한 게 아니다 — 그 직후 바로 촬영하면
    렌즈가 이동 중인 상태의 이미지를 얻을 수 있다. 기본값(`FOCUS_SETTLE_S_DEFAULT`)은 실기체
    실측(2026-07-22e, 노출 확보 후 실제 선명도 피크가 재현되는 focus 값으로 왕복 이동시키며
    지연을 0~500ms로 바꿔가며 측정)으로 정했다 — 상세 수치와 방법은 vision/CLAUDE.md 참조.
    하드웨어 필요(테스트 대상 아님 — `vision/tests/test_rpi_capture.py`는 이 함수를 호출하지
    않고 순수 로직(`laplacian_sharpness`/`crop_center_fraction`/`pick_best_focus`/
    `parse_focus_sweep_spec`)만 검증한다).
    """
    if not (_FOCUS_MIN <= value <= _FOCUS_MAX):
        raise ValueError(
            f"set_focus_absolute: value는 {_FOCUS_MIN}~{_FOCUS_MAX} 범위여야 한다 (입력: {value})"
        )
    _run([_V4L2_CTL, "-d", _LENS_SUBDEV, "-c", f"focus_absolute={value}"])
    if settle_s > 0:
        time.sleep(settle_s)


def set_exposure_gain(exposure: int = None, gain: int = None) -> None:
    """센서(imx708, `/dev/v4l-subdev2`)의 exposure/analogue_gain 컨트롤을 설정한다.
    둘 다 None이면 아무 것도 하지 않는다(no-op) — 둘 중 하나만 줘도 그것만 설정.

    **범위를 하드코딩하지 않는 이유:** focus_absolute(dw9807, 렌즈 드라이버 고유 범위, 센서
    모드와 무관)와 달리 exposure의 실제 최댓값은 센서 모드의 vertical_blanking(프레임 길이)에
    따라 달라질 수 있다(`v4l2-ctl --list-ctrls-menus`의 vertical_blanking 범위가 넓은 것이
    이 가변성의 근거) — 특정 모드에서 실측한 범위(예: 1~2602)를 다른 모드에도 그대로
    하드코딩해 검증하면 실제로는 유효한 값을 잘못 거부할 수 있다. 그래서 값을 그대로
    v4l2-ctl에 전달하고, 범위를 벗어나면 v4l2-ctl 자체가 `CalledProcessError`로 거부하도록
    맡긴다(하드웨어가 최종 판단 기준).
    """
    ctrls = []
    if exposure is not None:
        ctrls.append(f"exposure={exposure}")
    if gain is not None:
        ctrls.append(f"analogue_gain={gain}")
    if not ctrls:
        return
    _run([_V4L2_CTL, "-d", _SENSOR_SUBDEV, "-c", ",".join(ctrls)])


def run_focus_sweep(width: int, height: int, raw_tmp_path: Path, values: list,
                     settle_s: float = FOCUS_SETTLE_S_DEFAULT, bayer_pattern: str = "rggb",
                     white_balance: bool = True, roi_fraction: float = 0.6,
                     save_dir: Path = None) -> list:
    """focus_absolute를 `values` 각각으로 설정+정착 후 촬영, 중앙 ROI 기준 선명도를 계산한다.
    반환값: [(focus_value, sharpness), ...] (values 순서 그대로).

    하드웨어 필요(테스트 대상 아님) — 실제 조합 로직은 이미 테스트된 순수 함수
    (`set_focus_absolute`의 검증 부분 제외한 `laplacian_sharpness`/`crop_center_fraction`)를
    그대로 재사용하는 얇은 접착 코드. `save_dir`을 주면 각 프레임을 `sweep_focus_<값>.png`로
    저장(사용자가 어떤 값이 실제로 어떻게 나왔는지 육안으로 재확인할 수 있게)."""
    results = []
    for v in values:
        set_focus_absolute(v, settle_s=settle_s)
        bgr = capture_frame_bgr(width, height, raw_tmp_path, bayer_pattern, white_balance)
        sharpness = laplacian_sharpness(crop_center_fraction(bgr, roi_fraction))
        results.append((v, sharpness))
        print(f"  focus={v:4d} sharpness={sharpness:10.2f} mean={bgr.mean():.1f}")
        if save_dir is not None:
            cv2.imwrite(str(save_dir / f"sweep_focus_{v:04d}.png"), bgr)
    return results


def _render_page(focus: int, exposure: int, gain: int) -> str:
    """미리보기 페이지 HTML. 초점/노출/게인 현재값을 폼에 채워 넣는다 — 사용자가 체커보드를
    들고 화면을 보면서 실시간으로 세 값을 조정할 수 있게 하는 최소 확장(2026-07-22e).
    슬라이더 대신 숫자 입력 폼 + focus용 +/-10 빠른 버튼만 둔다(과설계 금지 지시에 따름) —
    적용은 `/controls`(GET, 쿼리스트링) 302 리다이렉트 왕복으로 처리해 새 JS 없이 기존
    `_PAGE` 패턴(순수 HTML 폼)을 그대로 따른다."""
    focus_minus = max(_FOCUS_MIN, focus - 10)
    focus_plus = min(_FOCUS_MAX, focus + 10)
    return f"""<!doctype html><html><body style="margin:0;background:#111;color:#eee;font-family:sans-serif">
<img id="p" src="/preview.jpg" style="width:100%;display:block"
     onerror="setTimeout(()=>{{this.src='/preview.jpg?t='+Date.now()}}, 500)"
     onload="setTimeout(()=>{{this.src='/preview.jpg?t='+Date.now()}}, 400)">
<form method="POST" action="/capture">
<button style="width:100%;padding:24px;font-size:24px" type="submit">촬영</button>
</form>
<form method="GET" action="/controls" style="padding:10px 12px;display:flex;gap:10px;flex-wrap:wrap;align-items:center">
  <label>focus({_FOCUS_MIN}~{_FOCUS_MAX}) <input name="focus" type="number" min="{_FOCUS_MIN}" max="{_FOCUS_MAX}" value="{focus}" style="width:70px"></label>
  <label>exposure <input name="exposure" type="number" min="1" value="{exposure}" style="width:80px"></label>
  <label>gain <input name="gain" type="number" min="1" value="{gain}" style="width:70px"></label>
  <button type="submit" style="padding:8px 16px">적용</button>
</form>
<div style="padding:0 12px 12px;display:flex;gap:10px">
  <a href="/controls?focus={focus_minus}" style="flex:1;text-align:center;padding:12px;background:#333;color:#eee;text-decoration:none;border-radius:4px">focus -10</a>
  <a href="/controls?focus={focus_plus}" style="flex:1;text-align:center;padding:12px;background:#333;color:#eee;text-decoration:none;border-radius:4px">focus +10</a>
</div>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    camera: "_CaptureSession"

    def log_message(self, *args):
        pass  # 콘솔에 요청 로그 스팸 안 냄

    def do_GET(self):
        if self.path == "/":
            focus, exposure, gain = self.camera.current_controls()
            self._send_bytes(_render_page(focus, exposure, gain).encode(), "text/html; charset=utf-8")
        elif self.path.startswith("/preview.jpg"):
            jpeg = self.camera.latest_preview_jpeg()
            if jpeg is None:
                self.send_response(503)
                self.end_headers()
            else:
                self._send_bytes(jpeg, "image/jpeg")
        elif self.path.startswith("/controls"):
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)

            def _get_int(name):
                raw = qs.get(name, [""])[0]
                return int(raw) if raw != "" else None

            try:
                self.camera.set_controls(
                    focus=_get_int("focus"), exposure=_get_int("exposure"), gain=_get_int("gain"))
            except ValueError as e:
                self._send_bytes(f"잘못된 값: {e}".encode(), "text/plain; charset=utf-8")
                return
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/capture":
            path = self.camera.capture()
            self._send_bytes(f"saved: {path}".encode(), "text/plain; charset=utf-8")
        else:
            self.send_response(404)
            self.end_headers()

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class _CaptureSession:
    """카메라(미디어 파이프라인)는 한 번에 한 스트림만 열 수 있어, 미리보기 루프와 촬영
    트리거가 같은 락을 공유해 v4l2-ctl 스트리밍 호출을 직렬화한다.

    초점/노출/게인은 "원하는 값"(`_focus`/`_exposure`/`_gain`)과 "마지막으로 하드웨어에
    실제 적용한 값"(`_applied_*`)을 분리해 둔다 — `/controls`는 즉시 v4l2-ctl을 부르지 않고
    원하는 값만 갱신하고, 실제 적용은 다음 `preview_tick()`/`capture()`가 이미 쥐고 있는
    `_lock` 안에서 값이 바뀐 경우에만 수행한다(변경 없으면 불필요한 v4l2-ctl 호출/정착시간
    낭비 없음, 그리고 스트리밍 직렬화 락과 충돌 안 남)."""

    def __init__(self, out_dir: Path, preview_size: tuple, main_size: tuple,
                 bayer_pattern: str = "rggb", white_balance: bool = True,
                 focus: int = None, exposure: int = None, gain: int = None,
                 focus_settle_s: float = FOCUS_SETTLE_S_DEFAULT):
        self._out_dir = out_dir
        self._preview_size = preview_size
        self._main_size = main_size
        self._bayer_pattern = bayer_pattern
        self._white_balance = white_balance
        self._lock = threading.Lock()
        self._count = 0
        self._raw_tmp = out_dir / "_last_capture.raw"
        self._preview_jpeg = None
        self._focus_settle_s = focus_settle_s
        self._focus = focus
        self._exposure = exposure
        self._gain = gain
        self._applied_focus = None
        self._applied_exposure = None
        self._applied_gain = None

    def set_controls(self, focus: int = None, exposure: int = None, gain: int = None) -> None:
        """원하는 초점/노출/게인 값을 갱신(주어진 것만) — 하드웨어 적용은 다음 촬영/미리보기
        틱에서 지연 수행(위 클래스 docstring 참조). 범위 검증은 focus만 여기서 미리 하고
        (실패를 호출자에게 바로 알려 HTTP 303 대신 에러 메시지를 보여줄 수 있게), exposure/gain은
        `set_exposure_gain()`과 동일 이유로 v4l2-ctl 자체 검증에 맡긴다."""
        if focus is not None and not (_FOCUS_MIN <= focus <= _FOCUS_MAX):
            raise ValueError(f"focus는 {_FOCUS_MIN}~{_FOCUS_MAX} 범위여야 한다 (입력: {focus})")
        with self._lock:
            if focus is not None:
                self._focus = focus
            if exposure is not None:
                self._exposure = exposure
            if gain is not None:
                self._gain = gain

    def current_controls(self) -> tuple:
        """(focus, exposure, gain) — 아직 한 번도 설정 안 된 값은 하드웨어 기본값으로 채워
        표시(미리보기 페이지 폼 초깃값용)."""
        with self._lock:
            focus = self._focus if self._focus is not None else _FOCUS_DEFAULT
            exposure = self._exposure if self._exposure is not None else _EXPOSURE_DEFAULT
            gain = self._gain if self._gain is not None else _GAIN_DEFAULT
        return focus, exposure, gain

    def _apply_pending_controls_locked(self) -> None:
        """`_lock`을 이미 쥔 상태에서 호출 — 원하는 값이 마지막 적용값과 다르면 하드웨어에
        반영한다. focus는 VCM 정착시간이 걸리므로 실제로 바뀌었을 때만 대기한다."""
        if self._focus is not None and self._focus != self._applied_focus:
            set_focus_absolute(self._focus, settle_s=self._focus_settle_s)
            self._applied_focus = self._focus
        if (self._exposure is not None and self._exposure != self._applied_exposure) or \
           (self._gain is not None and self._gain != self._applied_gain):
            set_exposure_gain(exposure=self._exposure, gain=self._gain)
            self._applied_exposure = self._exposure
            self._applied_gain = self._gain

    def preview_tick(self) -> None:
        w, h = self._preview_size
        with self._lock:
            try:
                self._apply_pending_controls_locked()
                bgr = capture_frame_bgr(w, h, self._raw_tmp, self._bayer_pattern, self._white_balance)
                ok, jpeg = cv2.imencode(".jpg", bgr)
                if ok:
                    self._preview_jpeg = jpeg.tobytes()
            except (subprocess.CalledProcessError, ValueError, RuntimeError) as e:
                print(f"[미리보기 실패] {e}", file=sys.stderr)

    def latest_preview_jpeg(self):
        return self._preview_jpeg

    def capture(self) -> str:
        w, h = self._main_size
        with self._lock:
            self._apply_pending_controls_locked()
            self._count += 1
            path = self._out_dir / f"calib_{self._count:03d}.png"
            bgr = capture_frame_bgr(w, h, self._raw_tmp, self._bayer_pattern, self._white_balance)
            cv2.imwrite(str(path), bgr)
            print(f"[촬영] {path} shape={bgr.shape} mean={bgr.mean():.1f} std={bgr.std():.1f} "
                  f"focus={self._applied_focus} exposure={self._applied_exposure} gain={self._applied_gain}")
            return str(path)


def _preview_loop(session: _CaptureSession, interval: float) -> None:
    while True:
        session.preview_tick()
        time.sleep(interval)


def _enter_key_trigger(session: _CaptureSession) -> None:
    """SSH 터미널에서 Enter만 눌러도 촬영 — 브라우저 버튼과 동등한 경로."""
    while True:
        try:
            input()
        except EOFError:
            return
        session.capture()


def _check_tools_available() -> None:
    missing = [t for t in (_V4L2_CTL, _MEDIA_CTL) if shutil.which(t) is None]
    if missing:
        print(
            f"다음 도구가 없다: {', '.join(missing)}\n"
            "  sudo apt install -y v4l-utils media-ctl",
            file=sys.stderr,
        )
        sys.exit(1)
    if cv2 is None:
        print("cv2(OpenCV)가 없다 — RPi에는 python3-opencv(또는 opencv-python-headless) 필요",
              file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="vision/data/calibration_raw", help="촬영 저장 폴더")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--main-size", default="4608x2592", help="풀해상도 촬영 크기 WxH")
    parser.add_argument("--preview-size", default="1536x864",
                         help="미리보기 스냅샷 크기 WxH (센서 지원 모드로 스냅될 수 있음)")
    parser.add_argument("--preview-interval", type=float, default=1.5, help="미리보기 갱신 주기(초)")
    parser.add_argument("--bayer-pattern", default="rggb",
                         choices=sorted(_BAYER_CV_CODE), help="센서 베이어 패턴 (IMX708=rggb)")
    parser.add_argument("--single-shot", action="store_true",
                         help="HTTP 서버 없이 프레임 1장만 main-size로 캡처해 저장하고 종료"
                              "(원격/스크립트 검증용)")
    parser.add_argument("--white-balance", action=argparse.BooleanOptionalAction, default=True,
                         help="Gray-world 가정 기반 화이트밸런스 보정(기본 켜짐). raw 베이어 경로는"
                              " ISP를 우회하므로 끄면 강한 색편향(이 카메라는 초록)이 그대로 남는다."
                              " --no-white-balance로 끌 수 있다.")
    parser.add_argument("--focus", type=int, default=None,
                         help=f"초점 절대값({_FOCUS_MIN}~{_FOCUS_MAX}, dw9807 VCM). libcamera 없이는"
                              " 연속 AF가 전혀 동작하지 않아 기본값(480)에서 한 번도 안 움직인 채"
                              " 방치되는 문제가 있었다 — 지정하면 촬영 전 설정+정착 대기"
                              "(--focus-settle-ms)까지 수행한다. 최적값은 --focus-sweep로 탐색.")
    parser.add_argument("--focus-settle-ms", type=int, default=int(FOCUS_SETTLE_S_DEFAULT * 1000),
                         help="focus_absolute 설정 후 VCM 물리 이동 정착 대기시간(ms). 실측 근거는"
                              " vision/CLAUDE.md 참조.")
    parser.add_argument("--exposure", type=int, default=None,
                         help="센서 exposure 컨트롤 값(정수, 클수록 밝음). libcamera AE가 없어 이"
                              " raw 경로는 노출이 방치돼 매우 어둡게 나온다 — 실내에서는 크게"
                              " 올려야 한다(근거는 vision/CLAUDE.md).")
    parser.add_argument("--gain", type=int, default=None,
                         help="센서 analogue_gain 컨트롤 값(정수, 클수록 밝음).")
    parser.add_argument("--focus-sweep", default=None, metavar="START:END:STEP",
                         help="HTTP 서버 없이 focus_absolute를 START부터 END까지 STEP 간격으로"
                              " 스윕하며 각각 촬영 → 중앙 영역(--sweep-roi) 라플라시안 분산이 가장"
                              " 높은 값을 보고하고 종료. 좁은 피크(실측 폭 ~100~150)를 놓치지 않게"
                              " STEP은 60 이하를 권장(근거는 vision/CLAUDE.md). 예: --focus-sweep"
                              " 300:800:40")
    parser.add_argument("--sweep-roi", type=float, default=0.6,
                         help="--focus-sweep 선명도 계산에 쓸 중앙 크롭 비율(0~1, 기본 0.6). 실기체"
                              " 검증에서 프레임 전체 기준으로는 근접 배경(항상 흐림)에 신호가 묻혀"
                              " 초점 변화를 못 잡았던 문제 때문에 기본으로 중앙만 본다.")
    args = parser.parse_args()

    _check_tools_available()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_w, main_h = (int(v) for v in args.main_size.split("x"))
    prev_w, prev_h = (int(v) for v in args.preview_size.split("x"))
    focus_settle_s = args.focus_settle_ms / 1000.0

    if args.focus_sweep:
        values = parse_focus_sweep_spec(args.focus_sweep)
        set_exposure_gain(exposure=args.exposure, gain=args.gain)
        raw_tmp = out_dir / "_focus_sweep.raw"
        print(f"[초점 스윕] {len(values)}개 값 {values[0]}~{values[-1]} "
              f"(roi={args.sweep_roi}, settle={args.focus_settle_ms}ms)")
        results = run_focus_sweep(main_w, main_h, raw_tmp, values, settle_s=focus_settle_s,
                                   bayer_pattern=args.bayer_pattern, white_balance=args.white_balance,
                                   roi_fraction=args.sweep_roi, save_dir=out_dir)
        best = pick_best_focus(results)
        print(f"[초점 스윕 완료] 최적 focus_absolute={best}")
        return

    if args.single_shot:
        set_exposure_gain(exposure=args.exposure, gain=args.gain)
        if args.focus is not None:
            set_focus_absolute(args.focus, settle_s=focus_settle_s)
        raw_tmp = out_dir / "_single_shot.raw"
        bgr = capture_frame_bgr(main_w, main_h, raw_tmp, args.bayer_pattern, args.white_balance)
        path = out_dir / "single_shot.png"
        cv2.imwrite(str(path), bgr)
        b_mean, g_mean, r_mean = (float(bgr[:, :, i].mean()) for i in range(3))
        print(f"[단발촬영] {path} shape={bgr.shape} dtype={bgr.dtype} "
              f"mean={bgr.mean():.2f} std={bgr.std():.2f} "
              f"min={int(bgr.min())} max={int(bgr.max())} "
              f"white_balance={args.white_balance} "
              f"focus={args.focus} exposure={args.exposure} gain={args.gain} "
              f"B={b_mean:.2f} G={g_mean:.2f} R={r_mean:.2f}")
        return

    session = _CaptureSession(out_dir, (prev_w, prev_h), (main_w, main_h),
                               args.bayer_pattern, args.white_balance,
                               focus=args.focus, exposure=args.exposure, gain=args.gain,
                               focus_settle_s=focus_settle_s)
    threading.Thread(target=_preview_loop, args=(session, args.preview_interval), daemon=True).start()
    threading.Thread(target=_enter_key_trigger, args=(session,), daemon=True).start()

    handler = type("_BoundHandler", (_Handler,), {"camera": session})
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(f"미리보기: http://<이 라즈베리파이의 IP>:{args.port}/  (SSH 터미널에서 Enter로도 촬영)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
