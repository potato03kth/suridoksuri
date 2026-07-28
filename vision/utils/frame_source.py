"""
FrameSource 어댑터 — Live/Dir/Bag (vision_plan.md §7.2 "카메라" 변화흡수 이음새, §7.5 기록·재생,
§7.9 "지금 당장 할 일" 4번).

세 모드:
  LiveFrameSource   실카메라(picamera2 백엔드). 연결 실패 시 재시도 후 명확한 ConnectionError.
                    2026-07-24 카메라 브링업(docs/vision_camera_bringup.md)으로 libcamera가
                    정식 경로로 살아나면서 picamera2 API로 재구현됨 — 이전 cv2.VideoCapture
                    구현은 V4L2 raw 경로와 비호환임이 실측 확인됐다(isOpened()는 성공하지만
                    read()가 실패, docs/vision_status.md 2026-07-22b). picamera2는 이 노트북
                    .venv에 없는 RPi 전용 하드웨어 라이브러리라 지연 import(open() 내부)로
                    격리한다 — 모듈 최상단에서 import하면 DirFrameSource/BagFrameSource를 쓰는
                    이 .venv의 모든 코드가 깨진다.
  DirFrameSource    녹화 폴더 재생 — 프레임 이미지 파일들 + 선택적 telemetry.jsonl.
                    §7.9 (a) "재생 오버레이 뷰어"의 주력 입력.
  BagFrameSource    단일 recording bag(비디오 파일) + 선택적 사이드카 telemetry.jsonl 재생.

모든 어댑터는 FrameRecord(frame_id, ts, image, telemetry)를 순서대로 내는 이터레이터다.
결정론적 재생(§7.5): 같은 입력 → 같은 순서의 같은 FrameRecord. wall-clock/난수 없음
(LiveFrameSource의 ts만 예외 — 실카메라는 실시간이므로 캡처 시각을 그대로 쓴다).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple, Union

import cv2
import numpy as np


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class FrameRecord:
    """한 프레임 + 그 시점 텔레메트리(§7.5 "raw 프레임 + 텔레메트리를 타임스탬프와 함께 기록")."""

    frame_id: int
    ts: float
    image: np.ndarray
    telemetry: dict = field(default_factory=dict)


def _load_telemetry_jsonl(path: Path) -> dict[int, dict]:
    """frame_id → 텔레메트리 dict. 파일이 없으면 빈 dict(텔레메트리 없이도 재생 가능)."""
    if not path.exists():
        return {}
    by_frame_id: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "frame_id" not in record:
            raise ValueError(f"텔레메트리 라인에 frame_id 없음: {path} — {line!r}")
        by_frame_id[record["frame_id"]] = record
    return by_frame_id


_DEFAULT_LIVE_RESOLUTION: Tuple[int, int] = (4608, 2592)
"""vision/calibration/cam109-imx708af75/nominal.yaml의 image_size와 동일 — solvePnP가 쓰는
camera_matrix가 이 해상도 기준이라, LiveFrameSource가 기본으로 다른 해상도를 내면 캘리브레이션이
조용히 어긋난다(ArUco Phase 3, core/target.py)."""


# ---------------------------------------------------------------------------
# AF(오토포커스) 제어 — 순수 로직 (libcamera 미의존, 하드웨어 없이 테스트 가능)
#
# **여기가 저장소의 AF 제어 단일 출처다.** 원래 `tools/h264_stream.py`(2026-07-25, 실기체
# 검증됨)에만 있었는데, 라이브 파이프라인 경로(`LiveFrameSource`)의 초점이 드라이버 기본
# 동작에 방치돼 있던 갭을 닫으면서 이쪽으로 옮겼다 — 두 벌로 복제하면 `LENS_POSITION_MAX`
# 같은 **실측 물리값**이 한쪽만 갱신돼 조용히 갈라진다. `h264_stream.py`가 여기서 import해
# 쓰므로 동작은 이전과 동일하다(import 규칙상 방향은 `tools/ → utils/`만 가능).
# ---------------------------------------------------------------------------

DEFAULT_AF_MODE = "continuous"
AF_MODES = ("continuous", "auto", "manual")

# VCM 실가동범위(실측 하드클램프) — 드라이버는 32.0을 광고하지만 15.0에서 하드클램프된다.
# 32를 상한으로 쓰지 않는다(실기체 확정 사실, 재확인 불필요).
LENS_POSITION_MIN = 0.0
LENS_POSITION_MAX = 15.0


def validate_lens_position(value: float) -> float:
    """VCM 실가동범위(0~15.0 디옵터) 밖이면 거부(클램프하지 않음 — 조용히 다른 값으로
    바뀌는 것보다 호출자가 명시적으로 알아채는 편이 안전).
    """
    if not (LENS_POSITION_MIN <= value <= LENS_POSITION_MAX):
        raise ValueError(
            f"lens-position은 {LENS_POSITION_MIN}~{LENS_POSITION_MAX} 범위여야 함 "
            f"(요청값={value}). 드라이버가 광고하는 상한 32.0은 실측 하드클램프로 무효 — "
            "32를 상한으로 쓰지 말 것."
        )
    return value


def validate_af_args(af_mode: str, lens_position: Optional[float]) -> None:
    """af_mode/lens_position 조합 검증.

    manual은 lens_position이 필수, 그 외 모드는 lens_position을 받지 않는다(모드 하나당
    의미가 명확한 조합만 허용 — 조용히 무시되는 인자를 남기지 않는다).
    """
    if af_mode not in AF_MODES:
        raise ValueError(f"af_mode는 {AF_MODES} 중 하나여야 함: {af_mode!r}")
    if af_mode == "manual":
        if lens_position is None:
            raise ValueError("af_mode manual 은 lens_position 이 필수")
        validate_lens_position(lens_position)
    elif lens_position is not None:
        raise ValueError(
            f"lens_position은 af_mode manual 에서만 사용 가능(현재 af_mode={af_mode!r})"
        )


def make_af_controls(
    af_mode: str, lens_position: Optional[float], controls_module: Any
) -> Tuple[dict, Optional[dict]]:
    """(초기 set_controls dict, 트리거용 2차 set_controls dict|None) 반환.

    `controls_module`은 `from libcamera import controls`로 얻는 실제 모듈(또는 테스트용
    가짜 객체) — 이 함수 자체는 libcamera를 import하지 않아 하드웨어 없이도 테스트 가능하다.

    - continuous: AfMode=Continuous 하나로 충분(연속 AF, 트리거 불필요).
    - auto: AfMode=Auto 로 전환한 뒤 AfTrigger=Start로 단발 스캔을 시작해야 한다(libcamera
      표준 동작 — Auto는 트리거 없이는 렌즈가 마지막 위치에 그대로 머문다).
    - manual: AfMode=Manual + LensPosition을 한 번에 건다(calib_capture.py 전례와 동일 패턴).
    """
    if af_mode == "continuous":
        return {"AfMode": controls_module.AfModeEnum.Continuous}, None
    if af_mode == "auto":
        return (
            {"AfMode": controls_module.AfModeEnum.Auto},
            {"AfTrigger": controls_module.AfTriggerEnum.Start},
        )
    if af_mode == "manual":
        return (
            {"AfMode": controls_module.AfModeEnum.Manual, "LensPosition": float(lens_position)},
            None,
        )
    raise ValueError(f"알 수 없는 af_mode: {af_mode!r}")  # validate_af_args가 이미 걸렀어야 함(방어)


class LiveFrameSource:
    """실카메라 프레임 소스 (picamera2 백엔드).

    2026-07-24 카메라 브링업(docs/vision_camera_bringup.md) 성공으로 libcamera/picamera2
    정식 경로가 열려, V4L2 raw 경로와 비호환임이 실측 확인된(§ 위 모듈 docstring) 이전
    cv2.VideoCapture 구현을 대체한다.

    ⚠️ 이 노트북 .venv에는 picamera2가 없다(RPi 전용 하드웨어 라이브러리) — 그래서 `open()`
    내부에서만 `from picamera2 import Picamera2`를 한다(지연 import). 이 클래스를 생성하는 것
    자체나 이 모듈을 import하는 것은 picamera2 없이도 항상 성공해야 한다 — 실패는 오직
    `open()`(또는 `open()`을 부르는 `__enter__`/`__iter__`) 호출 시점에만 일어난다.

    camera_num: picamera2가 받는 카메라 인덱스. cv2 구현의 `device`(정수 인덱스 또는
        V4L2/GStreamer 경로 문자열)를 대체 — picamera2는 정수 카메라 번호만 받고 장치 경로
        문자열 개념이 없어(내부적으로 libcamera가 열거) 생성자 인자를 자연스럽게 좁혔다.
    resolution: (width, height). 기본값은 위 `_DEFAULT_LIVE_RESOLUTION` 참조.
        calib_capture.py의 "단일 still config, 세션 내내 모드 전환 없음" 원칙과 일관되게
        `create_still_configuration()`으로 한 번만 설정하고 이후 모드를 바꾸지 않는다
        (모드 전환이 센서 크롭/비닝을 바꿔 인트린식이 흔들리는 것을 원천 차단).
    retries/retry_delay: 기존 cv2 구현과 동일한 재시도 계약 유지.

    af_mode/lens_position: **[2026-07-28 추가]** 오토포커스 제어. 그전까지 이 클래스는 AF를
        전혀 건드리지 않아 **라이브 파이프라인의 초점이 드라이버 기본 동작에 방치**돼
        있었다(AF 제어는 `tools/h264_stream.py`에만 들어가 있었다). 기본값
        `DEFAULT_AF_MODE`("continuous") — 기체가 10~40m를 오르내리므로 연속 AF가 맞다.
        `af_mode=None`을 주면 **AF에 손대지 않는다**(예전 동작 그대로 — 현장에서 AF가
        말썽이면 되돌릴 escape hatch). `manual`은 `lens_position`(0~15.0 디옵터) 필수.
        인자 조합 검증은 **생성자에서 즉시**(하드웨어 만지기 전에 실패).

        🔴 **실기체 미검증.** 이 세션은 RPi 접속이 금지돼 있어 단위테스트(가짜 picamera2 주입)
        까지만 했다. 실기체 확인 절차는 `docs/vision_camera_bringup.md` / 인수인계 참조 —
        `tools/h264_stream.py`의 AF 경로 자체는 2026-07-25에 실기체에서 "크래시 없이 동작"
        까지 확인됐지만, **초점이 실제로 이동했는지(선명도 지표)는 그때도 미검증**이다.
    """

    def __init__(
        self,
        camera_num: int = 0,
        resolution: Tuple[int, int] = _DEFAULT_LIVE_RESOLUTION,
        retries: int = 3,
        retry_delay: float = 1.0,
        af_mode: Optional[str] = DEFAULT_AF_MODE,
        lens_position: Optional[float] = None,
    ):
        if retries < 1:
            raise ValueError("retries는 1 이상이어야 한다")
        if af_mode is None:
            if lens_position is not None:
                raise ValueError("af_mode=None(AF 미개입)에는 lens_position을 줄 수 없다")
        else:
            validate_af_args(af_mode, lens_position)
        self.camera_num = camera_num
        self.resolution = resolution
        self.retries = retries
        self.retry_delay = retry_delay
        self.af_mode = af_mode
        self.lens_position = lens_position
        self._picam: Optional[Any] = None
        # AF 적용 결과 관측용(§7.4) — 적용 성공 시 True, 실패 시 사유 문자열이 af_error에 남는다.
        self.af_applied: bool = False
        self.af_error: Optional[str] = None

    def open(self) -> None:
        """연결 시도. 실패하면 retries회까지 retry_delay초 간격으로 재시도 후 ConnectionError.

        picamera2 자체를 import할 수 없는 경우(설치 안 됨)는 재시도 대상이 아니다 — 이건
        하드웨어의 일시적 연결 실패가 아니라 환경 문제이므로 ImportError를 그대로 전파한다.
        """
        from picamera2 import Picamera2

        last_attempt = 0
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.retries + 1):
            last_attempt = attempt
            picam = None
            try:
                picam = Picamera2(camera_num=self.camera_num)
                # "RGB888" 포맷 요청 → 실제 메모리 바이트 순서는 B,G,R (picamera2 명명 역전,
                # vision/tools/calib_capture.py에서 실기체로 확인된 사실 재사용) — 이 덕분에
                # cv2/DirFrameSource/BagFrameSource가 기대하는 BGR 배열을 별도 cv2.cvtColor
                # 변환 없이 바로 얻는다.
                config = picam.create_still_configuration(
                    main={"format": "RGB888", "size": self.resolution}
                )
                picam.configure(config)
                picam.start()
            except Exception as exc:  # picamera2/libcamera 구체 예외 타입은 버전마다 다름 — 폭넓게 잡아 재시도
                last_error = exc
                if picam is not None:
                    try:
                        picam.close()
                    except Exception:
                        pass
                if attempt < self.retries:
                    time.sleep(self.retry_delay)
                continue
            self._picam = picam
            self._apply_af()
            return
        raise ConnectionError(
            f"LiveFrameSource: 카메라 연결 실패 (camera_num={self.camera_num!r}), "
            f"{last_attempt}/{self.retries}회 재시도 후 포기."
        ) from last_error

    def _apply_af(self) -> None:
        """`picam.start()` 직후 AF 컨트롤을 건다 (`tools/h264_stream.py::run_server` 순서 동일).

        **실패해도 예외를 올리지 않는다** — AF를 못 걸면 카메라는 드라이버 기본 초점으로
        계속 동작하는데(=이 변경 전의 동작), 그것 때문에 라이브 파이프라인 전체를 죽이면
        얻는 것보다 잃는 게 크다. 대신 사유를 `af_error`에 남겨 조용히 묻히지 않게 한다
        (§7.4 "거절이유 로깅" 철학 — 침묵 금지). 인자 조합 오류는 이 지점까지 오지 않는다
        (생성자에서 이미 걸렀다).

        `libcamera`는 picamera2와 마찬가지로 RPi 전용이라 여기서 지연 import한다.
        """
        if self.af_mode is None:
            self.af_error = None
            self.af_applied = False
            return
        try:
            from libcamera import controls as libcamera_controls

            initial, trigger = make_af_controls(
                self.af_mode, self.lens_position, libcamera_controls
            )
            self._picam.set_controls(initial)
            if trigger is not None:
                self._picam.set_controls(trigger)
        except Exception as exc:  # libcamera 미설치/컨트롤 미지원/드라이버 거부 등
            self.af_applied = False
            self.af_error = f"{type(exc).__name__}: {exc}"
            return
        self.af_applied = True
        self.af_error = None

    def close(self) -> None:
        if self._picam is not None:
            try:
                self._picam.stop()
            except Exception:
                pass
            try:
                self._picam.close()
            except Exception:
                pass
            self._picam = None

    def __enter__(self) -> "LiveFrameSource":
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __iter__(self) -> Iterator[FrameRecord]:
        if self._picam is None:
            self.open()
        frame_id = 0
        while True:
            try:
                frame = self._picam.capture_array("main")
            except Exception as exc:
                raise ConnectionError(
                    f"LiveFrameSource: 프레임 읽기 실패 (camera_num={self.camera_num!r}) — "
                    "연결이 끊겼을 수 있다."
                ) from exc
            yield FrameRecord(frame_id=frame_id, ts=time.time(), image=frame, telemetry={})
            frame_id += 1


class DirFrameSource:
    """녹화 폴더 재생 (§7.9 (a) 재생 오버레이 뷰어 주력 입력).

    폴더 안 이미지 파일들(png/jpg/jpeg/bmp)을 파일명 정렬 순서로 읽는다.
    frame_id는 정렬 순서의 0-based 인덱스 — 같은 폴더는 항상 같은 순서를 낸다(결정론).
    선택적으로 같은 폴더에 telemetry.jsonl이 있으면 {"frame_id": ..., "ts": ..., ...} 라인들을
    frame_id로 매칭해 붙인다. 없으면 ts는 frame_id를 초 단위로 쓴 자리표시자.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.is_dir():
            raise NotADirectoryError(f"DirFrameSource: 디렉터리 없음 — {path}")
        self._files = sorted(
            p for p in self.path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not self._files:
            raise FileNotFoundError(f"DirFrameSource: 프레임 이미지가 없음 — {path}")
        self._telemetry = _load_telemetry_jsonl(self.path / "telemetry.jsonl")

    def __len__(self) -> int:
        return len(self._files)

    def __iter__(self) -> Iterator[FrameRecord]:
        for frame_id, file in enumerate(self._files):
            image = cv2.imread(str(file))
            if image is None:
                raise ValueError(f"DirFrameSource: 디코딩 실패 — {file}")
            telemetry = self._telemetry.get(frame_id, {})
            ts = telemetry.get("ts", float(frame_id))
            yield FrameRecord(frame_id=frame_id, ts=ts, image=image, telemetry=telemetry)

    def __enter__(self) -> "DirFrameSource":
        return self

    def __exit__(self, *_exc) -> None:
        pass


class BagFrameSource:
    """단일 recording bag 재생.

    이 코드베이스에는 rosbag 의존성이 없으므로 "bag"은 비디오 파일 하나(mp4/avi 등) +
    선택적 사이드카 텔레메트리(같은 basename, .jsonl 확장자)로 구현한다 — 폴더로 흩어진
    DirFrameSource보다 더 압축된 단일 파일 재생 경로.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"BagFrameSource: 파일 없음 — {path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"BagFrameSource: 열 수 없음 — {path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        sidecar = self.path.with_suffix(".jsonl")
        self._telemetry = _load_telemetry_jsonl(sidecar)

    def __iter__(self) -> Iterator[FrameRecord]:
        frame_id = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            telemetry = self._telemetry.get(frame_id, {})
            ts = telemetry.get("ts", frame_id / self._fps)
            yield FrameRecord(frame_id=frame_id, ts=ts, image=frame, telemetry=telemetry)
            frame_id += 1

    def __enter__(self) -> "BagFrameSource":
        return self

    def __exit__(self, *_exc) -> None:
        self._cap.release()


def open_dir_or_bag(path: Union[str, Path]) -> Union[DirFrameSource, BagFrameSource]:
    """재생 CLI용 팩토리 — 경로가 디렉터리면 Dir, 파일이면 Bag (§7.9 항목4 "<녹화폴더|bag>")."""
    p = Path(path)
    if p.is_dir():
        return DirFrameSource(p)
    if p.is_file():
        return BagFrameSource(p)
    raise FileNotFoundError(f"open_dir_or_bag: 경로 없음 — {path}")
