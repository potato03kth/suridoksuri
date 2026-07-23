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
    """

    def __init__(
        self,
        camera_num: int = 0,
        resolution: Tuple[int, int] = _DEFAULT_LIVE_RESOLUTION,
        retries: int = 3,
        retry_delay: float = 1.0,
    ):
        if retries < 1:
            raise ValueError("retries는 1 이상이어야 한다")
        self.camera_num = camera_num
        self.resolution = resolution
        self.retries = retries
        self.retry_delay = retry_delay
        self._picam: Optional[Any] = None

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
            return
        raise ConnectionError(
            f"LiveFrameSource: 카메라 연결 실패 (camera_num={self.camera_num!r}), "
            f"{last_attempt}/{self.retries}회 재시도 후 포기."
        ) from last_error

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
