"""
FrameSource 어댑터 — Live/Dir/Bag (vision_plan.md §7.2 "카메라" 변화흡수 이음새, §7.5 기록·재생,
§7.9 "지금 당장 할 일" 4번).

세 모드:
  LiveFrameSource   실카메라(V4L2/장치경로 또는 인덱스). 연결 실패 시 재시도 후
                    명확한 ConnectionError. 실제 하드웨어 검증은 이 세션 범위 밖(§ vision_status.md).
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
from typing import Iterator, Optional, Union

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


class LiveFrameSource:
    """실카메라 프레임 소스.

    ⚠️ 이 세션(2026-07-21)에서는 RPi 실카메라 하드웨어 작업이 금지되어 있다
    (docs/vision_status.md — libcamera PiSP IPA 브링업 미완). 아래는 인터페이스
    계약(연결 실패 시 재시도 → 명확한 에러)만 만족시키며, 실장치 연결 검증은
    RPi 작업 재개 시 별도로 한다.

    device: cv2.VideoCapture가 받는 장치(정수 인덱스 또는 V4L2/GStreamer 경로 문자열).
    """

    def __init__(self, device: Union[int, str], retries: int = 3, retry_delay: float = 1.0):
        if retries < 1:
            raise ValueError("retries는 1 이상이어야 한다")
        self.device = device
        self.retries = retries
        self.retry_delay = retry_delay
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """연결 시도. 실패하면 retries회까지 retry_delay초 간격으로 재시도 후 ConnectionError."""
        last_attempt = 0
        for attempt in range(1, self.retries + 1):
            last_attempt = attempt
            cap = cv2.VideoCapture(self.device)
            if cap.isOpened():
                self._cap = cap
                return
            cap.release()
            if attempt < self.retries:
                time.sleep(self.retry_delay)
        raise ConnectionError(
            f"LiveFrameSource: 카메라 연결 실패 (device={self.device!r}), "
            f"{last_attempt}/{self.retries}회 재시도 후 포기."
        )

    def __enter__(self) -> "LiveFrameSource":
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> Iterator[FrameRecord]:
        if self._cap is None:
            self.open()
        frame_id = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                raise ConnectionError(
                    f"LiveFrameSource: 프레임 읽기 실패 (device={self.device!r}) — "
                    "연결이 끊겼을 수 있다."
                )
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
