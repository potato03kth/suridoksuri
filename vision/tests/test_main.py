"""main.py CLI 헤드리스 안전성 회귀 테스트.

불변식: 기본값 --display none 은 어떤 GUI 함수(cv2.imshow 등)도 호출하지 않는다.
드론(디스플레이 없음)에서의 크래시를 방지하는 계약이므로 절대 깨지면 안 된다.
"""
import sys

import cv2
import numpy as np
import pytest

import vision.main as main_mod


def _write_image(tmp_path) -> str:
    img = np.full((120, 120, 3), 180, dtype=np.uint8)
    p = tmp_path / "frame.png"
    cv2.imwrite(str(p), img)
    return str(p)


def test_display_none_never_calls_imshow(tmp_path, monkeypatch):
    img_path = _write_image(tmp_path)
    calls = []
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(sys, "argv", ["vision.main", img_path])  # 기본 --display none
    main_mod.main()
    assert calls == [], "--display none 에서 imshow가 호출되면 헤드리스 크래시 위험"


def test_display_window_calls_imshow(tmp_path, monkeypatch):
    img_path = _write_image(tmp_path)
    calls = []
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(cv2, "waitKey", lambda *a, **k: ord("q"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(sys, "argv", ["vision.main", img_path, "--display", "window"])
    main_mod.main()
    assert len(calls) == 1


def test_display_file_requires_output(tmp_path, monkeypatch):
    img_path = _write_image(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vision.main", img_path, "--display", "file"])
    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 2


def test_display_stream_not_implemented(tmp_path, monkeypatch):
    img_path = _write_image(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vision.main", img_path, "--display", "stream"])
    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 2
