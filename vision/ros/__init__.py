"""`vision/ros/` — 컨테이너(fc, `ros:humble`) 안에서만 도는 ROS2 shim 패키지.

**이 `__init__.py`는 아무것도 import하지 않는다.** `shim_node`가 `rclpy`를 import하는
저장소 유일의 vision 파일인데 랩탑 `.venv`/RPi `picam-venv` 어디에도 rclpy가 없다 —
여기서 `from .shim_node import ...`를 하면 `import vision.ros.shim_core`(순수 stdlib,
단위테스트 대상)까지 통째로 죽는다. `vision/utils/frame_source.py`가 picamera2를
지연 import로 격리한 것과 같은 이유·같은 패턴이다.
"""
