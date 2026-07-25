"""pytest path 설정.

fc_bridge가 ROS2 colcon 환경 외부(Windows 로컬)에서도 import 가능하도록 project root를 추가한다.
"""
import sys
import os

_HERE = os.path.dirname(__file__)
# _HERE = <repo>/fc_ros/test → ".." = fc_ros, "../.." = repo root.
# 종전엔 "..", "..", ".." 로 저장소 **부모**를 가리켜, CWD가 저장소 루트일 때만
# 우연히 통과했다(2026-07-25 감사, `cd fc_ros && pytest test` 로 재현).
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_FC_ROS_ROOT  = os.path.abspath(os.path.join(_HERE, ".."))

for _p in (_PROJECT_ROOT, _FC_ROS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
