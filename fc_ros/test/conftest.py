"""pytest path 설정.

fc_bridge가 ROS2 colcon 환경 외부(Windows 로컬)에서도 import 가능하도록 project root를 추가한다.
"""
import sys
import os

_HERE = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_FC_ROS_ROOT  = os.path.abspath(os.path.join(_HERE, ".."))

for _p in (_PROJECT_ROOT, _FC_ROS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
