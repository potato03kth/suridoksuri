#!/usr/bin/env bash
# SITL-7 런 분석 래퍼 — ROS 소싱은 필요 없지만 호출 형태를 run_scenario.sh 와 맞춘다.
#
#   wsl.exe -d Ubuntu-22.04 -- bash /root/drone_ws/src/suridoksuri/tools/sitl/analyze_run.sh A1
#
# `set -u` 는 쓰지 않는다 — /opt/ros/humble/setup.bash 가
# AMENT_TRACE_SETUP_FILES 미설정 변수를 참조해 즉시 죽는다(실측).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${HERE}/analyze_run.py" "$@"
