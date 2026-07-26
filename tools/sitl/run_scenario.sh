#!/usr/bin/env bash
# SITL-7 시나리오 실행 래퍼 — ROS2/워크스페이스 소싱 후 run_scenario.py 호출.
#
# 이 래퍼가 존재하는 이유: `wsl.exe -d Ubuntu-22.04 -- bash -lc '...'` 는 복잡한
# 셸 구문(`$(seq)`+`for`+`if` 조합 등)이 깨지는 것이 실측됐다. 호스트에서는
# 항상 아래 한 줄 형태로만 호출한다:
#
#   wsl.exe -d Ubuntu-22.04 -- bash /root/drone_ws/src/suridoksuri/tools/sitl/run_scenario.sh A1
#
# `set -u` 는 쓰지 않는다 — /opt/ros/humble/setup.bash 가
# AMENT_TRACE_SETUP_FILES 미설정 변수를 참조해 즉시 죽는다(실측).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "${DRONE_WS:-/root/drone_ws}/install/setup.bash"

exec python3 "${HERE}/run_scenario.py" "$@"
