#!/usr/bin/env bash
# record_flight.sh — 비행 1회 = 폴더 1개 (logs/YYYY-MM-DD_flightNN/) 자동 수집 래퍼
#
# 하는 일:
#   1. 플라이트 폴더 생성 + notes.md 템플릿
#   2. rosbag2 record 백그라운드 시작 (토픽: topics.txt)
#   3. ros2 launch fc_ros phase2.launch.py <인자 그대로 통과> | tee launch.log
#   4. launch 종료(Ctrl-C) 시: rosbag 정지 → ulog 회수
#      --sitl  : SITL 로그 디렉터리에서 최신 .ulg 복사
#      (기본)  : pull_ulog.py 로 MAVLink 다운로드 (실패 시 SD 폴백 안내)
#
# 사용 예:
#   ./record_flight.sh vehicle_type:=mc                       # RPi 실기체
#   ./record_flight.sh --sitl v_cruise:=18.0                  # WSL SITL
#   ./record_flight.sh --no-ulog vehicle_type:=mc             # 벤치 (ulog 생략)
#
# 환경변수 오버라이드:
#   FLIGHT_LOG_ROOT   로그 루트 (기본: <repo>/logs)
#   PX4_SITL_LOG_DIR  SITL ulog 위치 (기본: ~/PX4-Autopilot/build/px4_sitl_default/rootfs/log)
#   PULL_ULOG_ARGS    pull_ulog.py 추가 인자 (예: "--url /dev/ttyACM1")

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_ROOT="${FLIGHT_LOG_ROOT:-$REPO_ROOT/logs}"
SITL_LOG_DIR="${PX4_SITL_LOG_DIR:-$HOME/PX4-Autopilot/build/px4_sitl_default/rootfs/log}"

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

SITL=0
NO_ULOG=0
LAUNCH_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --sitl)    SITL=1 ;;
        --no-ulog) NO_ULOG=1 ;;
        -h|--help) usage; exit 0 ;;
        *)         LAUNCH_ARGS+=("$arg") ;;
    esac
done

# ── 1. 플라이트 폴더 ────────────────────────────────────────────────
mkdir -p "$LOG_ROOT"
DIRNAME="$(python3 "$SCRIPT_DIR/pull_ulog.py" --next-dir --log-root "$LOG_ROOT")" || {
    echo "[record_flight] 폴더 넘버링 실패" >&2; exit 1; }
FLIGHT_DIR="$LOG_ROOT/$DIRNAME"
mkdir -p "$FLIGHT_DIR"
echo "[record_flight] 플라이트 폴더: $FLIGHT_DIR"

cat > "$FLIGHT_DIR/notes.md" <<EOF
# $DIRNAME

- **비행 조건:** (기체/모드/launch 인자: ${LAUNCH_ARGS[*]:-없음}$( [ "$SITL" -eq 1 ] && echo ' [SITL]' ))
- **관찰:**
- **결론:**
EOF

# ── 2. rosbag2 record 백그라운드 ───────────────────────────────────
TOPICS=()
while IFS= read -r line; do
    line="${line%%#*}"
    line="$(echo "$line" | tr -d '[:space:]')"
    [ -n "$line" ] && TOPICS+=("$line")
done < "$SCRIPT_DIR/topics.txt"

ros2 bag record -o "$FLIGHT_DIR/rosbag" "${TOPICS[@]}" \
    > "$FLIGHT_DIR/rosbag_record.log" 2>&1 &
BAG_PID=$!
echo "[record_flight] rosbag2 record 시작 (pid $BAG_PID, 토픽 ${#TOPICS[@]}개)"

stop_bag() {
    if kill -0 "$BAG_PID" 2>/dev/null; then
        kill -INT "$BAG_PID" 2>/dev/null
        wait "$BAG_PID" 2>/dev/null
        echo "[record_flight] rosbag2 정지"
    fi
}
trap stop_bag EXIT

sleep 2  # rosbag 기동 대기
if ! kill -0 "$BAG_PID" 2>/dev/null; then
    echo "[record_flight] 경고: rosbag2가 즉시 종료됨 — rosbag_record.log 확인. launch는 계속 진행" >&2
fi

# ── 3. launch (Ctrl-C 로 종료 — 스크립트는 살아서 수집 단계로 진행) ──
echo "[record_flight] ros2 launch 시작 — 종료는 Ctrl-C"
trap ':' INT
ros2 launch fc_ros phase2.launch.py ${LAUNCH_ARGS[@]+"${LAUNCH_ARGS[@]}"} 2>&1 \
    | tee "$FLIGHT_DIR/launch.log"
trap - INT
echo "[record_flight] launch 종료 — 수집 단계 시작"

# ── 4. rosbag 정지 + ulog 회수 ─────────────────────────────────────
stop_bag
trap - EXIT

if [ "$NO_ULOG" -eq 1 ]; then
    echo "[record_flight] --no-ulog: ulog 회수 생략"
elif [ "$SITL" -eq 1 ]; then
    LATEST_ULG="$(find "$SITL_LOG_DIR" -name '*.ulg' -printf '%T@ %p\n' 2>/dev/null \
                  | sort -n | tail -1 | cut -d' ' -f2-)"
    if [ -n "$LATEST_ULG" ]; then
        cp "$LATEST_ULG" "$FLIGHT_DIR/"
        echo "[record_flight] SITL ulog 복사: $LATEST_ULG"
    else
        echo "[record_flight] 경고: $SITL_LOG_DIR 에 .ulg 없음 (arm 안 했으면 정상)" >&2
    fi
else
    # 실기체: MAVROS가 내려간 뒤라 시리얼 포트가 비어 있다
    python3 "$SCRIPT_DIR/pull_ulog.py" --out "$FLIGHT_DIR" ${PULL_ULOG_ARGS:-} \
        || echo "[record_flight] ulog 자동 회수 실패 — 위 폴백 안내대로 SD 수동 회수" >&2
fi

# 컨테이너(fc, root)로 실행된 경우 $FLIGHT_DIR가 root 소유가 되어 호스트의
# suri 사용자가 이후 접근/전송하지 못하는 문제를 방지 — best-effort
chown -R "$(stat -c '%u:%g' "$LOG_ROOT")" "$FLIGHT_DIR" 2>/dev/null || true

echo
echo "[record_flight] 수집 완료: $FLIGHT_DIR"
ls -la "$FLIGHT_DIR"
echo "[record_flight] notes.md 에 비행 조건/관찰/결론을 채워라."
