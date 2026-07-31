#!/usr/bin/env bash
# 벤치 로거 시작 — SSH 가 끊겨도 계속 돈다 (setsid + nohup + SIGHUP 무시).
#   ./dpres_start.sh ["세션 메모"]
set -u
# 저장소로 들여오면서 바뀐 유일한 줄: 로거 경로를 $HOME/tmp 고정 -> 이 스크립트 옆.
# 로직은 스크래치패드 원본 그대로다 (실기체 검증본).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DPRES_LOG_PY=${DPRES_LOG_PY:-$HERE/dpres_log.py}
BENCH=${DPRES_BENCH_DIR:-$HOME/dpres_bench}
mkdir -p "$BENCH"
STAMP=$(date +%Y%m%d_%H%M%S)
CSV=$BENCH/dpres_$STAMP.csv
NOTE=${1:-}
nohup setsid python3 "$DPRES_LOG_PY" "$CSV" --hz 10 --note "$NOTE" \
      > "$BENCH/dpres_$STAMP.out" 2>&1 < /dev/null &
sleep 3
ln -sfn "$CSV" "$BENCH/latest.csv"
ln -sfn "$CSV.marks" "$BENCH/latest.marks"
echo "CSV     : $CSV"
echo "마커파일: $CSV.marks"
echo "stdout  : $BENCH/dpres_$STAMP.out"
pgrep -af dpres_log.py | grep -v grep
