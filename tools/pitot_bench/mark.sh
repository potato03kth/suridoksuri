#!/usr/bin/env bash
# 환경 변화 마커. 예: ./mark.sh "냉방 ON 설정 24도, 6평방 진입"
set -u
BENCH=${DPRES_BENCH_DIR:-$HOME/dpres_bench}
M=$(readlink -f "$BENCH/latest.marks" 2>/dev/null || echo "")
[ -z "$M" ] && { echo "로거가 안 돌고 있다 (latest.marks 없음)"; exit 1; }
echo "$*" >> "$M"
echo "기록: $*  -> $M"
