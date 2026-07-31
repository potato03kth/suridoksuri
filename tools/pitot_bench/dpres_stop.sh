#!/usr/bin/env bash
# 로거 정상 종료 (SIGTERM -> 핸들러가 CSV 에 stopped_ 푸터를 쓰고 닫는다)
pkill -TERM -f dpres_log.py
sleep 2
if pgrep -f dpres_log.py > /dev/null; then echo "아직 살아있음"; else echo "종료됨"; fi
tail -3 "${DPRES_BENCH_DIR:-$HOME/dpres_bench}/latest.csv"
