#!/usr/bin/env python3
"""피토 차압센서 다이온도 열평형 감시자 (RPi 로컬 실행)

판정: 최근 5분 창의 OLS 추세기울기 x 300s = 계통적 온도변화량.
      |변화량| < 0.2 도 가 연속 2회 성립하면 열평형으로 보고 exit 0.

왜 max-min 이 아니라 기울기인가:
  다이온도는 양자화 0.0977 도 + 에어컨 압축기 사이클링 때문에 방이 안정일 때도
  5분 창의 max-min 이 0.6~1.5 도로 나온다(실측). max-min 판정은 원리적으로
  절대 성립하지 않는다. 기울기는 진동을 평균으로 지우고 계통적 드리프트만 남긴다.

exit code:  0 = 열평형 도달   2 = 최대 감시시간 초과   3 = 로거 사망/데이터 정지
"""
import csv
import sys
import time

CSV = "/home/suri/dpres_bench/latest.csv"
WINDOW = 300.0        # 판정 창 (초)
THRESH = 0.2          # 도 / 창 길이
MIN_N = 1200          # 창 안 최소 표본수 (10Hz 면 3000 기대)
CONFIRM = 2           # 연속 성립 요구 횟수 (압축기 사이클 오탐 방지)
POLL = 30.0           # 폴링 간격 (초)
STALE = 90.0          # 마지막 표본이 이보다 오래되면 로거 사망
MAX_RUN = 4 * 3600


def read_window():
    """CSV 꼬리에서 최근 WINDOW 초 표본을 (t, dieT, dp) 로 돌려준다."""
    rows = []
    with open(CSV, "r", errors="replace") as f:
        lines = [l for l in f if not l.startswith("#")]
    for r in csv.DictReader(lines):
        try:
            rows.append((float(r["unix_s"]),
                         float(r["dp_die_temp_c"]),
                         float(r["dp_raw_pa"])))
        except (ValueError, TypeError, KeyError):
            continue
    if not rows:
        return None, None
    last_t = rows[-1][0]
    return rows, [r for r in rows if r[0] >= last_t - WINDOW]


def ols(xs, ys):
    """기울기, 절편, 잔차 sd."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        sd = (sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)) ** 0.5
    else:
        sd = 0.0
    return b, a, sd


def emit(msg):
    print("%s  %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def main():
    t_start = time.time()
    hits = 0
    emit("감시 시작 — 창 %.0fs, 임계 %.2f도/창, 연속 %d회 요구"
         % (WINDOW, THRESH, CONFIRM))

    while True:
        if time.time() - t_start > MAX_RUN:
            emit("최대 감시시간 %.1fh 초과 — 종료" % (MAX_RUN / 3600))
            return 2

        try:
            rows, win = read_window()
        except OSError as e:
            emit("CSV 읽기 실패(%s) — 재시도" % e)
            time.sleep(POLL)
            continue

        if not rows:
            emit("표본 없음 — 재시도")
            time.sleep(POLL)
            continue

        age = time.time() - rows[-1][0]
        if age > STALE:
            emit("로거 정지 감지 — 마지막 표본이 %.0fs 전. 중단" % age)
            return 3

        n = len(win)
        if n < MIN_N:
            emit("창 표본 %d < %d — 축적 대기 (dieT=%.2f)" % (n, MIN_N, win[-1][1]))
            hits = 0
            time.sleep(POLL)
            continue

        ts = [r[0] for r in win]
        Ts = [r[1] for r in win]
        dps = [r[2] for r in win]
        bT, _, sdT = ols(ts, Ts)
        bD, _, sdD = ols(ts, dps)
        driftT = bT * WINDOW
        driftD = bD * WINDOW
        meanT = sum(Ts) / n
        meanD = sum(dps) / n

        stable = abs(driftT) < THRESH
        hits = hits + 1 if stable else 0

        emit("dieT=%.2f (%+.3f도/5분)  dp=%.2f (%+.2fPa/5분)  n=%d  %s%s"
             % (meanT, driftT, meanD, driftD, n,
                "안정" if stable else "변화중",
                "  [%d/%d]" % (hits, CONFIRM) if stable else ""))

        if hits >= CONFIRM:
            emit("=== 열평형 도달 ===")
            emit("  다이온도 평균 %.3f 도  (5분 추세 %+.3f 도, 잔차sd %.3f)"
                 % (meanT, driftT, sdT))
            emit("  원시 차압 평균 %.3f Pa  (5분 추세 %+.3f Pa, 잔차sd %.3f)"
                 % (meanD, driftD, sdD))
            emit("  창 표본수 %d,  평균의 표준오차 %.4f Pa" % (n, sdD / n ** 0.5))
            return 0

        time.sleep(POLL)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        emit("사용자 중단")
        sys.exit(130)
