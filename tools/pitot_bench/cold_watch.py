#!/usr/bin/env python3
"""피토 강온 연장 감시자 — 저온 기준점 확정용 (RPi 로컬 실행)

앞선 감시자는 임계 0.2도/5분으로 통과시켰는데, 통과 시점의 추세가
-0.132도/5분(= 시간당 -1.58도)이라 아직 하강 중이었다. 사용자 지적에 따라
① 최소 유지시간을 강제하고 ② 온도 임계를 0.10도/5분으로 조이고
③ 기류 오염까지 함께 본다(dp 잔차 sd).

exit code: 0 = 저온 기준점 확정  2 = 감시시간 초과  3 = 로거 사망
"""
import csv
import sys
import time

CSV = "/home/suri/dpres_bench/latest.csv"
MIN_HOLD = 5.0        # 분 — 사용자 요청 추가 유지
WINDOW = 300.0
THRESH_T = 0.10       # 도 / 5분 (종전 0.20 에서 조임)
THRESH_SD = 2.60      # Pa — 청정 기준선 2.10 대비 여유. 넘으면 기류 오염
MIN_N = 1200
CONFIRM = 2
POLL = 30.0
STALE = 90.0
MAX_RUN = 3 * 3600


def read_window():
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
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    sd = ((sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)) ** 0.5
          if n > 2 else 0.0)
    return b, a, sd


def emit(msg):
    print("%s  %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def main():
    t_start = time.time()
    hits = 0
    emit("강온 연장 감시 시작 — 최소 %.0f분, 온도임계 %.2f도/5분, 기류임계 sd %.2f Pa"
         % (MIN_HOLD, THRESH_T, THRESH_SD))

    while True:
        if time.time() - t_start > MAX_RUN:
            emit("최대 감시시간 초과 — 종료")
            return 2

        try:
            rows, win = read_window()
        except OSError as e:
            emit("CSV 읽기 실패(%s) — 재시도" % e)
            time.sleep(POLL)
            continue
        if not rows:
            time.sleep(POLL)
            continue

        age = time.time() - rows[-1][0]
        if age > STALE:
            emit("로거 정지 감지 — 마지막 표본이 %.0fs 전. 중단" % age)
            return 3

        if len(win) < MIN_N:
            emit("창 표본 %d 축적중" % len(win))
            time.sleep(POLL)
            continue

        ts = [r[0] for r in win]
        Ts = [r[1] for r in win]
        dps = [r[2] for r in win]
        bT, _, _ = ols(ts, Ts)
        bD, _, sdD = ols(ts, dps)
        driftT = bT * WINDOW
        driftD = bD * WINDOW
        meanT = sum(Ts) / len(win)
        meanD = sum(dps) / len(win)
        held = (time.time() - t_start) / 60.0

        t_ok = abs(driftT) < THRESH_T
        air_ok = sdD < THRESH_SD
        ok = t_ok and air_ok
        hits = hits + 1 if ok else 0

        flags = []
        if not t_ok:
            flags.append("온도하강중")
        if not air_ok:
            flags.append("기류의심")
        emit("유지 %.1f/%.0f분  dieT=%.2f (%+.3f도/5분)  dp=%.2f (%+.2fPa/5분, sd %.2f)  %s%s"
             % (held, MIN_HOLD, meanT, driftT, meanD, driftD, sdD,
                "/".join(flags) if flags else "안정",
                "  [%d/%d]" % (hits, CONFIRM) if ok else ""))

        if held >= MIN_HOLD and hits >= CONFIRM:
            emit("=== 저온 기준점 확정 ===")
            emit("  다이온도 평균 %.3f 도  (5분 추세 %+.3f 도)" % (meanT, driftT))
            emit("  원시 차압 평균 %.3f Pa  (5분 추세 %+.3f Pa, 잔차sd %.3f)"
                 % (meanD, driftD, sdD))
            emit("  창 표본수 %d,  평균의 표준오차 %.4f Pa" % (len(win), sdD / len(win) ** 0.5))
            return 0

        time.sleep(POLL)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        emit("사용자 중단")
        sys.exit(130)
