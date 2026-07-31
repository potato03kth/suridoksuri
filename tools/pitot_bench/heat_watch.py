#!/usr/bin/env python3
"""피토 승온 구간 감시자 — 목표온도 도달 후 soak 완료까지 (RPi 로컬 실행)

2단계로 돈다:
  Phase A  다이온도가 TARGET 에 닿을 때까지 대기 (진행상황만 출력)
  Phase B  닿은 뒤 SOAK_MIN 분 경과 + 5분 창 온도추세 < THRESH 가 연속 2회

왜 "다이온도 안정"만으로 끝내면 안 되나:
  강온 실측에서 다이는 14분 만에 평형(추세 -0.13 도/5분)에 들었는데 영점은
  전혀 움직이지 않았다. 다이 평형 != 영점 평형이다. 영점 시정수는 실비행
  로그상 2분보다 훨씬 크고 100분 이하로만 묶여 있다. 그래서 다이가 멈춘
  뒤에도 SOAK_MIN 을 강제로 채운다.

exit code: 0 = soak 완료  2 = 감시시간 초과  3 = 로거 사망  4 = 과열(중단 권고)
"""
import csv
import sys
import time

CSV = "/home/suri/dpres_bench/latest.csv"
TARGET = 30.8         # 도 — 이 온도에 닿으면 soak 시계 시작
OVERHEAT = 43.5       # 도 — 넘으면 즉시 경고 종료
SOAK_MIN = 40.0       # 분 — 온도가 SOAK_BAND 안에 머문 채 유지돼야 하는 시간
SOAK_BAND = 0.5       # 도 — 이 폭 안에 머물러야 '정온'으로 인정
WINDOW = 300.0
THRESH = 0.2          # 도 / 5분 창
THRESH_SD = 2.60      # Pa — 기류 게이트. 청정 기준선 실측 2.08~2.16
STALL_DRIFT = 0.10    # 도/5분 — 이보다 작으면 승온 정체로 계수
STALL_HITS = 20       # 연속 회수 (POLL 30s x 20 = 10분)
MIN_N = 1200
CONFIRM = 2
POLL = 30.0
STALE = 90.0
MAX_RUN = 5 * 3600


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
    t_target = None
    hits = 0
    stall_hits = 0
    band_hist = []
    emit("승온 감시 시작 — 목표 %.1f도, 정온 soak %.0f분(폭 %.1f도 이내), "
         "과열상한 %.1f도, 정체감지 %.0f분"
         % (TARGET, SOAK_MIN, SOAK_BAND, OVERHEAT, STALL_HITS * POLL / 60))

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

        cur_T = rows[-1][1]
        if cur_T > OVERHEAT:
            emit("!!! 과열 %.2f도 > 상한 %.1f도 — 가열 중단 필요 !!!"
                 % (cur_T, OVERHEAT))
            return 4

        if len(win) < MIN_N:
            emit("창 표본 %d 축적중 (dieT=%.2f)" % (len(win), cur_T))
            time.sleep(POLL)
            continue

        ts = [r[0] for r in win]
        Ts = [r[1] for r in win]
        dps = [r[2] for r in win]
        bT, _, sdT = ols(ts, Ts)
        bD, _, sdD = ols(ts, dps)
        driftT = bT * WINDOW
        driftD = bD * WINDOW
        meanT = sum(Ts) / len(win)
        meanD = sum(dps) / len(win)

        if t_target is None:
            if meanT >= TARGET:
                t_target = time.time()
                emit(">>> 목표 %.1f도 도달 (dieT=%.2f) — soak %.0f분 시작 <<<"
                     % (TARGET, meanT, SOAK_MIN))
            else:
                # 정체 감지 — 목표 미달인데 승온이 멈추면 사람이 개입해야 한다.
                # 이게 없으면 감시자가 "승온중"만 몇 시간 찍고 아무도 안 깨어난다.
                if abs(driftT) < STALL_DRIFT:
                    stall_hits += 1
                else:
                    stall_hits = 0
                emit("승온중  dieT=%.2f (%+.3f도/5분)  dp=%.2f (%+.2fPa/5분, sd %.2f)  목표까지 %.2f도%s"
                     % (meanT, driftT, meanD, driftD, sdD, TARGET - meanT,
                        "  [정체 %d/%d]" % (stall_hits, STALL_HITS) if stall_hits else ""))
                if stall_hits >= STALL_HITS:
                    emit("=== 승온 정체 — 목표 %.1f도 미달 %.2f도, %.0f분간 상승 없음 ==="
                         % (TARGET, TARGET - meanT, STALL_HITS * POLL / 60))
                    emit("  현재 dieT=%.3f 도  dp=%.3f Pa (sd %.3f)" % (meanT, meanD, sdD))
                    return 5
                time.sleep(POLL)
                continue

        # soak 은 "목표를 넘은 뒤 40분"이 아니라 "온도가 SOAK_BAND 안에 머문 채 40분"
        # 이어야 한다. 전자로 세면 30분간 상승하다 마지막 10분만 평평해도 완료돼
        # 실제 열평형이 10분뿐인 값을 고온 앵커로 확정해 버린다.
        now_s = time.time()
        band_hist.append((now_s, meanT))
        band_hist[:] = [x for x in band_hist if x[0] >= now_s - SOAK_MIN * 60]
        span_min = (now_s - band_hist[0][0]) / 60.0
        band = max(x[1] for x in band_hist) - min(x[1] for x in band_hist)

        t_ok = abs(driftT) < THRESH
        air_ok = sdD < THRESH_SD          # 기류 게이트: 선풍기/사람 움직임 배제
        band_ok = span_min >= SOAK_MIN and band <= SOAK_BAND
        stable = t_ok and air_ok and band_ok
        hits = hits + 1 if stable else 0

        flags = []
        if not t_ok:
            flags.append("온도미안정")
        if not air_ok:
            flags.append("기류오염")
        if not band_ok:
            flags.append("정온구간 %.1f/%.0f분(폭%.2f도)" % (span_min, SOAK_MIN, band))
        emit("soak  dieT=%.2f (%+.3f도/5분)  dp=%.2f (%+.2fPa/5분, sd %.2f)  %s%s"
             % (meanT, driftT, meanD, driftD, sdD,
                " / ".join(flags) if flags else "정온 %.0f분 폭%.2f도 — 완료조건 충족"
                % (span_min, band),
                "  [%d/%d]" % (hits, CONFIRM) if stable else ""))

        if hits >= CONFIRM:
            emit("=== soak 완료 ===")
            emit("  다이온도 평균 %.3f 도  (5분 추세 %+.3f 도, 잔차sd %.3f)"
                 % (meanT, driftT, sdT))
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
