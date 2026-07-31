#!/usr/bin/env python3
"""벤치 로그 중간분석 — 오염구간 제외 후 dp ~ 1 + T + t 회귀 (RPi 로컬 실행)

왜 다중회귀인가: 단순 dp~T 는 시간 드리프트(누설/워밍업)에 오염된다.
왜 오염구간 제외인가: 문 개방 드래프트·선풍기·사람 조작은 진짜 Δp 를 만든다.
왜 분할검정인가: T 와 t 가 공선이면 b_T 가 불안정해진다. 반으로 갈라 부호가
                뒤집히면 그 추정은 못 쓴다 (실제로 한 번 뒤집힌 적 있다).
"""
import csv
import statistics as st

CSV = "/home/suri/dpres_bench/latest.csv"

# 오염구간: (시작마커 접두, 종료마커 접두). 종료가 None 이면 다음 마커까지.
DIRTY = [("C4", "X5"),      # 베란다 문 개방 ~ 기체 회전 (드래프트·선풍기·조작 전부 포함)
         ]
SETTLE = 90.0               # 오염구간 종료 후 추가로 버릴 초


def load():
    rows, ev = [], []
    with open(CSV, "r", errors="replace") as f:
        lines = [l for l in f if not l.startswith("#")]
    for r in csv.DictReader(lines):
        try:
            t = float(r["unix_s"])
            rows.append((t, float(r["dp_die_temp_c"]), float(r["dp_raw_pa"])))
            if r.get("event"):
                ev.append((t, r["iso_local"][11:19], r["event"]))
        except (ValueError, TypeError, KeyError):
            continue
    return rows, ev


def mreg(seg):
    """dp ~ 1 + T + t(시간). (b_T, se_T, b_t, se_t, resid_sd, corr(T,t), n)"""
    n = len(seg)
    T = [r[1] for r in seg]
    D = [r[2] for r in seg]
    ts = [(r[0] - seg[0][0]) / 3600.0 for r in seg]
    X = [[1.0, a, b] for a, b in zip(T, ts)]
    A = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(3)] for i in range(3)]
    bv = [sum(X[k][i] * D[k] for k in range(n)) for i in range(3)]
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(3)] for i, row in enumerate(A)]
    for i in range(3):                                    # 가우스-조던 (역행렬 동시)
        p = max(range(i, 3), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        bv[i], bv[p] = bv[p], bv[i]
        d = M[i][i]
        M[i] = [x / d for x in M[i]]
        bv[i] /= d
        for r in range(3):
            if r == i:
                continue
            f = M[r][i]
            M[r] = [a - f * b for a, b in zip(M[r], M[i])]
            bv[r] -= f * bv[i]
    c = bv
    inv = [row[3:] for row in M]
    res = [D[k] - (c[0] + c[1] * T[k] + c[2] * ts[k]) for k in range(n)]
    s2 = sum(e * e for e in res) / (n - 3)
    mT, mt = st.mean(T), st.mean(ts)
    num = sum((a - mT) * (b - mt) for a, b in zip(T, ts))
    den = (sum((a - mT) ** 2 for a in T) * sum((b - mt) ** 2 for b in ts)) ** 0.5
    return (c[1], (s2 * inv[1][1]) ** 0.5, c[2], (s2 * inv[2][2]) ** 0.5,
            s2 ** 0.5, num / den if den else 0.0, n)


def show(lab, seg):
    if len(seg) < 500:
        print("  %-16s 표본부족 (n=%d)" % (lab, len(seg)))
        return
    bT, seT, bt, set_, sd, r, n = mreg(seg)
    T = [x[1] for x in seg]
    print("  %-16s n=%6d  ΔT=%5.2f도  b_T=%+7.3f [%+.3f,%+.3f]  "
          "b_t=%+6.2f Pa/h  corr(T,t)=%+.3f" %
          (lab, n, max(T) - min(T), bT, bT - 1.96 * seT, bT + 1.96 * seT, bt, r))


def main():
    rows, ev = load()
    print("전체 %d 표본, %.1f 분" % (len(rows), (rows[-1][0] - rows[0][0]) / 60))
    print()
    print("마커:")
    for t, hhmmss, e in ev:
        print("  %s  %s" % (hhmmss, e[:72]))

    # 오염구간 산출
    bad = []
    for a, b in DIRTY:
        ta = next((t for t, _, e in ev if e.startswith(a)), None)
        tb = next((t for t, _, e in ev if e.startswith(b)), None)
        if ta is not None:
            bad.append((ta, (tb + SETTLE) if tb else rows[-1][0]))
    clean = [r for r in rows if not any(a <= r[0] <= b for a, b in bad)]
    print()
    print("오염구간 %d 개 제외 -> 청정 %d 표본 (%.0f%%)"
          % (len(bad), len(clean), 100.0 * len(clean) / len(rows)))
    T = [x[1] for x in clean]
    print("청정구간 다이온도 %.2f ~ %.2f 도 (폭 %.2f)" % (min(T), max(T), max(T) - min(T)))

    print()
    print("[다중회귀 dp ~ 1 + T + t]  대괄호는 b_T 의 95%% CI")
    show("청정 전체", clean)
    h = len(clean) // 2
    show("청정 전반", clean[:h])
    show("청정 후반", clean[h:])
    q = len(clean) // 4
    for i in range(4):
        show("4분할 %d" % (i + 1), clean[i * q:(i + 1) * q])

    print()
    print("[참고] 단순회귀 dp ~ T (시간 드리프트 미통제)")
    mx, my = st.mean(T), st.mean([x[2] for x in clean])
    sxx = sum((x - mx) ** 2 for x in T)
    b = sum((x - mx) * (y - my) for x, y in zip(T, [x[2] for x in clean])) / sxx
    print("  b = %+.3f Pa/도" % b)


if __name__ == "__main__":
    main()
