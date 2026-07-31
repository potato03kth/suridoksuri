#!/usr/bin/env python3
"""기존 ulog 에서 (MS4525DO 다이온도, raw 차압) 쌍을 뽑아 회귀한다.
벤치 실험 전에 "이미 가진 데이터로 어디까지 말할 수 있는가"를 보는 도구.

    python3 ulog_dpres_temp.py "logs/2026-07-*/*.ulg" [개시초=10]

주의: 로깅은 SDLOG_MODE=0 이라 **아밍 후**에만 시작된다. 개시 직후 구간도
프롭이 돌고 있어 프롭워시가 실린 진짜 차압이 섞일 수 있다 — 교란 요인.
"""
import glob, sys
import numpy as np
from pyulog import ULog

pat = sys.argv[1] if len(sys.argv) > 1 else "logs/2026-07-*/*.ulg"
win = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

pts = []
for p in sorted(glob.glob(pat)):
    if p.endswith("_1.ulg"):
        continue
    try:
        u = ULog(p, ["differential_pressure"])
        d = u.data_list[0]
    except Exception as e:
        print(f"skip {p}: {e}")
        continue
    t = d.data["timestamp"] / 1e6
    dp = d.data["differential_pressure_pa"]
    T = d.data["temperature"]
    m = (t - t[0]) < win
    if m.sum() >= 1:
        pts.append((T[m].mean(), dp[m].mean(), p,
                    u.initial_parameters.get("SENS_DPRES_OFF", float("nan"))))

a = np.array([(x, y) for x, y, _, _ in pts])
print(f"n={len(a)}  die T {a[:,0].min():.1f}~{a[:,0].max():.1f} °C  "
      f"dp_raw {a[:,1].min():+.1f}~{a[:,1].max():+.1f} Pa")
s, b = np.polyfit(a[:, 0], a[:, 1], 1)
r = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
print(f"▶ dp_raw = {s:+.3f} Pa/°C * dieT {b:+.2f} Pa   (r²={r*r:.3f}, "
      f"잔차 sd={np.std(a[:,1]-(s*a[:,0]+b)):.2f} Pa)")
offs = sorted({round(o, 4) for *_, o in pts})
print(f"이 표본에 등장한 SENS_DPRES_OFF: {offs}")
