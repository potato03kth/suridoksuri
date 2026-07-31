#!/usr/bin/env python3
"""dpres_log.py CSV 요약 — 구간 통계 + 다이온도 vs raw 영점 회귀.

    python3 dpres_stats.py LOG.csv [SENS_DPRES_OFF]
"""
import csv, statistics, sys

path = sys.argv[1]
off = float(sys.argv[2]) if len(sys.argv) > 2 else None
rows = [r for r in csv.DictReader(l for l in open(path) if not l.startswith('#'))]
if not rows:
    sys.exit("데이터 행 없음")

tcol = 'dp_die_temp_c' if 'dp_die_temp_c' in rows[0] else 'dp_temp_c'
data = [r for r in rows if not r.get('event', '').strip('"')]
dp = [float(r['dp_raw_pa']) for r in data]
t = [float(r[tcol]) for r in data]
bt = [float(r['baro_temp_c']) for r in data]
dur = float(data[-1]['unix_s']) - float(data[0]['unix_s'])

print(f"n={len(data)}  duration={dur/60:.1f} min  rate={len(data)/dur:.2f} Hz")
print(f"dp_raw    mean={statistics.mean(dp):+8.3f} Pa  sd={statistics.pstdev(dp):.3f}"
      f"  se={statistics.pstdev(dp)/len(dp)**0.5:.4f}  [{min(dp):+.3f} .. {max(dp):+.3f}]")
print(f"die_temp  mean={statistics.mean(t):7.3f} C   [{min(t):.2f} .. {max(t):.2f}]  "
      f"span={max(t)-min(t):.2f} C")
print(f"baro_temp mean={statistics.mean(bt):7.3f} C   [{min(bt):.2f} .. {max(bt):.2f}]")
q = sorted(set(round(x, 3) for x in dp))
print(f"dp 양자화 계단 {len(q)}종, 최소간격={min(b-a for a, b in zip(q, q[1:])) if len(q)>1 else 0:.3f} Pa")
if off is not None:
    print(f"SENS_DPRES_OFF={off:+.4f} -> 보정후 평균 dp = {statistics.mean(dp)-off:+.3f} Pa "
          f"(무풍이면 0 이어야 함)")

if max(t) - min(t) > 0.5:
    n = len(dp)
    mt, md = statistics.mean(t), statistics.mean(dp)
    sxx = sum((x-mt)**2 for x in t)
    sxy = sum((x-mt)*(y-md) for x, y in zip(t, dp))
    a = sxy/sxx
    b = md - a*mt
    print(f"\n▶ 회귀  dp_raw = {a:+.4f} Pa/°C * die_T {b:+.3f} Pa")
    res = [y-(a*x+b) for x, y in zip(t, dp)]
    print(f"  잔차 sd={statistics.pstdev(res):.3f} Pa,  "
          f"온도구간 {max(t)-min(t):.2f}°C 에서 예측 영점이동 {a*(max(t)-min(t)):+.2f} Pa")
else:
    print("\n(온도 변화폭 0.5°C 미만 -> 회귀 생략. 냉/온 구간을 만들어야 기울기가 나온다)")

ev = [r for r in rows if r.get('event', '').strip('"')]
if ev:
    print(f"\n마커 {len(ev)}건:")
    for r in ev:
        print(f"  {r['iso_local'][:19]}  dp={float(r['dp_raw_pa']):+8.3f} Pa  "
              f"T={float(r[tcol]):6.2f} C   {r['event']}")
