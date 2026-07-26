#!/usr/bin/env python3
"""SITL-7 S5 — 경로 기하 분석 (코너 오버슈트 / WP 최근접 / 표류 / 선회중 고도).

    python3 pathan.py <run_dir>

meta.json 의 launch_args.waypoints 와 metrics.json 의 상태창을 읽어
ulog vehicle_local_position 궤적에 대해 계산한다. 판정하지 않고 수치만 낸다.
"""
import json
import sys
from pathlib import Path

import numpy as np
from pyulog import ULog

run = Path(sys.argv[1])
meta = json.load(open(run / "meta.json"))
met = json.load(open(run / "metrics.json"))

wp_str = meta["launch_args"]["waypoints"].strip()[1:-1]
vals = [float(v) for v in wp_str.split(",")]
WP = np.array(vals).reshape(-1, 3)  # N, E, up
print(f"waypoints (N,E,up) x{len(WP)}:")
for i, w in enumerate(WP):
    print(f"  WP{i} = [{w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f}]")

ulog_name = met.get("ulog_file") or meta["ulogs"][0]
u = ULog(str(run / Path(ulog_name).name), ["vehicle_local_position"])
d = u.data_list[0]
t = np.array(d.data["timestamp"]) / 1e6
x = np.array(d.data["x"])
y = np.array(d.data["y"])
z = np.array(d.data["z"])
agl = -z - met["altitude"]["z_ground_local_m"]

win = met["state_windows_ulog_s"]


def seg(t0, t1):
    m = (t >= t0) & (t <= t1)
    return t[m], x[m], y[m], agl[m]


# ── 1. 헤딩 정렬 구간 표류 ────────────────────────────────────────────────
hw = met["heading"]["window_ulog_s"]
t_al = met["heading"].get("t_align_ulog_s")
if t_al:
    ts, xs, ys, _ = seg(hw[0], t_al)
    if len(ts):
        drift = np.hypot(xs - xs[0], ys - ys[0])
        print(f"\n[헤딩 정렬 구간 표류] ulog {ts[0]:.2f}~{ts[-1]:.2f}s ({ts[-1]-ts[0]:.2f}s)")
        print(f"  시작 pos=[{xs[0]:.2f},{ys[0]:.2f}] 종료 pos=[{xs[-1]:.2f},{ys[-1]:.2f}]")
        print(f"  시작점 대비 최대 이동 {drift.max():.2f} m / 종료 시 {drift[-1]:.2f} m")

# ── 2. FOLLOWING 구간 (경로 추종 본체) ────────────────────────────────────
fw = win.get("FOLLOWING")
if not fw:
    print("\nFOLLOWING 창 없음 — 이하 생략")
    sys.exit(0)
tf, xf, yf, af = seg(fw[0], fw[1])
print(f"\n[FOLLOWING 구간] ulog {fw[0]:.2f}~{fw[1]:.2f}s, 샘플 {len(tf)}")
print(f"  N 범위 {xf.min():.2f} ~ {xf.max():.2f} / E 범위 {yf.min():.2f} ~ {yf.max():.2f}")
print(f"  AGL 최소 {af.min():.2f} @t={tf[af.argmin()]:.2f}s pos=[{xf[af.argmin()]:.1f},{yf[af.argmin()]:.1f}]"
      f" / 최대 {af.max():.2f} @t={tf[af.argmax()]:.2f}s pos=[{xf[af.argmax()]:.1f},{yf[af.argmax()]:.1f}]")

# ── 3. 각 원본 WP 최근접 (FOLLOWING 구간 기준) ────────────────────────────
print("\n[원본 WP 최근접 — FOLLOWING 구간]")
for i, w in enumerate(WP):
    dd = np.hypot(xf - w[0], yf - w[1])
    j = dd.argmin()
    print(f"  WP{i} [{w[0]:.0f},{w[1]:.0f}] → 최근접 {dd[j]:.2f} m @t={tf[j]:.2f}s pos=[{xf[j]:.1f},{yf[j]:.1f}]")

# ── 4. 코너 오버슈트 ──────────────────────────────────────────────────────
# 코너 i (1..N-2): 입력 레그 방향 u_in 을 코너 너머로 연장한 초과 거리의 최대값.
print("\n[코너 오버슈트 — 입력 레그 진행축 기준 코너 초과량]")
if len(WP) < 3:
    print("  코너 없음 (2WP 직선)")
for i in range(1, len(WP) - 1):
    P = WP[i][:2]
    A = WP[i - 1][:2]
    B = WP[i + 1][:2]
    u_in = (P - A) / np.linalg.norm(P - A)
    u_out = (B - P) / np.linalg.norm(B - P)
    ang = np.degrees(np.arccos(np.clip(np.dot(u_in, u_out), -1, 1)))
    rel = np.stack([xf - P[0], yf - P[1]], axis=1)
    along = rel @ u_in           # 코너 통과 진행축 성분
    prog_out = rel @ u_out       # 출력 레그 진행 성분
    n_out = np.array([-u_out[1], u_out[0]])   # 진행방향 오른쪽
    lat_out = rel @ n_out
    cross = u_in[0] * u_out[1] - u_in[1] * u_out[0]
    right_turn = cross > 0
    # 선회 바깥쪽 = 우선회면 왼쪽(lat_out 음), 좌선회면 오른쪽(lat_out 양)
    outer = -lat_out if right_turn else lat_out
    # 코너 근방창: 출력 레그로 -60~+120m
    dcorner = np.hypot(rel[:, 0], rel[:, 1])
    m = (prog_out > -60) & (prog_out < 120) & (dcorner < 130)
    print(f"  코너 WP{i} [{P[0]:.0f},{P[1]:.0f}] 꺾임각 {ang:.1f}° ({'우선회' if right_turn else '좌선회'})")
    if not m.any():
        print("    해당 구간 샘플 없음")
        continue
    idx = np.where(m)[0]
    ko = idx[np.argmax(outer[idx])]
    print(f"    코너 오버슈트(출력레그 직선 바깥쪽 최대 이탈) {max(0.0, outer[ko]):.3f} m"
          f" @t={tf[ko]:.2f}s pos=[{xf[ko]:.1f},{yf[ko]:.1f}]")
    ki = idx[np.argmin(outer[idx])]
    print(f"    안쪽 최대(코너 컷) {max(0.0, -outer[ki]):.3f} m @t={tf[ki]:.2f}s")
    # 90° 코너에서 S4 corner.py 와 동일한 값: 입력축 초과 (코너 근방 한정)
    mn = (prog_out > -30) & (prog_out < 60)
    if mn.any():
        j = np.where(mn)[0][np.argmax(along[mn])]
        print(f"    (참고) 입력 진행축 초과 {max(0.0, along[j]):.3f} m @t={tf[j]:.2f}s")

# ── 5. 종점 근접 ─────────────────────────────────────────────────────────
end = WP[-1][:2]
dend = np.hypot(xf - end[0], yf - end[1])
print(f"\n[종점] WP{len(WP)-1} 까지 FOLLOWING 마지막 샘플 거리 {dend[-1]:.2f} m, 최소 {dend.min():.2f} m")
