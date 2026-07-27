#!/usr/bin/env python3
"""ulog `vehicle_local_position` → (t, n, e) CSV 덤프 (WSL 안에서 실행).

`tools/sitl/seg_replay.py` 에 조밀한(5Hz+) 위치 표본을 주기 위한 것이다.
node.log 의 FOLLOWING 진단 로그는 2초 간격이라 세그먼트 인덱스 시계열을
보기엔 성기다 — .ulg 는 대용량이라 WSL 안에만 두므로(캠페인 규약) 여기서
CSV 로 줄여 내보낸다.

t 는 ulog boot 시각(초)이고 x/y 는 EKF 로컬 NED 의 북/동이다.
`seg_replay.py` 가 쓰는 경로도 같은 프레임(node.log 의 "경로 원점" 으로
평행이동)이라 그대로 대응된다.

사용:
    python3 tools/sitl/dump_positions.py <run_dir> [--hz 5] [--out positions.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from pyulog import ULog


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    meta_p = run / "meta.json"
    names = []
    if meta_p.exists():
        names = json.loads(meta_p.read_text(encoding="utf-8")).get("ulogs") or []
    cand = [run / n for n in reversed(names) if (run / n).exists()]
    if not cand:
        cand = sorted(run.glob("*.ulg"))
    if not cand:
        print(f"ulg 없음: {run}", file=sys.stderr)
        return 4
    ulg = ULog(str(cand[0]), ["vehicle_local_position"])
    d = ulg.data_list[0].data
    t = np.asarray(d["timestamp"], dtype=float) / 1e6
    n = np.asarray(d["x"], dtype=float)
    e = np.asarray(d["y"], dtype=float)
    good = np.isfinite(n) & np.isfinite(e)
    t, n, e = t[good], n[good], e[good]

    step = 1.0 / float(args.hz)
    keep, last = [], -1e9
    for i, ti in enumerate(t):
        if ti - last >= step:
            keep.append(i)
            last = ti
    out = Path(args.out) if args.out else run / "positions.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "n", "e"])
        for i in keep:
            w.writerow([f"{t[i]:.3f}", f"{n[i]:.3f}", f"{e[i]:.3f}"])
    print(f"[dump_positions] {cand[0].name}: {len(keep)} rows @ {args.hz}Hz -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
