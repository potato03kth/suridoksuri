#!/usr/bin/env python3
"""`v_cruise` 별 플래너 경로 **형상** 대조 — SITL 노이즈 없이 순수 비교.

SITL 런은 바람·EKF·PX4 내부 상태 때문에 같은 조건에서도 cte 가 런마다 흔들린다
(캠페인 실측 대역 참조). 반면 `run_planner()` 는 결정적이므로 **경로 형상이
바뀌었는지 아닌지는 여기서 정확히 답할 수 있다** — 바뀌었다면 SITL 차이의
원인 후보가 되고, 안 바뀌었다면 SITL 차이는 전부 런간 변동이다.

기준 v_cruise 대비 각 후보의 (a) 경로점 수 (b) 전장 (c) 호길이 정규화 후
점대점 최대/평균 거리 를 낸다.

    python3 tools/sitl/compare_planner_paths.py --paths A3_L R2_closed \
        --ref 20 --cand 15 18
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.sitl.measure_planner_blocking import PATHS  # noqa: E402
from fc_bridge.planning.planner_runner import run_planner  # noqa: E402


def build(path_key: str, v_cruise: float) -> tuple[np.ndarray, np.ndarray, float]:
    wps = np.array(PATHS[path_key], dtype=float).reshape(-1, 3)
    p = run_planner("eta3", wps, {"v_cruise": v_cruise,
                                  "a_max_g": 0.3, "gravity": 9.81})
    xy = np.array([pt.pos[:2] for pt in p.points], dtype=float)
    s = np.array([pt.s for pt in p.points], dtype=float)
    return xy, s, float(p.total_length)


def resample(xy: np.ndarray, s: np.ndarray, n: int = 2000) -> np.ndarray:
    """호길이 정규화(0~1) 후 균등 재표본 — 점 개수가 달라도 비교 가능하게."""
    u = (s - s[0]) / max(s[-1] - s[0], 1e-9)
    g = np.linspace(0.0, 1.0, n)
    return np.column_stack([np.interp(g, u, xy[:, 0]), np.interp(g, u, xy[:, 1])])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", default=["A3_L", "R2_closed"],
                    choices=sorted(PATHS))
    ap.add_argument("--ref", type=float, default=20.0)
    ap.add_argument("--cand", nargs="+", type=float, default=[15.0, 18.0])
    args = ap.parse_args()

    print("| 경로 | v_cruise | 경로점 | 전장(m) | "
          f"ref={args.ref:g} 대비 최대괴리(m) | 평균괴리(m) |")
    print("|---|---|---|---|---|---|")
    for pk in args.paths:
        rxy, rs, rlen = build(pk, args.ref)
        rr = resample(rxy, rs)
        print(f"| {pk} | {args.ref:g} (기준) | {len(rxy)} | {rlen:.2f} | — | — |")
        for vc in args.cand:
            cxy, cs, clen = build(pk, vc)
            cc = resample(cxy, cs)
            d = np.linalg.norm(cc - rr, axis=1)
            print(f"| {pk} | {vc:g} | {len(cxy)} | {clen:.2f} | "
                  f"**{d.max():.3f}** | {d.mean():.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
