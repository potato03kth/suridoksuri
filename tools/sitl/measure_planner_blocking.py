#!/usr/bin/env python3
"""플래너 블로킹 시간 실측기 — `v_cruise` 3점 비교 (F-12 / §4 결정 4).

**왜 SITL 런이 아니라 이 스크립트인가.**
`OffboardNode.__init__` 이 블로킹되는 구간은 정확히 한 호출이다:

    path = run_planner(planner_name, raw_wps, vehicle_params)   # offboard_node.py:319

이 호출은 ROS2·MAVROS·gz 와 아무 상호작용이 없는 **순수 계산**이다(입력은
경로점·`v_cruise`·`a_max_g`·`gravity` 뿐). 따라서 같은 인자로 여기서 재면
`__init__` 블로킹과 같은 값이 나온다 — 실제로 SITL 런의 `경로 계획 완료 — 소요
Xs` 로그와 교차확인한다. SITL 로 3점을 재려면 런당 5~11분이 더 들고
(비행 구간은 `v_cruise` 와 무관하다) 폐회로 20.0 은 그중 3분 25초가 순수 대기다.

사용:
    python3 tools/sitl/measure_planner_blocking.py                  # 기본 2경로 × 3속도
    python3 tools/sitl/measure_planner_blocking.py --repeat 2
    python3 tools/sitl/measure_planner_blocking.py --json out.json

출력은 표 + (옵션) JSON. 이 저장소의 다른 하니스와 달리 SITL 이 필요 없으므로
호스트/WSL 어디서나 돈다. 다만 **캠페인 수치와 나란히 놓으려면 캠페인과 같은
환경(WSL Ubuntu-22.04)에서 재야 한다** — CPU 가 다르면 절대값이 달라진다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fc_bridge.planning.planner_runner import (  # noqa: E402
    planner_eta_s,
    resolve_planner_name,
    run_planner,
)

# 경로는 tools/sitl/scenarios.yaml 의 해당 시나리오와 **글자 그대로 같아야** 한다.
PATHS: dict[str, list[float]] = {
    # R2_a3 / A3 — L자(90° 1회), 3WP
    "A3_L": [0.0, 0.0, 50.0, 200.0, 0.0, 50.0, 200.0, 200.0, 50.0],
    # R2_closed — 완전 폐회로(종점 = 시점), 5WP
    "R2_closed": [0.0, 0.0, 50.0, 200.0, 0.0, 50.0, 200.0, 200.0, 50.0,
                  0.0, 200.0, 50.0, 0.0, 0.0, 50.0],
    # R2_b5 / B5 — 사각 폐곡선(종점 20m 벌림), 5WP
    "B5_square": [0.0, 0.0, 50.0, 200.0, 0.0, 50.0, 200.0, 200.0, 50.0,
                  0.0, 200.0, 50.0, 0.0, 20.0, 50.0],
    # R2_a1 / A1 — 직선 300m, 2WP (플래너 NR 우회 → 0s 기대)
    "A1_line": [0.0, 0.0, 50.0, 300.0, 0.0, 50.0],
    # R2_b7 / B7 — 단거리 40m, 2WP
    "B7_short": [0.0, 0.0, 50.0, 40.0, 0.0, 50.0],
}

DEFAULT_PATHS = ("A3_L", "R2_closed")
DEFAULT_VCRUISE = (15.0, 18.0, 20.0)


def measure(path_key: str, v_cruise: float, a_max_g: float, gravity: float,
            vehicle_type: str = "vtol") -> dict:
    """한 (경로, v_cruise) 조합의 블로킹 시간을 잰다."""
    wps = np.array(PATHS[path_key], dtype=float).reshape(-1, 3)
    planner_name = resolve_planner_name("auto", vehicle_type)
    vehicle_params = {"v_cruise": v_cruise,
                      "a_max_g": a_max_g, "gravity": gravity}

    t0 = time.monotonic()
    path = run_planner(planner_name, wps, vehicle_params)
    dt = time.monotonic() - t0

    return {
        "path": path_key,
        "n_wp": int(wps.shape[0]),
        "v_cruise": v_cruise,
        "planner": planner_name,
        "blocking_s": round(dt, 3),
        "eta_s": planner_eta_s(planner_name, wps.shape[0]),
        "n_points": len(path.points),
        "total_length_m": round(float(path.total_length), 2),
        "planning_time_s": round(float(getattr(path, "planning_time", float("nan"))), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", nargs="+", default=list(DEFAULT_PATHS),
                    choices=sorted(PATHS), help="측정할 경로 키")
    ap.add_argument("--v-cruise", nargs="+", type=float,
                    default=list(DEFAULT_VCRUISE))
    ap.add_argument("--a-max-g", type=float, default=0.3)
    ap.add_argument("--gravity", type=float, default=9.81)
    ap.add_argument("--repeat", type=int, default=1,
                    help="같은 조합 반복 횟수 (분산 확인용). 표에는 중앙값을 쓴다")
    ap.add_argument("--json", default=None, help="원본 결과 JSON 저장 경로")
    args = ap.parse_args()

    rows: list[dict] = []
    for pk in args.paths:
        for vc in args.v_cruise:
            samples = []
            for i in range(max(1, args.repeat)):
                r = measure(pk, vc, args.a_max_g, args.gravity)
                r["rep"] = i
                samples.append(r)
                print(f"  [{pk} v_cruise={vc} rep{i}] "
                      f"blocking={r['blocking_s']:.3f}s "
                      f"points={r['n_points']} len={r['total_length_m']}m",
                      flush=True)
            med = float(np.median([s["blocking_s"] for s in samples]))
            base = dict(samples[0])
            base["blocking_s"] = round(med, 3)
            base["samples_s"] = [s["blocking_s"] for s in samples]
            base.pop("rep", None)
            rows.append(base)

    print("\n| 경로 | WP | v_cruise | 블로킹(s) | 경로점 | 전장(m) | eta 예고(s) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['path']} | {r['n_wp']} | {r['v_cruise']:.1f} | "
              f"**{r['blocking_s']:.2f}** | {r['n_points']} | "
              f"{r['total_length_m']:.1f} | {r['eta_s']:.0f} |")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"\nJSON 저장: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
