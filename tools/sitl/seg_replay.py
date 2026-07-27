#!/usr/bin/env python3
"""세그먼트 탐색 수정 전/후 대조 재생기 (SITL-7 R2 게이트 증거용).

**무엇을 하는가**

이미 비행한 런의 *실제 위치 시계열*을 그대로 가져와, 같은 입력에
  ① 2026-07-27 이전 `_find_segment()` — 매 틱 경로 **전역** 최근접 탐색
  ② 수정 후 `_find_segment()` — 직전 인덱스 근방 **창** 탐색 + 역행 제한
두 알고리즘을 나란히 먹여 **세그먼트 인덱스 시계열과 그로부터 나오는 lookahead
목표점**이 어떻게 갈리는지 뽑는다.

**왜 이렇게 하는가**

수정 전/후를 각각 비행시켜 비교하면 궤적 자체가 달라져(제어 루프가 닫혀 있으니
당연하다) "인덱스 차이가 알고리즘 때문인지 궤적 때문인지"를 분리할 수 없다.
이 재생기는 **입력 궤적을 고정**하므로 차이의 원인이 알고리즘 하나로 좁혀진다.
비행 A/B(수정 전 코드로 같은 시나리오를 실제로 날린 런)와 함께 보면
"알고리즘이 다르게 고른다"와 "실비행에서 문제가 났다/안 났다"를 둘 다 말할 수 있다.

**위치 표본 두 가지**
  --positions <csv>  : ulog `vehicle_local_position` 덤프 (t,n,e). 조밀(5Hz+).
                       WSL 안에서 `tools/sitl/dump_positions.py` 로 만든다.
  (기본)             : run/node.log 의 FOLLOWING 진단 로그(2초 간격). 성기지만
                       추가 산출물 없이 기존 런에도 바로 쓸 수 있다.

**경로 재구성**
meta.json 의 `launch_args.waypoints` 로 플래너를 다시 돌린다(같은 파라미터면
결정론적이다). node.log 의 "경로 원점 = 이륙지점 [...]" 줄로 평행이동해
ulog/EKF 로컬 프레임에 맞춘다 — `waypoint_frame=local` 런은 원점 이동이 없다.

사용:
    python3 tools/sitl/seg_replay.py --run logs/<...>/B5_pxvehicle
    python3 tools/sitl/seg_replay.py --run <dir> --positions <dir>/positions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fc_bridge.guidance.l1_guidance import L1Guidance  # noqa: E402
from fc_bridge.planning.planner_runner import (  # noqa: E402
    run_planner, resolve_planner_name)

# offboard_node 의 FW lookahead 상수와 같아야 한다 (import 하면 rclpy 를 끌고 온다)
FW_LOOKAHEAD = 70.0
# 노드 진단(`_SEG_JUMP_WARN`)과 같은 임계 — "경로상 전진"이 아니라고 볼 점프 폭
JUMP_THRESH = 20

RE_ORIGIN = re.compile(
    r"경로 원점 = 이륙지점 \[N=(-?[\d.]+), E=(-?[\d.]+), h_up=(-?[\d.]+)\]")
RE_FOLLOW_TICK = re.compile(
    r"FOLLOWING tick=(\d+) mode=(\S*) cte=(-?[\d.]+)m "
    r"pos=\[(-?[\d.]+),(-?[\d.]+)\]")
RE_T = re.compile(r"^\[[^\]]+\]\s+\[\w+\]\s+\[(\d+\.\d+)\]")


class _OldFindSegment:
    """2026-07-27 이전 `L1Guidance._find_segment()` — 전역 최근접 탐색.

    당시 코드 그대로다: 캐시 `_seg_idx` 는 갱신만 되고 탐색에는 쓰이지 않아
    매 틱 경로 전체에서 가장 가까운 세그먼트를 고른다. 동점이면 작은 인덱스.
    """

    def __init__(self, pts):
        self.pts = np.asarray(pts, dtype=float)
        self.seg_idx = 0

    def find(self, p2) -> int:
        pts, p2 = self.pts, np.asarray(p2, dtype=float)[:2]
        if len(pts) < 2:
            return 0
        a = pts[:-1]
        b = pts[1:]
        ab = b - a
        ab2 = np.einsum("ij,ij->i", ab, ab)
        safe = ab2 > 1e-12
        t = np.zeros(len(a))
        t[safe] = np.einsum("ij,ij->i", p2 - a[safe], ab[safe]) / ab2[safe]
        np.clip(t, 0.0, 1.0, out=t)
        d = p2 - (a + t[:, None] * ab)
        i = int(np.argmin(np.einsum("ij,ij->i", d, d)))
        self.seg_idx = i
        return i


def _lookahead_point(pts, p2, start_seg, dist):
    """`L1Guidance._lookahead_point()` 와 동일한 계산 (전 구현 재생용)."""
    n = len(pts)
    remain = float(dist)
    for i in range(int(start_seg), n - 1):
        a, b = pts[i], pts[i + 1]
        seg_len = float(np.linalg.norm(b - a))
        if seg_len < 1e-9:
            continue
        ab = b - a
        t = float(np.clip((p2 - a) @ ab / (seg_len ** 2), 0.0, 1.0))
        along_remain = seg_len * (1.0 - t)
        if along_remain >= remain:
            return a + (t + remain / seg_len) * ab
        remain -= along_remain
    return pts[-1].copy()


def load_positions_from_nodelog(path: Path):
    """node.log 의 FOLLOWING 진단 로그에서 (t, N, E) 를 뽑는다 (2초 간격)."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_FOLLOW_TICK.search(line)
            if not m:
                continue
            mt = RE_T.match(line.strip())
            t = float(mt.group(1)) if mt else float(len(out))
            out.append((t, float(m.group(4)), float(m.group(5))))
    return out


def load_positions_csv(path: Path):
    out = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append((float(row["t"]), float(row["n"]), float(row["e"])))
    return out


def path_origin_from_nodelog(path: Path):
    if not path.exists():
        return np.zeros(2)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_ORIGIN.search(line)
            if m:
                return np.array([float(m.group(1)), float(m.group(2))])
    return np.zeros(2)


def build_path(meta: dict, cache: Path | None):
    if cache is not None and cache.exists():
        return np.load(cache)
    la = meta.get("launch_args") or {}
    wtxt = la.get("waypoints")
    if not wtxt:
        raise SystemExit(
            "meta.json 에 launch_args.waypoints 가 없다 — 경로를 재구성할 수 없다")
    vals = [float(x) for x in wtxt.strip().strip("[]").replace(",", " ").split()]
    wps = np.array(vals, dtype=float).reshape(-1, 3)
    planner = resolve_planner_name(la.get("planner", "auto"),
                                   la.get("vehicle_type", "vtol"))
    vp = {"v_cruise": float(la.get("v_cruise", 20.0)),
          "a_max_g": float(la.get("a_max_g", 0.3)),
          "gravity": float(la.get("gravity", 9.81))}
    print(f"[seg_replay] 플래너 재실행 planner={planner} WP={len(wps)} "
          f"v_cruise={vp['v_cruise']} (폐곡선이면 수 분 걸린다)", flush=True)
    path = run_planner(planner, wps, vp)
    pts = np.array([p.pos[:2] for p in path.points], dtype=float)
    if cache is not None:
        np.save(cache, pts)
    return pts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="런 디렉터리 (meta.json/node.log)")
    ap.add_argument("--positions", default=None, help="위치 CSV (t,n,e)")
    ap.add_argument("--out", default=None, help="출력 디렉터리 (기본: 런 디렉터리)")
    ap.add_argument("--lookahead", type=float, default=FW_LOOKAHEAD)
    args = ap.parse_args()

    run = Path(args.run)
    out_dir = Path(args.out) if args.out else run
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))

    pts = build_path(meta, out_dir / "replay_path_pts.npy")
    origin = path_origin_from_nodelog(run / "node.log")
    frame = (meta.get("launch_args") or {}).get("waypoint_frame", "takeoff")
    if str(frame).lower() != "local":
        pts = pts + origin
    else:
        origin = np.zeros(2)

    if args.positions:
        samples = load_positions_csv(Path(args.positions))
        src = f"csv:{Path(args.positions).name}"
    else:
        samples = load_positions_from_nodelog(run / "node.log")
        src = "node.log(FOLLOWING 2s)"
    if not samples:
        print("위치 표본이 없다", file=sys.stderr)
        return 4

    old = _OldFindSegment(pts)
    new = L1Guidance(l1_dist=20.0, path_pts=pts,
                     v_profile=np.full(len(pts), 20.0))

    rows = []
    for t, n, e in samples:
        p2 = np.array([n, e])
        i_old = old.find(p2)
        i_new = new._find_segment(p2)
        new._seg_idx = i_new
        tgt_old = _lookahead_point(pts, p2, i_old, args.lookahead)
        tgt_new = _lookahead_point(pts, p2, i_new, args.lookahead)
        rows.append({
            "t": t, "n": n, "e": e,
            "seg_old": i_old, "seg_new": i_new,
            "seg_diff": i_new - i_old,
            "tgt_old_n": tgt_old[0], "tgt_old_e": tgt_old[1],
            "tgt_new_n": tgt_new[0], "tgt_new_e": tgt_new[1],
            "tgt_dist_m": float(np.linalg.norm(tgt_new - tgt_old)),
        })

    so = np.array([r["seg_old"] for r in rows])
    sn = np.array([r["seg_new"] for r in rows])
    do, dn = np.diff(so), np.diff(sn)
    td = np.array([r["tgt_dist_m"] for r in rows])

    summary = {
        "run": run.name,
        "n_samples": len(rows),
        "position_source": src,
        "n_path_points": int(len(pts)),
        "path_origin_ne": [float(origin[0]), float(origin[1])],
        "waypoint_frame": frame,
        "lookahead_m": args.lookahead,
        "jump_threshold_points": JUMP_THRESH,
        "old": {
            "n_backward": int(np.count_nonzero(do < 0)),
            "max_backward": int(do.min()) if len(do) else 0,
            "max_forward": int(do.max()) if len(do) else 0,
            "n_jumps_over_thresh": int(np.count_nonzero(np.abs(do) > JUMP_THRESH)),
        },
        "new": {
            "n_backward": int(np.count_nonzero(dn < 0)),
            "max_backward": int(dn.min()) if len(dn) else 0,
            "max_forward": int(dn.max()) if len(dn) else 0,
            "n_jumps_over_thresh": int(np.count_nonzero(np.abs(dn) > JUMP_THRESH)),
        },
        "n_ticks_index_differs": int(np.count_nonzero(so != sn)),
        "n_ticks_index_differs_over_thresh": int(
            np.count_nonzero(np.abs(so - sn) > JUMP_THRESH)),
        "max_target_divergence_m": float(td.max()) if len(td) else 0.0,
        "n_ticks_target_differs_over_1m": int(np.count_nonzero(td > 1.0)),
        "worst_ticks": sorted(rows, key=lambda r: -r["tgt_dist_m"])[:10],
    }

    csv_path = out_dir / "seg_replay.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path = out_dir / "seg_replay.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[seg_replay] {run.name}: 표본 {len(rows)} ({src}) — "
          f"인덱스 상이 {summary['n_ticks_index_differs']}틱, "
          f"임계초과 상이 {summary['n_ticks_index_differs_over_thresh']}틱, "
          f"목표점 최대 괴리 {summary['max_target_divergence_m']:.1f}m")
    print(f"[seg_replay] old: 역행 {summary['old']['n_backward']}회 / "
          f"급변 {summary['old']['n_jumps_over_thresh']}회  |  "
          f"new: 역행 {summary['new']['n_backward']}회 / "
          f"급변 {summary['new']['n_jumps_over_thresh']}회")
    print(f"[seg_replay] → {csv_path}, {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
