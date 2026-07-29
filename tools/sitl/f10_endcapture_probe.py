#!/usr/bin/env python3
"""F-10 전용 ulog 프로브 — FOLLOWING 이 **종점을 포착했는가**를 직접 잰다.

`analyze_run.py` 는 "DONE 에 도달했는가"만 보고, 못 갔을 때 **얼마나 아깝게
놓쳤는지**를 남기지 않는다. C1b 는 그 여백이 전부인 시나리오다 — 종점 포착
조건은 `dist_to_end < d_end_thresh`(기본 10m) 하나뿐인데, FW 는 종점을 지나친
뒤 선회반경(≈110~140m)으로 되돌아오므로 **10m 원을 스치느냐 마느냐가 곧
완주/미완주**다. 그 여백을 재지 않으면 같은 코드가 런마다 완주/타임아웃을
오가는 이유를 설명할 수 없다.

재는 것 (FOLLOWING 상태창 안, ulog `vehicle_local_position` 전 샘플):
  - `min_dist_to_end_m` : 종점까지 수평거리의 최솟값 = **포착 여유**
  - `n_approach`        : 종점 반경 `--near` (기본 60m) 안으로 들어온 횟수(=선회 바퀴수)
  - `approaches`        : 바퀴마다 (시각, 최근접거리, 그때 AGL)
  - `agl_at_first_pass` : 첫 종점 통과 시 AGL — 순항고도와의 차이가 클수록 선회가 커진다
  - `min_agl_m`         : FOLLOWING 중 최저 AGL (음수면 접지)

종점 좌표는 **ulog 안에서** 얻는다 — 노드가 경로 끝을 지나면
`target_point_ned()` 가 `_pts[-1]` 로 클램프되므로, FOLLOWING 중 발행된
`trajectory_setpoint` 의 수평성분이 **한 점에 고정되는 구간**이 곧 종점이다.
waypoints/원점 이동 규약을 다시 구현하지 않아도 되고 프레임 불일치가 원리적으로
없다(둘 다 EKF 로컬 NED).

⚠️ **완료 틱은 FOLLOWING 상태창 밖에 있다.** 상태창 끝은 `TRANSITION_MC` 진입
시각이고, 완료를 만든 그 표본은 그 경계 위/직후에 있다 — 창을 그대로 자르면
완주한 런의 최근접거리가 **높게 편향**된다(실측: A1 이 거리 원(10m)으로 완료했는데
창 안 최솟값은 13.40m 였다). 그래서 창 뒤로 `--tail` 초(기본 1.5s)를 더 본다.
미완주 런은 최솟값이 창 한복판이라 이 보정과 무관하다.

사용: python3 f10_endcapture_probe.py <run_dir> [<run_dir> ...] [--near 60] [--tail 1.5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pyulog import ULog


def topic(ulog, name, multi=0):
    for d in ulog.data_list:
        if d.name == name and d.multi_id == multi:
            return d.data
    return None


def tsec(d):
    return np.asarray(d["timestamp"], dtype=float) / 1e6


def _f(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def _endpoint_xy(sp, t, win) -> tuple[np.ndarray | None, str | None]:
    """FOLLOWING 중 위치 setpoint 가 고정되는 좌표 = 경로 종점."""
    a, b = win
    px = np.asarray(sp["position[0]"], dtype=float)
    py = np.asarray(sp["position[1]"], dtype=float)
    m = (t >= a) & (t <= b) & np.isfinite(px) & np.isfinite(py)
    if np.count_nonzero(m) < 5:
        return None, "FOLLOWING 창에 유한한 위치 setpoint 가 없음"
    P = np.column_stack([px[m], py[m]])
    # 가장 오래 머문 좌표(0.5m 격자 최빈값)를 종점으로 본다.
    key = np.round(P / 0.5).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    top = int(np.argmax(cnt))
    sel = P[inv == top]
    if len(sel) < 3:
        return None, "위치 setpoint 가 한 점에 고정된 구간이 없음(종점 미도달)"
    return sel.mean(axis=0), None


def probe(run_dir: Path, near_m: float, tail_s: float = 1.5) -> dict:
    out = {"run": run_dir.name}
    ulgs = sorted(run_dir.glob("*.ulg"))
    if not ulgs:
        return {**out, "error": "ulg 없음"}
    mf = run_dir / "metrics.json"
    if not mf.exists():
        return {**out, "error": "metrics.json 없음 (analyze_run 먼저)"}
    met = json.loads(mf.read_text(encoding="utf-8"))
    win = (met.get("state_windows_ulog_s") or {}).get("FOLLOWING")
    if not win:
        return {**out, "error": "FOLLOWING 상태창 없음"}

    ulog = ULog(str(ulgs[-1]))
    lp = topic(ulog, "vehicle_local_position")
    sp = topic(ulog, "trajectory_setpoint")
    if lp is None or sp is None:
        return {**out, "error": "vehicle_local_position / trajectory_setpoint 없음"}

    end_xy, err = _endpoint_xy(sp, tsec(sp), win)
    if end_xy is None:
        return {**out, "error": err}

    t = tsec(lp)
    a, b = win
    # 완료 틱은 창 경계 위/직후에 있다 — 그만큼 뒤를 더 본다(위 ⚠️ 참조).
    b_ext = b + float(tail_s)
    m = (t >= a) & (t <= b_ext)
    xy = np.column_stack([lp["x"], lp["y"]]).astype(float)[m]
    z = np.asarray(lp["z"], dtype=float)[m]         # NED z, 아래가 +
    tt = t[m]
    ok = np.isfinite(xy).all(axis=1) & np.isfinite(z)
    xy, z, tt = xy[ok], z[ok], tt[ok]
    if len(xy) < 5:
        return {**out, "error": "FOLLOWING 창 위치 샘플 부족"}

    # 지면 기준: 이 프로브는 metrics.altitude.z_ground_local_m 을 그대로 쓴다.
    z_ground_up = float(((met.get("altitude") or {}).get("z_ground_local_m")) or 0.0)
    agl = (-z) - z_ground_up

    d = np.linalg.norm(xy - end_xy, axis=1)
    i_min = int(np.argmin(d))

    # 바퀴 세기 — near_m 반경에 들어갔다 나오는 구간마다 1회.
    inside = d < near_m
    approaches = []
    i = 0
    n = len(inside)
    while i < n:
        if not inside[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and inside[j + 1]:
            j += 1
        seg = slice(i, j + 1)
        k = i + int(np.argmin(d[seg]))
        approaches.append({
            "t_ulog_s": _f(tt[k]),
            "t_since_following_s": _f(tt[k] - a),
            "min_dist_m": _f(d[k]),
            "agl_m": _f(agl[k]),
        })
        i = j + 1

    return {
        **out,
        "endpoint_local_ne": [_f(end_xy[0]), _f(end_xy[1])],
        "following_window_ulog_s": [_f(a), _f(b)],
        "tail_s": _f(tail_s),
        "following_dwell_s": _f(b - a),
        "d_end_thresh_m": _f((met.get("resolved_params") or {}).get("d_end_thresh")),
        "min_dist_to_end_m": _f(d[i_min]),
        "min_dist_at_ulog_s": _f(tt[i_min]),
        "agl_at_min_dist_m": _f(agl[i_min]),
        "n_approach": len(approaches),
        "near_radius_m": _f(near_m),
        "approaches": approaches,
        "agl_at_first_pass_m": approaches[0]["agl_m"] if approaches else None,
        "min_agl_m": _f(agl.min()),
        "reached_done": bool((met.get("completion") or {}).get("reached_done")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="F-10 종점 포착 여유 프로브")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--near", type=float, default=60.0,
                    help="선회 바퀴를 세는 종점 반경 (m, 기본 60)")
    ap.add_argument("--tail", type=float, default=1.5,
                    help="FOLLOWING 상태창 뒤로 더 볼 시간 (s, 기본 1.5). "
                         "완료 틱이 창 경계 밖이라 필요하다")
    ap.add_argument("--json", action="store_true", help="JSON 만 출력")
    args = ap.parse_args()

    rows = [probe(Path(p), args.near, args.tail) for p in args.run_dirs]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    for r in rows:
        print(f"=== {r['run']}")
        if "error" in r:
            print(f"    ⚠️ {r['error']}")
            continue
        print(f"    완주={r['reached_done']}  FOLLOWING {r['following_dwell_s']}s  "
              f"종점={r['endpoint_local_ne']}")
        print(f"    최근접 {r['min_dist_to_end_m']}m "
              f"(임계 {r['d_end_thresh_m']}m, AGL {r['agl_at_min_dist_m']}m)  "
              f"최저AGL {r['min_agl_m']}m")
        print(f"    종점 반경 {r['near_radius_m']}m 진입 {r['n_approach']}회:")
        for i, ap_ in enumerate(r["approaches"], 1):
            print(f"      {i}) t+{ap_['t_since_following_s']}s  "
                  f"최근접 {ap_['min_dist_m']}m  AGL {ap_['agl_m']}m")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
