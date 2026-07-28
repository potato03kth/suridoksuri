#!/usr/bin/env python3
"""F-9 전용 ulog 프로브 — 위치 setpoint **고도 성분**의 계단을 직접 잰다.

`analyze_run.py` 의 `setpoint_jump` 은 3D 노름이라 F-6(수평 스냅백 214~300m)이
전부 가려버리고, 천이 첫 틱은 속도 setpoint 구간(position=NaN) 뒤의 "스트림 재개
갭"으로 분류돼 위반 집계에서 아예 빠진다. F-9 는 그 재개 틱의 **z 성분 하나**가
본질이므로 따로 잰다.

측정 구간은 **FW 구간**(첫 재개 틱 ~ 두 번째 스트림 갭 = 역천이)으로 자른다 —
HOLD 는 `_mc_pos_ramp` 3D 슬루(v_approach 5.0 m/s = 틱당 0.5m)라 고도 성분이
0.5m 까지 나오고, LANDING 은 PX4 자신의 setpoint 라 우리 코드와 무관하다.

사용: python3 f9_alt_probe.py <run_dir> [<run_dir> ...]
"""
import json
import sys
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


def probe(run_dir: Path) -> dict:
    ulgs = sorted(run_dir.glob("*.ulg"))
    if not ulgs:
        return {"run": run_dir.name, "error": "ulg 없음"}
    ulog = ULog(str(ulgs[-1]))

    sp = topic(ulog, "trajectory_setpoint")
    if sp is None:
        sp = topic(ulog, "vehicle_local_position_setpoint")
    lp = topic(ulog, "vehicle_local_position")
    if sp is None or lp is None:
        return {"run": run_dir.name, "error": "필요 토픽 없음"}

    t_sp = tsec(sp)
    z_key = "position[2]" if "position[2]" in sp else "z"
    z_sp = np.asarray(sp[z_key], dtype=float)      # NED down (음수 = 위)
    x_sp = np.asarray(sp["position[0]" if "position[0]" in sp else "x"], dtype=float)
    y_sp = np.asarray(sp["position[1]" if "position[1]" in sp else "y"], dtype=float)
    good = np.isfinite(z_sp) & np.isfinite(x_sp) & np.isfinite(y_sp)
    t_sp, z_sp, x_sp, y_sp = t_sp[good], z_sp[good], x_sp[good], y_sp[good]
    if len(t_sp) < 3:
        return {"run": run_dir.name, "error": "유한 setpoint 표본 부족"}

    alt_sp = -z_sp                                  # h_up
    t_lp = tsec(lp)
    alt_veh = -np.asarray(lp["z"], dtype=float)

    dt = np.diff(t_sp)
    gaps = np.where(dt > 0.5)[0]
    i0 = int(gaps[0] + 1) if len(gaps) else 0
    # FW 구간 끝 = 두 번째 스트림 갭(역천이) 직전. 없으면 끝까지.
    i1 = int(gaps[1]) if len(gaps) > 1 else len(t_sp) - 1

    t0 = float(t_sp[i0])
    veh0 = float(np.interp(t0, t_lp, alt_veh))
    step0 = float(alt_sp[i0] - veh0)

    # FW 구간의 연속 틱(dt<=0.5s) 고도/수평 변화
    seg = slice(i0, i1 + 1)
    ts, alts = t_sp[seg], alt_sp[seg]
    xs, ys = x_sp[seg], y_sp[seg]
    d = np.diff(ts)
    cont = (d > 1e-6) & (d <= 0.5)
    dalt = np.abs(np.diff(alts))[cont]
    dhor = np.hypot(np.diff(xs), np.diff(ys))[cont]

    cruise = float(np.median(alts[-max(len(alts) // 4, 1):]))
    conv = np.where(np.abs(alts - cruise) <= 0.5)[0]
    alt_conv_s = float(ts[conv[0]] - t0) if len(conv) else None
    m = (t_lp >= t0) & (t_lp <= float(ts[-1])) & (np.abs(alt_veh - cruise) <= 3.0)
    veh_conv_s = float(t_lp[m][0] - t0) if m.any() else None

    def r(v, n=4):
        return None if v is None else round(float(v), n)

    return {
        "run": run_dir.name,
        "fw_window_ulog_s": [round(t0, 2), round(float(ts[-1]), 2)],
        "vehicle_alt_at_resume_m": r(veh0, 3),
        "setpoint_alt_at_resume_m": r(float(alt_sp[i0]), 3),
        "first_sp_alt_step_m": r(step0, 3),
        "fw_max_tick_alt_step_m": r(dalt.max()) if len(dalt) else None,
        "fw_p99_tick_alt_step_m": r(np.percentile(dalt, 99)) if len(dalt) else None,
        "fw_max_tick_horiz_step_m": r(dhor.max()) if len(dhor) else None,
        "fw_n_tick_samples": int(len(dalt)),
        "fw_cruise_alt_sp_m": r(cruise, 3),
        "alt_sp_converge_s": r(alt_conv_s, 2),
        "veh_alt_converge_s": r(veh_conv_s, 2),
    }


if __name__ == "__main__":
    print(json.dumps([probe(Path(a)) for a in sys.argv[1:]],
                     indent=2, ensure_ascii=False))
