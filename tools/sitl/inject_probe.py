#!/usr/bin/env python3
"""장애주입 런 전용 ulog 검사기 — "주입이 실제로 먹혔는가" 를 답한다.

`analyze_run.py` 는 캠페인 4장 지표(cte·고도편차·천이 등)를 낸다. 이 스크립트는
그것과 겹치지 않는 **세 가지 사실**만 ulog 원본에서 뽑는다.

  1. `vehicle_status.nav_state` 구간 — 주입한 모드로 **실제로 바뀌었는지**.
     `/mavros/set_mode` 서비스의 `mode_sent: true` 는 "보냈다"일 뿐 "바뀌었다"가 아니다.
  2. `offboard_control_mode` / `trajectory_setpoint` 의 **마지막 발행 시각과 공백** —
     OVERRIDE·PILOT_TAKEOVER 의 핵심 안전 속성인 "setpoint 발행 중단" 판정 근거.
     (offboard_control_mode 는 MAVROS 가 setpoint 를 중계할 때만 나온다 →
      이 토픽이 끊긴 시점 = 우리 노드가 손을 뗀 시점.)
  3. `vehicle_command` 중 DO_SET_MODE(176) — 주입 명령이 FCU 에 도달한 시각.

사용:  python3 tools/sitl/inject_probe.py <run_dir>
출력:  JSON (stdout)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from pyulog import ULog

PX4_MSG_CANDIDATES = [Path("/root/PX4-vehicle/msg/versioned/VehicleStatus.msg"),
                      Path("/root/PX4-vehicle/msg/VehicleStatus.msg")]


def nav_state_names() -> dict[int, str]:
    """PX4 소스에서 NAVIGATION_STATE_* 를 읽는다 (버전 추정 금지)."""
    names: dict[int, str] = {}
    for cand in PX4_MSG_CANDIDATES:
        if not cand.exists():
            continue
        for line in cand.read_text(errors="replace").splitlines():
            m = re.match(r"\s*uint8\s+NAVIGATION_STATE_(\w+)\s*=\s*(\d+)", line)
            if m:
                names[int(m.group(2))] = m.group(1)
        if names:
            break
    return names


def topic(ulog, name):
    for d in ulog.data_list:
        if d.name == name:
            return d.data
    return None


def tsec(d):
    return np.asarray(d["timestamp"], dtype=float) / 1e6


def segments(t, v, names):
    segs, start = [], 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[start]:
            end_t = t[i - 1] if i == len(v) else t[i]
            segs.append({
                "value": int(v[start]),
                "name": names.get(int(v[start]), f"?{int(v[start])}"),
                "t_start_s": round(float(t[start]), 2),
                "t_end_s": round(float(end_t), 2),
                "duration_s": round(float(end_t - t[start]), 2),
            })
            start = i
    return segs


def stream_report(ulog, name, gap_s=1.0):
    d = topic(ulog, name)
    if d is None:
        return {"present": False}
    t = tsec(d)
    dt = np.diff(t)
    gaps = [{"from_s": round(float(t[i]), 2), "to_s": round(float(t[i + 1]), 2),
             "gap_s": round(float(dt[i]), 2)}
            for i in np.where(dt > gap_s)[0]]
    return {
        "present": True, "n": int(len(t)),
        "first_s": round(float(t[0]), 2), "last_s": round(float(t[-1]), 2),
        "mean_hz": round(float(len(t) / max(t[-1] - t[0], 1e-6)), 2),
        "gaps_over_{}s".format(gap_s): gaps[:20],
        "n_gaps": len(gaps),
    }


def main() -> int:
    run_dir = Path(sys.argv[1])
    meta = json.loads((run_dir / "meta.json").read_text())
    ulgs = meta.get("ulogs") or []
    if not ulgs:
        print(json.dumps({"error": "meta.json 에 ulogs 없음"}, ensure_ascii=False))
        return 1
    ulog = ULog(str(run_dir / ulgs[-1]))
    names = nav_state_names()

    out: dict = {"run": meta.get("run_id"), "ulog": ulgs[-1],
                 "exit_reason": meta.get("exit_reason"),
                 "px4_head": meta.get("px4_head")}

    vs = topic(ulog, "vehicle_status")
    if vs is None:
        out["nav_state"] = {"error": "vehicle_status 없음"}
    else:
        t = tsec(vs)
        out["nav_state"] = segments(t, np.asarray(vs["nav_state"], dtype=int), names)
        arm = np.asarray(vs["arming_state"], dtype=int)
        out["arming"] = segments(t, arm, {1: "STANDBY", 2: "ARMED"})

    vv = topic(ulog, "vtol_vehicle_status")
    if vv is not None:
        out["vtol_state"] = segments(
            tsec(vv), np.asarray(vv["vehicle_vtol_state"], dtype=int),
            {0: "UNDEFINED", 1: "TRANS_TO_FW", 2: "TRANS_TO_MC", 3: "MC", 4: "FW"})

    for tname in ("offboard_control_mode", "trajectory_setpoint",
                  "vehicle_local_position_setpoint"):
        out[tname] = stream_report(ulog, tname)

    # C4(바람) 검증용 — EKF 풍속 추정. gz 월드의 <wind> 가 실제 물리에 반영됐다면
    # 여기에 나타난다. "바람을 넣었다"와 "바람이 먹었다"를 가르는 유일한 내부 증거.
    w = topic(ulog, "wind")
    if w is not None:
        t = tsec(w)
        wn = np.asarray(w["windspeed_north"], dtype=float)
        we = np.asarray(w["windspeed_east"], dtype=float)
        mag = np.hypot(wn, we)
        out["wind_estimate"] = {
            "n": int(len(t)),
            "last_north": round(float(wn[-1]), 2),
            "last_east": round(float(we[-1]), 2),
            "max_mag": round(float(mag.max()), 2),
            "final_mag": round(float(mag[-1]), 2),
            "mean_mag_last_20pct": round(
                float(mag[int(len(mag) * 0.8):].mean()), 2),
        }
    else:
        out["wind_estimate"] = {"present": False}

    vc = topic(ulog, "vehicle_command")
    if vc is not None:
        t = tsec(vc)
        cmd = np.asarray(vc["command"], dtype=int)
        rows = []
        for i in range(len(t)):
            if cmd[i] in (176, 300, 3000, 22, 21, 400):  # SET_MODE/TAKEOFF/VTOL 등
                rows.append({
                    "t_s": round(float(t[i]), 2), "command": int(cmd[i]),
                    "p1": round(float(vc["param1"][i]), 3),
                    "p2": round(float(vc["param2"][i]), 3),
                    "p3": round(float(vc["param3"][i]), 3),
                })
        out["vehicle_command"] = rows

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
