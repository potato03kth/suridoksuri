#!/usr/bin/env python3
"""F-17 / F-4 천이 헤딩 결함 전용 ulog 프로브.

`docs/sitl_vtol_f17_transition_heading.md` §Q3-2/§Q3-3 의 표를 재현한다.
원래 이 분석은 `/mnt/c/sitl7_xfer/f17_probe*.py` 라는 임시 스크립트로 했고
그게 사라져서 재현이 불가능해졌다 — 그래서 저장소에 정식으로 넣는다.

무엇을 보나:
  전방천이(`vehicle_status.in_transition_to_fw == 1`) 구간에서 PX4 가 만든
  횡유도 지령 `fixed_wing_lateral_setpoint.course` 가 **기체 실제 헤딩을
  따르는지**, 아니면 **정북(0 rad)에 고정되는지**.

  F-17 미패치: `_pos_sp_triplet = {}` 가 yaw 를 0.0f 로 제로초기화 → 0.0f 는
  finite → `FixedWingModeManager.cpp:549` 의 `_yaw` 폴백이 죽음 →
  가상 WP 가 항상 정북 3000m → course ≈ atan2(-E, 3000) ≈ 0°.

사용:
  python3 tools/sitl/f17_transition_probe.py <run.ulg> [--post-s 15.0] [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

HDG_HOLD_DIST_NEXT = 3000.0  # FixedWingModeManager.hpp:106


def topic(ulog, name, multi=0):
    for d in ulog.data_list:
        if d.name == name and d.multi_id == multi:
            return d.data
    return None


def tsec(data):
    return np.asarray(data["timestamp"], dtype=float) / 1e6


def quat_yaw(data):
    q0 = np.asarray(data["q[0]"], dtype=float)
    q1 = np.asarray(data["q[1]"], dtype=float)
    q2 = np.asarray(data["q[2]"], dtype=float)
    q3 = np.asarray(data["q[3]"], dtype=float)
    return np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))


def wrap180(deg):
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


def interp_at(t_src, v_src, t_at):
    if len(t_src) == 0:
        return float("nan")
    return float(np.interp(t_at, t_src, v_src))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ulog")
    ap.add_argument("--post-s", type=float, default=15.0,
                    help="천이 종료 후 피해를 집계할 창(초). 기본 15s "
                         "(기준선 표의 북향 최대이탈 t+3.2s / 고도최저 t+3.7s 를 포함)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from pyulog import ULog
    ulog = ULog(args.ulog)

    st = topic(ulog, "vehicle_status")
    lp = topic(ulog, "vehicle_local_position")
    att = topic(ulog, "vehicle_attitude")
    fwsp = topic(ulog, "fixed_wing_lateral_setpoint")

    if st is None or lp is None or att is None:
        print("ERROR: vehicle_status / vehicle_local_position / vehicle_attitude 중 없는 토픽이 있다",
              file=sys.stderr)
        return 1

    # ── 천이 구간 ────────────────────────────────────────────────────────
    t_st = tsec(st)
    if "in_transition_to_fw" not in st:
        print("ERROR: vehicle_status 에 in_transition_to_fw 없음", file=sys.stderr)
        return 1
    intr = np.asarray(st["in_transition_to_fw"], dtype=float)
    on = np.nonzero(intr > 0.5)[0]
    if len(on) == 0:
        print("ERROR: in_transition_to_fw==1 샘플이 없다 (전방천이 미발생)", file=sys.stderr)
        return 1
    tr_a, tr_b = float(t_st[on[0]]), float(t_st[on[-1]])

    # ── 자세 / 위치 ──────────────────────────────────────────────────────
    t_att = tsec(att)
    yaw_deg = np.degrees(quat_yaw(att))
    t_lp = tsec(lp)
    N = np.asarray(lp["x"], dtype=float)
    E = np.asarray(lp["y"], dtype=float)
    D = np.asarray(lp["z"], dtype=float)
    alt = -D  # local frame: 이륙지점 기준 상대고도

    out = {
        "ulog": args.ulog,
        "transition_window_s": [round(tr_a, 3), round(tr_b, 3)],
        "transition_duration_s": round(tr_b - tr_a, 3),
    }

    # ── 천이 중 course 지령 (핵심 판정) ──────────────────────────────────
    if fwsp is None or "course" not in fwsp:
        out["course"] = {"error": "fixed_wing_lateral_setpoint.course 없음"}
        rows = []
    else:
        t_sp = tsec(fwsp)
        course_deg = np.degrees(np.asarray(fwsp["course"], dtype=float))
        finite = np.isfinite(course_deg)
        m_in = (t_sp >= tr_a) & (t_sp <= tr_b) & finite
        rows = []
        for i in np.nonzero(m_in)[0]:
            e_here = interp_at(t_lp, E, t_sp[i])
            rows.append({
                "t": round(float(t_sp[i]), 3),
                "course_deg": round(float(course_deg[i]), 2),
                "yaw_deg": round(interp_at(t_att, yaw_deg, t_sp[i]), 1),
                "N": round(interp_at(t_lp, N, t_sp[i]), 2),
                "E": round(e_here, 2),
                # 가상 WP 가 정북 3000m 이라면 course 는 이 값이어야 한다
                "predict_north3000_deg": round(
                    math.degrees(math.atan2(-e_here, HDG_HOLD_DIST_NEXT)), 3),
            })
        n_in = len(rows)
        if n_in:
            c_in = np.array([r["course_deg"] for r in rows])
            y_in = np.array([r["yaw_deg"] for r in rows])
            pred = np.array([r["predict_north3000_deg"] for r in rows])
            out["course"] = {
                "samples_in_transition": n_in,
                "course_deg_min": round(float(c_in.min()), 2),
                "course_deg_max": round(float(c_in.max()), 2),
                "course_deg_mean": round(float(c_in.mean()), 2),
                "yaw_deg_mean": round(float(y_in.mean()), 1),
                # |course - yaw| 이 작으면 지령이 기체 헤딩을 따른다 = 패치 정상
                "abs_course_minus_yaw_deg_mean": round(
                    float(np.abs(wrap180(c_in - y_in)).mean()), 2),
                # |course - 정북3000m예측| 이 작으면 F-17 발현 = 미패치
                "abs_course_minus_north3000_deg_max": round(
                    float(np.abs(wrap180(c_in - pred)).max()), 3),
            }
            near_north = float(np.abs(wrap180(c_in - pred)).max()) < 1.0
            follows_yaw = float(np.abs(wrap180(c_in - y_in)).mean()) < 20.0
            out["course"]["verdict"] = (
                "F-17 발현 (course 가 정북3000m 가상WP 를 따름)" if near_north else
                "F-17 해소 (course 가 기체 헤딩을 따름)" if follows_yaw else
                "판정보류 — 둘 다 아님")
        else:
            out["course"] = {"samples_in_transition": 0,
                             "note": "천이 중 FW 횡유도 비활성 (샘플 0). "
                                     "§Q3-5 의 간헐 발현 조건에 해당 — 이 런으로는 판정 불가"}
    out["course_rows"] = rows

    # ── 천이 전후 피해 지표 ──────────────────────────────────────────────
    yaw_a = interp_at(t_att, yaw_deg, tr_a)
    yaw_b = interp_at(t_att, yaw_deg, tr_b)
    post_b = tr_b + args.post_s
    m_post = (t_att >= tr_a) & (t_att <= post_b)
    m_post_lp = (t_lp >= tr_a) & (t_lp <= post_b)

    dmg = {
        "yaw_at_transition_start_deg": round(yaw_a, 1),
        "yaw_at_transition_end_deg": round(yaw_b, 1),
        "post_window_s": [round(tr_a, 3), round(post_b, 3)],
    }
    if np.any(m_post):
        yp, tp = yaw_deg[m_post], t_att[m_post]
        i_min, i_max = int(np.argmin(yp)), int(np.argmax(yp))
        dmg["yaw_min_deg"] = round(float(yp[i_min]), 1)
        dmg["yaw_min_t"] = round(float(tp[i_min]), 2)
        dmg["yaw_max_deg"] = round(float(yp[i_max]), 1)
        dmg["yaw_max_t"] = round(float(tp[i_max]), 2)
    if np.any(m_post_lp):
        Np, Ep, ap_, tl = N[m_post_lp], E[m_post_lp], alt[m_post_lp], t_lp[m_post_lp]
        i_n = int(np.argmax(np.abs(Np)))
        i_e = int(np.argmax(np.abs(Ep)))
        i_a = int(np.argmin(ap_))
        dmg["north_dev_max_abs_m"] = round(float(abs(Np[i_n])), 2)
        dmg["north_dev_max_abs_t"] = round(float(tl[i_n]), 2)
        dmg["east_dev_max_abs_m"] = round(float(abs(Ep[i_e])), 2)
        dmg["east_dev_max_abs_t"] = round(float(tl[i_e]), 2)
        dmg["alt_min_m"] = round(float(ap_[i_a]), 2)
        dmg["alt_min_t"] = round(float(tl[i_a]), 2)
        dmg["alt_at_transition_start_m"] = round(interp_at(t_lp, alt, tr_a), 2)
        dmg["alt_drop_m"] = round(dmg["alt_at_transition_start_m"] - dmg["alt_min_m"], 2)
    out["damage"] = dmg

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print(f"ulog: {args.ulog}")
    print(f"천이 구간: {tr_a:.3f} ~ {tr_b:.3f} s  ({tr_b - tr_a:.3f}s)")
    print()
    print("── 천이 중 PX4 횡유도 지령 (F-17 핵심 증거) ──")
    if rows:
        print(f"{'t(s)':>9} {'course°':>9} {'yaw°':>8} {'N(m)':>8} {'E(m)':>8} {'정북3000예측°':>14}")
        for r in rows:
            print(f"{r['t']:>9.3f} {r['course_deg']:>9.2f} {r['yaw_deg']:>8.1f} "
                  f"{r['N']:>8.2f} {r['E']:>8.2f} {r['predict_north3000_deg']:>14.3f}")
    else:
        print("  (샘플 없음)")
    print()
    for k, v in out["course"].items():
        print(f"  {k}: {v}")
    print()
    print("── 천이 + 이후 피해 ──")
    for k, v in dmg.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
