#!/usr/bin/env python3
"""F-5 전용 ulog 프로브 — **선회 중 고도 침하**를 런끼리 비교 가능하게 뽑는다.

`docs/fc_f5_turn_altitude_loss.md` 의 처방(`FW_T_SPDWEIGHT` 인하 등)을 SITL 로
검증할 때, `analyze_run.py` 의 `altitude.cruise_alt_dev` 만으로는 부족하다:

  - `max_abs_dev_m` 은 **상방 오버슈트도 포함**한다(보고서 §1) → 침하만 따로 봐야 한다
  - 침하의 원인 분해(고도→속도 교환 vs 총에너지 손실)는 아예 없다
  - `FW_T_SPDWEIGHT` 를 내리면 **속도가 흔들린다** → 실속 여유를 같이 봐야 판정이 된다

그래서 이 프로브는 FW 순항창 안에서 다음을 낸다:

  침하/오버슈트 · 뱅크 첨두 · TAS 변동 · 스로틀/피치 첨두 · `underspeed_ratio` ·
  실속 여유(CAS 최저 vs `FW_AIRSPD_MIN`/`FW_AIRSPD_STALL`) ·
  **뱅크 이벤트별 에너지 분해**(Δh / ΔKE / ΔE)

그리고 **그 런이 실제로 어떤 파라미터로 돌았는지**를 ulog 의 `initial_parameters`
에서 직접 읽어 함께 낸다 — `meta.json` 의 `px4_params` 되읽기와 **독립적인 2차
확인**이다(파라미터가 안 먹은 런을 결과로 쓰는 사고 방지).

사용:

    python3 tools/sitl/f5_turn_probe.py <run_dir> [<run_dir> ...]
    python3 tools/sitl/f5_turn_probe.py <run_dir> --json out.json

`<run_dir>` 은 `run_scenario.py` 산출 디렉터리(`meta.json` + `*.ulg`).
`metrics.json` 이 있으면 FOLLOWING 창과 순항고도 기준을 거기서 가져오고,
없으면 ulog 만으로 **FW 순항 구간**(`vtol_vehicle_status.vehicle_vtol_state==4`)을
잡아 대체한다(그 사실이 출력의 `window_source` 에 남는다).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from pyulog import ULog
except ImportError:                                   # pragma: no cover
    print("pyulog 가 없다: pip install pyulog", file=sys.stderr)
    raise

G = 9.80665

# 관심 파라미터 — 런 간 비교표의 provenance 로 쓴다.
PARAMS_OF_INTEREST = [
    "FW_T_SPDWEIGHT", "FW_T_ALT_TC", "FW_T_RLL2THR", "FW_T_CLMB_MAX",
    "FW_T_SINK_MAX", "FW_T_SINK_MIN", "FW_T_I_GAIN_PIT",
    "FW_R_LIM", "FW_P_LIM_MIN", "FW_P_LIM_MAX", "FW_PSP_OFF",
    "FW_THR_MAX", "FW_THR_TRIM", "FW_AIRSPD_TRIM", "FW_AIRSPD_MIN",
    "FW_AIRSPD_STALL", "NPFG_PERIOD",
]

BANK_EVENT_DEG = 20.0     # 이 각을 넘는 구간을 "선회 이벤트"로 본다
BANK_TAIL_S = 4.0         # 뱅크가 풀린 뒤에도 침하가 이어지는 꼬리


def _f(x, n=3):
    if x is None:
        return None
    v = float(x)
    return None if not np.isfinite(v) else round(v, n)


def tsec(d) -> np.ndarray:
    return np.asarray(d.data["timestamp"], dtype=float) / 1e6


def topic(u: ULog, name: str):
    for d in u.data_list:
        if d.name == name:
            return d
    return None


def quat_roll_pitch(d):
    """쿼터니언 → (t, roll[rad], pitch[rad])."""
    t = tsec(d)
    q0 = np.asarray(d.data["q[0]"], float)
    q1 = np.asarray(d.data["q[1]"], float)
    q2 = np.asarray(d.data["q[2]"], float)
    q3 = np.asarray(d.data["q[3]"], float)
    roll = np.arctan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
    pitch = np.arcsin(np.clip(2.0 * (q0 * q2 - q3 * q1), -1.0, 1.0))
    return t, roll, pitch


def agl_series(lp):
    """`analyze_run.metric_altitude` 와 **같은 정의**의 AGL (비교 가능성 유지)."""
    t = tsec(lp)
    z = np.asarray(lp.data["z"], float)
    good = np.isfinite(z)
    t, z = t[good], z[good]
    if len(t) == 0:
        return None, None, None
    pre = z[t <= t[0] + 5.0]
    z_ground = float(np.median(pre)) if len(pre) else float(z[0])
    return t, -(z - z_ground), z_ground


def fw_cruise_window(u: ULog):
    """`vtol_vehicle_state==4`(FW) 최장 구간 — metrics.json 이 없을 때의 폴백."""
    d = topic(u, "vtol_vehicle_status")
    if d is None:
        return None
    t = tsec(d)
    v = np.asarray(d.data["vehicle_vtol_state"], int)
    best, cur_start = None, None
    for i in range(len(v)):
        if v[i] == 4 and cur_start is None:
            cur_start = t[i]
        elif v[i] != 4 and cur_start is not None:
            if best is None or (t[i] - cur_start) > (best[1] - best[0]):
                best = (cur_start, t[i])
            cur_start = None
    if cur_start is not None:
        if best is None or (t[-1] - cur_start) > (best[1] - best[0]):
            best = (cur_start, t[-1])
    return best


def _interp(t_src, y_src, t_dst):
    return np.interp(t_dst, t_src, y_src)


def bank_events(t_att, roll, t_agl, agl, t_tas, tas, thr_t, thr, window):
    """|roll| > BANK_EVENT_DEG 구간마다 에너지 분해."""
    a, b = window
    m = (t_att >= a) & (t_att <= b)
    if not m.any():
        return []
    ta, ra = t_att[m], np.abs(np.degrees(roll[m]))
    hot = ra > BANK_EVENT_DEG
    events = []
    i = 0
    while i < len(hot):
        if not hot[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(hot) and hot[j + 1]:
            j += 1
        t_start, t_end = ta[i], ta[j]
        peak = float(ra[i:j + 1].max())
        # 침하는 뱅크가 풀린 뒤에도 이어진다 → 꼬리를 포함해 최저점을 찾는다
        ms = (t_agl >= t_start) & (t_agl <= min(t_end + BANK_TAIL_S, b))
        if ms.any() and len(t_tas) and len(thr_t):
            seg_t, seg_h = t_agl[ms], agl[ms]
            k = int(np.argmin(seg_h))
            t_min = seg_t[k]
            h0, h1 = float(seg_h[0]), float(seg_h[k])
            v0 = float(_interp(t_tas, tas, np.array([t_start]))[0])
            v1 = float(_interp(t_tas, tas, np.array([t_min]))[0])
            dh = h1 - h0
            dke = (v1 * v1 - v0 * v0) / (2 * G)
            de = dh + dke
            mt = (thr_t >= t_start) & (thr_t <= t_min + 0.5)
            events.append({
                "t_start_ulog_s": _f(t_start), "t_min_alt_ulog_s": _f(t_min),
                "bank_peak_deg": _f(peak),
                "duration_over_thresh_s": _f(t_end - t_start),
                "agl_start_m": _f(h0), "agl_min_m": _f(h1),
                "delta_h_m": _f(dh),
                "tas_start_ms": _f(v0), "tas_at_min_ms": _f(v1),
                "delta_ke_m": _f(dke), "delta_E_m": _f(de),
                "to_speed_pct": _f(100.0 * dke / -dh if dh < 0 else None, 1),
                "throttle_peak": _f(thr[mt].max() if mt.any() else None),
            })
        i = j + 1
    events.sort(key=lambda e: (e["delta_h_m"] if e["delta_h_m"] is not None else 0))
    return events


def probe_run(run_dir: Path) -> dict:
    meta_p = run_dir / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    ulgs = sorted(run_dir.glob("*.ulg"))
    named = meta.get("ulogs") or []
    if named:
        pick = [run_dir / n for n in named if (run_dir / n).exists()]
        if pick:
            ulgs = pick
    out: dict = {
        "run_dir": str(run_dir),
        "run_id": meta.get("run_id", run_dir.name),
        "scenario_id": meta.get("scenario_id"),
        "exit_code": meta.get("exit_code"),
        "exit_reason": meta.get("exit_reason"),
        "px4_params_requested": meta.get("px4_params"),
        "repo_head": meta.get("repo_head"),
        "px4_head": meta.get("px4_head"),
    }
    if not ulgs:
        out["error"] = "ulg 없음"
        return out
    u = ULog(str(ulgs[-1]), message_name_filter_list=[
        "vehicle_attitude", "vehicle_local_position", "tecs_status",
        "airspeed_validated", "vtol_vehicle_status"])
    out["ulg"] = ulgs[-1].name
    # 🔴 이 런이 **실제로** 어떤 파라미터로 돌았는지 (meta 되읽기와 독립)
    #
    # ⚠️ `initial_parameters` 만 보면 안 된다 — SITL ulog 는 **PX4 부팅 시점**부터
    #    기록되고, `--px4-param` 은 MAVROS 가 붙은 뒤(부팅 +30초쯤) 들어간다.
    #    그래서 주입한 값은 `initial_parameters` 가 아니라 `changed_parameters`
    #    에 들어 있다(B4_spd05 실측: initial=1.0, changed@32.9s=0.5).
    #    "initial 이 1.0 이니 파라미터가 안 먹었다"는 오판을 하기 딱 좋다.
    eff = dict(u.initial_parameters)
    changes = {}
    for ts, key, val in u.changed_parameters:
        eff[key] = val
        if key in PARAMS_OF_INTEREST:
            changes.setdefault(key, []).append((round(ts / 1e6, 2), val))
    out["params_effective"] = {k: _f(v, 4) for k, v in eff.items()
                               if k in PARAMS_OF_INTEREST}
    out["params_boot"] = {k: _f(v, 4) for k, v in u.initial_parameters.items()
                          if k in PARAMS_OF_INTEREST}
    out["params_changed_in_log"] = changes

    lp = topic(u, "vehicle_local_position")
    att = topic(u, "vehicle_attitude")
    tec = topic(u, "tecs_status")
    asp = topic(u, "airspeed_validated")
    if lp is None or att is None:
        out["error"] = "vehicle_local_position / vehicle_attitude 없음"
        return out

    t_agl, agl, z_ground = agl_series(lp)

    # ── 창과 순항고도 기준 ────────────────────────────────────────────────
    win, ref, src = None, None, None
    mp = run_dir / "metrics.json"
    if mp.exists():
        try:
            M = json.loads(mp.read_text(encoding="utf-8"))
            cd = (M.get("altitude") or {}).get("cruise_alt_dev") or {}
            if cd.get("window_ulog_s") and cd.get("ref_agl_m") is not None:
                win = tuple(cd["window_ulog_s"])
                ref = float(cd["ref_agl_m"])
                src = "metrics.json altitude.cruise_alt_dev (FOLLOWING 창)"
        except Exception:                              # noqa: BLE001
            pass
    if win is None:
        win = fw_cruise_window(u)
        src = "ulog vtol_vehicle_status==4 (FW 순항 구간) — metrics.json 없음/불완전"
        if win is None:
            out["error"] = "FOLLOWING 창도 FW 구간도 특정 불가"
            return out
    if ref is None:
        wpz = None
        try:
            wp = json.loads((meta.get("launch_args") or {}).get("waypoints", "[]"))
            wpz = float(wp[-1]) if wp else None
        except Exception:                              # noqa: BLE001
            pass
        ref = (wpz - (-z_ground)) if wpz is not None else float(
            np.median(agl[(t_agl >= win[0]) & (t_agl <= win[1])]))
        src = (src or "") + " / ref = waypoints[-1].z − 지면"
    out["window_raw_ulog_s"] = [_f(win[0]), _f(win[1])]
    out["window_source"] = src
    out["ref_agl_m"] = _f(ref)

    # 🔴 FOLLOWING 창을 FW 구간(vtol_state==4)과 **교집합** 낸다.
    #    node.log↔ulog 시각정렬 잔차(±1s)로 창의 양 끝이 천이 구간을 물면
    #    TAS 최저가 천이 중 6 m/s 로 찍혀 "실속 직전"처럼 보인다(B4 실측:
    #    창 시작 147.10 vs FW 진입 147.59 → TAS min 6.06 @ 147.19).
    fw = fw_cruise_window(u)
    if fw is not None:
        lo, hi = max(win[0], fw[0]), min(win[1], fw[1])
        if hi - lo > 1.0:
            win = (lo, hi)
    out["window_ulog_s"] = [_f(win[0]), _f(win[1])]
    out["window_note"] = "FOLLOWING ∩ vtol_state==4 (천이 경계 오염 제거)"
    # 파라미터 주입이 **분석창보다 먼저** 끝났는지 — 아니면 그 런은 무효다.
    late = {k: v for k, v in out.get("params_changed_in_log", {}).items()
            if any(ts >= win[0] for ts, _ in v)}
    out["params_settled_before_window"] = not late
    if late:
        out["params_late_change"] = late

    a, b = win
    m_raw = (t_agl >= out["window_raw_ulog_s"][0]) & (t_agl <= out["window_raw_ulog_s"][1])
    m_agl = (t_agl >= a) & (t_agl <= b)
    if not m_agl.any():
        out["error"] = "창 안에 위치 샘플 없음"
        return out
    out["altitude"] = {
        "min_agl_m": _f(agl[m_agl].min()),
        "max_agl_m": _f(agl[m_agl].max()),
        "sink_m": _f(ref - agl[m_agl].min()),          # 🔴 헤드라인 지표
        "overshoot_m": _f(agl[m_agl].max() - ref),
        "mean_dev_m": _f((agl[m_agl] - ref).mean()),
        "rms_dev_m": _f(np.sqrt(((agl[m_agl] - ref) ** 2).mean())),
        # 보고서 §1 표(7.45/4.73/0.91)와 직접 대조하기 위한 미교집합 값
        "sink_m_following_raw": _f(ref - agl[m_raw].min()) if m_raw.any() else None,
    }
    # 하강률 첨두
    tv = tsec(lp)
    vz = np.asarray(lp.data["vz"], float)
    mv = (tv >= a) & (tv <= b) & np.isfinite(vz)
    out["altitude"]["vz_down_peak_ms"] = _f(vz[mv].max() if mv.any() else None)

    # ── 자세 ──────────────────────────────────────────────────────────────
    t_att, roll, pitch = quat_roll_pitch(att)
    ma = (t_att >= a) & (t_att <= b)
    out["attitude"] = {
        "roll_abs_peak_deg": _f(np.degrees(np.abs(roll[ma])).max()) if ma.any() else None,
        "pitch_max_deg": _f(np.degrees(pitch[ma]).max()) if ma.any() else None,
        "pitch_min_deg": _f(np.degrees(pitch[ma]).min()) if ma.any() else None,
        "t_over_20deg_bank_s": _f(
            float(np.degrees(np.abs(roll[ma])).__gt__(20.0).sum())
            * float(np.median(np.diff(t_att))) if ma.any() else None),
    }

    # ── 속도 (실속 여유) ─────────────────────────────────────────────────
    spd_min = float(u.initial_parameters.get("FW_AIRSPD_MIN", 10.0))
    stall = float(u.initial_parameters.get("FW_AIRSPD_STALL", 7.0))
    t_tas, tas = np.array([]), np.array([])
    if asp is not None:
        t_a = tsec(asp)
        tas_all = np.asarray(asp.data["true_airspeed_m_s"], float)
        cas_all = np.asarray(asp.data["calibrated_airspeed_m_s"], float)
        ms = (t_a >= a) & (t_a <= b) & np.isfinite(tas_all)
        t_tas, tas = t_a[np.isfinite(tas_all)], tas_all[np.isfinite(tas_all)]
        if ms.any():
            dt = float(np.median(np.diff(t_a))) if len(t_a) > 1 else 0.0
            out["airspeed"] = {
                "tas_min_ms": _f(tas_all[ms].min()),
                "tas_max_ms": _f(tas_all[ms].max()),
                "tas_mean_ms": _f(tas_all[ms].mean()),
                "tas_span_ms": _f(tas_all[ms].max() - tas_all[ms].min()),
                "cas_min_ms": _f(cas_all[ms].min()),
                "FW_AIRSPD_MIN": _f(spd_min), "FW_AIRSPD_STALL": _f(stall),
                "cas_margin_to_min_ms": _f(cas_all[ms].min() - spd_min),
                "cas_margin_to_stall_ms": _f(cas_all[ms].min() - stall),
                "t_cas_below_min_s": _f((cas_all[ms] < spd_min).sum() * dt),
                "t_cas_below_stall_s": _f((cas_all[ms] < stall).sum() * dt),
            }
            # 🔴 뱅크가 걸리면 PX4 가 요구하는 최저속도 자체가 올라간다:
            #   min CAS = FW_AIRSPD_MIN·sqrt(1/cos φ)
            #   (PerformanceModel.cpp:147-152 ← FwLateralLongitudinalControl.cpp:619)
            # 그리고 그보다 0.15·FW_AIRSPD_TRIM 아래에서 underspeed 완화가 램프인
            # 된다(TECS.cpp:369-378, tas_error_percentage=0.15 하드코딩).
            # 정지 임계(10 m/s)와의 여유만 보면 선회 중 실제 여유를 과대평가한다.
            trim = float(u.initial_parameters.get("FW_AIRSPD_TRIM", 15.0))
            phi_at = np.abs(_interp(t_att, roll, t_a[ms]))
            lf = 1.0 / np.maximum(np.cos(phi_at), 1e-6)
            min_cas_dyn = spd_min * np.sqrt(lf)
            ramp_in = min_cas_dyn - 0.15 * trim
            margin = cas_all[ms] - min_cas_dyn
            out["airspeed"].update({
                "dyn_min_cas_peak_ms": _f(min_cas_dyn.max()),
                "dyn_margin_min_ms": _f(margin.min()),
                "t_below_dyn_min_s": _f((margin < 0).sum() * dt),
                "t_below_underspeed_rampin_s": _f(
                    (cas_all[ms] < ramp_in).sum() * dt),
                "note": ("여유 = CAS − FW_AIRSPD_MIN·√(1/cosφ). "
                         "underspeed 램프인은 그보다 0.15·TRIM 아래"),
            })

    # ── TECS ─────────────────────────────────────────────────────────────
    thr_t, thr = np.array([]), np.array([])
    if tec is not None:
        t_t = tsec(tec)
        mt = (t_t >= a) & (t_t <= b)
        thr_all = np.asarray(tec.data["throttle_sp"], float)
        thr_t, thr = t_t, thr_all
        if mt.any():
            us = np.asarray(tec.data["underspeed_ratio"], float)
            hrs = np.asarray(tec.data["height_rate_setpoint"], float)
            psp = np.degrees(np.asarray(tec.data["pitch_sp_rad"], float))
            asp_sp = np.asarray(tec.data["equivalent_airspeed_sp"], float)
            thr_max = float(u.initial_parameters.get("FW_THR_MAX", 1.0))
            out["tecs"] = {
                "throttle_sp_peak": _f(thr_all[mt].max()),
                "throttle_sp_mean": _f(thr_all[mt].mean()),
                "FW_THR_MAX": _f(thr_max),
                "throttle_headroom_pct": _f(100.0 * (1 - thr_all[mt].max() / thr_max), 1),
                "underspeed_ratio_max": _f(us[mt].max()),
                "height_rate_sp_min_ms": _f(hrs[mt].min()),
                "height_rate_sp_max_ms": _f(hrs[mt].max()),
                "pitch_sp_max_deg": _f(psp[mt].max()),
                "pitch_sp_min_deg": _f(psp[mt].min()),
                "eas_sp_mean_ms": _f(asp_sp[mt].mean()),
            }

    # ── 뱅크 이벤트별 에너지 분해 ────────────────────────────────────────
    if len(t_tas) and len(thr_t):
        evs = bank_events(t_att, roll, t_agl, agl, t_tas, tas, thr_t, thr, win)
        out["bank_events"] = evs
        if evs:
            w = evs[0]                                  # 가장 크게 침하한 이벤트
            out["worst_event"] = w
            dh = w["delta_h_m"] or 0.0
            out["worst_event_split"] = {
                "delta_h_m": dh,
                "to_speed_m": w["delta_ke_m"],
                "true_energy_loss_m": w["delta_E_m"],
                "to_speed_pct": w["to_speed_pct"],
            }
    return out


def fmt_table(rows: list[dict]) -> str:
    hdr = ("| 런 | SPDWEIGHT | ALT_TC | 침하(m) | 상방(m) | RMS(m) | 뱅크첨두° | "
           "vz첨두 | TAS min/max | CAS최저 여유 | thr첨두 | pitch_sp°첨두 | underspd |")
    sep = "|" + "---|" * 13
    out = [hdr, sep]
    for r in rows:
        if r.get("error"):
            out.append(f"| {r['run_id']} | — | — | **{r['error']}** |" + " |" * 10)
            continue
        p = r.get("params_effective", {})
        al = r.get("altitude", {})
        at = r.get("attitude", {})
        sp = r.get("airspeed", {})
        te = r.get("tecs", {})
        out.append(
            f"| {r['run_id']} | {p.get('FW_T_SPDWEIGHT')} | {p.get('FW_T_ALT_TC')} "
            f"| **{al.get('sink_m')}** | +{al.get('overshoot_m')} | {al.get('rms_dev_m')} "
            f"| {at.get('roll_abs_peak_deg')} | {al.get('vz_down_peak_ms')} "
            f"| {sp.get('tas_min_ms')}/{sp.get('tas_max_ms')} "
            f"| {sp.get('cas_min_ms')} (+{sp.get('cas_margin_to_min_ms')}) "
            f"| {te.get('throttle_sp_peak')} | {te.get('pitch_sp_max_deg')} "
            f"| {te.get('underspeed_ratio_max')} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="F-5 선회 침하 프로브")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--json", default=None, help="결과 JSON 저장 경로")
    ap.add_argument("--events", action="store_true", help="뱅크 이벤트 전량 출력")
    args = ap.parse_args()

    rows = [probe_run(Path(d)) for d in args.run_dirs]
    print(fmt_table(rows))
    if args.events:
        for r in rows:
            print(f"\n### {r['run_id']} — 뱅크 이벤트 "
                  f"(|roll|>{BANK_EVENT_DEG:.0f}°, 꼬리 {BANK_TAIL_S:.0f}s)")
            for e in r.get("bank_events", []):
                print(f"  φ={e['bank_peak_deg']}° t={e['t_start_ulog_s']}s "
                      f"Δh={e['delta_h_m']} ΔKE={e['delta_ke_m']} "
                      f"ΔE={e['delta_E_m']} (속도로 {e['to_speed_pct']}%) "
                      f"thr={e['throttle_peak']}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON → {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
