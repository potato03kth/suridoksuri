#!/usr/bin/env python3
"""차압센서(피토) 영점 ↔ 온도 회귀 — 기존 ulog 전수 추출·분석 고정 스크립트.

배경: 2026-07-30 flight02 VTOL 천이 실패 분석에서 `SENS_DPRES_OFF = -21.78 Pa`
(저장된 영점)와 호버 중 raw 차압 -38.97 Pa 사이에 약 17 Pa 의 격차가 관측됐다
(`docs/f30_compare/transition_success_vs_fail.md` §3). 가설은 "차압센서 영점이
온도에 따라 드리프트한다"였고, 이 스크립트는 그 가설을 기존 실비행 ulog 만으로
정량화하기 위해 만들어졌다. 결과 해석은 `docs/fc_pitot_temp_drift.md` 에 있다.

이 스크립트가 하는 일:
  1. `logs/` 아래 모든 .ulg 를 md5 로 중복제거하고 `differential_pressure` 가
     있는 것만 추린다 (SITL 로그는 ver_hw 로 배제).
  2. 각 dp 표본마다 **참 차압이 0 이라고 볼 수 있는 구간**을 엄격히 판정한다
     — 착륙판정 + 모터정지 + 무전류 + 대지속도 0. 프롭워시·기류 오염을 피한다.
  3. 같은 시각의 온도 채널 4종(`differential_pressure.temperature`,
     `pitot_temperature`, `sensor_baro.temperature`,
     `vehicle_air_data.ambient_temperature`)과 `SENS_DPRES_OFF`·`device_id` 를 붙인다.
  4. 영점 ↔ 온도 OLS 회귀를 채널별/세션별/일자별로 돌리고, 세션 내부(within-session)
     기울기와 세션 사이(between-session) 기울기를 **따로** 보고한다.
     — 이 둘이 갈리는 것이 이 분석의 핵심 소견이라 반드시 분리해서 낸다.
  5. CSV(표본·창·회귀) + 플롯을 저장한다.

사용:
    .venv/bin/python tools/flight_logs/pitot_zero_temp.py
    .venv/bin/python tools/flight_logs/pitot_zero_temp.py --logs-root logs --out results/pitot_zero_temp
    .venv/bin/python tools/flight_logs/pitot_zero_temp.py --no-plots --json

의존: pyulog, numpy (플롯에만 matplotlib). scipy 는 쓰지 않는다 — t 분위수는
자체 구현(`t_ppf`)이고 단위테스트(`test_pitot_zero_temp.py`)로 검증한다.
"""

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 순수 함수 (pytest 대상 — test_pitot_zero_temp.py). pyulog 불필요.
# ---------------------------------------------------------------------------

# PX4 device::Device::DeviceId 비트필드 (src/drivers/drv_sensor.h 규약)
#   [0:3) bus_type, [3:8) bus, [8:16) address, [16:24) devtype
BUS_TYPE_NAMES = {0: "UNKNOWN", 1: "I2C", 2: "SPI", 3: "UAVCAN", 4: "SIMULATION",
                  5: "SERIAL", 6: "MAVLINK"}

# MS4525DO(1 psi, DS5AI001DP 계열) 스케일 상수 — PX4 ms4525do 드라이버와 동일.
#   14bit ADC 중 10%~90% 구간(=0.8*16383 counts)이 -1..+1 psi 에 대응한다.
MS4525DO_PSI_SPAN = 2.0
MS4525DO_COUNT_SPAN = 0.8 * 16383
PSI_TO_PA = 6894.757

# 표준 해면 밀도 — PX4 가 CAS(교정대기속도)를 만들 때 쓰는 값.
RHO_SEA_LEVEL = 1.225


def decode_device_id(device_id):
    """PX4 device_id 정수를 (bus_type, bus, address, devtype) 로 분해한다."""
    did = int(device_id)
    return {
        "device_id": did,
        "bus_type": did & 0x7,
        "bus_type_name": BUS_TYPE_NAMES.get(did & 0x7, f"UNKNOWN({did & 0x7})"),
        "bus": (did >> 3) & 0x1F,
        "address": (did >> 8) & 0xFF,
        "devtype": (did >> 16) & 0xFF,
    }


def ms4525do_count_step_pa():
    """MS4525DO 1 LSB 가 몇 Pa 인지. 관측된 raw dp 양자화 간격과 대조용."""
    return MS4525DO_PSI_SPAN / MS4525DO_COUNT_SPAN * PSI_TO_PA


def cas_from_dp(dp_pa, rho=RHO_SEA_LEVEL):
    """차압(Pa) → CAS(m/s). PX4 와 같이 부호를 보존한다."""
    return math.copysign(math.sqrt(2.0 * abs(dp_pa) / rho), dp_pa)


def dp_from_cas(cas_mps, rho=RHO_SEA_LEVEL):
    """CAS(m/s) → 차압(Pa). `cas_from_dp` 의 역함수."""
    return math.copysign(rho * cas_mps * cas_mps / 2.0, cas_mps)


def recompute_cas(cas_reported, stored_offset_pa, true_zero_pa, rho=RHO_SEA_LEVEL):
    """보고된 CAS 를 '영점이 true_zero 였다면' 값으로 다시 계산한다.

    PX4 는 `dp_corrected = dp_raw - SENS_DPRES_OFF` 로 CAS 를 만든다. 따라서
    보고 CAS 로부터 raw 를 역산한 뒤 올바른 영점을 빼면 교정 CAS 가 나온다.
    """
    dp_corrected_reported = dp_from_cas(cas_reported, rho)
    dp_raw = dp_corrected_reported + stored_offset_pa
    return cas_from_dp(dp_raw - true_zero_pa, rho)


def betacf(a, b, x, itmax=300, eps=3e-12):
    """정규화 불완전베타함수용 연분수 (Numerical Recipes 방식)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """정규화 불완전베타 I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    """Student-t 누적분포."""
    x = df / (df + t * t)
    p = 0.5 * betainc(0.5 * df, 0.5, x)
    return 1.0 - p if t > 0 else p


def t_ppf(p, df):
    """Student-t 분위수 (이분법). scipy 미사용 환경용."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    if df <= 0:
        raise ValueError("df must be > 0")
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ols_fit(x, y, conf=0.95):
    """단순선형회귀 y = b0 + b1*x. 표준오차·신뢰구간·R²·잔차표준편차를 함께 낸다.

    표본이 3개 미만이거나 x 가 상수면 회귀 불가로 None 을 돌려준다
    (— '레버리지 부족'을 조용히 0 으로 채우지 않기 위해서다)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3 or np.ptp(x) <= 0:
        return None
    X = np.vstack([np.ones(n), x]).T
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    dof = n - 2
    s2 = ss_res / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tcrit = t_ppf(0.5 + conf / 2.0, dof)
    return {
        "n": n,
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "se_intercept": float(se[0]),
        "se_slope": float(se[1]),
        "slope_lo": float(beta[1] - tcrit * se[1]),
        "slope_hi": float(beta[1] + tcrit * se[1]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "resid_sd": math.sqrt(s2),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "x_span": float(np.ptp(x)),
        "dof": dof,
        "conf": conf,
    }


def split_sessions(times, gap_s=45 * 60):
    """벽시계 시각 목록을 전원 세션(간격 > gap_s)으로 나눠 세션 인덱스를 돌려준다."""
    idx = []
    s = 0
    prev = None
    for t in times:
        if prev is not None and (t - prev).total_seconds() > gap_s:
            s += 1
        idx.append(s)
        prev = t
    return idx


def nearest_join(t_query, t_src, v_src):
    """t_query 각 시각에 가장 가까운 t_src 표본의 값을 뽑는다 (최근접이웃)."""
    t_query = np.asarray(t_query, dtype=np.int64)
    t_src = np.asarray(t_src, dtype=np.int64)
    v_src = np.asarray(v_src, dtype=float)
    if len(t_src) == 0:
        return np.full(len(t_query), np.nan)
    if len(t_src) == 1:
        return np.full(len(t_query), v_src[0])
    i = np.clip(np.searchsorted(t_src, t_query), 1, len(t_src) - 1)
    pick = np.where(np.abs(t_query - t_src[i - 1]) <= np.abs(t_src[i] - t_query), i - 1, i)
    return v_src[pick]


# ---------------------------------------------------------------------------
# ulog 추출
# ---------------------------------------------------------------------------

# 영점 구간 판정 기준. 세 조건을 모두 만족해야 '참 차압 = 0' 으로 본다.
#   - 착륙판정(vehicle_land_detected.landed == 1)  → 공중이 아니다
#   - 모든 모터 지령 < MOTOR_OFF                   → 프롭워시 없음
#   - 배터리 전류 < CURRENT_OFF_A                  → 모터정지 교차확인
#   - 대지속도 < GROUNDSPEED_MAX                   → 기체가 안 움직인다
MOTOR_OFF = 0.01
CURRENT_OFF_A = 1.0
GROUNDSPEED_MAX = 0.3

# 호버 구간(2차 참고용). 프롭워시가 걸리므로 영점의 1차 근거로는 쓰지 않는다.
HOVER_GS_MAX = 0.6
HOVER_VZ_MAX = 0.4
HOVER_MOTOR_MIN = 0.3

WALL_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})")


def wall_time_from_name(path):
    """ulog 파일명에 박힌 UTC 벽시계 시각. 없으면 None."""
    m = WALL_RE.search(os.path.basename(path))
    if not m:
        return None
    return dt.datetime(*[int(g) for g in m.groups()])


def extract_log(path):
    """ulog 하나에서 dp 표본 + 문맥을 뽑는다. dp 가 없으면 None."""
    from pyulog import ULog

    try:
        ulog = ULog(path)
    except Exception as exc:  # 손상된 로그를 조용히 삼키지 않는다
        return {"path": path, "error": str(exc)}

    info = ulog.msg_info_dict
    hw = str(info.get("ver_hw", "?"))
    if "SITL" in hw.upper():
        return {"path": path, "skip": "SITL"}

    topics = {}
    for d in ulog.data_list:
        topics.setdefault(d.name, d.data)
    dp = topics.get("differential_pressure")
    if dp is None or len(dp["timestamp"]) == 0:
        return {"path": path, "skip": "no differential_pressure"}

    ts = np.array(dp["timestamp"], dtype=np.int64)

    def ctx(name, key):
        src = topics.get(name)
        if src is None or key not in src:
            return np.full(len(ts), np.nan)
        return nearest_join(ts, np.array(src["timestamp"], dtype=np.int64),
                            np.array(src[key], dtype=float))

    raw = np.array(dp["differential_pressure_pa"], dtype=float)
    t_dp = np.array(dp["temperature"], dtype=float)
    t_pitot = np.array(dp.get("pitot_temperature", np.full(len(ts), np.nan)), dtype=float)
    device_id = int(dp["device_id"][0])

    landed = ctx("vehicle_land_detected", "landed")
    current = ctx("battery_status", "current_a")
    vx = ctx("vehicle_local_position", "vx")
    vy = ctx("vehicle_local_position", "vy")
    vz = ctx("vehicle_local_position", "vz")
    armed = ctx("actuator_armed", "armed")
    vtol_state = ctx("vtol_vehicle_status", "vehicle_vtol_state")
    t_baro = ctx("sensor_baro", "temperature")
    t_amb = ctx("vehicle_air_data", "ambient_temperature")
    p_baro = ctx("sensor_baro", "pressure")

    motors = np.vstack([ctx("actuator_motors", f"control[{i}]") for i in range(8)])
    motor_max = np.nanmax(np.where(np.isnan(motors), -9.0, motors), axis=0)
    groundspeed = np.hypot(vx, vy)

    is_zero = (
        (landed == 1)
        & (motor_max < MOTOR_OFF)
        & ((current < CURRENT_OFF_A) | np.isnan(current))
        & ((groundspeed < GROUNDSPEED_MAX) | np.isnan(groundspeed))
    )
    is_hover = (
        (landed == 0)
        & (vtol_state == 3)
        & (groundspeed < HOVER_GS_MAX)
        & (np.abs(vz) < HOVER_VZ_MAX)
        & (motor_max > HOVER_MOTOR_MIN)
    )
    # 이 로그 안에서 한 번이라도 떴는지 → 영점 표본을 이륙전/착륙후로 나눈다
    airborne_before = np.cumsum((landed == 0).astype(int)) > 0

    t_rel = (ts - ulog.start_timestamp) / 1e6
    wall = wall_time_from_name(path)

    return {
        "path": path,
        "flight_dir": os.path.basename(os.path.dirname(path)),
        "file": os.path.basename(path),
        "wall": wall,
        "hw": hw,
        "sw": str(info.get("ver_sw", "?")),
        "sw_release": info.get("ver_sw_release", None),
        "device_id": device_id,
        "sens_dpres_off": float(ulog.initial_parameters.get("SENS_DPRES_OFF", float("nan"))),
        "sens_dpres_rev": ulog.initial_parameters.get("SENS_DPRES_REV", None),
        "sdlog_profile": ulog.initial_parameters.get("SDLOG_PROFILE", None),
        "vt_arsp_trans": ulog.initial_parameters.get("VT_ARSP_TRANS", None),
        "n_tc_params": sum(1 for k in ulog.initial_parameters if k.startswith("TC_")),
        "duration_s": float((ulog.last_timestamp - ulog.start_timestamp) / 1e6),
        "n_dp": len(ts),
        "boot_s_first": float(ts[0] / 1e6),
        # 배열
        "t_rel": t_rel,
        "boot_s": ts / 1e6,
        "raw": raw,
        "t_dp": t_dp,
        "t_pitot": t_pitot,
        "t_baro": t_baro,
        "t_amb": t_amb,
        "p_baro": p_baro,
        "landed": landed,
        "armed": armed,
        "motor_max": motor_max,
        "current": current,
        "groundspeed": groundspeed,
        "vz": vz,
        "vtol_state": vtol_state,
        "is_zero": is_zero,
        "is_hover": is_hover,
        "post_flight": airborne_before,
    }


def collect(logs_root):
    """logs_root 아래 .ulg 를 md5 중복제거해 전부 추출한다."""
    paths = sorted(glob.glob(os.path.join(logs_root, "**", "*.ulg"), recursive=True))
    seen = {}
    out, skipped, dups, errors = [], [], [], []
    for p in paths:
        with open(p, "rb") as fh:
            h = hashlib.md5(fh.read()).hexdigest()
        if h in seen:
            dups.append((p, seen[h]))
            continue
        seen[h] = p
        rec = extract_log(p)
        if "error" in rec:
            errors.append((p, rec["error"]))
        elif "skip" in rec:
            skipped.append((p, rec["skip"]))
        else:
            out.append(rec)
    return out, skipped, dups, errors, len(paths)


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------

CHANNELS = [
    ("t_dp", "differential_pressure.temperature"),
    ("t_pitot", "differential_pressure.pitot_temperature"),
    ("t_baro", "sensor_baro.temperature"),
    ("t_amb", "vehicle_air_data.ambient_temperature"),
]


def build_sample_rows(logs, mask_key):
    rows = []
    for lg in logs:
        m = lg[mask_key]
        for i in np.nonzero(m)[0]:
            rows.append({
                "file": lg["file"],
                "flight_dir": lg["flight_dir"],
                "wall_utc": lg["wall"].isoformat() if lg["wall"] else "",
                "t_rel_s": round(float(lg["t_rel"][i]), 2),
                "boot_s": round(float(lg["boot_s"][i]), 1),
                "raw_dp_pa": float(lg["raw"][i]),
                "t_dp_c": float(lg["t_dp"][i]),
                "t_pitot_c": float(lg["t_pitot"][i]),
                "t_baro_c": float(lg["t_baro"][i]),
                "t_amb_c": float(lg["t_amb"][i]),
                "p_baro_pa": float(lg["p_baro"][i]),
                "sens_dpres_off_pa": lg["sens_dpres_off"],
                "device_id": lg["device_id"],
                "motor_max": round(float(lg["motor_max"][i]), 4),
                "current_a": round(float(lg["current"][i]), 2),
                "groundspeed_mps": round(float(lg["groundspeed"][i]), 3),
                "post_flight": int(lg["post_flight"][i]),
            })
    return rows


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def fmt_fit(fit, label):
    if fit is None:
        return f"  {label:<40} 회귀 불가 (표본<3 또는 x 상수 — 레버리지 부족)"
    return (f"  {label:<40} n={fit['n']:4d}  기울기={fit['slope']:+7.3f} "
            f"[{fit['slope_lo']:+7.3f},{fit['slope_hi']:+7.3f}] Pa/°C  "
            f"절편={fit['intercept']:+8.2f}  R²={fit['r2']:.3f}  "
            f"잔차sd={fit['resid_sd']:5.2f} Pa  T범위={fit['x_span']:5.2f}°C")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-root", default="logs")
    ap.add_argument("--out", default="results/pitot_zero_temp")
    ap.add_argument("--session-gap-min", type=float, default=45.0,
                    help="이 간격(분)보다 크면 다른 전원 세션으로 본다")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--json", action="store_true", help="요약을 JSON 으로도 출력")
    args = ap.parse_args(argv)

    logs, skipped, dups, errors, n_paths = collect(args.logs_root)

    print("=" * 78)
    print("차압센서 영점 ↔ 온도 회귀")
    print("=" * 78)
    print(f"스캔한 .ulg: {n_paths}  중복(md5 동일): {len(dups)}  "
          f"제외: {len(skipped)}  오류: {len(errors)}  분석대상: {len(logs)}")
    for p, why in errors:
        print(f"  [오류] {p}: {why}")
    no_dp = sum(1 for _, w in skipped if "differential_pressure" in w)
    sitl = sum(1 for _, w in skipped if w == "SITL")
    print(f"  제외 내역: differential_pressure 없음 {no_dp} / SITL {sitl}")

    # --- 센서 동일성
    devs = sorted({lg["device_id"] for lg in logs})
    print("\n[센서 동일성]")
    for d in devs:
        info = decode_device_id(d)
        n = sum(1 for lg in logs if lg["device_id"] == d)
        print(f"  device_id={d} ({info['bus_type_name']} bus{info['bus']} "
              f"addr=0x{info['address']:02X} devtype=0x{info['devtype']:02X})  로그 {n}건")
    print(f"  MS4525DO 이론 양자화 스텝 = {ms4525do_count_step_pa():.4f} Pa/LSB")
    all_raw = np.concatenate([lg["raw"] for lg in logs])
    steps = np.diff(np.unique(np.round(all_raw, 4)))
    if len(steps):
        print(f"  실측 raw dp 최소 간격      = {steps.min():.4f} Pa  (일치하면 MS4525DO 확정)")

    offs = sorted({round(lg["sens_dpres_off"], 3) for lg in logs})
    print(f"\n[SENS_DPRES_OFF] 로그에 나타난 값: {offs}")
    tc = sorted({lg["n_tc_params"] for lg in logs})
    print(f"[TC_* 온도보상 파라미터] 로그별 개수: {tc}  (0 이면 PX4 온도보상 미적용)")
    profiles = sorted({lg["sdlog_profile"] for lg in logs})
    print(f"[SDLOG_PROFILE] {profiles}")
    rates = [lg["n_dp"] / lg["duration_s"] for lg in logs if lg["duration_s"] > 5]
    if rates:
        print(f"[differential_pressure 로깅률] {np.mean(rates):.2f} Hz "
              f"(min {np.min(rates):.2f} / max {np.max(rates):.2f}) "
              f"— PX4 logger 가 이 토픽을 1000 ms 간격으로 받는다")

    # --- 온도 채널 실제 분포 (회귀 전에 반드시 본다)
    print("\n[온도 채널 실측 분포 — 영점 구간]")
    zero_rows = build_sample_rows(logs, "is_zero")
    hover_rows = build_sample_rows(logs, "is_hover")
    print(f"  영점 표본 {len(zero_rows)}개 / 호버 표본 {len(hover_rows)}개")
    if not zero_rows:
        print("  영점 표본 0 — 중단")
        return 1
    for key, name in CHANNELS:
        col = key.replace("t_", "t_", 1)
        colname = {"t_dp": "t_dp_c", "t_pitot": "t_pitot_c",
                   "t_baro": "t_baro_c", "t_amb": "t_amb_c"}[key]
        v = np.array([r[colname] for r in zero_rows], dtype=float)
        finite = np.isfinite(v)
        if finite.sum() == 0:
            print(f"  {name:<48} 전부 NaN — 센서 미지원. 회귀 제외")
        else:
            print(f"  {name:<48} n_유효={finite.sum():3d}/{len(v)} "
                  f"범위 {v[finite].min():6.2f}~{v[finite].max():6.2f} °C")

    # --- 세션 분할
    dated = [(lg["wall"], lg) for lg in logs if lg["wall"] is not None]
    dated.sort(key=lambda kv: kv[0])
    sess_idx = split_sessions([w for w, _ in dated], gap_s=args.session_gap_min * 60)
    file_session = {lg["file"]: s for (w, lg), s in zip(dated, sess_idx)}
    undated = [lg["file"] for lg in logs if lg["wall"] is None]
    if undated:
        print(f"\n  [주의] 파일명에 벽시계 시각이 없어 세션 분할에서 빠진 로그 "
              f"{len(undated)}건: {undated}")
    for r in zero_rows:
        r["session"] = file_session.get(r["file"], -1)
    for r in hover_rows:
        r["session"] = file_session.get(r["file"], -1)

    y = np.array([r["raw_dp_pa"] for r in zero_rows])
    chan = {k: np.array([r[{"t_dp": "t_dp_c", "t_pitot": "t_pitot_c",
                            "t_baro": "t_baro_c", "t_amb": "t_amb_c"}[k]]
                        for r in zero_rows]) for k, _ in CHANNELS}

    reg_rows = []

    def record(label, scope, fit, xname="t_dp"):
        reg_rows.append({
            "scope": scope, "label": label, "x_channel": xname,
            "n": fit["n"] if fit else 0,
            "slope_pa_per_c": round(fit["slope"], 4) if fit else "",
            "slope_ci_lo": round(fit["slope_lo"], 4) if fit else "",
            "slope_ci_hi": round(fit["slope_hi"], 4) if fit else "",
            "intercept_pa": round(fit["intercept"], 3) if fit else "",
            "r2": round(fit["r2"], 4) if fit else "",
            "resid_sd_pa": round(fit["resid_sd"], 3) if fit else "",
            "x_span_c": round(fit["x_span"], 2) if fit else "",
            "note": "" if fit else "레버리지 부족",
        })

    print("\n[회귀 1 — 어느 온도 채널이 영점을 설명하는가 (영점 표본 전체)]")
    best = None
    for key, name in CHANNELS:
        fit = ols_fit(chan[key], y)
        print(fmt_fit(fit, name))
        record(name, "channel", fit, key)
        if fit and (best is None or fit["r2"] > best[1]["r2"]):
            best = (key, fit)
    if best is None:
        print("  회귀 가능한 온도 채널 없음 — 중단")
        return 1
    best_key, main_fit = best
    print(f"  → 최우수 채널: {dict(CHANNELS)[best_key]}")

    print("\n[회귀 2 — 하위집합 (오염·시점 분리)]")
    x = chan[best_key]
    pre = np.array([r["post_flight"] == 0 for r in zero_rows])
    for label, mask in [("이륙 전 지상 (프롭워시 이력 없음)", pre),
                        ("착륙 후 지상 (프롭워시 직후)", ~pre)]:
        fit = ols_fit(x[mask], y[mask])
        print(fmt_fit(fit, label))
        record(label, "subset", fit, best_key)

    print("\n[회귀 3 — 세션 내부 (within-session): 짧은 시간척도에서 온도를 따라가는가]")
    sessions = sorted({r["session"] for r in zero_rows})
    sess_mean_x, sess_mean_y, sess_label, sess_n = [], [], [], []
    for s in sessions:
        m = np.array([r["session"] == s for r in zero_rows])
        walls = sorted(r["wall_utc"] for r in zero_rows if r["session"] == s)
        label = f"세션{s} {walls[0][5:16]}..{walls[-1][11:16]}" if walls else f"세션{s}"
        fit = ols_fit(x[m], y[m])
        print(fmt_fit(fit, label))
        record(label, "within_session", fit, best_key)
        sess_mean_x.append(float(x[m].mean()))
        sess_mean_y.append(float(y[m].mean()))
        sess_label.append(label)
        sess_n.append(int(m.sum()))

    print("\n[회귀 4 — 세션 사이 (between-session): 세션 평균 1점씩]")
    sfit = ols_fit(sess_mean_x, sess_mean_y)
    print(fmt_fit(sfit, "세션 평균 회귀"))
    record("세션 평균 회귀", "between_session", sfit, best_key)
    if sfit:
        print("  세션별 잔차:")
        for lb, xx, yy, n_s in zip(sess_label, sess_mean_x, sess_mean_y, sess_n):
            print(f"    {lb:<32} n={n_s:3d} T={xx:6.2f} raw={yy:7.2f} "
                  f"잔차={yy - (sfit['intercept'] + sfit['slope'] * xx):+6.2f} Pa")

    print("\n[호버 구간 교차검증 — 프롭워시 오염이 있으므로 참고용]")
    if hover_rows:
        hx = np.array([r["t_dp_c"] for r in hover_rows])
        hy = np.array([r["raw_dp_pa"] for r in hover_rows])
        hfit = ols_fit(hx, hy)
        print(fmt_fit(hfit, "호버 표본 회귀"))
        record("호버 표본 회귀", "hover", hfit, "t_dp")
        print(f"  호버 표본 raw dp 산포 sd={hy.std(ddof=1):.2f} Pa "
              f"(영점 표본 잔차 sd={main_fit['resid_sd']:.2f} Pa) "
              f"→ 클수록 프롭워시·기류 오염이 크다")

    print("\n[baro 온도를 피토센서 온도의 대리지표로 쓸 수 있나 — 두 채널의 격차·상관]")
    print("  (피토관과 픽스호크가 약 60 cm 이격돼 있어 벤치에서 실제 쟁점이 된다)")
    for s in sessions:
        files = sorted({r["file"] for r in zero_rows if r["session"] == s})
        d = [r["t_baro_c"] - r["t_dp_c"] for r in zero_rows if r["session"] == s]
        boots = [r["boot_s"] for r in zero_rows if r["session"] == s]
        print(f"  세션{s}: n={len(d):3d} Tbaro-Tdp 평균={np.mean(d):+6.2f} °C "
              f"(범위 {np.min(d):+6.2f}~{np.max(d):+6.2f})  부팅경과 {np.min(boots):6.0f}~{np.max(boots):6.0f} s")
    alld = np.array([r["t_baro_c"] - r["t_dp_c"] for r in zero_rows])
    print(f"  → 전체: Tbaro-Tdp 는 {alld.min():+.2f}~{alld.max():+.2f} °C 로 "
          f"{alld.max() - alld.min():.1f} °C 흔들린다. 고정 오프셋이 아니다.")
    print("  로그 내부 상관(부호가 뒤집히면 대리지표로 못 쓴다는 직접 증거):")
    for lg in sorted(logs, key=lambda l: (l["wall"] or dt.datetime.min)):
        if lg["n_dp"] < 40:
            continue
        m = np.isfinite(lg["t_dp"]) & np.isfinite(lg["t_baro"])
        if m.sum() < 10 or np.ptp(lg["t_dp"][m]) < 0.3:
            continue
        r = float(np.corrcoef(lg["t_dp"][m], lg["t_baro"][m])[0, 1])
        print(f"    {lg['file']:<42} corr(Tdp,Tbaro)={r:+.3f}  "
              f"Tdp {lg['t_dp'][m].min():.2f}~{lg['t_dp'][m].max():.2f}  "
              f"Tbaro {lg['t_baro'][m].min():.2f}~{lg['t_baro'][m].max():.2f}")

    print("\n[비행 중 센서 온도 추세 — 벤치가 재현해야 할 실제 열이력]")
    trends = []
    for lg in sorted(logs, key=lambda l: (l["wall"] or dt.datetime.min)):
        if lg["duration_s"] < 20:
            continue
        rate = (lg["t_dp"][-1] - lg["t_dp"][0]) / lg["duration_s"] * 60.0
        trends.append(rate)
        print(f"  {lg['file']:<42} {lg['duration_s']:6.1f} s  "
              f"Tdp {lg['t_dp'][0]:6.2f} → {lg['t_dp'][-1]:6.2f}  "
              f"({lg['t_dp'][-1] - lg['t_dp'][0]:+5.2f} °C, {rate:+6.2f} °C/min)")
    if trends:
        print(f"  → 비행 중 추세 중앙값 {np.median(trends):+.2f} °C/min "
              f"(범위 {np.min(trends):+.2f}~{np.max(trends):+.2f}). "
              "음수 = 기류가 햇볕에 달궈진 센서를 식힌다.")
    for day in sorted({(lg["wall"].date() if lg["wall"] else None) for lg in logs} - {None}):
        vs = np.concatenate([lg["t_dp"] for lg in logs
                             if lg["wall"] and lg["wall"].date() == day])
        print(f"  {day} 전체 Tdp 범위: {vs.min():.2f} ~ {vs.max():.2f} °C  (n={len(vs)})")

    print("\n[SENS_DPRES_OFF 로부터 역산한 캘리브레이션 온도]")
    for off in offs:
        t_implied = (main_fit["intercept"] - off) / -main_fit["slope"]
        flag = "" if main_fit["x_min"] <= t_implied <= main_fit["x_max"] else "  ← 관측범위 밖 = 외삽"
        print(f"  SENS_DPRES_OFF={off:8.3f} Pa → T_cal ≈ {t_implied:5.2f} °C{flag}")

    # --- 2026-07-30 flight02 영향 계산 (이 분석의 출발점)
    f02 = next((lg for lg in logs if lg["file"].startswith("log_269_2026-07-30")), None)
    if f02 is not None:
        print("\n[2026-07-30 flight02 — 영점 오차가 천이 판정에 미친 영향]")
        stored = f02["sens_dpres_off"]
        sess6 = [r["raw_dp_pa"] for r in zero_rows if r["session"] == file_session.get(f02["file"], -1)]
        model_zero = main_fit["intercept"] + main_fit["slope"] * float(np.mean(
            [r["t_dp_c"] for r in zero_rows if r["session"] == file_session.get(f02["file"], -1)]))
        print(f"  저장 영점 SENS_DPRES_OFF        = {stored:8.3f} Pa")
        print(f"  같은 세션 지상 영점 실측 평균     = {np.mean(sess6):8.3f} Pa "
              f"(n={len(sess6)}, sd={np.std(sess6, ddof=1):.2f})")
        print(f"  회귀 모형이 예측한 영점          = {model_zero:8.3f} Pa")
        print(f"  → 영점 오차 = {np.mean(sess6) - stored:+.2f} Pa")
        for label, zero in [("저장값 그대로(=오차 없음 가정)", stored),
                            ("호버 raw 실측 -38.97 (선행 분석)", -38.97),
                            (f"같은 세션 지상 영점 실측 {np.mean(sess6):.2f}", float(np.mean(sess6))),
                            (f"회귀 모형 예측 {model_zero:.2f}", model_zero)]:
            print(f"    영점={zero:8.2f} Pa → 보고 CAS 9.56 m/s 는 실제 "
                  f"{recompute_cas(9.56, stored, zero):6.2f} m/s "
                  f"(VT_ARSP_TRANS={f02['vt_arsp_trans']})")
        need_dp = dp_from_cas(float(f02["vt_arsp_trans"] or 12.0))
        need_zero = dp_from_cas(9.56) + stored - need_dp
        print(f"  천이 임계 {f02['vt_arsp_trans']} m/s 를 넘기려면 영점이 {need_zero:.2f} Pa "
              f"여야 했다 (실측 영점보다 {need_zero - np.mean(sess6):+.1f} Pa 더 음수) "
              "— 즉 센서 교정만으로는 천이가 완주되지 않는다.")

    # --- 산출물
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "zero_samples.csv"), zero_rows)
    write_csv(os.path.join(args.out, "hover_samples.csv"), hover_rows)
    write_csv(os.path.join(args.out, "regressions.csv"), reg_rows)
    per_log = [{
        "file": lg["file"], "flight_dir": lg["flight_dir"],
        "wall_utc": lg["wall"].isoformat() if lg["wall"] else "",
        "session": file_session.get(lg["file"], -1),
        "duration_s": round(lg["duration_s"], 1), "n_dp": lg["n_dp"],
        "boot_s_first_dp": round(lg["boot_s_first"], 1),
        "device_id": lg["device_id"], "sens_dpres_off_pa": lg["sens_dpres_off"],
        "n_zero": int(lg["is_zero"].sum()), "n_hover": int(lg["is_hover"].sum()),
        "zero_raw_mean_pa": round(float(lg["raw"][lg["is_zero"]].mean()), 3) if lg["is_zero"].any() else "",
        "zero_raw_sd_pa": round(float(lg["raw"][lg["is_zero"]].std(ddof=1)), 3) if lg["is_zero"].sum() > 1 else "",
        "zero_t_dp_mean_c": round(float(lg["t_dp"][lg["is_zero"]].mean()), 3) if lg["is_zero"].any() else "",
        "t_dp_min_c": round(float(np.nanmin(lg["t_dp"])), 2),
        "t_dp_max_c": round(float(np.nanmax(lg["t_dp"])), 2),
        "t_dp_first_c": round(float(lg["t_dp"][0]), 2),
        "t_dp_last_c": round(float(lg["t_dp"][-1]), 2),
    } for lg in sorted(logs, key=lambda l: (l["wall"] or dt.datetime.min))]
    write_csv(os.path.join(args.out, "per_log.csv"), per_log)
    print(f"\nCSV 저장: {args.out}/{{zero_samples,hover_samples,per_log,regressions}}.csv")

    if not args.no_plots:
        make_plots(args.out, zero_rows, hover_rows, main_fit, sfit,
                   sess_mean_x, sess_mean_y, sess_label, chan, y, offs)
        print(f"플롯 저장: {args.out}/*.png")

    if args.json:
        print("\n" + json.dumps({
            "n_logs": len(logs), "n_zero_samples": len(zero_rows),
            "device_ids": devs, "sens_dpres_off_values": offs,
            "best_channel": dict(CHANNELS)[best_key],
            "main_fit": main_fit, "session_fit": sfit,
        }, indent=2, ensure_ascii=False, default=float))
    return 0


# 검증된 팔레트 (dataviz 기준 팔레트, light 모드, all-pairs 통과)
C_DATA = "#2a78d6"     # slot1 blue  — 영점 표본
C_FIT = "#eb6834"      # slot2 orange — 회귀선
C_HOVER = "#1baf7a"    # slot3 aqua  — 호버 표본(직접 라벨로 relief)
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e3e2de"
C_SURF = "#fcfcfb"


def make_plots(out, zero_rows, hover_rows, main_fit, sfit,
               sess_mean_x, sess_mean_y, sess_label, chan, y, offs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 축·제목이 한글이라 CJK 폰트가 없으면 두부(tofu)로 깨진다 — 있는 것 중 첫 번째를 쓴다.
    have = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("NanumGothic", "Noto Sans CJK KR", "NanumBarunGothic", "NanumSquare"):
        if cand in have:
            plt.rcParams["font.family"] = cand
            break
    else:
        print("  [주의] 한글 폰트를 못 찾았다 — 플롯 라벨이 깨질 수 있다 "
              "(apt install fonts-nanum 후 재실행)")
    plt.rcParams["axes.unicode_minus"] = False

    def style(ax):
        ax.set_facecolor(C_SURF)
        ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C_GRID)
        ax.tick_params(colors=C_INK2, labelsize=9)

    x = np.array([r["t_dp_c"] for r in zero_rows])

    # --- 1. 본 그림: 영점 vs 센서 다이 온도
    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=150, facecolor=C_SURF)
    style(ax)
    ax.scatter(x, y, s=34, color=C_DATA, alpha=0.75, linewidths=0, zorder=3,
               label=f"지상 영점 표본 (n={len(y)})")
    xs = np.linspace(x.min() - 1, x.max() + 1, 100)
    ax.plot(xs, main_fit["intercept"] + main_fit["slope"] * xs, color=C_FIT,
            linewidth=2, zorder=4,
            label=(f"OLS  {main_fit['slope']:+.2f} Pa/°C "
                   f"(95% CI {main_fit['slope_lo']:+.2f}…{main_fit['slope_hi']:+.2f}), "
                   f"R²={main_fit['r2']:.2f}"))
    for off in offs:
        ax.axhline(off, color=C_INK2, linestyle=(0, (5, 4)), linewidth=1.2, zorder=2)
    ax.annotate(f"저장된 SENS_DPRES_OFF ({', '.join(f'{o:.2f}' for o in offs)} Pa)",
                xy=(x.min() - 0.5, max(offs)), xytext=(0, 6),
                textcoords="offset points", color=C_INK2, fontsize=9, va="bottom")
    ax.set_xlabel("differential_pressure.temperature  [°C]", color=C_INK, fontsize=10)
    ax.set_ylabel("raw differential_pressure_pa  [Pa]", color=C_INK, fontsize=10)
    ax.set_title("피토 차압센서 영점 vs 센서 다이 온도 (실비행 ulog, 지상 정지·모터정지 구간)",
                 color=C_INK, fontsize=11.5, pad=12)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for t in leg.get_texts():
        t.set_color(C_INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "zero_vs_temp.png"), facecolor=C_SURF)
    plt.close(fig)

    # --- 2. 채널 비교 (small multiples — 카테고리 색 대신 패널 분리)
    usable = [(k, n) for k, n in CHANNELS if np.isfinite(chan[k]).sum() > 3
              and np.ptp(chan[k][np.isfinite(chan[k])]) > 0]
    fig, axes = plt.subplots(1, len(usable), figsize=(4.2 * len(usable), 4.2),
                             dpi=150, facecolor=C_SURF, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (k, name) in zip(axes, usable):
        style(ax)
        v = chan[k]
        m = np.isfinite(v)
        f = ols_fit(v[m], y[m])
        ax.scatter(v[m], y[m], s=22, color=C_DATA, alpha=0.7, linewidths=0, zorder=3)
        if f:
            xs = np.linspace(v[m].min(), v[m].max(), 50)
            ax.plot(xs, f["intercept"] + f["slope"] * xs, color=C_FIT, linewidth=2, zorder=4)
            ax.set_title(f"{name}\n{f['slope']:+.2f} Pa/°C · R²={f['r2']:.2f}",
                         color=C_INK, fontsize=9.5)
        ax.set_xlabel("[°C]", color=C_INK2, fontsize=9)
    axes[0].set_ylabel("raw differential_pressure_pa [Pa]", color=C_INK, fontsize=10)
    fig.suptitle("온도 채널별 설명력 비교 — 같은 영점 표본, x축만 교체",
                 color=C_INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "zero_vs_temp_channels.png"), facecolor=C_SURF)
    plt.close(fig)

    # --- 3. 세션 내부 vs 세션 사이
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, facecolor=C_SURF, sharey=True)
    ax = axes[0]
    style(ax)
    for s in sorted({r["session"] for r in zero_rows}):
        m = np.array([r["session"] == s for r in zero_rows])
        ax.plot(x[m], y[m], marker="o", markersize=5, linestyle="none",
                color=C_DATA, alpha=0.45, zorder=3)
        f = ols_fit(x[m], y[m])
        if f and f["x_span"] > 2:
            xs = np.linspace(x[m].min(), x[m].max(), 20)
            ax.plot(xs, f["intercept"] + f["slope"] * xs, color=C_DATA,
                    linewidth=1.6, alpha=0.9, zorder=4)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, main_fit["intercept"] + main_fit["slope"] * xs, color=C_FIT,
            linewidth=2, zorder=5, label="전체 회귀")
    ax.set_title("세션 내부 (각 전원세션의 자체 기울기)", color=C_INK, fontsize=10.5)
    ax.set_xlabel("differential_pressure.temperature [°C]", color=C_INK2, fontsize=9)
    ax.set_ylabel("raw differential_pressure_pa [Pa]", color=C_INK, fontsize=10)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for t in leg.get_texts():
        t.set_color(C_INK2)

    ax = axes[1]
    style(ax)
    ax.scatter(sess_mean_x, sess_mean_y, s=60, color=C_DATA, linewidths=0, zorder=3)
    for xx, yy, lb in zip(sess_mean_x, sess_mean_y, sess_label):
        ax.annotate(lb.split()[1][:11], (xx, yy), textcoords="offset points",
                    xytext=(6, 5), fontsize=7.5, color=C_INK2)
    if sfit:
        xs = np.linspace(min(sess_mean_x) - 1, max(sess_mean_x) + 1, 50)
        ax.plot(xs, sfit["intercept"] + sfit["slope"] * xs, color=C_FIT,
                linewidth=2, zorder=4,
                label=f"세션평균 OLS {sfit['slope']:+.2f} Pa/°C · R²={sfit['r2']:.2f}")
        leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
        for t in leg.get_texts():
            t.set_color(C_INK2)
    ax.set_title("세션 사이 (세션 평균 1점씩)", color=C_INK, fontsize=10.5)
    ax.set_xlabel("differential_pressure.temperature [°C]", color=C_INK2, fontsize=9)
    fig.suptitle("드리프트는 세션 사이에서만 보인다 — 세션 내부에서는 안 보인다",
                 color=C_INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "within_vs_between_session.png"), facecolor=C_SURF)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
