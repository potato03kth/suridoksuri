#!/usr/bin/env python3
"""비행 로그(.ulg) 표준 진단 리포트 — 매 비행마다 pyulog 코드를 새로 짜지 않기 위한 고정 스크립트.

배경: 2026-07-20/21 flight02·flight03 사고 분석 세션에서 매번 임시 pyulog 스크립트를
왔다갔다 짜며 세션 컨텍스트를 크게 소모했다. 그때 알아낸 진단 로직(쿼터니언 디코드,
CA_ROTOR 파라미터 기반 모터 위치매핑, nav_state_user_intention+failsafe로 모드전환
출처 판별, 축별 얼로케이터 포화(unallocated_torque) 시점 특정, ground_contact 기반
이함 순간 검출)을 이 스크립트 하나에 고정해, 다음 비행부터는 이 스크립트를 실행해
구조화된 요약만 받고 세션에서 코드를 재작성하지 않는 게 목적이다.

이 스크립트는 "원인이 뭐다"라는 해석은 하지 않는다 — 해석에 필요한 정확한 사실을
빠짐없이 뽑아 보여주는 것까지만 한다. 특히 스크립트가 명시적으로 다루지 않는
토픽도 목록으로 남겨(§11) 조용히 놓치는 부분이 없게 한다.

사용 예:
    pip install pyulog                                      # 1회
    python3 tools/flight_logs/analyze_flight.py logs/2026-07-20_flight03/
    python3 tools/flight_logs/analyze_flight.py logs/2026-07-20_flight03/ --json

플라이트 폴더 안의 .ulg가 여러 개면(재시동/재arm으로 인한 짧은 블립 포함) 가장 긴
것을 주 로그로 전체 분석하고, 나머지는 길이만 요약해 존재 자체를 놓치지 않게 한다.

기본으로 리포트를 <플라이트폴더>/analysis_auto.md 에도 저장한다(--no-write로 끔) —
notes.md 작성 시 이 파일을 인용하면 된다. 원본 데이터를 코드 판단 없이 그대로 뽑는
쪽이라 사람/LLM의 해석은 notes.md에 남기고, 이 파일은 갱신될 때마다 덮어써도 된다.
"""

import argparse
import glob
import json
import math
import os
import re
import sys

# ---------------------------------------------------------------------------
# 순수 함수 (pytest 대상 — test_flight_logs.py) — pyulog 불필요, 값만 받아 계산
# ---------------------------------------------------------------------------

NAV_STATE_NAMES = {
    0: "MANUAL", 1: "ALTCTL", 2: "POSCTL", 3: "AUTO_MISSION", 4: "AUTO_LOITER",
    5: "AUTO_RTL", 6: "POSITION_SLOW", 10: "ACRO", 12: "DESCEND",
    13: "TERMINATION", 14: "OFFBOARD", 15: "STAB", 17: "AUTO_TAKEOFF",
    18: "AUTO_LAND", 19: "AUTO_FOLLOW_TARGET", 20: "AUTO_PRECLAND",
    21: "ORBIT", 22: "AUTO_VTOL_TAKEOFF", 23: "EXTERNAL1",
}

ARMING_STATE_NAMES = {
    0: "INIT", 1: "STANDBY", 2: "ARMED", 3: "STANDBY_ERROR",
    4: "SHUTDOWN", 5: "IN_AIR_RESTORE",
}


def nav_state_name(n):
    return NAV_STATE_NAMES.get(int(n), f"UNKNOWN({int(n)})")


def arming_state_name(n):
    return ARMING_STATE_NAMES.get(int(n), f"UNKNOWN({int(n)})")


def quat_to_euler_deg(w, x, y, z):
    """PX4 쿼터니언(w,x,y,z) → roll/pitch/yaw(도). FRD 바디프레임 기준."""
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def motor_label(px, py):
    """CA_ROTORx_PX/PY 부호로 모터 위치 라벨을 만든다 (바디프레임: X전방+, Y우측+)."""
    fb = "전" if px > 1e-6 else ("후" if px < -1e-6 else "중")
    lr = "우" if py > 1e-6 else ("좌" if py < -1e-6 else "중")
    return f"{fb}{lr}"


def classify_mode_transition(user_intention_changed, failsafe_after):
    """모드전환 한 건의 출처를 (가능한 범위에서) 분류한다.

    user_intention_changed: 이 전환과 동시에 nav_state_user_intention도 바뀌었는가
    failsafe_after: 전환 직후 vehicle_status.failsafe 값(0/1)

    반환값은 확정적 라벨이 아니라 "이 필드들만으로 구분 가능한 것까지"다.
    INTENTIONAL은 조종기/지상국/컴패니언 커맨드를 다 포함할 수 있어 출처까지
    확정하려면 manual_control_switches·vehicle_command를 추가로 대조해야 한다.
    """
    if failsafe_after:
        return "FAILSAFE_FORCED"
    if user_intention_changed:
        return "INTENTIONAL_CHANGE"
    return "AUTO_RECOVERY_OR_UNCLASSIFIED"


def detect_rate_onset(times, rates_deg_s, baseline_end_s=1.5, k=4.0, consec=3,
                       floor_deg_s=8.0):
    """각속도가 '초반 정지/트림 구간' 잡음 수준을 벗어나 튀는 최초 시점을 찾는다
    (휴리스틱, 육안 재확인 필요 — 절대적 판정 아님).

    times[0]~baseline_end_s 구간을 '조용한 기준선'으로 삼아 평균·표준편차를 구하고,
    thresh = |평균| + max(k*표준편차, floor_deg_s) 를 넘는 상태가 consec 샘플
    연속으로 유지되는 첫 시점을 반환한다(단발 잡음 스파이크 배제). 기준선 구간이
    너무 짧거나(<5샘플) 기준선을 못 잡으면 None.

    고정 window로 '직전 N샘플 평균'을 볼 때, 값이 완만히 램프업하는 구간에서는
    직전 구간도 이미 높아져 있어 '조용했던 시점'을 못 찾고 탐지가 계속 뒤로
    밀리는 문제가 있었다 — 그래서 기준선을 로그 초반 한 번만 고정해서 쓴다.
    """
    baseline = [rates_deg_s[i] for i in range(len(times)) if times[i] <= baseline_end_s]
    if len(baseline) < 5:
        return None
    mean_b = sum(baseline) / len(baseline)
    var_b = sum((v - mean_b) ** 2 for v in baseline) / len(baseline)
    std_b = var_b ** 0.5
    thresh = abs(mean_b) + max(k * std_b, floor_deg_s)

    run = 0
    for i in range(len(times)):
        if times[i] <= baseline_end_s:
            continue
        if abs(rates_deg_s[i]) > thresh:
            run += 1
            if run >= consec:
                onset_i = i - consec + 1
                return times[onset_i], rates_deg_s[onset_i], thresh
        else:
            run = 0
    return None


def first_sustained_nonzero(times, vals, eps=1e-3, consec=3, skip_before=1.0):
    """0 근처에 머물던 값이 eps를 넘는 상태로 consec 샘플 연속 유지되는 첫 시점.

    arm 직후 한두 샘플만 튀는 트랜지언트(컨트롤러 초기화 노이즈)를 '포화 시작'으로
    오판하지 않기 위해 연속조건을 둔다 — 단발 블립과 실제 지속 포화를 구분.
    """
    run = 0
    for i in range(len(times)):
        if times[i] < skip_before:
            continue
        if abs(vals[i]) > eps:
            run += 1
            if run >= consec:
                idx = i - consec + 1
                return times[idx], vals[idx]
        else:
            run = 0
    return None


def parse_transition_alt(notes_md_text):
    """notes.md의 '비행 조건' 줄에서 transition_alt:=X 를 뽑는다. 없으면 None."""
    m = re.search(r"transition_alt:=([\d.]+)", notes_md_text or "")
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# I/O — pyulog 로딩 + 리포트 조립
# ---------------------------------------------------------------------------

def _get(ulog, name, multi_id=0):
    for d in ulog.data_list:
        if d.name == name and d.multi_id == multi_id:
            return d.data
    return None


def _to_native(v):
    """numpy 스칼라/배열을 JSON 직렬화 가능한 파이썬 값으로."""
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except ImportError:
        pass
    return v


def analyze_ulog(path):
    """단일 .ulg 하나를 분석해 report(dict) 반환. 섹션별로 best-effort — 토픽이
    없으면 해당 섹션에 사유를 남기고 넘어간다(예외로 전체를 죽이지 않음)."""
    from pyulog import ULog
    ulog = ULog(path)
    t0 = ulog.start_timestamp
    duration_s = (ulog.last_timestamp - t0) / 1e6

    report = {"file": path, "duration_s": round(duration_s, 2)}
    present_topics = sorted({(d.name, d.multi_id) for d in ulog.data_list})
    covered_topics = set()

    def mark(name, multi_id=0):
        covered_topics.add((name, multi_id))

    def t_of(data):
        return (data["timestamp"] - t0) / 1e6

    # -- §1 파라미터 스냅샷 (모터 위치매핑 + 주요 안전 파라미터) --------------
    params = ulog.initial_parameters
    rotor_count = int(params.get("CA_ROTOR_COUNT", 0) or 0)
    motors = []
    for i in range(rotor_count):
        px = params.get(f"CA_ROTOR{i}_PX")
        py = params.get(f"CA_ROTOR{i}_PY")
        km = params.get(f"CA_ROTOR{i}_KM")
        if px is None or py is None:
            continue
        motors.append({"index": i, "label": motor_label(px, py), "px": px, "py": py, "km": km})
    report["motors"] = motors
    report["params_snapshot"] = {
        k: params.get(k) for k in [
            "BAT_LOW_THR", "BAT_CRIT_THR", "BAT_EMERGEN_THR",
            "FD_IMB_PROP_THR", "MPC_TKO_SPEED", "COM_ARM_WO_GPS",
            "EKF2_HGT_REF",
        ] if k in params
    }

    # -- §2 고도/AMSL 정합성 ---------------------------------------------------
    hp = _get(ulog, "home_position"); mark("home_position")
    lp = _get(ulog, "vehicle_local_position"); mark("vehicle_local_position")
    agl_section = {}
    if hp is not None and lp is not None:
        home_alt = float(hp["alt"][0])
        ref_alt = float(lp["ref_alt"][0])
        z = lp["z"]
        t_lp = t_of(lp)
        agl = ref_alt - z - home_alt
        i_max = int(agl.argmax()); i_min = int(agl.argmin())
        agl_section = {
            "home_position_alt_amsl": round(home_alt, 2),
            "ekf_ref_alt_amsl": round(ref_alt, 2),
            "amsl_offset_note": (
                "두 값 차이가 그대로 초기 pos_ned[2] 오프셋이 된다 — 정합적이면 좌표버그 아님, "
                "그냥 EKF 원점이 지면과 다른 기준일 뿐"),
            "agl_max_m": round(float(agl[i_max]), 2), "agl_max_at_s": round(float(t_lp[i_max]), 2),
            "agl_min_m": round(float(agl[i_min]), 2), "agl_min_at_s": round(float(t_lp[i_min]), 2),
            "agl_end_m": round(float(agl[-1]), 2),
        }
    report["altitude"] = agl_section

    # -- §3 모드권한 타임라인 ---------------------------------------------------
    vs = _get(ulog, "vehicle_status"); mark("vehicle_status")
    mode_transitions = []
    if vs is not None and "nav_state_user_intention" in vs:
        t_vs = t_of(vs)
        nav = vs["nav_state"]; intent = vs["nav_state_user_intention"]
        failsafe = vs["failsafe"]; arming = vs["arming_state"]
        # 일부 로그의 최초 1~2개 샘플이 uint64 sentinel 타임스탬프(≈18446744073709.55s,
        # 즉 2**64-1을 마이크로초로 나눈 값)를 갖고 나오는 경우가 있어 걸러낸다.
        valid_start = next((i for i in range(len(t_vs)) if 0 <= t_vs[i] <= duration_s + 1), 0)
        nav = nav[valid_start:]; intent = intent[valid_start:]
        failsafe = failsafe[valid_start:]; arming = arming[valid_start:]
        t_vs = t_vs[valid_start:]
        prev_nav = None
        for i in range(len(nav)):
            if prev_nav is not None and nav[i] == prev_nav:
                continue
            intent_changed = (i > 0 and intent[i] != intent[i - 1])
            label = classify_mode_transition(intent_changed, bool(failsafe[i]))
            mode_transitions.append({
                "t_s": round(float(t_vs[i]), 2),
                "nav_state": f"{int(nav[i])}:{nav_state_name(nav[i])}",
                "user_intention": f"{int(intent[i])}:{nav_state_name(intent[i])}",
                "failsafe": int(failsafe[i]),
                "arming_state": arming_state_name(arming[i]),
                "classification": label,
            })
            prev_nav = nav[i]
    report["mode_transitions"] = mode_transitions
    report["mode_transitions_note"] = (
        "INTENTIONAL_CHANGE는 조종기/지상국/컴패니언 커맨드를 다 포함 — 출처를 더 좁히려면 "
        "manual_control_switches/vehicle_command를 별도로 대조할 것. FAILSAFE_FORCED는 "
        "PX4 자체 강제전환이 확실함(user_intention 불변).")

    # -- §4 자세 타임라인 (실측 + 목표 + 각속도 이상조짐) -----------------------
    attitude_section = {}
    va = _get(ulog, "vehicle_attitude"); mark("vehicle_attitude")
    if va is not None:
        t_va = t_of(va)
        rolls, pitches, yaws = [], [], []
        for i in range(len(va["q[0]"])):
            r, p, y = quat_to_euler_deg(va["q[0]"][i], va["q[1]"][i], va["q[2]"][i], va["q[3]"][i])
            rolls.append(r); pitches.append(p); yaws.append(y)
        i_r = max(range(len(rolls)), key=lambda k: abs(rolls[k]))
        i_p = max(range(len(pitches)), key=lambda k: abs(pitches[k]))
        attitude_section["roll_deg_max_abs"] = round(rolls[i_r], 2)
        attitude_section["roll_deg_max_abs_at_s"] = round(float(t_va[i_r]), 2)
        attitude_section["pitch_deg_max_abs"] = round(pitches[i_p], 2)
        attitude_section["pitch_deg_max_abs_at_s"] = round(float(t_va[i_p]), 2)

        vas = _get(ulog, "vehicle_attitude_setpoint"); mark("vehicle_attitude_setpoint")
        if vas is not None and "q_d[0]" in vas:
            t_vas = t_of(vas)
            sp_rolls = []
            for i in range(len(vas["q_d[0]"])):
                r, _, _ = quat_to_euler_deg(vas["q_d[0]"][i], vas["q_d[1]"][i], vas["q_d[2]"][i], vas["q_d[3]"][i])
                sp_rolls.append(r)
            i_sp = max(range(len(sp_rolls)), key=lambda k: abs(sp_rolls[k]))
            attitude_section["roll_setpoint_deg_max_abs"] = round(sp_rolls[i_sp], 2)
            attitude_section["roll_setpoint_deg_max_abs_at_s"] = round(float(t_vas[i_sp]), 2)
            attitude_section["roll_setpoint_note"] = (
                "이 값이 실측 roll_deg_max_abs 근처까지 크면 세트포인트 자체 결함(순간점프 등) 의심 — "
                "0 근처로 작으면 컨트롤러는 정상 수평을 명령했는데 실제가 못 따라간 것.")

        av = _get(ulog, "vehicle_angular_velocity"); mark("vehicle_angular_velocity")
        if av is not None:
            t_av = t_of(av)
            for axis, key in [("roll", "xyz[0]"), ("pitch", "xyz[1]"), ("yaw", "xyz[2]")]:
                rates_deg = [math.degrees(v) for v in av[key]]
                onset = detect_rate_onset(list(t_av), rates_deg)
                if onset:
                    attitude_section[f"{axis}_rate_onset_candidate_s"] = round(onset[0], 2)
                    attitude_section[f"{axis}_rate_onset_candidate_deg_s"] = round(onset[1], 1)
                    attitude_section[f"{axis}_rate_onset_threshold_deg_s"] = round(onset[2], 1)
            attitude_section["rate_onset_note"] = (
                "휴리스틱 후보일 뿐 — 잡음(±5deg/s 이내)에서 15deg/s 이상으로 튄 첫 시점. "
                "이 근처를 §5 얼로케이터 포화 시점·§9 이함 시점과 대조해 진짜 트리거인지 육안 확인할 것.")
    report["attitude"] = attitude_section

    # -- §5 얼로케이터 포화(축별) -----------------------------------------------
    saturation_section = {}
    cas = _get(ulog, "control_allocator_status"); mark("control_allocator_status")
    if cas is not None:
        t_cas = t_of(cas)
        axis_names = ["roll(x)", "pitch(y)", "yaw(z)"]
        for ai, axis in enumerate(axis_names):
            key = f"unallocated_torque[{ai}]"
            if key not in cas:
                continue
            vals = cas[key]
            onset = first_sustained_nonzero(list(t_cas), list(vals))
            if onset is not None:
                saturation_section[axis] = {
                    "first_saturation_at_s": round(onset[0], 2),
                    "unallocated_torque_at_onset": round(float(onset[1]), 4),
                }
            else:
                saturation_section[axis] = {"first_saturation_at_s": None, "note": "비행 내내 지속적 포화 없음(arm 직후 단발 블립은 제외)"}
        if "torque_setpoint_achieved" in cas:
            saturation_section["torque_setpoint_achieved_true_fraction"] = round(
                float(cas["torque_setpoint_achieved"].astype(bool).mean()), 3)
            saturation_section["achieved_flag_note"] = (
                "이 플래그는 roll/pitch/yaw/thrust 통합 — 축별 판단은 위 unallocated_torque[]를 볼 것. "
                "포화 시작 시점을 그 순간의 attitude.roll_deg_max_abs_at_s 등과 비교해 '오차가 이미 컸을 때부터 "
                "포화가 시작됐는지'를 확인하라 — 그렇다면 포화는 원인이 아니라 결과일 가능성이 크다.")
    report["allocator_saturation"] = saturation_section

    # -- §6 모터별 커맨드 --------------------------------------------------------
    motor_section = {}
    am = _get(ulog, "actuator_motors"); mark("actuator_motors")
    if am is not None:
        for i in range(rotor_count or 4):
            key = f"control[{i}]"
            if key not in am:
                continue
            vals = am[key]
            label = next((m["label"] for m in motors if m["index"] == i), f"motor{i}")
            motor_section[f"{i}:{label}"] = {
                "mean": round(float(vals.mean()), 3),
                "max": round(float(vals.max()), 3),
                "min": round(float(vals.min()), 3),
            }
    report["motors_command"] = motor_section

    # -- §7 배터리 ----------------------------------------------------------------
    battery_section = {}
    bat = _get(ulog, "battery_status"); mark("battery_status")
    if bat is not None:
        battery_section = {
            "voltage_rest_v": round(float(bat["voltage_v"][0]), 2),
            "voltage_min_v": round(float(bat["voltage_v"].min()), 2),
            "current_max_a": round(float(bat["current_a"].max()), 1),
        }
        if "remaining" in bat:
            battery_section["remaining_start"] = round(float(bat["remaining"][0]), 2)
            battery_section["remaining_end"] = round(float(bat["remaining"][-1]), 2)
    report["battery"] = battery_section

    # -- §8 실패감지기 --------------------------------------------------------------
    fd_section = {}
    fds = _get(ulog, "failure_detector_status"); mark("failure_detector_status")
    if fds is not None:
        if "imbalanced_prop_metric" in fds:
            m = fds["imbalanced_prop_metric"]
            i_max = int(abs(m).argmax())
            fd_section["imbalanced_prop_metric_max_abs"] = round(float(m[i_max]), 3)
        for flag in ["fd_imbalanced_prop", "fd_motor", "motor_failure_mask"]:
            if flag in fds:
                import numpy as np
                fd_section[f"{flag}_ever_nonzero"] = bool(np.any(fds[flag] != 0))
    report["failure_detector"] = fd_section

    # -- §9 착지감지기 / 이함 순간 ---------------------------------------------
    liftoff_section = {}
    ld = _get(ulog, "vehicle_land_detected"); mark("vehicle_land_detected")
    mc = _get(ulog, "manual_control_setpoint"); mark("manual_control_setpoint")
    if ld is not None and "ground_contact" in ld:
        t_ld = t_of(ld)
        gc = ld["ground_contact"]
        liftoff_t = None
        for i in range(1, len(gc)):
            if gc[i - 1] == 1 and gc[i] == 0:
                liftoff_t = float(t_ld[i])
                break
        liftoff_section["liftoff_t_s"] = round(liftoff_t, 2) if liftoff_t is not None else None
        if liftoff_t is not None and va is not None:
            t_va = t_of(va)
            t_mc = t_of(mc) if mc is not None else None
            samples = []
            for dt in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                t = liftoff_t + dt
                j = min(range(len(t_va)), key=lambda k: abs(t_va[k] - t))
                r, _, _ = quat_to_euler_deg(va["q[0]"][j], va["q[1]"][j], va["q[2]"][j], va["q[3]"][j])
                entry = {"t_s": round(t, 2), "roll_deg": round(r, 2)}
                if t_mc is not None:
                    k = min(range(len(t_mc)), key=lambda idx: abs(t_mc[idx] - t))
                    entry["stick_roll"] = round(float(mc["roll"][k]), 2)
                samples.append(entry)
            liftoff_section["roll_after_liftoff"] = samples
    report["liftoff"] = liftoff_section

    # -- §10 목표 대비 (notes.md의 transition_alt 파싱, 있으면) -------------------
    flight_dir = os.path.dirname(path)
    notes_path = os.path.join(flight_dir, "notes.md")
    target_alt = None
    if os.path.isfile(notes_path):
        with open(notes_path, "r", encoding="utf-8") as f:
            target_alt = parse_transition_alt(f.read())
    if target_alt is not None:
        report["target_transition_alt_m"] = target_alt
        if agl_section.get("agl_max_m") is not None:
            report["target_reached"] = agl_section["agl_max_m"] >= target_alt - 0.5

    # -- §11 안전망: 스크립트가 다루지 않은 토픽 -----------------------------------
    uncovered = sorted(name if mid == 0 else f"{name}(#{mid})"
                        for name, mid in present_topics if (name, mid) not in covered_topics)
    report["uncovered_topics"] = uncovered

    return report


def find_ulogs(flight_dir):
    return sorted(glob.glob(os.path.join(flight_dir, "*.ulg")))


def render_markdown(flight_dir, primary_report, other_logs):
    lines = [f"# 자동분석 — {flight_dir}", ""]
    lines.append(f"(analyze_flight.py 자동생성 — 해석 없이 사실만. notes.md에 결론 작성 시 이 파일 인용)")
    lines.append("")
    if other_logs:
        lines.append("## 이 폴더의 다른 ulog (참고, 미분석)")
        for name, dur in other_logs:
            lines.append(f"- `{name}`: {dur:.1f}s")
        lines.append("")
    if primary_report is None:
        lines.append("(주 로그 없음 — .ulg 파일을 찾지 못함)")
        return "\n".join(lines)

    r = primary_report
    lines.append(f"## 주 로그: `{os.path.basename(r['file'])}` ({r['duration_s']}s)")
    lines.append("")

    if r.get("motors"):
        lines.append("### 모터 배치 (CA_ROTOR 파라미터 기반)")
        for m in r["motors"]:
            lines.append(f"- motor{m['index']} = {m['label']} (PX={m['px']}, PY={m['py']}, KM={m['km']})")
        lines.append("")

    if r.get("params_snapshot"):
        lines.append("### 주요 파라미터")
        for k, v in r["params_snapshot"].items():
            lines.append(f"- `{k}` = {v}")
        lines.append("")

    a = r.get("altitude", {})
    if a:
        lines.append("### 고도")
        lines.append(f"- home_position AMSL={a['home_position_alt_amsl']}m, EKF ref_alt AMSL={a['ekf_ref_alt_amsl']}m")
        lines.append(f"- AGL 최대 {a['agl_max_m']}m (t={a['agl_max_at_s']}s), 최소 {a['agl_min_m']}m (t={a['agl_min_at_s']}s), 종료시 {a['agl_end_m']}m")
        if "target_transition_alt_m" in r:
            reached = "달성" if r.get("target_reached") else "미달"
            lines.append(f"- notes.md 목표 transition_alt={r['target_transition_alt_m']}m → **{reached}**")
        lines.append("")

    if r.get("mode_transitions"):
        lines.append("### 모드권한 타임라인")
        lines.append("| t(s) | nav_state | user_intention | failsafe | arming | 분류 |")
        lines.append("|---|---|---|---|---|---|")
        for m in r["mode_transitions"]:
            lines.append(f"| {m['t_s']} | {m['nav_state']} | {m['user_intention']} | {m['failsafe']} | {m['arming_state']} | {m['classification']} |")
        lines.append(f"\n> {r['mode_transitions_note']}")
        lines.append("")

    att = r.get("attitude", {})
    if att:
        lines.append("### 자세")
        lines.append(f"- roll 최대|값|={att.get('roll_deg_max_abs')}° (t={att.get('roll_deg_max_abs_at_s')}s)")
        lines.append(f"- pitch 최대|값|={att.get('pitch_deg_max_abs')}° (t={att.get('pitch_deg_max_abs_at_s')}s)")
        if "roll_setpoint_deg_max_abs" in att:
            lines.append(f"- roll 목표(setpoint) 최대|값|={att['roll_setpoint_deg_max_abs']}° (t={att['roll_setpoint_deg_max_abs_at_s']}s) — {att.get('roll_setpoint_note','')}")
        for axis in ["roll", "pitch", "yaw"]:
            key = f"{axis}_rate_onset_candidate_s"
            if key in att:
                lines.append(f"- {axis} rate 이상조짐 후보: t={att[key]}s ({att[f'{axis}_rate_onset_candidate_deg_s']}deg/s)")
        if "rate_onset_note" in att:
            lines.append(f"> {att['rate_onset_note']}")
        lines.append("")

    sat = r.get("allocator_saturation", {})
    if sat:
        lines.append("### 얼로케이터 포화 (축별)")
        for axis in ["roll(x)", "pitch(y)", "yaw(z)"]:
            if axis in sat:
                v = sat[axis]
                if v.get("first_saturation_at_s") is not None:
                    lines.append(f"- {axis}: t={v['first_saturation_at_s']}s부터 포화 시작 (unallocated={v['unallocated_torque_at_onset']})")
                else:
                    lines.append(f"- {axis}: {v.get('note')}")
        if "torque_setpoint_achieved_true_fraction" in sat:
            lines.append(f"- 통합 achieved 플래그 True 비율: {sat['torque_setpoint_achieved_true_fraction']}")
            lines.append(f"> {sat.get('achieved_flag_note','')}")
        lines.append("")

    if r.get("motors_command"):
        lines.append("### 모터별 커맨드 (0~1)")
        for k, v in r["motors_command"].items():
            lines.append(f"- {k}: mean={v['mean']} max={v['max']} min={v['min']}")
        lines.append("")

    bat = r.get("battery", {})
    if bat:
        lines.append("### 배터리")
        lines.append(f"- 휴지전압 {bat.get('voltage_rest_v')}V, 부하중 최저 {bat.get('voltage_min_v')}V, 전류 최대 {bat.get('current_max_a')}A")
        if "remaining_start" in bat:
            lines.append(f"- remaining: {bat['remaining_start']} → {bat['remaining_end']}")
        lines.append("")

    fd = r.get("failure_detector", {})
    if fd:
        lines.append("### 실패감지기")
        for k, v in fd.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lo = r.get("liftoff", {})
    if lo.get("liftoff_t_s") is not None:
        lines.append("### 이함(liftoff) 순간")
        lines.append(f"- ground_contact 1→0 시점: t={lo['liftoff_t_s']}s")
        if lo.get("roll_after_liftoff"):
            lines.append("| t(s) | roll(deg) | stick_roll |")
            lines.append("|---|---|---|")
            for s in lo["roll_after_liftoff"]:
                lines.append(f"| {s['t_s']} | {s['roll_deg']} | {s.get('stick_roll','-')} |")
        lines.append("")
    elif "liftoff_t_s" in lo:
        lines.append("### 이함(liftoff) 순간")
        lines.append("- ground_contact 1→0 전이 미검출 (짧은 로그이거나 실제 이함이 없었을 가능성)")
        lines.append("")

    if r.get("uncovered_topics"):
        lines.append("### 이 스크립트가 다루지 않은 토픽 (필요시 직접 확인)")
        lines.append(", ".join(f"`{t}`" for t in r["uncovered_topics"]))
        lines.append("")

    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("flight_dir", help="플라이트 폴더 경로 (logs/<날짜>_flightNN/)")
    p.add_argument("--json", action="store_true", help="analysis_auto.json 도 같이 생성")
    p.add_argument("--no-write", action="store_true", help="파일로 저장하지 않고 stdout만")
    p.add_argument("--out", default=None, help="markdown 출력 경로 (기본: <flight_dir>/analysis_auto.md)")
    args = p.parse_args(argv)

    flight_dir = args.flight_dir.rstrip("/")
    if not os.path.isdir(flight_dir):
        print(f"[analyze_flight] 폴더 없음: {flight_dir}", file=sys.stderr)
        return 1

    ulogs = find_ulogs(flight_dir)
    if not ulogs:
        print(f"[analyze_flight] .ulg 파일 없음: {flight_dir}", file=sys.stderr)
        return 1

    try:
        from pyulog import ULog  # noqa: F401  — 조기 실패로 안내 메시지 제공
    except ImportError:
        print("[analyze_flight] pyulog 미설치 — `pip install pyulog` 실행 후 재시도", file=sys.stderr)
        return 1

    durations = []
    for path in ulogs:
        u = ULog(path)
        durations.append((path, (u.last_timestamp - u.start_timestamp) / 1e6))

    primary_path, primary_dur = max(durations, key=lambda x: x[1])
    others = [(os.path.basename(p), d) for p, d in durations if p != primary_path]

    print(f"[analyze_flight] 주 로그: {primary_path} ({primary_dur:.1f}s), 그 외 {len(others)}개", file=sys.stderr)
    report = analyze_ulog(primary_path)

    md = render_markdown(flight_dir, report, others)
    print(md)

    if not args.no_write:
        out_md = args.out or os.path.join(flight_dir, "analysis_auto.md")
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[analyze_flight] 저장: {out_md}", file=sys.stderr)
        if args.json:
            out_json = os.path.splitext(out_md)[0] + ".json"
            full = {"primary": report, "other_logs": others}
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(full, f, ensure_ascii=False, indent=2, default=_to_native)
            print(f"[analyze_flight] 저장: {out_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
