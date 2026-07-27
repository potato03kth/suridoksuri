#!/usr/bin/env python3
"""SITL-7 VTOL 회귀 캠페인 — 로그 → 판정 지표.

입력:  logs/2026-07-27_sitl_vtol_campaign/<scenario_id>/ (node.log, *.ulg, meta.json)
출력:  같은 디렉터리에 metrics.json + verdict.md

    python3 tools/sitl/analyze_run.py A1
    python3 tools/sitl/analyze_run.py --dir /path/to/run

구현하는 지표는 docs/sitl_vtol_campaign.md 4장 전부:
  공통      — 상태 전이 타임라인 / 완주 여부 / vtol_state 시퀀스와 천이 소요시간
  부드러움  — setpoint 점프, 수직가속 피크, 역천이 감속률, TRANSITION_FW 헤딩오차
              시계열(오버슈트·단조수렴), 천이 중 고도손실, 순항 고도편차, FW cte,
              CLIMBING 오버슈트

원칙: **계산할 수 없는 지표는 거짓으로 채우지 않는다.** 값은 null 로 두고
`reason` 에 왜 못 구했는지 남긴다. verdict 는 그 항목을 PASS 로 세지 않는다(NULL).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = REPO_ROOT / "logs" / "2026-07-27_sitl_vtol_campaign"
PARAMS_YAML = REPO_ROOT / "fc_ros" / "fc_ros" / "params" / "fc_ros_params.yaml"

G = 9.80665

# 정상 상태 순서 (VTOL). 실제 시퀀스가 이것의 부분수열인지 본다.
CANONICAL_ORDER = ["ARM_TAKEOFF", "CLIMBING", "TRANSITION_FW", "STREAMING",
                   "ENTRY", "FOLLOWING", "TRANSITION_MC", "HOLD",
                   "LANDING", "DONE"]

VTOL_STATE_NAME = {0: "UNDEFINED", 1: "TRANS_TO_FW", 2: "TRANS_TO_MC",
                   3: "MC", 4: "FW"}

LOG_LINE_RE = re.compile(
    r"^\[(?P<proc>[^\]]+)\]\s+\[(?P<level>DEBUG|INFO|WARN|ERROR|FATAL)\]\s+"
    r"\[(?P<t>\d+\.\d+)\]\s+\[(?P<logger>[^\]]+)\]:\s*(?P<msg>.*)$")

STATE_ENTRY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ARM_TAKEOFF",    re.compile(r"^ARM 요청")),
    ("CLIMBING",       re.compile(r"CommandTOL 이륙 요청.*-> CLIMBING")),
    ("TRANSITION_FW",  re.compile(r"운용 고도 .* 도달 → transition_fw")),
    ("STREAMING",      re.compile(r"^(FW 전환 완료 -> STREAMING|운용 고도 .* 도달 → streaming)")),
    # ENTRY 는 F-14 수정(2026-07-27 R2) 전까지 로그에 나타나지 않았다 —
    # `entry_mode=mid_flight` 여도 노드가 "OFFBOARD 확인 → FOLLOWING" 만 찍어
    # R1_c10_entry 런의 ENTRY 구간이 통째로 FOLLOWING 창으로 잡혔다.
    ("ENTRY",          re.compile(r"^OFFBOARD 확인 → ENTRY")),
    ("FOLLOWING",      re.compile(r"^(OFFBOARD 확인 → FOLLOWING|ENTRY 완료 -> FOLLOWING)")),
    ("TRANSITION_MC",  re.compile(r"경로 추종 완료 -> transition_mc")),
    ("HOLD",           re.compile(r"MC 전환 완료 -> HOLD")),
    ("LANDING",        re.compile(r"(WP1 도달·안정 → LANDING|→ 강제 LANDING)")),
    ("OVERRIDE",       re.compile(r"긴급 수동 전환 실행 →")),
    ("PILOT_TAKEOVER", re.compile(r"조종사 인계 감지")),
    ("DONE",           re.compile(r"(착륙 완료 \(disarmed\) -> DONE|수동/안전 모드 진입 확인.*-> DONE)")),
]

RE_HEADING_DONE = re.compile(
    r"헤딩 정렬 완료 target=(-?[\d.]+)° current=(-?[\d.]+)° err=(-?[\d.]+)°")
RE_PATH_ORIGIN = re.compile(
    r"경로 원점 = 이륙지점 \[N=(-?[\d.]+), E=(-?[\d.]+), h_up=(-?[\d.]+)\].*"
    r"frame=(\w+).*순항고도 h_up=(-?[\d.]+)")
RE_TAKEOFF_REQ = re.compile(
    r"CommandTOL 이륙 요청 alt=(-?[\d.]+)m AMSL \(지면 (-?[\d.]+)\+(-?[\d.]+)\)")
RE_FOLLOW_TICK = re.compile(
    r"FOLLOWING tick=(\d+) mode=(\S*) cte=(-?[\d.]+)m "
    r"pos=\[(-?[\d.]+),(-?[\d.]+)\] tgt=\[(-?[\d.]+),(-?[\d.]+)\]")
# L1 세그먼트 인덱스 (FW 전용, 2026-07-27 R2 신설). 종전 런에는 없으므로
# **선택적**으로 뽑는다 — 없으면 None 이고 기존 런 재분석이 깨지지 않는다.
RE_FOLLOW_SEG = re.compile(r"\bseg=(\d+)/(\d+)")
RE_SEG_JUMP = re.compile(
    r"세그먼트 인덱스 급변 (\d+)→(\d+) \(Δ([+-]?\d+), 전체 (\d+)\)")
RE_FOLLOW_START = re.compile(
    r"FOLLOWING 시작 pos=\[(-?[\d.]+),(-?[\d.]+)\] tgt=\[(-?[\d.]+),(-?[\d.]+)\] "
    r"cte=(-?[\d.]+)m mode=(\S*)")
RE_HOLD_TICK = re.compile(r"WP1 홀드 dist=(-?[\d.]+)m speed=(-?[\d.]+)m/s")

# 숫자를 지워 같은 경고를 하나로 묶는다
RE_NUM = re.compile(r"-?\d+\.?\d*")

# ── ROS 로그 포맷이 아닌 줄 ────────────────────────────────────────────────
# `ros2 launch` 는 자식 프로세스의 stdout/stderr 를 `[proc] <원문>` 으로 **그대로**
# 중계한다. 이 줄들은 `[LEVEL] [t] [logger]:` 구조가 없어서 LOG_LINE_RE 가 통째로
# 버린다. 여기에 실제 경고가 들어 있었다 — 대표적으로 플래너 경고
#   [offboard_node-2] [Eta3ClothoidPlannerV3] WARNING: NR pos residual 9.451m is large. ...
# 가 B2/B3/B4/B5/A3 런에서 verdict 에 한 건도 실리지 않았다(S6 확인).
# 계획서 5장 "경고는 전부 보고" 를 만족시키려면 이 경로도 읽어야 한다.
# 무해성 판단은 하지 않는다 — 수집만 하고 출처(`raw_stdout`)·SIGINT 이후 여부만 표시한다.
RE_RAW_ERROR = re.compile(
    r"\bERROR\b|\bFATAL\b|\bCRITICAL\b|Traceback \(most recent call last\)"
    r"|KeyboardInterrupt|\bException\b")
RE_RAW_WARN = re.compile(r"\bWARNING\b|\bWARN\b")
# 하니스가 런 종료 시 launch 에 SIGINT 를 보낸다 — 이 줄 이후의 traceback 은
# 그 종료 절차에서 나온 것이라는 **사실**만 기록한다(무해 판단이 아니다).
RE_LAUNCH_SIGINT = re.compile(r"user interrupted with ctrl-c \(SIGINT\)")
RE_PLANNER_NR = re.compile(r"NR pos residual (-?[\d.]+)m is large")


# ── 유틸 ───────────────────────────────────────────────────────────────────

def _f(x):
    """JSON 직렬화 가능한 float (NaN/inf → None)."""
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, 4)


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def null(reason: str) -> dict:
    return {"value": None, "reason": reason}


def _seg_of(msg: str):
    """FOLLOWING 로그 줄에서 `seg=<i>/<n>` 의 i 를 뽑는다. 없으면 None.

    2026-07-27 R2 이전 런에는 이 꼬리표가 없다 — 그 런들을 다시 분석해도
    깨지지 않아야 하므로 선택적으로 취급한다.
    """
    m = RE_FOLLOW_SEG.search(msg)
    return int(m.group(1)) if m else None


# ── node.log 파싱 ──────────────────────────────────────────────────────────

def parse_node_log(path: Path) -> dict:
    out = {
        "present": path.exists(),
        "state_events": [],       # 첫 진입만
        "all_state_events": [],   # 재진입 포함
        "warnings": [],
        "errors": [],
        "heading_align": None,
        "path_origin": None,
        "takeoff_request": None,
        "cte_samples": [],        # {t, tick, cte, mode, pos, tgt}
        "hold_samples": [],
        "offboard_reacquire": [],
        "timeouts": [],
        "planner_nr_residual_m": [],
        "segment_jumps": [],      # 세그먼트 인덱스 급변 경고 (2026-07-27 R2)
        "line_count": 0,
        "t_first": None,
        "t_last": None,
    }
    if not path.exists():
        return out
    warn_groups: dict[str, dict] = {}
    raw_groups: dict[str, dict] = {}
    shutdown_seen = False
    seen_states: set[str] = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            out["line_count"] += 1
            m = LOG_LINE_RE.match(raw.strip())
            if not m:
                # ROS 로그 포맷이 아닌 줄 — 자식 프로세스 stdout/stderr 중계.
                s = raw.strip()
                if not s:
                    continue
                if RE_LAUNCH_SIGINT.search(s):
                    shutdown_seen = True
                mm = RE_PLANNER_NR.search(s)
                if mm:
                    out["planner_nr_residual_m"].append(float(mm.group(1)))
                lvl = ("ERROR" if RE_RAW_ERROR.search(s) else
                       "WARN" if RE_RAW_WARN.search(s) else None)
                if lvl:
                    key = RE_NUM.sub("#", s)[:160]
                    b = raw_groups.setdefault(key, {
                        "level": lvl, "count": 0,
                        # 이 줄엔 ROS 타임스탬프가 없다. 직전 ROS 로그 시각을
                        # 근사로 쓰고(없으면 None), 줄 번호를 같이 남긴다.
                        "first_t": out["t_last"], "last_t": out["t_last"],
                        "first_line": out["line_count"], "sample": s,
                        "raw_stdout": True, "after_sigint": shutdown_seen})
                    b["count"] += 1
                    b["last_t"] = out["t_last"]
                continue
            t = float(m.group("t"))
            msg, level = m.group("msg"), m.group("level")
            out["t_first"] = out["t_first"] if out["t_first"] is not None else t
            out["t_last"] = t

            if level == "WARN" or level in ("ERROR", "FATAL"):
                key = RE_NUM.sub("#", msg)[:160]
                bucket = warn_groups.setdefault(
                    key, {"level": level, "count": 0, "first_t": t,
                          "last_t": t, "sample": msg})
                bucket["count"] += 1
                bucket["last_t"] = t
                if "타임아웃" in msg:
                    out["timeouts"].append({"t": t, "msg": msg})
                if "재요청" in msg:
                    out["offboard_reacquire"].append({"t": t, "msg": msg})

            for state, pat in STATE_ENTRY_PATTERNS:
                if pat.search(msg):
                    out["all_state_events"].append(
                        {"state": state, "t": t, "msg": msg})
                    if state not in seen_states:
                        seen_states.add(state)
                        out["state_events"].append(
                            {"state": state, "t": t, "msg": msg})
                    break

            mm = RE_HEADING_DONE.search(msg)
            if mm and out["heading_align"] is None:
                out["heading_align"] = {
                    "t": t, "target_deg": float(mm.group(1)),
                    "current_deg": float(mm.group(2)),
                    "err_deg": float(mm.group(3))}
            mm = RE_PATH_ORIGIN.search(msg)
            if mm and out["path_origin"] is None:
                out["path_origin"] = {
                    "t": t, "n": float(mm.group(1)), "e": float(mm.group(2)),
                    "h_up": float(mm.group(3)), "frame": mm.group(4),
                    "cruise_alt": float(mm.group(5))}
            mm = RE_TAKEOFF_REQ.search(msg)
            if mm and out["takeoff_request"] is None:
                out["takeoff_request"] = {
                    "t": t, "target_amsl": float(mm.group(1)),
                    "ground_amsl": float(mm.group(2)),
                    "transition_alt": float(mm.group(3))}
            mm = RE_FOLLOW_TICK.search(msg)
            if mm:
                out["cte_samples"].append({
                    "t": t, "tick": int(mm.group(1)), "mode": mm.group(2),
                    "cte": float(mm.group(3)),
                    "pos": [float(mm.group(4)), float(mm.group(5))],
                    "tgt": [float(mm.group(6)), float(mm.group(7))],
                    "seg": _seg_of(msg)})
            mm = RE_FOLLOW_START.search(msg)
            if mm:
                out["cte_samples"].append({
                    "t": t, "tick": 0, "mode": mm.group(6),
                    "cte": float(mm.group(5)),
                    "pos": [float(mm.group(1)), float(mm.group(2))],
                    "tgt": [float(mm.group(3)), float(mm.group(4))],
                    "seg": _seg_of(msg)})
            mm = RE_SEG_JUMP.search(msg)
            if mm:
                out["segment_jumps"].append({
                    "t": t, "from": int(mm.group(1)), "to": int(mm.group(2)),
                    "delta": int(mm.group(3)), "n_seg": int(mm.group(4))})
            mm = RE_HOLD_TICK.search(msg)
            if mm:
                out["hold_samples"].append(
                    {"t": t, "dist": float(mm.group(1)),
                     "speed": float(mm.group(2))})

    for key, b in sorted(warn_groups.items(),
                         key=lambda kv: -kv[1]["count"]):
        rec = {"pattern": key, "level": b["level"], "count": b["count"],
               "first_t": b["first_t"], "last_t": b["last_t"],
               "sample": b["sample"]}
        (out["errors"] if b["level"] != "WARN" else out["warnings"]).append(rec)
    for key, b in sorted(raw_groups.items(),
                         key=lambda kv: -kv[1]["count"]):
        rec = {"pattern": key, "level": b["level"], "count": b["count"],
               "first_t": b["first_t"], "last_t": b["last_t"],
               "first_line": b["first_line"], "sample": b["sample"],
               "raw_stdout": True, "after_sigint": b["after_sigint"]}
        (out["errors"] if b["level"] != "WARN" else out["warnings"]).append(rec)
    return out


RE_MAVROS_LINE = re.compile(
    r"\[(?P<level>WARN|ERROR|FATAL)\]\s+\[(?P<t>\d+\.\d+)\][^:]*:\s*(?P<msg>.*)$")

# 계획서 1장이 "알려진 코스메틱"으로 명시한 유일한 경고. 그래도 숨기지 않고
# known_cosmetic 플래그만 달아 verdict 에 그대로 싣는다.
KNOWN_COSMETIC = [
    re.compile(r"PositionTargetGlobal failed because no origin"),
]


def parse_mavros_log(path: Path) -> dict:
    """MAVROS stdout 의 WARN/ERROR 를 묶어서 수집한다.

    node.log 만 보면 MAVROS 쪽 경고를 통째로 놓친다 — 계획서 5장이 "경고는 전부
    보고"를 요구하므로 여기도 읽는다."""
    out = {"present": path.exists(), "warnings": [], "line_count": 0}
    if not path.exists():
        return out
    groups: dict[str, dict] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            out["line_count"] += 1
            m = RE_MAVROS_LINE.search(raw.strip())
            if not m:
                continue
            msg, level, t = m.group("msg"), m.group("level"), float(m.group("t"))
            key = RE_NUM.sub("#", msg)[:160]
            b = groups.setdefault(key, {
                "level": level, "count": 0, "first_t": t, "last_t": t,
                "sample": msg,
                "known_cosmetic": any(p.search(msg) for p in KNOWN_COSMETIC)})
            b["count"] += 1
            b["last_t"] = t
    out["warnings"] = [dict(pattern=k, **v) for k, v in
                       sorted(groups.items(), key=lambda kv: -kv[1]["count"])]
    return out


# ── ulog 디코드 ────────────────────────────────────────────────────────────

def load_ulog(path: Path):
    from pyulog import ULog
    return ULog(str(path))


def topic(ulog, name: str, multi: int = 0):
    for d in ulog.data_list:
        if d.name == name and d.multi_id == multi:
            return d.data
    return None


def tsec(data) -> np.ndarray:
    return np.asarray(data["timestamp"], dtype=float) / 1e6


def quat_yaw(data) -> np.ndarray:
    q0 = np.asarray(data["q[0]"], dtype=float)
    q1 = np.asarray(data["q[1]"], dtype=float)
    q2 = np.asarray(data["q[2]"], dtype=float)
    q3 = np.asarray(data["q[3]"], dtype=float)
    return np.arctan2(2.0 * (q0 * q3 + q1 * q2),
                      1.0 - 2.0 * (q2 * q2 + q3 * q3))


# ── 시각 정렬 (node.log 벽시계 ↔ ulog boot 시각) ───────────────────────────

ANCHORS = [
    # (설명, ulog 조건 함수, node.log 메시지 정규식)
    ("COMPONENT_ARM_DISARM(400)",
     lambda cmd, p1: int(cmd) == 400 and p1 > 0.5,
     re.compile(r"^ARM 요청")),
    ("NAV_TAKEOFF(22)",
     lambda cmd, p1: int(cmd) == 22,
     re.compile(r"CommandTOL 이륙 요청")),
    ("DO_VTOL_TRANSITION→FW(3000,p1=4)",
     lambda cmd, p1: int(cmd) == 3000 and abs(p1 - 4.0) < 0.1,
     re.compile(r"MC→FW 천이 명령 요청")),
    ("DO_VTOL_TRANSITION→MC(3000,p1=3)",
     lambda cmd, p1: int(cmd) == 3000 and abs(p1 - 3.0) < 0.1,
     re.compile(r"FW->MC 역천이 명령 요청")),
]


def estimate_time_offset(ulog, nodelog: dict) -> dict:
    """node.log 벽시계 ↔ ulog boot 시각 변환을 **1차식**으로 추정한다.

        wall_t ≈ scale * ulog_t + offset      (ulog_t = (wall_t - offset)/scale)

    ⚠️ 상수 오프셋으로는 안 된다(실측): lockstep SITL 의 시뮬 클록은 벽시계와
    같은 속도로 흐르지 않는다. A1 실측에서 앵커 3개의 상수 오프셋이 65초 구간에
    5.40초나 벌어졌다(시뮬이 벽시계보다 약 8% 느림). 그대로 쓰면 상태창이
    수 초씩 밀리고 DONE 창은 끝<시작이 되어 창 기반 지표가 전부 오염된다.
    앵커가 1개뿐이면 scale=1 로 두고 그 사실을 reason 에 남긴다.
    """
    vc = topic(ulog, "vehicle_command")
    if vc is None:
        return {"offset_s": None, "scale": None,
                "reason": "ulog 에 vehicle_command 토픽 없음", "anchors": []}
    t = tsec(vc)
    cmd = np.asarray(vc["command"], dtype=float)
    p1 = np.asarray(vc.get("param1", np.zeros_like(cmd)), dtype=float)

    # 상태 이벤트가 아닌 명령 로그(천이 명령 등)도 앵커로 쓰므로 원문에서 찾는다
    anchors = []
    for label, cond, pat in ANCHORS:
        idx = [i for i in range(len(cmd)) if cond(cmd[i], p1[i])]
        if not idx:
            continue
        wall = [wt for wt, m in nodelog["_raw_msgs"] if pat.search(m)]
        if not wall:
            continue
        anchors.append({"anchor": label, "ulog_t": float(t[idx[0]]),
                        "wall_t": float(wall[0]),
                        "const_offset_s": float(wall[0] - t[idx[0]])})
    if not anchors:
        return {"offset_s": None, "scale": None,
                "reason": "node.log 와 ulog 에 공통 앵커(ARM/이륙/천이 명령)가 없음",
                "anchors": []}

    x = np.array([a["ulog_t"] for a in anchors])
    y = np.array([a["wall_t"] for a in anchors])
    consts = y - x
    if len(anchors) >= 2 and (x.max() - x.min()) > 5.0:
        scale, offset = np.polyfit(x, y, 1)
        resid = y - (scale * x + offset)
        note = None
    else:
        scale, offset = 1.0, float(np.median(consts))
        resid = y - (scale * x + offset)
        note = ("앵커가 1개뿐(또는 구간이 5초 미만)이라 scale=1 상수 오프셋으로 "
                "폴백 — 시뮬/벽시계 클록 드리프트가 보정되지 않았다")
    for a, r in zip(anchors, resid):
        a["fit_residual_s"] = float(r)
    return {
        "offset_s": float(offset),
        "scale": float(scale),
        "model": "wall_t = scale * ulog_t + offset_s",
        "n_anchors": len(anchors),
        "max_abs_residual_s": float(np.abs(resid).max()),
        "const_offset_spread_s": float(consts.max() - consts.min()),
        "anchors": anchors,
        "reason": note,
    }


def wall_to_ulog(wall_t: float, ta: dict) -> float:
    return (wall_t - ta["offset_s"]) / ta["scale"]


# ── 지표 계산 ──────────────────────────────────────────────────────────────

def state_windows(nodelog: dict, ta: dict,
                  t_end_ulog: float | None) -> dict:
    """상태별 [ulog 시각 구간]. 시각정렬 실패 시 빈 dict.

    마지막 상태(보통 DONE)의 끝은 로그 끝으로 잡되, 변환 결과가 로그 끝을
    넘거나 시작보다 앞서는 경우 클램프한다(상수 오프셋 시절 DONE 창이
    끝<시작으로 뒤집혔던 버그 방지)."""
    if ta.get("offset_s") is None or t_end_ulog is None:
        return {}
    evs = nodelog["state_events"]
    win = {}
    for i, ev in enumerate(evs):
        t0 = wall_to_ulog(ev["t"], ta)
        t1 = (wall_to_ulog(evs[i + 1]["t"], ta) if i + 1 < len(evs)
              else t_end_ulog)
        t0 = max(0.0, min(t0, t_end_ulog))
        t1 = max(t0, min(t1, t_end_ulog))
        win[ev["state"]] = (t0, t1)
    return win


def metric_completion(nodelog, ulog, meta) -> dict:
    states = [e["state"] for e in nodelog["state_events"]]
    # 순서 검사: 관측 시퀀스가 정상 순서의 부분수열인가
    it = iter(CANONICAL_ORDER)
    order_ok = all(any(s == c for c in it) for s in states
                   if s in CANONICAL_ORDER)
    res = {
        "states_seen": states,
        "order_is_subsequence_of_canonical": bool(order_ok),
        "reached_done": "DONE" in states,
        "pilot_takeover": "PILOT_TAKEOVER" in states,
        "override": "OVERRIDE" in states,
        "exit_reason": meta.get("exit_reason"),
        "exit_code": meta.get("exit_code"),
    }
    if nodelog["state_events"]:
        t_start = nodelog["state_events"][0]["t"]
        t_last = nodelog["state_events"][-1]["t"]
        res["mission_span_s"] = _f(t_last - t_start)
    else:
        res["mission_span_s"] = None

    durations = []
    evs = nodelog["state_events"]
    for i, ev in enumerate(evs):
        t1 = evs[i + 1]["t"] if i + 1 < len(evs) else nodelog["t_last"]
        durations.append({"state": ev["state"], "entry_wall_t": ev["t"],
                          "dwell_s": _f((t1 or ev["t"]) - ev["t"])})
    res["state_timeline"] = durations

    vs = topic(ulog, "vehicle_status") if ulog else None
    if vs is None:
        res["disarm"] = null("ulog vehicle_status 없음")
    else:
        t = tsec(vs)
        arm = np.asarray(vs["arming_state"], dtype=int)
        armed_idx = np.where(arm == 2)[0]
        if len(armed_idx) == 0:
            res["disarm"] = null("ulog 상 armed(=2) 구간이 없음 — ARM 실패")
        else:
            after = np.where((arm == 1) & (t > t[armed_idx[0]]))[0]
            res["disarm"] = {
                "armed_at_ulog_s": _f(t[armed_idx[0]]),
                "disarmed_at_ulog_s": _f(t[after[0]]) if len(after) else None,
                "armed_duration_s": (_f(t[after[0]] - t[armed_idx[0]])
                                     if len(after) else None),
                "value": bool(len(after) > 0),
                "reason": None if len(after) else "로그 끝까지 disarm 되지 않음",
            }
    return res


def metric_vtol_sequence(ulog) -> dict:
    d = topic(ulog, "vtol_vehicle_status") if ulog else None
    if d is None:
        return null("ulog 에 vtol_vehicle_status 토픽 없음 "
                    "(MC 기체이거나 optional topic 미기록)")
    t = tsec(d)
    v = np.asarray(d["vehicle_vtol_state"], dtype=int)
    segs = []
    start = 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[start]:
            segs.append({"vtol_state": int(v[start]),
                         "name": VTOL_STATE_NAME.get(int(v[start]), "?"),
                         "t_start_ulog_s": _f(t[start]),
                         "duration_s": _f((t[i - 1] if i == len(v) else t[i])
                                          - t[start])})
            start = i
    seq = [s["vtol_state"] for s in segs]
    fwd = [s for s in segs if s["vtol_state"] == 1]
    back = [s for s in segs if s["vtol_state"] == 2]
    return {
        "value": seq,
        "segments": segs,
        "forward_transition_s": _f(fwd[0]["duration_s"]) if fwd else None,
        "back_transition_s": _f(back[0]["duration_s"]) if back else None,
        "forward_ok_3_1_4": _has_sub(seq, [3, 1, 4]),
        "back_ok_4_2_3": _has_sub(seq, [4, 2, 3]),
        "reason": None,
    }


def _has_sub(seq, sub) -> bool:
    it = iter(seq)
    return all(any(x == s for x in it) for s in sub)


def metric_setpoint_jump(ulog, win: dict, params: dict) -> dict:
    d = topic(ulog, "trajectory_setpoint") if ulog else None
    src = "trajectory_setpoint"
    if d is None:
        d = topic(ulog, "vehicle_local_position_setpoint") if ulog else None
        src = "vehicle_local_position_setpoint"
    if d is None:
        return null("ulog 에 trajectory_setpoint / "
                    "vehicle_local_position_setpoint 둘 다 없음")
    t = tsec(d)
    if "position[0]" in d:
        p = np.column_stack([d["position[0]"], d["position[1]"],
                             d["position[2]"]]).astype(float)
    else:
        p = np.column_stack([d["x"], d["y"], d["z"]]).astype(float)
    good = np.isfinite(p).all(axis=1)
    t, p = t[good], p[good]
    if len(t) < 3:
        return null(f"{src} 에 유한한 위치 setpoint 샘플이 {len(t)}개뿐")

    dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
    dt = np.diff(t)
    tm = t[1:]

    # 위치 setpoint 가 끊겼다 재개된 구간(속도 setpoint 로 제어하던 동안 position
    # 이 NaN 이라 유한 샘플 사이가 벌어짐)은 "연속 두 틱의 점프"가 아니다.
    # 이것을 점프로 세면 16초 간격을 1틱 취급해 300m 점프가 잡힌다(실측) —
    # 별도 항목(resumption_gaps)으로 분리해 보고한다.
    MAX_GAP_S = 0.5
    cont = (dt > 1e-6) & (dt <= MAX_GAP_S)
    gaps = (dt > MAX_GAP_S)
    dpc, dtc, tmc = dp[cont], dt[cont], tm[cont]
    if len(dpc) == 0:
        return null(f"{src} 에 {MAX_GAP_S}s 이내 연속 샘플 쌍이 없음")
    rate = dpc / dtc

    dt_ctrl = 1.0 / float(params["control_hz"])
    thresh_m = 3.0 * float(params["v_approach"]) * dt_ctrl

    # 상태 전이 경계 ±1s 창
    bounds = [w[0] for w in win.values()]
    near = np.zeros(len(tmc), dtype=bool)
    for b in bounds:
        near |= np.abs(tmc - b) <= 1.0

    viol = dpc > thresh_m
    order = np.argsort(-dpc)[:10]

    def ev(i):
        return {"ulog_t_s": _f(tmc[i]), "jump_m": _f(dpc[i]),
                "rate_mps": _f(rate[i]), "dt_s": _f(dtc[i]),
                "near_state_boundary": bool(near[i]),
                "state": _state_at(win, float(tmc[i]))}

    bviol = np.where(viol & near)[0]
    bviol = bviol[np.argsort(-dpc[bviol])][:10]
    return {
        "source_topic": src,
        "max_gap_s": MAX_GAP_S,
        "threshold_m": _f(thresh_m),
        "threshold_note": (
            f"3 x v_approach({params['v_approach']}) x dt({dt_ctrl:.3f}s) — 계획서 4장. "
            f"참고: FW 순항 중 lookahead 목표점은 기체속도로 전진하므로 정상 상태에서도 "
            f"약 v_cruise({params['v_cruise']}) x dt = "
            f"{params['v_cruise']*dt_ctrl:.2f}m/틱 이 나온다"),
        "max_jump_m": _f(dpc.max()),
        "max_rate_mps": _f(rate.max()),
        "n_violations": int(np.count_nonzero(viol)),
        "n_violations_at_state_boundary": int(np.count_nonzero(viol & near)),
        "boundary_violations_top": [ev(i) for i in bviol],
        "top10": [ev(i) for i in order],
        "n_samples": int(len(tmc)),
        "resumption_gaps": [
            {"ulog_t_s": _f(tm[i]), "gap_s": _f(dt[i]),
             "distance_m": _f(dp[i]), "state": _state_at(win, float(tm[i]))}
            for i in np.where(gaps)[0]][:10],
        "n_resumption_gaps": int(np.count_nonzero(gaps)),
        "reason": None,
        "boundary_windows_available": bool(win),
    }


def metric_segment_index(nodelog) -> dict:
    """L1 세그먼트 인덱스 시계열 요약 (2026-07-27 SITL-7 R2).

    `_find_segment()` 가 잡은 "지금 경로의 어디를 타고 있는가"의 시계열이다.
    보고 싶은 것은 값 자체가 아니라 **되감김(backward)과 급변(jump)** 이다 —
    종전 전역 최근접 탐색은 폐회로·U턴에서 다른 레그를 잡을 수 있었고 그
    순간 인덱스가 불연속으로 튄다.

    ⚠️ 표본은 node.log 의 2초 간격 진단 로그다(10Hz 제어틱 전부가 아니다).
    따라서 `max_forward_step` 을 "한 틱 전진량"으로 읽으면 안 된다 —
    2초 동안의 전진량이다. 틱 단위 급변은 노드가 직접 찍는
    `segment_jumps`(임계 20점, 1s throttle)로 본다.
    """
    samples = [s for s in nodelog.get("cte_samples", [])
               if s.get("seg") is not None]
    jumps = nodelog.get("segment_jumps", [])
    if not samples:
        return null("node.log 에 seg= 꼬리표가 없음 "
                    "(2026-07-27 R2 이전 런이거나 MC 런)")
    seq = [int(s["seg"]) for s in samples]
    d = [b - a for a, b in zip(seq[:-1], seq[1:])]
    back = [{"t": samples[i + 1]["t"], "from": seq[i], "to": seq[i + 1],
             "delta": d[i]} for i in range(len(d)) if d[i] < 0]
    return {
        "n_samples": len(seq),
        "series": seq,
        "t_series": [_f(s["t"]) for s in samples],
        "first": seq[0], "last": seq[-1],
        "monotonic_nondecreasing": all(x >= 0 for x in d),
        "n_backward": len(back),
        "backward_events": back[:20],
        "max_backward_step": (min(d) if d else 0),
        "max_forward_step": (max(d) if d else 0),
        "n_tick_jump_warnings": len(jumps),
        "tick_jump_warnings": jumps[:20],
        "reason": None,
    }


def _state_at(win: dict, t: float):
    for s, (a, b) in win.items():
        if a <= t <= b:
            return s
    return None


def _mask_between(t, a, b):
    return (t >= a) & (t <= b)


def metric_vertical_accel(ulog, vtol_seq, win, disarm=None) -> dict:
    lp = topic(ulog, "vehicle_local_position") if ulog else None
    if lp is None:
        return null("ulog 에 vehicle_local_position 없음")
    t = tsec(lp)
    az = np.asarray(lp["az"], dtype=float)
    good = np.isfinite(az)
    t, az = t[good], az[good]
    if len(t) == 0:
        return null("az 샘플이 전부 비유한")

    excl = np.zeros(len(t), dtype=bool)
    excl_desc = []
    if isinstance(vtol_seq, dict) and vtol_seq.get("segments"):
        for seg in vtol_seq["segments"]:
            if seg["vtol_state"] in (1, 2) and seg["duration_s"]:
                a = seg["t_start_ulog_s"]
                excl |= _mask_between(t, a, a + seg["duration_s"])
                excl_desc.append(f"vtol_state={seg['vtol_state']} "
                                 f"{a:.1f}~{a+seg['duration_s']:.1f}s")
    elif win:
        for s in ("TRANSITION_FW", "TRANSITION_MC"):
            if s in win:
                excl |= _mask_between(t, *win[s])
                excl_desc.append(f"{s} 상태창")

    keep = ~excl
    if not keep.any():
        return null("천이 구간 제외 후 남는 샘플이 없음")
    i = int(np.argmax(np.abs(az[keep])))
    peak = float(np.abs(az[keep])[i])

    # 접지 충격 제외 보조값 — disarm 직전 TOUCHDOWN_S 초를 더 뺀 피크.
    # 계획서 4장 기준은 "천이 구간 제외"뿐이라 이 값은 판정에 쓰지 않는다.
    # 다만 착륙 접지 스파이크(A1 실측 40.3 m/s² = 4.1g @ disarm −3.1s)가
    # 판정을 지배하므로, 비행 중 부드러움을 보려면 이 보조값을 봐야 한다.
    TOUCHDOWN_S = 5.0
    td = None
    t_dis = (disarm or {}).get("disarmed_at_ulog_s")
    if t_dis is not None:
        keep2 = keep & (t < (t_dis - TOUCHDOWN_S))
        if keep2.any():
            j = int(np.argmax(np.abs(az[keep2])))
            td = {"value": _f(float(np.abs(az[keep2])[j])),
                  "peak_g": _f(float(np.abs(az[keep2])[j]) / G),
                  "peak_at_ulog_s": _f(t[keep2][j]),
                  "peak_state": _state_at(win, float(t[keep2][j])),
                  "window_note": f"disarm({t_dis}s) − {TOUCHDOWN_S}s 이후 제외"}
    return {
        "value": _f(peak),
        "peak_g": _f(peak / G),
        "peak_at_ulog_s": _f(t[keep][i]),
        "peak_state": _state_at(win, float(t[keep][i])),
        "excluded_windows": excl_desc or ["(천이 구간 정보 없음 — 제외 없이 계산)"],
        "max_including_transition_mps2": _f(np.abs(az).max()),
        "excl_touchdown": td or {
            "value": None,
            "reason": "disarm 시각을 몰라 접지 구간을 제외할 수 없음"},
        "reason": None,
    }


def metric_decel(ulog, vtol_seq, win) -> dict:
    """역천이 구간 수평 감속률 d(|v_xy|)/dt (m/s²)."""
    lp = topic(ulog, "vehicle_local_position") if ulog else None
    if lp is None:
        return null("ulog 에 vehicle_local_position 없음")
    t = tsec(lp)
    v = np.hypot(np.asarray(lp["vx"], float), np.asarray(lp["vy"], float))
    good = np.isfinite(v)
    t, v = t[good], v[good]

    seg = None
    if isinstance(vtol_seq, dict) and vtol_seq.get("segments"):
        back = [s for s in vtol_seq["segments"] if s["vtol_state"] == 2]
        if back:
            seg = (back[0]["t_start_ulog_s"],
                   back[0]["t_start_ulog_s"] + (back[0]["duration_s"] or 0))
    if seg is None and "TRANSITION_MC" in win:
        seg = win["TRANSITION_MC"]
    if seg is None:
        return null("역천이 구간(vtol_state==2 또는 TRANSITION_MC 상태창)을 "
                    "특정할 수 없음 — 역천이가 일어나지 않았을 수 있다")

    m = _mask_between(t, seg[0], seg[1])
    if np.count_nonzero(m) < 3:
        return null(f"역천이 구간 샘플 부족 ({int(np.count_nonzero(m))}개)")
    ts, vs = t[m], v[m]
    # 0.5초 창 유한차분 (센서 노이즈 완화)
    acc = []
    for i in range(len(ts)):
        j = np.searchsorted(ts, ts[i] + 0.5)
        if j >= len(ts):
            break
        dtv = ts[j] - ts[i]
        if dtv > 1e-3:
            acc.append(((vs[i] - vs[j]) / dtv, ts[i]))
    if not acc:
        return null("역천이 구간이 0.5s보다 짧아 감속률을 낼 수 없음")
    a = np.array([x[0] for x in acc])
    tt = np.array([x[1] for x in acc])
    i = int(np.argmax(a))
    return {
        "value": _f(a[i]),
        "peak_decel_g": _f(a[i] / G),
        "peak_at_ulog_s": _f(tt[i]),
        "window_ulog_s": [_f(seg[0]), _f(seg[1])],
        "v_at_window_start_mps": _f(vs[0]),
        "v_at_window_end_mps": _f(vs[-1]),
        "reason": None,
    }


def metric_heading(ulog, nodelog, win, params, waypoints) -> dict:
    att = topic(ulog, "vehicle_attitude") if ulog else None
    if att is None:
        return null("ulog 에 vehicle_attitude 없음")
    if "TRANSITION_FW" not in win:
        return null("TRANSITION_FW 상태창이 없음 (상태 미도달이거나 시각정렬 실패)")
    t = tsec(att)
    yaw = quat_yaw(att)
    a, b = win["TRANSITION_FW"]
    m = _mask_between(t, a, b)
    if np.count_nonzero(m) < 5:
        return null(f"TRANSITION_FW 구간 attitude 샘플 부족 "
                    f"({int(np.count_nonzero(m))}개)")
    ts, ys = t[m], yaw[m]

    if nodelog.get("heading_align"):
        chi = math.radians(nodelog["heading_align"]["target_deg"])
        chi_src = "node.log '헤딩 정렬 완료 target=' 값"
    elif waypoints is not None and len(waypoints) >= 2:
        d = waypoints[1][:2] - waypoints[0][:2]
        chi = float(np.arctan2(d[1], d[0]))
        chi_src = "waypoints[0]→waypoints[1] 방향(플래너 보간 전 근사)"
    else:
        return null("목표 헤딩(chi_wp)을 알 수 없음")

    err = _wrap(chi - ys)
    ae = np.abs(err)
    tol = float(params["wp0_heading_tol"])

    # 정렬 완료 시각: node.log 값 우선
    t_align = None
    if (nodelog.get("heading_align") and nodelog["heading_align"].get("t")
            and params.get("_offset") is not None):
        t_align = ((nodelog["heading_align"]["t"] - params["_offset"])
                   / params["_scale"])

    # 오버슈트: 최초 tol 진입 이후 다시 tol 밖으로 나가는가 / |err| 재증가량
    first_in = np.where(ae < tol)[0]
    overshoot_rad = None
    monotone = None
    if len(first_in):
        k = int(first_in[0])
        after = ae[k:]
        overshoot_rad = float(after.max() - ae[k]) if len(after) else 0.0
        pre = ae[:k + 1]
        # 단조수렴: 재증가 총량이 0.05 rad 미만이면 단조로 본다
        rises = np.diff(pre)
        monotone = bool(rises[rises > 0].sum() < 0.05) if len(pre) > 1 else True
    return {
        "target_chi_rad": _f(chi),
        "target_chi_deg": _f(math.degrees(chi)),
        "target_source": chi_src,
        "tol_rad": _f(tol),
        "window_ulog_s": [_f(a), _f(b)],
        "n_samples": int(len(ts)),
        "err_at_window_start_deg": _f(math.degrees(err[0])),
        "max_abs_err_deg": _f(math.degrees(ae.max())),
        "final_abs_err_deg": _f(math.degrees(ae[-1])),
        "first_within_tol_ulog_s": (_f(ts[int(first_in[0])])
                                    if len(first_in) else None),
        "align_time_s": (_f(ts[int(first_in[0])] - ts[0])
                         if len(first_in) else None),
        "overshoot_rad_after_first_in_tol": _f(overshoot_rad),
        "monotone_convergence": monotone,
        "node_log_align": nodelog.get("heading_align"),
        "t_align_ulog_s": _f(t_align),
        "reason": None,
    }


def _agl_series(lp):
    t = tsec(lp)
    z = np.asarray(lp["z"], dtype=float)
    good = np.isfinite(z)
    t, z = t[good], z[good]
    if len(t) == 0:
        return None, None, None
    # 지면 기준: 로그 첫 5초(이륙 전) z 중앙값
    pre = z[t <= t[0] + 5.0]
    z_ground = float(np.median(pre)) if len(pre) else float(z[0])
    return t, -(z - z_ground), z_ground


def metric_altitude(ulog, nodelog, win, vtol_seq, params) -> dict:
    lp = topic(ulog, "vehicle_local_position") if ulog else None
    if lp is None:
        return null("ulog 에 vehicle_local_position 없음")
    t, agl, z_ground = _agl_series(lp)
    if t is None:
        return null("z 샘플이 전부 비유한")
    trans_alt = float(params["transition_alt"])
    res = {"z_ground_local_m": _f(z_ground), "transition_alt_m": _f(trans_alt)}

    # CLIMBING 오버슈트 (C9 핵심)
    if "CLIMBING" in win:
        a, b = win["CLIMBING"]
        m = _mask_between(t, a, b)
        res["climb"] = ({"max_agl_m": _f(agl[m].max()),
                         "overshoot_pct": _f(100.0 * (agl[m].max() - trans_alt)
                                             / trans_alt),
                         "window_ulog_s": [_f(a), _f(b)],
                         "reason": None}
                        if m.any() else null("CLIMBING 창 샘플 없음"))
        # CLIMBING~STREAMING 전체 (STREAMING 오버슈트 이슈 대상 구간)
        b2 = win.get("FOLLOWING", (None, None))[0] or b
        m2 = _mask_between(t, a, b2)
        res["climb_to_following"] = (
            {"max_agl_m": _f(agl[m2].max()),
             "overshoot_pct": _f(100.0 * (agl[m2].max() - trans_alt) / trans_alt),
             "window_ulog_s": [_f(a), _f(b2)], "reason": None}
            if m2.any() else null("CLIMBING~FOLLOWING 창 샘플 없음"))
    else:
        res["climb"] = null("CLIMBING 상태창 없음 (시각정렬 실패 또는 상태 미도달)")
        res["climb_to_following"] = null("동상")

    # 정천이 중 고도손실
    seg = None
    if isinstance(vtol_seq, dict) and vtol_seq.get("segments"):
        fwd = [s for s in vtol_seq["segments"] if s["vtol_state"] == 1]
        if fwd:
            seg = (fwd[0]["t_start_ulog_s"],
                   fwd[0]["t_start_ulog_s"] + (fwd[0]["duration_s"] or 0))
    if seg is None and "TRANSITION_FW" in win:
        seg = win["TRANSITION_FW"]
    if seg is None:
        res["transition_alt_loss"] = null(
            "정천이 구간(vtol_state==1)을 특정할 수 없음 — 천이 미발생 가능")
    else:
        m = _mask_between(t, seg[0], seg[1])
        if not m.any():
            res["transition_alt_loss"] = null("정천이 창 샘플 없음")
        else:
            res["transition_alt_loss"] = {
                "value": _f(agl[m][0] - agl[m].min()),
                "agl_at_start_m": _f(agl[m][0]),
                "agl_min_m": _f(agl[m].min()),
                "window_ulog_s": [_f(seg[0]), _f(seg[1])],
                "reason": None}

    # 순항 고도편차 (FOLLOWING)
    cruise_ref = None
    cruise_src = None
    if nodelog.get("path_origin"):
        # path_origin.cruise_alt 은 EKF 로컬 h_up. AGL 로 바꾸려면 지면 h_up 을 뺀다.
        cruise_ref = nodelog["path_origin"]["cruise_alt"] - (-z_ground)
        cruise_src = "node.log '순항고도 h_up=' (EKF 로컬) − 지면 h_up"
    elif params.get("_cruise_alt_wp") is not None:
        cruise_ref = float(params["_cruise_alt_wp"]) - (-z_ground)
        cruise_src = ("waypoints[-1].z (EKF 로컬 절대, waypoint_frame=local) "
                      "− 지면 h_up")
    if "FOLLOWING" in win and cruise_ref is not None:
        a, b = win["FOLLOWING"]
        m = _mask_between(t, a, b)
        if m.any():
            dev = agl[m] - cruise_ref
            res["cruise_alt_dev"] = {
                "ref_agl_m": _f(cruise_ref), "ref_source": cruise_src,
                "mean_dev_m": _f(dev.mean()),
                "max_abs_dev_m": _f(np.abs(dev).max()),
                "min_agl_m": _f(agl[m].min()), "max_agl_m": _f(agl[m].max()),
                "window_ulog_s": [_f(a), _f(b)], "reason": None}
        else:
            res["cruise_alt_dev"] = null("FOLLOWING 창 샘플 없음")
    else:
        res["cruise_alt_dev"] = null(
            "FOLLOWING 상태창 또는 순항고도 기준을 알 수 없음")
    return res


def metric_cte(nodelog, ulog, win, waypoints, path_origin) -> dict:
    """FW 경로추종 cross-track error.

    1차 근거는 node.log 의 cte (L1Guidance 가 실제 계획경로에 대해 계산한 값).
    2차(참고)는 ulog 위치를 원본 waypoints 폴리라인에 투영한 기하 편차 —
    VTOL 기본 플래너(eta3)는 곡선이라 코너에서 둘이 다를 수 있다.
    """
    res = {}
    samples = nodelog.get("cte_samples") or []
    if samples:
        # ⚠️ L1Guidance 의 cte 는 **부호 있는** 값이다(좌/우). 그대로 max() 를
        #    쓰면 전부 음수인 런에서 "최대 −0.6m" 같은 무의미한 값이 나온다(실측).
        #    판정은 |cte| 로 한다.
        c = np.array([s["cte"] for s in samples])
        ac = np.abs(c)
        res["node_log_cte"] = {
            "n_samples": int(len(c)), "max_m": _f(ac.max()),
            "mean_m": _f(ac.mean()), "p95_m": _f(np.percentile(ac, 95)),
            "signed_min_m": _f(c.min()), "signed_max_m": _f(c.max()),
            "first_m": _f(c[0]), "last_m": _f(c[-1]),
            "note": ("offboard_node 가 20틱마다 찍는 L1Guidance cte. "
                     "max/mean/p95 는 절댓값, signed_* 는 부호 유지"),
            "reason": None}
    else:
        res["node_log_cte"] = null(
            "node.log 에 'FOLLOWING tick= ... cte=' 샘플이 없음 "
            "(FOLLOWING 미진입이거나 20틱 미만 체류)")

    lp = topic(ulog, "vehicle_local_position") if ulog else None
    if lp is None or waypoints is None or len(waypoints) < 2 or "FOLLOWING" not in win:
        res["geometric_cte"] = null(
            "ulog 위치 / waypoints / FOLLOWING 상태창 중 하나가 없음")
        return res
    t = tsec(lp)
    xy = np.column_stack([lp["x"], lp["y"]]).astype(float)
    a, b = win["FOLLOWING"]
    m = _mask_between(t, a, b) & np.isfinite(xy).all(axis=1)
    if np.count_nonzero(m) < 3:
        res["geometric_cte"] = null("FOLLOWING 창 위치 샘플 부족")
        return res
    wp = np.asarray(waypoints, dtype=float)[:, :2].copy()
    if path_origin is not None:
        wp += np.array([path_origin[0], path_origin[1]])
    P = xy[m]
    d = np.full(len(P), np.inf)
    for i in range(len(wp) - 1):
        A, B = wp[i], wp[i + 1]
        AB = B - A
        L2 = float(AB @ AB)
        if L2 < 1e-9:
            continue
        s = np.clip(((P - A) @ AB) / L2, 0.0, 1.0)
        proj = A + s[:, None] * AB
        d = np.minimum(d, np.linalg.norm(P - proj, axis=1))
    res["geometric_cte"] = {
        "n_samples": int(len(P)), "max_m": _f(d.max()),
        "mean_m": _f(d.mean()), "p95_m": _f(np.percentile(d, 95)),
        "note": ("원본 waypoints 직선 폴리라인 기준. VTOL 기본 플래너(eta3)는 "
                 "코너를 곡선으로 잇기 때문에 이 값은 '계획경로 이탈'이 아니라 "
                 "'직선 대비 편차'다 — 코너 구간에서 크게 나오는 것이 정상."),
        "reason": None}
    return res


# ── 판정 ───────────────────────────────────────────────────────────────────

def build_verdict(m: dict, params: dict) -> list[dict]:
    v: list[dict] = []

    def add(name, status, detail, criterion):
        v.append({"item": name, "status": status,
                  "criterion": criterion, "detail": detail})

    def le(val, limit, fail="FAIL"):
        """val ≤ limit 판정. val 이 None(비유한/미산출)이면 NULL."""
        if val is None:
            return "NULL"
        return "PASS" if float(val) <= limit else fail

    comp = m["completion"]
    if comp["reached_done"]:
        add("완주 (DONE 도달)", "PASS",
            f"상태 {' → '.join(comp['states_seen'])}, "
            f"소요 {comp['mission_span_s']}s", "DONE 상태 도달")
    else:
        add("완주 (DONE 도달)", "FAIL",
            f"관측 상태: {' → '.join(comp['states_seen']) or '(없음)'}; "
            f"종료사유={comp['exit_reason']}", "DONE 상태 도달")

    dis = comp.get("disarm") or {}
    if dis.get("value") is True:
        add("disarm 확인", "PASS",
            f"armed {dis['armed_at_ulog_s']}s → disarmed "
            f"{dis['disarmed_at_ulog_s']}s (비행 {dis['armed_duration_s']}s)",
            "ulog arming_state 2→1")
    elif dis.get("value") is False:
        add("disarm 확인", "FAIL", dis.get("reason"), "ulog arming_state 2→1")
    else:
        add("disarm 확인", "NULL", dis.get("reason"), "ulog arming_state 2→1")

    add("상태 순서", "PASS" if comp["order_is_subsequence_of_canonical"]
        else "FAIL",
        f"{' → '.join(comp['states_seen'])}",
        "정상 순서의 부분수열")

    seq = m["vtol_state_sequence"]
    if seq.get("value") is None:
        add("vtol_state 시퀀스", "NULL", seq.get("reason"), "3→1→4, 4→2→3")
    else:
        ok = seq["forward_ok_3_1_4"] and seq["back_ok_4_2_3"]
        add("vtol_state 시퀀스", "PASS" if ok else "FAIL",
            f"seq={seq['value']}, 정천이 {seq['forward_transition_s']}s / "
            f"역천이 {seq['back_transition_s']}s", "3→1→4, 4→2→3")

    sj = m["setpoint_jump"]
    if sj.get("reason") and sj.get("max_jump_m") is None:
        add("setpoint 점프", "NULL", sj["reason"], "상태 경계에서 급점프 없음")
    elif not sj.get("boundary_windows_available"):
        add("setpoint 점프", "NULL",
            f"최대 {sj['max_jump_m']}m — 그러나 상태 경계 시각을 알 수 없어 "
            f"'경계에서' 판정 불가", "상태 경계에서 급점프 없음")
    else:
        n = sj["n_violations_at_state_boundary"]
        worst = ", ".join(
            f"{e['jump_m']}m@{e['ulog_t_s']}s({e['state']})"
            for e in sj["boundary_violations_top"][:3]) or "-"
        add("setpoint 점프", "PASS" if n == 0 else "FAIL",
            f"임계 {sj['threshold_m']}m, 경계±1s 위반 {n}건 / 전체 위반 "
            f"{sj['n_violations']}건 / 샘플 {sj['n_samples']}개, 최대 "
            f"{sj['max_jump_m']}m ({sj['max_rate_mps']} m/s). "
            f"경계 최대: {worst}. 스트림 재개 갭 {sj['n_resumption_gaps']}건은 별도 집계",
            f"상태 경계 ±1s 에서 {sj['threshold_m']}m 초과 점프 없음")

    va = m["vertical_accel"]
    if va.get("value") is None:
        add("수직 가속", "NULL", va.get("reason"), "천이 제외 |az| ≤ 0.5g")
    else:
        td = va.get("excl_touchdown") or {}
        td_txt = (f"; 접지(disarm−5s) 제외 시 {td['value']} m/s² "
                  f"({td['peak_g']}g) @{td['peak_at_ulog_s']}s "
                  f"state={td['peak_state']}"
                  if td.get("value") is not None
                  else f"; 접지 제외값 없음({td.get('reason')})")
        add("수직 가속", le(va["peak_g"], 0.5),
            f"피크 |az|={va['value']} m/s² ({va['peak_g']}g) "
            f"@{va['peak_at_ulog_s']}s state={va['peak_state']}" + td_txt,
            "천이 제외 |az| ≤ 0.5g (4.90 m/s²)")

    dc = m["decel"]
    if dc.get("value") is None:
        add("역천이 감속률", "NULL", dc.get("reason"), "≤ 0.3g (2.94 m/s²)")
    else:
        add("역천이 감속률", le(dc["value"], 2.94),
            f"최대 {dc['value']} m/s² ({dc['peak_decel_g']}g), "
            f"{dc['v_at_window_start_mps']}→{dc['v_at_window_end_mps']} m/s",
            "≤ 0.3g (2.94 m/s²)")

    hd = m["heading"]
    if hd.get("reason"):
        add("TRANSITION_FW 헤딩", "NULL", hd["reason"],
            "오버슈트 없이 단조수렴, 정렬완료 err ≤ wp0_heading_tol")
    else:
        tol_deg = math.degrees(params["wp0_heading_tol"])
        ok = (hd["monotone_convergence"] is not False
              and hd["first_within_tol_ulog_s"] is not None)
        add("TRANSITION_FW 헤딩", "PASS" if ok else "FAIL",
            f"시작오차 {hd['err_at_window_start_deg']}° → 정렬 "
            f"{hd['align_time_s']}s 소요, 최대 {hd['max_abs_err_deg']}°, "
            f"tol 진입 후 재증가 {hd['overshoot_rad_after_first_in_tol']} rad, "
            f"단조수렴={hd['monotone_convergence']}",
            f"단조수렴 + err ≤ {tol_deg:.1f}°")

    alt = m["altitude"]
    alt_null = alt.get("reason") if alt.get("value", "x") is None else None
    cl = alt.get("climb") or ({"reason": alt_null} if alt_null else {})
    if cl.get("reason") or cl.get("max_agl_m") is None:
        add("CLIMBING 오버슈트", "NULL", cl.get("reason"),
            "transition_alt 대비 최대 AGL ≤ +10%")
    else:
        add("CLIMBING 오버슈트", le(cl["overshoot_pct"], 10.0),
            f"최대 AGL {cl['max_agl_m']}m vs transition_alt "
            f"{alt['transition_alt_m']}m → {cl['overshoot_pct']}%",
            "≤ +10%")
    tl = alt.get("transition_alt_loss") or ({"reason": alt_null} if alt_null else {})
    if tl.get("value") is None:
        add("정천이 고도손실", "NULL", tl.get("reason"), "≤ 5m")
    else:
        add("정천이 고도손실", le(tl["value"], 5.0),
            f"{tl['agl_at_start_m']}m → 최저 {tl['agl_min_m']}m "
            f"(손실 {tl['value']}m)", "≤ 5m")
    cd = alt.get("cruise_alt_dev") or ({"reason": alt_null} if alt_null else {})
    if cd.get("reason") or cd.get("max_abs_dev_m") is None:
        add("순항 고도편차", "NULL", cd.get("reason"), "±3m")
    else:
        add("순항 고도편차", le(cd["max_abs_dev_m"], 3.0),
            f"기준 AGL {cd['ref_agl_m']}m, 평균편차 {cd['mean_dev_m']}m, "
            f"최대 |편차| {cd['max_abs_dev_m']}m", "±3m")

    ct = m["cte"]["node_log_cte"]
    if ct.get("reason"):
        add("FW cte", "NULL", ct["reason"], "직선 ≤ 2m")
    else:
        add("FW cte", le(ct["max_m"], 2.0, fail="WARN"),
            f"최대 |cte| {ct['max_m']}m 평균 {ct['mean_m']}m "
            f"(부호 {ct['signed_min_m']}~{ct['signed_max_m']}m, "
            f"n={ct['n_samples']}) — 코너 오버슈트 포함값",
            "직선 ≤ 2m (코너는 별도 기록)")

    w = m["node_log"]["warnings"] + m["node_log"]["errors"]
    mw = (m.get("mavros_log") or {}).get("warnings") or []
    if w or mw:
        add("경고/타임아웃", "WARN",
            f"node.log {sum(x['count'] for x in w)}건/{len(w)}종, "
            f"mavros.log {sum(x['count'] for x in mw)}건/{len(mw)}종 "
            f"— verdict 하단 목록 참조",
            "무해성 판단은 사람이 한다(계획서 5장)")
    else:
        add("경고/타임아웃", "PASS", "node.log·mavros.log 모두 WARN/ERROR 없음", "-")
    return v


def render_verdict(meta, m, verdict, params) -> str:
    counts = {}
    for v in verdict:
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    lines = []
    a = lines.append
    a(f"# {meta.get('scenario_id','?')} — 판정")
    a("")
    a(f"- 목적: {meta.get('desc','')}")
    a(f"- 실행: {meta.get('started_utc','?')} ~ {meta.get('ended_utc','?')} "
      f"(경과 {meta.get('elapsed_s','?')}s)")
    a(f"- 종료: `{meta.get('exit_reason','?')}` (exit={meta.get('exit_code','?')})")
    a(f"- launch: `{' '.join(meta.get('launch_argv', []))}`")
    a(f"- 저장소 HEAD: `{meta.get('repo_head','?')}`")
    # PX4 빌드 커밋 — 어느 PX4 에서 돈 런인지 없으면 결과를 해석할 수 없다(S4).
    if meta.get("px4_head"):
        a(f"- PX4 빌드: `{meta['px4_head'][:10]}` "
          f"(`{meta.get('px4_dir','?')}`)")
    a(f"- ulog: {m.get('ulog_file') or '(없음)'} "
      f"(meta.json 기록: {', '.join(meta.get('ulogs') or []) or '없음'})")
    if m.get("ulog_selection_note"):
        a(f"  - ⚠️ {m['ulog_selection_note']}")
    a(f"- 요약: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    a("")
    ta = m.get("time_alignment", {})
    if ta.get("offset_s") is None:
        a(f"> ⚠️ node.log ↔ ulog 시각 정렬 실패 ({ta.get('reason')}) — "
          f"상태창 기반 지표는 전부 NULL 이다.")
    else:
        a(f"- 시각 정렬: `wall = {ta['scale']:.5f} x ulog + {ta['offset_s']:.3f}` "
          f"(앵커 {ta.get('n_anchors')}개, 최대 잔차 "
          f"{ta.get('max_abs_residual_s'):.3f}s). 시뮬 클록이 벽시계보다 "
          f"{(ta['scale']-1)*100:+.1f}% 빠름/느림 — 상수 오프셋만 쓰면 "
          f"{ta.get('const_offset_spread_s'):.2f}s 벌어진다")
        if ta.get("reason"):
            a(f"  - ⚠️ {ta['reason']}")
    a("")
    a("## 판정표")
    a("")
    a("| 항목 | 판정 | 근거 수치 | 기준 |")
    a("|---|---|---|---|")
    for v in verdict:
        det = str(v["detail"] or "").replace("|", "\\|").replace("\n", " ")
        a(f"| {v['item']} | **{v['status']}** | {det} | {v['criterion']} |")
    a("")

    a("## 상태 전이 타임라인")
    a("")
    a("| 상태 | 진입(벽시계) | 체류(s) |")
    a("|---|---|---|")
    t0 = None
    for e in m["completion"]["state_timeline"]:
        t0 = t0 if t0 is not None else e["entry_wall_t"]
        a(f"| {e['state']} | +{e['entry_wall_t']-t0:.1f}s | {e['dwell_s']} |")
    a("")

    seq = m["vtol_state_sequence"]
    a("## vtol_state 시퀀스")
    a("")
    if seq.get("value") is None:
        a(f"NULL — {seq.get('reason')}")
    else:
        a("| vtol_state | 이름 | 시작(ulog s) | 지속(s) |")
        a("|---|---|---|---|")
        for s in seq["segments"]:
            a(f"| {s['vtol_state']} | {s['name']} | {s['t_start_ulog_s']} "
              f"| {s['duration_s']} |")
    a("")

    a("## 경고 / 에러 (전량 — 무해성 판단은 사람이 한다)")
    a("")
    ws = ([dict(src="node.log", **x) for x in m["node_log"]["errors"]]
          + [dict(src="node.log", **x) for x in m["node_log"]["warnings"]]
          + [dict(src="mavros.log", **x)
             for x in ((m.get("mavros_log") or {}).get("warnings") or [])])
    if not ws:
        a("없음.")
    else:
        a("| 출처 | 레벨 | 건수 | 최초 | 비고 | 예시 |")
        a("|---|---|---|---|---|---|")
        for x in ws:
            smp = x["sample"].replace("|", "\\|")[:150]
            if x.get("first_t") is None:
                ft = f"(줄 {x.get('first_line', '?')})"
            elif x.get("raw_stdout"):
                ft = f"≈{x['first_t']:.1f}"
            else:
                ft = f"{x['first_t']:.1f}"
            note = []
            if x.get("known_cosmetic"):
                note.append("알려진 코스메틱")
            if x.get("raw_stdout"):
                note.append("stdout 중계(비-ROS 포맷)")
            if x.get("after_sigint"):
                note.append("하니스 SIGINT 이후")
            a(f"| {x['src']} | {x['level']} | {x['count']} | "
              f"{ft} | {', '.join(note)} | {smp} |")
    a("")

    inj = meta.get("inject_results") or []
    if inj:
        a("## 장애주입 결과")
        a("")
        for r in inj:
            a(f"- `{r.get('action')}` spec={json.dumps(r.get('spec'), ensure_ascii=False)} "
              f"→ 발화 +{r.get('fired_mono_s')}s rc={r.get('rc')}")
        a("")

    a("## 미산출 지표 (null)")
    a("")
    nulls = [v for v in verdict if v["status"] == "NULL"]
    if not nulls:
        a("없음 — 계획서 4장 지표 전부 산출됨.")
    else:
        for v in nulls:
            a(f"- **{v['item']}**: {v['detail']}")
    a("")
    a("---")
    a("")
    a("판정 기준 출처: `docs/sitl_vtol_campaign.md` 4장. "
      "경고의 무해성 판단은 이 스크립트가 하지 않는다.")
    return "\n".join(lines) + "\n"


# ── 메인 ───────────────────────────────────────────────────────────────────

def resolve_params(meta: dict) -> dict:
    with open(PARAMS_YAML, encoding="utf-8") as f:
        y = yaml.safe_load(f)["offboard_node"]["ros__parameters"]
    la = meta.get("launch_args") or {}
    p = {
        "control_hz": float(la.get("control_hz", y["control_hz"])),
        "v_approach": float(la.get("v_approach", y["v_approach"])),
        "v_cruise": float(la.get("v_cruise", y["v_cruise"])),
        "transition_alt": float(la.get("transition_alt", y["transition_alt"])),
        "d_end_thresh": float(la.get("d_end_thresh", y["d_end_thresh"])),
        "wp0_heading_tol": float(la.get("wp0_heading_tol",
                                        y["wp0_heading_tol"])),
        "l1_dist": float(la.get("l1_dist", y["l1_dist"])),
        "waypoint_frame": str(la.get("waypoint_frame", y["waypoint_frame"])),
        "entry_mode": str(la.get("entry_mode", y["entry_mode"])),
    }
    wp_raw = la.get("waypoints")
    if wp_raw:
        flat = [float(x) for x in yaml.safe_load(wp_raw)]
    else:
        flat = [float(x) for x in y["waypoints"]]
    p["_waypoints"] = np.array(flat, dtype=float).reshape(-1, 3)
    p["_cruise_alt_wp"] = float(p["_waypoints"][-1, 2])
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="SITL-7 런 로그 → 지표/판정")
    ap.add_argument("scenario_id", nargs="?")
    ap.add_argument("--dir", default=None, help="런 디렉터리 직접 지정")
    args = ap.parse_args()
    if args.dir:
        run = Path(args.dir)
    elif args.scenario_id:
        run = CAMPAIGN_DIR / args.scenario_id
    else:
        ap.error("scenario_id 또는 --dir 이 필요하다")
    if not run.is_dir():
        print(f"런 디렉터리 없음: {run}", file=sys.stderr)
        return 5

    meta_path = run / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) \
        if meta_path.exists() else {"scenario_id": run.name}
    params = resolve_params(meta)

    nodelog = parse_node_log(run / "node.log")
    # 시각 앵커용 원문 (state 패턴에 안 걸리는 명령 로그도 필요)
    raw_msgs = []
    if (run / "node.log").exists():
        with open(run / "node.log", encoding="utf-8", errors="replace") as f:
            for line in f:
                mm = LOG_LINE_RE.match(line.strip())
                if mm:
                    raw_msgs.append((float(mm.group("t")), mm.group("msg")))
    nodelog["_raw_msgs"] = raw_msgs

    # ⚠️ ulog 선택은 반드시 meta.json 의 "ulogs"(이번 런이 실제로 수집한 파일)를
    #    따른다. 크기/이름으로 고르면 같은 시나리오를 재실행했을 때 이전 런의
    #    ulog 를 새 node.log 와 짝지어 분석하는 사고가 난다(A1 재실행에서 실측).
    ulog_sel_note = None
    picked = None
    for name in reversed(meta.get("ulogs") or []):
        cand = run / name
        if cand.exists():
            picked = cand
            break
    if picked is None:
        ulgs = sorted(run.glob("*.ulg"), key=lambda p: p.stat().st_mtime)
        if ulgs:
            picked = ulgs[-1]
            ulog_sel_note = (
                "meta.json 의 ulogs 로 특정하지 못해 mtime 최신 파일로 폴백 — "
                "이 런의 ulog 가 맞는지 직접 확인할 것")
    ulog = None
    ulog_err = None
    if picked is not None:
        try:
            ulog = load_ulog(picked)
        except Exception as e:            # noqa: BLE001
            ulog_err = f"{type(e).__name__}: {e}"
    else:
        ulog_err = "런 디렉터리에 .ulg 파일이 없음"

    metrics: dict = {
        "scenario_id": meta.get("scenario_id", run.name),
        "run_dir": str(run),
        "ulog_file": picked.name if picked else None,
        "ulog_selection_note": ulog_sel_note,
        "ulog_error": ulog_err,
        "resolved_params": {k: v for k, v in params.items()
                            if not k.startswith("_")},
        "waypoints": params["_waypoints"].tolist(),
    }

    if ulog is not None:
        ta = estimate_time_offset(ulog, nodelog)
        lp = topic(ulog, "vehicle_local_position")
        t_end = float(tsec(lp)[-1]) if lp is not None else None
        metrics["ulog_topics"] = sorted({d.name for d in ulog.data_list})
        metrics["ulog_duration_s"] = _f(t_end)
    else:
        ta = {"offset_s": None, "reason": ulog_err, "anchors": []}
        t_end = None
        metrics["ulog_topics"] = []
        metrics["ulog_duration_s"] = None

    metrics["time_alignment"] = ta
    params["_offset"] = ta.get("offset_s")
    params["_scale"] = ta.get("scale")
    win = state_windows(nodelog, ta, t_end)
    metrics["state_windows_ulog_s"] = {k: [_f(v[0]), _f(v[1])]
                                       for k, v in win.items()}

    metrics["completion"] = metric_completion(nodelog, ulog, meta)
    metrics["vtol_state_sequence"] = (metric_vtol_sequence(ulog) if ulog
                                      else null(ulog_err))
    metrics["setpoint_jump"] = (metric_setpoint_jump(ulog, win, params)
                                if ulog else null(ulog_err))
    metrics["vertical_accel"] = (
        metric_vertical_accel(ulog, metrics["vtol_state_sequence"], win,
                              metrics["completion"].get("disarm"))
        if ulog else null(ulog_err))
    metrics["decel"] = (metric_decel(ulog, metrics["vtol_state_sequence"], win)
                        if ulog else null(ulog_err))
    metrics["heading"] = (
        metric_heading(ulog, nodelog, win, params, params["_waypoints"])
        if ulog else null(ulog_err))
    metrics["altitude"] = (
        metric_altitude(ulog, nodelog, win, metrics["vtol_state_sequence"],
                        params)
        if ulog else null(ulog_err))
    po = None
    if nodelog.get("path_origin"):
        po = (nodelog["path_origin"]["n"], nodelog["path_origin"]["e"])
    metrics["cte"] = metric_cte(nodelog, ulog, win, params["_waypoints"], po)
    metrics["segment_index"] = metric_segment_index(nodelog)

    nl = dict(nodelog)
    nl.pop("_raw_msgs", None)
    metrics["node_log"] = nl
    metrics["mavros_log"] = parse_mavros_log(run / "mavros.log")

    verdict = build_verdict(metrics, params)
    metrics["verdict"] = verdict

    (run / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run / "verdict.md").write_text(
        render_verdict(meta, metrics, verdict, params), encoding="utf-8")

    counts = {}
    for v in verdict:
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    print(f"[analyze_run] {run.name}: " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"[analyze_run] → {run/'metrics.json'}, {run/'verdict.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
