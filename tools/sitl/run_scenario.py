#!/usr/bin/env python3
"""SITL-7 VTOL 회귀 캠페인 — 시나리오 1건 실행기.

**WSL 배포판 `Ubuntu-22.04` 안에서** 돈다(호스트 쪽에서 호출하는 스크립트가 아니다).
ROS2/워크스페이스 소싱은 같은 폴더의 `run_scenario.sh` 래퍼가 해준다.

    bash tools/sitl/run_scenario.sh A1          # 권장 (소싱 포함)
    python3 tools/sitl/run_scenario.py A1       # 이미 소싱된 셸에서

하는 일 (docs/sitl_vtol_campaign.md 2장):
  1. scenarios.yaml 에서 시나리오 정의를 읽는다
  2. PX4 SITL 기동 (px4 바이너리 직접 실행 — make 를 거치지 않아 잔류 프로세스가 적다)
  3. MAVROS 기동 → /mavros/state 의 connected=true 대기
  4. SITL 프리플라이트 우회 파라미터 설정 (CBRK_SUPPLY_CHK / NAV_DLL_ACT)
  5. `ros2 launch fc_ros phase2.launch.py <시나리오 인자>` 실행, stdout → node.log
  6. 종료 감시: DONE 도달 / 노드 자체 종료 / timeout_s 초과.
     ⚠️ 미션 시계는 launch 기동이 아니라 **offboard_node 첫 로그**부터 잰다 —
     플래너가 __init__ 을 동기 블로킹하기 때문(정적 감사 E-11: 꺾임 경로 45~130초).
     그 앞 구간은 별도 boot_timeout_s(기본 300s)로 감시한다.
  7. 산출물 수집 → logs/2026-07-27_sitl_vtol_campaign/<id>/
  8. 프로세스 정리

⚠️ 절대 어기면 안 되는 것 (전부 실측된 함정, docs/sitl_vtol_campaign.md 1장):
  - **PX4 콘솔(pxh>)을 파일로 리다이렉트하지 않는다.** 비-TTY 재출력 루프로 20초에
    195MB까지 폭주한 실측이 있다. 이 스크립트는 PX4 stdout/stderr 를 /dev/null 로
    버리고 stdin 도 /dev/null 로 끊는다. offboard_node(ros2 launch) 로그는 pxh 가
    아니므로 파일 저장이 안전하다.
  - 시나리오 간 정리는 이 스크립트의 정리 루틴만 믿지 말고 호출자가
    `wsl.exe --terminate Ubuntu-22.04` 로 배포판을 통째 재기동하는 것을 기본으로 한다
    (gz sim 잔류 → 다음 런이 이전 gz 서버에 얹히는 중복 인스턴스 실측).

종료 코드:
  0  DONE 도달 (미션 완주)
  2  timeout_s 초과 (미완주 — 그 자체가 발견사항이다, 산출물은 정상 수집됨)
  3  환경 브링업 실패 (PX4/MAVROS 미기동, 또는 offboard_node 가 boot_timeout_s 안에
     첫 로그를 못 찍음 = 플래너 무한대기) — 산출물 불완전
  4  ros2 launch 프로세스가 DONE 없이 스스로 종료 (노드 크래시 등)
  5  사용법/정의 오류 (시나리오 없음 등)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_YAML = Path(__file__).resolve().parent / "scenarios.yaml"
CAMPAIGN_DIR = REPO_ROOT / "logs" / "2026-07-27_sitl_vtol_campaign"

PX4_DIR = Path(os.environ.get("PX4_DIR", "/root/PX4-Autopilot"))
PX4_BUILD = PX4_DIR / "build" / "px4_sitl_default"
PX4_BIN = PX4_BUILD / "bin" / "px4"
PX4_ROOTFS = PX4_BUILD / "rootfs"
PX4_LOG_DIR = PX4_ROOTFS / "log"

MAVROS_FCU_URL = os.environ.get(
    "MAVROS_FCU_URL", "udp://:14540@localhost:14580")

# SITL 벤치 전용 프리플라이트 우회 — 실기체 파라미터에는 절대 적용 금지.
PREFLIGHT_BYPASS = {
    "CBRK_SUPPLY_CHK": 894281,   # "system power unavailable" 우회
    "NAV_DLL_ACT": 0,            # "No connection to the GCS" 우회
}

# node.log 한 줄 형식:
#   [offboard_node-2] [INFO] [1753600000.123456789] [offboard_node]: 메시지
LOG_LINE_RE = re.compile(
    r"^\[(?P<proc>[^\]]+)\]\s+\[(?P<level>DEBUG|INFO|WARN|ERROR|FATAL)\]\s+"
    r"\[(?P<t>\d+\.\d+)\]\s+\[(?P<logger>[^\]]+)\]:\s*(?P<msg>.*)$")

# 상태 진입 판정 — offboard_node 가 실제로 찍는 문장 (fc_ros/nodes/offboard_node.py).
# 순서 중요: 위에서부터 먼저 맞는 것을 쓴다.
STATE_ENTRY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ARM_TAKEOFF",    re.compile(r"^ARM 요청")),
    ("CLIMBING",       re.compile(r"CommandTOL 이륙 요청.*-> CLIMBING")),
    ("TRANSITION_FW",  re.compile(r"운용 고도 .* 도달 → transition_fw")),
    ("STREAMING",      re.compile(r"^(FW 전환 완료 -> STREAMING|운용 고도 .* 도달 → streaming)")),
    # ⚠️ entry_mode=mid_flight 여도 노드는 "OFFBOARD 확인 → FOLLOWING" 을 찍고
    #    실제로는 ENTRY 로 간다(offboard_node.py STREAMING 분기의 로그 부정확).
    #    ENTRY 진입은 이 문장 다음의 "ENTRY 완료" 유무로만 사후 판별 가능하다.
    ("FOLLOWING",      re.compile(r"^(OFFBOARD 확인 → FOLLOWING|ENTRY 완료 -> FOLLOWING)")),
    ("TRANSITION_MC",  re.compile(r"경로 추종 완료 -> transition_mc")),
    ("HOLD",           re.compile(r"MC 전환 완료 -> HOLD")),
    ("LANDING",        re.compile(r"(WP1 도달·안정 → LANDING|→ 강제 LANDING)")),
    ("OVERRIDE",       re.compile(r"긴급 수동 전환 실행 →")),
    ("PILOT_TAKEOVER", re.compile(r"조종사 인계 감지")),
    ("DONE",           re.compile(r"(착륙 완료 \(disarmed\) -> DONE|수동/안전 모드 진입 확인.*-> DONE)")),
]

TERMINAL_STATES = {"DONE"}


def log(msg: str) -> None:
    print(f"[run_scenario] {msg}", flush=True)


# ── 시나리오 정의 ───────────────────────────────────────────────────────────

def load_scenario(scenario_id: str) -> dict:
    with open(SCENARIOS_YAML, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    defaults = doc.get("defaults", {}) or {}
    for sc in doc.get("scenarios", []):
        if sc.get("id") == scenario_id:
            merged = dict(defaults)
            merged.update(sc)
            return merged
    ids = [s.get("id") for s in doc.get("scenarios", [])]
    raise SystemExit(
        f"시나리오 '{scenario_id}' 없음. 정의된 id: {', '.join(map(str, ids))}")


# ── 프로세스 관리 ───────────────────────────────────────────────────────────

class Proc:
    """setsid 로 띄운 자식 프로세스 — 프로세스 그룹째 신호를 보낸다."""

    def __init__(self, name: str, argv: list[str], *, cwd=None, env=None,
                 stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT):
        self.name = name
        self.argv = argv
        self._out = stdout
        self.p = subprocess.Popen(
            argv, cwd=str(cwd) if cwd else None, env=env,
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            start_new_session=True)
        log(f"기동 {name} pid={self.p.pid}")

    @property
    def alive(self) -> bool:
        return self.p.poll() is None

    def stop(self, sig=signal.SIGINT, wait_s: float = 8.0) -> None:
        if not self.alive:
            return
        try:
            os.killpg(os.getpgid(self.p.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        t0 = time.monotonic()
        while self.alive and time.monotonic() - t0 < wait_s:
            time.sleep(0.2)
        if self.alive:
            log(f"{self.name}: SIGINT 무반응 → SIGKILL")
            try:
                os.killpg(os.getpgid(self.p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            if hasattr(self._out, "close"):
                self._out.close()
        except Exception:
            pass


def sh(argv: list[str], timeout: float = 20.0) -> tuple[int, str]:
    try:
        r = subprocess.run(argv, stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout)
        return r.returncode, r.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, "(timeout)"
    except FileNotFoundError as e:
        return 127, str(e)


# ── 브링업 ─────────────────────────────────────────────────────────────────

def start_px4(model: str, home: dict | None) -> Proc:
    env = dict(os.environ)
    env["PX4_SIM_MODEL"] = model            # 예: gz_standard_vtol
    env["GZ_IP"] = "127.0.0.1"
    env["HEADLESS"] = "1"                   # gz GUI 미기동 (px4-rc.gzsim)
    if home:
        for k, ek in (("lat", "PX4_HOME_LAT"), ("lon", "PX4_HOME_LON"),
                      ("alt", "PX4_HOME_ALT")):
            if home.get(k) is not None:
                env[ek] = str(home[k])
    if not PX4_BIN.exists():
        raise SystemExit(f"PX4 바이너리 없음: {PX4_BIN}")
    # ⚠️ stdout/stderr 는 반드시 DEVNULL. 파일로 받으면 pxh 재출력 루프로 폭주한다.
    return Proc("px4", [str(PX4_BIN)], cwd=PX4_ROOTFS, env=env)


def wait_mavros_connected(timeout_s: float) -> tuple[bool, str]:
    """/mavros/state 의 connected=true 를 기다린다."""
    t0 = time.monotonic()
    last = ""
    while time.monotonic() - t0 < timeout_s:
        # ⚠️ Humble 의 `ros2 topic echo` 에는 --timeout 옵션이 없다 (Jazzy 부터).
        #    붙이면 argparse 오류로 매번 rc!=0 이 되어 브링업이 통째로 실패한다(실측).
        #    대신 subprocess 타임아웃으로 끊는다.
        rc, out = sh(["ros2", "topic", "echo", "--once",
                      "/mavros/state"], timeout=10)
        last = out.strip()[-400:]
        if rc == 0 and re.search(r"connected:\s*true", out):
            return True, last
        time.sleep(1.0)
    return False, last


RE_PARAM_VALUE = re.compile(r"value is:\s*(-?\d+)")


def _param_get(name: str) -> int | None:
    rc, out = sh(["ros2", "param", "get", "/mavros/param", name], timeout=20)
    if rc != 0:
        return None
    m = RE_PARAM_VALUE.search(out)
    return int(m.group(1)) if m else None


def set_preflight_bypass(timeout_s: float = 180.0) -> dict:
    """SITL 프리플라이트 우회 파라미터 설정 + **읽기 되돌림 검증**.

    ⚠️ `ros2 param set /mavros/param X` 은 MAVROS 파라미터 목록 동기화가 끝나기
    전에도 `Set parameter successful` 을 돌려준다 — 실제로는 MAVROS 가
    `PR: Unknown parameter to set: CBRK_SUPPLY_CHK` 로 버린다(A1 1차 실행에서
    실측, 하니스는 OK 로 보고했는데 값은 안 들어감). 그래서 set 만으로 성공을
    판단하지 않고 반드시 get 으로 되읽어 확인한다.
    """
    results = {}
    for name, value in PREFLIGHT_BYPASS.items():
        verified, out, cur = False, "", None
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            cur = _param_get(name)
            if cur is None:           # MAVROS 가 아직 이 파라미터를 모른다
                time.sleep(3.0)
                continue
            if cur == value:
                verified = True
                break
            rc, out = sh(["ros2", "param", "set", "/mavros/param",
                          name, str(value)], timeout=25)
            time.sleep(2.0)
            cur = _param_get(name)
            if cur == value:
                verified = True
                break
            time.sleep(2.0)
        results[name] = {"ok": verified, "want": value, "readback": cur,
                         "output": out.strip()[-200:]}
        log(f"파라미터 {name}={value} → "
            f"{'검증됨' if verified else f'검증 실패(readback={cur})'}")
    return results


# ── 장애주입 훅 ────────────────────────────────────────────────────────────

class Injector(threading.Thread):
    """시나리오 정의의 `inject:` 항목을 트리거 성립 시 실행한다.

    트리거는 세 가지: on_state(node.log 상태 진입) / on_vtol_state(MAVROS
    /mavros/extended_state 의 vtol_state) / at_s(런 시작 후 절대 경과시간).
    실행 결과(발화 시각·명령·표준출력)는 meta.json 의 inject 항목에 남는다.
    """

    def __init__(self, specs: list[dict], t0: float, state_events: list[dict],
                 stop_evt: threading.Event, outdir: Path):
        super().__init__(daemon=True)
        self.specs = [dict(s) for s in specs]
        self.t0 = t0
        self.state_events = state_events    # Monitor 가 채우는 공유 리스트
        self.stop_evt = stop_evt
        self.outdir = outdir
        self.results: list[dict] = []
        self._vtol_state: int | None = None
        self._vtol_seen: set[int] = set()
        self._vtol_hit: dict[int, float] = {}

    # -- 트리거 판정 --------------------------------------------------
    def _trigger_time(self, spec: dict) -> float | None:
        """트리거가 성립한 monotonic 시각(성립 전이면 None)."""
        if "at_s" in spec:
            return self.t0 + float(spec["at_s"])
        if "on_state" in spec:
            want = str(spec["on_state"]).upper()
            for ev in list(self.state_events):
                if ev["state"] == want:
                    return ev["mono"]
            return None
        if "on_vtol_state" in spec:
            want = int(spec["on_vtol_state"])
            if want in self._vtol_seen:
                return self._vtol_hit.get(want)
            return None
        return None

    def _poll_vtol_state(self) -> None:
        rc, out = sh(["ros2", "topic", "echo", "--once",
                      "/mavros/extended_state"], timeout=8)
        if rc != 0:
            return
        m = re.search(r"vtol_state:\s*(\d+)", out)
        if not m:
            return
        v = int(m.group(1))
        self._vtol_state = v
        if v not in self._vtol_seen:
            self._vtol_seen.add(v)
            self._vtol_hit[v] = time.monotonic()

    # -- 액션 실행 ----------------------------------------------------
    def _fire(self, spec: dict) -> dict:
        action = str(spec.get("action", "")).lower()
        rec = {"spec": spec, "action": action,
               "fired_mono_s": round(time.monotonic() - self.t0, 3),
               "fired_utc": datetime.now(timezone.utc).isoformat()}
        if action == "set_mode":
            mode = str(spec["mode"])
            rc, out = sh(["ros2", "service", "call", "/mavros/set_mode",
                          "mavros_msgs/srv/SetMode",
                          f"{{base_mode: 0, custom_mode: '{mode}'}}"],
                         timeout=20)
        elif action == "override":
            rc, out = sh(["ros2", "topic", "pub", "--once", "/fc_ros/override",
                          "std_msgs/msg/Bool", "{data: true}"], timeout=20)
        elif action == "param_set":
            rc, out = sh(["ros2", "param", "set", "/mavros/param",
                          str(spec["param"]), str(spec["value"])], timeout=25)
        elif action == "probe":
            topic = str(spec.get("topic", "/mavros/state"))
            rc, out = sh(["ros2", "topic", "echo", "--once",
                          topic], timeout=12)
        else:
            rc, out = 5, f"알 수 없는 action: {action!r}"
        rec["rc"] = rc
        rec["output"] = out.strip()[-1500:]
        log(f"inject 발화 {action} rc={rc} (+{rec['fired_mono_s']}s)")
        return rec

    def run(self) -> None:
        pending = list(enumerate(self.specs))
        needs_vtol = any("on_vtol_state" in s for s in self.specs)
        while pending and not self.stop_evt.is_set():
            if needs_vtol:
                self._poll_vtol_state()
            now = time.monotonic()
            still = []
            for idx, spec in pending:
                trig = self._trigger_time(spec)
                if trig is None:
                    still.append((idx, spec))
                    continue
                due = trig + float(spec.get("delay_s", 0.0))
                if now >= due:
                    self.results.append(self._fire(spec))
                else:
                    still.append((idx, spec))
            pending = still
            self.stop_evt.wait(0.3)
        for _, spec in pending:
            self.results.append(
                {"spec": spec, "action": spec.get("action"),
                 "fired_mono_s": None, "rc": None,
                 "output": "트리거 미성립 — 발화하지 않음"})


# ── 실행 감시 ──────────────────────────────────────────────────────────────

class Monitor(threading.Thread):
    """node.log 를 증분으로 읽어 상태 전이/경고를 수집한다."""

    def __init__(self, path: Path, stop_evt: threading.Event):
        super().__init__(daemon=True)
        self.path = path
        self.stop_evt = stop_evt
        self.state_events: list[dict] = []
        self.seen_states: set[str] = set()
        self.warn_count = 0
        self.error_count = 0
        self.done = threading.Event()
        # offboard_node 가 첫 줄을 찍은 monotonic 시각.
        # 노드는 __init__ 에서 플래너를 **동기 실행**하므로(정적 감사 E-11:
        # 꺾임 경로는 45~130초) 그 전까지는 로그가 한 줄도 안 나온다. 미션
        # 타임아웃을 launch 기동 시점부터 재면 플래너 계산시간이 미션 예산을
        # 통째로 잡아먹는다 → 이 시각부터 미션 시계를 시작한다.
        self.node_alive_mono: float | None = None

    def run(self) -> None:
        while not self.path.exists() and not self.stop_evt.is_set():
            time.sleep(0.2)
        if self.stop_evt.is_set():
            return
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            buf = ""
            while not self.stop_evt.is_set():
                chunk = f.read()
                if not chunk:
                    time.sleep(0.2)
                    continue
                buf += chunk
                *lines, buf = buf.split("\n")
                for line in lines:
                    self._handle(line)

    def _handle(self, line: str) -> None:
        m = LOG_LINE_RE.match(line.strip())
        if not m:
            return
        level, msg, t = m.group("level"), m.group("msg"), float(m.group("t"))
        if (self.node_alive_mono is None
                and "offboard_node" in m.group("proc")):
            self.node_alive_mono = time.monotonic()
            log("offboard_node 첫 로그 관측 (플래너 계산 완료) — 미션 시계 시작")
        if level == "WARN":
            self.warn_count += 1
        elif level in ("ERROR", "FATAL"):
            self.error_count += 1
        for state, pat in STATE_ENTRY_PATTERNS:
            if pat.search(msg):
                if state not in self.seen_states:
                    self.seen_states.add(state)
                    self.state_events.append(
                        {"state": state, "ros_t": t,
                         "mono": time.monotonic(), "msg": msg})
                    log(f"상태 진입: {state}  ({msg[:70]})")
                if state in TERMINAL_STATES:
                    self.done.set()
                break


# ── 산출물 수집 ────────────────────────────────────────────────────────────

def snapshot_ulogs() -> set[Path]:
    if not PX4_LOG_DIR.exists():
        return set()
    return set(PX4_LOG_DIR.rglob("*.ulg"))


def collect_ulogs(before: set[Path], outdir: Path) -> list[str]:
    after = snapshot_ulogs()
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    copied = []
    for p in new:
        dst = outdir / p.name
        if dst.exists():
            dst = outdir / f"{p.stem}_{int(p.stat().st_mtime)}.ulg"
        shutil.copy2(p, dst)
        copied.append(dst.name)
        log(f"ulog 수집 {p} → {dst.name} ({p.stat().st_size/1e6:.1f} MB)")
    return copied


def trim_file(path: Path, max_bytes: int = 4_000_000) -> None:
    """폭주 대비 — 커진 로그는 head/tail 만 남긴다."""
    if not path.exists() or path.stat().st_size <= max_bytes:
        return
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(max_bytes // 2)
        f.seek(-max_bytes // 2, os.SEEK_END)
        tail = f.read()
    with open(path, "wb") as f:
        f.write(head)
        f.write(f"\n\n... [{size - max_bytes} bytes 중략: run_scenario.py 가 잘라냄] ...\n\n"
                .encode())
        f.write(tail)
    log(f"{path.name} 이 {size/1e6:.1f} MB → 중간 잘라냄")


# ── 메인 ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="SITL-7 시나리오 1건 실행")
    ap.add_argument("scenario_id")
    ap.add_argument("--outdir", default=None,
                    help="산출물 루트 (기본 logs/2026-07-27_sitl_vtol_campaign)")
    ap.add_argument("--run-id", default=None,
                    help="산출물 하위 디렉터리 이름 (기본: 시나리오 id). "
                         "같은 시나리오를 다른 PX4 빌드로 재실행할 때 기존 결과를 "
                         "덮어쓰지 않도록 구분한다 (예: A3 → A3_pxvehicle)")
    ap.add_argument("--px4-boot-timeout", type=float, default=90.0)
    ap.add_argument("--mavros-timeout", type=float, default=120.0)
    ap.add_argument("--post-done-s", type=float, default=6.0,
                    help="DONE 관측 후 로그를 더 받는 시간(초)")
    ap.add_argument("--boot-timeout-s", type=float, default=300.0,
                    help="ros2 launch 기동 후 offboard_node 첫 로그까지 허용 시간(초). "
                         "플래너가 __init__ 을 블로킹하므로 timeout_s 와 분리한다 "
                         "(정적 감사 E-11: 꺾임 경로 45~130초 실측)")
    ap.add_argument("--keep-running", action="store_true",
                    help="종료 후 PX4/MAVROS 를 죽이지 않는다 (디버그용)")
    args = ap.parse_args()

    sc = load_scenario(args.scenario_id)
    root = Path(args.outdir) if args.outdir else CAMPAIGN_DIR
    outdir = root / (args.run_id or sc["id"])
    outdir.mkdir(parents=True, exist_ok=True)
    for stale in ("node.log", "mavros.log", "meta.json",
                  "metrics.json", "verdict.md"):
        (outdir / stale).unlink(missing_ok=True)
    # ⚠️ 이전 런의 ulog 를 반드시 지운다. 남겨두면 analyze_run.py 가 엉뚱한 런의
    #    ulog 를 골라 새 node.log 와 짝지어 분석한다(A1 재실행에서 실측 —
    #    meta.json 은 17_20_05.ulg 인데 분석은 17_09_11.ulg 로 돌아갔다).
    for old_ulg in outdir.glob("*.ulg"):
        old_ulg.unlink()
        log(f"이전 런 잔여 ulog 삭제: {old_ulg.name}")

    launch_args = [f"{k}:={v}" for k, v in (sc.get("launch_args") or {}).items()]
    meta = {
        "scenario_id": sc["id"],
        "desc": sc.get("desc", ""),
        "model": sc.get("model", "gz_standard_vtol"),
        "launch_args": sc.get("launch_args") or {},
        "launch_argv": ["ros2", "launch", "fc_ros", "phase2.launch.py", *launch_args],
        "home": sc.get("home"),
        "timeout_s": float(sc.get("timeout_s", 420)),
        "inject_spec": sc.get("inject") or [],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id or sc["id"],
        "px4_dir": str(PX4_DIR),
        # ⚠️ 어느 PX4 빌드에서 돈 결과인지 반드시 남긴다. SITL-7 S4 에서 SITL 의
        #    PX4(9bb0d365c4)와 실기체 PX4(c890d9db0a)의 오프보드 course 처리가
        #    다르다는 것이 확인됐다 — 런 결과는 PX4 커밋과 짝지어야만 해석된다.
        "px4_head": sh(["git", "-C", str(PX4_DIR),
                        "rev-parse", "HEAD"])[1].strip(),
        "px4_bin": str(PX4_BIN),
        "px4_bin_mtime_utc": (
            datetime.fromtimestamp(PX4_BIN.stat().st_mtime, timezone.utc)
            .isoformat() if PX4_BIN.exists() else None),
        "mavros_fcu_url": MAVROS_FCU_URL,
        "host": os.uname().nodename,
        "repo_head": sh(["git", "-C", str(REPO_ROOT),
                         "rev-parse", "--short", "HEAD"])[1].strip(),
    }

    stop_evt = threading.Event()
    procs: list[Proc] = []
    exit_code = 0
    ulogs_before = snapshot_ulogs()
    t0 = time.monotonic()

    def finish(code: int, reason: str) -> int:
        meta["exit_reason"] = reason
        meta["exit_code"] = code
        meta["ended_utc"] = datetime.now(timezone.utc).isoformat()
        meta["elapsed_s"] = round(time.monotonic() - t0, 1)
        return code

    try:
        # 1) PX4 SITL
        log(f"=== {sc['id']} — {sc.get('desc','')}")
        px4 = start_px4(meta["model"], sc.get("home"))
        procs.append(px4)
        time.sleep(6.0)
        if not px4.alive:
            meta["bringup_error"] = "PX4 가 기동 직후 종료됨"
            return finish(3, "px4_dead")

        # 2) MAVROS — pxh 가 아니므로 파일 저장 안전
        mavros_log = open(outdir / "mavros.log", "wb")
        mavros = Proc("mavros",
                      ["ros2", "launch", "mavros", "px4.launch",
                       f"fcu_url:={MAVROS_FCU_URL}"],
                      stdout=mavros_log)
        procs.append(mavros)

        ok, snap = wait_mavros_connected(args.mavros_timeout)
        meta["mavros_state_snapshot"] = snap
        if not ok:
            meta["bringup_error"] = "/mavros/state connected=true 미도달"
            return finish(3, "mavros_not_connected")
        log("MAVROS connected=true 확인")

        # 3) 프리플라이트 우회
        meta["preflight_bypass"] = set_preflight_bypass()
        bad = [k for k, v in meta["preflight_bypass"].items() if not v["ok"]]
        if bad:
            log(f"경고: 프리플라이트 우회 파라미터 검증 실패 {bad} — "
                f"ARM 이 거부될 수 있다 (meta.json preflight_bypass 확인)")

        # 4) 미션 launch
        node_log_path = outdir / "node.log"
        node_log = open(node_log_path, "wb")
        launch = Proc("phase2.launch", meta["launch_argv"], stdout=node_log)
        procs.append(launch)
        launch_t0 = time.monotonic()
        meta["launch_started_utc"] = datetime.now(timezone.utc).isoformat()

        mon = Monitor(node_log_path, stop_evt)
        mon.start()

        inj = None
        if meta["inject_spec"]:
            inj = Injector(meta["inject_spec"], launch_t0,
                           mon.state_events, stop_evt, outdir)
            inj.start()

        # 5) 감시 루프
        timeout_s = meta["timeout_s"]
        meta["boot_timeout_s"] = args.boot_timeout_s
        reason, code = "timeout", 2
        while True:
            # 플래너 블로킹 구간 — 미션 시계는 아직 안 돈다
            if mon.node_alive_mono is None:
                if time.monotonic() - launch_t0 > args.boot_timeout_s:
                    reason, code = "node_boot_timeout", 3
                    log(f"offboard_node 가 {args.boot_timeout_s:.0f}s 안에 "
                        f"첫 로그를 못 찍음 — 플래너 무한대기/기동 실패 의심")
                    break
                if not launch.alive:
                    reason, code = "launch_exited", 4
                    log(f"ros2 launch 가 노드 기동 전에 종료 "
                        f"(rc={launch.p.returncode})")
                    break
                time.sleep(0.5)
                continue
            mission_t0 = mon.node_alive_mono
            if mon.done.is_set():
                log(f"DONE 관측 — {args.post_done_s}s 더 수집 후 종료")
                time.sleep(args.post_done_s)
                reason, code = "done", 0
                break
            if not launch.alive:
                reason, code = "launch_exited", 4
                log(f"ros2 launch 가 DONE 없이 종료 (rc={launch.p.returncode})")
                break
            if not px4.alive:
                reason, code = "px4_died", 4
                log("PX4 프로세스가 죽었다")
                break
            if time.monotonic() - mission_t0 > timeout_s:
                log(f"timeout {timeout_s:.0f}s 초과 (미션 시계 기준) — 강제 종료")
                break
            time.sleep(0.5)

        meta["planner_blocking_s"] = (
            round(mon.node_alive_mono - launch_t0, 2)
            if mon.node_alive_mono else None)
        base = mon.node_alive_mono or launch_t0
        meta["state_timeline"] = [
            {"state": e["state"],
             "ros_t": e["ros_t"],
             "t_since_launch_s": round(e["mono"] - launch_t0, 2),
             "t_since_node_alive_s": round(e["mono"] - base, 2),
             "msg": e["msg"]}
            for e in mon.state_events]
        meta["warn_count"] = mon.warn_count
        meta["error_count"] = mon.error_count
        exit_code = code

        stop_evt.set()
        if inj:
            inj.join(timeout=10.0)
            meta["inject_results"] = inj.results

        return finish(code, reason)

    except KeyboardInterrupt:
        stop_evt.set()
        return finish(2, "interrupted")
    finally:
        stop_evt.set()
        # 역순 정리: launch → mavros → px4 (PX4 는 마지막에 SIGINT 로 ulog flush)
        for p in reversed(procs):
            p.stop()
        # gz 잔류 청소 — 이것만으로는 부족할 수 있다(wsl --terminate 권장)
        subprocess.run(["pkill", "-f", "gz sim"],
                       stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)
        meta["ulogs"] = collect_ulogs(ulogs_before, outdir)
        trim_file(outdir / "mavros.log")
        trim_file(outdir / "node.log", max_bytes=20_000_000)
        meta.setdefault("exit_reason", "unknown")
        meta.setdefault("exit_code", exit_code)
        with open(outdir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        log(f"산출물: {outdir}  (exit={meta['exit_code']} "
            f"reason={meta['exit_reason']})")


if __name__ == "__main__":
    sys.exit(main())
