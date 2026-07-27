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
  6  거리 상한 초과 (--range-limit-m) — 기체가 임무영역 밖으로 이탈. 미완주.
     C10 실측(장애주입 없이 480초 동안 5.85km 이탈, 스트림이 살아 있어 PX4
     offboard-loss 페일세이프도 안 걸림)에서 도입. 시나리오 최장 경로가 500m 이므로
     기본 1500m 는 정상 런에 절대 걸리지 않는다. **실행 감시 전용 — 판정 로직이 아니다.**
"""
from __future__ import annotations

import argparse
import hashlib
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

class RosInjectClient:
    """장애주입용 in-process ROS2 클라이언트 (set_mode 서비스 + override 퍼블리셔).

    **왜 CLI 를 안 쓰는가** — `ros2 service call` 은 매번 새 rclpy 노드를 띄우고
    디스커버리를 다시 하므로 호출 1회에 **4.04초**가 걸린다(C3 2차 실행 실측:
    트리거 로그 21:27:31.961 → Injector 발화 21:27:32.168(+0.21s) → PX4 가
    실제로 모드를 바꾼 시각 ulog 80.64s ≈ 21:27:36.21). 그런데 노려야 하는
    MC→FW 천이 구간은 **3.19초**뿐이라 CLI 로는 구조적으로 못 맞춘다 —
    2차 실행에서 주입이 천이 종료 1.66초 **뒤**(FOLLOWING 중)에 착탄했다.

    여기서는 런 시작 시점에 노드·클라이언트·퍼블리셔를 미리 만들어 디스커버리를
    끝내둔다. 발화 순간엔 이미 연결돼 있으므로 지연이 수십 ms 로 줄어든다.
    실패하면 조용히 CLI 폴백으로 돌아간다(`available()` 이 False).
    """

    def __init__(self):
        self.ok = False
        self._node = None
        self._cli = None
        self._pub = None
        self._exec = None
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from mavros_msgs.srv import SetMode
            from std_msgs.msg import Bool
            self._rclpy = rclpy
            self._SetMode = SetMode
            self._Bool = Bool
            rclpy.init(args=None)
            self._node = rclpy.create_node("sitl7_injector")
            self._cli = self._node.create_client(SetMode, "/mavros/set_mode")
            self._pub = self._node.create_publisher(Bool, "/fc_ros/override", 10)
            self._exec = SingleThreadedExecutor()
            self._exec.add_node(self._node)
            self.ok = True
            log("inject: in-process ROS 클라이언트 준비됨 (CLI 폴백 불필요)")
        except Exception as e:                        # noqa: BLE001
            log(f"inject: in-process ROS 클라이언트 실패 → CLI 폴백 ({e})")
            self.ok = False

    def wait_ready(self, timeout_s: float = 60.0) -> bool:
        """서비스 발견까지 대기 — 발화 전에 미리 끝내둔다."""
        if not self.ok:
            return False
        try:
            return bool(self._cli.wait_for_service(timeout_sec=timeout_s))
        except Exception:                             # noqa: BLE001
            return False

    def set_mode(self, mode: str, timeout_s: float = 8.0) -> tuple[int, str]:
        if not self.ok:
            return 127, "in-process 클라이언트 없음"
        try:
            req = self._SetMode.Request()
            req.base_mode = 0
            req.custom_mode = mode
            fut = self._cli.call_async(req)
            t0 = time.monotonic()
            while not fut.done() and time.monotonic() - t0 < timeout_s:
                self._exec.spin_once(timeout_sec=0.05)
            if not fut.done():
                return 124, f"set_mode({mode}) 응답 없음 ({timeout_s}s)"
            r = fut.result()
            return (0 if getattr(r, "mode_sent", False) else 1,
                    f"in-process SetMode(custom_mode={mode!r}) "
                    f"→ mode_sent={getattr(r, 'mode_sent', None)} "
                    f"(왕복 {time.monotonic()-t0:.3f}s)")
        except Exception as e:                        # noqa: BLE001
            return 126, f"set_mode 예외: {e}"

    def override(self, wait_sub_s: float = 5.0) -> tuple[int, str]:
        if not self.ok:
            return 127, "in-process 클라이언트 없음"
        try:
            t0 = time.monotonic()
            # `ros2 topic pub --once` 가 구독자 매칭 전에 종료돼 메시지가 통째로
            # 사라지는 사고를 피하려고, 구독자 확인 후 발행한다.
            while (self._pub.get_subscription_count() == 0
                   and time.monotonic() - t0 < wait_sub_s):
                self._exec.spin_once(timeout_sec=0.05)
            n = self._pub.get_subscription_count()
            msg = self._Bool()
            msg.data = True
            self._pub.publish(msg)
            for _ in range(6):
                self._exec.spin_once(timeout_sec=0.02)
            return (0 if n > 0 else 1,
                    f"in-process /fc_ros/override <- Bool(true), 구독자 {n}개 "
                    f"(대기 {time.monotonic()-t0:.3f}s)")
        except Exception as e:                        # noqa: BLE001
            return 126, f"override 예외: {e}"

    def close(self) -> None:
        if not self.ok:
            return
        try:
            self._exec.remove_node(self._node)
            self._node.destroy_node()
            self._rclpy.shutdown()
        except Exception:                             # noqa: BLE001
            pass


class Injector(threading.Thread):
    """시나리오 정의의 `inject:` 항목을 트리거 성립 시 실행한다.

    트리거는 네 가지: on_state(node.log 상태 진입) / on_log(node.log 임의 문장
    정규식) / on_vtol_state(MAVROS /mavros/extended_state 의 vtol_state) /
    at_s(런 시작 후 절대 경과시간).
    실행 결과(발화 시각·명령·표준출력)는 meta.json 의 inject 항목에 남는다.

    ⚠️ **트리거 지연** — `on_vtol_state` 는 `ros2 topic echo --once` 폴링이라
    한 바퀴가 1~2초다. C3 1차 실행(`C3_pxvehicle_try1_noinject`)에서 MC→FW
    천이(vtol_state==1)가 **2.60초**밖에 안 돼 폴링이 통째로 놓쳤다
    (`inject_results` = "트리거 미성립"). 짧은 구간을 노릴 때는 Monitor 가 이미
    증분으로 읽고 있는 node.log 에 거는 `on_log` 를 써야 한다 — 지연이
    폴링주기가 아니라 루프주기(0.1s)로 줄어든다.
    """

    def __init__(self, specs: list[dict], t0: float, state_events: list[dict],
                 stop_evt: threading.Event, outdir: Path,
                 log_hits: dict[str, float] | None = None,
                 ros: "RosInjectClient | None" = None):
        super().__init__(daemon=True)
        self.ros = ros
        self.specs = [dict(s) for s in specs]
        self.t0 = t0
        self.state_events = state_events    # Monitor 가 채우는 공유 리스트
        self.log_hits = log_hits if log_hits is not None else {}
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
        if "on_log" in spec:
            # Monitor 가 채우는 {패턴: 최초 매치 monotonic} 사전
            return self.log_hits.get(str(spec["on_log"]))
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
        via = "cli"
        if action == "set_mode":
            mode = str(spec["mode"])
            if self.ros and self.ros.ok:
                rc, out = self.ros.set_mode(mode)
                via = "in-process"
            else:
                rc, out = sh(["ros2", "service", "call", "/mavros/set_mode",
                              "mavros_msgs/srv/SetMode",
                              f"{{base_mode: 0, custom_mode: '{mode}'}}"],
                             timeout=20)
        elif action == "override":
            if self.ros and self.ros.ok:
                rc, out = self.ros.override()
                via = "in-process"
            else:
                rc, out = sh(["ros2", "topic", "pub", "--once",
                              "/fc_ros/override", "std_msgs/msg/Bool",
                              "{data: true}"], timeout=20)
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
        rec["via"] = via
        rec["done_mono_s"] = round(time.monotonic() - self.t0, 3)
        rec["output"] = out.strip()[-1500:]
        log(f"inject 발화 {action} via={via} rc={rc} "
            f"(+{rec['fired_mono_s']}s, 소요 "
            f"{rec['done_mono_s'] - rec['fired_mono_s']:.3f}s)")
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
            # on_log 은 짧은 구간(2~3초)을 노리는 트리거다 — 루프 지연을 줄인다.
            self.stop_evt.wait(0.1 if any("on_log" in s for _, s in pending)
                               else 0.3)
        for _, spec in pending:
            self.results.append(
                {"spec": spec, "action": spec.get("action"),
                 "fired_mono_s": None, "rc": None,
                 "output": "트리거 미성립 — 발화하지 않음"})


# ── 거리 상한 감시 ─────────────────────────────────────────────────────────

# `/mavros/local_position/pose` 의 position 블록만 뽑는다.
# (orientation 에도 x/y 가 있으므로 position: 앵커를 반드시 걸어야 한다.)
RE_LOCAL_POS = re.compile(
    r"position:\s*\n\s*x:\s*(?P<x>-?[\d.]+(?:[eE][-+]?\d+)?)\s*\n"
    r"\s*y:\s*(?P<y>-?[\d.]+(?:[eE][-+]?\d+)?)\s*\n"
    r"\s*z:\s*(?P<z>-?[\d.]+(?:[eE][-+]?\d+)?)")


class RangeGuard(threading.Thread):
    """이륙지점 기준 수평거리가 상한을 넘으면 런을 즉시 끝내게 하는 **실행 감시**.

    도입 근거 — C10_pxvehicle 실측: 장애주입이 하나도 없는데 노드가 ENTRY 에서
    무한대기하는 동안 기체가 WP0 반대방향으로 **5.85km** 이탈했고, 하니스는
    `timeout_s`(480s) 를 다 채울 때까지 붙잡고 있으면서 ulog 만 키웠다.
    스트림이 살아 있으면 PX4 의 offboard-loss 페일세이프가 안 걸린다는 것도
    같은 런에서 확인됐다 — 즉 **하니스 밖에는 아무 제동장치가 없다.**

    기준점은 `/mavros/local_position/pose` 의 로컬 원점(= EKF 원점 ≈ 이륙지점)이다.
    수평거리만 본다(고도는 무관).

    ⚠️ 이것은 **판정이 아니다.** PASS/FAIL 을 결정하지 않고 verdict 에도 관여하지
    않는다. 오직 "언제 손을 떼고 런을 끝낼 것인가"만 정한다. 임계값은 시나리오
    최장 경로(B1 500m)의 3배로, 정상 런에서는 절대 성립하지 않는다.
    """

    def __init__(self, limit_m: float, stop_evt: threading.Event,
                 poll_s: float = 4.0):
        super().__init__(daemon=True)
        self.limit_m = float(limit_m)
        self.stop_evt = stop_evt
        self.poll_s = poll_s
        self.breached = threading.Event()
        self.samples = 0
        self.max_horiz_m: float | None = None
        self.max_at: tuple[float, float, float] | None = None
        self.breach: dict | None = None

    def run(self) -> None:
        if self.limit_m <= 0:
            return
        while not self.stop_evt.is_set():
            rc, out = sh(["ros2", "topic", "echo", "--once",
                          "/mavros/local_position/pose"], timeout=8)
            if rc == 0:
                m = RE_LOCAL_POS.search(out)
                if m:
                    x, y, z = (float(m.group("x")), float(m.group("y")),
                               float(m.group("z")))
                    d = (x * x + y * y) ** 0.5
                    self.samples += 1
                    if self.max_horiz_m is None or d > self.max_horiz_m:
                        self.max_horiz_m = d
                        self.max_at = (x, y, z)
                    if d > self.limit_m and not self.breached.is_set():
                        self.breach = {
                            "horiz_m": round(d, 1),
                            "limit_m": self.limit_m,
                            "local_enu": [round(x, 1), round(y, 1), round(z, 1)],
                            "utc": datetime.now(timezone.utc).isoformat(),
                        }
                        log(f"⚠️ 거리 상한 초과: 이륙지점에서 {d:.0f}m "
                            f"(상한 {self.limit_m:.0f}m) — 런 강제 종료")
                        self.breached.set()
                        return
            self.stop_evt.wait(self.poll_s)

    def report(self) -> dict:
        return {
            "enabled": self.limit_m > 0,
            "limit_m": self.limit_m,
            "samples": self.samples,
            "max_horiz_m": (round(self.max_horiz_m, 1)
                            if self.max_horiz_m is not None else None),
            "max_at_local_enu": ([round(v, 1) for v in self.max_at]
                                 if self.max_at else None),
            "breach": self.breach,
        }


# ── 실행 감시 ──────────────────────────────────────────────────────────────

class Monitor(threading.Thread):
    """node.log 를 증분으로 읽어 상태 전이/경고를 수집한다."""

    def __init__(self, path: Path, stop_evt: threading.Event,
                 watch_patterns: list[str] | None = None):
        super().__init__(daemon=True)
        self.path = path
        self.stop_evt = stop_evt
        # Injector 의 on_log 트리거용 — {정규식 원문: 최초 매치 monotonic}
        self.watch = [(p, re.compile(p)) for p in (watch_patterns or [])]
        self.log_hits: dict[str, float] = {}
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
        for raw, pat in self.watch:
            if raw not in self.log_hits and pat.search(msg):
                self.log_hits[raw] = time.monotonic()
                log(f"on_log 트리거 성립: {raw!r}")
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
    ap.add_argument("--range-limit-m", type=float, default=1500.0,
                    help="이륙지점 기준 수평거리 상한(m). 초과하면 즉시 종료하고 "
                         "exit 6 / reason=range_exceeded 로 기록한다. 0 이하면 비활성. "
                         "판정이 아니라 실행 감시다 (C10 의 5.85km 이탈 실측에서 도입)")
    ap.add_argument("--keep-running", action="store_true",
                    help="종료 후 PX4/MAVROS 를 죽이지 않는다 (디버그용)")
    ap.add_argument("--launch-arg", action="append", default=[], metavar="KEY=VALUE",
                    help="phase2.launch.py 에 넘길 추가 launch 인자 (반복 지정 가능). "
                         "시나리오의 launch_args 를 덮어쓴다. 테스트용 임시 파라미터는 "
                         "fc_ros_params.yaml 을 고치지 말고 이 옵션으로만 줄 것 "
                         "(예: --launch-arg range_limit_m=1200.0). "
                         "적용된 값은 meta.json 의 launch_args/launch_argv 에 그대로 남는다")
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

    # 시나리오 정의 + CLI 추가분. CLI 가 나중에 오므로 같은 키는 CLI 가 이긴다.
    effective_launch_args = dict(sc.get("launch_args") or {})
    for kv in args.launch_arg:
        if "=" not in kv:
            log(f"⚠️ --launch-arg 형식 오류(무시): {kv!r} — KEY=VALUE 여야 한다")
            continue
        k, v = kv.split("=", 1)
        effective_launch_args[k.strip()] = v

    launch_args = [f"{k}:={v}" for k, v in effective_launch_args.items()]
    meta = {
        "scenario_id": sc["id"],
        "desc": sc.get("desc", ""),
        "model": sc.get("model", "gz_standard_vtol"),
        "launch_args": effective_launch_args,
        "launch_args_scenario": sc.get("launch_args") or {},
        "launch_args_cli": list(args.launch_arg),
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
        # ⚠️ px4_head 만으로는 부족하다 — F-17/F-4 패치처럼 **커밋하지 않고
        #    워킹트리에만 얹는** 변경이 있으면 커밋 해시는 순정과 똑같다.
        #    (펌웨어 쪽도 같은 문제다: PX4 의 버전 헤더 생성은 `--dirty` 를
        #     붙이지 않아 `git_identity` 로 패치 여부를 구별할 수 없다.
        #     `docs/px4_v6c_patch_build.md` §4-2)
        #    그래서 워킹트리 상태와 diff 해시를 함께 남긴다.
        "px4_dirty": bool(sh(["git", "-C", str(PX4_DIR),
                              "status", "--porcelain"])[1].strip()),
        "px4_diff_sha256": hashlib.sha256(
            sh(["git", "-C", str(PX4_DIR), "diff"])[1].encode()).hexdigest()[:16],
        "px4_bin": str(PX4_BIN),
        "px4_bin_mtime_utc": (
            datetime.fromtimestamp(PX4_BIN.stat().st_mtime, timezone.utc)
            .isoformat() if PX4_BIN.exists() else None),
        # gz 월드 — C4(바람)처럼 월드를 바꿔 끼우는 런의 provenance.
        # 비어 있으면 PX4 기본(`default`).
        "px4_gz_world": os.environ.get("PX4_GZ_WORLD"),
        "px4_gz_worlds_dir": os.environ.get("PX4_GZ_WORLDS"),
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

        watch = [str(s["on_log"]) for s in meta["inject_spec"] if "on_log" in s]
        mon = Monitor(node_log_path, stop_evt, watch_patterns=watch)
        mon.start()

        # 거리 상한 감시 — 브링업이 끝난 뒤부터 런 종료까지 계속 돈다.
        guard = RangeGuard(args.range_limit_m, stop_evt)
        if guard.limit_m > 0:
            guard.start()
            log(f"거리 상한 감시 시작: {guard.limit_m:.0f}m (이륙지점 기준 수평)")

        inj = None
        ros_inj = None
        if meta["inject_spec"]:
            # 발화 전에 디스커버리를 끝내둔다 (CLI 4.04s → 수십 ms).
            ros_inj = RosInjectClient()
            meta["inject_transport"] = "in-process" if ros_inj.ok else "cli"
            if ros_inj.ok:
                ready = ros_inj.wait_ready(60.0)
                meta["inject_setmode_service_ready"] = ready
                log(f"inject: /mavros/set_mode 발견={ready}")
            inj = Injector(meta["inject_spec"], launch_t0,
                           mon.state_events, stop_evt, outdir,
                           log_hits=mon.log_hits, ros=ros_inj)
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
            # 거리 상한은 DONE 판정보다 먼저 본다 — 이탈 중이면 그 사실이 결론이다.
            if guard.breached.is_set():
                reason, code = "range_exceeded", 6
                break
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
        meta["range_guard"] = guard.report()
        exit_code = code

        stop_evt.set()
        if inj:
            inj.join(timeout=10.0)
            meta["inject_results"] = inj.results
        if ros_inj:
            ros_inj.close()

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
