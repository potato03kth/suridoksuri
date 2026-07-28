#!/usr/bin/env python3
"""신규 비행 로그를 RPi에서 개발컴으로 회수 + 커밋 — 매 비행 후 서브에이전트가 실행하는 표준 절차.

배경: "비행 → 로그수집 → 로그분석/문제인식 → 평가 → 비행" 사이클을 수십 번 반복할
예정인데, 로그수집은 ①정해진 원격(RPi)으로 가서 ②정해진 로컬 위치로 파일을 복사하고
③소소한 정리(notes.md 스켈레톤) 후 ④커밋하는, 판단이 필요 없는 기계적 작업이다.
그래서 이 스크립트 하나로 고정해 서브에이전트가 매번 그대로 실행하게 한다 —
"관찰/결론" 같은 해석은 하지 않고 빈칸으로 남긴다(사람 몫).

하는 일:
    1. 원격 logs/YYYY-MM-DD_* 폴더 중 로컬에 없는 것을 rsync (record_flight.sh가
       이미 만들어둔 rosbag+launch.log+ulog 세트 — flightNN류는 1회성으로 skip,
       manual/unlogged류 catch-all은 증분 병합)
    2. 원격에서 pull_ulog.py --list로 FC의 전체 ulog id를 확인해, 로컬에 아직
       하나도 없는 id(= record_flight.sh도 못 받았고 수동으로도 안 받은 것)를
       찾아 원격 catch-all 폴더(<오늘날짜>_manual/)로 받아온 뒤 다시 rsync
    3. 새로 생긴 폴더에 notes.md가 없으면 스켈레톤 생성(내용은 비워둠)
    4. analyze_flight.py를 새 폴더마다 실행 (pyulog 없으면 경고만 하고 건너뜀)
    5. git add + commit([mc-hw] 등 --tag) + push (기본 켜짐, --no-push로 끔)

**원격(RPi)의 git 상태/코드는 건드리지 않는다** — 로그 파일만 읽는다. RPi 코드
배포(git pull+colcon build)는 이 스크립트의 책임 밖, 별도의 의도적 절차다.

사용 예:
    python3 tools/flight_logs/collect_new_logs.py                # 기본 실행
    python3 tools/flight_logs/collect_new_logs.py --dry-run       # 뭘 할지만 확인
    python3 tools/flight_logs/collect_new_logs.py --no-push       # 로컬 커밋까지만
    python3 tools/flight_logs/collect_new_logs.py --tag '[vtol-hw]'
"""

import argparse
import datetime
import glob
import os
import re
import subprocess
import sys

DEFAULT_REMOTE = "suri@100.67.27.83"
DEFAULT_REMOTE_PATH = "~/drone_ws/src/suridoksuri"
DEFAULT_BRANCH = "dev--vision-computing-module"


# ---------------------------------------------------------------------------
# 순수 함수 (pytest 대상 — test_flight_logs.py)
# ---------------------------------------------------------------------------

def is_dated_dirname(name):
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}_", name))


def classify_dirname(name):
    """flight = record_flight.sh가 만든 1회성 폴더(로컬에 있으면 절대 건드리지 않음).
    catchall = manual/unlogged 등 하루 동안 계속 추가될 수 있는 폴더(증분 병합)."""
    return "flight" if re.match(r"^\d{4}-\d{2}-\d{2}_flight\d+$", name) else "catchall"


def plan_dir_sync(remote_dirs, local_dirs):
    """회수할 원격 폴더 계획. flight류는 로컬에 없을 때만 1회 전체복사(full),
    catchall류는 항상 증분 병합(merge, rsync -av라 로컬 전용 파일은 안 지움).

    반환: [(dirname, mode)] — mode in {"full", "merge"}, dirname 오름차순.
    """
    plan = []
    for d in sorted(remote_dirs):
        if not is_dated_dirname(d):
            continue
        if classify_dirname(d) == "flight":
            if d not in local_dirs:
                plan.append((d, "full"))
        else:
            plan.append((d, "merge"))
    return plan


def parse_ulog_list(list_output):
    """pull_ulog.py --list 표준출력 파싱 → [(id, utc_str_or_None, size_bytes), ...]"""
    entries = []
    row = re.compile(r"^\s*(\d+)\s+(\S.*\S|\(GPS 시각 없음\))\s+([\d,]+)\s*$")
    for line in list_output.splitlines():
        m = row.match(line)
        if not m:
            continue
        log_id = int(m.group(1))
        ts = m.group(2)
        size = int(m.group(3).replace(",", ""))
        entries.append((log_id, None if "GPS" in ts else ts, size))
    return entries


def collected_ulog_ids(local_logs_root):
    """logs/**/log_<id>[_...].ulg 파일명에서 이미 로컬에 있는 id 집합을 구한다."""
    ids = set()
    pat = re.compile(r"^log_(\d+)(?:_|\.ulg$)")
    for p in glob.glob(os.path.join(local_logs_root, "*", "log_*.ulg")):
        m = pat.match(os.path.basename(p))
        if m:
            ids.add(int(m.group(1)))
    return ids


def catchall_dirname(date_str):
    return f"{date_str}_manual"


def notes_skeleton(dirname, extra_note=""):
    lines = [f"# {dirname}", ""]
    lines.append(f"- **비행 조건:** {extra_note or '(자동수집 — 직접 채울 것)'}")
    lines.append("- **관찰:**")
    lines.append("- **결론:**")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# I/O — ssh/rsync/git 실행
# ---------------------------------------------------------------------------

def log(msg):
    print(f"[collect_new_logs] {msg}")


class _TimedOut:
    """subprocess.run이 하드 타임아웃으로 죽었을 때 쓰는 가짜 CompletedProcess.

    네트워크 순간끊김(Tailscale/WiFi) 하나로 전체 스크립트가 uncaught 예외로
    죽는 것을 막는다 — 이 경우 호출부는 그냥 '이번 요청 실패'로 취급하면 된다.
    """
    returncode = 124
    stdout = ""

    def __init__(self, cmd):
        self.stderr = f"타임아웃: {' '.join(cmd)}"


def run(cmd, timeout=None, **kw):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    except subprocess.TimeoutExpired:
        return _TimedOut(cmd)


def ssh_run(remote, remote_cmd, timeout=30):
    # subprocess 레벨 타임아웃은 ssh 자체 ConnectTimeout보다 여유를 둬서, 가능하면
    # ssh가 먼저 정상적으로(비정상 종료코드로) 실패하게 하고 하드킬은 최후수단으로만 쓴다.
    connect_timeout = min(timeout, 15)
    return run(["ssh", "-o", f"ConnectTimeout={connect_timeout}", "-o", "BatchMode=yes",
                remote, remote_cmd], timeout=timeout + 10)


def rsync_pull(remote, remote_dir, local_dir, dry_run=False):
    os.makedirs(local_dir, exist_ok=True)
    args = ["rsync", "-a"] + (["--dry-run", "-v"] if dry_run else [])
    args += [f"{remote}:{remote_dir}/", f"{local_dir}/"]
    r = run(args, timeout=300)
    if r.returncode != 0:
        log(f"경고: rsync 실패 ({remote_dir}): {r.stderr.strip()}")
        return False
    return True


def git(repo_root, *args):
    return run(["git", "-C", repo_root, *args], timeout=60)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--remote", default=os.environ.get("RPI_REMOTE", DEFAULT_REMOTE))
    p.add_argument("--remote-path", default=os.environ.get("RPI_REMOTE_PATH", DEFAULT_REMOTE_PATH))
    p.add_argument("--branch", default=DEFAULT_BRANCH, help="push 대상 브랜치")
    p.add_argument("--tag", default="[mc-hw]", help="커밋 메시지 태그 (기본: [mc-hw])")
    p.add_argument("--dry-run", action="store_true", help="아무것도 쓰지 않고 계획만 출력")
    p.add_argument("--no-push", action="store_true", help="로컬 커밋까지만, push 생략")
    p.add_argument("--no-analyze", action="store_true", help="analyze_flight.py 실행 생략")
    args = p.parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    local_logs = os.path.join(repo_root, "logs")
    os.makedirs(local_logs, exist_ok=True)

    # -- 연결 확인 --------------------------------------------------------------
    r = ssh_run(args.remote, "echo ok", timeout=10)
    if r.returncode != 0 or "ok" not in r.stdout:
        log(f"SSH 접속 실패: {args.remote} — {r.stderr.strip()}")
        return 1
    log(f"연결 OK: {args.remote}")

    # -- 로컬 최신화(다른 세션이 먼저 push했을 수 있음) — fast-forward만, 코드 배포 아님 --
    if not args.dry_run:
        fr = git(repo_root, "fetch", "origin", args.branch)
        if fr.returncode == 0:
            mr = git(repo_root, "merge", "--ff-only", f"origin/{args.branch}")
            if mr.returncode != 0:
                log(f"경고: origin/{args.branch}로 fast-forward 불가 — 로컬이 갈라졌을 수 있음, 수동 확인 필요")
                log(mr.stderr.strip())
                return 1

    # -- §1 신규 플라이트 폴더 rsync --------------------------------------------
    # 디렉터리만 — logs/ 밑에는 review.md 같은 일반 파일도 있어서 ls -1로 훑으면
    # 그 파일 이름으로 로컬 디렉터리를 만들려다 FileExistsError로 죽는다.
    remote_ls = ssh_run(
        args.remote,
        f"find {args.remote_path}/logs -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' 2>/dev/null")
    remote_dirs = [d for d in remote_ls.stdout.splitlines() if d.strip()]
    local_dirs = [d for d in os.listdir(local_logs) if os.path.isdir(os.path.join(local_logs, d))] \
        if os.path.isdir(local_logs) else []

    plan = plan_dir_sync(remote_dirs, local_dirs)
    collected_dirs = set()
    for dirname, mode in plan:
        remote_dir = f"{args.remote_path}/logs/{dirname}"
        local_dir = os.path.join(local_logs, dirname)
        if args.dry_run:
            log(f"[dry-run] {mode}: {dirname}")
            continue
        log(f"{'전체복사' if mode == 'full' else '증분병합'}: {dirname}")
        if rsync_pull(args.remote, remote_dir, local_dir):
            collected_dirs.add(dirname)

    # -- §2 FC 미회수 ulog 스윕 --------------------------------------------------
    list_res = ssh_run(args.remote, f"python3 {args.remote_path}/tools/flight_logs/pull_ulog.py --list", timeout=20)
    if list_res.returncode != 0:
        log("경고: FC 로그목록 조회 실패(FC 미연결이거나 MAVROS/docker가 시리얼 점유 중일 수 있음) "
            "— 신규 폴더 회수만 수행하고 미회수 ulog 스윕은 건너뜀")
    else:
        fc_entries = parse_ulog_list(list_res.stdout)
        have = collected_ulog_ids(local_logs)
        missing = sorted(i for i, _, _ in fc_entries if i not in have)
        if not missing:
            log("FC에 미회수 ulog 없음")
        else:
            today = datetime.date.today().isoformat()
            catchall = catchall_dirname(today)
            remote_catchall = f"{args.remote_path}/logs/{catchall}"
            log(f"미회수 ulog {len(missing)}개 발견(id: {missing}) → {catchall}/ 로 회수")
            if args.dry_run:
                log(f"[dry-run] pull_ulog.py --log-id {{{','.join(map(str, missing))}}} --out {remote_catchall}/")
            else:
                ssh_run(args.remote, f"mkdir -p {remote_catchall}", timeout=10)
                for log_id in missing:
                    pr = ssh_run(
                        args.remote,
                        f"python3 {args.remote_path}/tools/flight_logs/pull_ulog.py "
                        f"--log-id {log_id} --out {remote_catchall}",
                        timeout=120,
                    )
                    if pr.returncode != 0:
                        log(f"경고: id {log_id} 회수 실패 — {pr.stderr.strip() or pr.stdout.strip()}")
                if rsync_pull(args.remote, remote_catchall, os.path.join(local_logs, catchall)):
                    collected_dirs.add(catchall)

    if not collected_dirs:
        log("신규 로그 없음 — 종료")
        return 0

    if args.dry_run:
        log(f"[dry-run] 여기까지 — 실제 회수는 --dry-run 없이 재실행")
        return 0

    # -- §3 notes.md 스켈레톤 (없을 때만) ----------------------------------------
    for dirname in sorted(collected_dirs):
        notes_path = os.path.join(local_logs, dirname, "notes.md")
        if not os.path.exists(notes_path):
            with open(notes_path, "w", encoding="utf-8") as f:
                f.write(notes_skeleton(dirname))
            log(f"notes.md 스켈레톤 생성: {dirname}")

    # -- §4 analyze_flight.py -----------------------------------------------------
    if not args.no_analyze:
        analyzer = os.path.join(script_dir, "analyze_flight.py")
        for dirname in sorted(collected_dirs):
            flight_dir = os.path.join(local_logs, dirname)
            if not glob.glob(os.path.join(flight_dir, "*.ulg")):
                continue
            ar = run([sys.executable, analyzer, flight_dir])
            if ar.returncode != 0:
                log(f"경고: analyze_flight.py 실패({dirname}) — {ar.stderr.strip()}")
            else:
                log(f"analysis_auto.md 생성: {dirname}")

    # -- §5 git commit + push ------------------------------------------------------
    for dirname in sorted(collected_dirs):
        git(repo_root, "add", os.path.join("logs", dirname))
    st = git(repo_root, "status", "--porcelain")
    if not st.stdout.strip():
        log("커밋할 변경 없음(이미 최신) — 종료")
        return 0

    summary = ", ".join(sorted(collected_dirs))
    msg = f"{args.tag} 비행로그 자동회수: {summary}\n\ncollect_new_logs.py 로 회수 — 관찰/결론은 미기록(다음 세션 채울 것).\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
    cr = git(repo_root, "commit", "-m", msg)
    if cr.returncode != 0:
        log(f"커밋 실패: {cr.stderr.strip()}")
        return 1
    log(f"커밋 완료: {summary}")

    if not args.no_push:
        pr = git(repo_root, "push", "origin", f"HEAD:{args.branch}")
        if pr.returncode != 0:
            log(f"push 실패 — 로컬 커밋은 남아있음: {pr.stderr.strip()}")
            return 1
        log(f"push 완료 → origin/{args.branch}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
