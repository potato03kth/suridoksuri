#!/usr/bin/env python3
"""PX4 비행 로그(.ulg) 자동 회수 — MAVLink 로그 전송 프로토콜 (pymavlink).

비행 종료(launch/MAVROS 종료) 후에만 실행할 것 — 시리얼 링크를 MAVROS가
단독 점유하며, 비행 중 대용량 전송은 제어 링크를 오염시킨다.

사용 예:
    python3 pull_ulog.py --list                     # FC의 로그 목록만 출력
    python3 pull_ulog.py --out logs/2026-07-06_flight01/
    python3 pull_ulog.py --url /dev/ttyACM0 --baud 921600 --log-id 7 --out .

record_flight.sh 가 내부적으로 쓰는 헬퍼:
    python3 pull_ulog.py --next-dir --log-root logs/   # 다음 플라이트 폴더명 출력

구현 노트: 계획서에는 MAVLink FTP로 적혀 있으나, PX4 로그 회수의 표준 경로인
로그 전송 프로토콜(LOG_REQUEST_LIST/LOG_REQUEST_DATA — QGC가 쓰는 방식)로
구현했다. 같은 링크로 같은 파일을 받되 디렉터리 탐색이 필요 없어 더 견고하다.
"""

import argparse
import datetime
import os
import re
import sys
import time
from dataclasses import dataclass

DEFAULT_URL = "/dev/ttyACM0"
DEFAULT_BAUD = 921600

# LOG_DATA 메시지의 페이로드 크기 (MAVLink 규격 고정값)
LOG_DATA_CHUNK = 90

# 손실 링크(UDP SITL 등) 대비 다운로드 튜닝값. 무손실 serial(실기체 기본, /dev/ttyACM0)
# 에서는 첫 윈도우가 순서대로 채워져 라운드 1회로 완결된다 — serial 경로 성능·동작 불변.
REQUEST_WINDOW = 256 * 1024   # 라운드당 요청 바이트 상한 — PX4의 1회 버스트 크기를 제한
RECV_IDLE = 1.0               # s: 이 시간 동안 LOG_DATA 없으면 현재 버스트 종료로 판단
MAX_NO_PROGRESS = 6           # 연속 무진행 라운드 → 실패(무한 hang 방지)
DL_HARD_TIMEOUT = 1800.0      # s: 전체 하드 상한(백스톱)
UDP_RCVBUF = 4 * 1024 * 1024  # UDP 수신 버퍼 확대로 버스트 손실 완화(serial엔 무영향)


@dataclass(frozen=True)
class LogEntry:
    log_id: int
    time_utc: int  # epoch 초. GPS 락 없이 기록된 로그는 0
    size: int      # bytes


# ---------------------------------------------------------------------------
# 순수 함수 (pytest 대상 — test_flight_logs.py)
# ---------------------------------------------------------------------------

def next_flight_dirname(existing_names, date_str):
    """당일 플라이트 폴더명을 넘버링해 반환한다. 비행 1회 = 폴더 1개 규약.

    existing_names: 로그 루트에 이미 있는 항목 이름들 (파일/폴더 무관)
    date_str: "YYYY-MM-DD"
    반환: "YYYY-MM-DD_flightNN" (NN = 당일 최대 순번 + 1, 최소 01)
    """
    pat = re.compile(re.escape(date_str) + r"_flight(\d+)$")
    nums = [int(m.group(1)) for n in existing_names for m in [pat.match(n)] if m]
    nn = max(nums) + 1 if nums else 1
    return f"{date_str}_flight{nn:02d}"


def pick_latest_log(entries):
    """로그 목록에서 가장 최근 로그를 고른다.

    전 항목에 GPS 시각이 있으면 time_utc 기준(동률이면 log_id),
    시각 0인 항목이 섞여 있으면(GPS 락 없던 비행) log_id 기준으로 폴백 —
    PX4 로그 목록은 오래된 순으로 열거되므로 최대 id가 최신이다.
    """
    if not entries:
        return None
    if all(e.time_utc > 0 for e in entries):
        return max(entries, key=lambda e: (e.time_utc, e.log_id))
    return max(entries, key=lambda e: e.log_id)


def ulog_filename(entry):
    """다운로드 파일명: 시각이 있으면 log_<id>_<UTC>.ulg, 없으면 log_<id>.ulg"""
    if entry.time_utc > 0:
        ts = datetime.datetime.utcfromtimestamp(entry.time_utc)
        return f"log_{entry.log_id}_{ts.strftime('%Y-%m-%d-%H-%M-%S')}.ulg"
    return f"log_{entry.log_id}.ulg"


def merge_intervals(intervals):
    """[start, end) 구간들을 정렬·병합해 겹침/인접을 하나로 합친다."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for a, b in ordered[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def missing_ranges(received, size):
    """수신 구간(received) 기준 [0, size) 안에서 아직 못 받은 구간 목록."""
    gaps = []
    cur = 0
    for a, b in merge_intervals(received):
        if a > cur:
            gaps.append((cur, min(a, size)))
        cur = max(cur, b)
        if cur >= size:
            break
    if cur < size:
        gaps.append((cur, size))
    return gaps


def coverage(received, size):
    """[0, size)로 클램프한 실제 수신 바이트 총합."""
    total = 0
    for a, b in merge_intervals(received):
        lo, hi = max(0, a), min(size, b)
        if hi > lo:
            total += hi - lo
    return total


# ---------------------------------------------------------------------------
# MAVLink 통신 (pymavlink는 실제 사용 시에만 import — --next-dir 등은 불필요)
# ---------------------------------------------------------------------------

def connect(url, baud, timeout=10):
    from pymavlink import mavutil
    print(f"[pull_ulog] 연결 중: {url} (baud {baud})")
    m = mavutil.mavlink_connection(url, baud=baud)
    if m.wait_heartbeat(timeout=timeout) is None:
        raise TimeoutError(f"heartbeat 수신 실패 ({timeout}s) — FC 전원/케이블/MAVROS 종료 여부 확인")
    print(f"[pull_ulog] heartbeat OK (sys {m.target_system} comp {m.target_component})")
    return m


def _grow_udp_rcvbuf(m, nbytes=UDP_RCVBUF):
    """UDP 소켓 수신 버퍼를 키워 대용량 버스트의 손실을 줄인다.

    serial 등 비소켓 연결에는 아무 영향이 없다(무손실 링크는 애초에 손실이 없다).
    """
    try:
        import socket
        port = getattr(m, "port", None)
        if isinstance(port, socket.socket):
            port.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, nbytes)
    except Exception:
        pass


def request_log_list(m, timeout=10):
    m.mav.log_request_list_send(m.target_system, m.target_component, 0, 0xFFFF)
    entries = {}
    num_logs = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = m.recv_match(type="LOG_ENTRY", blocking=True, timeout=2)
        if msg is None:
            continue
        num_logs = msg.num_logs
        if num_logs == 0:
            return []
        entries[msg.id] = LogEntry(msg.id, msg.time_utc, msg.size)
        if len(entries) >= num_logs:
            break
        deadline = time.time() + timeout
    if num_logs is None:
        raise TimeoutError("LOG_ENTRY 응답 없음 — FC가 로그 전송 프로토콜에 응답하지 않는다")
    if len(entries) < num_logs:
        print(f"[pull_ulog] 경고: 로그 목록 불완전 ({len(entries)}/{num_logs}) — 수신분으로 진행")
    return [entries[k] for k in sorted(entries)]


def download_log(m, entry, out_path, window=REQUEST_WINDOW, idle=RECV_IDLE,
                 max_no_progress=MAX_NO_PROGRESS, hard_timeout=DL_HARD_TIMEOUT):
    """LOG_REQUEST_DATA로 entry 전체를 out_path에 저장한다.

    윈도우드 요청 + 누락 구간 재요청 방식. 무손실 serial(실기체 기본)에서는 첫
    윈도우가 순서대로 채워져 한 라운드에 끝나고, 손실 링크(UDP SITL)에서는 아직
    못 받은 구간만 반복 요청해 수렴한다 — 예전처럼 매 gap마다 남은 로그 전체를
    재요청하지 않는다(그게 fire-hose 재전송 → livelock의 원인이었다).

    각 청크는 msg.ofs 위치에 직접 기록(seek)하므로 순서가 뒤섞여 도착해도 안전하다.
    진행이 멈추거나(무진행 라운드 누적) 하드 상한을 넘으면 예외를 던진다(무한 hang
    방지). 불완전 종료 시에도 부분 파일을 지우고 예외 — 잘린 .ulg를 완결본으로
    오인하는 조용한 실패를 막는다.
    """
    size = entry.size
    _grow_udp_rcvbuf(m)
    t0 = time.time()
    received = []          # 실제로 기록한 [start, end) 구간(병합 상태 유지)
    stalls = 0
    last_report = 0
    with open(out_path, "wb") as f:
        if size:
            f.truncate(size)  # 미리 할당 — 못 받은 구간은 채워질 때까지 0
        while True:
            gaps = missing_ranges(received, size)
            if not gaps or time.time() - t0 > hard_timeout:
                break
            req_ofs, gap_end = gaps[0]
            req_len = min(gap_end - req_ofs, window)
            m.mav.log_request_data_send(
                m.target_system, m.target_component, entry.log_id, req_ofs, req_len)
            got = 0
            round_intervals = []
            while True:
                msg = m.recv_match(type="LOG_DATA", blocking=True, timeout=idle)
                if msg is None:
                    break  # 이번 버스트 종료(침묵)
                if msg.id != entry.log_id:
                    continue
                n = min(int(msg.count), LOG_DATA_CHUNK)
                if n <= 0 or msg.ofs >= size:
                    continue
                end = min(msg.ofs + n, size)
                f.seek(msg.ofs)
                f.write(bytes(msg.data[: end - msg.ofs]))
                round_intervals.append((msg.ofs, end))
                got += end - msg.ofs
                if got >= req_len:
                    break  # 요청 윈도우를 다 받음 — idle 대기 없이 다음 라운드
            before = coverage(received, size)
            if round_intervals:
                received = merge_intervals(received + round_intervals)
            after = coverage(received, size)
            if after > before:
                stalls = 0
                if after - last_report >= 256 * 1024 or after == size:
                    last_report = after
                    rate = after / max(time.time() - t0, 1e-6) / 1024
                    print(f"[pull_ulog] {after}/{size} bytes "
                          f"({100 * after // max(size, 1)}%, {rate:.1f} KB/s)")
            else:
                stalls += 1
                if stalls >= max_no_progress:
                    break
    m.mav.log_request_end_send(m.target_system, m.target_component)
    cov = coverage(received, size)
    elapsed = time.time() - t0
    if cov < size:
        try:
            os.remove(out_path)
        except OSError:
            pass
        raise TimeoutError(
            f"다운로드 미완결: {cov}/{size} bytes ({size - cov} 누락, {elapsed:.0f}s, "
            f"무진행 {stalls}회) — 링크 확인 후 재시도하거나 SD 카드 수동 회수")
    print(f"[pull_ulog] 완료: {out_path} ({cov} bytes, {elapsed:.0f}s, "
          f"{cov / max(elapsed, 1e-6) / 1024:.1f} KB/s)")


FALLBACK_MSG = """\
[pull_ulog] ── 폴백 안내 ─────────────────────────────────────
  MAVLink 다운로드에 실패했다. Pixhawk의 SD 카드를 수동 회수해서
  /fs/microsd/log/<날짜>/ 안의 최신 .ulg 파일을
  해당 플라이트 폴더(logs/YYYY-MM-DD_flightNN/)에 직접 복사하라.
────────────────────────────────────────────────────────────"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL,
                   help=f"MAVLink 연결 주소 (기본: {DEFAULT_URL} — RPi USB 직결)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--out", default=".", help="저장 폴더 (기본: 현재 폴더)")
    p.add_argument("--list", action="store_true", help="로그 목록만 출력")
    p.add_argument("--log-id", type=int, default=None,
                   help="받을 로그 id (기본: 최신 자동 선택)")
    # record_flight.sh 내부용 헬퍼 — MAVLink 불필요
    p.add_argument("--next-dir", action="store_true",
                   help="다음 플라이트 폴더명 출력 후 종료 (--log-root 필요)")
    p.add_argument("--log-root", default=None, help="--next-dir 용 로그 루트 경로")
    args = p.parse_args(argv)

    if args.next_dir:
        if not args.log_root:
            p.error("--next-dir 에는 --log-root 가 필요하다")
        existing = os.listdir(args.log_root) if os.path.isdir(args.log_root) else []
        print(next_flight_dirname(existing, datetime.date.today().isoformat()))
        return 0

    try:
        m = connect(args.url, args.baud)
        entries = request_log_list(m)
    except Exception as e:
        print(f"[pull_ulog] 실패: {e}", file=sys.stderr)
        print(FALLBACK_MSG, file=sys.stderr)
        return 1

    if not entries:
        print("[pull_ulog] FC에 로그가 없다 (SD 카드 미장착이거나 기록 전).", file=sys.stderr)
        print(FALLBACK_MSG, file=sys.stderr)
        return 1

    if args.list:
        print(f"{'id':>4}  {'UTC 시각':<20}  {'크기':>12}")
        for e in entries:
            ts = (datetime.datetime.utcfromtimestamp(e.time_utc).isoformat(" ")
                  if e.time_utc > 0 else "(GPS 시각 없음)")
            print(f"{e.log_id:>4}  {ts:<20}  {e.size:>12,}")
        return 0

    if args.log_id is not None:
        matched = [e for e in entries if e.log_id == args.log_id]
        if not matched:
            print(f"[pull_ulog] id {args.log_id} 로그 없음. --list로 확인하라.", file=sys.stderr)
            return 1
        target = matched[0]
    else:
        target = pick_latest_log(entries)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, ulog_filename(target))
    print(f"[pull_ulog] 대상: id {target.log_id}, {target.size:,} bytes → {out_path}")
    try:
        download_log(m, target, out_path)
    except Exception as e:
        print(f"[pull_ulog] 다운로드 실패: {e}", file=sys.stderr)
        print(FALLBACK_MSG, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
