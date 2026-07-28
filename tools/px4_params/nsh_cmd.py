#!/usr/bin/env python3
"""임의 nsh 명령 실행 — MAVLink SERIAL_CONTROL(SHELL) 읽기 전용 조회 도구.

`ver_all.py` 의 셸 경로를 그대로 쓰되 명령을 인자로 받는다.
상태 변경 명령은 화이트리스트 밖이면 거부한다 (파라미터 쓰기·ARM·캘리브레이션 금지).

    python3 nsh_cmd.py /dev/ttyACM0 "rc_input status"
"""
import os
import re
import sys
import termios
import time
import tty

from pymavlink.dialects.v20 import common as mavlink2

SERIAL_CONTROL_DEV_SHELL = 10
FLAG_REPLY, FLAG_RESPOND, FLAG_EXCLUSIVE, FLAG_BLOCKING, FLAG_MULTI = 1, 2, 4, 8, 16

# 읽기 전용 명령만 허용한다. `param set`·`commander`·`*_calibration` 등은 전부 걸린다.
ALLOWED = re.compile(
    r"^(ver\s+\w+|"
    r"\w[\w-]*\s+status|"
    r"listener\s+\w+(\s+-n\s+\d+)?|"
    r"param\s+(show|get)\s+\S*|"
    r"dmesg|uptime|free|top\s+once|serial_test|"
    r"ls\s+\S*|cat\s+/proc/\S*)$"
)


class RawTTY:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
        tty.setraw(self.fd)
        os.set_blocking(self.fd, False)

    def write(self, buf):
        n = 0
        while n < len(buf):
            try:
                n += os.write(self.fd, buf[n:])
            except BlockingIOError:
                time.sleep(0.001)
        return n

    def read(self, n=4096):
        try:
            return os.read(self.fd, n)
        except BlockingIOError:
            return b""

    def close(self):
        os.close(self.fd)


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    cmds = sys.argv[2:] or ["ver all"]
    for c in cmds:
        if not ALLOWED.match(c.strip()):
            print(f"[!] REFUSED (읽기 전용 화이트리스트 밖): {c}")
            return 3

    port = RawTTY(dev)
    mav = mavlink2.MAVLink(port, srcSystem=250, srcComponent=190)
    mav.robust_parsing = True

    tgt = None
    t_end = time.time() + 12
    while time.time() < t_end and tgt is None:
        mav.heartbeat_send(6, 8, 0, 0, 0)
        time.sleep(0.2)
        for msg in mav.parse_buffer(port.read()) or []:
            if msg.get_type() == "HEARTBEAT" and msg.get_srcSystem() != 250:
                tgt = (msg.get_srcSystem(), msg.get_srcComponent())
                break
    if tgt is None:
        print("[!] FAIL: no heartbeat")
        return 2

    def shell(data=b""):
        buf = data.ljust(70, b"\x00")
        mav.serial_control_send(SERIAL_CONTROL_DEV_SHELL,
                                FLAG_RESPOND | FLAG_EXCLUSIVE | FLAG_MULTI,
                                0, 0, len(data), buf)

    shell(b"\n")
    time.sleep(0.3)

    for c in cmds:
        print(f"\n===== nsh> {c} =====")
        shell(c.encode() + b"\n")
        out = bytearray()
        t0 = time.time()
        last = time.time()
        while time.time() - t0 < 15:
            for msg in mav.parse_buffer(port.read()) or []:
                if msg.get_type() == "SERIAL_CONTROL" and msg.count:
                    out += bytes(msg.data[:msg.count])
                    last = time.time()
            if out and time.time() - last > 2.0:
                break
            shell()          # poll
            time.sleep(0.1)
        print(out.decode("utf-8", "replace"))

    port.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
