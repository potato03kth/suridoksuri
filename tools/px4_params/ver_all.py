#!/usr/bin/env python3
"""플래시 전 펌웨어 baseline 채취 — `ver all` 만 실행하는 읽기 전용 조회.

MAVLink SERIAL_CONTROL(SHELL)로 문자열 "ver all\\n" 하나만 보낸다.
파라미터 쓰기·ARM·설정 변경 명령은 일절 보내지 않는다.
"""
import os
import sys
import termios
import time
import tty

from pymavlink.dialects.v20 import common as mavlink2

DEV = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
SERIAL_CONTROL_DEV_SHELL = 10
FLAG_REPLY, FLAG_RESPOND, FLAG_EXCLUSIVE, FLAG_BLOCKING, FLAG_MULTI = 1, 2, 4, 8, 16


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
    port = RawTTY(DEV)
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
    shell(b"ver all\n")

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

    port.close()
    print(out.decode("utf-8", "replace"))
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
