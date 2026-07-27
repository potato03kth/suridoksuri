#!/usr/bin/env python3
"""PX4 파라미터 전량 덤프 (읽기 전용, pyserial 불요).

/dev/ttyACM0 을 raw tty 로 직접 열고 pymavlink 의 MAVLink 파서만 사용한다.
보내는 메시지는 HEARTBEAT(GCS) / PARAM_REQUEST_LIST / PARAM_REQUEST_READ 뿐.
PARAM_SET·COMMAND_LONG(ARM) 은 일절 보내지 않는다.
"""
import json
import os
import struct
import sys
import termios
import time
import tty

from pymavlink.dialects.v20 import common as mavlink2

DEV = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/px4_params.json"

TYPE_NAMES = {1: "UINT8", 2: "INT8", 3: "UINT16", 4: "INT16", 5: "UINT32",
              6: "INT32", 7: "UINT64", 8: "INT64", 9: "REAL32", 10: "REAL64"}
INT_TYPES = {1, 2, 3, 4, 5, 6, 7, 8}


class RawTTY:
    """pymavlink MAVLink 가 요구하는 최소 file-like 래퍼."""

    def __init__(self, path):
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
        tty.setraw(self.fd)
        attrs = termios.tcgetattr(self.fd)
        attrs[4] = attrs[5] = termios.B57600  # USB CDC 는 무시하지만 안전하게
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
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


def bitcast_to_int(f):
    return struct.unpack("<i", struct.pack("<f", f))[0]


def main():
    print(f"[*] opening {DEV}", flush=True)
    port = RawTTY(DEV)
    mav = mavlink2.MAVLink(port, srcSystem=250, srcComponent=190)
    mav.robust_parsing = True

    def send_hb():
        mav.heartbeat_send(6, 8, 0, 0, 0)  # MAV_TYPE_GCS, AUTOPILOT_INVALID

    # --- heartbeat 대기 ---
    tgt_sys = tgt_comp = None
    t_end = time.time() + 15
    last_hb = 0
    while time.time() < t_end and tgt_sys is None:
        if time.time() - last_hb > 1:
            send_hb()
            last_hb = time.time()
        data = port.read()
        if not data:
            time.sleep(0.01)
            continue
        for msg in mav.parse_buffer(data) or []:
            if msg.get_type() == "HEARTBEAT" and msg.get_srcSystem() != 250:
                tgt_sys = msg.get_srcSystem()
                tgt_comp = msg.get_srcComponent()
                print(f"[*] heartbeat sys={tgt_sys} comp={tgt_comp} "
                      f"type={msg.type} autopilot={msg.autopilot}", flush=True)
                break
    if tgt_sys is None:
        print("[!] FAIL: no heartbeat from FC (연결 안 됨 / USB 뽑힘?)", flush=True)
        port.close()
        return 2

    params = {}
    total = None

    def ingest(msg):
        nonlocal total
        total = msg.param_count
        name = msg.param_id
        if isinstance(name, bytes):
            name = name.decode("ascii", "ignore")
        params[msg.param_index] = {
            "name": name.rstrip("\x00"),
            "index": msg.param_index,
            "type": msg.param_type,
            "type_name": TYPE_NAMES.get(msg.param_type, str(msg.param_type)),
            "raw_float": float(msg.param_value),
        }

    mav.param_request_list_send(tgt_sys, tgt_comp)
    t_last = time.time()
    t_hb = time.time()
    deadline = time.time() + 180
    while time.time() < deadline:
        if time.time() - t_hb > 1:
            send_hb()
            t_hb = time.time()
        data = port.read()
        if not data:
            if total is not None and len(params) >= total:
                break
            if time.time() - t_last > 8:
                break
            time.sleep(0.005)
            continue
        for msg in mav.parse_buffer(data) or []:
            if msg.get_type() == "PARAM_VALUE":
                ingest(msg)
                t_last = time.time()
        if total is not None and len(params) >= total:
            break
        if len(params) and len(params) % 250 < 3:
            print(f"    {len(params)}/{total}", flush=True)

    print(f"[*] first pass: {len(params)}/{total}", flush=True)

    if total:
        for attempt in range(5):
            missing = [i for i in range(total) if i not in params]
            if not missing:
                break
            print(f"[*] retry {attempt + 1}: {len(missing)} missing", flush=True)
            for idx in missing:
                mav.param_request_read_send(tgt_sys, tgt_comp, b"", idx)
                t0 = time.time()
                got = False
                while time.time() - t0 < 0.8 and not got:
                    data = port.read()
                    if not data:
                        time.sleep(0.003)
                        continue
                    for msg in mav.parse_buffer(data) or []:
                        if msg.get_type() == "PARAM_VALUE":
                            ingest(msg)
                            if msg.param_index == idx:
                                got = True

    out = {}
    for p in params.values():
        p["value"] = (bitcast_to_int(p["raw_float"]) if p["type"] in INT_TYPES
                      else p["raw_float"])
        out[p["name"]] = p

    result = {
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": DEV,
        "target_system": tgt_sys,
        "target_component": tgt_comp,
        "param_count_reported": total,
        "param_count_received": len(out),
        "params": out,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(f"[*] wrote {OUT}: {len(out)}/{total}", flush=True)
    for k in ("SYS_AUTOSTART", "FW_AIRSPD_TRIM", "FW_AIRSPD_MIN",
              "FW_AIRSPD_MAX", "MPC_THR_HOVER", "CAL_MAG0_ID"):
        if k in out:
            print(f"    SANITY {k} = {out[k]['value']} ({out[k]['type_name']})",
                  flush=True)
    port.close()
    return 0 if (total and len(out) >= total) else 1


if __name__ == "__main__":
    sys.exit(main())
