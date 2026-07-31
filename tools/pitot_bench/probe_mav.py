#!/usr/bin/env python3
"""읽기 전용 프로브: USB MAVLink 로 어떤 메시지에 raw 차압/온도가 실려오는지 확인."""
import sys, time, collections
from pymavlink import mavutil

dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0

m = mavutil.mavlink_connection(dev, baud=57600, source_system=250, source_component=190)
print("waiting heartbeat...", flush=True)
hb = m.wait_heartbeat(timeout=15)
if hb is None:
    print("FAIL no heartbeat"); sys.exit(2)
print(f"heartbeat from sys={m.target_system} comp={m.target_component}", flush=True)

# SCALED_PRESSURE(29), HIGHRES_IMU(105), VFR_HUD(74) 를 10Hz 로 요청 (런타임 명령, 파라미터 변경 아님)
for msgid, hz in ((29, 10.0), (105, 10.0), (74, 5.0)):
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                            msgid, 1e6 / hz, 0, 0, 0, 0, 0)
    time.sleep(0.15)

counts = collections.Counter()
samples = {}
t0 = time.time()
while time.time() - t0 < secs:
    msg = m.recv_match(blocking=True, timeout=1.0)
    if msg is None:
        continue
    t = msg.get_type()
    counts[t] += 1
    if t in ("SCALED_PRESSURE", "SCALED_PRESSURE2", "SCALED_PRESSURE3",
             "HIGHRES_IMU", "VFR_HUD") and counts[t] <= 2:
        samples.setdefault(t, []).append(msg.to_dict())

print("\n=== message counts ===")
for k, v in counts.most_common():
    print(f"{k:28s} {v}")
print("\n=== samples ===")
for k, v in samples.items():
    for d in v:
        print(f"{k}: {d}")
