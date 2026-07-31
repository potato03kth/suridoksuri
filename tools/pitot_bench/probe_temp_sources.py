#!/usr/bin/env python3
"""MAVROS 가 재발행할 수 있는 온도/차압 필드가 각각 무엇인지 실측 대조.
  SCALED_PRESSURE.press_diff             -> /mavros/imu/diff_pressure  (Pa)
  SCALED_PRESSURE.temperature            -> /mavros/imu/temperature_baro
  SCALED_PRESSURE.temperature_press_diff -> (mavros 가 발행하지 않음)
  HIGHRES_IMU.temperature                -> /mavros/imu/temperature_imu
"""
import sys, time
from pymavlink import mavutil

dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
m = mavutil.mavlink_connection(dev, baud=57600, source_system=250, source_component=190)
if m.wait_heartbeat(timeout=15) is None:
    sys.exit("no heartbeat")
for mid in (29, 105):
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                            mid, 1e6 / 5.0, 0, 0, 0, 0, 0)
    time.sleep(0.15)

print(f"{'t':>5} | {'SP.press_diff Pa':>16} {'SP.temp C':>10} {'SP.temp_pdiff C':>16} | "
      f"{'HR.diff_pres Pa':>16} {'HR.temperature C':>17}")
sp = hr = None
t0 = time.time(); last = 0
while time.time() - t0 < secs:
    msg = m.recv_match(type=["SCALED_PRESSURE", "HIGHRES_IMU"], blocking=True, timeout=2)
    if msg is None:
        continue
    if msg.get_type() == "SCALED_PRESSURE":
        sp = msg
    else:
        hr = msg
    now = time.time() - t0
    if sp and hr and now - last >= 2.0:
        last = now
        print(f"{now:5.1f} | {sp.press_diff*100:16.5f} {sp.temperature/100:10.2f} "
              f"{sp.temperature_press_diff/100:16.2f} | {hr.diff_pressure*100:16.5f} "
              f"{hr.temperature:17.5f}")
