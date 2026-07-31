#!/usr/bin/env python3
"""차압(피토) 영점 온도드리프트 벤치 계측 로거 — 읽기 전용 / 파라미터 변경 없음 / ARM 없음.

데이터 출처 (전부 PX4 USB MAVLink, 실측 확인 완료 2026-07-31):
  SCALED_PRESSURE.press_diff             = differential_pressure.differential_pressure_pa
                                           ** 오프셋(SENS_DPRES_OFF) 적용 前 RAW ** (hPa)
  SCALED_PRESSURE.temperature_press_diff = MS4525DO 다이 온도 (cdegC, 분해능 0.0977 C)
  SCALED_PRESSURE.temperature            = 내장 기압계 온도 (cdegC) = FC 케이스 내부 온도
  HIGHRES_IMU.temperature                = (이 빌드에서) MS4525DO 다이 온도와 동일 — 교차검증용
  VFR_HUD.airspeed                       = PX4 가 최종 산출한 IAS (오프셋 적용 後) — 정합 확인용

이벤트 마커:
    mark.sh "냉방 ON 24도"      <- 다른 터미널에서 실행하면 같은 CSV 에 event 행이 들어간다
    (구현: 마커 파일에 append -> 로거가 새 줄을 읽어 CSV 에 기록)

사용:
    python3 dpres_log.py OUT.csv [--dev /dev/ttyACM0] [--hz 10] [--note "..."]
"""
import argparse
import os
import signal
import sys
import time
from datetime import datetime, timezone

from pymavlink import mavutil

RUNNING = True
COLS = ("unix_s,iso_local,iso_utc,boot_ms,dp_raw_pa,dp_corr_pa,dp_die_temp_c,"
        "hr_imu_temp_c,baro_temp_c,press_abs_hpa,airspeed_vfr_ms,event\n")


def _stop(signum, frame):
    global RUNNING
    RUNNING = False


def connect(dev, baud):
    m = mavutil.mavlink_connection(dev, baud=baud, source_system=250, source_component=190)
    if m.wait_heartbeat(timeout=20) is None:
        try:
            m.close()
        except Exception:
            pass
        raise TimeoutError("no heartbeat")
    return m


def request_streams(m, hz):
    """런타임 스트림 요청만 보낸다 (파라미터 쓰기 아님, 재부팅 시 사라짐)."""
    for msgid in (mavutil.mavlink.MAVLINK_MSG_ID_SCALED_PRESSURE,
                  mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU,
                  mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD):
        m.mav.command_long_send(m.target_system, m.target_component,
                                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                msgid, 1e6 / hz, 0, 0, 0, 0, 0)
        time.sleep(0.08)


def read_param(m, name, timeout=8.0):
    """PARAM_REQUEST_READ — 읽기만 한다."""
    for _ in range(3):
        m.mav.param_request_read_send(m.target_system, m.target_component,
                                      name.encode(), -1)
        t0 = time.time()
        while time.time() - t0 < timeout / 3:
            msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.0)
            if msg and msg.param_id.strip("\x00") == name:
                return msg.param_value
    return None


class MarkerTail:
    """마커 파일을 tail 한다. 이미 있던 내용은 건너뛴다."""

    def __init__(self, path):
        self.path = path
        open(path, "a").close()
        self.pos = os.path.getsize(path)

    def poll(self):
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return []
        if size < self.pos:          # 파일이 잘렸으면 처음부터
            self.pos = 0
        if size == self.pos:
            return []
        with open(self.path, "r", errors="replace") as fh:
            fh.seek(self.pos)
            data = fh.read()
            self.pos = fh.tell()
        return [ln.strip() for ln in data.splitlines() if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dev", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--note", default="")
    ap.add_argument("--marks", default="", help="마커 파일 (기본: <out>.marks)")
    ap.add_argument("--fsync-s", type=float, default=10.0)
    ap.add_argument("--duration", type=float, default=0.0, help="초, 0=무한")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)   # SSH 끊겨도 계속

    marks_path = args.marks or (args.out + ".marks")
    tail = MarkerTail(marks_path)

    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    f = open(args.out, "a", buffering=1)           # 라인 단위 flush

    m = None
    while RUNNING and m is None:
        try:
            m = connect(args.dev, args.baud)
        except Exception as e:
            print(f"[connect retry] {e}", file=sys.stderr, flush=True)
            time.sleep(2.0)
    if m is None:
        return 1

    dpres_off = read_param(m, "SENS_DPRES_OFF")
    request_streams(m, args.hz)

    if new_file:
        f.write("# suridoksuri 차압 영점 온도드리프트 벤치로그\n")
        f.write(f"# started_local={datetime.now().isoformat()}\n")
        f.write(f"# host={os.uname().nodename} dev={args.dev} hz={args.hz}\n")
        f.write(f"# SENS_DPRES_OFF_at_start={dpres_off}\n")
        f.write(f"# marks_file={marks_path}\n")
        f.write(f"# note={args.note}\n")
        f.write("# dp_raw_pa   = SCALED_PRESSURE.press_diff*100  (오프셋 적용 전 RAW)\n")
        f.write("# dp_corr_pa  = dp_raw_pa - SENS_DPRES_OFF      (PX4 airspeed_selector 와 동일 식)\n")
        f.write("# dp_die_temp_c = MS4525DO 다이 온도 / baro_temp_c = FC 내장 기압계 온도\n")
        f.write("# event 컬럼이 비어있지 않은 행 = 사용자 마커\n")
        f.write(COLS)

    NAN = float("nan")

    def row(now, sp, hr_t, hud, event=""):
        # ⚠️ temperature_press_diff 는 MAVLink **확장 필드**다. PX4 가 후행 0 바이트를
        #    잘라 보내면 pymavlink 메시지에 속성 자체가 없다 -> getattr 필수.
        #    (실측: v2 첫 기동에서 AttributeError 로 즉사했다.)
        #    없을 때는 HIGHRES_IMU.temperature 로 대체한다 — 이 빌드에서 두 값이
        #    동일함을 40초 대조로 확인했다.
        if sp is not None:
            dp_raw = sp.press_diff * 100.0
            dp_corr = (dp_raw - dpres_off) if dpres_off is not None else NAN
            tpd = getattr(sp, "temperature_press_diff", None)
            die_t = (tpd / 100.0) if tpd not in (None, 0) else hr_t
            baro_t = sp.temperature / 100.0
            pabs = sp.press_abs
            boot = sp.time_boot_ms
        else:
            dp_raw = dp_corr = die_t = baro_t = pabs = NAN
            boot = ""
        lt = datetime.fromtimestamp(now)
        ut = datetime.fromtimestamp(now, timezone.utc)
        f.write(f"{now:.3f},{lt.isoformat()},{ut.isoformat()},"
                f"{boot},{dp_raw:.5f},{dp_corr:.5f},{die_t:.5f},"
                f"{hr_t:.5f},{baro_t:.2f},{pabs:.4f},{hud:.4f},"
                f"\"{event}\"\n")

    n = 0
    sp = None
    hr_t = float("nan")
    hud = float("nan")
    t_start = time.time()
    last_req = last_sync = last_mark = time.time()
    while RUNNING:
        if args.duration and time.time() - t_start > args.duration:
            break
        try:
            msg = m.recv_match(type=["SCALED_PRESSURE", "HIGHRES_IMU", "VFR_HUD"],
                               blocking=True, timeout=3.0)
        except Exception as e:
            print(f"[recv err] {e}", file=sys.stderr, flush=True)
            msg = None

        now = time.time()

        # ── 마커 처리 (0.5 s 마다) ─────────────────────────────
        if now - last_mark > 0.5:
            last_mark = now
            for line in tail.poll():
                row(now, sp, hr_t, hud, event=line.replace('"', "'"))
                print(f"[MARK] {line}", flush=True)

        if msg is None:
            print("[no data 3s] 스트림 재요청", file=sys.stderr, flush=True)
            try:
                request_streams(m, args.hz)
            except Exception:
                try:
                    m.close()
                except Exception:
                    pass
                m = None
                while RUNNING and m is None:
                    try:
                        m = connect(args.dev, args.baud)
                        request_streams(m, args.hz)
                        print("[reconnected]", file=sys.stderr, flush=True)
                    except Exception as e:
                        print(f"[reconnect retry] {e}", file=sys.stderr, flush=True)
                        time.sleep(2.0)
            continue

        t = msg.get_type()
        if t == "VFR_HUD":
            hud = getattr(msg, "airspeed", hud)
            continue
        if t == "HIGHRES_IMU":
            hr_t = getattr(msg, "temperature", hr_t)
            continue

        # 한 건의 이상 메시지가 수 시간짜리 기록을 죽이면 안 된다.
        sp = msg
        try:
            row(now, sp, hr_t, hud)
        except Exception as e:
            print(f"[row skip] {e}", file=sys.stderr, flush=True)
            continue
        n += 1
        if now - last_req > 30.0:
            request_streams(m, args.hz)
            last_req = now
        if now - last_sync > args.fsync_s:
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
            last_sync = now
        if n % 300 == 0:
            print(f"[{n:7d}] {(now-t_start)/60:6.1f}min  "
                  f"dp_raw={sp.press_diff*100:+8.3f} Pa  "
                  f"die_T={sp.temperature_press_diff/100:6.2f} C  "
                  f"baro_T={sp.temperature/100:6.2f} C  aspd={hud:5.2f} m/s", flush=True)

    f.write(f"# stopped_local={datetime.now().isoformat()} samples={n}\n")
    try:
        os.fsync(f.fileno())
    except Exception:
        pass
    f.close()
    print(f"done, {n} samples -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
