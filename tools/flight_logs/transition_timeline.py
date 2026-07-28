#!/usr/bin/env python3
"""VTOL 천이 구간 타임라인 — rosbag2 db3 하나에서 pose/vel/setpoint/vtol_state/mode 를 한 표로.

`bagdump.py` 와 마찬가지로 ROS 설치가 필요 없다(CDR 직접 파싱) — 개발컴에서 바로 돈다.
2026-07-28 flight02(첫 실기체 MC→FW 천이 시도) 분석에 쓴 것을 정식 편입한 것.

사용:
    python3 tools/flight_logs/transition_timeline.py logs/<폴더>/rosbag/rosbag_0.db3 55 75
    python3 tools/flight_logs/transition_timeline.py <db3> 57.8 62.2 --step 0.2 --t0 1785234421.4

`--t0` 를 주지 않으면 첫 pose 메시지 시각을 0 으로 잡는다. 표기는 NED(N,E,U) 로 통일한다
(mavros local_position 은 ENU 로 오므로 여기서 뒤집는다) — launch.log 의 telemetry_node
출력과 같은 기준이라 그대로 대조된다.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bagdump import read, p_pose, p_twist, p_ext, p_state  # noqa: E402


def _at(series, t):
    """t 이하의 마지막 샘플 (zero-order hold).

    시각은 `r[1]`(메시지 헤더 stamp)로 본다 — `r[0]`(bag 기록시각)은 수신 지연이 얹혀
    있어 토픽마다 어긋난다.
    """
    best = None
    for r in series:
        if r[1] <= t + 1e-6:
            best = r
        else:
            break
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="rosbag_0.db3 경로")
    ap.add_argument("lo", type=float, help="시작 t_rel (s)")
    ap.add_argument("hi", type=float, help="끝 t_rel (s)")
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--t0", type=float, default=None,
                    help="t_rel=0 에 해당하는 절대시각. 생략하면 첫 pose 시각")
    args = ap.parse_args(argv)

    pose = read(args.bag, '/mavros/local_position/pose', p_pose)
    vel = read(args.bag, '/mavros/local_position/velocity_local', p_twist)
    ext = read(args.bag, '/mavros/extended_state', p_ext)
    sp_pos = read(args.bag, '/mavros/setpoint_position/local', p_pose)
    sp_vel = read(args.bag, '/mavros/setpoint_velocity/cmd_vel', p_twist)
    state = read(args.bag, '/mavros/state', p_state)

    if not pose:
        print("pose 토픽이 비어 있다 — 다른 bag 인지 확인할 것", file=sys.stderr)
        return 1
    t0 = args.t0 if args.t0 is not None else pose[0][1]

    def mode_at(t):
        m = 'n/a'
        for r in state:
            if r[1] <= t:
                m = r[3]   # (bag_ts, msg_ts, armed, mode)
        return m

    print(f"{'t_rel':>7} {'mode':>10} {'vtol':>4} | {'N':>7} {'E':>7} {'U':>6} {'yawNED':>7} | "
          f"{'vN':>6} {'vE':>6} {'vU':>6} {'course':>6} | {'spN':>7} {'spE':>7} {'spU':>6} | "
          f"{'cmd vx,vy,vz,wz':>22}")

    t = args.lo
    while t <= args.hi + 1e-9:
        ts = t0 + t
        p, v, e = _at(pose, ts), _at(vel, ts), _at(ext, ts)
        sp, sv = _at(sp_pos, ts), _at(sp_vel, ts)
        if not p:
            t += args.step
            continue
        # bagdump 의 pose/twist 는 (bag_ts, msg_ts, x, y, z, yaw_deg) 형태 (ENU)
        E, N, U, yaw_enu = p[2], p[3], p[4], p[5]
        yaw_ned = (90.0 - yaw_enu + 180) % 360 - 180
        vE, vN, vU = (v[2], v[3], v[4]) if v else (float('nan'),) * 3
        vh = math.hypot(vE, vN)
        chi = math.degrees(math.atan2(vE, vN)) if vh > 0.3 else float('nan')
        spN = spE = spU = float('nan')
        if sp:
            spE, spN, spU = sp[2], sp[3], sp[4]
        cmd = f"{sv[2]:.2f},{sv[3]:.2f},{sv[4]:.2f},{sv[7]:.2f}" if sv else "-"
        print(f"{t:7.2f} {mode_at(ts):>10} {int(e[2]) if e else -1:>4} | "
              f"{N:7.2f} {E:7.2f} {U:6.2f} {yaw_ned:7.1f} | "
              f"{vN:6.2f} {vE:6.2f} {vU:6.2f} {chi:6.1f} | "
              f"{spN:7.2f} {spE:7.2f} {spU:6.2f} | {cmd:>22}")
        t += args.step
    return 0


if __name__ == '__main__':
    sys.exit(main())
