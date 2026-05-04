"""
Phase 2 실행 스크립트: 경로 생성 → Offboard 실시간 속도 세트포인트 제어.

사용법:
  cd fc_bridge
  python run_phase2.py [--planner eta3|diterpin] \
                       [--entry pre_takeoff|mid_flight] \
                       [--conn udp:127.0.0.1:14550]

시나리오별 준비:
  pre_takeoff: 기체 arm 후 즉시 실행. arm 상태 감지 후 Offboard 전환.
  mid_flight : 기체가 이미 비행 중일 때 실행. 스트림 시작 후 Offboard 전환,
               WP0 진입 기동 후 경로 추종.

확인:
  SITL: MAVProxy 또는 QGC에서 비행 궤적 vs 계획 경로 비교.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

import fc_bridge.config as cfg
from fc_bridge.comm.mavlink_conn import MavlinkConn
from fc_bridge.comm.telemetry import Telemetry
from fc_bridge.planning.planner_runner import run_planner
from fc_bridge.planning.speed_profile import compute_speed_profile
from fc_bridge.guidance.l1_guidance import L1Guidance
from fc_bridge.execution.offboard_follower import OffboardFollower


DEFAULT_WAYPOINTS = np.array([
    [0.0,    0.0,   50.0],
    [200.0,  0.0,   50.0],
    [200.0, 200.0,  50.0],
    [0.0,   200.0,  50.0],
])


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Offboard 경로 추종")
    parser.add_argument("--planner", choices=["eta3", "diterpin"],
                        default="eta3")
    parser.add_argument("--entry", choices=["pre_takeoff", "mid_flight"],
                        default=cfg.OFFBOARD_ENTRY_MODE,
                        help="Offboard 진입 시점")
    parser.add_argument("--conn", default=cfg.CONNECTION_STR)
    parser.add_argument("--duration", type=float, default=300.0,
                        help="최대 실행 시간 (s)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # 1. 경로 생성 (arm 전에 미리)
    print(f"[Phase2] {args.planner} 플래너로 경로 생성 중...")
    path = run_planner(
        planner_name=args.planner,
        waypoints_ned=DEFAULT_WAYPOINTS,
        vehicle_params=cfg.VEHICLE_PARAMS,
    )
    print(f"[Phase2] 경로 생성 완료: {len(path.points)}점, {path.total_length:.1f}m")

    # 경로 데이터 추출
    pts_arr = np.array([p.pos[:2] for p in path.points])   # (N, 2) [N, E]
    kappa_arr = np.array([p.curvature for p in path.points])
    gamma_arr = np.array([p.gamma_ref for p in path.points])
    g = cfg.VEHICLE_PARAMS.get("gravity", 9.81)
    a_max = cfg.VEHICLE_PARAMS["a_max_g"] * g

    # 곡률 기반 속도 프로필
    v_arr = compute_speed_profile(
        kappa_arr,
        v_cruise=cfg.VEHICLE_PARAMS["v_cruise"],
        a_max=a_max,
    )

    # 2. 연결 & 텔레메트리
    print(f"[Phase2] 연결 중: {args.conn}")
    conn = MavlinkConn(connection_str=args.conn, baud=cfg.BAUD)
    conn.connect(timeout=15.0)
    print("[Phase2] 연결 완료")

    tel = Telemetry(conn, poll_timeout=0.02)
    tel.start()

    # 3. arm 대기 (pre_takeoff 시나리오)
    if args.entry == "pre_takeoff":
        print("[Phase2] ARM 대기 중... (QGC 또는 RC로 arm 하세요)")
        while not tel.is_armed:
            time.sleep(0.5)
        print("[Phase2] ARM 감지!")

    # 4. Offboard 추종 실행
    follower = OffboardFollower(
        conn=conn,
        telemetry=tel,
        path_pts=pts_arr,
        v_profile=v_arr,
        gamma_profile=gamma_arr,
        entry_mode=args.entry,
        control_hz=cfg.CONTROL_HZ,
        l1_dist=cfg.L1_DISTANCE,
        wp0_entry_radius=cfg.WP0_ENTRY_RADIUS,
        wp0_heading_tol=cfg.WP0_HEADING_TOL,
        a_max=a_max,
        verbose=args.verbose,
    )

    print(f"[Phase2] Offboard 시작 (entry={args.entry})")
    try:
        follower.run(duration_s=args.duration)
    except KeyboardInterrupt:
        print("\n[Phase2] 사용자 중단 (Ctrl+C)")

    print(f"[Phase2] 종료 상태: {follower.state}")
    tel.stop()
    conn.close()


if __name__ == "__main__":
    main()
