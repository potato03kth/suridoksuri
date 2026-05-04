"""
Phase 1 실행 스크립트: 경로 생성 → PX4 Mission 업로드.

사용법:
  cd fc_bridge
  python run_phase1.py [--planner eta3|diterpin] [--conn udp:127.0.0.1:14550]

확인:
  QGroundControl > Plan 탭에서 업로드된 경로 시각 확인.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Windows 콘솔 UTF-8 출력 보장
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# fc_bridge를 패키지로 인식하기 위해 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parents[1]))

import fc_bridge.config as cfg
from fc_bridge.comm.mavlink_conn import MavlinkConn
from fc_bridge.planning.planner_runner import run_planner
from fc_bridge.execution.mission_uploader import MissionUploader


# ── 기본 테스트 웨이포인트 (NED, m) ──────────────────────────
DEFAULT_WAYPOINTS = np.array([
    [0.0,    0.0,   50.0],   # WP0: 출발점 (고도 50m)
    [200.0,  0.0,   50.0],   # WP1
    [200.0, 200.0,  50.0],   # WP2
    [0.0,   200.0,  50.0],   # WP3
    [0.0,    0.0,   50.0],   # WP4: 귀환
])


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Mission Upload")
    parser.add_argument("--planner", choices=["eta3", "diterpin"],
                        default="eta3", help="경로 생성 알고리즘")
    parser.add_argument("--conn", default=cfg.CONNECTION_STR,
                        help="MAVLink 연결 문자열")
    parser.add_argument("--dry-run", action="store_true",
                        help="연결 없이 경로 생성만 확인")
    args = parser.parse_args()

    # 1. 경로 생성
    print(f"[Phase1] {args.planner} 플래너로 경로 생성 중...")
    path = run_planner(
        planner_name=args.planner,
        waypoints_ned=DEFAULT_WAYPOINTS,
        vehicle_params=cfg.VEHICLE_PARAMS,
    )
    print(f"[Phase1] 경로 생성 완료: {len(path.points)}점, "
          f"총 길이 {path.total_length:.1f}m, "
          f"계획 시간 {path.planning_time*1000:.1f}ms")

    if args.dry_run:
        print("[Phase1] --dry-run: 연결 생략, 종료.")
        return

    # 2. 연결
    print(f"[Phase1] 연결 중: {args.conn}")
    conn = MavlinkConn(connection_str=args.conn, baud=cfg.BAUD)
    conn.connect(timeout=15.0)
    print("[Phase1] 연결 완료")

    # 3. Mission 업로드
    uploader = MissionUploader(conn, timeout=5.0)
    print(f"[Phase1] {len(DEFAULT_WAYPOINTS)}개 WP 업로드 중...")
    ok = uploader.upload(
        waypoints_ned=DEFAULT_WAYPOINTS,
        accept_radius=2.0,
        speed=cfg.VEHICLE_PARAMS["v_cruise"],
    )
    if ok:
        print("[Phase1] ✓ Mission 업로드 성공. QGC에서 경로를 확인하세요.")
    else:
        print("[Phase1] ✗ Mission 업로드 실패.")
        sys.exit(1)

    conn.close()


if __name__ == "__main__":
    main()
