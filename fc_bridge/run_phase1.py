"""
Phase 1 실행 스크립트: 경로 생성 → PX4 Mission 업로드.

사용법:
  cd fc_bridge
  python run_phase1.py [--planner eta3|diterpin] [--conn udp:127.0.0.1:14550]
  python run_phase1.py --dry-run --plot
  python run_phase1.py --dry-run --save-plot results/path.png

확인:
  QGroundControl > Plan 탭에서 업로드된 경로 시각 확인.
"""
from __future__ import annotations
import sys
from pathlib import Path

# fc_bridge를 패키지로 인식하기 위해 프로젝트 루트를 path에 추가
# the line beneath must be set on between of import sys;import path and other import part
# fmt: off
sys.path.insert(0, str(Path(__file__).parents[1]))

from fc_bridge.execution.mission_uploader import MissionUploader
from fc_bridge.planning.planner_runner import run_planner
from fc_bridge.comm.mavlink_conn import MavlinkConn
import fc_bridge.config as cfg
import argparse
import numpy as np


# Windows 콘솔 UTF-8 출력 보장
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── 기본 테스트 웨이포인트 (NED, m) ──────────────────────────
DEFAULT_WAYPOINTS = np.array([
    [0.0,    0.0,   50.0],   # WP0: 출발점 (고도 50m)
    [200.0,  0.0,   50.0],   # WP1
    [200.0, 200.0,  50.0],   # WP2
    [0.0,   200.0,  50.0],   # WP3
    [0.0,    0.0,   50.0],   # WP4: 귀환
])


def _plot_path(path, waypoints_ned: np.ndarray, planner_name: str,
               save_path: str | None = None) -> None:
    """
    생성된 경로를 6패널로 시각화.

    패널 구성
    ---------
    [0:2, 0:2]  Top-down 2D (|κ| 컬러맵, 이상 WP 직선 오버레이)
    [0,   2  ]  고도 vs arc length
    [1,   2  ]  곡률 (부호) vs arc length
    [2,   0  ]  위치잔차 (WP 직선 기준 CTE)
    [2,   1  ]  속도 프로필 (κ 기반 추정)
    [2,   2  ]  헤딩 χ_ref vs arc length
    """
    import matplotlib
    matplotlib.use("Agg" if save_path else "TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.collections import LineCollection
    from fc_bridge.planning.speed_profile import compute_speed_profile

    pts = path.points
    pos = np.array([p.pos for p in pts])   # (N, 3)
    kappa = np.array([p.curvature for p in pts])   # (N,)
    chi = np.array([p.chi_ref for p in pts])   # (N,)
    s_arr = np.array([p.s for p in pts])   # (N,)

    wps = waypoints_ned  # (M, 3)

    # WP 통과 인덱스 (wp_index 마킹된 점)
    wp_path_idx = [i for i, p in enumerate(pts) if p.wp_index is not None]

    # 속도 프로필
    v_cruise = float(cfg.VEHICLE_PARAMS["v_cruise"])
    a_max = float(cfg.VEHICLE_PARAMS["a_max_g"]) * \
        float(cfg.VEHICLE_PARAMS["gravity"])
    v_profile = compute_speed_profile(kappa, v_cruise=v_cruise, a_max=a_max)

    # 위치잔차(CTE): 각 경로점 → 가장 가까운 이상 WP 선분까지 수직 거리
    cte = np.zeros(len(pos))
    for i in range(len(pos)):
        p2 = pos[i, :2]
        best = np.inf
        for j in range(len(wps) - 1):
            a, b = wps[j, :2], wps[j + 1, :2]
            ab = b - a
            denom = np.dot(ab, ab)
            if denom < 1e-12:
                d = np.linalg.norm(p2 - a)
            else:
                t = np.clip(np.dot(p2 - a, ab) / denom, 0.0, 1.0)
                d = np.linalg.norm(p2 - (a + t * ab))
            best = min(best, d)
        cte[i] = best

    # ── 레이아웃 ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 18))
    fig.patch.set_facecolor("#0e1117")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.32)

    fig.suptitle(
        f"path visualization  |  planner: {planner_name}  |  {len(wps)} WPs  |  "
        f"tot {path.total_length:.1f} m  |  est.drive time {path.planning_time * 1000:.1f} ms  |  {len(pts)} points.",
        fontsize=13, color="white",
    )

    kappa_abs = np.abs(kappa)

    # ── dark 스타일 헬퍼 ────────────────────────────────────────────────
    def _dark(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor("#1a1d24")
        ax.set_title(title, color="white", fontsize=9)
        ax.set_xlabel(xlabel, color="white", fontsize=8)
        ax.set_ylabel(ylabel, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.grid(True, alpha=0.2, color="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")

    def _vlines(ax):
        for idx in wp_path_idx:
            ax.axvline(s_arr[idx], color="lime", lw=0.6, alpha=0.35)

    # ── 1. Top-down 2D ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.set_facecolor("#1a1d24")

    # 이상 WP 직선 (점선 오버레이)
    ax1.plot(wps[:, 1], wps[:, 0], "w--", lw=0.9,
             alpha=0.35, label="idial path(WP straight)")

    # 경로를 |κ|로 컬러링
    segs = np.stack(
        [np.c_[pos[:-1, 1], pos[:-1, 0]],
         np.c_[pos[1:,  1], pos[1:,  0]]],
        axis=1,
    )  # (N-1, 2, 2) in [E, N]
    lc = LineCollection(segs, cmap="plasma", linewidth=1.8, alpha=0.9)
    lc.set_array(kappa_abs[:-1])
    ax1.add_collection(lc)
    cb = plt.colorbar(lc, ax=ax1, fraction=0.025, pad=0.02)
    cb.set_label("|κ| (1/m)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # WP 마커
    ax1.scatter(wps[:, 1], wps[:, 0], c="cyan", s=130, zorder=6, marker="^")
    for i, wp in enumerate(wps):
        ax1.annotate(
            f"WP{i}", (wp[1], wp[0]), fontsize=9, color="cyan", fontweight="bold",
            xytext=(7, 7), textcoords="offset points",
        )

    # 경로 위 WP 통과점 강조 (lime 원)
    for idx in wp_path_idx:
        ax1.scatter(pos[idx, 1], pos[idx, 0], c="lime",
                    s=55, zorder=7, marker="o")

    margin = 80
    ax1.set_xlim(min(pos[:, 1].min(), wps[:, 1].min()) - margin,
                 max(pos[:, 1].max(), wps[:, 1].max()) + margin)
    ax1.set_ylim(min(pos[:, 0].min(), wps[:, 0].min()) - margin,
                 max(pos[:, 0].max(), wps[:, 0].max()) + margin)
    ax1.set_aspect("equal")
    ax1.set_xlabel("E (m)", color="white")
    ax1.set_ylabel("N (m)", color="white")
    ax1.set_title(
        "Top-down 2D  (col = |kappa|,  lime = WP thrupass point)", color="white")
    ax1.tick_params(colors="white")
    ax1.grid(True, alpha=0.2, color="gray")
    for spine in ax1.spines.values():
        spine.set_edgecolor("gray")
    ax1.legend(fontsize=8, facecolor="#2a2d34",
               labelcolor="white", loc="upper right")

    # ── 2. 고도 vs arc length ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(s_arr, pos[:, 2], color="#4fc3f7", lw=1.3)
    for idx in wp_path_idx:
        ax2.scatter(s_arr[idx], pos[idx, 2], color="cyan", s=45, zorder=5)
    _vlines(ax2)
    _dark(ax2, "altitude", "arc length (m)", "h (m)")

    # ── 3. 곡률 (부호) vs arc length ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(s_arr, kappa, color="#ef9a9a", lw=1.0, alpha=0.85)
    ax3.axhline(0, color="white", lw=0.5, alpha=0.4)
    R_min = v_cruise ** 2 / a_max
    kappa_lim = 1.0 / R_min
    ax3.axhline(kappa_lim, color="#ff6b6b", lw=0.9, ls="--", alpha=0.7,
                label=f"κ_max  (R_min={R_min:.0f} m)")
    ax3.axhline(-kappa_lim, color="#ff6b6b", lw=0.9, ls="--", alpha=0.7)
    _vlines(ax3)
    _dark(ax3, "curverture κ  (sign: R+ / L−)", "arc length (m)", "κ (1/m)")
    ax3.legend(fontsize=7, facecolor="#2a2d34", labelcolor="white")

    # ── 4. 위치잔차 (CTE) ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.fill_between(s_arr, cte, alpha=0.30, color="#80cbc4")
    ax4.plot(s_arr, cte, color="#4db6ac", lw=1.1)
    _vlines(ax4)

    # y축: 데이터 범위로 꽉 채워 미묘한 차이 강조
    cte_max = cte.max()
    ax4.set_ylim(0, cte_max * 1.18 if cte_max > 0 else 1.0)

    peak_idx = int(cte.argmax())
    ax4.annotate(
        f"max {cte_max:.2f} m",
        xy=(s_arr[peak_idx], cte_max),
        xytext=(12, -16), textcoords="offset points",
        color="#80cbc4", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#80cbc4", lw=0.8),
    )
    _dark(ax4, "pos residual  (CTE, when WP straight)",
          "arc length (m)", "CTE (m)")

    # ── 5. 속도 프로필 ───────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(s_arr, v_profile, color="#ce93d8",
             lw=1.3, label="v_profile (κ 기반)")
    ax5.axhline(v_cruise, color="white", lw=0.6, ls="--", alpha=0.5,
                label=f"v_cruise = {v_cruise} m/s")
    _vlines(ax5)
    _dark(ax5, "velocity profile (est. by curverture)",
          "arc length (m)", "v (m/s)")
    ax5.legend(fontsize=7, facecolor="#2a2d34", labelcolor="white")

    # ── 6. 헤딩 ─────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(s_arr, np.rad2deg(chi), color="#ffcc80", lw=1.1)
    _vlines(ax6)
    _dark(ax6, "heading χ_ref", "arc length (m)", "χ (deg)")

    # ── 출력 ─────────────────────────────────────────────────────────────
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Phase1] 경로 그래프 저장: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Mission Upload")
    parser.add_argument("--planner", choices=["eta3", "diterpin"],
                        default="eta3", help="경로 생성 알고리즘")
    parser.add_argument("--conn", default=cfg.CONNECTION_STR,
                        help="MAVLink 연결 문자열")
    parser.add_argument("--dry-run", action="store_true",
                        help="연결 없이 경로 생성만 확인")
    parser.add_argument("--plot", action="store_true",
                        help="경로 시각화 창 표시 (--dry-run과 함께 사용)")
    parser.add_argument("--save-plot", metavar="PATH",
                        help="경로 시각화를 파일로 저장 (예: results/path.png)")
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

    # 2. 시각화
    if args.plot or args.save_plot:
        _plot_path(path, DEFAULT_WAYPOINTS, args.planner,
                   save_path=args.save_plot)

    if args.dry_run:
        print("[Phase1] --dry-run: 연결 생략, 종료.")
        return

    # 3. 연결
    print(f"[Phase1] 연결 중: {args.conn}")
    conn = MavlinkConn(connection_str=args.conn, baud=cfg.BAUD)
    conn.connect(timeout=15.0)
    print("[Phase1] 연결 완료")

    # 4. Mission 업로드
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
