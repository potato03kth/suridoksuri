"""
L1Guidance 단위 테스트.
"""
import numpy as np
import pytest
from fc_bridge.guidance.l1_guidance import (
    L1Guidance, cumulative_arclen, segment_search_range,
    nearest_segment_in_range,
)


def _straight_path(length=200.0, n=100, direction="north"):
    """북쪽 또는 동쪽 방향 직선 경로 생성."""
    pts = np.zeros((n, 2))
    s = np.linspace(0, length, n)
    if direction == "north":
        pts[:, 0] = s
    else:
        pts[:, 1] = s
    v = np.full(n, 15.0)
    return pts, v


def test_on_path_heading_aligned():
    """경로 위에 있고 속도 방향이 경로 방향과 일치할 때 chi_cmd ≈ 경로 헤딩."""
    pts, v = _straight_path(direction="north")   # 경로 헤딩 = 0 rad (북쪽)
    guide = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)

    pos = np.array([50.0, 0.0, 30.0])       # 경로 위
    vel = np.array([15.0, 0.0, 0.0])        # 북쪽으로 비행

    chi_cmd, v_cmd, cte = guide.compute(pos, vel)

    assert chi_cmd == pytest.approx(0.0, abs=0.15)   # 헤딩 ≈ 0 rad
    assert v_cmd == pytest.approx(15.0, rel=0.01)
    assert abs(cte) < 0.5


def test_lateral_offset_corrects_heading():
    """경로에서 옆으로 벗어났을 때 chi_cmd가 경로 쪽으로 회전한다."""
    pts, v = _straight_path(direction="north")
    guide = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)

    # 경로(N축)에서 동쪽으로 10m 벗어남
    pos = np.array([50.0, 10.0, 30.0])
    vel = np.array([15.0, 0.0, 0.0])   # 현재는 북쪽으로 비행 중

    chi_cmd, _, cte = guide.compute(pos, vel)

    # 경로가 서쪽(좌측)에 있으므로 chi_cmd가 약간 서쪽 방향(음수)이어야 함
    assert chi_cmd < 0.0 or abs(chi_cmd) < 0.5   # 경로 쪽으로 틀어짐


def test_cross_track_error_sign():
    """횡방향 오차 부호: 경로 **오른쪽 = 양수, 왼쪽 = 음수** (2026-07-25 정정).

    종전 docstring은 반대로 적혀 있었으나 assert 자체는 코드와 맞았다 —
    틀린 규약을 "검증된 것"으로 고착시키고 있었다."""
    pts, v = _straight_path(direction="north")
    guide = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)

    vel = np.array([15.0, 0.0, 0.0])

    _, _, cte_left  = guide.compute(np.array([50.0, -5.0, 0.0]), vel)  # 서쪽(좌)
    _, _, cte_right = guide.compute(np.array([50.0,  5.0, 0.0]), vel)  # 동쪽(우)

    assert cte_left  < 0   # 북진 경로에서 서쪽은 왼쪽 → 음수
    assert cte_right > 0


def test_ned_velocity_cmd_shape():
    """ned_velocity_cmd 출력은 shape (3,)."""
    pts, v = _straight_path()
    guide = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)
    vel_cmd = guide.ned_velocity_cmd(
        pos_ned=np.array([30.0, 0.0, 50.0]),
        vel_ned=np.array([12.0, 0.0, 0.0]),
    )
    assert vel_cmd.shape == (3,)


def test_near_zero_velocity_does_not_crash():
    """속도가 거의 0일 때 예외 없음."""
    pts, v = _straight_path()
    guide = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)
    chi_cmd, v_cmd, cte = guide.compute(
        pos_ned=np.array([10.0, 0.0, 0.0]),
        vel_ned=np.array([0.0, 0.0, 0.0]),
    )
    assert np.isfinite(chi_cmd)
    assert np.isfinite(v_cmd)


def test_path_end_does_not_crash():
    """경로 끝 근처에서 예외 없음."""
    pts, v = _straight_path(length=100.0, n=50)
    guide = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)
    chi_cmd, v_cmd, cte = guide.compute(
        pos_ned=np.array([99.0, 0.0, 0.0]),
        vel_ned=np.array([10.0, 0.0, 0.0]),
    )
    assert np.isfinite(chi_cmd)
    assert np.isfinite(v_cmd)


# ── FW 위치 setpoint lookahead (target_point_ned) ────────────────
# FW 오프보드는 위치 setpoint만 추종하므로, 이 목표점이 경로 위 전방에
# 위치해야 직선 추종이 된다. (목표점이 잘못되면 SITL에서 flower-pattern 발생)

def test_target_point_ahead_on_path():
    """경로 위에서 lookahead 만큼 전방의 점을 경로 위에 반환."""
    pts, v = _straight_path(length=200.0, n=200, direction="north")
    guide = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)
    tgt = guide.target_point_ned(np.array([50.0, 0.0, 30.0]), lookahead=70.0)
    assert tgt.shape == (2,)
    assert tgt[0] == pytest.approx(120.0, abs=2.0)   # N=50 → 70m 전방
    assert abs(tgt[1]) < 0.5                          # 경로(E=0) 위


def test_target_point_clamps_to_path_end():
    """lookahead가 경로 끝을 넘으면 끝점으로 클램프."""
    pts, v = _straight_path(length=100.0, n=100, direction="north")
    guide = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)
    tgt = guide.target_point_ned(np.array([60.0, 0.0, 30.0]), lookahead=70.0)
    assert tgt[0] == pytest.approx(100.0, abs=1.0)   # 60+70=130 > 100 → 끝점
    assert abs(tgt[1]) < 0.5


def test_target_point_pulls_back_to_path_when_offset():
    """경로에서 옆으로 벗어나도 목표점은 경로 위(전방)에 놓인다.

    이것이 FW를 경로로 복귀시키는 핵심 — 목표가 현재 진행방향이 아닌
    경로 위에 있어야 GPS 경로를 따라간다."""
    pts, v = _straight_path(length=200.0, n=200, direction="north")
    guide = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)
    # 경로(N축)에서 동쪽 30m 벗어남
    tgt = guide.target_point_ned(np.array([50.0, 30.0, 30.0]), lookahead=70.0)
    assert abs(tgt[1]) < 1.0     # 목표는 경로 위(E≈0), 현재 E=30 이 아님
    assert tgt[0] > 50.0         # 전방(북)


def test_target_point_lookahead_exceeds_turn_radius():
    """기본 _FW_LOOKAHEAD(70m)는 선회반경(~37m)보다 커야 한다(orbit 방지)."""
    pts, v = _straight_path(length=300.0, n=300, direction="north")
    guide = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)
    pos = np.array([50.0, 0.0, 30.0])
    tgt = guide.target_point_ned(pos, lookahead=70.0)
    dist = float(np.linalg.norm(tgt - pos[:2]))
    assert dist == pytest.approx(70.0, abs=2.0)
    assert dist > 37.0


# ── 세그먼트 탐색 (2026-07-27 SITL-7 R2) ──────────────────────────
#
# 종전 `_find_segment()` 는 캐시를 무시하고 매 틱 경로 전체를 훑었다.
# 아래 `_global_nearest_segment()` 가 **그 종전 구현 그대로**이며,
# 회귀 테스트가 "무엇이 달라졌는지"를 코드로 못박기 위해 남겨둔 것이다
# (비행 코드에는 존재하지 않는다).

def _global_nearest_segment(pts, p2):
    """2026-07-27 이전 `L1Guidance._find_segment()` 구현 (전역 최근접 탐색)."""
    pts = np.asarray(pts, dtype=float)
    p2 = np.asarray(p2, dtype=float)[:2]
    best_d2, best_i = np.inf, 0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        ab2 = float(ab @ ab)
        proj = a if ab2 < 1e-12 else a + np.clip((p2 - a) @ ab / ab2, 0.0, 1.0) * ab
        d2 = float((p2 - proj) @ (p2 - proj))
        if d2 < best_d2:
            best_d2, best_i = d2, i
    return best_i


def _closed_square(side=200.0, ds=1.0):
    """폐회로(닫힌 코스) 경로점. 시작점 = 종점 = (0, 0).

    (0,0) → (side,0) → (side,side) → (0,side) → (0,0) 을 ds 간격으로 샘플링.
    대회 경로가 폐회로로 확정됐으므로(2026-07-27 사용자) 이것이 기준 형상이다.
    """
    corners = [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side), (0.0, 0.0)]
    pts = []
    for (n0, e0), (n1, e1) in zip(corners[:-1], corners[1:]):
        seg_len = float(np.hypot(n1 - n0, e1 - e0))
        k = int(seg_len / ds)
        for j in range(k):
            f = j / k
            pts.append((n0 + (n1 - n0) * f, e0 + (e1 - e0) * f))
    pts.append(corners[-1])
    return np.array(pts), np.full(len(pts), 20.0)


def test_cumulative_arclen_basic():
    pts = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    c = cumulative_arclen(pts)
    assert c.tolist() == pytest.approx([0.0, 3.0, 7.0])
    assert len(cumulative_arclen(np.zeros((0, 2)))) == 0
    assert cumulative_arclen(np.array([[1.0, 2.0]])).tolist() == [0.0]


def test_segment_search_range_contains_prev_and_clamps():
    pts, _ = _straight_path(length=300.0, n=301)   # 1m 간격
    c = cumulative_arclen(pts)
    lo, hi = segment_search_range(c, 100, window_m=150.0, back_m=5.0)
    assert lo <= 100 <= hi
    assert lo == pytest.approx(95, abs=1)
    assert hi == pytest.approx(250, abs=1)
    # 경로 끝: hi 가 마지막 세그먼트를 넘지 않는다
    lo, hi = segment_search_range(c, 299, window_m=150.0, back_m=5.0)
    assert hi == len(pts) - 2
    # 경로 시작: lo 가 음수가 되지 않는다
    lo, hi = segment_search_range(c, 0, window_m=150.0, back_m=5.0)
    assert lo == 0
    # 퇴화 입력
    assert segment_search_range(np.zeros(1), 0) == (0, 0)


def test_nearest_segment_in_range_picks_projection():
    pts, _ = _straight_path(length=300.0, n=301)
    i = nearest_segment_in_range(pts, np.array([120.4, 3.0]), 0, 299)
    assert i == pytest.approx(120, abs=1)
    # 구간 밖은 절대 고르지 않는다
    i = nearest_segment_in_range(pts, np.array([120.4, 3.0]), 200, 250)
    assert i == 200


def test_closed_loop_final_leg_is_not_confused_with_first_leg():
    """**R2 핵심 회귀 테스트** — 폐회로 종점 부근에서 첫 레그를 잡지 않는다.

    폐회로는 마지막 레그가 첫 레그의 시작점으로 돌아온다. 종점 근방에서
    기체가 횡오차를 조금만 가져도 **기하학적으로는 첫 레그가 더 가깝다** —
    전역 최근접 탐색은 그 순간 인덱스를 0 근처로 되돌리고,
    `_lookahead_point()` 는 첫 레그를 따라 70m 전방(즉 진행방향과 90° 다른
    곳)을 목표로 내놓는다. 캠페인 cte 실측이 7~19m 였으므로 이 조건은
    가정이 아니라 실제 발현 대역이다.
    """
    pts, v = _closed_square(side=200.0)
    # 마지막 레그(E: 200→0, N=0) 위, 종점까지 4m. 횡오차 +6m(북쪽).
    pos = np.array([6.0, 4.0, 50.0])

    # 종전 구현: 첫 레그(N축)를 잡는다 — 거리 4m < 6m
    old = _global_nearest_segment(pts, pos)
    assert old < 10, f"전역탐색이 첫 레그를 잡지 않으면 이 테스트의 전제가 깨진다 (old={old})"

    guide = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)
    guide._seg_idx = len(pts) - 20          # 직전 틱: 마지막 레그 위
    new = guide._find_segment(pos[:2])
    assert new >= len(pts) - 30, f"창 탐색이 첫 레그로 되감겼다 (new={new})"

    # 목표점도 함께 확인 — 종전은 첫 레그를 따라 북쪽으로, 수정 후는 종점으로.
    guide2 = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)
    guide2._seg_idx = len(pts) - 20
    tgt = guide2.target_point_ned(pos, lookahead=70.0)
    assert float(np.linalg.norm(tgt - pts[-1])) < 1.0
    old_tgt = pts[min(old + 70, len(pts) - 1)]
    assert float(np.linalg.norm(tgt - old_tgt)) > 50.0


def test_segment_index_does_not_run_backward():
    """경로에서 뒤로 밀려도 인덱스는 back_m 이상 되감기지 않는다."""
    pts, v = _straight_path(length=300.0, n=301)
    guide = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)
    guide._seg_idx = 200
    # 실제로는 50m 지점에 있어도(EKF 리셋·급후퇴 등) 200 근방을 유지한다
    assert guide._find_segment(np.array([50.0, 0.0])) >= 200 - 6
    # 소폭(3m) 후퇴는 흡수한다
    guide._seg_idx = 200
    assert guide._find_segment(np.array([197.0, 0.0])) == pytest.approx(197, abs=1)


def test_segment_index_catches_up_forward_within_few_ticks():
    """진입 시 기체가 경로 한참 앞에 있어도 몇 틱 안에 따라잡는다.

    창 폭이 좁아 영원히 뒤처지면 목표점이 뒤에 남아 추종이 통째로 깨진다 —
    회귀 위험이 실재하므로 명시적으로 못박는다.
    """
    pts, v = _straight_path(length=500.0, n=501)
    guide = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)
    pos = np.array([400.0, 0.0])
    for _ in range(4):
        guide._seg_idx = guide._find_segment(pos)
    assert guide._seg_idx == pytest.approx(400, abs=2)


def test_windowed_search_matches_global_on_simple_forward_flight():
    """단순 전진 비행에서는 종전 구현과 결과가 같아야 한다(무회귀).

    A1/A3 같은 기존 기준선 시나리오의 거동이 바뀌지 않는다는 뜻이다.
    """
    pts, v = _straight_path(length=300.0, n=301)
    guide = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)
    for n in np.arange(0.0, 295.0, 2.0):
        p = np.array([n, 0.6 * np.sin(n / 20.0)])
        new = guide._find_segment(p)
        guide._seg_idx = new
        assert abs(new - _global_nearest_segment(pts, p)) <= 1
