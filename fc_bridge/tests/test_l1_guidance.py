"""
L1Guidance 단위 테스트.
"""
import numpy as np
import pytest
from fc_bridge.guidance.l1_guidance import L1Guidance


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
    """횡방향 오차 부호: 경로 왼쪽 = 양수, 오른쪽 = 음수."""
    pts, v = _straight_path(direction="north")
    guide = L1Guidance(l1_dist=20.0, path_pts=pts, v_profile=v)

    vel = np.array([15.0, 0.0, 0.0])

    _, _, cte_left  = guide.compute(np.array([50.0, -5.0, 0.0]), vel)  # 서쪽(좌)
    _, _, cte_right = guide.compute(np.array([50.0,  5.0, 0.0]), vel)  # 동쪽(우)

    assert cte_left  < 0   # 북쪽 방향 경로에서 서쪽은 오른쪽
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
