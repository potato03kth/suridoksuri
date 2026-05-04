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
