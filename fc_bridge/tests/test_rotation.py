import numpy as np
import pytest

from fc_bridge.utils.rotation import quat_to_euler_xyz, yaw_ned_to_quat_enu


def test_yaw_ned_zero_is_east_facing_enu():
    # NED yaw=0(북) → ENU yaw=+90°(북=ENU에서 +y축)
    w, x, y, z = yaw_ned_to_quat_enu(0.0)
    assert x == 0.0 and y == 0.0
    roll, pitch, yaw_enu = quat_to_euler_xyz(w, x, y, z)
    assert roll == pytest.approx(0.0)
    assert pitch == pytest.approx(0.0)
    assert yaw_enu == pytest.approx(np.pi / 2.0)


def test_yaw_ned_90_is_identity_quat():
    # NED yaw=90°(동) → ENU yaw=0 → 단위쿼터니언 (이전까지의 암묵적 기본값과 일치)
    w, x, y, z = yaw_ned_to_quat_enu(np.pi / 2.0)
    assert w == pytest.approx(1.0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(0.0)


@pytest.mark.parametrize("yaw_ned", [
    0.0, 0.3, -0.3, np.pi / 2, -np.pi / 2, np.pi, -np.pi + 0.01,
    -1.704,  # flight04 사고 당시 실측 헤딩(-97.64°) 근사
])
def test_round_trip_through_decode(yaw_ned):
    w, x, y, z = yaw_ned_to_quat_enu(yaw_ned)
    _, _, yaw_enu = quat_to_euler_xyz(w, x, y, z)
    recovered_ned = np.pi / 2.0 - yaw_enu
    wrapped = (recovered_ned - yaw_ned + np.pi) % (2 * np.pi) - np.pi
    assert wrapped == pytest.approx(0.0, abs=1e-9)


def test_quaternion_is_normalized():
    for yaw_ned in np.linspace(-np.pi, np.pi, 13):
        w, x, y, z = yaw_ned_to_quat_enu(float(yaw_ned))
        assert w * w + x * x + y * y + z * z == pytest.approx(1.0)
