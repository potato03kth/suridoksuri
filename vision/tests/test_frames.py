"""`vision/core/frames.py` — 좌표 프레임 체인 왕복검증.

`docs/vision_fc_interface.md` §4.3의 RT-1~RT-6를 그대로 구현한다(RT-7 = ENU 경로는 §4.4가
"vision은 NED 변환을 하지 않는다"로 확정해 **구현 자체가 없으므로** 테스트도 없다 —
`core/frames.py` docstring의 "이 파일이 하지 않는 것" 참조).

스타일은 `fc_bridge/tests/test_rotation.py`의 왕복검증 관례를 따른다(읽기만 함).

⚠️ **RT-3~RT-5가 이 파일의 핵심이다.** 축 이름만 맞고 부호가 뒤집힌 회전행렬도 RT-1(직교성)과
RT-2(왕복)는 **멀쩡히 통과한다** — 알려진 방향쌍을 박아두는 것만이 부호 실수를 잡는다.
"""
import numpy as np
import pytest

from vision.core import frames


PSIS = [0.0, 0.3, -0.3, np.pi / 4, np.pi / 2, -np.pi / 2, np.pi, 2.0, -2.7]


# ── RT-1: 진짜 회전인가 ────────────────────────────────────────────────────────

@pytest.mark.parametrize("psi", PSIS)
def test_rt1_rotation_matrices_are_orthonormal_with_unit_determinant(psi):
    for R in (frames.R_frd_cam(psi), frames.R_flu_cam(psi)):
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


# ── RT-2: 왕복 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("psi", PSIS)
def test_rt2_cam_to_frd_and_back_round_trip(psi):
    rng = np.random.default_rng(0)
    for _ in range(20):
        p_cam = rng.uniform(-40.0, 40.0, size=3)
        recovered = frames.frd_to_cam(frames.cam_to_frd(p_cam, psi), psi)
        np.testing.assert_allclose(recovered, p_cam, atol=1e-9)


@pytest.mark.parametrize("psi", PSIS)
def test_rt2_full_chain_cam_flu_frd_back_to_cam(psi):
    """§4.3 체인 전체를 지나 되돌아온다: cam → flu → frd → flu → cam."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        p_cam = rng.uniform(-40.0, 40.0, size=3)
        p_flu = frames.cam_to_flu(p_cam, psi)
        p_frd = frames.flu_to_frd(p_flu)
        recovered = frames.flu_to_cam(frames.frd_to_flu(p_frd), psi)
        np.testing.assert_allclose(recovered, p_cam, atol=1e-9)


@pytest.mark.parametrize("psi", PSIS)
def test_flu_and_frd_paths_agree(psi):
    """`cam_to_frd`와 `cam_to_flu`+`flu_to_frd`가 같은 값이어야 한다 —
    `R_flu_cam`을 `R_frd_cam`에서 유도한 설계가 실제로 일관적인지 확인."""
    rng = np.random.default_rng(2)
    for _ in range(10):
        p_cam = rng.uniform(-10.0, 10.0, size=3)
        np.testing.assert_allclose(
            frames.cam_to_frd(p_cam, psi),
            frames.flu_to_frd(frames.cam_to_flu(p_cam, psi)),
            atol=1e-12,
        )


# ── RT-3 ~ RT-5: 알려진 방향쌍 (부호 실수를 잡는 실질적 그물) ──────────────────

def test_rt3_target_directly_below_camera():
    """카메라 정중앙 바로 아래 타겟 `p_cam=(0,0,h)` → FRD `(0,0,h)`(정하방), FLU `(0,0,-h)`."""
    h = 12.5
    np.testing.assert_allclose(frames.cam_to_frd((0.0, 0.0, h)), (0.0, 0.0, h), atol=1e-12)
    np.testing.assert_allclose(frames.cam_to_flu((0.0, 0.0, h)), (0.0, 0.0, -h), atol=1e-12)


def test_rt4_target_on_image_right_is_to_the_aircraft_right():
    """이미지 **오른쪽**의 타겟 `p_cam=(+d,0,h)` → FRD `(0,+d,h)`(기체 **우측**), FLU `(0,-d,-h)`."""
    d, h = 3.0, 12.5
    np.testing.assert_allclose(frames.cam_to_frd((d, 0.0, h)), (0.0, d, h), atol=1e-12)
    np.testing.assert_allclose(frames.cam_to_flu((d, 0.0, h)), (0.0, -d, -h), atol=1e-12)


def test_rt5_target_on_image_top_is_ahead_of_the_aircraft():
    """이미지 **위쪽**의 타겟 `p_cam=(0,-d,h)` → FRD `(+d,0,h)`(기체 **전방**), FLU `(+d,0,-h)`."""
    d, h = 3.0, 12.5
    np.testing.assert_allclose(frames.cam_to_frd((0.0, -d, h)), (d, 0.0, h), atol=1e-12)
    np.testing.assert_allclose(frames.cam_to_flu((0.0, -d, h)), (d, 0.0, -h), atol=1e-12)


def test_matrices_match_the_interface_document_literals():
    """§4.3이 문서에 직접 적어둔 두 행렬 리터럴과 일치하는가.

    `R_flu_cam`은 `R_frd_cam`에서 유도하도록 구현했으므로(일관성 보장), 그 유도 결과가 문서가
    독립적으로 적어둔 값과 같은지 확인해야 유도 자체의 오류를 잡는다."""
    np.testing.assert_allclose(
        frames.R_flu_cam(0.0),
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        frames.R_frd_cam(0.0),
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )


# ── 마운트 요각 ψ_m 이 실제로 작동하는가 ───────────────────────────────────────

def test_mount_yaw_90deg_rotates_forward_target_to_the_right():
    """🔴 §4.3의 경고 "잘못 잡으면 정확히 90° 틀린 방향으로 정확하게 이동한다"를 코드로 고정.

    ψ_m=0에서 **전방**(RT-5)으로 나오던 이미지-위쪽 타겟이, ψ_m=90°에서는 **우측**으로 나와야
    한다. ψ_m이 하드코딩돼 무시되면 이 테스트가 red가 된다."""
    d, h = 3.0, 12.5
    p_cam = (0.0, -d, h)
    np.testing.assert_allclose(frames.cam_to_frd(p_cam, 0.0), (d, 0.0, h), atol=1e-12)
    np.testing.assert_allclose(frames.cam_to_frd(p_cam, np.pi / 2), (0.0, d, h), atol=1e-12)


@pytest.mark.parametrize("psi", PSIS)
def test_psi_composition_is_left_multiplication_by_rz(psi):
    """§4.3 `R_{frd←cam}(ψ_m) = Rz(ψ_m) · R_{frd←cam}(0)`."""
    c, s = np.cos(psi), np.sin(psi)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(frames.R_frd_cam(psi), rz @ frames.R_frd_cam(0.0), atol=1e-12)


def test_mount_yaw_default_is_declared_unmeasured():
    """🔴 §7 U3 — ψ_m은 미측정이다. 누가 실측 없이 `MEASURED=True`로 뒤집으면 red가 된다
    (그 플래그가 와이어 레코드를 타고 소비자까지 가므로 조용히 바뀌면 안 된다)."""
    assert frames.MOUNT_YAW_PSI_M_RAD_DEFAULT == 0.0
    assert frames.MOUNT_YAW_PSI_M_MEASURED is False


# ── FLU ↔ FRD 자체 성질 ────────────────────────────────────────────────────────

def test_flu_frd_conversion_is_an_involution():
    """`(x,-y,-z)`는 두 번 적용하면 원래대로 — 두 함수가 같은 행렬을 쓴다는 설계의 근거."""
    rng = np.random.default_rng(3)
    for _ in range(10):
        p = rng.uniform(-5.0, 5.0, size=3)
        np.testing.assert_allclose(frames.flu_to_frd(frames.frd_to_flu(p)), p, atol=1e-12)


def test_flu_to_frd_flips_exactly_y_and_z():
    np.testing.assert_allclose(frames.flu_to_frd((1.0, 2.0, 3.0)), (1.0, -2.0, -3.0), atol=1e-12)


# ── RT-6: 쿼터니언 성분 순서 ───────────────────────────────────────────────────

def test_rt6_quaternion_order_round_trip():
    """(x,y,z,w) → (w,x,y,z) → (x,y,z,w) 왕복. §4.2의 "파일 하나 사이로 규약이 다르다" 함정."""
    q_xyzw = (0.1, -0.2, 0.3, 0.927)
    assert frames.quat_wxyz_to_xyzw(frames.quat_xyzw_to_wxyz(q_xyzw)) == pytest.approx(
        q_xyzw, abs=1e-9
    )


def test_rt6_quaternion_order_actually_moves_the_scalar_component():
    """단순 항등함수로 구현하면(=순서를 안 바꾸면) red가 되게 하는 테스트."""
    assert frames.quat_xyzw_to_wxyz((1.0, 2.0, 3.0, 4.0)) == (4.0, 1.0, 2.0, 3.0)
    assert frames.quat_wxyz_to_xyzw((4.0, 1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0, 4.0)


# ── 입력 방어 / 결정론 ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [(1.0, 2.0), (1.0, 2.0, 3.0, 4.0), ()])
def test_non_3vector_input_raises(bad):
    with pytest.raises(ValueError):
        frames.cam_to_frd(bad)


def test_bad_quaternion_size_raises():
    with pytest.raises(ValueError):
        frames.quat_xyzw_to_wxyz((1.0, 2.0, 3.0))


def test_deterministic():
    """§7.5 — 같은 입력이면 항상 같은 출력(난수·wall-clock 미사용)."""
    p = (1.234, -5.678, 9.012)
    a = frames.cam_to_frd(p, 0.7)
    b = frames.cam_to_frd(p, 0.7)
    np.testing.assert_array_equal(a, b)


def test_accepts_list_tuple_and_ndarray_alike():
    expected = frames.cam_to_frd((1.0, 2.0, 3.0))
    np.testing.assert_allclose(frames.cam_to_frd([1.0, 2.0, 3.0]), expected, atol=1e-15)
    np.testing.assert_allclose(frames.cam_to_frd(np.array([1.0, 2.0, 3.0])), expected, atol=1e-15)
