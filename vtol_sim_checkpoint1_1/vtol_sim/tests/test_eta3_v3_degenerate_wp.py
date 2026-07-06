"""
Eta3ClothoidPlannerV3 (v3.3) — 2D 퇴화 WP 회귀 테스트
======================================================

배경: 실기체(RPi5) 브링업 중, 이륙점 수직 상방에 놓인 천이고도 WP(같은
N,E, 고도만 다름)가 planner에 들어가 경로 시작에 s가 동일한 중복점이
생겼고, offboard_node의 np.gradient(f, s) 계산이 divide-by-zero → NaN
궤적을 만들어 setpoint가 전부 무효화되었다.
(2026-07-03 비행 로그의 numpy RuntimeWarning 4종이 그 증상)

이 테스트는 다음을 보장한다:
  1. 2D 퇴화 WP 입력 시에도 plan()이 성공한다 (병합 + 경고).
  2. 출력 경로의 s는 strictly increasing이다.
  3. np.gradient(pos, s)에 NaN이 없다 (offboard FF 계산 안전).
"""
from __future__ import annotations
import sys
import os
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from path_planning.eta3clothoid_v3_1_planner import Eta3ClothoidPlannerV3

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

AIRCRAFT = {"gravity": 9.81, "v_cruise": 18.0, "a_max_g": 0.5}


def _check_path_safe_for_gradient(path):
    pts = path.positions_array()
    s = np.array([p.s for p in path.points])
    ds = np.diff(s)
    assert np.all(ds > 1e-9), f"s가 strictly increasing이 아님: min(ds)={ds.min():.3e}"
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        grad = np.gradient(pts, s, axis=0)
    assert not np.isnan(grad).any(), "np.gradient 결과에 NaN 존재"
    assert len(wlist) == 0, f"np.gradient에서 RuntimeWarning 발생: {[str(w.message) for w in wlist]}"


def test_vertical_climb_wp_merged():
    print("\n[Test 1] 이륙점 수직 상방 천이고도 WP (2D 퇴화) → 병합 + NaN 없는 경로")
    wps = np.array([
        [0.0,    0.0,   0.0],
        [0.0,    0.0,  20.0],   # 같은 N,E — 수직 상승 WP
        [150.0,  50.0, 30.0],
        [300.0, -40.0, 30.0],
    ])
    path = Eta3ClothoidPlannerV3(ds=1.0).plan(wps, AIRCRAFT)
    assert len(path.waypoints_ned) == 3, "퇴화 WP가 병합되지 않음"
    assert path.waypoints_ned[0, 2] == 20.0, "병합 시 나중 WP 고도를 채택해야 함"
    _check_path_safe_for_gradient(path)
    print("  ✓ 통과")


def test_normal_wps_unaffected():
    print("\n[Test 2] 정상 WP 세트는 병합 없이 기존과 동일하게 동작")
    wps = np.array([
        [0.0,    0.0,   0.0],
        [100.0, 10.0,  20.0],
        [250.0, 60.0,  30.0],
        [400.0, -40.0, 30.0],
    ])
    path = Eta3ClothoidPlannerV3(ds=1.0).plan(wps, AIRCRAFT)
    assert len(path.waypoints_ned) == 4
    marks = [p.wp_index for p in path.points if p.wp_index is not None]
    assert sorted(marks) == [0, 1, 2, 3], f"wp_index 마킹 불일치: {marks}"
    _check_path_safe_for_gradient(path)
    print("  ✓ 통과")


def test_all_wps_degenerate_raises():
    print("\n[Test 3] 전 WP가 한 지점(수직 전용 미션) → 명확한 ValueError")
    wps = np.array([
        [0.0, 0.0,  0.0],
        [0.0, 0.0, 10.0],
        [0.0, 0.0, 20.0],
    ])
    try:
        Eta3ClothoidPlannerV3(ds=1.0).plan(wps, AIRCRAFT)
    except ValueError as e:
        print(f"  기대한 예외: {e}")
        print("  ✓ 통과")
        return
    raise AssertionError("퇴화 WP만 있는데 ValueError가 발생하지 않음")


if __name__ == "__main__":
    tests = [
        test_vertical_climb_wp_merged,
        test_normal_wps_unaffected,
        test_all_wps_degenerate_raises,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  ✗ 실패: {e}")
            failed.append(t.__name__)
        except Exception as e:
            print(f"  ✗ 예외: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(t.__name__)

    print("\n" + "=" * 60)
    if not failed:
        print("✓ 모든 테스트 통과")
    else:
        print(f"✗ {len(failed)}개 실패: {failed}")
        sys.exit(1)
