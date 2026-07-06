"""
StraightLinePlanner 단위 테스트 — 쿼드 검증 비행용 planner
============================================================
브링업 요구사항 검증:
  1. 이륙고도 10 m 기준 WP 세트(첫 WP = [0,0,10])에서 즉시 계획 성공
  2. s strictly increasing + np.gradient(pos, s) NaN 없음 (offboard 안전)
  3. 수직 상승 WP(같은 N,E)도 유효하게 표현
  4. wp_index 마킹 표준 준수 (각 원본 WP 정확히 하나)
"""
from __future__ import annotations
import sys
import os
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from path_planning.straight_line_planner import StraightLinePlanner

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

AIRCRAFT = {"gravity": 9.81, "v_cruise": 5.0, "a_max_g": 0.5}


def _assert_gradient_safe(path):
    pts = path.positions_array()
    s = np.array([p.s for p in path.points])
    ds = np.diff(s)
    assert np.all(ds > 1e-9), f"s가 strictly increasing이 아님: min(ds)={ds.min():.3e}"
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        grad = np.gradient(pts, s, axis=0)
    assert not np.isnan(grad).any(), "np.gradient 결과에 NaN 존재"
    assert len(wlist) == 0, f"RuntimeWarning 발생: {[str(w.message) for w in wlist]}"


def test_quad_bringup_wps_alt10():
    print("\n[Test 1] 이륙고도 10 m 기준 쿼드 검증 WP — 직선 경로")
    wps = np.array([
        [0.0,  0.0, 10.0],    # 이륙 완료 지점 (호버 시작점)
        [30.0, 0.0, 10.0],    # 전방 30 m
        [30.0, 20.0, 10.0],   # 우측 20 m
    ])
    path = StraightLinePlanner(ds=1.0).plan(wps, AIRCRAFT)
    assert abs(path.total_length - 50.0) < 1e-6
    marks = {p.wp_index for p in path.points if p.wp_index is not None}
    assert marks == {0, 1, 2}, f"wp_index 마킹 불일치: {marks}"
    n_marks = sum(1 for p in path.points if p.wp_index is not None)
    assert n_marks == 3, "각 WP에 정확히 하나의 마크 필요"
    assert all(p.curvature == 0.0 for p in path.points)
    assert path.planning_time < 0.5, "직선 planner는 즉시 끝나야 함"
    _assert_gradient_safe(path)
    print(f"  총 길이 {path.total_length:.1f} m, 점 {len(path.points)}개, "
          f"계획시간 {path.planning_time*1000:.1f} ms  ✓ 통과")


def test_vertical_segment_supported():
    print("\n[Test 2] 수직 상승 WP(같은 N,E) — 3D 직선이라 유효")
    wps = np.array([
        [0.0,  0.0,  0.0],
        [0.0,  0.0, 10.0],    # 수직 상승 10 m
        [30.0, 0.0, 10.0],
    ])
    path = StraightLinePlanner(ds=1.0).plan(wps, AIRCRAFT)
    assert abs(path.total_length - 40.0) < 1e-6
    _assert_gradient_safe(path)
    # 수직 구간 gamma ≈ +90°
    g0 = path.points[1].gamma_ref
    assert abs(g0 - np.pi / 2) < 1e-6, f"수직 구간 gamma 오류: {g0}"
    print("  ✓ 통과")


def test_duplicate_wp_merged():
    print("\n[Test 3] 완전 중복 WP 병합")
    wps = np.array([
        [0.0,  0.0, 10.0],
        [0.0,  0.0, 10.0],    # 완전 중복
        [30.0, 0.0, 10.0],
    ])
    path = StraightLinePlanner(ds=1.0).plan(wps, AIRCRAFT)
    assert len(path.waypoints_ned) == 2
    _assert_gradient_safe(path)
    print("  ✓ 통과")


if __name__ == "__main__":
    tests = [
        test_quad_bringup_wps_alt10,
        test_vertical_segment_supported,
        test_duplicate_wp_merged,
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
