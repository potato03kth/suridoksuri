"""pitot_zero_temp.py 의 순수 함수 단위테스트.

이 분석의 결론(기울기·신뢰구간·R²)이 전부 여기 있는 수식 위에 서 있고, scipy 가
없는 환경이라 t 분위수·불완전베타를 직접 구현했다. 그 구현이 틀리면 신뢰구간이
조용히 틀린 채로 문서에 실린다 — 그래서 알려진 값과 대조하는 테스트를 둔다.
pyulog·matplotlib 없이 돌아간다.
"""

import datetime as dt
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pitot_zero_temp import (  # noqa: E402
    betainc,
    cas_from_dp,
    decode_device_id,
    dp_from_cas,
    ms4525do_count_step_pa,
    nearest_join,
    ols_fit,
    recompute_cas,
    split_sessions,
    t_cdf,
    t_ppf,
    wall_time_from_name,
)


# --- device_id 디코드 -------------------------------------------------------

def test_decode_device_id_matches_actual_airspeed_sensor():
    """실기체 로그에 찍힌 값. I2C bus4, addr 0x28(MS4525DO 고정주소), devtype 0x48."""
    d = decode_device_id(4728865)
    assert d["bus_type"] == 1
    assert d["bus_type_name"] == "I2C"
    assert d["bus"] == 4
    assert d["address"] == 0x28
    assert d["devtype"] == 0x48


def test_decode_device_id_roundtrip():
    bus_type, bus, addr, devtype = 2, 3, 0x76, 0x21
    did = bus_type | (bus << 3) | (addr << 8) | (devtype << 16)
    d = decode_device_id(did)
    assert (d["bus_type"], d["bus"], d["address"], d["devtype"]) == (bus_type, bus, addr, devtype)


def test_ms4525do_quantization_step_matches_observed():
    """실측 raw dp 의 최소 간격이 1.0521 Pa 였다 — 드라이버 스케일과 맞아야 한다."""
    assert ms4525do_count_step_pa() == pytest.approx(1.0521, abs=1e-4)


# --- 차압 ↔ CAS ------------------------------------------------------------

def test_cas_dp_roundtrip_preserves_sign():
    for v in (-9.0, -0.5, 0.0, 3.3, 12.0):
        assert cas_from_dp(dp_from_cas(v)) == pytest.approx(v, abs=1e-9)


def test_dp_from_cas_known_value():
    # q = 0.5*rho*v^2, rho=1.225, v=12 -> 88.2 Pa
    assert dp_from_cas(12.0) == pytest.approx(0.5 * 1.225 * 144.0, rel=1e-12)


def test_recompute_cas_no_change_when_zero_matches_stored():
    assert recompute_cas(9.56, -21.783, -21.783) == pytest.approx(9.56, abs=1e-9)


def test_recompute_cas_more_negative_zero_raises_cas():
    """영점이 실제로 더 음수였다면 참 차압은 더 컸다는 뜻 → 교정 CAS 는 올라간다."""
    out = recompute_cas(9.56, -21.783, -38.97)
    assert out > 9.56
    assert out == pytest.approx(10.93, abs=0.01)


def test_recompute_cas_matches_manual_computation():
    cas, stored, zero = 9.56, -21.783, -44.28
    dp_corr = 0.5 * 1.225 * cas ** 2
    dp_raw = dp_corr + stored
    expect = math.sqrt(2 * (dp_raw - zero) / 1.225)
    assert recompute_cas(cas, stored, zero) == pytest.approx(expect, rel=1e-12)


# --- 통계 -------------------------------------------------------------------

def test_betainc_symmetry_and_bounds():
    assert betainc(2.0, 3.0, 0.0) == 0.0
    assert betainc(2.0, 3.0, 1.0) == 1.0
    # I_x(a,b) = 1 - I_{1-x}(b,a)
    assert betainc(2.5, 4.0, 0.3) == pytest.approx(1 - betainc(4.0, 2.5, 0.7), abs=1e-10)


def test_t_cdf_is_half_at_zero():
    for df in (1, 5, 30, 200):
        assert t_cdf(0.0, df) == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize("df,expect", [
    (1, 12.706), (2, 4.303), (5, 2.571), (10, 2.228),
    (30, 2.042), (60, 2.000), (118, 1.9803),
])
def test_t_ppf_matches_published_table(df, expect):
    """양측 95% 임계값 표(t_{0.975,df})와 대조."""
    assert t_ppf(0.975, df) == pytest.approx(expect, abs=2e-3)


def test_t_ppf_symmetric():
    assert t_ppf(0.025, 12) == pytest.approx(-t_ppf(0.975, 12), abs=1e-6)


def test_t_ppf_rejects_bad_input():
    with pytest.raises(ValueError):
        t_ppf(0.0, 10)
    with pytest.raises(ValueError):
        t_ppf(0.5, 0)


def test_ols_fit_recovers_exact_line():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 3.0 - 1.5 * x
    f = ols_fit(x, y)
    assert f["slope"] == pytest.approx(-1.5, abs=1e-12)
    assert f["intercept"] == pytest.approx(3.0, abs=1e-12)
    assert f["r2"] == pytest.approx(1.0, abs=1e-12)
    assert f["resid_sd"] == pytest.approx(0.0, abs=1e-9)


def test_ols_fit_ci_brackets_slope_and_is_symmetric():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 40)
    y = 2.0 - 0.8 * x + rng.normal(0, 0.5, x.size)
    f = ols_fit(x, y)
    assert f["slope_lo"] < f["slope"] < f["slope_hi"]
    assert (f["slope"] - f["slope_lo"]) == pytest.approx(f["slope_hi"] - f["slope"], rel=1e-9)
    assert f["slope_lo"] < -0.8 < f["slope_hi"]


def test_ols_fit_se_matches_textbook_formula():
    x = np.array([1.0, 2.0, 4.0, 7.0, 9.0, 11.0])
    y = np.array([2.0, 1.0, 4.0, 3.0, 8.0, 7.0])
    f = ols_fit(x, y)
    b1 = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    resid = y - (y.mean() - b1 * x.mean() + b1 * x)
    s2 = resid @ resid / (len(x) - 2)
    se = math.sqrt(s2 / np.sum((x - x.mean()) ** 2))
    assert f["slope"] == pytest.approx(b1, rel=1e-12)
    assert f["se_slope"] == pytest.approx(se, rel=1e-10)


def test_ols_fit_returns_none_without_leverage():
    """x 가 상수면 '기울기 0' 이 아니라 회귀 불가여야 한다 — 조용히 0 을 채우면 안 된다."""
    assert ols_fit([5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0]) is None
    assert ols_fit([1.0, 2.0], [1.0, 2.0]) is None


def test_ols_fit_drops_nan_pairs():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    f = ols_fit(x, y)
    assert f["n"] == 3


# --- 세션 분할 / 최근접 조인 -------------------------------------------------

def test_split_sessions_by_gap():
    base = dt.datetime(2026, 7, 30, 1, 0, 0)
    times = [base, base + dt.timedelta(minutes=2), base + dt.timedelta(minutes=100),
             base + dt.timedelta(minutes=103)]
    assert split_sessions(times, gap_s=45 * 60) == [0, 0, 1, 1]


def test_split_sessions_single_and_empty():
    assert split_sessions([]) == []
    assert split_sessions([dt.datetime(2026, 7, 30)]) == [0]


def test_nearest_join_picks_closest_sample():
    t_src = np.array([0, 100, 200], dtype=np.int64)
    v_src = np.array([10.0, 20.0, 30.0])
    out = nearest_join(np.array([-5, 40, 60, 199, 500], dtype=np.int64), t_src, v_src)
    assert list(out) == [10.0, 10.0, 20.0, 30.0, 30.0]


def test_nearest_join_handles_empty_and_single_source():
    q = np.array([1, 2, 3], dtype=np.int64)
    assert np.isnan(nearest_join(q, np.array([], dtype=np.int64), np.array([]))).all()
    assert list(nearest_join(q, np.array([7], dtype=np.int64), np.array([4.2]))) == [4.2] * 3


def test_wall_time_from_name():
    assert wall_time_from_name("logs/x/log_269_2026-07-30-01-15-54.ulg") == \
        dt.datetime(2026, 7, 30, 1, 15, 54)
    assert wall_time_from_name("logs/2026-07-23_flight01/11_32_15.ulg") is None
