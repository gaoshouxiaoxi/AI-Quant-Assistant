"""波动率估计量的数学正确性测试。

这些估计量都有闭式定义，必须用解析解验证，而不是"跑通就行"。
"""

import numpy as np
import pandas as pd
import pytest

from rv_estimators import (TRADING_DAYS, daily_rs, daily_var_close,
                           daily_var_garman_klass, daily_var_parkinson,
                           rolling_vol_yang_zhang)

LN2 = np.log(2.0)


@pytest.fixture
def ohlc():
    """构造一个可手算的 30 天 OHLC 序列。"""
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    close = pd.Series(np.linspace(100, 130, 30), index=idx)
    open_ = close.shift(1).fillna(100.0)
    high = close * 1.02
    low = close * 0.98
    return open_, high, low, close


def test_parkinson_matches_formula():
    """Parkinson 单日方差 = log(H/L)² / (4·ln2)。"""
    high = pd.Series([110.0, 120.0])
    low = pd.Series([100.0, 90.0])
    expected = pd.Series([np.log(1.1) ** 2 / (4 * LN2),
                          np.log(120 / 90) ** 2 / (4 * LN2)])
    got = daily_var_parkinson(high, low)
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_parkinson_zero_when_no_range():
    high = pd.Series([100.0])
    low = pd.Series([100.0])
    assert daily_var_parkinson(high, low).iloc[0] == pytest.approx(0.0)


def test_garman_klass_matches_formula():
    o = pd.Series([100.0])
    h = pd.Series([110.0])
    lo = pd.Series([95.0])
    c = pd.Series([105.0])
    expected = (0.5 * np.log(h / lo) ** 2
                - (2 * LN2 - 1) * np.log(c / o) ** 2).iloc[0]
    assert daily_var_garman_klass(o, h, lo, c).iloc[0] == pytest.approx(
        expected, rel=1e-12)


def test_rogers_satchell_matches_formula():
    o = pd.Series([100.0])
    h = pd.Series([110.0])
    lo = pd.Series([95.0])
    c = pd.Series([105.0])
    expected = (np.log(h / c) * np.log(h / o)
                + np.log(lo / c) * np.log(lo / o)).iloc[0]
    assert daily_rs(o, h, lo, c).iloc[0] == pytest.approx(expected, rel=1e-12)


def test_close_variance_is_squared_log_return():
    close = pd.Series([100.0, 110.0])
    got = daily_var_close(close)
    assert got.iloc[1] == pytest.approx(np.log(1.1) ** 2, rel=1e-12)
    assert np.isnan(got.iloc[0])  # 第一天没有前收盘


def test_yang_zhang_positive_and_finite(ohlc):
    o, h, lo, c = ohlc
    vol = rolling_vol_yang_zhang(o, h, lo, c, 10)
    valid = vol.dropna()
    assert len(valid) > 0
    assert (valid > 0).all()
    assert np.isfinite(valid).all()


def test_yang_zhang_annualization_scale(ohlc):
    """年化后应显著大于日波动（约 sqrt(252) ≈ 15.9 倍）。"""
    o, h, lo, c = ohlc
    vol = rolling_vol_yang_zhang(o, h, lo, c, 10).dropna()
    daily_scale = vol.iloc[-1] / np.sqrt(TRADING_DAYS)
    assert 0 < daily_scale < 0.5


def test_vol_scales_with_range():
    """把高低价区间放大一倍，波动率应随之上升。"""
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    close = pd.Series(np.linspace(100, 110, 40), index=idx)
    o = close.shift(1).fillna(100.0)

    narrow = rolling_vol_yang_zhang(o, close * 1.01, close * 0.99, close, 10)
    wide = rolling_vol_yang_zhang(o, close * 1.05, close * 0.95, close, 10)
    assert wide.dropna().iloc[-1] > narrow.dropna().iloc[-1]
