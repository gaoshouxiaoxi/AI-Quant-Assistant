"""指标计算的单元测试。

原则：用**能手算的**简单序列，断言精确值，而不是只断言"不报错"。
"""

import numpy as np
import pandas as pd
import pytest

from metrics import (annualized_return, annualized_volatility, daily_returns,
                     drawdown_series, equity_curve, max_drawdown,
                     sharpe_ratio, total_return, TRADING_DAYS)


@pytest.fixture
def rising() -> pd.Series:
    """每日 +1% 的序列，共 101 个点（100 个交易日）。"""
    idx = pd.date_range("2024-01-01", periods=101, freq="B")
    return pd.Series(1.01 ** np.arange(101), index=idx)


def test_daily_returns_length(rising):
    r = daily_returns(rising)
    assert len(r) == len(rising) - 1
    assert r.dropna().iloc[0] == pytest.approx(0.01, rel=1e-9)


def test_total_return(rising):
    # 100 个交易日，每天 1%
    assert total_return(rising) == pytest.approx(1.01 ** 100 - 1, rel=1e-9)


def test_annualized_return(rising):
    # 年化用 len(close)/252 作为年数（101 个点 → 101 个交易日），
    # 所以是 (1.01^100)^(252/101) - 1
    expected = (1.01 ** 100) ** (TRADING_DAYS / 101) - 1
    assert annualized_return(rising) == pytest.approx(expected, rel=1e-6)


def test_volatility_zero_for_constant_growth(rising):
    # 每日固定 +1%，标准差为 0
    assert annualized_volatility(rising) == pytest.approx(0.0, abs=1e-10)


def test_sharpe_undefined_when_vol_zero(rising):
    # 波动为 0 时夏普应有定义缺口，返回 NaN 而不是抛异常或返回 inf
    assert np.isnan(sharpe_ratio(rising))


def test_max_drawdown_known_path():
    # 100 → 120 (+20%) → 90 (-25%) → 110
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    close = pd.Series([100.0, 120.0, 90.0, 110.0], index=idx)
    # 从峰值 120 跌到 90 = -25%
    assert max_drawdown(close) == pytest.approx(-0.25, rel=1e-12)


def test_max_drawdown_monotonic_rising_is_zero(rising):
    assert max_drawdown(rising) == pytest.approx(0.0, abs=1e-12)


def test_drawdown_series_non_positive(rising):
    dd = drawdown_series(rising)
    assert (dd <= 1e-12).all()


def test_equity_curve_starts_at_one(rising):
    eq = equity_curve(rising)
    assert eq.iloc[0] == pytest.approx(1.0)
    assert eq.iloc[-1] == pytest.approx(1.01 ** 100, rel=1e-9)


def test_short_series_returns_nan():
    idx = pd.date_range("2024-01-01", periods=1, freq="B")
    close = pd.Series([100.0], index=idx)
    assert np.isnan(total_return(close))
    assert np.isnan(annualized_return(close))
    assert np.isnan(max_drawdown(close))
