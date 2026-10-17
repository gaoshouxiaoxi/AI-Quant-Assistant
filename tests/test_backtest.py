"""回测引擎的测试。

重点：
1. 手续费与滑点必须**单向**作用于净值（成本越高，收益越低）
2. 不能有前瞻偏差——改未来价格不能影响过去的成交
3. 整股成交，不出现小数股
4. 期末未平仓要如实标记
"""

import numpy as np
import pandas as pd
import pytest

from backtest import backtest_ma


@pytest.fixture
def oscillating() -> pd.Series:
    """明显震荡的序列，能触发多次交易。"""
    idx = pd.date_range("2022-01-03", periods=300, freq="B")
    t = np.arange(300)
    close = 100 + 18 * np.sin(t / 12) + 6 * np.sin(t / 4)
    return pd.Series(close, index=idx)


def test_returns_expected_keys(oscillating):
    result = backtest_ma(oscillating, 5, 20)
    assert set(result) >= {"equity", "trades", "stats", "frame"}
    assert len(result["equity"]) == len(oscillating)
    assert "final_equity" in result["stats"]


def test_higher_costs_lower_final_equity(oscillating):
    free = backtest_ma(oscillating, 5, 20, fee_rate=0.0, slippage=0.0)
    costly = backtest_ma(oscillating, 5, 20, fee_rate=0.002, slippage=0.002)
    assert costly["stats"]["final_equity"] < free["stats"]["final_equity"]


def test_no_lookahead(oscillating):
    """改掉最后 40 天的价格，之前的成交不应变化。"""
    base = backtest_ma(oscillating, 5, 20)
    modified = oscillating.copy()
    modified.iloc[-40:] = modified.iloc[-40:] * 1.5
    alt = backtest_ma(modified, 5, 20)

    cut = oscillating.index[-40]
    base_before = [t for t in base["trades"] if t["exit_date"] < cut]
    alt_before = [t for t in alt["trades"] if t["exit_date"] < cut]
    assert len(base_before) == len(alt_before)
    for a, b in zip(base_before, alt_before):
        assert a["entry_date"] == b["entry_date"]
        assert a["entry_price"] == pytest.approx(b["entry_price"], rel=1e-12)


def test_integer_shares(oscillating):
    result = backtest_ma(oscillating, 5, 20)
    for t in result["trades"]:
        assert float(t["shares"]).is_integer(), "出现小数股，现实中不可成交"


def test_buy_price_includes_slippage(oscillating):
    slip = 0.01
    result = backtest_ma(oscillating, 5, 20, slippage=slip, fee_rate=0.0)
    frame = result["frame"]
    for t in result["trades"][:3]:
        raw_close = float(frame.loc[t["entry_date"], "close"])
        assert t["entry_price"] == pytest.approx(raw_close * (1 + slip), rel=1e-9)


def test_sell_price_subtracts_slippage(oscillating):
    slip = 0.01
    result = backtest_ma(oscillating, 5, 20, slippage=slip, fee_rate=0.0)
    frame = result["frame"]
    for t in result["trades"][:3]:
        raw_close = float(frame.loc[t["exit_date"], "close"])
        assert t["exit_price"] == pytest.approx(raw_close * (1 - slip), rel=1e-9)


def test_benchmark_matches_price_change(oscillating):
    result = backtest_ma(oscillating, 5, 20)
    expected = oscillating.iloc[-1] / oscillating.iloc[0] - 1
    assert result["stats"]["benchmark_return"] == pytest.approx(expected, rel=1e-12)


def test_everything_in_cash_when_no_signals():
    """单调上涨但均线始终同向时，至少不会亏成负数。"""
    idx = pd.date_range("2022-01-03", periods=120, freq="B")
    close = pd.Series(np.linspace(100, 200, 120), index=idx)
    result = backtest_ma(close, 5, 20)
    assert result["stats"]["final_equity"] > 0
    assert result["stats"]["final_equity"] <= 100_000 * (1 + 1.01)


def test_too_short_raises():
    idx = pd.date_range("2022-01-03", periods=10, freq="B")
    close = pd.Series(np.linspace(100, 110, 10), index=idx)
    with pytest.raises(ValueError):
        backtest_ma(close, 5, 20)


def test_open_position_flagged(oscillating):
    result = backtest_ma(oscillating, 5, 20)
    if result["stats"]["open_position"]:
        assert result["stats"]["open_trade"] is not None
        assert result["stats"]["open_trade"]["exit_date"] is None
