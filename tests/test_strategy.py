"""双均线信号与持仓状态的测试。

这里重点守住两个曾经踩过的坑：
1. 持仓状态必须能**延续**（不是只有金叉当天持仓）——ffill 用错会退化成 1 天
2. 持仓必须**次日生效**，不能用到当天收盘后才知道的信号
"""

import numpy as np
import pandas as pd
import pytest

from strategy import add_moving_averages, crossover_dates, generate_signals


@pytest.fixture
def up_then_down() -> pd.DataFrame:
    """横盘 → 上涨 → 下跌，确保先出现金叉再出现死叉。

    注意不能只用"单边上涨→下跌"：线性上涨期间快线从一开始就在慢线上方，
    根本不存在穿越点（实测会得到 0 个信号），无法测试信号逻辑。
    """
    idx = pd.date_range("2023-01-02", periods=160, freq="B")
    flat = np.full(40, 100.0)
    up = np.linspace(100, 150, 50)
    down = np.linspace(150, 100, 70)
    close = np.r_[flat, up, down]
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": 1_000_000.0,
    }, index=idx)


def test_moving_averages_match_rolling_mean(up_then_down):
    df = add_moving_averages(up_then_down, 5, 20)
    expected = up_then_down["Close"].rolling(5).mean()
    np.testing.assert_allclose(df["MA5"].to_numpy(), expected.to_numpy(),
                               rtol=1e-12, equal_nan=True)


def test_signals_are_crossing_events_not_states(up_then_down):
    """信号数应远少于"快线在慢线上方"的天数——说明识别的是穿越而不是状态。"""
    df = generate_signals(up_then_down, 5, 20)
    above_days = int((df["MA5"] > df["MA20"]).sum())
    signal_days = int((df["signal"] != 0).sum())
    assert signal_days > 0
    assert signal_days < above_days / 4


def test_position_extends_beyond_signal_day(up_then_down):
    """守住 ffill 的坑：持仓必须延续多天，而不是只有金叉当天。

    曾经的 bug 是把"无信号日"写成 0，导致 ffill 失效、全区间只持仓
    与金叉次数相同的天数。这里直接检查最长连续持仓长度。
    """
    df = generate_signals(up_then_down, 5, 20)
    pos = df["position"].to_numpy()

    longest = current = 0
    for v in pos:
        current = current + 1 if v == 1 else 0
        longest = max(longest, current)

    assert longest >= 20, f"最长连续持仓只有 {longest} 天，疑似 ffill 失效"


def test_position_never_exceeds_signal_count_windows(up_then_down):
    """持仓天数应接近"买入到卖出"的累计区间，而不是信号次数。"""
    df = generate_signals(up_then_down, 5, 20)
    pos_days = int(df["position"].sum())
    n_buys = int((df["signal"] == 1).sum())
    # 若只有信号当天持仓，pos_days 会等于 n_buys
    assert pos_days > n_buys * 5, f"持仓 {pos_days} 天 / 金叉 {n_buys} 次，疑似提前平仓"


def test_position_is_next_day(up_then_down):
    """金叉次日才持仓，绝不当天建仓。"""
    df = generate_signals(up_then_down, 5, 20)
    for i in range(1, len(df)):
        if df["signal"].iloc[i - 1] == 1:
            assert df["position"].iloc[i] == 1
        if df["signal"].iloc[i - 1] == -1:
            assert df["position"].iloc[i] == 0


def test_no_lookahead_in_position(up_then_down):
    """截断未来数据不应改变历史持仓。

    若实现里用到了未来信息，截断前后的历史部分会不一致。
    """
    full = generate_signals(up_then_down, 5, 20)["position"]
    cut = generate_signals(up_then_down.iloc[:100], 5, 20)["position"]
    # 前 90 天应完全一致（留出 20 日均线的预热期）
    np.testing.assert_array_equal(full.iloc[30:90].to_numpy(),
                                  cut.iloc[30:90].to_numpy())


def test_crossover_dates_alternate(up_then_down):
    """金叉与死叉应交替出现，不会连续两个同向信号。"""
    df = generate_signals(up_then_down, 5, 20)
    events = [k for _, k in crossover_dates(df)]
    for a, b in zip(events, events[1:]):
        assert a != b, "出现连续同向信号，穿越判定有误"
