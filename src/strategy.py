"""第 3 周：技术指标与均线策略。

策略（双均线）：
- 金叉（买入）：MA5 上穿 MA20
- 死叉（卖出）：MA5 下穿 MA20

重要约定：
- `position` 是**实际可执行**的持仓状态，从金叉的**次一交易日**才变为 1。
  信号当天收盘后才能知道是否交叉，无法当天成交，否则会引入前视偏差
  （look-ahead bias）——这是回测里最常见、也最致命的错误来源。
"""

from __future__ import annotations

import pandas as pd

BUY = "buy"
SELL = "sell"


def add_moving_averages(stock: pd.DataFrame,
                        fast: int = 5, slow: int = 20) -> pd.DataFrame:
    """加上快慢均线。"""
    df = stock.copy()
    df[f"MA{fast}"] = df["Close"].rolling(fast).mean()
    df[f"MA{slow}"] = df["Close"].rolling(slow).mean()
    return df


def generate_signals(stock: pd.DataFrame, fast: int = 5,
                     slow: int = 20) -> pd.DataFrame:
    """生成金叉/死叉信号与可执行持仓。

    返回的 DataFrame 额外包含：
    - MA{fast} / MA{slow}
    - signal_diff：均线差（快 - 慢），画图用
    - signal：1 金叉、-1 死叉、0 无
    - position：当日收盘后应持有的仓位（次日才生效）
    - returns / equity：策略与买入持有的净值对比
    """
    df = add_moving_averages(stock, fast, slow)
    fast_col, slow_col = f"MA{fast}", f"MA{slow}"
    diff = df[fast_col] - df[slow_col]

    # 真正发生穿越：前一天还在下方/上方，今天越过去了
    cross_up = (diff > 0) & (diff.shift(1) <= 0)
    cross_down = (diff < 0) & (diff.shift(1) >= 0)

    signal = pd.Series(0, index=df.index, dtype="int64")
    signal[cross_up.fillna(False)] = 1
    signal[cross_down.fillna(False)] = -1

    # 用 NaN 表示"当天没有新信号"，这样 ffill 才能把持仓状态延续下去。
    # 注意：不能把无信号日写成 0——0 是"空仓"这个有效状态，会阻断 ffill，
    # 导致持仓只剩信号当天（曾因此让策略净值从 ~1.85 掉到 1.02）。
    state = pd.Series(float("nan"), index=df.index)
    state[cross_up.fillna(False)] = 1.0
    state[cross_down.fillna(False)] = 0.0

    # 信号次日才可能成交：整体后移一天，再向前填充；首个信号之前视为空仓
    position = state.shift(1).ffill().fillna(0.0).astype("int64")

    df["signal_diff"] = diff
    df["signal"] = signal
    df["position"] = position

    # 净值对比（不含手续费，仅用于直观比较；正式回测在第 4 周）
    df["returns"] = df["Close"].pct_change().fillna(0.0)
    df["equity_hold"] = (1.0 + df["returns"]).cumprod()
    df["equity_ma"] = (1.0 + df["returns"] * df["position"].shift(1).fillna(0)).cumprod()
    return df


def crossover_dates(df: pd.DataFrame) -> list[tuple[pd.Timestamp, str]]:
    """返回 [(日期, 'buy'/'sell'), ...]。"""
    out: list[tuple[pd.Timestamp, str]] = []
    for date, value in df["signal"].items():
        if value == 1:
            out.append((date, BUY))
        elif value == -1:
            out.append((date, SELL))
    return out
