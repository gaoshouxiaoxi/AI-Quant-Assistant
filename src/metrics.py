"""风险收益指标计算。

全部基于日线收盘价，统一口径：
- 日收益率 = 收盘价的百分比变化
- 年化收益：按 252 个交易日复利年化
- 年化波动率：日收益率标准差 × sqrt(252)
- 夏普比率：按无风险利率 0 计算（即 年化收益 / 年化波动率）
- 最大回撤：净值从历史最高点的最大跌幅（负数）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(close: pd.Series) -> pd.Series:
    """日收益率序列。"""
    return close.pct_change().dropna()


def total_return(close: pd.Series) -> float:
    """区间总收益率。"""
    if len(close) < 2:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[0] - 1.0)


def annualized_return(close: pd.Series) -> float:
    """年化收益率。"""
    n = len(close)
    if n < 2:
        return float("nan")
    years = n / TRADING_DAYS
    if years <= 0:
        return float("nan")
    return float((close.iloc[-1] / close.iloc[0]) ** (1.0 / years) - 1.0)


def annualized_volatility(close: pd.Series) -> float:
    """年化波动率。"""
    r = daily_returns(close)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(close: pd.Series, risk_free: float = 0.0) -> float:
    """夏普比率（默认无风险利率为 0）。"""
    vol = annualized_volatility(close)
    if not np.isfinite(vol) or vol == 0:
        return float("nan")
    return float((annualized_return(close) - risk_free) / vol)


def max_drawdown(close: pd.Series) -> float:
    """最大回撤（负数，例如 -0.42 表示最大跌去 42%）。"""
    if len(close) < 2:
        return float("nan")
    running_max = close.cummax()
    drawdown = close / running_max - 1.0
    return float(drawdown.min())


def drawdown_series(close: pd.Series) -> pd.Series:
    """回撤曲线，用于画图。"""
    return close / close.cummax() - 1.0


def equity_curve(close: pd.Series) -> pd.Series:
    """归一化净值曲线（起点为 1.0）。"""
    if len(close) == 0:
        return close
    return close / close.iloc[0]


def summarize(close: pd.Series) -> dict[str, float]:
    """一次性算出所有指标。"""
    return {
        "start": close.index.min().date(),
        "end": close.index.max().date(),
        "days": len(close),
        "total_return": total_return(close),
        "annualized_return": annualized_return(close),
        "annualized_volatility": annualized_volatility(close),
        "sharpe": sharpe_ratio(close),
        "max_drawdown": max_drawdown(close),
    }
