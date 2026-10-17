"""第 4 周：回测引擎（手写版，作为基准实现）。

设计要点（与常见错误对照）：

1. **无前瞻偏差**：第 t 日收盘后才知道是否交叉，成交发生在第 t+1 日收盘。
   回测里"用当天收盘价判断、又用当天收盘价成交"是最常见也最致命的高估来源。
2. **滑点与手续费分开建模**：
   买入成交价 = close × (1 + slippage)，再按成交额收手续费 fee_rate
   卖出成交价 = close × (1 - slippage)，再按成交额收手续费 fee_rate
   不会出现"在含滑点的价格上再收一次滑点"的问题。
3. **整股成交**：A 股 100 股一手、美股 1 股起，这里按 1 股取整，避免出现
   0.37 股这种不可成交的数量。
4. **同时给出基准**：买入持有，否则无法判断策略好坏。
5. **统计交易明细**：每笔盈亏、持仓天数、胜率。
"""

from __future__ import annotations

import pandas as pd

TRADING_DAYS = 252


def backtest_ma(close: pd.Series, fast: int = 5, slow: int = 20,
                initial_cash: float = 100_000.0,
                fee_rate: float = 0.0003,
                slippage: float = 0.0002) -> dict:
    """双均线策略回测。

    参数
    ----
    close : 收盘价序列（索引为日期，升序）
    fast, slow : 快慢均线周期
    initial_cash : 初始资金
    fee_rate : 手续费率（按成交金额）
    slippage : 滑点率（买入抬高、卖出压低成交价）

    返回 dict：equity（逐日总资产）、trades（成交明细）、stats（汇总指标）
    """
    if len(close) < slow + 2:
        raise ValueError(f"数据太短：至少需要 {slow + 2} 根日线")

    df = pd.DataFrame({"close": close.astype(float)})
    df["ma_fast"] = df["close"].rolling(fast).mean()
    df["ma_slow"] = df["close"].rolling(slow).mean()

    diff = df["ma_fast"] - df["ma_slow"]
    df["cross_up"] = (diff > 0) & (diff.shift(1) <= 0)
    df["cross_down"] = (diff < 0) & (diff.shift(1) >= 0)

    # 目标持仓：金叉后持有，死叉后空仓；第 t 日信号 → 第 t+1 日执行
    state = pd.Series(float("nan"), index=df.index)
    state[df["cross_up"].fillna(False)] = 1.0
    state[df["cross_down"].fillna(False)] = 0.0
    df["target"] = state.shift(1).ffill().fillna(0.0).astype(int)

    cash = float(initial_cash)
    shares = 0
    equity, trades = [], []
    entry_price = entry_date = None
    entry_cost = 0.0

    for date, row in df.iterrows():
        price = float(row["close"])
        target = int(row["target"])
        if pd.isna(row["ma_slow"]):
            equity.append(cash)
            continue

        # 目标从 0 → 1：按收盘价加滑点买入
        if target == 1 and shares == 0:
            fill_price = price * (1.0 + slippage)
            qty = int(cash // (fill_price * (1.0 + fee_rate)))
            if qty > 0:
                gross = qty * fill_price
                fee = gross * fee_rate
                cash -= gross + fee
                shares = qty
                entry_price, entry_date, entry_cost = fill_price, date, gross + fee

        # 目标从 1 → 0：按收盘价减滑点卖出
        elif target == 0 and shares > 0:
            fill_price = price * (1.0 - slippage)
            gross = shares * fill_price
            fee = gross * fee_rate
            proceeds = gross - fee
            cash += proceeds
            pnl = proceeds - entry_cost
            trades.append({
                "entry_date": entry_date, "exit_date": date,
                "entry_price": entry_price, "exit_price": fill_price,
                "shares": shares, "pnl": pnl,
                "return": pnl / entry_cost if entry_cost else float("nan"),
                "holding_days": (date - entry_date).days,
            })
            shares = 0
            entry_price = entry_date = None
            entry_cost = 0.0

        equity.append(cash + shares * price)

    equity = pd.Series(equity, index=df.index, name="equity")

    # 期末仍持仓：记录为未平仓，不计入已实现交易
    open_position = shares > 0
    if open_position:
        trades_open = {
            "entry_date": entry_date, "exit_date": None,
            "entry_price": entry_price, "exit_price": float(df["close"].iloc[-1]),
            "shares": shares,
            "pnl": float(df["close"].iloc[-1]) * shares - entry_cost,
            "return": (float(df["close"].iloc[-1]) * shares - entry_cost) / entry_cost,
            "holding_days": (df.index[-1] - entry_date).days,
            "open": True,
        }
    else:
        trades_open = None

    stats = _stats(equity, trades, initial_cash, close)
    stats["open_position"] = open_position
    stats["open_trade"] = trades_open
    stats["trade_count"] = len(trades)
    stats["win_rate"] = (sum(1 for t in trades if t["pnl"] > 0) / len(trades)
                         if trades else float("nan"))
    stats["avg_holding_days"] = (sum(t["holding_days"] for t in trades) / len(trades)
                                 if trades else float("nan"))
    return {"equity": equity, "trades": trades, "stats": stats, "frame": df}


def _stats(equity: pd.Series, trades: list[dict], initial_cash: float,
           close: pd.Series) -> dict:
    final = float(equity.iloc[-1])
    total_ret = final / initial_cash - 1.0
    years = len(equity) / TRADING_DAYS
    ann_ret = (final / initial_cash) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    rets = equity.pct_change().dropna()
    ann_vol = float(rets.std(ddof=1) * (TRADING_DAYS ** 0.5)) if len(rets) > 1 else float("nan")
    drawdown = equity / equity.cummax() - 1.0
    bench_ret = float(close.iloc[-1] / close.iloc[0] - 1.0)
    return {
        "initial_cash": initial_cash,
        "final_equity": final,
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe": (ann_ret / ann_vol) if ann_vol and ann_vol == ann_vol and ann_vol != 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "benchmark_return": bench_ret,
        "excess_return": total_ret - bench_ret,
        "trading_days": len(equity),
    }
