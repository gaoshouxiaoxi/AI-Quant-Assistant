"""用 Backtrader 独立复算双均线策略，与 src/backtest.py 交叉验证。

**验证结论（2026-10 实测）**

两个引擎的结果不能逐笔完全对齐，差异来自 **Backtrader 的订单生命周期**：
在 `next()` 里下的单，其平仓动作要到下一根 bar 才结算，而我们手写引擎的语义是
"目标日收盘价成交"。因此成交日期会差 1~3 个交易日，总资产差约 3~5%。

已验证一致的部分：
- 成交**次数**一致（手写 12 笔往返 = 2×12 次事件 + 1 次未平仓买入 = 25，与 BT 相符）
- 成交**价格**取自同一天的真实收盘价（例如 2022-11-08 close = 139.50）

这个脚本的作用是**防腐**：以后有人改动了 backtest.py 的成交逻辑，
两边的差异会明显放大，从而被这里发现。

用法：
    python src/backtrader_check.py AAPL
    python src/backtrader_check.py 600519
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows 控制台默认 GBK，遇到 ✅/❌ 这类字符会抛 UnicodeEncodeError，
# 而异常发生在 print 里时，之前的输出还留在缓冲区，表现为"退出码 0 但零输出"。
# 这里强制 UTF-8，保证诊断信息一定能打出来。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from backtest import backtest_ma  # noqa: E402
from data_loader import load_daily  # noqa: E402

import backtrader as bt  # noqa: E402

TOLERANCE = 0.06  # 6%：超过说明逻辑有实质变化，需要人工复核


class SignalStrategy(bt.Strategy):
    """按预设目标仓位调仓：只做多，全仓进出。

    两点 Backtrader 特有的坑，都在这里踩过：
    1. 信号表不能用 `params` 传 dict——参数元类会把它转成 None，
       报 "'NoneType' object has no attribute 'addindicator'"。
    2. 不能用 `if not self.position`——Backtrader 对象重载了 __bool__，
       会触发 LineOwnOperation 并报同样的错。必须用 `.position.size`。
    """

    targets_map: dict = {}
    dates: list = []

    def __init__(self):
        self.equity_log: list[float] = []
        self.fill_log: list[tuple] = []

    def notify_order(self, order):
        if order.status == order.Completed:
            self.fill_log.append((
                self.datas[0].datetime.date(0),
                "buy" if order.isbuy() else "sell",
                float(order.executed.price),
            ))

    def next(self):
        i = len(self) - 1
        if i + 1 >= len(SignalStrategy.dates):
            self.equity_log.append(self.broker.getvalue())
            return
        # 提前一根 bar 下单，使成交落在目标日期
        target = SignalStrategy.targets_map.get(SignalStrategy.dates[i + 1])
        if target == 1 and self.position.size == 0:
            size = int(self.broker.getcash() / self.data.close[0])
            if size > 0:
                self.buy(size=size)
        elif target == 0 and self.position.size > 0:
            self.close()
        self.equity_log.append(self.broker.getvalue())


def run_backtrader(close: pd.Series, targets: dict,
                   initial_cash: float = 100_000.0,
                   fee_rate: float = 0.0003,
                   slippage: float = 0.0002) -> dict:
    SignalStrategy.targets_map = targets
    SignalStrategy.dates = list(close.index.date)

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(SignalStrategy)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.set_coo(True)          # 下一根 bar 开盘成交
    cerebro.broker.setcommission(commission=fee_rate)
    cerebro.broker.set_slippage_perc(perc=slippage)

    # open 设为与 close 相同，使"下一根 bar 开盘"等价于"目标日收盘"
    frame = pd.DataFrame({
        "open": close.values, "high": close.values,
        "low": close.values, "close": close.values,
        "volume": 1_000_000,
    }, index=close.index)
    frame.index.name = "date"

    cerebro.adddata(bt.feeds.PandasData(dataname=frame))
    cerebro.run()

    strat = cerebro.runstrats[0][0]
    return {"final_equity": float(cerebro.broker.getvalue()),
            "fills": strat.fill_log,
            "events": len(strat.fill_log)}


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    start = sys.argv[2] if len(sys.argv) > 2 else "2022-06-15"
    end = sys.argv[3] if len(sys.argv) > 3 else "2024-12-31"

    stock, source = load_daily(ticker, start, end)
    close = stock["Close"]

    fee, slip = 0.0003, 0.0002
    mine = backtest_ma(close, 5, 20, 100_000.0, fee, slip)
    stats = mine["stats"]
    targets = dict(zip(close.index.date,
                       mine["frame"]["target"].astype(int).tolist()))

    print(f"标的     : {ticker}   数据源: {source}   "
          f"{close.index.min().date()} ~ {close.index.max().date()}")
    print(f"参数     : MA5/MA20  初始资金 100000  手续费 {fee:.4%}  滑点 {slip:.4%}")
    print()

    bt_result = run_backtrader(close, targets, 100_000.0, fee, slip)
    bt_final = bt_result["final_equity"]
    rel = abs(bt_final - stats["final_equity"]) / stats["final_equity"]
    expected_events = stats["trade_count"] * 2 + (1 if stats["open_position"] else 0)

    print(f"{'指标':<20}{'手写引擎':>14}{'Backtrader':>14}")
    print("-" * 50)
    print(f"{'最终资产':<20}{stats['final_equity']:>14.2f}{bt_final:>14.2f}")
    print(f"{'总收益率':<20}{stats['total_return']:>13.2%}{bt_final / 100_000 - 1:>13.2%}")
    print(f"{'成交事件数':<20}{expected_events:>14}{bt_result['events']:>14}")
    print(f"{'相对偏差':<20}{'':>14}{rel:>13.2%}")
    print()
    if bt_result["events"] != expected_events:
        print("[WARN] 成交次数不一致——成交逻辑可能被改动，请人工复核")
    elif rel <= TOLERANCE:
        print(f"[OK] 偏差 {rel:.2%} 在容差 {TOLERANCE:.0%} 内，"
              "差异可由订单结算时序解释")
    else:
        print(f"[WARN] 偏差 {rel:.2%} 超出容差 {TOLERANCE:.0%}，请检查成交逻辑")

    print()
    print("--- 手写引擎完整指标")
    for key in ("final_equity", "total_return", "annualized_return",
                "annualized_volatility", "sharpe", "max_drawdown",
                "benchmark_return", "excess_return", "trade_count",
                "win_rate", "avg_holding_days", "open_position"):
        value = stats.get(key)
        if isinstance(value, float) and value == value:
            if key in ("total_return", "annualized_return", "annualized_volatility",
                       "max_drawdown", "benchmark_return", "excess_return", "win_rate"):
                print(f"  {key:<24}{value:>10.2%}")
            else:
                print(f"  {key:<24}{value:>10.4f}")
        else:
            print(f"  {key:<24}{value}")


if __name__ == "__main__":
    main()
