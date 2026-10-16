"""第 2 周：数据分析（K 线图 + 收益率 + 波动率 + 回撤）。

用法：
    python src/analyze.py                          # AAPL 与 600519
    python src/analyze.py AAPL                     # 只分析一个
    python src/analyze.py AAPL,600519 2022-06-15 2024-12-31

关于数据区间：默认区间是 2022-06-15 ~ 2024-12-31，这是当前免费数据源
（腾讯前复权、单次 640 根日线）实际能稳定给到的范围。yfinance 若可用会自动
覆盖更长的区间，脚本会打印每个标的的真实覆盖区间，请注意核对。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from charts import (  # noqa: E402
    plot_candlestick,
    plot_equity_and_drawdown,
    plot_signals,
    plot_strategy_equity,
    setup_chinese_font,
)
from data_loader import load_daily  # noqa: E402
from metrics import summarize  # noqa: E402
from strategy import crossover_dates, generate_signals  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHART_DIR = ROOT / "charts"

DEFAULT_TICKERS = ["AAPL", "600519"]
DEFAULT_START = "2022-06-15"
DEFAULT_END = "2024-12-31"

FAST, SLOW = 5, 20

LABELS = {"AAPL": "Apple (AAPL)", "600519": "贵州茅台 (600519)"}


def fmt_pct(value: float) -> str:
    return "n/a" if value != value else f"{value * 100:.2f}%"


def open_position_return(df) -> float:
    """若期末仍持仓，给出该笔持仓的浮动收益（仅供展示，不是已实现收益）。"""
    buys = df.index[df["signal"] == 1]
    sells = df.index[df["signal"] == -1]
    last_buy = buys[-1] if len(buys) else None
    last_sell = sells[-1] if len(sells) else None
    if last_buy is None:
        return float("nan")
    if last_sell is not None and last_sell > last_buy:
        return float("nan")
    return float(df.loc[df.index[-1], "Close"] / df.loc[last_buy, "Close"] - 1.0)


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) >= 3:
        tickers = argv[0].split(",")
        start, end = argv[1], argv[2]
    elif len(argv) == 1:
        tickers, start, end = argv[0].split(","), DEFAULT_START, DEFAULT_END
    else:
        tickers, start, end = DEFAULT_TICKERS, DEFAULT_START, DEFAULT_END

    font = setup_chinese_font()
    print(f"中文字体: {font or '未找到（图上中文可能显示为方框）'}")
    print(f"请求区间: {start} ~ {end}\n")

    rows: list[dict] = []

    for ticker in tickers:
        label = LABELS.get(ticker, ticker)
        print("=" * 64)
        print(f"{label}")
        print("=" * 64)

        stock, source = load_daily(ticker, start, end)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        stock.to_csv(DATA_DIR / f"{ticker}_{start}_{end}.csv")

        stats = summarize(stock["Close"])
        print(f"数据源  : {source}")
        print(f"覆盖区间: {stats['start']} ~ {stats['end']}（{stats['days']} 个交易日）")
        print(f"区间收益: {fmt_pct(stats['total_return'])}")
        print(f"年化收益: {fmt_pct(stats['annualized_return'])}")
        print(f"年化波动: {fmt_pct(stats['annualized_volatility'])}")
        print(f"夏普比率: {stats['sharpe']:.2f}")
        print(f"最大回撤: {fmt_pct(stats['max_drawdown'])}")
        print("参考口径: 年化按 252 交易日复利；夏普按无风险利率 0；")
        print("         结论仅基于上述覆盖区间，不代表更长历史。")

        kline = plot_candlestick(stock, f"{label} K线图（MA5 / MA20）",
                                 CHART_DIR / f"{ticker}_kline.png")
        curve = plot_equity_and_drawdown(stock, f"{label} 净值与回撤",
                                         CHART_DIR / f"{ticker}_equity.png")

        # ---- 第 3 周：均线策略信号 ----
        sig = generate_signals(stock, FAST, SLOW)
        events = crossover_dates(sig)
        buys = [d for d, kind in events if kind == "buy"]
        sells = [d for d, kind in events if kind == "sell"]

        print()
        print(f"MA{FAST}/MA{SLOW} 策略")
        print(f"  金叉(买入): {len(buys)} 次   死叉(卖出): {len(sells)} 次")
        if events:
            shown = ", ".join(f"{d.date()}{'买' if k == 'buy' else '卖'}"
                              for d, k in events[:8])
            print(f"  信号日期  : {shown}{' ...' if len(events) > 8 else ''}")
        print(f"  期末状态  : {'持仓中' if sig['position'].iloc[-1] else '空仓'}")

        equity_hold = float(sig['equity_hold'].iloc[-1])
        equity_ma = float(sig['equity_ma'].iloc[-1])
        open_ret = open_position_return(sig)
        print(f"  买入持有  : 净值 {equity_hold:.4f}（{fmt_pct(equity_hold - 1)}）")
        print(f"  均线策略  : 净值 {equity_ma:.4f}（{fmt_pct(equity_ma - 1)}）")
        if open_ret == open_ret:
            print(f"  期末浮动  : {fmt_pct(open_ret)}（该笔尚未平仓）")
        print("  口径说明  : 信号次日生效，未计手续费与滑点；正式回测见第 4 周。")

        sig_chart = plot_signals(sig, f"{label} MA{FAST}/MA{SLOW} 均线策略信号",
                                 CHART_DIR / f"{ticker}_signals.png", FAST, SLOW)
        eq_chart = plot_strategy_equity(sig, f"{label} 策略净值 vs 买入持有",
                                        CHART_DIR / f"{ticker}_strategy_equity.png")
        print(f"图表    : charts/{kline.name}、charts/{curve.name}、"
              f"charts/{sig_chart.name}、charts/{eq_chart.name}\n")

        rows.append({"ticker": ticker, "label": label, "source": source,
                     "buys": len(buys), "sells": len(sells),
                     "equity_hold": equity_hold, "equity_ma": equity_ma,
                     **stats})

    print("=" * 64)
    print("对比汇总")
    print("=" * 64)
    print(f"{'标的':<18}{'年化收益':>10}{'年化波动':>10}{'夏普':>8}{'最大回撤':>10}")
    print("-" * 64)
    for r in rows:
        print(f"{r['label']:<18}{fmt_pct(r['annualized_return']):>10}"
              f"{fmt_pct(r['annualized_volatility']):>10}"
              f"{r['sharpe']:>8.2f}{fmt_pct(r['max_drawdown']):>10}")

    print()
    print("=" * 64)
    print(f"MA{FAST}/MA{SLOW} 策略对比（信号次日生效，未计费用）")
    print("=" * 64)
    print(f"{'标的':<18}{'买入持有':>10}{'均线策略':>10}{'金叉':>6}{'死叉':>6}")
    print("-" * 64)
    for r in rows:
        print(f"{r['label']:<18}{r['equity_hold']:>10.4f}{r['equity_ma']:>10.4f}"
              f"{r['buys']:>6}{r['sells']:>6}")
    print()
    print("注意：均线策略净值未计手续费与滑点，也未做参数优化；")
    print("     这是策略逻辑的验证，不是可直接交易的结果。")
    print("     若各标的覆盖区间不同，对比结论只在重叠区间内成立。")


if __name__ == "__main__":
    main()
