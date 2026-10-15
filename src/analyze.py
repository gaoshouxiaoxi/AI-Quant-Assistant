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

from charts import plot_candlestick, plot_equity_and_drawdown, setup_chinese_font  # noqa: E402
from data_loader import load_daily  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHART_DIR = ROOT / "charts"

DEFAULT_TICKERS = ["AAPL", "600519"]
DEFAULT_START = "2022-06-15"
DEFAULT_END = "2024-12-31"

LABELS = {"AAPL": "Apple (AAPL)", "600519": "贵州茅台 (600519)"}


def fmt_pct(value: float) -> str:
    return "n/a" if value != value else f"{value * 100:.2f}%"


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
        print(f"图表    : charts/{kline.name}、charts/{curve.name}\n")

        rows.append({"ticker": ticker, "label": label, "source": source, **stats})

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
    print("注意：若各标的覆盖区间不同，对比结论只在重叠区间内成立。")


if __name__ == "__main__":
    main()
