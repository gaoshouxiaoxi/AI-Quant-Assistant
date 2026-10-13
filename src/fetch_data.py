"""第 1 周：股票数据获取。

用法：
    python src/fetch_data.py                      # 默认 AAPL 2020-2025
    python src/fetch_data.py TSLA                 # 换股票代码
    python src/fetch_data.py 600519 2020-01-01 2025-01-01   # 也可以抓 A 股
"""

import sys
from pathlib import Path

# 允许直接以脚本方式运行（python src/fetch_data.py）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_daily  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    start = sys.argv[2] if len(sys.argv) > 2 else "2020-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2025-01-01"

    print(f"正在获取 {ticker} 从 {start} 到 {end} ...")
    stock, source = load_daily(ticker, start, end)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"{ticker}_{start}_{end}.csv"
    stock.to_csv(out)

    print(f"数据源: {source}")
    print(f"共 {len(stock)} 个交易日\n")
    print(stock.head())
    print(f"\n已保存到 {out}")


if __name__ == "__main__":
    main()
