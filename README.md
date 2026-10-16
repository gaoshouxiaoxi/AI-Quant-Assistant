# AI Quant Assistant

An AI-powered quantitative analysis platform.

## Features

- Stock data collection
- Technical analysis
- Backtesting
- Machine learning prediction

## Tech Stack

- Python
- Pandas
- Machine Learning
- Streamlit

## Getting Started

```bash
# 1. 创建并激活虚拟环境
python -m venv venv
venv\Scripts\activate        # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 抓取数据并保存 CSV
python src/fetch_data.py
python src/fetch_data.py TSLA
python src/fetch_data.py 600519 2020-01-01 2025-01-01   # A 股

# 4. 数据分析：K 线图 + 收益率 + 波动率 + 回撤 + 均线策略信号
python src/analyze.py                       # 默认 AAPL 与 600519
python src/analyze.py AAPL                  # 只分析一个
python src/analyze.py AAPL,600519 2022-06-15 2024-12-31
```

> **注意（Windows）**：本机 `python3` 指向微软商店占位程序，请一律使用 `python`。

## 示例结果

`python src/analyze.py`（区间 2022-06-15 ~ 2024-12-31，前复权日线）：

| 标的 | 年化收益 | 年化波动 | 夏普 | 最大回撤 |
| --- | --- | --- | --- | --- |
| Apple (AAPL) | +27.38% | 25.22% | 1.09 | -28.38% |
| 贵州茅台 (600519) | -5.31% | 28.18% | -0.19 | -37.70% |

指标口径：年化按 252 个交易日复利；夏普比率按无风险利率 0 计算。

### MA5 / MA20 均线策略

规则：**MA5 上穿 MA20 买入（金叉），下穿卖出（死叉）**，信号次日生效，
未计手续费与滑点（正式回测见第 4 周）。

| 标的 | 买入持有 | MA5/MA20 策略 | 金叉 | 死叉 |
| --- | --- | --- | --- | --- |
| Apple (AAPL) | 1.8491 (+84.91%) | 1.4351 (+43.51%) | 13 | 13 |
| 贵州茅台 (600519) | 0.8743 (-12.57%) | 0.6706 (-32.94%) | 21 | 21 |

**结论：这段区间里双均线策略明显跑输买入持有。** 均线是滞后指标，上涨途中反复
被"洗"出场（AAPL 的 13 次金叉多数落在震荡区）；茅台 21 次信号意味着约每 30 个
交易日换一次手，摩擦成本极高。这不是 bug，而是均线策略在趋势市 + 震荡市中的
常见表现，也正是第 4 周正式回测要量化的核心问题。

| K 线图 | 净值与回撤 | 策略信号 | 策略 vs 买入持有 |
| --- | --- | --- | --- |
| ![AAPL K线](charts/AAPL_kline.png) | ![AAPL 净值回撤](charts/AAPL_equity.png) | ![AAPL 信号](charts/AAPL_signals.png) | ![AAPL 策略净值](charts/AAPL_strategy_equity.png) |
| ![600519 K线](charts/600519_kline.png) | ![600519 净值回撤](charts/600519_equity.png) | ![600519 信号](charts/600519_signals.png) | ![600519 策略净值](charts/600519_strategy_equity.png) |

## 踩过的坑

### 1. 未复权数据会造成假暴跌（已在代码中拦截）

AAPL 在 2020-08-31 拆股（4:1），未复权数据会出现 **-74.15%** 的假单日跌幅：

```
Date          Close
2020-08-28    499.23
2020-08-31    129.04   ← 假的 -74.15%，实际是拆股
```

不拦截的话最大回撤会被算成 -78.89%、年化波动 45.90%，之后所有策略回测都是错的。
`data_loader._validate()` 会拒绝单日跌幅超过 50% 的数据源（真实个股极少如此）。

### 2. `ffill` 会把"空仓"当成有效状态，导致持仓只剩信号当天

持仓状态用 `1` 表示持仓、`0` 表示空仓。若把"当天无新信号"也写成 `0`，
`ffill()` 会因为序列中没有 `NaN` 空洞而**完全失效**，持仓退化成只剩金叉当天：

```python
# 错误：无信号日写成 0，ffill 填不动 → 全区间只持仓 13 天
state = pd.Series(0, index=df.index)
# 正确：无信号日用 NaN 表示，空仓仍用 0 → 持仓 350 天
state = pd.Series(float("nan"), index=df.index)
```

这个 bug 曾让 AAPL 策略净值从 1.4351 变成 1.0176（几乎全程空仓），
而且表面数字"看起来正常"，只有把持仓天数与手工推算对照才会暴露。

## 数据源说明

`src/data_loader.py` 采用**多源自动降级 + 体检**策略：

1. 依次尝试 `yfinance`（支持复权、覆盖长）→ 腾讯行情（前复权、较稳）→ `akshare`；
2. 每个源取到数据后必须通过**复权体检**，否则视为不可用并继续降级；
3. 全部失败时报错，并列出每个源的具体原因。

### 为什么需要复权体检

免费数据源里存在**未复权**的坑。真实案例：AAPL 在 2020-08-31 拆股（4:1），
未复权数据会出现一个 **-74.15%** 的假单日跌幅——

```
Date          Close
2020-08-28    499.23
2020-08-31    129.04   ← 假的 -74.15%，实际是拆股
```

如果不拦截，最大回撤会被算成 **-78.89%**，年化波动被算成 45.90%，
基于这种数据的任何策略回测都是错的。因此 `_validate()` 会拒绝单日跌幅
超过 50% 的数据源（真实个股极少出现这种波动）。

### 已知限制

- **腾讯接口单次最多返回 640 根日线**，因此默认分析区间取
  `2022-06-15 ~ 2024-12-31`，这是当前免费源能稳定给到、且能被验证的范围。
- 东方财富系接口（`akshare` 的 `stock_us_hist` / `stock_zh_a_hist`）在部分
  网络环境下会 `RemoteDisconnected`，且本机系统代理会拒绝该域名，
  代码里对 A 股接口做了绕过代理处理，但仍不保证稳定。
- **yfinance 可能被 Yahoo 限流**（`YFRateLimitError: Too Many Requests`）。
- 跨标的对比时，请先核对各自的实际覆盖区间；结论只在重叠区间内成立。

## 项目结构

```
AI-Quant-Assistant/
├── charts/              # 生成的图表（入库，供 README 引用）
├── data/                # 抓取的 CSV（不入库，可重新生成）
├── notebooks/
├── src/
│   ├── data_loader.py   # 多源降级 + 复权体检
│   ├── fetch_data.py    # 第 1 周：数据获取
│   ├── metrics.py       # 收益率 / 波动率 / 夏普 / 最大回撤
│   ├── charts.py        # K 线图、净值与回撤图、策略信号图
│   ├── strategy.py      # 第 3 周：MA5/MA20 金叉死叉信号
│   └── analyze.py       # 分析主程序
├── README.md
└── requirements.txt
```

## Roadmap

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 第 1 周 | 环境 + 股票数据获取 | ✅ 完成 |
| 第 2 周 | 数据分析（K线、收益率、波动率） | ✅ 完成 |
| 第 3 周 | 技术指标（MA5 / MA20 均线策略） | ✅ 完成 |
| 第 4 周 | 回测系统（Backtrader） | ⬜ 未开始 |
| 第 5-6 周 | 机器学习预测（RandomForest → XGBoost） | ⬜ 未开始 |
| 第 7 周 | 网页界面（Streamlit） | ⬜ 未开始 |
| 第 8 周 | 开源包装（文档 + Demo） | ⬜ 未开始 |

## License

MIT
