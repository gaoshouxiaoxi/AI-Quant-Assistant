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

# 3. 抓取数据（默认 AAPL 2020-2025）
python src/fetch_data.py
python src/fetch_data.py TSLA
python src/fetch_data.py 600519 2020-01-01 2025-01-01   # A 股
```

> **注意（Windows）**：本机 `python3` 指向微软商店占位程序，请一律使用 `python`。

### 数据源说明

`src/data_loader.py` 采用**自动降级**策略：优先 `yfinance`，一旦被 Yahoo 限流
（`YFRateLimitError: Too Many Requests`）就自动切换到腾讯行情接口，调用方无感。
两个源都失败才会报错，并列出各自的具体原因。

## Roadmap

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 第 1 周 | 环境 + 股票数据获取 | 🚧 进行中 |
| 第 2 周 | 数据分析（K线、收益率、波动率） | ⬜ 未开始 |
| 第 3 周 | 技术指标（MA5 / MA20 均线策略） | ⬜ 未开始 |
| 第 4 周 | 回测系统（Backtrader） | ⬜ 未开始 |
| 第 5-6 周 | 机器学习预测（RandomForest → XGBoost） | ⬜ 未开始 |
| 第 7 周 | 网页界面（Streamlit） | ⬜ 未开始 |
| 第 8 周 | 开源包装（文档 + Demo） | ⬜ 未开始 |

## License

MIT
