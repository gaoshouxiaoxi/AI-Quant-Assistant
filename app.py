"""AI Quant Assistant —— Streamlit 界面（第 7 周）。

把前 6 周验证过的能力包装成可以直接操作的产品：
- 行情与风险指标（年化收益、波动、夏普、最大回撤）
- K 线 + MA5/MA20 与金叉死叉信号
- 双均线策略回测（含手续费、滑点、基准对比）
- 波动率预测（Yang-Zhang 目标 + HAR 结构，带样本外记录）
- 明确的"能预测什么、不能预测什么"说明

运行：
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from backtest import backtest_ma  # noqa: E402
from charts import setup_chinese_font  # noqa: E402
from data_loader import load_daily  # noqa: E402
from metrics import summarize  # noqa: E402
from rv_estimators import (TRADING_DAYS, rolling_vol_yang_zhang)  # noqa: E402
from strategy import crossover_dates, generate_signals  # noqa: E402

DATA_DIR = ROOT / "data"
FONT = setup_chinese_font()

LABELS = {
    "600519": "贵州茅台 (600519)",
    "000858": "五粮液 (000858)",
    "000001": "平安银行 (000001)",
    "AAPL": "Apple (AAPL)",
}

st.set_page_config(page_title="AI Quant Assistant", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- 数据

def available_tickers() -> list[str]:
    """从 data/ 里已缓存的 CSV 推断可用标的。"""
    found = []
    for p in DATA_DIR.glob("*_*.csv"):
        name = p.name.split("_")[0]
        if name and name not in found:
            found.append(name)
    return sorted(found)


@st.cache_data(show_spinner=False)
def get_prices(ticker: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """读取行情。优先本地缓存，避免每次刷新都打网络接口。"""
    df, source = load_daily(ticker, start, end, cache_dir=DATA_DIR)
    return df, source


@st.cache_data(show_spinner=False)
def har_forecast(close: pd.Series, high: pd.Series, low: pd.Series,
                 open_: pd.Series, horizon: int = 5) -> dict:
    """用 HAR 结构预测未来 horizon 日的平均波动率。

    做法：以 Yang-Zhang 日方差为 RV 代理，构造 HAR 三尺度特征，
    用扩展窗口前向验证给出样本外记录，最后用全量数据拟合做当期预测。
    """
    from sklearn.linear_model import LinearRegression

    yz = rolling_vol_yang_zhang(open_, high, low, close, 20)
    rv = (yz / np.sqrt(TRADING_DAYS)) ** 2
    rv = rv.replace(0, np.nan)

    feat = pd.DataFrame({
        "rv": rv,
        "rv_week": rv.rolling(5).mean(),
        "rv_month": rv.rolling(22).mean(),
    })
    feat["target"] = rv.rolling(horizon).mean().shift(-horizon)
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    feat = feat[(feat["rv"] > 0) & (feat["target"] > 0)]
    if len(feat) < 120:
        return {"ok": False, "reason": f"样本不足（{len(feat)} 天）"}

    X = np.log(feat[["rv", "rv_week", "rv_month"]].to_numpy())
    y = np.log(feat["target"].to_numpy())

    # 样本外记录（3 折扩展窗口）
    n = len(X)
    start = int(n * 0.5)
    step = max(1, (n - start) // 3)
    preds, trues = [], []
    for k in range(3):
        b = start + k * step
        end = n if k == 2 else b + step
        if b < 60 or end <= b:
            continue
        model = LinearRegression().fit(X[:b], y[:b])
        preds.extend(model.predict(X[b:end]).tolist())
        trues.extend(y[b:end].tolist())
    oos_r2 = float("nan")
    if preds:
        preds_a, trues_a = np.array(preds), np.array(trues)
        ss_res = float(np.sum((trues_a - preds_a) ** 2))
        ss_tot = float(np.sum((trues_a - trues_a.mean()) ** 2))
        oos_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # 当期预测：用全量数据拟合
    model = LinearRegression().fit(X, y)
    last = np.log(feat[["rv", "rv_week", "rv_month"]].iloc[-1].to_numpy())
    forecast = float(np.exp(model.predict(last.reshape(1, -1))[0]))
    return {
        "ok": True,
        "current": float(np.sqrt(feat["rv"].iloc[-1] * TRADING_DAYS)),
        "forecast": float(np.sqrt(forecast * TRADING_DAYS)),
        "oos_r2": oos_r2,
        "n_oos": len(preds),
        "n_total": n,
        "history": feat,
    }


# ---------------------------------------------------------------- 绘图

def fig_kline(df: pd.DataFrame, sig: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(df.index, df["Close"], color="#37474f", linewidth=1.0, label="收盘价")
    ax.plot(sig.index, sig["MA5"], color="#1565c0", linewidth=1.2, label="MA5")
    ax.plot(sig.index, sig["MA20"], color="#ef6c00", linewidth=1.2, label="MA20")
    buys = sig[sig["signal"] == 1]
    sells = sig[sig["signal"] == -1]
    ax.scatter(buys.index, buys["Close"], marker="^", s=55, color="#d32f2f",
               zorder=5, label=f"金叉 ({len(buys)})")
    ax.scatter(sells.index, sells["Close"], marker="v", s=55, color="#2e7d32",
               zorder=5, label=f"死叉 ({len(sells)})")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_equity(close: pd.Series, title: str):
    equity = close / close.iloc[0]
    dd = close / close.cummax() - 1
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(equity.index, equity, color="#1565c0", linewidth=1.3)
    ax1.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_title(title)
    ax1.set_ylabel("净值")
    ax1.grid(alpha=0.25)
    ax2.fill_between(dd.index, dd, 0, color="#c62828", alpha=0.35)
    ax2.set_ylabel("回撤")
    ax2.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_vol(history: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    vol = np.sqrt(history["rv"] * TRADING_DAYS)
    ax.plot(vol.index, vol, color="#212121", linewidth=1.1, label="20 日 Yang-Zhang 波动")
    ax.set_title(title)
    ax.set_ylabel("年化波动率")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 主界面

def main() -> None:
    st.title("📈 AI Quant Assistant")
    st.caption("AI 辅助量化分析平台 · 数据、指标、回测、波动率预测")

    tickers = available_tickers()
    if not tickers:
        st.error("data/ 目录下没有缓存行情。请先运行 `python src/fetch_data.py 600519`。")
        return

    with st.sidebar:
        st.header("参数")
        default = tickers.index("600519") if "600519" in tickers else 0
        ticker = st.selectbox("标的", tickers, index=default,
                              format_func=lambda t: LABELS.get(t, t))
        start = st.date_input("开始日期", value=pd.Timestamp("2022-06-15"))
        end = st.date_input("结束日期", value=pd.Timestamp("2024-12-31"))
        st.divider()
        st.subheader("回测参数")
        cash = st.number_input("初始资金", value=100_000, step=10_000)
        fee = st.number_input("手续费率", value=0.0003, format="%.4f", step=0.0001)
        slip = st.number_input("滑点率", value=0.0002, format="%.4f", step=0.0001)
        st.divider()
        st.caption(f"数据源：本地缓存 + 网络降级\n标的数：{len(tickers)}")

    try:
        df, source = get_prices(ticker, str(start), str(end))
    except Exception as exc:  # noqa: BLE001
        st.error(f"数据获取失败：{type(exc).__name__}: {exc}")
        return

    close = df["Close"]
    stats = summarize(close)
    sig = generate_signals(df, 5, 20)

    tab_overview, tab_strategy, tab_vol, tab_about = st.tabs(
        ["概览", "均线策略", "波动率预测", "说明与局限"])

    # ---- 概览 ----
    with tab_overview:
        st.subheader(f"{LABELS.get(ticker, ticker)}")
        st.caption(f"数据源 {source} · {stats['start']} ~ {stats['end']} "
                   f"· {stats['days']} 个交易日")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("区间收益", f"{stats['total_return']:+.2%}")
        c2.metric("年化收益", f"{stats['annualized_return']:+.2%}")
        c3.metric("年化波动", f"{stats['annualized_volatility']:.2%}")
        c4.metric("夏普比率", f"{stats['sharpe']:.2f}")
        c5.metric("最大回撤", f"{stats['max_drawdown']:.2%}")

        st.pyplot(fig_kline(df, sig, "K 线与金叉/死叉信号"), width="stretch")

    # ---- 均线策略 ----
    with tab_strategy:
        st.subheader("MA5 / MA20 双均线策略")
        st.caption("金叉买入、死叉卖出，信号次日生效；含手续费与滑点。")

        events = crossover_dates(sig)
        result = backtest_ma(close, 5, 20, float(cash), float(fee), float(slip))
        s = result["stats"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最终资产", f"{s['final_equity']:,.0f}")
        c2.metric("策略收益", f"{s['total_return']:+.2%}")
        c3.metric("买入持有", f"{s['benchmark_return']:+.2%}")
        c4.metric("超额", f"{s['excess_return']:+.2%}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("已平仓笔数", s["trade_count"])
        c6.metric("胜率", f"{s['win_rate']:.1%}" if s["win_rate"] == s["win_rate"] else "—")
        c7.metric("平均持仓天数", f"{s['avg_holding_days']:.0f}"
                  if s["avg_holding_days"] == s["avg_holding_days"] else "—")
        c8.metric("期末持仓", "是" if s["open_position"] else "否")

        st.pyplot(fig_equity(close, "净值与回撤（买入持有）"), width="stretch")

        with st.expander("最近 10 次信号"):
            if events:
                rows = [{"日期": d.date(), "方向": "买入" if k == "buy" else "卖出"}
                        for d, k in events[-10:]]
                st.table(pd.DataFrame(rows))
            else:
                st.write("区间内无信号")

    # ---- 波动率预测 ----
    with tab_vol:
        st.subheader("波动率预测（HAR 结构）")
        st.caption("以 Yang-Zhang 日方差为 RV 代理，用日/周/月三尺度做线性预测。")

        fc = har_forecast(close, df["High"], df["Low"], df["Open"])
        if not fc["ok"]:
            st.warning(fc["reason"])
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("当前波动（20日）", f"{fc['current']:.2%}")
            c2.metric("未来 5 日预测", f"{fc['forecast']:.2%}",
                      delta=f"{(fc['forecast'] - fc['current']) * 100:+.2f} pct")
            r2 = fc["oos_r2"]
            c3.metric("样本外 R²", f"{r2:.3f}" if r2 == r2 else "—",
                      help="3 折扩展窗口的样本外拟合优度")

            st.pyplot(fig_vol(fc["history"], "历史波动率（20 日 Yang-Zhang）"),
                      width="stretch")
            st.info("HAR 擅长预测波动的**水平**，不擅长预测**尖峰**。"
                    "按研究结论（AUC 0.506），突发事件导致的跳跃不可提前识别，"
                    "建议对预测值留 1.5~2 倍安全边际。")

    # ---- 说明与局限 ----
    with tab_about:
        st.subheader("这个工具能做什么、不能做什么")
        st.markdown("""
| 目标 | 可预测性 | 结论 |
| --- | --- | --- |
| 涨跌方向 | AUC ≈ 0.50 | ❌ 与随机猜无异 |
| 波动率水平 | R² ≈ 0.77，Skill +16% | ✅ 可预测 |
| 高波动 regime | AUC ≈ 0.93 | ✅ 高度可预测 |
| 波动尖峰 | AUC ≈ 0.506 | ❌ 不可提前识别 |

**核心结论：能估"风险有多大"，不能估"什么时候涨跌、什么时候出事"。**
""")
        st.markdown("""
#### 方法论要点
- 一律使用**扩展窗口前向验证**，不做随机切分（避免未来信息泄漏）
- 标准化放在每折内部完成
- 每个结论都配**基线对照**（持续性、多数派、随机）
- 数据经过**复权体检**，拦截拆股造成的假暴跌
""")
        st.warning("本工具仅用于研究与教学，不构成任何投资建议。")


if __name__ == "__main__":
    main()
