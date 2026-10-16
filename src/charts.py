"""绘图：K 线图（含 MA5 / MA20）与净值、回撤曲线。

中文字体：优先使用 Windows 自带字体（微软雅黑 / 黑体），找不到就退回默认字体
并给出提示，避免图上出现一排方框。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import pandas as pd

matplotlib.use("Agg")  # 无界面环境也能出图
import matplotlib.pyplot as plt  # noqa: E402

CANDIDATE_FONTS = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
UP_COLOR = "#d32f2f"     # 中国市场习惯：红涨
DOWN_COLOR = "#2e7d32"   # 绿跌


def setup_chinese_font() -> str | None:
    """配置中文字体，返回实际使用的字体名（找不到则返回 None）。"""
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in CANDIDATE_FONTS:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


def plot_candlestick(stock: pd.DataFrame, title: str, out_path: Path,
                     ma_windows: tuple[int, ...] = (5, 20)) -> Path:
    """画 K 线图 + 均线。"""
    df = stock.copy()
    for w in ma_windows:
        df[f"MA{w}"] = df["Close"].rolling(w).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.6
    for date, row in df.iterrows():
        color = UP_COLOR if row["Close"] >= row["Open"] else DOWN_COLOR
        ax.vlines(date, row["Low"], row["High"], color=color, linewidth=0.7)
        bottom = min(row["Open"], row["Close"])
        height = abs(row["Close"] - row["Open"])
        ax.add_patch(plt.Rectangle(
            (mdates.date2num(date) - width / 2, bottom), width,
            height if height > 0 else 1e-6,
            facecolor=color, edgecolor=color, linewidth=0.5,
        ))

    palette = {5: "#1565c0", 20: "#ef6c00"}
    for w in ma_windows:
        ax.plot(df.index, df[f"MA{w}"], label=f"MA{w}",
                linewidth=1.2, color=palette.get(w))

    ax.set_title(title)
    ax.set_ylabel("价格")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_equity_and_drawdown(stock: pd.DataFrame, title: str,
                             out_path: Path) -> Path:
    """上图画净值曲线，下图画回撤。"""
    close = stock["Close"]
    equity = close / close.iloc[0]
    drawdown = close / close.cummax() - 1.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax1.plot(equity.index, equity, color="#1565c0", linewidth=1.4)
    ax1.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_title(title)
    ax1.set_ylabel("净值（起点=1）")
    ax1.grid(alpha=0.25)

    ax2.fill_between(drawdown.index, drawdown, 0,
                     color="#c62828", alpha=0.35)
    ax2.set_ylabel("回撤")
    ax2.grid(alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_signals(df: pd.DataFrame, title: str, out_path: Path,
                 fast: int = 5, slow: int = 20) -> Path:
    """画均线策略信号图。

    上图：收盘价 + MA5/MA20，并在金叉/死叉处标注 B / S
    下图：均线差（快 - 慢），穿过 0 的位置就是交叉点
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(df.index, df["Close"], color="#37474f", linewidth=1.0,
             label="收盘价")
    ax1.plot(df.index, df[f"MA{fast}"], color="#1565c0", linewidth=1.4,
             label=f"MA{fast}")
    ax1.plot(df.index, df[f"MA{slow}"], color="#ef6c00", linewidth=1.4,
             label=f"MA{slow}")

    buys = df[df["signal"] == 1]
    sells = df[df["signal"] == -1]
    ax1.scatter(buys.index, buys["Close"], marker="^", s=70,
                color="#d32f2f", zorder=5, label=f"金叉买入 ({len(buys)})")
    ax1.scatter(sells.index, sells["Close"], marker="v", s=70,
                color="#2e7d32", zorder=5, label=f"死叉卖出 ({len(sells)})")

    ax1.set_title(title)
    ax1.set_ylabel("价格")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.plot(df.index, df["signal_diff"], color="#6a1b9a", linewidth=1.0)
    ax2.axhline(0, color="gray", linewidth=0.9, linestyle="--")
    ax2.fill_between(df.index, df["signal_diff"], 0,
                     where=(df["signal_diff"] > 0),
                     color="#d32f2f", alpha=0.18, interpolate=True)
    ax2.fill_between(df.index, df["signal_diff"], 0,
                     where=(df["signal_diff"] < 0),
                     color="#2e7d32", alpha=0.18, interpolate=True)
    ax2.set_ylabel(f"MA{fast} - MA{slow}")
    ax2.grid(alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_strategy_equity(df: pd.DataFrame, title: str, out_path: Path) -> Path:
    """策略净值 vs 买入持有净值。"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["equity_hold"], color="#455a64", linewidth=1.4,
            label="买入持有")
    ax.plot(df.index, df["equity_ma"], color="#d32f2f", linewidth=1.4,
            label="MA5/MA20 策略")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("净值（起点=1）")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
