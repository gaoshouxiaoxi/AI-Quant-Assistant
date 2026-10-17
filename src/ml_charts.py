"""图表模块补充：机器学习相关的两张图。

放到独立文件里，避免 charts.py 继续膨胀；charts.py 仍然负责行情类图表。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_feature_importance(columns, importances, title: str,
                            out_path: Path, top_n: int = 15) -> Path:
    """横向条形图：特征重要性 Top N。"""
    series = pd.Series(np.asarray(importances, dtype=float),
                       index=list(columns)).sort_values(ascending=False)
    top = series.head(top_n)[::-1]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(top) + 1.6))
    ax.barh(top.index, top.values, color="#1565c0")
    ax.set_title(title)
    ax.set_xlabel("重要性")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_prediction_curve(proba: pd.Series, y_true: pd.Series, title: str,
                          out_path: Path) -> Path:
    """上图：样本外预测概率与 0.5 阈值；下图：实际涨跌。"""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax1.plot(proba.index, proba.values, color="#6a1b9a", linewidth=1.0,
             label="预测上涨概率")
    ax1.axhline(0.5, color="gray", linewidth=0.9, linestyle="--",
                label="决策阈值 0.5")
    ax1.axhline(float(y_true.mean()), color="#ef6c00", linewidth=0.9,
                linestyle=":", label=f"上涨占比 {y_true.mean():.1%}")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("概率")
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    colors = np.where(y_true.values > 0, "#d32f2f", "#2e7d32")
    ax2.bar(y_true.index, np.ones(len(y_true)), color=colors, width=1.0)
    ax2.set_yticks([])
    ax2.set_ylabel("实际")
    ax2.set_xlabel("红色=次日涨  绿色=次日跌")

    fig.autofmt_xdate()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
