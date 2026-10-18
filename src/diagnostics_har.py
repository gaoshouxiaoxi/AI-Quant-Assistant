"""诊断：为什么 HAR 的 Skill 高、却 Spearman 低？

背景
----
在严格样本外（仅真实 RV，492 天）上出现了一个矛盾：

| 标的 | HAR-3 Skill | HAR-3 Spearman | 基线 Spearman |
| --- | --- | --- | --- |
| 贵州茅台 | +25.8% | 0.448 | 0.472 |
| 五粮液 | +30.5% | 0.297 | 0.368 |
| 平安银行 | +29.1% | 0.225 | 0.321 |

Skill 是 1 − MSE_model/MSE_baseline（幅度误差），Spearman 是排序能力。
两者衡量的是**不同能力**，本脚本把它拆开看：

1. **散点与校准**：预测值 vs 真实值的 45° 线关系、分箱后的均值对均值
2. **按波动区间的误差**：把真实 RV 分低/中/高三档，比各档的 MAE 与排序
3. **极端日主导性**：剔除真实波动最大的 1%/5%/10% 交易日后重算 Skill 与 Spearman
4. **逐折稳定性**：每一折的 Skill / Spearman 是否一致

判读原则：若剔除少数极端日后 Skill 大幅下降而 Spearman 上升，说明
"幅度优势主要来自对尖峰的补偿，排序能力并不成立"。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from har_extended import (BASELINES, FORWARD, HAR3, HARX, N_SPLITS,  # noqa: E402
                          NAMES, SYMBOLS, build_features, build_series,
                          evaluate)
from rv_estimators import TRADING_DAYS  # noqa: E402
from validation import common_index  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHART_DIR = ROOT / "charts"
REPORT_DIR = ROOT / "reports"


def metrics(pred: pd.Series, true: pd.Series, base: pd.Series) -> dict:
    mse_m = float(np.mean((pred - true) ** 2))
    mse_b = float(np.mean((base - true) ** 2))
    return {
        "n": len(true),
        "skill": 1 - mse_m / mse_b if mse_b > 0 else float("nan"),
        "spearman": float(true.corr(pred, method="spearman")),
        "spearman_base": float(true.corr(base, method="spearman")),
        "mae": float(np.mean(np.abs(pred - true))),
        "mae_base": float(np.mean(np.abs(base - true))),
    }


def run_strict(ticker: str, symbol: str) -> tuple[dict, pd.DataFrame]:
    """在严格样本（仅真实 RV）上跑 HAR-3，返回样本外预测序列。

    样本对齐口径必须与 `har_extended.run_block` 的 D 组完全一致
    （特征并集 = HAR3 + HARX + 基线列），否则同一份结果会出现两个数字。
    """
    strict_raw, _, _ = build_series(ticker, symbol)
    data = build_features(strict_raw)
    all_cols = sorted(set(HAR3 + HARX
                          + [c for cs in BASELINES.values() for c in cs]))
    idx = common_index(data, all_cols)
    data = data.loc[idx]

    m = evaluate(data, HAR3, lambda: LinearRegression(), "HAR-3",
                 purge=FORWARD, align_index=data.index)
    df = pd.DataFrame({"pred": m["pred"], "true": m["true"], "base": m["base"]})
    df["actual_vol"] = np.sqrt(np.exp(df["true"]) * TRADING_DAYS)
    df["pred_vol"] = np.sqrt(np.exp(df["pred"]) * TRADING_DAYS)
    df["base_vol"] = np.sqrt(np.exp(df["base"]) * TRADING_DAYS)
    df["err"] = df["pred"] - df["true"]
    df["err_base"] = df["base"] - df["true"]
    # evaluate() 不返回基线的排序指标，这里统一用自己的 metrics 重算
    return metrics(df["pred"], df["true"], df["base"]), df


def buckets(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """按真实波动分档，比较各档误差与排序。"""
    q = pd.qcut(df["actual_vol"], n, labels=["低波动", "中波动", "高波动"])
    rows = []
    for name, g in df.groupby(q, observed=True):
        rows.append({
            "区间": str(name), "n": len(g),
            "平均真实波动": g["actual_vol"].mean(),
            "MAE(模型)": g["err"].abs().mean(),
            "MAE(基线)": g["err_base"].abs().mean(),
            "Spearman(模型)": g["true"].corr(g["pred"], method="spearman"),
            "Spearman(基线)": g["true"].corr(g["base"], method="spearman"),
            "有偏(模型)": g["err"].mean(),
            "有偏(基线)": g["err_base"].mean(),
        })
    return pd.DataFrame(rows)


def calibration(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """分箱校准：预测值分位 vs 真实均值。"""
    q = pd.qcut(df["pred"], n_bins, labels=False, duplicates="drop")
    rows = []
    for b, g in df.groupby(q):
        rows.append({"bin": int(b), "n": len(g),
                     "预测均值(log)": g["pred"].mean(),
                     "真实均值(log)": g["true"].mean(),
                     "预测均值(年化vol)": g["pred_vol"].mean(),
                     "真实均值(年化vol)": g["actual_vol"].mean(),
                     "偏差(log)": (g["pred"] - g["true"]).mean()})
    return pd.DataFrame(rows)


def tail_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """剔除真实波动最大的 k% 交易日后，重算 Skill 与 Spearman。"""
    rows = []
    for pct in (0, 1, 5, 10, 20):
        if pct == 0:
            sub = df
        else:
            thr = df["actual_vol"].quantile(1 - pct / 100)
            sub = df[df["actual_vol"] <= thr]
        if len(sub) < 60:
            continue
        m = metrics(sub["pred"], sub["true"], sub["base"])
        m["剔除比例"] = f"{pct}%"
        rows.append(m)
    return pd.DataFrame(rows)


def plot_scatter(df: pd.DataFrame, title: str, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from charts import setup_chinese_font
    setup_chinese_font()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))

    ax1.scatter(df["actual_vol"], df["pred_vol"], s=14, alpha=0.55,
                color="#d32f2f", label="HAR 预测")
    ax1.scatter(df["actual_vol"], df["base_vol"], s=14, alpha=0.35,
                color="#1565c0", label="持续性基线")
    lim = [min(df["actual_vol"].min(), df["base_vol"].min()) * 0.9,
           max(df["actual_vol"].max(), df["base_vol"].max()) * 1.05]
    ax1.plot(lim, lim, color="gray", linestyle="--", linewidth=1,
             label="完美预测（45°）")
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_xlabel("实际年化波动率")
    ax1.set_ylabel("预测年化波动率")
    ax1.set_title(f"{title} · 预测 vs 实际")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    cal = calibration(df)
    ax2.plot(cal["真实均值(年化vol)"], cal["预测均值(年化vol)"],
             marker="o", color="#d32f2f", label="HAR 分箱均值")
    ax2.plot(lim, lim, color="gray", linestyle="--", linewidth=1,
             label="45° 线")
    ax2.set_xlabel("真实波动（分箱均值）")
    ax2.set_ylabel("预测波动（同箱均值）")
    ax2.set_title(f"{title} · 校准关系")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_tail(df: pd.DataFrame, title: str, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from charts import setup_chinese_font
    setup_chinese_font()

    t = tail_sensitivity(df)
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - 0.2, t["skill"] * 100, width=0.4, color="#d32f2f",
           label="Skill（%）")
    ax.bar(x + 0.2, t["spearman"], width=0.4, color="#1565c0",
           label="Spearman")
    ax.set_xticks(x)
    ax.set_xticklabels([f"剔除最大 {v}" if v != "0%" else "全样本"
                        for v in t["剔除比例"]])
    ax.axhline(0, color="gray", linewidth=0.9)
    ax.set_title(f"{title} · 剔除极端波动日后的指标变化")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    print("=" * 86)
    print("诊断：Skill（幅度误差）与 Spearman（排序能力）为何矛盾")
    print("=" * 86)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows, bucket_rows, tail_rows = [], [], []

    for symbol, ticker in SYMBOLS.items():
        name = NAMES[ticker]
        m, df = run_strict(ticker, symbol)
        print(f"\n----- {name}（严格样本外 {len(df)} 天）")
        print(f"  整体：Skill {m['skill']:+.2%}   Spearman {m['spearman']:.3f}"
              f"（基线 {m['spearman_base']:.3f}）   "
              f"MAE {m['mae']:.4f}（基线 {m['mae_base']:.4f}）")

        b = buckets(df)
        print("\n  按真实波动分档：")
        print("    " + b.to_string(index=False).replace("\n", "\n    "))

        t = tail_sensitivity(df)
        print("\n  剔除极端波动日：")
        for _, r in t.iterrows():
            print(f"    剔除最大 {r['剔除比例']:>3}：n={int(r['n']):>3}  "
                  f"Skill {r['skill']:+7.2%}  Spearman {r['spearman']:.3f}"
                  f"（基线 {r['spearman_base']:.3f}）  MAE {r['mae']:.4f}")

        # 校准斜率与离散度：判断预测是否被"压缩"到均值附近
        slope, intercept = np.polyfit(df["pred"], df["true"], 1)
        disp_ratio = float(df["pred"].std() / df["true"].std())
        disp_ratio_base = float(df["base"].std() / df["true"].std())
        print(f"\n  校准斜率（true~pred）= {slope:.2f}"
              f"（1=完美，<1 表示预测向均值收缩）")
        print(f"  预测离散度/真实离散度：模型 {disp_ratio:.2f}"
              f"   基线 {disp_ratio_base:.2f}")

        # 校准偏差方向
        cal = calibration(df)
        hi_bin = cal.iloc[-1]
        print(f"  最高预测分箱：预测 {hi_bin['预测均值(年化vol)']:.1%} vs "
              f"真实 {hi_bin['真实均值(年化vol)']:.1%}"
              f"（偏差 {(hi_bin['预测均值(年化vol)'] - hi_bin['真实均值(年化vol)']):+.1%}）")

        for _, r in t.iterrows():
            tail_rows.append({"ticker": ticker, "name": name,
                              "drop_pct": r["剔除比例"], **r.to_dict()})
        for _, r in b.iterrows():
            bucket_rows.append({"ticker": ticker, "name": name,
                                "bucket": r["区间"], **r.to_dict()})
        summary_rows.append({"ticker": ticker, "name": name, **m,
                             "calib_slope": float(slope),
                             "disp_ratio": disp_ratio,
                             "disp_ratio_base": disp_ratio_base})

        try:
            plot_scatter(df, name, CHART_DIR / f"diag_scatter_{ticker}.png")
            plot_tail(df, name, CHART_DIR / f"diag_tail_{ticker}.png")
        except Exception as exc:  # noqa: BLE001
            print(f"  图表失败: {type(exc).__name__}: {exc}")

    pd.DataFrame(summary_rows).to_csv(REPORT_DIR / "diag_summary.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(REPORT_DIR / "diag_buckets.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(REPORT_DIR / "diag_tail.csv", index=False)

    print("\n" + "=" * 86)
    print("结论要点")
    print("=" * 86)
    print("  - Skill 衡量幅度误差（MSE），Spearman 衡量排序，两者是不同能力")
    print("  - 若剔除极端波动日后 Skill 明显下降、Spearman 反而上升，")
    print("    说明幅度优势主要来自对尖峰的补偿，排序能力并不成立")
    print("  - 分档结果可看出模型在哪一档最差，避免用总体指标掩盖局部失效")


if __name__ == "__main__":
    main()
