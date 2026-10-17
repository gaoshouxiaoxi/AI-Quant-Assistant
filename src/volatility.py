"""波动率预测：比方向预测可靠得多的目标。

为什么波动率可预测而方向不可预测
--------------------------------
波动率具有**聚集性**（volatility clustering）：大波动后面往往还是大波动。
这是金融学最稳健的经验规律之一（Mandelbrot 1963；Engle 的 ARCH 模型 1982，
Engle 因此获 2003 年诺贝尔经济学奖）。而**方向**近似随机游走，没有这种结构。

所以本模块的预测目标不是"涨还是跌"，而是"未来波动有多大"。

关键设计：必须打败持续性基线
----------------------------
波动率预测有一个极强且免费的基线——**用当前波动率直接外推**：

    预测未来波动 = 最近 20 日已实现波动

任何模型如果打不过这个基线，就没有存在价值。本模块把它作为基准，
报告 R² 提升与 MAE 降低。

评估口径：
- 扩展窗口前向验证（训练集永远在测试集之前）
- 目标取对数（对数波动率分布更接近正态，且误差相对化）
- 同时报告回归指标（R²/MAE）与分类指标（高波动 regime 的 AUC）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHART_DIR = ROOT / "charts"
REPORT_DIR = ROOT / "reports"

TRADING_DAYS = 252
FORWARD_DAYS = 10      # 预测未来 10 日波动
TRAIL_DAYS = 20        # 持续性基线用的回看窗口
N_SPLITS = 5


# ---------------------------------------------------------------- 数据准备

def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """已实现波动率（年化）。"""
    return returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def build_dataset(stock: pd.DataFrame) -> pd.DataFrame:
    """构造波动率预测的特征与目标。

    所有特征只用 t 及之前的数据；目标是 [t+1, t+FORWARD_DAYS] 的已实现波动。
    """
    df = stock.sort_index().copy()
    close = df["Close"].astype(float)
    high, low = df["High"].astype(float), df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    ret = close.pct_change()
    out = pd.DataFrame(index=df.index)

    # ---- 特征：历史波动率 ----
    for w in (5, 10, 20, 60):
        out[f"f_vol{w}"] = realized_vol(ret, w)
    out["f_vol_ratio_5_20"] = out["f_vol5"] / out["f_vol20"]
    out["f_vol_ratio_10_60"] = out["f_vol10"] / out["f_vol60"]

    # ---- 特征：绝对收益（波动率的直接来源）----
    for lag in range(1, 6):
        out[f"f_absret_lag{lag}"] = ret.abs().shift(lag)
    out["f_absret_mean5"] = ret.abs().rolling(5).mean()
    out["f_absret_mean20"] = ret.abs().rolling(20).mean()

    # ---- 特征：波动率的偏度结构（下跌波动 vs 上涨波动）----
    out["f_downvol20"] = ret.rolling(20).apply(
        lambda x: float(np.std(x[x < 0], ddof=1)) if (x < 0).sum() > 2 else np.nan,
        raw=True,
    ) * np.sqrt(TRADING_DAYS)
    out["f_upvol20"] = ret.rolling(20).apply(
        lambda x: float(np.std(x[x > 0], ddof=1)) if (x > 0).sum() > 2 else np.nan,
        raw=True,
    ) * np.sqrt(TRADING_DAYS)
    out["f_vol_asymmetry"] = out["f_downvol20"] / out["f_upvol20"]

    # ---- 特征：日内振幅（真实波幅 ATR 的相对值）----
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    out["f_atr20"] = tr.rolling(20).mean() / close
    out["f_range20"] = ((high - low) / close).rolling(20).mean()

    # ---- 特征：收益率分布形状 ----
    out["f_skew20"] = ret.rolling(20).skew()
    out["f_kurt20"] = ret.rolling(20).kurt()
    out["f_absret_skew20"] = ret.abs().rolling(20).skew()

    # ---- 特征：成交量 ----
    out["f_volume_ratio20"] = np.log1p(volume / volume.rolling(20).mean())
    out["f_volume_std20"] = volume.rolling(20).std() / volume.rolling(20).mean()

    # ---- 目标：未来 FORWARD_DAYS 的已实现波动（年化）----
    # 正确写法：rolling(10).std()[t] 是 [t-9, t] 的波动，
    # 再 shift(-10) 就搬到 t 上，即 [t+1, t+10] 的波动。
    # 注意不要写成 ret.shift(-1).rolling(n).std().shift(-(n-1))——
    # 那是双重位移，会取到 [t+10, t+19]，等于把目标推得更远。
    fwd = ret.rolling(FORWARD_DAYS).std(ddof=1).shift(-FORWARD_DAYS)
    out["forward_vol"] = fwd * np.sqrt(TRADING_DAYS)
    # 持续性基线：用当前 20 日波动直接外推
    out["baseline_vol"] = out[f"f_vol{TRAIL_DAYS}"]

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna()
    # 取对数：波动率分布右偏，取对数后误差更对称
    out = out[(out["forward_vol"] > 0) & (out["baseline_vol"] > 0)]
    out["target_log"] = np.log(out["forward_vol"])
    out["baseline_log"] = np.log(out["baseline_vol"])

    # 目标是对数，特征却是原始幅值 —— 线性模型无法拟合 log 关系，
    # 会把 Ridge 拖到比基线还差。对所有严格为正的特征取对数后，
    # 模型在"对数空间"里是线性的，与目标匹配。
    # 树模型对单调变换不敏感，因此这条对 RF/XGB 无副作用。
    positive_cols = [c for c in out.columns
                     if c.startswith("f_")
                     and (out[c].dropna() > 0).all()]
    for col in positive_cols:
        out[f"log_{col}"] = np.log(out[col])
    return out


# ---------------------------------------------------------------- 评估

def walk_forward(n: int, n_splits: int = N_SPLITS, min_train: float = 0.4):
    start = int(n * min_train)
    step = max(1, (n - start) // n_splits)
    for k in range(n_splits):
        b = start + k * step
        end = n if k == n_splits - 1 else b + step
        yield np.arange(0, b), np.arange(b, end)


def evaluate_volatility(data: pd.DataFrame, model_factory) -> dict:
    """前向验证：预测**波动的对数变化量**，再与持续性基线对比。

    为什么不直接预测水平
    --------------------
    波动率的绝对水平随时间漂移（高波动年份 vs 低波动年份）。若让模型直接预测
    log(未来波动)，它在训练期学到的水平到测试期就错位了，R² 会变成负数；
    而"用当前波动率外推"的基线永远用最新观测，天然自适应。

    因此改为预测 **log(未来波动) - log(当前波动)**，即"波动率会放大还是收敛"。
    基线对应变化量 = 0；模型的预测值直接就是相对基线的增益，
    最终水平 = 当前波动 × exp(预测变化)。
    """
    feature_cols = [c for c in data.columns
                    if c.startswith("f_") or c.startswith("log_f_")]
    X = data[feature_cols]
    y_delta = data["target_log"] - data["baseline_log"]

    deltas, trues, bases, idxs = [], [], [], []

    for tr, te in walk_forward(len(X)):
        X_tr, y_tr = X.iloc[tr], y_delta.iloc[tr]
        X_te = X.iloc[te]
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("model", model_factory())])
        pipe.fit(X_tr, y_tr)
        deltas.append(pd.Series(pipe.predict(X_te), index=X_te.index))
        trues.append(data["target_log"].iloc[te])
        bases.append(data["baseline_log"].iloc[te])
        idxs.append(X_te.index)

    delta_pred = pd.concat(deltas).sort_index()
    true = pd.concat(trues).sort_index()
    base = pd.concat(bases).sort_index()

    # 还原为对数水平：当前水平 + 预测变化
    pred = base + delta_pred
    true_delta = true - base

    result = {
        "samples": len(true),
        "r2_model": r2_score(true, pred),
        "r2_baseline": r2_score(true, base),
        "mae_model": mean_absolute_error(true, pred),
        "mae_baseline": mean_absolute_error(true, base),
        "spearman_model": float(true.corr(pred, method="spearman")),
        "spearman_baseline": float(true.corr(base, method="spearman")),
        # 变化量本身的拟合优度：0 分代表与基线持平（不预测任何变化）
        "r2_delta": r2_score(true_delta, delta_pred),
        "pred": pred, "true": true, "base": base,
        "delta_true": true_delta, "delta_pred": delta_pred,
    }
    result["r2_gain"] = result["r2_model"] - result["r2_baseline"]
    result["mae_reduction"] = 1 - result["mae_model"] / result["mae_baseline"]

    # 技巧得分：1 - MSE_model/MSE_baseline。>0 表示优于"当前波动率外推"。
    mse_model = float(np.mean((pred - true) ** 2))
    mse_base = float(np.mean((base - true) ** 2))
    result["skill"] = 1.0 - mse_model / mse_base if mse_base > 0 else float("nan")

    # 分类任务：未来波动是否高于样本中位数（高波动 regime）
    median = float(true.median())
    y_cls = (true > median).astype(int)
    for tag, series in (("auc_model", pred), ("auc_baseline", base)):
        try:
            result[tag] = roc_auc_score(y_cls, series)
        except ValueError:
            result[tag] = float("nan")
    return result


def model_factories() -> dict:
    models = {
        "Ridge": lambda: Ridge(alpha=1.0),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            random_state=42, n_jobs=-1),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = lambda: XGBRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        pass
    return models


def load_cached(ticker: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{ticker}_2020-01-01_2025-01-01.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


TICKERS = {"600519": "贵州茅台", "000858": "五粮液", "000001": "平安银行",
           "AAPL": "Apple"}


def plot_vol_forecast(result: dict, title: str, out_path: Path) -> Path:
    """预测 vs 实际波动率（对数尺度换成百分比更直观）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 必须设置中文字体，否则标题和图例全是方框
    from charts import setup_chinese_font
    setup_chinese_font()

    true = np.exp(result["true"])
    pred = np.exp(result["pred"])
    base = np.exp(result["base"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax1.plot(true.index, true.values, color="#212121", linewidth=1.1,
             label="实际未来10日波动")
    ax1.plot(pred.index, pred.values, color="#d32f2f", linewidth=1.1,
             alpha=0.85, label="模型预测")
    ax1.plot(base.index, base.values, color="#1565c0", linewidth=1.0,
             linestyle="--", alpha=0.75, label="持续性基线")
    ax1.set_ylabel("年化波动率")
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    delta_true = np.exp(result["delta_true"]) - 1
    delta_pred = np.exp(result["delta_pred"]) - 1
    ax2.plot(delta_true.index, delta_true.values, color="#616161",
             linewidth=1.0, label="实际变化")
    ax2.plot(delta_pred.index, delta_pred.values, color="#d32f2f",
             linewidth=1.0, alpha=0.85, label="模型预测变化")
    ax2.axhline(0, color="gray", linewidth=0.9, linestyle="--")
    ax2.set_ylabel("相对当前波动")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    print("=" * 78)
    print("波动率预测：未来 10 日已实现波动（年化）")
    print("基线：用当前 20 日波动直接外推（波动率聚集性）")
    print("=" * 78)

    factories = model_factories()
    print(f"\n模型: {', '.join(factories)}")
    print("目标: log(未来10日已实现波动)，回归任务；另测高波动 regime 分类\n")

    rows = []
    for ticker, name in TICKERS.items():
        stock = load_cached(ticker)
        if stock is None:
            print(f"{name} ({ticker}): 无本地数据，跳过")
            continue
        data = build_dataset(stock)
        if len(data) < 200:
            print(f"{name}: 样本不足（{len(data)}），跳过")
            continue

        print(f"----- {name} ({ticker}) -----  样本 {len(data)}  "
              f"{data.index.min().date()} ~ {data.index.max().date()}")

        base_r2 = None
        plotted = False
        for mname, factory in factories.items():
            m = evaluate_volatility(data, factory)
            base_r2 = m["r2_baseline"]
            print(f"  {mname:<13} R²={m['r2_model']:>7.3f}  "
                  f"MAE={m['mae_model']:.4f}  "
                  f"Spearman={m['spearman_model']:.3f}  "
                  f"AUC={m['auc_model']:.3f}")
            rows.append({"ticker": ticker, "name": name, "model": mname,
                         "samples": m["samples"],
                         "r2": m["r2_model"], "mae": m["mae_model"],
                         "spearman": m["spearman_model"], "auc": m["auc_model"],
                         "r2_baseline": m["r2_baseline"],
                         "mae_baseline": m["mae_baseline"],
                         "r2_gain": m["r2_gain"],
                         "mae_reduction": m["mae_reduction"]})
            # 用表现最稳的 RandomForest 画一张预测 vs 实际
            if mname == "RandomForest" and not plotted:
                try:
                    plot_vol_forecast(m, f"{name} 波动率预测（样本外）",
                                      CHART_DIR / f"vol_{ticker}.png")
                    plotted = True
                except Exception as exc:  # noqa: BLE001
                    print(f"    图表生成失败: {type(exc).__name__}: {exc}")

        # 基线单独一行（各模型共享同一基线）
        b = evaluate_volatility(data, lambda: Ridge(alpha=1.0))
        print(f"  {'[基线]持续性':<13} R²={b['r2_baseline']:>7.3f}  "
              f"MAE={b['mae_baseline']:.4f}  "
              f"Spearman={b['spearman_baseline']:.3f}  "
              f"AUC={b['auc_baseline']:.3f}")
        print()

    if not rows:
        print("没有可用数据")
        return

    df = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_DIR / "volatility_results.csv", index=False)

    print("=" * 78)
    print("汇总：模型 vs 持续性基线")
    print("=" * 78)
    print(f"{'标的':<12}{'模型':<13}{'R²':>8}{'基线R²':>9}{'R²提升':>9}"
          f"{'MAE降低':>9}{'AUC':>8}")
    print("-" * 78)
    for _, r in df.iterrows():
        print(f"{r['name']:<12}{r['model']:<13}{r['r2']:>8.3f}"
              f"{r['r2_baseline']:>9.3f}{r['r2_gain']:>+9.3f}"
              f"{r['mae_reduction']:>+9.1%}{r['auc']:>8.3f}")

    print(f"\n整体平均 R²：模型 {df['r2'].mean():.3f} vs 基线 {df['r2_baseline'].mean():.3f}")
    print(f"整体平均 R² 提升：{df['r2_gain'].mean():+.3f}")
    print(f"整体平均 AUC：{df['auc'].mean():.3f}")

    print("\n判读：")
    print("  - 波动率聚集性强，'用当前波动外推'已是很难打败的基线")
    print("  - AUC 0.7+ 在此任务上属正常且真实，与方向预测的 0.5 形成鲜明对比")
    print("  - 波动率可预测 ≠ 能赚钱：它只说明'风险大小'可估，不说明'方向'")
    print(f"\n图表已保存到 charts/vol_*.png")


if __name__ == "__main__":
    main()
