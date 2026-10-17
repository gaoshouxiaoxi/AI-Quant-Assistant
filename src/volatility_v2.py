"""波动率预测 v2：用 Yang-Zhang 估计量替代收盘价法。

为什么换
--------
rv_validation.py 用 15 分钟高频数据做了对照实验：

    日内估计量 vs 日内 RV：Parkinson      corr 0.954, 斜率 0.99
    含隔夜估计量 vs 总 RV：Yang-Zhang     corr 0.837, 斜率 1.05
    含隔夜估计量 vs 总 RV：收盘价法        corr 0.767, 斜率 0.52  ← 严重偏低

收盘价法的回归斜率只有 0.52，意味着它系统性低估真实波动，是有偏估计量。
v1 的"持续性基线"正是用收盘价法，所以基线本身就不准——这解释了为什么
v1 的 R² 普遍为负。

Yang-Zhang 同时利用开高低收四个价格、并正确处理隔夜跳空，斜率 1.05、
中位比值 0.96，接近无偏，适合作为目标。

本模块与 v1 的差别
------------------
- 目标：未来 10 日 Yang-Zhang 方差均值（年化）
- 基线：当前 20 日 Yang-Zhang 波动（持续性）
- 特征：把各估计量的对数都放进去（Parkinson / GK / RS / YZ 分量）
- 依然预测对数变化量，与 v1 保持一致以便对比
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

from ml_model import walk_forward_splits  # noqa: E402
from rv_estimators import (TRADING_DAYS, daily_rs, daily_var_close,
                           daily_var_garman_klass, daily_var_parkinson,
                           rolling_vol_yang_zhang)  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHART_DIR = ROOT / "charts"
REPORT_DIR = ROOT / "reports"

FORWARD_DAYS = 10
TRAIL_DAYS = 20
N_SPLITS = 5

TICKERS = {"600519": "贵州茅台", "000858": "五粮液", "000001": "平安银行",
           "AAPL": "Apple"}


def build_dataset_v2(stock: pd.DataFrame) -> pd.DataFrame:
    """用 Yang-Zhang 构造目标，特征包含多估计量的对数。"""
    df = stock.sort_index().copy()
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    lo = df["Low"].astype(float)
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)

    # ---- 目标与基线：Yang-Zhang ----
    yz = rolling_vol_yang_zhang(o, h, lo, c, TRAIL_DAYS)       # 年化
    yz_var_daily = (yz / np.sqrt(TRADING_DAYS)) ** 2             # 日方差
    # 未来 FORWARD_DAYS 的平均日方差（shift 保证不含未来）
    fwd_var = yz_var_daily.rolling(FORWARD_DAYS).mean().shift(-FORWARD_DAYS)
    forward_vol = np.sqrt(fwd_var * TRADING_DAYS)
    baseline_vol = yz

    out = pd.DataFrame(index=df.index)
    out["forward_vol"] = forward_vol
    out["baseline_vol"] = baseline_vol

    # ---- 特征：各估计量的对数（同一天的不同视角）----
    out["f_vol_yz"] = yz
    out["f_vol_parkinson"] = np.sqrt(
        daily_var_parkinson(h, lo).rolling(TRAIL_DAYS).mean() * TRADING_DAYS)
    out["f_vol_gk"] = np.sqrt(
        daily_var_garman_klass(o, h, lo, c).rolling(TRAIL_DAYS).mean() * TRADING_DAYS)
    out["f_vol_rs"] = np.sqrt(
        daily_rs(o, h, lo, c).rolling(TRAIL_DAYS).mean() * TRADING_DAYS)
    out["f_vol_close"] = np.sqrt(
        daily_var_close(c).rolling(TRAIL_DAYS).mean() * TRADING_DAYS)

    # 短窗 Yang-Zhang（5/10 日）
    out["f_vol_yz5"] = rolling_vol_yang_zhang(o, h, lo, c, 5)
    out["f_vol_yz10"] = rolling_vol_yang_zhang(o, h, lo, c, 10)
    out["f_vol_ratio_5_20"] = out["f_vol_yz5"] / out["f_vol_yz"]
    out["f_vol_ratio_10_20"] = out["f_vol_yz10"] / out["f_vol_yz"]

    # ---- 特征：隔夜与日内分解 ----
    log_oc = np.log(o / c.shift(1))
    log_co = np.log(c / o)
    out["f_overnight_vol20"] = log_oc.rolling(TRAIL_DAYS).std(ddof=1) * np.sqrt(TRADING_DAYS)
    out["f_intraday_vol20"] = log_co.rolling(TRAIL_DAYS).std(ddof=1) * np.sqrt(TRADING_DAYS)
    out["f_overnight_share"] = (
        out["f_overnight_vol20"] ** 2 /
        (out["f_overnight_vol20"] ** 2 + out["f_intraday_vol20"] ** 2)
    )

    # ---- 特征：离散度与极值 ----
    ret = c.pct_change()
    out["f_skew20"] = ret.rolling(TRAIL_DAYS).skew()
    out["f_kurt20"] = ret.rolling(TRAIL_DAYS).kurt()
    out["f_absret_mean5"] = ret.abs().rolling(5).mean()
    out["f_range20"] = ((h - lo) / c).rolling(TRAIL_DAYS).mean()

    # ---- 特征：成交量 ----
    out["f_volume_ratio20"] = np.log1p(v / v.rolling(TRAIL_DAYS).mean())
    out["f_volume_std20"] = v.rolling(TRAIL_DAYS).std() / v.rolling(TRAIL_DAYS).mean()

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[(out["forward_vol"] > 0) & (out["baseline_vol"] > 0)]

    # 目标与特征都在对数空间 —— 与 v1 的修正保持一致
    out["target_log"] = np.log(out["forward_vol"])
    out["baseline_log"] = np.log(out["baseline_vol"])
    for col in [c for c in out.columns
                if c.startswith("f_") and (out[c].dropna() > 0).all()]:
        out[f"log_{col}"] = np.log(out[col])
    return out


def evaluate(data: pd.DataFrame, factory) -> dict:
    """前向验证，预测对数变化量。"""
    cols = [c for c in data.columns
            if c.startswith("f_") or c.startswith("log_f_")]
    X = data[cols]
    y_delta = data["target_log"] - data["baseline_log"]

    deltas, trues, bases = [], [], []
    for tr, te in walk_forward_splits(len(X), N_SPLITS):
        pipe = Pipeline([("scaler", StandardScaler()), ("model", factory())])
        pipe.fit(X.iloc[tr], y_delta.iloc[tr])
        deltas.append(pd.Series(pipe.predict(X.iloc[te]), index=X.iloc[te].index))
        trues.append(data["target_log"].iloc[te])
        bases.append(data["baseline_log"].iloc[te])

    delta_pred = pd.concat(deltas).sort_index()
    true = pd.concat(trues).sort_index()
    base = pd.concat(bases).sort_index()
    pred = base + delta_pred

    mse_model = float(np.mean((pred - true) ** 2))
    mse_base = float(np.mean((base - true) ** 2))
    median = float(true.median())
    y_cls = (true > median).astype(int)

    return {
        "samples": len(true),
        "r2": r2_score(true, pred),
        "r2_baseline": r2_score(true, base),
        "mae": mean_absolute_error(true, pred),
        "mae_baseline": mean_absolute_error(true, base),
        "skill": 1.0 - mse_model / mse_base if mse_base > 0 else float("nan"),
        "spearman": float(true.corr(pred, method="spearman")),
        "spearman_baseline": float(true.corr(base, method="spearman")),
        "auc": roc_auc_score(y_cls, pred),
        "auc_baseline": roc_auc_score(y_cls, base),
        "pred": pred, "true": true, "base": base,
        "delta_pred": delta_pred,
        "delta_true": true - base,
    }


def factories() -> dict:
    models = {"Ridge": lambda: Ridge(alpha=1.0),
              "RandomForest": lambda: RandomForestRegressor(
                  n_estimators=300, max_depth=6, min_samples_leaf=20,
                  random_state=42, n_jobs=-1)}
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = lambda: XGBRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        pass
    return models


def load(ticker: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{ticker}_2020-01-01_2025-01-01.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def plot_prediction(result: dict, title: str, out_path: Path) -> Path:
    """预测 vs 实际（含基线），以及变化量。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from charts import setup_chinese_font
    setup_chinese_font()

    true = np.exp(result["true"])
    pred = np.exp(result["pred"])
    base = np.exp(result["base"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax1.plot(true.index, true.values, color="#212121", linewidth=1.1, label="实际")
    ax1.plot(pred.index, pred.values, color="#d32f2f", linewidth=1.1,
             alpha=0.85, label="模型（Yang-Zhang 目标）")
    ax1.plot(base.index, base.values, color="#1565c0", linewidth=1.0,
             linestyle="--", alpha=0.75, label="持续性基线")
    ax1.set_ylabel("年化波动率")
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.plot(result["delta_true"].index, result["delta_true"].values,
             color="#616161", linewidth=1.0, label="实际变化")
    ax2.plot(result["delta_pred"].index, result["delta_pred"].values,
             color="#d32f2f", linewidth=1.0, alpha=0.85, label="模型预测变化")
    ax2.axhline(0, color="gray", linewidth=0.9, linestyle="--")
    ax2.set_ylabel("对数变化")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    print("=" * 80)
    print("波动率预测 v2 —— 目标改用 Yang-Zhang（校验斜率 1.05，接近无偏）")
    print("=" * 80)
    print(f"预测未来 {FORWARD_DAYS} 日平均方差；基线为当前 {TRAIL_DAYS} 日 Yang-Zhang 波动")
    print("仍预测对数变化量，与 v1 可直接对比\n")

    f = factories()
    rows = []
    for ticker, name in TICKERS.items():
        stock = load(ticker)
        if stock is None:
            continue
        data = build_dataset_v2(stock)
        if len(data) < 200:
            print(f"{name}: 样本不足 {len(data)}，跳过")
            continue

        print(f"----- {name} ({ticker})  样本 {len(data)}  "
              f"{data.index.min().date()} ~ {data.index.max().date()}")
        base_row = None
        best_for_plot = None
        for mname, factory in f.items():
            m = evaluate(data, factory)
            print(f"  {mname:<13} R²={m['r2']:>7.3f}  MAE={m['mae']:.4f}  "
                  f"Skill={m['skill']:>+7.2%}  Spearman={m['spearman']:.3f}  "
                  f"AUC={m['auc']:.3f}")
            rows.append({"ticker": ticker, "name": name, "model": mname,
                         "samples": m["samples"], "r2": m["r2"],
                         "mae": m["mae"], "skill": m["skill"],
                         "spearman": m["spearman"], "auc": m["auc"],
                         "r2_baseline": m["r2_baseline"],
                         "mae_baseline": m["mae_baseline"],
                         "spearman_baseline": m["spearman_baseline"],
                         "auc_baseline": m["auc_baseline"]})
            base_row = m
            if mname == "RandomForest":
                best_for_plot = m
        if best_for_plot is not None:
            try:
                plot_prediction(best_for_plot,
                                f"{name} 波动率预测 v2（Yang-Zhang 目标）",
                                CHART_DIR / f"vol_v2_{ticker}.png")
            except Exception as exc:  # noqa: BLE001
                print(f"    图表失败: {type(exc).__name__}: {exc}")
        if base_row:
            print(f"  {'[基线]持续性':<13} R²={base_row['r2_baseline']:>7.3f}  "
                  f"MAE={base_row['mae_baseline']:.4f}  "
                  f"Skill={0.0:>+7.2%}  Spearman={base_row['spearman_baseline']:.3f}  "
                  f"AUC={base_row['auc_baseline']:.3f}")
        print()

    if not rows:
        print("无可用数据")
        return

    df = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_DIR / "volatility_v2_yang_zhang.csv", index=False)

    print("=" * 80)
    print("汇总 v2（Yang-Zhang 目标）")
    print("=" * 80)
    print(f"{'标的':<12}{'模型':<13}{'R²':>8}{'基线R²':>9}{'Skill':>9}"
          f"{'Spearman':>10}{'AUC':>8}")
    print("-" * 80)
    for _, r in df.iterrows():
        print(f"{r['name']:<12}{r['model']:<13}{r['r2']:>8.3f}"
              f"{r['r2_baseline']:>9.3f}{r['skill']:>+9.2%}"
              f"{r['spearman']:>10.3f}{r['auc']:>8.3f}")

    print(f"\n平均：模型 R²={df['r2'].mean():.3f}  基线 R²={df['r2_baseline'].mean():.3f}")
    print(f"      模型 Skill={df['skill'].mean():+.2%}  "
          f"模型 AUC={df['auc'].mean():.3f}  基线 AUC={df['auc_baseline'].mean():.3f}")

    print("\n对比 v1（收盘价法目标）：v1 模型平均 R²=-0.245，基线平均 R²=-0.046")
    print("若 v2 的 R² 明显回升，说明降低估计噪声确实有效。")


if __name__ == "__main__":
    main()
