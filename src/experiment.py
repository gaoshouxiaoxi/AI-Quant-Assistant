"""预测能力实验：在**不引入泄漏**的前提下，测试哪些设置能提升信号。

测试维度：
1. 预测周期（1 日 / 3 日 / 5 日 / 10 日）——周期越长，信噪比通常越高
2. 标签阈值（收益率超过 ±x% 才算涨/跌，过滤微小波动）
3. 特征集（仅量价 vs 加入大盘环境）

所有设置都使用同一套扩展窗口前向验证，不做随机切分。
输出 AUC 与相对基线的增益，用于判断「是否真的比随机猜好」。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from data_loader import load_daily  # noqa: E402
from ml_model import evaluate_model, make_pipeline, walk_forward_splits  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

HORIZONS = [1, 3, 5, 10]
THRESHOLDS = [0.0, 0.005, 0.01]


def build_dataset(stock: pd.DataFrame, horizon: int, threshold: float
                  ) -> tuple[pd.DataFrame, pd.Series]:
    """构造特征与标签。

    特征只用到 t 及之前的数据；标签用 t 到 t+horizon 的累计收益。
    标准化在交叉验证内部完成（见 ml_model.make_pipeline）。
    """
    df = stock.sort_index().copy()
    close = df["Close"].astype(float)
    high, low = df["High"].astype(float), df["Low"].astype(float)
    open_, volume = df["Open"].astype(float), df["Volume"].astype(float)
    ret = close.pct_change()

    feats = pd.DataFrame(index=df.index)
    # 滞后收益（缩短到 10 个，避免维度爆炸）
    for lag in range(1, 11):
        feats[f"f_ret_lag{lag}"] = ret.shift(lag)
    # 技术指标
    for w in (5, 10, 20):
        feats[f"f_ma{w}_ratio"] = close / close.rolling(w).mean() - 1.0
    feats["f_vol5"] = ret.rolling(5).std()
    feats["f_vol20"] = ret.rolling(20).std()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    feats["f_rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    feats["f_volume_ratio"] = np.log1p(volume / volume.rolling(20).mean())
    feats["f_momentum10"] = close / close.shift(10) - 1.0
    # 形态
    rng = (high - low).replace(0, np.nan)
    feats["f_body"] = (close - open_) / rng
    feats["f_upper_wick"] = (high - close.combine(open_, max)) / rng
    feats["f_lower_wick"] = (close.combine(open_, min) - low) / rng

    # 标签：t → t+horizon 的累计收益，超过阈值才算方向
    fwd = close.shift(-horizon) / close - 1.0
    label = pd.Series(index=df.index, dtype="float")
    label[fwd > threshold] = 1
    label[fwd < -threshold] = 0
    # |收益| ≤ 阈值的行为中性，丢弃（减少噪声样本）

    data = feats.copy()
    data["label"] = label
    data["forward_return"] = fwd          # 仅用于记录，绝不进特征
    data = data.dropna()
    data = data[data["label"].isin([0, 1])]
    X = data[[c for c in data.columns if c.startswith("f_")]]
    y = data["label"].astype(int)
    return X, y


def main() -> None:
    tickers = ["AAPL", "600519"]
    start, end = "2022-06-15", "2024-12-31"

    rows = []
    print("== 预测能力实验（扩展窗口前向验证，无泄漏）==")
    print(f"{'标的':<10}{'周期':>6}{'阈值':>8}{'样本':>7}{'上涨占比':>10}"
          f"{'准确率':>9}{'AUC':>8}{'超基线':>10}")
    print("-" * 74)

    for ticker in tickers:
        stock, _ = load_daily(ticker, start, end, cache_dir=DATA_DIR)
        for horizon in HORIZONS:
            for threshold in THRESHOLDS:
                X, y = build_dataset(stock, horizon, threshold)
                if len(X) < 150 or y.nunique() < 2:
                    continue
                model = RandomForestClassifier(
                    n_estimators=250, max_depth=5, min_samples_leaf=25,
                    random_state=42, n_jobs=-1,
                )
                m = evaluate_model(model, X, y, purge=horizon)
                rows.append({
                    "ticker": ticker, "horizon": horizon, "threshold": threshold,
                    "samples": len(X), "up_rate": m["up_rate"],
                    "accuracy": m["accuracy"], "auc": m["auc"],
                    "edge": m["edge_over_baseline"],
                })
                print(f"{ticker:<10}{horizon:>6}{threshold:>8.1%}{len(X):>7}"
                      f"{m['up_rate']:>10.2%}{m['accuracy']:>9.2%}"
                      f"{m['auc']:>8.3f}{m['edge_over_baseline']:>+10.2%}")

    report = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "horizon_threshold_experiment.csv"
    report.to_csv(out, index=False)

    print()
    print("== 按周期汇总（两标的平均 AUC）==")
    by_h = report.groupby("horizon")[["auc", "accuracy", "edge"]].mean()
    for h, r in by_h.iterrows():
        print(f"  周期 {h:>2} 日:  AUC {r['auc']:.3f}   准确率 {r['accuracy']:.2%}"
              f"   超基线 {r['edge']:+.2%}")

    best = report.loc[report["auc"].idxmax()]
    print()
    print(f"最优配置: {best['ticker']} 周期={int(best['horizon'])}日 "
          f"阈值={best['threshold']:.1%}  AUC={best['auc']:.3f} "
          f"准确率={best['accuracy']:.2%}")
    print(f"明细已保存到 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
