"""多标的稳健性验证：最优配置（10 日 / 1% 阈值）能否推广？

用 2020-2025 全窗口（约 1212 个交易日）在 3 个 A 股上分别验证。
样本量比之前（2.5 年）多了近一倍，且跨标的——如果信号只在某一只股票上出现，
大概率是过拟合；如果多只都出现且方向一致，可信度才够。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from ml_model import make_pipeline, walk_forward_splits  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

TICKERS = {
    "600519": "贵州茅台",
    "000858": "五粮液",
    "000001": "平安银行",
}
HORIZON = 10
THRESHOLD = 0.01
N_SPLITS = 5


def load_cached(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}_2020-01-01_2025-01-01.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def build_dataset(stock: pd.DataFrame, horizon: int = HORIZON,
                  threshold: float = THRESHOLD):
    df = stock.sort_index().copy()
    close = df["Close"].astype(float)
    high, low = df["High"].astype(float), df["Low"].astype(float)
    open_, volume = df["Open"].astype(float), df["Volume"].astype(float)
    ret = close.pct_change()

    feats = pd.DataFrame(index=df.index)
    for lag in range(1, 11):
        feats[f"f_ret_lag{lag}"] = ret.shift(lag)
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
    rng = (high - low).replace(0, np.nan)
    feats["f_body"] = (close - open_) / rng
    feats["f_upper_wick"] = (high - close.combine(open_, max)) / rng
    feats["f_lower_wick"] = (close.combine(open_, min) - low) / rng

    fwd = close.shift(-horizon) / close - 1.0
    label = pd.Series(index=df.index, dtype="float")
    label[fwd > threshold] = 1
    label[fwd < -threshold] = 0

    data = feats.copy()
    data["label"] = label
    data = data.dropna()
    data = data[data["label"].isin([0, 1])]
    X = data[[c for c in data.columns if c.startswith("f_")]]
    y = data["label"].astype(int)
    return X, y


def run_folds(X: pd.DataFrame, y: pd.Series):
    rows = []
    for k, (tr, te) in enumerate(walk_forward_splits(len(X), N_SPLITS, purge=HORIZON), 1):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_te, y_te = X.iloc[te], y.iloc[te]
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue
        clf = make_pipeline(RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=25,
            random_state=42, n_jobs=-1))
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y_te, proba)
        except ValueError:
            auc = float("nan")
        rows.append({
            "ticker": None, "fold": k, "n_test": len(X_te),
            "up_rate": float(y_te.mean()),
            "accuracy": float((pred == y_te.values).mean()),
            "baseline": max(float(y_te.mean()), 1 - float(y_te.mean())),
            "auc": auc,
        })
    return rows


def main() -> None:
    print("=" * 78)
    print(f"多标的验证：周期 {HORIZON} 日，阈值 {THRESHOLD:.0%}，"
          f"区间 2020-01-02 ~ 2024-12-31")
    print("=" * 78)

    all_rows: list[dict] = []
    summary: list[dict] = []

    for ticker, name in TICKERS.items():
        stock = load_cached(ticker)
        X, y = build_dataset(stock)
        rows = run_folds(X, y)
        for r in rows:
            r["ticker"] = ticker
        all_rows.extend(rows)

        aucs = [r["auc"] for r in rows if r["auc"] == r["auc"]]
        accs = [r["accuracy"] for r in rows]
        edges = [r["accuracy"] - r["baseline"] for r in rows]
        above = sum(1 for a in aucs if a > 0.5)

        print(f"\n{name} ({ticker})  样本 {len(X)}  次日(10日)上涨占比 {y.mean():.1%}")
        print(f"{'折':>3}{'样本':>7}{'上涨占比':>10}{'准确率':>9}{'基线':>9}{'AUC':>8}")
        print("-" * 52)
        for r in rows:
            print(f"{r['fold']:>3}{r['n_test']:>7}{r['up_rate']:>10.1%}"
                  f"{r['accuracy']:>9.1%}{r['baseline']:>9.1%}{r['auc']:>8.3f}")
        print(f"  平均 AUC {np.mean(aucs):.3f}  >0.5 折数 {above}/{len(aucs)}  "
              f"平均超基线 {np.mean(edges):+.1%}")

        summary.append({
            "ticker": ticker, "name": name, "samples": len(X),
            "mean_auc": float(np.mean(aucs)),
            "folds_above_05": above, "n_folds": len(aucs),
            "mean_accuracy": float(np.mean(accs)),
            "mean_edge": float(np.mean(edges)),
        })

    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"{'标的':<16}{'样本':>7}{'平均AUC':>10}{'>0.5折数':>11}{'平均准确率':>12}{'平均超基线':>12}")
    print("-" * 78)
    for s in summary:
        print(f"{s['name']:<16}{s['samples']:>7}{s['mean_auc']:>10.3f}"
              f"{str(s['folds_above_05']) + '/' + str(s['n_folds']):>11}"
              f"{s['mean_accuracy']:>12.1%}{s['mean_edge']:>+12.1%}")

    avg_auc = np.mean([s["mean_auc"] for s in summary])
    print(f"\n三只标的平均 AUC: {avg_auc:.3f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(REPORT_DIR / "multi_ticker_folds.csv", index=False)
    pd.DataFrame(summary).to_csv(REPORT_DIR / "multi_ticker_summary.csv", index=False)
    print(f"明细已保存到 reports/multi_ticker_*.csv")

    print("\n判读：若三只标的都稳定 >0.5 且超基线为正，才可认为该周期存在可利用信号；")
    print("      若只有个别标的突出，应视为过拟合，不能作为交易依据。")


if __name__ == "__main__":
    main()
