"""稳健性检验：最优配置是不是"运气"？

单个标的上出现 AUC 0.63 有两种可能：
  1. 真实存在的可预测成分
  2. 在特定区间、特定参数上碰巧拟合（数据挖掘偏差）

本脚本用三种方式区分：
  A. 分折稳定性 —— 5 个前向折的 AUC 是否都 > 0.5，还是靠某一折拉高平均
  B. 随机标签对照 —— 把标签打乱后重跑，AUC 应该回到 0.5
  C. 跨区间检验 —— 换一段时间窗口，看结论是否还成立
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

from data_loader import load_daily  # noqa: E402
from ml_model import make_pipeline, walk_forward_splits  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

HORIZON = 10
THRESHOLD = 0.01


def build_dataset(stock: pd.DataFrame, horizon: int, threshold: float):
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


def fold_aucs(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """逐折计算 AUC，返回明细。"""
    rows = []
    for k, (tr, te) in enumerate(walk_forward_splits(len(X), n_splits), 1):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_te, y_te = X.iloc[te], y.iloc[te]
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue
        model = RandomForestClassifier(
            n_estimators=250, max_depth=5, min_samples_leaf=25,
            random_state=42, n_jobs=-1,
        )
        clf = make_pipeline(model)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        try:
            auc = roc_auc_score(y_te, proba)
        except ValueError:
            auc = float("nan")
        rows.append({
            "fold": k,
            "test_start": X_te.index[0].date(),
            "test_end": X_te.index[-1].date(),
            "n_test": len(X_te),
            "up_rate": float(y_te.mean()),
            "auc": auc,
        })
    return rows


def main() -> None:
    print("=" * 74)
    print(f"稳健性检验：600519  周期={HORIZON}日  阈值={THRESHOLD:.0%}")
    print("=" * 74)

    stock, _ = load_daily("600519", "2022-06-15", "2024-12-31", cache_dir=DATA_DIR)
    X, y = build_dataset(stock, HORIZON, THRESHOLD)

    # ---- A. 分折稳定性 ----
    print(f"\n[A] 逐折 AUC（共 {len(X)} 个样本，5 折前向验证）")
    rows = fold_aucs(X, y)
    print(f"{'折':>3}{'测试区间':>26}{'样本':>7}{'上涨占比':>10}{'AUC':>8}")
    print("-" * 74)
    for r in rows:
        print(f"{r['fold']:>3}{str(r['test_start']) + ' ~ ' + str(r['test_end']):>26}"
              f"{r['n_test']:>7}{r['up_rate']:>10.2%}{r['auc']:>8.3f}")
    aucs = [r["auc"] for r in rows if r["auc"] == r["auc"]]
    above = sum(1 for a in aucs if a > 0.5)
    print(f"\n  平均 AUC {np.mean(aucs):.3f}   >0.5 的折数 {above}/{len(aucs)}")
    if above == len(aucs) and len(aucs) >= 4:
        print("  → 每一折都强于随机，稳定性较好")
    elif above >= len(aucs) * 0.6:
        print("  → 多数折强于随机，但存在波动，需更多样本确认")
    else:
        print("  → 多数折不优于随机，之前的 AUC 很可能来自个别幸运折")

    # ---- B. 随机标签对照 ----
    print("\n[B] 随机打乱标签后重跑（AUC 应回到 0.5 附近）")
    rng = np.random.default_rng(42)
    shuffled = pd.Series(rng.permutation(y.values), index=y.index)
    rows_b = fold_aucs(X, shuffled)
    aucs_b = [r["auc"] for r in rows_b if r["auc"] == r["auc"]]
    print(f"  随机标签平均 AUC: {np.mean(aucs_b):.3f}")
    print("  （若接近 0.5，说明前面的信号不是评估流程本身造出来的）")

    # ---- C. 跨区间检验 ----
    print("\n[C] 跨区间检验（把样本前后对半，分别看 AUC）")
    mid = len(X) // 2
    for tag, xs, ys in (("前半段", X.iloc[:mid], y.iloc[:mid]),
                        ("后半段", X.iloc[mid:], y.iloc[mid:])):
        if len(xs) < 120 or ys.nunique() < 2:
            print(f"  {tag}: 样本不足，跳过")
            continue
        rows_c = fold_aucs(xs, ys, n_splits=3)
        a = [r["auc"] for r in rows_c if r["auc"] == r["auc"]]
        print(f"  {tag} ({xs.index[0].date()} ~ {xs.index[-1].date()}, "
              f"{len(xs)} 样本): 平均 AUC {np.mean(a):.3f}")

    print("\n判读：只有当 [A] 多数折 > 0.5、[B] 接近 0.5、[C] 两个子区间都存在时，")
    print("      才能认为存在真实可预测成分。")


if __name__ == "__main__":
    main()
