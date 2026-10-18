"""第 5-6 周：模型训练与评估（扩展窗口前向验证）。

为什么不用 train_test_split 随机切分
------------------------------------
金融时序里随机打乱会让"未来"样本进入训练集，模型相当于偷看了答案，
指标会虚高到 70%+，实盘完全不可复现。这里用**扩展窗口前向验证**：
每次训练都用测试折之前的所有数据，测试折紧接其后，绝不重叠。

为什么必须给基线
----------------
股价上涨的天数天然多于下跌（本数据里约 55%），"永远预测涨"就能拿到 55%
准确率。所以准确率必须和基线对照，否则无法判断模型是否真的学到了东西。
本模块统一报告：准确率、AUC、精确率、召回率、F1，以及两个基线。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

N_SPLITS = 5


def make_pipeline(model) -> Pipeline:
    """标准化 + 模型。

    标准化必须在 Pipeline 里，随每折的训练数据单独 fit——
    先全量标准化再切分会把测试集的均值/方差泄漏进训练。
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def walk_forward_splits(n_samples: int, n_splits: int = N_SPLITS,
                        min_train_frac: float = 0.4, purge: int = 1):
    """生成扩展窗口的 (train_idx, test_idx)，并在训练集末尾剔除 purge 行。

    [审计修复] `features.py` 的标签是 `close.pct_change().shift(-1)`，
    即索引 t 的标签用到 **t+1 的收盘价**，前视窗口 h=1。
    按 b 切开后，训练行 t=b-1 的标签用到 close[b]——落在测试区间内。
    因此必须 purge 至少 1 行。

    调用方应按各自标签的前视窗口传 purge：
    - `ml_predict.py`（标签 h=1）→ purge=1
    - `experiment.py`（h 为循环变量）→ purge=horizon
    - `robustness.py` / `validate_multi.py`（h=10）→ purge=10
    - `volatility_v2.py`（目标为未来 10 日均值）→ purge=10
    """
    from validation import walk_forward_indices

    yield from walk_forward_indices(n_samples, n_splits, min_train_frac, purge)


def evaluate_model(model, X: pd.DataFrame, y: pd.Series,
                   n_splits: int = N_SPLITS, purge: int = 1) -> dict:
    """前向验证并汇总指标（各折样本外结果的合并）。"""
    proba_all, y_all = [], []
    fold_rows = []

    for train_idx, test_idx in walk_forward_splits(len(X), n_splits,
                                                    purge=purge):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        if y_tr.nunique() < 2 or len(X_te) == 0:
            continue

        clf = make_pipeline(model)
        clf.fit(X_tr, y_tr)

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_te)[:, 1]
        else:
            proba = clf.predict(X_te).astype(float)
        pred = (proba >= 0.5).astype(int)

        proba_all.append(pd.Series(proba, index=X_te.index))
        y_all.append(y_te)
        fold_rows.append({
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": accuracy_score(y_te, pred),
            "test_up_rate": float(y_te.mean()),
        })

    if not y_all:
        raise RuntimeError("所有折都无效，无法评估")

    proba_series = pd.concat(proba_all).sort_index()
    y_true = pd.concat(y_all).sort_index()
    pred_all = (proba_series >= 0.5).astype(int)

    # 基线：永远预测涨 / 永远预测跌
    up_rate = float(y_true.mean())
    baseline_always_up = max(up_rate, 1 - up_rate)
    baseline_always_down = 1 - up_rate

    metrics = {
        "samples": len(y_true),
        "accuracy": accuracy_score(y_true, pred_all),
        "precision": precision_score(y_true, pred_all, zero_division=0),
        "recall": recall_score(y_true, pred_all, zero_division=0),
        "f1": f1_score(y_true, pred_all, zero_division=0),
        "up_rate": up_rate,
        "baseline_always_up": baseline_always_up,
        "baseline_always_down": baseline_always_down,
        "edge_over_baseline": accuracy_score(y_true, pred_all) - baseline_always_up,
        "folds": fold_rows,
        "proba": proba_series,
        "y_true": y_true,
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, proba_series)
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def build_models() -> dict:
    """返回 {名称: 未训练的模型实例}。"""
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            random_state=42, n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, eval_metric="logloss",
            verbosity=0,
        )
    except ImportError:
        pass
    return models


def format_metrics(name: str, m: dict) -> str:
    """把指标格式化成对齐的一行。"""
    return (f"{name:<14}{m['accuracy']:>9.2%}{m['auc']:>9.3f}"
            f"{m['precision']:>9.2%}{m['recall']:>9.2%}{m['f1']:>9.3f}")
