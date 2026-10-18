"""B1：滚动截距修正（Rolling Intercept Correction）。

严格实现 `docs/preregistration_b1.md` 冻结的规则。**不修改**任何既有文件：
本模块是一条独立的预测路径，只 import 原 HAR-3 的特征与切分函数。

三条最容易写错的规则，在代码里逐条标注：

1. **标签成熟**：`trading_day(t) − trading_day(s) ≥ 5` 才可用 `e_s`。
   交易日序号取自**原始日线序列**（未经特征 NaN 过滤），不是日历、也不是数组位置。
2. **初始化**：训练期**样本外**残差（嵌套扩展窗口），不是训练集内拟合残差。
3. **残差定义**：`e_s = y_s − ŷ_s`（基础预测），永远不是 B1 修正后的预测。

依赖：pandas / numpy / scikit-learn + 项目内轻量模块。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from har_features import ALL_COLS, HAR3, build_features  # noqa: E402
from research_pipeline import daily_rv, read_bars  # noqa: E402
from rv_estimators import TRADING_DAYS  # noqa: E402
from validation import common_index, walk_forward_indices  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 数据准备

def build_frames(bars: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, np.ndarray]:
    """返回 (对齐后的特征表, 每个日期在**原始日线**中的交易日序号)。

    关键：交易日序号来自 `daily.index`（聚合后的完整交易日序列），
    而不是特征数据集的行号——特征对齐会丢掉前若干行，用行号会让成熟条件偏早。
    """
    daily = daily_rv(bars, 2)
    feats = build_features(daily, forward=horizon)
    idx = common_index(feats, ALL_COLS)
    ds = feats.loc[idx]

    day_pos = pd.Series(np.arange(len(daily), dtype=int), index=daily.index)
    daynum = day_pos.reindex(ds.index).to_numpy()
    if np.isnan(daynum).any():
        raise ValueError("特征数据集中出现了原始日线里不存在的日期")
    return ds, daynum.astype(int)


def matured_residuals(history: list[tuple[int, float]], up_to_day: int,
                      horizon: int) -> list[float]:
    """截至 `up_to_day` 已成熟的残差，按时间顺序返回。

    成熟条件：`residual_day + horizon <= up_to_day`。
    这是 B1 唯一允许的"可用残差"定义，抽成独立函数是为了可测试。
    """
    return [e for (d, e) in history if d + horizon <= up_to_day]


def apply_window(pool: list[float], window: int) -> float | None:
    """取最近 `window` 个残差的均值；不足则返回 None（触发失败规则）。"""
    if len(pool) < window:
        return None
    return float(np.mean(pool[-window:]))


def _fit(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    model = Pipeline([("scaler", StandardScaler()),
                      ("model", LinearRegression())])
    model.fit(X, y)
    return model


# ---------------------------------------------------------------- 单折

def run_fold(ds: pd.DataFrame, daynum: np.ndarray,
             train_idx: np.ndarray, test_idx: np.ndarray,
             horizon: int, window: int, min_history_train: int) -> dict:
    """跑一个外层折，返回该折的基础预测、B1 预测与残差记录。

    状态严格限制在本折内：函数返回后不保留任何滚动状态。
    """
    X, y = ds[HAR3], ds["target"]
    test_day = daynum[test_idx]

    # 外层折的基础 HAR-3：只用训练折拟合，测试折内部系数冻结
    base_model = _fit(X.iloc[train_idx], y.iloc[train_idx])

    # ---- 训练期嵌套样本外残差 ----
    # 对训练区间内每个起点 s，只用标签已成熟的训练行 [.., s-horizon] 拟合
    train_day = daynum[train_idx]
    history: list[tuple[int, float]] = []      # (该残差所属日期的交易日序号, 残差)
    for k in range(min_history_train, len(train_idx)):
        s_pos = int(train_idx[k])
        s_day = int(daynum[s_pos])
        # 训练行 i 的标签成熟条件是 daynum[i] + horizon <= daynum[s]
        usable_mask = train_day[:k] + horizon <= s_day
        n_fit = int(usable_mask.sum())
        if n_fit < 10:
            continue
        fit_positions = train_idx[:n_fit]
        inner = _fit(X.iloc[fit_positions], y.iloc[fit_positions])
        yhat_s = float(inner.predict(X.iloc[[s_pos]])[0])
        # 注意：s_pos 是**位置**，y 的索引是日期，必须用 iloc
        history.append((s_day, float(y.iloc[s_pos]) - yhat_s))

    def matured(up_to_day: int) -> list[float]:
        return matured_residuals(history, up_to_day, horizon)

    # ---- 初始化：测试起点处最近 20 个已成熟残差 ----
    init_pool = matured(int(test_day[0]))
    init_delta = apply_window(init_pool, window)
    if init_delta is None:
        return {"ok": False,
                "reason": f"训练期已成熟残差仅 {len(init_pool)} 个（需要 {window}）",
                "fold_test_size": len(test_idx)}

    # ---- 测试折逐日推进 ----
    rows = []
    for pos, d_now in zip(test_idx, test_day):
        pos = int(pos)
        base_pred = float(base_model.predict(X.iloc[[pos]])[0])

        pool = matured(int(d_now))
        delta = apply_window(pool, window)
        if delta is None:
            return {"ok": False,
                    "reason": f"测试期第 {len(rows)+1} 天可用残差仅 {len(pool)} 个",
                    "fold_test_size": len(test_idx)}

        rows.append({
            "date": ds.index[pos],
            "y": float(y.iloc[pos]),
            "pred_base": base_pred,
            "pred_b1": base_pred + delta,
            "delta": delta,
            "n_resid": len(pool),
        })

        # 当日残差现在就登记，但成熟条件会阻止它在 t+5 之前被使用。
        # 这与"等 t+5 再登记"完全等价，且更难写错——有测试守护。
        history.append((int(d_now), float(y.iloc[pos]) - base_pred))

    out = pd.DataFrame(rows).set_index("date")
    out["base_pred_ref"] = base_model.predict(X.iloc[test_idx])
    return {"ok": True, "frame": out.drop(columns=["base_pred_ref"]),
            "base_check": float(np.max(np.abs(
                out["base_pred_ref"].to_numpy() - out["pred_base"].to_numpy())))}


def run_stock(item: dict, cfg: dict) -> dict:
    """单只股票的 B1 全流程。"""
    corr = cfg["correction"]
    horizon, window = corr["horizon"], corr["window"]
    min_hist = cfg["initialization"]["min_history_train"]

    bars = read_bars(ROOT / item["bars"])
    ds, daynum = build_frames(bars, horizon)

    folds = walk_forward_indices(
        len(ds), cfg["split"]["n_splits"],
        cfg["split"]["min_train_frac"], cfg["split"]["purge_rows"])

    frames, failures = [], []
    for fi, (tr, te) in enumerate(folds, 1):
        res = run_fold(ds, daynum, tr, te, horizon, window, min_hist)
        if not res["ok"]:
            failures.append({"ticker": item["ticker"], "fold": fi,
                             "reason": res["reason"],
                             "fold_test_size": res.get("fold_test_size")})
            continue
        if res["base_check"] > 1e-9:
            raise AssertionError(
                f"基础预测不一致（{res['base_check']:.2e}）：B1 必须与 HAR-3 同源")
        f = res["frame"].copy()
        f["fold"] = fi
        frames.append(f)

    if not frames:
        return {"ok": False, "failures": failures}

    oos = pd.concat(frames).sort_index()
    return {"ok": True, "oos": oos, "failures": failures,
            "dataset": ds, "daynum": daynum}


# ---------------------------------------------------------------- 指标

def stock_metrics(oos: pd.DataFrame, base_series: pd.Series) -> dict:
    """B1 与 HAR-3 在同一批日期上的指标与配对损失差。"""
    y = oos["y"]
    err_base = oos["pred_base"] - y
    err_b1 = oos["pred_b1"] - y
    err_rv = base_series.loc[oos.index] - y

    mse_base = float(np.mean(err_base ** 2))
    mse_b1 = float(np.mean(err_b1 ** 2))
    mse_rv = float(np.mean(err_rv ** 2))
    return {
        "n": int(len(oos)),
        "mse_base": mse_base,
        "mse_b1": mse_b1,
        "mse_rv_w": mse_rv,
        "skill_base": float(1 - mse_base / mse_rv) if mse_rv > 0 else float("nan"),
        "skill_b1": float(1 - mse_b1 / mse_rv) if mse_rv > 0 else float("nan"),
        "paired_rel_improvement": float(1 - mse_b1 / mse_base) if mse_base > 0 else float("nan"),
        "mean_delta": float(oos["delta"].mean()),
        "std_delta": float(oos["delta"].std()),
        "spearman_base": float(y.corr(oos["pred_base"], method="spearman")),
        "spearman_b1": float(y.corr(oos["pred_b1"], method="spearman")),
        "disp_ratio_base": float(oos["pred_base"].std() / y.std()),
        "disp_ratio_b1": float(oos["pred_b1"].std() / y.std()),
        "mean_n_resid": float(oos["n_resid"].mean()),
    }


def high_vol_group(ds: pd.DataFrame, daynum: np.ndarray,
                   cfg: dict) -> float:
    """用**第 1 折训练区间**的日度 RV 折算年化波动，取其中位数。

    只用预测起点之前的信息，禁止使用测试期数据。
    """
    folds = walk_forward_indices(
        len(ds), cfg["split"]["n_splits"],
        cfg["split"]["min_train_frac"], cfg["split"]["purge_rows"])
    first_train = folds[0][0]
    rv = ds["rv"].iloc[first_train]
    vol = np.sqrt(rv * TRADING_DAYS)
    return float(np.median(vol))


def paired_block_bootstrap(err_base: np.ndarray, err_b1: np.ndarray,
                           block: int = 10, n_boot: int = 1000,
                           seed: int = 42) -> tuple[float, float]:
    """配对损失差的移动块 bootstrap（处理 5 日重叠标签的自相关）。"""
    n = len(err_base)
    if n < block * 3:
        return (float("nan"), float("nan"))
    starts = np.arange(0, n - block + 1)
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(starts, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in pick])[:n]
        a, b = err_base[idx], err_b1[idx]
        mb = float(np.mean(b ** 2))
        stats[i] = 1 - mb / float(np.mean(a ** 2)) if mb >= 0 else np.nan
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def cluster_bootstrap_median(values: np.ndarray, n_boot: int = 1000,
                             seed: int = 42) -> tuple[float, float]:
    v = np.asarray([x for x in values if np.isfinite(x)])
    if len(v) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = np.array([np.median(rng.choice(v, size=len(v), replace=True))
                     for _ in range(n_boot)])
    lo, hi = np.nanpercentile(meds, [2.5, 97.5])
    return float(lo), float(hi)
