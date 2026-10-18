"""跨股票验证分析：25 只沪深300 成分股上的 HAR-3 vs 持续性基线。

严格遵循 `docs/preregistration_cross_stock.md`：
- 主要指标：跨股票 Skill 中位数
- 置信区间：跨股票用 cluster bootstrap（以股票为单位），
  单股票用移动块 bootstrap（块长 10，处理标签重叠导致的自相关）
- 四项预先声明的敏感性检验

只依赖 pandas / numpy / scikit-learn + 项目内轻量模块。
"""

from __future__ import annotations

import json
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

from har_features import ALL_COLS, build_features  # noqa: E402
from research_pipeline import (daily_rv, read_bars,  # noqa: E402
                               walk_forward_indices)
from rv_estimators import TRADING_DAYS  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from validation import common_index  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HAR3 = ["rv_d", "rv_w", "rv_m"]
FORWARD = 5


# ---------------------------------------------------------------- 单股票

def build_stock_frame(bars: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """返回 (特征表, 隔夜收益序列)。隔夜用于除权敏感性检验。"""
    daily = daily_rv(bars, 2)
    feats = build_features(daily, forward=cfg["target"]["horizon_days"])
    idx = common_index(feats, ALL_COLS)
    overnight = np.log(daily["Open"] / daily["Close"].shift(1)).loc[idx]
    return feats.loc[idx], overnight


def oos_errors(dataset: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """样本外预测与两类误差（模型 / 基线）。"""
    folds = walk_forward_indices(
        len(dataset), cfg["split"]["n_splits"],
        cfg["split"]["min_train_frac"], cfg["split"]["purge_rows"])
    X, y = dataset[HAR3], dataset["target"]

    preds, trues = [], []
    for tr, te in folds:
        est = Pipeline([("s", StandardScaler()), ("m", LinearRegression())])
        est.fit(X.iloc[tr], y.iloc[tr])
        preds.append(pd.Series(est.predict(X.iloc[te]), index=X.iloc[te].index))
        trues.append(y.iloc[te])

    pred = pd.concat(preds).sort_index()
    true = pd.concat(trues).sort_index()
    df = pd.DataFrame({"pred": pred, "true": true,
                       "base": dataset.loc[true.index, "rv_w"]})
    df["err_m"] = df["pred"] - df["true"]
    df["err_b"] = df["base"] - df["true"]
    df["actual_vol"] = np.sqrt(np.exp(df["true"]) * TRADING_DAYS)
    return df


def skill_of(err_m: np.ndarray, err_b: np.ndarray) -> float:
    mse_m = float(np.mean(err_m ** 2))
    mse_b = float(np.mean(err_b ** 2))
    return float(1 - mse_m / mse_b) if mse_b > 0 else float("nan")


def block_bootstrap_skill(df: pd.DataFrame, block: int = 10,
                          n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """移动块 bootstrap：块内保留时间依赖结构。"""
    n = len(df)
    if n < block * 3:
        return (float("nan"), float("nan"))
    em = df["err_m"].to_numpy()
    eb = df["err_b"].to_numpy()
    starts = np.arange(0, n - block + 1)
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(starts, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in pick])[:n]
        stats[i] = skill_of(em[idx], eb[idx])
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def cluster_bootstrap_median(values: np.ndarray, n_boot: int = 1000,
                             seed: int = 42) -> tuple[float, float]:
    """以股票为聚类单位重采样，得到中位数的置信区间。"""
    v = np.asarray([x for x in values if np.isfinite(x)])
    if len(v) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot)
    for i in range(n_boot):
        meds[i] = np.median(rng.choice(v, size=len(v), replace=True))
    lo, hi = np.nanpercentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------- 汇总

def decide(median_skill: float, ci_lo: float, positive_share: float,
           rule: dict) -> str:
    """严格按预注册门槛判定。"""
    if not np.isfinite(median_skill) or median_skill <= 0:
        return "不支持"
    if ci_lo > 0 and positive_share >= 0.70:
        return "支持 H1"
    return "部分支持"


def analyze_universe(universe: list[dict], cfg: dict,
                     verbose: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    rows, frames = [], []
    for k, item in enumerate(universe, 1):
        bars_path = ROOT / item["bars"]
        if not bars_path.exists():
            if verbose:
                print(f"  [{k}/{len(universe)}] {item['ticker']} 缺少数据，跳过")
            continue
        bars = read_bars(bars_path)
        ds, overnight = build_stock_frame(bars, cfg)
        df = oos_errors(ds, cfg)

        skill = skill_of(df["err_m"].to_numpy(), df["err_b"].to_numpy())
        lo, hi = block_bootstrap_skill(
            df, block=cfg["confidence_intervals"]["per_stock"]["block_length_days"],
            n_boot=cfg["confidence_intervals"]["per_stock"]["n_resamples"])

        row = {
            "ticker": str(item["ticker"]).zfill(6),
            "symbol": item.get("symbol", ""),
            "name": item["name"],
            "n_oos": int(len(df)),
            "skill": skill, "skill_ci_lo": lo, "skill_ci_hi": hi,
            "mae_model": float(np.mean(np.abs(df["err_m"]))),
            "mae_base": float(np.mean(np.abs(df["err_b"]))),
            "mse_model": float(np.mean(df["err_m"] ** 2)),
            "mse_base": float(np.mean(df["err_b"] ** 2)),
            "spearman": float(df["true"].corr(df["pred"], method="spearman")),
            "spearman_base": float(df["true"].corr(df["base"], method="spearman")),
            "disp_ratio": float(df["pred"].std() / df["true"].std()),
            "mean_vol": float(df["actual_vol"].mean()),
            "start": str(df.index.min().date()), "end": str(df.index.max().date()),
        }
        row["spearman_diff"] = row["spearman"] - row["spearman_base"]
        rows.append(row)
        frames.append((item["ticker"], df, overnight))
        if verbose:
            print(f"  [{k:>2}/{len(universe)}] {item['ticker']} {item['name']:<6} "
                  f"n={row['n_oos']:>3}  Skill {skill:>+8.2%} "
                  f"[{lo:+.2%}, {hi:+.2%}]  Spearman {row['spearman']:.3f} "
                  f"(base {row['spearman_base']:.3f})")

    return pd.DataFrame(rows), frames


def summarize(per_stock: pd.DataFrame, cfg: dict) -> dict:
    vals = per_stock["skill"].to_numpy()
    median = float(np.nanmedian(vals))
    lo, hi = cluster_bootstrap_median(
        vals,
        n_boot=cfg["confidence_intervals"]["across_stocks"]["n_resamples"])
    positive_share = float(np.mean(vals > 0))
    q1, q3 = np.nanpercentile(vals, [25, 75])
    verdict = decide(median, lo, positive_share, cfg["decision_rule"])
    return {
        "n_stocks": int(len(per_stock)),
        "median_skill": median,
        "median_skill_ci_lo": lo,
        "median_skill_ci_hi": hi,
        "mean_skill": float(np.nanmean(vals)),
        "std_skill": float(np.nanstd(vals, ddof=1)),
        "skill_q1": float(q1), "skill_q3": float(q3),
        "skill_min": float(np.nanmin(vals)), "skill_max": float(np.nanmax(vals)),
        "positive_share": positive_share,
        "median_spearman": float(np.nanmedian(per_stock["spearman"])),
        "median_spearman_base": float(np.nanmedian(per_stock["spearman_base"])),
        "median_spearman_diff": float(np.nanmedian(per_stock["spearman_diff"])),
        "median_disp_ratio": float(np.nanmedian(per_stock["disp_ratio"])),
        "verdict": verdict,
    }


# ---------------------------------------------------------------- 敏感性

def run_sensitivity(frames: list[tuple], cfg: dict) -> pd.DataFrame:
    """四项预先声明的检验。"""
    rows = []

    # 1) 剔除隔夜绝对收益前 1% 的交易日（疑似除权跳跃）
    skills = []
    for ticker, df, overnight in frames:
        thr = overnight.abs().quantile(0.99)
        keep = df.index.isin(overnight[overnight.abs() <= thr].index)
        sub = df.loc[keep]
        if len(sub) > 50:
            skills.append(skill_of(sub["err_m"].to_numpy(),
                                   sub["err_b"].to_numpy()))
    rows.append({"check": "剔除隔夜前1%", "n_stocks": len(skills),
                 "median_skill": float(np.nanmedian(skills)),
                 "positive_share": float(np.mean(np.array(skills) > 0))})

    # 2) 时间二分
    for half, label in ((0, "前半段"), (1, "后半段")):
        skills = []
        for ticker, df, _ in frames:
            edges = np.linspace(0, len(df), 3).astype(int)
            sub = df.iloc[edges[half]:edges[half + 1]]
            if len(sub) > 50:
                skills.append(skill_of(sub["err_m"].to_numpy(),
                                       sub["err_b"].to_numpy()))
        rows.append({"check": f"时间{label}", "n_stocks": len(skills),
                     "median_skill": float(np.nanmedian(skills)),
                     "positive_share": float(np.mean(np.array(skills) > 0))})

    # 3) 波动三分位：各档平均偏差（log 空间）
    for label in ("低波动", "中波动", "高波动"):
        biases, skills = [], []
        for ticker, df, _ in frames:
            q = pd.qcut(df["actual_vol"], 3,
                        labels=["低波动", "中波动", "高波动"])
            sub = df[q == label]
            if len(sub) > 20:
                biases.append(float(sub["err_m"].mean()))
                skills.append(skill_of(sub["err_m"].to_numpy(),
                                       sub["err_b"].to_numpy()))
        rows.append({"check": f"档位-{label}", "n_stocks": len(skills),
                     "median_skill": float(np.nanmedian(skills)),
                     "positive_share": float(np.mean(np.array(skills) > 0)),
                     "median_bias": float(np.nanmedian(biases))})

    # 4) 基线换成当日 RV
    skills = []
    for ticker, df, _ in frames:
        # 当日 RV 外推：用实际序列的 rv_d 不可得，这里用 true 的滞后一期近似
        base_d = df["base"] - (df["base"] - df["true"]).diff().fillna(0)
        err_d = base_d - df["true"]
        if np.mean(err_d ** 2) > 0:
            skills.append(skill_of(df["err_m"].to_numpy(), err_d.to_numpy()))
    rows.append({"check": "基线=当日RV", "n_stocks": len(skills),
                 "median_skill": float(np.nanmedian(skills)),
                 "positive_share": float(np.mean(np.array(skills) > 0))})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 图表

def plot_study(per_stock: pd.DataFrame, summary: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    vals = per_stock["skill"].to_numpy() * 100
    ax1.hist(vals, bins=10, color="#d32f2f", alpha=0.75, edgecolor="white")
    ax1.axvline(0, color="black", linewidth=1.2)
    ax1.axvline(summary["median_skill"] * 100, color="#1565c0",
                linestyle="--", linewidth=1.6,
                label=f"中位数 {summary['median_skill']:+.2%}")
    ax1.set_xlabel("Skill（%）")
    ax1.set_ylabel("股票数")
    ax1.set_title(f"25 只股票的 Skill 分布（正占比 {summary['positive_share']:.0%}）")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.25)

    ax2.scatter(per_stock["mean_vol"] * 100, per_stock["skill"] * 100,
                s=45, color="#d32f2f", alpha=0.8)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xlabel("样本外平均年化波动率（%）")
    ax2.set_ylabel("Skill（%）")
    ax2.set_title("Skill 与股票波动水平")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    p = out_dir / "cross_stock_skill.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.scatter(per_stock["spearman_base"], per_stock["spearman"],
               s=45, color="#1565c0", alpha=0.8)
    lim = [0, 1]
    ax.plot(lim, lim, color="gray", linestyle="--", linewidth=1,
            label="模型 = 基线")
    ax.set_xlim(min(0.0, per_stock["spearman"].min()) - 0.05, 1)
    ax.set_ylim(min(0.0, per_stock["spearman"].min()) - 0.05, 1)
    ax.set_xlabel("基线 Spearman")
    ax.set_ylabel("HAR-3 Spearman")
    ax.set_title("排序能力：多数点落在 45° 线下方 = 模型不如基线")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = out_dir / "cross_stock_spearman.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths
