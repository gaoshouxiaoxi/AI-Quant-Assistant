"""多因子 HAR + 延长的日度 RV 目标（审计修复版）。

本文件在 2026-10 的代码审查后做了三处修复，修复点都标注了 `[审计修复]`：

1. **purging**：标签是"未来 5 日均值 RV"，训练集末尾 5 行的标签窗口会伸进测试区间。
   现在切分时剔除这 5 行（`validation.walk_forward_indices(..., purge=5)`）。
2. **统一样本**：所有模型与基线使用同一批有效日期（特征并集无缺失），
   否则"特征多的模型样本少"会污染 Skill 对比。
3. **严格/探索分离**：
   - 严格实验只在**真实 RV**（60 分钟 bar 算出的 492 天）上做，样本外完全独立；
   - 延长到 2020 年的序列依赖"用后段高频数据校准的代理"，属于**探索性分析**，
     单独报告、明确标注。

另修复图表口径：`RV` 是**日度方差**，年化波动率应画 `sqrt(252 × RV)`，
原实现画的是 `252 × RV`（年化方差）却标成"年化波动率"。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from daily_rv import build_daily_rv, fetch_sina_bars  # noqa: E402
from data_loader import load_daily  # noqa: E402,F401
from rv_estimators import TRADING_DAYS, daily_var_parkinson  # noqa: E402
from validation import (assert_same_samples, common_index,  # noqa: E402
                        verify_no_overlap, walk_forward_indices)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHART_DIR = ROOT / "charts"
REPORT_DIR = ROOT / "reports"

SYMBOLS = {"sh600519": "600519", "sz000858": "000858", "sz000001": "000001"}
NAMES = {"600519": "贵州茅台", "000858": "五粮液", "000001": "平安银行"}

FORWARD = 5
N_SPLITS = 5

HAR3 = ["rv_d", "rv_w", "rv_m"]
HARX = HAR3 + ["vol_ratio", "overnight_share", "range_ratio",
               "skew22", "kurt22", "vol_asym", "absret"]
BASELINES = {"持续性(当日 RV)": ["rv_d"], "持续性(周均 RV)": ["rv_w"]}


# ---------------------------------------------------------------- 数据

def hf_daily(symbol: str) -> pd.DataFrame:
    return build_daily_rv(fetch_sina_bars(symbol, "60"))


def daily_ohlc(ticker: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{ticker}_2020-01-01_2025-01-01.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def calibrate_proxy(hf: pd.DataFrame) -> dict:
    """拟合 log(RV) ~ log(Parkinson)，并给出样本外诊断。"""
    park = daily_var_parkinson(hf["High"], hf["Low"])
    d = pd.DataFrame({"rv": hf["rv_total"], "park": park})
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d["rv"] > 0) & (d["park"] > 0)]
    lo_q, hi_q = d["rv"].quantile([0.01, 0.99])
    d = d[(d["rv"] >= lo_q) & (d["rv"] <= hi_q)]

    x, y = np.log(d["park"].to_numpy()), np.log(d["rv"].to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)

    cut = int(len(x) * 0.6)
    s2, i2 = np.polyfit(x[:cut], y[:cut], 1)
    pred = i2 + s2 * x[cut:]
    ss_res = float(np.sum((y[cut:] - pred) ** 2))
    ss_tot = float(np.sum((y[cut:] - y[cut:].mean()) ** 2))

    return {"slope": float(slope), "intercept": float(intercept),
            "r2": float(1 - np.var(resid) / np.var(y)),
            "rmse_log": float(np.sqrt(np.mean(resid ** 2))),
            "holdout_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "n": int(len(d)), "park_median": float(d["park"].median())}


def build_series(ticker: str, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """返回 (严格=仅真实 RV, 延长=真实+校准代理, 校准信息)。"""
    hf = hf_daily(symbol)
    cal = calibrate_proxy(hf)

    strict = hf[["Open", "High", "Low", "Close", "Volume", "rv_total"]].rename(
        columns={"rv_total": "rv"}).copy()
    strict["rv_is_true"] = 1

    early = daily_ohlc(ticker)
    park_early = None
    if early is not None:
        park = daily_var_parkinson(early["High"], early["Low"])
        park_early = float(park.median())
        proxy = np.exp(cal["intercept"] + cal["slope"] * np.log(park))
        early = early.assign(rv=proxy).loc[lambda x: x.index < hf.index.min()]

    if early is not None and len(early):
        early = early[["Open", "High", "Low", "Close", "Volume", "rv"]]
        early["rv_is_true"] = 0
        extended = pd.concat([early, strict]).sort_index()
    else:
        extended = strict.copy()

    extended["rv"] = extended["rv"].clip(lower=1e-12)
    strict["rv"] = strict["rv"].clip(lower=1e-12)

    cal["n_true"] = int(len(strict))
    cal["n_proxy"] = int((extended["rv_is_true"] == 0).sum())
    if park_early is not None and cal.get("park_median"):
        cal["park_ratio"] = cal["park_median"] / park_early
        cal["park_ratio_pow_slope"] = cal["park_ratio"] ** cal["slope"]
    return strict, extended, cal


# ---------------------------------------------------------------- 特征

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    lo = df["Low"].astype(float)
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)
    rv = df["rv"].clip(lower=1e-12)

    out = pd.DataFrame(index=df.index)
    out["rv_is_true"] = df["rv_is_true"] if "rv_is_true" in df else 1
    out["rv"] = rv
    out["rv_d"] = np.log(rv)
    out["rv_w"] = np.log(rv.rolling(5).mean())
    out["rv_m"] = np.log(rv.rolling(22).mean())

    out["vol_ratio"] = np.log1p(v / v.rolling(22).mean())
    overnight = np.log(o / c.shift(1))
    out["overnight_share"] = (overnight ** 2 / rv).clip(0, 5)
    out["range_ratio"] = (h - lo) / c

    ret = c.pct_change()
    out["skew22"] = ret.rolling(22, min_periods=10).skew()
    out["kurt22"] = ret.rolling(22, min_periods=10).kurt()
    out["absret"] = ret.abs()

    sq = ret ** 2
    down = np.sqrt(sq.where(ret < 0).rolling(22, min_periods=5).mean()
                   * TRADING_DAYS)
    up = np.sqrt(sq.where(ret > 0).rolling(22, min_periods=5).mean()
                 * TRADING_DAYS)
    out["vol_asym"] = down / up

    # 标签：未来 FORWARD 日的平均日度方差（对数）
    out["target"] = np.log(rv.rolling(FORWARD).mean().shift(-FORWARD))
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# ---------------------------------------------------------------- 评估

def evaluate(data: pd.DataFrame, cols: list[str], factory, label: str,
             purge: int, align_index: pd.Index | None = None) -> dict:
    """purge=0 复现旧行为；purge=FORWARD 为修复后行为。

    align_index 给定时，所有模型强制使用同一批日期（审计修复第 2 点）。
    """
    if align_index is not None:
        d = data.loc[align_index]
    else:
        d = data
    d = d.dropna(subset=["target"] + cols)
    if len(d) < 150:
        return {"label": label, "n": 0, "index": pd.Index([])}

    X, y = d[cols], d["target"]
    folds = walk_forward_indices(len(X), N_SPLITS, 0.4, purge)
    if purge:
        verify_no_overlap(len(X), purge, FORWARD, folds)

    preds, trues = [], []
    for tr, te in folds:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", factory())])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        preds.append(pd.Series(pipe.predict(X.iloc[te]), index=X.iloc[te].index))
        trues.append(y.iloc[te])
    if not trues:
        return {"label": label, "n": 0, "index": pd.Index([])}

    p = pd.concat(preds).sort_index()
    t = pd.concat(trues).sort_index()

    # 基线在**同一批测试行**上评估
    ref = data.loc[t.index, "rv_w"]

    mse_m = float(np.mean((p - t) ** 2))
    mse_b = float(np.mean((ref - t) ** 2))
    return {
        "label": label, "n": len(t), "index": t.index,
        "r2": r2_score(t, p), "r2_base": r2_score(t, ref),
        "mae": mean_absolute_error(t, p),
        "skill": 1.0 - mse_m / mse_b if mse_b > 0 else float("nan"),
        "spearman": float(t.corr(p, method="spearman")),
        "pred": p, "true": t, "base": ref,
    }


def plot_compare(result: dict, title: str, out_path: Path) -> Path:
    """[审计修复] 纵轴改为真正的年化波动率：sqrt(252 × RV)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from charts import setup_chinese_font
    setup_chinese_font()

    def to_vol(logrv):
        return np.sqrt(np.exp(logrv) * TRADING_DAYS)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result["true"].index, to_vol(result["true"]),
            color="#212121", linewidth=1.1, label="实际年化波动率")
    ax.plot(result["pred"].index, to_vol(result["pred"]),
            color="#d32f2f", linewidth=1.1, alpha=0.85, label="多因子 HAR")
    ax.plot(result["base"].index, to_vol(result["base"]),
            color="#1565c0", linewidth=1.0, linestyle="--", alpha=0.75,
            label="持续性基线")
    ax.set_title(title)
    ax.set_ylabel("年化波动率")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def run_block(data: pd.DataFrame, title: str, purge: int,
              align: bool) -> tuple[list[dict], dict]:
    """跑一组模型，返回 (指标行, 用于画图的完整结果)。"""
    all_cols = sorted(set(HAR3 + HARX
                          + [c for cs in BASELINES.values() for c in cs]))
    if align:
        idx = common_index(data, all_cols)
        data = data.loc[idx]

    models = [
        ("HAR-3", HAR3, lambda: LinearRegression()),
        ("HAR-3+kurt22", HAR3 + ["kurt22"], lambda: LinearRegression()),
        ("HAR-X（多因子 OLS）", HARX, lambda: LinearRegression()),
        ("HAR-X + Ridge", HARX, lambda: Ridge(alpha=1.0)),
        ("RandomForest", HARX, lambda: RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=15,
            random_state=42, n_jobs=-1)),
    ]
    rows, plot_res, results = [], {}, {}
    for label, cols, factory in models:
        m = evaluate(data, cols, factory, label, purge,
                     align_index=data.index if align else None)
        if not m.get("n"):
            print(f"    {label}: 样本不足")
            continue
        results[label] = m
        rows.append({"block": title, "model": label, "n": m["n"],
                     "r2": m["r2"], "r2_base": m["r2_base"],
                     "mae": m["mae"], "skill": m["skill"],
                     "spearman": m["spearman"]})
        if label == "HAR-X + Ridge":
            plot_res = m

    if results:
        ref_m = next(iter(results.values()))
        rows.append({"block": title, "model": "持续性(周均 RV)",
                     "n": ref_m["n"], "r2": ref_m["r2_base"],
                     "r2_base": ref_m["r2_base"], "mae": float("nan"),
                     "skill": 0.0,
                     "spearman": float(ref_m["true"].corr(
                         ref_m["base"], method="spearman"))})
        if align:
            try:
                assert_same_samples(results)
            except AssertionError as exc:
                print(f"    [样本一致性检查] {exc}")
    return rows, plot_res


def main() -> None:
    print("=" * 84)
    print("多因子 HAR（审计修复版）：purging + 统一样本 + 严格/探索分离")
    print("=" * 84)
    print(f"标签窗口 = 未来 {FORWARD} 日；切分 purge = {FORWARD} 行\n")

    all_rows, cals = [], []
    for symbol, ticker in SYMBOLS.items():
        name = NAMES[ticker]
        strict_raw, extended_raw, cal = build_series(ticker, symbol)
        cals.append({"ticker": ticker, "name": name, **cal})

        strict = build_features(strict_raw)
        extended = build_features(extended_raw)

        print(f"----- {name} ({ticker})")
        print(f"  严格样本（仅真实 RV）：{len(strict)} 天  "
              f"{strict.index.min().date()} ~ {strict.index.max().date()}")
        print(f"  探索样本（真实 {cal['n_true']} + 代理 {cal['n_proxy']}）："
              f"{len(extended)} 天  {extended.index.min().date()} ~ "
              f"{extended.index.max().date()}")
        print(f"  校准：斜率 {cal['slope']:.3f}  R² {cal['r2']:.3f}  "
              f"holdout R² {cal['holdout_r2']:.3f}"
              f"（系数由全高频样本拟合，用于更早年份属探索性）")

        blocks = [
            ("A 旧做法(不 purge/各自 dropna)", extended, 0, False),
            ("B 仅加 purging", extended, FORWARD, False),
            ("C purging + 统一样本", extended, FORWARD, True),
            ("D 严格：仅真实 RV + purging + 统一样本", strict, FORWARD, True),
        ]
        for title, data, purge, align in blocks:
            rows, plot_res = run_block(data, title, purge, align)
            all_rows.extend([{**r, "ticker": ticker, "name": name} for r in rows])
            if plot_res and title.startswith("D"):
                try:
                    plot_compare(plot_res,
                                 f"{name} 多因子 HAR（严格样本外）",
                                 CHART_DIR / f"harx_{ticker}.png")
                except Exception as exc:  # noqa: BLE001
                    print(f"    图表失败: {type(exc).__name__}")

        sub = pd.DataFrame([r for r in all_rows if r["ticker"] == ticker])
        print(f"\n  {'配置 / 模型':<44}{'n':>6}{'R²':>8}{'Skill':>9}{'Spearman':>10}")
        print("  " + "-" * 76)
        for block, g in sub.groupby("block", sort=False):
            print(f"  [{block}]")
            for _, r in g.iterrows():
                print(f"    {r['model']:<42}{int(r['n']):>6}{r['r2']:>8.3f}"
                      f"{r['skill']:>+9.2%}{r['spearman']:>10.3f}")
        print()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cals).to_csv(REPORT_DIR / "rv_proxy_calibration.csv", index=False)
    df = pd.DataFrame(all_rows)
    df.to_csv(REPORT_DIR / "har_audit_blocks.csv", index=False)

    print("=" * 84)
    print("跨标的平均 Skill（按配置）")
    print("=" * 84)
    print(f"{'配置':<46}{'HAR-3':>10}{'HAR-X':>10}{'RF':>10}")
    print("-" * 84)
    for block, g in df.groupby("block", sort=False):
        def skill_of(model, g=g):
            s = g[g["model"] == model]["skill"]
            return f"{s.mean():+.2%}" if len(s) else "—"
        print(f"{block:<46}{skill_of('HAR-3'):>10}"
              f"{skill_of('HAR-X（多因子 OLS）'):>10}"
              f"{skill_of('RandomForest'):>10}")

    print("\n判读：")
    print("  A → B = purging 的影响")
    print("  B → C = 统一样本的影响")
    print("  C → D = 去掉探索性代理样本后的严格结果")
    print("  若修复后 Skill 变负，如实保留，不做二次调参掩盖。")


if __name__ == "__main__":
    main()
