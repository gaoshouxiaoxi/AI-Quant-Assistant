"""多因子 HAR + 延长的日度 RV 目标。

一、怎么把 RV 目标做长
----------------------
免费数据源能拿到的最长高频历史是新浪的 60 分钟线（datalen 上限 1970 根）：

| 周期 | 覆盖 | 每日 bar 数 |
| --- | --- | --- |
| 15 分钟 | 约 122 个交易日 | 16 |
| 30 分钟 | 约 250 个交易日 | 8 |
| **60 分钟** | **约 492 个交易日** | 4 |

（`money.finance.sina` 的更大 datalen 返回空、`scale=120/240` 是 404、
腾讯 mkline 主机不可达、东方财富整站不通。）

因此只剩 492 天真实 RV，对多因子模型偏少。解决办法是**用代理把序列往前延**：

1. 在 60 分钟覆盖的 492 天上，同时算出两种量：
   - 真实 RV（bar 收益平方和 + 隔夜²）
   - Parkinson 方差（用同一批 bar 聚合出的日线 OHLC）
2. 在对数空间回归 `log(RV) = a + b·log(Parkinson)`，得到校准系数
3. 对 2020-2024 的日线 OHLC 套用该系数，得到**校准后的 RV 代理**

这样目标序列从 492 天扩到约 1600 天。论文里这类做法叫"proxy calibration"，
关键是把校准误差和代理的局限**如实报告**，而不是把代理当真值。

二、多因子 HAR
--------------
经典 HAR（Corsi 2009）只用三个尺度的 RV：

    RV_{t+1} = c + b1·RV_d + b2·RV_w + b3·RV_m

多因子版本在此基础上加入市场微观结构信息：

| 因子 | 含义 | 为什么可能有信息 |
| --- | --- | --- |
| 成交量比 | log(成交量 / 22 日均量) | 放量常伴随波动上升 |
| 隔夜占比 | 隔夜² / 总 RV | 隔夜跳空多的品种，日内波动结构不同 |
| 振幅比 | (高-低)/收盘 | 与 RV 互补的日内波动刻画 |
| 收益偏度/峰度 | 22 日窗口 | 尾部风险信号 |
| RV 不对称 | 下跌波动 / 上涨波动 | 杠杆效应（跌时波动更大） |

输出会同时给出：持续性基线、HAR-3、HAR-X、随机森林，方便看"多加因子到底有没有用"。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from data_loader import load_daily  # noqa: E402
from daily_rv import build_daily_rv, fetch_sina_bars  # noqa: E402
from rv_estimators import TRADING_DAYS, daily_var_parkinson  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHART_DIR = ROOT / "charts"
REPORT_DIR = ROOT / "reports"

SYMBOLS = {"sh600519": "600519", "sz000858": "000858", "sz000001": "000001"}
NAMES = {"600519": "贵州茅台", "000858": "五粮液", "000001": "平安银行"}

FORWARD = 5
N_SPLITS = 5


# ---------------------------------------------------------------- 数据拼接

def hf_daily(symbol: str) -> pd.DataFrame:
    """60 分钟 bar → 日线 OHLC + 真实 RV。"""
    return build_daily_rv(fetch_sina_bars(symbol, "60"))


def daily_ohlc(ticker: str) -> pd.DataFrame | None:
    """日线 OHLC（用于更早的历史）。"""
    path = DATA_DIR / f"{ticker}_2020-01-01_2025-01-01.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def calibrate_proxy(hf: pd.DataFrame) -> dict:
    """在同一批 bar 上拟合 log(RV) ~ log(Parkinson)，并做样本外验证。"""
    o, h, lo, c = hf["Open"], hf["High"], hf["Low"], hf["Close"]
    park = daily_var_parkinson(h, lo)

    d = pd.DataFrame({
        "rv": hf["rv_total"],
        "park": park,
    }).replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d["rv"] > 0) & (d["park"] > 0)]
    lo_q, hi_q = d["rv"].quantile([0.01, 0.99])
    d = d[(d["rv"] >= lo_q) & (d["rv"] <= hi_q)]

    x = np.log(d["park"].to_numpy())
    y = np.log(d["rv"].to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)

    # 校准能否外推：前 60% 拟合、后 40% 检验。
    # 若这里 R² 很低，说明映射不稳定，把代理外推到 2020-2024 就不可靠。
    cut = int(len(x) * 0.6)
    s2, i2 = np.polyfit(x[:cut], y[:cut], 1)
    pred = i2 + s2 * x[cut:]
    ss_res = float(np.sum((y[cut:] - pred) ** 2))
    ss_tot = float(np.sum((y[cut:] - y[cut:].mean()) ** 2))
    oos_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1 - np.var(resid) / np.var(y)),
        "rmse_log": float(np.sqrt(np.mean(resid ** 2))),
        "holdout_r2": float(oos_r2),
        "n": int(len(d)),
        "start": d.index.min().date(),
        "end": d.index.max().date(),
        "park_median": float(d["park"].median()),
    }


def build_extended_series(ticker: str, symbol: str) -> tuple[pd.DataFrame, dict]:
    """拼接：早期用校准代理，有 60 分钟数据的区间用真实 RV。"""
    hf = hf_daily(symbol)
    cal = calibrate_proxy(hf)

    # 阶段 1：用日线 OHLC 构造代理（覆盖 2020 至今）
    early = daily_ohlc(ticker)
    park_early = None
    if early is not None:
        park = daily_var_parkinson(early["High"], early["Low"])
        park_early = float(park.median())
        proxy = np.exp(cal["intercept"] + cal["slope"] * np.log(park))
        early = early.assign(rv_proxy=proxy)
        # 60 分钟已覆盖的日期不重复计算，避免两段水平跳变
        early = early.loc[early.index < hf.index.min()]
    else:
        early = pd.DataFrame(columns=["Open", "High", "Low", "Close",
                                      "Volume", "rv_proxy"])

    # 阶段 2：60 分钟区间用真实 RV
    late = hf[["Open", "High", "Low", "Close", "Volume", "rv_total"]].rename(
        columns={"rv_total": "rv_proxy"})
    late["rv_is_true"] = 1

    combined = pd.concat([early.assign(rv_is_true=0), late])
    combined = combined.sort_index()
    combined["rv_proxy"] = combined["rv_proxy"].clip(lower=1e-12)

    # 记录两段的水平，便于检查拼接处有没有跳变
    boundary = hf.index.min()
    before = combined.loc[combined.index < boundary, "rv_proxy"]
    after = combined.loc[combined.index >= boundary, "rv_proxy"]
    cal["seam_ratio"] = float(after.median() / before.median()) if len(before) else np.nan
    cal["n_true"] = int(combined["rv_is_true"].sum())
    cal["n_proxy"] = int((combined["rv_is_true"] == 0).sum())

    # 归因：若两段的 Parkinson 中位数之比 ≈ 拼接处比值，
    # 说明水平差异来自"波动率状态不同"，而不是两段用了不同的计量单位。
    if park_early is not None and cal.get("park_median"):
        cal["park_median_early"] = park_early
        cal["park_ratio"] = cal["park_median"] / park_early
        cal["park_ratio_pow_slope"] = cal["park_ratio"] ** cal["slope"]
    return combined, cal


# ---------------------------------------------------------------- 特征

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """多因子 HAR 特征。"""
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    c = df["Close"].astype(float)
    lo = df["Low"].astype(float)
    v = df["Volume"].astype(float)
    rv = df["rv_proxy"].clip(lower=1e-12)

    out = pd.DataFrame(index=df.index)
    out["rv"] = rv
    out["rv_d"] = np.log(rv)
    out["rv_w"] = np.log(rv.rolling(5).mean())
    out["rv_m"] = np.log(rv.rolling(22).mean())
    out["rv_ratio"] = out["rv_d"] - out["rv_m"]

    # 成交量
    out["vol_ratio"] = np.log1p(v / v.rolling(22).mean())
    out["vol_z"] = (np.log1p(v) - np.log1p(v).rolling(22).mean()) / \
                   np.log1p(v).rolling(22).std()

    # 隔夜与日内结构
    overnight = np.log(o / c.shift(1))
    out["overnight_share"] = (overnight ** 2 / rv).clip(0, 5)
    out["overnight_abs"] = overnight.abs()

    # 振幅与形状
    out["range_ratio"] = ((h - lo) / c)
    ret = c.pct_change()
    out["skew22"] = ret.rolling(22, min_periods=10).skew()
    out["kurt22"] = ret.rolling(22, min_periods=10).kurt()
    out["absret"] = ret.abs()

    # RV 不对称（杠杆效应）。
    # 注意：不能写 ret.where(ret<0).rolling(22).std()——被 mask 后窗口里几乎
    # 全是 NaN，rolling 默认要求满 22 个非空值，会整列变 NaN，进而把多因子
    # 版本的所有样本都 dropna 掉（实测会导致"HAR-X 样本不足"）。
    # 改用「负收益平方的滚动均值」再开方，并放低 min_periods。
    sq = ret ** 2
    down_sq = sq.where(ret < 0)
    up_sq = sq.where(ret > 0)
    out["down_vol"] = np.sqrt(down_sq.rolling(22, min_periods=5).mean() * TRADING_DAYS)
    out["up_vol"] = np.sqrt(up_sq.rolling(22, min_periods=5).mean() * TRADING_DAYS)
    out["vol_asym"] = out["down_vol"] / out["up_vol"]

    # 目标：未来 FORWARD 日均值 RV（对数）
    out["target"] = np.log(rv.rolling(FORWARD).mean().shift(-FORWARD))
    out["target_level"] = rv.rolling(FORWARD).mean().shift(-FORWARD)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


HAR3 = ["rv_d", "rv_w", "rv_m"]
# 多因子版本：在 HAR 三尺度之外加入微观结构信息
HARX = HAR3 + ["vol_ratio", "overnight_share", "range_ratio",
               "skew22", "kurt22", "vol_asym", "absret"]


# ---------------------------------------------------------------- 评估

def walk_forward(n: int, n_splits: int = N_SPLITS, min_train: float = 0.4):
    start = int(n * min_train)
    step = max(1, (n - start) // n_splits)
    for k in range(n_splits):
        b = start + k * step
        end = n if k == n_splits - 1 else b + step
        yield np.arange(0, b), np.arange(b, end)


def evaluate(data: pd.DataFrame, cols: list[str], factory, label: str) -> dict:
    d = data.dropna(subset=["target"] + cols)
    if len(d) < 150:
        return {"label": label, "n": 0}
    X, y = d[cols], d["target"]

    preds, trues = [], []
    for tr, te in walk_forward(len(X)):
        if len(tr) < 60:
            continue
        pipe = Pipeline([("scaler", StandardScaler()), ("model", factory())])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        preds.append(pd.Series(pipe.predict(X.iloc[te]), index=X.iloc[te].index))
        trues.append(y.iloc[te])

    if not trues:
        return {"label": label, "n": 0}
    p = pd.concat(preds).sort_index()
    t = pd.concat(trues).sort_index()

    # 持续性基线：直接用周均值 RV 外推（对数空间）
    base = d.loc[t.index, "rv_w"]

    mse_m = float(np.mean((p - t) ** 2))
    mse_b = float(np.mean((base - t) ** 2))
    return {
        "label": label, "n": len(t),
        "r2": r2_score(t, p), "r2_base": r2_score(t, base),
        "mae": mean_absolute_error(t, p),
        "skill": 1.0 - mse_m / mse_b if mse_b > 0 else float("nan"),
        "spearman": float(t.corr(p, method="spearman")),
        "pred": p, "true": t, "base": base,
    }


def plot_compare(result: dict, title: str, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from charts import setup_chinese_font
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result["true"].index, np.exp(result["true"]) * TRADING_DAYS,
            color="#212121", linewidth=1.1, label="实际 RV（年化）")
    ax.plot(result["pred"].index, np.exp(result["pred"]) * TRADING_DAYS,
            color="#d32f2f", linewidth=1.1, alpha=0.85, label="多因子 HAR")
    ax.plot(result["base"].index, np.exp(result["base"]) * TRADING_DAYS,
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


def main() -> None:
    print("=" * 80)
    print("多因子 HAR + 延长的日度 RV 目标")
    print("=" * 80)

    factories = {
        "HAR-3": lambda: LinearRegression(),
        "HAR-X": lambda: LinearRegression(),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=15,
            random_state=42, n_jobs=-1),
    }

    # 最后一轮：验证"因子多反而变差"是不是过拟合——
    # 若 Ridge 正则化后能追上或超过 HAR-3，就说明是过拟合而非因子无用。
    from sklearn.linear_model import Ridge

    rows, cals = [], []
    for symbol, ticker in SYMBOLS.items():
        name = NAMES[ticker]
        combined, cal = build_extended_series(ticker, symbol)
        cals.append({"ticker": ticker, "name": name, **cal})

        print(f"\n----- {name} ({ticker})")
        print(f"  校准（log RV ~ log Parkinson）：n={cal['n']}  "
              f"斜率={cal['slope']:.3f}  R²={cal['r2']:.3f}  "
              f"RMSE(log)={cal['rmse_log']:.3f}")
        print(f"  校准外推检验（前 60% 拟合 → 后 40% 预测）："
              f"R²={cal.get('holdout_r2', float('nan')):.3f}")
        print(f"  样本：真实 RV {cal['n_true']} 天 + 代理 {cal['n_proxy']} 天 "
              f"= {cal['n_true'] + cal['n_proxy']} 天 "
              f"({combined.index.min().date()} ~ {combined.index.max().date()})")
        seam = cal.get("seam_ratio", float("nan"))
        pr = cal.get("park_ratio", float("nan"))
        prs = cal.get("park_ratio_pow_slope", float("nan"))
        print(f"  拼接处 RV 中位数比（后/前）= {seam:.2f}")
        print(f"  两段 Parkinson 中位数比 = {pr:.2f}，按校准斜率折算 = {prs:.2f}")
        gap = abs(seam - prs) if seam == seam and prs == prs else float("nan")
        verdict = "可由波动率状态差异解释" if gap < 0.15 else "存在口径不一致，代理需谨慎"
        print(f"  → 归因：{verdict}（差值 {gap:.2f}）")

        data = build_features(combined)
        for label, cols, factory in (
            ("持续性基线", ["rv_w"], lambda: LinearRegression()),
            ("HAR-3", HAR3, factories["HAR-3"]),
            ("HAR-3+kurt22", HAR3 + ["kurt22"], lambda: LinearRegression()),
            ("HAR-X（多因子 OLS）", HARX, factories["HAR-X"]),
            ("HAR-X + Ridge", HARX, lambda: Ridge(alpha=1.0)),
            ("RandomForest", HARX, factories["RandomForest"]),
        ):
            m = evaluate(data, cols, factory, label)
            if not m.get("n"):
                print(f"  {label}: 样本不足")
                continue
            print(f"  {label:<16} R²={m['r2']:>7.3f}  MAE={m['mae']:.4f}  "
                  f"Skill={m['skill']:>+7.2%}  Spearman={m['spearman']:.3f}  n={m['n']}")
            rows.append({"ticker": ticker, "name": name, "model": label,
                         "n": m["n"], "r2": m["r2"], "r2_base": m["r2_base"],
                         "mae": m["mae"], "skill": m["skill"],
                         "spearman": m["spearman"]})
            if label == "HAR-X + Ridge":
                try:
                    plot_compare(m, f"{name} 多因子 HAR vs 持续性基线",
                                 CHART_DIR / f"harx_{ticker}.png")
                except Exception as exc:  # noqa: BLE001
                    print(f"    图表失败: {type(exc).__name__}")

        # ---- 逐因子消融：HAR-3 + 单个因子，看谁真的有增量 ----
        print("  逐因子消融（HAR-3 + 单个因子）：")
        ablation_base = None
        for factor in ["vol_ratio", "overnight_share", "range_ratio",
                       "skew22", "kurt22", "vol_asym", "absret"]:
            m = evaluate(data, HAR3 + [factor], lambda: LinearRegression(),
                         f"HAR-3+{factor}")
            if not m.get("n"):
                continue
            if ablation_base is None:
                base_m = evaluate(data, HAR3, lambda: LinearRegression(), "HAR-3")
                ablation_base = base_m.get("skill", float("nan"))
            delta = m["skill"] - ablation_base
            marker = "↑ 有增量" if delta > 0.005 else (
                "↓ 变差" if delta < -0.005 else "≈ 无变化")
            print(f"    +{factor:<16} Skill={m['skill']:>+7.2%}  "
                  f"Δvs HAR-3={delta:>+7.2%}  {marker}")
            rows.append({"ticker": ticker, "name": name,
                         "model": f"HAR-3+{factor}", "n": m["n"],
                         "r2": m["r2"], "r2_base": m["r2_base"],
                         "mae": m["mae"], "skill": m["skill"],
                         "spearman": m["spearman"], "ablation": 1})

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cals).to_csv(REPORT_DIR / "rv_proxy_calibration.csv", index=False)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(REPORT_DIR / "har_multifactor.csv", index=False)
        print("\n" + "=" * 80)
        print("汇总（跨标平均）")
        print("=" * 80)
        print(f"{'模型':<18}{'平均 R²':>10}{'平均 Skill':>12}{'平均 Spearman':>15}")
        print("-" * 80)
        for label, g in df.groupby("model", sort=False):
            print(f"{label:<18}{g['r2'].mean():>10.3f}{g['skill'].mean():>+12.2%}"
                  f"{g['spearman'].mean():>15.3f}")

    print("\n判读：")
    print("  - HAR-X 若明显优于 HAR-3，说明成交量/隔夜/不对称等因子有增量信息")
    print("  - 代理段（2020-2024）是校准后的 Parkinson，不是真实 RV，")
    print("    结论应理解为'在该代理目标上'的表现，不能等同于真实 RV 上的精度")


if __name__ == "__main__":
    main()
