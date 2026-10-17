"""跳跃建模：检测波动尖峰并尝试预测。

一、怎么把"跳跃"和"连续波动"分开
----------------------------------
连续的布朗运动会让已实现波动 RV 与双幂变差 BV 相近；
一旦出现跳跃（突发利空、公告、闪崩），RV 会显著大于 BV。
所以定义跳跃占比：

    J_t / RV_t = max(RV_t − BV_t, 0) / RV_t

其中双幂变差（Barndorff-Nielsen & Shephard, 2004）：

    BV_t = (π/2) · Σ |r_i| · |r_{i−1}|

直觉：BV 用相邻收益的**乘积**，跳跃只影响单根 bar，乘积会被"稀释"；
而 RV 用**平方**，跳跃会被放大。两者之差就是跳跃的贡献。

二、数据与分辨率
----------------
- 60 分钟 bar：492 个交易日（2024-10 ~ 2026-10），每交易日 4 根
- 15 分钟 bar：122 个交易日（2026-04 ~ 2026-10），每交易日 16 根

两者在 2026-04 ~ 2026-10 有重叠，因此可以做**跨分辨率一致性检查**：
如果两种分辨率检出的是同一批跳跃日，说明跳跃是数据本身的特征，
而不是某一种采样频率的假象。

三、预测任务
------------
目标：未来 5 个交易日内是否出现跳跃日（二分类）。
特征只用到 t 及之前的信息。基线用"当前波动水平"直接判断，
模型必须打败它才算有增量价值。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
MINUTE_DIR = ROOT / "data" / "minute"
CHART_DIR = ROOT / "charts"
REPORT_DIR = ROOT / "reports"

SYMBOLS = {"sh600519": "贵州茅台", "sz000858": "五粮液", "sz000001": "平安银行"}
JUMP_RATIO_THRESHOLD = 0.35     # (RV-BV)/RV 超过此值判为跳跃日
FORWARD_DAYS = 5
N_SPLITS = 3

# MAD 离群检测参数：log RV 超过 expanding 中位数 + SPIKE_K × MAD 即判为尖峰
SPIKE_K = 2.0
MAD_TO_SIGMA = 1.4826          # 正态下 MAD 换算成标准差的系数


def theoretical_jump_bias(n_bars: int) -> float:
    """BV 口径的跳跃占比在"无跳跃"情形下的期望值。

    对 iid 收益，E[RV] = n·σ²，E[BV] = (π/2)·(n−1)·(E|r|)² = (n−1)·σ²，
    因此 (RV−BV)/RV 的期望约为 1/n。

    这解释了为什么 60 分钟（n=4）用 0.35 作阈值会误判约一半的日子：
    噪声本身就有 0.25 的基线。BV 方法需要高频数据才可靠。
    """
    return 1.0 / n_bars if n_bars > 0 else float("nan")


def detect_spikes(daily: pd.DataFrame, k: float = SPIKE_K,
                  min_periods: int = 40) -> pd.DataFrame:
    """用稳健 z 分数（MAD 口径）标记波动尖峰。

    为什么不用 BV
    -------------
    60 分钟每日仅 4 根 bar，BV 只剩 3 个乘积项，其"无跳跃期望值"高达 1/4，
    阈值法基本失效。MAD 口径不依赖 bar 数量，直接看当日波动相对历史是否异常。

    为什么用 expanding 而非 rolling
    -------------------------------
    expanding 只用 t 之前的数据算中位数与 MAD，天然不会用到未来信息；
    rolling 也可以，但 expanding 在每个时点的样本更多、更稳定。

    稳健性：中位数和 MAD 都不受极端值影响，不会被"尖峰自己"抬高门槛。
    """
    out = daily.copy()
    log_rv = np.log(out["rv_total"].clip(lower=1e-12))

    median = log_rv.expanding(min_periods=min_periods).median()
    mad = (log_rv - median).abs().expanding(min_periods=min_periods).median()
    robust_sigma = (mad * MAD_TO_SIGMA).replace(0, np.nan)
    out["log_rv"] = log_rv
    out["rv_z"] = (log_rv - median) / robust_sigma
    out["is_spike"] = (out["rv_z"] > k).astype(int)
    # expanding 未成形时（前 60 天）无判定
    out.loc[median.isna(), "is_spike"] = np.nan
    return out


# ---------------------------------------------------------------- 数据

def load_bars(symbol: str, scale: str) -> pd.DataFrame:
    path = MINUTE_DIR / f"{symbol}_{scale}m.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path.name}，请先运行 src/daily_rv.py 或抓取脚本")
    df = pd.read_csv(path, parse_dates=["day"]).set_index("day").sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_rv_bv(bars: pd.DataFrame) -> pd.DataFrame:
    """按日计算 RV、BV、跳跃占比。

    RV = Σ r_i²，BV = (π/2)·Σ|r_i||r_{i−1}|（r 为日内 bar 收益）。
    注意只用日内收益（不含隔夜），因为隔夜跳空会同时污染 RV 和 BV。
    """
    bars = bars.copy()
    bars["date"] = bars.index.date
    rows = []

    for date, g in bars.groupby("date"):
        g = g.sort_index()
        closes = g["close"].to_numpy()
        opens = g["open"].to_numpy()
        if len(closes) < 3:
            continue
        prev = np.r_[opens[0], closes[:-1]]
        r = np.log(closes / prev)
        r = r[np.isfinite(r)]
        if len(r) < 3:
            continue

        rv = float(np.sum(r ** 2))
        bv = float((np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
        # 若某根 bar 完全没有波动，BV 可能为 0，此时跳跃占比无定义
        jump_ratio = (rv - bv) / rv if rv > 0 else np.nan

        rows.append({
            "date": pd.Timestamp(date),
            "rv": rv,
            "bv": bv,
            "jump_ratio": jump_ratio,
            "n_bars": len(r),
            "Open": float(opens[0]),
            "High": float(g["high"].max()),
            "Low": float(g["low"].min()),
            "Close": float(closes[-1]),
            "Volume": float(g["volume"].sum()),
        })

    df = pd.DataFrame(rows).set_index("date").sort_index()
    df["overnight"] = np.log(df["Open"] / df["Close"].shift(1))
    df = df.dropna(subset=["overnight"])
    df["rv_total"] = df["rv"] + df["overnight"] ** 2
    df["is_jump"] = (df["jump_ratio"] > JUMP_RATIO_THRESHOLD).astype(int)
    return df


# ---------------------------------------------------------------- 特征与目标

def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """构造尖峰预测的特征与目标（未来 5 日内是否出现尖峰）。"""
    out = daily.copy()
    rv = out["rv_total"].clip(lower=1e-12)

    # 波动水平与结构（对数化，波动率分布右偏）
    out["f_log_rv"] = np.log(rv)
    out["f_log_rv5"] = np.log(rv.rolling(5).mean())
    out["f_log_rv20"] = np.log(rv.rolling(22).mean())
    out["f_rv_ratio_5_20"] = out["f_log_rv5"] - out["f_log_rv20"]
    out["f_rv_ratio_1_20"] = out["f_log_rv"] - out["f_log_rv20"]

    # 尖峰有聚集性：近期出现过尖峰，则短期再出现的概率更高
    out["f_rv_z"] = out["rv_z"]
    out["f_rv_z_max5"] = out["rv_z"].rolling(5).max()
    out["f_spike_count_22"] = out["is_spike"].rolling(22).sum()

    # 隔夜与日内分解
    log_oc = np.log(out["Open"] / out["Close"].shift(1))
    out["f_overnight_abs"] = log_oc.abs().rolling(5).mean()
    out["f_overnight_share"] = (
        out["overnight"] ** 2 / out["rv_total"].clip(lower=1e-12)
    )

    # 收益形状
    ret = out["Close"].pct_change()
    out["f_absret"] = ret.abs()
    out["f_skew20"] = ret.rolling(22).skew()
    out["f_kurt20"] = ret.rolling(22).kurt()

    # 成交量
    out["f_volume_ratio"] = np.log1p(
        out["Volume"] / out["Volume"].rolling(22).mean())

    # 目标：未来 FORWARD_DAYS 日内是否出现尖峰
    future_spike = out["is_spike"].rolling(FORWARD_DAYS).max().shift(-FORWARD_DAYS)
    out["target"] = future_spike
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]


def walk_forward(n: int, n_splits: int = N_SPLITS, min_train: float = 0.4):
    start = int(n * min_train)
    step = max(1, (n - start) // n_splits)
    for k in range(n_splits):
        b = start + k * step
        end = n if k == n_splits - 1 else b + step
        yield np.arange(0, b), np.arange(b, end)


def auc_ci(p: pd.Series, t: pd.Series, n_boot: int = 500,
           seed: int = 42) -> tuple[float, float, float]:
    """AUC 的 bootstrap 置信区间。

    尖峰事件只有 8~11 个，AUC 的点估计极不稳定。若 95% 区间覆盖 0.5，
    就不能宣称"有预测能力"——这是稀疏事件分类里最容易被忽略的一步。
    """
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(t))
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        tt = t.iloc[pick]
        if tt.nunique() < 2:
            continue
        vals.append(roc_auc_score(tt, p.iloc[pick]))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return (float(np.mean(vals)),
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def evaluate(df: pd.DataFrame, factory, label: str) -> dict:
    """扩展窗口前向验证，返回样本外 AUC 与基础比率。"""
    data = df.dropna(subset=["target"] + feature_columns(df))
    X = data[feature_columns(df)]
    y = data["target"].astype(int)

    proba, truth = [], []
    for tr, te in walk_forward(len(X)):
        if y.iloc[tr].nunique() < 2 or len(te) == 0:
            continue
        pipe = Pipeline([("scaler", StandardScaler()), ("model", factory())])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        proba.append(pd.Series(pipe.predict_proba(X.iloc[te])[:, 1],
                               index=X.iloc[te].index))
        truth.append(y.iloc[te])

    if not truth:
        return {"label": label, "n": 0}
    p = pd.concat(proba).sort_index()
    t = pd.concat(truth).sort_index()
    try:
        auc = roc_auc_score(t, p)
    except ValueError:
        auc = float("nan")
    _, lo, hi = auc_ci(p, t)
    return {"label": label, "n": len(t), "base_rate": float(t.mean()),
            "events": int(t.sum()),
            "auc": auc, "auc_lo": lo, "auc_hi": hi,
            "pred": p, "true": t,
            "hit_rate_top20": float(t.loc[p.nlargest(max(1, len(p) // 5)).index].mean())}


# ---------------------------------------------------------------- 主流程

def main() -> None:
    print("=" * 78)
    print("跳跃建模：检测波动尖峰 + 预测未来 5 日是否出现尖峰")
    print("=" * 78)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary, agreement_rows = [], []

    for symbol, name in SYMBOLS.items():
        raw = compute_rv_bv(load_bars(symbol, "60"))
        daily = detect_spikes(raw)
        n_bars = int(daily["n_bars"].median())

        print(f"----- {name}（60 分钟，{len(daily)} 个交易日，每日约 {n_bars} 根 bar）")

        # BV 口径为什么不可用：给出无跳跃时的理论基线
        bias = theoretical_jump_bias(n_bars)
        bv_rate = float((raw["jump_ratio"] > JUMP_RATIO_THRESHOLD).mean())
        print(f"  BV 口径阈值 {JUMP_RATIO_THRESHOLD:.0%}：判定跳跃 "
              f"{bv_rate:.1%} 的日子")
        print(f"  但 n={n_bars} 时，无跳跃情形下 (RV−BV)/RV 的期望就有 "
              f"{bias:.2f} —— 阈值被噪声穿透，因此改用 MAD 口径")

        valid = daily.dropna(subset=["is_spike"])
        spikes = valid[valid["is_spike"] == 1]
        print(f"  MAD 口径（z > {SPIKE_K}）：尖峰 {len(spikes)} 天"
              f"（{len(spikes) / len(valid):.1%}）")
        if len(spikes):
            for d, row in spikes.nlargest(3, "rv_z").iterrows():
                print(f"    {d.date()}  z={row['rv_z']:.2f}  "
                      f"年化波动 {np.sqrt(row['rv_total'] * 252):.1%}")

        # ---- 跨分辨率一致性（仅重叠区间）----
        try:
            d15 = detect_spikes(compute_rv_bv(load_bars(symbol, "15")))
            common = daily.index.intersection(d15.index)
            common = [d for d in common
                      if not pd.isna(daily.loc[d, "is_spike"])
                      and not pd.isna(d15.loc[d, "is_spike"])]
            if len(common) >= 30:
                a = daily.loc[common, "is_spike"]
                b = d15.loc[common, "is_spike"]
                both = int(((a == 1) & (b == 1)).sum())
                either = int(((a == 1) | (b == 1)).sum())
                jacc = both / either if either else float("nan")
                print(f"  跨分辨率一致性（重叠 {len(common)} 天，MAD 口径）："
                      f"都判尖峰 {both} 天，任一 {either} 天，Jaccard={jacc:.2f}")
                agreement_rows.append({"symbol": symbol, "name": name,
                                       "overlap_days": len(common),
                                       "both_spike": both, "either_spike": either,
                                       "jaccard": jacc})
        except FileNotFoundError:
            pass

        # ---- 预测 ----
        feat = build_features(daily)
        models = {
            "Logistic": lambda: LogisticRegression(max_iter=1000, C=1.0),
            "RandomForest": lambda: RandomForestClassifier(
                n_estimators=300, max_depth=4, min_samples_leaf=20,
                random_state=42, n_jobs=-1),
        }
        # 基线只用"当前波动水平"一个特征
        base_df = pd.DataFrame({"target": feat["target"],
                                "f_log_rv5": feat["f_log_rv5"]})
        base = evaluate(base_df, lambda: LogisticRegression(max_iter=1000),
                        "基线:仅波动水平")
        if base.get("n"):
            print(f"  [基线] 仅波动水平 → AUC={base['auc']:.3f} "
                  f"[{base['auc_lo']:.2f},{base['auc_hi']:.2f}]  "
                  f"事件={base['events']}/{base['n']}  "
                  f"基础率={base['base_rate']:.1%}  "
                  f"前20%命中率={base['hit_rate_top20']:.1%}")

        for mname, factory in models.items():
            m = evaluate(feat, factory, mname)
            if not m.get("n"):
                continue
            sig = "显著" if m["auc_lo"] > 0.5 else "不显著"
            print(f"  {mname:<14} AUC={m['auc']:.3f} "
                  f"[{m['auc_lo']:.2f},{m['auc_hi']:.2f}] {sig}  "
                  f"事件={m['events']}/{m['n']}  "
                  f"基础率={m['base_rate']:.1%}  "
                  f"前20%命中率={m['hit_rate_top20']:.1%}")
            summary.append({"symbol": symbol, "name": name, "model": mname,
                            "n": m["n"], "base_rate": m["base_rate"],
                            "events": m["events"],
                            "auc": m["auc"], "auc_lo": m["auc_lo"],
                            "auc_hi": m["auc_hi"],
                            "hit_rate_top20": m["hit_rate_top20"],
                            "baseline_auc": base.get("auc")})
        if base.get("n"):
            summary.append({"symbol": symbol, "name": name,
                            "model": "基线:仅波动水平", "n": base["n"],
                            "base_rate": base["base_rate"],
                            "auc": base["auc"],
                            "hit_rate_top20": base["hit_rate_top20"],
                            "baseline_auc": base["auc"]})
        print()

    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(REPORT_DIR / "jump_prediction.csv", index=False)
        print("=" * 78)
        print("汇总")
        print("=" * 78)
        print(f"{'标的':<12}{'模型':<18}{'AUC':>8}{'基础率':>9}{'前20%命中':>11}")
        print("-" * 78)
        for _, r in df.iterrows():
            print(f"{r['name']:<12}{r['model']:<18}{r['auc']:>8.3f}"
                  f"{r['base_rate']:>9.1%}{r['hit_rate_top20']:>11.1%}")
        model_rows = df[df["model"] != "基线:仅波动水平"]
        base_rows = df[df["model"] == "基线:仅波动水平"]
        if not model_rows.empty:
            print(f"\n模型平均 AUC={model_rows['auc'].mean():.3f}  "
                  f"基线平均 AUC={base_rows['auc'].mean():.3f}")

    if agreement_rows:
        pd.DataFrame(agreement_rows).to_csv(
            REPORT_DIR / "jump_cross_resolution.csv", index=False)

    print("\n判读：")
    print("  - AUC 明显高于 0.5 且超过基线，才说明尖峰可被提前识别")
    print("  - 前20%命中率是实用指标：只在高概率时预警，看预警准不准")
    print("  - 基础率若接近 100%，说明目标定义退化了，结论无意义")


if __name__ == "__main__":
    main()
